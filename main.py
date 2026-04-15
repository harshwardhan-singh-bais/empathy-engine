import os
import time
import logging
import re
from pathlib import Path

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from emotion_engine import detect_emotion
from mapping_engine import compute_voice_parameters, generate_ssml
from tts_engine import synthesize_speech, get_available_voices

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv(".env")        # Load general .env
load_dotenv(".env.local")  # Load local overrides

# ---------------------------------------------------------------------------
# Logging setup — plain text, no emojis, every step visible in console
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("empathy-engine")

# ---------------------------------------------------------------------------
# Startup directory scaffolding
# ---------------------------------------------------------------------------
logger.info("STARTUP: Creating required directories")
try:
    Path("outputs").mkdir(exist_ok=True)
    logger.info("STARTUP: [OK] outputs/ directory ready")
except Exception as exc:
    logger.error("STARTUP: [FAIL] Could not create outputs/ directory - %s", exc)
    raise

try:
    Path("static").mkdir(exist_ok=True)
    logger.info("STARTUP: [OK] static/ directory ready")
except Exception as exc:
    logger.error("STARTUP: [FAIL] Could not create static/ directory - %s", exc)
    raise

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
logger.info("STARTUP: Initialising FastAPI application")
app = FastAPI(
    title="Empathy Engine",
    description="Emotionally-aware Text-to-Speech synthesis service",
    version="1.0.0",
)
logger.info("STARTUP: [OK] FastAPI app created")

# ---------------------------------------------------------------------------
# CORS — allow all origins so browser fetch() works from any context
# This fixes the CORS block when testing locally
# ---------------------------------------------------------------------------
logger.info("STARTUP: Registering CORS middleware (allow all origins)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("STARTUP: [OK] CORS middleware registered")

# ---------------------------------------------------------------------------
# Static file mounts
# ---------------------------------------------------------------------------
logger.info("STARTUP: Mounting static file directories")
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("STARTUP: [OK] /static -> static/ mounted")
except Exception as exc:
    logger.error("STARTUP: [FAIL] Could not mount /static - %s", exc)

try:
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    logger.info("STARTUP: [OK] /outputs -> outputs/ mounted")
except Exception as exc:
    logger.error("STARTUP: [FAIL] Could not mount /outputs - %s", exc)

# ---------------------------------------------------------------------------
# Jinja2 templates
# ---------------------------------------------------------------------------
logger.info("STARTUP: Loading Jinja2 templates from templates/")
try:
    templates = Jinja2Templates(directory="templates")
    logger.info("STARTUP: [OK] Jinja2 templates loaded")
except Exception as exc:
    logger.error("STARTUP: [FAIL] Could not load Jinja2 templates - %s", exc)
    raise

# ---------------------------------------------------------------------------
# ElevenLabs key check at startup
# ---------------------------------------------------------------------------
_api_key_present = bool(os.getenv("ELEVENLABS_API_KEY", "").strip())
if _api_key_present:
    logger.info("STARTUP: [OK] ELEVENLABS_API_KEY found - primary TTS engine = ElevenLabs")
else:
    logger.warning(
        "STARTUP: [WARN] ELEVENLABS_API_KEY not set - "
        "TTS will fall back to gTTS then pyttsx3 (offline mode)"
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = None


class SynthesizeResponse(BaseModel):
    text: str
    dominant_emotion: str
    granular_label: str
    category: str
    intensity: float
    raw_intensity: float
    punctuation_boost: float
    fine_emotions: dict
    voice_params: dict
    ssml: str
    audio_url: str
    engine_used: str
    voice_id: str
    voice_selection: dict = {}
    shaped_text: str = ""



# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info("GET / - serving index page to client %s", request.client)

    logger.info("ROUTE index: Fetching available voices for UI dropdown")
    try:
        voices = get_available_voices()
        logger.info("ROUTE index: [OK] Retrieved %d voice(s)", len(voices))
    except Exception as exc:
        logger.error("ROUTE index: [FAIL] get_available_voices() raised - %s", exc)
        voices = []

    logger.info("ROUTE index: Rendering templates/index.html")
    try:
        response = templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"voices": voices},
        )
        logger.info("ROUTE index: [OK] Template rendered, sending response")
        return response
    except Exception as exc:
        logger.error("ROUTE index: [FAIL] Template render failed - %s", exc)
        raise HTTPException(status_code=500, detail=f"Template rendering failed: {exc}")


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(body: SynthesizeRequest):
    pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info("POST /synthesize - new request received")
    logger.info("REQUEST: raw text length = %d chars", len(body.text))
    logger.info("REQUEST: voice_id = %s", body.voice_id or "(default)")

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    logger.info("STEP 0: Input validation")
    text = body.text.strip()

    if not text:
        logger.warning("STEP 0: [FAIL] Text is empty after stripping whitespace")
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if len(text) > 2000:
        logger.warning(
            "STEP 0: [FAIL] Text length %d exceeds 2000 character limit", len(text)
        )
        raise HTTPException(status_code=400, detail="Text exceeds 2000 character limit.")

    logger.info("STEP 0: [OK] Input valid - %d chars after strip", len(text))
    logger.info("STEP 0: Text preview = %s...", text[:80].replace("\n", " "))

    # ------------------------------------------------------------------
    # Step 1: Emotion detection
    # ------------------------------------------------------------------
    logger.info("STEP 1: Starting emotion detection via HuggingFace model")
    step1_start = time.time()

    try:
        emotion_result = detect_emotion(text)
        step1_elapsed = time.time() - step1_start

        logger.info(
            "STEP 1: [OK] Emotion detection completed in %.3fs", step1_elapsed
        )
        logger.info(
            "STEP 1: Dominant emotion    = %s", emotion_result["dominant_emotion"]
        )
        logger.info(
            "STEP 1: Granular label      = %s", emotion_result["granular_label"]
        )
        logger.info(
            "STEP 1: Coarse category     = %s", emotion_result["category"]
        )
        logger.info(
            "STEP 1: Raw model score     = %.4f", emotion_result["raw_intensity"]
        )
        logger.info(
            "STEP 1: Punctuation boost   = %.4f", emotion_result["punctuation_boost"]
        )
        logger.info(
            "STEP 1: Combined intensity  = %.4f", emotion_result["intensity"]
        )
        logger.info("STEP 1: All 7 emotion scores:")
        for emotion, score in sorted(
            emotion_result["fine_emotions"].items(), key=lambda x: -x[1]
        ):
            logger.info("STEP 1:   %-12s = %.4f", emotion, score)

    except Exception as exc:
        step1_elapsed = time.time() - step1_start
        logger.error(
            "STEP 1: [FAIL] detect_emotion() raised after %.3fs - %s", step1_elapsed, exc
        )
        raise HTTPException(
            status_code=500, detail=f"Emotion detection failed: {exc}"
        )

    # ------------------------------------------------------------------
    # Step 2: Voice parameter computation (weighted parametric mapping)
    # ------------------------------------------------------------------
    logger.info("STEP 2: Computing voice parameters via weighted emotion aggregation")
    step2_start = time.time()

    try:
        voice_params = compute_voice_parameters(
            fine_emotions=emotion_result["fine_emotions"],
            intensity=emotion_result["intensity"],
        )
        step2_elapsed = time.time() - step2_start

        logger.info(
            "STEP 2: [OK] Voice parameter mapping completed in %.3fs", step2_elapsed
        )
        logger.info(
            "STEP 2: Pitch           = %s  (%+.3f semitones)",
            voice_params["pitch_percent"],
            voice_params["pitch_semitones"],
        )
        logger.info(
            "STEP 2: Rate            = %s  (ratio %.3f x)",
            voice_params["rate_tag"],
            voice_params["rate_ratio"],
        )
        logger.info(
            "STEP 2: Volume          = %s  (%.3f dB)",
            voice_params["ssml_volume"],
            voice_params["volume_db"],
        )

    except Exception as exc:
        step2_elapsed = time.time() - step2_start
        logger.error(
            "STEP 2: [FAIL] compute_voice_parameters() raised after %.3fs - %s",
            step2_elapsed, exc,
        )
        raise HTTPException(
            status_code=500, detail=f"Voice parameter mapping failed: {exc}"
        )

    # ------------------------------------------------------------------
    # Step 3: SSML generation
    # ------------------------------------------------------------------
    logger.info("STEP 3: Generating SSML markup from voice parameters")
    step3_start = time.time()

    try:
        ssml = generate_ssml(text, voice_params)
        step3_elapsed = time.time() - step3_start

        logger.info(
            "STEP 3: [OK] SSML generated in %.3fs - total length = %d chars",
            step3_elapsed, len(ssml),
        )
        logger.info("STEP 3: SSML preview = %s...", ssml[:120].replace("\n", " "))

    except Exception as exc:
        step3_elapsed = time.time() - step3_start
        logger.error(
            "STEP 3: [FAIL] generate_ssml() raised after %.3fs - %s", step3_elapsed, exc
        )
        raise HTTPException(
            status_code=500, detail=f"SSML generation failed: {exc}"
        )

    # ------------------------------------------------------------------
    # Step 4: TTS synthesis with automatic engine fallback
    # ------------------------------------------------------------------
    voice_id  = body.voice_id or None   # let tts_engine auto-select per emotion
    timestamp = f"audio_{int(time.time() * 1000)}"

    logger.info("STEP 4: Starting TTS synthesis")
    logger.info("STEP 4: User voice override = %s", voice_id or "(auto — emotion-matched)")
    logger.info("STEP 4: Output filename     = %s.mp3", timestamp)
    logger.info("STEP 4: Engine priority     = ElevenLabs -> gTTS -> pyttsx3")
    logger.info(
        "STEP 4: Expected voice character for emotion=%s = %s",
        emotion_result["dominant_emotion"],
        {
            "joy": "Rachel", "surprise": "Rachel", "anger": "Adam",
            "disgust": "Arnold", "fear": "Antoni", "sadness": "Bella",
            "neutral": "Sam",
        }.get(emotion_result["dominant_emotion"], "Sam"),
    )

    step4_start = time.time()

    try:
        tts_result = synthesize_speech(
            text=text,
            voice_params=voice_params,
            emotion_result=emotion_result,
            voice_id=voice_id,
            filename=timestamp,
        )
        step4_elapsed = time.time() - step4_start

        logger.info(
            "STEP 4: [OK] TTS synthesis completed in %.3fs", step4_elapsed
        )
        logger.info("STEP 4: Engine used     = %s", tts_result["engine_used"])
        logger.info("STEP 4: Voice ID used   = %s", tts_result["voice_id"])
        logger.info("STEP 4: Voice name      = %s", tts_result.get("voice_name", "unknown"))
        logger.info("STEP 4: Shaped text     = %s", tts_result.get("shaped_text", text)[:80])
        logger.info("STEP 4: Audio saved to  = %s", tts_result["audio_path"])

        audio_path = tts_result["audio_path"]
        audio_size = Path(audio_path).stat().st_size if Path(audio_path).exists() else 0
        logger.info("STEP 4: Audio file size = %d bytes", audio_size)

    except Exception as exc:
        step4_elapsed = time.time() - step4_start
        logger.error(
            "STEP 4: [FAIL] synthesize_speech() raised after %.3fs - %s",
            step4_elapsed, exc,
        )
        raise HTTPException(
            status_code=500, detail=f"TTS synthesis failed: {exc}"
        )

    # ------------------------------------------------------------------
    # Step 5: Build response
    # ------------------------------------------------------------------
    logger.info("STEP 5: Building API response object")

    audio_filename = Path(tts_result["audio_path"]).name
    audio_url      = f"/outputs/{audio_filename}"

    pipeline_elapsed = time.time() - pipeline_start
    logger.info("STEP 5: [OK] Response built successfully")
    logger.info("STEP 5: Audio URL       = %s", audio_url)
    logger.info(
        "PIPELINE: Total synthesis time = %.3fs", pipeline_elapsed
    )
    logger.info("=" * 60)

    logger.info("STEP 5: Voice character   = %s", tts_result.get("voice_name", "unknown"))
    logger.info("STEP 5: Voice selection   = %s", tts_result.get("voice_selection", {}))
    logger.info("STEP 5: Shaped text       = %s", tts_result.get("shaped_text", "")[:80])

    return SynthesizeResponse(
        text=text,
        dominant_emotion=emotion_result["dominant_emotion"],
        granular_label=emotion_result["granular_label"],
        category=emotion_result["category"],
        intensity=emotion_result["intensity"],
        raw_intensity=emotion_result["raw_intensity"],
        punctuation_boost=emotion_result["punctuation_boost"],
        fine_emotions=emotion_result["fine_emotions"],
        voice_params=voice_params,
        ssml=ssml,
        audio_url=audio_url,
        engine_used=tts_result["engine_used"],
        voice_id=tts_result["voice_id"],
        voice_selection=tts_result.get("voice_selection", {}),
        shaped_text=tts_result.get("shaped_text", text),
    )



@app.get("/voices")
async def voices():
    logger.info("GET /voices - fetching available TTS voices")
    try:
        voice_list = get_available_voices()
        logger.info("GET /voices: [OK] Returning %d voice(s)", len(voice_list))
        return {"voices": voice_list}
    except Exception as exc:
        logger.error("GET /voices: [FAIL] get_available_voices() raised - %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch voices.")


@app.get("/health")
async def health():
    logger.info("GET /health - health check requested")
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    el_status = "configured" if api_key else "not configured (offline fallback active)"
    logger.info("GET /health: [OK] ElevenLabs status = %s", el_status)
    return {
        "status":     "ok",
        "elevenlabs": el_status,
    }


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    logger.info("GET /audio/%s - audio file requested", filename)
    path = Path("outputs") / filename
    if not path.exists():
        logger.warning("GET /audio/%s: [FAIL] File not found at %s", filename, path)
        raise HTTPException(status_code=404, detail="Audio file not found.")
    logger.info(
        "GET /audio/%s: [OK] Serving file (%d bytes)", filename, path.stat().st_size
    )
    return FileResponse(path, media_type="audio/mpeg")
