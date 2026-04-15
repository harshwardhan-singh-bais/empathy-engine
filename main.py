import os
import time
import logging
import re
from pathlib import Path

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from emotion_engine import detect_emotion
from mapping_engine import compute_voice_parameters, generate_ssml
from tts_engine import synthesize_speech, get_available_voices

load_dotenv(".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("empathy-engine")

Path("outputs").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

app = FastAPI(
    title="Empathy Engine",
    description="Emotionally-aware Text-to-Speech synthesis service",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
templates = Jinja2Templates(directory="templates")


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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    voices = get_available_voices()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "voices": voices},
    )


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(body: SynthesizeRequest):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text exceeds 2000 character limit.")

    logger.info("Synthesize request → %d chars", len(text))

    # Step 1: Emotion detection
    emotion_result = detect_emotion(text)
    logger.info(
        "Emotion → %s (%s) | intensity=%.3f",
        emotion_result["dominant_emotion"],
        emotion_result["category"],
        emotion_result["intensity"],
    )

    # Step 2: Voice parameter computation (weighted + non-linear scaling)
    voice_params = compute_voice_parameters(
        fine_emotions=emotion_result["fine_emotions"],
        intensity=emotion_result["intensity"],
    )

    # Step 3: SSML generation
    ssml = generate_ssml(text, voice_params)

    # Step 4: TTS synthesis with fallback chain
    voice_id = body.voice_id or os.getenv("DEFAULT_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
    timestamp = f"audio_{int(time.time() * 1000)}"
    tts_result = synthesize_speech(
        text=text,
        voice_params=voice_params,
        voice_id=voice_id,
        filename=timestamp,
    )

    audio_path = tts_result["audio_path"]
    audio_filename = Path(audio_path).name
    audio_url = f"/outputs/{audio_filename}"

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
    )


@app.get("/voices")
async def voices():
    return {"voices": get_available_voices()}


@app.get("/health")
async def health():
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    return {
        "status":    "ok",
        "elevenlabs": "configured" if api_key else "not configured (offline mode)",
    }


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    path = Path("outputs") / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(path, media_type="audio/mpeg")
