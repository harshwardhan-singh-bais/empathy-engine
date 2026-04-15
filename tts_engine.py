import os
import io
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE_ID   = "EXAVITQu4vr4xnSDxMaL"   # "Bella" — warm, expressive


# ---------------------------------------------------------------------------
# ElevenLabs TTS
# ---------------------------------------------------------------------------

def _synthesize_elevenlabs(
    text: str,
    voice_params: dict,
    voice_id: str,
    api_key: str,
    output_path: Path,
) -> Path:
    """
    Call ElevenLabs TTS via HTTP.
    Maps our internal voice_params → ElevenLabs voice_settings.
    Rate/pitch/volume are baked into SSML for maximum compatibility.
    """
    # ElevenLabs stability: lower = more expressive (inverse of neutral)
    # Map intensity-driven volume_db to stability [0.3 → 0.7]
    volume_db = voice_params.get("volume_db", 0.0)
    stability = round(max(0.25, min(0.75, 0.50 - (volume_db / 18.0))), 2)

    # Similarity boost: keep voice recognisable
    similarity_boost = 0.80

    # Style exaggeration scales with absolute pitch change
    pitch_st       = abs(voice_params.get("pitch_semitones", 0.0))
    style          = round(min(1.0, pitch_st / 8.0), 2)

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability":        stability,
            "similarity_boost": similarity_boost,
            "style":            style,
            "use_speaker_boost": True,
        },
    }

    headers = {
        "xi-api-key":   api_key,
        "Content-Type": "application/json",
        "Accept":       "audio/mpeg",
    }

    response = requests.post(
        f"{ELEVENLABS_API_URL}/text-to-speech/{voice_id}",
        json=payload,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    output_path.write_bytes(response.content)
    logger.info("ElevenLabs audio saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# gTTS Fallback
# ---------------------------------------------------------------------------

def _synthesize_gtts(text: str, voice_params: dict, output_path: Path) -> Path:
    """
    gTTS fallback — no pitch/rate control, but produces real TTS output.
    Uses MP3 output directly.
    """
    try:
        from gtts import gTTS
        rate_ratio = voice_params.get("rate_ratio", 1.0)
        slow = rate_ratio < 0.85
        tts = gTTS(text=text, lang="en", slow=slow)
        tts.save(str(output_path))
        logger.info("gTTS audio saved → %s", output_path)
        return output_path
    except Exception as e:
        logger.warning("gTTS failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# pyttsx3 Offline Fallback
# ---------------------------------------------------------------------------

def _synthesize_pyttsx3(text: str, voice_params: dict, output_path: Path) -> Path:
    """
    pyttsx3 offline TTS — full pitch + rate control via engine properties.
    Saves to WAV, then converts to MP3 if pydub is available.
    """
    import pyttsx3

    engine = pyttsx3.init()

    # Rate: words per minute; default ~200
    rate_ratio = voice_params.get("rate_ratio", 1.0)
    engine.setProperty("rate", int(200 * rate_ratio))

    # Volume: [0.0, 1.0]; map from dB shift
    volume_db = voice_params.get("volume_db", 0.0)
    vol = max(0.2, min(1.0, 0.75 + (volume_db / 20.0)))
    engine.setProperty("volume", vol)

    wav_path = output_path.with_suffix(".wav")
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()
    engine.stop()

    # Convert WAV → MP3 if pydub available
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(str(wav_path))
        audio.export(str(output_path), format="mp3")
        wav_path.unlink(missing_ok=True)
        logger.info("pyttsx3 audio saved → %s", output_path)
        return output_path
    except Exception:
        logger.info("pyttsx3 WAV saved → %s (pydub not available for conversion)", wav_path)
        return wav_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_speech(
    text: str,
    voice_params: dict,
    voice_id: str = DEFAULT_VOICE_ID,
    filename: str = None,
) -> dict:
    """
    Synthesize speech with automatic fallback: ElevenLabs → gTTS → pyttsx3.

    Returns:
        {
          "audio_path": str,
          "engine_used": "elevenlabs" | "gtts" | "pyttsx3",
          "voice_id": str,
        }
    """
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    timestamp = filename or f"audio_{int(time.time())}"
    output_path = OUTPUTS_DIR / f"{timestamp}.mp3"

    # 1. ElevenLabs (primary)
    if api_key:
        try:
            path = _synthesize_elevenlabs(text, voice_params, voice_id, api_key, output_path)
            return {"audio_path": str(path), "engine_used": "elevenlabs", "voice_id": voice_id}
        except Exception as e:
            logger.warning("ElevenLabs failed (%s), falling back to gTTS.", e)

    # 2. gTTS (first fallback)
    try:
        path = _synthesize_gtts(text, voice_params, output_path)
        return {"audio_path": str(path), "engine_used": "gtts", "voice_id": "gtts-en"}
    except Exception as e:
        logger.warning("gTTS failed (%s), falling back to pyttsx3.", e)

    # 3. pyttsx3 (offline fallback)
    path = _synthesize_pyttsx3(text, voice_params, output_path)
    return {"audio_path": str(path), "engine_used": "pyttsx3", "voice_id": "pyttsx3-local"}


def get_available_voices() -> list[dict]:
    """Return list of ElevenLabs voices, or a static fallback list."""
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        return [
            {"voice_id": "gtts-en",      "name": "gTTS English (Offline Fallback)"},
            {"voice_id": "pyttsx3-local","name": "pyttsx3 Local (Fully Offline)"},
        ]
    try:
        headers = {"xi-api-key": api_key}
        response = requests.get(f"{ELEVENLABS_API_URL}/voices", headers=headers, timeout=10)
        response.raise_for_status()
        voices = response.json().get("voices", [])
        return [{"voice_id": v["voice_id"], "name": v["name"]} for v in voices]
    except Exception as e:
        logger.warning("Could not fetch ElevenLabs voices: %s", e)
        return [{"voice_id": DEFAULT_VOICE_ID, "name": "Bella (ElevenLabs)"}]
