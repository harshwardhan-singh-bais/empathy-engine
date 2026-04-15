import os
import re
import logging
import time
from pathlib import Path
import requests
from voice_selector import select_voice_profile, describe_selection

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"


def _shape_text_for_emotion(text: str, dominant_emotion: str, intensity: float) -> str:
    logger.info("TEXT SHAPING: emotion=%s, intensity=%.3f", dominant_emotion, intensity)
    shaped = text.strip()

    if dominant_emotion == "joy":
        shaped = re.sub(r'\.\s+', '! ', shaped)
        if not shaped.endswith("!"):
            shaped = shaped.rstrip(".?") + "!"

    elif dominant_emotion == "surprise":
        words = shaped.split()
        if len(words) > 4:
            mid = len(words) // 2
            words.insert(mid, "—")
            shaped = " ".join(words)
        if not shaped.endswith(("!", "?")):
            shaped = shaped.rstrip(".") + "!"

    elif dominant_emotion == "anger":
        if intensity > 0.70:
            if not shaped.lower().startswith("listen"):
                shaped = "Listen. " + shaped
        STRONG_WORDS = {
            "never", "always", "terrible", "awful", "hate",
            "wrong", "unacceptable", "disgusting", "ridiculous", "worst",
        }
        words = shaped.split()
        shaped = " ".join(
            w.upper() if w.lower().strip(".,!?;:") in STRONG_WORDS else w
            for w in words
        )

    elif dominant_emotion == "disgust":
        openers = ("honestly", "frankly", "seriously", "unbelievable", "pathetic")
        if not shaped.lower().startswith(openers):
            shaped = "Honestly... " + shaped

    elif dominant_emotion == "fear":
        words = shaped.split()
        if len(words) > 5:
            new_words = []
            for i, w in enumerate(words):
                new_words.append(w)
                if i > 0 and i % 5 == 0 and i < len(words) - 1:
                    new_words.append("...")
            shaped = " ".join(new_words)

    elif dominant_emotion == "sadness":
        shaped = re.sub(r',\s+', '... ', shaped)
        shaped = re.sub(r'\.\s+', '... ', shaped)
        if not shaped.endswith(("...", ".")):
            shaped = shaped + "..."

    return shaped


def _synthesize_elevenlabs(
    text: str,
    voice_profile,
    api_key: str,
    output_path: Path,
    dominant_emotion: str,
    intensity: float,
) -> dict:
    logger.info("ELEVENLABS: Starting synthesis")
    shaped_text = _shape_text_for_emotion(text, dominant_emotion, intensity)

    payload = {
        "text":     shaped_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability":         voice_profile.stability,
            "similarity_boost":  voice_profile.similarity_boost,
            "style":             voice_profile.style,
            "use_speaker_boost": voice_profile.use_speaker_boost,
        },
    }

    headers = {
        "xi-api-key":   api_key,
        "Content-Type": "application/json",
        "Accept":       "audio/mpeg",
    }

    response = requests.post(
        f"{ELEVENLABS_API_URL}/text-to-speech/{voice_profile.voice_id}",
        json=payload,
        headers=headers,
        timeout=30,
    )

    if not response.ok:
        logger.error(
            "ELEVENLABS: [FAIL] HTTP %d - %s",
            response.status_code, response.text[:300],
        )
    response.raise_for_status()

    output_path.write_bytes(response.content)
    return {
        "audio_path":       str(output_path),
        "engine_used":      "elevenlabs",
        "voice_id":         voice_profile.voice_id,
        "voice_name":       voice_profile.voice_name,
        "voice_selection":  describe_selection(voice_profile),
        "shaped_text":      shaped_text,
    }


def _synthesize_gtts(
    text: str,
    dominant_emotion: str,
    intensity: float,
    output_path: Path,
) -> dict:
    from gtts import gTTS
    shaped_text = _shape_text_for_emotion(text, dominant_emotion, intensity)

    low_energy    = {"sadness", "neutral", "fear"}
    slow          = dominant_emotion in low_energy

    tts = gTTS(text=shaped_text, lang="en", slow=slow)
    tts.save(str(output_path))

    return {
        "audio_path":      str(output_path),
        "engine_used":     "gtts",
        "voice_id":        "gtts-en",
        "voice_name":      "gTTS English",
        "voice_selection": {"note": "gTTS fallback — no voice_settings support"},
        "shaped_text":     shaped_text,
    }


def _synthesize_pyttsx3(
    text: str,
    dominant_emotion: str,
    intensity: float,
    voice_params: dict,
    output_path: Path,
) -> dict:
    import pyttsx3
    engine = pyttsx3.init()

    rate_ratio = voice_params.get("rate_ratio", 1.0)
    wpm        = int(200 * rate_ratio)
    engine.setProperty("rate", wpm)

    volume_db  = voice_params.get("volume_db", 0.0)
    vol        = max(0.2, min(1.0, 0.75 + (volume_db / 20.0)))
    engine.setProperty("volume", vol)

    shaped_text = _shape_text_for_emotion(text, dominant_emotion, intensity)

    wav_path = output_path.with_suffix(".wav")
    engine.save_to_file(shaped_text, str(wav_path))
    engine.runAndWait()
    engine.stop()

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(str(wav_path))
        audio.export(str(output_path), format="mp3")
        wav_path.unlink(missing_ok=True)
        final_path = str(output_path)
    except Exception:
        final_path = str(wav_path)

    return {
        "audio_path":      final_path,
        "engine_used":     "pyttsx3",
        "voice_id":        "pyttsx3-local",
        "voice_name":      "pyttsx3 Offline",
        "voice_selection": {"note": "pyttsx3 offline — no voice_settings support"},
        "shaped_text":     shaped_text,
    }


def synthesize_speech(
    text: str,
    voice_params: dict,
    emotion_result: dict,
    voice_id: str = None,
    filename: str = None,
) -> dict:
    api_key          = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if api_key == "your_elevenlabs_api_key_here":
        api_key = ""
        
    dominant_emotion = emotion_result.get("dominant_emotion", "neutral")
    fine_emotions    = emotion_result.get("fine_emotions", {})
    intensity        = emotion_result.get("intensity", 0.5)
    timestamp        = filename or f"audio_{int(time.time())}"
    output_path      = OUTPUTS_DIR / f"{timestamp}.mp3"

    voice_profile = select_voice_profile(
        dominant_emotion=dominant_emotion,
        fine_emotions=fine_emotions,
        intensity=intensity,
        voice_id_override=voice_id or "",
    )

    if api_key:
        try:
            return _synthesize_elevenlabs(
                text=text,
                voice_profile=voice_profile,
                api_key=api_key,
                output_path=output_path,
                dominant_emotion=dominant_emotion,
                intensity=intensity,
            )
        except Exception as el_err:
            logger.warning("SYNTHESIZE: ElevenLabs failed (%s) — falling back to gTTS", el_err)

    try:
        return _synthesize_gtts(text, dominant_emotion, intensity, output_path)
    except Exception as gtts_err:
        logger.warning("SYNTHESIZE: gTTS failed (%s) — falling back to pyttsx3", gtts_err)

    return _synthesize_pyttsx3(
        text=text,
        dominant_emotion=dominant_emotion,
        intensity=intensity,
        voice_params=voice_params,
        output_path=output_path,
    )


def get_available_voices() -> list[dict]:
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if api_key == "your_elevenlabs_api_key_here":
        api_key = ""
        
    if not api_key:
        return [
            {"voice_id": "gtts-en",       "name": "gTTS English (No API key)"},
            {"voice_id": "pyttsx3-local", "name": "pyttsx3 Offline"},
        ]

    try:
        headers  = {"xi-api-key": api_key}
        response = requests.get(f"{ELEVENLABS_API_URL}/voices", headers=headers, timeout=10)
        response.raise_for_status()
        voices = response.json().get("voices", [])
        return [{"voice_id": v["voice_id"], "name": v["name"]} for v in voices]
    except Exception:
        return [
            {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel — Joy / Surprise"},
            {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam — Anger"},
            {"voice_id": "VR6AewLTigWG4xSOukaG", "name": "Arnold — Disgust"},
            {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni — Fear"},
            {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella — Sadness"},
            {"voice_id": "yoZ06aMxZJJ28mfd3POQ", "name": "Sam — Neutral"},
        ]
