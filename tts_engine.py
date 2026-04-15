"""
tts_engine.py

Handles speech synthesis with a 3-engine fallback chain:
  ElevenLabs (primary) -> gTTS (fallback) -> pyttsx3 (offline fallback)

Voice selection is fully delegated to voice_selector.py.
This module only handles:
  - Text shaping (emotional punctuation/rhythm)
  - API calls
  - File I/O
  - Fallback logic
"""

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


# ---------------------------------------------------------------------------
# Text shaping
#
# ElevenLabs is very sensitive to punctuation cues in the text itself.
# Ellipses, dashes, exclamation marks — all affect pacing and rhythm.
# This is the second layer of expressiveness after voice_settings.
# ---------------------------------------------------------------------------
def _shape_text_for_emotion(text: str, dominant_emotion: str, intensity: float) -> str:
    """
    Injects emotion-appropriate punctuation and rhythm cues into the text.

    ElevenLabs interprets these as pacing signals:
      - "..."   → pause, hesitation, slowdown
      - "—"     → dramatic mid-sentence break
      - "!"     → energy, emphasis
      - ALL CAPS → forceful emphasis on that word
    """
    logger.info("TEXT SHAPING: emotion=%s, intensity=%.3f", dominant_emotion, intensity)
    shaped = text.strip()

    if dominant_emotion == "joy":
        # Convert sentence endings to enthusiastic rhythm
        shaped = re.sub(r'\.\s+', '! ', shaped)
        if not shaped.endswith("!"):
            shaped = shaped.rstrip(".?") + "!"
        logger.info("TEXT SHAPING: Joy — added enthusiasm punctuation")

    elif dominant_emotion == "surprise":
        # Inject a dramatic mid-sentence dash for impact
        words = shaped.split()
        if len(words) > 4:
            mid = len(words) // 2
            words.insert(mid, "—")
            shaped = " ".join(words)
        if not shaped.endswith(("!", "?")):
            shaped = shaped.rstrip(".") + "!"
        logger.info("TEXT SHAPING: Surprise — injected mid-sentence dash break")

    elif dominant_emotion == "anger":
        # Forceful framing at high intensity + caps for strong words
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
        logger.info("TEXT SHAPING: Anger — added framing and emphasis caps")

    elif dominant_emotion == "disgust":
        # Contemptuous opener if not already present
        openers = ("honestly", "frankly", "seriously", "unbelievable", "pathetic")
        if not shaped.lower().startswith(openers):
            shaped = "Honestly... " + shaped
        logger.info("TEXT SHAPING: Disgust — added contemptuous opener")

    elif dominant_emotion == "fear":
        # Hesitation pauses every ~5 words
        words = shaped.split()
        if len(words) > 5:
            new_words = []
            for i, w in enumerate(words):
                new_words.append(w)
                if i > 0 and i % 5 == 0 and i < len(words) - 1:
                    new_words.append("...")
            shaped = " ".join(new_words)
        logger.info("TEXT SHAPING: Fear — inserted hesitation ellipses")

    elif dominant_emotion == "sadness":
        # Heavy comma pauses and trailing ellipsis
        shaped = re.sub(r',\s+', '... ', shaped)
        shaped = re.sub(r'\.\s+', '... ', shaped)
        if not shaped.endswith(("...", ".")):
            shaped = shaped + "..."
        logger.info("TEXT SHAPING: Sadness — inserted pause ellipses for slow delivery")

    elif dominant_emotion == "neutral":
        logger.info("TEXT SHAPING: Neutral — no reshaping applied")

    logger.info("TEXT SHAPING: Original = '%s'", text[:70])
    logger.info("TEXT SHAPING: Shaped   = '%s'", shaped[:80])
    return shaped


# ---------------------------------------------------------------------------
# ElevenLabs primary synthesis
# ---------------------------------------------------------------------------
def _synthesize_elevenlabs(
    text: str,
    voice_profile,
    api_key: str,
    output_path: Path,
    dominant_emotion: str,
    intensity: float,
) -> dict:
    """
    Calls ElevenLabs API using the voice profile from voice_selector.

    Uses voice_settings (stability/style/similarity_boost) exclusively.
    SSML prosody tags are NOT used — ElevenLabs largely ignores them.
    Text shaping provides the rhythm/pacing control instead.
    """
    logger.info("ELEVENLABS: Starting synthesis")
    logger.info("ELEVENLABS: Voice     = %s (%s)", voice_profile.voice_name, voice_profile.voice_id)
    logger.info("ELEVENLABS: Stability = %.3f", voice_profile.stability)
    logger.info("ELEVENLABS: Style     = %.3f", voice_profile.style)
    logger.info("ELEVENLABS: Similarity= %.3f", voice_profile.similarity_boost)
    logger.info("ELEVENLABS: SpeakerBo = %s",   voice_profile.use_speaker_boost)

    # Shape text before sending
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

    logger.info("ELEVENLABS: POST /text-to-speech/%s", voice_profile.voice_id)
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
    logger.info(
        "ELEVENLABS: [OK] Audio saved to %s (%d bytes)",
        output_path, len(response.content),
    )

    return {
        "audio_path":       str(output_path),
        "engine_used":      "elevenlabs",
        "voice_id":         voice_profile.voice_id,
        "voice_name":       voice_profile.voice_name,
        "voice_selection":  describe_selection(voice_profile),
        "shaped_text":      shaped_text,
    }


# ---------------------------------------------------------------------------
# gTTS fallback
# ---------------------------------------------------------------------------
def _synthesize_gtts(
    text: str,
    dominant_emotion: str,
    intensity: float,
    output_path: Path,
) -> dict:
    """
    gTTS fallback. Shapes text identically to ElevenLabs path.
    Rate is approximate via slow=True for low-energy emotions.
    """
    logger.info("GTTS: Starting fallback synthesis")
    from gtts import gTTS

    shaped_text = _shape_text_for_emotion(text, dominant_emotion, intensity)

    low_energy    = {"sadness", "neutral", "fear"}
    slow          = dominant_emotion in low_energy
    logger.info("GTTS: slow=%s for emotion=%s", slow, dominant_emotion)

    tts = gTTS(text=shaped_text, lang="en", slow=slow)
    tts.save(str(output_path))
    logger.info("GTTS: [OK] Audio saved to %s", output_path)

    return {
        "audio_path":      str(output_path),
        "engine_used":     "gtts",
        "voice_id":        "gtts-en",
        "voice_name":      "gTTS English",
        "voice_selection": {"note": "gTTS fallback — no voice_settings support"},
        "shaped_text":     shaped_text,
    }


# ---------------------------------------------------------------------------
# pyttsx3 offline fallback
# ---------------------------------------------------------------------------
def _synthesize_pyttsx3(
    text: str,
    dominant_emotion: str,
    intensity: float,
    voice_params: dict,
    output_path: Path,
) -> dict:
    """
    pyttsx3 fully offline fallback.
    Controls rate and volume from voice_params.
    """
    logger.info("PYTTSX3: Starting offline fallback synthesis")
    import pyttsx3

    engine = pyttsx3.init()

    rate_ratio = voice_params.get("rate_ratio", 1.0)
    wpm        = int(200 * rate_ratio)
    engine.setProperty("rate", wpm)
    logger.info("PYTTSX3: Rate set to %d wpm (ratio %.2f)", wpm, rate_ratio)

    volume_db  = voice_params.get("volume_db", 0.0)
    vol        = max(0.2, min(1.0, 0.75 + (volume_db / 20.0)))
    engine.setProperty("volume", vol)
    logger.info("PYTTSX3: Volume set to %.2f", vol)

    shaped_text = _shape_text_for_emotion(text, dominant_emotion, intensity)

    wav_path = output_path.with_suffix(".wav")
    engine.save_to_file(shaped_text, str(wav_path))
    engine.runAndWait()
    engine.stop()
    logger.info("PYTTSX3: WAV saved to %s", wav_path)

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(str(wav_path))
        audio.export(str(output_path), format="mp3")
        wav_path.unlink(missing_ok=True)
        logger.info("PYTTSX3: [OK] Converted WAV -> MP3 at %s", output_path)
        final_path = str(output_path)
    except Exception as conv_err:
        logger.warning("PYTTSX3: pydub conversion failed (%s) — keeping WAV", conv_err)
        final_path = str(wav_path)

    return {
        "audio_path":      final_path,
        "engine_used":     "pyttsx3",
        "voice_id":        "pyttsx3-local",
        "voice_name":      "pyttsx3 Offline",
        "voice_selection": {"note": "pyttsx3 offline — no voice_settings support"},
        "shaped_text":     shaped_text,
    }


# ---------------------------------------------------------------------------
# Public API — main entry point
# ---------------------------------------------------------------------------
def synthesize_speech(
    text: str,
    voice_params: dict,
    emotion_result: dict,
    voice_id: str = None,
    filename: str = None,
) -> dict:
    """
    Main synthesis entry point.

    Pipeline:
      1. voice_selector selects voice character + settings (deterministic)
      2. text shaping injects emotion rhythm into the text
      3. ElevenLabs API called with voice_settings (not SSML)
      4. Auto-fallback: gTTS -> pyttsx3 if ElevenLabs unavailable

    Args:
        text:           Raw input text
        voice_params:   Computed params from mapping_engine (used by pyttsx3)
        emotion_result: Full result dict from emotion_engine.detect_emotion()
        voice_id:       Optional manual override (system ignores if None)
        filename:       Output file stem (without extension)
    """
    api_key          = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if api_key == "your_elevenlabs_api_key_here":
        api_key = ""
        
    dominant_emotion = emotion_result.get("dominant_emotion", "neutral")
    fine_emotions    = emotion_result.get("fine_emotions", {})
    intensity        = emotion_result.get("intensity", 0.5)
    timestamp        = filename or f"audio_{int(time.time())}"
    output_path      = OUTPUTS_DIR / f"{timestamp}.mp3"

    logger.info("SYNTHESIZE: dominant_emotion = %s", dominant_emotion)
    logger.info("SYNTHESIZE: intensity        = %.4f", intensity)
    logger.info("SYNTHESIZE: output           = %s", output_path)
    logger.info("SYNTHESIZE: api_key present  = %s", bool(api_key))

    # ------------------------------------------------------------------
    # Run voice selector — determines voice + settings from emotion
    # ------------------------------------------------------------------
    logger.info("SYNTHESIZE: Invoking voice_selector for profile selection")
    voice_profile = select_voice_profile(
        dominant_emotion=dominant_emotion,
        fine_emotions=fine_emotions,
        intensity=intensity,
        voice_id_override=voice_id or "",
    )
    logger.info(
        "SYNTHESIZE: Voice profile ready — %s (stability=%.3f, style=%.3f)",
        voice_profile.voice_name,
        voice_profile.stability,
        voice_profile.style,
    )

    # ------------------------------------------------------------------
    # Engine 1: ElevenLabs (primary)
    # ------------------------------------------------------------------
    if api_key:
        logger.info("SYNTHESIZE: Attempting ElevenLabs engine")
        try:
            result = _synthesize_elevenlabs(
                text=text,
                voice_profile=voice_profile,
                api_key=api_key,
                output_path=output_path,
                dominant_emotion=dominant_emotion,
                intensity=intensity,
            )
            logger.info("SYNTHESIZE: [OK] ElevenLabs synthesis successful")
            return result
        except Exception as el_err:
            logger.warning(
                "SYNTHESIZE: ElevenLabs failed (%s) — falling back to gTTS", el_err
            )
    else:
        logger.warning(
            "SYNTHESIZE: No ELEVENLABS_API_KEY — skipping to gTTS fallback"
        )

    # ------------------------------------------------------------------
    # Engine 2: gTTS fallback
    # ------------------------------------------------------------------
    logger.info("SYNTHESIZE: Attempting gTTS fallback engine")
    try:
        result = _synthesize_gtts(text, dominant_emotion, intensity, output_path)
        logger.info("SYNTHESIZE: [OK] gTTS synthesis successful")
        return result
    except Exception as gtts_err:
        logger.warning(
            "SYNTHESIZE: gTTS failed (%s) — falling back to pyttsx3", gtts_err
        )

    # ------------------------------------------------------------------
    # Engine 3: pyttsx3 fully offline fallback
    # ------------------------------------------------------------------
    logger.info("SYNTHESIZE: Attempting pyttsx3 offline fallback engine")
    result = _synthesize_pyttsx3(
        text=text,
        dominant_emotion=dominant_emotion,
        intensity=intensity,
        voice_params=voice_params,
        output_path=output_path,
    )
    logger.info("SYNTHESIZE: [OK] pyttsx3 synthesis successful")
    return result


# ---------------------------------------------------------------------------
# Voice list for UI dropdown (or just show "auto-selected")
# ---------------------------------------------------------------------------
def get_available_voices() -> list[dict]:
    """
    Returns ElevenLabs voice list if API key is present and valid.
    Falls back to the curated emotion-matched roster.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if api_key == "your_elevenlabs_api_key_here":
        api_key = ""
        
    if not api_key:
        logger.info("VOICES: No API key — returning offline fallback list")
        return [
            {"voice_id": "gtts-en",       "name": "gTTS English (No API key)"},
            {"voice_id": "pyttsx3-local", "name": "pyttsx3 Offline"},
        ]

    logger.info("VOICES: Fetching voice list from ElevenLabs API")
    try:
        headers  = {"xi-api-key": api_key}
        response = requests.get(
            f"{ELEVENLABS_API_URL}/voices", headers=headers, timeout=10
        )
        response.raise_for_status()
        voices = response.json().get("voices", [])
        logger.info("VOICES: [OK] Retrieved %d voices", len(voices))
        return [{"voice_id": v["voice_id"], "name": v["name"]} for v in voices]
    except Exception as err:
        logger.warning("VOICES: Could not fetch ElevenLabs voices: %s", err)
        curated = [
            {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel — Joy / Surprise"},
            {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam — Anger"},
            {"voice_id": "VR6AewLTigWG4xSOukaG", "name": "Arnold — Disgust"},
            {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni — Fear"},
            {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella — Sadness"},
            {"voice_id": "yoZ06aMxZJJ28mfd3POQ", "name": "Sam — Neutral"},
        ]
        logger.info("VOICES: Returning curated emotion-matched roster (%d voices)", len(curated))
        return curated
