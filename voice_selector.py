import os
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SINGLE_VOICE_MODE = os.getenv("SINGLE_VOICE_MODE", "false").lower() == "true"
SINGLE_VOICE_ID   = os.getenv("SINGLE_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
SINGLE_VOICE_NAME = os.getenv("SINGLE_VOICE_NAME", "Bella")


@dataclass
class VoiceProfile:
    voice_id:          str
    voice_name:        str
    stability:         float
    similarity_boost:  float
    style:             float
    use_speaker_boost: bool
    selection_reason:  str
    emotion:           str
    intensity:         float
    mode:              str


VOICE_ROSTER = {
    "rachel": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "name":     "Rachel",
        "note":     "Bright, warm, energetic — suits joy and surprise",
    },
    "adam": {
        "voice_id": "pNInz6obpgDQGcFmaJgB",
        "name":     "Adam",
        "note":     "Deep, firm, authoritative — suits anger",
    },
    "bella": {
        "voice_id": "EXAVITQu4vr4xnSDxMaL",
        "name":     "Bella",
        "note":     "Soft, gentle, empathetic — suits sadness. Also used as single-voice default.",
    },
    "antoni": {
        "voice_id": "ErXwobaYiN019PkySvjV",
        "name":     "Antoni",
        "note":     "Warm but conveys tension and nervousness — suits fear",
    },
    "arnold": {
        "voice_id": "VR6AewLTigWG4xSOukaG",
        "name":     "Arnold",
        "note":     "Dry, heavy, low-energy — suits disgust",
    },
    "sam": {
        "voice_id": "yoZ06aMxZJJ28mfd3POQ",
        "name":     "Sam",
        "note":     "Balanced, clean, professional — suits neutral",
    },
}

EMOTION_TO_VOICE_KEY = {
    "joy":      "rachel",
    "surprise": "rachel",
    "anger":    "adam",
    "disgust":  "arnold",
    "fear":     "antoni",
    "sadness":  "bella",
    "neutral":  "sam",
}

BASE_VOICE_SETTINGS = {
    "joy": {
        "stability":         0.22,
        "similarity_boost":  0.78,
        "style":             0.82,
        "use_speaker_boost": True,
        "reason": "Low stability + high style = energetic, expressive, elated delivery",
    },
    "surprise": {
        "stability":         0.18,
        "similarity_boost":  0.76,
        "style":             0.78,
        "use_speaker_boost": True,
        "reason": "Very low stability = sudden expressive peaks for surprise",
    },
    "anger": {
        "stability":         0.15,
        "similarity_boost":  0.70,
        "style":             0.90,
        "use_speaker_boost": True,
        "reason": "Minimum stability + max style = forceful, aggressive, intense delivery",
    },
    "disgust": {
        "stability":         0.35,
        "similarity_boost":  0.75,
        "style":             0.65,
        "use_speaker_boost": True,
        "reason": "Moderate stability + moderate style = dry, contemptuous tone",
    },
    "fear": {
        "stability":         0.30,
        "similarity_boost":  0.80,
        "style":             0.58,
        "use_speaker_boost": True,
        "reason": "Moderate-low stability = hesitant, trembling, nervous quality",
    },
    "sadness": {
        "stability":         0.75,
        "similarity_boost":  0.88,
        "style":             0.15,
        "use_speaker_boost": False,
        "reason": "High stability + low style = slow, flat, heavy, quiet delivery",
    },
    "neutral": {
        "stability":         0.52,
        "similarity_boost":  0.82,
        "style":             0.28,
        "use_speaker_boost": False,
        "reason": "Balanced settings — clean, professional, no emotional lean",
    },
}

INTENSITY_SCALING_DIRECTION = {
    "joy":      {"stability_delta": -0.12, "style_delta": +0.15},
    "surprise": {"stability_delta": -0.10, "style_delta": +0.18},
    "anger":    {"stability_delta": -0.08, "style_delta": +0.10},
    "disgust":  {"stability_delta": -0.10, "style_delta": +0.12},
    "fear":     {"stability_delta": -0.12, "style_delta": +0.08},
    "sadness":  {"stability_delta": +0.10, "style_delta": -0.08},
    "neutral":  {"stability_delta": +0.05, "style_delta": -0.05},
}


def _apply_intensity_scaling(base: dict, emotion: str, intensity: float) -> tuple[float, float]:
    scale_factor = intensity ** 1.3
    direction    = INTENSITY_SCALING_DIRECTION.get(emotion, {"stability_delta": 0, "style_delta": 0})

    raw_stability = base["stability"] + (direction["stability_delta"] * scale_factor)
    raw_style     = base["style"]     + (direction["style_delta"]     * scale_factor)

    final_stability = round(max(0.05, min(1.0, raw_stability)), 3)
    final_style     = round(max(0.00, min(1.0, raw_style)),     3)

    return final_stability, final_style


def select_voice_profile(
    dominant_emotion: str,
    fine_emotions: dict,
    intensity: float,
    voice_id_override: str = None,
) -> VoiceProfile:
    logger.info("VOICE SELECTOR: Starting voice profile selection")
    logger.info("VOICE SELECTOR: Mode            = %s", "SINGLE_VOICE" if SINGLE_VOICE_MODE else "MULTI_VOICE")
    logger.info("VOICE SELECTOR: Input emotion   = %s", dominant_emotion)
    logger.info("VOICE SELECTOR: Input intensity = %.4f", intensity)

    if voice_id_override and voice_id_override not in ("gtts-en", "pyttsx3-local", ""):
        voice_id   = voice_id_override
        voice_name = "user-override"
        mode       = "user_override"
        logger.info("VOICE SELECTOR: User override applied   = %s", voice_id)

    elif SINGLE_VOICE_MODE:
        voice_id   = SINGLE_VOICE_ID
        voice_name = SINGLE_VOICE_NAME
        mode       = "single_voice"
        logger.info(
            "VOICE SELECTOR: Single-voice mode active — voice = %s (%s)",
            voice_name, voice_id,
        )
        logger.info(
            "VOICE SELECTOR: Emotion '%s' will be expressed via parameter modulation only",
            dominant_emotion,
        )

    else:
        voice_key  = EMOTION_TO_VOICE_KEY.get(dominant_emotion, "sam")
        voice_data = VOICE_ROSTER[voice_key]
        voice_id   = voice_data["voice_id"]
        voice_name = voice_data["name"]
        mode       = "multi_voice"
        logger.info(
            "VOICE SELECTOR: Multi-voice mode — emotion '%s' matched to %s (%s)",
            dominant_emotion, voice_name, voice_id,
        )
        logger.info("VOICE SELECTOR: Character note = %s", voice_data["note"])

    base   = BASE_VOICE_SETTINGS.get(dominant_emotion, BASE_VOICE_SETTINGS["neutral"])
    reason = base["reason"]

    logger.info("VOICE SELECTOR: Base settings (PRIMARY expression layer):")
    logger.info("VOICE SELECTOR:   stability       = %.3f", base["stability"])
    logger.info("VOICE SELECTOR:   style           = %.3f", base["style"])
    logger.info("VOICE SELECTOR:   similarity      = %.3f", base["similarity_boost"])
    logger.info("VOICE SELECTOR:   speaker_boost   = %s",   base["use_speaker_boost"])
    logger.info("VOICE SELECTOR:   reason          = %s",   reason)

    scaled_stability, scaled_style = _apply_intensity_scaling(base, dominant_emotion, intensity)
    scale_factor = round(intensity ** 1.3, 4)

    logger.info("VOICE SELECTOR: Intensity scaling (curve = intensity ^ 1.3):")
    logger.info("VOICE SELECTOR:   raw intensity   = %.4f", intensity)
    logger.info("VOICE SELECTOR:   scale_factor    = %.4f", scale_factor)
    logger.info(
        "VOICE SELECTOR:   stability  %.3f -> %.3f  (delta %+.3f)",
        base["stability"], scaled_stability, scaled_stability - base["stability"],
    )
    logger.info(
        "VOICE SELECTOR:   style      %.3f -> %.3f  (delta %+.3f)",
        base["style"], scaled_style, scaled_style - base["style"],
    )

    if SINGLE_VOICE_MODE:
        selection_reason = (
            f"Single-voice mode: {voice_name} used for all emotions. "
            f"Emotion '{dominant_emotion}' (intensity {intensity:.2f}) expressed via "
            f"parameter modulation — stability={scaled_stability:.3f}, style={scaled_style:.3f}. "
            f"{reason}"
        )
    else:
        selection_reason = (
            f"Multi-voice mode: '{dominant_emotion}' (intensity {intensity:.2f}) "
            f"matched to {voice_name}. {reason} "
            f"Voice character adds secondary timbre reinforcement."
        )

    profile = VoiceProfile(
        voice_id=voice_id,
        voice_name=voice_name,
        stability=scaled_stability,
        similarity_boost=base["similarity_boost"],
        style=scaled_style,
        use_speaker_boost=base["use_speaker_boost"],
        selection_reason=selection_reason,
        emotion=dominant_emotion,
        intensity=intensity,
        mode=mode,
    )

    logger.info("VOICE SELECTOR: Final profile:")
    logger.info("VOICE SELECTOR:   voice           = %s (%s)", profile.voice_name, profile.voice_id)
    logger.info("VOICE SELECTOR:   mode            = %s", profile.mode)
    logger.info("VOICE SELECTOR:   stability       = %.3f", profile.stability)
    logger.info("VOICE SELECTOR:   style           = %.3f", profile.style)
    logger.info("VOICE SELECTOR:   similarity      = %.3f", profile.similarity_boost)
    logger.info("VOICE SELECTOR:   speaker_boost   = %s",   profile.use_speaker_boost)
    logger.info("VOICE SELECTOR:   reason          = %s",   profile.selection_reason)

    return profile


def describe_selection(profile: VoiceProfile) -> dict:
    return {
        "voice_name":        profile.voice_name,
        "voice_id":          profile.voice_id,
        "mode":              profile.mode,
        "stability":         profile.stability,
        "style":             profile.style,
        "similarity_boost":  profile.similarity_boost,
        "use_speaker_boost": profile.use_speaker_boost,
        "selection_reason":  profile.selection_reason,
    }
