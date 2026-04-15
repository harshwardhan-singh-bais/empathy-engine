import math
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emotion Parameter Profiles
# Each emotion contributes a direction vector in vocal-parameter space.
# Values are relative deltas: pitch (semitones), rate (ratio), volume (dB).
# ---------------------------------------------------------------------------
EMOTION_PROFILES = {
    "joy":      {"pitch_st": +4.0,  "rate_delta": +0.30, "volume_db": +3.0},
    "surprise": {"pitch_st": +3.0,  "rate_delta": +0.25, "volume_db": +2.0},
    "anger":    {"pitch_st": +2.0,  "rate_delta": +0.40, "volume_db": +5.0},
    "disgust":  {"pitch_st": -1.0,  "rate_delta": -0.10, "volume_db": +1.0},
    "fear":     {"pitch_st": +1.5,  "rate_delta": +0.20, "volume_db": -2.0},
    "sadness":  {"pitch_st": -4.0,  "rate_delta": -0.30, "volume_db": -3.0},
    "neutral":  {"pitch_st":  0.0,  "rate_delta":  0.00, "volume_db":  0.0},
}

# Base speech parameters (ElevenLabs / SSML baseline)
BASE_RATE_RATIO = 1.0    # 1.0 = normal speed
BASE_VOLUME_DB  = 0.0    # 0 dB = normal
BASE_PITCH_ST   = 0.0    # 0 semitones = normal

# Non-linear exponent — keeps subtle emotions subtle, makes strong ones dramatic
INTENSITY_EXPONENT = 1.5

# Human-readable ElevenLabs/SSML rate tags
def _rate_to_ssml_tag(rate_ratio: float) -> str:
    if rate_ratio >= 1.40:
        return "x-fast"
    if rate_ratio >= 1.15:
        return "fast"
    if rate_ratio <= 0.65:
        return "x-slow"
    if rate_ratio <= 0.85:
        return "slow"
    return "medium"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_voice_parameters(fine_emotions: dict, intensity: float) -> dict:
    """
    Weighted aggregation of all emotion probabilities into final voice params.

    Pipeline:
      1. For each fine emotion, scale its profile by its probability score.
      2. Sum all weighted directions → blended parameter delta.
      3. Apply non-linear intensity curve (exponent) to the blended delta.
      4. Add to base values, then clamp within safe bounds.

    Args:
        fine_emotions: {label: probability} — all 7 raw scores from model.
        intensity: combined intensity score (model + punctuation boost).

    Returns:
        {
          "pitch_semitones": float,   # relative pitch shift in semitones
          "pitch_percent":   str,     # SSML-compatible "+10%"
          "rate_ratio":      float,   # speed multiplier
          "rate_tag":        str,     # "fast" / "slow" / "medium" etc.
          "volume_db":       float,   # dB shift
          "ssml_volume":     str,     # "+3dB" / "-2dB"
        }
    """
    blended_pitch  = 0.0
    blended_rate   = 0.0
    blended_volume = 0.0
    total_weight   = sum(fine_emotions.values())

    for emotion, prob in fine_emotions.items():
        profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])
        weight  = prob / total_weight if total_weight > 0 else 0.0

        blended_pitch  += weight * profile["pitch_st"]
        blended_rate   += weight * profile["rate_delta"]
        blended_volume += weight * profile["volume_db"]

    # Non-linear intensity scaling → subtle emotions stay subtle
    scale_factor = intensity ** INTENSITY_EXPONENT

    final_pitch_st     = _clamp(blended_pitch  * scale_factor, -8.0,  +8.0)
    final_rate_delta   = _clamp(blended_rate   * scale_factor, -0.50, +0.60)
    final_volume_db    = _clamp(blended_volume * scale_factor, -6.0,  +6.0)

    final_rate_ratio   = _clamp(BASE_RATE_RATIO + final_rate_delta, 0.5, 1.8)

    # Convert semitones → percent string for SSML
    pitch_percent_val  = int(round(final_pitch_st / 12 * 100))
    pitch_percent_str  = f"+{pitch_percent_val}%" if pitch_percent_val >= 0 else f"{pitch_percent_val}%"

    volume_db_int     = int(round(final_volume_db))
    ssml_volume_str   = f"+{volume_db_int}dB" if volume_db_int >= 0 else f"{volume_db_int}dB"

    logger.debug(
        "Voice params → pitch=%s, rate=%.2f (%s), volume=%s",
        pitch_percent_str, final_rate_ratio, _rate_to_ssml_tag(final_rate_ratio), ssml_volume_str,
    )

    return {
        "pitch_semitones": round(final_pitch_st, 3),
        "pitch_percent":   pitch_percent_str,
        "rate_ratio":      round(final_rate_ratio, 3),
        "rate_tag":        _rate_to_ssml_tag(final_rate_ratio),
        "volume_db":       round(final_volume_db, 3),
        "ssml_volume":     ssml_volume_str,
    }


def generate_ssml(text: str, voice_params: dict) -> str:
    """
    Wrap cleaned text in SSML <prosody> tags.
    Adds strategic <break> and <emphasis> for natural rhythm.
    """
    pitch   = voice_params["pitch_percent"]
    rate    = voice_params["rate_tag"]
    volume  = voice_params["ssml_volume"]

    # Identify capitalized words for emphasis (heuristic for strong expression)
    words = text.split()
    processed_words = []
    for word in words:
        stripped = word.strip(".,!?;:")
        if stripped.isupper() and len(stripped) > 1:
            processed_words.append(f'<emphasis level="strong">{word}</emphasis>')
        else:
            processed_words.append(word)

    processed_text = " ".join(processed_words)

    # Add a brief pause after sentence-ending punctuation for natural breathing
    processed_text = processed_text.replace(". ", '. <break time="200ms"/> ')
    processed_text = processed_text.replace("! ", '! <break time="150ms"/> ')
    processed_text = processed_text.replace("? ", '? <break time="200ms"/> ')

    ssml = (
        f'<speak>'
        f'<prosody pitch="{pitch}" rate="{rate}" volume="{volume}">'
        f'{processed_text}'
        f'</prosody>'
        f'</speak>'
    )
    return ssml
