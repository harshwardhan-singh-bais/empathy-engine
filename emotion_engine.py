from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logger.info("Loading emotion model: %s", MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
        logger.info("Emotion model loaded successfully.")


EMOTION_TO_CATEGORY = {
    "joy":      "HAPPY",
    "surprise": "HAPPY",
    "anger":    "FRUSTRATED",
    "disgust":  "FRUSTRATED",
    "fear":     "CONCERNED",
    "sadness":  "SAD",
    "neutral":  "NEUTRAL",
}

GRANULAR_LABELS = {
    "joy":      "Joyful",
    "surprise": "Surprised",
    "anger":    "Angry",
    "disgust":  "Disgusted",
    "fear":     "Fearful",
    "sadness":  "Sad",
    "neutral":  "Neutral",
}


def _compute_punctuation_boost(text: str) -> float:
    exclamation_count = text.count("!")
    question_count = text.count("?")
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    boost = 0.0
    boost += min(exclamation_count * 0.04, 0.12)
    boost += min(question_count * 0.02, 0.06)
    boost += min(caps_ratio * 0.5, 0.08)
    return min(boost, 0.20)


def detect_emotion(text: str) -> dict:
    _load_model()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    labels = _model.config.id2label

    fine_emotions = {labels[i]: float(probs[i]) for i in range(len(probs))}

    dominant_emotion = max(fine_emotions, key=fine_emotions.get)
    raw_intensity = fine_emotions[dominant_emotion]

    punctuation_boost = _compute_punctuation_boost(text)
    combined_intensity = min(raw_intensity + punctuation_boost, 1.0)

    return {
        "fine_emotions":     fine_emotions,
        "dominant_emotion":  dominant_emotion,
        "category":          EMOTION_TO_CATEGORY.get(dominant_emotion, "NEUTRAL"),
        "granular_label":    GRANULAR_LABELS.get(dominant_emotion, "Neutral"),
        "raw_intensity":     round(raw_intensity, 4),
        "punctuation_boost": round(punctuation_boost, 4),
        "intensity":         round(combined_intensity, 4),
    }
