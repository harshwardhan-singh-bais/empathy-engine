# 🧠 Empathy Engine — Giving AI a Human Voice

> A production-grade, emotionally-aware Text-to-Speech service that dynamically modulates vocal characteristics based on detected emotion.

---

## 📌 Project Overview

The **Empathy Engine** bridges the gap between cold, robotic TTS and genuinely expressive human-like speech. It analyses raw text, classifies fine-grained emotions using a transformer model, then computes a precise set of vocal parameters (pitch, rate, volume) through a **weighted parametric mapping system** — not naive if-else logic.

---

## ✨ Features

| Feature | Details |
|---|---|
| **7 Fine-grained emotions** | joy, surprise, anger, disgust, fear, sadness, neutral |
| **4 Coarse categories** | HAPPY · FRUSTRATED · SAD · CONCERNED · NEUTRAL |
| **Weighted probability aggregation** | All 7 emotion scores contribute to final voice params |
| **Non-linear intensity scaling** | Subtle emotions → subtle changes; strong emotions → dramatic |
| **Punctuation intensity boost** | `!`, `?`, ALL-CAPS detected and factored in |
| **SSML generation** | `<prosody>`, `<emphasis>`, `<break>` inserted automatically |
| **ElevenLabs primary TTS** | World-class expressive voice synthesis |
| **gTTS → pyttsx3 fallback** | Fully offline fallback chain, zero config required |
| **Web UI** | Glassmorphic dark UI, animated emotion bars, embedded audio player |

---

## 🏗 Architecture & Pipeline

```
Input Text
    │
    ▼
┌───────────────────────────────────────┐
│          Emotion Engine               │
│  j-hartmann/emotion-english-          │
│  distilroberta-base (HuggingFace)     │
│  → 7 raw probability scores           │
│  → punctuation boost (+0–20%)         │
│  → combined intensity score           │
└───────────────────────────────────────┘
    │
    ▼  fine_emotions dict + intensity
┌───────────────────────────────────────┐
│         Mapping Engine                │
│  Weighted aggregation:                │
│    Σ (prob_i × profile_i)             │
│  Non-linear scaling: × intensity^1.5  │
│  → pitch_percent, rate_tag,           │
│     volume_db, ssml                   │
└───────────────────────────────────────┘
    │
    ▼  voice params + SSML
┌───────────────────────────────────────┐
│          TTS Engine                   │
│  1. ElevenLabs API (expressive)       │
│  2. gTTS (fallback)                   │
│  3. pyttsx3 offline (fallback)        │
│  → audio.mp3                          │
└───────────────────────────────────────┘
    │
    ▼
  FastAPI → Jinja2 Web UI
```

---

## 🧠 Design Decisions: Emotion → Voice Mapping

### Why NOT simple if-else?

```python
# ❌ Naive (weak):
if emotion == "happy":
    pitch += 10
```

### Why Weighted Aggregation?

```python
# ✅ Professional (parametric):
for emotion, prob in fine_emotions.items():
    weight = prob / total_weight
    blended_pitch += weight * profile[emotion]["pitch_st"]

# Then non-linear intensity scaling:
final_pitch = blended_pitch * (intensity ** 1.5)
```

**Benefits:**
- All 7 emotion channels contribute simultaneously → **smooth emotional blending**
- Intensity automatically derived from model confidence (no separate logic)
- Punctuation surface signals augment model score (real-world robustness)
- Non-linear exponent keeps subtle emotions subtle and amplifies strong ones

### Emotion Profile Matrix

| Emotion  | Pitch (st) | Rate Δ | Volume (dB) |
|----------|-----------|--------|-------------|
| joy      | +4.0      | +0.30  | +3.0        |
| surprise | +3.0      | +0.25  | +2.0        |
| anger    | +2.0      | +0.40  | +5.0        |
| disgust  | -1.0      | -0.10  | +1.0        |
| fear     | +1.5      | +0.20  | -2.0        |
| sadness  | -4.0      | -0.30  | -3.0        |
| neutral  |  0.0      |  0.00  |  0.0        |

### Intensity Calculation

```
intensity = clamp(model_confidence + punctuation_boost, 0.0, 1.0)

punctuation_boost =
    min(exclamation_marks × 0.04, 0.12)
  + min(question_marks  × 0.02, 0.06)
  + min(caps_ratio      × 0.50, 0.08)
```

---

## 🚀 Setup & Run

### Prerequisites

- Python 3.12+
- `uv` (installed) — for virtual environment management
- ElevenLabs API key (optional — offline mode works without it)

### 1. Install dependencies

```bash
uv pip install -r requirements.txt
```

Or with regular pip:

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.local` and add your ElevenLabs API key:

```bash
# Edit .env.local
ELEVENLABS_API_KEY=your_key_here
```

> Without an API key, the service automatically falls back to **gTTS → pyttsx3** (no internet required for pyttsx3).

### 3. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the Web UI

```
http://localhost:8000
```

### 5. API usage (curl)

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "This is absolutely incredible news!"}' \
  | python -m json.tool
```

---

## 📁 Project Structure

```
empathy-engine/
│
├── main.py              ← FastAPI app, routes, pipeline orchestration
├── emotion_engine.py    ← HuggingFace model, 7-class detection, intensity
├── mapping_engine.py    ← Weighted parametric voice mapping + SSML generation
├── tts_engine.py        ← ElevenLabs / gTTS / pyttsx3 synthesis + fallback
│
├── templates/
│   └── index.html       ← Jinja2 web UI (glassmorphic dark theme)
│
├── static/              ← (reserved for future static assets)
├── outputs/             ← Generated audio files (.mp3)
│
├── .env.local           ← API keys (never commit to git)
├── requirements.txt     ← Python dependencies
└── README.md
```

---

## 🔑 API Reference

### `POST /synthesize`

```json
// Request
{ "text": "I can't believe this happened!", "voice_id": "EXAVITQu4vr4xnSDxMaL" }

// Response
{
  "dominant_emotion": "anger",
  "granular_label":   "Angry",
  "category":         "FRUSTRATED",
  "intensity":        0.88,
  "fine_emotions":    { "joy": 0.02, "anger": 0.82, ... },
  "voice_params":     { "pitch_percent": "+5%", "rate_tag": "fast", "ssml_volume": "+4dB" },
  "ssml":             "<speak><prosody ...>...</prosody></speak>",
  "audio_url":        "/outputs/audio_1234567890.mp3",
  "engine_used":      "elevenlabs"
}
```

### `GET /voices` — List available voices
### `GET /health` — Service health + engine status

---

## 🎯 Bonus Features Implemented

- [x] **Granular Emotions** — 7 fine-grained states, 5 coarse categories
- [x] **Intensity Scaling** — non-linear `intensity^1.5` curve
- [x] **Punctuation boost** — `!`, `?`, ALL-CAPS augment intensity
- [x] **SSML Integration** — prosody, emphasis, break tags
- [x] **Web Interface** — animated, glassmorphic single-page app
- [x] **Multiple TTS engines** — ElevenLabs, gTTS, pyttsx3 with auto-fallback
- [x] **ElevenLabs style mapping** — stability & style exaggeration computed from emotion

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| ML Model | `j-hartmann/emotion-english-distilroberta-base` |
| TTS Primary | ElevenLabs API (`eleven_multilingual_v2`) |
| TTS Fallback | gTTS → pyttsx3 (offline) |
| Frontend | Jinja2 + Vanilla HTML/CSS/JS |
| Config | python-dotenv |