# The Empathy Engine: Giving AI a Human Voice

## 1. Project Description

The Empathy Engine is an advanced Speech Synthesis service designed to solve the "uncanny valley" problem in AI-driven communication. While modern LLMs generate text with high accuracy, the final vocal delivery often remains robotic, monotonic, and emotionally disconnected. This disconnect can frustrate users in customer service or sales scenarios where rapport and trust are essential.

Our solution is a deterministic emotion-to-voice mapping system that dynamically modulates vocal characteristics based on the detected sentiment of the source text. It moves beyond simple "text-to-speech" by transforming raw text into an emotionally resonant performance. The service analyzes input text, identifies granular emotional states (such as Joy, Sadness, or Anger), and programmatically alters vocal parameters including Pitch, Rate, Volume, Stability, and Style to match the speaker's intent.

## 2. Design Choices and Mapping Logic

### Emotion Detection and Classification
The engine uses a pre-trained Transformer model (`j-hartmann/emotion-english-distilroberta-base`) to achieve high-accuracy classification across seven granular categories: Joy, Surprise, Anger, Disgust, Fear, Sadness, and Neutral. This provides a more nuanced foundation than simple Positive/Negative models.

### Weighted Parametric Mapping
A critical design choice was the implementation of a Weighted Aggregator for parameter calculation. Instead of using a simple lookup table, the system analyzes the probability distribution of all detected emotions. If a sentence is 70% Neutral and 30% Joy, the vocal modulation will reflect a subtle "cheerfulness" rather than a full transition to elation. This produces a much smoother and more natural vocal transition between sentences.

### The Two-Layer Expression System
Emotional resonance is achieved through two compounding layers:
1. Primary Layer (Parameter Modulation): Stability, Style, and text-shaping (punctuation/rhythm) are the primary drivers of emotion. For example, high Stability creates the "heaviness" associated with Sadness, while low Stability creates the "erratic energy" of Joy or Surprise.
2. Secondary Layer (Timbre Matching): The system selects a specific voice character whose base timbre reinforces the emotion (e.g., the authoritative "Adam" for anger, or the empathetic "Bella" for sadness).

### Triple-Engine Fallback Chain
To ensure high availability and accessibility, the engine implements a three-tier execution pipeline:
- Tier 1 (ElevenLabs): Used for high-fidelity, expressive synthesis with fine-grained stability and style control.
- Tier 2 (Google TTS): Used as an online fallback if the primary API is unavailable or if credit limits are reached.
- Tier 3 (pyttsx3 Offline): An fully offline fallback that ensures the system remains functional even without internet connectivity, utilizing local system drivers.

## 3. Detailed Setup Instructions

Follow these steps to set up and run the Empathy Engine on your local machine.

### Prerequisites
- Python 3.12 or higher.
- A terminal (PowerShell, Bash, or CMD).

### Step 1: Clone and Infrastructure
1. Clone this repository to your local directory.
2. Open your terminal in the `empathy-engine` folder.

### Step 2: Virtual Environment
It is recommended to use a virtual environment to manage dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Dependencies
Install the required libraries and the PyTorch/Transformers stack:
```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration
1. Create a `.env` file in the root directory.
2. Populate it with your ElevenLabs API Key:
   ```env
   ELEVENLABS_API_KEY=your_api_key_here
   ```
3. (Optional) To demo the "Power of Parameters", set `SINGLE_VOICE_MODE=true`. This forces the engine to use only one voice for all emotions, proving that our modulation logic (not just the voice choice) is the primary driver of expression.

## 4. Running the Application

### Start the Backend
Execute the following command to start the FastAPI server:
```bash
uvicorn main:app --reload
```

### Access the Web Interface
Once the terminal shows "Application startup complete", navigate to:
```text
http://localhost:8000
```
This interface provides a real-time dashboard visualizing the emotion detection scores, the computed vocal parameters, and the generated audio output.

## 5. Mapping Reference
| Emotion | Pitch Shift | Rate Shift | Stability | Style |
|:--- |:--- |:--- |:--- |:--- |
| Joy | High (+) | Fast (+) | Very Low | Max |
| Anger | Mid (+) | Very Fast (++) | Min | Max |
| Sadness | Low (-) | Slow (-) | High | Min |
| Fear | High (+) | Fast (+) | Low | Mid |
| Disgust | Low (-) | Slow (-) | Mid | Mid |
| Neutral | Baseline | Baseline | Mid | Low |

## 6. Project Deliverables Checklist
- [x] Functional Text-to-Speech service.
- [x] Multi-layer Emotion Detection (7 granular states).
- [x] Programmatic modulation of Rate, Pitch, Volume, Stability, and Style.
- [x] Web UI with real-time parameter visualization and audio playback.
- [x] Automated fallback logic (Online -> Offline).
- [x] Comprehensive documentation on design and logic.