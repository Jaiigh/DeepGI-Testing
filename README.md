# DeepGI ASR

Voice-activated speech recognition pipeline for colonoscopy findings reporting on macOS.

Say **"Hey DeepGI"** to activate, speak your finding, and the system transcribes and logs it — all offline, no cloud required.

---

## How It Works

```
Microphone → VAD (CNN wake word) → ASR (Whisper or Qwen3-ASR) → TTS (macOS say) → Log file
```

| Component | What it does |
|-----------|-------------|
| **VAD** | Listens continuously for "Hey DeepGI" using a CNN wake word model |
| **ASR** | Records 8 seconds and transcribes with Whisper or optional Qwen3-ASR |
| **TTS** | Reads the finding back using macOS `say` (or Kokoro neural TTS) |

---

## Setup

```bash
# 1. Clone the repo and enter the directory
git clone <repo-url>
cd DeepGI-Testing

# 2. Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start (using pre-trained model)

If a trained `classifier_cnn.pt` is already provided:

```bash
source venv/bin/activate
python pipeline.py
```

Say **"Hey DeepGI"**, speak your finding, and it will be transcribed and logged to `outputs/reports/`.

---

## Training the CNN Wake Word Model (your own voice)

The CNN VAD must be trained on voices that will use the system. If it does not detect your voice, follow these steps.

### Step 1 — Check your microphone device ID

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Find your microphone in the list and set its ID in `config.py`:

```python
AUDIO_DEVICE = 1   # replace with your device ID
```

### Step 2 — Record your voice samples

```bash
python training/record_for_vad.py
```

- Follow the on-screen prompts
- Record **trigger phrases** ("Hey DeepGI" and variants) — aim for 50+ samples
- Record **non-trigger speech** (random sentences) — aim for 100+ samples
- Files are saved to `training_data/audio_vad/` and logged in `training_data/audio_vad/metadata.csv`

> Record in a quiet environment, at a normal speaking distance from the microphone.  
> More speakers = more robust model. Have everyone who will use the system record samples.

### Step 3 — Train the CNN

```bash
python training/train_vad_cnn.py
```

- Trains for 40 epochs (~2–5 minutes on CPU)
- Saves the best checkpoint to `outputs/models/vad-deepgi/classifier_cnn.pt`
- Watch the F1 score — anything above 0.85 is good for demo use

### Step 4 — Enable the CNN in config

In `config.py`:

```python
USE_FINETUNED_VAD = True   # already True by default
```

### Step 5 — Run the pipeline

```bash
python pipeline.py
```

You should see:
```
[VAD] CNN model loaded. threshold=0.65, sr=16000, clip=3.0s
[VAD] Listening for wake word...
```

Say "Hey DeepGI" — the confidence score prints on each attempt.

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `AUDIO_DEVICE` | `1` | Microphone device ID (`None` = system default) |
| `USE_FINETUNED_VAD` | `True` | Use CNN wake word model |
| `USE_FINETUNED_ASR` | `False` | Use fine-tuned Whisper (requires training) |
| `USE_KOKORO_TTS` | `False` | Use Kokoro neural TTS (requires 400MB download) |
| `WHISPER_MODEL` | `"small"` | Whisper model size |
| `ASR_BACKEND` | `"whisper"` | Select `"whisper"` or optional `"qwen"` backend |
| `QWEN_ASR_MODEL` | `"Qwen/Qwen3-ASR-0.6B"` | Qwen model for the optional backend |
| `TRIGGER_CHUNK_DURATION` | `3` | Seconds of audio passed to CNN |
| `FINDING_DURATION` | `8` | Seconds recorded after trigger |

---

## Optional: Kokoro Neural TTS

By default the system uses macOS `say` for speech output. To use the higher quality Kokoro TTS:

```bash
# Download the model (~400MB, one time only)
export HF_TOKEN=your_huggingface_token
huggingface-cli download hexgrad/Kokoro-82M --token $HF_TOKEN
```

Then in `config.py`:

```python
USE_KOKORO_TTS = True
```

---

## Optional: Fine-tune Whisper ASR

To improve transcription accuracy on medical terminology:

```bash
# 1. Record medical finding samples
python training/record_samples.py

# 2. Fine-tune Whisper
python training/train_asr.py

# 3. Enable in config
# USE_FINETUNED_ASR = True
```

## Optional: Compare Whisper with Qwen3-ASR

Qwen's official ASR family is **Qwen3-ASR**, available in 0.6B and 1.7B
checkpoints. The project defaults to 0.6B because it is the more practical
local comparison model. Whisper remains the default and is not replaced.

```bash
# Install Qwen's official local inference package (one time)
pip install -U qwen-asr

# Evaluate both models against the same files in training_data/metadata.csv
python training/evaluate.py
```

The report at `outputs/evaluate_results.txt` lists WER and average inference
latency for Whisper and Qwen side by side. Qwen weights download automatically
on first use.

To try Qwen in the live pipeline instead, change only this setting in
`config.py`:

```python
ASR_BACKEND = "qwen"
```

For a higher-accuracy (and substantially heavier) comparison, change
`QWEN_ASR_MODEL` to `"Qwen/Qwen3-ASR-1.7B"`. On a CPU-only Mac, keep
`QWEN_ASR_DEVICE = "cpu"` and `QWEN_ASR_DTYPE = "float32"`.

---

## Evaluating the VAD Model

After training, run evaluation to see accuracy, precision, recall, and confusion matrix:

```bash
python training/validate_vad.py
```

---

## Project Structure

```
DeepGI-Testing/
├── config.py                          # all settings in one place
├── pipeline.py                        # main entry point
├── requirements.txt
│
├── modules/
│   ├── voice_activation/
│   │   └── detector.py                # CNN wake word detection
│   ├── asr/
│   │   └── transcriber.py             # Whisper transcription
│   └── tts/
│       └── speaker.py                 # say / Kokoro TTS
│
├── training/
│   ├── record_for_vad.py              # record wake word samples
│   ├── train_vad_cnn.py               # train CNN wake word model
│   ├── validate_vad.py                # evaluate VAD model
│   ├── train_vad.py                   # (legacy) sklearn logistic regression
│   ├── record_samples.py              # record ASR training samples
│   └── train_asr.py                   # fine-tune Whisper
│
├── training_data/
│   ├── audio_vad/                     # wake word .wav files
│   │   └── metadata.csv
│   └── metadata.csv                   # ASR training metadata
│
└── outputs/
    ├── models/
    │   └── vad-deepgi/
    │       └── classifier_cnn.pt      # trained CNN model
    └── reports/                       # session transcription logs
```
