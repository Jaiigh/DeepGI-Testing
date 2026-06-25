# DeepGI ASR

Voice-activated speech recognition pipeline for colonoscopy findings reporting on macOS.

Say **"Hey DeepGI"** to activate, speak your finding, and the system transcribes and logs it — all offline, no cloud required.

---

## Overview

| Component | What it does |
|-----------|-------------|
| **VAD** (Voice Activation Detection) | Listens for the "Hey DeepGI" wake word using Whisper |
| **ASR** (Automatic Speech Recognition) | Transcribes GI findings with Whisper + medical prompt |
| **TTS** (Text-to-Speech) | Reads back findings using macOS `say` command |

The pipeline can run with base Whisper models out of the box, or switch to fine-tuned models after training on your own GI audio samples.

---

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **macOS note:** `fp16=False` is set throughout — CUDA is not required.

---

## Running the Demo

```bash
python demo.py
```

- Say **"Hey DeepGI"** to trigger recording
- Speak your finding (8 seconds)
- Finding is transcribed, read back, and saved to `outputs/reports/session_YYYY-MM-DD.txt`
- Press **Ctrl+C** to end the session

---

## Recording Training Samples

```bash
python training/record_samples.py
```

- Follow the on-screen prompts
- Records 30 scripted lines (trigger phrases + GI findings)
- Saves `.wav` files to `training_data/audio/`
- Appends rows to `training_data/metadata.csv`

---

## Training the ASR Model

```bash
python training/train_asr.py
```

Fine-tunes Whisper small on your recorded samples. Training takes ~15–30 minutes on CPU.  
Model is saved to `outputs/models/whisper-deepgi`.

---

## Training the Wake Word Detector

```bash
python training/train_vad.py
```

Trains a custom openWakeWord model on your "hey deepgi" samples.  
Model is saved to `outputs/models/vad-deepgi`.

---

## Switching to Fine-Tuned Models

Edit [config.py](config.py):

```python
USE_FINETUNED_ASR = True   # use fine-tuned Whisper
USE_FINETUNED_VAD = True   # use trained wake word detector
```

Then run `python demo.py` as normal.

---

## Evaluating Performance

```bash
python training/evaluate.py
```

Prints a comparison table: WER, trigger accuracy, and latency for base vs. fine-tuned models.  
Results saved to `outputs/evaluate_results.txt`.

---

## Project Structure

```
deepgi-asr/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                    # central config for all settings
├── demo.py                      # single entry point for professor demo
├── pipeline.py                  # connects VAD → ASR → TTS
│
├── modules/
│   ├── __init__.py
│   ├── voice_activation/
│   │   ├── __init__.py
│   │   └── detector.py          # wake word detection
│   ├── asr/
│   │   ├── __init__.py
│   │   └── transcriber.py       # speech to text
│   └── tts/
│       ├── __init__.py
│       └── speaker.py           # text to speech
│
├── training/
│   ├── __init__.py
│   ├── train_asr.py             # fine-tune Whisper on medical terms
│   ├── train_vad.py             # train custom wake word detector
│   ├── record_samples.py        # record training audio samples
│   └── evaluate.py              # measure WER and accuracy after training
│
├── training_data/
│   ├── audio/                   # recorded .wav samples (git-ignored)
│   └── metadata.csv             # headers: audio_filepath, text
│
└── outputs/
    ├── reports/                 # session transcription logs (git-ignored)
    └── models/                  # saved fine-tuned models (git-ignored)
```
