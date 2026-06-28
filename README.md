# DeepGI ASR

Voice-activated speech recognition pipeline for colonoscopy findings reporting.

Say **"Hey DeepGI"** to activate, speak your finding, and the system transcribes and logs it offline.

---

## Overview

| Component | What it does |
|-----------|-------------|
| **VAD** (Voice Activation Detection) | Listens for the "Hey DeepGI" wake word using Whisper or the trained CNN detector |
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

There are two wake-word training paths in this repository:

- `training/train_vad.py` is the older baseline that trains a logistic regression classifier from MFCC summary features and saves `classifier.pkl`.
- `training/train_vad.ipynb` is the notebook used for the current CNN wake-word flow. It saves `classifier_cnn.pt`, which is what `modules/voice_activation/detector.py` loads when `USE_FINETUNED_VAD = True`.

```bash
python training/train_vad.py
```

This command trains the older MFCC + logistic regression baseline and saves `outputs/models/vad-deepgi/classifier.pkl`.

For the current runtime CNN detector, train with `training/train_vad.ipynb` and export `outputs/models/vad-deepgi/classifier_cnn.pt`.

---

## VAD Dataset Recording

`training/record_for_vad.py` records a balanced wake-word dataset for the CNN wake-word detector.

Run it with:

```bash
python training/record_for_vad.py
```

What it does:

- Records fixed-duration mono audio clips using `sounddevice`.
- Uses `config.SAMPLE_RATE` and `config.TRIGGER_CHUNK_DURATION`.
- Saves WAV files to `training_data/audio_vad/` by default.
- Appends metadata to `training_data/audio_vad/metadata.csv` by default.
- Writes metadata columns: `audio_filepath,text,label,category,speaker,take`.
- Supports positive wake-word samples, hard negatives, medical negative phrases, command negatives, noise/silence, and general Thai speech.

The script is interactive. It asks for a speaker id, repeat count per selected script, and which script numbers to record. Selection supports:

- `all`
- one item, such as `10`
- comma-separated items, such as `1,3,10`
- ranges, such as `1-10`
- mixed ranges and items, such as `1-5,10`

Each take can be saved, redone, skipped, or the session can be stopped.

---

## VAD Training Notebook

`training/train_vad.ipynb` is the notebook workflow for training the CNN wake-word classifier used by runtime detection.

The notebook:

- Loads WAV files and labels from a metadata CSV.
- Uses the `label` column where `1` means trigger and `0` means non-trigger.
- Pads or trims audio clips to the configured trigger length.
- Converts audio to 40-coefficient MFCC features.
- Trains a PyTorch CNN binary classifier.
- Tracks train and validation metrics including loss, accuracy, precision, recall, false positives, and false negatives.
- Saves the best model checkpoint as `outputs/models/vad-deepgi/classifier_cnn.pt`.
- Can also run prediction/evaluation against a validation metadata file and produce a confusion matrix.

Use the notebook when you want the model format expected by `modules/voice_activation/detector.py`.

---

## Voice Activation Runtime

`modules/voice_activation/detector.py` exposes the runtime trigger function:

```python
from modules.voice_activation.detector import wait_for_trigger

wait_for_trigger()
```

`wait_for_trigger()` blocks until the wake word is detected. It chooses the detection path from `config.USE_FINETUNED_VAD`.

### Base mode

When `USE_FINETUNED_VAD = False`, the detector:

- Records short audio chunks with `sounddevice`.
- Skips quiet chunks using RMS thresholding.
- Transcribes each chunk with Whisper.
- Checks the transcript against `config.TRIGGER_VARIANTS`.
- Uses fuzzy matching with `config.FUZZY_THRESHOLD`.

This mode does not require a trained VAD model, but it is slower because it runs Whisper repeatedly.

### Fine-tuned CNN mode

When `USE_FINETUNED_VAD = True`, the detector:

- Loads `outputs/models/vad-deepgi/classifier_cnn.pt`.
- Captures microphone audio through `sounddevice.RawInputStream`.
- Uses WebRTC VAD to find voiced segments before classification.
- Keeps a short pre-roll so the beginning of speech is not dropped.
- Ends a segment after enough silence or after `config.TRIGGER_CHUNK_DURATION`.
- Converts PCM audio to float32.
- Rejects very quiet segments.
- Converts the segment to MFCC features.
- Runs the CNN and triggers when confidence is at least `0.7`.

On a successful trigger, it saves the detected segment to `debug_voice_frames/voice_segment.wav` for debugging.

Important compatibility notes:

- WebRTC VAD supports only 10 ms, 20 ms, or 30 ms frames. This repo uses 20 ms.
- WebRTC VAD requires a sample rate of 8000, 16000, 32000, or 48000 Hz.
- The model sample rate should match `config.SAMPLE_RATE`.
- The CNN architecture in the notebook and `detector.py` must stay compatible, otherwise `load_state_dict` will fail.

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
│   ├── train_vad.py             # older MFCC + logistic regression VAD baseline
│   ├── train_vad.ipynb          # CNN wake-word training notebook
│   ├── record_samples.py        # record ASR / general training audio samples
│   ├── record_for_vad.py        # record labeled wake-word VAD samples
│   └── evaluate.py              # measure WER and accuracy after training
│
├── training_data/
│   ├── audio/                   # recorded .wav samples (git-ignored)
│   ├── metadata.csv             # ASR/general metadata: audio_filepath, text
│   └── audio_vad/               # wake-word VAD .wav samples
│       └── metadata.csv         # audio_filepath, text, label, category, speaker, take
│
└── outputs/
    ├── reports/                 # session transcription logs (git-ignored)
    └── models/                  # saved fine-tuned models (git-ignored)
```
