import difflib
import os
import sys
import numpy as np
import sounddevice as sd
import config

_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"[VAD] Loading Whisper {config.WHISPER_MODEL} for trigger detection...")
        _whisper_model = whisper.load_model(config.WHISPER_MODEL)
    return _whisper_model


def _record_audio(duration: float) -> np.ndarray:
    audio = sd.rec(
        int(duration * config.SAMPLE_RATE),
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def _is_trigger(text: str) -> bool:
    text = text.strip().lower()
    for variant in config.TRIGGER_VARIANTS:
        if variant in text:
            return True
    # Fuzzy match against each variant
    for variant in config.TRIGGER_VARIANTS:
        ratio = difflib.SequenceMatcher(None, text, variant).ratio()
        if ratio >= config.FUZZY_THRESHOLD:
            return True
    return False


_SILENCE_RMS_THRESHOLD = 0.01  # skip transcription if audio is this quiet


def _wait_for_trigger_base():
    model = _get_whisper_model()
    while True:
        audio = _record_audio(config.TRIGGER_CHUNK_DURATION)

        # Skip silent/noise chunks — Whisper hallucinates garbage on silence
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < _SILENCE_RMS_THRESHOLD:
            continue

        # No initial_prompt here — it causes Whisper to parrot back the medical
        # vocabulary instead of transcribing what was actually said
        result = model.transcribe(
            audio,
            language=config.LANGUAGE,
            fp16=config.FP16,
            condition_on_previous_text=False,
        )
        heard = result["text"]
        print(f"[VAD] Heard: {heard.strip()!r}")
        if _is_trigger(heard):
            print("[VAD] Trigger detected!")
            return


def _wait_for_trigger_finetuned():
    import joblib
    import torch
    import torchaudio.transforms as T

    model_path = os.path.join(config.FINETUNED_VAD_PATH, "classifier.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Classifier not found at {model_path}. Run python training/train_vad.py first."
        )
    print(f"[VAD] Loading classifier from {model_path}...")
    clf = joblib.load(model_path)

    mfcc_transform = T.MFCC(
        sample_rate=config.SAMPLE_RATE,
        n_mfcc=40,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
    )

    print("[VAD] Listening for wake word (fine-tuned classifier)...")
    while True:
        audio = _record_audio(config.TRIGGER_CHUNK_DURATION)

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < _SILENCE_RMS_THRESHOLD:
            continue

        tensor = torch.tensor(audio).unsqueeze(0)
        mfcc = mfcc_transform(tensor)
        features = torch.cat([mfcc.mean(dim=-1), mfcc.std(dim=-1)], dim=-1)
        features_np = features.numpy().flatten().reshape(1, -1)

        prob = clf.predict_proba(features_np)[0][1]
        if prob >= config.FUZZY_THRESHOLD:
            print(f"[VAD] Wake word detected! (confidence: {prob:.1%})")
            return


def wait_for_trigger():
    """Block until the trigger phrase is detected, then return."""
    if config.USE_FINETUNED_VAD:
        _wait_for_trigger_finetuned()
    else:
        _wait_for_trigger_base()
