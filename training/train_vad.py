"""Train a wake word classifier using MFCC features + logistic regression.

Approach:
  1. Load all .wav files from metadata.csv
  2. Extract 40-coefficient MFCCs (mean + std over time = 80-dim feature vector)
  3. Label trigger samples (containing "hey deepgi" variants) as positive
  4. Train a logistic regression with StandardScaler
  5. Save to outputs/models/vad-deepgi/classifier.pkl
"""

import csv
import os
import sys

import numpy as np
import scipy.io.wavfile as wav
import torch
import torchaudio.transforms as T
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

METADATA_CSV = "training_data/audio_vad/metadata.csv"
OUTPUT_MODEL_DIR = config.FINETUNED_VAD_PATH
TRIGGER_KEYWORDS = ["hey deepgi", "hey deep gi", "hey dgi", "hey deepgee", "hey deep gee"]

_MFCC = T.MFCC(
    sample_rate=config.SAMPLE_RATE,
    n_mfcc=40,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
)


def extract_features(filepath: str) -> np.ndarray:
    rate, data = wav.read(filepath)
    audio = data.astype(np.float32) / 32767.0
    if audio.ndim > 1:
        audio = audio[:, 0]
    tensor = torch.tensor(audio).unsqueeze(0)  # (1, samples)
    mfcc = _MFCC(tensor)                        # (1, 40, time)
    features = torch.cat([mfcc.mean(dim=-1), mfcc.std(dim=-1)], dim=-1)
    return features.numpy().flatten()           # 80-dim vector


def load_samples():
    if not os.path.exists(METADATA_CSV):
        print(f"[ERROR] {METADATA_CSV} not found. Run training/record_samples.py first.")
        sys.exit(1)

    X, y = [], []
    with open(METADATA_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Fix Windows backslashes in paths recorded on macOS
            path = row["audio_filepath"].replace("\\", "/")
            if not os.path.exists(path):
                print(f"  [WARN] Missing file: {path} — skipping")
                continue
            features = extract_features(path)
            # Use label column directly if present, otherwise infer from text
            if "label" in row and row["label"].strip() in ("0", "1"):
                label = int(row["label"].strip())
            else:
                text = row["text"].strip().lower()
                label = 1 if any(kw in text for kw in TRIGGER_KEYWORDS) else 0
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


def main():
    print("Loading samples and extracting MFCC features...")
    X, y = load_samples()

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"  {n_pos} trigger samples (positive)")
    print(f"  {n_neg} non-trigger samples (negative)")

    if n_pos == 0:
        print("[ERROR] No trigger samples found in metadata.csv.")
        print("Record 'hey deepgi' clips with training/record_samples.py first.")
        sys.exit(1)

    if n_neg == 0:
        print("[ERROR] No non-trigger samples found. Need both classes to train.")
        sys.exit(1)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
    ])
    clf.fit(X, y)

    train_acc = clf.score(X, y)
    print(f"  Training accuracy: {train_acc:.1%}")

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_MODEL_DIR, "classifier.pkl")
    joblib.dump(clf, model_path)

    print(f"\nModel saved to {model_path}")
    print("Training complete. Set USE_FINETUNED_VAD = True in config.py to use it.")


if __name__ == "__main__":
    main()
