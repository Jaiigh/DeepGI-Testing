"""Train the WakeWordCNN model locally — no Colab needed.

Saves outputs/models/vad-deepgi/classifier_cnn.pt, which detector.py loads.
Run from the repo root:  python training/train_vad_cnn.py
"""

import csv
import os
import sys

import numpy as np
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from modules.voice_activation.detector import WakeWordCNN

# ── Hyperparameters ────────────────────────────────────────────────────────────
METADATA_CSV = "training_data/audio_vad/metadata.csv"
OUTPUT_DIR = config.FINETUNED_VAD_PATH
SAMPLE_RATE = config.SAMPLE_RATE
CLIP_SECONDS = 3.0
N_MFCC = 40
N_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3
VAL_SPLIT = 0.15
THRESHOLD = 0.65

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MFCC = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
)


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_audio(path: str) -> torch.Tensor:
    rate, data = wav.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    audio = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        audio /= np.iinfo(data.dtype).max
    tensor = torch.tensor(audio).unsqueeze(0)
    if rate != SAMPLE_RATE:
        import torchaudio.functional as AF
        tensor = AF.resample(tensor, rate, SAMPLE_RATE)
    n = tensor.shape[-1]
    if n < N_SAMPLES:
        tensor = F.pad(tensor, (0, N_SAMPLES - n))
    else:
        tensor = tensor[:, :N_SAMPLES]
    return tensor


def extract_mfcc(path: str) -> torch.Tensor:
    audio = load_audio(path)
    mfcc = _MFCC(audio)                              # (1, N_MFCC, time)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    return mfcc.float()


class VADDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        x = extract_mfcc(path)
        return x, torch.tensor(label, dtype=torch.float32)


def load_rows():
    rows = []
    missing = 0
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = row["audio_filepath"].replace("\\", "/")
            if not os.path.exists(path):
                missing += 1
                continue
            label = int(row["label"].strip())
            rows.append((path, label))
    if missing:
        print(f"[WARN] {missing} files missing — skipped")
    return rows


# ── Training loop ──────────────────────────────────────────────────────────────

def train():
    rows = load_rows()
    n_pos = sum(1 for _, l in rows if l == 1)
    n_neg = len(rows) - n_pos
    print(f"Dataset: {len(rows)} samples  ({n_pos} trigger / {n_neg} non-trigger)")

    dataset = VADDataset(rows)
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = WakeWordCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        tp = fp = fn = tn = 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                preds = (torch.sigmoid(logits) >= THRESHOLD).int()
                labels = y.int()
                tp += int(((preds == 1) & (labels == 1)).sum())
                fp += int(((preds == 1) & (labels == 0)).sum())
                fn += int(((preds == 0) & (labels == 1)).sum())
                tn += int(((preds == 0) & (labels == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"loss={train_loss/len(train_loader):.4f}  "
              f"val_loss={val_loss/len(val_loader):.4f}  "
              f"acc={acc:.3f}  f1={f1:.3f}")

        if f1 >= best_val_f1:
            best_val_f1 = f1
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, "classifier_cnn.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "threshold": THRESHOLD,
                "sample_rate": SAMPLE_RATE,
                "clip_seconds": CLIP_SECONDS,
                "n_mfcc": N_MFCC,
            }, out_path)
            print(f"  -> saved best model (f1={f1:.3f}) to {out_path}")

    print(f"\nTraining done. Best val F1: {best_val_f1:.3f}")
    print(f"Model saved to {os.path.join(OUTPUT_DIR, 'classifier_cnn.pt')}")


if __name__ == "__main__":
    train()
