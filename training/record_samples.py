"""Interactive CLI to record labeled audio samples for training."""

import csv
import os
import sys

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

AUDIO_DIR = "training_data/audio"
METADATA_CSV = "training_data/metadata.csv"
RECORD_DURATION = 5  # seconds per sample

SCRIPTS = [
    # Trigger phrases (10)
    "hey deepgi",
    "hey deepgi polyp found",
    "hey deepgi bleeding noted",
    "hey deepgi lesion identified",
    "hey deepgi diverticulum seen",
    "hey deep gi",
    "hey dgi",
    "hey deepgee",
    "hey deep gee",
    "hey deepgi finding recorded",
    # GI findings (20)
    "polyp at sigmoid colon 5 millimeters",
    "bleeding at rectum",
    "diverticulum at descending colon",
    "Paris classification one p pedunculated polyp",
    "sessile polyp at ascending colon 8 millimeters",
    "flat lesion at transverse colon",
    "biopsy taken from hepatic flexure",
    "resection performed at splenic flexure",
    "Boston Bowel Score 8 excellent preparation",
    "cecum reached intubation time 4 minutes",
    "polyp at sigmoid colon 12 millimeters pedunculated",
    "no lesions identified throughout the colon",
    "diverticulum at sigmoid colon mild diverticulosis",
    "bleeding at ascending colon active oozing",
    "Paris classification 2a flat elevated lesion at rectum",
    "hot snare polypectomy performed at cecum",
    "tattoo placed distal to polyp at descending colon",
    "Paris classification 1 s sessile polyp at transverse colon",
    "Boston Bowel Score 6 adequate preparation",
    "withdrawal time 9 minutes complete examination",
]


def record_audio(duration: int) -> np.ndarray:
    audio = sd.rec(
        int(duration * config.SAMPLE_RATE),
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def save_wav(audio: np.ndarray, filepath: str) -> None:
    pcm = (audio * 32767).astype(np.int16)
    wav.write(filepath, config.SAMPLE_RATE, pcm)


def main() -> None:
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # Find next sample index
    existing = [f for f in os.listdir(AUDIO_DIR) if f.startswith("sample_") and f.endswith(".wav")]
    next_index = len(existing) + 1

    write_header = not os.path.exists(METADATA_CSV)
    csv_file = open(METADATA_CSV, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(["audio_filepath", "text"])

    recorded = 0
    print(f"\nDeepGI Sample Recorder — {len(SCRIPTS)} scripts to record")
    print(f"Each recording is {RECORD_DURATION} seconds. Press Enter to start each one.\n")

    for i, script in enumerate(SCRIPTS, start=1):
        filename = f"sample_{next_index:03d}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)
        print(f"[{i}/{len(SCRIPTS)}] Say: \"{script}\"")
        input("  Press Enter to start recording...")
        print(f"  Recording {RECORD_DURATION}s... ", end="", flush=True)
        audio = record_audio(RECORD_DURATION)
        save_wav(audio, filepath)
        writer.writerow([filepath, script])
        csv_file.flush()
        recorded += 1
        next_index += 1
        print("saved.")

    csv_file.close()
    print(f"\nDone. {recorded} samples recorded and saved to {METADATA_CSV}.")


if __name__ == "__main__":
    main()
