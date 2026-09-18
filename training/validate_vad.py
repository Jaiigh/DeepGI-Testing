"""
Interactive CLI to record balanced wake word dataset.

Metadata format:
audio_filepath,text,label,category,speaker,take

label:
  1 = trigger / wake word
  0 = non-trigger

Selection examples:
  all      = record all scripts
  10       = record script number 10
  1,3,10   = record script 1, 3, 10
  1-10     = record script 1 to 10
  1-5,10   = record script 1 to 5 and 10

Recording flow:
  script 10 take 1
  script 10 take 2
  script 10 take 3
  ...
  then next script
"""

import csv
import os
import sys

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# =========================
# Config
# =========================

AUDIO_DIR = getattr(
    config,
    "WAKEWORD_AUDIO_DIR",
    "validate_data/audio_vad"
)

METADATA_CSV = getattr(
    config,
    "WAKEWORD_METADATA_CSV",
    "validate_data/audio_vad/metadata.csv"
)

RECORD_DURATION = getattr(
    config,
    "TRIGGER_CHUNK_DURATION",
    3
)

SAMPLE_RATE = config.SAMPLE_RATE


# =========================
# Dataset scripts
# =========================

DATASET_PLAN = [
    # =========================
    # Positive: wake word only
    # ต้อง trigger
    # =========================
    {"text": "hey deepgi", "label": 1, "category": "positive_validate"},
    {"text": "hey deep gi", "label": 1, "category": "positive_validate"},
    {"text": "hey dgi", "label": 1, "category": "positive_validate"},
    {"text": "hey d gi", "label": 1, "category": "positive_validate"},
    {"text": "hey deepgee", "label": 1, "category": "positive_validate"},
    {"text": "hey deep gee", "label": 1, "category": "positive_validate"},
    {"text": "เฮ้ deepgi", "label": 1, "category": "positive_validate"},
    {"text": "เฮ้ deep gi", "label": 1, "category": "positive_validate"},
    {"text": "เฮ deepgi", "label": 1, "category": "positive_validate"},
    {"text": "เฮ dgi", "label": 1, "category": "positive_validate"},

    # =========================
    # Hard negative:
    # คล้าย trigger แต่ห้าม trigger
    # =========================
    {"text": "deepgi", "label": 0, "category": "hard_negative_validate"},
    {"text": "deep gi", "label": 0, "category": "hard_negative_validate"},
    {"text": "dgi", "label": 0, "category": "hard_negative_validate"},
    {"text": "deepgee", "label": 0, "category": "hard_negative_validate"},
    {"text": "hey", "label": 0, "category": "hard_negative_validate"},
    {"text": "เฮ้", "label": 0, "category": "hard_negative_validate"},
    {"text": "hey deep", "label": 0, "category": "hard_negative_validate"},
    {"text": "เฮ้ deep", "label": 0, "category": "hard_negative_validate"},
    {"text": "hey gi", "label": 0, "category": "hard_negative_validate"},
    {"text": "deepgi please", "label": 0, "category": "hard_negative_validate"},
    {"text": "โอเค deepgi", "label": 0, "category": "hard_negative_validate"},
    {"text": "เรียก deepgi", "label": 0, "category": "hard_negative_validate"},
    {"text": "เปิด deepgi", "label": 0, "category": "hard_negative_validate"},
    {"text": "ปิด deepgi", "label": 0, "category": "hard_negative_validate"},
    {"text": "ระบบ deepgi", "label": 0, "category": "hard_negative_validate"},
    {"text": "ชื่อ deepgi", "label": 0, "category": "hard_negative_validate"},
    {"text": "deepgi ทำงานไหม", "label": 0, "category": "hard_negative_validate"},
    {"text": "deepgi ยังไม่ติด", "label": 0, "category": "hard_negative_validate"},

    # =========================
    # Confusing wake words:
    # คำปลุกอื่น ๆ ห้าม trigger
    # =========================
    {"text": "hey siri", "label": 0, "category": "other_wakeword_validate"},
    {"text": "hey google", "label": 0, "category": "other_wakeword_validate"},
    {"text": "hey alexa", "label": 0, "category": "other_wakeword_validate"},
    {"text": "hey jarvis", "label": 0, "category": "other_wakeword_validate"},
    {"text": "hey doctor", "label": 0, "category": "other_wakeword_validate"},
    {"text": "hey หมอ", "label": 0, "category": "other_wakeword_validate"},
    {"text": "เฮ้ หมอ", "label": 0, "category": "other_wakeword_validate"},
    {"text": "hello deepgi", "label": 0, "category": "other_wakeword_validate"},

    # =========================
    # Medical negative:
    # คำพูดจริงหลัง trigger แต่ไม่มี wake word
    # =========================
    {"text": "เจอ polyp", "label": 0, "category": "negative_medical_validate"},
    {"text": "พบ polyp", "label": 0, "category": "negative_medical_validate"},
    {"text": "เจอ bleeding", "label": 0, "category": "negative_medical_validate"},
    {"text": "พบ bleeding", "label": 0, "category": "negative_medical_validate"},
    {"text": "มี lesion", "label": 0, "category": "negative_medical_validate"},
    {"text": "พบ lesion", "label": 0, "category": "negative_medical_validate"},
    {"text": "เจอ diverticulum", "label": 0, "category": "negative_medical_validate"},
    {"text": "พบ diverticulum", "label": 0, "category": "negative_medical_validate"},
    {"text": "ทำ biopsy", "label": 0, "category": "negative_medical_validate"},
    {"text": "ทำ resection", "label": 0, "category": "negative_medical_validate"},
    {"text": "polyp ที่ rectum", "label": 0, "category": "negative_medical_validate"},
    {"text": "polyp ที่ sigmoid", "label": 0, "category": "negative_medical_validate"},
    {"text": "bleeding ที่ rectum", "label": 0, "category": "negative_medical_validate"},
    {"text": "lesion ที่ colon", "label": 0, "category": "negative_medical_validate"},
    {"text": "flat lesion", "label": 0, "category": "negative_medical_validate"},
    {"text": "sessile polyp", "label": 0, "category": "negative_medical_validate"},
    {"text": "small polyp", "label": 0, "category": "negative_medical_validate"},
    {"text": "large polyp", "label": 0, "category": "negative_medical_validate"},
    {"text": "cecum reached", "label": 0, "category": "negative_medical_validate"},
    {"text": "ตรวจถึง cecum", "label": 0, "category": "negative_medical_validate"},
    {"text": "ไม่พบ lesion", "label": 0, "category": "negative_medical_validate"},
    {"text": "ไม่พบ bleeding", "label": 0, "category": "negative_medical_validate"},

    # =========================
    # Command negative:
    # คำสั่งทั่วไป แต่ไม่มี wake word
    # =========================
    {"text": "save image", "label": 0, "category": "negative_command_validate"},
    {"text": "capture image", "label": 0, "category": "negative_command_validate"},
    {"text": "save ภาพนี้", "label": 0, "category": "negative_command_validate"},
    {"text": "capture ภาพนี้", "label": 0, "category": "negative_command_validate"},
    {"text": "สร้าง report", "label": 0, "category": "negative_command_validate"},
    {"text": "เปิด report", "label": 0, "category": "negative_command_validate"},
    {"text": "ปิด report", "label": 0, "category": "negative_command_validate"},
    {"text": "ดูภาพล่าสุด", "label": 0, "category": "negative_command_validate"},
    {"text": "บันทึกข้อมูล", "label": 0, "category": "negative_command_validate"},
    {"text": "เพิ่ม finding", "label": 0, "category": "negative_command_validate"},
    {"text": "ลบ finding", "label": 0, "category": "negative_command_validate"},
    {"text": "ตรวจต่อเลย", "label": 0, "category": "negative_command_validate"},
    {"text": "หยุดอัดเสียง", "label": 0, "category": "negative_command_validate"},
    {"text": "เริ่มอัดเสียง", "label": 0, "category": "negative_command_validate"},

    # =========================
    # Normal Thai speech:
    # คำพูดทั่วไป ห้าม trigger
    # =========================
    {"text": "โอเค เริ่มได้", "label": 0, "category": "negative_general_validate"},
    {"text": "เดี๋ยวดูตรงนี้", "label": 0, "category": "negative_general_validate"},
    {"text": "ขยับกล้องนิดนึง", "label": 0, "category": "negative_general_validate"},
    {"text": "ภาพชัดดี", "label": 0, "category": "negative_general_validate"},
    {"text": "ตรงนี้ดูปกติ", "label": 0, "category": "negative_general_validate"},
    {"text": "ขอเวลาสักครู่", "label": 0, "category": "negative_general_validate"},
    {"text": "เตรียมเครื่องมือ", "label": 0, "category": "negative_general_validate"},
    {"text": "ไปต่อได้เลย", "label": 0, "category": "negative_general_validate"},

    # =========================
    # Noise / silence:
    # ห้าม trigger
    # =========================
    {"text": "silence", "label": 0, "category": "noise_validate"},
    {"text": "background noise", "label": 0, "category": "noise_validate"},
    {"text": "room noise", "label": 0, "category": "noise_validate"},
    {"text": "keyboard sound", "label": 0, "category": "noise_validate"},
    {"text": "mouse click", "label": 0, "category": "noise_validate"},
    {"text": "chair movement", "label": 0, "category": "noise_validate"},
    {"text": "เสียงห้องตรวจ", "label": 0, "category": "noise_validate"},
    {"text": "เสียงคนคุย", "label": 0, "category": "noise_validate"},
    {"text": "เสียงเครื่องมือ", "label": 0, "category": "noise_validate"},
    {"text": "เสียงแอร์", "label": 0, "category": "noise_validate"},
]

def print_dataset_plan() -> None:
    print("\nAvailable scripts:")

    for i, item in enumerate(DATASET_PLAN, start=1):
        print(
            f"{i:02d}. "
            f"[{item['category']}] "
            f"label={item['label']} | "
            f"{item['text']}"
        )

    print()


def parse_selection(selection: str):
    """
    Return list of (original_index, item)

    Examples:
      all      -> all scripts
      10       -> script number 10
      1,3,10   -> script 1, 3, 10
      1-10     -> script 1 to 10
      1-5,10   -> script 1 to 5 and 10
    """
    selection = selection.strip().lower()

    if selection == "" or selection == "all":
        return list(enumerate(DATASET_PLAN, start=1))

    selected_indices = set()
    parts = selection.split(",")

    for part in parts:
        part = part.strip()

        if part == "":
            continue

        try:
            if "-" in part:
                start_str, end_str = part.split("-", 1)

                start = int(start_str.strip())
                end = int(end_str.strip())

                if start > end:
                    start, end = end, start

                for idx in range(start, end + 1):
                    selected_indices.add(idx)
            else:
                selected_indices.add(int(part))

        except ValueError:
            print(f"[WARN] Invalid selection: {part}. Skipping.")

    selected_plan = []

    for idx in sorted(selected_indices):
        if idx < 1 or idx > len(DATASET_PLAN):
            print(f"[WARN] Script number {idx} is out of range. Skipping.")
            continue

        selected_plan.append((idx, DATASET_PLAN[idx - 1]))

    return selected_plan


def record_audio(duration: float) -> np.ndarray:
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    return audio.flatten()


def save_wav(audio: np.ndarray, filepath: str) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    wav.write(filepath, SAMPLE_RATE, pcm)


def get_next_index(audio_dir: str) -> int:
    if not os.path.exists(audio_dir):
        return 1

    existing = [
        f for f in os.listdir(audio_dir)
        if f.startswith("sample_") and f.endswith(".wav")
    ]

    if not existing:
        return 1

    numbers = []

    for filename in existing:
        try:
            number = int(filename.replace("sample_", "").replace(".wav", ""))
            numbers.append(number)
        except ValueError:
            pass

    return max(numbers) + 1 if numbers else 1


def write_header_if_needed(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "audio_filepath",
                "text",
                "label",
                "category",
                "speaker",
                "take",
            ])


def main() -> None:
    os.makedirs(AUDIO_DIR, exist_ok=True)

    metadata_dir = os.path.dirname(METADATA_CSV)
    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)

    write_header_if_needed(METADATA_CSV)

    speaker = input("Speaker name/id เช่น tatthon_01: ").strip()
    if speaker == "":
        speaker = "unknown"

    try:
        repeat_per_script = int(
            input("Record each selected script how many times? เช่น 3: ").strip()
        )
    except ValueError:
        repeat_per_script = 1

    if repeat_per_script <= 0:
        repeat_per_script = 1

    print_dataset_plan()

    selection = input(
        "Select scripts เช่น 10 หรือ 1,3,10 หรือ 1-10 หรือ all: "
    ).strip()

    selected_plan = parse_selection(selection)

    if len(selected_plan) == 0:
        print("[ERROR] No valid scripts selected.")
        return

    next_index = get_next_index(AUDIO_DIR)

    total = len(selected_plan) * repeat_per_script
    recorded = 0

    print("\nDeepGI Wake Word Dataset Recorder")
    print(f"Audio dir: {AUDIO_DIR}")
    print(f"Metadata: {METADATA_CSV}")
    print(f"Duration: {RECORD_DURATION}s")
    print(f"Sample rate: {SAMPLE_RATE}")
    print(f"Speaker: {speaker}")
    print(f"Selected scripts: {len(selected_plan)}")
    print(f"Repeat per script: {repeat_per_script}")
    print(f"Total recordings: {total}")

    print("\nSelected:")
    for original_idx, item in selected_plan:
        print(
            f"  {original_idx:02d}. "
            f"[{item['category']}] "
            f"label={item['label']} | "
            f"{item['text']}"
        )

    print("\nCommands:")
    print("  Enter = record")
    print("  s     = skip this take")
    print("  q     = quit\n")

    with open(METADATA_CSV, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        # สำคัญ:
        # loop นี้จะอัดซ้ำประโยคเดิมให้ครบก่อน
        # แล้วค่อยไปประโยคถัดไป
        for current_i, (original_idx, item) in enumerate(selected_plan, start=1):
            text = item["text"]
            label = item["label"]
            category = item["category"]

            print("\n" + "=" * 60)
            print(
                f"[{current_i}/{len(selected_plan)}] "
                f"script_no={original_idx} "
                f"category={category} "
                f"label={label}"
            )
            print(f"Text: \"{text}\"")
            print(f"Will record this script {repeat_per_script} time(s).")
            print("=" * 60 + "\n")

            for take in range(1, repeat_per_script + 1):
                filename = f"sample_{next_index:04d}.wav"
                filepath = os.path.join(AUDIO_DIR, filename)

                print(f"Take {take}/{repeat_per_script}")
                print(f"Say / record: \"{text}\"")

                cmd = input(
                    "Press Enter to record, s=skip this take, q=quit: "
                ).strip().lower()

                if cmd == "q":
                    print("\nStopped.")
                    return

                if cmd == "s":
                    print("Skipped this take.\n")
                    continue

                while True:
                    print(f"Recording {RECORD_DURATION}s... ", end="", flush=True)
                    audio = record_audio(RECORD_DURATION)

                    rms = float(np.sqrt(np.mean(audio ** 2)))
                    print(f"done. RMS={rms:.4f}")

                    confirm = input(
                        "Save? Enter=yes, r=redo this take, s=skip this take, q=quit: "
                    ).strip().lower()

                    if confirm == "q":
                        print("\nStopped.")
                        return

                    if confirm == "r":
                        print("Redoing this take...\n")
                        continue

                    if confirm == "s":
                        print("Skipped this take.\n")
                        break

                    save_wav(audio, filepath)

                    writer.writerow([
                        filepath,
                        text,
                        label,
                        category,
                        speaker,
                        take,
                    ])

                    csv_file.flush()

                    recorded += 1
                    next_index += 1

                    print(f"Saved: {filepath}\n")
                    break

    print(f"\nDone. Recorded {recorded} samples.")
    print(f"Metadata saved to: {METADATA_CSV}")


if __name__ == "__main__":
    main()