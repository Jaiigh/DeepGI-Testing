"""Evaluate base vs. fine-tuned models: WER, trigger accuracy, and latency."""

import csv
import os
import sys
import time

import numpy as np
import scipy.io.wavfile as wav

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

METADATA_CSV = "training_data/metadata.csv"
RESULTS_FILE = "outputs/evaluate_results.txt"
TRIGGER_KEYWORDS = ["hey deepgi", "hey deep gi", "hey dgi", "hey deepgee"]


def load_test_set():
    """Load all samples from metadata.csv as (audio_path, transcript) pairs."""
    if not os.path.exists(METADATA_CSV):
        print(f"[ERROR] {METADATA_CSV} not found. Record samples first.")
        sys.exit(1)
    samples = []
    with open(METADATA_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((row["audio_filepath"], row["text"].strip().lower()))
    return samples


def load_audio(path: str) -> np.ndarray:
    rate, data = wav.read(path)
    audio = data.astype(np.float32) / 32767.0
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def evaluate_whisper(model, samples: list, label: str) -> dict:
    import jiwer

    latencies = []
    references = []
    hypotheses = []

    for path, transcript in samples:
        if not os.path.exists(path):
            continue
        audio = load_audio(path)
        t0 = time.time()
        result = model.transcribe(
            audio,
            language=config.LANGUAGE,
            initial_prompt=config.INITIAL_PROMPT,
            fp16=config.FP16,
        )
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)
        references.append(transcript)
        hypotheses.append(result["text"].strip().lower())

    wer = jiwer.wer(references, hypotheses) if references else float("nan")
    return {
        "label": label,
        "samples": len(latencies),
        "wer": wer,
        "avg_latency_ms": np.mean(latencies) if latencies else 0.0,
    }


def evaluate_trigger_accuracy(model, samples: list, label: str) -> dict:
    import difflib

    true_positives = 0
    false_negatives = 0
    true_negatives = 0
    false_positives = 0

    for path, transcript in samples:
        if not os.path.exists(path):
            continue
        audio = load_audio(path)
        result = model.transcribe(
            audio,
            language=config.LANGUAGE,
            initial_prompt=config.INITIAL_PROMPT,
            fp16=config.FP16,
        )
        heard = result["text"].strip().lower()
        is_trigger_sample = any(kw in transcript for kw in TRIGGER_KEYWORDS)

        # Check if model detected a trigger
        detected = any(kw in heard for kw in TRIGGER_KEYWORDS)
        if not detected:
            for kw in TRIGGER_KEYWORDS:
                if difflib.SequenceMatcher(None, heard, kw).ratio() >= config.FUZZY_THRESHOLD:
                    detected = True
                    break

        if is_trigger_sample and detected:
            true_positives += 1
        elif is_trigger_sample and not detected:
            false_negatives += 1
        elif not is_trigger_sample and detected:
            false_positives += 1
        else:
            true_negatives += 1

    total_triggers = true_positives + false_negatives
    accuracy = (true_positives / total_triggers * 100) if total_triggers > 0 else 0.0
    return {"label": label, "accuracy_pct": accuracy, "tp": true_positives, "fn": false_negatives}


def print_table(rows: list[dict], headers: list[str]) -> str:
    col_widths = [max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, col_widths)) + "|"
    lines = [sep, header_row, sep]
    for row in rows:
        line = "|" + "|".join(f" {str(row.get(h, '')):<{w}} " for h, w in zip(headers, col_widths)) + "|"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def main():
    import whisper

    samples = load_test_set()
    if not samples:
        print("[ERROR] No samples found in metadata.csv.")
        sys.exit(1)

    print(f"Loaded {len(samples)} samples from {METADATA_CSV}.\n")
    results_lines = [f"DeepGI ASR Evaluation — {len(samples)} samples\n"]

    # --- ASR: Base Whisper ---
    print(f"[1/3] Loading base Whisper ({config.WHISPER_MODEL}) for WER evaluation...")
    base_model = whisper.load_model(config.WHISPER_MODEL)
    base_asr = evaluate_whisper(base_model, samples, f"Base Whisper ({config.WHISPER_MODEL})")

    asr_rows = [{"Model": base_asr["label"], "Samples": base_asr["samples"],
                 "WER": f"{base_asr['wer']:.2%}", "Avg Latency (ms)": f"{base_asr['avg_latency_ms']:.0f}"}]

    # --- ASR: Fine-tuned (if available) ---
    if os.path.isdir(config.FINETUNED_ASR_PATH):
        print(f"[2/3] Loading fine-tuned ASR from {config.FINETUNED_ASR_PATH}...")
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        import torch

        ft_processor = WhisperProcessor.from_pretrained(config.FINETUNED_ASR_PATH)
        ft_model_hf = WhisperForConditionalGeneration.from_pretrained(config.FINETUNED_ASR_PATH)
        ft_model_hf.eval()

        latencies, references, hypotheses = [], [], []
        for path, transcript in samples:
            if not os.path.exists(path):
                continue
            audio = load_audio(path)
            inputs = ft_processor(audio, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
            t0 = time.time()
            with torch.no_grad():
                ids = ft_model_hf.generate(inputs["input_features"])
            latency_ms = (time.time() - t0) * 1000
            text = ft_processor.batch_decode(ids, skip_special_tokens=True)[0].strip().lower()
            latencies.append(latency_ms)
            references.append(transcript)
            hypotheses.append(text)

        import jiwer
        ft_wer = jiwer.wer(references, hypotheses) if references else float("nan")
        asr_rows.append({
            "Model": f"Fine-tuned ({config.FINETUNED_ASR_PATH})",
            "Samples": len(latencies),
            "WER": f"{ft_wer:.2%}",
            "Avg Latency (ms)": f"{np.mean(latencies):.0f}" if latencies else "N/A",
        })
    else:
        print(f"[2/3] Fine-tuned ASR model not found at {config.FINETUNED_ASR_PATH} — skipping.")

    # --- VAD trigger accuracy ---
    print("[3/3] Evaluating trigger detection accuracy (base Whisper)...")
    trig = evaluate_trigger_accuracy(base_model, samples, f"Base Whisper ({config.WHISPER_MODEL})")
    trig_rows = [{"Model": trig["label"], "Trigger Accuracy": f"{trig['accuracy_pct']:.1f}%",
                  "True Pos": trig["tp"], "False Neg": trig["fn"]}]

    # --- Print results ---
    asr_table = print_table(asr_rows, ["Model", "Samples", "WER", "Avg Latency (ms)"])
    trig_table = print_table(trig_rows, ["Model", "Trigger Accuracy", "True Pos", "False Neg"])

    output = "\n".join([
        "=== ASR Performance ===",
        asr_table,
        "",
        "=== VAD Trigger Detection ===",
        trig_table,
    ])
    print("\n" + output)

    os.makedirs("outputs", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(results_lines) + "\n" + output + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
