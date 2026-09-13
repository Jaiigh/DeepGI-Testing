import os
import sys
import json
from datetime import datetime

from modules.voice_activation.detector import wait_for_trigger
from modules.asr.transcriber import transcribe_finding
from modules.tts.speaker import speak, warmup
from modules.llm.qwen_lora import extract_finding, warmup as warmup_llm
import config


def _session_log_path() -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("outputs/reports", exist_ok=True)
    return f"outputs/reports/session_{date_str}.txt"


def _append_finding(finding: str, structured: dict) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    os.makedirs("outputs/reports", exist_ok=True)
    path = _session_log_path().replace(".txt", ".jsonl")
    record = {"timestamp": timestamp, "transcription": finding, "result": structured}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _spoken_summary(result: dict) -> str:
    parts = []
    if result.get("lesion_type"):
        parts.append(str(result["lesion_type"]))
    if result.get("location"):
        parts.append(f"in {result['location']}")
    if result.get("size_mm") is not None:
        parts.append(f"{result['size_mm']} millimeters")
    if result.get("procedure"):
        parts.append(f"procedure {result['procedure']}")
    if result.get("pathology"):
        parts.append("sent to pathology")
    return " ".join(parts) if parts else "Finding processed."


def run_pipeline() -> None:
    try:
        print("[TTS] Loading Kokoro...")
        warmup()
        print("[LLM] Loading Qwen + LoRA...")
        warmup_llm()

        while True:
            print("\n👂 Listening for 'Hey DeepGI'...")
            wait_for_trigger()
            print(f"🔴 Trigger detected! Recording finding ({config.FINDING_DURATION} seconds)...")
            speak("Please say your finding.")
            finding = transcribe_finding()
            print(f"📝 Finding: {finding}")
            if not finding:
                print("[LLM] ASR returned empty text; skipping LLM extraction.")
                speak("I did not hear a finding. Please try again.")
                continue
            print("[LLM] Extracting structured JSON...")
            structured = extract_finding(finding)
            print(json.dumps(structured, ensure_ascii=False, indent=2))
            speak(_spoken_summary(structured))
            _append_finding(finding, structured)
            print("👂 Listening again...")
    except KeyboardInterrupt:
        print("\nSession ended.")
        speak("Session ended.")


if __name__ == "__main__":
    run_pipeline()
