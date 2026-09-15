import os
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


class FindingProcessingError(RuntimeError):
    def __init__(self, stage, transcription, cause):
        super().__init__(str(cause))
        self.stage = stage
        self.transcription = transcription


def process_finding(on_status=None, on_result=None, strict=False):
    """Capture one finding; acknowledge/persist the result before speaking it.

    on_result(text, result) may return False to suppress spoken feedback when
    persistence fails. Empty ASR returns None. Errors retain stage and text.
    """
    status = on_status or (lambda value: print(f"[Voice] {value}"))
    stage, finding = "Prompting", ""
    try:
        status(stage)
        speak("Please say your finding.")
        stage = "Recording"
        status(stage)

        def asr_status(value):
            nonlocal stage
            stage = value
            status(value)

        finding = transcribe_finding(on_status=asr_status)
        if not finding:
            status("No finding heard. Say Hey DeepGI to try again.")
            return None
        stage = "Extracting"
        status(stage)
        result = extract_finding(finding, strict=strict)
        stage = "Saving"
        saved = (on_result or _append_finding)(finding, result)
        if saved is not False:
            stage = "Speaking"
            status(stage)
            speak(_spoken_summary(result))
        return {"transcription": finding, "result": result}
    except Exception as exc:
        raise FindingProcessingError(stage, finding, exc) from exc


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
            try:
                process_finding()
            except FindingProcessingError as exc:
                print(f"[ERROR] {exc.stage}: {exc}")
            print("👂 Listening again...")
    except KeyboardInterrupt:
        print("\nSession ended.")
        speak("Session ended.")


if __name__ == "__main__":
    run_pipeline()
