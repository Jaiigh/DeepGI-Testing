import os
import sys
from datetime import datetime

from modules.voice_activation.detector import wait_for_trigger
from modules.asr.transcriber import transcribe_finding
from modules.tts.speaker import speak
import config


def _session_log_path() -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("outputs/reports", exist_ok=True)
    return f"outputs/reports/session_{date_str}.txt"


def _append_finding(finding: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(_session_log_path(), "a") as f:
        f.write(f"[{timestamp}] {finding}\n")


def run_pipeline() -> None:
    try:
        while True:
            print("\n👂 Listening for 'Hey DeepGI'...")
            wait_for_trigger()
            print(f"🔴 Trigger detected! Recording finding ({config.FINDING_DURATION} seconds)...")
            speak("Trigger detected. Please say your finding.")
            finding = transcribe_finding()
            print(f"📝 Finding: {finding}")
            speak(f"Finding logged: {finding}")
            _append_finding(finding)
            print("👂 Listening again...")
    except KeyboardInterrupt:
        print("\nSession ended.")
        speak("Session ended.")
