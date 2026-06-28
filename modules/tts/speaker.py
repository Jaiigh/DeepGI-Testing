import subprocess
import sys


def _tts_command(text: str, voice: str) -> list[str] | None:
    if sys.platform == "darwin":
        return ["say", "-v", voice, text]
    if sys.platform == "win32":
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speaker.Speak($args[0])"
        )
        return ["powershell", "-NoProfile", "-Command", command, text]
    return None


def speak(text: str, voice: str = "Samantha") -> None:
    """Print text and speak it when a platform TTS command is available."""
    print(f"[TTS] {text}")
    command = _tts_command(text, voice)
    if command is None:
        return
    try:
        subprocess.run(command, check=False)
    except (OSError, subprocess.SubprocessError):
        return
