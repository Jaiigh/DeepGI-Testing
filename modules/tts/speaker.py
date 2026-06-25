import subprocess


def speak(text: str, voice: str = "Samantha") -> None:
    """Print text to console and speak it using the macOS say command."""
    print(f"[TTS] {text}")
    subprocess.run(["say", "-v", voice, text], check=False)
