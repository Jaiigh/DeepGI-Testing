import platform
import subprocess


def speak(text: str, voice: str = None) -> None:
    """Speak text using the native TTS system on macOS or Windows."""

    system = platform.system()

    if system == "Darwin":
        # macOS
        # Default voice can be overridden with the voice argument.
        cmd = ["say"]

        if voice:
            cmd.extend(["-v", voice])

        cmd.append(text)
        subprocess.run(cmd, check=False)

    elif system == "Windows":
        # Windows PowerShell + System.Speech
        escaped_text = text.replace("'", "''")

        ps_command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speak.Speak('{escaped_text}')"
        )

        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_command],
            check=False
        )

    else:
        # Linux / other systems
        print(f"[TTS] Unsupported operating system: {system}")
        print(f"[TTS] Text: {text}")
