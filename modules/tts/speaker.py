import subprocess

import config

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        import sounddevice as sd
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code=config.KOKORO_LANG_CODE)
    return _pipeline


def speak(text: str, voice: str = "Samantha") -> None:
    print(f"[TTS] {text}")
    if config.USE_KOKORO_TTS:
        import sounddevice as sd
        pipeline = _get_pipeline()
        for _, _, audio in pipeline(text, voice=config.KOKORO_VOICE):
            sd.play(audio, config.KOKORO_SAMPLE_RATE)
            sd.wait()
    else:
        subprocess.run(["say", "-v", voice, text], check=False)
