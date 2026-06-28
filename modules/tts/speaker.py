import sounddevice as sd

import config


_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline

        _pipeline = KPipeline(lang_code=config.KOKORO_LANG_CODE)
    return _pipeline


def speak(text: str, voice: str = "Samantha") -> None:
    """Print text to console and speak it using Kokoro TTS."""
    print(f"[TTS] {text}")
    pipeline = _get_pipeline()

    for _, _, audio in pipeline(text, voice=config.KOKORO_VOICE):
        sd.play(audio, config.KOKORO_SAMPLE_RATE)
        sd.wait()
