import subprocess

import config

_pipeline = None
_audio_cache = {}


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        import sounddevice as sd
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code=config.KOKORO_LANG_CODE)
    return _pipeline


def _get_audio(text: str):
    """Render text once per process and reuse the generated audio."""
    if text not in _audio_cache:
        pipeline = _get_pipeline()
        _audio_cache[text] = [audio for _, _, audio in pipeline(
            text,
            voice=config.KOKORO_VOICE,
        )]
    return _audio_cache[text]


def warmup() -> None:
    """Load Kokoro and pre-render the first fixed response before listening."""
    if config.USE_KOKORO_TTS:
        _get_audio("Please say your finding.")


def speak(text: str, voice: str = "Samantha") -> None:
    print(f"[TTS] {text}")
    if config.USE_KOKORO_TTS:
        import sounddevice as sd
        for audio in _get_audio(text):
            sd.play(audio, config.KOKORO_SAMPLE_RATE)
            sd.wait()
    else:
        subprocess.run(["say", "-v", voice, text], check=False)
