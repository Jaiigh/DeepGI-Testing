import numpy as np
import sounddevice as sd
import config

_whisper_model = None
_hf_pipeline = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"[ASR] Loading Whisper {config.WHISPER_MODEL}...")
        _whisper_model = whisper.load_model(config.WHISPER_MODEL)
    return _whisper_model


def _get_hf_pipeline():
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch
        print(f"[ASR] Loading fine-tuned model from {config.FINETUNED_ASR_PATH}...")
        processor = WhisperProcessor.from_pretrained(config.FINETUNED_ASR_PATH)
        model = WhisperForConditionalGeneration.from_pretrained(config.FINETUNED_ASR_PATH)
        model.eval()
        _hf_pipeline = (processor, model)
    return _hf_pipeline


def _record_audio(duration: float) -> np.ndarray:
    audio = sd.rec(
        int(duration * config.SAMPLE_RATE),
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def transcribe_finding() -> str:
    """Record FINDING_DURATION seconds and return the transcribed text."""
    audio = _record_audio(config.FINDING_DURATION)

    if config.USE_FINETUNED_ASR:
        import torch
        processor, model = _get_hf_pipeline()
        inputs = processor(
            audio,
            sampling_rate=config.SAMPLE_RATE,
            return_tensors="pt",
        )
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                forced_decoder_ids=processor.get_decoder_prompt_ids(
                    language=config.LANGUAGE, task="transcribe"
                ),
            )
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    else:
        model = _get_whisper_model()
        result = model.transcribe(
            audio,
            language=config.LANGUAGE,
            initial_prompt=config.INITIAL_PROMPT,
            fp16=config.FP16,
        )
        text = result["text"]

    return text.strip()
