"""Optional local Qwen3-ASR backend.

The qwen-asr package and model weights are loaded only when this backend is
selected, so the existing Whisper-only setup remains unchanged.
"""

from __future__ import annotations

import numpy as np

import config


class QwenASRTranscriber:
    """Lazy wrapper around the official Qwen3-ASR transformers backend."""

    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                import torch
                from qwen_asr import Qwen3ASRModel
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen ASR is not installed. Run `pip install -U qwen-asr` "
                    "in this project's virtual environment."
                ) from exc

            try:
                dtype = getattr(torch, config.QWEN_ASR_DTYPE)
            except AttributeError as exc:
                raise ValueError(
                    f"Unsupported QWEN_ASR_DTYPE: {config.QWEN_ASR_DTYPE!r}"
                ) from exc

            print(f"[ASR] Loading Qwen ASR {config.QWEN_ASR_MODEL} on "
                  f"{config.QWEN_ASR_DEVICE}...")
            self._model = Qwen3ASRModel.from_pretrained(
                config.QWEN_ASR_MODEL,
                dtype=dtype,
                device_map=config.QWEN_ASR_DEVICE,
                max_inference_batch_size=1,
                max_new_tokens=256,
            )
        return self._model

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        """Return text for a mono float waveform at ``sample_rate`` Hz."""
        result = self._get_model().transcribe(
            audio=(np.asarray(audio, dtype=np.float32), sample_rate),
            language="English" if config.LANGUAGE == "en" else None,
        )
        # The official API returns a list of ASRTranscription objects.
        return result[0].text.strip()
