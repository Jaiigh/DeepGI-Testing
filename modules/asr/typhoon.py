import torch
from transformers import pipeline


MODEL_ID = "typhoon-ai/typhoon-whisper-turbo"


class TyphoonASR:
    def __init__(self):
        if torch.cuda.is_available():
            device = 0
            torch_dtype = torch.float16
            print("[ASR] Using CUDA")
        else:
            device = -1
            torch_dtype = torch.float32
            print("[ASR] Using CPU")

        print(f"[ASR] Loading Typhoon Whisper Turbo...")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            device=device,
            torch_dtype=torch_dtype,
        )

        print("[ASR] Typhoon loaded.")

    def transcribe(self, audio):
        result = self.pipe(
            audio,
            generate_kwargs={
                "language": "thai",
                "task": "transcribe",
            }
        )

        return result["text"].strip()