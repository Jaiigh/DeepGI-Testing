import collections
import difflib
import os
import queue
import tempfile
import wave

import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import webrtcvad

import config


# =========================
# Global state
# =========================

DEVICE = "cpu"

_whisper_model = None

_vad_model = None
_vad_mfcc_transform = None
_vad_threshold = None
_vad_n_samples = None
_vad_sample_rate = None

audio_queue = queue.Queue()


# =========================
# Audio / VAD config
# =========================

SAMPLE_RATE = config.SAMPLE_RATE
CHANNELS = 1

# WebRTC VAD รองรับเฉพาะ 10, 20, 30 ms
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

PRE_ROLL_SECONDS = 0.30
END_SILENCE_SECONDS = 0.50
MAX_TRIGGER_SECONDS = config.TRIGGER_CHUNK_DURATION

PRE_ROLL_FRAMES = int(PRE_ROLL_SECONDS * 1000 / FRAME_MS)
END_SILENCE_FRAMES = int(END_SILENCE_SECONDS * 1000 / FRAME_MS)
MAX_TRIGGER_FRAMES = int(MAX_TRIGGER_SECONDS * 1000 / FRAME_MS)

SILENCE_RMS_THRESHOLD = 0.01

VAD_MODE = 2
vad = webrtcvad.Vad(VAD_MODE)


# =========================
# Sounddevice callback
# =========================

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)

    # RawInputStream ให้ข้อมูลเป็น bytes-like object
    audio_queue.put(bytes(indata))


def clear_audio_queue():
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


# =========================
# CNN Wake Word Model
# =========================

class WakeWordCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # input: (batch, 1, 40, time)

            # มองกว้างขึ้นจากเดิม 3x5 เป็น 5x9
            nn.Conv2d(1, 32, kernel_size=(5, 9), padding=(2, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # มอง pattern ใหญ่ขึ้นอีก
            nn.Conv2d(32, 64, kernel_size=(5, 9), padding=(2, 4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # ใช้ dilation ให้มองแกนเวลาไกลขึ้น
            # kernel จริงเหมือนมองประมาณ 3 x 13
            nn.Conv2d(
                64,
                128,
                kernel_size=(3, 7),
                padding=(1, 6),
                dilation=(1, 2),
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # มองทั้ง frequency และ time กว้างขึ้นอีก
            # kernel จริงประมาณ 5 x 13
            nn.Conv2d(
                128,
                128,
                kernel_size=(3, 7),
                padding=(2, 6),
                dilation=(2, 2),
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Dropout2d(0.20),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (batch, 1, 40, time)
        x = self.net(x)
        x = x.flatten(1)
        logits = self.classifier(x).squeeze(1)
        return logits


def load_finetuned_vad_model():
    global _vad_model, _vad_mfcc_transform, _vad_threshold, _vad_n_samples, _vad_sample_rate

    if _vad_model is not None:
        return _vad_model

    model_path = os.path.join(config.FINETUNED_VAD_PATH, "classifier_cnn.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"CNN model not found at {model_path}. "
            f"Train it first by running training/train_vad.ipynb in Google Colab, "
            f"then download classifier_cnn.pt to {config.FINETUNED_VAD_PATH}/."
        )

    print(f"[VAD] Loading CNN model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        _vad_threshold = checkpoint.get("threshold", 0.65)
        sr = checkpoint.get("sample_rate", config.SAMPLE_RATE)
        clip_sec = checkpoint.get("clip_seconds", 3.0)
        n_mfcc = checkpoint.get("n_mfcc", 40)
    else:
        state_dict = checkpoint
        _vad_threshold = 0.65
        sr = config.SAMPLE_RATE
        clip_sec = 3.0
        n_mfcc = 40

    _vad_sample_rate = sr
    _vad_n_samples = int(sr * clip_sec)

    model = WakeWordCNN().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    _vad_model = model

    _vad_mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
    )

    print(f"[VAD] CNN model loaded. threshold={_vad_threshold:.2f}, sr={sr}, clip={clip_sec}s")
    return _vad_model


# =========================
# Audio helpers
# =========================

def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """
    แปลง raw int16 PCM bytes เป็น float32 ช่วงประมาณ -1.0 ถึง 1.0
    """
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32


def save_temp_wav(audio_bytes: bytes) -> str:
    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    with wave.open(temp.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_bytes)

    return temp.name


def calc_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


# =========================
# CNN prediction
# =========================

def predict_wake_word_from_audio(audio: np.ndarray):
    global _vad_model, _vad_mfcc_transform, _vad_threshold, _vad_n_samples

    if _vad_model is None:
        load_finetuned_vad_model()

    audio = np.asarray(audio, dtype=np.float32)
    tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # (1, samples)

    # Pad or trim to the fixed clip length the CNN was trained on
    n = tensor.shape[-1]
    if n < _vad_n_samples:
        tensor = F.pad(tensor, (0, _vad_n_samples - n))
    else:
        tensor = tensor[:, :_vad_n_samples]

    mfcc = _vad_mfcc_transform(tensor)              # (1, n_mfcc, time)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)  # same normalization as training
    x = mfcc.unsqueeze(0).to(DEVICE)                # (1, 1, n_mfcc, time)

    with torch.no_grad():
        logits = _vad_model(x)
        confidence = torch.sigmoid(logits).item()

    is_trigger = confidence >= _vad_threshold
    return is_trigger, confidence


def process_voiced_frames(voiced_frames):
    """
    เอา voiced_frames ที่เก็บจาก WebRTC VAD มา save เป็น wav
    แล้วส่งเข้า CNN wake word classifier
    """
    if not voiced_frames:
        return False

    audio_bytes = b"".join(voiced_frames)

    # save voice segment ก่อนส่งเข้า CNN

    # แปลง bytes -> float32 เพื่อเข้า CNN
    audio = pcm16_bytes_to_float32(audio_bytes)

    rms = calc_rms(audio)

    if rms < SILENCE_RMS_THRESHOLD:
        print(f"[VAD] Skip quiet segment. rms={rms:.5f}")
        return False

    is_trigger, confidence = predict_wake_word_from_audio(audio)

    if is_trigger:
        print(f"[VAD] Wake word detected! confidence={confidence:.1%}")
        os.makedirs(DEBUG_VOICE_DIR, exist_ok=True)
        audio_path = os.path.join(DEBUG_VOICE_DIR, f"voice_segment.wav")

        with wave.open(audio_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)

        print(f"[VAD] Saved voice segment before CNN: {audio_path}")
        print(f"[VAD] CNN input audio: {audio_path}")
        return True

    print(f"[VAD] Not trigger. confidence={confidence:.1%}")
    return False


# =========================
# Whisper fallback
# =========================

def _get_whisper_model():
    global _whisper_model

    if _whisper_model is None:
        import whisper
        print(f"[VAD] Loading Whisper {config.WHISPER_MODEL} for trigger detection...")
        _whisper_model = whisper.load_model(config.WHISPER_MODEL)

    return _whisper_model


def _record_audio(duration: float) -> np.ndarray:
    audio = sd.rec(
        int(duration * config.SAMPLE_RATE),
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=config.AUDIO_DEVICE,
    )
    sd.wait()
    return audio.flatten()


def _is_trigger(text: str) -> bool:
    text = text.strip().lower()

    for variant in config.TRIGGER_VARIANTS:
        if variant in text:
            return True

    for variant in config.TRIGGER_VARIANTS:
        ratio = difflib.SequenceMatcher(None, text, variant).ratio()
        if ratio >= config.FUZZY_THRESHOLD:
            return True

    return False


def _wait_for_trigger_base():
    model = _get_whisper_model()

    while True:
        audio = _record_audio(config.TRIGGER_CHUNK_DURATION)

        rms = calc_rms(audio)

        if rms < SILENCE_RMS_THRESHOLD:
            continue

        result = model.transcribe(
            audio,
            language=config.LANGUAGE,
            fp16=config.FP16,
            condition_on_previous_text=False,
        )

        heard = result["text"]
        print(f"[VAD] Heard: {heard.strip()!r}")

        if _is_trigger(heard):
            print("[VAD] Trigger detected!")
            return


# =========================
# Finetuned VAD flow
# =========================
DEBUG_VOICE_DIR = "debug_voice_frames"
def _wait_for_trigger_finetuned():
    load_finetuned_vad_model()

    if FRAME_MS not in (10, 20, 30):
        raise ValueError("FRAME_MS must be 10, 20, or 30 for WebRTC VAD.")

    if SAMPLE_RATE not in (8000, 16000, 32000, 48000):
        raise ValueError("SAMPLE_RATE must be 8000, 16000, 32000, or 48000 for WebRTC VAD.")

    ring_buffer = collections.deque(maxlen=PRE_ROLL_FRAMES)
    voiced_frames = []

    triggered = False
    silence_count = 0

    clear_audio_queue()

    print('[VAD] Listening for wake word...')

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=CHANNELS,
        device=config.AUDIO_DEVICE,
        callback=audio_callback,
    ):
        while True:
            frame = audio_queue.get()

            try:
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
            except Exception as e:
                print(f"[VAD] vad.is_speech error: {e}")
                continue

            if not triggered:
                ring_buffer.append(frame)

                if is_speech:
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
                    silence_count = 0
                    print("[VAD] Speech started.")

            else:
                voiced_frames.append(frame)

                if is_speech:
                    silence_count = 0
                else:
                    silence_count += 1

                too_much_silence = silence_count >= END_SILENCE_FRAMES
                too_long = len(voiced_frames) >= MAX_TRIGGER_FRAMES

                if too_much_silence or too_long:
                    print("[VAD] Speech ended.")

                    found = process_voiced_frames(voiced_frames)

                    triggered = False
                    voiced_frames = []
                    silence_count = 0
                    ring_buffer.clear()

                    if found:
                        return


# =========================
# Public function
# =========================

def wait_for_trigger():
    """
    Block until the trigger phrase is detected, then return.
    """
    if getattr(config, "USE_FINETUNED_VAD", False):
        _wait_for_trigger_finetuned()
    else:
        _wait_for_trigger_base()