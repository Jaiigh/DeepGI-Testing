import sys
import config
import pipeline


def _print_banner() -> None:
    asr_mode = f"fine-tuned ({config.FINETUNED_ASR_PATH})" if config.USE_FINETUNED_ASR else f"base Whisper ({config.WHISPER_MODEL})"
    vad_mode = f"fine-tuned ({config.FINETUNED_VAD_PATH})" if config.USE_FINETUNED_VAD else f"base Whisper ({config.WHISPER_MODEL})"
    print("=" * 56)
    print("  DeepGI ASR Demo | VAD + ASR + TTS Pipeline")
    print("  Say 'Hey DeepGI' to activate")
    print("  Press Ctrl+C to end session")
    print("=" * 56)
    print(f"  ASR model  : {asr_mode}")
    print(f"  VAD model  : {vad_mode}")
    print(f"  Trigger    : {config.TRIGGER_PHRASE!r}")
    print(f"  Language   : {config.LANGUAGE}")
    print(f"  Sample rate: {config.SAMPLE_RATE} Hz")
    print("=" * 56)
    print()


if __name__ == "__main__":
    try:
        _print_banner()
        pipeline.run_pipeline()
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("Check that model paths in config.py are correct.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        print("Check your microphone permissions and audio settings.")
        sys.exit(1)
