# Model settings
WHISPER_MODEL = "medium"          # tiny, base, small, medium, large-v3
TRIGGER_WHISPER_MODEL = "tiny"  # tiny, base, small, medium, large-v3
LANGUAGE = "en"                  # Language for the spoken colonoscopy finding
TRIGGER_LANGUAGE = "en"           # Wake phrase is "Hey DeepGI"
ASR_DEVICE = "cpu"             # ASR/Whisper device: "cuda", "cuda:0", or "cpu"
FP16 = False                      # Use half precision for Whisper on CUDA

# ASR backend. Leave this as "whisper" to retain the existing pipeline.
# Set to "qwen" to run Qwen3-ASR instead.
ASR_BACKEND = "whisper"          # "whisper" or "qwen"
QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"  # Official Qwen3-ASR: 0.6B or 1.7B
QWEN_ASR_DEVICE = "cuda:0"       # "cpu", "mps", or e.g. "cuda:0"
QWEN_ASR_DTYPE = "float16"       # "float32", "float16", or "bfloat16"

# Qwen LLM + LoRA inference
# The LoRA adapter in llm_qwen was trained from Qwen/Qwen2.5-3B-Instruct.
LLM_BASE_MODEL_PATH = "models/Qwen2.5-3B-Instruct"
LLM_HF_REPO_ID = "Qwen/Qwen2.5-3B-Instruct"
LLM_LORA_PATH = "llm_qwen"
LLM_DEVICE = "cpu"             # Keep the LLM on GPU; use "cpu" to override
LLM_DTYPE = "float32"              # "auto", "float32", "float16", "bfloat16"
LLM_MAX_NEW_TOKENS = 384
LLM_LOCAL_FILES_ONLY = True       # Load locally after the explicit completeness check

# Wake word
TRIGGER_PHRASE = "hey deepgi"
TRIGGER_VARIANTS = [
    "hey deepgi", "hey deep gi", "hey dgi", "hey d gi",
    "hey deepgee", "hey deep gee", "hey travis", "hey davis", "hey jarvis"
]
FUZZY_THRESHOLD = 0.7

# Audio settings
SAMPLE_RATE = 16000
TRIGGER_CHUNK_DURATION = 3       # seconds to listen for trigger
FINDING_DURATION = 8             # seconds to record after trigger
AUDIO_DEVICE = None              # None = system default (set to device ID if needed)

# TTS settings
USE_KOKORO_TTS = False            # True to use Kokoro neural TTS, False for platform say
KOKORO_LANG_CODE = "a"           # American English
KOKORO_VOICE = "af_heart"
KOKORO_SAMPLE_RATE = 24000

# Medical vocabulary hint for Whisper
INITIAL_PROMPT = """DeepGI colonoscopy findings: polyp, lesion, bleeding,
diverticulum, sigmoid colon, cecum, rectum, ascending colon, descending colon,
transverse colon, hepatic flexure, splenic flexure, Paris classification,
Boston Bowel Score, pedunculated, sessile, flat, biopsy, resection"""

# Training mode toggle
USE_FINETUNED_ASR = False        # True to use fine-tuned model
USE_FINETUNED_VAD = False        # True to use trained wake word model
FINETUNED_ASR_PATH = "outputs/models/whisper-deepgi"
FINETUNED_VAD_PATH = "outputs/models/vad-deepgi"
