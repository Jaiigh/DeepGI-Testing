# Model settings
WHISPER_MODEL = "small"          # tiny, base, small, medium, large-v3
LANGUAGE = "en"
FP16 = False                     # False for macOS CPU

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

# Medical vocabulary hint for Whisper
INITIAL_PROMPT = """DeepGI colonoscopy findings: polyp, lesion, bleeding,
diverticulum, sigmoid colon, cecum, rectum, ascending colon, descending colon,
transverse colon, hepatic flexure, splenic flexure, Paris classification,
Boston Bowel Score, pedunculated, sessile, flat, biopsy, resection"""

# Training mode toggle
USE_FINETUNED_ASR = False        # True to use fine-tuned model
USE_FINETUNED_VAD = True       # True to use trained wake word model
FINETUNED_ASR_PATH = "outputs/models/whisper-deepgi"
FINETUNED_VAD_PATH = "outputs/models/vad-deepgi"
