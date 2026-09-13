"""Qwen2.5 + PEFT LoRA inference for structured DeepGI findings."""

import json
import os
import re
from pathlib import Path
from typing import Any

import config


OUTPUT_FIELDS = (
    "lesion_type",
    "location",
    "size_mm",
    "procedure",
    "biopsy_forceps",
    "biopsy_pieces",
    "pathology",
)

EXTRACTION_SYSTEM_PROMPT = r'''คุณเป็นระบบ Information Extraction สำหรับรายงาน colonoscopy ภาษาไทย

อ่านข้อความเพียง 1 ข้อความ และคืน JSON object เท่านั้น

ต้องคืน EXACTLY 7 fields ต่อไปนี้:

{
  "lesion_type": null,
  "location": null,
  "size_mm": null,
  "procedure": null,
  "biopsy_forceps": false,
  "biopsy_pieces": null,
  "pathology": false
}

กฎสำคัญมาก:

1. ห้ามอนุมาน ห้ามเดา ห้ามใช้ความรู้ทางการแพทย์เติมข้อมูล
2. ใช้เฉพาะสิ่งที่ปรากฏชัดเจนในข้อความ
3. ถ้าไม่มีข้อมูล ให้ใช้ค่า default ตาม schema ด้านบน
4. ห้ามใช้ null กับ biopsy_forceps และ pathology
   สอง field นี้ต้องเป็น true หรือ false เท่านั้น

lesion_type:
- ถ้ามีคำว่า "sessile polyp" -> "sessile polyp"
- ถ้ามีคำว่า "polyp" -> "polyp"
- ถ้ามีคำว่า "lesion" -> "lesion"
- ถ้าไม่ได้พูดคำเหล่านี้ -> null
- ห้ามเดา lesion_type จาก procedure

location:
- ascending colon
- transverse colon
- descending colon
- sigmoid colon
- rectum
- ถ้าไม่ระบุ -> null

size_mm:
- คืน integer หน่วย mm
- 1 cm = 10 mm
- ถ้าไม่ระบุ -> null

procedure:
- ถ้ามีคำว่า cold snare -> "cold snare"
- ถ้ามีคำว่า hot snare -> "hot snare"
- ถ้ามีคำว่า biopsy หรือ "ขอ biopsy" -> "biopsy"
- ถ้าพูดแค่ว่า "ตัดออกแล้ว" แต่ไม่ได้บอกวิธี -> null
- ห้ามเดา cold snare หรือ hot snare จากขนาดของ lesion

biopsy_forceps:
- true เฉพาะเมื่อมีคำว่า forceps ชัดเจน
- ถ้าไม่มีคำว่า forceps -> false
- ห้ามคืน null

biopsy_pieces:
- ถ้ามีจำนวนชิ้น biopsy ให้คืนจำนวนนั้น
- ไม่ระบุ -> null

pathology:
- true เฉพาะเมื่อมีคำว่า pathology หรือระบุว่าส่งชิ้นเนื้อไปตรวจ
- ถ้าไม่มี -> false
- ห้ามคืน null

SELF-CORRECTION:
เมื่อผู้พูดแก้ข้อมูล ให้ทิ้งค่าก่อนหน้าและใช้ค่าหลังสุดเท่านั้น

ตอบ JSON object เท่านั้น
ห้าม markdown
ห้ามคำอธิบาย'''


class QwenLoraExtractor:
    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None

    @staticmethod
    def _ensure_base_model(base_path: str) -> None:
        """Download missing Qwen base files into the configured D: drive folder."""
        base_dir = Path(base_path)
        base_dir.mkdir(parents=True, exist_ok=True)
        index_path = base_dir / "model.safetensors.index.json"
        shard_files = list(base_dir.glob("model-*-of-*.safetensors"))
        # Git LFS pointer files are only a few hundred bytes and do not count.
        has_weights = (
            index_path.exists()
            and bool(shard_files)
            and all(path.stat().st_size > 1_000_000 for path in shard_files)
        )
        if has_weights:
            return

        print(f"[LLM] Base model is incomplete; downloading {config.LLM_HF_REPO_ID}...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=config.LLM_HF_REPO_ID,
                local_dir=str(base_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for automatic model download. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        shard_files = list(base_dir.glob("model-*-of-*.safetensors"))
        if (
            not index_path.exists()
            or not shard_files
            or not all(path.stat().st_size > 1_000_000 for path in shard_files)
        ):
            raise RuntimeError(
                f"Qwen download did not complete. Check the model folder: {base_dir}"
            )

    def _load(self) -> None:
        if self._model is not None:
            return

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        base_path = os.path.abspath(config.LLM_BASE_MODEL_PATH)
        adapter_path = os.path.abspath(config.LLM_LORA_PATH)
        self._ensure_base_model(base_path)
        if not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

        dtype = config.LLM_DTYPE
        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        if config.LLM_DEVICE == "cpu":
            device_map = None
        elif config.LLM_DEVICE == "auto":
            device_map = "auto"
        else:
            # Transformers expects a module-to-device map for a fixed GPU.
            device_map = {"": config.LLM_DEVICE}

        print(f"[LLM] Loading Qwen base model from {base_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            base_path, local_files_only=config.LLM_LOCAL_FILES_ONLY
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=config.LLM_LOCAL_FILES_ONLY,
        )
        if device_map is None:
            base_model = base_model.to(config.LLM_DEVICE)

        print(f"[LLM] Loading LoRA adapter from {adapter_path}...")
        self._model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            local_files_only=config.LLM_LOCAL_FILES_ONLY,
        )
        self._model.eval()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    value = json.loads(match.group(0))
                    return value if isinstance(value, dict) else {}
                except json.JSONDecodeError:
                    pass
        return {}

    @staticmethod
    def _normalize_result(value: dict[str, Any]) -> dict[str, Any]:
        """Return exactly the seven-field contract, even if the model adds fields."""
        result = {
            "lesion_type": value.get("lesion_type"),
            "location": value.get("location"),
            "size_mm": value.get("size_mm"),
            "procedure": value.get("procedure"),
            "biopsy_forceps": value.get("biopsy_forceps") is True,
            "biopsy_pieces": value.get("biopsy_pieces"),
            "pathology": value.get("pathology") is True,
        }
        if isinstance(result["size_mm"], float) and result["size_mm"].is_integer():
            result["size_mm"] = int(result["size_mm"])
        return result

    def extract(self, finding: str) -> dict[str, Any]:
        self._load()
        import torch

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": finding},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        model_device = next(self._model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return self._normalize_result(self._extract_json(text))

    def warmup(self) -> None:
        """Load the base model and adapter before microphone capture starts."""
        self._load()


_extractor = QwenLoraExtractor()


def extract_finding(finding: str) -> dict[str, Any]:
    return _extractor.extract(finding)


def warmup() -> None:
    _extractor.warmup()
