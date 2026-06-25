"""Fine-tune Whisper small on DeepGI colonoscopy audio samples."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate


METADATA_CSV = "training_data/metadata.csv"
OUTPUT_DIR = config.FINETUNED_ASR_PATH
BASE_MODEL = f"openai/whisper-{config.WHISPER_MODEL}"


def load_data():
    dataset = load_dataset(
        "csv",
        data_files={"train": METADATA_CSV},
        split="train",
    )
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=config.SAMPLE_RATE))
    dataset = dataset.rename_column("audio_filepath", "audio")
    return dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch


def compute_metrics(pred, tokenizer, wer_metric):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def main():
    print(f"Loading base model: {BASE_MODEL}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL)
    tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL, language=config.LANGUAGE, task="transcribe")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=config.LANGUAGE, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.generation_config.language = config.LANGUAGE
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    print(f"Loading dataset from {METADATA_CSV}...")
    dataset = load_data()
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=dataset.column_names,
    )

    # Simple 90/10 train/eval split
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=500,
        fp16=False,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer, wer_metric),
        processing_class=processor.feature_extractor,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"\nTraining complete. Set USE_FINETUNED_ASR = True in config.py to use it.")


if __name__ == "__main__":
    main()
