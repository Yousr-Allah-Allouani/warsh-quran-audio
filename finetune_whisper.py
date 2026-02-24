#!/usr/bin/env python3
"""
Fine-tune Whisper-base on Warsh Quran recitation (Surahs 1-2).

Optimized for M1 MacBook Air, 8GB RAM, CPU training.
Expected runtime: ~15 min for 292 clips x 2 epochs on M1 CPU.

Usage:
    caffeinate -dims ~/.warsh-venv/bin/python finetune_whisper.py

    # Dry-run (1 step, no saving):
    caffeinate -dims ~/.warsh-venv/bin/python finetune_whisper.py --dry-run
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import torch
import torchaudio
from torch_audiomentations import Compose, PitchShift
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / "dataset"
TRAIN_JSONL = DATASET_DIR / "train.jsonl"
OUTPUT_DIR = PROJECT_DIR / "whisper-warsh-v1"

MODEL_NAME = "openai/whisper-base"
LANGUAGE = "ar"
TASK = "transcribe"
SAMPLE_RATE = 16_000

# Surahs to include in this proof-of-concept
TARGET_SURAHS = {1, 2}


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_entries(jsonl_path: Path, surahs: set[int]) -> list[dict]:
    """Load JSONL entries filtered to specific surahs."""
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry["surah"] in surahs:
                # Resolve audio path relative to dataset dir
                entry["audio_path"] = str(DATASET_DIR / entry["audio_path"])
                entries.append(entry)
    return entries


def split_train_eval(
    entries: list[dict], eval_ratio: float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split entries into train and eval sets deterministically."""
    import random

    rng = random.Random(seed)
    shuffled = entries.copy()
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - eval_ratio)))
    return shuffled[:split_idx], shuffled[split_idx:]


# ──────────────────────────────────────────────
# Audio augmentation
# ──────────────────────────────────────────────
def build_augmentation():
    """Build audio augmentation pipeline using torch-audiomentations."""
    return Compose(
        transforms=[
            PitchShift(
                min_transpose_semitones=-3.0,
                max_transpose_semitones=3.0,
                sample_rate=SAMPLE_RATE,
                p=0.5,
                output_type="tensor",
            ),
        ],
        output_type="tensor",
    )


# ──────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────
class WarshDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        entries: list[dict],
        processor: WhisperProcessor,
        augment=None,
    ):
        self.entries = entries
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load audio
        waveform, sr = torchaudio.load(entry["audio_path"])

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Apply augmentation (training only)
        if self.augment is not None:
            # torch-audiomentations expects (batch, channels, samples)
            waveform_3d = waveform.unsqueeze(0)
            waveform_3d = self.augment(waveform_3d, sample_rate=SAMPLE_RATE)
            waveform = waveform_3d.squeeze(0)

        # Extract features
        audio_array = waveform.squeeze().numpy()
        input_features = self.processor.feature_extractor(
            audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features[0]

        # Tokenize labels (truncate to Whisper's 448-token max)
        labels = self.processor.tokenizer(
            entry["warsh_text"],
            return_tensors="pt",
            max_length=448,
            truncation=True,
        ).input_ids[0]

        return {"input_features": input_features, "labels": labels}


# ──────────────────────────────────────────────
# Data collator
# ──────────────────────────────────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        # Stack input features (all same length from Whisper feature extractor)
        input_features = torch.stack(
            [f["input_features"] for f in features]
        )

        # Pad labels to max length in batch
        label_features = [f["labels"] for f in features]
        max_label_length = max(len(l) for l in label_features)
        padded_labels = []
        for labels in label_features:
            remainder = torch.full(
                (max_label_length - len(labels),),
                -100,
                dtype=labels.dtype,
            )
            padded_labels.append(torch.cat([labels, remainder]))

        labels = torch.stack(padded_labels)

        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_features": input_features,
            "labels": labels,
        }


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def build_compute_metrics(processor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 training step + 1 eval step to verify setup",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Whisper Fine-tuning: Warsh Quran (Surahs 1-2)")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {TRAIN_JSONL}...")
    entries = load_entries(TRAIN_JSONL, TARGET_SURAHS)
    print(f"  Found {len(entries)} clips for surahs {TARGET_SURAHS}")

    train_entries, eval_entries = split_train_eval(entries)
    print(f"  Train: {len(train_entries)}, Eval: {len(eval_entries)}")

    # Verify a few audio files exist
    for e in entries[:3]:
        if not os.path.exists(e["audio_path"]):
            print(f"  ERROR: Audio file not found: {e['audio_path']}")
            sys.exit(1)
    print("  Audio files verified.")

    # Load model and processor
    print(f"\nLoading model: {MODEL_NAME}...")
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Set generation config for Arabic transcription
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.generation_config.no_repeat_ngram_size = 3  # prevent repetition loops
    model.generation_config.repetition_penalty = 1.2
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Increase dropout to fight overfitting on small dataset
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.2  # default is 0.0 for whisper-base

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model loaded: {param_count:.1f}M parameters")

    # Build augmentation
    augment = build_augmentation()

    # Build datasets
    print("\nBuilding datasets...")
    train_dataset = WarshDataset(train_entries, processor, augment=augment)
    eval_dataset = WarshDataset(eval_entries, processor, augment=None)

    # Test one sample
    sample = train_dataset[0]
    print(f"  Sample input_features shape: {sample['input_features'].shape}")
    print(f"  Sample labels shape: {sample['labels'].shape}")
    decoded = processor.tokenizer.decode(
        sample["labels"], skip_special_tokens=True
    )
    print(f"  Sample decoded text: {decoded}")

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Training arguments
    num_epochs = 1 if args.dry_run else 2
    max_steps = 2 if args.dry_run else -1

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        weight_decay=0.05,  # L2 regularization to prevent overfitting
        warmup_steps=50 if not args.dry_run else 0,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        fp16=False,
        eval_strategy="epoch" if not args.dry_run else "steps",
        eval_steps=1 if args.dry_run else None,
        save_strategy="no",  # save manually at end to avoid disk issues
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=5 if not args.dry_run else 1,
        report_to="none",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        use_cpu=True,
    )

    # Compute metrics
    compute_metrics = build_compute_metrics(processor)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # Train
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN: Training 2 steps + eval to verify setup")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Starting training: 2 epochs (anti-overfitting config)")
        print(f"  - weight_decay=0.05, dropout=0.2")
        print(f"  - no_repeat_ngram_size=3, repetition_penalty=1.2")
        print(f"Output: {OUTPUT_DIR}")
        print("=" * 60)

    trainer.train()

    # Final eval
    print("\n" + "=" * 60)
    print("Final evaluation")
    print("=" * 60)
    metrics = trainer.evaluate()
    print(f"  WER: {metrics.get('eval_wer', 'N/A')}")

    if not args.dry_run:
        # Save model + processor
        print(f"\nSaving model to {OUTPUT_DIR}...")
        trainer.save_model(str(OUTPUT_DIR))
        processor.save_pretrained(str(OUTPUT_DIR))
        print("  Done!")

        # Quick test inference on 5 eval samples
        print("\n" + "=" * 60)
        print("Quick inference test (5 eval samples)")
        print("=" * 60)
        for i, test_entry in enumerate(eval_entries[:5], 1):
            waveform, sr = torchaudio.load(test_entry["audio_path"])
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(
                    waveform
                )
            audio_array = waveform.squeeze().numpy()

            input_features = processor(
                audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).input_features

            with torch.no_grad():
                predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            print(f"\n  [{i}] Surah {test_entry['surah']}, Ayah {test_entry['ayah']}")
            print(f"      Expected: {test_entry['warsh_text']}")
            print(f"      Got:      {transcription}")
    else:
        print("\nDry run completed successfully!")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
