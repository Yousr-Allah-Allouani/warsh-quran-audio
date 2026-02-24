#!/usr/bin/env python3
"""
Test fine-tuned Whisper model on Warsh Quran audio clips.

Usage:
    # Test on a few random clips from eval set:
    ~/.warsh-venv/bin/python test_whisper.py

    # Test on a specific audio file:
    ~/.warsh-venv/bin/python test_whisper.py --audio dataset/audio/001/001_001.mp3

    # Compare fine-tuned vs base model:
    ~/.warsh-venv/bin/python test_whisper.py --compare
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / "dataset"
TRAIN_JSONL = DATASET_DIR / "train.jsonl"
MODEL_DIR = PROJECT_DIR / "whisper-warsh-v1"
BASE_MODEL = "openai/whisper-base"
SAMPLE_RATE = 16_000
TARGET_SURAHS = {1, 2}


def load_audio(audio_path: str) -> torch.Tensor:
    """Load and preprocess audio to 16kHz mono."""
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    return waveform.squeeze()


def transcribe(model, processor, audio_array):
    """Transcribe an audio array."""
    input_features = processor(
        audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, help="Path to a specific audio file")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare fine-tuned vs base model",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of random clips to test (default: 5)",
    )
    args = parser.parse_args()

    # Load fine-tuned model
    print(f"Loading fine-tuned model from {MODEL_DIR}...")
    if not MODEL_DIR.exists():
        print(f"ERROR: Model not found at {MODEL_DIR}")
        print("Run finetune_whisper.py first.")
        return

    ft_processor = WhisperProcessor.from_pretrained(str(MODEL_DIR))
    ft_model = WhisperForConditionalGeneration.from_pretrained(str(MODEL_DIR))
    ft_model.eval()
    print("  Fine-tuned model loaded.")

    # Optionally load base model for comparison
    base_model = None
    base_processor = None
    if args.compare:
        print(f"Loading base model: {BASE_MODEL}...")
        base_processor = WhisperProcessor.from_pretrained(
            BASE_MODEL, language="ar", task="transcribe"
        )
        base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
        base_model.eval()
        print("  Base model loaded.")

    if args.audio:
        # Test single file
        audio_path = args.audio
        audio_array = load_audio(audio_path).numpy()

        print(f"\nAudio: {audio_path}")
        result = transcribe(ft_model, ft_processor, audio_array)
        print(f"  Fine-tuned: {result}")

        if base_model:
            base_result = transcribe(base_model, base_processor, audio_array)
            print(f"  Base model: {base_result}")
    else:
        # Test random clips from eval set
        entries = []
        with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry["surah"] in TARGET_SURAHS:
                    entry["audio_path"] = str(DATASET_DIR / entry["audio_path"])
                    entries.append(entry)

        # Use same split as training (seed=42, 10% eval)
        rng = random.Random(42)
        shuffled = entries.copy()
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * 0.9))
        eval_entries = shuffled[split_idx:]

        # Pick random samples
        test_samples = random.sample(
            eval_entries, min(args.n, len(eval_entries))
        )

        print(f"\nTesting on {len(test_samples)} clips from eval set:")
        print("-" * 70)

        for i, entry in enumerate(test_samples, 1):
            audio_array = load_audio(entry["audio_path"]).numpy()
            result = transcribe(ft_model, ft_processor, audio_array)

            print(f"\n[{i}] Surah {entry['surah']}, Ayah {entry['ayah']}")
            print(f"    Audio:    {Path(entry['audio_path']).name}")
            print(f"    Expected: {entry['warsh_text']}")
            print(f"    Got:      {result}")

            if base_model:
                base_result = transcribe(base_model, base_processor, audio_array)
                print(f"    Base:     {base_result}")

        print("\n" + "-" * 70)
        print("Done.")


if __name__ == "__main__":
    main()
