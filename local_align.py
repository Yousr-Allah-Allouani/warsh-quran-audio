#!/usr/bin/env python3
"""Batch-process Warsh Quran audio locally using the Quran Multi-Aligner pipeline."""

import argparse
import json
import sys
import time
from pathlib import Path

# Add the aligner to the path
ALIGNER_DIR = Path("../Quran-multi-aligner")  # adjust to your local path
sys.path.insert(0, str(ALIGNER_DIR))

import numpy as np
import librosa

AUDIO_DIR = Path(__file__).parent
OUTPUT_DIR = AUDIO_DIR / "alignments"
TOTAL_SURAHS = 114


def process_surah(surah_num: int, model_name: str = "Base", device: str = "mps") -> dict | None:
    """Process a single surah through the aligner pipeline."""
    from src.pipeline import process_audio

    audio_path = AUDIO_DIR / f"{surah_num:03d}.mp3"
    if not audio_path.exists():
        print(f"  Audio file not found: {audio_path}")
        return None

    # Load audio
    y, sr = librosa.load(str(audio_path), sr=None)
    audio_data = (sr, y.astype(np.float32))

    # Run pipeline
    result = process_audio(
        audio_data,
        min_silence_ms=200,
        min_speech_ms=1000,
        pad_ms=100,
        model_name=model_name,
        device=device,
    )

    # result is a tuple: (html, json_output, raw_intervals, is_complete, audio, sr, intervals, seg_dir, log_row)
    json_output = result[1]
    return json_output


def main():
    parser = argparse.ArgumentParser(description="Local batch-align Warsh Quran audio")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=TOTAL_SURAHS)
    parser.add_argument("--only", type=int, nargs="+")
    parser.add_argument("--model", default="Base", choices=["Base", "Large"])
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.only:
        surah_list = sorted(args.only)
    else:
        surah_list = list(range(args.start, args.end + 1))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    to_process = []
    for n in surah_list:
        out_path = OUTPUT_DIR / f"{n:03d}.json"
        if not args.force and out_path.exists():
            continue
        audio_path = AUDIO_DIR / f"{n:03d}.mp3"
        if not audio_path.exists():
            print(f"WARNING: {audio_path} missing, skipping.")
            continue
        to_process.append(n)

    skipped = len(surah_list) - len(to_process)
    if skipped:
        print(f"Skipping {skipped} surah(s) with existing alignments (use --force to redo).")
    if not to_process:
        print("Nothing to process. All done!")
        return

    print(f"Will process {len(to_process)} surah(s)")
    print(f"Model: {args.model} | Device: {args.device}")
    print()

    # Preload models
    print("Preloading models...")
    from src.segmenter.segmenter_model import load_segmenter
    from src.alignment.phoneme_asr import load_phoneme_asr
    from src.alignment.ngram_index import get_ngram_index
    from src.alignment.phoneme_matcher_cache import preload_all_chapters

    load_segmenter()
    load_phoneme_asr(args.model)
    get_ngram_index()
    preload_all_chapters()
    print("Models preloaded.\n")

    succeeded = 0
    failed = 0
    start_time = time.time()

    for i, surah_num in enumerate(to_process, 1):
        print(f"[{i}/{len(to_process)}] Processing surah {surah_num:03d} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = process_surah(surah_num, model_name=args.model, device=args.device)
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
            continue

        elapsed = time.time() - t0

        if result is None:
            failed += 1
            print(f"FAILED ({elapsed:.1f}s)")
            continue

        out_path = OUTPUT_DIR / f"{surah_num:03d}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        seg_count = len(result.get("segments", [])) if isinstance(result, dict) else "?"
        succeeded += 1
        print(f"OK  {seg_count} segments  ({elapsed:.1f}s)  -> {out_path.name}")

    total_time = time.time() - start_time
    print(f"\nDone. {succeeded} succeeded, {failed} failed, {skipped} skipped.")
    print(f"Total time: {total_time / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
