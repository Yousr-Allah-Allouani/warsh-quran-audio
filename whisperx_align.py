#!/usr/bin/env python3
"""Forced alignment of Warsh Quran audio using WhisperX.

Uses Whisper for initial transcription, then forced alignment
against the known Warsh text to get precise word-level timestamps.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import whisperx

AUDIO_DIR = Path(__file__).parent
WARSH_TEXT_DIR = AUDIO_DIR / "warsh_text"
OUTPUT_DIR = AUDIO_DIR / "alignments_whisperx"
TOTAL_SURAHS = 114


def load_warsh_text(surah_num: int) -> list[dict]:
    """Load Warsh verse text for a surah."""
    text_file = WARSH_TEXT_DIR / f"{surah_num:03d}.json"
    with open(text_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["ayahs"]


def process_surah(
    model,
    align_model,
    align_metadata,
    surah_num: int,
    device: str,
) -> dict:
    """Process a single surah: transcribe then force-align against Warsh text."""
    audio_path = str(AUDIO_DIR / f"{surah_num:03d}.mp3")

    # Load audio
    audio = whisperx.load_audio(audio_path)

    # Step 1: Transcribe with Whisper (to get initial segments/timing)
    result = model.transcribe(audio, batch_size=4, language="ar")

    # Step 2: Force-align against audio to get word-level timestamps
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=True,
    )

    # Step 3: Load the correct Warsh text
    warsh_ayahs = load_warsh_text(surah_num)

    # Build output
    output = {
        "surah_number": surah_num,
        "whisperx_segments": result["segments"],
        "word_segments": result.get("word_segments", []),
        "warsh_ayahs": warsh_ayahs,
    }

    return output


def main():
    parser = argparse.ArgumentParser(description="WhisperX forced alignment for Warsh Quran")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=TOTAL_SURAHS)
    parser.add_argument("--only", type=int, nargs="+")
    parser.add_argument("--model-size", default="medium", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
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
        print(f"Skipping {skipped} surah(s) with existing alignments.")
    if not to_process:
        print("Nothing to process. All done!")
        return

    print(f"Will process {len(to_process)} surah(s)")
    print(f"Whisper model: {args.model_size} | Device: {args.device}")
    print()

    # Load models once
    print("Loading Whisper model...")
    compute_type = "float32"  # MPS needs float32
    model = whisperx.load_model(args.model_size, args.device, compute_type=compute_type, language="ar")
    print("Whisper loaded.")

    print("Loading alignment model...")
    align_model, align_metadata = whisperx.load_align_model(language_code="ar", device=args.device)
    print("Alignment model loaded.\n")

    succeeded = 0
    failed = 0
    start_time = time.time()

    for i, surah_num in enumerate(to_process, 1):
        print(f"[{i}/{len(to_process)}] Processing surah {surah_num:03d} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = process_surah(model, align_model, align_metadata, surah_num, args.device)
            elapsed = time.time() - t0

            out_path = OUTPUT_DIR / f"{surah_num:03d}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            seg_count = len(result.get("whisperx_segments", []))
            word_count = len(result.get("word_segments", []))
            succeeded += 1
            print(f"OK  {seg_count} segments, {word_count} words  ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR ({elapsed:.1f}s): {e}")
            failed += 1
            continue

    total_time = time.time() - start_time
    print(f"\nDone. {succeeded} succeeded, {failed} failed, {skipped} skipped.")
    print(f"Total time: {total_time / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
