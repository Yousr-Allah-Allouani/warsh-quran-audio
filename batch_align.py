#!/usr/bin/env python3
"""Batch-process Warsh Quran audio files through the Quran Multi-Aligner API.

Sends each of the 114 surah MP3 files to the HuggingFace Space and saves
the alignment JSON results.  Supports resuming (skips files that already
have output) and retries on transient failures.

Requirements:
    pip install gradio_client

Usage:
    python batch_align.py                  # process all 114 surahs
    python batch_align.py --start 50       # start from surah 50
    python batch_align.py --only 2 36 67   # process specific surahs only
    python batch_align.py --model Large    # use the Large model
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from gradio_client import Client, handle_file
except ImportError:
    print("ERROR: gradio_client is not installed. Run:")
    print("  pip install gradio_client")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_SPACE = "hetchyy/Quran-multi-aligner"
API_ENDPOINT = "/process_audio_session"

AUDIO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = AUDIO_DIR / "alignments"

TOTAL_SURAHS = 114

# Default VAD / segmentation parameters
DEFAULT_MIN_SILENCE_MS = 200
DEFAULT_MIN_SPEECH_MS = 1000
DEFAULT_PAD_MS = 100
DEFAULT_MODEL = "Base"
DEFAULT_DEVICE = "GPU"

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 30  # seconds; doubles each retry


def get_audio_path(surah_num: int) -> Path:
    return AUDIO_DIR / f"{surah_num:03d}.mp3"


def get_output_path(surah_num: int) -> Path:
    return OUTPUT_DIR / f"{surah_num:03d}.json"


def process_surah(
    client: Client,
    surah_num: int,
    *,
    min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS,
    pad_ms: int = DEFAULT_PAD_MS,
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
) -> dict:
    """Send one surah to the aligner and return the JSON result."""
    audio_path = get_audio_path(surah_num)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    result = client.predict(
        audio_data=handle_file(str(audio_path)),
        min_silence_ms=min_silence_ms,
        min_speech_ms=min_speech_ms,
        pad_ms=pad_ms,
        model_name=model_name,
        device=device,
        api_name=API_ENDPOINT,
    )
    return result


def save_result(surah_num: int, data: dict) -> Path:
    """Write alignment JSON to disk."""
    out = get_output_path(surah_num)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out


def process_with_retries(
    client: Client,
    surah_num: int,
    **kwargs,
) -> dict | None:
    """Attempt to process a surah with exponential-backoff retries."""
    delay = RETRY_DELAY_BASE
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return process_surah(client, surah_num, **kwargs)
        except FileNotFoundError:
            raise  # no point retrying a missing file
        except Exception as e:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] Error: {e}")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {delay}s ...")
                time.sleep(delay)
                delay *= 2
                # Reconnect client in case the connection went stale
                try:
                    import httpx
                    client = Client(HF_SPACE, upload_size_limit=500 * 1024 * 1024)
                    client.httpx_client = httpx.Client(timeout=httpx.Timeout(600.0))
                except Exception:
                    pass
            else:
                print(f"  FAILED after {MAX_RETRIES} attempts. Skipping surah {surah_num:03d}.")
                return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Batch-align Warsh Quran audio")
    parser.add_argument("--start", type=int, default=1, help="First surah number (default: 1)")
    parser.add_argument("--end", type=int, default=TOTAL_SURAHS, help="Last surah number (default: 114)")
    parser.add_argument("--only", type=int, nargs="+", help="Process only these surah numbers")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=["Base", "Large"], help="Whisper model (default: Base)")
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["GPU", "CPU"], help="Device (default: GPU)")
    parser.add_argument("--silence", type=int, default=DEFAULT_MIN_SILENCE_MS, help="min_silence_ms (default: 200)")
    parser.add_argument("--speech", type=int, default=DEFAULT_MIN_SPEECH_MS, help="min_speech_ms (default: 1000)")
    parser.add_argument("--pad", type=int, default=DEFAULT_PAD_MS, help="pad_ms (default: 100)")
    parser.add_argument("--force", action="store_true", help="Re-process even if output JSON exists")
    args = parser.parse_args()

    # Determine which surahs to process
    if args.only:
        surah_list = sorted(args.only)
    else:
        surah_list = list(range(args.start, args.end + 1))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Count how many are already done / skipped
    to_process = []
    for n in surah_list:
        if not args.force and get_output_path(n).exists():
            continue
        if not get_audio_path(n).exists():
            print(f"WARNING: {get_audio_path(n)} does not exist, skipping.")
            continue
        to_process.append(n)

    skipped = len(surah_list) - len(to_process)
    if skipped:
        print(f"Skipping {skipped} surah(s) with existing alignments (use --force to redo).")
    if not to_process:
        print("Nothing to process. All done!")
        return

    print(f"Will process {len(to_process)} surah(s): {to_process[0]:03d} .. {to_process[-1]:03d}")
    print(f"Model: {args.model} | Device: {args.device}")
    print(f"Params: silence={args.silence}ms  speech={args.speech}ms  pad={args.pad}ms")
    print()

    # Connect to the HF Space with extended timeout for long surahs
    print("Connecting to HuggingFace Space ...")
    import httpx
    client = Client(HF_SPACE)
    # Patch the httpx client timeout to 10 minutes for long audio files
    client.httpx_client = httpx.Client(timeout=httpx.Timeout(600.0))
    print("Connected.\n")

    succeeded = 0
    failed = 0
    start_time = time.time()

    for i, surah_num in enumerate(to_process, 1):
        print(f"[{i}/{len(to_process)}] Processing surah {surah_num:03d} ...", end=" ", flush=True)
        t0 = time.time()

        result = process_with_retries(
            client,
            surah_num,
            min_silence_ms=args.silence,
            min_speech_ms=args.speech,
            pad_ms=args.pad,
            model_name=args.model,
            device=args.device,
        )

        elapsed = time.time() - t0

        if result is None:
            failed += 1
            print(f"FAILED ({elapsed:.1f}s)")
            continue

        # Check for API-level errors
        if isinstance(result, dict) and result.get("error") and not result.get("segments"):
            print(f"API error: {result['error']} ({elapsed:.1f}s)")
            failed += 1
            continue

        out_path = save_result(surah_num, result)
        seg_count = len(result.get("segments", [])) if isinstance(result, dict) else "?"
        succeeded += 1
        print(f"OK  {seg_count} segments  ({elapsed:.1f}s)  -> {out_path.name}")

        # Brief pause between requests to be polite to the API
        if i < len(to_process):
            time.sleep(2)

    total_time = time.time() - start_time
    print(f"\nDone. {succeeded} succeeded, {failed} failed, {skipped} skipped.")
    print(f"Total time: {total_time / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
