#!/usr/bin/env python3
"""Build Warsh Whisper fine-tuning dataset using FORCED ALIGNMENT.

Instead of relying on WhisperX transcription (which is bad for Warsh),
this script feeds the KNOWN Warsh text directly to the wav2vec2 alignment
model, bypassing Whisper entirely. This ensures:
- Every word gets a timestamp (no dropped words)
- Verse boundaries are precise
- No Hafs/Warsh confusion

Pipeline per surah:
1. Load audio + Warsh text
2. Estimate rough verse timing (word-count proportional)
3. Run forced alignment: wav2vec2 aligns Warsh text → audio
4. Get precise word-level timestamps for ALL words
5. Compute verse boundaries (midpoint between adjacent verses)
6. Cut audio clips
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

AUDIO_DIR = Path(__file__).parent
WARSH_TEXT_DIR = AUDIO_DIR / "warsh_text"
DATASET_DIR = AUDIO_DIR / "dataset"
ALIGNMENT_DIR = AUDIO_DIR / "alignments_forced"
TOTAL_SURAHS = 114

# Diacritics to strip for alignment model (wav2vec2 works on base characters)
DIACRITICS_RE = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED'
    r'\u08D3-\u08FF\u0657\u0656ۣۖۗۘۙۚۛۜ۟۠ۡۢۤۥۦ۪ۭۧۨ۬ـ]'
)


def strip_diacritics(text: str) -> str:
    return DIACRITICS_RE.sub('', text)


def estimate_verse_timing(ayahs: list[dict], total_duration: float, offset: float = 0.0) -> list[dict]:
    """Estimate rough verse timing using word-count proportional allocation.

    Args:
        ayahs: List of verse dicts with 'text' field
        total_duration: Total audio duration in seconds
        offset: Seconds to skip at start (for isti'adha + basmala)

    Returns:
        List of dicts with 'text', 'start', 'end' for each verse
    """
    remaining = total_duration - offset
    total_words = sum(len(strip_diacritics(a['text']).split()) for a in ayahs)

    if total_words == 0:
        return []

    secs_per_word = remaining / total_words

    segments = []
    t = offset
    for ayah in ayahs:
        text = strip_diacritics(ayah['text'])
        n_words = max(1, len(text.split()))
        dur = n_words * secs_per_word
        segments.append({
            'text': text,
            'start': round(t, 3),
            'end': round(t + dur, 3),
        })
        t += dur

    return segments


def build_chunked_transcript(ayahs: list[dict], total_dur: float, offset: float, max_chunk_dur: float = 30.0) -> list[dict]:
    """Group consecutive verses into ~30s chunks for efficient alignment.

    Each chunk concatenates multiple verse texts into one segment.
    We track which verses are in each chunk so we can split words back later.

    Returns list of dicts: {'text', 'start', 'end', 'verse_word_counts': [n1, n2, ...]}
    """
    remaining = total_dur - offset
    total_words = sum(max(1, len(strip_diacritics(a['text']).split())) for a in ayahs)

    if total_words == 0:
        return []

    secs_per_word = remaining / total_words

    chunks = []
    chunk_texts = []
    chunk_word_counts = []
    chunk_start = offset
    chunk_dur = 0.0

    for ayah in ayahs:
        text = strip_diacritics(ayah['text'])
        n_words = max(1, len(text.split()))
        verse_dur = n_words * secs_per_word

        # If adding this verse would exceed max chunk duration, flush current chunk
        if chunk_dur + verse_dur > max_chunk_dur and chunk_texts:
            chunks.append({
                'text': ' '.join(chunk_texts),
                'start': round(chunk_start, 3),
                'end': round(chunk_start + chunk_dur, 3),
                'verse_word_counts': chunk_word_counts.copy(),
            })
            chunk_start += chunk_dur
            chunk_texts = []
            chunk_word_counts = []
            chunk_dur = 0.0

        chunk_texts.append(text)
        chunk_word_counts.append(n_words)
        chunk_dur += verse_dur

    # Flush last chunk
    if chunk_texts:
        chunks.append({
            'text': ' '.join(chunk_texts),
            'start': round(chunk_start, 3),
            'end': round(total_dur, 3),  # extend to end of audio
            'verse_word_counts': chunk_word_counts.copy(),
        })

    return chunks


def forced_align_surah(
    audio_path: Path,
    warsh_file: Path,
    align_model,
    align_metadata,
    device: str,
    surah_num: int,
) -> dict | None:
    """Run forced alignment for one surah using chunked processing.

    Groups verses into ~30s chunks for efficient wav2vec2 processing.
    Each chunk is aligned independently, then words are split back to verses.

    Returns dict with 'segments' (one per verse, with word timestamps).
    """
    import whisperx

    # Load audio
    audio = whisperx.load_audio(str(audio_path))
    total_dur = len(audio) / 16000

    # Load Warsh text
    with open(warsh_file, 'r', encoding='utf-8') as f:
        warsh_data = json.load(f)
    ayahs = warsh_data['ayahs']

    if not ayahs:
        return None

    # Estimate preamble (isti'adha + basmala)
    if surah_num == 1:
        offset = 5.0   # just isti'adha
    elif surah_num == 9:
        offset = 5.0   # just isti'adha, no basmala
    else:
        offset = 10.0  # isti'adha + basmala

    # Build chunked transcript (~30s blocks)
    chunks = build_chunked_transcript(ayahs, total_dur, offset, max_chunk_dur=30.0)

    if not chunks:
        return None

    # Prepare transcript for whisperx.align() (just text + start + end)
    transcript = [{'text': c['text'], 'start': c['start'], 'end': c['end']} for c in chunks]

    # Run forced alignment on all chunks
    result = whisperx.align(
        transcript,
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Split aligned words back into per-verse segments
    aligned_chunks = result.get('segments', [])
    verse_segments = []
    verse_idx = 0

    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx >= len(aligned_chunks):
            # No alignment result for this chunk
            for n_words in chunk['verse_word_counts']:
                verse_segments.append({'text': '', 'words': [], 'start': None, 'end': None})
                verse_idx += 1
            continue

        aligned = aligned_chunks[chunk_idx]
        all_words = aligned.get('words', [])

        # Split words back to verses using word counts
        word_pos = 0
        for n_words in chunk['verse_word_counts']:
            verse_words = all_words[word_pos:word_pos + n_words]
            timed = [w for w in verse_words if 'start' in w and 'end' in w]

            seg_text = ' '.join(w.get('word', '') for w in verse_words)
            seg_start = timed[0]['start'] if timed else None
            seg_end = timed[-1]['end'] if timed else None

            verse_segments.append({
                'text': seg_text,
                'start': seg_start,
                'end': seg_end,
                'words': verse_words,
            })
            word_pos += n_words
            verse_idx += 1

    return {'segments': verse_segments}


def compute_verse_boundaries(result: dict, ayahs: list[dict], surah_num: int, total_dur: float) -> list[dict]:
    """Compute verse boundaries from forced alignment result.

    Uses midpoint between adjacent verses for clean cuts.
    """
    segments_raw = result.get('segments', [])

    if len(segments_raw) != len(ayahs):
        print(f"  WARNING: got {len(segments_raw)} aligned segments but expected {len(ayahs)} verses")

    # Extract timing for each verse
    verse_data = []
    for i, (seg, ayah) in enumerate(zip(segments_raw, ayahs)):
        words = seg.get('words', [])
        timed_words = [w for w in words if 'start' in w and 'end' in w]

        if not timed_words:
            verse_data.append(None)
            continue

        verse_data.append({
            'ayah_number': ayah['ayah_number'],
            'warsh_text': ayah['text'],
            'first_word_start': timed_words[0]['start'],
            'last_word_end': timed_words[-1]['end'],
            'words': timed_words,
            'n_words_expected': len(strip_diacritics(ayah['text']).split()),
            'n_words_timed': len(timed_words),
        })

    # Compute boundaries using midpoints
    segments = []
    active = [(i, v) for i, v in enumerate(verse_data) if v is not None]

    for pos, (idx, vd) in enumerate(active):
        # Start: midpoint between previous verse end and this verse start
        if pos > 0:
            prev_vd = active[pos - 1][1]
            start = (prev_vd['last_word_end'] + vd['first_word_start']) / 2
        else:
            start = max(0, vd['first_word_start'] - 0.15)

        # End: midpoint between this verse end and next verse start
        if pos < len(active) - 1:
            next_vd = active[pos + 1][1]
            end = (vd['last_word_end'] + next_vd['first_word_start']) / 2
        else:
            end = min(total_dur, vd['last_word_end'] + 0.5)

        # Confidence: fraction of words that got timestamps
        alignment_ratio = vd['n_words_timed'] / vd['n_words_expected'] if vd['n_words_expected'] > 0 else 0

        # Word-level confidence from alignment scores
        scores = [w.get('score', 0) for w in vd['words']]
        avg_score = sum(scores) / len(scores) if scores else 0

        segments.append({
            'surah': surah_num,
            'ayah': vd['ayah_number'],
            'start': round(start, 3),
            'end': round(end, 3),
            'duration': round(end - start, 3),
            'warsh_text': vd['warsh_text'],
            'aligned_text': ' '.join(w['word'] for w in vd['words']),
            'confidence': round(avg_score, 3),
            'n_words': vd['n_words_expected'],
            'n_matched': vd['n_words_timed'],
            'alignment_ratio': round(alignment_ratio, 3),
        })

    return segments


def cut_audio(audio_path: Path, output_path: Path, start: float, end: float):
    """Cut a segment from an audio file using ffmpeg."""
    duration = end - start
    subprocess.run(
        [
            'ffmpeg', '-y', '-i', str(audio_path),
            '-ss', str(start), '-t', str(duration),
            '-c', 'copy', '-loglevel', 'error',
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def process_surah(
    surah_num: int,
    align_model,
    align_metadata,
    device: str,
    cut_clips: bool = True,
    save_alignment: bool = True,
) -> list[dict]:
    """Process one surah: forced align + cut audio clips."""
    import whisperx

    audio_path = AUDIO_DIR / f"{surah_num:03d}.mp3"
    warsh_file = WARSH_TEXT_DIR / f"{surah_num:03d}.json"

    if not audio_path.exists() or not warsh_file.exists():
        return []

    # Load audio for duration
    audio = whisperx.load_audio(str(audio_path))
    total_dur = len(audio) / 16000

    # Load Warsh data
    with open(warsh_file, 'r', encoding='utf-8') as f:
        warsh_data = json.load(f)
    ayahs = warsh_data['ayahs']

    # Check for cached alignment
    alignment_file = ALIGNMENT_DIR / f"{surah_num:03d}.json"
    if alignment_file.exists():
        with open(alignment_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
    else:
        # Run forced alignment
        result = forced_align_surah(
            audio_path, warsh_file, align_model, align_metadata, device, surah_num
        )
        if result is None:
            return []

        # Cache alignment
        if save_alignment:
            ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)
            # Convert to JSON-serializable format
            serializable = {
                'segments': [
                    {
                        'text': seg.get('text', ''),
                        'start': seg.get('start'),
                        'end': seg.get('end'),
                        'words': [
                            {k: v for k, v in w.items() if k in ('word', 'start', 'end', 'score')}
                            for w in seg.get('words', [])
                        ],
                    }
                    for seg in result.get('segments', [])
                ],
            }
            alignment_file.write_text(
                json.dumps(serializable, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )

    # Compute verse boundaries
    segments = compute_verse_boundaries(result, ayahs, surah_num, total_dur)

    # Cut audio clips
    if cut_clips and segments:
        surah_dir = DATASET_DIR / "audio" / f"{surah_num:03d}"
        surah_dir.mkdir(parents=True, exist_ok=True)

        for seg in segments:
            clip_path = surah_dir / f"{surah_num:03d}_{seg['ayah']:03d}.mp3"
            try:
                cut_audio(audio_path, clip_path, seg['start'], seg['end'])
                seg['audio_path'] = str(clip_path.relative_to(DATASET_DIR))
            except Exception as e:
                print(f"    Error cutting {clip_path.name}: {e}")
                seg['audio_path'] = None

    return segments


def main():
    parser = argparse.ArgumentParser(description="Build Warsh dataset using forced alignment")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=TOTAL_SURAHS)
    parser.add_argument("--only", type=int, nargs="+")
    parser.add_argument("--no-cut", action="store_true", help="Skip audio cutting")
    parser.add_argument("--force", action="store_true", help="Re-run alignment even if cached")
    args = parser.parse_args()

    import whisperx

    if args.only:
        surah_list = sorted(args.only)
    else:
        surah_list = list(range(args.start, args.end + 1))

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)

    # Delete cached alignments if --force
    if args.force:
        for f in ALIGNMENT_DIR.glob("*.json"):
            f.unlink()

    # Load alignment model once (no Whisper model needed!)
    print("Loading Arabic wav2vec2 alignment model...")
    device = "cpu"
    align_model, align_metadata = whisperx.load_align_model(language_code="ar", device=device)
    print("Model loaded.\n")

    all_segments = []
    total_duration = 0
    total_time = 0

    for surah_num in surah_list:
        audio_path = AUDIO_DIR / f"{surah_num:03d}.mp3"
        if not audio_path.exists():
            continue

        print(f"Surah {surah_num:03d} ...", end=" ", flush=True)
        t0 = time.time()

        segments = process_surah(
            surah_num, align_model, align_metadata, device,
            cut_clips=not args.no_cut,
        )

        elapsed = time.time() - t0
        total_time += elapsed

        if segments:
            duration = sum(s['duration'] for s in segments)
            total_duration += duration
            avg_conf = sum(s['confidence'] for s in segments) / len(segments)
            avg_ratio = sum(s['alignment_ratio'] for s in segments) / len(segments)
            print(f"{len(segments)} verses, {duration:.0f}s, "
                  f"conf={avg_conf:.2f}, align={avg_ratio:.0%}, "
                  f"({elapsed:.1f}s)")
            all_segments.extend(segments)
        else:
            print(f"no segments ({elapsed:.1f}s)")

    # Save metadata
    metadata_file = DATASET_DIR / "metadata.json"
    metadata_file.write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    # Save as JSONL
    jsonl_file = DATASET_DIR / "train.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for seg in all_segments:
            f.write(json.dumps(seg, ensure_ascii=False) + '\n')

    # Summary
    print(f"\n{'='*60}")
    print(f"Dataset built: {len(all_segments)} verse segments")
    print(f"Total audio: {total_duration/3600:.1f} hours")
    print(f"Processing time: {total_time/60:.1f} minutes")
    print(f"Alignments cached: {ALIGNMENT_DIR}/")
    print(f"Training file: {jsonl_file}")
    if not args.no_cut:
        print(f"Audio clips: {DATASET_DIR / 'audio'}/")

    # Quality summary
    if all_segments:
        ratios = [s['alignment_ratio'] for s in all_segments]
        confs = [s['confidence'] for s in all_segments]
        print(f"\nQuality:")
        print(f"  Avg alignment ratio: {sum(ratios)/len(ratios):.1%}")
        print(f"  Perfect alignment (100%): {sum(1 for r in ratios if r >= 1.0)}/{len(ratios)}")
        print(f"  Low alignment (<80%): {sum(1 for r in ratios if r < 0.8)}/{len(ratios)}")
        print(f"  Avg confidence: {sum(confs)/len(confs):.3f}")


if __name__ == "__main__":
    main()
