#!/usr/bin/env python3
"""Build the Whisper fine-tuning dataset by merging WhisperX alignments with Warsh text.

For each surah:
1. Load WhisperX output (word-level timestamps from transcription)
2. Load Warsh ground-truth text (verse-level)
3. Match transcribed words to Warsh verses using word-count alignment
4. Cut audio into verse-level clips using timestamps
5. Output: (audio_clip, warsh_verse_text) pairs

The matching works by:
- Stripping diacritics from Warsh text to get plain words
- Counting words per verse in Warsh
- Walking through the WhisperX word list, consuming N words per verse
- Using the first word's start and last word's end as verse boundaries
- Skipping non-Quran segments (isti'adha, basmala) at the start
"""

import argparse
import difflib
import json
import re
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path

AUDIO_DIR = Path(__file__).parent
WHISPERX_DIR = AUDIO_DIR / "alignments_whisperx"
WARSH_TEXT_DIR = AUDIO_DIR / "warsh_text"
DATASET_DIR = AUDIO_DIR / "dataset"
TOTAL_SURAHS = 114

# Arabic diacritics range (tashkeel) + special Warsh marks
DIACRITICS_RE = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF\u0657\u0656ۣۖۗۘۙۚۛۜ۟۠ۡۢۤۥۦ۪ۭۧۨ۬]')
# Alef variants to normalize
ALEF_RE = re.compile(r'[إأآٱاٰ]')
# Tatweel
TATWEEL_RE = re.compile(r'ـ')


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics/tashkeel from text."""
    text = DIACRITICS_RE.sub('', text)
    text = TATWEEL_RE.sub('', text)
    return text


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for fuzzy matching."""
    text = strip_diacritics(text)
    text = ALEF_RE.sub('ا', text)
    text = text.replace('ة', 'ه')
    text = text.replace('ى', 'ي')
    text = text.replace('ؤ', 'و')
    text = text.replace('ئ', 'ي')
    # Remove non-letter/space chars
    text = re.sub(r'[^\u0621-\u064A\s]', '', text)
    return ' '.join(text.split())


def get_warsh_words(verse_text: str) -> list[str]:
    """Get normalized word list from a Warsh verse."""
    normalized = normalize_arabic(verse_text)
    return [w for w in normalized.split() if w]


def is_isti_adha_or_basmala(words: list[dict], start_idx: int) -> int:
    """Detect isti'adha or basmala at the start, return number of words to skip."""
    if start_idx >= len(words):
        return 0

    # Look at the first ~10 words
    window = words[start_idx:start_idx + 12]
    text = ' '.join(normalize_arabic(w['word']) for w in window)

    skip = 0

    # Isti'adha: أعوذ بالله من الشيطان الرجيم (5 words)
    if 'اعوذ' in text[:30] or 'اعود' in text[:30]:
        skip = 5

    # Basmala: بسم الله الرحمن الرحيم (4 words)
    remaining = ' '.join(normalize_arabic(w['word']) for w in words[start_idx + skip:start_idx + skip + 6])
    if 'بسم' in remaining[:20]:
        skip += 4

    return skip


def sequence_align(warsh_words_flat: list[str], whisperx_words: list[dict]) -> list[int | None]:
    """Align Warsh words to WhisperX words using difflib sequence matching.

    Returns a list, same length as warsh_words_flat.
    alignment[i] = index into whisperx_words (the best matching wx word), or None
    if WhisperX dropped that Warsh word entirely.

    This handles:
    - WhisperX dropping a word → alignment[i] = None (verse boundary doesn't drift)
    - WhisperX adding an extra word → ignored (no Warsh counterpart)
    - Word-level Hafs/Warsh variant → still matched by position in a replace block
    """
    wx_normalized = [normalize_arabic(w['word']) for w in whisperx_words]
    alignment = [None] * len(warsh_words_flat)

    matcher = difflib.SequenceMatcher(None, warsh_words_flat, wx_normalized, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Exact match block: 1-to-1 correspondence
            for offset in range(i2 - i1):
                alignment[i1 + offset] = j1 + offset
        elif tag == 'replace':
            # WhisperX heard something different (Hafs/Warsh variant, noise, etc.)
            # Map as many as we can in order; leftover Warsh words stay None
            for offset in range(min(i2 - i1, j2 - j1)):
                alignment[i1 + offset] = j1 + offset
        # 'delete': Warsh words WhisperX missed → alignment stays None
        # 'insert': WhisperX extra words → we just skip them

    return alignment


def match_words_to_verses(whisperx_words: list[dict], warsh_ayahs: list[dict], surah_num: int) -> list[dict]:
    """Match WhisperX transcribed words to Warsh verse boundaries using sequence alignment.

    Uses difflib to align the full WhisperX word stream to the full Warsh word list,
    tolerating dropped/added/substituted words without propagating drift to later verses.

    Key: verse N's audio ends where verse N+1's first detected word begins,
    NOT where verse N's last detected word ends. This captures the full verse
    audio even when WhisperX misses words in the middle/end of a verse.
    """
    # Skip isti'adha and basmala at the start
    skip_count = is_isti_adha_or_basmala(whisperx_words, 0)
    whisperx_words = whisperx_words[skip_count:]

    # Build flat list of (normalized_warsh_word, ayah_index)
    warsh_flat = []
    for ayah_idx, ayah in enumerate(warsh_ayahs):
        for w in get_warsh_words(ayah['text']):
            warsh_flat.append((w, ayah_idx))

    if not warsh_flat or not whisperx_words:
        return []

    # Sequence-align all Warsh words to all WhisperX words at once
    warsh_words_only = [w for w, _ in warsh_flat]
    alignment = sequence_align(warsh_words_only, whisperx_words)

    # Group aligned WhisperX word indices by ayah
    ayah_wx_indices: dict[int, list[int]] = defaultdict(list)
    for i, (_, ayah_idx) in enumerate(warsh_flat):
        wx_idx = alignment[i]
        if wx_idx is not None:
            ayah_wx_indices[ayah_idx].append(wx_idx)

    # First pass: find each verse's anchor timestamps (first/last detected word)
    verse_anchors = []  # (ayah_idx, first_wx_start, last_wx_end, wx_indices)
    for ayah_idx, ayah in enumerate(warsh_ayahs):
        wx_indices = sorted(ayah_wx_indices.get(ayah_idx, []))
        if not wx_indices:
            verse_anchors.append(None)
            continue
        first_start = whisperx_words[wx_indices[0]].get('start')
        last_end = whisperx_words[wx_indices[-1]].get('end')
        if first_start is None or last_end is None:
            verse_anchors.append(None)
            continue
        verse_anchors.append((ayah_idx, first_start, last_end, wx_indices))

    # Second pass: set verse boundaries using neighboring verse anchors
    # verse N ends where verse N+1 starts (minus tiny gap)
    # verse N starts where verse N-1 ends (plus tiny gap)
    segments = []
    active_anchors = [(i, a) for i, a in enumerate(verse_anchors) if a is not None]

    for pos, (list_idx, anchor) in enumerate(active_anchors):
        ayah_idx, first_start, last_end, wx_indices = anchor
        ayah = warsh_ayahs[ayah_idx]
        warsh_words = get_warsh_words(ayah['text'])
        n_warsh = len(warsh_words)

        # Start: midpoint between previous verse's last word and this verse's first word
        if pos > 0:
            prev_anchor = active_anchors[pos - 1][1]
            prev_end = prev_anchor[2]  # previous verse's last detected word end
            start_time = (prev_end + first_start) / 2
        else:
            start_time = max(0, first_start - 0.15)

        # End: midpoint between this verse's last word and next verse's first word
        if pos < len(active_anchors) - 1:
            next_anchor = active_anchors[pos + 1][1]
            next_start = next_anchor[1]  # next verse's first detected word start
            end_time = (last_end + next_start) / 2
        else:
            end_time = last_end + 0.5

        # Collect all WhisperX words that fall within this verse's time range
        # (includes unaligned words that WhisperX detected but couldn't match)
        verse_all_words = [
            w for w in whisperx_words
            if w.get('start') is not None
            and w['start'] >= start_time - 0.05
            and w['start'] <= end_time + 0.05
        ]

        # Confidence from aligned words only
        aligned_words = [whisperx_words[i] for i in wx_indices]
        avg_score = sum(w.get('score', 0) for w in aligned_words) / len(aligned_words)
        alignment_ratio = len(wx_indices) / n_warsh if n_warsh > 0 else 0

        segments.append({
            'surah': surah_num,
            'ayah': ayah['ayah_number'],
            'start': round(start_time, 3),
            'end': round(end_time, 3),
            'duration': round(end_time - start_time, 3),
            'warsh_text': ayah['text'],
            'whisper_text': ' '.join(w['word'] for w in verse_all_words),
            'confidence': round(avg_score, 3),
            'n_words': n_warsh,
            'n_matched': len(wx_indices),
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
            str(output_path)
        ],
        check=True,
        capture_output=True,
    )


def process_surah(surah_num: int, cut_clips: bool = True) -> list[dict]:
    """Process one surah: match words to verses and optionally cut audio clips."""
    whisperx_file = WHISPERX_DIR / f"{surah_num:03d}.json"
    warsh_file = WARSH_TEXT_DIR / f"{surah_num:03d}.json"
    audio_file = AUDIO_DIR / f"{surah_num:03d}.mp3"

    if not whisperx_file.exists():
        return []

    with open(whisperx_file, 'r', encoding='utf-8') as f:
        whisperx_data = json.load(f)
    with open(warsh_file, 'r', encoding='utf-8') as f:
        warsh_data = json.load(f)

    # Get flat word list from WhisperX
    words = whisperx_data.get('word_segments', [])
    if not words:
        # Fall back to extracting words from segments
        for seg in whisperx_data.get('whisperx_segments', []):
            words.extend(seg.get('words', []))

    ayahs = warsh_data['ayahs']

    # Match words to verses
    segments = match_words_to_verses(words, ayahs, surah_num)

    # Cut audio clips
    if cut_clips and segments:
        surah_dir = DATASET_DIR / "audio" / f"{surah_num:03d}"
        surah_dir.mkdir(parents=True, exist_ok=True)

        for seg in segments:
            clip_path = surah_dir / f"{surah_num:03d}_{seg['ayah']:03d}.mp3"
            try:
                cut_audio(audio_file, clip_path, seg['start'], seg['end'])
                seg['audio_path'] = str(clip_path.relative_to(DATASET_DIR))
            except Exception as e:
                print(f"    Error cutting {clip_path.name}: {e}")
                seg['audio_path'] = None

    return segments


def main():
    parser = argparse.ArgumentParser(description="Build Warsh Whisper fine-tuning dataset")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=TOTAL_SURAHS)
    parser.add_argument("--only", type=int, nargs="+")
    parser.add_argument("--no-cut", action="store_true", help="Skip audio cutting (metadata only)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.only:
        surah_list = sorted(args.only)
    else:
        surah_list = list(range(args.start, args.end + 1))

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    all_segments = []
    total_duration = 0

    for surah_num in surah_list:
        whisperx_file = WHISPERX_DIR / f"{surah_num:03d}.json"
        if not whisperx_file.exists():
            continue

        print(f"Surah {surah_num:03d} ...", end=" ", flush=True)

        segments = process_surah(surah_num, cut_clips=not args.no_cut)

        if segments:
            duration = sum(s['duration'] for s in segments)
            total_duration += duration
            avg_conf = sum(s['confidence'] for s in segments) / len(segments)
            print(f"{len(segments)} verses, {duration:.0f}s, avg confidence {avg_conf:.2f}")
            all_segments.extend(segments)
        else:
            print("no segments (alignment not ready?)")

    # Save metadata
    metadata_file = DATASET_DIR / "metadata.json"
    metadata_file.write_text(json.dumps(all_segments, ensure_ascii=False, indent=2), encoding='utf-8')

    # Also save as JSONL (one line per sample, common for training)
    jsonl_file = DATASET_DIR / "train.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for seg in all_segments:
            f.write(json.dumps(seg, ensure_ascii=False) + '\n')

    # Summary
    print(f"\n{'='*50}")
    print(f"Dataset built: {len(all_segments)} verse segments")
    print(f"Total audio: {total_duration/3600:.1f} hours")
    print(f"Metadata: {metadata_file}")
    print(f"Training file: {jsonl_file}")
    if not args.no_cut:
        print(f"Audio clips: {DATASET_DIR / 'audio'}/")


if __name__ == "__main__":
    main()
