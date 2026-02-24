#!/usr/bin/env python3
"""Parse Warsh.txt into clean per-surah JSON with complete verses."""

import json
import re
from pathlib import Path

INPUT_FILE = Path("Warsh.txt")  # path to your Warsh.txt from Tanzil
OUTPUT_DIR = Path(__file__).parent / "warsh_text"

SURAH_PATTERN = re.compile(r"^سُورَةُ\s+(.+)$")
# Match Arabic verse numbers (١٢٣٤٥٦٧٨٩٠)
VERSE_NUM_PATTERN = re.compile(r'\s*([\u0660-\u0669]+)\s*')

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def arabic_to_int(s: str) -> int:
    return int(s.translate(ARABIC_DIGITS))


def parse_warsh():
    text = INPUT_FILE.read_text(encoding="utf-8")
    lines = text.splitlines()

    surahs = []
    current_surah_name = None
    current_surah_text = []

    for line in lines:
        line = line.strip()
        # Remove ۞ (hizb marker)
        line = line.replace("۞", "").strip()

        match = SURAH_PATTERN.match(line)
        if match:
            # Save previous surah
            if current_surah_name is not None:
                surahs.append((current_surah_name, " ".join(current_surah_text)))
            current_surah_name = match.group(1)
            current_surah_text = []
            continue

        if current_surah_name is not None and line:
            current_surah_text.append(line)

    # Last surah
    if current_surah_name is not None:
        surahs.append((current_surah_name, " ".join(current_surah_text)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_surahs = []

    for surah_idx, (name, full_text) in enumerate(surahs, 1):
        # Split by verse numbers
        parts = VERSE_NUM_PATTERN.split(full_text)
        # parts alternates: [text_before_num, num, text_before_next_num, num, ...]
        ayahs = []
        for i in range(0, len(parts) - 1, 2):
            verse_text = parts[i].strip()
            verse_num = arabic_to_int(parts[i + 1])
            if verse_text:
                ayahs.append({
                    "ayah_number": verse_num,
                    "text": verse_text,
                })

        surah_data = {
            "surah_number": surah_idx,
            "name": name,
            "number_of_ayahs": len(ayahs),
            "ayahs": ayahs,
        }

        out_file = OUTPUT_DIR / f"{surah_idx:03d}.json"
        out_file.write_text(json.dumps(surah_data, ensure_ascii=False, indent=2), encoding="utf-8")
        all_surahs.append({"surah_number": surah_idx, "name": name, "ayahs": len(ayahs)})
        print(f"Surah {surah_idx:03d} ({name}): {len(ayahs)} ayahs")

    # Write index
    index_file = OUTPUT_DIR / "index.json"
    index_file.write_text(json.dumps(all_surahs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. {len(surahs)} surahs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    parse_warsh()
