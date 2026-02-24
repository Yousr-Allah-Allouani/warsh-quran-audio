# Warsh Quran Audio — Training Data Pipeline

A pipeline for converting a full Warsh Quran recitation into **ayah-level audio/text pairs** — the format needed to train Quran speech recognition models for the Warsh reading.

**The main contribution is the pipeline, not the model.** If you have a Warsh recitation from any reciter, these scripts can produce a clean labeled dataset with no manual audio editing and no manual text labeling. Everything here was built through conversations with [Claude Code CLI](https://claude.ai/claude-code).

---

## Why This Exists

> *"Warsh has been planned for multiple years, the training data just isn't there at the moment. Someone should create a resource to collect audio data for Warsh — it'll make training much easier. Biggest bottleneck has been the lack of quality data."*
>
> — Reliable source

[Tarteel](https://tarteel.ai) is the leading Quran recitation app. Their production system is built on **NVIDIA NeMo**, trained on 1,250+ hours of Quran data using A100/V100 GPUs — [see the NVIDIA case study](https://nvidia.com/en-us/case-studies/automating-real-time-arabic-speech-recognition/). Warsh support has been on their roadmap for years. The blocker is training data.

This repo documents an attempt to produce that data and — more importantly — **how** to produce more of it. Also worth reading: [this community thread](https://x.com/qasimtwt/status/2025960052096725112?s=20).

---

## What's in This Repo

```
audio/                       113 full-surah MP3s — Abd al-Basit Abd as-Samad, Warsh 'an Nafi'
                             (audio/002.mp3 excluded — 154MB, see note below)
warsh_text/                  Complete Warsh Quran text, 114 JSON files with full diacritics
alignments_whisperx/         Word-level timestamps for all 114 surahs
alignments/                  Segment-level alignments for 38 surahs (Quran Multi-Aligner)
dataset/
  train.jsonl                6,201 ayah-level entries: audio path + Warsh text + metadata
  review.html                Browser tool to listen to clips and validate alignment
parse_warsh.py               Step 1 — Parse raw Warsh text into per-surah JSON
whisperx_align.py            Step 2 — Align full-surah audio → word-level timestamps
build_dataset_v3.py          Step 3 — Cut aligned audio into ayah-level clips
finetune_whisper.py          Reference fine-tuning script
live_recitation/             Browser-based live recitation tracker (proof of concept)
```

> **Missing: `audio/002.mp3`** — Al-Baqara (154MB) exceeds GitHub's 100MB file limit and is not in this repo. To use the full pipeline, download any complete Abd al-Basit Abd as-Samad Warsh 'an Nafi' recitation and place the Al-Baqara audio at `audio/002.mp3`. The recitation used here is widely available on Islamic audio sites and YouTube.

---

## The Core Contribution: How the Chunking Works

This is the part that took the most work to figure out, and it's worth explaining clearly.

**The problem:** We have a full-surah MP3 (e.g. 1 hour of Al-Baqara) and a text file with 286 verses. How do you automatically cut the audio into 286 labeled clips without listening to any of it manually?

**The naive approach (that fails):** Ask Whisper to transcribe the audio, match its output to the Warsh text. Problem: Whisper was trained on Hafs, not Warsh. It frequently transcribes Warsh-specific words incorrectly, confuses verse boundaries, and hallucinates text it "knows" from pretraining.

**What actually works — the key insight:**

We use Whisper *only for its timing information*, not for its text output. The correct Warsh text is already known (from `warsh_text/`). The process:

1. **WhisperX** transcribes the audio with Whisper-medium → gives rough segment timestamps
2. **wav2vec2 forced alignment** (inside WhisperX) then aligns those segments to the audio at word level — this gives us precise timestamps for every spoken word
3. We **fuzzy-match** WhisperX's transcription against the known Warsh text to figure out which words belong to which ayah
4. We use those ayah boundaries to **cut the audio**, and assign the correct Warsh text (from `warsh_text/`) as the label — not Whisper's transcription

The result: even when Whisper mishears a Warsh-specific word, the timing is still valid, and the label is still the correct Warsh text. The `train.jsonl` always uses `warsh_text` as the ground truth, with `whisper_text` stored separately for reference.

This is what makes the pipeline viable without manual labor.

---

## Running the Pipeline

### Prerequisites

```bash
pip install whisperx transformers torch torchaudio rapidfuzz
brew install ffmpeg
```

### Step 1 — Parse the Warsh text (`parse_warsh.py`)

Takes a raw Warsh Quran text file (e.g. from [Tanzil](https://tanzil.net/download/)) and outputs clean per-surah JSON with full diacritics and Warsh-specific orthography.

```bash
python parse_warsh.py
# Output: warsh_text/001.json … warsh_text/114.json
```

The `warsh_text/` directory is already included in this repo — you only need this step if you want to use a different Warsh text source.

### Step 2 — Align audio to text (`whisperx_align.py`)

Runs WhisperX on each full-surah MP3. Takes ~15-20 min per surah on M1 CPU — run overnight for all 114.

```bash
python whisperx_align.py --model-size medium

# Or specific surahs only:
python whisperx_align.py --only 1 2 3
```

Output in `alignments_whisperx/` — already included in this repo for Abd al-Basit's recitation. You only need to re-run this if you're processing a **new reciter's audio**.

### Step 3 — Cut into ayah clips (`build_dataset_v3.py`)

Uses the alignments to slice full-surah MP3s into individual ayah clips, validated against the Warsh text.

```bash
python build_dataset_v3.py
# Output: dataset/audio/001/001_001.mp3 … + dataset/train.jsonl
```

Each entry in `train.jsonl`:
```json
{
  "surah": 1, "ayah": 1,
  "start": 12.98, "end": 18.53, "duration": 5.55,
  "warsh_text": "اِ۬لْحَمْدُ لِلهِ رَبِّ اِ۬لْعَٰلَمِينَ",
  "whisper_text": "الحمد لله رب العالمين",
  "alignment_ratio": 1.0,
  "audio_path": "audio/001/001_001.mp3"
}
```

### Step 4 — Review (`dataset/review.html`)

Open in a browser. Plays each clip with the Warsh text alongside Whisper's transcription. Keyboard shortcuts for fast pass/fail review.

```bash
open dataset/review.html
```

---

## What This Produced

6,201 ayah-level clips from one complete Warsh recitation (Abd al-Basit Abd as-Samad). Average clip duration ~5 seconds.

**This is a starting point, not a training corpus.** A single reciter causes overfitting. The dataset needs at minimum 2–3 more complete Warsh recitations run through the same pipeline before it's useful for fine-tuning.

The 6,201 clips (1.9GB) are available as a HuggingFace dataset — link coming soon.

---

## What's Needed Next: More Reciters

The pipeline is built and documented. Running it again on a different reciter's audio requires no new code — just Steps 2–3 above on new MP3s.

Three complete Warsh recitations identified as good candidates:

- [Playlist 1](https://youtube.com/playlist?list=PL9qjIA25Xw9HqMHhYgYT0CJGy6cS8_Jde&si=dwj02uHWMs4i1o60)
- [Single recitation](https://youtu.be/kHjpM8W7Z9I?si=xFxjInIh6ACRPreL)
- [Playlist 2](https://youtube.com/playlist?list=PL9qjIA25Xw9E3Yt0irSVYyd0Nj8LXmduB&si=hVn_t2Mp8bqBLf66)

To process any of these: download the audio (e.g. with `yt-dlp`), split into per-surah MP3s named `001.mp3`–`114.mp3`, and run Steps 2–3. The `warsh_text/` and `whisperx_align.py` script are already here.

---

## What We Learned

### Fine-tuning Whisper on Warsh doesn't work well

Fine-tuning Whisper on a single Warsh dataset does not yield meaningful improvements. The reasons are structural:
- **Single reciter = overfitting** — one voice and one reading style, the model memorizes rather than generalizes
- **Data volume** — a single reciter's dataset is insufficient; meaningful generalization requires multiple reciters
- **Whisper hallucinates Quran** — it "knows" the text from pretraining and completes ayahs it didn't actually hear. This is the fundamental problem with Whisper for live tracking and no amount of fine-tuning resolves it — see the CTC section below.

### Tarteel doesn't use the open-source Whisper model

The `tarteel-ai/whisper-base-ar-quran` on HuggingFace is a 3-year-old research artifact. Their production model is NVIDIA NeMo, trained on 1,250+ hours. Whisper is not their stack.

### Models explored for live tracking

| Model | Speed (M1 CPU) | Quality | Verdict |
|---|---|---|---|
| `tarteel-ai/whisper-base-ar-quran` int8 | ~0.9s / 10s audio | Decent for Hafs, weaker on Warsh | Best option on CPU |
| `openai/whisper-large-v3-turbo` int8 | ~8-9s / 10s audio | Better quality, wrong language detection on short clips | Too slow for real-time |
| `IJyad/whisper-large-v3-Tarteel` | 44s / 10s audio | Outputs English with `language='ar'` | Unusable |
| `MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix` | ~8-9s / 10s audio (estimated) | 12.69% WER on Hafs test set, LoRA on turbo | Not tested — same CPU bottleneck as turbo, Hafs-trained |

### The most promising direction: CTC + text matching (not tried yet)

Every approach explored here uses Whisper, which is a **seq2seq model with an autoregressive decoder**. That decoder is the root of the hallucination problem — it generates text token by token from a learned distribution, and since it was pretrained on Quran audio, it will confidently generate the "likely next ayah" even when the audio doesn't match.

The architectural fix is to use a **CTC model** (Connectionist Temporal Classification) instead:

```
Current approach:  audio → Whisper encoder+decoder → transcription → fuzzy match Quran text
CTC approach:      audio → encoder only → frame-level scores → match against known Quran text
```

CTC models (like wav2vec2, MMS, or a distilled Conformer) are frame-level: each audio frame gets assigned a character or phoneme score directly. There is no decoder, no autoregressive generation, no memory of "what usually comes next." The model can only output what it actually heard. Hallucination is **architecturally impossible**.

For Quran live tracking, this maps naturally to a matching/search problem:
1. Short audio chunk (2–3s) → CTC encoder → character/phoneme sequence
2. Slide a window over the known Quran text (we have it in `warsh_text/`)
3. Pick the verse with the best combined edit distance + acoustic confidence score
4. Advance position when match is strong enough

This is closer to what Tarteel likely does under the hood — confirmed by their team as "ASR + text search."

**What's missing to try this:**
- A CTC model fine-tuned on Quranic Arabic (ideally Warsh). `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` exists for general Arabic but has no Quran-specific training. The dataset in this repo could be used to fine-tune it.
- A fast on-device CTC encoder — the inference would be much lighter than Whisper since there's no decoder.

This is the direction worth pursuing next.

### The live tracker works, with limitations

`live_recitation/` is a working browser proof of concept — real-time word highlighting as you recite. Accurate enough to demo, not production-quality. See instructions below.

---

## Running the Live Tracker

```bash
pip install aiohttp faster-whisper rapidfuzz numpy
brew install ffmpeg

# Download the model
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tarteel-ai/whisper-base-ar-quran', local_dir='tarteel-base-ct2')
"

python live_recitation/server.py
open http://localhost:8080
```

---

## License

Code: MIT
Warsh text: derived from [Tanzil](https://tanzil.net) (public domain)
Audio: Abd al-Basit Abd as-Samad Warsh recitation (waqf)
