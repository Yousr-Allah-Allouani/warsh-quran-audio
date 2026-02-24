"""
Live Warsh Quran Recitation Tracker — Server (v5: Per-chunk ASR, strict matching)

Each 1.5s audio chunk is transcribed independently (no accumulation).
Only explicitly matched words are marked correct — gaps are never auto-filled.
Words more than ERROR_LAG positions behind the cursor are flagged as errors.
"""

import asyncio
import json
import logging
import re
import subprocess
from pathlib import Path

import numpy as np
from aiohttp import web
from faster_whisper import WhisperModel
from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
TEXT_DIR = BASE_DIR / "warsh_text"

# ── Text normalization ────────────────────────────────────────────────────────
DIACRITICS_RE = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED'
    r'\u08D3-\u08FF\u0657\u0656ۣۖۗۘۙۚۛۜ۟۠ۡۢۤۥۦ۪ۭۧۨ۬ـ]'
)
ALEF_RE = re.compile(r'[أإآٱ]')

def normalize(text: str) -> str:
    text = DIACRITICS_RE.sub('', text)
    text = ALEF_RE.sub('ا', text)
    text = text.replace('ة', 'ه')
    text = text.replace('ى', 'ي')
    return text


# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = str(BASE_DIR / "tarteel-base-ct2")
log.info("Loading faster-whisper int8 from %s …", MODEL_PATH)
model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")
log.info("Model ready.")


# ── Load surah text ────────────────────────────────────────────────────────────
def load_surah(num: int) -> dict:
    path = TEXT_DIR / f"{num:03d}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)

SURAHS = {1: load_surah(1), 2: load_surah(2)}

def surah_words(surah: dict) -> list[dict]:
    words = []
    for ayah in surah["ayahs"]:
        for w in ayah["text"].split():
            words.append({
                "word": w,
                "ayah": ayah["ayah_number"],
                "index": len(words),
                "norm": normalize(w),
            })
    return words


# ── Audio decode ───────────────────────────────────────────────────────────────
async def decode_audio(audio_bytes: bytes) -> np.ndarray | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", "pipe:0",
            "-f", "f32le", "-ar", "16000", "-ac", "1",
            "-loglevel", "error", "pipe:1",
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=audio_bytes)
        if proc.returncode != 0 or len(stdout) < 4:
            return None
        return np.frombuffer(stdout, dtype=np.float32)
    except Exception as e:
        log.error("decode failed: %s", e)
        return None


# ── Whisper transcribe ─────────────────────────────────────────────────────────
def transcribe(audio: np.ndarray) -> str:
    segments, _ = model.transcribe(
        audio,
        language="ar",
        beam_size=5,       # higher = more accurate (model needs this for long phrases)
        vad_filter=False,  # VAD kills Quranic elongations — process raw audio
    )
    return " ".join(seg.text.strip() for seg in segments)


# ── Recitation tracker ─────────────────────────────────────────────────────────
LOOKAHEAD    = 3    # tight window — prevents jumping far ahead
MATCH_THRESH = 70   # fuzzy ratio threshold
ERROR_LAG    = 6    # mark pending words as error if cursor is this many ahead
MAX_WPS      = 2.5  # max words per second in Quran recitation (caps hallucination)
MAX_BUF_SECS = 10.0 # accumulate ~10s — model needs full phrases, not 1.5s scraps
OVERLAP_SECS = 2.0  # keep this much audio after cursor advances (context)

class RecitationTracker:
    def __init__(self, words: list[dict]):
        self.words    = words
        self.cursor   = 0
        self.states   = ["pending"] * len(words)
        self.busy     = False
        self.audio_buf = np.array([], dtype=np.float32)
        self.buf_secs  = 0.0  # seconds of NEW audio accumulated since last trim

    def reset(self):
        self.cursor    = 0
        self.states    = ["pending"] * len(self.words)
        self.audio_buf = np.array([], dtype=np.float32)
        self.buf_secs  = 0.0

    def append(self, pcm: np.ndarray):
        self.audio_buf = np.concatenate([self.audio_buf, pcm])
        self.buf_secs += len(pcm) / 16000

    def process(self) -> dict:
        if self.cursor >= len(self.words):
            return self._result("done")

        raw = transcribe(self.audio_buf)
        trans_words = normalize(raw).split()

        # Only trust words that could physically have been spoken in the NEW audio
        # (buf_secs tracks fresh audio since last trim, not the overlap context)
        max_words = max(2, int(self.buf_secs * MAX_WPS))
        trans_words = trans_words[:max_words]

        log.info("  transcript (%ds buf, cap=%d): %s",
                 int(self.buf_secs), max_words, raw[:120])

        if not trans_words:
            return self._result("(silence)")

        cursor = self.cursor

        for tw in trans_words:
            if cursor >= len(self.words):
                break
            if len(tw) <= 1:
                continue

            # Find best match within a tight lookahead window
            best_score, best_idx = 0, -1
            end = min(cursor + LOOKAHEAD, len(self.words))
            for i in range(cursor, end):
                score = fuzz.ratio(tw, self.words[i]["norm"])
                if score > best_score:
                    best_score, best_idx = score, i

            if best_score >= MATCH_THRESH and best_idx >= 0:
                # Mark only the matched word — skipped words stay pending
                self.states[best_idx] = "correct"
                cursor = best_idx + 1

        # Mark words that the cursor has moved well past as errors
        for i in range(self.cursor, max(0, cursor - ERROR_LAG)):
            if self.states[i] == "pending":
                self.states[i] = "error"

        if cursor > self.cursor:
            log.info("  Cursor %d → %d", self.cursor, cursor)
            # Trim buffer — keep only overlap context, reset new-audio counter
            keep = int(OVERLAP_SECS * 16000)
            self.audio_buf = self.audio_buf[-keep:] if len(self.audio_buf) > keep else self.audio_buf
            self.buf_secs = 0.0

        self.cursor = cursor
        return self._result(raw[:80])

    def _result(self, debug: str) -> dict:
        return {
            "type": "update",
            "cursor": self.cursor,
            "total": len(self.words),
            "states": self.states,
            "transcript": debug,
        }


# ── HTTP handlers ──────────────────────────────────────────────────────────────
async def index_handler(request):
    return web.FileResponse(Path(__file__).parent / "index.html")

async def surah_handler(request):
    num = int(request.match_info["num"])
    if num not in SURAHS:
        return web.json_response({"error": "not found"}, status=404)
    return web.json_response(SURAHS[num], ensure_ascii=False)


# ── WebSocket handler ─────────────────────────────────────────────────────────
async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    log.info("WS connected")

    tracker = None

    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            data = json.loads(msg.data)

            if data.get("cmd") == "select_surah":
                num = data["surah"]
                if num in SURAHS:
                    words = surah_words(SURAHS[num])
                    tracker = RecitationTracker(words)
                    await ws.send_json({
                        "type": "surah_loaded",
                        "surah": num,
                        "words": [{"word": w["word"], "ayah": w["ayah"]} for w in words],
                        "states": tracker.states,
                    })
                    log.info("Surah %d (%d words)", num, len(words))

            elif data.get("cmd") == "reset":
                if tracker:
                    tracker.reset()
                    await ws.send_json(tracker._result("reset"))

        elif msg.type == web.WSMsgType.BINARY:
            if not tracker:
                continue

            audio = await decode_audio(msg.data)
            if audio is None or len(audio) < 3200:
                continue

            # Always accumulate audio
            tracker.append(audio)

            # Only transcribe when we have enough new audio AND not already busy
            if tracker.busy or tracker.buf_secs < MAX_BUF_SECS:
                continue

            tracker.busy = True
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, tracker.process
                )
                if not ws.closed:
                    await ws.send_json(result)
            except Exception as e:
                log.error("Error: %s", e)
            finally:
                tracker.busy = False

        elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
            break

    log.info("WS disconnected")
    return ws


app = web.Application()
app.router.add_get("/", index_handler)
app.router.add_get("/api/surah/{num}", surah_handler)
app.router.add_get("/ws", ws_handler)

if __name__ == "__main__":
    log.info("Starting at http://localhost:8080")
    web.run_app(app, host="0.0.0.0", port=8080)
