"""Arabic-aware streaming text chunker.

First-audio latency depends on handing the TTS engine a *speakable* chunk as
soon as one exists, rather than waiting for the agent to finish its reply. This
module splits on Arabic and Latin sentence punctuation and falls back to clause
boundaries (and finally hard length limits) so a long run-on sentence still
starts playing quickly.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

# Arabic full stop is '.', question mark '؟', semicolon '؛', comma '،'.
SENTENCE_END = ".!?؟…\n"
CLAUSE_END = "،,;؛:"

_WHITESPACE = re.compile(r"\s+")
# Protect decimals ("3.5") and common abbreviations from being split.
_DECIMAL = re.compile(r"\d[.,]\d")

PAUSE_SPLIT_RE = re.compile(r"(\[(?:pause|silence)\s+\d+\s*(?:ms|s)?\])", re.IGNORECASE)
PAUSE_MATCH_RE = re.compile(r"^\[(?:pause|silence)\s+(\d+)\s*(ms|s)?\]$", re.IGNORECASE)


def parse_pause_ms(text: str) -> int | None:
    """If text is an inline pause tag like [pause 500ms] or [silence 1s], return duration in ms."""
    m = PAUSE_MATCH_RE.match(text.strip())
    if not m:
        return None
    val, unit = m.groups()
    ms = int(val)
    if unit and unit.lower() == "s":
        ms *= 1000
    return ms


def extract_speed_tags(text: str) -> tuple[str, float]:
    """Extract and strip [slow] or [fast] tags, returning cleaned text and speed multiplier."""
    speed = 1.0
    cleaned = text
    if "[slow]" in cleaned or "[/slow]" in cleaned:
        cleaned = cleaned.replace("[slow]", "").replace("[/slow]", "")
        speed = 0.85
    elif "[fast]" in cleaned or "[/fast]" in cleaned:
        cleaned = cleaned.replace("[fast]", "").replace("[/fast]", "")
        speed = 1.15
    return cleaned.strip(), speed


MIN_CHUNK_CHARS = 12
MAX_CHUNK_CHARS = 220
#: Hard cap on the opening chunk. Short = audio starts sooner; too short and
#: the TTS model has no prosodic context to work with.
FIRST_CHUNK_CHARS = 70
#: The opening chunk may be shorter than a normal one -- a greeting like
#: "مرحبا بك،" is perfectly speakable and gets audio playing much sooner.
FIRST_MIN_CHUNK_CHARS = 8


def normalize(text: str) -> str:
    """Light normalization before synthesis."""
    text = text.replace("\u200f", "").replace("\u200e", "")  # bidi marks
    text = text.replace("ـ", "")  # tatweel / kashida
    return _WHITESPACE.sub(" ", text).strip()


def _is_protected(text: str, idx: int) -> bool:
    window = text[max(0, idx - 1) : idx + 2]
    return bool(_DECIMAL.search(window))


def split_sentences(text: str) -> list[str]:
    """Split into sentence-ish units, keeping terminal punctuation and pause tags isolated."""
    text = normalize(text)
    if not text:
        return []

    segments = PAUSE_SPLIT_RE.split(text)
    out: list[str] = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        if parse_pause_ms(segment) is not None:
            out.append(segment)
            continue

        start = 0
        for i, ch in enumerate(segment):
            if ch in SENTENCE_END and not _is_protected(segment, i):
                piece = segment[start : i + 1].strip()
                if piece:
                    out.append(piece)
                start = i + 1
        tail = segment[start:].strip()
        if tail:
            out.append(tail)
    return out


def _split_long(piece: str) -> Iterator[str]:
    """Break an over-long sentence at clause marks, then at spaces."""
    if parse_pause_ms(piece) is not None:
        yield piece
        return
    while len(piece) > MAX_CHUNK_CHARS:
        window = piece[:MAX_CHUNK_CHARS]
        cut = max((window.rfind(c) for c in CLAUSE_END), default=-1)
        if cut < MIN_CHUNK_CHARS:
            cut = window.rfind(" ")
        if cut < MIN_CHUNK_CHARS:
            cut = MAX_CHUNK_CHARS - 1
        yield piece[: cut + 1].strip()
        piece = piece[cut + 1 :].strip()
    if piece:
        yield piece


def chunk_text(text: str) -> list[str]:
    """Split a complete string into synthesis chunks."""
    chunks: list[str] = []
    for sentence in split_sentences(text):
        chunks.extend(c for c in _split_long(sentence) if c)
    return chunks


class StreamingChunker:
    """Feed token deltas in, get speakable chunks out as early as possible.

    The *first* chunk of a reply also breaks at clause marks (، , ; :), because
    nothing is playing yet and every millisecond before first audio is dead
    air. Later chunks hold out for a sentence end, which gives the TTS model
    more context and better prosody -- by then playback is already running, so
    the extra wait is hidden.
    """

    def __init__(
        self,
        min_chars: int = MIN_CHUNK_CHARS,
        max_chars: int = MAX_CHUNK_CHARS,
        first_chunk_chars: int = FIRST_CHUNK_CHARS,
        first_min_chars: int = FIRST_MIN_CHUNK_CHARS,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.first_chunk_chars = first_chunk_chars
        self.first_min_chars = first_min_chars
        self._buf = ""
        self._emitted = 0
        #: How far into ``_buf`` we have already looked for a cut. Without this
        #: every delta rescans the whole buffer, which is quadratic over a long
        #: reply.
        self._scanned = 0

    def push(self, delta: str) -> list[str]:
        self._buf += delta
        ready: list[str] = []

        while True:
            cut = self._find_cut(self._buf)
            if cut is None:
                break
            piece = self._buf[: cut + 1].strip()
            self._buf = self._buf[cut + 1 :]
            self._scanned = 0
            if piece:
                ready.append(piece)
                self._emitted += 1
        return ready

    def _find_cut(self, buf: str) -> int | None:
        # Opening a reply: a clause mark is a good enough place to start
        # talking. Afterwards, hold out for a sentence boundary.
        first = self._emitted == 0
        breaks = SENTENCE_END + CLAUSE_END if first else SENTENCE_END
        limit = self.first_chunk_chars if first else self.max_chars
        floor = min(self.first_min_chars, self.min_chars) if first else self.min_chars

        start = max(self._scanned, floor - 1)
        for i in range(start, len(buf)):
            if buf[i] in breaks and not _is_protected(buf, i):
                return i
        self._scanned = len(buf)

        if len(buf) >= limit:
            window = buf[:limit]
            cut = max((window.rfind(c) for c in CLAUSE_END), default=-1)
            if cut < floor:
                cut = window.rfind(" ")
            if cut < floor:
                cut = limit - 1
            return cut
        return None

    def flush(self) -> list[str]:
        """Drain whatever is left when the upstream stream ends."""
        rest = normalize(self._buf)
        self._buf = ""
        self._scanned = 0
        if not rest:
            return []
        pieces = list(_split_long(rest))
        self._emitted += len(pieces)
        return pieces


def chunk_stream(deltas: Iterable[str]) -> Iterator[str]:
    chunker = StreamingChunker()
    for delta in deltas:
        yield from chunker.push(delta)
    yield from chunker.flush()
