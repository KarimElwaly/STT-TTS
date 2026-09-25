"""Voice activity detection and noise gating.

This is **not optional**. The Cohere model card explicitly warns that the model
is "eager to transcribe, even non-speech sounds" and benefits from a noise gate
or VAD in front of it -- without one, room tone turns into hallucinated text.

Two implementations:
  * :class:`SileroVad`  -- ONNX, ~1 MB, realtime on CPU. Preferred.
  * :class:`EnergyVad`  -- dependency-free RMS gate. Fallback / tests.

Both run identically in the CPU and GPU profiles.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..common.protocols import SAMPLE_RATE_IN

log = logging.getLogger(__name__)

FRAME_MS = 32
FRAME_SAMPLES = SAMPLE_RATE_IN * FRAME_MS // 1000


@dataclass(slots=True)
class Segment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(slots=True)
class VadConfig:
    threshold: float = 0.5
    #: Utterance must contain at least this much speech to be transcribed.
    min_speech_ms: int = 250
    #: Silence this long ends the utterance.
    min_silence_ms: int = 600
    #: Padding kept around detected speech so words aren't clipped.
    speech_pad_ms: int = 160
    #: Hard RMS floor applied before the model; kills constant low-level hiss.
    noise_gate_rms: float = 0.005


class EnergyVad:
    """RMS gate. Crude but dependency-free and deterministic for tests."""

    name = "energy"

    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()

    def reset(self) -> None:  # pragma: no cover - stateless
        return

    def frame_probability(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        if rms < self.config.noise_gate_rms:
            return 0.0
        return min(1.0, rms / (self.config.noise_gate_rms * 6.0))


class SileroVad:
    """Silero VAD v5 via onnxruntime."""

    name = "silero"

    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()
        self._model = None
        self._fallback = EnergyVad(self.config)

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad(onnx=True)
            log.info("VAD: silero-vad (onnx) loaded")
        except Exception as exc:
            log.warning("VAD: silero unavailable (%s); falling back to energy gate", exc)
            self._model = None

    def reset(self) -> None:
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def frame_probability(self, frame: np.ndarray) -> float:
        # The noise gate applies regardless of which backend answers.
        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        if rms < self.config.noise_gate_rms:
            return 0.0

        if self._model is None:
            return self._fallback.frame_probability(frame)

        import torch

        # Silero v5 expects exactly 512 samples at 16 kHz.
        chunk = frame[:512]
        if chunk.size < 512:
            chunk = np.pad(chunk, (0, 512 - chunk.size))
        with torch.no_grad():
            return float(self._model(torch.from_numpy(chunk).float().unsqueeze(0), SAMPLE_RATE_IN))


def build_vad(config: VadConfig | None = None, prefer_silero: bool = True):
    if prefer_silero:
        vad = SileroVad(config)
        vad.load()
        return vad
    return EnergyVad(config)


def segment_audio(audio: np.ndarray, vad=None, config: VadConfig | None = None) -> list[Segment]:
    """Find speech segments in a complete recording."""
    config = config or VadConfig()
    vad = vad or build_vad(config)
    vad.reset()

    pad = config.speech_pad_ms / 1000.0
    min_speech = config.min_speech_ms / 1000.0
    max_silence_frames = max(1, config.min_silence_ms // FRAME_MS)

    segments: list[Segment] = []
    start: float | None = None
    silence_run = 0

    for i in range(0, len(audio), FRAME_SAMPLES):
        frame = audio[i : i + FRAME_SAMPLES]
        t = i / SAMPLE_RATE_IN
        speech = vad.frame_probability(frame) >= config.threshold

        if speech:
            silence_run = 0
            if start is None:
                start = t
        elif start is not None:
            silence_run += 1
            if silence_run >= max_silence_frames:
                end = t - (silence_run - 1) * FRAME_MS / 1000.0
                if end - start >= min_speech:
                    segments.append(Segment(max(0.0, start - pad), end + pad))
                start = None
                silence_run = 0

    if start is not None:
        end = len(audio) / SAMPLE_RATE_IN
        if end - start >= min_speech:
            segments.append(Segment(max(0.0, start - pad), end))

    return segments


def gate_audio(audio: np.ndarray, config: VadConfig | None = None, vad=None) -> np.ndarray:
    """Return only the speech parts. Empty array means "nothing was said"."""
    segments = segment_audio(audio, vad=vad, config=config)
    if not segments:
        return np.zeros(0, dtype=np.float32)
    parts = [
        audio[int(s.start_s * SAMPLE_RATE_IN) : int(s.end_s * SAMPLE_RATE_IN)] for s in segments
    ]
    return np.concatenate(parts).astype(np.float32)


def chunk_long_audio(
    audio: np.ndarray,
    max_chunk_s: float = 28.0,
    config: VadConfig | None = None,
    vad=None,
) -> Iterator[np.ndarray]:
    """Split >30s audio at silence boundaries so the AED decoder stays in-distribution."""
    total_s = len(audio) / SAMPLE_RATE_IN
    if total_s <= max_chunk_s:
        yield audio
        return

    segments = segment_audio(audio, vad=vad, config=config)
    if not segments:
        return

    buf_start = segments[0].start_s
    buf_end = segments[0].end_s
    for seg in segments[1:]:
        if seg.end_s - buf_start > max_chunk_s:
            yield audio[int(buf_start * SAMPLE_RATE_IN) : int(buf_end * SAMPLE_RATE_IN)]
            buf_start = seg.start_s
        buf_end = seg.end_s
    yield audio[int(buf_start * SAMPLE_RATE_IN) : int(buf_end * SAMPLE_RATE_IN)]


class UtteranceDetector:
    """Streaming end-of-utterance detection for the realtime loop.

    Push PCM frames; get ``speech_started`` / ``utterance`` events back.
    """

    def __init__(self, config: VadConfig | None = None, vad=None) -> None:
        self.config = config or VadConfig()
        self.vad = vad or build_vad(self.config)
        self._buf = np.zeros(0, dtype=np.float32)
        self._speech = np.zeros(0, dtype=np.float32)
        self._in_speech = False
        self._silence_frames = 0
        self._speech_ms = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._speech = np.zeros(0, dtype=np.float32)
        self._in_speech = False
        self._silence_frames = 0
        self._speech_ms = 0
        self.vad.reset()

    def push(self, samples: np.ndarray) -> list[tuple[str, np.ndarray | None]]:
        """Return a list of ``(event, payload)``; events: ``speech_started``, ``utterance``."""
        self._buf = np.concatenate([self._buf, samples.astype(np.float32)])
        events: list[tuple[str, np.ndarray | None]] = []
        max_silence_frames = max(1, self.config.min_silence_ms // FRAME_MS)

        while len(self._buf) >= FRAME_SAMPLES:
            frame, self._buf = self._buf[:FRAME_SAMPLES], self._buf[FRAME_SAMPLES:]
            speech = self.vad.frame_probability(frame) >= self.config.threshold

            if speech:
                if not self._in_speech:
                    self._in_speech = True
                    self._speech_ms = 0
                    events.append(("speech_started", None))
                self._silence_frames = 0
                self._speech_ms += FRAME_MS
                self._speech = np.concatenate([self._speech, frame])
            elif self._in_speech:
                self._silence_frames += 1
                self._speech = np.concatenate([self._speech, frame])
                if self._silence_frames >= max_silence_frames:
                    utterance = self._speech
                    long_enough = self._speech_ms >= self.config.min_speech_ms
                    self._in_speech = False
                    self._silence_frames = 0
                    self._speech = np.zeros(0, dtype=np.float32)
                    self.vad.reset()
                    if long_enough:
                        events.append(("utterance", utterance))
        return events

    def force_end(self) -> np.ndarray | None:
        """End the utterance now (push-to-talk release)."""
        if not self._in_speech or self._speech_ms < self.config.min_speech_ms:
            self.reset()
            return None
        utterance = self._speech
        self.reset()
        return utterance
