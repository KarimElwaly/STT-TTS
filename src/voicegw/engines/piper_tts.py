"""Piper TTS -- ONNX CPU fallback.

Fast and light (RTF ~0.05 on CPU) but **preset voices only, no cloning**. The
voice registry routes clone voices to OmniVoice and preset voices here, so the
CPU profile degrades gracefully instead of failing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..common.protocols import AudioChunk, EngineInfo, Voice
from ..common.text_chunker import chunk_text, extract_speed_tags, parse_pause_ms

log = logging.getLogger(__name__)

DEFAULT_VOICE_MODEL = "ar_JO-kareem-medium"
PIPER_SAMPLE_RATE = 22_050


def voices_dir() -> Path:
    """Where piper voice models live.

    ``PiperVoice.load`` defaults ``download_dir`` to the *current working
    directory*, which scatters .onnx files wherever the server happened to be
    started. Keep them in one predictable cache instead.
    """
    return Path.home() / ".cache" / "voicegw" / "piper"


def is_cached(model_name: str = DEFAULT_VOICE_MODEL) -> bool:
    return (voices_dir() / f"{model_name}.onnx").is_file()


class PiperTts:
    def __init__(self, model_name: str = DEFAULT_VOICE_MODEL) -> None:
        self.model_name = model_name
        self._voice = None
        self.info = EngineInfo(
            engine_id="piper",
            model_id=model_name,
            device="cpu",
            dtype="int8",
            rtf_estimate=0.05,
            realtime_capable=True,
            allocator="none",
            notes="Preset voices only -- no zero-shot cloning.",
        )

    def load(self) -> None:
        if self._voice is not None:
            return
        from piper import PiperVoice  # type: ignore[import-not-found]

        from ..common.config import get_settings

        target = voices_dir()
        target.mkdir(parents=True, exist_ok=True)
        if get_settings().offline and not is_cached(self.model_name):
            raise FileNotFoundError(
                f"Piper voice {self.model_name!r} is not in {target} and offline "
                "mode is on. Run `voicegw fetch` with a network connection."
            )

        self._voice = PiperVoice.load(self.model_name, download_dir=target)
        self.info.model_id = self.model_name
        log.info("Loaded piper voice %s", self.model_name)

    def unload(self) -> None:
        self._voice = None

    def supports(self, voice: Voice) -> bool:
        return not voice.is_clone

    def _sample_rate(self) -> int:
        cfg = getattr(self._voice, "config", None)
        return int(getattr(cfg, "sample_rate", PIPER_SAMPLE_RATE))

    def _generate(self, text: str) -> np.ndarray:
        buf = bytearray()
        for chunk in self._voice.synthesize_stream_raw(text):
            buf.extend(chunk)
        return np.frombuffer(bytes(buf), dtype="<i2").astype(np.float32) / 32768.0

    def synthesize(self, text: str, voice: Voice, urgent: bool = False) -> Iterator[AudioChunk]:
        if voice.is_clone:
            raise ValueError(
                f"Voice {voice.id!r} requires zero-shot cloning, which the piper engine "
                "does not support. Use a preset voice or switch to the gpu profile."
            )
        self.load()
        sr = self._sample_rate()
        chunks = chunk_text(text)
        if not chunks:
            yield AudioChunk(np.zeros(0, dtype=np.float32), sr, is_final=True)
            return
        for i, piece in enumerate(chunks):
            is_final = (i == len(chunks) - 1)
            pause_ms = parse_pause_ms(piece)
            if pause_ms is not None:
                num_samples = int(sr * (pause_ms / 1000.0))
                samples = np.zeros(num_samples, dtype=np.float32)
                yield AudioChunk(samples, sr, is_final=is_final)
                continue

            clean_piece, speed = extract_speed_tags(piece)
            if not clean_piece:
                continue

            samples = self._generate(clean_piece)
            if speed != 1.0 and len(samples) > 0:
                import librosa

                samples = librosa.effects.time_stretch(samples, rate=speed)
            yield AudioChunk(samples, sr, is_final=is_final)


def build_cpu() -> PiperTts:
    return PiperTts()
