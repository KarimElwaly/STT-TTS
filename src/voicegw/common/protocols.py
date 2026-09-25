"""Engine contracts.

Every STT/TTS implementation satisfies these protocols. The REST API, the
realtime WebSocket loop and the MCP server code against *these types only* --
they never import a concrete model class. That is what makes the CPU and GPU
profiles interchangeable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

SAMPLE_RATE_IN = 16_000
SAMPLE_RATE_OUT = 24_000


@dataclass(slots=True)
class Transcript:
    text: str
    language: str
    duration_s: float = 0.0
    engine: str = ""


@dataclass(slots=True)
class AudioChunk:
    """A slice of synthesized audio, float32 mono in ``[-1, 1]``."""

    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE_OUT
    is_final: bool = False

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(slots=True)
class Voice:
    id: str
    label: str
    language: str = "ar"
    # Zero-shot cloning reference (OmniVoice). Absent for preset voices (Piper).
    ref_audio: str | None = None
    ref_text: str | None = None
    # Voice-design attribute string, used when no reference audio is given.
    description: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def is_clone(self) -> bool:
        return self.ref_audio is not None


@dataclass(slots=True)
class EngineInfo:
    engine_id: str
    model_id: str
    device: str
    dtype: str
    #: Measured/declared real-time factor. < 1.0 means faster than realtime.
    rtf_estimate: float
    #: Whether this engine is fast enough to serve the /v1/realtime loop.
    realtime_capable: bool
    notes: str = ""
    #: Which CUDA memory allocator this engine uses: ``torch``, ``ct2`` or
    #: ``none`` (CPU-only). Engines from different families cannot share the
    #: VRAM one of them has cached, which matters under exclusive residency.
    allocator: str = "torch"
    #: Measured VRAM footprint of the loaded weights, in MiB. ``0`` means
    #: "unknown" and makes residency fall back to a coarse VRAM threshold.
    vram_mib: int = 0
    #: Whether the weights can move between devices after loading. Quantized
    #: models are pinned to the device they were loaded onto.
    movable: bool = True


@runtime_checkable
class SttEngine(Protocol):
    info: EngineInfo

    def load(self) -> None:
        """Materialize weights. Idempotent."""

    def transcribe(self, audio: np.ndarray, language: str = "ar") -> Transcript:
        """Transcribe float32 mono audio at :data:`SAMPLE_RATE_IN`."""

    def unload(self) -> None: ...


@runtime_checkable
class TtsEngine(Protocol):
    info: EngineInfo

    def load(self) -> None: ...

    def synthesize(self, text: str, voice: Voice, urgent: bool = False) -> Iterator[AudioChunk]:
        """Yield audio chunks as they become available (streaming-first).

        ``urgent`` marks the chunk that gates the start of playback: the
        listener is waiting in silence for it, so an engine may trade fidelity
        for latency. Later chunks are generated while earlier audio is still
        playing and should not make that trade.
        """

    def supports(self, voice: Voice) -> bool:
        """Whether this engine can render the given voice (e.g. cloning)."""

    def unload(self) -> None: ...
