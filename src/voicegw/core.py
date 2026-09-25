"""The voice core: warm model singletons shared by REST, WebSocket and MCP.

One process hosts all three façades so the ~2.6B parameters are loaded exactly
once. Each model is guarded by its own lock so concurrent requests serialize
instead of OOM-ing a laptop GPU.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import asdict

import numpy as np

from .common.audio import decode_audio
from .common.config import (
    Profile,
    ProfileResolution,
    Residency,
    Settings,
    get_settings,
    resolve_profile,
    resolve_residency,
)
from .common.errors import EngineUnavailable, hint_for
from .common.protocols import SAMPLE_RATE_IN, AudioChunk, Transcript, Voice
from .engines import registry
from .engines.vad import VadConfig, build_vad, gate_audio
from .engines.voices import VoiceRegistry

log = logging.getLogger(__name__)


class VoiceCore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.resolution: ProfileResolution | None = None
        self.stt = None
        self.tts = None
        self._tts_alternates: list = []
        self.vad = None
        self.vad_config = VadConfig()
        self.voices: VoiceRegistry | None = None
        self._stt_lock = asyncio.Lock()
        self._tts_lock = asyncio.Lock()
        self.residency = Residency.SHARED
        self.residency_reason = ""
        #: Set when an engine fails to load, so we can report it instead of
        #: pretending to be healthy and 500-ing on every request.
        self.stt_error: EngineUnavailable | None = None
        self.tts_error: EngineUnavailable | None = None
        #: Which engine currently owns the GPU in EXCLUSIVE mode.
        self._gpu_occupant: str | None = None
        self._ready = False

    # -- lifecycle ---------------------------------------------------------
    @property
    def profile(self) -> Profile:
        assert self.resolution is not None
        return self.resolution.profile

    @property
    def realtime_capable(self) -> bool:
        return bool(
            self._ready
            and self.stt_error is None
            and self.tts_error is None
            and self.stt.info.realtime_capable
            and self.tts.info.realtime_capable
        )

    async def startup(self, warmup: bool = True) -> None:
        if self._ready:
            return
        self.resolution = resolve_profile(self.settings)
        log.info(
            "Profile %s (%s) on device %s",
            self.resolution.profile.value,
            self.resolution.reason,
            self.resolution.device,
        )

        self.voices = VoiceRegistry(self.settings.voices_dir)
        self.stt = registry.build_stt(self.resolution, self.settings)
        self.tts = registry.build_tts(self.resolution, self.settings)
        self._tts_alternates = registry.build_tts_alternates(
            self.resolution, self.tts.info.engine_id, self.settings
        )
        self.vad = build_vad(self.vad_config)

        self.residency, self.residency_reason = resolve_residency(
            self.resolution, self.settings, self._vram_needed()
        )
        if self.residency is Residency.EXCLUSIVE:
            # STT and TTS alternate within a turn, so a single lock serializes
            # them and guarantees only one model is on the GPU at any moment.
            self._tts_lock = self._stt_lock
            log.info("GPU residency: exclusive (%s)", self.residency_reason)
            self._resolve_allocator_conflict()
        else:
            log.info("GPU residency: shared (%s)", self.residency_reason)

        if not self.stt.info.realtime_capable:
            log.warning(
                "STT engine %s is not realtime-capable (RTF ~%.2f): /v1/realtime will refuse "
                "connections. Batch transcription still works.",
                self.stt.info.engine_id,
                self.stt.info.rtf_estimate,
            )

        if warmup:
            await asyncio.to_thread(self._warmup)
        self._ready = True

    def _warmup(self) -> None:
        """Pay the lazy-init cost now, not on the user's first utterance.

        A failure here is recorded rather than raised: half a working gateway
        (say, TTS but no STT) is more useful than none, and /healthz reports
        which half is down.
        """
        t0 = time.perf_counter()
        try:
            self.stt.load()
            self._claim_gpu(self.stt)
            self.stt.transcribe(np.zeros(SAMPLE_RATE_IN, dtype=np.float32), "ar")
        except Exception as exc:
            self.stt_error = EngineUnavailable(self.stt.info.engine_id, str(exc), hint_for(exc))
            log.error("STT unavailable: %s", self.stt_error)
        try:
            self.tts.load()
            self._claim_gpu(self.tts)
            voice = self.voices.get(None)
            for _ in self.tts.synthesize("مرحبا", voice):
                break
        except Exception as exc:
            self.tts_error = EngineUnavailable(self.tts.info.engine_id, str(exc), hint_for(exc))
            log.error("TTS unavailable: %s", self.tts_error)
        log.info("Warmup finished in %.1fs", time.perf_counter() - t0)

    async def shutdown(self) -> None:
        for engine in (self.stt, self.tts, *self._tts_alternates):
            if engine is not None:
                engine.unload()
        self._ready = False

    # -- GPU residency -----------------------------------------------------
    def _vram_needed(self) -> int:
        """Summed VRAM footprint of the engines that would be resident at once.

        Only the *active* STT and TTS count: alternates are built lazily and
        never share a turn with the primary. Returns 0 if any engine declines
        to declare a footprint, which makes residency fall back to a threshold
        rather than act on a number it half-knows.
        """
        engines = [e for e in (self.stt, self.tts) if e is not None]
        if any(e.info.vram_mib == 0 for e in engines):
            return 0
        return sum(e.info.vram_mib for e in engines)

    def _resolve_allocator_conflict(self) -> None:
        """Keep every GPU-resident engine inside one CUDA allocator family.

        Exclusive residency swaps models in and out of VRAM. That only works if
        the memory one engine frees is visible to the next. PyTorch's caching
        allocator hangs on to hundreds of MiB of *reserved* VRAM after
        ``.to("cpu")`` -- ``empty_cache()`` cannot reclaim fragmented segments
        and ``expandable_segments`` is unsupported on Windows -- and that pool
        is invisible to CTranslate2. Mixing the two families therefore OOMs on
        a small card, so we demote the odd one out to the CPU up front.
        """
        engines = [e for e in (self.stt, self.tts, *self._tts_alternates) if e is not None]
        families = {e.info.allocator for e in engines} - {"none"}
        if len(families) < 2:
            return
        for engine in engines:
            if engine.info.allocator == "ct2" and hasattr(engine, "demote_to_cpu"):
                engine.demote_to_cpu(
                    "exclusive GPU residency cannot share VRAM between CTranslate2 "
                    "and PyTorch engines"
                )

    def _claim_gpu(self, engine) -> None:
        """In EXCLUSIVE mode, give `engine` the GPU and park the other model.

        Callers must already hold the (shared) engine lock.
        """
        if self.residency is not Residency.EXCLUSIVE:
            return
        if engine.info.allocator == "none":
            # Lives in host RAM: it needs no VRAM, so leave the occupant alone.
            return
        engine_id = engine.info.engine_id
        if self._gpu_occupant == engine_id:
            return

        t0 = time.perf_counter()
        for other in (self.stt, self.tts, *self._tts_alternates):
            if other is not None and other is not engine and hasattr(other, "offload"):
                other.offload()
        if hasattr(engine, "onload"):
            try:
                engine.onload()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if not hasattr(engine, "demote_to_cpu"):
                    raise
                # Another allocator is still holding VRAM we cannot reclaim.
                # Falling back to the CPU is slow but keeps the turn alive.
                engine.demote_to_cpu(f"GPU onload failed: {exc}")
        self._gpu_occupant = engine_id
        log.debug("GPU swapped to %s in %.0f ms", engine_id, (time.perf_counter() - t0) * 1000)

    def _run_stt(self, audio: np.ndarray, language: str) -> Transcript:
        self._claim_gpu(self.stt)
        return self.stt.transcribe(audio, language)

    # -- STT ---------------------------------------------------------------
    def _gate(self, audio: np.ndarray) -> np.ndarray:
        """Mandatory VAD pass: the ASR model hallucinates on silence."""
        return gate_audio(audio, self.vad_config, self.vad)

    async def transcribe(
        self, audio: np.ndarray, language: str | None = None, apply_vad: bool = True
    ) -> Transcript:
        if self.stt_error is not None:
            raise self.stt_error
        language = language or self.settings.default_language
        if apply_vad:
            audio = await asyncio.to_thread(self._gate, audio)
            if audio.size == 0:
                log.debug("VAD found no speech; returning empty transcript")
                return Transcript("", language, 0.0, self.stt.info.engine_id)
        async with self._stt_lock:
            return await asyncio.to_thread(self._run_stt, audio, language)

    async def transcribe_bytes(
        self, data: bytes, language: str | None = None, apply_vad: bool = True
    ) -> Transcript:
        audio = await asyncio.to_thread(decode_audio, data, SAMPLE_RATE_IN)
        return await self.transcribe(audio, language, apply_vad)

    # -- TTS ---------------------------------------------------------------
    def resolve_voice(self, voice_id: str | None) -> Voice:
        return self.voices.get(voice_id)

    def _engine_for(self, voice: Voice):
        if self.tts.supports(voice):
            return self.tts
        for alt in self._tts_alternates:
            if alt.supports(voice):
                log.info(
                    "Voice %s unsupported by %s; routing to %s",
                    voice.id,
                    self.tts.info.engine_id,
                    alt.info.engine_id,
                )
                return alt
        raise ValueError(
            f"No loaded TTS engine can render voice {voice.id!r} "
            f"(clone={voice.is_clone}, profile={self.profile.value})."
        )

    async def synthesize(
        self, text: str, voice_id: str | None = None, urgent: bool = False
    ) -> AsyncIterator[AudioChunk]:
        if self.tts_error is not None:
            raise self.tts_error
        voice = self.resolve_voice(voice_id)
        engine = self._engine_for(voice)

        async with self._tts_lock:
            queue: asyncio.Queue = asyncio.Queue(maxsize=8)
            loop = asyncio.get_running_loop()
            sentinel = object()

            def produce() -> None:
                try:
                    self._claim_gpu(engine)
                    for chunk in engine.synthesize(text, voice, urgent=urgent):
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
                except Exception as exc:  # surfaced to the consumer below
                    asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

            task = asyncio.create_task(asyncio.to_thread(produce))
            try:
                while True:
                    item = await queue.get()
                    if item is sentinel:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                task.cancel()

    # -- introspection -----------------------------------------------------
    def health(self) -> dict:
        if not self._ready:
            return {"status": "loading"}
        failures = {
            kind: {"engine": err.engine_id, "reason": err.reason, "hint": err.hint}
            for kind, err in (("stt", self.stt_error), ("tts", self.tts_error))
            if err is not None
        }
        info: dict = {
            "status": "degraded" if failures else "ok",
            "errors": failures,
            "profile": self.profile.value,
            "resolved_reason": self.resolution.reason,
            "device": self.resolution.device,
            "realtime_capable": self.realtime_capable,
            "residency": self.residency.value,
            "residency_reason": self.residency_reason,
            "gpu_occupant": self._gpu_occupant,
            "vad": getattr(self.vad, "name", None),
            "offline": self.settings.offline,
            # EngineInfo is a slots dataclass: it has no __dict__.
            "asr_engine": asdict(self.stt.info),
            "tts_engine": asdict(self.tts.info),
            "tts_alternates": [e.info.engine_id for e in self._tts_alternates],
            "voices": [v.id for v in self.voices.list()],
        }
        info.update(self._vram())
        return info

    def _vram(self) -> dict:
        if self.profile is Profile.CPU:
            return {"vram_free_mib": None, "vram_total_mib": None}
        try:
            import torch

            free, total = torch.cuda.mem_get_info()
            return {
                "vram_free_mib": free // (1024 * 1024),
                "vram_total_mib": total // (1024 * 1024),
            }
        except Exception:
            return {"vram_free_mib": None, "vram_total_mib": None}


_core: VoiceCore | None = None


def get_core() -> VoiceCore:
    global _core
    if _core is None:
        _core = VoiceCore()
    return _core
