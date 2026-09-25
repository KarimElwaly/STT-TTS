"""faster-whisper STT engine -- the CPU profile's realtime lane.

CTranslate2 int8 inference is genuinely faster than realtime on a laptop CPU,
which the 2B Cohere model is not. Arabic (especially dialectal) accuracy is
lower; ``scripts/bench.py`` quantifies the gap on your own audio.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ..common.protocols import SAMPLE_RATE_IN, EngineInfo, Transcript

log = logging.getLogger(__name__)

#: Set VOICEGW_WHISPER_MODEL to pin a size and override the per-device picks.
GPU_MODEL = "large-v3-turbo"
#: large-v3-turbo measures RTF ~2.2 on a laptop CPU -- far too slow for the
#: realtime loop -- so the CPU lane drops to a size that actually keeps up.
CPU_MODEL = "small"


def pinned_model() -> str | None:
    """Read from Settings, so `.env` works -- not just the process environment."""
    from ..common.config import get_settings

    return get_settings().whisper_model or None


#: Rough int8 CPU real-time factors on a laptop. ``large-v3-turbo`` is measured
#: (~2.2 on an 8-core laptop); the rest scale by parameter count. Replace these
#: with your own numbers from ``scripts/bench.py``.
_CPU_RTF = {
    "tiny": 0.06,
    "base": 0.11,
    "small": 0.30,
    "medium": 0.90,
    "large-v3-turbo": 2.2,
    "large-v3": 5.0,
}


def _cpu_rtf(model_id: str) -> float:
    for name, rtf in sorted(_CPU_RTF.items(), key=lambda kv: -len(kv[0])):
        if name in model_id:
            return rtf
    return 1.0


class FasterWhisperAsr:
    def __init__(
        self,
        model_id: str = GPU_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._offloaded = False
        rtf = 0.05 if device != "cpu" else _cpu_rtf(model_id)
        self.info = EngineInfo(
            engine_id="faster-whisper",
            model_id=model_id,
            device=device,
            dtype=compute_type,
            rtf_estimate=rtf,
            realtime_capable=rtf < 1.0,
            notes="CPU realtime lane. Lower dialectal-Arabic accuracy than Cohere.",
            allocator="ct2" if device != "cpu" else "none",
        )

    def demote_to_cpu(self, reason: str) -> None:
        """Permanently move this engine to the CPU.

        CTranslate2 has its own CUDA allocator and cannot use the VRAM PyTorch
        keeps cached, so on a small GPU a torch/CT2 swap can OOM forever. Going
        to the CPU is slower but always works.
        """
        if self.device == "cpu":
            return
        log.warning("Demoting faster-whisper to CPU: %s", reason)
        self.device = "cpu"
        self.compute_type = "int8"
        self.model_id = pinned_model() or CPU_MODEL
        self._model = None
        self._offloaded = False
        self.info.model_id = self.model_id
        self.info.device = "cpu"
        self.info.dtype = "int8"
        self.info.rtf_estimate = _cpu_rtf(self.model_id)
        self.info.realtime_capable = self.info.rtf_estimate < 1.0
        self.info.allocator = "none"
        self.info.notes = f"Demoted to CPU: {reason}"

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        from ..common.config import get_settings

        cpu_threads = max(1, (os.cpu_count() or 4) // 2)
        self._model = WhisperModel(
            self.model_id,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=cpu_threads if self.device == "cpu" else 0,
            local_files_only=get_settings().offline,
        )
        log.info("Loaded faster-whisper %s (%s/%s)", self.model_id, self.device, self.compute_type)

    def unload(self) -> None:
        self._model = None
        self._offloaded = False

    # -- GPU residency -----------------------------------------------------
    def offload(self) -> None:
        """CTranslate2 can park its weights in host RAM without a full reload."""
        if self._model is None or self.device == "cpu" or self._offloaded:
            return
        self._model.model.unload_model(to_cpu=True)
        self._offloaded = True

    def onload(self) -> None:
        if self._model is None or self.device == "cpu" or not self._offloaded:
            return
        self._model.model.load_model()
        self._offloaded = False

    def transcribe(self, audio: np.ndarray, language: str = "ar") -> Transcript:
        self.load()
        duration = len(audio) / SAMPLE_RATE_IN
        if audio.size == 0:
            return Transcript("", language, 0.0, self.info.engine_id)

        segments, _info = self._model.transcribe(
            audio.astype(np.float32),
            language=language,
            beam_size=1,
            condition_on_previous_text=False,
            # Our own VAD already gated the audio; this is a second safety net.
            vad_filter=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(text, language, duration, self.info.engine_id)


def build_cpu() -> FasterWhisperAsr:
    return FasterWhisperAsr(model_id=pinned_model() or CPU_MODEL, device="cpu", compute_type="int8")


def build_gpu() -> FasterWhisperAsr:
    return FasterWhisperAsr(
        model_id=pinned_model() or GPU_MODEL, device="cuda", compute_type="float16"
    )
