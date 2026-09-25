"""Cohere Transcribe Arabic STT engine (GPU and CPU variants).

Model: ``CohereLabs/cohere-transcribe-arabic-07-2026`` -- 2B conformer
encoder-decoder, Apache-2.0, but the repo is **gated**: you must accept the
terms on the model page with your Hugging Face account and provide ``HF_TOKEN``.

The model has no automatic language detection, no timestamps and no
diarization. It is also eager to transcribe silence, so callers must gate audio
through :mod:`voicegw.engines.vad` first.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from ..common.config import Quantization, get_settings
from ..common.protocols import SAMPLE_RATE_IN, EngineInfo, Transcript
from .vad import chunk_long_audio

log = logging.getLogger(__name__)

MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
MAX_NEW_TOKENS = 256

#: Measured loaded-weight footprint per quantization mode, in MiB
#: (RTX 4050 Laptop, `python scripts/quant_compare.py`).
VRAM_MIB = {
    Quantization.NONE: 3940,
    Quantization.INT8: 2244,
    Quantization.NF4: 1413,
}

#: Measured single-utterance latency per mode, in ms, on the same card.
#: Used only for the RTF estimate the realtime loop gates on.
RTF_ESTIMATE = {
    Quantization.NONE: 0.08,
    Quantization.INT8: 0.40,
    Quantization.NF4: 0.12,
}


def resolve_quantization(settings=None) -> Quantization:
    """Pick a weight precision, shrinking the model only when it has to.

    Full precision is both the fastest and the most faithful, so ``auto`` only
    quantizes when the alternative is worse: a card that cannot hold STT and
    TTS at once pays ~3.3 s of offload/onload traffic *per turn*, which dwarfs
    the ~100 ms that nf4 adds to a transcription.
    """
    settings = settings or get_settings()
    if settings.asr_quantization is not Quantization.AUTO:
        return settings.asr_quantization

    try:
        import torch

        free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
    except Exception:  # pragma: no cover - driver quirks
        return Quantization.NF4

    # OmniVoice needs ~1937 MiB; see scripts/vram_budget.py.
    both_fit = VRAM_MIB[Quantization.NONE] + 1937 + settings.vram_headroom_mib <= free_mib
    return Quantization.NONE if both_fit else Quantization.NF4


class CohereAsr:
    """Shared implementation; the CPU/GPU split is dtype + device + threading."""

    def __init__(
        self,
        engine_id: str,
        device: str,
        dtype: str,
        rtf_estimate: float,
        realtime_capable: bool,
        notes: str = "",
        model_id: str = MODEL_ID,
        quantization: Quantization = Quantization.NONE,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype_name = dtype
        self.quantization = quantization
        self._model = None
        self._processor = None
        self._offloaded = False
        self.info = EngineInfo(
            engine_id=engine_id,
            model_id=model_id,
            device=device,
            dtype=dtype if quantization is Quantization.NONE else f"{quantization.value}/{dtype}",
            rtf_estimate=rtf_estimate,
            realtime_capable=realtime_capable,
            notes=notes,
            vram_mib=VRAM_MIB[quantization] if device != "cpu" else 0,
            # bitsandbytes pins quantized weights to the device they were
            # quantized on; `.to()` raises. They are small enough not to need
            # swapping anyway -- that is the whole point of quantizing.
            movable=quantization is Quantization.NONE,
        )

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        settings = get_settings()
        token = settings.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            log.warning(
                "HF_TOKEN is not set. %s is a gated repo -- accept the terms at "
                "https://huggingface.co/%s and export a read token.",
                self.model_id,
                self.model_id,
            )

        torch_dtype = getattr(torch, self.dtype_name)
        if self.device == "cpu":
            # Saturate the laptop's physical cores; oversubscribing hurts.
            threads = max(1, (os.cpu_count() or 4) // 2)
            torch.set_num_threads(threads)
            log.info("CohereAsr CPU: torch threads=%d", threads)

        t0 = time.perf_counter()
        # Explicit, not just HF_HUB_OFFLINE: the env var is snapshotted at
        # import time, so it may have been read before we set it.
        local_only = settings.offline
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, token=token, local_files_only=local_only
        )
        self._model = CohereAsrForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            token=token,
            local_files_only=local_only,
            **self._placement_kwargs(),
        )
        if self.device == "cpu":
            self._model.to("cpu")
        self._model.eval()
        log.info(
            "Loaded %s on %s (%s) in %.1fs",
            self.model_id,
            self.device,
            self.info.dtype,
            time.perf_counter() - t0,
        )

    def _placement_kwargs(self) -> dict:
        """Device/quantization arguments for ``from_pretrained``."""
        if self.device == "cpu":
            return {}
        if self.quantization is Quantization.NONE:
            return {"device_map": self.device}

        import torch
        from transformers import BitsAndBytesConfig

        if self.quantization is Quantization.INT8:
            config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        # bitsandbytes dispatches the shards itself; passing device_map
        # alongside it is redundant and fights that dispatch.
        return {"quantization_config": config}

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._offloaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- GPU residency -----------------------------------------------------
    def offload(self) -> None:
        """Park the weights in system RAM, freeing VRAM but avoiding a reload."""
        if self._model is None or self.device == "cpu" or self._offloaded:
            return
        if not self.info.movable:
            # Quantized weights cannot leave the device they were quantized
            # on. Ignoring the request is safe rather than wrong: such a model
            # is small by construction, which is why residency chose SHARED.
            log.debug("%s is quantized and stays resident", self.info.engine_id)
            return
        import torch

        self._model.to("cpu")
        self._offloaded = True
        torch.cuda.empty_cache()

    def onload(self) -> None:
        if self._model is None or self.device == "cpu" or not self._offloaded:
            return
        self._model.to(self.device)
        self._offloaded = False

    # -- inference ---------------------------------------------------------
    def _transcribe_one(self, audio: np.ndarray, language: str) -> str:
        import torch

        inputs = self._processor(
            audio,
            sampling_rate=SAMPLE_RATE_IN,
            return_tensors="pt",
            language=language,
        )
        # The feature extractor emits float32 while the encoder's bias is
        # bf16, and the mismatch raises instead of promoting. A quantized
        # model reports a misleading `.dtype` (its weights are uint8), so key
        # off the compute dtype we asked bitsandbytes for.
        inputs = inputs.to(self._model.device, dtype=self._compute_dtype(torch))
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        # `generate` returns a (batch, seq) tensor, so `decode` on it yields a
        # *list*. Use batch_decode and take the single row we asked for.
        decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

    def _compute_dtype(self, torch):
        if self.quantization is Quantization.NONE:
            return getattr(torch, self.dtype_name)
        return torch.bfloat16

    def transcribe(self, audio: np.ndarray, language: str = "ar") -> Transcript:
        self.load()
        duration = len(audio) / SAMPLE_RATE_IN
        if audio.size == 0:
            return Transcript("", language, 0.0, self.info.engine_id)

        pieces = [
            self._transcribe_one(chunk, language) for chunk in chunk_long_audio(audio) if chunk.size
        ]
        return Transcript(" ".join(p for p in pieces if p), language, duration, self.info.engine_id)


def build_gpu(device: str = "cuda:0") -> CohereAsr:
    quant = resolve_quantization()
    notes = "Best Arabic/dialect accuracy. Requires accepted gated-repo terms."
    if quant is not Quantization.NONE:
        notes += f" Quantized to {quant.value} so TTS can stay resident too."
    return CohereAsr(
        engine_id="cohere-asr",
        device=device,
        dtype="bfloat16",
        rtf_estimate=RTF_ESTIMATE[quant],
        realtime_capable=True,
        notes=notes,
        quantization=quant,
    )


def build_cpu() -> CohereAsr:
    return CohereAsr(
        engine_id="cohere-asr-cpu",
        device="cpu",
        dtype="float32",
        rtf_estimate=3.0,
        # 2B params in fp32 on a laptop CPU: accurate but far slower than
        # realtime. Offered for offline file transcription only; the realtime
        # loop refuses it and the CPU profile defaults to faster-whisper.
        realtime_capable=False,
        notes="Accurate but slow (~3x realtime). Batch/file transcription only.",
    )
