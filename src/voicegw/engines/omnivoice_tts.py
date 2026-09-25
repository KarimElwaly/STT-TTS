"""OmniVoice TTS engine (GPU and CPU variants).

Model: ``k2-fsa/OmniVoice`` -- 0.6B diffusion-LM zero-shot TTS, 600+ languages,
24 kHz output, voice cloning and voice design.

LICENSING: the *code* is Apache-2.0 but the *weights* are CC-BY-NC --
non-commercial use only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import numpy as np

from ..common.protocols import SAMPLE_RATE_OUT, AudioChunk, EngineInfo, Voice
from ..common.text_chunker import chunk_text

log = logging.getLogger(__name__)

MODEL_ID = "k2-fsa/OmniVoice"


#: Diffusion steps. The library default is 32, which costs ~1.4 s for a 4 s
#: utterance -- far too slow for conversation. 16 roughly halves that at a
#: modest quality cost; run `scripts/tune_tts.py`, listen, and override with
#: VOICEGW_TTS_NUM_STEP.
def default_num_step() -> int:
    """Read from Settings, so `.env` works -- not just the process environment."""
    from ..common.config import get_settings

    return get_settings().tts_num_step


#: OmniVoice uses ISO 639-3 style ids, so the ISO 639-1 codes the REST API
#: accepts ('ar', 'en') must be translated. An unmapped code makes the library
#: silently fall back to language-agnostic mode, which degrades pronunciation.
LANGUAGE_MAP = {
    "ar": "arb",  # Modern Standard Arabic
    "ar-eg": "arz",  # Egyptian
    "ar-lv": "apc",  # Levantine
    "ar-gf": "afb",  # Gulf
    "ar-ma": "ary",  # Moroccan
    "ar-iq": "acm",  # Mesopotamian
    "ar-sa": "ars",  # Najdi
    "ar-tn": "aeb",  # Tunisian
    "en": "en",
}


def resolve_language(code: str | None) -> str | None:
    if not code:
        return None
    key = code.lower().replace("_", "-")
    if key in LANGUAGE_MAP:
        return LANGUAGE_MAP[key]
    try:
        from omnivoice.utils.lang_map import LANG_IDS

        if code in LANG_IDS:
            return code
    except ImportError:
        pass
    log.warning("Unknown TTS language %r; using language-agnostic mode", code)
    return None


class OmniVoiceTts:
    def __init__(
        self,
        engine_id: str,
        device: str,
        dtype: str,
        rtf_estimate: float,
        realtime_capable: bool,
        notes: str = "",
        num_step: int | None = None,
    ) -> None:
        self.device = device
        self.dtype_name = dtype
        #: Diffusion steps. The library default is 32; fewer is faster but
        #: lower quality. This is the main TTS latency knob.
        self.num_step = num_step
        self._model = None
        self._offloaded = False
        self.info = EngineInfo(
            engine_id=engine_id,
            model_id=MODEL_ID,
            device=device,
            dtype=dtype,
            rtf_estimate=rtf_estimate,
            realtime_capable=realtime_capable,
            notes=notes,
        )

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from omnivoice import OmniVoice

        t0 = time.perf_counter()
        self._model = OmniVoice.from_pretrained(
            MODEL_ID,
            device_map=self.device,
            dtype=getattr(torch, self.dtype_name),
        )
        log.info("Loaded %s on %s in %.1fs", MODEL_ID, self.device, time.perf_counter() - t0)

    def unload(self) -> None:
        self._model = None
        self._offloaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- GPU residency -----------------------------------------------------
    def offload(self) -> None:
        if self._model is None or self.device == "cpu" or self._offloaded:
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

    def supports(self, voice: Voice) -> bool:
        # OmniVoice handles both cloning (ref audio) and voice design (instruct).
        return True

    def _generate(self, text: str, voice: Voice) -> np.ndarray:
        from omnivoice import OmniVoiceGenerationConfig

        kwargs: dict[str, object] = {"text": text, "language": resolve_language(voice.language)}
        if voice.is_clone:
            kwargs["ref_audio"] = voice.ref_audio
            if voice.ref_text:
                kwargs["ref_text"] = voice.ref_text
        elif voice.description:
            # NOTE: the parameter is `instruct`. `generate()` swallows unknown
            # keywords via **kwargs, so a wrong name fails silently and yields
            # a random voice instead of the designed one.
            kwargs["instruct"] = voice.description

        if self.num_step is not None:
            kwargs["generation_config"] = OmniVoiceGenerationConfig(num_step=self.num_step)

        audio = self._model.generate(**kwargs)
        # `.generate` returns a list of np.ndarray (T,) at 24 kHz.
        if isinstance(audio, list):
            audio = audio[0]
        return np.asarray(audio, dtype=np.float32)

    def synthesize(self, text: str, voice: Voice) -> Iterator[AudioChunk]:
        self.load()
        chunks = chunk_text(text)
        if not chunks:
            yield AudioChunk(np.zeros(0, dtype=np.float32), SAMPLE_RATE_OUT, is_final=True)
            return

        for i, piece in enumerate(chunks):
            samples = self._generate(piece, voice)
            yield AudioChunk(samples, SAMPLE_RATE_OUT, is_final=(i == len(chunks) - 1))


def build_gpu(device: str = "cuda:0") -> OmniVoiceTts:
    return OmniVoiceTts(
        engine_id="omnivoice",
        device=device,
        dtype="float16",
        rtf_estimate=0.025,
        realtime_capable=True,
        notes="Zero-shot cloning + voice design. Weights are CC-BY-NC.",
        num_step=default_num_step(),
    )


def build_cpu() -> OmniVoiceTts:
    return OmniVoiceTts(
        engine_id="omnivoice-cpu",
        device="cpu",
        dtype="float32",
        # 0.6B on CPU is borderline; scripts/bench.py replaces this estimate
        # with a measured value and decides whether Piper should take over.
        rtf_estimate=0.8,
        realtime_capable=True,
        notes="CPU cloning. Benchmark before trusting for realtime.",
        num_step=default_num_step(),
    )
