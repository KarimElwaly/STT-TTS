"""Engine registry -- the *only* module that knows about concrete model classes.

Everything else resolves engines through :func:`build_stt` / :func:`build_tts`,
which is what keeps the REST, WebSocket and MCP layers identical across the
``cpu`` and ``gpu`` profiles.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..common.config import Profile, ProfileResolution, Settings, get_settings
from ..common.protocols import SttEngine, TtsEngine

log = logging.getLogger(__name__)


def _cohere_gpu(res: ProfileResolution) -> Any:
    from . import cohere_asr

    return cohere_asr.build_gpu(res.device)


def _cohere_cpu(_res: ProfileResolution) -> Any:
    from . import cohere_asr

    return cohere_asr.build_cpu()


def _whisper(res: ProfileResolution) -> Any:
    from . import faster_whisper_asr

    if res.profile is Profile.GPU:
        return faster_whisper_asr.build_gpu()
    return faster_whisper_asr.build_cpu()


def _omnivoice_gpu(res: ProfileResolution) -> Any:
    from . import omnivoice_tts

    return omnivoice_tts.build_gpu(res.device)


def _omnivoice_cpu(_res: ProfileResolution) -> Any:
    from . import omnivoice_tts

    return omnivoice_tts.build_cpu()


def _piper(_res: ProfileResolution) -> Any:
    from . import piper_tts

    return piper_tts.build_cpu()


def _omnivoice_gguf(res: ProfileResolution) -> Any:
    from . import omnivoice_gguf

    if res.profile is Profile.GPU:
        return omnivoice_gguf.build_gpu(res.device)
    return omnivoice_gguf.build_cpu()


STT_ENGINES: dict[str, Callable[[ProfileResolution], Any]] = {
    "cohere-asr": _cohere_gpu,
    "cohere-asr-cpu": _cohere_cpu,
    "faster-whisper": _whisper,
}

TTS_ENGINES: dict[str, Callable[[ProfileResolution], Any]] = {
    "omnivoice": _omnivoice_gpu,
    "omnivoice-cpu": _omnivoice_cpu,
    "omnivoice-gguf": _omnivoice_gguf,
    "piper": _piper,
}


#: Defaults per profile. The CPU STT default is faster-whisper rather than
#: Cohere-on-CPU because the 2B model cannot hit the conversational latency
#: budget on a laptop CPU -- run ``scripts/bench.py`` to confirm on your box,
#: and pin VOICEGW_ASR_ENGINE=cohere-asr-cpu if you prefer accuracy over speed.
PROFILE_DEFAULTS: dict[Profile, dict[str, str]] = {
    Profile.GPU: {"stt": "cohere-asr", "tts": "omnivoice"},
    Profile.CPU: {"stt": "faster-whisper", "tts": "omnivoice-cpu"},
}

#: Used when the requested voice is not supported by the primary engine.
TTS_ALTERNATES: dict[Profile, list[str]] = {
    Profile.GPU: ["omnivoice"],
    Profile.CPU: ["omnivoice-cpu", "piper"],
}


def _select(kind: str, table: dict, pinned: str | None, res: ProfileResolution) -> str:
    if pinned:
        if pinned not in table:
            raise ValueError(
                f"Unknown {kind} engine {pinned!r}. Available: {', '.join(sorted(table))}"
            )
        return pinned
    return PROFILE_DEFAULTS[res.profile][kind]


def build_stt(res: ProfileResolution, settings: Settings | None = None) -> SttEngine:
    settings = settings or get_settings()
    engine_id = _select("stt", STT_ENGINES, settings.asr_engine, res)
    engine = STT_ENGINES[engine_id](res)
    log.info("STT engine: %s (%s, %s)", engine_id, engine.info.device, engine.info.dtype)
    return engine


def build_tts(res: ProfileResolution, settings: Settings | None = None) -> TtsEngine:
    settings = settings or get_settings()
    engine_id = _select("tts", TTS_ENGINES, settings.tts_engine, res)
    engine = TTS_ENGINES[engine_id](res)
    log.info("TTS engine: %s (%s, %s)", engine_id, engine.info.device, engine.info.dtype)
    return engine


def build_tts_alternates(
    res: ProfileResolution, primary_id: str, settings: Settings | None = None
) -> list[TtsEngine]:
    """Secondary TTS engines for voices the primary can't render."""
    settings = settings or get_settings()
    if settings.tts_engine:  # explicit pin means "use only this one"
        return []
    return [TTS_ENGINES[eid](res) for eid in TTS_ALTERNATES[res.profile] if eid != primary_id]
