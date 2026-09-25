"""Offline operation: run entirely from locally cached weights.

Every model this gateway uses lives in the Hugging Face cache, so "offline"
means two things:

1. Nothing may reach the network at load time (``local_files_only``), and
2. a missing file must fail immediately with a clear message instead of
   hanging on a connection attempt or retry loop.

:func:`enable` must run *before* ``huggingface_hub`` is imported, because that
library snapshots ``HF_HUB_OFFLINE`` into a module constant at import time.
Engines also pass ``local_files_only`` explicitly, so offline mode still holds
if something imported the hub early.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Env vars that make the HF stack refuse network access.
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    # Progress bars for downloads that will never happen are just noise.
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
}


#: OmniVoice falls back to this separate repo when its snapshot does not
#: bundle an ``audio_tokenizer/`` subdirectory.
AUDIO_TOKENIZER_REPO = "eustlb/higgs-audio-v2-tokenizer"


@dataclass(frozen=True, slots=True)
class ModelAsset:
    """A downloadable weight set, identified by its Hugging Face repo."""

    key: str
    repo_id: str
    purpose: str
    #: Gated repos need an accepted licence and a token before they can be
    #: fetched even once.
    gated: bool = False
    #: Approximate on-disk size, for the pre-fetch summary.
    size_hint: str = ""


def whisper_repo(model_id: str) -> str:
    """Map a faster-whisper size alias to the HF repo it downloads from."""
    if "/" in model_id:
        return model_id
    from faster_whisper.utils import _MODELS

    return _MODELS.get(model_id, model_id)


def assets(settings=None) -> list[ModelAsset]:
    """Every model the current configuration could need, in load order."""
    from ..engines import cohere_asr, faster_whisper_asr, omnivoice_tts

    return [
        ModelAsset(
            "cohere-asr",
            cohere_asr.MODEL_ID,
            "Arabic ASR (gpu profile default)",
            gated=True,
            size_hint="~4 GB",
        ),
        ModelAsset(
            "omnivoice",
            omnivoice_tts.MODEL_ID,
            "TTS (both profiles)",
            size_hint="~1.5 GB",
        ),
        ModelAsset(
            "faster-whisper",
            whisper_repo(faster_whisper_asr.pinned_model() or faster_whisper_asr.CPU_MODEL),
            "ASR (cpu profile / small-GPU fallback)",
            size_hint="~0.5-1.6 GB",
        ),
    ]


def is_cached(repo_id: str) -> bool:
    """True if `repo_id` can be loaded with no network access."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return False
    return True


def cache_dir() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def enable() -> None:
    """Forbid network access for model loading, process-wide.

    Setting the env vars is not sufficient on its own: ``huggingface_hub``
    copies ``HF_HUB_OFFLINE`` into ``constants.HF_HUB_OFFLINE`` at import time,
    so if anything imported it first the env var is silently ignored. Patch the
    constant too when the module is already loaded. Verified: patching the
    constant does block a download; setting the env var afterwards does not.
    """
    for key, value in _OFFLINE_ENV.items():
        os.environ[key] = value

    hub = sys.modules.get("huggingface_hub")
    if hub is not None:
        from huggingface_hub import constants

        constants.HF_HUB_OFFLINE = True

    log.info("Offline mode: models load from %s, no network access", cache_dir())


def is_enabled() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1"


def needs_external_audio_tokenizer() -> bool:
    """True if OmniVoice would have to fetch the tokenizer as a separate repo.

    Current releases bundle it as an ``audio_tokenizer/`` subdirectory, but the
    loader falls back to a second repo when that is absent -- which would
    reach the network on an otherwise-offline box.
    """
    from ..engines import omnivoice_tts

    try:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(omnivoice_tts.MODEL_ID, local_files_only=True))
    except Exception:
        return False  # the main repo is not cached; that is the bigger problem
    return not (root / "audio_tokenizer").is_dir()


def missing(settings=None) -> list[ModelAsset]:
    """Assets that offline mode would fail to load."""
    required = assets(settings)
    if needs_external_audio_tokenizer():
        required.append(
            ModelAsset(
                "audio-tokenizer",
                AUDIO_TOKENIZER_REPO,
                "OmniVoice audio tokenizer (not bundled in this revision)",
                size_hint="~1 GB",
            )
        )
    return [a for a in required if not is_cached(a.repo_id)]
