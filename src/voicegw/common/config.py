"""Runtime configuration and profile resolution.

The single switch that decides whether the gateway runs on CPU or GPU is
``VOICEGW_PROFILE``. Everything downstream of :func:`resolve_profile` is
profile-agnostic: the profile only selects *which engine implementations* are
constructed, never the shape of the API.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class Profile(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    AUTO = "auto"


class Residency(str, Enum):
    """How STT and TTS share the GPU."""

    #: Both models stay resident. Fastest, needs enough VRAM for both.
    SHARED = "shared"
    #: Only the model in use holds VRAM; the other parks in system RAM.
    #: Costs a few hundred ms per swap but prevents OOM/thrashing on small GPUs.
    EXCLUSIVE = "exclusive"
    #: Pick based on total VRAM at startup.
    AUTO = "auto"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOICEGW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    profile: Profile = Profile.AUTO

    # Explicit engine pins win over profile defaults.
    asr_engine: str | None = None
    tts_engine: str | None = None

    host: str = "127.0.0.1"
    port: int = 8000
    default_language: str = "ar"
    voices_dir: Path = Path("voices")

    #: Free VRAM needed for `auto` to pick the gpu profile.
    #: Cohere ASR bf16 (~4.0 GiB) + OmniVoice fp16 (~1.2 GiB) plus activations.
    min_vram_mib: int = 4500

    residency: Residency = Residency.AUTO
    #: Below this much *total* VRAM, `residency=auto` chooses EXCLUSIVE.
    shared_residency_min_vram_mib: int = 10_000

    # Realtime agent hook (any OpenAI-compatible chat completions endpoint).
    agent_base_url: str = "http://localhost:11434/v1"
    agent_model: str = "qwen2.5:7b"
    agent_api_key: str = "not-needed"
    agent_system_prompt: str = "أنت مساعد صوتي مفيد. أجب بإيجاز."

    #: OmniVoice diffusion steps -- the dominant TTS latency/quality knob.
    tts_num_step: int = 16
    #: Override the faster-whisper size for both profiles (e.g. "medium").
    whisper_model: str | None = None

    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")

    #: Load every model from the local Hugging Face cache and never touch the
    #: network. Run `voicegw fetch` once while online to populate it.
    offline: bool = False


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.offline:
        from .offline import enable

        enable()
    return settings


class ProfileResolution:
    """The concrete profile chosen at startup, plus why."""

    def __init__(self, profile: Profile, reason: str, device: str) -> None:
        if profile is Profile.AUTO:  # pragma: no cover - defensive
            raise ValueError("resolved profile cannot be AUTO")
        self.profile = profile
        self.reason = reason
        self.device = device

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ProfileResolution(profile={self.profile.value!r}, device={self.device!r})"


def _cuda_probe(min_vram_mib: int) -> tuple[bool, str, str]:
    """Return ``(usable, reason, device)`` without importing torch on the CPU path."""
    try:
        import torch
    except ImportError:
        return False, "torch is not installed", "cpu"

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False", "cpu"

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception as exc:  # pragma: no cover - driver quirks
        return True, f"CUDA present (VRAM probe failed: {exc})", "cuda:0"

    free_mib = free_bytes // (1024 * 1024)
    name = torch.cuda.get_device_name(0)
    if free_mib < min_vram_mib:
        return (
            False,
            f"{name} has only {free_mib} MiB free, need {min_vram_mib} MiB",
            "cpu",
        )
    return True, f"{name} with {free_mib} MiB free", "cuda:0"


def resolve_profile(settings: Settings | None = None) -> ProfileResolution:
    settings = settings or get_settings()

    if settings.profile is Profile.CPU:
        # Make the choice binding: keep CUDA out of the process entirely so a
        # stray `.to("cuda")` fails loudly instead of silently using the GPU.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        return ProfileResolution(Profile.CPU, "forced by VOICEGW_PROFILE=cpu", "cpu")

    if settings.profile is Profile.GPU:
        usable, reason, device = _cuda_probe(settings.min_vram_mib)
        if not usable:
            raise RuntimeError(
                f"VOICEGW_PROFILE=gpu was requested but CUDA is unusable: {reason}. "
                "Set VOICEGW_PROFILE=cpu or auto."
            )
        return ProfileResolution(Profile.GPU, f"forced by VOICEGW_PROFILE=gpu; {reason}", device)

    usable, reason, device = _cuda_probe(settings.min_vram_mib)
    if usable:
        return ProfileResolution(Profile.GPU, f"auto-selected: {reason}", device)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    return ProfileResolution(Profile.CPU, f"auto-selected: {reason}", "cpu")


def resolve_residency(resolution: ProfileResolution, settings: Settings) -> tuple[Residency, str]:
    """Decide whether STT and TTS may hold VRAM at the same time."""
    if resolution.profile is Profile.CPU:
        return Residency.SHARED, "cpu profile: no VRAM contention"
    if settings.residency is not Residency.AUTO:
        return settings.residency, f"forced by VOICEGW_RESIDENCY={settings.residency.value}"

    try:
        import torch

        total_mib = torch.cuda.mem_get_info()[1] // (1024 * 1024)
    except Exception as exc:  # pragma: no cover - driver quirks
        return Residency.EXCLUSIVE, f"VRAM probe failed ({exc}); assuming a small GPU"

    if total_mib < settings.shared_residency_min_vram_mib:
        return (
            Residency.EXCLUSIVE,
            f"{total_mib} MiB total VRAM is too small to hold both models at once",
        )
    return Residency.SHARED, f"{total_mib} MiB total VRAM fits both models"
