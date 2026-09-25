"""Profile resolution and engine selection -- the CPU/GPU switching logic."""

import pytest

from voicegw.common.config import Profile, Settings, resolve_profile
from voicegw.engines import registry


@pytest.fixture(autouse=True)
def _no_cuda_leak(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def _no_gpu(monkeypatch):
    monkeypatch.setattr(
        registry_config := __import__("voicegw.common.config", fromlist=["_cuda_probe"]),
        "_cuda_probe",
        lambda _mib: (False, "no CUDA device", "cpu"),
    )
    return registry_config


def _with_gpu(monkeypatch, free_mib=8000):
    import voicegw.common.config as config

    monkeypatch.setattr(
        config, "_cuda_probe", lambda mib: (free_mib >= mib, f"fake GPU {free_mib} MiB", "cuda:0")
    )
    return config


def test_cpu_profile_is_forced_and_hides_cuda(monkeypatch):
    _with_gpu(monkeypatch)  # a GPU is present...
    res = resolve_profile(Settings(profile=Profile.CPU))
    assert res.profile is Profile.CPU  # ...and is still not used
    assert res.device == "cpu"
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_gpu_profile_raises_when_cuda_unusable(monkeypatch):
    _no_gpu(monkeypatch)
    with pytest.raises(RuntimeError, match="CUDA is unusable"):
        resolve_profile(Settings(profile=Profile.GPU))


def test_auto_picks_gpu_when_vram_is_sufficient(monkeypatch):
    _with_gpu(monkeypatch, free_mib=8000)
    res = resolve_profile(Settings(profile=Profile.AUTO, min_vram_mib=4500))
    assert res.profile is Profile.GPU
    assert res.device == "cuda:0"
    assert "auto-selected" in res.reason


def test_auto_falls_back_to_cpu_when_vram_is_short(monkeypatch):
    _with_gpu(monkeypatch, free_mib=2000)
    res = resolve_profile(Settings(profile=Profile.AUTO, min_vram_mib=4500))
    assert res.profile is Profile.CPU


def test_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    _no_gpu(monkeypatch)
    assert resolve_profile(Settings(profile=Profile.AUTO)).profile is Profile.CPU


def test_resolution_always_reports_a_reason(monkeypatch):
    _no_gpu(monkeypatch)
    assert resolve_profile(Settings(profile=Profile.AUTO)).reason


# --- engine selection ------------------------------------------------------


@pytest.mark.parametrize("profile", [Profile.CPU, Profile.GPU])
def test_every_profile_has_defaults_that_exist(profile):
    defaults = registry.PROFILE_DEFAULTS[profile]
    assert defaults["stt"] in registry.STT_ENGINES
    assert defaults["tts"] in registry.TTS_ENGINES


@pytest.mark.parametrize("profile", [Profile.CPU, Profile.GPU])
def test_tts_alternates_are_registered(profile):
    assert all(eid in registry.TTS_ENGINES for eid in registry.TTS_ALTERNATES[profile])


def test_cpu_default_stt_is_realtime_capable():
    """The CPU profile must stay usable for the realtime loop."""
    assert registry.PROFILE_DEFAULTS[Profile.CPU]["stt"] == "faster-whisper"


def test_pinned_engine_overrides_profile_default():
    from voicegw.common.config import ProfileResolution

    res = ProfileResolution(Profile.CPU, "test", "cpu")
    settings = Settings(profile=Profile.CPU, asr_engine="cohere-asr-cpu")
    assert registry._select("stt", registry.STT_ENGINES, settings.asr_engine, res) == (
        "cohere-asr-cpu"
    )


def test_unknown_engine_is_rejected():
    from voicegw.common.config import ProfileResolution

    res = ProfileResolution(Profile.CPU, "test", "cpu")
    with pytest.raises(ValueError, match="Unknown stt engine"):
        registry._select("stt", registry.STT_ENGINES, "does-not-exist", res)


def test_pinning_disables_alternates():
    """An explicit pin means 'use only this engine'."""
    from voicegw.common.config import ProfileResolution

    res = ProfileResolution(Profile.CPU, "test", "cpu")
    settings = Settings(profile=Profile.CPU, tts_engine="piper")
    assert registry.build_tts_alternates(res, "piper", settings) == []
