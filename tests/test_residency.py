"""GPU residency: only one model may hold VRAM on a small card."""

import numpy as np
import pytest

from voicegw.common.config import Profile, ProfileResolution, Residency, Settings, resolve_residency
from voicegw.common.protocols import EngineInfo, Transcript


class FakeEngine:
    def __init__(self, engine_id: str):
        self.info = EngineInfo(engine_id, "fake", "cuda:0", "float16", 0.1, True)
        self.on_gpu = True
        self.offload_calls = 0
        self.onload_calls = 0

    def offload(self):
        self.on_gpu = False
        self.offload_calls += 1

    def onload(self):
        self.on_gpu = True
        self.onload_calls += 1

    def load(self):
        pass

    def unload(self):
        pass

    def transcribe(self, audio, language="ar"):
        assert self.on_gpu, "transcribe ran while offloaded"
        return Transcript("ok", language, 0.0, self.info.engine_id)

    def supports(self, voice):
        return True


GPU = ProfileResolution(Profile.GPU, "test", "cuda:0")
CPU = ProfileResolution(Profile.CPU, "test", "cpu")


# --- resolution ------------------------------------------------------------


def test_cpu_profile_never_needs_exclusive():
    mode, reason = resolve_residency(CPU, Settings())
    assert mode is Residency.SHARED
    assert "cpu" in reason


def test_explicit_setting_wins(monkeypatch):
    mode, reason = resolve_residency(GPU, Settings(residency=Residency.EXCLUSIVE))
    assert mode is Residency.EXCLUSIVE
    assert "forced" in reason


def test_small_gpu_auto_selects_exclusive(monkeypatch):
    import torch

    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda: (1 << 30, 6141 * 1024 * 1024), raising=False
    )
    mode, reason = resolve_residency(GPU, Settings(shared_residency_min_vram_mib=10_000))
    assert mode is Residency.EXCLUSIVE
    assert "too small" in reason


def test_large_gpu_auto_selects_shared(monkeypatch):
    import torch

    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda: (1 << 30, 24_000 * 1024 * 1024), raising=False
    )
    mode, _ = resolve_residency(GPU, Settings(shared_residency_min_vram_mib=10_000))
    assert mode is Residency.SHARED


def test_probe_failure_is_conservative(monkeypatch):
    import torch

    def boom():
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "mem_get_info", boom, raising=False)
    mode, reason = resolve_residency(GPU, Settings())
    assert mode is Residency.EXCLUSIVE
    assert "probe failed" in reason


# --- footprint-based resolution -------------------------------------------
#
# A declared footprint turns the decision from a guess into arithmetic, which
# is what lets a quantized ASR model flip a 6 GiB card to SHARED.


def _free(monkeypatch, mib: int):
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda: (mib * 1024 * 1024, 6141 * 1024 * 1024),
        raising=False,
    )


def test_declared_footprint_beats_the_total_vram_threshold(monkeypatch):
    """nf4 ASR (1413) + OmniVoice (1937) fits in 5080 MiB free."""
    _free(monkeypatch, 5080)
    settings = Settings(shared_residency_min_vram_mib=10_000, vram_headroom_mib=700)
    mode, reason = resolve_residency(GPU, settings, needed_mib=1413 + 1937)
    assert mode is Residency.SHARED
    assert "3350 MiB" in reason


def test_full_precision_footprint_still_forces_exclusive(monkeypatch):
    """bf16 ASR (3940) + OmniVoice (1937) does not."""
    _free(monkeypatch, 5080)
    settings = Settings(shared_residency_min_vram_mib=10_000, vram_headroom_mib=700)
    mode, reason = resolve_residency(GPU, settings, needed_mib=3940 + 1937)
    assert mode is Residency.EXCLUSIVE
    assert "only 5080 MiB is free" in reason


def test_headroom_is_respected(monkeypatch):
    """A footprint that fits only by ignoring activations must not pass."""
    _free(monkeypatch, 3400)
    settings = Settings(vram_headroom_mib=700)
    mode, _ = resolve_residency(GPU, settings, needed_mib=3350)
    assert mode is Residency.EXCLUSIVE


def test_unknown_footprint_falls_back_to_threshold(monkeypatch):
    _free(monkeypatch, 5080)
    settings = Settings(shared_residency_min_vram_mib=10_000)
    mode, reason = resolve_residency(GPU, settings, needed_mib=0)
    assert mode is Residency.EXCLUSIVE
    assert "total VRAM" in reason


def test_explicit_setting_still_wins_over_footprint(monkeypatch):
    _free(monkeypatch, 5080)
    settings = Settings(residency=Residency.EXCLUSIVE)
    mode, _ = resolve_residency(GPU, settings, needed_mib=100)
    assert mode is Residency.EXCLUSIVE


# --- swapping --------------------------------------------------------------


@pytest.fixture
def core():
    from voicegw.core import VoiceCore

    instance = VoiceCore(Settings(profile=Profile.GPU))
    instance.resolution = GPU
    instance.stt = FakeEngine("stt")
    instance.tts = FakeEngine("tts")
    instance._tts_alternates = []
    instance.residency = Residency.EXCLUSIVE
    instance._ready = True
    return instance


def test_only_one_engine_holds_the_gpu(core):
    core._claim_gpu(core.stt)
    assert core.stt.on_gpu and not core.tts.on_gpu

    core._claim_gpu(core.tts)
    assert core.tts.on_gpu and not core.stt.on_gpu


def test_repeat_claim_does_not_swap(core):
    core._claim_gpu(core.stt)
    before = core.tts.offload_calls
    core._claim_gpu(core.stt)
    assert core.tts.offload_calls == before


def test_shared_mode_never_swaps(core):
    core.residency = Residency.SHARED
    core._claim_gpu(core.stt)
    assert core.tts.offload_calls == 0
    assert core.stt.on_gpu and core.tts.on_gpu


def test_alternates_are_offloaded_too(core):
    alternate = FakeEngine("alt")
    core._tts_alternates = [alternate]
    core._claim_gpu(core.stt)
    assert not alternate.on_gpu


def test_transcribe_claims_the_gpu_first(core):
    core._claim_gpu(core.tts)  # TTS currently owns it
    # _run_stt must swap before invoking the model, or the assert inside
    # FakeEngine.transcribe fires.
    result = core._run_stt(np.zeros(16_000, dtype=np.float32), "ar")
    assert result.text == "ok"
    assert core.stt.on_gpu and not core.tts.on_gpu


class OomEngine(FakeEngine):
    """Mimics CTranslate2 failing to claim VRAM that torch still has reserved."""

    def __init__(self, engine_id: str):
        super().__init__(engine_id)
        self.info.allocator = "ct2"
        self.demoted_with = None

    def onload(self):
        raise RuntimeError("CUDA failed with error out of memory")

    def demote_to_cpu(self, reason: str) -> None:
        self.demoted_with = reason
        self.on_gpu = True  # now trivially true: it lives in host RAM
        self.info.device = "cpu"
        self.info.allocator = "none"


def test_onload_oom_demotes_to_cpu_instead_of_failing(core):
    core.stt = OomEngine("stt")
    core._claim_gpu(core.tts)
    core._claim_gpu(core.stt)
    assert core.stt.demoted_with is not None
    assert core.stt.info.device == "cpu"
    assert core._gpu_occupant == "stt"


def test_onload_errors_other_than_oom_propagate(core):
    class Boom(FakeEngine):
        def onload(self):
            raise RuntimeError("driver exploded")

    core.stt = Boom("stt")
    core._claim_gpu(core.tts)
    with pytest.raises(RuntimeError, match="driver exploded"):
        core._claim_gpu(core.stt)


def test_mixed_allocators_demote_the_ct2_engine(core):
    core.stt = OomEngine("stt")  # ct2
    core.tts = FakeEngine("tts")  # torch
    core._resolve_allocator_conflict()
    assert core.stt.demoted_with is not None
    assert "CTranslate2" in core.stt.demoted_with


def test_single_allocator_family_is_left_alone(core):
    core.stt = FakeEngine("stt")
    core.tts = FakeEngine("tts")
    core._resolve_allocator_conflict()
    assert core.stt.info.device == "cuda:0"
    assert core.tts.info.device == "cuda:0"


def test_cpu_only_alternates_do_not_count_as_a_conflict(core):
    alternate = FakeEngine("alt")
    alternate.info.allocator = "none"
    core._tts_alternates = [alternate]
    core._resolve_allocator_conflict()
    assert core.stt.info.device == "cuda:0"


def test_health_reports_residency(core):
    from voicegw.engines.voices import VoiceRegistry

    core.voices = VoiceRegistry("voices")
    core.residency_reason = "test reason"
    core._claim_gpu(core.tts)
    health = core.health()
    assert health["residency"] == "exclusive"
    assert health["gpu_occupant"] == "tts"
