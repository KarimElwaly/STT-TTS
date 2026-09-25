"""The playback-gating chunk may trade fidelity for latency; nothing else may.

TTS latency is ~linear in diffusion steps and nearly flat in text length below
~65 characters (scripts/tts_latency_profile.py), so the only chunk worth
degrading is the one the listener waits on in silence. Every later chunk is
rendered while earlier audio is still playing, so degrading it would cost
quality for no perceived speed.
"""

import numpy as np
import pytest

from voicegw.common.config import Settings
from voicegw.common.protocols import SAMPLE_RATE_OUT, AudioChunk, EngineInfo, Voice
from voicegw.engines import omnivoice_tts


class RecordingOmniVoice:
    """Stands in for the real model and records the config it was handed."""

    def __init__(self):
        self.steps: list[int | None] = []

    def generate(self, **kwargs):
        config = kwargs.get("generation_config")
        self.steps.append(getattr(config, "num_step", None))
        return [np.zeros(240, dtype=np.float32)]


@pytest.fixture
def engine(monkeypatch):
    tts = omnivoice_tts.OmniVoiceTts(
        engine_id="omnivoice",
        device="cuda:0",
        dtype="float16",
        rtf_estimate=0.025,
        realtime_capable=True,
        num_step=16,
        first_num_step=8,
    )
    tts._model = RecordingOmniVoice()
    monkeypatch.setattr(tts, "load", lambda: None)
    return tts


VOICE = Voice(id="default", label="Default", language="ar")


def _drain(engine, text, **kw):
    return list(engine.synthesize(text, VOICE, **kw))


def test_normal_synthesis_uses_the_full_step_count(engine):
    _drain(engine, "مرحبا بك.")
    assert engine._model.steps == [16]


def test_urgent_synthesis_uses_the_reduced_step_count(engine):
    _drain(engine, "مرحبا بك.", urgent=True)
    assert engine._model.steps == [8]


def test_only_the_first_piece_of_an_urgent_call_is_degraded(engine):
    """A long urgent text still splits; the tail must keep full quality."""
    _drain(engine, "الجملة الأولى. الجملة الثانية. الجملة الثالثة.", urgent=True)
    assert len(engine._model.steps) > 1
    assert engine._model.steps[0] == 8
    assert set(engine._model.steps[1:]) == {16}


def test_asymmetry_is_disabled_when_the_counts_match(monkeypatch):
    tts = omnivoice_tts.OmniVoiceTts(
        engine_id="omnivoice",
        device="cuda:0",
        dtype="float16",
        rtf_estimate=0.025,
        realtime_capable=True,
        num_step=16,
        first_num_step=16,
    )
    tts._model = RecordingOmniVoice()
    monkeypatch.setattr(tts, "load", lambda: None)
    list(tts.synthesize("مرحبا.", VOICE, urgent=True))
    assert tts._model.steps == [16]


def test_settings_drive_both_step_counts(monkeypatch):
    """Both knobs must come from Settings so `.env` is honoured."""
    from voicegw.common import config

    monkeypatch.setenv("VOICEGW_TTS_NUM_STEP", "20")
    monkeypatch.setenv("VOICEGW_TTS_FIRST_CHUNK_NUM_STEP", "5")
    config.get_settings.cache_clear()
    try:
        assert omnivoice_tts.default_num_step() == 20
        assert omnivoice_tts.default_first_num_step() == 5
    finally:
        config.get_settings.cache_clear()


# --- the hint must survive the trip through core ---------------------------


class UrgencySpyTts:
    def __init__(self):
        self.info = EngineInfo("spy-tts", "fake", "cpu", "float32", 0.1, True)
        self.seen: list[bool] = []

    def load(self):
        pass

    def unload(self):
        pass

    def supports(self, voice):
        return True

    def synthesize(self, text, voice, urgent=False):
        self.seen.append(urgent)
        yield AudioChunk(np.zeros(240, dtype=np.float32), SAMPLE_RATE_OUT, is_final=True)


@pytest.mark.asyncio
async def test_core_forwards_the_urgent_hint():
    from voicegw.core import VoiceCore
    from voicegw.engines.voices import VoiceRegistry

    core = VoiceCore(Settings())
    core.tts = UrgencySpyTts()
    core.voices = VoiceRegistry()

    async for _ in core.synthesize("مرحبا", urgent=True):
        pass
    async for _ in core.synthesize("مرحبا"):
        pass
    assert core.tts.seen == [True, False]
