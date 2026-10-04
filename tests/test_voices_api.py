"""Test voices API endpoints: creation, quality control report, and prosody describe."""

import io
import numpy as np
import pytest
from starlette.testclient import TestClient

from voicegw.common.audio import encode_wav
from voicegw.common.config import Profile, ProfileResolution, Settings
from voicegw.common.protocols import EngineInfo, SAMPLE_RATE_OUT, AudioChunk, Voice
from voicegw.engines.voices import VoiceRegistry


class FakeTts:
    def __init__(self):
        self.info = EngineInfo("fake-tts", "fake", "cpu", "float32", 0.1, True)

    def load(self):
        pass

    def unload(self):
        pass

    def supports(self, voice: Voice) -> bool:
        return True

    def synthesize(self, text, voice, urgent=False):
        yield AudioChunk(np.full(2400, 0.1, dtype=np.float32), SAMPLE_RATE_OUT, is_final=True)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from voicegw import core as core_module
    from voicegw.api.app import app

    instance = core_module.VoiceCore(Settings(profile=Profile.CPU, voices_dir=str(tmp_path)))
    instance.resolution = ProfileResolution(Profile.CPU, "test", "cpu")
    instance.tts = FakeTts()
    instance._tts_alternates = []
    instance.voices = VoiceRegistry(tmp_path)
    instance._ready = True

    monkeypatch.setattr(core_module, "_core", instance)
    with TestClient(app) as test_client:
        yield test_client


def test_describe_endpoint(client):
    # Create 1 second of 200 Hz tone
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 200.0 * t)
    wav_bytes = encode_wav(tone, sr)

    res = client.post(
        "/v1/voices/describe",
        files={"file": ("tone.wav", wav_bytes, "audio/wav")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "description" in data
    assert "metrics" in data
    assert data["metrics"]["f0_hz"] is not None


def test_create_voice_endpoint(client):
    sr = 16000
    # 4 seconds of clean sine speech simulation
    t = np.linspace(0, 4.0, sr * 4, endpoint=False, dtype=np.float32)
    tone = 0.4 * np.sin(2 * np.pi * 180.0 * t)
    wav_bytes = encode_wav(tone, sr)

    res = client.post(
        "/v1/voices",
        data={
            "id": "test_narrator",
            "label": "Test Narrator",
            "ref_text": "هذا تسجيل تجريبي للنظام الصوتي",
            "language": "ar",
        },
        files={"file": ("ref.wav", wav_bytes, "audio/wav")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == "test_narrator"
    assert data["is_clone"] is True
    assert data["quality"] is not None
    assert data["quality"]["is_valid"] is True

    # Check that voice is now in /v1/voices list
    list_res = client.get("/v1/voices")
    assert list_res.status_code == 200
    voice_ids = [v["id"] for v in list_res.json()["data"]]
    assert "test_narrator" in voice_ids
