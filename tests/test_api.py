"""API tests with fake engines.

These prove the REST/WebSocket contract is identical regardless of which
engines a profile picked -- no model weights are loaded.
"""

import numpy as np
import pytest

from voicegw.common.config import Profile, ProfileResolution, Settings
from voicegw.common.protocols import SAMPLE_RATE_OUT, AudioChunk, EngineInfo, Transcript, Voice


class FakeStt:
    def __init__(self, engine_id="fake-stt", realtime=True, text="مرحبا بك"):
        self.text = text
        self.calls: list[np.ndarray] = []
        self.info = EngineInfo(engine_id, "fake", "cpu", "float32", 0.1, realtime)

    def load(self):
        pass

    def unload(self):
        pass

    def transcribe(self, audio, language="ar"):
        self.calls.append(audio)
        return Transcript(self.text, language, len(audio) / 16_000, self.info.engine_id)


class FakeTts:
    def __init__(self, engine_id="fake-tts", supports_clone=True, chunks=3):
        self.supports_clone = supports_clone
        self.chunks = chunks
        self.urgent_calls: list[bool] = []
        self.info = EngineInfo(engine_id, "fake", "cpu", "float32", 0.1, True)

    def load(self):
        pass

    def unload(self):
        pass

    def supports(self, voice: Voice) -> bool:
        return self.supports_clone or not voice.is_clone

    def synthesize(self, text, voice, urgent=False):
        self.urgent_calls.append(urgent)
        for i in range(self.chunks):
            yield AudioChunk(
                np.full(2400, 0.1, dtype=np.float32),
                SAMPLE_RATE_OUT,
                is_final=(i == self.chunks - 1),
            )


@pytest.fixture
def core(monkeypatch):
    """A VoiceCore wired to fakes, bypassing profile probing and model loads."""
    from voicegw import core as core_module

    instance = core_module.VoiceCore(Settings(profile=Profile.CPU))
    instance.resolution = ProfileResolution(Profile.CPU, "test", "cpu")
    instance.stt = FakeStt()
    instance.tts = FakeTts()
    instance._tts_alternates = []
    instance.vad = None
    instance._ready = True

    from voicegw.engines.voices import VoiceRegistry

    instance.voices = VoiceRegistry("voices")
    # No real VAD in these tests: pass audio straight through.
    monkeypatch.setattr(instance, "_gate", lambda audio: audio)
    monkeypatch.setattr(core_module, "_core", instance)
    return instance


@pytest.fixture
def client(core, monkeypatch):
    from fastapi.testclient import TestClient

    from voicegw.api import app as app_module

    # Skip the lifespan so startup() never runs.
    monkeypatch.setattr(app_module.app.router, "lifespan_context", _noop_lifespan)
    with TestClient(app_module.app) as test_client:
        yield test_client


import contextlib  # noqa: E402


@contextlib.asynccontextmanager
async def _noop_lifespan(app):
    yield


def wav_bytes(seconds=1.0, sr=16_000):
    from voicegw.common.audio import encode_wav

    return encode_wav(np.full(int(seconds * sr), 0.1, dtype=np.float32), sr)


# --- health / introspection ------------------------------------------------


def test_healthz_reports_profile_and_engines(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["profile"] == "cpu"
    assert body["device"] == "cpu"
    assert body["realtime_capable"] is True
    assert body["asr_engine"]["engine_id"] == "fake-stt"


def test_healthz_returns_503_while_loading(client, core):
    core._ready = False
    assert client.get("/healthz").status_code == 503


# --- degraded engines ------------------------------------------------------


def _fail(core, kind="stt"):
    from voicegw.common.errors import EngineUnavailable

    err = EngineUnavailable(
        "cohere-asr",
        "You are trying to access a gated repo. 401 Client Error.",
        "Accept the model terms on its Hugging Face page, then set HF_TOKEN in .env.",
    )
    setattr(core, f"{kind}_error", err)
    return err


def test_healthz_reports_degraded_when_an_engine_failed(client, core):
    _fail(core, "stt")
    resp = client.get("/healthz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["errors"]["stt"]["engine"] == "cohere-asr"
    assert "HF_TOKEN" in body["errors"]["stt"]["hint"]
    assert body["realtime_capable"] is False


def test_transcription_returns_503_with_a_hint_not_500(client, core):
    _fail(core, "stt")
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 503
    err = resp.json()["error"]
    assert err["type"] == "engine_unavailable"
    assert "HF_TOKEN" in err["hint"]


def test_speech_fails_before_streaming_starts(client, core):
    """A dead TTS must not yield an empty 200 body."""
    _fail(core, "tts")
    resp = client.post("/v1/audio/speech", json={"input": "مرحبا"})
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "engine_unavailable"


def test_tts_failure_does_not_break_transcription(client, core):
    _fail(core, "tts")
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 200


def test_realtime_reports_engine_failure_not_slow_engine(client, core):
    _fail(core, "stt")
    with client.websocket_connect("/v1/realtime") as ws:
        event = ws.receive_json()
    assert event["type"] == "error"
    assert event["code"] == "engine_unavailable"
    assert "HF_TOKEN" in event["hint"]


def test_healthy_core_reports_no_errors(client):
    assert client.get("/healthz").json()["errors"] == {}


def test_voices_endpoint_lists_builtin(client):
    ids = {v["id"] for v in client.get("/v1/voices").json()["data"]}
    assert "default" in ids


# --- transcription ---------------------------------------------------------


def test_transcription_returns_openai_shape(client):
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav_bytes(), "audio/wav")},
        data={"language": "ar"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "مرحبا بك"
    assert body["language"] == "ar"


def test_transcription_text_format(client):
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav_bytes(), "audio/wav")},
        data={"response_format": "text"},
    )
    assert resp.text == "مرحبا بك"


def test_empty_upload_is_rejected(client):
    resp = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"", "audio/wav")})
    assert resp.status_code == 400


def test_vad_gate_suppresses_silence(client, core, monkeypatch):
    """No speech in => empty transcript, and the model is never called."""
    monkeypatch.setattr(core, "_gate", lambda audio: np.zeros(0, dtype=np.float32))
    resp = client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(), "audio/wav")}
    )
    assert resp.json()["text"] == ""
    assert core.stt.calls == []


# --- speech ----------------------------------------------------------------


def test_speech_streams_wav(client):
    resp = client.post("/v1/audio/speech", json={"input": "مرحبا", "voice": "default"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    assert resp.content.startswith(b"RIFF")
    assert len(resp.content) > 44


def test_speech_non_streaming_returns_complete_wav(client):
    resp = client.post(
        "/v1/audio/speech", json={"input": "مرحبا", "voice": "default", "stream": False}
    )
    assert resp.content.startswith(b"RIFF")

    import io

    import soundfile as sf

    samples, sr = sf.read(io.BytesIO(resp.content))
    assert sr == SAMPLE_RATE_OUT
    assert len(samples) == 3 * 2400


def test_unknown_voice_returns_404(client):
    resp = client.post("/v1/audio/speech", json={"input": "x", "voice": "nope"})
    assert resp.status_code == 404


def test_bad_response_format_rejected(client):
    resp = client.post("/v1/audio/speech", json={"input": "x", "response_format": "ogg"})
    assert resp.status_code == 400


def test_empty_input_rejected(client):
    assert client.post("/v1/audio/speech", json={"input": ""}).status_code == 422


# --- realtime --------------------------------------------------------------


def test_realtime_announces_session(client):
    with client.websocket_connect("/v1/realtime") as ws:
        event = ws.receive_json()
        assert event["type"] == "session.created"
        assert event["profile"] == "cpu"
        assert event["input_sample_rate"] == 16_000
        assert event["realtime_capable"] is True


def test_realtime_refuses_non_realtime_engine(client, core):
    """A batch-only engine must fail loudly, not hang."""
    core.stt = FakeStt(realtime=False)
    with client.websocket_connect("/v1/realtime") as ws:
        event = ws.receive_json()
        assert event["type"] == "error"
        assert event["code"] == "not_realtime_capable"


def test_realtime_rejects_unknown_voice(client):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "config", "voice": "nope"})
        event = ws.receive_json()
        assert event["type"] == "error"
        assert event["code"] == "unknown_voice"


def test_realtime_rejects_malformed_control_frame(client):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_text("not json")
        assert ws.receive_json()["code"] == "bad_json"


def test_realtime_bounds_text_length_like_the_rest_facade(client):
    """Otherwise the socket is the cheap way to ask for unbounded synthesis."""
    from voicegw.common.protocols import MAX_SYNTHESIS_CHARS

    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "text", "text": "ا" * (MAX_SYNTHESIS_CHARS + 1)})
        event = ws.receive_json()
        assert event["type"] == "error"
        assert event["code"] == "text_too_long"


def test_realtime_accepts_text_at_the_limit(client):
    from voicegw.common.protocols import MAX_SYNTHESIS_CHARS

    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "text", "text": "ا" * MAX_SYNTHESIS_CHARS})
        # Anything but the rejection means it was accepted for synthesis.
        assert ws.receive_json()["type"] != "error"


# --- TTS routing -----------------------------------------------------------


def test_clone_voice_routes_to_supporting_engine(core):
    """Piper can't clone, so a clone request must be routed to OmniVoice."""
    core.tts = FakeTts("piper-like", supports_clone=False)
    core._tts_alternates = [FakeTts("omnivoice-like", supports_clone=True)]
    clone = Voice(id="c", label="c", ref_audio="x.wav", ref_text="t")
    assert core._engine_for(clone).info.engine_id == "omnivoice-like"


def test_unsupported_voice_without_alternate_raises(core):
    core.tts = FakeTts("piper-like", supports_clone=False)
    core._tts_alternates = []
    clone = Voice(id="c", label="c", ref_audio="x.wav", ref_text="t")
    with pytest.raises(ValueError, match="No loaded TTS engine"):
        core._engine_for(clone)
