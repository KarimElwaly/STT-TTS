"""FastAPI app: OpenAI-compatible REST + realtime WebSocket + static web client.

The response shapes here are identical in the ``cpu`` and ``gpu`` profiles --
the profile only decides which engines the core constructed.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..common.audio import CONTENT_TYPES, encode_wav, float32_to_pcm16, wav_header
from ..common.errors import EngineUnavailable
from ..core import VoiceCore, get_core
from ..realtime.session import RealtimeSession

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 64 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    core = get_core()
    await core.startup()
    yield
    await core.shutdown()


app = FastAPI(title="voicegw", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Localhost-only dev gateway; tighten before exposing beyond your machine.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def core_dep() -> VoiceCore:
    return get_core()


@app.exception_handler(EngineUnavailable)
async def engine_unavailable_handler(_request, exc: EngineUnavailable) -> JSONResponse:
    """A dead engine is a 503 the caller can act on, not an opaque 500."""
    return JSONResponse(
        {
            "error": {
                "message": str(exc),
                "type": "engine_unavailable",
                "engine": exc.engine_id,
                "hint": exc.hint,
            }
        },
        status_code=503,
    )


# ---------------------------------------------------------------------------
# Health / introspection
# ---------------------------------------------------------------------------
@app.get("/healthz")
async def healthz(core: VoiceCore = Depends(core_dep)) -> JSONResponse:
    health = core.health()
    return JSONResponse(health, status_code=200 if health["status"] == "ok" else 503)


@app.get("/v1/models")
async def list_models(core: VoiceCore = Depends(core_dep)) -> dict:
    return {
        "object": "list",
        "data": [
            {"id": core.stt.info.engine_id, "object": "model", "owned_by": "voicegw"},
            {"id": core.tts.info.engine_id, "object": "model", "owned_by": "voicegw"},
        ],
    }


@app.get("/v1/voices")
async def list_voices(core: VoiceCore = Depends(core_dep)) -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": v.id,
                "label": v.label,
                "language": v.language,
                "is_clone": v.is_clone,
                "tags": v.tags,
            }
            for v in core.voices.list()
        ],
    }


# ---------------------------------------------------------------------------
# STT -- OpenAI-compatible
# ---------------------------------------------------------------------------
@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    response_format: str = Form("json"),
    vad: bool = Form(True),
    core: VoiceCore = Depends(core_dep),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty audio file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "audio file too large")

    try:
        transcript = await core.transcribe_bytes(data, language, apply_vad=vad)
    except EngineUnavailable:
        raise
    except Exception as exc:
        log.exception("transcription failed")
        raise HTTPException(500, f"transcription failed: {exc}") from exc

    if response_format == "text":
        return StreamingResponse(iter([transcript.text]), media_type="text/plain")
    return {
        "text": transcript.text,
        "language": transcript.language,
        "duration": transcript.duration_s,
        "engine": transcript.engine,
    }


# ---------------------------------------------------------------------------
# TTS -- OpenAI-compatible
# ---------------------------------------------------------------------------
class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=8000)
    model: str | None = None
    voice: str | None = None
    response_format: str = "wav"
    stream: bool = True


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest, core: VoiceCore = Depends(core_dep)):
    if req.response_format not in {"wav", "pcm"}:
        raise HTTPException(400, "response_format must be 'wav' or 'pcm'")
    # Raise before the streaming response starts: once the 200 and the first
    # bytes are out, there is no way to report the failure to the client.
    if core.tts_error is not None:
        raise core.tts_error
    try:
        core.resolve_voice(req.voice)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    if not req.stream:
        try:
            chunks = [c async for c in core.synthesize(req.input, req.voice)]
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not chunks:
            raise HTTPException(500, "synthesis produced no audio")
        sr = chunks[0].sample_rate
        samples = np.concatenate([c.samples for c in chunks])
        if req.response_format == "wav":
            body = encode_wav(samples, sr)
        else:
            body = float32_to_pcm16(samples)
        return StreamingResponse(iter([body]), media_type=CONTENT_TYPES[req.response_format])

    async def stream() -> AsyncIterator[bytes]:
        header_sent = False
        try:
            async for chunk in core.synthesize(req.input, req.voice):
                if not header_sent and req.response_format == "wav":
                    yield wav_header(chunk.sample_rate)
                    header_sent = True
                yield float32_to_pcm16(chunk.samples)
        except ValueError as exc:
            log.error("synthesis rejected: %s", exc)
        except Exception:
            log.exception("synthesis failed mid-stream")

    return StreamingResponse(stream(), media_type=CONTENT_TYPES[req.response_format])


# ---------------------------------------------------------------------------
# Realtime
# ---------------------------------------------------------------------------
@app.websocket("/v1/realtime")
async def realtime(ws: WebSocket, core: VoiceCore = Depends(core_dep)) -> None:
    await RealtimeSession(ws, core).run()


_WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
if _WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
