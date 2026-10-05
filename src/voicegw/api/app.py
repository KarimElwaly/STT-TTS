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
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..common.audio import (
    CONTENT_TYPES,
    concat_with_crossfade,
    decode_audio,
    encode_aac,
    encode_flac,
    encode_mp3,
    encode_opus,
    encode_wav,
    float32_to_pcm16,
    wav_header,
)
from ..common.errors import EngineUnavailable
from ..common.protocols import MAX_SYNTHESIS_CHARS, Transcript
from ..core import VoiceCore, get_core
from ..engines.prosody import describe_voice
from ..realtime.session import RealtimeSession

log = logging.getLogger(__name__)


def _format_timestamp(seconds: float, decimal_sep: str) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d}{decimal_sep}{millis:03d}"


def format_srt(transcript: Transcript) -> str:
    lines = []
    for i, seg in enumerate(transcript.segments, 1):
        s_start = seg.get("start", 0.0)
        s_end = seg.get("end", 0.0)
        start_ts = _format_timestamp(s_start, ",")
        end_ts = _format_timestamp(s_end, ",")
        text = seg.get("text", "").strip()
        lines.append(f"{i}\n{start_ts} --> {end_ts}\n{text}\n")
    if not lines and transcript.text:
        end_ts = _format_timestamp(transcript.duration_s, ",")
        lines.append(f"1\n00:00:00,000 --> {end_ts}\n{transcript.text}\n")
    return "\n".join(lines).strip() + "\n"


def format_vtt(transcript: Transcript) -> str:
    lines = ["WEBVTT\n"]
    for i, seg in enumerate(transcript.segments, 1):
        s_start = seg.get("start", 0.0)
        s_end = seg.get("end", 0.0)
        start_ts = _format_timestamp(s_start, ".")
        end_ts = _format_timestamp(s_end, ".")
        text = seg.get("text", "").strip()
        lines.append(f"{start_ts} --> {end_ts}\n{text}\n")
    if len(lines) == 1 and transcript.text:
        end_ts = _format_timestamp(transcript.duration_s, ".")
        lines.append(f"00:00:00.000 --> {end_ts}\n{transcript.text}\n")
    return "\n".join(lines).strip() + "\n"


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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc: RequestValidationError) -> JSONResponse:
    """Translate FastAPI 422 validation errors into OpenAI HTTP 400 error envelopes."""
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = ".".join(str(l) for l in first.get("loc", []))
    msg = first.get("msg", "Invalid request parameter")
    return JSONResponse(
        {
            "error": {
                "message": f"{loc}: {msg}" if loc else msg,
                "type": "invalid_request_error",
                "param": loc or None,
                "code": first.get("type", None),
            }
        },
        status_code=400,
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


@app.post("/v1/voices")
async def create_voice(
    id: str = Form(...),
    label: str = Form(...),
    ref_text: str = Form(...),
    file: UploadFile = File(...),
    language: str = Form("ar"),
    core: VoiceCore = Depends(core_dep),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty audio file")
    try:
        audio = decode_audio(data, target_sr=16000)
    except Exception as exc:
        raise HTTPException(400, f"Failed to decode audio: {exc}") from exc
    wav_bytes = encode_wav(audio, 16000)
    try:
        voice = core.voices.add_voice(
            voice_id=id,
            label=label,
            audio_bytes=wav_bytes,
            ref_text=ref_text,
            language=language,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    qc = core.voices.check_quality(voice)
    return {
        "id": voice.id,
        "label": voice.label,
        "language": voice.language,
        "is_clone": voice.is_clone,
        "quality": {
            "is_valid": qc.is_valid if qc else True,
            "duration_s": qc.duration_s if qc else 0.0,
            "rms_dbfs": qc.rms_dbfs if qc else 0.0,
            "clipped_samples": qc.clipped_samples if qc else 0,
            "speech_duration_s": qc.speech_duration_s if qc else 0.0,
            "warnings": qc.warnings if qc else [],
        }
        if qc
        else None,
    }


@app.post("/v1/voices/describe")
async def describe_audio(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty audio file")
    try:
        audio = decode_audio(data, target_sr=16000)
    except Exception as exc:
        log.warning("Audio decoding failed in /v1/voices/describe: %s", exc)
        raise HTTPException(400, f"Failed to decode audio: {exc}") from exc
    description, metrics = describe_voice(audio, 16000)
    return {
        "description": description,
        "metrics": metrics,
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
    if response_format not in {"json", "text", "verbose_json", "srt", "vtt"}:
        raise HTTPException(
            400, "response_format must be 'json', 'text', 'verbose_json', 'srt', or 'vtt'"
        )
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
    if response_format == "srt":
        return StreamingResponse(iter([format_srt(transcript)]), media_type="text/plain")
    if response_format == "vtt":
        return StreamingResponse(iter([format_vtt(transcript)]), media_type="text/vtt")
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": transcript.language,
            "duration": transcript.duration_s,
            "text": transcript.text,
            "words": transcript.words,
            "segments": transcript.segments,
        }
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
    input: str = Field(..., min_length=1, max_length=MAX_SYNTHESIS_CHARS)
    model: str | None = None
    voice: str | None = None
    response_format: str = "wav"
    stream: bool = True


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest, core: VoiceCore = Depends(core_dep)):
    valid_formats = {"wav", "pcm", "mp3", "flac", "opus", "aac"}
    if req.response_format not in valid_formats:
        raise HTTPException(
            400, f"response_format must be one of: {', '.join(sorted(valid_formats))}"
        )
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
        fade_samples = int(sr * 0.035)
        samples = concat_with_crossfade([c.samples for c in chunks], fade_samples)
        if req.response_format == "wav":
            body = encode_wav(samples, sr)
        elif req.response_format == "flac":
            body = encode_flac(samples, sr)
        elif req.response_format == "mp3":
            body = encode_mp3(samples, sr)
        elif req.response_format == "opus":
            body = encode_opus(samples, sr)
        elif req.response_format == "aac":
            body = encode_aac(samples, sr)
        else:
            body = float32_to_pcm16(samples)
        return StreamingResponse(iter([body]), media_type=CONTENT_TYPES[req.response_format])


    async def stream() -> AsyncIterator[bytes]:
        header_sent = False
        try:
            # Streaming callers play as bytes arrive, so the first chunk gates
            # what they hear. (A non-streaming caller waits for the whole file
            # either way, so it is not marked urgent above: degrading it would
            # cost quality and save nothing.)
            async for chunk in core.synthesize(req.input, req.voice, urgent=True):
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
