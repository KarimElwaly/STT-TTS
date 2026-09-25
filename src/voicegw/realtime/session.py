"""Realtime WebSocket session: mic PCM in, events + audio deltas out.

Wire protocol
-------------
Client -> server:
  * binary frames: little-endian PCM16 mono @16 kHz
  * text frames (JSON): ``{"type": "commit"}`` (push-to-talk release),
    ``{"type": "cancel"}`` (barge-in / stop playback),
    ``{"type": "text", "text": "..."}`` (skip STT, synthesize directly),
    ``{"type": "config", "voice": "...", "language": "ar"}``

Server -> client:
  * text frames (JSON): ``session.created``, ``speech_started``,
    ``transcript.final``, ``response.text.delta``, ``response.audio.start``,
    ``response.done``, ``error``
  * binary frames: PCM16 audio at the rate given in ``response.audio.start``

Barge-in: any speech detected while audio is playing cancels the in-flight
response task and flushes queued chunks.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

from fastapi import WebSocket, WebSocketDisconnect

from ..common.audio import float32_to_pcm16, pcm16_to_float32
from ..common.text_chunker import StreamingChunker
from ..core import VoiceCore
from ..engines.vad import UtteranceDetector
from .agent import build_agent

log = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 12


class RealtimeSession:
    def __init__(self, ws: WebSocket, core: VoiceCore) -> None:
        self.ws = ws
        self.core = core
        self.detector = UtteranceDetector(core.vad_config)
        self.agent = build_agent(core.settings)
        self.history: list[dict] = []
        self.voice: str | None = None
        self.language: str | None = None
        self._response: asyncio.Task | None = None

    # -- helpers -----------------------------------------------------------
    async def send(self, **event) -> None:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await self.ws.send_json(event)

    async def _cancel_response(self, reason: str) -> None:
        if self._response and not self._response.done():
            self._response.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._response
            await self.send(type="response.cancelled", reason=reason)
        self._response = None

    @property
    def _speaking(self) -> bool:
        return self._response is not None and not self._response.done()

    # -- main loop ---------------------------------------------------------
    async def run(self) -> None:
        await self.ws.accept()

        if not self.core.realtime_capable:
            failure = self.core.stt_error or self.core.tts_error
            if failure is not None:
                await self.send(
                    type="error",
                    code="engine_unavailable",
                    message=str(failure),
                    engine=failure.engine_id,
                    hint=failure.hint,
                )
            else:
                await self.send(
                    type="error",
                    code="not_realtime_capable",
                    message=(
                        f"Profile '{self.core.profile.value}' loaded engine "
                        f"'{self.core.stt.info.engine_id}' "
                        f"(RTF ~{self.core.stt.info.rtf_estimate}), "
                        "which is too slow for the realtime loop. Use POST "
                        "/v1/audio/transcriptions for batch transcription, "
                        "pin VOICEGW_ASR_ENGINE=faster-whisper, or run the gpu profile."
                    ),
                )
            await self.ws.close(code=1011)
            return

        await self.send(
            type="session.created",
            profile=self.core.profile.value,
            device=self.core.resolution.device,
            asr_engine=self.core.stt.info.engine_id,
            tts_engine=self.core.tts.info.engine_id,
            realtime_capable=True,
            input_sample_rate=16_000,
            voices=[v.id for v in self.core.voices.list()],
        )

        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (data := message.get("bytes")) is not None:
                    await self._on_audio(data)
                elif (text := message.get("text")) is not None:
                    await self._on_control(text)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("realtime session error")
        finally:
            await self._cancel_response("session_closed")
            if hasattr(self.agent, "aclose"):
                await self.agent.aclose()

    async def _on_audio(self, data: bytes) -> None:
        samples = pcm16_to_float32(data)
        for event, payload in self.detector.push(samples):
            if event == "speech_started":
                await self.send(type="speech_started")
                if self._speaking:
                    # Barge-in: the user talked over the assistant.
                    await self._cancel_response("barge_in")
            elif event == "utterance" and payload is not None:
                await self._handle_utterance(payload)

    async def _on_control(self, raw: str) -> None:
        import json

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self.send(type="error", code="bad_json", message="control frame is not JSON")
            return

        kind = msg.get("type")
        if kind == "commit":
            utterance = self.detector.force_end()
            if utterance is None:
                await self.send(type="transcript.final", text="", reason="no_speech")
            else:
                await self._handle_utterance(utterance)
        elif kind == "cancel":
            await self._cancel_response("client_cancel")
            self.detector.reset()
        elif kind == "config":
            if "voice" in msg:
                try:
                    self.core.resolve_voice(msg["voice"])
                except KeyError as exc:
                    await self.send(type="error", code="unknown_voice", message=str(exc))
                    return
                self.voice = msg["voice"]
            if "language" in msg:
                self.language = msg["language"]
            await self.send(type="config.updated", voice=self.voice, language=self.language)
        elif kind == "text":
            await self._cancel_response("new_turn")
            self._response = asyncio.create_task(self._respond(msg.get("text", "")))
        else:
            await self.send(type="error", code="unknown_type", message=f"unknown type {kind!r}")

    async def _handle_utterance(self, audio) -> None:
        t0 = time.perf_counter()
        transcript = await self.core.transcribe(audio, self.language, apply_vad=False)
        await self.send(
            type="transcript.final",
            text=transcript.text,
            language=transcript.language,
            latency_ms=round((time.perf_counter() - t0) * 1000),
        )
        if not transcript.text.strip():
            return
        await self._cancel_response("new_turn")
        self._response = asyncio.create_task(self._respond(transcript.text))

    # -- response pipeline -------------------------------------------------
    async def _respond(self, user_text: str) -> None:
        """Agent deltas -> sentence chunks -> streamed TTS audio."""
        started = time.perf_counter()
        chunker = StreamingChunker()
        reply_parts: list[str] = []
        header_sent = False

        async def speak(piece: str) -> None:
            nonlocal header_sent
            async for chunk in self.core.synthesize(piece, self.voice):
                if not header_sent:
                    await self.send(
                        type="response.audio.start",
                        sample_rate=chunk.sample_rate,
                        first_audio_ms=round((time.perf_counter() - started) * 1000),
                    )
                    header_sent = True
                if chunk.samples.size:
                    await self.ws.send_bytes(float32_to_pcm16(chunk.samples))

        try:
            async for delta in self.agent(user_text, self.history):
                reply_parts.append(delta)
                await self.send(type="response.text.delta", delta=delta)
                for piece in chunker.push(delta):
                    await speak(piece)
            for piece in chunker.flush():
                await speak(piece)

            reply = "".join(reply_parts).strip()
            self.history.extend(
                [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
            )
            del self.history[: max(0, len(self.history) - MAX_HISTORY_TURNS * 2)]

            await self.send(
                type="response.done",
                text=reply,
                total_ms=round((time.perf_counter() - started) * 1000),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("response pipeline failed")
            await self.send(type="error", code="response_failed", message=str(exc))
