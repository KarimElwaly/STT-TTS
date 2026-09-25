"""MCP server exposing the gateway as agent tools.

This is a **thin HTTP client** over the running voice core -- it never loads
model weights itself, so your agent host doesn't duplicate ~2.6B parameters.
Start the gateway first (``voicegw serve``), then run this over stdio.

VS Code / Claude Desktop config:

    {
      "servers": {
        "voicegw": {
          "command": "voicegw",
          "args": ["mcp"],
          "env": { "VOICEGW_URL": "http://127.0.0.1:8000" }
        }
      }
    }
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

GATEWAY_URL = os.environ.get("VOICEGW_URL", "http://127.0.0.1:8000")
OUTPUT_DIR = Path(os.environ.get("VOICEGW_MCP_OUTPUT_DIR", tempfile.gettempdir())) / "voicegw"

mcp = FastMCP("voicegw")


def _client() -> httpx.Client:
    return httpx.Client(base_url=GATEWAY_URL, timeout=httpx.Timeout(300.0, connect=5.0))


def _unreachable(exc: Exception) -> str:
    return (
        f"Could not reach the voicegw gateway at {GATEWAY_URL} ({exc}). "
        "Start it with `voicegw serve`."
    )


def _status_error(exc: httpx.HTTPStatusError) -> str:
    """Turn a gateway error response into advice the agent can act on."""
    try:
        err = exc.response.json().get("error", {})
    except ValueError:
        err = {}
    if err.get("type") == "engine_unavailable":
        msg = f"Error: the {err.get('engine')} engine is unavailable: {err.get('message')}"
        return f"{msg}\nFix: {err['hint']}" if err.get("hint") else msg
    return f"Error: gateway returned {exc.response.status_code}: {exc.response.text[:400]}"


@mcp.tool()
def transcribe_audio(
    path: str | None = None,
    audio_base64: str | None = None,
    language: str = "ar",
) -> str:
    """Transcribe speech to text. Provide either a local file `path` or `audio_base64`.

    Optimized for Arabic (including dialects) and English.
    """
    if not path and not audio_base64:
        return "Error: provide either `path` or `audio_base64`."

    if path:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            return f"Error: file not found: {source}"
        data = source.read_bytes()
        filename = source.name
    else:
        try:
            data = base64.b64decode(audio_base64, validate=True)
        except Exception as exc:
            return f"Error: audio_base64 is not valid base64 ({exc})."
        filename = "audio.wav"

    try:
        with _client() as client:
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": (filename, data, "application/octet-stream")},
                data={"language": language},
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return _status_error(exc)
    except httpx.HTTPError as exc:
        return _unreachable(exc)

    text = resp.json().get("text", "").strip()
    return text or "(no speech detected)"


@mcp.tool()
def synthesize_speech(text: str, voice: str = "default", filename: str | None = None) -> str:
    """Synthesize `text` to speech and return the path of the generated WAV file."""
    if not text.strip():
        return "Error: text is empty."

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Never let the model choose an arbitrary write path.
    safe = Path(filename or "speech.wav").name
    if not safe.endswith(".wav"):
        safe += ".wav"
    out = OUTPUT_DIR / safe

    try:
        with _client() as client:
            resp = client.post(
                "/v1/audio/speech",
                json={"input": text, "voice": voice, "response_format": "wav", "stream": False},
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return _status_error(exc)
    except httpx.HTTPError as exc:
        return _unreachable(exc)

    out.write_bytes(resp.content)
    return str(out)


@mcp.tool()
def list_voices() -> str:
    """List the voices available for synthesis."""
    try:
        with _client() as client:
            resp = client.get("/v1/voices")
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        return _unreachable(exc)

    lines = [
        f"- {v['id']}: {v['label']} [{v['language']}]" + (" (cloned)" if v["is_clone"] else "")
        for v in resp.json()["data"]
    ]
    return "\n".join(lines) or "(no voices configured)"


@mcp.tool()
def gateway_status() -> str:
    """Report the active runtime profile, engines and device of the voice gateway."""
    try:
        with _client() as client:
            health = client.get("/healthz").json()
    except httpx.HTTPError as exc:
        return _unreachable(exc)

    status = health.get("status")
    if status == "loading":
        return "Gateway status: loading (models are still warming up)"
    if status != "ok" and not health.get("profile"):
        return f"Gateway status: {status}"

    lines = [
        f"profile={health['profile']} device={health['device']} "
        f"asr={health['asr_engine']['engine_id']} tts={health['tts_engine']['engine_id']} "
        f"realtime_capable={health['realtime_capable']} ({health['resolved_reason']})"
    ]
    # Surface the remediation hint so the agent can tell the user how to fix it
    # instead of just reporting that something is broken.
    for kind, err in (health.get("errors") or {}).items():
        lines.append(f"{kind.upper()} UNAVAILABLE ({err['engine']}): {err['reason']}")
        if err.get("hint"):
            lines.append(f"  fix: {err['hint']}")
    return "\n".join(lines)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
