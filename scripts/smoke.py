"""Smoke test against a running gateway: latency, silence guard, streaming.

python scripts/smoke.py [--url http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import io
import sys
import time

import httpx
import numpy as np
import soundfile as sf


def stt(client: httpx.Client, data: bytes, label: str) -> str:
    t0 = time.perf_counter()
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", data, "audio/wav")},
        data={"language": "ar"},
    )
    resp.raise_for_status()
    text = resp.json()["text"]
    print(f"{label:22} {round((time.perf_counter() - t0) * 1000):>6} ms  {text!r}")
    return text


def wav(samples: np.ndarray, sr: int = 16_000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--text", default="السلام عليكم، كيف حالك اليوم؟")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.url, timeout=300)
    failures: list[str] = []

    health = client.get("/healthz").json()
    print(
        f"profile={health['profile']} asr={health['asr_engine']['engine_id']} "
        f"tts={health['tts_engine']['engine_id']}\n"
    )

    # 1. Synthesize, then transcribe back.
    t0 = time.perf_counter()
    resp = client.post("/v1/audio/speech", json={"input": args.text, "stream": False})
    resp.raise_for_status()
    audio = resp.content
    print(f"{'TTS (non-stream)':22} {round((time.perf_counter() - t0) * 1000):>6} ms")

    stt(client, audio, "STT round-trip #1")
    text = stt(client, audio, "STT round-trip #2 (warm)")
    if not text.strip():
        failures.append("round-trip transcription was empty")

    # 2. The VAD must stop the ASR model hallucinating on non-speech.
    if stt(client, wav(np.zeros(16_000 * 5, dtype=np.float32)), "digital silence 5s").strip():
        failures.append("silence produced a transcript")

    rng = np.random.default_rng(0)
    room_tone = rng.normal(0, 0.002, 16_000 * 5).astype(np.float32)
    if stt(client, wav(room_tone), "room tone 5s").strip():
        failures.append("room tone produced a transcript")

    # 3. Streaming should deliver audio well before synthesis completes.
    long_text = "الجملة الأولى قصيرة. أما الجملة الثانية فهي أطول قليلا وتحتاج وقتا أكبر للتوليد."
    t0 = time.perf_counter()
    first_ms: float | None = None
    total = 0
    with client.stream("POST", "/v1/audio/speech", json={"input": long_text}) as resp:
        for chunk in resp.iter_bytes():
            if first_ms is None and len(chunk) > 44:
                first_ms = (time.perf_counter() - t0) * 1000
            total += len(chunk)
    full_ms = (time.perf_counter() - t0) * 1000
    print(
        f"\n{'TTS streaming':22} first byte {round(first_ms or 0):>5} ms, "
        f"complete {round(full_ms)} ms, {total} bytes"
    )
    if first_ms and first_ms >= full_ms * 0.95:
        failures.append("streaming did not deliver audio early")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
