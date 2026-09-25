"""Measure a conversational turn: STT then TTS, repeatedly.

This is the pattern that matters for the <800 ms target, and the one that pays
a GPU swap on every call when residency is EXCLUSIVE.

    python scripts/turn_latency.py --turns 5
"""

from __future__ import annotations

import argparse
import io
import statistics
import time

import httpx
import soundfile as sf

TEXT = "السلام عليكم، كيف حالك اليوم؟"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--turns", type=int, default=5)
    args = parser.parse_args()

    client = httpx.Client(base_url=args.url, timeout=300)
    health = client.get("/healthz").json()
    print(
        f"profile={health['profile']} residency={health['residency']} "
        f"asr={health['asr_engine']['engine_id']} tts={health['tts_engine']['engine_id']}\n"
    )

    # One utterance to replay every turn.
    resp = client.post("/v1/audio/speech", json={"input": TEXT, "stream": False})
    utterance = resp.content
    samples, sr = sf.read(io.BytesIO(utterance))
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    utterance = buf.getvalue()
    audio_s = len(samples) / sr

    stt_ms: list[float] = []
    first_audio_ms: list[float] = []

    print(f"{'turn':>4}  {'STT':>8}  {'TTS 1st':>8}  {'total':>8}")
    for turn in range(1, args.turns + 1):
        t0 = time.perf_counter()
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", utterance, "audio/wav")},
            data={"language": "ar"},
        )
        r.raise_for_status()
        stt = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        first: float | None = None
        with client.stream("POST", "/v1/audio/speech", json={"input": TEXT}) as stream:
            for chunk in stream.iter_bytes():
                if first is None and len(chunk) > 44:
                    first = (time.perf_counter() - t1) * 1000
                    break
        stt_ms.append(stt)
        first_audio_ms.append(first or 0)
        print(f"{turn:>4}  {stt:>7.0f}ms  {first or 0:>7.0f}ms  {stt + (first or 0):>7.0f}ms")

    # Ignore the first turn: it still carries one-time init.
    warm_stt = stt_ms[1:] or stt_ms
    warm_tts = first_audio_ms[1:] or first_audio_ms
    total = statistics.median(warm_stt) + statistics.median(warm_tts)
    print(
        f"\nwarm median  STT {statistics.median(warm_stt):.0f} ms "
        f"(RTF {statistics.median(warm_stt) / 1000 / audio_s:.2f})  "
        f"TTS first audio {statistics.median(warm_tts):.0f} ms"
    )
    print(f"end-to-end to first audio: {total:.0f} ms  (target < 800 ms)")


if __name__ == "__main__":
    main()
