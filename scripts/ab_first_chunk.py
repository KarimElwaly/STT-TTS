"""A/B the first-chunk quality tradeoff, through the real server code paths.

Time-to-first-audio was halved by rendering only the *opening* chunk of a
reply with fewer diffusion steps. That is a quality claim, and quality claims
need ears, not benchmarks. This writes two files of the same sentence:

    uniform.wav  -- every chunk at VOICEGW_TTS_NUM_STEP (non-streaming path)
    urgent.wav   -- first chunk at VOICEGW_TTS_FIRST_CHUNK_NUM_STEP, rest full
                    (streaming path: exactly what a caller hears)

Listen for two things: whether the opening words sound worse on their own, and
whether the seam where the step count changes is audible. If either bothers
you, set VOICEGW_TTS_FIRST_CHUNK_NUM_STEP equal to VOICEGW_TTS_NUM_STEP.

    python scripts/ab_first_chunk.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
OUT = Path("out/ab_first_chunk")
TEXT = (
    "أهلا بك، كيف أساعدك اليوم؟ يمكنني الاستماع إليك والرد عليك بصوت طبيعي، "
    "ويمكنني أيضا مساعدتك في الترجمة والتلخيص."
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url=BASE, timeout=120)

    try:
        health = client.get("/healthz").json()
    except httpx.ConnectError:
        print("start the gateway first:  python -m voicegw.cli serve")
        return 1

    tts = health["tts_engine"]
    print(f"engine={tts['engine_id']} device={tts['device']}")

    uniform = client.post("/v1/audio/speech", json={"input": TEXT, "stream": False}).content
    (OUT / "uniform.wav").write_bytes(uniform)

    with client.stream("POST", "/v1/audio/speech", json={"input": TEXT, "stream": True}) as r:
        streamed = b"".join(r.iter_bytes())
    (OUT / "urgent.wav").write_bytes(streamed)

    print(f"\nwrote {OUT / 'uniform.wav'}  ({len(uniform) / 1024:.0f} KiB)")
    print(f"wrote {OUT / 'urgent.wav'}   ({len(streamed) / 1024:.0f} KiB)")
    print("\nListen to both. If `urgent.wav` sounds materially worse, put this in .env:")
    print("    VOICEGW_TTS_FIRST_CHUNK_NUM_STEP=16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
