"""Drive the realtime socket the way the browser's Speak box does.

Checks the agent-bypass mode end to end against a *running* gateway with real
models: type text, skip STT and the LLM, get audio back.

    python scripts/check_speak.py "نص للاختبار"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from voicegw.common.audio import encode_wav, pcm16_to_float32  # noqa: E402

URL = "ws://127.0.0.1:8000/v1/realtime"


def main() -> int:
    import json

    text = sys.argv[1] if len(sys.argv) > 1 else "مرحبا، هذا اختبار النطق المباشر"
    samples: list[np.ndarray] = []
    sample_rate = 24_000
    first_audio_ms = None
    reply = None

    with connect(URL, max_size=None) as ws:
        created = json.loads(ws.recv())
        print(f"session: {created['asr_engine']} -> {created['tts_engine']}")
        print(f"agent available: {created['agent_available']}")

        ws.send(json.dumps({"type": "config", "agent": False}))
        updated = json.loads(ws.recv())
        print(f"agent enabled: {updated['agent']}")

        t0 = time.perf_counter()
        ws.send(json.dumps({"type": "text", "text": text}))

        while True:
            message = ws.recv()
            if isinstance(message, bytes):
                samples.append(pcm16_to_float32(message))
                continue
            event = json.loads(message)
            kind = event.get("type")
            if kind == "response.audio.start":
                sample_rate = event["sample_rate"]
                first_audio_ms = event["first_audio_ms"]
            elif kind == "response.done":
                reply = event["text"]
                total_ms = event["total_ms"]
                break
            elif kind == "error":
                print(f"ERROR {event['code']}: {event['message']}")
                return 1

    elapsed = round((time.perf_counter() - t0) * 1000)
    audio = np.concatenate(samples) if samples else np.zeros(0, dtype=np.float32)
    out = Path("out/speak_test.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(encode_wav(audio, sample_rate))

    print(f"\nsent:  {text!r}")
    print(f"spoke: {reply!r}")
    print(f"first audio: {first_audio_ms} ms · total: {total_ms} ms · wall: {elapsed} ms")
    print(f"audio: {len(audio) / sample_rate:.2f}s @ {sample_rate} Hz -> {out}")

    failures = []
    if reply != text:
        failures.append(f"bypassed agent changed the text: {reply!r}")
    if audio.size == 0:
        failures.append("no audio was produced")
    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: typed text was spoken back verbatim, no LLM involved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
