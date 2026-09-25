"""End-to-end check of the real ASR engine through the core.

The model card warns that the model is "eager to transcribe, even non-speech
sounds". This verifies the mandatory VAD gate actually stops that, and reports
warm latency for the configured profile.

    python scripts/verify_asr.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicegw.common.audio import decode_audio  # noqa: E402
from voicegw.core import VoiceCore  # noqa: E402

SAMPLE = Path("out/offline.wav")


async def main() -> int:
    if not SAMPLE.is_file():
        print(f'missing {SAMPLE}; run: voicegw say "مرحبا" --out {SAMPLE}')
        return 1

    core = VoiceCore()
    await core.startup()
    print(
        f"\nprofile={core.profile.value} asr={core.stt.info.engine_id} "
        f"residency={core.residency.value}"
    )
    if core.stt_error:
        print(f"STT unavailable: {core.stt_error}")
        return 1

    audio = decode_audio(SAMPLE.read_bytes())
    result = await core.transcribe(audio, "ar")
    print(f"speech     -> {result.text!r}")

    silence = np.zeros(16_000 * 3, dtype=np.float32)
    print(f"silence    -> {(await core.transcribe(silence, 'ar')).text!r}")

    rng = np.random.default_rng(0)
    room = (rng.standard_normal(16_000 * 3) * 0.002).astype(np.float32)
    print(f"room tone  -> {(await core.transcribe(room, 'ar')).text!r}")

    # Ungated: what the model would emit without the VAD in front of it.
    raw = await core.transcribe(silence, "ar", apply_vad=False)
    print(f"silence, VAD OFF -> {raw.text!r}   <- why the gate is mandatory")

    timings = []
    for _ in range(3):
        start = time.perf_counter()
        await core.transcribe(audio, "ar")
        timings.append((time.perf_counter() - start) * 1000)
    warm = sorted(timings)[len(timings) // 2]
    print(
        f"\nwarm STT {warm:.0f} ms for {result.duration_s:.1f}s  "
        f"RTF {warm / 1000 / result.duration_s:.2f}"
    )

    await core.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
