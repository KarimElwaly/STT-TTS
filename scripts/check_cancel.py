"""Does cancelling a synthesis (barge-in) actually stop the worker thread?

`core.synthesize` runs the engine in a thread and bridges chunks through a
bounded queue. Cancelling the *task* does not stop a thread, so this checks
what really happens to the producer when the consumer walks away -- which is
exactly what barge-in does, on every interrupted reply.
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from voicegw.common.config import Settings  # noqa: E402
from voicegw.common.protocols import SAMPLE_RATE_OUT, AudioChunk, EngineInfo  # noqa: E402
from voicegw.core import VoiceCore  # noqa: E402
from voicegw.engines.voices import VoiceRegistry  # noqa: E402

CHUNKS = 30
CHUNK_SECONDS = 0.05


class SlowTts:
    """Stands in for OmniVoice: each chunk takes real time to produce."""

    def __init__(self):
        self.info = EngineInfo("slow-tts", "fake", "cpu", "float32", 0.1, True)
        self.produced = 0
        self.finished = threading.Event()

    def load(self):
        pass

    def unload(self):
        pass

    def supports(self, voice):
        return True

    def synthesize(self, text, voice, urgent=False):
        try:
            for _ in range(CHUNKS):
                time.sleep(CHUNK_SECONDS)
                self.produced += 1
                yield AudioChunk(np.zeros(240, dtype=np.float32), SAMPLE_RATE_OUT)
        finally:
            self.finished.set()


async def main() -> int:
    core = VoiceCore(Settings())
    core.tts = SlowTts()
    core.voices = VoiceRegistry()

    async def consume_two():
        n = 0
        async for _ in core.synthesize("مرحبا"):
            n += 1
            if n == 2:
                break  # the consumer walks away, like barge-in

    t0 = time.perf_counter()
    await consume_two()
    walked_away_at = core.tts.produced
    print(f"consumer stopped after {walked_away_at} chunks ({(time.perf_counter() - t0):.2f}s)")

    # The generator is suspended at its `yield` until it is finalized, so the
    # lock is legitimately still held here. What matters is that it is released
    # promptly afterwards -- if it were not, every later turn would deadlock.
    print(f"tts lock held immediately after: {core._tts_lock.locked()}")
    await asyncio.sleep(0.5)
    still_locked = core._tts_lock.locked()
    print(f"tts lock held 0.5s later:         {still_locked}")

    await asyncio.sleep(CHUNKS * CHUNK_SECONDS)
    after_abandon = core.tts.produced
    print(f"producer generated {after_abandon}/{CHUNKS} chunks in total")
    print(f"producer generator finished: {core.tts.finished.is_set()}")

    # A second turn must be able to run at all -- this is what a leaked lock or
    # a stranded executor thread would break.
    second = 0
    async for _ in core.synthesize("مرحبا مرة أخرى"):
        second += 1
    print(f"a later turn produced {second} chunks")

    live = [t for t in threading.enumerate() if "asyncio_" in t.name or "ThreadPool" in t.name]
    print(f"worker threads still alive: {len(live)}")

    failures = []
    if after_abandon > walked_away_at + 2:
        failures.append("the engine kept generating long after the consumer left")
    if still_locked:
        failures.append("the TTS lock was never released")
    if not core.tts.finished.is_set():
        failures.append("the engine generator was never finalized")
    if second == 0:
        failures.append("a later turn could not run")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: producer stopped promptly, lock released, later turns work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
