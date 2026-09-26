"""Abandoning a synthesis must not strand the engine.

Barge-in is a feature, and HTTP clients disconnect, so a consumer walking away
mid-reply is normal traffic rather than an error path. The generation runs in a
worker thread, and `task.cancel()` does not stop a thread -- so without an
explicit stop signal the producer blocks forever on a full buffer, holding an
executor thread and the engine with it.
"""

import asyncio
import threading

import numpy as np
import pytest

from voicegw.common.config import Settings
from voicegw.common.protocols import SAMPLE_RATE_OUT, AudioChunk, EngineInfo
from voicegw.core import VoiceCore
from voicegw.engines.voices import VoiceRegistry

#: Comfortably more than any sane look-ahead buffer, so a producer that is not
#: told to stop would keep going (or wedge) rather than finish on its own.
CHUNKS = 40


class SlowTts:
    def __init__(self, chunks: int = CHUNKS):
        self.info = EngineInfo("slow-tts", "fake", "cpu", "float32", 0.1, True)
        self.chunks = chunks
        self.produced = 0
        self.finalized = threading.Event()

    def load(self):
        pass

    def unload(self):
        pass

    def supports(self, voice):
        return True

    def synthesize(self, text, voice, urgent=False):
        try:
            for _ in range(self.chunks):
                self.produced += 1
                yield AudioChunk(np.zeros(240, dtype=np.float32), SAMPLE_RATE_OUT)
        finally:
            self.finalized.set()


class ExplodingTts(SlowTts):
    def synthesize(self, text, voice, urgent=False):
        yield AudioChunk(np.zeros(240, dtype=np.float32), SAMPLE_RATE_OUT)
        raise RuntimeError("engine blew up mid-stream")


@pytest.fixture
def core():
    instance = VoiceCore(Settings())
    instance.tts = SlowTts()
    instance.voices = VoiceRegistry()
    return instance


async def _take(core, n: int) -> int:
    seen = 0
    async for _ in core.synthesize("مرحبا"):
        seen += 1
        if seen == n:
            break
    return seen


@pytest.mark.asyncio
async def test_abandoned_producer_stops_instead_of_running_on(core):
    await _take(core, 2)
    await asyncio.sleep(0.3)
    # The old bridge ran ahead until the bounded queue filled, then blocked in
    # `put` forever. A few chunks of look-ahead are fine; half the reply is not.
    assert core.tts.produced < CHUNKS // 2


@pytest.mark.asyncio
async def test_abandoned_engine_generator_is_finalized(core):
    """The engine's own `finally` must run so it can release its resources."""
    await _take(core, 2)
    await asyncio.sleep(0.3)
    assert core.tts.finalized.is_set()


@pytest.mark.asyncio
async def test_lock_is_released_after_abandonment(core):
    await _take(core, 2)
    await asyncio.sleep(0.3)
    assert not core._tts_lock.locked()


@pytest.mark.asyncio
async def test_a_later_turn_still_works_after_abandonment(core):
    """The failure mode this guards against is a permanently wedged gateway."""
    await _take(core, 2)
    await asyncio.sleep(0.3)

    core.tts = SlowTts(chunks=3)
    got = [c async for c in core.synthesize("مرحبا مرة أخرى")]
    assert len(got) == 3


@pytest.mark.asyncio
async def test_repeated_abandonment_does_not_exhaust_the_executor(core):
    """Every barge-in used to leak one worker thread permanently."""
    before = threading.active_count()
    for _ in range(12):
        await _take(core, 1)
    await asyncio.sleep(0.5)
    assert threading.active_count() <= before + 4


@pytest.mark.asyncio
async def test_full_consumption_still_yields_everything(core):
    core.tts = SlowTts(chunks=5)
    got = [c async for c in core.synthesize("مرحبا")]
    assert len(got) == 5
    assert core.tts.finalized.is_set()
    assert not core._tts_lock.locked()


@pytest.mark.asyncio
async def test_engine_errors_still_reach_the_caller():
    """Backpressure must not swallow the exception path."""
    instance = VoiceCore(Settings())
    instance.tts = ExplodingTts()
    instance.voices = VoiceRegistry()

    with pytest.raises(RuntimeError, match="blew up"):
        async for _ in instance.synthesize("مرحبا"):
            pass
    assert not instance._tts_lock.locked()
