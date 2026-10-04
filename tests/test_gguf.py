"""Test OmniVoice GGUF engine wrapper and synthesis."""

import numpy as np
import pytest

from voicegw.common.protocols import Voice
from voicegw.engines.omnivoice_gguf import OmniVoiceGgufEngine, build_cpu, build_gpu


def test_gguf_engine_info():
    engine_gpu = build_gpu()
    assert engine_gpu.info.engine_id == "omnivoice-gguf"
    assert engine_gpu.info.vram_mib == 945
    assert engine_gpu.info.allocator == "cuda"

    engine_cpu = build_cpu()
    assert engine_cpu.info.vram_mib == 0
    assert engine_cpu.info.allocator == "none"
    assert engine_cpu.info.realtime_capable is True


def test_gguf_supports_voice():
    engine = build_cpu()
    clone_voice = Voice(id="custom", label="Custom", ref_audio="custom.wav", ref_text="نص")
    preset_voice = Voice(id="design", label="Design", description="female, low pitch")
    assert engine.supports(clone_voice) is True
    assert engine.supports(preset_voice) is True


def test_gguf_synthesize_ssml_pause():
    engine = build_cpu()
    voice = Voice(id="design", label="Design", description="female, moderate pitch")
    chunks = list(engine.synthesize("مرحبا [pause 200ms] كيف حالك؟", voice))
    assert len(chunks) >= 2
    # Verify pause chunk has zero samples and ~200ms duration at 24000 Hz
    pause_chunk = [c for c in chunks if np.all(c.samples == 0.0)]
    assert len(pause_chunk) >= 1
    assert abs(pause_chunk[0].duration_s - 0.2) < 0.01
