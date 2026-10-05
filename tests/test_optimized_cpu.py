"""Tests for Optimized CPU mode: Cohere ASR INT8 and OmniVoice 4-step diffusion."""

import pytest

from voicegw.common.config import Profile, ProfileResolution, Quantization, Settings
from voicegw.engines import cohere_asr, omnivoice_gguf, omnivoice_tts, registry


def test_cohere_asr_cpu_int8_is_realtime_capable():
    """INT8 quantized Cohere on CPU should be marked realtime_capable."""
    engine = cohere_asr.build_cpu(quantization=Quantization.INT8)
    assert engine.quantization is Quantization.INT8
    assert engine.info.realtime_capable is True
    assert engine.info.device == "cpu"
    assert engine.info.rtf_estimate < 1.0


def test_cohere_asr_cpu_float32_remains_offline_only():
    """Default float32 Cohere on CPU without quantization remains non-realtime."""
    engine = cohere_asr.build_cpu(quantization=Quantization.NONE, dtype="float32")
    assert engine.quantization is Quantization.NONE
    assert engine.info.dtype == "float32"
    assert engine.info.realtime_capable is False


def test_cohere_asr_cpu_bfloat16_is_realtime_capable():
    """Bfloat16 Cohere on CPU cuts memory in half and is realtime capable."""
    engine = cohere_asr.build_cpu(quantization=Quantization.NONE, dtype="bfloat16")
    assert engine.info.dtype == "bfloat16"
    assert engine.info.realtime_capable is True



def test_registry_has_cohere_cpu_int8():
    res = ProfileResolution(Profile.CPU, "CPU Mode", "cpu")
    engine = registry.STT_ENGINES["cohere-asr-cpu-int8"](res)
    assert engine.quantization is Quantization.INT8
    assert engine.info.realtime_capable is True


def test_omnivoice_cpu_defaults_to_4_steps():
    """OmniVoice on CPU should default to 4 steps (and 2 first-chunk steps) to cut latency by 75%."""
    engine = omnivoice_tts.build_cpu()
    assert engine.num_step == 4
    assert engine.first_num_step == 2
    assert engine.info.rtf_estimate <= 0.40


def test_omnivoice_gguf_cpu_defaults_to_4_steps():
    """OmniVoice GGUF on CPU should default to 4 steps."""
    engine = omnivoice_gguf.build_cpu()
    assert engine.num_step == 4
    assert engine.first_num_step == 2
    assert engine.info.realtime_capable is True


def test_cpu_mode_profile_configurations():
    """Verify switching between GPU and CPU modes."""
    # GPU profile defaults
    assert registry.PROFILE_DEFAULTS[Profile.GPU]["stt"] == "cohere-asr"
    assert registry.PROFILE_DEFAULTS[Profile.GPU]["tts"] == "omnivoice"

    # CPU profile defaults
    assert registry.PROFILE_DEFAULTS[Profile.CPU]["stt"] == "faster-whisper"
    assert registry.PROFILE_DEFAULTS[Profile.CPU]["tts"] == "omnivoice-cpu"

    # Explicit pinning of optimized CPU engines
    res = ProfileResolution(Profile.CPU, "CPU Mode", "cpu")
    pinned_stt = registry._select("stt", registry.STT_ENGINES, "cohere-asr-cpu-int8", res)
    assert pinned_stt == "cohere-asr-cpu-int8"

    pinned_tts = registry._select("tts", registry.TTS_ENGINES, "omnivoice-gguf", res)
    assert pinned_tts == "omnivoice-gguf"


def test_cpu_threads_setting():
    settings = Settings(cpu_threads=8)
    assert settings.cpu_threads == 8

