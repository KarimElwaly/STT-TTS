import numpy as np
import pytest

from voicegw.common.audio import (
    decode_audio,
    encode_wav,
    float32_to_pcm16,
    pcm16_to_float32,
    to_mono,
)
from voicegw.common.protocols import SAMPLE_RATE_IN


def test_pcm16_roundtrip():
    original = np.array([0.0, 0.5, -0.5, 0.999], dtype=np.float32)
    restored = pcm16_to_float32(float32_to_pcm16(original))
    assert np.allclose(original, restored, atol=1e-3)


def test_pcm16_clips_out_of_range():
    restored = pcm16_to_float32(float32_to_pcm16(np.array([2.0, -2.0], dtype=np.float32)))
    assert np.all(np.abs(restored) <= 1.0)


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(100), np.zeros(100)], axis=1)
    assert np.allclose(to_mono(stereo), 0.5)


def test_wav_roundtrip_preserves_duration():
    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE_IN) / SAMPLE_RATE_IN)).astype(
        np.float32
    )
    decoded = decode_audio(encode_wav(tone, SAMPLE_RATE_IN))
    assert decoded.shape[0] == pytest.approx(SAMPLE_RATE_IN, rel=0.01)
    assert np.allclose(decoded, tone, atol=1e-3)


def test_decode_resamples_to_16k():
    tone = np.zeros(48_000, dtype=np.float32)
    decoded = decode_audio(encode_wav(tone, 48_000))
    assert decoded.shape[0] == pytest.approx(SAMPLE_RATE_IN, rel=0.02)
