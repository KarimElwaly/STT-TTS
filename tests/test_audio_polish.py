import numpy as np
import pytest

from voicegw.common.audio import concat_with_crossfade, crossfade, trim_edge_silence


def test_trim_edge_silence_empty():
    empty = np.zeros(0, dtype=np.float32)
    assert len(trim_edge_silence(empty, 16000)) == 0


def test_trim_edge_silence_all_silence():
    silence = np.zeros(16000, dtype=np.float32)
    # When no sample exceeds threshold, returns unchanged
    trimmed = trim_edge_silence(silence, 16000)
    assert len(trimmed) == 16000


def test_trim_edge_silence_padded_tone():
    sr = 16000
    # 0.5s silence + 0.5s tone + 0.5s silence
    pad_len = 8000
    tone_len = 8000
    tone = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, tone_len, dtype=np.float32))
    audio = np.concatenate([np.zeros(pad_len, dtype=np.float32), tone, np.zeros(pad_len, dtype=np.float32)])

    trimmed = trim_edge_silence(audio, sr, threshold_db=-40.0, keep_ms=35)
    keep_samples = int(sr * 35 / 1000)

    # Trimmed length should be approximately tone_len + 2 * keep_samples
    assert len(trimmed) < len(audio)
    assert abs(len(trimmed) - (tone_len + 2 * keep_samples)) < 100


def test_crossfade_empty_chunks():
    a = np.ones(100, dtype=np.float32)
    empty = np.zeros(0, dtype=np.float32)
    assert np.array_equal(crossfade(a, empty, 20), a)
    assert np.array_equal(crossfade(empty, a, 20), a)


def test_crossfade_zero_samples():
    a = np.ones(10, dtype=np.float32)
    b = np.full(10, 2.0, dtype=np.float32)
    out = crossfade(a, b, 0)
    assert len(out) == 20
    assert np.array_equal(out[:10], a)
    assert np.array_equal(out[10:], b)


def test_crossfade_equal_power():
    sr = 16000
    # Two constant value chunks
    chunk_a = np.ones(1000, dtype=np.float32)
    chunk_b = np.ones(1000, dtype=np.float32)
    fade_len = 200
    out = crossfade(chunk_a, chunk_b, fade_len)

    # In equal-power cosine crossfade of identical signals, cos^2 + sin^2 = 1
    # For constant 1.0 signals, cos(t)*1 + sin(t)*1 has a slight gentle crest (~1.41 at pi/4),
    # but never dips to zero or clips negatively.
    assert len(out) == 2000 - fade_len
    fade_region = out[1000 - fade_len : 1000]
    assert np.all(fade_region >= 1.0)
    assert np.all(fade_region <= 1.45)


def test_concat_with_crossfade():
    chunks = [np.ones(1000, dtype=np.float32) for _ in range(3)]
    out = concat_with_crossfade(chunks, 100)
    # Total length: 3000 - 2 * 100 = 2800
    assert len(out) == 2800
