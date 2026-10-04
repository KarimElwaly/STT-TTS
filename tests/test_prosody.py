"""Test pure-NumPy prosody analysis and OmniVoice prompt tag generation."""

import numpy as np
import pytest

from voicegw.engines.prosody import describe_voice, estimate_f0, estimate_speech_rate
from voicegw.engines.voices import INSTRUCT_VOCAB


def test_estimate_f0_sine_wave():
    # 200 Hz tone at 16000 Hz
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 200.0 * t)

    f0, voicing = estimate_f0(tone, sr)
    assert f0 is not None
    assert abs(f0 - 200.0) < 5.0
    assert voicing > 0.8


def test_estimate_f0_silence():
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    f0, voicing = estimate_f0(silence, sr)
    assert f0 is None
    assert voicing == 0.0


def test_describe_voice_valid_vocab():
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False, dtype=np.float32)
    # High pitch female tone ~220 Hz
    tone = 0.5 * np.sin(2 * np.pi * 220.0 * t)

    description, metrics = describe_voice(tone, sr)
    tags = [t.strip() for t in description.split(",")]
    for tag in tags:
        assert tag in INSTRUCT_VOCAB

    assert "female" in tags
    assert "high pitch" in tags or "moderate pitch" in tags
    assert metrics["f0_hz"] is not None
    assert metrics["rms_dbfs"] > -20.0
