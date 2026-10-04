import numpy as np
import pytest

from voicegw.common.audio_qc import analyze_audio_quality


def test_qc_empty_audio():
    report = analyze_audio_quality(np.zeros(0, dtype=np.float32), 16000)
    assert not report.is_valid
    assert "audio is empty" in report.warnings


def test_qc_short_audio():
    sr = 16000
    audio = np.random.uniform(-0.1, 0.1, sr * 2).astype(np.float32)
    report = analyze_audio_quality(audio, sr, min_seconds=3.0, max_seconds=15.0)
    assert any("want >=" in w for w in report.warnings)


def test_qc_clipped_audio():
    sr = 16000
    audio = np.random.uniform(-0.2, 0.2, sr * 4).astype(np.float32)
    # Clip 1000 samples
    audio[:1000] = 1.0
    report = analyze_audio_quality(audio, sr)
    assert report.clipped_samples >= 1000
    assert any("clipped samples" in w for w in report.warnings)


def test_qc_very_quiet_audio():
    sr = 16000
    # Very low amplitude
    audio = np.random.uniform(-0.001, 0.001, sr * 4).astype(np.float32)
    report = analyze_audio_quality(audio, sr)
    assert report.rms_dbfs < -35.0
    assert any("quiet" in w for w in report.warnings)


def test_qc_clean_speech_signal():
    sr = 16000
    # Modulated sinusoidal signal simulating normal speech level (-18 dBFS)
    t = np.linspace(0, 4.0, sr * 4, endpoint=False, dtype=np.float32)
    carrier = np.sin(2 * np.pi * 200 * t)
    modulator = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    audio = (0.25 * carrier * modulator).astype(np.float32)

    report = analyze_audio_quality(audio, sr, min_seconds=3.0, max_seconds=15.0)
    assert report.is_valid
    assert len(report.warnings) == 0
    assert report.duration_s == 4.0
    assert report.clipped_samples == 0
