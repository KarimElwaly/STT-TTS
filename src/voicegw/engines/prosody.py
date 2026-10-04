"""Pure-NumPy acoustic prosody mirror.

Extracts acoustic metrics (pitch F0, loudness, speech rate, voicing)
from speech audio and maps them to valid OmniVoice descriptor tags.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .voices import INSTRUCT_VOCAB


def estimate_f0(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
) -> tuple[Optional[float], float]:
    """Estimate median F0 (pitch in Hz) and voicing ratio using autocorrelation."""
    if len(audio) < int(sample_rate * (frame_ms / 1000.0)):
        return None, 0.0

    frame_len = int(sample_rate * (frame_ms / 1000.0))
    hop_len = int(sample_rate * (hop_ms / 1000.0))
    min_lag = int(sample_rate / f0_max)
    max_lag = int(sample_rate / f0_min)

    num_frames = (len(audio) - frame_len) // hop_len
    if num_frames <= 0:
        return None, 0.0

    f0_estimates: list[float] = []
    voiced_count = 0

    for i in range(num_frames):
        start = i * hop_len
        frame = audio[start : start + frame_len]
        # Remove DC offset
        frame = frame - np.mean(frame)
        energy = np.sum(frame**2)
        if energy < 1e-4:
            continue

        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode="full")
        center = len(frame) - 1
        lags = autocorr[center + min_lag : center + max_lag + 1]
        if len(lags) == 0:
            continue

        best_lag_idx = int(np.argmax(lags))
        peak_val = lags[best_lag_idx]

        # Normalized autocorrelation threshold
        norm_peak = peak_val / energy
        if norm_peak > 0.35:
            voiced_count += 1
            lag = min_lag + best_lag_idx
            f0_estimates.append(sample_rate / lag)

    voicing_ratio = voiced_count / num_frames if num_frames > 0 else 0.0
    median_f0 = float(np.median(f0_estimates)) if f0_estimates else None
    return median_f0, round(voicing_ratio, 2)


def estimate_speech_rate(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate syllable rate (syllables per second) from amplitude envelope peaks."""
    if len(audio) < sample_rate:
        return 0.0

    # Rectified smoothed envelope (15ms window)
    window_len = int(sample_rate * 0.015)
    kernel = np.ones(window_len, dtype=np.float32) / window_len
    envelope = np.convolve(np.abs(audio), kernel, mode="same")

    # Downsample envelope to ~100 Hz
    step = sample_rate // 100
    sub_env = envelope[::step]
    if len(sub_env) < 2:
        return 0.0

    thresh = np.mean(sub_env) * 0.8
    # Count local maxima above threshold with minimum spacing (150ms ~ 6.6 Hz max)
    peaks = 0
    last_peak = -15
    for i in range(1, len(sub_env) - 1):
        if sub_env[i] > thresh and sub_env[i] > sub_env[i - 1] and sub_env[i] >= sub_env[i + 1]:
            if i - last_peak >= 12:  # at least 120ms gap
                peaks += 1
                last_peak = i

    duration_s = len(audio) / sample_rate
    return round(peaks / duration_s, 2) if duration_s > 0 else 0.0


def describe_voice(audio: np.ndarray, sample_rate: int) -> tuple[str, dict]:
    """Map acoustic features from audio to a valid OmniVoice descriptor string."""
    median_f0, voicing_ratio = estimate_f0(audio, sample_rate)
    speech_rate = estimate_speech_rate(audio, sample_rate)
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-5))

    tags: list[str] = []

    # 1. Gender heuristic based on fundamental pitch
    if median_f0 is not None:
        if median_f0 >= 165.0:
            tags.append("female")
        else:
            tags.append("male")
    else:
        tags.append("male")

    # 2. Age band
    tags.append("young adult")

    # 3. Pitch descriptor
    if median_f0 is not None:
        if median_f0 < 95.0:
            tags.append("very low pitch")
        elif median_f0 < 130.0:
            tags.append("low pitch")
        elif median_f0 < 200.0:
            tags.append("moderate pitch")
        elif median_f0 < 280.0:
            tags.append("high pitch")
        else:
            tags.append("very high pitch")
    else:
        tags.append("moderate pitch")

    # 4. Whisper / dynamics
    if voicing_ratio < 0.30 and rms_dbfs < -28.0:
        tags.append("whisper")

    # Filter against known INSTRUCT_VOCAB to guarantee 100% validity
    valid_tags = [t for t in tags if t in INSTRUCT_VOCAB]
    description = ", ".join(valid_tags)

    metrics = {
        "f0_hz": round(median_f0, 1) if median_f0 is not None else None,
        "voicing_ratio": voicing_ratio,
        "syllables_per_s": speech_rate,
        "rms_dbfs": round(rms_dbfs, 1),
    }
    return description, metrics
