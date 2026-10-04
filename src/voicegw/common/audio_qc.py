"""Reference audio quality control (QC).

Inspects voice cloning reference clips for common acoustic defects:
clipping, excessive noise / low volume, and insufficient active speech.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class AudioQualityReport:
    duration_s: float
    rms_dbfs: float
    peak_dbfs: float
    clipped_samples: int
    speech_duration_s: float
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.warnings) == 0


def analyze_audio_quality(
    audio: np.ndarray,
    sample_rate: int,
    min_seconds: float = 3.0,
    max_seconds: float = 15.0,
) -> AudioQualityReport:
    """Analyze audio samples for levels, clipping, and active speech."""
    warnings: list[str] = []
    total_samples = len(audio)
    duration_s = total_samples / sample_rate if sample_rate > 0 else 0.0

    if total_samples == 0:
        return AudioQualityReport(
            duration_s=0.0,
            rms_dbfs=-100.0,
            peak_dbfs=-100.0,
            clipped_samples=0,
            speech_duration_s=0.0,
            warnings=["audio is empty"],
        )

    # 1. Duration bounds
    if duration_s < min_seconds:
        warnings.append(f"duration is {duration_s:.1f}s, want >= {min_seconds:.1f}s")
    elif duration_s > max_seconds:
        warnings.append(f"duration is {duration_s:.1f}s, want <= {max_seconds:.1f}s")

    # 2. Clipping detection
    abs_audio = np.abs(audio)
    peak = float(np.max(abs_audio))
    clipped = int(np.sum(abs_audio >= 0.999))
    clip_ratio = clipped / total_samples

    peak_dbfs = 20.0 * math.log10(max(peak, 1e-5))
    if clip_ratio > 0.001 or (clipped > 10 and peak >= 0.999):
        warnings.append(
            f"detected {clipped} clipped samples ({clip_ratio * 100:.2f}%); "
            "reference may produce harsh artifacts"
        )

    # 3. Overall RMS loudness
    rms = float(np.sqrt(np.mean(audio**2)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-5))

    if rms_dbfs < -35.0:
        warnings.append(
            f"recording is very quiet ({rms_dbfs:.1f} dBFS < -35 dBFS); "
            "clone output may be noisy or whispery"
        )
    elif rms_dbfs > -6.0:
        warnings.append(
            f"recording is excessively loud ({rms_dbfs:.1f} dBFS > -6 dBFS); "
            "may have limited dynamic range"
        )

    # 4. Energy-based active speech estimation (30ms frames)
    frame_len = max(1, int(sample_rate * 0.030))
    num_frames = total_samples // frame_len
    active_frames = 0

    if num_frames > 0:
        reshaped = audio[: num_frames * frame_len].reshape(num_frames, frame_len)
        frame_rms = np.sqrt(np.mean(reshaped**2, axis=1))
        # Consider a frame active if its level is above -45 dBFS and above 15% of median speech
        active_thresh = max(10.0 ** (-45.0 / 20.0), rms * 0.15)
        active_frames = int(np.sum(frame_rms > active_thresh))

    speech_duration_s = (active_frames * frame_len) / sample_rate
    if speech_duration_s < 2.0 and duration_s >= min_seconds:
        warnings.append(
            f"only {speech_duration_s:.1f}s of active speech detected; "
            "reference audio contains too much silence or background noise"
        )

    return AudioQualityReport(
        duration_s=round(duration_s, 2),
        rms_dbfs=round(rms_dbfs, 1),
        peak_dbfs=round(peak_dbfs, 1),
        clipped_samples=clipped,
        speech_duration_s=round(speech_duration_s, 2),
        warnings=warnings,
    )
