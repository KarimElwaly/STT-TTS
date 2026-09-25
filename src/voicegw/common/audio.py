"""Audio I/O helpers: everything entering an engine is float32 mono at 16 kHz."""

from __future__ import annotations

import io

import numpy as np

from .protocols import SAMPLE_RATE_IN


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    # soundfile gives (frames, channels); librosa gives (channels, frames).
    axis = 1 if audio.shape[0] > audio.shape[1] else 0
    return audio.mean(axis=axis)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE_IN) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    import librosa

    return librosa.resample(
        audio.astype(np.float32, copy=False), orig_sr=orig_sr, target_sr=target_sr
    )


def decode_audio(data: bytes, target_sr: int = SAMPLE_RATE_IN) -> np.ndarray:
    """Decode an audio file (wav/flac/ogg/mp3/...) to float32 mono at ``target_sr``."""
    import soundfile as sf

    try:
        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        # soundfile can't read mp3/m4a on every platform; fall back to librosa/audioread.
        import librosa

        samples, sr = librosa.load(io.BytesIO(data), sr=target_sr, mono=True)
    return resample(to_mono(np.asarray(samples)), sr, target_sr)


def load_audio_file(path: str, target_sr: int = SAMPLE_RATE_IN) -> np.ndarray:
    with open(path, "rb") as fh:
        return decode_audio(fh.read(), target_sr)


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Decode little-endian PCM16 (the realtime WebSocket wire format)."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def wav_header(sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    """A streaming WAV header with an unknown/maximal data length.

    Lets us start sending audio bytes before we know the total duration.
    """
    import struct

    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size = 0xFFFFFFFF - 44
    return (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data"
        + struct.pack("<I", data_size)
    )


CONTENT_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/L16",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
}
