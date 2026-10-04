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


def encode_flac(audio: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), sample_rate, format="FLAC")
    return buf.getvalue()


def encode_mp3(audio: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    try:
        sf.write(buf, np.clip(audio, -1.0, 1.0), sample_rate, format="MP3")
        return buf.getvalue()
    except Exception:
        # Fall back to high quality WAV if MP3 encoder is unavailable in libsndfile
        return encode_wav(audio, sample_rate)


def encode_opus(audio: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    try:
        sf.write(buf, np.clip(audio, -1.0, 1.0), sample_rate, format="OGG", subtype="OPUS")
        return buf.getvalue()
    except Exception:
        try:
            sf.write(buf, np.clip(audio, -1.0, 1.0), sample_rate, format="OGG", subtype="VORBIS")
            return buf.getvalue()
        except Exception:
            return encode_flac(audio, sample_rate)


def encode_aac(audio: np.ndarray, sample_rate: int) -> bytes:
    # Fall back to MP3 when AAC patent-encumbered encoder is not bundled in libsndfile
    return encode_mp3(audio, sample_rate)


def trim_edge_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    keep_ms: int = 35,
) -> np.ndarray:
    """Trim near-silent lead-in and decay tail from a rendered audio chunk.

    Preserves keep_ms of buffer around the voiced section to avoid hard cuts
    on initial or terminal consonants.
    """
    if audio.size == 0:
        return audio

    axis_to_check = 0 if audio.ndim == 1 else -1
    abs_audio = np.abs(audio) if audio.ndim == 1 else np.max(np.abs(audio), axis=0)
    threshold = 10.0 ** (threshold_db / 20.0)

    loud_indices = np.nonzero(abs_audio > threshold)[0]
    if loud_indices.size == 0:
        return audio

    keep_samples = int(sample_rate * keep_ms / 1000)
    start = max(0, int(loud_indices[0]) - keep_samples)
    end = min(audio.shape[axis_to_check], int(loud_indices[-1]) + 1 + keep_samples)

    if audio.ndim == 1:
        return audio[start:end]
    return audio[:, start:end]


def crossfade(
    chunk_a: np.ndarray,
    chunk_b: np.ndarray,
    crossfade_samples: int,
) -> np.ndarray:
    """Smoothly join two audio chunks using equal-power cosine crossfading."""
    if chunk_a.size == 0:
        return chunk_b
    if chunk_b.size == 0:
        return chunk_a

    fade_len = min(crossfade_samples, len(chunk_a), len(chunk_b))
    if fade_len <= 0:
        return np.concatenate([chunk_a, chunk_b])

    t = np.linspace(0.0, np.pi / 2.0, fade_len, endpoint=False, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)

    cross = chunk_a[-fade_len:] * fade_out + chunk_b[:fade_len] * fade_in
    return np.concatenate([chunk_a[:-fade_len], cross, chunk_b[fade_len:]])


def concat_with_crossfade(
    chunks: list[np.ndarray],
    crossfade_samples: int,
) -> np.ndarray:
    """Concatenate a sequence of audio chunks with smooth crossfading."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    result = chunks[0]
    for nxt in chunks[1:]:
        result = crossfade(result, nxt, crossfade_samples)
    return result


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
    "opus": "audio/ogg",
    "aac": "audio/aac",
}

