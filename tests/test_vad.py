import numpy as np
import pytest

from voicegw.common.protocols import SAMPLE_RATE_IN
from voicegw.engines.vad import (
    EnergyVad,
    UtteranceDetector,
    VadConfig,
    chunk_long_audio,
    gate_audio,
    segment_audio,
)

CONFIG = VadConfig(threshold=0.5, min_speech_ms=200, min_silence_ms=300, noise_gate_rms=0.005)


def tone(seconds: float, amplitude: float = 0.4, freq: int = 220) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE_IN)) / SAMPLE_RATE_IN
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, amplitude: float = 0.0) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE_IN)
    if amplitude == 0.0:
        return np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    return (rng.normal(0, amplitude, n)).astype(np.float32)


def test_silence_produces_no_segments():
    """The ASR model hallucinates on silence, so the gate must reject it."""
    assert segment_audio(silence(5.0), EnergyVad(CONFIG), CONFIG) == []


def test_room_tone_is_rejected_by_noise_gate():
    quiet = silence(5.0, amplitude=0.001)
    assert gate_audio(quiet, CONFIG, EnergyVad(CONFIG)).size == 0


def test_speech_is_detected():
    audio = np.concatenate([silence(0.5), tone(1.0), silence(0.5)])
    segments = segment_audio(audio, EnergyVad(CONFIG), CONFIG)
    assert len(segments) == 1
    assert segments[0].duration_s == pytest.approx(1.0, abs=0.4)


def test_two_utterances_are_split():
    audio = np.concatenate([tone(0.8), silence(1.0), tone(0.8)])
    assert len(segment_audio(audio, EnergyVad(CONFIG), CONFIG)) == 2


def test_gate_drops_silence_but_keeps_speech():
    audio = np.concatenate([silence(2.0), tone(1.0), silence(2.0)])
    gated = gate_audio(audio, CONFIG, EnergyVad(CONFIG))
    assert 0 < gated.size < audio.size


def test_short_blip_below_min_speech_is_ignored():
    audio = np.concatenate([silence(0.5), tone(0.05), silence(1.0)])
    assert segment_audio(audio, EnergyVad(CONFIG), CONFIG) == []


def test_long_audio_is_chunked_under_the_limit():
    audio = np.concatenate([np.concatenate([tone(2.0), silence(0.6)]) for _ in range(20)])
    chunks = list(chunk_long_audio(audio, max_chunk_s=10.0, config=CONFIG, vad=EnergyVad(CONFIG)))
    assert len(chunks) > 1
    assert all(len(c) / SAMPLE_RATE_IN <= 12.0 for c in chunks)


def test_short_audio_is_not_chunked():
    audio = tone(3.0)
    chunks = list(chunk_long_audio(audio, max_chunk_s=28.0, config=CONFIG, vad=EnergyVad(CONFIG)))
    assert len(chunks) == 1


class TestUtteranceDetector:
    def _detector(self) -> UtteranceDetector:
        return UtteranceDetector(CONFIG, EnergyVad(CONFIG))

    def test_emits_speech_started_then_utterance(self):
        detector = self._detector()
        events = detector.push(np.concatenate([silence(0.3), tone(1.0), silence(1.0)]))
        kinds = [e for e, _ in events]
        assert kinds == ["speech_started", "utterance"]
        assert events[1][1] is not None and events[1][1].size > 0

    def test_no_events_for_silence(self):
        assert self._detector().push(silence(3.0)) == []

    def test_works_across_frame_boundaries(self):
        detector = self._detector()
        audio = np.concatenate([tone(1.0), silence(1.0)])
        kinds: list[str] = []
        # 100-sample writes don't align with the 512-sample VAD frame.
        for i in range(0, len(audio), 100):
            kinds.extend(e for e, _ in detector.push(audio[i : i + 100]))
        assert "speech_started" in kinds
        assert "utterance" in kinds

    def test_force_end_returns_buffered_speech(self):
        detector = self._detector()
        detector.push(tone(1.0))
        assert detector.in_speech
        utterance = detector.force_end()
        assert utterance is not None and utterance.size > 0
        assert not detector.in_speech

    def test_force_end_on_silence_returns_none(self):
        detector = self._detector()
        detector.push(silence(1.0))
        assert detector.force_end() is None
