import json

import numpy as np
import pytest
import soundfile as sf

from voicegw.common.protocols import SAMPLE_RATE_IN
from voicegw.engines.voices import VoiceRegistry


@pytest.fixture
def voices_dir(tmp_path):
    clip = tmp_path / "ref.wav"
    sf.write(clip, np.zeros(SAMPLE_RATE_IN * 5, dtype=np.float32), SAMPLE_RATE_IN)
    return tmp_path


def write_manifest(directory, voices):
    (directory / "manifest.json").write_text(json.dumps({"voices": voices}), encoding="utf-8")


def test_builtin_voice_always_available(tmp_path):
    registry = VoiceRegistry(tmp_path)
    assert registry.get(None).id == "default"
    assert registry.get("default").description


def test_unknown_voice_lists_alternatives(tmp_path):
    with pytest.raises(KeyError, match="Available"):
        VoiceRegistry(tmp_path).get("nope")


def test_design_voice_is_loaded(voices_dir):
    write_manifest(voices_dir, [{"id": "d", "description": "female, young adult"}])
    voice = VoiceRegistry(voices_dir).get("d")
    assert not voice.is_clone
    assert voice.description == "female, young adult"


def test_clone_voice_resolves_ref_audio(voices_dir):
    write_manifest(voices_dir, [{"id": "c", "ref_audio": "ref.wav", "ref_text": "مرحبا"}])
    voice = VoiceRegistry(voices_dir).get("c")
    assert voice.is_clone
    assert voice.ref_audio.endswith("ref.wav")


def test_clone_without_ref_text_is_skipped(voices_dir):
    write_manifest(voices_dir, [{"id": "c", "ref_audio": "ref.wav"}])
    with pytest.raises(KeyError):
        VoiceRegistry(voices_dir).get("c")


def test_voice_without_ref_or_description_is_skipped(voices_dir):
    write_manifest(voices_dir, [{"id": "bad"}])
    with pytest.raises(KeyError):
        VoiceRegistry(voices_dir).get("bad")


def test_path_traversal_is_rejected(voices_dir):
    """`ref_audio` must not become an arbitrary file-read primitive."""
    write_manifest(
        voices_dir, [{"id": "evil", "ref_audio": "../../../etc/passwd", "ref_text": "x"}]
    )
    with pytest.raises(KeyError):
        VoiceRegistry(voices_dir).get("evil")


def test_missing_ref_file_is_skipped(voices_dir):
    write_manifest(voices_dir, [{"id": "c", "ref_audio": "absent.wav", "ref_text": "x"}])
    with pytest.raises(KeyError):
        VoiceRegistry(voices_dir).get("c")


def test_corrupt_manifest_degrades_to_builtin(voices_dir):
    (voices_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    assert VoiceRegistry(voices_dir).get(None).id == "default"


def test_short_reference_is_flagged(voices_dir):
    sf.write(voices_dir / "short.wav", np.zeros(SAMPLE_RATE_IN, dtype=np.float32), SAMPLE_RATE_IN)
    write_manifest(voices_dir, [{"id": "s", "ref_audio": "short.wav", "ref_text": "x"}])
    registry = VoiceRegistry(voices_dir)
    problems = registry.validate_reference(registry.get("s"))
    assert problems and "want >=" in problems[0]


def test_good_reference_has_no_problems(voices_dir):
    write_manifest(voices_dir, [{"id": "c", "ref_audio": "ref.wav", "ref_text": "x"}])
    registry = VoiceRegistry(voices_dir)
    assert registry.validate_reference(registry.get("c")) == []


def test_shipped_manifest_is_valid():
    """The repo's own voices/manifest.json must parse."""
    registry = VoiceRegistry("voices")
    ids = {v.id for v in registry.list()}
    assert "default" in ids
