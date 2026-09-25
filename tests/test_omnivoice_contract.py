"""OmniVoice input contracts.

Two parameters have strict, non-obvious requirements that fail silently or
late if they are wrong, so they are validated up front:

* `language` uses ISO 639-3 style ids ('arb', not 'ar').
* `instruct` (voice design) must come from a fixed vocabulary, not prose.
"""

import pytest

from voicegw.engines.omnivoice_tts import LANGUAGE_MAP, resolve_language
from voicegw.engines.voices import INSTRUCT_VOCAB, VoiceRegistry, validate_instruct


def test_arabic_maps_to_iso_639_3():
    assert resolve_language("ar") == "arb"


def test_english_maps_to_en():
    assert resolve_language("en") == "en"


def test_dialects_are_mapped():
    assert resolve_language("ar-EG") == "arz"
    assert resolve_language("ar_lv") == "apc"


def test_none_is_language_agnostic():
    assert resolve_language(None) is None
    assert resolve_language("") is None


def test_unknown_language_degrades_to_none():
    assert resolve_language("klingon") is None


@pytest.mark.parametrize("code", sorted(LANGUAGE_MAP))
def test_every_mapped_code_resolves(code):
    assert resolve_language(code) is not None


def test_valid_instruct_accepted():
    assert validate_instruct("male, young adult, moderate pitch") == []


def test_prose_instruct_rejected():
    """Free-form descriptions are silently ignored by the model, so reject them."""
    bad = validate_instruct("A clear, warm adult male voice with a neutral accent.")
    assert bad


def test_instruct_is_case_insensitive():
    assert validate_instruct("Male, British Accent") == []


def test_unknown_item_is_reported():
    assert validate_instruct("male, wizard") == ["wizard"]


def test_builtin_voice_uses_valid_instruct():
    registry = VoiceRegistry("voices")
    assert validate_instruct(registry.get("default").description) == []


def test_shipped_manifest_voices_all_valid():
    registry = VoiceRegistry("voices")
    for voice in registry.list():
        if voice.description:
            assert validate_instruct(voice.description) == [], voice.id


def test_manifest_with_prose_description_is_skipped(tmp_path):
    import json

    (tmp_path / "manifest.json").write_text(
        json.dumps({"voices": [{"id": "p", "description": "a lovely warm voice"}]}),
        encoding="utf-8",
    )
    with pytest.raises(KeyError):
        VoiceRegistry(tmp_path).get("p")


def test_vocabulary_is_not_empty():
    assert "male" in INSTRUCT_VOCAB
    assert "whisper" in INSTRUCT_VOCAB
