import pytest

from voicegw.common.text_chunker import (
    chunk_text,
    extract_speed_tags,
    parse_pause_ms,
    split_sentences,
)


def test_parse_pause_ms():
    assert parse_pause_ms("[pause 500ms]") == 500
    assert parse_pause_ms("[PAUSE 300MS]") == 300
    assert parse_pause_ms("[silence 1s]") == 1000
    assert parse_pause_ms("[pause 2s]") == 2000
    assert parse_pause_ms("not a pause") is None
    assert parse_pause_ms("[pause]") is None


def test_extract_speed_tags():
    text, speed = extract_speed_tags("هذا نص [slow]بطيء[/slow] جدا")
    assert speed == 0.85
    assert text == "هذا نص بطيء جدا"

    text2, speed2 = extract_speed_tags("[fast]سريع للغاية[/fast]")
    assert speed2 == 1.15
    assert text2 == "سريع للغاية"

    text3, speed3 = extract_speed_tags("نص عادي")
    assert speed3 == 1.0
    assert text3 == "نص عادي"


def test_split_sentences_with_pause():
    text = "مرحبا بك. [pause 500ms] كيف حالك اليوم؟"
    pieces = split_sentences(text)
    assert len(pieces) == 3
    assert pieces[0] == "مرحبا بك."
    assert pieces[1] == "[pause 500ms]"
    assert pieces[2] == "كيف حالك اليوم؟"


def test_chunk_text_preserves_pause():
    text = "الجملة الأولى. [pause 1s] الجملة الثانية."
    chunks = chunk_text(text)
    assert "[pause 1s]" in chunks
    assert chunks[1] == "[pause 1s]"
