from voicegw.common.text_chunker import StreamingChunker, chunk_text, normalize, split_sentences


def test_splits_on_arabic_question_mark():
    assert split_sentences("كيف حالك؟ أنا بخير.") == ["كيف حالك؟", "أنا بخير."]


def test_keeps_decimals_intact():
    assert split_sentences("السعر 3.5 ريال.") == ["السعر 3.5 ريال."]


def test_normalize_strips_tatweel_and_bidi_marks():
    assert normalize("\u200fمرحبــا   بك") == "مرحبا بك"


def test_long_sentence_is_broken_at_clause_marks():
    text = "و".join(["كلمة طويلة جدا"] * 40)
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)


def test_streaming_emits_before_stream_ends():
    chunker = StreamingChunker()
    assert chunker.push("مرحبا بك في ") == []
    emitted = chunker.push("هذا الاختبار. ")
    assert emitted == ["مرحبا بك في هذا الاختبار."]


def test_streaming_flush_drains_remainder():
    chunker = StreamingChunker()
    chunker.push("جملة بلا نقطة")
    assert chunker.flush() == ["جملة بلا نقطة"]
    assert chunker.flush() == []


def test_streaming_preserves_full_text():
    deltas = ["السلام ", "عليكم. ", "كيف ", "حالك اليوم؟ ", "أتمنى أن تكون بخير."]
    chunker = StreamingChunker()
    out = []
    for d in deltas:
        out.extend(chunker.push(d))
    out.extend(chunker.flush())
    assert "".join(out).replace(" ", "") == "".join(deltas).replace(" ", "")


# --- first-chunk latency ---------------------------------------------------
# Nothing is playing while the opening chunk is being built, so it breaks at
# clause marks too. Later chunks wait for a sentence end (better prosody,
# hidden behind playback).


def test_first_chunk_breaks_at_a_clause_mark():
    chunker = StreamingChunker()
    # The opening clause is emitted immediately; the short tail waits for more
    # text (or flush) because it is below the normal minimum.
    assert chunker.push("مرحبا بك، كيف يمكنني مساعدتك؟") == [
        "مرحبا بك،",
        "كيف يمكنني مساعدتك؟",
    ]


def test_later_chunks_ignore_clause_marks():
    chunker = StreamingChunker()
    chunker.push("أهلا وسهلا بك. ")  # consume the first chunk
    assert chunker.push("جملة طويلة، بها فاصلة ولا تنتهي بعد") == []


def test_a_too_short_opening_clause_is_not_split_off():
    """A 4-char fragment is not worth its own TTS call, so it stays joined."""
    chunker = StreamingChunker()
    assert chunker.push("نعم، أكيد.") == ["نعم، أكيد."]


def test_first_chunk_is_capped_even_without_punctuation():
    chunker = StreamingChunker()
    emitted = chunker.push("كلمة " * 30)
    assert emitted, "an unpunctuated opening must still start playback"
    assert len(emitted[0]) <= 70


def test_flush_counts_as_emitted_so_a_resumed_reply_is_not_treated_as_first():
    chunker = StreamingChunker()
    chunker.push("بداية بلا علامات")
    chunker.flush()
    assert chunker.push("جملة، بها فاصلة فقط") == []


def test_streaming_is_linear_not_quadratic():
    """A long reply must not rescan its whole buffer on every delta."""
    import time

    chunker = StreamingChunker()
    text = "كلمة " * 3000
    start = time.perf_counter()
    for i in range(0, len(text), 4):
        chunker.push(text[i : i + 4])
    chunker.flush()
    assert time.perf_counter() - start < 1.0


def test_decimals_survive_clause_splitting_in_the_first_chunk():
    chunker = StreamingChunker()
    out = chunker.push("السعر 3,5 دينار فقط، شكرا.") + chunker.flush()
    assert "3,5" in "".join(out)
