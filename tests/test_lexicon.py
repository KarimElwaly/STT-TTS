import pytest

from voicegw.common.lexicon import LexiconManager, apply_lexicon, compile_lexicon, normalize_lexicon


def test_normalize_lexicon():
    raw = {
        " AI ": " إيه آي ",
        "empty_val": None,
        None: "ignored",
        "": "ignored",
        "GPU": "جي بي يو",
    }
    cleaned = normalize_lexicon(raw)
    assert "AI" in cleaned
    assert cleaned["AI"] == "إيه آي"
    assert "GPU" in cleaned
    assert "" not in cleaned
    assert len(cleaned) == 2


def test_apply_lexicon_single_pass():
    lex = {
        "AI": "إيه آي",
        "API": "إيه بي آي",
        "FastAPI": "فاست إيه بي آي",
    }
    # Longest match first: FastAPI should match as a single token, not Fast + API
    text = "نحن نستخدم FastAPI مع نماذج AI"
    res = apply_lexicon(text, lex)
    assert "فاست إيه بي آي" in res
    assert "إيه آي" in res
    assert "Fast" not in res


def test_lexicon_word_boundaries():
    lex = {"cat": "قطة"}
    text = "The cat is in the category"
    res = apply_lexicon(text, lex)
    assert "The قطة is in the category" == res


def test_lexicon_manager_empty(tmp_path):
    mgr = LexiconManager(tmp_path / "absent.json")
    assert mgr.apply("Hello World") == "Hello World"


def test_lexicon_manager_file(tmp_path):
    json_path = tmp_path / "lexicon.json"
    json_path.write_text('{"STT": "إس تي تي", "TTS": "تي تي إس"}', encoding="utf-8")
    mgr = LexiconManager(json_path)
    res = mgr.apply("نظام STT و TTS")
    assert "إس تي تي" in res
    assert "تي تي إس" in res
