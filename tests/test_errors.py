"""Failure messages must tell the user what to do about it."""

import pytest

from voicegw.common.errors import EngineUnavailable, hint_for


def test_message_includes_engine_reason_and_hint():
    exc = EngineUnavailable("cohere-asr", "401 Client Error", "Set HF_TOKEN.")
    text = str(exc)
    assert "cohere-asr" in text
    assert "401 Client Error" in text
    assert "Set HF_TOKEN." in text


def test_message_without_a_hint_has_no_dangling_separator():
    assert str(EngineUnavailable("x", "boom")) == "x is unavailable: boom"


@pytest.mark.parametrize(
    "message, expected",
    [
        ("You are trying to access a gated repo.", "HF_TOKEN"),
        ("401 Client Error. Cannot access gated repo", "HF_TOKEN"),
        ("Access to model CohereLabs/x is restricted", "HF_TOKEN"),
        ("CUDA failed with error out of memory", "VRAM"),
        ("No module named 'omnivoice'", "dependency"),
        ("OSError: [WinError 127] The specified procedure could not be found", "dependency"),
        ("Max retries exceeded with url", "downloaded"),
    ],
)
def test_known_failures_get_actionable_hints(message, expected):
    assert expected in hint_for(RuntimeError(message))


def test_unknown_failure_yields_no_hint():
    assert hint_for(RuntimeError("something entirely novel")) == ""


def test_hint_matching_is_case_insensitive():
    assert hint_for(RuntimeError("GATED REPO")) == hint_for(RuntimeError("gated repo"))
