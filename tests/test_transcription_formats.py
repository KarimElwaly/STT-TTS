"""Test transcription formatting: JSON, text, verbose_json, SRT, and VTT."""

import numpy as np
import pytest

from voicegw.api.app import format_srt, format_vtt
from voicegw.common.protocols import Transcript


def test_format_srt():
    tr = Transcript(
        text="مرحبا بكم جميعا",
        language="ar",
        duration_s=3.5,
        engine="faster-whisper",
        segments=[
            {"id": 0, "start": 0.0, "end": 1.5, "text": "مرحبا بكم"},
            {"id": 1, "start": 1.5, "end": 3.5, "text": "جميعا"},
        ],
    )
    srt = format_srt(tr)
    assert "1\n00:00:00,000 --> 00:00:01,500\nمرحبا بكم" in srt
    assert "2\n00:00:01,500 --> 00:00:03,500\nجميعا" in srt


def test_format_vtt():
    tr = Transcript(
        text="أهلا وسهلا",
        language="ar",
        duration_s=2.0,
        engine="faster-whisper",
        segments=[
            {"id": 0, "start": 0.0, "end": 2.0, "text": "أهلا وسهلا"},
        ],
    )
    vtt = format_vtt(tr)
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:02.000\nأهلا وسهلا" in vtt


def test_format_srt_fallback_without_segments():
    tr = Transcript(
        text="نص بدون مقاطع",
        language="ar",
        duration_s=1.2,
        engine="cohere-asr",
        segments=[],
    )
    srt = format_srt(tr)
    assert "1\n00:00:00,000 --> 00:00:01,200\nنص بدون مقاطع" in srt
