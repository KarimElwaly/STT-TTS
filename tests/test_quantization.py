"""ASR weight quantization.

Quantization here is a *latency* decision, not a memory one. Full precision is
the fastest single transcription, but on a 6 GiB card it leaves no room for
TTS, so every turn pays ~3.3 s of offload/onload traffic. nf4 costs ~100 ms per
transcription and buys back all of it.
"""

import pytest

from voicegw.common.config import Quantization, Settings
from voicegw.engines import cohere_asr


def _free(monkeypatch, mib: int):
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda: (mib * 1024 * 1024, 6141 * 1024 * 1024),
        raising=False,
    )


# --- mode selection --------------------------------------------------------


@pytest.mark.parametrize("mode", [Quantization.NONE, Quantization.INT8, Quantization.NF4])
def test_explicit_mode_is_honoured(mode, monkeypatch):
    _free(monkeypatch, 5080)
    assert cohere_asr.resolve_quantization(Settings(asr_quantization=mode)) is mode


def test_auto_quantizes_when_both_models_would_not_fit(monkeypatch):
    """The real RTX 4050 case: 5080 MiB free, 3940 + 1937 needed."""
    _free(monkeypatch, 5080)
    assert cohere_asr.resolve_quantization(Settings()) is Quantization.NF4


def test_auto_keeps_full_precision_when_there_is_room(monkeypatch):
    _free(monkeypatch, 16_000)
    assert cohere_asr.resolve_quantization(Settings()) is Quantization.NONE


def test_auto_quantizes_when_the_probe_fails(monkeypatch):
    import torch

    def boom():
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "mem_get_info", boom, raising=False)
    assert cohere_asr.resolve_quantization(Settings()) is Quantization.NF4


# --- what the mode implies -------------------------------------------------


def test_quantized_engines_declare_a_smaller_footprint(monkeypatch):
    _free(monkeypatch, 5080)
    engine = cohere_asr.build_gpu()
    assert engine.quantization is Quantization.NF4
    assert engine.info.vram_mib == cohere_asr.VRAM_MIB[Quantization.NF4]
    assert engine.info.vram_mib < cohere_asr.VRAM_MIB[Quantization.NONE]


def test_quantized_engines_are_not_movable(monkeypatch):
    """bitsandbytes pins weights to the device it quantized them on."""
    _free(monkeypatch, 5080)
    assert cohere_asr.build_gpu().info.movable is False

    _free(monkeypatch, 16_000)
    assert cohere_asr.build_gpu().info.movable is True


def test_offload_is_a_noop_for_quantized_weights(monkeypatch):
    """Rather than raise: a quantized model is small enough to stay put."""
    _free(monkeypatch, 5080)
    engine = cohere_asr.build_gpu()
    engine._model = object()  # a real `.to()` would raise; we must not reach it
    engine.offload()
    assert engine._offloaded is False


def test_dtype_advertises_the_mode(monkeypatch):
    _free(monkeypatch, 5080)
    assert cohere_asr.build_gpu().info.dtype == "nf4/bfloat16"

    _free(monkeypatch, 16_000)
    assert cohere_asr.build_gpu().info.dtype == "bfloat16"


def test_cpu_build_is_never_quantized():
    engine = cohere_asr.build_cpu()
    assert engine.quantization is Quantization.NONE
    assert engine.info.vram_mib == 0


def test_every_mode_has_a_footprint_and_an_rtf():
    modes = set(Quantization) - {Quantization.AUTO}
    assert modes == set(cohere_asr.VRAM_MIB) == set(cohere_asr.RTF_ESTIMATE)


def test_quantized_inference_uses_bfloat16_inputs(monkeypatch):
    """The feature extractor emits float32; the encoder bias will not promote."""
    import torch

    _free(monkeypatch, 5080)
    engine = cohere_asr.build_gpu()
    assert engine._compute_dtype(torch) is torch.bfloat16
    assert cohere_asr.build_cpu()._compute_dtype(torch) is torch.float32
