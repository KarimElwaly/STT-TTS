"""Does 8-bit quantization make both models fit -- and at what cost?

Exclusive residency costs ~3.3 s of PCIe traffic per conversational turn. If
the ASR model can be halved so both models stay resident, that cost vanishes.
This measures VRAM, latency and the actual transcription text for bf16 vs int8
vs nf4, so the tradeoff is a measurement rather than a guess.

    python scripts/quant_compare.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicegw.common.audio import decode_audio  # noqa: E402
from voicegw.common.protocols import SAMPLE_RATE_IN  # noqa: E402

MIB = 1024 * 1024
SAMPLE = Path("out/offline.wav")
MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
TTS_MIB = 1937  # measured by scripts/vram_budget.py


def run(mode: str, audio) -> dict:
    from transformers import AutoProcessor, BitsAndBytesConfig, CohereAsrForConditionalGeneration

    kwargs: dict = {"local_files_only": True, "low_cpu_mem_usage": True}
    if mode == "bf16":
        kwargs |= {"dtype": torch.bfloat16, "device_map": "cuda:0"}
    elif mode == "int8":
        kwargs |= {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    elif mode == "nf4":
        kwargs |= {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        }

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = CohereAsrForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    load_s = time.perf_counter() - t0
    vram = torch.cuda.memory_allocated() // MIB

    def once() -> str:
        inputs = processor(audio, sampling_rate=SAMPLE_RATE_IN, return_tensors="pt", language="ar")
        # The feature extractor emits float32; the encoder's weights are bf16
        # (the compute dtype in every mode, including the quantized ones).
        inputs = inputs.to(model.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=256)
        return processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    text = once()  # warm up
    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        text = once()
        timings.append((time.perf_counter() - t0) * 1000)

    model = processor = None
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "vram": vram,
        "load_s": load_s,
        "ms": sorted(timings)[1],
        "text": text,
    }


def main() -> int:
    if not SAMPLE.is_file():
        print(f"missing {SAMPLE}")
        return 1
    audio = decode_audio(SAMPLE.read_bytes())
    free = torch.cuda.mem_get_info()[0] // MIB
    print(f"free VRAM {free} MiB | TTS needs {TTS_MIB} MiB | audio {len(audio) / 16000:.1f}s\n")

    baseline = None
    for mode in ("bf16", "int8", "nf4"):
        try:
            r = run(mode, audio)
        except Exception as exc:
            print(f"{mode:<6} FAILED: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        if baseline is None:
            baseline = r
        both = r["vram"] + TTS_MIB
        fits = "FITS with TTS" if both < free else f"does NOT fit (needs {both})"
        same = " (same text)" if r["text"] == baseline["text"] else " (TEXT DIFFERS)"
        print(
            f"{mode:<6} vram={r['vram']:>5} MiB  load={r['load_s']:>5.1f}s  "
            f"stt={r['ms']:>6.0f} ms  {fits}"
        )
        print(f"       {r['text']!r}{same}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
