"""How much VRAM does each model actually need, and do they fit together?

Exclusive residency costs a full PCIe round-trip per turn. If both models fit
at once, that cost disappears entirely. This measures the real footprints
instead of guessing from parameter counts.

    python scripts/vram_budget.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MIB = 1024 * 1024


def mib(n: int) -> int:
    return n // MIB


def report(label: str) -> None:
    print(
        f"{label:<28} allocated={mib(torch.cuda.memory_allocated()):>5} "
        f"reserved={mib(torch.cuda.memory_reserved()):>5}"
    )


def main() -> int:
    free, total = torch.cuda.mem_get_info()
    print(f"GPU total {mib(total)} MiB, free {mib(free)} MiB (desktop uses the rest)\n")

    from voicegw.engines import cohere_asr, omnivoice_tts

    report("baseline")

    t0 = time.perf_counter()
    asr = cohere_asr.build_gpu()
    asr.load()
    asr_mib = mib(torch.cuda.memory_allocated())
    report(f"ASR loaded ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    tts = omnivoice_tts.build_gpu()
    tts.load()
    both_mib = mib(torch.cuda.memory_allocated())
    report(f"+ TTS loaded ({time.perf_counter() - t0:.1f}s)")

    tts_mib = both_mib - asr_mib
    headroom = mib(free) - both_mib
    print(f"\nASR {asr_mib} MiB + TTS {tts_mib} MiB = {both_mib} MiB")
    print(f"free was {mib(free)} MiB -> headroom {headroom} MiB")
    print(
        "\nboth fit simultaneously: "
        + ("YES -- shared residency is viable" if headroom > 0 else "NO")
    )

    # What a swap actually costs.
    for label, engine in (("ASR", asr), ("TTS", tts)):
        t0 = time.perf_counter()
        engine.offload()
        off = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        engine.onload()
        on = (time.perf_counter() - t0) * 1000
        print(f"{label} swap: offload {off:.0f} ms + onload {on:.0f} ms = {off + on:.0f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
