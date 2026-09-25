"""Benchmark every registered engine on your own machine.

The CPU-profile defaults in :mod:`voicegw.engines.registry` are a starting
point, not a measurement. Run this to get real RTF / latency / WER numbers and
then pin ``VOICEGW_ASR_ENGINE`` / ``VOICEGW_TTS_ENGINE`` accordingly.

    python scripts/bench.py --audio-dir samples/ --profile gpu
    python scripts/bench.py --audio-dir samples/ --profile cpu

``samples/`` should contain audio files; put an optional ``<name>.txt`` next to
each one holding the reference transcript to get a WER column.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from voicegw.common.audio import load_audio_file  # noqa: E402
from voicegw.common.config import Profile, get_settings, resolve_profile  # noqa: E402
from voicegw.common.protocols import SAMPLE_RATE_IN  # noqa: E402
from voicegw.engines import registry  # noqa: E402
from voicegw.engines.vad import gate_audio  # noqa: E402
from voicegw.engines.voices import VoiceRegistry  # noqa: E402

console = Console()

TTS_TEXTS = [
    "مرحبا، كيف يمكنني مساعدتك اليوم؟",
    "الطقس اليوم مشمس ودرجة الحرارة سبعة وعشرون درجة مئوية.",
    "Hello, this is a short English sentence for comparison.",
]


def _wer(reference: str, hypothesis: str) -> float | None:
    try:
        import jiwer
    except ImportError:
        return None
    if not reference.strip():
        return None
    return float(jiwer.wer(reference, hypothesis))


def _peak_memory_mib(device: str) -> float | None:
    if device == "cpu":
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return None
    try:
        import torch

        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None


def bench_stt(engine_ids: list[str], res, clips: list[tuple[Path, np.ndarray, str]]) -> list[dict]:
    rows: list[dict] = []
    for engine_id in engine_ids:
        console.print(f"[cyan]STT[/] {engine_id} …")
        try:
            engine = registry.STT_ENGINES[engine_id](res)
            t0 = time.perf_counter()
            engine.load()
            load_s = time.perf_counter() - t0
        except Exception as exc:
            console.print(f"  [red]load failed: {exc}[/]")
            continue

        # Warm up so the first clip doesn't carry lazy-init cost.
        try:
            engine.transcribe(np.zeros(SAMPLE_RATE_IN, dtype=np.float32), "ar")
        except Exception as exc:
            console.print(f"  [red]warmup failed: {exc}[/]")
            engine.unload()
            continue

        rtfs: list[float] = []
        latencies: list[float] = []
        wers: list[float] = []
        for path, audio, reference in clips:
            gated = gate_audio(audio)
            if gated.size == 0:
                console.print(f"  [yellow]{path.name}: VAD found no speech, skipped[/]")
                continue
            duration = len(gated) / SAMPLE_RATE_IN
            t0 = time.perf_counter()
            try:
                result = engine.transcribe(gated, "ar")
            except Exception as exc:
                console.print(f"  [red]{path.name}: {exc}[/]")
                continue
            elapsed = time.perf_counter() - t0
            rtfs.append(elapsed / duration)
            latencies.append(elapsed * 1000)
            if reference and (w := _wer(reference, result.text)) is not None:
                wers.append(w)

        rows.append(
            {
                "engine": engine_id,
                "device": engine.info.device,
                "load_s": round(load_s, 1),
                "rtf_p50": round(statistics.median(rtfs), 3) if rtfs else None,
                "latency_p50_ms": round(statistics.median(latencies)) if latencies else None,
                "latency_p95_ms": round(max(latencies)) if latencies else None,
                "wer": round(statistics.mean(wers), 3) if wers else None,
                "peak_mib": round(_peak_memory_mib(engine.info.device) or 0),
                "declared_realtime": engine.info.realtime_capable,
            }
        )
        engine.unload()
    return rows


def bench_tts(engine_ids: list[str], res, voice_id: str) -> list[dict]:
    voices = VoiceRegistry()
    try:
        voice = voices.get(voice_id)
    except KeyError as exc:
        console.print(f"[red]{exc}[/]")
        return []

    rows: list[dict] = []
    for engine_id in engine_ids:
        console.print(f"[cyan]TTS[/] {engine_id} …")
        try:
            engine = registry.TTS_ENGINES[engine_id](res)
            if not engine.supports(voice):
                console.print(f"  [yellow]does not support voice {voice.id!r}, skipped[/]")
                continue
            t0 = time.perf_counter()
            engine.load()
            load_s = time.perf_counter() - t0
        except Exception as exc:
            console.print(f"  [red]load failed: {exc}[/]")
            continue

        try:
            next(iter(engine.synthesize("تجربة", voice)), None)  # warmup
        except Exception as exc:
            console.print(f"  [red]warmup failed: {exc}[/]")
            engine.unload()
            continue

        first_chunk_ms: list[float] = []
        rtfs: list[float] = []
        for text in TTS_TEXTS:
            t0 = time.perf_counter()
            total_samples = 0
            sample_rate = 24_000
            first: float | None = None
            try:
                for chunk in engine.synthesize(text, voice):
                    if first is None:
                        first = (time.perf_counter() - t0) * 1000
                    total_samples += chunk.samples.size
                    sample_rate = chunk.sample_rate
            except Exception as exc:
                console.print(f"  [red]{exc}[/]")
                continue
            elapsed = time.perf_counter() - t0
            audio_s = total_samples / sample_rate
            if first is not None:
                first_chunk_ms.append(first)
            if audio_s > 0:
                rtfs.append(elapsed / audio_s)

        rows.append(
            {
                "engine": engine_id,
                "device": engine.info.device,
                "load_s": round(load_s, 1),
                "rtf_p50": round(statistics.median(rtfs), 3) if rtfs else None,
                "first_chunk_p50_ms": round(statistics.median(first_chunk_ms))
                if first_chunk_ms
                else None,
                "peak_mib": round(_peak_memory_mib(engine.info.device) or 0),
                "declared_realtime": engine.info.realtime_capable,
            }
        )
        engine.unload()
    return rows


def _render(title: str, rows: list[dict]) -> None:
    if not rows:
        console.print(f"[yellow]{title}: no results[/]")
        return
    table = Table(title=title)
    for column in rows[0]:
        table.add_column(column)
    for row in rows:
        table.add_row(*("—" if v is None else str(v) for v in row.values()))
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=Path("samples"))
    parser.add_argument("--profile", choices=[p.value for p in Profile], default=None)
    parser.add_argument("--voice", default="default")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.profile:
        os.environ["VOICEGW_PROFILE"] = args.profile
        get_settings.cache_clear()

    # Benchmark every engine the profile could use, not just the defaults.
    settings = get_settings()
    settings.asr_engine = None
    settings.tts_engine = None
    res = resolve_profile(settings)
    console.print(f"[bold]profile[/] {res.profile.value} · {res.reason}\n")

    clips: list[tuple[Path, np.ndarray, str]] = []
    if not args.skip_stt:
        if not args.audio_dir.is_dir():
            console.print(
                f"[yellow]{args.audio_dir} not found — skipping STT. "
                "Create it and drop in a few Arabic audio files.[/]"
            )
        else:
            for path in sorted(args.audio_dir.iterdir()):
                if path.suffix.lower() not in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}:
                    continue
                reference_file = path.with_suffix(".txt")
                reference = (
                    reference_file.read_text(encoding="utf-8").strip()
                    if reference_file.exists()
                    else ""
                )
                clips.append((path, load_audio_file(str(path)), reference))
            console.print(f"loaded {len(clips)} clip(s) from {args.audio_dir}\n")

    results: dict[str, list[dict]] = {}
    if clips:
        candidates = (
            ["cohere-asr", "faster-whisper"]
            if res.profile is Profile.GPU
            else [
                "faster-whisper",
                "cohere-asr-cpu",
            ]
        )
        results["stt"] = bench_stt(candidates, res, clips)
        _render("STT", results["stt"])

    if not args.skip_tts:
        candidates = ["omnivoice"] if res.profile is Profile.GPU else ["omnivoice-cpu", "piper"]
        results["tts"] = bench_tts(candidates, res, args.voice)
        _render("TTS", results["tts"])

    console.print("\n[dim]Pin the winners with VOICEGW_ASR_ENGINE / VOICEGW_TTS_ENGINE in .env.[/]")

    if args.json_out:
        args.json_out.write_text(
            json.dumps({"profile": res.profile.value, **results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
