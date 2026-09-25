"""Sweep OmniVoice diffusion steps to pick a latency/quality operating point.

Runs offline (no server) and writes a WAV per setting so you can listen.

    python scripts/tune_tts.py --steps 4 8 16 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from voicegw.common.audio import encode_wav  # noqa: E402
from voicegw.engines.omnivoice_tts import resolve_language  # noqa: E402
from voicegw.engines.voices import VoiceRegistry  # noqa: E402

TEXT = "السلام عليكم، كيف حالك اليوم؟ أتمنى أن تكون بخير."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16, 24, 32])
    parser.add_argument("--voice", default="default")
    parser.add_argument("--out", type=Path, default=Path("out/tts_steps"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

    voice = VoiceRegistry().get(args.voice)
    args.out.mkdir(parents=True, exist_ok=True)

    print("loading OmniVoice …")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)

    kwargs: dict[str, object] = {"text": TEXT, "language": resolve_language(voice.language)}
    if voice.is_clone:
        kwargs["ref_audio"] = voice.ref_audio
        kwargs["ref_text"] = voice.ref_text
    elif voice.description:
        kwargs["instruct"] = voice.description

    model.generate(**kwargs, generation_config=OmniVoiceGenerationConfig(num_step=4))  # warm

    print(f"\n{'steps':>6}  {'median ms':>10}  {'audio s':>8}  {'RTF':>6}  file")
    for steps in args.steps:
        timings: list[float] = []
        audio = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            audio = model.generate(
                **kwargs, generation_config=OmniVoiceGenerationConfig(num_step=steps)
            )
            timings.append((time.perf_counter() - t0) * 1000)
        samples = np.asarray(audio[0], dtype=np.float32)
        audio_s = len(samples) / 24_000
        median = sorted(timings)[len(timings) // 2]
        path = args.out / f"steps_{steps:02d}.wav"
        path.write_bytes(encode_wav(samples, 24_000))
        print(
            f"{steps:>6}  {median:>9.0f}  {audio_s:>8.2f}  {median / 1000 / audio_s:>6.3f}  {path}"
        )

    print("\nListen to the files and pick the lowest step count that still sounds good.")
    print("Then set VOICEGW_TTS_NUM_STEP in .env.")


if __name__ == "__main__":
    main()
