"""Where does the 873 ms of TTS first-audio latency actually go?

With both models now resident, TTS is 71% of the time-to-first-audio. Before
tuning anything we need to know what the 873 ms is made of: a fixed per-call
overhead, the diffusion steps, or the length of the chunk being synthesized.
Only the last two are ours to control, and they pull in different directions.

    python scripts/tts_latency_profile.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from voicegw.engines.omnivoice_tts import resolve_language  # noqa: E402
from voicegw.engines.voices import VoiceRegistry  # noqa: E402

# Realistic opening chunks of a spoken Arabic reply, shortest first. These are
# what StreamingChunker actually emits for the first TTS call.
CHUNKS = [
    "نعم،",
    "أهلا بك، كيف أساعدك؟",
    "بالطبع، يمكنني مساعدتك في ذلك، فقط أخبرني بالتفاصيل التي تحتاجها.",
    "مرحبا بك في النظام الصوتي. يمكنني الاستماع إليك والرد عليك بصوت طبيعي، "
    "كما يمكنني مساعدتك في مهام كثيرة مثل الترجمة والتلخيص والإجابة عن الأسئلة.",
]
STEPS = [4, 8, 16, 32]
REPEATS = 3


def main() -> int:
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

    voice = VoiceRegistry().get("default")
    base: dict[str, object] = {"language": resolve_language(voice.language)}
    if voice.is_clone:
        base |= {"ref_audio": voice.ref_audio, "ref_text": voice.ref_text}
    elif voice.description:
        base["instruct"] = voice.description

    print("loading OmniVoice …")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    model.generate(
        text="تجربة", **base, generation_config=OmniVoiceGenerationConfig(num_step=4)
    )  # warm

    def run(text: str, steps: int) -> tuple[float, float]:
        timings = []
        audio = None
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            audio = model.generate(
                text=text, **base, generation_config=OmniVoiceGenerationConfig(num_step=steps)
            )
            timings.append((time.perf_counter() - t0) * 1000)
        secs = len(np.asarray(audio[0], dtype=np.float32)) / 24_000
        return sorted(timings)[len(timings) // 2], secs

    header = "chars  audio_s  " + "".join(f"{s:>7}st" for s in STEPS)
    print(f"\n{header}")
    print("-" * len(header))

    rows = []
    for text in CHUNKS:
        cells, audio_s = [], 0.0
        for steps in STEPS:
            ms, audio_s = run(text, steps)
            cells.append(ms)
        rows.append((len(text), audio_s, cells))
        cell_str = "".join(f"{c:>9.0f}" for c in cells)
        print(f"{len(text):>5}  {audio_s:>7.2f}  {cell_str}")

    # Fixed overhead is what survives at the smallest chunk and fewest steps;
    # anything above it is work we can trade against quality.
    floor = rows[0][2][0]
    print(f"\nfloor (shortest chunk, {STEPS[0]} steps): {floor:.0f} ms")
    print("cost of each extra step at that length:")
    for i, steps in enumerate(STEPS[1:], start=1):
        delta = rows[0][2][i] - floor
        print(f"  {STEPS[0]:>2} -> {steps:>2} steps: +{delta:>5.0f} ms")
    print("\ncost of a longer first chunk (at 16 steps):")
    idx = STEPS.index(16)
    for chars, audio_s, cells in rows:
        print(f"  {chars:>4} chars -> {cells[idx]:>5.0f} ms for {audio_s:.2f}s of audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
