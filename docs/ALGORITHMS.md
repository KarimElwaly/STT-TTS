# Algorithms

Every number here was measured on the development machine — RTX 4050 Laptop
(6140 MiB total, ~5080 MiB free with a desktop session running), Windows,
Python 3.11, torch 2.6.0+cu124. Reproduce with the scripts named in each
section. Treat them as *this machine's* numbers, not universal constants.

---

## 1. Latency budget

Conversational turn-taking feels broken past roughly a second. The budget:

```
user stops speaking
   │
   ├─ VAD endpoint detection      600 ms  (min_silence_ms — a design choice)
   ├─ STT                         326 ms
   ├─ agent first token        (external, local LLM)
   └─ TTS first chunk             466 ms
                                 ───────
   time to first audio            792 ms
```

The 600 ms silence window is not overhead to be optimized away — cut it and the
assistant interrupts people who pause mid-sentence. It is excluded from the
measurements below, which start at "audio is ready to transcribe".

Progress, via `scripts/turn_latency.py`:

| configuration | STT | TTS 1st | total |
| --- | --- | --- | --- |
| bf16 ASR, exclusive residency | 3683 ms | 5081 ms | 8764 ms |
| nf4 ASR, shared residency | 351 ms | 873 ms | 1224 ms |
| + reduced first-chunk steps | 326 ms | 466 ms | **792 ms** |

---

## 2. Residency: an arithmetic decision

**Problem.** ASR (3940 MiB) + TTS (1937 MiB) = 5877 MiB against 5080 MiB free.
Short by 797 MiB.

The obvious fix — swap models in and out — was implemented first and measured
(`scripts/vram_budget.py`):

| operation | cost |
| --- | --- |
| ASR offload → onload | 2203 ms |
| TTS offload → onload | 1062 ms |
| **per turn** (STT then TTS) | **~3265 ms** |

That is four times the entire latency budget, paid on every single turn. The
swap machinery is correct and still exists as a fallback, but the real answer
was to make the models fit.

**Algorithm.**

```
if profile is CPU:                      → SHARED   (no VRAM contention)
if VOICEGW_RESIDENCY is explicit:       → that value
needed = Σ engine.info.vram_mib
if any engine declares 0 (unknown):     → fall back to total-VRAM threshold
if needed + headroom ≤ free VRAM:       → SHARED
else:                                   → EXCLUSIVE
```

Headroom (`VOICEGW_VRAM_HEADROOM_MIB`, default 700) covers activations, KV
cache and fragmentation — the declared footprints are *weights only*.

The "any engine declares 0" branch is deliberate: acting on a half-known total
would be worse than the coarse threshold, because a confident wrong answer here
means an OOM at inference time rather than a slow-but-working gateway.

---

## 3. Quantization: the decision that unblocked everything

Since swapping costs 3.3 s/turn, anything that makes both models fit is worth
up to 3.3 s of added inference time. That reframes quantization entirely: the
question is not "is nf4 as fast as bf16?" but "is nf4 within 3.3 s of bf16?"

Measured with `scripts/quant_compare.py`, 2.7 s Arabic utterance:

| mode | VRAM | load | STT | fits beside TTS? | transcript |
| --- | --- | --- | --- | --- | --- |
| bf16 | 3940 MiB | 5.0 s | 210 ms | no (5877 > 5080) | — |
| int8 | 2244 MiB | 9.3 s | 1067 ms | yes | identical |
| **nf4** | **1413 MiB** | 6.9 s | **310 ms** | **yes** | identical |

**nf4 wins decisively**: 64 % smaller for 100 ms. `int8` is the trap — 5× the
latency for less than half the saving, because bitsandbytes' 8-bit matmul path
is far slower than its 4-bit one on this hardware. It stays selectable because
that ratio is model- and card-dependent, but `auto` never picks it.

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4: matches a normal weight prior
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # quantizes the quantization constants too
)
```

`auto` quantizes **only when it has to** — full precision is both faster and
more faithful, so on a card with room it picks `none`. It also checks that
bitsandbytes is actually installed: `transformers` exposes `BitsAndBytesConfig`
even when it is not, so the config importing cleanly says nothing about whether
the weights will load. Without it, `auto` falls back to full precision and
accepts the swap cost — a slower gateway beats a dead one. An *explicit*
`VOICEGW_ASR_QUANTIZATION=nf4` is still honoured and fails loudly, matching how
`VOICEGW_PROFILE=gpu` behaves on a machine with no usable CUDA.

### Two sharp edges

**Quantized weights cannot move between devices.** bitsandbytes pins them to
the device it quantized them on; `.to()` raises. Hence
`EngineInfo.movable = False`, and `offload()` returns early instead of raising.
Ignoring the request is correct rather than lazy: a quantized model is small by
construction, which is precisely why residency chose SHARED and why nothing
should be asking it to swap.

**The input dtype must be set explicitly.** The feature extractor emits
float32; the encoder bias is bfloat16; PyTorch raises rather than promoting:

```
RuntimeError: Input type (float) and bias type (struct c10::BFloat16) should be the same
```

A quantized model's `.dtype` is *misleading* (its weights are uint8), so the
cast keys off the compute dtype we requested, not off the model:

```python
inputs = inputs.to(self._model.device, dtype=self._compute_dtype(torch))
```

### Verification caveat

"Identical transcript" rests on **one utterance**. That is a good signal, not
proof. Before trusting nf4 on dialect-heavy or noisy audio, re-run
`scripts/quant_compare.py` against a representative sample.

---

## 4. TTS: steps, not chunk size

The intuition — "synthesize a shorter first chunk to start talking sooner" —
turns out to be **wrong on this model**. Measured with
`scripts/tts_latency_profile.py`:

| chars | audio | 4 steps | 8 steps | 16 steps | 32 steps |
| --- | --- | --- | --- | --- | --- |
| 4 | 1.3 s | 206 ms | 398 ms | 785 ms | 1559 ms |
| 20 | 2.0 s | 199 ms | 386 ms | 740 ms | 1464 ms |
| 65 | 5.7 s | 226 ms | 402 ms | 784 ms | 1506 ms |
| 145 | 12.7 s | 363 ms | 609 ms | 1094 ms | 2083 ms |

Read across the 16-step column: a **4-character** chunk costs 785 ms and a
**65-character** chunk costs 784 ms. Below ~65 characters, chunk length is
free. Read down any column: cost is ~linear in steps, about **48 ms/step**.

So the lever is `num_step`. Shrinking chunks below ~65 characters buys nothing
and costs prosody.

### First-chunk asymmetry

Only the **first** chunk of a reply is heard in silence. Every later chunk is
generated while earlier audio is still playing, so its latency is hidden behind
playback. Degrading a hidden chunk costs quality for no perceived speed.

Hence `VOICEGW_TTS_FIRST_CHUNK_NUM_STEP` (default 8) vs
`VOICEGW_TTS_NUM_STEP` (default 16), plumbed through as `urgent=True` from the
callers that actually gate playback:

- realtime WebSocket — the first `speak()` of a turn
- `/v1/audio/speech` with `stream=true`
- **not** `stream=false` — that caller waits for the whole file either way, so
  degrading it would cost quality and save nothing

This is the single largest remaining win: 873 ms → 466 ms of TTS first-audio.

**It is a quality claim, and quality claims need ears.**
`scripts/ab_first_chunk.py` renders the same sentence both ways through the
real server paths. Listen for whether the opening words degrade and whether the
seam where the step count changes is audible. Set the two knobs equal to
disable the asymmetry.

---

## 5. Streaming text chunker

Agent tokens arrive one at a time; TTS needs a speakable unit. Waiting for the
full reply wastes the entire generation time.

`StreamingChunker` applies **asymmetric break rules** for the same reason as
above — the first chunk is heard in silence:

| | break characters | max | min |
| --- | --- | --- | --- |
| first chunk | sentence ends **+ clause marks** `، , ; :` | 70 | 8 |
| later chunks | sentence ends `. ! ? ؟ … \n` | 220 | 12 |

So `"مرحبا بك،"` (9 characters) starts playback immediately, where a
sentence-only rule would have waited for the full
`"مرحبا بك، كيف يمكنني مساعدتك؟"` (29 characters). Later chunks wait for a
sentence end and get better prosody — a wait hidden by playback.

Two implementation details that matter:

- **`_scanned` cursor.** Without it, every token delta rescans the whole
  buffer — quadratic over a long reply. `test_streaming_is_linear_not_quadratic`
  pushes a 15 000-character reply in 4-character deltas and asserts it
  completes in under a second.
- **Decimal protection.** `_is_protected()` prevents splitting `"3.5"` at its
  period, in both the sentence splitter and the first-chunk clause path.

Round-trip invariant: concatenating all emitted chunks reproduces the input
(modulo whitespace). Currently covered by one deterministic case,
`test_streaming_preserves_full_text` — a randomized property test would be a
stronger net here, since a chunker that silently drops text is near-impossible
to debug from audio alone.

---

## 6. VAD: mandatory, not optional

The Cohere model card warns it is "eager to transcribe, even non-speech
sounds". This is not a theoretical concern — measured directly:

| input | VAD on | VAD off |
| --- | --- | --- |
| digital silence | `""` | `"@@@فراغ"` |
| room tone | `""` | hallucinated text |

So `gate_audio()` runs before every transcription and the ASR engines are never
handed raw microphone audio.

**Frame loop** (32 ms frames, 16 kHz):

```
rms < noise_gate_rms        → probability 0.0   (hard gate, before the model)
p ≥ threshold               → speech
silence ≥ min_silence_ms    → utterance ends
speech < min_speech_ms      → discarded
                            → ±speech_pad_ms so words aren't clipped
```

The RMS floor runs *before* Silero regardless of backend: it is nearly free and
catches constant low-level hiss that a neural VAD may score above threshold.

Silero (ONNX, ~1 MB, expects exactly 512-sample chunks) falls back to the
energy gate if unavailable — a degraded gate is much better than none.

**Long audio.** The AED decoder goes out of distribution past ~30 s, so
`chunk_long_audio()` splits at *silence boundaries* below a 28 s cap — never
mid-word, which is what a fixed-size split would do.

---

## 7. Offline mode

Four distinct problems, each needing its own fix:

**1. `HF_HUB_OFFLINE` is snapshotted at import.** Setting the env var after
`huggingface_hub` is imported does nothing. `offline.enable()` patches
`huggingface_hub.constants.HF_HUB_OFFLINE` directly *and* sets the env var for
subprocesses.

**2. A hidden dependency.** OmniVoice pulls `eustlb/higgs-audio-v2-tokenizer`,
which appears in no documentation — found only by watching network traffic on a
supposedly-offline run. `voicegw fetch` now includes it.

**3. Piper downloads into the CWD.** Redirected to `~/.cache/voicegw/piper`.

**4. `.env` knobs silently ignored.** Two settings were read via `os.environ`
at import time, before `.env` was loaded. Both moved into `Settings`.

Offline also changes *failure mode*, which is the real benefit: a missing model
fails immediately with `voicegw fetch` instructions instead of hanging on a
connection attempt.

---

## 8. Reproducing

```bash
python scripts/vram_budget.py          # footprints + swap cost
python scripts/quant_compare.py        # bf16 vs int8 vs nf4
python scripts/tts_latency_profile.py  # steps vs chunk length
python scripts/turn_latency.py         # end-to-end (gateway must be running)
python scripts/ab_first_chunk.py       # quality A/B, needs your ears
python scripts/tune_tts.py             # sweep steady-state num_step
python scripts/bench.py                # per-engine RTF
python scripts/check_cancel.py         # barge-in does not strand the engine
```
