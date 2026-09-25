# Architecture

## The problem this shape solves

A voice assistant needs two large models (~2.6 B parameters combined) and three
different callers: a browser, an HTTP client, and an MCP agent. The naive
design — one process per façade — loads the models three times and OOMs a
laptop GPU before it answers anything.

So: **one process, one copy of each model, three thin façades.**

```
                        ┌──────────────────────────────────┐
  browser / curl ──────▶│ REST        /v1/audio/*          │
                        │             /healthz  /v1/voices │
  browser (mic) ───────▶│ WebSocket   /v1/realtime         │──┐
                        │                                  │  │
  MCP agent ───────────▶│ (separate stdio process, calls   │  │
                        │  the REST API over localhost)    │  │
                        └──────────────────────────────────┘  │
                                                              ▼
                                              ┌───────────────────────────┐
                                              │        VoiceCore          │
                                              │  warm singletons + locks  │
                                              │  residency + VAD + voices │
                                              └───────────────────────────┘
                                                      │            │
                                              ┌───────▼───┐  ┌─────▼──────┐
                                              │ SttEngine │  │ TtsEngine  │
                                              └───────────┘  └────────────┘
```

The MCP server is the one deliberate exception: it runs as its own stdio
process because that is what MCP clients spawn, and it talks to the gateway
over localhost HTTP. It holds no models.

## Layers

| Layer | Module | Responsibility |
| --- | --- | --- |
| Façades | `api/app.py`, `realtime/session.py`, `mcp/server.py`, `cli.py` | Protocol translation only. No model code, no device logic. |
| Core | `core.py` | Owns the engines, the locks, GPU residency, VAD gating, voice resolution, health. |
| Registry | `engines/registry.py` | The **only** module that imports concrete engine classes. |
| Engines | `engines/*.py` | One class per model. Implement `SttEngine` / `TtsEngine`. |
| Contracts | `common/protocols.py` | `Transcript`, `AudioChunk`, `Voice`, `EngineInfo`, and the two Protocols. |
| Config | `common/config.py` | `Settings`, profile resolution, residency resolution. |

The layering rule that keeps this honest: **nothing above the registry may
import an engine module.** Adding a model means writing one class and adding
one line to a dict; no façade changes.

## Why profile resolution happens exactly once

`resolve_profile()` runs at startup and returns a frozen `ProfileResolution`
(profile, reason, device). Everything downstream reads that object rather than
re-probing CUDA.

This matters because the CPU path must be *binding*. When the profile resolves
to CPU, `CUDA_VISIBLE_DEVICES` is set to empty — so a stray `.to("cuda")`
anywhere in the codebase fails loudly instead of silently allocating VRAM the
user asked us not to touch.

The resolution carries its `reason` string through to `/healthz` and
`voicegw doctor`, because "why did it pick CPU?" is the first question anyone
asks when it's slow.

## Concurrency model

Two `asyncio.Lock`s, one per model. Inference itself runs in a worker thread
(`asyncio.to_thread`) so the event loop keeps serving other requests during a
multi-second synthesis.

Under **exclusive** residency the two locks are deliberately collapsed into
one:

```python
self._tts_lock = self._stt_lock
```

That single line is what guarantees a model is never swapped out from under an
in-flight inference. STT and TTS alternate within a turn anyway, so serializing
them costs nothing that the GPU wasn't already forcing.

Streaming synthesis bridges the thread boundary with a bounded
`asyncio.Queue(maxsize=8)`. Bounded matters: an unbounded queue lets a fast
producer buffer an entire reply's audio into RAM while a slow client dribbles
it out.

## GPU residency

The core problem on a 6 GB laptop card: full-precision ASR (3940 MiB) plus TTS
(1937 MiB) exceeds free VRAM (~5080 MiB). Two ways out, and the core picks
between them arithmetically.

**Shared** — both models stay resident. Requires
`sum(engine.info.vram_mib) + headroom <= free VRAM`.

**Exclusive** — one model holds the GPU; the other parks in system RAM via
`.to("cpu")`. Correct, but measured at **~3.3 s of PCIe traffic per turn**
(ASR swap 2203 ms + TTS swap 1062 ms) — which is why the real fix was to shrink
the model instead (see ALGORITHMS.md).

Engines declare `vram_mib` in `EngineInfo`; if *any* engine declares `0`
("unknown"), the core refuses to act on a half-known total and falls back to a
coarse total-VRAM threshold.

### The allocator trap

Exclusive residency assumes memory freed by one engine is visible to the next.
**That is false across allocator families.** PyTorch's caching allocator keeps
hundreds of MiB *reserved* after `.to("cpu")` — `empty_cache()` cannot reclaim
fragmented segments, and `expandable_segments` is unsupported on Windows — and
that pool is invisible to CTranslate2.

So `EngineInfo.allocator` is `"torch"`, `"ct2"`, or `"none"`. If two families
would share the GPU, `_resolve_allocator_conflict()` demotes the CTranslate2
engine to CPU at startup with a logged reason. An `onload` OOM at runtime
triggers the same demotion rather than failing the request.

## Degraded operation

A model that fails to load takes down **only the half that needs it**. Gated
ASR weights with no token shouldn't stop synthesis from working.

`_warmup()` records failures as `EngineUnavailable` rather than raising. That
exception carries `engine_id`, `reason`, and a **remediation hint** matched
from the failure text by `common/errors.py`:

| Failure signature | Hint |
| --- | --- |
| gated repo, 401 | Accept the terms on the model page, set `HF_TOKEN`. |
| out of memory | Try `VOICEGW_RESIDENCY=exclusive` or `VOICEGW_PROFILE=cpu`. |
| offline cache miss | Run `voicegw fetch` once online. |
| connection, timeout | Network or pre-fetch the weights. |

Hint order matters: an offline cache miss *also* mentions disabled outgoing
traffic, so it is matched **before** the generic network hint.

Every façade surfaces the same hint — REST as a 503 `engine_unavailable`,
WebSocket as an `error` event, MCP as text the agent can relay to the user.
`/healthz` returns 503 with `"status": "degraded"`.

One ordering detail in `/v1/audio/speech`: the TTS health check runs **before**
the `StreamingResponse` starts. Once a 200 and the first bytes are out, there
is no way to report a failure — so a dead engine can never produce an empty
`200`.

## Request paths

**Batch STT** — decode (thread) → VAD gate → claim GPU → transcribe (thread).
If the gate returns empty, the answer is `""`; the model is never asked to
transcribe silence.

**Batch TTS** — resolve voice → pick engine → chunk → synthesize per chunk,
streaming each as it completes.

**Realtime turn** — PCM frames → `UtteranceDetector` → on utterance end: STT →
agent (streaming tokens) → `StreamingChunker` → TTS per chunk → PCM16 frames
out. Speech detected while the assistant is talking cancels the response task
(barge-in).

## Engine contract

```python
class SttEngine(Protocol):
    info: EngineInfo
    def load(self) -> None: ...          # idempotent
    def transcribe(self, audio, language="ar") -> Transcript: ...
    def unload(self) -> None: ...

class TtsEngine(Protocol):
    info: EngineInfo
    def load(self) -> None: ...
    def synthesize(self, text, voice, urgent=False) -> Iterator[AudioChunk]: ...
    def supports(self, voice) -> bool: ...
    def unload(self) -> None: ...
```

Optional, discovered via `hasattr`: `offload()` / `onload()` for residency
swapping, `demote_to_cpu(reason)` for graceful GPU failure.

`urgent=True` marks the chunk that gates the start of playback — the listener
is sitting in silence waiting for it, so an engine may trade fidelity for
latency there. Engines that don't care ignore the flag.

`supports(voice)` drives fallback routing: Piper can't clone, so a clone voice
is routed to an OmniVoice alternate automatically.

## Configuration

Pydantic `BaseSettings`, prefix `VOICEGW_`, reads `.env`.

One hard-won rule: **every tunable must live in `Settings`.** Two knobs were
once read via `os.environ` at import time, so `.env` values were silently
ignored — the file was read *after* the module had already snapshotted the
environment. Same class of bug as `HF_HUB_OFFLINE`, which `huggingface_hub`
snapshots at import; `common/offline.py` therefore patches
`huggingface_hub.constants.HF_HUB_OFFLINE` directly rather than trusting the
env var.

## Testing strategy

178 tests, no GPU or network required. Engines are hidden behind Protocols, so
fakes substitute cleanly and the core's logic — residency arithmetic, allocator
conflict, degraded reporting, chunker invariants, urgency routing — is tested
without loading 2.6 B parameters.

What tests *cannot* cover, and is verified by script instead:

| Question | Script |
| --- | --- |
| Real VRAM footprints and swap cost | `scripts/vram_budget.py` |
| Quantization quality/latency tradeoff | `scripts/quant_compare.py` |
| End-to-end turn latency | `scripts/turn_latency.py` |
| TTS latency decomposition | `scripts/tts_latency_profile.py` |
| First-chunk quality (needs ears) | `scripts/ab_first_chunk.py` |
| Per-engine RTF on your machine | `scripts/bench.py` |
