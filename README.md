# voicegw

Arabic voice gateway: speech-to-text with
[Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
and text-to-speech with [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice),
exposed over an OpenAI-compatible REST API, a realtime WebSocket loop, and an
MCP server — so any agent or app can use it.

Runs fully local. Measured **792 ms** from end-of-speech to first audio on a
6 GB laptop GPU.

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — how the pieces fit and why
- **[docs/ALGORITHMS.md](docs/ALGORITHMS.md)** — the measurements behind every tuning decision
- **[docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md)** — step-by-step presentation & demo walkthrough with examples

## Quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# GPU profile
pip install -e ".[gpu,dev,mcp]" --extra-index-url https://download.pytorch.org/whl/cu124

# CPU profile
pip install -e ".[cpu,dev,mcp]" --extra-index-url https://download.pytorch.org/whl/cpu
```

The Cohere ASR repo is **gated**: accept its terms on the model page with your
Hugging Face account, then copy `.env.example` to `.env` and set `HF_TOKEN`.
It is needed only for the first download.

```powershell
voicegw doctor          # environment + profile check
voicegw fetch           # download all models for offline use
voicegw serve           # REST + WebSocket on :8000
```

OmniVoice **weights are CC-BY-NC** — non-commercial use only. The code is
Apache-2.0.

## Runtime profiles

One switch, `VOICEGW_PROFILE`, selects the engines. The API surface is identical
in both profiles.

| | `gpu` (Original Models on CUDA) | `cpu` (Optimized Models on CPU) |
| --- | --- | --- |
| STT | **Original `cohere-asr`** (2B, PyTorch on CUDA) | **Optimized `cohere-asr-cpu`** (bfloat16, bounded) / `faster-whisper` |
| TTS | **Original `omnivoice`** (fp16, PyTorch on CUDA, 16/8 steps) | **Optimized `omnivoice-cpu`** (4/2 steps) / `omnivoice-gguf` |
| Runtime | **Native PyTorch + CUDA** (No GGUF, no CPU fallbacks) | PyTorch CPU / GGUF / CTranslate2 |
| Realtime | **Yes (~792 ms sub-second)** | **Yes (~1.8s first audio with cloning, ~300 ms with Piper)** |
| Memory | **~3.3 GB VRAM** (GPU), ~1.5 GB Host RAM | **~5.3 GB Host RAM** (bfloat16 prevents 98% RAM paging) |

`auto` (the default) probes CUDA and free VRAM, resolves to one of the two, and
logs the reason. When it picks CPU it also clears `CUDA_VISIBLE_DEVICES`, so the
choice is binding rather than advisory.

### Launch Commands for the Two Modes

Pin individual engines with `--stt` / `--tts` CLI options or `VOICEGW_ASR_ENGINE` / `VOICEGW_TTS_ENGINE`.

#### Mode 1: Original Models on GPU (CUDA)
Runs the **100% original PyTorch models** (Cohere 2B ASR + OmniVoice FP16) on CUDA with shared VRAM residency (~792 ms sub-second turn latency):
```powershell
python -m voicegw.cli serve --profile gpu --stt cohere-asr --tts omnivoice --port 8005
```
*(Or simply `python -m voicegw.cli serve --profile gpu --port 8005`)*

#### Mode 2: Optimized CPU Mode (Cohere ASR + OmniVoice)
Runs the **optimized CPU pipeline** (Cohere ASR bfloat16 + OmniVoice 4-step diffusion) with multi-threading:
```powershell
python -m voicegw.cli serve --profile cpu --stt cohere-asr-cpu --tts omnivoice-cpu --port 8005
```

#### Alternative: Ultra-Lightweight Instant CPU Mode (~300 ms Turn)
Runs Faster-Whisper (300 MB) + Piper TTS (100 MB) for instant sub-350 ms turns on CPU:
```powershell
python -m voicegw.cli serve --profile cpu --stt faster-whisper --tts piper --port 8005
```

## Performance

| configuration | STT | TTS 1st | total |
| --- | --- | --- | --- |
| **GPU:** Cohere nf4 + OmniVoice fp16 (shared residency) | 326 ms | 466 ms | **792 ms** |
| GPU: bf16 ASR, exclusive residency (PCIe swapping) | 3683 ms | 5081 ms | 8764 ms |
| **CPU:** Cohere bfloat16 + OmniVoice 4-step (cached) | 1450 ms | 1850 ms | **3.3 s** |
| CPU: Cohere unquantized float32 + OmniVoice reload | 3900 ms | 8200 ms | 12.1 s |

RTX 4050 Laptop (6 GB), Arabic, `python scripts/turn_latency.py`. The two
changes that mattered:

**Both models must stay resident.** Swapping them across PCIe costs ~3.3 s per
turn. Quantizing ASR to nf4 cuts it from 3940 → 1413 MiB, which makes both fit
alongside TTS (1937 MiB) — for only ~100 ms of added inference.

**Only the first chunk of a reply is heard in silence.** TTS latency is ~linear
in diffusion steps (~48 ms/step) and nearly flat in text length below ~65
characters — so the lever is steps, not chunk size, and it is only worth pulling
on the chunk that gates playback.

Full measurements, including the `int8` trap and the A/B procedure for the
quality tradeoff, are in [docs/ALGORITHMS.md](docs/ALGORITHMS.md).

### Tuning knobs

| setting | default | effect |
| --- | --- | --- |
| `VOICEGW_ASR_QUANTIZATION` | `auto` | `none` \| `int8` \| `nf4`. `auto` quantizes only when both models would not otherwise fit. |
| `VOICEGW_RESIDENCY` | `auto` | `shared` \| `exclusive`. `auto` decides from declared footprints vs free VRAM. |
| `VOICEGW_VRAM_HEADROOM_MIB` | `700` | Reserved for activations and fragmentation. |
| `VOICEGW_TTS_NUM_STEP` | `16` | Diffusion steps on GPU — the dominant TTS cost. |
| `VOICEGW_TTS_FIRST_CHUNK_NUM_STEP` | `8` | Steps for the playback-gating GPU chunk only. |
| `VOICEGW_TTS_NUM_STEP_CPU` | `4` | Diffusion steps on CPU (cuts compute by 75%). |
| `VOICEGW_TTS_FIRST_CHUNK_NUM_STEP_CPU` | `2` | Steps for the playback-gating CPU chunk only. |
| `VOICEGW_CPU_DTYPE` | `bfloat16` | Precision for CPU PyTorch models (`bfloat16` cuts RAM in half vs `float32`). |
| `VOICEGW_CPU_THREADS` | `auto` | Number of CPU threads for PyTorch/CTranslate2. |

```bash
python scripts/ab_first_chunk.py   # hear the first-chunk tradeoff
python scripts/tune_tts.py         # sweep steady-state num_step
python scripts/quant_compare.py    # bf16 vs int8 vs nf4 on your card
```

`GET /healthz` reports the resolved profile, engines, residency mode, the
arithmetic behind it, and live VRAM.

## Fully local / offline

Everything runs on your machine; nothing is sent to a third-party service. To
also remove the *network* dependency:

```powershell
voicegw fetch          # once, with a connection — downloads every model
# then, in .env:
VOICEGW_OFFLINE=1
```

`fetch` reports what it downloaded, skips what is already cached, and exits
non-zero if anything is missing. With `VOICEGW_OFFLINE=1` the gateway loads only
from the local Hugging Face cache, and a missing file fails **immediately** with
instructions instead of hanging on a connection attempt.

`voicegw doctor` lists each model and whether it is cached.

Already local without extra steps:

| component | why |
| --- | --- |
| Silero VAD | ships inside the `silero-vad` package |
| espeak-ng data (Piper) | ships inside the `piper-tts` package |
| Web client | no CDN — all assets bundled by Vite |
| Realtime agent | defaults to a local Ollama endpoint |

Two caveats:

- The **Cohere ASR repo is gated**, so the first `fetch` needs `HF_TOKEN`.
- Piper stores voices outside the HF cache, in `~/.cache/voicegw/piper`.
  `fetch` populates it — the library otherwise downloads into the current
  working directory.

## Usage

```powershell
voicegw serve --profile cpu
voicegw transcribe sample.wav
voicegw say "مرحبا بك" --out out.wav
voicegw mcp             # MCP server over stdio (gateway must be running)
```

### REST

```powershell
curl.exe -F file=@sample.wav -F language=ar http://127.0.0.1:8000/v1/audio/transcriptions
curl.exe -X POST http://127.0.0.1:8000/v1/audio/speech `
  -H "Content-Type: application/json" `
  -d '{\"input\":\"مرحبا\",\"voice\":\"default\"}' -o out.wav
```

OpenAI-compatible, so an existing client works unmodified:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
client.audio.transcriptions.create(model="cohere-asr", file=open("sample.wav", "rb"))
```

### Realtime WebSocket

`ws://127.0.0.1:8000/v1/realtime` — send PCM16 mono @16 kHz binary frames,
receive JSON events (`speech_started`, `transcript.final`,
`response.text.delta`, `response.audio.start`, `response.done`) plus PCM16 audio
frames. Speaking over the assistant cancels playback (barge-in).

Control frames: `{"type": "commit"}` (push-to-talk release),
`{"type": "cancel"}`, `{"type": "text", "text": "…"}` (skip STT),
`{"type": "config", "voice": "…", "language": "ar", "agent": true}`.

The agent is pluggable: point `VOICEGW_AGENT_BASE_URL` at any OpenAI-compatible
chat endpoint. Setting `"agent": false` on a session bypasses it entirely and
speaks the input back verbatim — which exercises STT, TTS, streaming and
barge-in without an LLM running. `session.created` reports `agent_available`
so a client can show which mode is actually possible.

### Web client

```powershell
cd web; npm install; npm run dev     # http://localhost:5173
```

**Connect** first — nothing works before the socket is open. Then either hold
**Hold to talk**, or tick **Open mic (VAD)** and just speak. The **Speak** box
synthesizes typed text directly, with no microphone and no STT. Untick **Send
to agent** to test the audio path with no LLM at all.

The HUD reports per-turn `ASR`, `First audio` and `Total` from server events,
so the latency you see is measured on your machine rather than quoted here.

### MCP

```json
{
  "servers": {
    "voicegw": {
      "command": "voicegw",
      "args": ["mcp"],
      "env": { "VOICEGW_URL": "http://127.0.0.1:8000" }
    }
  }
}
```

Tools: `transcribe_audio`, `synthesize_speech`, `list_voices`, `gateway_status`.
The MCP server holds no models — it calls the running gateway over localhost.

## When an engine fails to load

A model that won't load takes down only the half of the gateway that needs it —
if the ASR weights are gated but OmniVoice loaded, synthesis keeps working. The
core records the failure instead of pretending to be healthy:

- `GET /healthz` returns **503** with `"status": "degraded"` and an `errors`
  object naming the engine, the reason, and a remediation `hint`.
- Affected endpoints return **503** `engine_unavailable` carrying that hint, not
  an opaque 500.
- `/v1/audio/speech` checks *before* streaming starts, so a dead TTS can never
  produce an empty `200`.
- `/v1/realtime` reports `engine_unavailable` rather than blaming engine speed.
- The MCP `gateway_status` tool and the CLI print the same hint.

Hints are matched from the failure text — gated repo, OOM, missing dependency,
offline cache miss, network — because the usual cause is fixable in under a
minute if we actually say what to do.

## Voices

`voices/manifest.json` maps voice ids to either a cloning reference
(`ref_audio` + `ref_text`, 3–15 s of clean speech) or a voice-design
`description`. Clone voices require OmniVoice; requests for them are routed away
from Piper automatically.

Design descriptions are **not free-form prose**. OmniVoice accepts a fixed
vocabulary (gender, age band, pitch, speed, emotion, …) such as
`"male, young adult, moderate pitch"`; anything outside it raises at synthesis
time, so the manifest is validated at load.

## Notes

- **There is no authentication.** The gateway binds to `127.0.0.1` by default
  and is meant to stay there; `serve` warns if you bind it anywhere else.
  Anyone who can reach the port can use your GPU. Put a reverse proxy in front
  of it before exposing it.
- A VAD/noise gate always runs before the ASR model — its card warns it
  hallucinates text from silence, and it does: with the gate off, digital
  silence transcribes as `"@@@فراغ"`.
- The ASR model provides no timestamps and no speaker diarization.
- `python scripts/bench.py` measures RTF and latency per engine on your machine.

## Development

```powershell
python -m pytest -q                        # 196 tests, no GPU or network needed
python -m ruff check src tests scripts
python -m ruff format src tests scripts
```

Adding an engine means writing one class against the `SttEngine` / `TtsEngine`
Protocol and adding one line to `engines/registry.py`. No façade changes — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
