# voicegw

Arabic voice gateway: speech-to-text with
[Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
and text-to-speech with [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice),
exposed over an OpenAI-compatible REST API, a realtime WebSocket loop, and an
MCP server — so any agent or app can use it.

## Runtime profiles

One switch, `VOICEGW_PROFILE`, selects the engines. The API surface is identical
in both profiles.

| | `gpu` | `cpu` |
| --- | --- | --- |
| STT | `cohere-asr` (2B, bf16) | `faster-whisper` (int8) |
| TTS | `omnivoice` (fp16) | `omnivoice-cpu`, `piper` fallback |
| Realtime | yes | yes |

`VOICEGW_PROFILE=auto` (the default) probes CUDA and free VRAM, then resolves to
one of the two and logs the reason. `voicegw doctor` shows the decision.

Pin an individual engine with `VOICEGW_ASR_ENGINE` / `VOICEGW_TTS_ENGINE`. For
example `VOICEGW_ASR_ENGINE=cohere-asr-cpu` trades speed for accuracy on CPU;
that engine is marked non-realtime, so `/v1/realtime` refuses it while file
transcription keeps working.

### GPU residency

Both models together need more VRAM than a 6 GB laptop card has once the
desktop takes its share. When total VRAM is below
`VOICEGW_SHARED_RESIDENCY_MIN_VRAM_MIB` (default 10000), the core switches to
**exclusive** residency: one model holds the GPU at a time and a single lock
serializes STT and TTS so they swap cleanly. On a 6 GB RTX 4050 this took warm
STT from 3920 ms (both resident, thrashing) to 361 ms. Force it either way with
`VOICEGW_RESIDENCY=exclusive|shared|auto`; `GET /healthz` reports the mode, the
reason, and the current occupant.

Exclusive residency requires every GPU-resident engine to use the **same CUDA
allocator**. PyTorch keeps hundreds of MiB *reserved* after `.to("cpu")` —
`empty_cache()` cannot reclaim fragmented segments and `expandable_segments` is
unsupported on Windows — and that pool is invisible to CTranslate2. So pairing
`faster-whisper` (CTranslate2) with `omnivoice` (PyTorch) would OOM on swap. The
core detects the mix at startup and moves the CTranslate2 engine to the CPU,
logging the reason; an `onload` OOM at runtime triggers the same fallback rather
than failing the request. The default GPU pairing (`cohere-asr` + `omnivoice`)
is all-PyTorch and swaps without this penalty.

### Latency knobs

`VOICEGW_TTS_NUM_STEP` sets OmniVoice's diffusion steps and is the dominant TTS
cost — on a 4 s utterance: 4 steps = 207 ms, 8 = 403 ms, 16 = 747 ms (default),
32 = 1442 ms. Run `python scripts/tune_tts.py`, listen to `out/tts_steps/*.wav`
and pick the lowest step count you find acceptable.

## Setup

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

OmniVoice **weights are CC-BY-NC** — non-commercial use only. The code is
Apache-2.0.

## Fully local / offline

Everything runs on your machine; nothing is sent to a third-party service. To
also remove the *network* dependency:

```powershell
voicegw fetch          # once, with a connection — downloads every model
# then, in .env:
VOICEGW_OFFLINE=1
```

`fetch` reports what it downloaded, skips what is already cached, and exits
non-zero if anything is missing. With `VOICEGW_OFFLINE=1` the gateway loads
only from the local Hugging Face cache and a missing file fails **immediately**
with instructions, instead of hanging on a connection attempt.

`voicegw doctor` lists each model and whether it is cached.

What is already local without any extra steps:

| component | offline? |
| --- | --- |
| Silero VAD | ships inside the `silero-vad` package |
| espeak-ng data (Piper) | ships inside the `piper-tts` package |
| Web client | no CDN — all assets are bundled by Vite |
| Realtime agent | defaults to a local Ollama endpoint |

Two caveats worth knowing:

- The **Cohere ASR repo is gated**, so the very first `fetch` needs `HF_TOKEN`.
  After that it is cached like anything else.
- Piper stores its voices outside the Hugging Face cache, in
  `~/.cache/voicegw/piper`. `fetch` populates it. (The library otherwise
  downloads into the current working directory.)

## Usage

```powershell
voicegw doctor          # environment + profile check
voicegw fetch           # download all models for offline use
voicegw serve           # REST + WebSocket on :8000
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

It is OpenAI-compatible, so an existing client works unmodified:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
client.audio.transcriptions.create(model="cohere-asr", file=open("sample.wav", "rb"))
```

`GET /healthz` reports the active profile, engines, device and VRAM.

### When an engine fails to load

A model that won't load takes down only the half of the gateway that needs it —
if the ASR weights are gated but OmniVoice loaded, synthesis keeps working. The
core records the failure instead of pretending to be healthy:

- `GET /healthz` returns **503** with `"status": "degraded"` and an `errors`
  object naming the engine, the reason, and a remediation `hint`.
- The affected endpoints return **503** `engine_unavailable` carrying that hint,
  not an opaque 500.
- `/v1/audio/speech` checks before streaming starts, so a dead TTS can never
  produce an empty `200`.
- `/v1/realtime` reports `engine_unavailable` rather than blaming engine speed.
- The MCP `gateway_status` tool and the local CLI commands print the same hint.

### Realtime WebSocket

`ws://127.0.0.1:8000/v1/realtime` — send PCM16 mono @16 kHz binary frames,
receive JSON events (`speech_started`, `transcript.final`,
`response.text.delta`, `response.done`) plus PCM16 audio frames. Speaking over
the assistant cancels playback (barge-in).

The agent is pluggable: point `VOICEGW_AGENT_BASE_URL` at any OpenAI-compatible
chat endpoint.

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

- A VAD/noise gate always runs before the ASR model — its card warns it
  hallucinates text from silence.
- The model provides no timestamps and no speaker diarization.
- `python scripts/bench.py` measures RTF and latency per engine on your machine.
