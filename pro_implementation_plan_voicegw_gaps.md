# Professional Implementation Plan: Closing the Feature & Quality Gaps in `voicegw`

## Executive Objective

Transform `voicegw` from an ultra-fast conversational prototype into an enterprise-grade Arabic voice gateway. We will integrate VoiceStudio's proven audio post-processing, lexical normalization, GGUF compute efficiency, and OpenAI protocol parity—**while strictly preserving our 792 ms sub-second turn latency and studio-grade voice quality**.

---

## Architecture Overview of Changes

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                     API FAÇADE LAYER                                    │
│   app.py: Multi-format encoder (mp3/opus/aac/flac), OpenAI 400 error envelope,          │
│           verbose_json word timestamps, /v1/audio/voices metadata                       │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                       VOICE CORE                                        │
│   core.py: Audio chunk edge-silence trimming (-40 dBFS), 50ms smooth crossfading,       │
│            SSML-lite pause generator, Arabic pronunciation pre-filter                   │
└──────────────────────┬───────────────────────────────────────────────────┬──────────────┘
                       │                                                   │
                       ▼                                                   ▼
┌──────────────────────────────────────────────┐  ┌────────────────────────────────────────┐
│             NEW COMMON UTILITIES             │  │            ENGINE ADAPTERS             │
│  - audio.py: trim_edge_silence, crossfade,   │  │  - voices.py: Audio QC (SNR, clips)   │
│              encode_mp3, encode_opus,        │  │  - prosody.py: Pure-NumPy extractor   │
│              encode_flac                     │  │  - omnivoice_gguf.py: C++ GGUF engine │
│  - text_chunker.py: [pause 300ms], [slow]    │  │    (Q8_0 default, Q4_K_M low-RAM)     │
│  - lexicon.py: ReDoS-safe dialect dictionary │  │                                        │
└──────────────────────────────────────────────┘  └────────────────────────────────────────┘
```

---

## Phase 1: Audio Quality & Ingestion Polishing (Immediate High-Value Wins)

### Task 1.1: Audio Edge-Silence Trimming & Chunk Crossfading
* **Problem:** Neural TTS models add ~50 ms of lead-in and ~200 ms of trailing decay. In batch synthesis or streaming playback, concatenated chunks have audible micro-silences and phase-cancelling clicks.
* **Target Files:**
  - [src/voicegw/common/audio.py](file:///d:/projects/stttts/src/voicegw/common/audio.py)
  - [src/voicegw/core.py](file:///d:/projects/stttts/src/voicegw/core.py)
* **Implementation Details:**
  1. Add `trim_edge_silence(audio: np.ndarray, sample_rate: int, threshold_db: float = -40.0, keep_ms: int = 35) -> np.ndarray` in `audio.py`.
     - Scans samples to find indices where level exceeds $10^{\text{threshold\_db}/20}$.
     - Preserves `keep_ms` (default 35 ms) on each side to avoid clipping onset/decay consonants.
  2. Add `crossfade(chunk_a: np.ndarray, chunk_b: np.ndarray, crossfade_samples: int) -> np.ndarray` in `audio.py`.
     - Equal-power or linear crossfade between consecutive audio frames.
  3. Update `VoiceCore.synthesize()` and non-streaming `/v1/audio/speech`:
     - Trim each rendered chunk before streaming.
     - When building full audio for non-streaming callers, join chunks using `crossfade()` instead of raw `np.concatenate`.
* **Testing:**
  - Create `tests/test_audio_polish.py` testing pure silence, impulse signals, edge trimming bounds, and crossfade energy continuity.

---

### Task 1.2: Reference Audio Quality Control (QC)
* **Problem:** Cloned voice quality depends on the reference clip. If a user supplies a quiet, clipped, or reverberant 3-second recording, the clone sounds muffled or distorted.
* **Target Files:**
  - [src/voicegw/engines/voices.py](file:///d:/projects/stttts/src/voicegw/engines/voices.py)
  - `src/voicegw/common/audio_qc.py` (New)
* **Implementation Details:**
  1. Create `AudioQualityReport` dataclass in `audio_qc.py`:
     ```python
     @dataclass
     class AudioQualityReport:
         duration_s: float
         rms_dbfs: float
         peak_dbfs: float
         clipped_samples: int
         speech_duration_s: float
         is_valid: bool
         warnings: list[str]
     ```
  2. Implement `analyze_reference_audio(audio: np.ndarray, sr: int, vad_model=None) -> AudioQualityReport`:
     - **Clipping Detection:** Counts samples where $|x| \ge 0.999$. Flag if $> 0.1\%$ of samples clip.
     - **Dynamic Range / RMS:** Warn if RMS level is $<-32\text{ dBFS}$ (too quiet) or $>-3\text{ dBFS}$ (dangerously hot).
     - **VAD Speech Density:** Run Silero VAD to verify that speech occupies at least 2.5 seconds of the clip (preventing 10s of background noise with 1s of speech).
  3. Integrate into `VoiceRegistry.validate_reference()`:
     - Log warnings during startup; return structured diagnostics via `/v1/voices` and CLI `voicegw doctor`.
* **Testing:**
  - Test with synthetic clipped audio, silent audio, and clean speech samples.

---

### Task 1.3: SSML-Lite & Inline Pause Markers
* **Problem:** Text containing `[pause 500ms]` or conversational pauses currently gets fed directly to the TTS model, which hallucinates or speaks the word "pause".
* **Target Files:**
  - [src/voicegw/common/text_chunker.py](file:///d:/projects/stttts/src/voicegw/common/text_chunker.py)
  - [src/voicegw/core.py](file:///d:/projects/stttts/src/voicegw/core.py)
* **Implementation Details:**
  1. In `text_chunker.py`, implement an inline tag parser:
     ```python
     PAUSE_REGEX = re.compile(r"\[(?:pause|silence)\s+(\d+)(ms|s)\]", re.IGNORECASE)
     ```
  2. Define chunk token types:
     ```python
     @dataclass
     class TextChunk:
         text: str
         speed_multiplier: float = 1.0

     @dataclass
     class PauseChunk:
         duration_ms: int
     ```
  3. When `VoiceCore.synthesize()` encounters a `PauseChunk`:
     - Instantly yields an `AudioChunk` containing pure zeros for `duration_ms` at the engine's sample rate (0 ms inference cost, perfect silence!).
  4. Parse `[slow]...[/slow]` (0.85x) and `[fast]...[/fast]` (1.15x) tags, adjusting generation rate.
* **Testing:**
  - Verify that `"[pause 500ms]"` produces exactly 12,000 samples of zero audio at 24 kHz.

---

### Task 1.4: Arabic Pronunciation & Dialect Respelling Lexicon
* **Problem:** Numbers, acronyms, foreign brand names, and dialectal words often mispronounce without diacritics.
* **Target Files:**
  - `src/voicegw/common/lexicon.py` (New)
  - `voices/lexicon.json` (New default rules)
* **Implementation Details:**
  1. Build a ReDoS-safe dictionary replacement engine in `lexicon.py`:
     - Sort keys by length (longest match first).
     - Construct a single compiled regex with word boundary guards (`\b`).
     - Safe against recursive substitution.
  2. Populate default `voices/lexicon.json`:
     - Arabic numerals expansion hints.
     - Tech terms: `"AI" -> "إيه آي"`, `"API" -> "إيه بي آي"`, `"GPT" -> "جي بي تي"`.
     - Standard MSA vocalization helpers.
  3. Apply in `VoiceCore.synthesize()` before `chunk_text()`.
* **Testing:**
  - Test with overlapping keys, Latin/Arabic mixed strings, and boundary edge cases.

---

## Phase 2: OmniVoice GGUF / GGML Integration (VRAM & CPU Breakthrough)

### Strategic Guardrail: Quality First
* As confirmed by VoiceStudio's research, `Q4_K_M` degrades audio quality into a robotic timbre.
* **Policy:**
  - **GPU Mode:** Primary GPU default remains **PyTorch FP16** (or **GGUF Q8_0** for 4GB GPUs).
  - **CPU Mode:** Default to **GGUF Q8_0** via C++ GGML (4x–8x faster than PyTorch CPU).
  - `Q4_K_M` is strictly an opt-in fallback for ultra-low RAM devices.

### Task 2.1: GGUF Model & Binary Resolution
* **Target Files:**
  - `src/voicegw/engines/omnivoice_gguf.py` (New)
  - [src/voicegw/engines/registry.py](file:///d:/projects/stttts/src/voicegw/engines/registry.py)
  - [src/voicegw/common/config.py](file:///d:/projects/stttts/src/voicegw/common/config.py)
* **Implementation Details:**
  1. Download and pin `Serveurperso/OmniVoice-GGUF` model artifacts:
     - Base: `omnivoice-base-Q8_0.gguf` (~656 MB)
     - Codec/Tokenizer: `omnivoice-tokenizer-Q8_0.gguf` (~289 MB)
  2. Create `OmniVoiceGgufEngine` implementing `TtsEngine` Protocol:
     - Wrap the native `omnivoice-tts` binary or GGML C-API binding.
     - Declare `EngineInfo(vram_mib=945, allocator="none" if cpu else "cuda")`.
  3. Register in `registry.py`:
     - `PROFILE_DEFAULTS[Profile.CPU]["tts"] = "omnivoice-gguf"` (elevating CPU performance above Piper).
* **Testing:**
  - Verify synthesize round-trip against a synthetic voice profile and measure RTF on CPU.

---

## Phase 3: Protocol Hardening & OpenAI Feature Parity

### Task 3.1: Multi-Format Audio Output Encoding
* **Problem:** Official OpenAI clients (like OpenAI Python SDK, Pipecat, or mobile apps) often request `mp3`, `opus`, or `aac`. `voicegw` currently returns 400 for anything other than `wav` or `pcm`.
* **Target Files:**
  - [src/voicegw/common/audio.py](file:///d:/projects/stttts/src/voicegw/common/audio.py)
  - [src/voicegw/api/app.py](file:///d:/projects/stttts/src/voicegw/api/app.py)
* **Implementation Details:**
  1. Implement in `audio.py`:
     - `encode_mp3(audio: np.ndarray, sr: int, bitrate="128k") -> bytes`
     - `encode_opus(audio: np.ndarray, sr: int) -> bytes`
     - `encode_flac(audio: np.ndarray, sr: int) -> bytes`
     - Uses `soundfile` (for FLAC/OGG) with fallback to `ffmpeg` via streaming subprocess pipe.
  2. Update `SpeechRequest.response_format`:
     - Allow `"mp3"`, `"opus"`, `"aac"`, `"flac"`, `"wav"`, `"pcm"`.
     - Map `CONTENT_TYPES` headers accordingly (`audio/mpeg`, `audio/ogg`, etc.).
* **Testing:**
  - Add API tests fetching `/v1/audio/speech` with each `response_format` and validating magic byte headers.

---

### Task 3.2: OpenAI Standardized Error Envelope
* **Problem:** FastAPI default validation returns HTTP 422 with a FastAPI-specific body. The official `openai` Python SDK expects HTTP 400 with `{"error": {"message": "...", "type": "...", "code": "..."}}`.
* **Target Files:**
  - [src/voicegw/api/app.py](file:///d:/projects/stttts/src/voicegw/api/app.py)
* **Implementation Details:**
  1. Register custom exception handler for `RequestValidationError`:
     ```python
     @app.exception_handler(RequestValidationError)
     async def validation_exception_handler(request: Request, exc: RequestValidationError):
         first_err = exc.errors()[0]
         param = ".".join(str(loc) for loc in first_err.get("loc", []))
         return JSONResponse(
             status_code=400,
             content={
                 "error": {
                     "message": first_err.get("msg", "Invalid request"),
                     "type": "invalid_request_error",
                     "param": param,
                     "code": first_err.get("type", None),
                 }
             },
         )
     ```
* **Testing:**
  - Verify sending invalid parameters yields HTTP 400 and parses cleanly via the official `openai` SDK.

---

### Task 3.3: Word-Level Timestamps in Transcription (`verbose_json`)
* **Target Files:**
  - [src/voicegw/api/app.py](file:///d:/projects/stttts/src/voicegw/api/app.py)
  - [src/voicegw/engines/faster_whisper_asr.py](file:///d:/projects/stttts/src/voicegw/engines/faster_whisper_asr.py)
* **Implementation Details:**
  1. When `response_format="verbose_json"`, return OpenAI segments and word arrays:
     ```json
     {
       "task": "transcribe",
       "language": "ar",
       "duration": 4.5,
       "text": "...",
       "words": [{"word": "مرحبا", "start": 0.0, "end": 0.6}]
     }
     ```
  2. Implement word timestamp extraction in `faster_whisper_asr.py` via `word_timestamps=True`.
* **Testing:**
  - Test `/v1/audio/transcriptions` with `response_format=verbose_json`.

---

## Phase 4: Directorial Intelligence & Web Studio

### Task 4.1: Pure-NumPy Acoustic Prosody Mirror
* **Problem:** Writing OmniVoice voice-design descriptors (`"female, young adult, moderate pitch"`) requires manual guesswork.
* **Target Files:**
  - `src/voicegw/engines/prosody.py` (New)
* **Implementation Details:**
  1. Pure-NumPy signal analysis module (adapted from VoiceStudio, zero heavy dependencies):
     - Calculate $F_0$ pitch via autocorrelation / harmonic peak extraction.
     - Calculate RMS energy (loudness in dBFS).
     - Calculate speaking rate via envelope onset frequency (syllables/sec).
     - Calculate voicing ratio (fraction of voiced frames).
  2. Map metrics to OmniVoice vocabulary:
     - Pitch: $< 115\text{ Hz} \to \text{"low pitch"}$, $115\text{–}180\text{ Hz} \to \text{"moderate pitch"}$, $> 180\text{ Hz} \to \text{"high pitch"}$.
     - Gender heuristic from fundamental frequency.
     - Energy: low energy + low voicing $\to \text{"whisper"}$.
  3. Expose via CLI: `voicegw describe sample.wav` and API `POST /v1/voices/describe`.
* **Testing:**
  - Test synthetic sine waves and known voice clips to verify accurate categorization.

---

### Task 4.2: Web Recording Studio & Voice Cloner
* **Target Files:**
  - [web/src/main.ts](file:///d:/projects/stttts/web/src/main.ts)
  - [web/src/style.css](file:///d:/projects/stttts/web/src/style.css)
  - [web/index.html](file:///d:/projects/stttts/web/index.html)
* **Implementation Details:**
  1. Add a **Voice Studio** tab to the Vite frontend:
     - Live audio visualizer (canvas waveform / RMS bar).
     - "Record Reference" button: captures 5–10 seconds of microphone input.
     - Runs instant client-side Audio QC (clipping, volume check).
     - Plays back the recording and prompts for reference transcript.
     - Saves to backend via `POST /v1/voices` directly into `voices/manifest.json`.
* **Testing:**
  - Verify end-to-end recording in browser, manifest update, and immediate synthesis using the newly cloned voice.

---

## Execution Matrix & Verification Schedule

| Step | Scope | Verification Command | Gate / Criteria |
| :--- | :--- | :--- | :--- |
| **1.1** | Edge trimming & Crossfading | `pytest tests/test_audio_polish.py` | Zero pops, clean 50 ms crossfade, no clipped consonants |
| **1.2** | Reference Audio QC | `pytest tests/test_voices.py` | Detects clipping & low RMS; flags bad voice clips on load |
| **1.3** | SSML-Lite Pause Markers | `pytest tests/test_text_chunker.py` | `[pause 500ms]` generates exact zero-samples; latency unchanged |
| **1.4** | Arabic Lexicon | `pytest tests/test_lexicon.py` | Correct word substitutions without ReDoS or recursive loops |
| **2.1** | OmniVoice GGUF Engine | `pytest tests/test_gguf.py` | Loads Q8_0 weights, synthesizes on CPU without PyTorch VRAM |
| **3.1** | Multi-Format Encoding | `pytest tests/test_api.py` | Valid MP3, Opus, FLAC bytes with correct HTTP headers |
| **3.2** | OpenAI Error Envelope | `pytest tests/test_api.py` | HTTP 400 status with standard OpenAI error dictionary |
| **4.1** | Prosody Mirroring | `pytest tests/test_prosody.py` | Correctly predicts pitch and descriptor tags on reference clips |
| **4.2** | Web Studio | Browser verification | In-browser recording, QC check, voice registered and speakable |

---

## Ready for Execution

With this plan established:
- **Phase 1** can be executed immediately as a non-breaking, pure-Python enhancement that directly elevates our audio quality and user control.
- Each phase is completely modular and regression-tested against our 191 existing tests.
