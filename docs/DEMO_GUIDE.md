# voicegw Live Demonstration Guide

This guide provides a structured, step-by-step walkthrough to present **voicegw** to stakeholders, technical teams, or clients. It showcases how `voicegw` solves the latency and locality challenges of Arabic Conversational AI on consumer hardware.

---

## ⏱️ At-A-Glance: 5-Minute Demo Flow

| Phase | Duration | Feature Highlighted | Wow Factor |
| :--- | :---: | :--- | :--- |
| **1. Startup & Health** | 45s | GPU Shared Residency & VRAM Budget | Runs Cohere 2B + OmniVoice on a 6 GB laptop GPU |
| **2. Live Conversation** | 90s | Sub-Second Turn Latency & HUD | **792 ms** end-of-speech to first audio playback |
| **3. Studio & Cloning** | 90s | Audio QC, Prosody Mirror & Voice Cloning | Zero-shot cloning with real-time acoustic analysis |
| **4. Speech Polish** | 45s | SSML-Lite (`[pause]`, `[speed]`) & Lexicon | Natural phrasing & phonetic acronyms (`AI`, `GPU`) |
| **5. Developer Ecosystem** | 30s | OpenAI Drop-In & MCP Server | Seamless plug-and-play with any AI ecosystem |

---

## 🚀 Pre-Demo Checklist

1. **Activate Environment & Start Server:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   voicegw serve
   ```
2. **Open Browser:**
   Navigate to [http://localhost:8000/](http://localhost:8000/).
3. **Verify Audio Input/Output:**
   Ensure your microphone and speakers are unmuted.
4. **Bilingual Check:**
   Notice the header language switch (`🌐 English` / `🌐 العربية`) at the top right/left.

---

## 🎬 Step-by-Step Presentation Script

### Step 1: Pre-Flight & System Architecture Check (30 seconds)
**Talking Point:**
> *"Most voice assistants take 4 to 8 seconds to reply because they send audio to cloud APIs. `voicegw` is 100% local, privacy-first, and runs both a 2-Billion parameter Arabic ASR and a Zero-Shot Diffusion TTS concurrently in just 3.3 GB of VRAM."*

1. Point to the top status badges:
   * **`gpu · cuda:0`** (Green badge confirming hardware acceleration).
   * **`غير متصل / disconnected`** (Clean connection state).
2. Show the `/healthz` endpoint in a separate tab or terminal:
   ```bash
   curl http://127.0.0.1:8000/healthz
   ```
   *Show the audience that both models stay resident in shared VRAM (Cohere nf4 ~1.4 GB + OmniVoice fp16 ~1.9 GB), completely eliminating PCIe swapping.*

---

### Step 2: Live Conversation & The 792 ms Sub-Second Turn (90 seconds)

1. Click **`اتصال / Connect`**. The status turns to green: **`متصل / connected`**.
2. **Demo 2.1: Instant Text Synthesis (Zero LLM Noise)**
   * Enter in the **Speak** input:
     ```text
     أهلاً بك! نظام voicegw يقدم استجابة صوتية فورية فائقة السرعة.
     ```
   * Click **`نطق / Speak`**.
   * *Highlight:* Audio starts streaming almost instantaneously with natural Arabic intonation.
3. **Demo 2.2: Live Spoken Turn with HUD Latency**
   * Keep **`إرسال إلى الوكيل / Send to agent`** **unchecked** (to demonstrate raw audio pipeline latency without LLM jitter).
   * Click and hold **`اضغط وتحدث / Hold to talk`** (or tick **`الميكروفون التلقائي (VAD)`**).
   * Speak clearly:
     ```text
     "ما هي أحدث التقنيات المستخدمة في الذكاء الاصطناعي؟"
     ```
   * Release the button and watch the **HUD metrics** update:
     * **ASR:** ~326 ms
     * **First Audio:** ~466 ms
     * **Total Turn Latency:** **~792 ms (Sub-second!)**
4. **Demo 2.3: Live Barge-In (Interruption)**
   * Type a longer sentence into the input and hit Speak.
   * While the assistant is actively talking, press **`اضغط وتحدث / Hold to talk`** or speak into the microphone.
   * *Highlight:* The assistant immediately cuts off its speech, flushes the playback buffer, and shifts back to listening without stutter.

---

### Step 3: Voice Studio — Cloning, Audio QC & Prosody (90 seconds)

1. Click the **`استوديو استنساخ الصوت / Voice Studio`** tab.
2. **Demo 3.1: Reference Recording & Audio QC**
   * Click **`بدء التسجيل / Start Recording`** and speak a sample for ~5 seconds:
     ```text
     "إن التعلم الآلي والأنظمة الصوتية المتقدمة تمثل مستقبل التفاعل بين الإنسان والآلة."
     ```
   * Click **`إيقاف التسجيل / Stop Recording`**.
   * Point to the **Audio QC** card:
     * **RMS Signal Level:** Shows whether volume is optimal (-35 to -14 dBFS).
     * **Digital Clipping:** Confirms zero saturation ($|x| < 0.999$).
     * **Speech Duration:** Confirms 3-10 second sweet spot.
3. **Demo 3.2: Prosody Mirror Diagnostics**
   * Point to the **Prosody Mirror** card:
     * Shows Fundamental Frequency ($F_0$), Voicing Ratio, and Speech Rate.
     * Shows auto-extracted tags (e.g. `[pitch-moderate]`, `[rate-moderate]`).
4. **Demo 3.3: Registering & Testing the New Voice**
   * Fill out the form:
     * **Voice ID:** `presenter_clone`
     * **Voice Name:** `صوت تجريبي مخصص`
     * **Ref Text:** Paste the exact sentence you spoke.
   * Click **`حفظ وتسجيل الصوت / Register Voice`**.
   * Under **`اختبار النطق الفوري / Instant Voice Test`**, enter:
     ```text
     "أنا سعيد جداً بالتحدث بهذا الصوت المستنسخ الجديد!"
     ```
   * Click **`نطق / Speak`** and play the generated audio!

---

### Step 4: Speech Polish — SSML-Lite & Pronunciation Lexicon (45 seconds)

Switch back to the **`المحادثة المباشرة / Live Conversation`** tab or use the Instant Test box to show advanced linguistic polish:

1. **SSML-Lite Pause Injection:**
   ```text
   نحن سعداء بوجودكم معنا اليوم. [pause 700ms] والآن سننتقل إلى الخطوة التالية.
   ```
   *Audience hears an exact, natural 700 ms pause between sentences without awkward artifacts.*
2. **SSML-Lite Speed Modulation:**
   ```text
   [slow] هذا كلام يتم نطقه بتمهل ووضوح [/slow] بينما يمكننا نطق [fast] جمل سريعة وموجزة [/fast] بنفس الجودة.
   ```
3. **Phonetic Lexicon Replacement (`voices/lexicon.json`):**
   ```text
   نظام voicegw يدعم تقنيات AI و LLM ويعمل على GPU بكفاءة.
   ```
   *The system transparently expands `AI` → `إيه آي`, `LLM` → `إل إل إم`, and `GPU` → `جي بي يو`.*

---

### Step 5: Developer API & Ecosystem Parity (30 seconds)

Show how easy it is for any application to consume `voicegw`:

1. **OpenAI Drop-In Compatibility (Python):**
   ```python
   from openai import OpenAI

   client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

   # Drop-in Speech
   response = client.audio.speech.create(
       model="omnivoice",
       voice="default",
       input="مرحباً بالعالم من خلال واجهة برمجة التطبيقات!",
   )
   response.stream_to_file("output.wav")
   ```
2. **Timestamped Transcriptions (`srt` / `vtt`):**
   ```powershell
   curl.exe -F file=@sample.wav -F response_format=srt http://127.0.0.1:8000/v1/audio/transcriptions
   ```
3. **Model Context Protocol (MCP):**
   * Mention that `voicegw mcp` exposes tools (`transcribe_audio`, `synthesize_speech`, `list_voices`) directly to Claude Desktop, Cursor, or AI IDEs over stdio.

---

## 📋 Ready-to-Use Demo Prompts

### Arabic Demonstration Prompts
* **General Introduction:**
  `"مرحباً بكم في العرض التوضيحي لنظام voicegw، البوابة الصوتية الفورية باللغة العربية."`
* **Technical Showcase (Lexicon + Pause):**
  `"يعتمد النظام على نموذج ASR ونموذج TTS محليين بالكامل، [pause 500ms] مع تسريع كامل على كرت الشاشة GPU."`
* **Conversational AI Question (with LLM enabled):**
  `"ما هي عاصمة سلطنة عمان، وما أهم معالمها؟"`

### English Demonstration Prompts
* Switch UI via **`🌐 English`** button.
* **General Introduction:**
  `"Welcome to the voicegw demonstration, showcasing real-time voice intelligence."`
* **Speed Modulation:**
  `"[slow] High precision voice cloning [/slow] combined with [fast] sub-second response times [/fast]."`

---

---

## ⚡ Dual Operational Modes (Cohere ASR + OmniVoice)

`voicegw` provides two dedicated, production-ready operational modes for running **Cohere Transcribe Arabic** and **OmniVoice**:

### Mode 1: Original Models on GPU (CUDA)
* **Engines:** **Original `cohere-asr`** (PyTorch conformer on CUDA) + **Original `omnivoice`** (PyTorch fp16 on CUDA, 16/8 steps).
* **VRAM Footprint:** ~3.3 GB total (shared residency, fits comfortably on 6 GB RTX 4050/3060).
* **Latency:** **~792 ms sub-second turn latency**.
* **Launch Command:**
  ```powershell
  python -m voicegw.cli serve --profile gpu --stt cohere-asr --tts omnivoice --port 8005
  ```

### Mode 2: Memory-Optimized CPU Mode
* **Engines:** **Optimized `cohere-asr-cpu`** (bfloat16 16-bit, bounded tokens, ~4.1 GB RAM) + **Optimized `omnivoice-cpu`** / `omnivoice-gguf` (4-step diffusion, ~1.2 GB RAM).
* **RAM Footprint:** ~5.3 GB total (prevents 98% RAM exhaustion; eliminates SSD page thrashing).
* **CPU Tuning:** Uses `VOICEGW_CPU_THREADS` across all CPU cores with `torch.inference_mode()`.
* **Latency:** ~1.8s first audio chunk / ~3.5s full turn.
* **Launch Command:**
  ```powershell
  python -m voicegw.cli serve --profile cpu --stt cohere-asr-cpu --tts omnivoice-cpu --port 8005
  ```

---

## 💡 Troubleshooting & Presenter Tips

* **Microphone Muted:** If the level meter isn't moving, check browser microphone permissions in Chrome/Edge settings.
* **Low GPU VRAM Alert:** If the system is started on a machine with insufficient VRAM, `voicegw` automatically selects `--profile cpu` or exclusive swapping mode.
* **Testing without Microphone:** Use the **Speak** input field on the conversation tab to run pure TTS without background room noise.

