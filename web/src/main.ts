import { MicCapture, StreamPlayer, floatToPcm16, pcm16ToFloat } from "./audio";
import "./style.css";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

// Navigation
const tabBtnConv = $<HTMLButtonElement>("tab-btn-conversation");
const tabBtnStudio = $<HTMLButtonElement>("tab-btn-studio");
const tabPaneConv = $<HTMLElement>("tab-conversation");
const tabPaneStudio = $<HTMLElement>("tab-studio");

tabBtnConv.onclick = () => {
  tabBtnConv.classList.add("active");
  tabBtnConv.setAttribute("aria-selected", "true");
  tabBtnStudio.classList.remove("active");
  tabBtnStudio.setAttribute("aria-selected", "false");
  tabPaneConv.classList.add("active");
  tabPaneStudio.classList.remove("active");
};

tabBtnStudio.onclick = () => {
  tabBtnStudio.classList.add("active");
  tabBtnStudio.setAttribute("aria-selected", "true");
  tabBtnConv.classList.remove("active");
  tabBtnConv.setAttribute("aria-selected", "false");
  tabPaneStudio.classList.add("active");
  tabPaneConv.classList.remove("active");
  void refreshVoicesList();
};

// Conversation elements
const connectBtn = $<HTMLButtonElement>("connect");
const talkBtn = $<HTMLButtonElement>("talk");
const stopBtn = $<HTMLButtonElement>("stop");
const openMic = $<HTMLInputElement>("open-mic");
const useAgent = $<HTMLInputElement>("use-agent");
const sayText = $<HTMLInputElement>("say-text");
const sayBtn = $<HTMLButtonElement>("say");
const voiceSelect = $<HTMLSelectElement>("voice");
const levelBar = $<HTMLProgressElement>("level");
const transcript = $<HTMLDivElement>("transcript");
const profileBadge = $<HTMLSpanElement>("profile-badge");
const engineBadge = $<HTMLSpanElement>("engine-badge");
const statusBadge = $<HTMLSpanElement>("status-badge");
const hudAsr = $<HTMLElement>("hud-asr");
const hudAudio = $<HTMLElement>("hud-audio");
const hudTotal = $<HTMLElement>("hud-total");

let socket: WebSocket | undefined;
let assistantTurn: HTMLDivElement | undefined;

const player = new StreamPlayer((playing) => {
  stopBtn.disabled = !playing;
});

const mic = new MicCapture(
  (frame) => {
    if (socket?.readyState === WebSocket.OPEN) socket.send(floatToPcm16(frame));
  },
  (rms) => {
    levelBar.value = Math.min(1, rms * 8);
  },
);

function setStatus(text: string, kind: "" | "ok" | "warn" = ""): void {
  statusBadge.textContent = text;
  statusBadge.className = `badge ${kind}`;
}

function addTurn(who: "user" | "assistant" | "error", text: string): HTMLDivElement {
  const div = document.createElement("div");
  div.className = `turn ${who}`;
  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who;
  div.append(label, document.createTextNode(text));
  transcript.append(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
  return div;
}

function wsUrl(): string {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}/v1/realtime`;
}

async function connect(): Promise<void> {
  setStatus("connecting…");
  socket = new WebSocket(wsUrl());
  socket.binaryType = "arraybuffer";

  socket.onopen = async () => {
    await mic.start();
    talkBtn.disabled = false;
    voiceSelect.disabled = false;
    sayText.disabled = false;
    sayBtn.disabled = false;
    connectBtn.textContent = "Disconnect";
  };

  socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      handleEvent(JSON.parse(event.data));
    } else {
      player.push(pcm16ToFloat(event.data as ArrayBuffer));
    }
  };

  socket.onclose = () => {
    setStatus("disconnected");
    talkBtn.disabled = true;
    voiceSelect.disabled = true;
    sayText.disabled = true;
    sayBtn.disabled = true;
    connectBtn.textContent = "Connect";
    void mic.stop();
  };

  socket.onerror = () => setStatus("connection error", "warn");
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function handleEvent(event: any): void {
  switch (event.type) {
    case "session.created": {
      setStatus("connected", "ok");
      profileBadge.textContent = `${event.profile} · ${event.device}`;
      profileBadge.className = `badge ${event.profile === "gpu" ? "ok" : "warn"}`;
      engineBadge.textContent = `${event.asr_engine} → ${event.tts_engine}`;
      voiceSelect.replaceChildren(...event.voices.map((id: string) => new Option(id, id)));
      useAgent.checked = event.agent;
      if (!event.agent_available) {
        useAgent.checked = false;
        useAgent.disabled = true;
        useAgent.parentElement!.title =
          "No agent endpoint configured (VOICEGW_AGENT_BASE_URL) — replies are echoed.";
      }
      void player.prepare(24_000);
      break;
    }
    case "speech_started":
      setStatus("listening…", "ok");
      player.flush();
      break;
    case "transcript.final":
      setStatus("thinking…");
      hudAsr.textContent = event.latency_ms ? `${event.latency_ms} ms` : "—";
      if (event.text) addTurn("user", event.text);
      assistantTurn = undefined;
      break;
    case "response.text.delta":
      if (!assistantTurn) assistantTurn = addTurn("assistant", "");
      assistantTurn.append(document.createTextNode(event.delta));
      break;
    case "response.audio.start":
      setStatus("speaking", "ok");
      hudAudio.textContent = `${event.first_audio_ms} ms`;
      void player.prepare(event.sample_rate);
      break;
    case "response.done":
      setStatus("connected", "ok");
      hudTotal.textContent = `${event.total_ms} ms`;
      break;
    case "response.cancelled":
      player.flush();
      setStatus(`cancelled (${event.reason})`, "warn");
      break;
    case "error":
      addTurn("error", `${event.code}: ${event.message}`);
      setStatus(event.code, "warn");
      break;
  }
}

function send(payload: unknown): void {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
}

connectBtn.onclick = () => {
  if (socket && socket.readyState <= WebSocket.OPEN) {
    socket.close();
  } else {
    void connect();
  }
};

const startTalking = () => {
  if (openMic.checked) return;
  talkBtn.classList.add("active");
  player.flush();
  send({ type: "cancel" });
  mic.setMuted(false);
};

const stopTalking = () => {
  if (openMic.checked) return;
  talkBtn.classList.remove("active");
  mic.setMuted(true);
  send({ type: "commit" });
};

talkBtn.addEventListener("pointerdown", startTalking);
talkBtn.addEventListener("pointerup", stopTalking);
talkBtn.addEventListener("pointerleave", stopTalking);

openMic.onchange = () => {
  mic.setMuted(!openMic.checked);
  talkBtn.disabled = openMic.checked || !socket;
};

voiceSelect.onchange = () => send({ type: "config", voice: voiceSelect.value });
useAgent.onchange = () => send({ type: "config", agent: useAgent.checked });

const speakTyped = () => {
  const text = sayText.value.trim();
  if (!text) return;
  addTurn("user", text);
  assistantTurn = undefined;
  player.flush();
  send({ type: "text", text });
  sayText.value = "";
};

sayBtn.onclick = speakTyped;
sayText.addEventListener("keydown", (e) => {
  if (e.key === "Enter") speakTyped();
});

stopBtn.onclick = () => {
  player.flush();
  send({ type: "cancel" });
};

// ---------------------------------------------------------------------------
// Studio View Logic
// ---------------------------------------------------------------------------
const recBtn = $<HTMLButtonElement>("studio-record-btn");
const stopRecBtn = $<HTMLButtonElement>("studio-stop-rec-btn");
const playRecBtn = $<HTMLButtonElement>("studio-play-rec-btn");
const fileUpload = $<HTMLInputElement>("studio-file-upload");
const timerEl = $<HTMLElement>("studio-rec-timer");
const canvas = $<HTMLCanvasElement>("studio-visualizer");
const canvasCtx = canvas.getContext("2d");

const analysisSection = $<HTMLElement>("studio-analysis-section");
const qcBadge = $<HTMLElement>("qc-badge");
const qcMetricsList = $<HTMLElement>("qc-metrics-list");
const prosodyTags = $<HTMLElement>("prosody-tags");
const prosodyMetricsList = $<HTMLElement>("prosody-metrics-list");

const studioForm = $<HTMLFormElement>("studio-voice-form");
const voiceIdInput = $<HTMLInputElement>("new-voice-id");
const voiceLabelInput = $<HTMLInputElement>("new-voice-label");
const voiceRefTextInput = $<HTMLTextAreaElement>("new-voice-ref-text");
const saveBtn = $<HTMLButtonElement>("studio-save-btn");

const testText = $<HTMLInputElement>("studio-test-text");
const testBtn = $<HTMLButtonElement>("studio-test-btn");
const audioPlayer = $<HTMLAudioElement>("studio-audio-player");
const voicesListEl = $<HTMLElement>("studio-voices-list");

let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];
let recordedBlob: Blob | null = null;
let recStartTime = 0;
let recTimerInterval: number | undefined;
let animFrameId: number | undefined;
let audioCtx: AudioContext | null = null;
let analyser: AnalyserNode | null = null;
let sourceNode: MediaStreamAudioSourceNode | null = null;

function drawVisualizer() {
  if (!analyser || !canvasCtx) return;
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteFrequencyData(dataArray);

  canvasCtx.fillStyle = "#090b10";
  canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

  const barWidth = (canvas.width / bufferLength) * 2.5;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    const barHeight = (dataArray[i] / 255) * canvas.height;
    canvasCtx.fillStyle = `hsl(${210 + (i / bufferLength) * 40}, 95%, ${45 + (dataArray[i] / 255) * 25}%)`;
    canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
    x += barWidth + 1;
    if (x > canvas.width) break;
  }

  animFrameId = requestAnimationFrame(drawVisualizer);
}

recBtn.onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    audioCtx = new AudioContext();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 128;
    sourceNode = audioCtx.createMediaStreamSource(stream);
    sourceNode.connect(analyser);

    drawVisualizer();

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      if (animFrameId) cancelAnimationFrame(animFrameId);
      if (sourceNode) sourceNode.disconnect();
      if (audioCtx) void audioCtx.close();

      stream.getTracks().forEach((track) => track.stop());
      recordedBlob = new Blob(audioChunks, { type: "audio/wav" });
      playRecBtn.disabled = false;
      saveBtn.disabled = false;
      testBtn.disabled = false;
      await runAnalysis(recordedBlob);
    };

    mediaRecorder.start();
    recBtn.classList.add("recording");
    recBtn.disabled = true;
    stopRecBtn.disabled = false;
    playRecBtn.disabled = true;

    recStartTime = Date.now();
    recTimerInterval = window.setInterval(() => {
      const elapsed = Math.floor((Date.now() - recStartTime) / 1000);
      const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
      const secs = String(elapsed % 60).padStart(2, "0");
      timerEl.textContent = `${mins}:${secs}`;
    }, 200);
  } catch (err) {
    alert("تعذر الوصول إلى الميكروفون: " + err);
  }
};

stopRecBtn.onclick = () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    recBtn.classList.remove("recording");
    recBtn.disabled = false;
    stopRecBtn.disabled = true;
    window.clearInterval(recTimerInterval);
  }
};

playRecBtn.onclick = () => {
  if (recordedBlob) {
    const url = URL.createObjectURL(recordedBlob);
    const audio = new Audio(url);
    void audio.play();
  }
};

fileUpload.onchange = async () => {
  if (fileUpload.files && fileUpload.files[0]) {
    recordedBlob = fileUpload.files[0];
    playRecBtn.disabled = false;
    saveBtn.disabled = false;
    testBtn.disabled = false;
    await runAnalysis(recordedBlob);
  }
};

async function runAnalysis(blob: Blob) {
  analysisSection.style.display = "block";
  prosodyTags.innerHTML = "<span class='tag-chip'>جاري التحليل النغمي…</span>";
  qcMetricsList.innerHTML = "<li>جاري فحص جودة الصوت…</li>";

  const formData = new FormData();
  formData.append("file", blob, "sample.wav");

  try {
    const res = await fetch("/v1/voices/describe", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    // Render prosody tags
    prosodyTags.innerHTML = "";
    const tags = data.description ? data.description.split(",") : [];
    tags.forEach((tag: string) => {
      const chip = document.createElement("span");
      chip.className = "tag-chip";
      chip.textContent = tag.trim();
      prosodyTags.appendChild(chip);
    });

    // Render prosody metrics
    prosodyMetricsList.innerHTML = `
      <li><span>تردد النغمة الأساسي (F0):</span> <b>${data.metrics.f0_hz ? data.metrics.f0_hz + " Hz" : "—"}</b></li>
      <li><span>نسبة النطق الصوتي (Voicing):</span> <b>${Math.round(data.metrics.voicing_ratio * 100)}%</b></li>
      <li><span>معدل المقاطع الصوتية:</span> <b>${data.metrics.syllables_per_s} syllables/s</b></li>
      <li><span>شدة الصوت (RMS):</span> <b>${data.metrics.rms_dbfs} dBFS</b></li>
    `;

    // Render QC report
    const rms = data.metrics.rms_dbfs;
    let qcStatus = "ممتاز";
    let qcClass = "ok";
    const warnings: string[] = [];

    if (rms < -35) {
      qcStatus = "منخفض جداً";
      qcClass = "warn";
      warnings.push("الصوت هادئ جداً، يفضل الاقتراب من الميكروفون");
    } else if (rms > -6) {
      qcStatus = "مرتفع جداً";
      qcClass = "warn";
      warnings.push("الصوت يقترب من حد التشويش");
    }

    qcBadge.textContent = qcStatus;
    qcBadge.className = `badge ${qcClass}`;

    qcMetricsList.innerHTML = `
      <li><span>شدة الإشارة:</span> <b>${rms} dBFS</b></li>
      <li><span>الحالة:</span> <b>${qcStatus}</b></li>
      ${warnings.map((w) => `<li style="color: var(--warn); font-size: 0.75rem;">⚠ ${w}</li>`).join("")}
    `;
  } catch (err) {
    prosodyTags.innerHTML = `<span style="color: var(--danger);">فشل التحليل: ${err}</span>`;
  }
}

studioForm.onsubmit = async (e) => {
  e.preventDefault();
  if (!recordedBlob) {
    alert("يرجى تسجيل أو رفع عينة صوتية أولاً.");
    return;
  }

  saveBtn.disabled = true;
  saveBtn.textContent = "جاري الحفظ…";

  const formData = new FormData();
  formData.append("id", voiceIdInput.value.trim());
  formData.append("label", voiceLabelInput.value.trim());
  formData.append("ref_text", voiceRefTextInput.value.trim());
  formData.append("file", recordedBlob, `${voiceIdInput.value.trim()}.wav`);
  formData.append("language", "ar");

  try {
    const res = await fetch("/v1/voices", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || JSON.stringify(err));
    }
    const created = await res.json();
    alert(`تم تسجيل الصوت '${created.label}' بنجاح!`);
    void refreshVoicesList();
  } catch (err) {
    alert("حدث خطأ أثناء الحفظ: " + err);
  } finally {
    saveBtn.disabled = false;
    saveBtn.textContent = "حفظ وتسجيل الصوت";
  }
};

testBtn.onclick = async () => {
  const vId = voiceIdInput.value.trim() || voiceSelect.value || "default";
  const text = testText.value.trim();
  if (!text) return;

  testBtn.disabled = true;
  testBtn.textContent = "جاري النطق…";

  try {
    const res = await fetch("/v1/audio/speech", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input: text,
        voice: vId,
        response_format: "wav",
        stream: false,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    audioPlayer.src = url;
    audioPlayer.style.display = "block";
    void audioPlayer.play();
  } catch (err) {
    alert("فشل اختبار النطق: " + err);
  } finally {
    testBtn.disabled = false;
    testBtn.textContent = "نطق";
  }
};

async function refreshVoicesList() {
  try {
    const res = await fetch("/v1/voices");
    if (!res.ok) return;
    const data = await res.json();
    const voices = data.data || [];

    // Populate select
    voiceSelect.replaceChildren(...voices.map((v: { id: string; label: string }) => new Option(v.label || v.id, v.id)));

    // Populate chips
    voicesListEl.innerHTML = "";
    voices.forEach((v: { id: string; label: string; is_clone: boolean; tags: string[] }) => {
      const chip = document.createElement("div");
      chip.className = "voice-badge-card";
      chip.innerHTML = `
        <span>${v.label}</span>
        <span class="v-type">${v.is_clone ? "Clone" : "Design"}</span>
      `;
      chip.onclick = () => {
        voiceIdInput.value = v.id;
        voiceLabelInput.value = v.label;
        voiceSelect.value = v.id;
      };
      voicesListEl.appendChild(chip);
    });
  } catch (err) {
    console.error("Could not fetch voices:", err);
  }
}

// Initial health check & voices
void fetch("/healthz")
  .then((r) => r.json())
  .then((health) => {
    if (health.status !== "ok") return;
    profileBadge.textContent = `${health.profile} · ${health.device}`;
    profileBadge.className = `badge ${health.profile === "gpu" ? "ok" : "warn"}`;
  })
  .catch(() => setStatus("gateway unreachable", "warn"));

void refreshVoicesList();
