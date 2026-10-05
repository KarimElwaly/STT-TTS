import { MicCapture, StreamPlayer, floatToPcm16, pcm16ToFloat } from "./audio";
import { Lang, translations } from "./i18n";
import "./style.css";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

// Language state
let currentLang: Lang = (localStorage.getItem("voicegw_lang") as Lang) || "ar";
const t = () => translations[currentLang];

// Navigation
const tabBtnConv = $<HTMLButtonElement>("tab-btn-conversation");
const tabBtnStudio = $<HTMLButtonElement>("tab-btn-studio");
const tabPaneConv = $<HTMLElement>("tab-conversation");
const tabPaneStudio = $<HTMLElement>("tab-studio");
const langToggleBtn = $<HTMLButtonElement>("lang-toggle-btn");
const langToggleText = $<HTMLSpanElement>("lang-toggle-text");

// Brand & Titles
const docTitle = $<HTMLTitleElement>("doc-title");
const subbrand = $<HTMLSpanElement>("subbrand");

// Conversation elements
const connectBtn = $<HTMLButtonElement>("connect");
const talkBtn = $<HTMLButtonElement>("talk");
const stopBtn = $<HTMLButtonElement>("stop");
const openMic = $<HTMLInputElement>("open-mic");
const openMicLabel = $<HTMLSpanElement>("open-mic-label");
const useAgent = $<HTMLInputElement>("use-agent");
const useAgentLabel = $<HTMLSpanElement>("use-agent-label");
const useAgentWrapper = $<HTMLElement>("use-agent-wrapper");
const sayText = $<HTMLInputElement>("say-text");
const sayBtn = $<HTMLButtonElement>("say");
const voiceSelect = $<HTMLSelectElement>("voice");
const levelBar = $<HTMLProgressElement>("level");
const micMeterLabel = $<HTMLSpanElement>("mic-meter-label");
const hudAsrLabel = $<HTMLSpanElement>("hud-asr-label");
const hudFirstAudioLabel = $<HTMLSpanElement>("hud-first-audio-label");
const hudTotalLabel = $<HTMLSpanElement>("hud-total-label");
const transcript = $<HTMLDivElement>("transcript");

// Badges
const profileBadge = $<HTMLSpanElement>("profile-badge");
const engineBadge = $<HTMLSpanElement>("engine-badge");
const statusBadge = $<HTMLSpanElement>("status-badge");
const hudAsr = $<HTMLElement>("hud-asr");
const hudAudio = $<HTMLElement>("hud-audio");
const hudTotal = $<HTMLElement>("hud-total");

// Studio Elements
const studioRecTitle = $<HTMLElement>("studio-record-title");
const studioRecDesc = $<HTMLElement>("studio-record-desc");
const recBtn = $<HTMLButtonElement>("studio-record-btn");
const stopRecBtn = $<HTMLButtonElement>("studio-stop-rec-btn");
const playRecBtn = $<HTMLButtonElement>("studio-play-rec-btn");
const uploadLabel = $<HTMLElement>("studio-upload-label");
const fileUpload = $<HTMLInputElement>("studio-file-upload");
const timerEl = $<HTMLElement>("studio-rec-timer");
const canvas = $<HTMLCanvasElement>("studio-visualizer");
const canvasCtx = canvas.getContext("2d");

const analysisSection = $<HTMLElement>("studio-analysis-section");
const studioAnalysisTitle = $<HTMLElement>("studio-analysis-title");
const qcCardTitle = $<HTMLElement>("qc-card-title");
const qcBadge = $<HTMLElement>("qc-badge");
const qcMetricsList = $<HTMLElement>("qc-metrics-list");
const prosodyCardTitle = $<HTMLElement>("prosody-card-title");
const prosodyTags = $<HTMLElement>("prosody-tags");
const prosodyMetricsList = $<HTMLElement>("prosody-metrics-list");

const studioCreateTitle = $<HTMLElement>("studio-create-title");
const studioCreateDesc = $<HTMLElement>("studio-create-desc");
const studioForm = $<HTMLFormElement>("studio-voice-form");
const voiceIdLabel = $<HTMLElement>("new-voice-id-label");
const voiceIdInput = $<HTMLInputElement>("new-voice-id");
const voiceNameLabel = $<HTMLElement>("new-voice-label-label");
const voiceLabelInput = $<HTMLInputElement>("new-voice-label");
const refTextLabel = $<HTMLElement>("new-voice-ref-text-label");
const voiceRefTextInput = $<HTMLTextAreaElement>("new-voice-ref-text");
const saveBtn = $<HTMLButtonElement>("studio-save-btn");

const studioTestTitle = $<HTMLElement>("studio-test-title");
const testText = $<HTMLInputElement>("studio-test-text");
const testBtn = $<HTMLButtonElement>("studio-test-btn");
const audioPlayer = $<HTMLAudioElement>("studio-audio-player");
const studioVoicesTitle = $<HTMLElement>("studio-voices-title");
const voicesListEl = $<HTMLElement>("studio-voices-list");

let socket: WebSocket | undefined;
let assistantTurn: HTMLDivElement | undefined;
let pendingUserTurn: HTMLDivElement | undefined;
let rawVoices: Array<{ id: string; label: string; is_clone: boolean; tags: string[] }> = [];

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

function applyLanguage(lang: Lang): void {
  currentLang = lang;
  localStorage.setItem("voicegw_lang", lang);
  const cur = t();

  document.documentElement.lang = lang;
  document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";

  // Document metadata
  if (docTitle) docTitle.textContent = cur.pageTitle;
  subbrand.textContent = cur.subbrand;
  langToggleText.textContent = cur.toggleToLang;

  // Tabs
  tabBtnConv.textContent = cur.tabConversation;
  tabBtnStudio.textContent = cur.tabStudio;

  // Controls
  const isConnected = socket?.readyState === WebSocket.OPEN;
  connectBtn.textContent = isConnected ? cur.disconnect : cur.connect;
  talkBtn.textContent = cur.holdToTalk;
  openMicLabel.textContent = cur.openMicVad;
  voiceSelect.setAttribute("aria-label", cur.voiceSelectLabel);
  stopBtn.textContent = cur.stopAudio;
  useAgentLabel.textContent = cur.sendToAgent;
  if (!useAgent.disabled) {
    useAgentWrapper.title = cur.agentTooltipOff;
  }
  sayText.placeholder = cur.sayPlaceholder;
  sayBtn.textContent = cur.speakBtn;

  // Meters & HUD
  micMeterLabel.textContent = cur.micLabel;
  hudAsrLabel.textContent = cur.hudAsr;
  hudFirstAudioLabel.textContent = cur.hudFirstAudio;
  hudTotalLabel.textContent = cur.hudTotal;

  // Studio Left
  studioRecTitle.textContent = cur.studioRecordTitle;
  studioRecDesc.textContent = cur.studioRecordDesc;
  if (!recBtn.classList.contains("recording")) {
    recBtn.textContent = cur.studioRecordBtn;
  }
  stopRecBtn.textContent = cur.studioStopRecBtn;
  playRecBtn.textContent = cur.studioPlayRecBtn;
  uploadLabel.textContent = cur.studioUploadPrompt;
  studioAnalysisTitle.textContent = cur.studioAnalysisTitle;
  qcCardTitle.textContent = cur.qcCardTitle;
  prosodyCardTitle.textContent = cur.prosodyCardTitle;

  // Studio Right
  studioCreateTitle.textContent = cur.studioCreateTitle;
  studioCreateDesc.textContent = cur.studioCreateDesc;
  voiceIdLabel.textContent = cur.voiceIdLabel;
  voiceIdInput.placeholder = cur.voiceIdPlaceholder;
  voiceNameLabel.textContent = cur.voiceNameLabel;
  voiceLabelInput.placeholder = cur.voiceNamePlaceholder;
  refTextLabel.textContent = cur.refTextLabel;
  voiceRefTextInput.placeholder = cur.refTextPlaceholder;
  saveBtn.textContent = cur.saveVoiceBtn;

  // Instant Test
  studioTestTitle.textContent = cur.instantTestTitle;
  testText.placeholder = cur.instantTestPlaceholder;
  if (
    !testText.value ||
    testText.value === translations.ar.instantTestDefault ||
    testText.value === translations.en.instantTestDefault
  ) {
    testText.value = cur.instantTestDefault;
  }
  testBtn.textContent = cur.instantTestBtn;
  studioVoicesTitle.textContent = cur.registeredVoicesTitle;

  // Re-render voice chips to localize badges
  renderVoicesChips();
}

langToggleBtn.onclick = () => {
  const nextLang: Lang = currentLang === "ar" ? "en" : "ar";
  applyLanguage(nextLang);
};

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

function wsUrl(): string {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}/v1/realtime`;
}

async function connect(): Promise<void> {
  const cur = t();
  setStatus(cur.statusConnecting);
  socket = new WebSocket(wsUrl());
  socket.binaryType = "arraybuffer";

  socket.onopen = async () => {
    await mic.start();
    talkBtn.disabled = false;
    voiceSelect.disabled = false;
    sayText.disabled = false;
    sayBtn.disabled = false;
    connectBtn.textContent = cur.disconnect;
  };

  socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      handleEvent(JSON.parse(event.data));
    } else {
      player.push(pcm16ToFloat(event.data as ArrayBuffer));
    }
  };

  socket.onclose = () => {
    const cur = t();
    setStatus(cur.statusDisconnected);
    talkBtn.disabled = true;
    voiceSelect.disabled = true;
    sayText.disabled = true;
    sayBtn.disabled = true;
    connectBtn.textContent = cur.connect;
    void mic.stop();
  };

  socket.onerror = () => setStatus(t().statusConnError, "warn");
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function handleEvent(event: any): void {
  const cur = t();
  switch (event.type) {
    case "session.created": {
      setStatus(cur.statusConnected, "ok");
      profileBadge.textContent = `${event.profile} · ${event.device}`;
      profileBadge.className = `badge ${event.profile === "gpu" ? "ok" : "warn"}`;
      engineBadge.textContent = `${event.asr_engine} → ${event.tts_engine}`;
      voiceSelect.replaceChildren(...event.voices.map((id: string) => new Option(id, id)));
      const savedAgent = localStorage.getItem("voicegw_agent");
      useAgent.checked = savedAgent !== null ? savedAgent === "true" : false;
      send({ type: "config", agent: useAgent.checked });
      if (!event.agent_available) {
        useAgent.checked = false;
        useAgent.disabled = true;
        useAgentWrapper.title = cur.agentTooltipDisabled;
      }
      void player.prepare(24_000);
      break;
    }
    case "speech_started":
      setStatus(cur.statusListening, "ok");
      player.flush();
      break;
    case "speech_stopped":
      setStatus(cur.statusTranscribing || "transcribing…", "ok");
      if (!pendingUserTurn) {
        pendingUserTurn = addTurn("user", cur.statusTranscribing || "🎙️ ...");
        pendingUserTurn.classList.add("pending");
      }
      break;
    case "transcript.final":
      setStatus(useAgent.checked ? cur.statusThinking : cur.statusConnected, "ok");
      hudAsr.textContent = event.latency_ms ? `${event.latency_ms} ms` : "—";
      if (pendingUserTurn) {
        if (event.text) {
          pendingUserTurn.classList.remove("pending");
          const whoSpan = pendingUserTurn.querySelector(".who");
          pendingUserTurn.replaceChildren(whoSpan ?? document.createTextNode(""), document.createTextNode(event.text));
        } else {
          pendingUserTurn.remove();
        }
        pendingUserTurn = undefined;
      } else if (event.text) {
        addTurn("user", event.text);
      }
      assistantTurn = undefined;
      break;
    case "response.text.delta":
      if (!assistantTurn) assistantTurn = addTurn("assistant", "");
      assistantTurn.append(document.createTextNode(event.delta));
      break;
    case "response.audio.start":
      setStatus(cur.statusSpeaking, "ok");
      hudAudio.textContent = `${event.first_audio_ms} ms`;
      void player.prepare(event.sample_rate);
      break;
    case "response.done":
      setStatus(cur.statusConnected, "ok");
      hudTotal.textContent = `${event.total_ms} ms`;
      break;
    case "response.cancelled":
      player.flush();
      if (pendingUserTurn) {
        pendingUserTurn.remove();
        pendingUserTurn = undefined;
      }
      setStatus(`${cur.statusCancelled} (${event.reason})`, "warn");
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
  setStatus(t().statusTranscribing || "transcribing…", "ok");
  if (!pendingUserTurn) {
    pendingUserTurn = addTurn("user", t().statusTranscribing || "🎙️ ...");
    pendingUserTurn.classList.add("pending");
  }
};

talkBtn.addEventListener("pointerdown", startTalking);
talkBtn.addEventListener("pointerup", stopTalking);
talkBtn.addEventListener("pointerleave", stopTalking);

openMic.onchange = () => {
  mic.setMuted(!openMic.checked);
  talkBtn.disabled = openMic.checked || !socket;
};

voiceSelect.onchange = () => send({ type: "config", voice: voiceSelect.value });
useAgent.onchange = () => {
  localStorage.setItem("voicegw_agent", useAgent.checked ? "true" : "false");
  send({ type: "config", agent: useAgent.checked });
};

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
  if (pendingUserTurn) {
    pendingUserTurn.remove();
    pendingUserTurn = undefined;
  }
  send({ type: "cancel" });
};

// ---------------------------------------------------------------------------
// Studio View Logic
// ---------------------------------------------------------------------------
let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];
let recordedBlob: Blob | null = null;
let recStartTime = 0;
let recTimerInterval: number | undefined;
let animFrameId: number | undefined;
let audioCtx: AudioContext | null = null;
let analyser: AnalyserNode | null = null;
let sourceNode: MediaStreamAudioSourceNode | null = null;

function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numChannels = 1; // force mono for voice models
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;

  const length = buffer.length;
  const samples = new Float32Array(length);
  if (buffer.numberOfChannels === 1) {
    samples.set(buffer.getChannelData(0));
  } else {
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      const channel = buffer.getChannelData(c);
      for (let i = 0; i < length; i++) {
        samples[i] += channel[i] / buffer.numberOfChannels;
      }
    }
  }

  const byteRate = (sampleRate * numChannels * bitDepth) / 8;
  const blockAlign = (numChannels * bitDepth) / 8;
  const dataSize = length * (bitDepth / 8);
  const bufferSize = 44 + dataSize;
  const arrayBuffer = new ArrayBuffer(bufferSize);
  const view = new DataView(arrayBuffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, format, true); // AudioFormat (1 = PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

async function convertBlobToWav(blob: Blob): Promise<Blob> {
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const decodeCtx = new (window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
    await decodeCtx.close();
    return audioBufferToWav(audioBuffer);
  } catch (err) {
    console.warn("AudioContext decode fallback, using raw blob:", err);
    return blob;
  }
}

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
  const cur = t();
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
      const rawBlob = new Blob(audioChunks, { type: mediaRecorder?.mimeType || "audio/webm" });
      recordedBlob = await convertBlobToWav(rawBlob);
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
    alert(cur.micAccessError + err);
  }
};

stopRecBtn.onclick = () => {
  const cur = t();
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    recBtn.classList.remove("recording");
    recBtn.textContent = cur.studioRecordBtn;
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
    const file = fileUpload.files[0];
    recordedBlob = await convertBlobToWav(file);
    playRecBtn.disabled = false;
    saveBtn.disabled = false;
    testBtn.disabled = false;
    await runAnalysis(recordedBlob);
  }
};

async function runAnalysis(blob: Blob) {
  const cur = t();
  analysisSection.style.display = "block";
  prosodyTags.innerHTML = `<span class='tag-chip'>${cur.prosodyAnalyzing}</span>`;
  qcMetricsList.innerHTML = `<li>${cur.qcAnalyzing}</li>`;

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
      <li><span>${cur.prosodyF0}</span> <b>${data.metrics.f0_hz ? data.metrics.f0_hz + " Hz" : "—"}</b></li>
      <li><span>${cur.prosodyVoicing}</span> <b>${Math.round(data.metrics.voicing_ratio * 100)}%</b></li>
      <li><span>${cur.prosodySyllables}</span> <b>${data.metrics.syllables_per_s} syllables/s</b></li>
      <li><span>${cur.prosodyLoudness}</span> <b>${data.metrics.rms_dbfs} dBFS</b></li>
    `;

    // Render QC report
    const rms = data.metrics.rms_dbfs;
    let qcStatus = cur.qcStatusExcellent;
    let qcClass = "ok";
    const warnings: string[] = [];

    if (rms < -35) {
      qcStatus = cur.qcStatusTooQuiet;
      qcClass = "warn";
      warnings.push(cur.qcWarnQuiet);
    } else if (rms > -6) {
      qcStatus = cur.qcStatusTooLoud;
      qcClass = "warn";
      warnings.push(cur.qcWarnLoud);
    }

    qcBadge.textContent = qcStatus;
    qcBadge.className = `badge ${qcClass}`;

    qcMetricsList.innerHTML = `
      <li><span>${cur.qcSignalRms}</span> <b>${rms} dBFS</b></li>
      <li><span>${cur.qcStatusLabel}</span> <b>${qcStatus}</b></li>
      ${warnings.map((w) => `<li style="color: var(--warn); font-size: 0.75rem;">⚠ ${w}</li>`).join("")}
    `;
  } catch (err) {
    prosodyTags.innerHTML = `<span style="color: var(--danger);">Error: ${err}</span>`;
  }
}

studioForm.onsubmit = async (e) => {
  e.preventDefault();
  const cur = t();
  if (!recordedBlob) {
    alert(cur.recordFirstPrompt);
    return;
  }

  saveBtn.disabled = true;
  saveBtn.textContent = cur.savingVoiceBtn;

  const formData = new FormData();
  formData.append("id", voiceIdInput.value.trim());
  formData.append("label", voiceLabelInput.value.trim());
  formData.append("ref_text", voiceRefTextInput.value.trim());
  formData.append("file", recordedBlob, `${voiceIdInput.value.trim()}.wav`);
  formData.append("language", currentLang);

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
    alert(`${cur.voiceSavedSuccess} ('${created.label}')`);
    void refreshVoicesList();
  } catch (err) {
    alert(cur.saveError + err);
  } finally {
    saveBtn.disabled = false;
    saveBtn.textContent = cur.saveVoiceBtn;
  }
};

testBtn.onclick = async () => {
  const cur = t();
  const vId = voiceIdInput.value.trim() || voiceSelect.value || "default";
  const text = testText.value.trim();
  if (!text) return;

  testBtn.disabled = true;
  testBtn.textContent = cur.instantTestingBtn;

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
    alert(cur.synthTestError + err);
  } finally {
    testBtn.disabled = false;
    testBtn.textContent = cur.instantTestBtn;
  }
};

function renderVoicesChips() {
  const cur = t();
  voicesListEl.innerHTML = "";
  rawVoices.forEach((v) => {
    const chip = document.createElement("div");
    chip.className = "voice-badge-card";
    const badgeText = v.is_clone ? cur.cloneBadge : cur.designBadge;
    chip.innerHTML = `
      <span>${v.label}</span>
      <span class="v-type">${badgeText}</span>
    `;
    chip.onclick = () => {
      document.querySelectorAll(".voice-badge-card").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      voiceIdInput.value = v.id;
      voiceLabelInput.value = v.label;
      voiceSelect.value = v.id;
    };
    voicesListEl.appendChild(chip);
  });
}

async function refreshVoicesList() {
  try {
    const res = await fetch("/v1/voices");
    if (!res.ok) return;
    const data = await res.json();
    rawVoices = data.data || [];

    // Populate select
    voiceSelect.replaceChildren(
      ...rawVoices.map((v) => new Option(v.label || v.id, v.id)),
    );

    renderVoicesChips();
  } catch (err) {
    console.error("Could not fetch voices:", err);
  }
}

// Initial initialization
applyLanguage(currentLang);

void fetch("/healthz")
  .then((r) => r.json())
  .then((health) => {
    if (health.status !== "ok") return;
    profileBadge.textContent = `${health.profile} · ${health.device}`;
    profileBadge.className = `badge ${health.profile === "gpu" ? "ok" : "warn"}`;
  })
  .catch(() => setStatus(t().statusUnreachable, "warn"));

void refreshVoicesList();
