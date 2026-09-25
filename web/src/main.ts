import { MicCapture, StreamPlayer, floatToPcm16, pcm16ToFloat } from "./audio";
import "./style.css";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const connectBtn = $<HTMLButtonElement>("connect");
const talkBtn = $<HTMLButtonElement>("talk");
const stopBtn = $<HTMLButtonElement>("stop");
const openMic = $<HTMLInputElement>("open-mic");
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
      voiceSelect.replaceChildren(
        ...event.voices.map((id: string) => new Option(id, id)),
      );
      void player.prepare(24_000);
      break;
    }
    case "speech_started":
      setStatus("listening…", "ok");
      // Local echo of the server's barge-in: stop playback right away.
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

// Push-to-talk: unmute while held, then commit the utterance on release.
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
  // Open mic streams continuously and lets the server's VAD segment speech.
  mic.setMuted(!openMic.checked);
  talkBtn.disabled = openMic.checked || !socket;
};

voiceSelect.onchange = () => send({ type: "config", voice: voiceSelect.value });

stopBtn.onclick = () => {
  player.flush();
  send({ type: "cancel" });
};

void fetch("/healthz")
  .then((r) => r.json())
  .then((health) => {
    if (health.status !== "ok") return;
    profileBadge.textContent = `${health.profile} · ${health.device}`;
    profileBadge.className = `badge ${health.profile === "gpu" ? "ok" : "warn"}`;
    if (!health.realtime_capable) {
      addTurn(
        "error",
        `Batch mode: engine '${health.asr_engine.engine_id}' is too slow for realtime. ` +
          `Expect delays, or switch profile.`,
      );
    }
  })
  .catch(() => setStatus("gateway unreachable", "warn"));
