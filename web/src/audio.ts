/** Mic capture -> PCM16 @16 kHz, and gapless playback of streamed PCM16. */

export const INPUT_SAMPLE_RATE = 16_000;

export function floatToPcm16(samples: Float32Array): ArrayBuffer {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out.buffer;
}

export function pcm16ToFloat(buffer: ArrayBuffer): Float32Array {
  const input = new Int16Array(buffer);
  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) out[i] = input[i] / 0x8000;
  return out;
}

export class MicCapture {
  private context?: AudioContext;
  private stream?: MediaStream;
  private node?: AudioWorkletNode;
  private muted = true;

  constructor(
    private readonly onFrame: (frame: Float32Array) => void,
    private readonly onLevel: (rms: number) => void,
  ) {}

  async start(): Promise<void> {
    if (this.context) return;

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    // Requesting 16 kHz directly avoids a resampling step on the main thread.
    this.context = new AudioContext({ sampleRate: INPUT_SAMPLE_RATE });
    await this.context.audioWorklet.addModule("/capture-worklet.js");

    const source = this.context.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.context, "capture-processor");
    this.node.port.onmessage = (event) => {
      const frame = event.data as Float32Array;
      let sum = 0;
      for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
      this.onLevel(Math.sqrt(sum / frame.length));
      if (!this.muted) this.onFrame(frame);
    };
    source.connect(this.node);
    // Keep the graph alive without routing the mic to the speakers.
    this.node.connect(this.context.destination);
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
  }

  async stop(): Promise<void> {
    this.node?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    await this.context?.close();
    this.context = undefined;
    this.node = undefined;
    this.stream = undefined;
  }
}

export class StreamPlayer {
  private context?: AudioContext;
  private node?: AudioWorkletNode;
  private sampleRate = 24_000;

  constructor(private readonly onStateChange: (playing: boolean) => void) {}

  async prepare(sampleRate: number): Promise<void> {
    if (this.context && this.sampleRate === sampleRate) {
      await this.context.resume();
      return;
    }
    await this.context?.close();
    this.sampleRate = sampleRate;
    this.context = new AudioContext({ sampleRate });
    await this.context.audioWorklet.addModule("/playback-worklet.js");
    this.node = new AudioWorkletNode(this.context, "playback-processor", {
      outputChannelCount: [1],
    });
    this.node.port.onmessage = (event) => {
      this.onStateChange(event.data.type === "started");
    };
    this.node.connect(this.context.destination);
  }

  push(samples: Float32Array): void {
    this.node?.port.postMessage({ type: "push", payload: samples }, [samples.buffer]);
  }

  flush(): void {
    this.node?.port.postMessage({ type: "flush" });
  }

  async close(): Promise<void> {
    this.node?.disconnect();
    await this.context?.close();
    this.context = undefined;
    this.node = undefined;
  }
}
