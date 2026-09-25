/**
 * Playback worklet: a ring buffer that plays streamed TTS chunks gaplessly.
 *
 * Chunks arrive at irregular intervals as the server synthesizes them, so the
 * audio thread must never block waiting for one -- it emits silence on
 * underrun instead of glitching, and `flush` empties the buffer instantly for
 * barge-in.
 */
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.capacity = 24000 * 30; // 30 s at 24 kHz
    this.ring = new Float32Array(this.capacity);
    this.read = 0;
    this.write = 0;
    this.available = 0;
    this.playing = false;

    this.port.onmessage = (event) => {
      const { type, payload } = event.data;
      if (type === "push") {
        this.push(payload);
      } else if (type === "flush") {
        // Barge-in: drop everything queued so playback stops immediately.
        this.read = this.write = this.available = 0;
        this.setPlaying(false);
      }
    };
  }

  setPlaying(value) {
    if (this.playing !== value) {
      this.playing = value;
      this.port.postMessage({ type: value ? "started" : "ended" });
    }
  }

  push(samples) {
    for (let i = 0; i < samples.length; i++) {
      this.ring[this.write] = samples[i];
      this.write = (this.write + 1) % this.capacity;
      if (this.available < this.capacity) {
        this.available++;
      } else {
        this.read = (this.read + 1) % this.capacity; // overwrite oldest
      }
    }
    this.setPlaying(true);
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    if (!out) return true;

    for (let i = 0; i < out.length; i++) {
      if (this.available > 0) {
        out[i] = this.ring[this.read];
        this.read = (this.read + 1) % this.capacity;
        this.available--;
      } else {
        out[i] = 0;
      }
    }
    if (this.available === 0) this.setPlaying(false);
    return true;
  }
}

registerProcessor("playback-processor", PlaybackProcessor);
