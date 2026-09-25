/**
 * Capture worklet: forwards mono float32 mic frames to the main thread, which
 * converts them to PCM16 and sends them over the WebSocket.
 *
 * The AudioContext is created at 16 kHz, so no resampling is needed here.
 */
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(0);
    // ~32 ms at 16 kHz, matching the server's VAD frame size.
    this.frameSize = 512;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    const merged = new Float32Array(this.buffer.length + channel.length);
    merged.set(this.buffer);
    merged.set(channel, this.buffer.length);

    let offset = 0;
    while (merged.length - offset >= this.frameSize) {
      const frame = merged.slice(offset, offset + this.frameSize);
      this.port.postMessage(frame, [frame.buffer]);
      offset += this.frameSize;
    }
    this.buffer = merged.slice(offset);
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
