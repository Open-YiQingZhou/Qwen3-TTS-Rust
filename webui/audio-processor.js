// Audio Worklet Processor for real-time PCM playback
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = [];
        this.maxBufferSize = 24000 * 5; // 5 seconds of buffer at 24kHz
        
        // Handle incoming audio data from main thread
        this.port.onmessage = (event) => {
            const data = event.data;
            if (data instanceof Float32Array) {
                // Add samples to buffer (limit buffer size to prevent memory issues)
                for (let i = 0; i < data.length; i++) {
                    if (this.buffer.length < this.maxBufferSize) {
                        // Clamp values to valid range
                        const sample = Math.max(-1, Math.min(1, data[i]));
                        this.buffer.push(sample);
                    }
                }
            }
        };
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        const channel = output[0];

        // Fill output from buffer
        for (let i = 0; i < channel.length; i++) {
            if (this.buffer.length > 0) {
                channel[i] = this.buffer.shift();
            } else {
                channel[i] = 0;
            }
        }

        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);
