// Audio Worklet Processor for real-time PCM playback
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // 使用环形缓冲区，更高效
        this.bufferSize = 24000 * 10; // 10秒缓冲
        this.buffer = new Float32Array(this.bufferSize);
        this.writeIndex = 0;
        this.readIndex = 0;
        this.available = 0;
        
        this.port.onmessage = (event) => {
            const data = event.data;
            if (data instanceof Float32Array) {
                // 写入环形缓冲区
                for (let i = 0; i < data.length; i++) {
                    if (this.available < this.bufferSize) {
                        const sample = Math.max(-1, Math.min(1, data[i]));
                        this.buffer[this.writeIndex] = sample;
                        this.writeIndex = (this.writeIndex + 1) % this.bufferSize;
                        this.available++;
                    }
                }
            }
        };
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        const channel = output[0];

        for (let i = 0; i < channel.length; i++) {
            if (this.available > 0) {
                channel[i] = this.buffer[this.readIndex];
                this.readIndex = (this.readIndex + 1) % this.bufferSize;
                this.available--;
            } else {
                channel[i] = 0;
            }
        }

        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);
