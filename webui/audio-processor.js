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
        
        // 淡入淡出参数
        this.fadeSamples = 48; // 2ms 淡入淡出（24000 * 0.002 = 48）
        this.lastSample = 0;
        this.isFirstChunk = true;
        
        // 缓冲区状态
        this.totalReceived = 0;
        this.totalPlayed = 0;
        this.isDraining = false;
        
        this.port.onmessage = (event) => {
            const data = event.data;
            if (data instanceof Float32Array) {
                // 写入环形缓冲区，应用淡入淡出
                const fadeLen = Math.min(this.fadeSamples, data.length);
                
                for (let i = 0; i < data.length; i++) {
                    if (this.available < this.bufferSize) {
                        let sample = Math.max(-1, Math.min(1, data[i]));
                        
                        // 淡入：开头几帧从上一个样本平滑过渡
                        if (i < fadeLen && !this.isFirstChunk) {
                            const fadeRatio = i / fadeLen;
                            sample = this.lastSample * (1 - fadeRatio) + sample * fadeRatio;
                        }
                        
                        this.buffer[this.writeIndex] = sample;
                        this.writeIndex = (this.writeIndex + 1) % this.bufferSize;
                        this.available++;
                    }
                }
                
                // 保存最后一个样本用于下次淡入
                this.lastSample = data[data.length - 1];
                this.isFirstChunk = false;
                this.totalReceived += data.length;
            } else if (data === 'drain') {
                // 标记为正在排空
                this.isDraining = true;
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
                this.totalPlayed++;
            } else {
                channel[i] = 0;
            }
        }

        // 如果正在排空且缓冲区为空，通知主线程
        if (this.isDraining && this.available === 0) {
            this.port.postMessage({ type: 'buffer_drained', played: this.totalPlayed, received: this.totalReceived });
            this.isDraining = false;
        }

        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);
