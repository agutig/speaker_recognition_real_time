/* AudioWorkletProcessor: agrupa exactamente 1 segundo de audio mono */
class PCMCollectorProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.target = sampleRate;      // nº de muestras en 1 s
      this.buf    = new Float32Array(0);
    }
    process(inputs) {
      const input = inputs[0][0];    // canal 0 mono
      if (!input) return true;
  
      // concatenar
      const tmp = new Float32Array(this.buf.length + input.length);
      tmp.set(this.buf); tmp.set(input, this.buf.length);
      this.buf = tmp;
  
      while (this.buf.length >= this.target) {
        const chunk = this.buf.slice(0, this.target);
        this.port.postMessage(chunk);
        this.buf = this.buf.slice(this.target);
      }
      return true;
    }
  }
  registerProcessor("pcm-collector", PCMCollectorProcessor);
  