// static/client.js
console.log("✔ client.js (AudioWorklet + 5 s historial + etiquetas fluidas) cargado");

window.addEventListener("DOMContentLoaded", () => {
  // ─── DOM ─────────────────────────────────────────
  const startBtn     = document.getElementById("start");
  const stopBtn      = document.getElementById("stop");
  const numSpeakersI = document.getElementById("numSpeakers");
  const canvas       = document.getElementById("waveformChart");
  const ctx          = canvas.getContext("2d");

  // ─── ESTADO GLOBAL ────────────────────────────────
  let audioCtx, analyser, dataArray, bufferLen;
  let ws, socketReady = false;
  let workletNode, animationId;
  let currentDom = 0;       // hablante vigente
  let lastTime = 0, leftover = 0;

  const WINDOW_SEC  = 5;                       // segundos a mostrar
  const RES_PER_SEC = 800;                     // pts/s
  const HIST_PTS    = WINDOW_SEC * RES_PER_SEC;// 4000
  const FPS_TARGET  = 60;                      // fps aproximado

  // buffers circulares
  let waveHist  = new Float32Array(HIST_PTS).fill(0);
  let labelHist = new Uint8Array   (HIST_PTS).fill(0);

  // colores por hablante
  const speakerColors = {
    0: "#ccc",
    1: "rgba(255,0,0,0.45)",
    2: "rgba(0,0,255,0.45)",
    3: "rgba(0,200,0,0.45)",
    4: "rgba(255,165,0,0.45)",
  };

  // ─── START ────────────────────────────────────────
  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    stopBtn.disabled  = false;

    // reset buffers y temporizador
    waveHist.fill(0);
    labelHist.fill(0);
    currentDom = 0;
    lastTime   = performance.now();
    leftover   = 0;

    // WebSocket con número de hablantes
    const nSpeakers = parseInt(numSpeakersI.value, 10) || 4;
    ws = new WebSocket(`ws://${location.host}/ws/diarize?ns=${nSpeakers}`);
    ws.onopen = () => { socketReady = true; };
    ws.onerror = e => console.error("🔴 WS error", e);
    ws.onmessage = ({ data }) => {
      const { labels, error } = JSON.parse(data);
      if (error) return console.error(error);
      // cuando llega la predicción, actualizamos currentDom
      currentDom = dominantSpeaker(labels);
      const lastSec = new Uint8Array(RES_PER_SEC).fill(currentDom);
      labelHist.set(lastSec, HIST_PTS - RES_PER_SEC);
    };

    // AudioContext + AnalyserNode + Worklet
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioCtx = new AudioContext();

    analyser         = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    bufferLen        = analyser.fftSize;
    dataArray        = new Float32Array(bufferLen);

    await audioCtx.audioWorklet.addModule("/static/audio-processor.js");
    const src = audioCtx.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(audioCtx, "pcm-collector");

    // conectar el grafo: mic → analyser → worklet → destino
    src.connect(analyser);
    analyser.connect(workletNode);
    workletNode.connect(audioCtx.destination);

    // envío de chunks al backend
    workletNode.port.onmessage = ({ data: chunk }) => {
      if (!socketReady) return;
      ws.send(encodeWAV(chunk, audioCtx.sampleRate));
    };

    // bucle de animación (≈60 fps)
    function renderLoop(ts) {
      const dt = ts - lastTime;
      lastTime = ts;

      // punto A) cuántas muestras necesita la historia
      const want = dt * RES_PER_SEC / 1000 + leftover;
      const nNew = Math.floor(want);
      leftover   = want - nNew;

      // punto B) leer muestra actual
      analyser.getFloatTimeDomainData(dataArray);

      // punto C) extraer nNew muestras representativas
      const step = Math.max(1, Math.floor(bufferLen / nNew));
      const freshW = new Float32Array(nNew);
      for (let i = 0, j = 0; j < nNew && i < bufferLen; i += step, j++) {
        freshW[j] = dataArray[i];
      }

      // punto D) desplazar y añadir onda
      waveHist.copyWithin(0, nNew);
      waveHist.set(freshW, HIST_PTS - nNew);

      // punto E) desplazar y añadir etiquetas
      labelHist.copyWithin(0, nNew);
      // rellenamos la zona nueva con currentDom
      labelHist.fill(currentDom, HIST_PTS - nNew, HIST_PTS);

      // punto F) dibujar etiqueta como banda superior
      const w = canvas.width / HIST_PTS;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < HIST_PTS; i++) {
        ctx.fillStyle = speakerColors[labelHist[i]] || "#999";
        ctx.fillRect(i * w, 0, w, 4);
      }

      // punto G) dibujar la onda
      ctx.beginPath();
      for (let i = 0; i < HIST_PTS; i++) {
        const x = (i / (HIST_PTS - 1)) * canvas.width;
        const y = (1 - waveHist[i]) * (canvas.height / 2);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = "#0066cc";
      ctx.lineWidth   = 1;
      ctx.stroke();

      animationId = requestAnimationFrame(renderLoop);
    }
    animationId = requestAnimationFrame(renderLoop);
  });

  // ─── STOP / RESET ─────────────────────────────────
  stopBtn.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    socketReady = false;
    if (workletNode) { workletNode.disconnect(); workletNode = null; }
    if (audioCtx)    { audioCtx.close();    audioCtx    = null; }
    if (animationId) { cancelAnimationFrame(animationId); animationId = null; }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    waveHist.fill(0);
    labelHist.fill(0);

    startBtn.disabled = false;
    stopBtn.disabled  = true;
  });

  // ─── HELPERS ─────────────────────────────────────
  function dominantSpeaker(labels) {
    const cnt = {};
    labels.forEach(l => cnt[l] = (cnt[l] || 0) + 1);
    return +Object.entries(cnt)
      .sort((a, b) => b[1] - a[1])[0][0];
  }

  function encodeWAV(samples, sr) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const view= new DataView(buf);
    write(view, 0, "RIFF"); view.setUint32(4,36+samples.length*2,true);
    write(view, 8, "WAVE"); write(view,12,"fmt "); view.setUint32(16,16,true);
    view.setUint16(20,1,true); view.setUint16(22,1,true);
    view.setUint32(24,sr,true); view.setUint32(28,sr*2,true);
    view.setUint16(32,2,true); view.setUint16(34,16,true);
    write(view,36,"data"); view.setUint32(40,samples.length*2,true);
    let off = 44;
    for (const s of samples) {
      const v = Math.max(-1, Math.min(1, s));
      view.setInt16(off, v<0?v*0x8000:v*0x7FFF, true);
      off += 2;
    }
    return buf;
  }
  function write(view, off, str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(off + i, str.charCodeAt(i));
    }
  }
});
