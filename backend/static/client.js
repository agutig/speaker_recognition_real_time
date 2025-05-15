// static/client.js
console.log("✔ client.js (AudioWorklet + 5 s historial dinámico) cargado");

window.addEventListener("DOMContentLoaded", () => {
  // ─── DOM ─────────────────────────────────────────
  const startBtn     = document.getElementById("start");
  const stopBtn      = document.getElementById("stop");
  const numSpeakersI = document.getElementById("numSpeakers");
  const wfCanvas     = document.getElementById("waveformChart");
  const wfCtx        = wfCanvas.getContext("2d");
  const tlCtx        = document.getElementById("timelineChart").getContext("2d");

  // ─── ESTADO GLOBAL ────────────────────────────────
  let audioCtx, analyser, dataArray, bufferLen;
  let ws, socketReady = false;
  let workletNode, animationId;
  let lastTimestamp = 0, leftover = 0;

  const WINDOW_SEC  = 5;                       // segundos en pantalla
  const RES_PER_SEC = 800;                     // pts/s de resolución
  const HIST_POINTS = WINDOW_SEC * RES_PER_SEC;// 4000 puntos totales

  // historial circular de 5 s (inicialmente todo ceros)
  let waveformHist = new Float32Array(HIST_POINTS).fill(0);

  const timelineData = [];
  const speakerColors = {
    0: "#ccc",
    1: "rgba(255,0,0,0.45)",
    2: "rgba(0,0,255,0.45)",
    3: "rgba(0,200,0,0.45)",
    4: "rgba(255,165,0,0.45)",
  };

  // ─── Chart.js – Timeline (últimos 5 s) ───────────
  const timelineChart = new Chart(tlCtx, {
    type: "bar",
    data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderWidth: 0 }] },
    options: {
      animation: false,
      scales: {
        x: { title: { display: true, text: "Hace (s)" }, grid: { display: false } },
        y: {
          beginAtZero: true,
          max: Object.keys(speakerColors).length,
          ticks: { stepSize: 1, callback: v => (v === 0 ? "Sil" : `Spk ${v}`) },
        },
      },
      plugins: { legend: { display: false } },
    },
  });

  // ─── BOTÓN START ──────────────────────────────────
  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    stopBtn.disabled  = false;

    // 1) Reinicializar el buffer de 5 s
    waveformHist.fill(0);
    lastTimestamp = performance.now();
    leftover = 0;

    // 2) Abrir WebSocket con nº de hablantes
    const nSpeakers = parseInt(numSpeakersI.value, 10) || 4;
    ws = new WebSocket(`ws://${location.host}/ws/diarize?ns=${nSpeakers}`);
    ws.onopen    = () => { socketReady = true; };
    ws.onerror   = e => console.error("🔴 WS error", e);
    ws.onmessage = ({ data }) => {
      const { labels, error } = JSON.parse(data);
      if (error) return console.error(error);
      const dom = dominantSpeaker(labels);
      timelineData.push(dom);
      if (timelineData.length > WINDOW_SEC) timelineData.shift();
      timelineChart.data.labels = timelineData.map((_, i, a) => `${-(a.length - i - 1)}s`);
      timelineChart.data.datasets[0].data            = timelineData;
      timelineChart.data.datasets[0].backgroundColor =
        timelineData.map(id => speakerColors[id] || "#999");
      timelineChart.update("none");
    };

    // 3) Crear AudioContext + Analyser + Worklet
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioCtx = new AudioContext();

    // 3a) Analizador para onda fluida
    analyser         = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    bufferLen        = analyser.fftSize;
    dataArray        = new Float32Array(bufferLen);

    // 3b) Worklet para chunks de 1 s
    await audioCtx.audioWorklet.addModule("/static/audio-processor.js");
    const sourceNode = audioCtx.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(audioCtx, "pcm-collector");

    // 3c) Conectar el grafo: mic → analyser → worklet → salida
    sourceNode.connect(analyser);
    analyser.connect(workletNode);
    workletNode.connect(audioCtx.destination);

    workletNode.port.onmessage = ({ data: chunk }) => {
      if (!socketReady) return;
      ws.send(encodeWAV(chunk, audioCtx.sampleRate));
    };

    // 4) Bucle de renderizado para onda suave + 5 s de historial
    function renderLoop(ts) {
      // A) Tiempo transcurrido desde el último frame
      const dt = ts - lastTimestamp;
      lastTimestamp = ts;

      // B) Determinar cuántos puntos nuevos necesitamos
      const neededF = (dt * RES_PER_SEC) / 1000 + leftover;
      const nNew    = Math.floor(neededF);
      leftover      = neededF - nNew;

      // C) Leer señal actual
      analyser.getFloatTimeDomainData(dataArray);

      // D) Extraer nNew muestras representativas
      const step  = Math.max(1, Math.floor(bufferLen / nNew));
      const fresh = new Float32Array(nNew);
      for (let i = 0, j = 0; j < nNew && i < bufferLen; i += step, j++) {
        fresh[j] = dataArray[i];
      }

      // E) Desplazar el historial y anexar fresh al final
      waveformHist.copyWithin(0, nNew);
      waveformHist.set(fresh, HIST_POINTS - nNew);

      // F) Dibujar todo el buffer de 5 s
      wfCtx.clearRect(0, 0, wfCanvas.width, wfCanvas.height);
      wfCtx.beginPath();
      for (let i = 0; i < HIST_POINTS; i++) {
        const x = (i / (HIST_POINTS - 1)) * wfCanvas.width;
        const y = (1 - waveformHist[i]) * (wfCanvas.height / 2);
        i === 0 ? wfCtx.moveTo(x, y) : wfCtx.lineTo(x, y);
      }
      wfCtx.strokeStyle = "#0066cc";
      wfCtx.lineWidth   = 1;
      wfCtx.stroke();

      animationId = requestAnimationFrame(renderLoop);
    }
    animationId = requestAnimationFrame(renderLoop);
  });

  // ─── BOTÓN STOP / RESET ──────────────────────────
  stopBtn.addEventListener("click", () => {
    console.log("⏹️ Detener / Reset");
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    socketReady = false;
    if (workletNode) { workletNode.disconnect(); workletNode = null; }
    if (audioCtx)    { audioCtx.close();    audioCtx    = null; }
    if (animationId) { cancelAnimationFrame(animationId); animationId = null; }

    // limpiar onda
    wfCtx.clearRect(0, 0, wfCanvas.width, wfCanvas.height);
    waveformHist.fill(0);

    // limpiar timeline
    timelineData.length = 0;
    timelineChart.data.labels           = [];
    timelineChart.data.datasets[0].data = [];
    timelineChart.update("none");

    startBtn.disabled = false;
    stopBtn.disabled  = true;
  });

  // ─── HELPERS ─────────────────────────────────────
  function dominantSpeaker(labels) {
    const cnt = {};
    labels.forEach(l => {
      cnt[l] = (cnt[l] || 0) + 1;
    });
    return +Object.entries(cnt)
      .sort((a, b) => b[1] - a[1])[0][0];
  }

  function encodeWAV(samples, sr) {
    const buf  = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buf);
    write(view, 0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true);
    write(view, 8, "WAVE"); write(view,12,"fmt "); view.setUint32(16,16, true);
    view.setUint16(20,1, true); view.setUint16(22,1, true);
    view.setUint32(24,sr,true); view.setUint32(28,sr*2,true);
    view.setUint16(32,2,true); view.setUint16(34,16,true);
    write(view,36,"data"); view.setUint32(40, samples.length*2, true);
    let off = 44;
    for (const s of samples) {
      const v = Math.max(-1, Math.min(1, s));
      view.setInt16(off, v < 0 ? v * 0x8000 : v * 0x7FFF, true);
      off += 2;
    }
    return buf;
  }

  function write(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }
});
