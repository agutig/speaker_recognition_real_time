/* client.js – versión AudioWorklet */
console.log("✔ client.js (AudioWorklet) cargado");

window.addEventListener("DOMContentLoaded", () => {
  //----------------------------------------------------------------
  // DOM y Chart.js
  //----------------------------------------------------------------
  const startBtn = document.getElementById("start");
  const stopBtn  = document.getElementById("stop");
  const wfCtx    = document.getElementById("waveformChart").getContext("2d");
  const tlCtx    = document.getElementById("timelineChart").getContext("2d");
  const numSpeakersI = document.getElementById("numSpeakers");
  

  // Estado global
  let audioCtx, ws, socketReady = false;
  const MAX_CHUNKS   = 10;
  const timelineData = [];
  let   waveformData = [];
  let waveformHist = new Float32Array(0); 
  let workletNode;

  const WINDOW_SEC     = 5;        // ⬅️  Ventana visible
  const CHUNK_SEC      = 1;        //     Duración de cada mensaje Worklet
  const RES_PER_SEC    = 800;      //     Puntos por segundo en waveform
  const POINTS_VISIBLE = WINDOW_SEC * RES_PER_SEC;   // 4 000 puntos

  // Colores por hablante
  const speakerColors = {
    0:"#ccc", 1:"rgba(255,0,0,.45)", 2:"rgba(0,0,255,.45)",
    3:"rgba(0,200,0,.45)", 4:"rgba(255,165,0,.45)"
  };

  //------------------ Chart – Waveform ------------------
  const waveformChart = new Chart(wfCtx,{
    type:"line",
    data:{ labels:[], datasets:[{
      label:"Onda", data:waveformData,
      pointRadius:0, borderWidth:2, borderColor:"#0066cc", fill:false
    }]},
    options:{ animation:false, scales:{
      x:{display:false}, y:{min:-1,max:1,ticks:{stepSize:0.5}}
    }}
  });

  //------------------ Chart – Timeline ------------------
  const timelineChart = new Chart(tlCtx,{
    type:"bar",
    data:{ labels:[], datasets:[{data:[],backgroundColor:[],borderWidth:0}]},
    options:{ animation:false, scales:{
      x:{title:{display:true,text:"Hace (s)"},grid:{display:false}},
      y:{beginAtZero:true,max:4,ticks:{stepSize:1,
          callback:v=>v===0?"Sil":`Spk ${v}`}}
    }, plugins:{legend:{display:false}} }
  });

  //----------------------------------------------------------------
  // BOTÓN INICIAR
  //----------------------------------------------------------------
  startBtn.addEventListener("click", async () => {

    startBtn.disabled = true;
    stopBtn.disabled  = false;

    const nSpeakers = parseInt(numSpeakersI.value, 10) || 4;
    console.log("▶️ Iniciar con", nSpeakers, "hablantes");


    //---------------- 1. WebSocket ----------------
    ws = new WebSocket(
      `ws://${location.host}/ws/diarize?ns=${nSpeakers}`
    );
    ws.onopen  = () => { console.log("🟢 WS abierto"); socketReady = true; };
    ws.onerror = e  => console.error("🔴 WS error:", e);

    //---------------- 2. AudioContext + Worklet ----
    const stream = await navigator.mediaDevices.getUserMedia({ audio:true });
    audioCtx     = new AudioContext();                // 44 100 / 48 000 Hz
    await audioCtx.audioWorklet.addModule("/static/audio-processor.js");

    const source  = audioCtx.createMediaStreamSource(stream);
    const worklet = new AudioWorkletNode(audioCtx, "pcm-collector");
    source.connect(worklet);
    worklet.connect(audioCtx.destination);            // ↓ silencio

    // Mensajes de 1 s desde el Worklet
    worklet.port.onmessage = ({ data: chunk }) => {
      if (!socketReady) return;
      ws.send(encodeWAV(chunk, audioCtx.sampleRate));
      appendWaveform(chunk);
      drawWaveform();                    // gráfica
    };

    //---------------- 3. Etiquetas ↓ WebSocket ----
    ws.onmessage = ({ data }) => {
      const { labels, error } = JSON.parse(data);
      if (error) { console.error(error); return; }

      const dom = dominantSpeaker(labels);
      timelineData.push(dom);
      if (timelineData.length > MAX_CHUNKS) timelineData.shift();

      timelineChart.data.labels =
        timelineData.map((_,i,a)=>`${-(a.length-i-1)}s`);
      timelineChart.data.datasets[0].data = timelineData;
      timelineChart.data.datasets[0].backgroundColor =
        timelineData.map(id=>speakerColors[id]||"#999");
      timelineChart.update("none");
      updateTimeline(dom);
    };

  });

  // ─── STOP / RESET ─────────────────────────────────
  stopBtn.addEventListener("click", () => {
    console.log("⏹️ Botón Detener pulsado — reiniciando todo");
    // 1) Cerrar WS
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    socketReady = false;

    // 2) Detener AudioWorklet y AudioContext
    if (workletNode) {
      workletNode.disconnect();
      workletNode = null;
    }
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }

    // 5) Restaurar botones
    startBtn.disabled = false;
    stopBtn.disabled  = true;
  });


  //----------------------------------------------------------------
  // HELPERS
  //----------------------------------------------------------------

  function appendWaveform(chunk) {
    // 1 s de audio → decimar a RES_PER_SEC (=800) puntos
    const factor = Math.max(1, Math.floor(chunk.length / RES_PER_SEC));
    const decimated = new Float32Array(RES_PER_SEC);
    for (let i = 0, j = 0; j < RES_PER_SEC; i += factor, j++) {
      decimated[j] = chunk[i];
    }
  
    // Añadir y recortar para mantener 5 s
    const tmp = new Float32Array(waveformHist.length + decimated.length);
    tmp.set(waveformHist);
    tmp.set(decimated, waveformHist.length);
    // si excede los 5 s → descartar el principio
    if (tmp.length > POINTS_VISIBLE) {
      waveformHist = tmp.slice(tmp.length - POINTS_VISIBLE);
    } else {
      waveformHist = tmp;
    }
  }

  function drawWaveform() {
    const ctx = wfCtx;
    const W   = ctx.canvas.width;
    const H   = ctx.canvas.height;
    ctx.clearRect(0, 0, W, H);
  
    const L = waveformHist.length;
    if (L < 2) return;
  
    ctx.beginPath();
    for (let i = 0; i < L; i++) {
      const x = (i / (L - 1)) * W;
      const y = (1 - waveformHist[i]) * (H / 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#0066cc";
    ctx.lineWidth   = 1;
    ctx.stroke();
  }

  function updateTimeline(dominant) {
    timelineData.push(dominant);
    if (timelineData.length > WINDOW_SEC) timelineData.shift();
  
    timelineChart.data.labels =
      timelineData.map((_, i, a) => `${-(a.length - i - 1)} s`);
    timelineChart.data.datasets[0].data            = timelineData;
    timelineChart.data.datasets[0].backgroundColor =
      timelineData.map(id => speakerColors[id] || "#999");
    timelineChart.update("none");
  }
  


  function dominantSpeaker(labels){
    const c={}; labels.forEach(l=>c[l]=(c[l]||0)+1);
    return +Object.entries(c).sort((a,b)=>b[1]-a[1])[0][0];
  }

  function updateWaveform(chunk) {
    console.log("📈 updateWaveform() llamado con chunk.length:", chunk.length);
  
    const factor = Math.max(1, Math.floor(chunk.length / 800));
    console.log("📈 factor de decimado calculado:", factor);
  
    waveformData = [];
    for (let i = 0; i < chunk.length; i += factor) {
      waveformData.push(chunk[i]);
    }
  
    console.log("📈 waveformData.length después del decimado:", waveformData.length);
    console.log("📈 primeros 10 valores de waveformData:", waveformData.slice(0, 10));
  
    waveformChart.data.labels = waveformData.map((_, i) => i);
    waveformChart.data.datasets[0].data = waveformData;  // <- ¿estaba esto antes?
    waveformChart.update("none");
  }
  
  function encodeWAV(samples,sr){
    const buf=new ArrayBuffer(44+samples.length*2);
    const v=new DataView(buf);
    write(v,0,"RIFF"); v.setUint32(4,36+samples.length*2,true);
    write(v,8,"WAVE"); write(v,12,"fmt "); v.setUint32(16,16,true);
    v.setUint16(20,1,true); v.setUint16(22,1,true);
    v.setUint32(24,sr,true); v.setUint32(28,sr*2,true);
    v.setUint16(32,2,true); v.setUint16(34,16,true);
    write(v,36,"data"); v.setUint32(40,samples.length*2,true);
    let o=44;
    for(const s of samples){
      const val = Math.max(-1,Math.min(1,s));
      v.setInt16(o, val<0 ? val*0x8000 : val*0x7FFF, true); o+=2;
    }
    return buf;
  }
  function write(view,off,str){ for(let i=0;i<str.length;i++)
      view.setUint8(off+i,str.charCodeAt(i)); }


});
