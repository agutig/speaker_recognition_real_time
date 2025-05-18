import io
import json
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.speaker_diarizer import SpeakerDiarizer  # ajusta si tu paquete cambia

app = FastAPI()

# ──────────────────────────────────────────────────────────────
# 1)  ESTÁTICOS → /static
# ──────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR, html=False),
    name="static",
)

# ──────────────────────────────────────────────────────────────
# 2)  Índice en la raíz
# ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


# ──────────────────────────────────────────────────────────────
# 3)  Gestión de clientes WebSocket
# ──────────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self) -> None:
        self.active: dict[WebSocket, SpeakerDiarizer] = {}

    async def connect(self, ws: WebSocket, n_speakers: int = 4) -> None:
        await ws.accept()
        # Crea un diarizador nuevo con ese número de clusters
        self.active[ws] = SpeakerDiarizer(
            device="cpu",
            n_speakers=n_speakers,
            chunk_sec=1
        )

    def disconnect(self, ws: WebSocket) -> None:
        self.active.pop(ws, None)
        
    async def receive_and_diarize(self, ws: WebSocket, data: bytes) -> None:
        diar = self.active[ws]
        try:
            # El cliente envía WAV 16 kHz mono
            waveform, sr = torchaudio.load(io.BytesIO(data), format="wav")
            labels = diar.predict(waveform.squeeze(0), sr)

            await ws.send_json({"labels": labels.tolist()})
        except Exception as e:
            import traceback, sys

            traceback.print_exc(file=sys.stderr)
            await ws.send_json({"error": str(e)})


manager = ConnectionManager()


# ──────────────────────────────────────────────────────────────
# 4)  Endpoint WebSocket
# ──────────────────────────────────────────────────────────────
@app.websocket("/ws/diarize")
async def ws_diarize(ws: WebSocket):
    # Extraer query string “ns”
    query = ws.scope.get("query_string", b"").decode()
    params = dict(pair.split("=") for pair in query.split("&") if pair)
    n = int(params.get("ns", 4))

    # Conectar con ese número de clusters
    await manager.connect(ws, n)

    try:
        while True:
            data = await ws.receive_bytes()
            await manager.receive_and_diarize(ws, data)
    except WebSocketDisconnect:
        manager.disconnect(ws)