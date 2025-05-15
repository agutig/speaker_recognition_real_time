# speaker_diarizer.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import warnings

# Suprime los FutureWarnings y UserWarnings de SpeechBrain, sklearn, etc.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torchaudio
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from speechbrain.pretrained import SpeakerRecognition


def _load_silero() -> Tuple[torch.nn.Module, dict]:
    """Carga Silero-VAD vía torch.hub y devuelve (modelo, utils)."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
        force_reload=False,
    )
    return model, utils


def _ensure_mono(wav: torch.Tensor) -> torch.Tensor:
    """Si el tensor llega (C, N) → lo convierte a (N,) promediando canales."""
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    return wav


class SpeakerDiarizer:
    """
    Pipeline online de diarización (VAD → embeddings → K-Means).

    Parámetros
    ----------
    device : {"cpu", "cuda"}
        Dispositivo donde se correrán los modelos.
    n_speakers : int
        Número conocido de hablantes en la grabación.
    chunk_sec : float
        Duración, en segundos, de cada ventana fija para extraer embeddings.
    sr : int
        Frecuencia de muestreo objetivo (Silero y ECAPA usan 16 000 Hz).
    max_embeddings : int | None
        Número máximo de embeddings a mantener en memoria.
        `None` = sin límite.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        n_speakers: int,
        chunk_sec: float = 1.0,
        sr: int = 16_000,
        max_embeddings: Optional[int] = None,
    ) -> None:
        self.device = device
        self.n_speakers = n_speakers
        self.chunk_sec = float(chunk_sec)
        self.sr = int(sr)
        self.max_embeddings = max_embeddings

        # -------------------- Silero VAD --------------------
        self.vad_model, vad_utils = _load_silero()
        (
            self._get_speech_timestamps,
            _save_audio,
            _read_audio,
            _,
            self._collect_chunks,
        ) = vad_utils
        self.vad_model.to(self.device)

        # -------------------- ECAPA-TDNN --------------------
        self.spkrec = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=Path("pretrained_models") / "ecapa",
            run_opts={"device": device},
        )
        self.spkrec.eval()

        # -------------------- K-Means -----------------------
        self.clusterer = KMeans(
            n_clusters=self.n_speakers,
            random_state=0,
            n_init="auto",
        )

        # Almacenes de estado (crecen con el tiempo)
        self._emb_store: List[np.ndarray] = []
        self._meta_store: List[Tuple[int, int]] = []  # (start, end) en muestras

    def _vad_activity(
        self, waveform: torch.Tensor
    ) -> Tuple[bool, List[dict]]:
        """Ejecuta Silero-VAD y devuelve (hay_voz, timestamps)."""
        wav = _ensure_mono(waveform).to(self.device).float()
        timestamps = self._get_speech_timestamps(
            wav, self.vad_model, sampling_rate=self.sr
        )
        return bool(timestamps), timestamps

    def _extract_embeddings(
        self, waveform: torch.Tensor, timestamps: List[dict]
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Divide cada segmento VAD en ventanas fijas y calcula embeddings ECAPA.

        Devuelve
        --------
        embeds : list[np.ndarray]
            Vectores de 192 dimensiones, uno por ventana.
        meta : list[tuple[int, int]]
            (start, end) en muestras de cada ventana.
        """
        wav = _ensure_mono(waveform).to(self.device)
        step = int(self.chunk_sec * self.sr)

        embeds, meta = [], []
        for ts in timestamps:
            seg_start, seg_end = ts["start"], ts["end"]
            for s in range(seg_start, seg_end, step):
                e = s + step
                chunk = wav[s:e]
                if chunk.shape[-1] < step:
                    pad = step - chunk.shape[-1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad))
                chunk = chunk.unsqueeze(0)  # (1, N)

                with torch.no_grad():
                    emb = (
                        self.spkrec.encode_batch(chunk)
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
                embeds.append(emb)
                meta.append((s, min(e, seg_end)))

        return embeds, meta

    def _cluster_embeddings(self, new_embeds: List[np.ndarray]) -> np.ndarray:
        if not new_embeds:
            return np.empty(0, dtype=int)

        # 1) Añade los nuevos embeddings a la memoria
        self._emb_store.extend(new_embeds)
        if (
            self.max_embeddings is not None
            and len(self._emb_store) > self.max_embeddings
        ):
            overflow = len(self._emb_store) - self.max_embeddings
            self._emb_store = self._emb_store[overflow:]
            self._meta_store = self._meta_store[overflow:]

        # 2) Prepara la matriz X
        X = normalize(np.vstack(self._emb_store))

        # 3) Ajusta dinámicamente el número de clusters
        n_available = X.shape[0]
        n_clusters = min(self.n_speakers, n_available)
        if n_clusters < 1:
            # No hay samples: devolvemos etiquetas vacías
            return np.empty(len(new_embeds), dtype=int)

        # 4) Crea y entrena un KMeans con el número correcto de clusters
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init="auto",
        )
        clusterer.fit(X)
        labels = clusterer.labels_

        # 5) Devuelve sólo las etiquetas de los nuevos embeddings
        return labels[-len(new_embeds):]


    def predict(
        self, waveform: torch.Tensor, sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Procesa el audio completo y devuelve un vector de etiquetas por muestra.

        Etiquetas
        ---------
        0  →  fondo  
        1, 2, …  →  hablantes (cluster_id + 1)
        """
        # Resample a 16 kHz si fuera necesario
        if sr and sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)

        waveform = _ensure_mono(waveform)

        # 1. VAD
        has_voice, timestamps = self._vad_activity(waveform)
        label_signal = np.zeros(waveform.shape[-1], dtype=np.int16)
        if not has_voice:
            return label_signal  # todo es fondo

        # 2. Embeddings
        embeds, meta = self._extract_embeddings(waveform, timestamps)

        # 3. Clustering
        seg_labels = self._cluster_embeddings(embeds)

        # 4. Construir señal de salida
        for (s, e), lab in zip(meta, seg_labels):
            # lab en [0..n_speakers-1], convertimos a [1..n_speakers]
            label_signal[s:e] = lab + 1

        # Guardamos meta por si alguien la quiere luego
        self._meta_store.extend(meta)

        return label_signal

    def reset(self) -> None:
        """Vacía buffers y reinicia el clusterer (útil en streaming)."""
        self._emb_store.clear()
        self._meta_store.clear()
        self.clusterer = KMeans(
            n_clusters=self.n_speakers,
            random_state=0,
            n_init="auto",
        )
