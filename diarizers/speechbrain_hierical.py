# speaker_diarizer_improved.py

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

# Sólo suprime los warnings más ruidosos de sklearn/speechbrain
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torchaudio
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from speechbrain.pretrained import SpeakerRecognition
from scipy.optimize import linear_sum_assignment

# speaker_diarizer_improved.py


from pathlib import Path
from typing import List, Tuple, Optional
import warnings

# Sólo suprime los warnings más ruidosos de sklearn/speechbrain
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torchaudio
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from speechbrain.pretrained import SpeakerRecognition
from scipy.optimize import linear_sum_assignment



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
    Pipeline de diarización (VAD → embeddings → clustering jerárquico).

    Parámetros
    ----------
    device : {"cpu", "cuda"}
    n_speakers : int, número conocido de hablantes.
    chunk_sec : float, duración en segundos de cada ventana.
    sr : int, frecuencia de muestreo (Silero & ECAPA usan 16 000 Hz).
    max_embeddings : Optional[int], tope de embeddings en memoria.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        n_speakers: int,
        chunk_sec: float = 1.0,
        sr: int = 16_000,
        max_embeddings: Optional[int] = None,
        clustering_kwargs: Optional[dict] = None,


    ) -> None:
        self.device = device
        self.n_speakers = n_speakers
        self.chunk_sec = float(chunk_sec)
        self.sr = int(sr)
        self.max_embeddings = max_embeddings

        # --- Silero VAD ---
        self.vad_model, vad_utils = _load_silero()
        (
            self._get_speech_timestamps,
            _save_audio,
            _read_audio,
            _,
            self._collect_chunks,
        ) = vad_utils
        self.vad_model.to(self.device)

        # --- ECAPA-TDNN (SpeechBrain) ---
        self.spkrec = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=Path("pretrained_models") / "ecapa",
            run_opts={"device": device},
        )
        self.spkrec.eval()





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
    Pipeline de diarización (VAD → embeddings → clustering jerárquico).

    Parámetros
    ----------
    device : {"cpu", "cuda"}
    n_speakers : int, número conocido de hablantes.
    chunk_sec : float, duración en segundos de cada ventana.
    sr : int, frecuencia de muestreo (Silero & ECAPA usan 16 000 Hz).
    max_embeddings : Optional[int], tope de embeddings en memoria.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        n_speakers: int,
        chunk_sec: float = 1.0,
        sr: int = 16_000,
        max_embeddings: Optional[int] = None


    ) -> None:
        self.device = device
        self.n_speakers = n_speakers
        self.chunk_sec = float(chunk_sec)
        self.sr = int(sr)
        self.max_embeddings = max_embeddings

        # --- Silero VAD ---
        self.vad_model, vad_utils = _load_silero()
        (
            self._get_speech_timestamps,
            _save_audio,
            _read_audio,
            _,
            self._collect_chunks,
        ) = vad_utils
        self.vad_model.to(self.device)

        # --- ECAPA-TDNN (SpeechBrain) ---
        self.spkrec = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=Path("pretrained_models") / "ecapa",
            run_opts={"device": device},
        )
        self.spkrec.eval()

        # --- Estado de embeddings y metadata ---
        self._emb_store: List[np.ndarray] = []
        self._meta_store: List[Tuple[int, int]] = []
        self._prev_labels: np.ndarray = np.empty(0, dtype=int)  # ← aquí



    def _vad_activity(
        self, waveform: torch.Tensor
    ) -> Tuple[bool, List[dict]]:
        """Ejecuta Silero-VAD y devuelve (hay_voz, timestamps)."""
        wav = _ensure_mono(waveform).to(self.device).float()
        timestamps = self._get_speech_timestamps(wav, self.vad_model, sampling_rate=self.sr)
        return bool(timestamps), timestamps


    def _match_and_remap(
        self,
        old_labels: np.ndarray,
        cur_old: np.ndarray,
        cur_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1) Hungarian solo para los clusters que aparecen en ambos conjuntos
        uniq_old = np.unique(old_labels)
        uniq_cur = np.unique(cur_old)
        C = np.zeros((len(uniq_old), len(uniq_cur)), int)
        for i, uo in enumerate(uniq_old):
            for j, uc in enumerate(uniq_cur):
                C[i, j] = -np.sum((old_labels == uo) & (cur_old == uc))
        row, col = linear_sum_assignment(C)
        mapping = { uniq_cur[c]: uniq_old[r] for r, c in zip(row, col) }

        # 2) Identity‐mapping para cualquier cluster ID que falte
        #    (asegura que no aparezcan nuevos IDs fuera de 0..n_speakers−1)
        all_ids = np.unique(np.concatenate([cur_old, cur_new]))
        for cid in all_ids:
            mapping.setdefault(cid, cid)

        # 3) Remapeo
        old_mapped = np.array([mapping[l] for l in cur_old])
        new_mapped = np.array([mapping[l] for l in cur_new])
        
        return old_mapped, new_mapped


    def _extract_embeddings(
        self, waveform: torch.Tensor, timestamps: List[dict]
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Ventanas solapadas + batching de embeddings.
        Retorna listas ‘embeds’ y sus metadatos ‘meta’.
        """
        wav = _ensure_mono(waveform).to(self.device)
        window_len = int(self.chunk_sec * self.sr)
        step = window_len // 2  # 50% de solapamiento

        # 1) Recoger *todos* los chunks y metadatos
        chunks: List[torch.Tensor] = []
        meta: List[Tuple[int, int]] = []
        for ts in timestamps:
            for start in range(ts["start"], ts["end"], step):
                end = min(start + window_len, ts["end"])
                chunk = wav[start:end]
                if chunk.shape[-1] < window_len:
                    pad = window_len - chunk.shape[-1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad))
                chunks.append(chunk.unsqueeze(0))  # (1, N)
                meta.append((start, end))

        if not chunks:
            return [], []

        # 2) Batching: concat y un solo encode_batch
        batch = torch.cat(chunks, dim=0)  # (N_chunks, N_samples)
        with torch.no_grad():
            emb_batch = self.spkrec.encode_batch(batch)  # (N_chunks, 192, 1) o similar
        # Aplanar si hace falta y pasar a numpy
        emb_batch = emb_batch.squeeze(-1).cpu().numpy()

        # Convertir a lista de vectores
        embeds = [emb_batch[i] for i in range(emb_batch.shape[0])]
        return embeds, meta
    


    def _cluster_embeddings(
        self,
        new_embeds: List[np.ndarray],
        new_meta: List[Tuple[int,int]]
    ) -> np.ndarray:
        if not new_embeds:
            return np.empty(0, dtype=int)

        # 1) Añadir y podar
        self._emb_store.extend(new_embeds)
        self._meta_store.extend(new_meta)
        if self.max_embeddings is not None and len(self._emb_store) > self.max_embeddings:
            of = len(self._emb_store) - self.max_embeddings
            self._emb_store = self._emb_store[of:]
            self._meta_store = self._meta_store[of:]
            self._prev_labels = self._prev_labels[of:]  # ¡poda sincronizada!

        # 2) Matriz y clustering
        X = normalize(np.vstack(self._emb_store))
        n_avail = X.shape[0]
        n_clust = min(self.n_speakers, n_avail)
        if n_clust < 1:
            return np.zeros(len(new_embeds), dtype=int)

        hc = AgglomerativeClustering(
            n_clusters=n_clust,
            metric="cosine",
            linkage="average"
        )
        labels_all = hc.fit_predict(X)  # etiquetas para todos los embeddings

        # 3) Si no es la primera ronda, emparejamos
        N_old = len(self._prev_labels)
        if N_old > 0 and N_old == (len(labels_all) - len(new_embeds)):
            old_cur = labels_all[:N_old]
            new_cur = labels_all[N_old:]
            old_mapped, new_mapped = self._match_and_remap(self._prev_labels, old_cur, new_cur)
            # 4) Actualizamos prev_labels (para la próxima iteración)
            self._prev_labels = np.concatenate([old_mapped, new_mapped])
            return new_mapped
        else:
            # primera vez o mismatch de tamaños: inicializamos
            self._prev_labels = labels_all
            return labels_all[-len(new_embeds):]
        

    def predict(
        self, waveform: torch.Tensor, sr: Optional[int] = None
    ) -> np.ndarray:
        # --- 0) Mono + normalización global ---
        if sr and sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        waveform = _ensure_mono(waveform)
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform #/ peak

        # 1) VAD
        has_voice, timestamps = self._vad_activity(waveform)
        labels = np.zeros(waveform.shape[-1], dtype=np.int16)
        if not has_voice:
            return labels

        # 2) Embeddings
        embeds, meta = self._extract_embeddings(waveform, timestamps)

        # 3) Clustering
        seg_labels = self._cluster_embeddings(embeds, meta)

        # 4) Construir señal de salida
        for (start, end), lab in zip(meta, seg_labels):
            labels[start:end] = lab + 1

        return labels


    def reset(self) -> None:
        """Vacía buffers de embeddings y metadatos."""
        self._emb_store.clear()
        self._meta_store.clear()
        self._prev_labels = np.empty(0, dtype=int)

        # --- Estado de embeddings y metadata ---
        self._emb_store: List[np.ndarray] = []
        self._meta_store: List[Tuple[int, int]] = []
        self._prev_labels: np.ndarray = np.empty(0, dtype=int)  # ← aquí


    def _vad_activity(
        self, waveform: torch.Tensor
    ) -> Tuple[bool, List[dict]]:
        """Ejecuta Silero-VAD y devuelve (hay_voz, timestamps)."""
        wav = _ensure_mono(waveform).to(self.device).float()
        timestamps = self._get_speech_timestamps(wav, self.vad_model, sampling_rate=self.sr)
        return bool(timestamps), timestamps


    def _match_and_remap(
        self,
        old_labels: np.ndarray,
        cur_old: np.ndarray,
        cur_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1) Hungarian solo para los clusters que aparecen en ambos conjuntos
        uniq_old = np.unique(old_labels)
        uniq_cur = np.unique(cur_old)
        C = np.zeros((len(uniq_old), len(uniq_cur)), int)
        for i, uo in enumerate(uniq_old):
            for j, uc in enumerate(uniq_cur):
                C[i, j] = -np.sum((old_labels == uo) & (cur_old == uc))
        row, col = linear_sum_assignment(C)
        mapping = { uniq_cur[c]: uniq_old[r] for r, c in zip(row, col) }

        # 2) Identity‐mapping para cualquier cluster ID que falte
        #    (asegura que no aparezcan nuevos IDs fuera de 0..n_speakers−1)
        all_ids = np.unique(np.concatenate([cur_old, cur_new]))
        for cid in all_ids:
            mapping.setdefault(cid, cid)

        # 3) Remapeo
        old_mapped = np.array([mapping[l] for l in cur_old])
        new_mapped = np.array([mapping[l] for l in cur_new])
        
        return old_mapped, new_mapped


    def _extract_embeddings(
        self, waveform: torch.Tensor, timestamps: List[dict]
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Ventanas solapadas + batching de embeddings.
        Retorna listas ‘embeds’ y sus metadatos ‘meta’.
        """
        wav = _ensure_mono(waveform).to(self.device)
        window_len = int(self.chunk_sec * self.sr)
        step = window_len // 2  # 50% de solapamiento

        # 1) Recoger *todos* los chunks y metadatos
        chunks: List[torch.Tensor] = []
        meta: List[Tuple[int, int]] = []
        for ts in timestamps:
            for start in range(ts["start"], ts["end"], step):
                end = min(start + window_len, ts["end"])
                chunk = wav[start:end]
                if chunk.shape[-1] < window_len:
                    pad = window_len - chunk.shape[-1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad))
                chunks.append(chunk.unsqueeze(0))  # (1, N)
                meta.append((start, end))

        if not chunks:
            return [], []

        # 2) Batching: concat y un solo encode_batch
        batch = torch.cat(chunks, dim=0)  # (N_chunks, N_samples)
        with torch.no_grad():
            emb_batch = self.spkrec.encode_batch(batch)  # (N_chunks, 192, 1) o similar
        # Aplanar si hace falta y pasar a numpy
        emb_batch = emb_batch.squeeze(-1).cpu().numpy()

        # Convertir a lista de vectores
        embeds = [emb_batch[i] for i in range(emb_batch.shape[0])]
        return embeds, meta
    


    def _cluster_embeddings(
        self,
        new_embeds: List[np.ndarray],
        new_meta: List[Tuple[int,int]]
    ) -> np.ndarray:
        if not new_embeds:
            return np.empty(0, dtype=int)

        # 1) Añadir y podar
        self._emb_store.extend(new_embeds)
        self._meta_store.extend(new_meta)
        if self.max_embeddings is not None and len(self._emb_store) > self.max_embeddings:
            of = len(self._emb_store) - self.max_embeddings
            self._emb_store = self._emb_store[of:]
            self._meta_store = self._meta_store[of:]
            self._prev_labels = self._prev_labels[of:]  # ¡poda sincronizada!

        # 2) Matriz y clustering
        X = normalize(np.vstack(self._emb_store))
        n_avail = X.shape[0]
        n_clust = min(self.n_speakers, n_avail)
        if n_clust < 1:
            return np.zeros(len(new_embeds), dtype=int)

        hc = AgglomerativeClustering(
            n_clusters=n_clust,
            metric="cosine",
            linkage="average"
        )
        labels_all = hc.fit_predict(X)  # etiquetas para todos los embeddings

        # 3) Si no es la primera ronda, emparejamos
        N_old = len(self._prev_labels)
        if N_old > 0 and N_old == (len(labels_all) - len(new_embeds)):
            old_cur = labels_all[:N_old]
            new_cur = labels_all[N_old:]
            old_mapped, new_mapped = self._match_and_remap(self._prev_labels, old_cur, new_cur)
            # 4) Actualizamos prev_labels (para la próxima iteración)
            self._prev_labels = np.concatenate([old_mapped, new_mapped])
            return new_mapped
        else:
            # primera vez o mismatch de tamaños: inicializamos
            self._prev_labels = labels_all
            return labels_all[-len(new_embeds):]
        

    def predict(
        self, waveform: torch.Tensor, sr: Optional[int] = None
    ) -> np.ndarray:
        # --- 0) Mono + normalización global ---
        if sr and sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        waveform = _ensure_mono(waveform)
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform #/ peak

        # 1) VAD
        has_voice, timestamps = self._vad_activity(waveform)
        labels = np.zeros(waveform.shape[-1], dtype=np.int16)
        if not has_voice:
            return labels

        # 2) Embeddings
        embeds, meta = self._extract_embeddings(waveform, timestamps)

        # 3) Clustering
        seg_labels = self._cluster_embeddings(embeds, meta)

        # 4) Construir señal de salida
        for (start, end), lab in zip(meta, seg_labels):
            labels[start:end] = lab + 1

        return labels


    def reset(self) -> None:
        """Vacía buffers de embeddings y metadatos."""
        self._emb_store.clear()
        self._meta_store.clear()
        self._prev_labels = np.empty(0, dtype=int)
