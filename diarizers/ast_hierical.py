# speaker_diarizer_improved.py

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

# Suprime los warnings de sklearn/transformers
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torchaudio
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment

from transformers import (
    AutoProcessor,
    ASTConfig,
    ASTForAudioClassification,
)


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
    Pipeline de diarización (VAD → embeddings AST → clustering jerárquico).

    Fija internamente los modelos:
      - Base processor/config: MIT/ast-finetuned-audioset-10-10-0.4593
      - Fine-tuned head:      agutig/AST_diarizer

    Parámetros
    ----------
    device : {"cpu","cuda"}
    num_labels : int, n.º de hablantes con que fine-tunaste (p.ej. 50)
    chunk_sec : float, segundos por ventana (1.0)
    sr : int, muestreo (16000)
    max_embeddings : Optional[int], tope de vectores acumulados
    clustering_kwargs : dict, args para AgglomerativeClustering
    """

    # IDs fijos de HuggingFace
    _BASE_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
    _FT_ID   = "agutig/AST_diarizer"

    def __init__(
        self,
        *,
        device: str = "cpu",
        n_speakers,
        chunk_sec: float = 1.0,
        sr: int = 16_000,
        max_embeddings: Optional[int] = None,
        clustering_kwargs: Optional[dict] = None,
    ) -> None:
        self.device = device
        self.chunk_sec = float(chunk_sec)
        self.sr = int(sr)
        self.max_embeddings = max_embeddings
        self.n_speakers = n_speakers
        self.clustering_kwargs = clustering_kwargs or {}

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

        # --- AST Processor + Modelo ---
        # 1) Processor del checkpoint base
        self.processor = AutoProcessor.from_pretrained(self._BASE_ID)

        # 2) Config original adaptada a tu número de labels
        config = ASTConfig.from_pretrained(
            self._BASE_ID,
            num_labels=50
        )

        # 3) Carga tu modelo fine-tuned con esa config
        self.embed_model = ASTForAudioClassification.from_pretrained(
            self._FT_ID,
            config=config
        )
        self.embed_model.to(self.device).eval()

        # Estado interno
        self.reset()

    def reset(self) -> None:
        """Vacía buffers de embeddings y metadatos."""
        self._emb_store: List[np.ndarray] = []
        self._meta_store: List[Tuple[int, int]] = []
        self._prev_labels: np.ndarray = np.empty(0, dtype=int)

    def _vad_activity(
        self, waveform: torch.Tensor
    ) -> Tuple[bool, List[dict]]:
        wav = _ensure_mono(waveform).to(self.device).float()
        timestamps = self._get_speech_timestamps(
            wav, self.vad_model, sampling_rate=self.sr, threshold=0.5
        )
        return bool(timestamps), timestamps

    def _match_and_remap(
        self,
        old_labels: np.ndarray,
        cur_old: np.ndarray,
        cur_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        uniq_old = np.unique(old_labels)
        uniq_cur = np.unique(cur_old)
        C = np.zeros((len(uniq_old), len(uniq_cur)), int)
        for i, uo in enumerate(uniq_old):
            for j, uc in enumerate(uniq_cur):
                C[i, j] = -np.sum((old_labels == uo) & (cur_old == uc))
        row, col = linear_sum_assignment(C)
        mapping = { uniq_cur[c]: uniq_old[r] for r, c in zip(row, col) }
        all_ids = np.unique(np.concatenate([cur_old, cur_new]))
        for cid in all_ids:
            mapping.setdefault(cid, cid)
        old_m = np.array([mapping[l] for l in cur_old])
        new_m = np.array([mapping[l] for l in cur_new])
        return old_m, new_m

    def _extract_embeddings(
        self, waveform: torch.Tensor, timestamps: List[dict]
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extrae embeddings en mini-batches para ahorro de memoria.
        Retorna listas de vectores `embeds` y sus metadatos `meta`.
        """
        # 1) Mono y a dispositivo
        wav = _ensure_mono(waveform).to(self.device)
        win = int(self.chunk_sec * self.sr)
        step = win // 2

        # 2) Fragmentar en chunks solapados y recolectar metadatos
        chunks: List[torch.Tensor] = []
        meta:   List[Tuple[int,int]] = []
        for ts in timestamps:
            for st in range(ts["start"], ts["end"], step):
                en = min(st + win, ts["end"])
                chunk = wav[st:en]
                if chunk.shape[-1] < win:
                    chunk = torch.nn.functional.pad(chunk, (0, win - chunk.shape[-1]))
                chunks.append(chunk.unsqueeze(0))  # (1, samples)
                meta.append((st, en))

        if not chunks:
            return [], []

        # 3) Mini-batching para no saturar GPU
        BATCH_SIZE = 8  # ajusta según tu VRAM disponible
        embeds: List[np.ndarray] = []

        for i in range(0, len(chunks), BATCH_SIZE):
            # Crear sub-batch de tamaño <= BATCH_SIZE
            sub_batch = torch.cat(chunks[i : i + BATCH_SIZE], dim=0)  # (b, samples)

            # Preprocesar con el processor de HuggingFace
            inputs = self.processor(
                sub_batch.cpu().numpy(),
                sampling_rate=self.sr,
                return_tensors="pt",
                padding=True,
            )
            # Mover tensores a GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inferencia sin gradiente
            with torch.no_grad():
                out = self.embed_model(**inputs)
                logits = out.logits.cpu().numpy()

            # Añadir cada vector al listado final
            for vec in logits:
                embeds.append(vec)

        return embeds, meta


    def _cluster_embeddings(
        self,
        new_embeds: List[np.ndarray],
        new_meta: List[Tuple[int,int]]
    ) -> np.ndarray:
        if not new_embeds:
            return np.empty(0, dtype=int)

        self._emb_store.extend(new_embeds)
        self._meta_store.extend(new_meta)
        if self.max_embeddings and len(self._emb_store) > self.max_embeddings:
            of = len(self._emb_store) - self.max_embeddings
            self._emb_store = self._emb_store[of:]
            self._meta_store = self._meta_store[of:]
            self._prev_labels = self._prev_labels[of:]

        X = normalize(np.vstack(self._emb_store))

        n_avail    = X.shape[0]
        n_clusters = min(self.n_speakers, n_avail)

        hc = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        labels_all = hc.fit_predict(X)

        N_old = len(self._prev_labels)
        if N_old > 0 and N_old == (len(labels_all) - len(new_embeds)):
            old_c = labels_all[:N_old]
            new_c = labels_all[N_old:]
            old_m, new_m = self._match_and_remap(self._prev_labels, old_c, new_c)
            self._prev_labels = np.concatenate([old_m, new_m])
            return new_m
        else:
            self._prev_labels = labels_all
            return labels_all[-len(new_embeds):]

    def predict(
        self, waveform: torch.Tensor, sr: Optional[int] = None
    ) -> np.ndarray:
        if sr and sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        waveform = _ensure_mono(waveform)
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform  # opcional: / peak

        has_voice, timestamps = self._vad_activity(waveform)
        labels = np.zeros(waveform.shape[-1], dtype=np.int16)
        if not has_voice:
            return labels

        embeds, meta = self._extract_embeddings(waveform, timestamps)
        seg_labels = self._cluster_embeddings(embeds, meta)

        for (st, en), lab in zip(meta, seg_labels):
            labels[st:en] = lab + 1

        return labels
