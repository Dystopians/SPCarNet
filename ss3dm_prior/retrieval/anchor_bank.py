"""Clean-latent anchor bank for CarNet_v0 / Phase 4 / A3 (retrieval-augmented decoding).

Maintains a collection of ``(embedding, latent)`` records gathered by
forwarding the training set in eval mode. At inference time the decoder
queries this bank with a corrupted-input embedding and receives the top-K
nearest clean latents, which are re-injected as extra cross-attention
context tokens. This gives the model an explicit shape memory instead of
requiring everything to live in the backbone's parameters.

The bank is intentionally source-agnostic (D3 future-proof): records come
from an iterable of ``(embedding, latent, metadata)`` tuples, so the same
object supports:
  - MeshFleet-only bank for CarNet_v0 (the default)
  - mixed ShapeNetCore + MeshFleet bank for CarNet_v0.1 (deferred)
  - external-dataset anchor banks during transfer evaluation

FAISS is preferred; if unavailable the module falls back to a pure-PyTorch
cosine search that is fine for < 10k entries (our MeshFleet training set
is 1456 samples after val split).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:  # noqa: BLE001 — any import failure falls through to fallback
    faiss = None  # type: ignore
    _HAS_FAISS = False


@dataclass
class AnchorRecord:
    embedding: np.ndarray
    latent: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class AnchorBank:
    """L2-normalised cosine anchor bank.

    Expected usage:
        bank = AnchorBank(embedding_dim=256, latent_dim=512)
        bank.update(iterable_of_records)
        top_embeddings, top_latents, top_sims = bank.query(batch_embeddings, k=3)

    All persisted tensors are contiguous float32 numpy arrays.
    """

    def __init__(self, embedding_dim: int, latent_dim: int) -> None:
        self.embedding_dim = int(embedding_dim)
        self.latent_dim = int(latent_dim)
        self._embeddings: np.ndarray = np.zeros((0, self.embedding_dim), dtype=np.float32)
        self._latents: np.ndarray = np.zeros((0, self.latent_dim), dtype=np.float32)
        self._metadata: list[dict[str, Any]] = []
        self._index = None  # lazy-built

    # ------------------------------------------------------------------
    # Construction / persistence
    # ------------------------------------------------------------------
    def update(self, records: Iterable[AnchorRecord]) -> None:
        """Replace the bank contents from an iterable of records."""
        embeddings: list[np.ndarray] = []
        latents: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []
        for record in records:
            emb = np.asarray(record.embedding, dtype=np.float32).reshape(-1)
            if emb.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"AnchorBank expects {self.embedding_dim}-d embeddings; got {emb.shape[0]}"
                )
            lat = np.asarray(record.latent, dtype=np.float32).reshape(-1)
            if lat.shape[0] != self.latent_dim:
                raise ValueError(
                    f"AnchorBank expects {self.latent_dim}-d latents; got {lat.shape[0]}"
                )
            # L2-normalise embeddings for cosine retrieval.
            emb_norm = emb / max(float(np.linalg.norm(emb)), 1e-8)
            embeddings.append(emb_norm)
            latents.append(lat)
            metas.append(dict(record.metadata))
        if embeddings:
            self._embeddings = np.stack(embeddings, axis=0).astype(np.float32)
            self._latents = np.stack(latents, axis=0).astype(np.float32)
        else:
            self._embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)
            self._latents = np.zeros((0, self.latent_dim), dtype=np.float32)
        self._metadata = metas
        self._index = None  # invalidate

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self._embeddings,
            latents=self._latents,
            metadata=np.asarray(
                [
                    "" if not m else __import__("json").dumps(m, sort_keys=True)
                    for m in self._metadata
                ],
                dtype=object,
            ),
            embedding_dim=np.asarray(self.embedding_dim, dtype=np.int32),
            latent_dim=np.asarray(self.latent_dim, dtype=np.int32),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AnchorBank":
        path = Path(path).expanduser().resolve()
        with np.load(path, allow_pickle=True) as handle:
            embedding_dim = int(handle["embedding_dim"].item())
            latent_dim = int(handle["latent_dim"].item())
            embeddings = np.asarray(handle["embeddings"], dtype=np.float32)
            latents = np.asarray(handle["latents"], dtype=np.float32)
            meta_array = handle["metadata"]
        bank = cls(embedding_dim=embedding_dim, latent_dim=latent_dim)
        bank._embeddings = embeddings
        bank._latents = latents
        import json as _json
        bank._metadata = [
            (_json.loads(m) if isinstance(m, str) and m else {})
            for m in meta_array.tolist()
        ]
        return bank

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self._embeddings.shape[0])

    @property
    def latents(self) -> np.ndarray:
        return self._latents

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        if _HAS_FAISS and len(self) > 0:
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            self._index.add(np.ascontiguousarray(self._embeddings))

    def query(
        self,
        queries: torch.Tensor | np.ndarray,
        *,
        k: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return top-K ``(embeddings, latents, cosine_sims)`` for each query.

        Parameters
        ----------
        queries : (B, embedding_dim) tensor or ndarray
        k : int

        Returns
        -------
        top_embeddings : (B, k, embedding_dim) float32 tensor
        top_latents    : (B, k, latent_dim)    float32 tensor
        top_sims       : (B, k)                float32 tensor
            Cosine similarities of the k nearest anchors to each query.
        """
        if isinstance(queries, torch.Tensor):
            device = queries.device
            q_np = queries.detach().to(torch.float32).cpu().numpy()
        else:
            device = torch.device("cpu")
            q_np = np.asarray(queries, dtype=np.float32)

        if len(self) == 0:
            zeros_emb = torch.zeros((q_np.shape[0], k, self.embedding_dim), dtype=torch.float32, device=device)
            zeros_lat = torch.zeros((q_np.shape[0], k, self.latent_dim), dtype=torch.float32, device=device)
            zeros_sim = torch.zeros((q_np.shape[0], k), dtype=torch.float32, device=device)
            return zeros_emb, zeros_lat, zeros_sim

        # L2-normalise queries for cosine retrieval.
        q_norm = q_np / np.clip(np.linalg.norm(q_np, axis=1, keepdims=True), 1e-8, None)
        effective_k = int(min(k, len(self)))

        self._ensure_index()
        if self._index is not None:
            sims, idx = self._index.search(np.ascontiguousarray(q_norm), effective_k)
        else:
            sims_full = q_norm @ self._embeddings.T  # (B, N)
            idx = np.argpartition(-sims_full, effective_k - 1, axis=1)[:, :effective_k]
            # Sort within the partitioned window.
            row_idx = np.arange(idx.shape[0])[:, None]
            sort = np.argsort(-sims_full[row_idx, idx], axis=1)
            idx = idx[row_idx, sort]
            sims = sims_full[row_idx, idx]

        out_embeddings = self._embeddings[idx]  # (B, k_eff, embedding_dim)
        out_latents = self._latents[idx]        # (B, k_eff, latent_dim)

        if effective_k < k:
            pad_n = k - effective_k
            out_embeddings = np.concatenate(
                [out_embeddings, np.zeros((out_embeddings.shape[0], pad_n, self.embedding_dim), dtype=np.float32)],
                axis=1,
            )
            out_latents = np.concatenate(
                [out_latents, np.zeros((out_latents.shape[0], pad_n, self.latent_dim), dtype=np.float32)],
                axis=1,
            )
            sims = np.concatenate([sims, np.zeros((sims.shape[0], pad_n), dtype=np.float32)], axis=1)

        return (
            torch.as_tensor(out_embeddings, dtype=torch.float32, device=device),
            torch.as_tensor(out_latents, dtype=torch.float32, device=device),
            torch.as_tensor(sims, dtype=torch.float32, device=device),
        )


def build_bank_from_forward_results(
    *,
    embeddings: torch.Tensor,
    latents: torch.Tensor,
    patch_ids: list[str],
    sequence_ids: list[str],
    embedding_dim: int,
    latent_dim: int,
) -> AnchorBank:
    """Convenience builder: stack tensors into a fresh bank."""
    if embeddings.shape[0] != latents.shape[0]:
        raise ValueError(
            f"embeddings and latents count mismatch: {embeddings.shape[0]} vs {latents.shape[0]}"
        )
    if len(patch_ids) != embeddings.shape[0]:
        raise ValueError("patch_ids length must match embedding count")
    bank = AnchorBank(embedding_dim=embedding_dim, latent_dim=latent_dim)
    records = [
        AnchorRecord(
            embedding=embeddings[i].detach().cpu().numpy(),
            latent=latents[i].detach().cpu().numpy(),
            metadata={
                "patch_id": patch_ids[i],
                "sequence_id": sequence_ids[i] if i < len(sequence_ids) else "",
            },
        )
        for i in range(embeddings.shape[0])
    ]
    bank.update(records)
    return bank
