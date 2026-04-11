"""Metrics for reconstruction, defect prediction, and retrieval."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
import warnings


def _to_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.asarray(x))


def recon_chamfer_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(x, y, p=1)
    return dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()


def recon_normal_cosine(
    recon_points: torch.Tensor,
    recon_normals: torch.Tensor,
    clean_points: torch.Tensor,
    clean_normals: torch.Tensor,
) -> torch.Tensor:
    idx = torch.cdist(recon_points, clean_points, p=2).argmin(dim=2)
    matched = torch.gather(clean_normals, 1, idx[..., None].expand(-1, -1, clean_normals.shape[-1]))
    return F.cosine_similarity(recon_normals, matched, dim=-1).mean()


def denoise_gain_chamfer(
    corrupted_points: torch.Tensor,
    recon_points: torch.Tensor,
    clean_points: torch.Tensor,
) -> torch.Tensor:
    return recon_chamfer_l1(corrupted_points, clean_points) - recon_chamfer_l1(recon_points, clean_points)


def score_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def score_spearman(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    target_np = target.detach().cpu().numpy().reshape(-1)
    if len(pred_np) < 2:
        warnings.warn("score_spearman undefined for fewer than two samples.", stacklevel=2)
        return float("nan")
    if np.allclose(pred_np, pred_np[0]) or np.allclose(target_np, target_np[0]):
        warnings.warn("score_spearman undefined for constant inputs.", stacklevel=2)
        return float("nan")
    corr = spearmanr(pred_np, target_np).correlation
    if corr is None or np.isnan(corr):
        warnings.warn("score_spearman undefined for constant inputs.", stacklevel=2)
        return float("nan")
    return float(corr)


def point_defect_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def retrieval_topk(
    query_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    k: int,
) -> float:
    similarity = torch.matmul(
        F.normalize(query_embedding, dim=-1),
        F.normalize(target_embedding, dim=-1).transpose(0, 1),
    )
    topk = similarity.topk(k=min(k, similarity.shape[1]), dim=1).indices
    labels = torch.arange(similarity.shape[0], device=similarity.device)[:, None]
    return float((topk == labels).any(dim=1).float().mean().item())


def retrieval_top1(query_embedding: torch.Tensor, target_embedding: torch.Tensor) -> float:
    return retrieval_topk(query_embedding, target_embedding, 1)


def retrieval_top5(query_embedding: torch.Tensor, target_embedding: torch.Tensor) -> float:
    return retrieval_topk(query_embedding, target_embedding, 5)
