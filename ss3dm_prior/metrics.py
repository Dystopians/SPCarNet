"""Metrics for reconstruction, defect prediction, and retrieval."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
import warnings


def _to_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.asarray(x))


def _cdist_fp32_safe(x: torch.Tensor, y: torch.Tensor, *, p: float) -> torch.Tensor:
    """``torch.cdist`` wrapper that guarantees a fp32 CUDA computation.

    ``cdist_cuda`` has no Half kernel, so metrics called on AMP fp16 outputs
    fail on GPU. We upcast both inputs to float32 and wrap the call in
    ``autocast(enabled=False)`` so the outer trainer's autocast context can't
    demote us back to fp16.
    """
    device_type = x.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return torch.cdist(x.float(), y.float(), p=p)


def recon_chamfer_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = _cdist_fp32_safe(x, y, p=1)
    return dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()


def recon_chamfer_l1_or_nan(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.shape[1] == 0 or y.shape[1] == 0:
        warnings.warn("recon_chamfer_l1_or_nan undefined for empty point sets.", stacklevel=2)
        return float("nan")
    return float(recon_chamfer_l1(x, y).detach().cpu().item())


def recon_normal_cosine(
    recon_points: torch.Tensor,
    recon_normals: torch.Tensor,
    clean_points: torch.Tensor,
    clean_normals: torch.Tensor,
) -> torch.Tensor:
    idx = _cdist_fp32_safe(recon_points, clean_points, p=2).argmin(dim=2)
    matched = torch.gather(clean_normals, 1, idx[..., None].expand(-1, -1, clean_normals.shape[-1]))
    return F.cosine_similarity(recon_normals, matched, dim=-1).mean()


def recon_normal_cosine_or_nan(
    recon_points: torch.Tensor,
    recon_normals: torch.Tensor,
    clean_points: torch.Tensor,
    clean_normals: torch.Tensor,
) -> float:
    if clean_points.shape[1] == 0 or clean_normals.shape[1] == 0:
        warnings.warn("recon_normal_cosine_or_nan undefined for empty clean point or normal sets.", stacklevel=2)
        return float("nan")
    return float(recon_normal_cosine(recon_points, recon_normals, clean_points, clean_normals).detach().cpu().item())


def denoise_gain_chamfer(
    corrupted_points: torch.Tensor,
    recon_points: torch.Tensor,
    clean_points: torch.Tensor,
) -> torch.Tensor:
    return recon_chamfer_l1(corrupted_points, clean_points) - recon_chamfer_l1(recon_points, clean_points)


def hidden_completion_gain_or_nan(
    corrupted_points: torch.Tensor,
    recon_points: torch.Tensor,
    hidden_clean_points: torch.Tensor,
) -> float:
    if hidden_clean_points.shape[1] == 0:
        warnings.warn("hidden_completion_gain_or_nan undefined for empty hidden clean point sets.", stacklevel=2)
        return float("nan")
    before = recon_chamfer_l1_or_nan(corrupted_points, hidden_clean_points)
    after = recon_chamfer_l1_or_nan(recon_points, hidden_clean_points)
    if not np.isfinite(before) or not np.isfinite(after):
        return float("nan")
    return float(before - after)


def score_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def score_spearman(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    target_np = target.detach().cpu().numpy().reshape(-1)
    if len(pred_np) < 2:
        return float("nan")
    if np.allclose(pred_np, pred_np[0]) or np.allclose(target_np, target_np[0]):
        return float("nan")
    corr = spearmanr(pred_np, target_np).correlation
    if corr is None or np.isnan(corr):
        return float("nan")
    return float(corr)


def point_defect_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def occupancy_iou_visible(
    query_occupancy_logits: torch.Tensor,
    query_labels_all: torch.Tensor,
    query_ignore_mask: torch.Tensor | None = None,
) -> float:
    pred = query_occupancy_logits > 0.0
    labels = query_labels_all > 0.5
    valid_mask = ~query_ignore_mask.bool() if query_ignore_mask is not None else torch.ones_like(labels, dtype=torch.bool)
    if not bool(valid_mask.any()):
        warnings.warn("occupancy_iou_visible undefined because no visible queries remain.", stacklevel=2)
        return float("nan")
    pred = pred[valid_mask]
    labels = labels[valid_mask]
    intersection = torch.logical_and(pred, labels).sum().item()
    union = torch.logical_or(pred, labels).sum().item()
    if union == 0:
        return 1.0
    return float(intersection / union)


def free_space_violation_rate(
    query_occupancy_logits: torch.Tensor,
    query_labels_all: torch.Tensor,
    query_ignore_mask: torch.Tensor | None = None,
) -> float:
    labels = query_labels_all > 0.5
    free_mask = ~labels
    if query_ignore_mask is not None:
        free_mask &= ~query_ignore_mask.bool()
    if not bool(free_mask.any()):
        warnings.warn("free_space_violation_rate undefined because no free-space queries remain.", stacklevel=2)
        return float("nan")
    violations = (query_occupancy_logits[free_mask] > 0.0).float().mean()
    return float(violations.item())


def free_space_fp_rate(
    query_occupancy_logits: torch.Tensor,
    query_labels_all: torch.Tensor,
    query_ignore_mask: torch.Tensor | None = None,
) -> float:
    return free_space_violation_rate(query_occupancy_logits, query_labels_all, query_ignore_mask)


def intrinsic_difficulty_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred.reshape(-1) - target.reshape(-1))).item())


def intrinsic_difficulty_calibration_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return intrinsic_difficulty_mae(pred, target)


def intrinsic_difficulty_spearman(pred: torch.Tensor, target: torch.Tensor) -> float:
    return score_spearman(pred.reshape(-1), target.reshape(-1))


def prototype_usage_entropy(code_indices: torch.Tensor, codebook_size: int | None = None) -> float:
    indices = code_indices.reshape(-1)
    if indices.numel() == 0:
        warnings.warn("prototype_usage_entropy undefined for empty code indices.", stacklevel=2)
        return float("nan")
    if codebook_size is None:
        codebook_size = int(indices.max().item()) + 1
    usage = torch.bincount(indices, minlength=int(codebook_size)).float()
    probs = usage / torch.clamp(usage.sum(), min=1.0)
    entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-8)))
    return float(entropy.item())


def _as_string_array(values: Sequence[str] | None, *, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required for filtered retrieval metrics.")
    return np.asarray([str(value) for value in values], dtype=object)


def _normalized_similarity(query_embedding: torch.Tensor, target_embedding: torch.Tensor) -> torch.Tensor:
    return torch.matmul(
        F.normalize(query_embedding, dim=-1),
        F.normalize(target_embedding, dim=-1).transpose(0, 1),
    )


def retrieval_topk(
    query_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    k: int,
) -> float:
    similarity = _normalized_similarity(query_embedding, target_embedding)
    topk = similarity.topk(k=min(k, similarity.shape[1]), dim=1).indices
    labels = torch.arange(similarity.shape[0], device=similarity.device)[:, None]
    return float((topk == labels).any(dim=1).float().mean().item())


def _neighbor_recall_at_k(
    query_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    *,
    query_patch_ids: Sequence[str],
    target_patch_ids: Sequence[str],
    query_sequence_ids: Sequence[str] | None = None,
    target_sequence_ids: Sequence[str] | None = None,
    k: int,
    require_different_sequence: bool = False,
) -> float:
    if query_embedding.shape[0] != target_embedding.shape[0]:
        raise ValueError("Filtered retrieval metrics require aligned query/target batches.")

    query_patch = _as_string_array(query_patch_ids, name="query_patch_ids")
    target_patch = _as_string_array(target_patch_ids, name="target_patch_ids")
    if len(query_patch) != query_embedding.shape[0] or len(target_patch) != target_embedding.shape[0]:
        raise ValueError("Patch id metadata must align with query and target embeddings.")

    candidate_mask = query_patch[:, None] != target_patch[None, :]
    if require_different_sequence:
        query_sequence = _as_string_array(query_sequence_ids, name="query_sequence_ids")
        target_sequence = _as_string_array(target_sequence_ids, name="target_sequence_ids")
        if len(query_sequence) != query_embedding.shape[0] or len(target_sequence) != target_embedding.shape[0]:
            raise ValueError("Sequence id metadata must align with query and target embeddings.")
        candidate_mask &= query_sequence[:, None] != target_sequence[None, :]

    if not bool(candidate_mask.any()):
        warnings.warn("Filtered retrieval metric undefined because no valid non-self candidates remain.", stacklevel=2)
        return float("nan")

    similarity_query = _normalized_similarity(query_embedding, target_embedding)
    similarity_anchor = _normalized_similarity(target_embedding, target_embedding)
    candidate_mask_tensor = torch.as_tensor(candidate_mask, dtype=torch.bool, device=similarity_query.device)
    similarity_query = similarity_query.masked_fill(~candidate_mask_tensor, float("-inf"))
    similarity_anchor = similarity_anchor.masked_fill(~candidate_mask_tensor, float("-inf"))

    valid_queries = torch.isfinite(similarity_anchor.max(dim=1).values)
    if not bool(valid_queries.any()):
        warnings.warn("Filtered retrieval metric undefined because every query lost all valid candidates.", stacklevel=2)
        return float("nan")

    pred_topk = similarity_query.topk(k=min(k, similarity_query.shape[1]), dim=1).indices
    reference_top1 = similarity_anchor.argmax(dim=1, keepdim=True)
    hits = (pred_topk == reference_top1).any(dim=1)
    return float(hits[valid_queries].float().mean().item())


def retrieval_top1(query_embedding: torch.Tensor, target_embedding: torch.Tensor) -> float:
    return retrieval_topk(query_embedding, target_embedding, 1)


def retrieval_top5(query_embedding: torch.Tensor, target_embedding: torch.Tensor) -> float:
    return retrieval_topk(query_embedding, target_embedding, 5)


def retrieval_top1_self_aligned(query_embedding: torch.Tensor, target_embedding: torch.Tensor) -> float:
    return retrieval_topk(query_embedding, target_embedding, 1)


def retrieval_top5_self_aligned(query_embedding: torch.Tensor, target_embedding: torch.Tensor) -> float:
    return retrieval_topk(query_embedding, target_embedding, 5)


def retrieval_top1_nonself(
    query_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    *,
    query_patch_ids: Sequence[str],
    target_patch_ids: Sequence[str],
    query_sequence_ids: Sequence[str] | None = None,
    target_sequence_ids: Sequence[str] | None = None,
) -> float:
    return _neighbor_recall_at_k(
        query_embedding,
        target_embedding,
        query_patch_ids=query_patch_ids,
        target_patch_ids=target_patch_ids,
        query_sequence_ids=query_sequence_ids,
        target_sequence_ids=target_sequence_ids,
        k=1,
    )


def retrieval_top5_nonself(
    query_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    *,
    query_patch_ids: Sequence[str],
    target_patch_ids: Sequence[str],
    query_sequence_ids: Sequence[str] | None = None,
    target_sequence_ids: Sequence[str] | None = None,
) -> float:
    return _neighbor_recall_at_k(
        query_embedding,
        target_embedding,
        query_patch_ids=query_patch_ids,
        target_patch_ids=target_patch_ids,
        query_sequence_ids=query_sequence_ids,
        target_sequence_ids=target_sequence_ids,
        k=5,
    )


def retrieval_top1_cross_sequence(
    query_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    *,
    query_patch_ids: Sequence[str],
    target_patch_ids: Sequence[str],
    query_sequence_ids: Sequence[str],
    target_sequence_ids: Sequence[str],
) -> float:
    return _neighbor_recall_at_k(
        query_embedding,
        target_embedding,
        query_patch_ids=query_patch_ids,
        target_patch_ids=target_patch_ids,
        query_sequence_ids=query_sequence_ids,
        target_sequence_ids=target_sequence_ids,
        k=1,
        require_different_sequence=True,
    )
