from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


def load_sparse_depth_parent_rollback_cache(path: str | Path, *, allow_test_cache: bool = False) -> dict[str, Any]:
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"sparse depth parent rollback cache not found: {cache_path}")
    payload = np.load(cache_path, allow_pickle=True)
    manifest_raw = payload.get("manifest_json", "{}")
    manifest = json.loads(str(manifest_raw.item() if hasattr(manifest_raw, "item") else manifest_raw))
    split = str(manifest.get("split", "")).lower()
    if split == "test" and not bool(allow_test_cache):
        raise RuntimeError("Refusing to use test split sentinel cache for training parent rollback loss.")
    arrays = {k: payload[k] for k in payload.files if k != "manifest_json"}
    image_key = np.asarray(arrays["image_key"], dtype=object).reshape(-1)
    by_image_key: dict[str, dict[str, np.ndarray]] = {}
    for key in sorted(set(image_key.tolist()), key=str):
        mask = image_key == key
        by_image_key[str(key)] = {name: np.asarray(value)[mask] for name, value in arrays.items()}
    return {
        "path": str(cache_path),
        "manifest": manifest,
        "arrays": arrays,
        "by_image_key": by_image_key,
    }


def sparse_depth_parent_rollback_lambda(iteration: int, opt) -> float:
    if not bool(getattr(opt, "enable_sparse_depth_parent_rollback_loss", False)):
        return 0.0
    base = float(getattr(opt, "lambda_sparse_depth_parent_rollback", 0.0))
    if base <= 0.0:
        return 0.0
    start_iter = int(getattr(opt, "sparse_depth_parent_rollback_start_iter", 0))
    if int(iteration) < start_iter:
        return 0.0
    warmup_iters = max(1, int(getattr(opt, "sparse_depth_parent_rollback_warmup_iters", 1)))
    warmup = min(1.0, max(0.0, float(int(iteration) - start_iter) / float(warmup_iters)))
    return float(base * warmup)


def _select_indices(weights: np.ndarray, max_points: int) -> np.ndarray:
    n = int(weights.shape[0])
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    order = np.argsort(-weights, kind="stable")
    return order[: int(max_points)].astype(np.int64)


def compute_sparse_depth_parent_rollback_loss(
    *,
    current_depth: torch.Tensor | None,
    cache_by_image_key: Mapping[str, Mapping[str, np.ndarray]],
    image_key: str,
    lam: float,
    margin_abs: float,
    margin_rel: float,
    huber_delta: float,
    loss_space: str,
    combined_mae_beta: float = 1.0,
    max_points_per_view: int = 0,
    cluster_balance: bool = False,
    regressed_only: bool = False,
    cluster_top_k: int = 0,
    strict: bool = False,
) -> dict[str, Any]:
    if lam <= 0.0:
        return {"loss_pure": 0.0, "loss_weighted": 0.0, "reason": "lambda_zero", "active_points": 0}
    if current_depth is None:
        if strict:
            raise RuntimeError("sparse parent rollback loss requires surf_depth in render package.")
        return {"loss_pure": 0.0, "loss_weighted": 0.0, "reason": "render_missing_output", "active_points": 0}
    entry = cache_by_image_key.get(str(image_key), None)
    if entry is None:
        return {"loss_pure": 0.0, "loss_weighted": 0.0, "reason": "missing_camera_key", "active_points": 0}

    px_np = np.asarray(entry["px"], dtype=np.int64).reshape(-1)
    py_np = np.asarray(entry["py"], dtype=np.int64).reshape(-1)
    width_np = np.asarray(entry.get("width", np.zeros_like(px_np)), dtype=np.int64).reshape(-1)
    height_np = np.asarray(entry.get("height", np.zeros_like(py_np)), dtype=np.int64).reshape(-1)
    gt_np = np.asarray(entry["gt_depth"], dtype=np.float64).reshape(-1)
    parent_abs_np = np.asarray(entry["parent_abs_error"], dtype=np.float64).reshape(-1)
    parent_rel_np = np.asarray(entry["parent_abs_rel"], dtype=np.float64).reshape(-1)
    weights_np = np.asarray(entry.get("sentinel_weight", np.ones_like(gt_np)), dtype=np.float64).reshape(-1)
    cluster_np = np.asarray(entry.get("cluster_id", np.full_like(gt_np, -1)), dtype=np.int64).reshape(-1)
    regressed_np = np.asarray(entry.get("is_regressed_candidate", np.ones_like(gt_np, dtype=bool)), dtype=bool).reshape(-1)
    candidate_delta_abs_np = np.asarray(entry.get("candidate_delta_abs_error", np.zeros_like(gt_np)), dtype=np.float64).reshape(-1)
    candidate_delta_rel_np = np.asarray(entry.get("candidate_delta_abs_rel", np.zeros_like(gt_np)), dtype=np.float64).reshape(-1)
    finite = (
        np.isfinite(px_np)
        & np.isfinite(py_np)
        & np.isfinite(gt_np)
        & (gt_np > 1e-6)
        & np.isfinite(parent_abs_np)
        & np.isfinite(parent_rel_np)
        & np.isfinite(weights_np)
        & (weights_np > 0)
    )
    if not np.any(finite):
        return {"loss_pure": 0.0, "loss_weighted": 0.0, "reason": "no_valid_cache_points", "active_points": 0}
    px_np = px_np[finite]
    py_np = py_np[finite]
    width_np = width_np[finite] if width_np.shape[0] == finite.shape[0] else np.zeros_like(px_np)
    height_np = height_np[finite] if height_np.shape[0] == finite.shape[0] else np.zeros_like(py_np)
    gt_np = gt_np[finite]
    parent_abs_np = parent_abs_np[finite]
    parent_rel_np = parent_rel_np[finite]
    weights_np = weights_np[finite]
    cluster_np = cluster_np[finite] if cluster_np.shape[0] == finite.shape[0] else np.full_like(gt_np, -1, dtype=np.int64)
    regressed_np = regressed_np[finite] if regressed_np.shape[0] == finite.shape[0] else np.ones_like(gt_np, dtype=bool)
    candidate_delta_abs_np = (
        candidate_delta_abs_np[finite] if candidate_delta_abs_np.shape[0] == finite.shape[0] else np.zeros_like(gt_np)
    )
    candidate_delta_rel_np = (
        candidate_delta_rel_np[finite] if candidate_delta_rel_np.shape[0] == finite.shape[0] else np.zeros_like(gt_np)
    )
    if bool(regressed_only):
        keep = regressed_np
        if not np.any(keep):
            return {"loss_pure": 0.0, "loss_weighted": 0.0, "reason": "no_regressed_candidate_sentinels", "active_points": 0}
        px_np = px_np[keep]
        py_np = py_np[keep]
        width_np = width_np[keep] if width_np.shape[0] == keep.shape[0] else np.zeros_like(px_np)
        height_np = height_np[keep] if height_np.shape[0] == keep.shape[0] else np.zeros_like(py_np)
        gt_np = gt_np[keep]
        parent_abs_np = parent_abs_np[keep]
        parent_rel_np = parent_rel_np[keep]
        weights_np = weights_np[keep]
        cluster_np = cluster_np[keep]
        regressed_np = regressed_np[keep]
        candidate_delta_abs_np = candidate_delta_abs_np[keep]
        candidate_delta_rel_np = candidate_delta_rel_np[keep]
    if int(cluster_top_k) > 0 and cluster_np.shape[0] == gt_np.shape[0] and np.any(cluster_np >= 0):
        severity = np.maximum(candidate_delta_rel_np, 0.0) + float(max(0.0, combined_mae_beta)) * np.maximum(candidate_delta_abs_np, 0.0)
        cluster_scores: dict[int, float] = {}
        for cid, score in zip(cluster_np.tolist(), severity.tolist()):
            if int(cid) < 0:
                continue
            cluster_scores[int(cid)] = max(float(cluster_scores.get(int(cid), 0.0)), float(score))
        selected_clusters = {
            cid
            for cid, _ in sorted(cluster_scores.items(), key=lambda kv: (-kv[1], kv[0]))[: int(cluster_top_k)]
        }
        keep_cluster = np.asarray([int(cid) in selected_clusters for cid in cluster_np], dtype=bool)
        if not np.any(keep_cluster):
            return {"loss_pure": 0.0, "loss_weighted": 0.0, "reason": "no_selected_rollback_clusters", "active_points": 0}
        px_np = px_np[keep_cluster]
        py_np = py_np[keep_cluster]
        width_np = width_np[keep_cluster] if width_np.shape[0] == keep_cluster.shape[0] else np.zeros_like(px_np)
        height_np = height_np[keep_cluster] if height_np.shape[0] == keep_cluster.shape[0] else np.zeros_like(py_np)
        gt_np = gt_np[keep_cluster]
        parent_abs_np = parent_abs_np[keep_cluster]
        parent_rel_np = parent_rel_np[keep_cluster]
        weights_np = weights_np[keep_cluster]
        cluster_np = cluster_np[keep_cluster]
        regressed_np = regressed_np[keep_cluster]
        candidate_delta_abs_np = candidate_delta_abs_np[keep_cluster]
        candidate_delta_rel_np = candidate_delta_rel_np[keep_cluster]
    if bool(cluster_balance) and cluster_np.shape[0] == weights_np.shape[0] and np.any(cluster_np >= 0):
        unique, counts = np.unique(cluster_np[cluster_np >= 0], return_counts=True)
        count_map = {int(cid): int(count) for cid, count in zip(unique.tolist(), counts.tolist())}
        weights_np = np.asarray(
            [
                float(wi) / np.sqrt(max(1, count_map.get(int(cid), 1)))
                for wi, cid in zip(weights_np.tolist(), cluster_np.tolist())
            ],
            dtype=np.float64,
        )
    pick = _select_indices(weights_np, int(max_points_per_view))
    px_np = px_np[pick]
    py_np = py_np[pick]
    width_np = width_np[pick] if width_np.shape[0] == pick.shape[0] or width_np.shape[0] == gt_np.shape[0] else np.zeros_like(px_np)
    height_np = height_np[pick] if height_np.shape[0] == pick.shape[0] or height_np.shape[0] == gt_np.shape[0] else np.zeros_like(py_np)
    gt_np = gt_np[pick]
    parent_abs_np = parent_abs_np[pick]
    parent_rel_np = parent_rel_np[pick]
    weights_np = weights_np[pick]

    pred_depth = current_depth
    if pred_depth.dim() == 3:
        pred_depth = pred_depth[0]
    h, w = int(pred_depth.shape[0]), int(pred_depth.shape[1])
    if width_np.shape[0] == px_np.shape[0] and np.any(width_np > 0):
        sx = float(w) / np.maximum(width_np.astype(np.float64), 1.0)
        px_np = np.rint(px_np.astype(np.float64) * sx).astype(np.int64)
    if height_np.shape[0] == py_np.shape[0] and np.any(height_np > 0):
        sy = float(h) / np.maximum(height_np.astype(np.float64), 1.0)
        py_np = np.rint(py_np.astype(np.float64) * sy).astype(np.int64)
    px_np = np.clip(px_np, 0, w - 1)
    py_np = np.clip(py_np, 0, h - 1)
    device = pred_depth.device
    dtype = pred_depth.dtype
    px = torch.from_numpy(px_np).to(device=device, dtype=torch.long)
    py = torch.from_numpy(py_np).to(device=device, dtype=torch.long)
    gt = torch.from_numpy(gt_np).to(device=device, dtype=dtype)
    parent_abs = torch.from_numpy(parent_abs_np).to(device=device, dtype=dtype)
    parent_rel = torch.from_numpy(parent_rel_np).to(device=device, dtype=dtype)
    weights = torch.from_numpy(weights_np).to(device=device, dtype=dtype)

    current = pred_depth[py, px]
    valid_current = torch.isfinite(current) & (current > 1e-6)
    if not torch.any(valid_current):
        return {"loss_pure": 0.0, "loss_weighted": 0.0, "reason": "no_valid_current_depth", "active_points": 0}
    current = current[valid_current]
    gt = gt[valid_current]
    parent_abs = parent_abs[valid_current]
    parent_rel = parent_rel[valid_current]
    weights = weights[valid_current]

    current_abs = torch.abs(current - gt)
    current_rel = current_abs / torch.clamp(torch.abs(gt), min=1e-6)
    violation_rel = torch.relu(current_rel - parent_rel - float(margin_rel))
    violation_abs = torch.relu(current_abs - parent_abs - float(margin_abs))
    space = str(loss_space or "combined").strip().lower()
    if space == "absrel":
        violation = violation_rel
    elif space == "mae":
        violation = violation_abs
    elif space == "combined":
        violation = violation_rel + float(combined_mae_beta) * violation_abs
    else:
        raise ValueError(f"unknown sparse parent rollback loss space: {loss_space}")
    zeros = torch.zeros_like(violation)
    per_point = F.smooth_l1_loss(violation, zeros, reduction="none", beta=max(1e-6, float(huber_delta)))
    weight_sum = torch.clamp(torch.sum(weights), min=1e-6)
    loss_pure = torch.sum(per_point * weights) / weight_sum
    loss_weighted = float(lam) * loss_pure
    active = violation > 0
    return {
        "loss_pure": loss_pure,
        "loss_weighted": loss_weighted,
        "reason": "ok",
        "active_points": int(torch.count_nonzero(active).detach().item()),
        "total_points": int(violation.numel()),
        "active_fraction": float(torch.mean(active.to(torch.float32)).detach().item()) if violation.numel() else 0.0,
        "mean_violation_rel": float(torch.mean(violation_rel).detach().item()) if violation_rel.numel() else 0.0,
        "max_violation_rel": float(torch.max(violation_rel).detach().item()) if violation_rel.numel() else 0.0,
        "mean_violation_abs": float(torch.mean(violation_abs).detach().item()) if violation_abs.numel() else 0.0,
        "max_violation_abs": float(torch.max(violation_abs).detach().item()) if violation_abs.numel() else 0.0,
    }
