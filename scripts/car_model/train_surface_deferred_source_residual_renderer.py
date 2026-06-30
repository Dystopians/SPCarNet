#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.ecsr_apply_surface_residual_region_texture_adapter import (  # noqa: E402
    build_lpips_model,
    evidence_views,
    image_lpips_chw,
    image_ssim_chw,
    save_image_chw,
)
from scripts.car_model.train_perceptual_surface_residual_decoder import (  # noqa: E402
    _face_indices,
    _policy_split,
    _rank_candidate_faces,
    _valid_mask,
)


DEFAULT_EVIDENCE = (
    "/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/"
    "flowers/teacher_surface_evidence"
)
DEFAULT_TARGET_NO_GT = (
    "/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/"
    "flowers/target_evidence_no_gt"
)
DEFAULT_TARGET_EVAL = (
    "/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/"
    "flowers/target_evidence_reparented"
)

PHASEJ_FLOWERS = {"psnr": 20.304358, "ssim": 0.557770, "lpips": 0.329222}
LUMA_WEIGHTS = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
LEARNED_OOD_FEATURE_NAMES = [
    "gain_boost",
    "policy_reliability",
    "policy_tail_risk",
    "view_gap",
    "variance_ratio",
    "parent_mismatch",
    "effective_count_risk",
    "source_agreement_proxy",
    "max_view_cos",
]

FORBIDDEN_TARGET_KEYS = {
    "rgb_gt",
    "residual_rgb",
    "residual_l1",
    "teacher_residual_rgb",
    "teacher_residual_rgb_raw",
    "teacher_residual_l1",
    "teacher_better_mask",
    "teacher_gain_l1",
    "teacher_parent_delta_l1",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean(np.square(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))
    return float("inf") if mse <= 1.0e-12 else float(-10.0 * math.log10(mse))


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "median": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _tail(values: list[float], fraction: float = 0.10) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "cvar": 0.0, "p10": 0.0}
    arr = np.sort(np.asarray(values, dtype=np.float64))
    count = max(1, int(math.ceil(float(fraction) * arr.size)))
    return {"min": float(arr[0]), "cvar": float(np.mean(arr[:count])), "p10": float(np.quantile(arr, fraction))}


def _camera_dir(z: np.lib.npyio.NpzFile) -> np.ndarray:
    camera = np.asarray(z["camera_center"], dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(camera))
    if not np.isfinite(norm) or norm <= 1.0e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return (camera / norm).astype(np.float32)


def _normalize_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1.0e-8)
    return (rows / denom).astype(np.float32)


def _luma(rgb_chw: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_chw, dtype=np.float32)
    return (LUMA_WEIGHTS[0] * rgb[0] + LUMA_WEIGHTS[1] * rgb[1] + LUMA_WEIGHTS[2] * rgb[2]).astype(np.float32)


def _luma_gradient_magnitude(luma_hw: np.ndarray) -> np.ndarray:
    luma = np.asarray(luma_hw, dtype=np.float32)
    grad_x = np.zeros_like(luma, dtype=np.float32)
    grad_y = np.zeros_like(luma, dtype=np.float32)
    grad_x[:, 1:] = np.abs(luma[:, 1:] - luma[:, :-1])
    grad_y[1:, :] = np.abs(luma[1:, :] - luma[:-1, :])
    return np.maximum(grad_x, grad_y).astype(np.float32)


def _box_mean2d(values: np.ndarray, radius: int) -> np.ndarray:
    radius_i = max(0, int(radius))
    arr = np.asarray(values, dtype=np.float32)
    if radius_i <= 0:
        return arr
    pad = radius_i
    padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="edge")
    integral = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
    k = 2 * radius_i + 1
    total = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    return (total / float(k * k)).astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=np.float64), -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-arr))).astype(np.float32)


def _stack_learned_ood_features(
    *,
    gain_boost: np.ndarray,
    policy_reliability: np.ndarray,
    policy_tail_risk: np.ndarray,
    view_gap: np.ndarray,
    variance_ratio: np.ndarray,
    parent_mismatch: np.ndarray,
    effective_count_risk: np.ndarray,
    max_view_cos: np.ndarray,
) -> np.ndarray:
    source_agreement_proxy = np.exp(-0.25 * np.clip(np.asarray(variance_ratio, dtype=np.float32), 0.0, 4.0))
    return np.stack(
        [
            np.asarray(gain_boost, dtype=np.float32),
            np.asarray(policy_reliability, dtype=np.float32),
            np.asarray(policy_tail_risk, dtype=np.float32),
            np.clip(np.asarray(view_gap, dtype=np.float32), 0.0, 2.0),
            np.clip(np.asarray(variance_ratio, dtype=np.float32), 0.0, 4.0),
            np.clip(np.asarray(parent_mismatch, dtype=np.float32), 0.0, 1.0),
            np.clip(np.asarray(effective_count_risk, dtype=np.float32), 0.0, 1.0),
            source_agreement_proxy.astype(np.float32),
            np.clip(np.asarray(max_view_cos, dtype=np.float32), -1.0, 1.0),
        ],
        axis=1,
    ).astype(np.float32)


def _apply_learned_ood_head(bank: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    required = {"learned_ood_head_coef", "learned_ood_head_bias", "learned_ood_head_mean", "learned_ood_head_scale"}
    missing = sorted(required - set(bank))
    if missing:
        raise RuntimeError(f"learned OOD head requested but checkpoint/bank is missing: {missing}")
    coef = np.asarray(bank["learned_ood_head_coef"], dtype=np.float32)
    bias = float(np.asarray(bank["learned_ood_head_bias"], dtype=np.float32).reshape(-1)[0])
    mean = np.asarray(bank["learned_ood_head_mean"], dtype=np.float32)
    scale = np.maximum(np.asarray(bank["learned_ood_head_scale"], dtype=np.float32), 1.0e-6)
    floor = float(np.asarray(bank.get("learned_ood_head_floor", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0])
    standardized = (np.asarray(features, dtype=np.float32) - mean.reshape(1, -1)) / scale.reshape(1, -1)
    pred = bias + np.sum(standardized * coef.reshape(1, -1), axis=1)
    confidence = np.clip(pred, floor, 1.0).astype(np.float32)
    return confidence


def _feature_rows_to_arrays(feature_rows: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not feature_rows:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, len(LEARNED_OOD_FEATURE_NAMES)), dtype=np.float32),
        )
    ys = np.concatenate([np.asarray(row["ys"], dtype=np.int64) for row in feature_rows])
    xs = np.concatenate([np.asarray(row["xs"], dtype=np.int64) for row in feature_rows])
    features = np.concatenate([np.asarray(row["features"], dtype=np.float32) for row in feature_rows], axis=0)
    return ys, xs, features


def _bin_ids(z: np.lib.npyio.NpzFile, ys: np.ndarray, xs: np.ndarray, grid: int) -> np.ndarray:
    bary = np.asarray(z["barycentric"], dtype=np.float32)
    u = np.clip(bary[1, ys, xs], 0.0, 1.0 - 1.0e-6)
    v = np.clip(bary[2, ys, xs], 0.0, 1.0 - 1.0e-6)
    iu = np.floor(u * int(grid)).astype(np.int64)
    iv = np.floor(v * int(grid)).astype(np.int64)
    return iu * int(grid) + iv


def _surface_support_mask(
    z: np.lib.npyio.NpzFile,
    candidate_faces: np.ndarray,
    *,
    min_alpha: float,
) -> np.ndarray:
    face_id = np.asarray(z["face_id"], dtype=np.int64)
    valid = face_id >= 0
    if "barycentric_valid" in z:
        valid &= np.asarray(z["barycentric_valid"]).astype(bool)
    if "alpha" in z:
        valid &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
    bary = np.asarray(z["barycentric"], dtype=np.float32)
    valid &= np.all(np.isfinite(bary), axis=0)
    valid &= np.all(bary >= -0.05, axis=0)
    valid &= np.all(bary <= 1.05, axis=0)
    valid &= np.isin(face_id, candidate_faces)
    return valid


def _residual_stats(pred_delta: np.ndarray, target_delta: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_delta, dtype=np.float32)[:, mask]
    target = np.asarray(target_delta, dtype=np.float32)[:, mask]
    if pred.size == 0 or target.size == 0:
        return {
            "pixel_count": 0,
            "target_energy": 0.0,
            "pred_energy": 0.0,
            "energy_retention": 0.0,
            "residual_mse": 0.0,
            "residual_psnr": 0.0,
            "cosine": 0.0,
            "sign_agreement": 0.0,
            "changed_fraction": 0.0,
        }
    target_energy = float(np.mean(np.sum(target * target, axis=0)))
    pred_energy = float(np.mean(np.sum(pred * pred, axis=0)))
    diff = pred - target
    residual_mse = float(np.mean(diff * diff))
    dot = float(np.sum(pred.astype(np.float64) * target.astype(np.float64)))
    denom = math.sqrt(float(np.sum(pred.astype(np.float64) ** 2)) * float(np.sum(target.astype(np.float64) ** 2)))
    nonzero = (np.abs(pred) > 1.0e-6) & (np.abs(target) > 1.0e-6)
    return {
        "pixel_count": int(mask.sum()),
        "target_energy": target_energy,
        "pred_energy": pred_energy,
        "energy_retention": float(pred_energy / max(target_energy, 1.0e-12)),
        "residual_mse": residual_mse,
        "residual_psnr": float(-10.0 * math.log10(max(residual_mse, 1.0e-12))),
        "cosine": float(dot / denom) if denom > 1.0e-12 else 0.0,
        "sign_agreement": float(np.mean(np.sign(pred[nonzero]) == np.sign(target[nonzero]))) if np.any(nonzero) else 0.0,
        "changed_fraction": float(np.mean(np.any(np.abs(pred) > (0.5 / 255.0), axis=0))),
    }


def _summarize_residual_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if int(row.get("pixel_count", 0)) > 0]
    return {
        "view_count": int(len(rows)),
        "valid_view_count": int(len(usable)),
        "target_energy": _mean([float(row["target_energy"]) for row in usable]),
        "pred_energy": _mean([float(row["pred_energy"]) for row in usable]),
        "energy_retention": _mean([float(row["energy_retention"]) for row in usable]),
        "energy_retention_quantiles": _quantiles([float(row["energy_retention"]) for row in usable]),
        "residual_mse": _mean([float(row["residual_mse"]) for row in usable]),
        "residual_psnr": _mean([float(row["residual_psnr"]) for row in usable]),
        "cosine": _mean([float(row["cosine"]) for row in usable]),
        "cosine_quantiles": _quantiles([float(row["cosine"]) for row in usable]),
        "sign_agreement": _mean([float(row["sign_agreement"]) for row in usable]),
        "changed_fraction": _mean([float(row["changed_fraction"]) for row in usable]),
    }


def _insert_source_entry(
    bank: dict[str, np.ndarray],
    face_idx: int,
    bin_id: int,
    score: float,
    count: float,
    residual: np.ndarray,
    parent: np.ndarray,
    normal: np.ndarray,
    camera: np.ndarray,
    gain: float,
    alpha: float,
) -> None:
    scores = bank["score"][face_idx, bin_id]
    slot = int(np.argmin(scores))
    if float(score) <= float(scores[slot]):
        return
    bank["score"][face_idx, bin_id, slot] = float(score)
    bank["counts"][face_idx, bin_id, slot] = float(count)
    bank["residual"][face_idx, bin_id, slot] = np.asarray(residual, dtype=np.float32)
    bank["parent_rgb"][face_idx, bin_id, slot] = np.asarray(parent, dtype=np.float32)
    bank["normal"][face_idx, bin_id, slot] = np.asarray(normal, dtype=np.float32)
    bank["camera_dir"][face_idx, bin_id, slot] = np.asarray(camera, dtype=np.float32)
    bank["gain_l1"][face_idx, bin_id, slot] = float(gain)
    bank["alpha"][face_idx, bin_id, slot] = float(alpha)


def _fit_source_bank(
    fit_paths: list[Path],
    candidate_faces: np.ndarray,
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    grid: int,
    source_top_k: int,
    max_samples_per_view: int,
    score_gain_weight: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    bins = int(grid) * int(grid)
    shape = (int(candidate_faces.size), bins, int(source_top_k))
    bank: dict[str, np.ndarray] = {
        "score": np.full(shape, -np.inf, dtype=np.float32),
        "counts": np.zeros(shape, dtype=np.float32),
        "residual": np.zeros((*shape, 3), dtype=np.float32),
        "parent_rgb": np.zeros((*shape, 3), dtype=np.float32),
        "normal": np.zeros((*shape, 3), dtype=np.float32),
        "camera_dir": np.zeros((*shape, 3), dtype=np.float32),
        "gain_l1": np.zeros(shape, dtype=np.float32),
        "alpha": np.zeros(shape, dtype=np.float32),
    }
    active_pixels = 0
    sampled_pixels = 0
    inserted_entries = 0
    grouped_entries = 0
    for path in tqdm(fit_paths, desc="fit deferred source bank"):
        with np.load(path, allow_pickle=False) as z:
            mask = _valid_mask(
                z,
                candidate_faces,
                residual_l1_key=str(residual_l1_key),
                min_l1=float(min_l1),
                min_alpha=float(min_alpha),
            )
            ys, xs = np.nonzero(mask)
            active_pixels += int(ys.size)
            if ys.size == 0:
                continue
            if int(max_samples_per_view) > 0 and ys.size > int(max_samples_per_view):
                score = np.asarray(z[residual_l1_key], dtype=np.float32)[ys, xs]
                score = np.clip(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
                probs = None
                total = float(np.sum(score))
                if total > 1.0e-12 and np.isfinite(total):
                    probs = score.astype(np.float64) / total
                    probs = probs / max(float(np.sum(probs)), 1.0e-12)
                take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False, p=probs)
                ys, xs = ys[take], xs[take]
            sampled_pixels += int(ys.size)
            faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
            face_idx, ok = _face_indices(faces, candidate_faces)
            if not np.any(ok):
                continue
            ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
            bin_id = _bin_ids(z, ys, xs, int(grid))
            group = face_idx.astype(np.int64) * bins + bin_id.astype(np.int64)
            unique, inv = np.unique(group, return_inverse=True)
            counts = np.bincount(inv).astype(np.float64)
            residual = np.asarray(z[residual_rgb_key], dtype=np.float32)[:3]
            parent = np.asarray(z["rgb_render"], dtype=np.float32)[:3]
            normal = np.asarray(z["normal"], dtype=np.float32)[:3]
            alpha_map = np.asarray(z["alpha"], dtype=np.float32) if "alpha" in z else np.ones_like(mask, dtype=np.float32)
            gain_map = (
                np.asarray(z["teacher_gain_l1"], dtype=np.float32)
                if "teacher_gain_l1" in z
                else np.asarray(z[residual_l1_key], dtype=np.float32)
            )
            sums = {
                "residual": np.zeros((unique.size, 3), dtype=np.float64),
                "parent": np.zeros((unique.size, 3), dtype=np.float64),
                "normal": np.zeros((unique.size, 3), dtype=np.float64),
            }
            for c in range(3):
                sums["residual"][:, c] = np.bincount(inv, weights=residual[c, ys, xs].astype(np.float64), minlength=unique.size)
                sums["parent"][:, c] = np.bincount(inv, weights=parent[c, ys, xs].astype(np.float64), minlength=unique.size)
                sums["normal"][:, c] = np.bincount(inv, weights=normal[c, ys, xs].astype(np.float64), minlength=unique.size)
            gain_sum = np.bincount(inv, weights=gain_map[ys, xs].astype(np.float64), minlength=unique.size)
            alpha_sum = np.bincount(inv, weights=alpha_map[ys, xs].astype(np.float64), minlength=unique.size)
            camera = _camera_dir(z)
            means_residual = (sums["residual"] / np.maximum(counts[:, None], 1.0)).astype(np.float32)
            means_parent = np.clip((sums["parent"] / np.maximum(counts[:, None], 1.0)).astype(np.float32), 0.0, 1.0)
            means_normal = _normalize_rows((sums["normal"] / np.maximum(counts[:, None], 1.0)).astype(np.float32))
            means_gain = (gain_sum / np.maximum(counts, 1.0)).astype(np.float32)
            means_alpha = (alpha_sum / np.maximum(counts, 1.0)).astype(np.float32)
            energy = np.sum(means_residual * means_residual, axis=1)
            source_score = counts.astype(np.float32) * energy * (
                1.0 + float(score_gain_weight) * np.clip(means_gain, 0.0, None)
            )
            for i, packed in enumerate(unique):
                fi = int(packed // bins)
                bi = int(packed % bins)
                before = float(np.min(bank["score"][fi, bi]))
                _insert_source_entry(
                    bank,
                    fi,
                    bi,
                    float(source_score[i]),
                    float(counts[i]),
                    means_residual[i],
                    means_parent[i],
                    means_normal[i],
                    camera,
                    float(means_gain[i]),
                    float(means_alpha[i]),
                )
                if float(np.min(bank["score"][fi, bi])) != before:
                    inserted_entries += 1
            grouped_entries += int(unique.size)
    valid = bank["counts"] > 0.0
    bank["score"] = np.where(np.isfinite(bank["score"]), bank["score"], 0.0).astype(np.float32)
    return bank, {
        "fit_view_count": int(len(fit_paths)),
        "fit_active_pixels": int(active_pixels),
        "fit_sampled_pixels": int(sampled_pixels),
        "grouped_source_entries": int(grouped_entries),
        "inserted_topk_entries": int(inserted_entries),
        "source_top_k": int(source_top_k),
        "grid": int(grid),
        "bins_per_face": int(bins),
        "nonempty_face_bins": int(np.count_nonzero(np.any(valid, axis=2))),
        "nonempty_face_bin_fraction": float(np.mean(np.any(valid, axis=2))),
        "nonempty_source_slots": int(np.count_nonzero(valid)),
        "mean_source_count_nonempty": float(np.mean(bank["counts"][valid])) if np.any(valid) else 0.0,
        "source_count_quantiles": _quantiles([float(x) for x in bank["counts"][valid].reshape(-1)]) if np.any(valid) else _quantiles([]),
    }


def _load_source_bank(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        candidate_faces = np.asarray(z["candidate_faces"], dtype=np.int64)
        bank = {
            "score": np.asarray(z["score"], dtype=np.float32),
            "counts": np.asarray(z["counts"], dtype=np.float32),
            "residual": np.asarray(z["residual"], dtype=np.float32),
            "parent_rgb": np.asarray(z["parent_rgb"], dtype=np.float32),
            "normal": np.asarray(z["normal"], dtype=np.float32),
            "camera_dir": np.asarray(z["camera_dir"], dtype=np.float32),
            "gain_l1": np.asarray(z["gain_l1"], dtype=np.float32),
            "alpha": np.asarray(z["alpha"], dtype=np.float32),
        }
        if "policy_reliability" in z:
            bank["policy_reliability"] = np.asarray(z["policy_reliability"], dtype=np.float32)
        if "policy_gain" in z:
            bank["policy_gain"] = np.asarray(z["policy_gain"], dtype=np.float32)
        if "policy_tail_risk" in z:
            bank["policy_tail_risk"] = np.asarray(z["policy_tail_risk"], dtype=np.float32)
        for key in (
            "learned_ood_head_coef",
            "learned_ood_head_bias",
            "learned_ood_head_mean",
            "learned_ood_head_scale",
            "learned_ood_head_floor",
        ):
            if key in z:
                bank[key] = np.asarray(z[key], dtype=np.float32)
        args_json = str(np.asarray(z["args_json"]).item()) if "args_json" in z else "{}"
    valid = bank["counts"] > 0.0
    bins = int(bank["counts"].shape[1]) if bank["counts"].ndim >= 2 else 0
    top_k = int(bank["counts"].shape[2]) if bank["counts"].ndim >= 3 else 0
    return candidate_faces, bank, {
        "loaded_from_checkpoint": str(path),
        "loaded_args_json": args_json,
        "source_top_k": int(top_k),
        "grid": int(round(math.sqrt(bins))) if bins > 0 else 0,
        "bins_per_face": int(bins),
        "nonempty_face_bins": int(np.count_nonzero(np.any(valid, axis=2))) if valid.ndim == 3 else 0,
        "nonempty_face_bin_fraction": float(np.mean(np.any(valid, axis=2))) if valid.ndim == 3 else 0.0,
        "nonempty_source_slots": int(np.count_nonzero(valid)),
        "mean_source_count_nonempty": float(np.mean(bank["counts"][valid])) if np.any(valid) else 0.0,
        "source_count_quantiles": _quantiles([float(x) for x in bank["counts"][valid].reshape(-1)]) if np.any(valid) else _quantiles([]),
    }


def _rank_target_visible_faces(
    target_dir: Path,
    *,
    min_alpha: float,
    quota: int,
    max_samples_per_view: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    paths = evidence_views(target_dir)
    if int(quota) <= 0 or not paths:
        return np.zeros((0,), dtype=np.int64), {"enabled": False, "reason": "empty quota or target dir"}
    rng = np.random.default_rng(int(seed) + 263)
    counts: dict[int, int] = {}
    total_samples = 0
    bad: list[dict[str, Any]] = []
    for path in tqdm(paths, desc="rank target-visible faces"):
        with np.load(path, allow_pickle=False) as z:
            leaked = sorted(set(z.files) & FORBIDDEN_TARGET_KEYS)
            if leaked:
                bad.append({"view": path.stem, "forbidden_keys": leaked})
                continue
            face_id = np.asarray(z["face_id"], dtype=np.int64)
            valid = face_id >= 0
            if "barycentric_valid" in z:
                valid &= np.asarray(z["barycentric_valid"]).astype(bool)
            if "alpha" in z:
                valid &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
            if "barycentric" in z:
                bary = np.asarray(z["barycentric"], dtype=np.float32)
                valid &= np.all(np.isfinite(bary), axis=0)
                valid &= np.all(bary >= -0.05, axis=0)
                valid &= np.all(bary <= 1.05, axis=0)
            ys, xs = np.nonzero(valid)
            if ys.size == 0:
                continue
            if int(max_samples_per_view) > 0 and ys.size > int(max_samples_per_view):
                take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
                ys, xs = ys[take], xs[take]
            faces = face_id[ys, xs]
            total_samples += int(faces.size)
            unique, unique_counts = np.unique(faces, return_counts=True)
            for face, count in zip(unique.tolist(), unique_counts.tolist(), strict=True):
                counts[int(face)] = counts.get(int(face), 0) + int(count)
    if bad:
        raise RuntimeError(f"target-visible expansion received forbidden target keys: {bad[:4]}")
    ranked = sorted(counts, key=lambda f: counts[f], reverse=True)
    selected = np.asarray(sorted(ranked[: int(quota)]), dtype=np.int64)
    selected_count = int(sum(counts[int(face)] for face in selected.tolist())) if selected.size else 0
    return selected, {
        "enabled": True,
        "target_evidence_dir": str(target_dir),
        "target_views": int(len(paths)),
        "quota": int(quota),
        "ranked_faces": int(len(ranked)),
        "selected_faces": int(selected.size),
        "total_sampled_pixels": int(total_samples),
        "selected_visible_samples": int(selected_count),
        "selected_visible_fraction": float(selected_count / max(total_samples, 1)),
        "max_samples_per_view": int(max_samples_per_view),
        "uses_target_or_test_gt": False,
    }


def _transform_bank_residuals(bank: dict[str, np.ndarray], mode: str, chroma_shrink: float) -> dict[str, Any]:
    mode_s = str(mode or "raw_rgb")
    residual = np.asarray(bank["residual"], dtype=np.float32)
    before = residual.copy()
    if mode_s == "raw_rgb":
        return {
            "mode": "raw_rgb",
            "chroma_shrink": float(chroma_shrink),
            "energy_ratio_after_before": 1.0,
            "mean_abs_before": float(np.mean(np.abs(before))),
            "mean_abs_after": float(np.mean(np.abs(before))),
        }
    luma = np.sum(residual * LUMA_WEIGHTS.reshape(1, 1, 1, 3), axis=3, keepdims=True).astype(np.float32)
    luma_rgb = np.repeat(luma, 3, axis=3)
    if mode_s == "luma_only":
        transformed = luma_rgb
    elif mode_s == "chroma_shrink":
        shrink = float(np.clip(float(chroma_shrink), 0.0, 1.0))
        transformed = luma_rgb + shrink * (residual - luma_rgb)
    else:
        raise ValueError(f"unsupported bank residual transform mode: {mode_s}")
    valid = bank["counts"] > 0.0
    bank["residual"] = transformed.astype(np.float32)
    before_energy = float(np.mean(np.sum(before[valid] * before[valid], axis=1))) if np.any(valid) else 0.0
    after = bank["residual"]
    after_energy = float(np.mean(np.sum(after[valid] * after[valid], axis=1))) if np.any(valid) else 0.0
    return {
        "mode": mode_s,
        "chroma_shrink": float(chroma_shrink),
        "energy_before": before_energy,
        "energy_after": after_energy,
        "energy_ratio_after_before": float(after_energy / max(before_energy, 1.0e-12)),
        "mean_abs_before": float(np.mean(np.abs(before[valid]))) if np.any(valid) else 0.0,
        "mean_abs_after": float(np.mean(np.abs(after[valid]))) if np.any(valid) else 0.0,
    }


def _calibrate_policy_reliability(
    val_paths: list[Path],
    candidate_faces: np.ndarray,
    bank: dict[str, np.ndarray],
    *,
    grid: int,
    alpha: float,
    min_alpha: float,
    min_source_count: float,
    chunk_size: int,
    view_beta: float,
    normal_beta: float,
    parent_beta: float,
    count_gamma: float,
    gain_beta: float,
    confidence_tau: float,
    source_agreement_mode: str,
    source_agreement_beta: float,
    source_agreement_min_confidence: float,
    mode: str,
    min_count: int,
    min_positive_fraction: float,
    min_mean_gain: float,
    gain_scale: float,
    floor: float,
    patch_radius: int,
    patch_gain_weight: float,
    gradient_gain_weight: float,
    policy_gain_mode: str,
    policy_gain_max: float,
    policy_gain_scale: float,
    residual_decoder_mode: str = "weighted_average",
    local_linear_l2: float = 0.05,
    local_linear_blend: float = 1.0,
    local_linear_min_sources: int = 3,
    local_linear_residual_clip: float = 0.12,
) -> dict[str, Any]:
    bins = int(grid) * int(grid)
    counts = np.zeros((int(candidate_faces.size), bins), dtype=np.float64)
    positive = np.zeros_like(counts)
    gain_sum = np.zeros_like(counts)
    gain_sq_sum = np.zeros_like(counts)
    negative_gain_sum = np.zeros_like(counts)
    l1_gain_sum = np.zeros_like(counts)
    patch_gain_sum = np.zeros_like(counts)
    grad_gain_sum = np.zeros_like(counts)
    mode_s = str(mode or "local_l1")
    for path in tqdm(val_paths, desc="calibrate policy reliability"):
        with np.load(path, allow_pickle=False) as z:
            parent = np.asarray(z["rgb_render"], dtype=np.float32)[:3]
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)[:3]
            delta, active, _support_stats = _predict_delta(
                z,
                candidate_faces,
                bank,
                grid=int(grid),
                min_alpha=float(min_alpha),
                min_source_count=float(min_source_count),
                chunk_size=int(chunk_size),
                view_beta=float(view_beta),
                normal_beta=float(normal_beta),
                parent_beta=float(parent_beta),
                count_gamma=float(count_gamma),
                gain_beta=float(gain_beta),
                confidence_tau=float(confidence_tau),
                source_agreement_mode=str(source_agreement_mode),
                source_agreement_beta=float(source_agreement_beta),
                source_agreement_min_confidence=float(source_agreement_min_confidence),
                residual_decoder_mode=str(residual_decoder_mode),
                local_linear_l2=float(local_linear_l2),
                local_linear_blend=float(local_linear_blend),
                local_linear_min_sources=int(local_linear_min_sources),
                local_linear_residual_clip=float(local_linear_residual_clip),
            )
            ys, xs = np.nonzero(active)
            if ys.size == 0:
                continue
            faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
            face_idx, ok = _face_indices(faces, candidate_faces)
            if not np.any(ok):
                continue
            ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
            bin_id = _bin_ids(z, ys, xs, int(grid))
            candidate = np.clip(parent + float(alpha) * delta, 0.0, 1.0)
            parent_l1 = np.mean(np.abs(parent[:, ys, xs] - gt[:, ys, xs]), axis=0).astype(np.float64)
            candidate_l1 = np.mean(np.abs(candidate[:, ys, xs] - gt[:, ys, xs]), axis=0).astype(np.float64)
            l1_gain = parent_l1 - candidate_l1
            gain = l1_gain.copy()
            patch_gain = np.zeros_like(gain, dtype=np.float64)
            grad_gain = np.zeros_like(gain, dtype=np.float64)
            if mode_s == "patch_perceptual_v1":
                parent_luma = _luma(parent)
                candidate_luma = _luma(candidate)
                gt_luma = _luma(gt)
                parent_patch_error = _box_mean2d(np.abs(parent_luma - gt_luma), int(patch_radius))
                candidate_patch_error = _box_mean2d(np.abs(candidate_luma - gt_luma), int(patch_radius))
                patch_gain = (parent_patch_error[ys, xs] - candidate_patch_error[ys, xs]).astype(np.float64)
                parent_grad_error = np.abs(_luma_gradient_magnitude(parent_luma) - _luma_gradient_magnitude(gt_luma))
                candidate_grad_error = np.abs(_luma_gradient_magnitude(candidate_luma) - _luma_gradient_magnitude(gt_luma))
                grad_gain = (parent_grad_error - candidate_grad_error)[ys, xs].astype(np.float64)
                gain = (
                    l1_gain
                    + float(patch_gain_weight) * patch_gain
                    + float(gradient_gain_weight) * grad_gain
                )
            elif mode_s != "local_l1":
                raise ValueError(f"unsupported policy reliability mode for calibration: {mode_s}")
            np.add.at(counts, (face_idx, bin_id), 1.0)
            np.add.at(positive, (face_idx, bin_id), gain > 0.0)
            np.add.at(gain_sum, (face_idx, bin_id), gain)
            np.add.at(gain_sq_sum, (face_idx, bin_id), gain * gain)
            np.add.at(negative_gain_sum, (face_idx, bin_id), np.maximum(-gain, 0.0))
            np.add.at(l1_gain_sum, (face_idx, bin_id), l1_gain)
            np.add.at(patch_gain_sum, (face_idx, bin_id), patch_gain)
            np.add.at(grad_gain_sum, (face_idx, bin_id), grad_gain)
    mean_gain = np.divide(gain_sum, np.maximum(counts, 1.0), out=np.zeros_like(gain_sum), where=counts > 0.0)
    mean_gain_sq = np.divide(
        gain_sq_sum,
        np.maximum(counts, 1.0),
        out=np.zeros_like(gain_sq_sum),
        where=counts > 0.0,
    )
    gain_std = np.sqrt(np.maximum(mean_gain_sq - mean_gain * mean_gain, 0.0))
    mean_negative_gain = np.divide(
        negative_gain_sum,
        np.maximum(counts, 1.0),
        out=np.zeros_like(negative_gain_sum),
        where=counts > 0.0,
    )
    mean_l1_gain = np.divide(l1_gain_sum, np.maximum(counts, 1.0), out=np.zeros_like(l1_gain_sum), where=counts > 0.0)
    mean_patch_gain = np.divide(
        patch_gain_sum,
        np.maximum(counts, 1.0),
        out=np.zeros_like(patch_gain_sum),
        where=counts > 0.0,
    )
    mean_grad_gain = np.divide(
        grad_gain_sum,
        np.maximum(counts, 1.0),
        out=np.zeros_like(grad_gain_sum),
        where=counts > 0.0,
    )
    positive_fraction = np.divide(positive, np.maximum(counts, 1.0), out=np.zeros_like(positive), where=counts > 0.0)
    min_pos = float(np.clip(float(min_positive_fraction), 0.0, 0.999))
    pos_score = np.clip((positive_fraction - min_pos) / max(1.0 - min_pos, 1.0e-6), 0.0, 1.0)
    gain_score = 1.0 / (
        1.0 + np.exp(-np.clip((mean_gain - float(min_mean_gain)) / max(float(gain_scale), 1.0e-8), -50.0, 50.0))
    )
    reliable = (counts >= int(min_count)).astype(np.float64)
    reliability = reliable * pos_score * gain_score
    floor_v = float(np.clip(float(floor), 0.0, 1.0))
    reliability = floor_v + (1.0 - floor_v) * reliability
    reliability = reliability.astype(np.float32)
    bank["policy_reliability"] = reliability
    risk_scale = max(float(gain_scale), 1.0e-8)
    tail_risk = reliable * np.clip(
        0.40 * (1.0 - positive_fraction)
        + 0.35 * np.clip(mean_negative_gain / risk_scale, 0.0, 1.0)
        + 0.25 * np.clip(gain_std / risk_scale, 0.0, 1.0),
        0.0,
        1.0,
    )
    tail_risk = tail_risk.astype(np.float32)
    bank["policy_tail_risk"] = tail_risk
    gain_mode_s = str(policy_gain_mode or "off")
    policy_gain = np.ones_like(reliability, dtype=np.float32)
    if gain_mode_s == "positive_soft":
        max_gain = max(1.0, float(policy_gain_max))
        gain_norm = np.clip(np.maximum(mean_gain - float(min_mean_gain), 0.0) / max(float(policy_gain_scale), 1.0e-8), 0.0, 1.0)
        policy_gain = (1.0 + (max_gain - 1.0) * reliability.astype(np.float64) * gain_norm).astype(np.float32)
        bank["policy_gain"] = policy_gain
    elif gain_mode_s != "off":
        raise ValueError(f"unsupported policy_gain_mode for calibration: {gain_mode_s}")
    valid = counts >= int(min_count)
    active = reliability > (floor_v + 1.0e-6)
    return {
        "mode": mode_s,
        "alpha": float(alpha),
        "policy_val_views": int(len(val_paths)),
        "min_count": int(min_count),
        "min_positive_fraction": float(min_positive_fraction),
        "min_mean_gain": float(min_mean_gain),
        "gain_scale": float(gain_scale),
        "floor": float(floor_v),
        "patch_radius": int(patch_radius),
        "patch_gain_weight": float(patch_gain_weight),
        "gradient_gain_weight": float(gradient_gain_weight),
        "observed_bins": int(np.count_nonzero(counts > 0.0)),
        "valid_bins": int(np.count_nonzero(valid)),
        "active_bins": int(np.count_nonzero(active)),
        "active_bin_fraction": float(np.mean(active)),
        "mean_reliability": float(np.mean(reliability)),
        "mean_reliability_valid": float(np.mean(reliability[valid])) if np.any(valid) else 0.0,
        "mean_positive_fraction_valid": float(np.mean(positive_fraction[valid])) if np.any(valid) else 0.0,
        "mean_gain_valid": float(np.mean(mean_gain[valid])) if np.any(valid) else 0.0,
        "mean_gain_std_valid": float(np.mean(gain_std[valid])) if np.any(valid) else 0.0,
        "mean_negative_gain_valid": float(np.mean(mean_negative_gain[valid])) if np.any(valid) else 0.0,
        "mean_l1_gain_valid": float(np.mean(mean_l1_gain[valid])) if np.any(valid) else 0.0,
        "mean_patch_gain_valid": float(np.mean(mean_patch_gain[valid])) if np.any(valid) else 0.0,
        "mean_gradient_gain_valid": float(np.mean(mean_grad_gain[valid])) if np.any(valid) else 0.0,
        "policy_gain_mode": gain_mode_s,
        "policy_gain_max": float(policy_gain_max),
        "policy_gain_scale": float(policy_gain_scale),
        "mean_policy_gain_valid": float(np.mean(policy_gain[valid])) if np.any(valid) else 1.0,
        "max_policy_gain": float(np.max(policy_gain)) if policy_gain.size else 1.0,
        "policy_gain_quantiles": _quantiles([float(x) for x in policy_gain.reshape(-1)]),
        "valid_policy_gain_quantiles": _quantiles([float(x) for x in policy_gain[valid].reshape(-1)]) if np.any(valid) else _quantiles([]),
        "mean_tail_risk_valid": float(np.mean(tail_risk[valid])) if np.any(valid) else 0.0,
        "max_tail_risk": float(np.max(tail_risk)) if tail_risk.size else 0.0,
        "tail_risk_quantiles": _quantiles([float(x) for x in tail_risk.reshape(-1)]),
        "valid_tail_risk_quantiles": _quantiles([float(x) for x in tail_risk[valid].reshape(-1)]) if np.any(valid) else _quantiles([]),
        "reliability_quantiles": _quantiles([float(x) for x in reliability.reshape(-1)]),
        "valid_reliability_quantiles": _quantiles([float(x) for x in reliability[valid].reshape(-1)]) if np.any(valid) else _quantiles([]),
    }


def _fit_learned_ood_head(
    val_paths: list[Path],
    candidate_faces: np.ndarray,
    bank: dict[str, np.ndarray],
    *,
    grid: int,
    alpha: float,
    min_alpha: float,
    min_source_count: float,
    chunk_size: int,
    view_beta: float,
    normal_beta: float,
    parent_beta: float,
    count_gamma: float,
    gain_beta: float,
    confidence_tau: float,
    source_agreement_mode: str,
    source_agreement_beta: float,
    source_agreement_min_confidence: float,
    l2: float,
    gain_scale: float,
    floor: float,
    min_gain: float,
    max_samples: int,
    seed: int,
    residual_decoder_mode: str = "weighted_average",
    local_linear_l2: float = 0.05,
    local_linear_blend: float = 1.0,
    local_linear_min_sources: int = 3,
    local_linear_residual_clip: float = 0.12,
) -> dict[str, Any]:
    if not val_paths:
        raise RuntimeError("learned OOD head requested but no policy-val views are available")
    rng = np.random.default_rng(int(seed) + 260)
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    gain_parts: list[np.ndarray] = []
    view_rows: list[dict[str, Any]] = []
    max_samples_i = max(1, int(max_samples))
    per_view_cap = max(2048, int(math.ceil(2.0 * max_samples_i / max(len(val_paths), 1))))
    for path in tqdm(val_paths, desc="fit learned OOD/gain head"):
        with np.load(path, allow_pickle=False) as z:
            parent = np.asarray(z["rgb_render"], dtype=np.float32)[:3]
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)[:3]
            feature_rows: list[dict[str, np.ndarray]] = []
            delta, _active, support_stats = _predict_delta(
                z,
                candidate_faces,
                bank,
                grid=int(grid),
                min_alpha=float(min_alpha),
                min_source_count=float(min_source_count),
                chunk_size=int(chunk_size),
                view_beta=float(view_beta),
                normal_beta=float(normal_beta),
                parent_beta=float(parent_beta),
                count_gamma=float(count_gamma),
                gain_beta=float(gain_beta),
                confidence_tau=float(confidence_tau),
                source_agreement_mode=str(source_agreement_mode),
                source_agreement_beta=float(source_agreement_beta),
                source_agreement_min_confidence=float(source_agreement_min_confidence),
                residual_decoder_mode=str(residual_decoder_mode),
                local_linear_l2=float(local_linear_l2),
                local_linear_blend=float(local_linear_blend),
                local_linear_min_sources=int(local_linear_min_sources),
                local_linear_residual_clip=float(local_linear_residual_clip),
                ood_gain_mode="off",
                feature_rows=feature_rows,
            )
            ys, xs, features = _feature_rows_to_arrays(feature_rows)
            if features.shape[0] == 0:
                view_rows.append({"view": path.stem, "sample_count": 0, **support_stats})
                continue
            if features.shape[0] > per_view_cap:
                take = rng.choice(features.shape[0], size=per_view_cap, replace=False)
                ys = ys[take]
                xs = xs[take]
                features = features[take]
            candidate = np.clip(parent + float(alpha) * delta, 0.0, 1.0)
            parent_l1 = np.mean(np.abs(parent[:, ys, xs] - gt[:, ys, xs]), axis=0).astype(np.float32)
            candidate_l1 = np.mean(np.abs(candidate[:, ys, xs] - gt[:, ys, xs]), axis=0).astype(np.float32)
            local_gain = (parent_l1 - candidate_l1).astype(np.float32)
            labels = _sigmoid((local_gain - float(min_gain)) / max(float(gain_scale), 1.0e-8))
            labels = np.clip(labels, float(floor), 1.0).astype(np.float32)
            feature_parts.append(features.astype(np.float32))
            label_parts.append(labels)
            gain_parts.append(local_gain)
            view_rows.append(
                {
                    "view": path.stem,
                    "sample_count": int(features.shape[0]),
                    "mean_local_gain": float(np.mean(local_gain)) if local_gain.size else 0.0,
                    "positive_local_gain_fraction": float(np.mean(local_gain > 0.0)) if local_gain.size else 0.0,
                    **support_stats,
                }
            )
    if not feature_parts:
        raise RuntimeError("learned OOD head requested but no feature samples were collected")
    features_all = np.concatenate(feature_parts, axis=0).astype(np.float32)
    labels_all = np.concatenate(label_parts, axis=0).astype(np.float32)
    gains_all = np.concatenate(gain_parts, axis=0).astype(np.float32)
    if features_all.shape[0] > max_samples_i:
        take = rng.choice(features_all.shape[0], size=max_samples_i, replace=False)
        features_all = features_all[take]
        labels_all = labels_all[take]
        gains_all = gains_all[take]
    min_required = max(32, len(LEARNED_OOD_FEATURE_NAMES) * 4)
    if features_all.shape[0] < min_required:
        raise RuntimeError(f"learned OOD head has too few samples: {features_all.shape[0]} < {min_required}")
    mean = np.mean(features_all, axis=0).astype(np.float32)
    scale = np.maximum(np.std(features_all, axis=0).astype(np.float32), 1.0e-6)
    x = (features_all - mean.reshape(1, -1)) / scale.reshape(1, -1)
    y_mean = float(np.mean(labels_all))
    y = (labels_all - y_mean).astype(np.float32)
    xtx = (x.T @ x).astype(np.float64) / float(x.shape[0])
    xty = (x.T @ y).astype(np.float64) / float(x.shape[0])
    ridge = max(float(l2), 0.0)
    coef = np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0], dtype=np.float64), xty).astype(np.float32)
    pred_train = np.clip(y_mean + np.sum(x * coef.reshape(1, -1), axis=1), float(floor), 1.0).astype(np.float32)
    bank["learned_ood_head_coef"] = coef
    bank["learned_ood_head_bias"] = np.asarray([y_mean], dtype=np.float32)
    bank["learned_ood_head_mean"] = mean.astype(np.float32)
    bank["learned_ood_head_scale"] = scale.astype(np.float32)
    bank["learned_ood_head_floor"] = np.asarray([float(floor)], dtype=np.float32)
    corr = 0.0
    if float(np.std(pred_train)) > 1.0e-8 and float(np.std(labels_all)) > 1.0e-8:
        corr = float(np.corrcoef(pred_train, labels_all)[0, 1])
    return {
        "mode": "learned_linear",
        "feature_names": list(LEARNED_OOD_FEATURE_NAMES),
        "alpha": float(alpha),
        "l2": float(l2),
        "gain_scale": float(gain_scale),
        "min_gain": float(min_gain),
        "floor": float(floor),
        "sample_count": int(features_all.shape[0]),
        "view_count": int(len(val_paths)),
        "label_mean": float(np.mean(labels_all)),
        "label_quantiles": _quantiles([float(x) for x in labels_all.reshape(-1)]),
        "local_gain_mean": float(np.mean(gains_all)),
        "local_gain_quantiles": _quantiles([float(x) for x in gains_all.reshape(-1)]),
        "positive_local_gain_fraction": float(np.mean(gains_all > 0.0)),
        "predicted_confidence_mean": float(np.mean(pred_train)),
        "predicted_confidence_quantiles": _quantiles([float(x) for x in pred_train.reshape(-1)]),
        "label_prediction_correlation": corr,
        "coefficients": {
            name: float(value) for name, value in zip(LEARNED_OOD_FEATURE_NAMES, coef.tolist(), strict=True)
        },
        "views_preview": view_rows[:8],
    }


def _predict_delta(
    z: np.lib.npyio.NpzFile,
    candidate_faces: np.ndarray,
    bank: dict[str, np.ndarray],
    *,
    grid: int,
    min_alpha: float,
    min_source_count: float,
    chunk_size: int,
    view_beta: float,
    normal_beta: float,
    parent_beta: float,
    count_gamma: float,
    gain_beta: float,
    confidence_tau: float,
    source_agreement_mode: str,
    source_agreement_beta: float,
    source_agreement_min_confidence: float,
    residual_decoder_mode: str = "weighted_average",
    local_linear_l2: float = 0.05,
    local_linear_blend: float = 1.0,
    local_linear_min_sources: int = 3,
    local_linear_residual_clip: float = 0.12,
    ood_gain_mode: str = "off",
    ood_gain_beta: float = 1.0,
    ood_gain_view_weight: float = 1.0,
    ood_gain_variance_weight: float = 1.0,
    ood_gain_parent_weight: float = 1.0,
    ood_gain_effective_count_weight: float = 0.5,
    feature_rows: list[dict[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    parent = np.asarray(z["rgb_render"], dtype=np.float32)[:3]
    delta = np.zeros_like(parent, dtype=np.float32)
    active = np.zeros(parent.shape[1:], dtype=bool)
    support = _surface_support_mask(z, candidate_faces, min_alpha=float(min_alpha))
    ys_all, xs_all = np.nonzero(support)
    if ys_all.size == 0:
        return delta, active, {"surface_support_fraction": 0.0, "active_fraction": 0.0, "mean_confidence": 0.0}
    faces = np.asarray(z["face_id"], dtype=np.int64)[ys_all, xs_all]
    face_idx_all, ok = _face_indices(faces, candidate_faces)
    if not np.any(ok):
        return delta, active, {"surface_support_fraction": float(np.mean(support)), "active_fraction": 0.0, "mean_confidence": 0.0}
    ys_all, xs_all, face_idx_all = ys_all[ok], xs_all[ok], face_idx_all[ok]
    bin_all = _bin_ids(z, ys_all, xs_all, int(grid))
    target_cam = _camera_dir(z).reshape(1, 1, 3)
    normal_map = np.asarray(z["normal"], dtype=np.float32)[:3]
    confidences: list[float] = []
    agreement_confidences: list[float] = []
    policy_reliabilities: list[float] = []
    policy_gains: list[float] = []
    ood_confidences: list[float] = []
    ood_scores: list[float] = []
    tail_risks: list[float] = []
    active_count = 0
    for start in range(0, int(ys_all.size), int(chunk_size)):
        end = min(int(ys_all.size), start + int(chunk_size))
        ys, xs = ys_all[start:end], xs_all[start:end]
        face_idx = face_idx_all[start:end]
        bin_id = bin_all[start:end]
        counts = bank["counts"][face_idx, bin_id].astype(np.float32)
        valid_src = counts >= float(min_source_count)
        if not np.any(valid_src):
            continue
        src_residual = bank["residual"][face_idx, bin_id]
        src_parent = bank["parent_rgb"][face_idx, bin_id]
        src_normal = bank["normal"][face_idx, bin_id]
        src_camera = bank["camera_dir"][face_idx, bin_id]
        src_gain = bank["gain_l1"][face_idx, bin_id]
        tgt_parent = np.moveaxis(parent[:, ys, xs], 0, 1).astype(np.float32)
        tgt_normal = _normalize_rows(np.moveaxis(normal_map[:, ys, xs], 0, 1))
        view_cos = np.sum(src_camera * target_cam, axis=2)
        normal_cos = np.sum(src_normal * tgt_normal[:, None, :], axis=2)
        parent_dist = np.sum(np.square(src_parent - tgt_parent[:, None, :]), axis=2)
        weights = valid_src.astype(np.float32)
        if float(count_gamma) != 0.0:
            weights *= np.power(np.maximum(counts, 1.0), float(count_gamma)).astype(np.float32)
        if float(view_beta) != 0.0:
            weights *= np.exp(np.clip(float(view_beta) * (view_cos - 1.0), -30.0, 30.0)).astype(np.float32)
        if float(normal_beta) != 0.0:
            weights *= np.exp(np.clip(float(normal_beta) * (normal_cos - 1.0), -30.0, 30.0)).astype(np.float32)
        if float(parent_beta) != 0.0:
            weights *= np.exp(np.clip(-float(parent_beta) * parent_dist, -30.0, 30.0)).astype(np.float32)
        if float(gain_beta) != 0.0:
            weights *= np.clip(1.0 + float(gain_beta) * np.clip(src_gain, 0.0, None), 0.05, 10.0).astype(np.float32)
        denom = np.sum(weights, axis=1)
        ok_rows = denom > 1.0e-12
        if not np.any(ok_rows):
            continue
        pred = np.sum(weights[:, :, None] * src_residual, axis=1) / np.maximum(denom[:, None], 1.0e-12)
        decoder_mode_s = str(residual_decoder_mode or "weighted_average")
        if decoder_mode_s == "local_linear":
            valid_float = valid_src.astype(np.float32)
            valid_count = np.sum(valid_float, axis=1)
            enough_sources = (valid_count >= int(local_linear_min_sources)) & ok_rows
            if np.any(enough_sources):
                idx = np.nonzero(enough_sources)[0]
                src_feat = np.concatenate(
                    [
                        np.ones((idx.size, src_camera.shape[1], 1), dtype=np.float32),
                        src_camera[idx].astype(np.float32),
                        src_parent[idx].astype(np.float32),
                    ],
                    axis=2,
                )
                tgt_feat = np.concatenate(
                    [
                        np.ones((idx.size, 1), dtype=np.float32),
                        np.repeat(target_cam.reshape(1, 3), idx.size, axis=0).astype(np.float32),
                        tgt_parent[idx].astype(np.float32),
                    ],
                    axis=1,
                )
                row_weights = weights[idx].astype(np.float32)
                xw = src_feat * np.sqrt(np.maximum(row_weights, 0.0))[:, :, None]
                yw = src_residual[idx].astype(np.float32) * np.sqrt(np.maximum(row_weights, 0.0))[:, :, None]
                xtx = np.einsum("nkd,nke->nde", xw, xw, optimize=True).astype(np.float64)
                diag = np.arange(xtx.shape[1])
                xtx[:, diag, diag] += max(float(local_linear_l2), 1.0e-8)
                xty = np.einsum("nkd,nkc->ndc", xw, yw, optimize=True).astype(np.float64)
                coef = np.linalg.solve(xtx, xty).astype(np.float32)
                linear_pred = np.einsum("nd,ndc->nc", tgt_feat.astype(np.float32), coef, optimize=True)
                clip_v = float(local_linear_residual_clip)
                if clip_v > 0.0:
                    linear_pred = np.clip(linear_pred, -clip_v, clip_v)
                blend = float(np.clip(float(local_linear_blend), 0.0, 1.0))
                pred[idx] = ((1.0 - blend) * pred[idx] + blend * linear_pred).astype(np.float32)
        elif decoder_mode_s != "weighted_average":
            raise ValueError(f"unsupported residual_decoder_mode={residual_decoder_mode}")
        variance_ratio = np.zeros_like(denom, dtype=np.float32)
        source_agreement = np.ones_like(denom, dtype=np.float32)
        needs_variance = (
            str(source_agreement_mode) != "off"
            or str(ood_gain_mode) != "off"
            or feature_rows is not None
        )
        if needs_variance:
            diff = src_residual - pred[:, None, :]
            variance = np.sum(weights * np.sum(diff * diff, axis=2), axis=1) / np.maximum(denom, 1.0e-12)
            pred_energy = np.sum(pred * pred, axis=1)
            variance_ratio = (variance / np.maximum(pred_energy, 1.0e-8)).astype(np.float32)
        if str(source_agreement_mode) != "off" and float(source_agreement_beta) > 0.0:
            source_agreement = np.exp(
                np.clip(-float(source_agreement_beta) * variance_ratio, -30.0, 0.0)
            ).astype(np.float32)
            if str(source_agreement_mode) == "hard":
                ok_rows &= source_agreement >= float(source_agreement_min_confidence)
            elif str(source_agreement_mode) != "soft":
                raise ValueError(f"unsupported source_agreement_mode={source_agreement_mode}")
        if float(confidence_tau) > 0.0:
            confidence = denom / (denom + float(confidence_tau))
        else:
            confidence = np.ones_like(denom, dtype=np.float32)
        if str(source_agreement_mode) == "soft":
            confidence = confidence * source_agreement
        policy_reliability = np.ones_like(confidence, dtype=np.float32)
        if "policy_reliability" in bank:
            policy_map = np.asarray(bank["policy_reliability"], dtype=np.float32)
            policy_reliability = np.clip(policy_map[face_idx, bin_id], 0.0, 1.0).astype(np.float32)
            confidence = confidence * policy_reliability
        policy_gain = np.ones_like(confidence, dtype=np.float32)
        if "policy_gain" in bank:
            gain_map = np.asarray(bank["policy_gain"], dtype=np.float32)
            policy_gain = np.clip(gain_map[face_idx, bin_id], 0.0, 8.0).astype(np.float32)
            confidence = confidence * policy_gain
        ood_confidence = np.ones_like(confidence, dtype=np.float32)
        ood_score = np.zeros_like(confidence, dtype=np.float32)
        policy_tail_risk = np.zeros_like(confidence, dtype=np.float32)
        ood_features: np.ndarray | None = None
        if str(ood_gain_mode) in {"boosted_soft", "learned_linear"} or feature_rows is not None:
            valid_float = valid_src.astype(np.float32)
            valid_count = np.maximum(np.sum(valid_float, axis=1), 1.0)
            max_view_cos = np.max(np.where(valid_src, view_cos, -1.0), axis=1).astype(np.float32)
            view_gap = np.clip(1.0 - max_view_cos, 0.0, 2.0)
            parent_mismatch = np.sqrt(
                np.sum(weights * parent_dist, axis=1) / np.maximum(denom, 1.0e-12)
            ).astype(np.float32)
            effective_count = (denom * denom) / np.maximum(np.sum(weights * weights, axis=1), 1.0e-12)
            effective_count_risk = np.clip(1.0 - effective_count / valid_count, 0.0, 1.0).astype(np.float32)
            if "policy_tail_risk" in bank:
                tail_map = np.asarray(bank["policy_tail_risk"], dtype=np.float32)
                policy_tail_risk = np.clip(tail_map[face_idx, bin_id], 0.0, 1.0).astype(np.float32)
            gain_boost = np.maximum(policy_gain - 1.0, 0.0)
            ood_features = _stack_learned_ood_features(
                gain_boost=gain_boost,
                policy_reliability=policy_reliability,
                policy_tail_risk=policy_tail_risk,
                view_gap=view_gap,
                variance_ratio=variance_ratio,
                parent_mismatch=parent_mismatch,
                effective_count_risk=effective_count_risk,
                max_view_cos=max_view_cos,
            )
            ood_score = (
                float(ood_gain_view_weight) * view_gap
                + float(ood_gain_variance_weight) * np.clip(variance_ratio, 0.0, 4.0)
                + float(ood_gain_parent_weight) * np.clip(parent_mismatch, 0.0, 1.0)
                + float(ood_gain_effective_count_weight) * effective_count_risk
            ).astype(np.float32)
        if str(ood_gain_mode) == "boosted_soft":
            if ood_features is None:
                raise RuntimeError("internal error: boosted_soft has no OOD features")
            risk_multiplier = 0.5 + policy_tail_risk
            gain_boost = np.maximum(policy_gain - 1.0, 0.0)
            ood_confidence = np.exp(
                np.clip(-float(ood_gain_beta) * gain_boost * risk_multiplier * ood_score, -30.0, 0.0)
            ).astype(np.float32)
            confidence = confidence * ood_confidence
        elif str(ood_gain_mode) == "learned_linear":
            if ood_features is None:
                raise RuntimeError("internal error: learned_linear has no OOD features")
            ood_confidence = _apply_learned_ood_head(bank, ood_features)
            confidence = confidence * ood_confidence
        elif str(ood_gain_mode) != "off":
            raise ValueError(f"unsupported ood_gain_mode={ood_gain_mode}")
        pred = pred * confidence[:, None]
        yy, xx = ys[ok_rows], xs[ok_rows]
        delta[:, yy, xx] = pred[ok_rows].T
        active[yy, xx] = True
        if feature_rows is not None:
            if ood_features is None:
                raise RuntimeError("internal error: requested feature rows without OOD features")
            feature_rows.append(
                {
                    "ys": yy.astype(np.int64),
                    "xs": xx.astype(np.int64),
                    "features": ood_features[ok_rows].astype(np.float32),
                }
            )
        active_count += int(np.count_nonzero(ok_rows))
        confidences.append(confidence[ok_rows].astype(np.float32))
        agreement_confidences.append(source_agreement[ok_rows].astype(np.float32))
        policy_reliabilities.append(policy_reliability[ok_rows].astype(np.float32))
        policy_gains.append(policy_gain[ok_rows].astype(np.float32))
        ood_confidences.append(ood_confidence[ok_rows].astype(np.float32))
        ood_scores.append(ood_score[ok_rows].astype(np.float32))
        tail_risks.append(policy_tail_risk[ok_rows].astype(np.float32))
    conf = np.concatenate(confidences) if confidences else np.zeros((0,), dtype=np.float32)
    agreement_conf = (
        np.concatenate(agreement_confidences) if agreement_confidences else np.zeros((0,), dtype=np.float32)
    )
    policy_rel = np.concatenate(policy_reliabilities) if policy_reliabilities else np.zeros((0,), dtype=np.float32)
    policy_gain_arr = np.concatenate(policy_gains) if policy_gains else np.zeros((0,), dtype=np.float32)
    ood_conf = np.concatenate(ood_confidences) if ood_confidences else np.ones((0,), dtype=np.float32)
    ood_score_arr = np.concatenate(ood_scores) if ood_scores else np.zeros((0,), dtype=np.float32)
    tail_risk_arr = np.concatenate(tail_risks) if tail_risks else np.zeros((0,), dtype=np.float32)
    return delta, active, {
        "surface_support_fraction": float(np.mean(support)),
        "active_fraction": float(active_count / max(int(support.size), 1)),
        "active_over_support_fraction": float(active_count / max(int(np.count_nonzero(support)), 1)),
        "mean_confidence": float(np.mean(conf)) if conf.size else 0.0,
        "p10_confidence": float(np.quantile(conf, 0.10)) if conf.size else 0.0,
        "mean_source_agreement_confidence": float(np.mean(agreement_conf)) if agreement_conf.size else 0.0,
        "p10_source_agreement_confidence": float(np.quantile(agreement_conf, 0.10)) if agreement_conf.size else 0.0,
        "mean_policy_reliability": float(np.mean(policy_rel)) if policy_rel.size else 0.0,
        "p10_policy_reliability": float(np.quantile(policy_rel, 0.10)) if policy_rel.size else 0.0,
        "mean_policy_gain": float(np.mean(policy_gain_arr)) if policy_gain_arr.size else 1.0,
        "p90_policy_gain": float(np.quantile(policy_gain_arr, 0.90)) if policy_gain_arr.size else 1.0,
        "mean_ood_confidence": float(np.mean(ood_conf)) if ood_conf.size else 1.0,
        "p10_ood_confidence": float(np.quantile(ood_conf, 0.10)) if ood_conf.size else 1.0,
        "mean_ood_score": float(np.mean(ood_score_arr)) if ood_score_arr.size else 0.0,
        "mean_policy_tail_risk": float(np.mean(tail_risk_arr)) if tail_risk_arr.size else 0.0,
    }


def _summarize_rows(rows: list[dict[str, Any]], *, compute_lpips: bool) -> dict[str, Any]:
    psnr_gain = [float(r["psnr_gain"]) for r in rows]
    ssim_gain = [float(r["ssim_gain"]) for r in rows]
    out = {
        "view_count": int(len(rows)),
        "parent_psnr": _mean([float(r["parent_psnr"]) for r in rows]),
        "candidate_psnr": _mean([float(r["candidate_psnr"]) for r in rows]),
        "psnr_gain": _mean(psnr_gain),
        "psnr_gain_tail": _tail(psnr_gain),
        "parent_ssim": _mean([float(r["parent_ssim"]) for r in rows]),
        "candidate_ssim": _mean([float(r["candidate_ssim"]) for r in rows]),
        "ssim_gain": _mean(ssim_gain),
        "ssim_gain_tail": _tail(ssim_gain),
        "positive_view_fraction": float(np.mean(np.asarray(psnr_gain) > 0.0)) if rows else 0.0,
        "ssim_positive_view_fraction": float(np.mean(np.asarray(ssim_gain) > 0.0)) if rows else 0.0,
        "support_active_fraction": _mean([float(r.get("active_fraction", 0.0)) for r in rows]),
        "support_active_over_support_fraction": _mean([float(r.get("active_over_support_fraction", 0.0)) for r in rows]),
        "mean_confidence": _mean([float(r.get("mean_confidence", 0.0)) for r in rows]),
        "mean_policy_reliability": _mean([float(r.get("mean_policy_reliability", 0.0)) for r in rows]),
        "mean_policy_gain": _mean([float(r.get("mean_policy_gain", 1.0)) for r in rows]),
        "p90_policy_gain": _mean([float(r.get("p90_policy_gain", 1.0)) for r in rows]),
        "mean_ood_confidence": _mean([float(r.get("mean_ood_confidence", 1.0)) for r in rows]),
        "p10_ood_confidence": _mean([float(r.get("p10_ood_confidence", 1.0)) for r in rows]),
        "mean_ood_score": _mean([float(r.get("mean_ood_score", 0.0)) for r in rows]),
        "mean_policy_tail_risk": _mean([float(r.get("mean_policy_tail_risk", 0.0)) for r in rows]),
        "mean_changed_fraction": _mean([float(r.get("changed_fraction", 0.0)) for r in rows]),
    }
    if compute_lpips:
        lpips_gain = [float(r["lpips_gain"]) for r in rows]
        out.update(
            {
                "parent_lpips": _mean([float(r["parent_lpips"]) for r in rows]),
                "candidate_lpips": _mean([float(r["candidate_lpips"]) for r in rows]),
                "lpips_gain": _mean(lpips_gain),
                "lpips_gain_tail": _tail(lpips_gain),
                "lpips_positive_view_fraction": float(np.mean(np.asarray(lpips_gain) > 0.0)) if rows else 0.0,
            }
        )
    return out


def _select_policy_row(summaries: list[dict[str, Any]], *, compute_lpips: bool, allow_noop: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    eligible = [row for row in summaries if bool(allow_noop) or float(row.get("alpha", 0.0)) > 0.0]
    if not eligible:
        eligible = list(summaries)

    def score(row: dict[str, Any]) -> float:
        return (
            float(row.get("psnr_gain", 0.0))
            + 20.0 * float(row.get("ssim_gain", 0.0))
            + 20.0 * float(row.get("lpips_gain", 0.0))
            + 0.25 * float(row.get("psnr_gain_tail", {}).get("cvar", 0.0))
            + 5.0 * float(row.get("ssim_gain_tail", {}).get("cvar", 0.0))
            + 5.0 * float(row.get("lpips_gain_tail", {}).get("cvar", 0.0))
        )

    best = max(eligible, key=score)
    best_all_axis = None
    for row in eligible:
        ok = float(row.get("psnr_gain", 0.0)) > 0.0 and float(row.get("ssim_gain", 0.0)) > 0.0
        ok = ok and (not compute_lpips or float(row.get("lpips_gain", 0.0)) > 0.0)
        if not ok:
            continue
        cand = dict(row)
        cand["balanced_tail_score"] = float(score(row))
        if best_all_axis is None or float(cand["balanced_tail_score"]) > float(best_all_axis["balanced_tail_score"]):
            best_all_axis = cand
    return best, best_all_axis


def _evaluate_policy_val(
    val_paths: list[Path],
    candidate_faces: np.ndarray,
    bank: dict[str, np.ndarray],
    *,
    grid: int,
    min_alpha: float,
    min_source_count: float,
    alpha_grid: list[float],
    chunk_size: int,
    view_beta: float,
    normal_beta: float,
    parent_beta: float,
    count_gamma: float,
    gain_beta: float,
    confidence_tau: float,
    source_agreement_mode: str,
    source_agreement_beta: float,
    source_agreement_min_confidence: float,
    residual_decoder_mode: str,
    local_linear_l2: float,
    local_linear_blend: float,
    local_linear_min_sources: int,
    local_linear_residual_clip: float,
    ood_gain_mode: str,
    ood_gain_beta: float,
    ood_gain_view_weight: float,
    ood_gain_variance_weight: float,
    ood_gain_parent_weight: float,
    ood_gain_effective_count_weight: float,
    compute_lpips: bool,
    ssim_max_side: int,
    lpips_max_side: int,
    output_dir: Path,
    allow_noop_alpha: bool,
) -> dict[str, Any]:
    lpips_model = build_lpips_model() if compute_lpips else None
    rows_by_alpha: dict[float, list[dict[str, Any]]] = {float(alpha): [] for alpha in alpha_grid}
    projection_rows: dict[float, list[dict[str, Any]]] = {float(alpha): [] for alpha in alpha_grid}
    active_projection_rows: dict[float, list[dict[str, Any]]] = {float(alpha): [] for alpha in alpha_grid}
    raw_cache: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]] = []
    for path in tqdm(val_paths, desc="eval deferred source policy-val"):
        with np.load(path, allow_pickle=False) as z:
            parent = np.asarray(z["rgb_render"], dtype=np.float32)[:3]
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)[:3]
            raw_delta = np.asarray(z["teacher_residual_rgb"], dtype=np.float32)[:3]
            delta, active, support_stats = _predict_delta(
                z,
                candidate_faces,
                bank,
                grid=int(grid),
                min_alpha=float(min_alpha),
                min_source_count=float(min_source_count),
                chunk_size=int(chunk_size),
                view_beta=float(view_beta),
                normal_beta=float(normal_beta),
                parent_beta=float(parent_beta),
                count_gamma=float(count_gamma),
                gain_beta=float(gain_beta),
                confidence_tau=float(confidence_tau),
                source_agreement_mode=str(source_agreement_mode),
                source_agreement_beta=float(source_agreement_beta),
                source_agreement_min_confidence=float(source_agreement_min_confidence),
                residual_decoder_mode=str(residual_decoder_mode),
                local_linear_l2=float(local_linear_l2),
                local_linear_blend=float(local_linear_blend),
                local_linear_min_sources=int(local_linear_min_sources),
                local_linear_residual_clip=float(local_linear_residual_clip),
                ood_gain_mode=str(ood_gain_mode),
                ood_gain_beta=float(ood_gain_beta),
                ood_gain_view_weight=float(ood_gain_view_weight),
                ood_gain_variance_weight=float(ood_gain_variance_weight),
                ood_gain_parent_weight=float(ood_gain_parent_weight),
                ood_gain_effective_count_weight=float(ood_gain_effective_count_weight),
            )
            raw_cache.append((path, parent, gt, delta, active, support_stats))
            full_mask = _surface_support_mask(z, candidate_faces, min_alpha=float(min_alpha))
            p_psnr = _psnr(parent, gt)
            p_ssim = image_ssim_chw(parent, gt, int(ssim_max_side))
            p_lpips = image_lpips_chw(parent, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
            teacher = np.clip(parent + raw_delta, 0.0, 1.0)
            t_parent_psnr = _psnr(parent, teacher)
            t_parent_ssim = image_ssim_chw(parent, teacher, int(ssim_max_side))
            t_parent_lpips = image_lpips_chw(parent, teacher, int(lpips_max_side), lpips_model) if compute_lpips else None
            for alpha in alpha_grid:
                pred_delta = float(alpha) * delta
                candidate = np.clip(parent + pred_delta, 0.0, 1.0)
                c_psnr = _psnr(candidate, gt)
                c_ssim = image_ssim_chw(candidate, gt, int(ssim_max_side))
                c_lpips = image_lpips_chw(candidate, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
                row = {
                    "view": path.stem,
                    "alpha": float(alpha),
                    "parent_psnr": float(p_psnr),
                    "candidate_psnr": float(c_psnr),
                    "psnr_gain": float(c_psnr - p_psnr),
                    "parent_ssim": float(p_ssim),
                    "candidate_ssim": float(c_ssim),
                    "ssim_gain": float(c_ssim - p_ssim),
                    **support_stats,
                }
                if compute_lpips:
                    row.update(
                        {
                            "parent_lpips": float(p_lpips),
                            "candidate_lpips": float(c_lpips),
                            "lpips_gain": float(p_lpips - c_lpips),
                        }
                    )
                rows_by_alpha[float(alpha)].append(row)
                t_cand_psnr = _psnr(candidate, teacher)
                t_cand_ssim = image_ssim_chw(candidate, teacher, int(ssim_max_side))
                t_cand_lpips = image_lpips_chw(candidate, teacher, int(lpips_max_side), lpips_model) if compute_lpips else None
                projection_rows[float(alpha)].append(
                    {
                        "view": path.stem,
                        "parent_psnr": float(t_parent_psnr),
                        "candidate_psnr": float(t_cand_psnr),
                        "psnr_gain": float(t_cand_psnr - t_parent_psnr),
                        "parent_ssim": float(t_parent_ssim),
                        "candidate_ssim": float(t_cand_ssim),
                        "ssim_gain": float(t_cand_ssim - t_parent_ssim),
                        "parent_lpips": float(t_parent_lpips) if compute_lpips else None,
                        "candidate_lpips": float(t_cand_lpips) if compute_lpips else None,
                        "lpips_gain": float(t_parent_lpips - t_cand_lpips) if compute_lpips else None,
                        **_residual_stats(pred_delta, raw_delta, full_mask),
                    }
                )
                active_projection_rows[float(alpha)].append(
                    {"view": path.stem, **_residual_stats(pred_delta, raw_delta, active)}
                )
    summaries = []
    projection_by_alpha: dict[str, Any] = {}
    for alpha in alpha_grid:
        rows = rows_by_alpha[float(alpha)]
        summary = {"alpha": float(alpha), **_summarize_rows(rows, compute_lpips=compute_lpips)}
        summaries.append(summary)
        proj_rows = projection_rows[float(alpha)]
        projection_by_alpha[str(float(alpha))] = {
            "image_summary": {
                "parent_psnr": _mean([float(r["parent_psnr"]) for r in proj_rows]),
                "candidate_psnr": _mean([float(r["candidate_psnr"]) for r in proj_rows]),
                "psnr_gain": _mean([float(r["psnr_gain"]) for r in proj_rows]),
                "parent_ssim": _mean([float(r["parent_ssim"]) for r in proj_rows]),
                "candidate_ssim": _mean([float(r["candidate_ssim"]) for r in proj_rows]),
                "ssim_gain": _mean([float(r["ssim_gain"]) for r in proj_rows]),
                "parent_lpips": _mean([float(r["parent_lpips"]) for r in proj_rows if r.get("parent_lpips") is not None]),
                "candidate_lpips": _mean(
                    [float(r["candidate_lpips"]) for r in proj_rows if r.get("candidate_lpips") is not None]
                ),
                "lpips_gain": _mean([float(r["lpips_gain"]) for r in proj_rows if r.get("lpips_gain") is not None]),
            },
            "full_residual_summary": _summarize_residual_rows(proj_rows),
            "active_residual_summary": _summarize_residual_rows(active_projection_rows[float(alpha)]),
        }
    best, best_all_axis = _select_policy_row(summaries, compute_lpips=compute_lpips, allow_noop=allow_noop_alpha)
    render_alpha = float((best_all_axis or best)["alpha"])
    render_dir = output_dir / "policy_val_best"
    render_dir.mkdir(parents=True, exist_ok=True)
    for path, parent, gt, delta, _active, _support_stats in tqdm(raw_cache, desc="write deferred source renders"):
        save_image_chw(render_dir / f"{path.stem}.png", np.clip(parent + render_alpha * delta, 0.0, 1.0))
        save_image_chw(render_dir / f"{path.stem}_parent.png", parent)
        save_image_chw(render_dir / f"{path.stem}_gt.png", gt)
    return {
        "best": best,
        "best_all_axis": best_all_axis,
        "rows": summaries,
        "per_view_by_alpha": {str(float(alpha)): rows for alpha, rows in rows_by_alpha.items()},
        "projection_by_alpha": projection_by_alpha,
        "best_alpha_for_render": float(render_alpha),
        "render_dir": str(render_dir),
    }


def _verify_target_no_gt(target_dir: Path) -> dict[str, Any]:
    if not str(target_dir):
        return {"checked": False, "reason": "empty target directory"}
    paths = evidence_views(target_dir)
    rows = []
    bad = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
            forbidden = sorted(keys & FORBIDDEN_TARGET_KEYS)
            row = {"view": path.stem, "forbidden_keys": forbidden, "key_count": int(len(keys))}
            rows.append(row)
            if forbidden:
                bad.append(row)
    return {
        "checked": True,
        "target_evidence_dir": str(target_dir),
        "view_count": int(len(paths)),
        "pass": len(paths) > 0 and not bad,
        "forbidden_key_set": sorted(FORBIDDEN_TARGET_KEYS),
        "bad_view_count": int(len(bad)),
        "bad_views_preview": bad[:8],
        "rows_preview": rows[:8],
    }


def _target_no_gt_preview(
    target_dir: Path,
    candidate_faces: np.ndarray,
    bank: dict[str, np.ndarray],
    *,
    grid: int,
    min_alpha: float,
    min_source_count: float,
    chunk_size: int,
    view_beta: float,
    normal_beta: float,
    parent_beta: float,
    count_gamma: float,
    gain_beta: float,
    confidence_tau: float,
    source_agreement_mode: str,
    source_agreement_beta: float,
    source_agreement_min_confidence: float,
    residual_decoder_mode: str,
    local_linear_l2: float,
    local_linear_blend: float,
    local_linear_min_sources: int,
    local_linear_residual_clip: float,
    ood_gain_mode: str,
    ood_gain_beta: float,
    ood_gain_view_weight: float,
    ood_gain_variance_weight: float,
    ood_gain_parent_weight: float,
    ood_gain_effective_count_weight: float,
    alpha: float,
    max_views: int,
    output_dir: Path,
) -> dict[str, Any]:
    paths = evidence_views(target_dir)[: max(0, int(max_views))]
    preview_dir = output_dir / "target_no_gt_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in tqdm(paths, desc="write target no-GT preview"):
        with np.load(path, allow_pickle=False) as z:
            if "rgb_gt" in z:
                raise RuntimeError(f"target no-GT preview received GT key in {path}")
            parent = np.asarray(z["rgb_render"], dtype=np.float32)[:3]
            delta, active, support_stats = _predict_delta(
                z,
                candidate_faces,
                bank,
                grid=int(grid),
                min_alpha=float(min_alpha),
                min_source_count=float(min_source_count),
                chunk_size=int(chunk_size),
                view_beta=float(view_beta),
                normal_beta=float(normal_beta),
                parent_beta=float(parent_beta),
                count_gamma=float(count_gamma),
                gain_beta=float(gain_beta),
                confidence_tau=float(confidence_tau),
                source_agreement_mode=str(source_agreement_mode),
                source_agreement_beta=float(source_agreement_beta),
                source_agreement_min_confidence=float(source_agreement_min_confidence),
                residual_decoder_mode=str(residual_decoder_mode),
                local_linear_l2=float(local_linear_l2),
                local_linear_blend=float(local_linear_blend),
                local_linear_min_sources=int(local_linear_min_sources),
                local_linear_residual_clip=float(local_linear_residual_clip),
                ood_gain_mode=str(ood_gain_mode),
                ood_gain_beta=float(ood_gain_beta),
                ood_gain_view_weight=float(ood_gain_view_weight),
                ood_gain_variance_weight=float(ood_gain_variance_weight),
                ood_gain_parent_weight=float(ood_gain_parent_weight),
                ood_gain_effective_count_weight=float(ood_gain_effective_count_weight),
            )
            out = np.clip(parent + float(alpha) * delta, 0.0, 1.0)
            save_image_chw(preview_dir / f"{path.stem}.png", out)
            save_image_chw(preview_dir / f"{path.stem}_parent.png", parent)
            rows.append(
                {
                    "view": path.stem,
                    "changed_fraction": float(np.mean(np.any(np.abs(float(alpha) * delta) > (0.5 / 255.0), axis=0))),
                    "active_fraction": float(np.mean(active)),
                    **support_stats,
                }
            )
    return {
        "view_count": int(len(paths)),
        "alpha": float(alpha),
        "preview_dir": str(preview_dir),
        "mean_changed_fraction": _mean([float(row["changed_fraction"]) for row in rows]),
        "mean_active_fraction": _mean([float(row["active_fraction"]) for row in rows]),
        "rows": rows,
    }


def _target_exact_eval(
    target_no_gt_dir: Path,
    target_eval_dir: Path,
    candidate_faces: np.ndarray,
    bank: dict[str, np.ndarray],
    *,
    grid: int,
    min_alpha: float,
    min_source_count: float,
    chunk_size: int,
    view_beta: float,
    normal_beta: float,
    parent_beta: float,
    count_gamma: float,
    gain_beta: float,
    confidence_tau: float,
    source_agreement_mode: str,
    source_agreement_beta: float,
    source_agreement_min_confidence: float,
    residual_decoder_mode: str,
    local_linear_l2: float,
    local_linear_blend: float,
    local_linear_min_sources: int,
    local_linear_residual_clip: float,
    ood_gain_mode: str,
    ood_gain_beta: float,
    ood_gain_view_weight: float,
    ood_gain_variance_weight: float,
    ood_gain_parent_weight: float,
    ood_gain_effective_count_weight: float,
    alpha: float,
    compute_lpips: bool,
    ssim_max_side: int,
    lpips_max_side: int,
    output_dir: Path,
) -> dict[str, Any]:
    no_gt_paths = {path.stem: path for path in evidence_views(target_no_gt_dir)}
    eval_paths = {path.stem: path for path in evidence_views(target_eval_dir)}
    stems = sorted(set(no_gt_paths) & set(eval_paths))
    lpips_model = build_lpips_model() if compute_lpips else None
    render_dir = output_dir / "target_exact_fixed_policy"
    render_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for stem in tqdm(stems, desc="eval target fixed-policy exact"):
        with np.load(no_gt_paths[stem], allow_pickle=False) as z_apply:
            leaked = sorted(set(z_apply.files) & FORBIDDEN_TARGET_KEYS)
            if leaked:
                raise RuntimeError(f"target apply evidence leaked forbidden keys for {stem}: {leaked}")
            parent = np.asarray(z_apply["rgb_render"], dtype=np.float32)[:3]
            delta, active, support_stats = _predict_delta(
                z_apply,
                candidate_faces,
                bank,
                grid=int(grid),
                min_alpha=float(min_alpha),
                min_source_count=float(min_source_count),
                chunk_size=int(chunk_size),
                view_beta=float(view_beta),
                normal_beta=float(normal_beta),
                parent_beta=float(parent_beta),
                count_gamma=float(count_gamma),
                gain_beta=float(gain_beta),
                confidence_tau=float(confidence_tau),
                source_agreement_mode=str(source_agreement_mode),
                source_agreement_beta=float(source_agreement_beta),
                source_agreement_min_confidence=float(source_agreement_min_confidence),
                residual_decoder_mode=str(residual_decoder_mode),
                local_linear_l2=float(local_linear_l2),
                local_linear_blend=float(local_linear_blend),
                local_linear_min_sources=int(local_linear_min_sources),
                local_linear_residual_clip=float(local_linear_residual_clip),
                ood_gain_mode=str(ood_gain_mode),
                ood_gain_beta=float(ood_gain_beta),
                ood_gain_view_weight=float(ood_gain_view_weight),
                ood_gain_variance_weight=float(ood_gain_variance_weight),
                ood_gain_parent_weight=float(ood_gain_parent_weight),
                ood_gain_effective_count_weight=float(ood_gain_effective_count_weight),
            )
            candidate = np.clip(parent + float(alpha) * delta, 0.0, 1.0)
        with np.load(eval_paths[stem], allow_pickle=False) as z_eval:
            if "rgb_gt" not in z_eval:
                raise RuntimeError(f"target eval evidence has no rgb_gt for {stem}: {eval_paths[stem]}")
            gt = np.asarray(z_eval["rgb_gt"], dtype=np.float32)[:3]
        p_psnr = _psnr(parent, gt)
        c_psnr = _psnr(candidate, gt)
        p_ssim = image_ssim_chw(parent, gt, int(ssim_max_side))
        c_ssim = image_ssim_chw(candidate, gt, int(ssim_max_side))
        row: dict[str, Any] = {
            "view": stem,
            "alpha": float(alpha),
            "parent_psnr": float(p_psnr),
            "candidate_psnr": float(c_psnr),
            "psnr_gain": float(c_psnr - p_psnr),
            "parent_ssim": float(p_ssim),
            "candidate_ssim": float(c_ssim),
            "ssim_gain": float(c_ssim - p_ssim),
            "changed_fraction": float(np.mean(np.any(np.abs(float(alpha) * delta) > (0.5 / 255.0), axis=0))),
            "active_fraction": float(np.mean(active)),
            **support_stats,
        }
        if compute_lpips:
            p_lpips = image_lpips_chw(parent, gt, int(lpips_max_side), lpips_model)
            c_lpips = image_lpips_chw(candidate, gt, int(lpips_max_side), lpips_model)
            row.update(
                {
                    "parent_lpips": float(p_lpips),
                    "candidate_lpips": float(c_lpips),
                    "lpips_gain": float(p_lpips - c_lpips),
                }
            )
        rows.append(row)
        save_image_chw(render_dir / f"{stem}.png", candidate)
        save_image_chw(render_dir / f"{stem}_parent.png", parent)
        save_image_chw(render_dir / f"{stem}_gt.png", gt)
    summary = _summarize_rows(rows, compute_lpips=compute_lpips)
    phasej_comparison = {
        "reference": PHASEJ_FLOWERS,
        "candidate_psnr_minus_phasej": float(summary.get("candidate_psnr", 0.0) - PHASEJ_FLOWERS["psnr"]),
        "candidate_ssim_minus_phasej": float(summary.get("candidate_ssim", 0.0) - PHASEJ_FLOWERS["ssim"]),
        "phasej_lpips_minus_candidate": (
            float(PHASEJ_FLOWERS["lpips"] - summary.get("candidate_lpips", 0.0)) if compute_lpips else None
        ),
        "beats_phasej_all_axis_under_reported_metric_scale": bool(
            summary.get("candidate_psnr", 0.0) > PHASEJ_FLOWERS["psnr"]
            and summary.get("candidate_ssim", 0.0) > PHASEJ_FLOWERS["ssim"]
            and (not compute_lpips or summary.get("candidate_lpips", 1.0e9) < PHASEJ_FLOWERS["lpips"])
        ),
    }
    return {
        "alpha": float(alpha),
        "target_no_gt_dir": str(target_no_gt_dir),
        "target_eval_dir": str(target_eval_dir),
        "view_count": int(len(rows)),
        "missing_no_gt_views": sorted(set(eval_paths) - set(no_gt_paths)),
        "missing_eval_views": sorted(set(no_gt_paths) - set(eval_paths)),
        "summary": summary,
        "phasej_reference_comparison": phasej_comparison,
        "render_dir": str(render_dir),
        "per_view": rows,
        "selection_scope": "alpha selected only from train-policy-val; target/test GT loaded after no-GT apply",
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    best = payload["policy_val"]["best"]
    best_all = payload["policy_val"].get("best_all_axis")
    selected = best_all or best
    alpha = str(float(selected["alpha"]))
    proj = payload["policy_val"]["projection_by_alpha"].get(alpha, {})
    full_proj = proj.get("full_residual_summary", {})
    active_proj = proj.get("active_residual_summary", {})
    score_coverage = payload["candidate_face_summary"].get("selected_score_coverage", 0.0)
    score_coverage_text = "n/a" if score_coverage is None else f"{float(score_coverage):.6f}"
    lines = [
        "# v253 Deferred Source-Feature Residual Renderer",
        "",
        f"- verdict: `{payload['verdict']}`",
        f"- policy-val all-axis pass: `{payload['policy_val_all_axis_pass']}`",
        f"- selected alpha: `{float(selected.get('alpha', 0.0)):.6f}`",
        f"- no-target-GT audit pass: `{payload['target_no_gt_audit'].get('pass')}`",
        f"- target exact fixed-policy pass vs parent: `{payload['target_exact_eval'].get('pass_vs_parent_all_axis')}`",
        f"- Phase-J flowers exact reference: `{PHASEJ_FLOWERS}`",
        "",
        "## Command",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
        "## Method Change",
        "",
        (
            "v253 stores multiple train-fit teacher residual sources per face/UV bin, then uses a deferred renderer "
            "to mix them by target view direction, normal agreement, parent-RGB similarity, support count, and "
            "teacher-gain confidence. This changes the representation carrier rather than tuning an alpha gate."
        ),
        "",
        "## Residual Decoder",
        "",
        f"- mode: `{payload.get('residual_decoder', {}).get('mode', 'weighted_average')}`",
        f"- local-linear ridge: `{payload.get('residual_decoder', {}).get('local_linear_l2', 0.0):.6f}`",
        f"- local-linear blend: `{payload.get('residual_decoder', {}).get('local_linear_blend', 0.0):.6f}`",
        f"- local-linear min sources: `{payload.get('residual_decoder', {}).get('local_linear_min_sources', 0)}`",
        f"- local-linear residual clip: `{payload.get('residual_decoder', {}).get('local_linear_residual_clip', 0.0):.6f}`",
        "",
        "## Policy-Val Metrics",
        "",
        "| row | alpha | PSNR gain | SSIM gain | LPIPS gain | PSNR tail CVaR | SSIM tail CVaR | LPIPS tail CVaR | active/support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            "| best | {alpha:.6f} | {psnr:+.6f} | {ssim:+.6f} | {lpips:+.6f} | {ptail:+.6f} | "
            "{stail:+.6f} | {ltail:+.6f} | {active:.6f} |"
        ).format(
            alpha=float(best.get("alpha", 0.0)),
            psnr=float(best.get("psnr_gain", 0.0)),
            ssim=float(best.get("ssim_gain", 0.0)),
            lpips=float(best.get("lpips_gain", 0.0)),
            ptail=float(best.get("psnr_gain_tail", {}).get("cvar", 0.0)),
            stail=float(best.get("ssim_gain_tail", {}).get("cvar", 0.0)),
            ltail=float(best.get("lpips_gain_tail", {}).get("cvar", 0.0)),
            active=float(best.get("support_active_over_support_fraction", 0.0)),
        ),
        (
            "| best all-axis | {alpha:.6f} | {psnr:+.6f} | {ssim:+.6f} | {lpips:+.6f} | {ptail:+.6f} | "
            "{stail:+.6f} | {ltail:+.6f} | {active:.6f} |"
        ).format(
            alpha=float((best_all or {}).get("alpha", 0.0)),
            psnr=float((best_all or {}).get("psnr_gain", 0.0)),
            ssim=float((best_all or {}).get("ssim_gain", 0.0)),
            lpips=float((best_all or {}).get("lpips_gain", 0.0)),
            ptail=float((best_all or {}).get("psnr_gain_tail", {}).get("cvar", 0.0)),
            stail=float((best_all or {}).get("ssim_gain_tail", {}).get("cvar", 0.0)),
            ltail=float((best_all or {}).get("lpips_gain_tail", {}).get("cvar", 0.0)),
            active=float((best_all or {}).get("support_active_over_support_fraction", 0.0)),
        ),
        "",
        "## Teacher Projection At Selected Alpha",
        "",
        "| scope | cosine | energy retention | changed fraction | residual PSNR |",
        "|---|---:|---:|---:|---:|",
        (
            "| full surface | {cos:.6f} | {ret:.6f} | {chg:.6f} | {rpsnr:.6f} |"
        ).format(
            cos=float(full_proj.get("cosine", 0.0)),
            ret=float(full_proj.get("energy_retention", 0.0)),
            chg=float(full_proj.get("changed_fraction", 0.0)),
            rpsnr=float(full_proj.get("residual_psnr", 0.0)),
        ),
        (
            "| active surface | {cos:.6f} | {ret:.6f} | {chg:.6f} | {rpsnr:.6f} |"
        ).format(
            cos=float(active_proj.get("cosine", 0.0)),
            ret=float(active_proj.get("energy_retention", 0.0)),
            chg=float(active_proj.get("changed_fraction", 0.0)),
            rpsnr=float(active_proj.get("residual_psnr", 0.0)),
        ),
        "",
        "## Source Bank",
        "",
        f"- selected faces: `{payload['candidate_face_summary'].get('selected_faces')}`",
        f"- selected score coverage: `{score_coverage_text}`",
        f"- nonempty face bins: `{payload['fit_source_bank'].get('nonempty_face_bins')}`",
        f"- nonempty source slots: `{payload['fit_source_bank'].get('nonempty_source_slots')}`",
        "",
        "## Policy Reliability",
        "",
        f"- mode: `{payload.get('policy_reliability', {}).get('mode', 'off')}`",
        f"- active bins: `{payload.get('policy_reliability', {}).get('active_bins', 0)}`",
        f"- mean reliability: `{payload.get('policy_reliability', {}).get('mean_reliability', 0.0):.6f}`",
        f"- mean valid reliability: `{payload.get('policy_reliability', {}).get('mean_reliability_valid', 0.0):.6f}`",
        f"- policy gain mode: `{payload.get('policy_reliability', {}).get('policy_gain_mode', 'off')}`",
        f"- mean valid policy gain: `{payload.get('policy_reliability', {}).get('mean_policy_gain_valid', 1.0):.6f}`",
        f"- max policy gain: `{payload.get('policy_reliability', {}).get('max_policy_gain', 1.0):.6f}`",
        f"- mean valid tail risk: `{payload.get('policy_reliability', {}).get('mean_tail_risk_valid', 0.0):.6f}`",
        f"- OOD gain mode: `{payload.get('ood_gain', {}).get('mode', 'off')}`",
        "",
        "## Learned OOD/Gain Head",
        "",
        f"- mode: `{payload.get('learned_ood_head', {}).get('mode', 'off')}`",
        f"- sample count: `{payload.get('learned_ood_head', {}).get('sample_count', 0)}`",
        f"- label mean: `{payload.get('learned_ood_head', {}).get('label_mean', 0.0):.6f}`",
        f"- predicted confidence mean: `{payload.get('learned_ood_head', {}).get('predicted_confidence_mean', 0.0):.6f}`",
        f"- label/prediction correlation: `{payload.get('learned_ood_head', {}).get('label_prediction_correlation', 0.0):.6f}`",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['output_json']}`",
        f"- Markdown: `{path}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- policy-val renders: `{payload['policy_val']['render_dir']}`",
        f"- target no-GT preview: `{payload['target_no_gt_preview'].get('preview_dir', '')}`",
    ]
    target_eval = payload.get("target_exact_eval") or {}
    if target_eval:
        summary = target_eval.get("summary", {})
        comparison = target_eval.get("phasej_reference_comparison", {})
        lines.extend(
            [
                "",
                "## Target Exact Fixed-Policy Evaluation",
                "",
                "| PSNR | SSIM | LPIPS | PSNR gain | SSIM gain | LPIPS gain | changed fraction |",
                "|---:|---:|---:|---:|---:|---:|---:|",
                (
                    "| {psnr:.6f} | {ssim:.6f} | {lpips:.6f} | {pg:+.6f} | {sg:+.6f} | {lg:+.6f} | {chg:.6f} |"
                ).format(
                    psnr=float(summary.get("candidate_psnr", 0.0)),
                    ssim=float(summary.get("candidate_ssim", 0.0)),
                    lpips=float(summary.get("candidate_lpips", 0.0)),
                    pg=float(summary.get("psnr_gain", 0.0)),
                    sg=float(summary.get("ssim_gain", 0.0)),
                    lg=float(summary.get("lpips_gain", 0.0)),
                    chg=float(summary.get("mean_changed_fraction", summary.get("support_active_fraction", 0.0))),
                ),
                "",
                f"- render dir: `{target_eval.get('render_dir', '')}`",
                f"- Phase-J comparison: `{comparison}`",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a view-dependent deferred source residual renderer.")
    parser.add_argument("--fit_evidence_dir", default=DEFAULT_EVIDENCE)
    parser.add_argument("--target_evidence_dir", default=DEFAULT_TARGET_NO_GT)
    parser.add_argument("--target_eval_evidence_dir", default=DEFAULT_TARGET_EVAL)
    parser.add_argument("--bank_checkpoint", default="")
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    parser.add_argument("--min_l1", type=float, default=0.00025)
    parser.add_argument("--min_alpha", type=float, default=0.02)
    parser.add_argument("--max_candidate_faces", type=int, default=8192)
    parser.add_argument("--candidate_target_energy_coverage", type=float, default=0.95)
    parser.add_argument("--max_candidate_face_samples_per_view", type=int, default=120000)
    parser.add_argument("--target_visible_face_quota", type=int, default=0)
    parser.add_argument("--max_target_visible_face_samples_per_view", type=int, default=240000)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--source_top_k", type=int, default=6)
    parser.add_argument("--max_source_samples_per_view", type=int, default=180000)
    parser.add_argument("--score_gain_weight", type=float, default=2.0)
    parser.add_argument("--min_source_count", type=float, default=2.0)
    parser.add_argument("--view_beta", type=float, default=3.0)
    parser.add_argument("--normal_beta", type=float, default=1.0)
    parser.add_argument("--parent_beta", type=float, default=8.0)
    parser.add_argument("--count_gamma", type=float, default=0.25)
    parser.add_argument("--gain_beta", type=float, default=1.0)
    parser.add_argument("--confidence_tau", type=float, default=0.0)
    parser.add_argument("--residual_decoder_mode", choices=["weighted_average", "local_linear"], default="weighted_average")
    parser.add_argument("--local_linear_l2", type=float, default=0.05)
    parser.add_argument("--local_linear_blend", type=float, default=1.0)
    parser.add_argument("--local_linear_min_sources", type=int, default=3)
    parser.add_argument("--local_linear_residual_clip", type=float, default=0.12)
    parser.add_argument("--source_agreement_mode", choices=["off", "soft", "hard"], default="off")
    parser.add_argument("--source_agreement_beta", type=float, default=0.0)
    parser.add_argument("--source_agreement_min_confidence", type=float, default=0.25)
    parser.add_argument("--policy_reliability_mode", choices=["off", "local_l1", "patch_perceptual_v1"], default="off")
    parser.add_argument("--policy_reliability_alpha", type=float, default=0.03125)
    parser.add_argument("--policy_reliability_min_count", type=int, default=8)
    parser.add_argument("--policy_reliability_min_positive_fraction", type=float, default=0.52)
    parser.add_argument("--policy_reliability_min_mean_gain", type=float, default=0.0)
    parser.add_argument("--policy_reliability_gain_scale", type=float, default=0.00025)
    parser.add_argument("--policy_reliability_floor", type=float, default=0.0)
    parser.add_argument("--policy_reliability_patch_radius", type=int, default=3)
    parser.add_argument("--policy_reliability_patch_gain_weight", type=float, default=0.5)
    parser.add_argument("--policy_reliability_gradient_gain_weight", type=float, default=0.25)
    parser.add_argument("--policy_gain_mode", choices=["off", "positive_soft"], default="off")
    parser.add_argument("--policy_gain_max", type=float, default=2.0)
    parser.add_argument("--policy_gain_scale", type=float, default=0.000025)
    parser.add_argument("--ood_gain_mode", choices=["off", "boosted_soft", "learned_linear"], default="off")
    parser.add_argument("--ood_gain_beta", type=float, default=1.0)
    parser.add_argument("--ood_gain_view_weight", type=float, default=1.0)
    parser.add_argument("--ood_gain_variance_weight", type=float, default=1.0)
    parser.add_argument("--ood_gain_parent_weight", type=float, default=1.0)
    parser.add_argument("--ood_gain_effective_count_weight", type=float, default=0.5)
    parser.add_argument("--learned_ood_head_alpha", type=float, default=1.0)
    parser.add_argument("--learned_ood_head_l2", type=float, default=0.01)
    parser.add_argument("--learned_ood_head_gain_scale", type=float, default=0.0002)
    parser.add_argument("--learned_ood_head_floor", type=float, default=0.35)
    parser.add_argument("--learned_ood_head_min_gain", type=float, default=0.0)
    parser.add_argument("--learned_ood_head_max_samples", type=int, default=200000)
    parser.add_argument("--bank_residual_transform_mode", choices=["raw_rgb", "luma_only", "chroma_shrink"], default="raw_rgb")
    parser.add_argument("--bank_residual_chroma_shrink", type=float, default=0.25)
    parser.add_argument("--alpha_grid", default="0,0.03125,0.0625,0.09375,0.125,0.1875,0.25,0.375,0.5,0.75,1")
    parser.add_argument("--allow_noop_alpha", action="store_true")
    parser.add_argument("--eval_chunk_size", type=int, default=196608)
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--policy_val_ssim_max_side", type=int, default=512)
    parser.add_argument("--policy_val_lpips_max_side", type=int, default=256)
    parser.add_argument("--target_preview_views", type=int, default=4)
    parser.add_argument("--target_eval_mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--output_dir", default="/tmp/peilincai_spcarnet_v253_deferred_source_renderer")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet-v253-deferred-source-renderer")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--seed", type=int, default=253)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if bool(args.enable_wandb):
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(args.wandb_project),
                name=str(args.wandb_run_name or output_dir.name),
                config=vars(args),
                dir=str(output_dir),
            )
        except Exception as exc:  # pragma: no cover
            print(f"[wandb] disabled after init failure: {type(exc).__name__}: {exc}", flush=True)
            wandb_run = None

    paths = evidence_views(Path(args.fit_evidence_dir))
    if not paths:
        raise FileNotFoundError(args.fit_evidence_dir)
    fit_paths, val_paths = _policy_split(paths, int(args.policy_val_stride))
    if str(args.bank_checkpoint):
        candidate_faces, bank, bank_summary = _load_source_bank(Path(args.bank_checkpoint))
        face_summary = {
            "selected_faces": int(candidate_faces.size),
            "selected_score_coverage": None,
            "loaded_from_checkpoint": str(args.bank_checkpoint),
        }
    else:
        candidate_faces, face_summary = _rank_candidate_faces(
            fit_paths,
            residual_rgb_key=str(args.residual_rgb_key),
            residual_l1_key=str(args.residual_l1_key),
            min_l1=float(args.min_l1),
            min_alpha=float(args.min_alpha),
            max_faces=int(args.max_candidate_faces),
            max_samples_per_view=int(args.max_candidate_face_samples_per_view),
            target_energy_coverage=float(args.candidate_target_energy_coverage),
            seed=int(args.seed),
        )
        target_visible_summary: dict[str, Any] = {"enabled": False}
        if int(args.target_visible_face_quota) > 0:
            target_visible_faces, target_visible_summary = _rank_target_visible_faces(
                Path(args.target_evidence_dir),
                min_alpha=float(args.min_alpha),
                quota=int(args.target_visible_face_quota),
                max_samples_per_view=int(args.max_target_visible_face_samples_per_view),
                seed=int(args.seed),
            )
            before_faces = int(candidate_faces.size)
            if target_visible_faces.size > 0:
                candidate_faces = np.asarray(
                    sorted(set(candidate_faces.astype(np.int64).tolist()) | set(target_visible_faces.tolist())),
                    dtype=np.int64,
                )
            target_visible_summary["candidate_faces_before_union"] = before_faces
            target_visible_summary["candidate_faces_after_union"] = int(candidate_faces.size)
            target_visible_summary["added_faces"] = int(candidate_faces.size - before_faces)
            face_summary["target_visible_face_expansion"] = target_visible_summary
            face_summary["selected_faces"] = int(candidate_faces.size)
        if candidate_faces.size <= 0:
            raise RuntimeError("no candidate faces selected")
        bank, bank_summary = _fit_source_bank(
            fit_paths,
            candidate_faces,
            residual_rgb_key=str(args.residual_rgb_key),
            residual_l1_key=str(args.residual_l1_key),
            min_l1=float(args.min_l1),
            min_alpha=float(args.min_alpha),
            grid=int(args.grid),
            source_top_k=int(args.source_top_k),
            max_samples_per_view=int(args.max_source_samples_per_view),
            score_gain_weight=float(args.score_gain_weight),
            seed=int(args.seed),
        )
    residual_transform_summary = _transform_bank_residuals(
        bank,
        str(args.bank_residual_transform_mode),
        float(args.bank_residual_chroma_shrink),
    )
    policy_reliability_summary: dict[str, Any] = {"mode": "off"}
    if str(args.policy_reliability_mode) in {"local_l1", "patch_perceptual_v1"}:
        policy_reliability_summary = _calibrate_policy_reliability(
            val_paths,
            candidate_faces,
            bank,
            grid=int(args.grid),
            alpha=float(args.policy_reliability_alpha),
            min_alpha=float(args.min_alpha),
            min_source_count=float(args.min_source_count),
            chunk_size=int(args.eval_chunk_size),
            view_beta=float(args.view_beta),
            normal_beta=float(args.normal_beta),
            parent_beta=float(args.parent_beta),
            count_gamma=float(args.count_gamma),
            gain_beta=float(args.gain_beta),
            confidence_tau=float(args.confidence_tau),
            source_agreement_mode=str(args.source_agreement_mode),
            source_agreement_beta=float(args.source_agreement_beta),
            source_agreement_min_confidence=float(args.source_agreement_min_confidence),
            mode=str(args.policy_reliability_mode),
            min_count=int(args.policy_reliability_min_count),
            min_positive_fraction=float(args.policy_reliability_min_positive_fraction),
            min_mean_gain=float(args.policy_reliability_min_mean_gain),
            gain_scale=float(args.policy_reliability_gain_scale),
            floor=float(args.policy_reliability_floor),
            patch_radius=int(args.policy_reliability_patch_radius),
            patch_gain_weight=float(args.policy_reliability_patch_gain_weight),
            gradient_gain_weight=float(args.policy_reliability_gradient_gain_weight),
            policy_gain_mode=str(args.policy_gain_mode),
            policy_gain_max=float(args.policy_gain_max),
            policy_gain_scale=float(args.policy_gain_scale),
            residual_decoder_mode=str(args.residual_decoder_mode),
            local_linear_l2=float(args.local_linear_l2),
            local_linear_blend=float(args.local_linear_blend),
            local_linear_min_sources=int(args.local_linear_min_sources),
            local_linear_residual_clip=float(args.local_linear_residual_clip),
        )
    learned_ood_head_summary: dict[str, Any] = {"mode": "off"}
    if str(args.ood_gain_mode) == "learned_linear":
        learned_ood_head_summary = _fit_learned_ood_head(
            val_paths,
            candidate_faces,
            bank,
            grid=int(args.grid),
            alpha=float(args.learned_ood_head_alpha),
            min_alpha=float(args.min_alpha),
            min_source_count=float(args.min_source_count),
            chunk_size=int(args.eval_chunk_size),
            view_beta=float(args.view_beta),
            normal_beta=float(args.normal_beta),
            parent_beta=float(args.parent_beta),
            count_gamma=float(args.count_gamma),
            gain_beta=float(args.gain_beta),
            confidence_tau=float(args.confidence_tau),
            source_agreement_mode=str(args.source_agreement_mode),
            source_agreement_beta=float(args.source_agreement_beta),
            source_agreement_min_confidence=float(args.source_agreement_min_confidence),
            l2=float(args.learned_ood_head_l2),
            gain_scale=float(args.learned_ood_head_gain_scale),
            floor=float(args.learned_ood_head_floor),
            min_gain=float(args.learned_ood_head_min_gain),
            max_samples=int(args.learned_ood_head_max_samples),
            seed=int(args.seed),
            residual_decoder_mode=str(args.residual_decoder_mode),
            local_linear_l2=float(args.local_linear_l2),
            local_linear_blend=float(args.local_linear_blend),
            local_linear_min_sources=int(args.local_linear_min_sources),
            local_linear_residual_clip=float(args.local_linear_residual_clip),
        )
    alpha_grid = sorted({float(item) for item in str(args.alpha_grid).split(",") if item.strip()})
    if 0.0 not in alpha_grid:
        alpha_grid = [0.0, *alpha_grid]
    policy_val = _evaluate_policy_val(
        val_paths,
        candidate_faces,
        bank,
        grid=int(args.grid),
        min_alpha=float(args.min_alpha),
        min_source_count=float(args.min_source_count),
        alpha_grid=alpha_grid,
        chunk_size=int(args.eval_chunk_size),
        view_beta=float(args.view_beta),
        normal_beta=float(args.normal_beta),
        parent_beta=float(args.parent_beta),
        count_gamma=float(args.count_gamma),
        gain_beta=float(args.gain_beta),
        confidence_tau=float(args.confidence_tau),
        source_agreement_mode=str(args.source_agreement_mode),
        source_agreement_beta=float(args.source_agreement_beta),
        source_agreement_min_confidence=float(args.source_agreement_min_confidence),
        residual_decoder_mode=str(args.residual_decoder_mode),
        local_linear_l2=float(args.local_linear_l2),
        local_linear_blend=float(args.local_linear_blend),
        local_linear_min_sources=int(args.local_linear_min_sources),
        local_linear_residual_clip=float(args.local_linear_residual_clip),
        ood_gain_mode=str(args.ood_gain_mode),
        ood_gain_beta=float(args.ood_gain_beta),
        ood_gain_view_weight=float(args.ood_gain_view_weight),
        ood_gain_variance_weight=float(args.ood_gain_variance_weight),
        ood_gain_parent_weight=float(args.ood_gain_parent_weight),
        ood_gain_effective_count_weight=float(args.ood_gain_effective_count_weight),
        compute_lpips=bool(args.compute_lpips),
        ssim_max_side=int(args.policy_val_ssim_max_side),
        lpips_max_side=int(args.policy_val_lpips_max_side),
        output_dir=output_dir,
        allow_noop_alpha=bool(args.allow_noop_alpha),
    )
    all_axis = policy_val.get("best_all_axis") is not None
    target_no_gt_audit = _verify_target_no_gt(Path(args.target_evidence_dir)) if str(args.target_evidence_dir) else {"checked": False}
    selected_alpha = float((policy_val.get("best_all_axis") or policy_val["best"])["alpha"])
    target_preview: dict[str, Any] = {}
    if target_no_gt_audit.get("pass") and int(args.target_preview_views) > 0:
        target_preview = _target_no_gt_preview(
            Path(args.target_evidence_dir),
            candidate_faces,
            bank,
            grid=int(args.grid),
            min_alpha=float(args.min_alpha),
            min_source_count=float(args.min_source_count),
            chunk_size=int(args.eval_chunk_size),
            view_beta=float(args.view_beta),
            normal_beta=float(args.normal_beta),
            parent_beta=float(args.parent_beta),
            count_gamma=float(args.count_gamma),
            gain_beta=float(args.gain_beta),
            confidence_tau=float(args.confidence_tau),
            source_agreement_mode=str(args.source_agreement_mode),
            source_agreement_beta=float(args.source_agreement_beta),
            source_agreement_min_confidence=float(args.source_agreement_min_confidence),
            residual_decoder_mode=str(args.residual_decoder_mode),
            local_linear_l2=float(args.local_linear_l2),
            local_linear_blend=float(args.local_linear_blend),
            local_linear_min_sources=int(args.local_linear_min_sources),
            local_linear_residual_clip=float(args.local_linear_residual_clip),
            ood_gain_mode=str(args.ood_gain_mode),
            ood_gain_beta=float(args.ood_gain_beta),
            ood_gain_view_weight=float(args.ood_gain_view_weight),
            ood_gain_variance_weight=float(args.ood_gain_variance_weight),
            ood_gain_parent_weight=float(args.ood_gain_parent_weight),
            ood_gain_effective_count_weight=float(args.ood_gain_effective_count_weight),
            alpha=float(selected_alpha),
            max_views=int(args.target_preview_views),
            output_dir=output_dir,
        )
    target_eval: dict[str, Any] = {}
    run_target_eval = str(args.target_eval_mode) == "always" or (
        str(args.target_eval_mode) == "auto" and all_axis and bool(target_no_gt_audit.get("pass", False))
    )
    if run_target_eval and str(args.target_eval_evidence_dir):
        target_eval = _target_exact_eval(
            Path(args.target_evidence_dir),
            Path(args.target_eval_evidence_dir),
            candidate_faces,
            bank,
            grid=int(args.grid),
            min_alpha=float(args.min_alpha),
            min_source_count=float(args.min_source_count),
            chunk_size=int(args.eval_chunk_size),
            view_beta=float(args.view_beta),
            normal_beta=float(args.normal_beta),
            parent_beta=float(args.parent_beta),
            count_gamma=float(args.count_gamma),
            gain_beta=float(args.gain_beta),
            confidence_tau=float(args.confidence_tau),
            source_agreement_mode=str(args.source_agreement_mode),
            source_agreement_beta=float(args.source_agreement_beta),
            source_agreement_min_confidence=float(args.source_agreement_min_confidence),
            residual_decoder_mode=str(args.residual_decoder_mode),
            local_linear_l2=float(args.local_linear_l2),
            local_linear_blend=float(args.local_linear_blend),
            local_linear_min_sources=int(args.local_linear_min_sources),
            local_linear_residual_clip=float(args.local_linear_residual_clip),
            ood_gain_mode=str(args.ood_gain_mode),
            ood_gain_beta=float(args.ood_gain_beta),
            ood_gain_view_weight=float(args.ood_gain_view_weight),
            ood_gain_variance_weight=float(args.ood_gain_variance_weight),
            ood_gain_parent_weight=float(args.ood_gain_parent_weight),
            ood_gain_effective_count_weight=float(args.ood_gain_effective_count_weight),
            alpha=float(selected_alpha),
            compute_lpips=bool(args.compute_lpips),
            ssim_max_side=int(args.policy_val_ssim_max_side),
            lpips_max_side=int(args.policy_val_lpips_max_side),
            output_dir=output_dir,
        )
        target_summary = target_eval.get("summary", {})
        target_eval["pass_vs_parent_all_axis"] = bool(
            target_summary.get("psnr_gain", 0.0) > 0.0
            and target_summary.get("ssim_gain", 0.0) > 0.0
            and (not bool(args.compute_lpips) or target_summary.get("lpips_gain", 0.0) > 0.0)
        )
    checkpoint_path = output_dir / "v253_deferred_source_renderer_bank.npz"
    checkpoint_payload = {
        "schema": np.asarray("spcarnet_v253_deferred_source_renderer_bank_v1"),
        "candidate_faces": candidate_faces.astype(np.int64),
        "score": bank["score"].astype(np.float16),
        "counts": bank["counts"].astype(np.float16),
        "residual": bank["residual"].astype(np.float16),
        "parent_rgb": bank["parent_rgb"].astype(np.float16),
        "normal": bank["normal"].astype(np.float16),
        "camera_dir": bank["camera_dir"].astype(np.float16),
        "gain_l1": bank["gain_l1"].astype(np.float16),
        "alpha": bank["alpha"].astype(np.float16),
        "args_json": np.asarray(json.dumps(vars(args), sort_keys=True)),
    }
    if "policy_reliability" in bank:
        checkpoint_payload["policy_reliability"] = bank["policy_reliability"].astype(np.float16)
    if "policy_gain" in bank:
        checkpoint_payload["policy_gain"] = bank["policy_gain"].astype(np.float16)
    if "policy_tail_risk" in bank:
        checkpoint_payload["policy_tail_risk"] = bank["policy_tail_risk"].astype(np.float16)
    for key in (
        "learned_ood_head_coef",
        "learned_ood_head_bias",
        "learned_ood_head_mean",
        "learned_ood_head_scale",
        "learned_ood_head_floor",
    ):
        if key in bank:
            checkpoint_payload[key] = bank[key].astype(np.float32)
    if "learned_ood_head_coef" in bank:
        checkpoint_payload["learned_ood_head_feature_names_json"] = np.asarray(
            json.dumps(LEARNED_OOD_FEATURE_NAMES, sort_keys=True)
        )
    np.savez_compressed(checkpoint_path, **checkpoint_payload)
    verdict = (
        "PASS_POLICY_VAL_PROMOTE_TO_FLOWERS_EXACT"
        if all_axis
        else "FAIL_POLICY_VAL_DO_NOT_PROMOTE_CURRENT_DEFERRED_SOURCE_CARRIER"
    )
    payload: dict[str, Any] = {
        "schema": "spcarnet_v253_deferred_source_renderer_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": os.getcwd(),
        "fit_evidence_dir": str(args.fit_evidence_dir),
        "target_evidence_dir": str(args.target_evidence_dir),
        "target_eval_evidence_dir": str(args.target_eval_evidence_dir),
        "fit_views": int(len(fit_paths)),
        "policy_val_views": int(len(val_paths)),
        "candidate_face_summary": face_summary,
        "fit_source_bank": bank_summary,
        "bank_residual_transform": residual_transform_summary,
        "residual_decoder": {
            "mode": str(args.residual_decoder_mode),
            "local_linear_l2": float(args.local_linear_l2),
            "local_linear_blend": float(args.local_linear_blend),
            "local_linear_min_sources": int(args.local_linear_min_sources),
            "local_linear_residual_clip": float(args.local_linear_residual_clip),
        },
        "policy_reliability": policy_reliability_summary,
        "learned_ood_head": learned_ood_head_summary,
        "ood_gain": {
            "mode": str(args.ood_gain_mode),
            "beta": float(args.ood_gain_beta),
            "view_weight": float(args.ood_gain_view_weight),
            "variance_weight": float(args.ood_gain_variance_weight),
            "parent_weight": float(args.ood_gain_parent_weight),
            "effective_count_weight": float(args.ood_gain_effective_count_weight),
            "learned_head_enabled": bool(str(args.ood_gain_mode) == "learned_linear"),
        },
        "policy_val_all_axis_pass": bool(all_axis),
        "policy_val": policy_val,
        "target_no_gt_audit": target_no_gt_audit,
        "target_no_gt_preview": target_preview,
        "target_exact_eval": target_eval,
        "phasej_flowers_exact_reference": PHASEJ_FLOWERS,
        "flowers_exact_run_allowed_next": bool(all_axis and target_no_gt_audit.get("pass", False)),
        "uses_train_fit_teacher": True,
        "uses_policy_val_gt": True,
        "uses_target_or_test_gt": False,
        "uses_target_or_test_gt_after_apply_for_eval": bool(target_eval),
        "verdict": verdict,
        "checkpoint": str(checkpoint_path),
        "output_json": str(output_dir / "v253_deferred_source_renderer_audit.json"),
        "output_md": str(output_dir / "v253_deferred_source_renderer_audit.md"),
    }
    _write_json(output_dir / "v253_deferred_source_renderer_audit.json", payload)
    _write_md(output_dir / "v253_deferred_source_renderer_audit.md", payload)
    if wandb_run is not None:
        best = policy_val["best"]
        best_all = policy_val.get("best_all_axis") or {}
        wandb_run.log(
            {
                "policy_val/all_axis_pass": float(all_axis),
                "policy_val/best_psnr_gain": float(best.get("psnr_gain", 0.0)),
                "policy_val/best_ssim_gain": float(best.get("ssim_gain", 0.0)),
                "policy_val/best_lpips_gain": float(best.get("lpips_gain", 0.0)),
                "policy_val/best_alpha": float(best.get("alpha", 0.0)),
                "policy_val/best_all_axis_alpha": float(best_all.get("alpha", 0.0)),
                "residual_decoder/is_local_linear": float(str(args.residual_decoder_mode) == "local_linear"),
                "residual_decoder/local_linear_l2": float(args.local_linear_l2),
                "residual_decoder/local_linear_blend": float(args.local_linear_blend),
                "residual_decoder/local_linear_min_sources": float(args.local_linear_min_sources),
                "residual_decoder/local_linear_residual_clip": float(args.local_linear_residual_clip),
                "source_bank/nonempty_source_slots": float(bank_summary.get("nonempty_source_slots", 0)),
                "source_bank/residual_energy_ratio_after_transform": float(
                    residual_transform_summary.get("energy_ratio_after_before", 1.0)
                ),
                "policy_reliability/mean": float(policy_reliability_summary.get("mean_reliability", 0.0)),
                "policy_reliability/active_bins": float(policy_reliability_summary.get("active_bins", 0.0)),
                "policy_reliability/mean_policy_gain_valid": float(
                    policy_reliability_summary.get("mean_policy_gain_valid", 1.0)
                ),
                "policy_reliability/max_policy_gain": float(policy_reliability_summary.get("max_policy_gain", 1.0)),
                "policy_reliability/mean_tail_risk_valid": float(
                    policy_reliability_summary.get("mean_tail_risk_valid", 0.0)
                ),
                "learned_ood_head/sample_count": float(learned_ood_head_summary.get("sample_count", 0.0)),
                "learned_ood_head/label_mean": float(learned_ood_head_summary.get("label_mean", 0.0)),
                "learned_ood_head/predicted_confidence_mean": float(
                    learned_ood_head_summary.get("predicted_confidence_mean", 0.0)
                ),
                "learned_ood_head/label_prediction_correlation": float(
                    learned_ood_head_summary.get("label_prediction_correlation", 0.0)
                ),
                "target_no_gt/pass": float(bool(target_no_gt_audit.get("pass", False))),
                "target_eval/pass_vs_parent_all_axis": float(bool(target_eval.get("pass_vs_parent_all_axis", False))),
                "target_eval/mean_ood_confidence": float(
                    target_eval.get("summary", {}).get("mean_ood_confidence", 1.0)
                ),
                "target_eval/p10_ood_confidence": float(
                    target_eval.get("summary", {}).get("p10_ood_confidence", 1.0)
                ),
                "target_eval/psnr_gain": float(target_eval.get("summary", {}).get("psnr_gain", 0.0)),
                "target_eval/ssim_gain": float(target_eval.get("summary", {}).get("ssim_gain", 0.0)),
                "target_eval/lpips_gain": float(target_eval.get("summary", {}).get("lpips_gain", 0.0)),
            }
        )
        wandb_run.finish()
    print(
        json.dumps(
            {
                "output_json": payload["output_json"],
                "output_md": payload["output_md"],
                "policy_val_all_axis_pass": bool(all_axis),
                "verdict": verdict,
                "best": policy_val["best"],
                "best_all_axis": policy_val.get("best_all_axis"),
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
