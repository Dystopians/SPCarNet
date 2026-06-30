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
        source_agreement = np.ones_like(denom, dtype=np.float32)
        if str(source_agreement_mode) != "off" and float(source_agreement_beta) > 0.0:
            diff = src_residual - pred[:, None, :]
            variance = np.sum(weights * np.sum(diff * diff, axis=2), axis=1) / np.maximum(denom, 1.0e-12)
            pred_energy = np.sum(pred * pred, axis=1)
            source_agreement = np.exp(
                np.clip(-float(source_agreement_beta) * variance / np.maximum(pred_energy, 1.0e-8), -30.0, 0.0)
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
        pred = pred * confidence[:, None]
        yy, xx = ys[ok_rows], xs[ok_rows]
        delta[:, yy, xx] = pred[ok_rows].T
        active[yy, xx] = True
        active_count += int(np.count_nonzero(ok_rows))
        confidences.append(confidence[ok_rows].astype(np.float32))
        agreement_confidences.append(source_agreement[ok_rows].astype(np.float32))
    conf = np.concatenate(confidences) if confidences else np.zeros((0,), dtype=np.float32)
    agreement_conf = (
        np.concatenate(agreement_confidences) if agreement_confidences else np.zeros((0,), dtype=np.float32)
    )
    return delta, active, {
        "surface_support_fraction": float(np.mean(support)),
        "active_fraction": float(active_count / max(int(support.size), 1)),
        "active_over_support_fraction": float(active_count / max(int(np.count_nonzero(support)), 1)),
        "mean_confidence": float(np.mean(conf)) if conf.size else 0.0,
        "p10_confidence": float(np.quantile(conf, 0.10)) if conf.size else 0.0,
        "mean_source_agreement_confidence": float(np.mean(agreement_conf)) if agreement_conf.size else 0.0,
        "p10_source_agreement_confidence": float(np.quantile(agreement_conf, 0.10)) if agreement_conf.size else 0.0,
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
    parser.add_argument("--source_agreement_mode", choices=["off", "soft", "hard"], default="off")
    parser.add_argument("--source_agreement_beta", type=float, default=0.0)
    parser.add_argument("--source_agreement_min_confidence", type=float, default=0.25)
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
    np.savez_compressed(
        checkpoint_path,
        schema=np.asarray("spcarnet_v253_deferred_source_renderer_bank_v1"),
        candidate_faces=candidate_faces.astype(np.int64),
        score=bank["score"].astype(np.float16),
        counts=bank["counts"].astype(np.float16),
        residual=bank["residual"].astype(np.float16),
        parent_rgb=bank["parent_rgb"].astype(np.float16),
        normal=bank["normal"].astype(np.float16),
        camera_dir=bank["camera_dir"].astype(np.float16),
        gain_l1=bank["gain_l1"].astype(np.float16),
        alpha=bank["alpha"].astype(np.float16),
        args_json=np.asarray(json.dumps(vars(args), sort_keys=True)),
    )
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
                "source_bank/nonempty_source_slots": float(bank_summary.get("nonempty_source_slots", 0)),
                "source_bank/residual_energy_ratio_after_transform": float(
                    residual_transform_summary.get("energy_ratio_after_before", 1.0)
                ),
                "target_no_gt/pass": float(bool(target_no_gt_audit.get("pass", False))),
                "target_eval/pass_vs_parent_all_axis": float(bool(target_eval.get("pass_vs_parent_all_axis", False))),
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
