#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
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
    transform_residual_samples_for_fit,
)
from scripts.car_model.train_perceptual_surface_residual_decoder import (  # noqa: E402
    DEFAULT_EVIDENCE,
    _face_indices,
    _policy_split,
    _rank_candidate_faces,
    _valid_mask,
)
from scripts.car_model.train_surface_uv_residual_texture import (  # noqa: E402
    _bin_ids,
    _mean,
    _psnr,
    _quantiles,
    _residual_stats,
    _summarize_residual_rows,
    _write_json,
)


def _basis_dim(basis_mode: str) -> int:
    mode = str(basis_mode)
    if mode == "dir_v1":
        return 6
    if mode == "dir_uv_v1":
        return 8
    if mode == "dir_uv_parent_v1":
        return 10
    raise ValueError(f"unknown basis_mode={basis_mode}")


def _basis_rows(z: np.lib.npyio.NpzFile, ys: np.ndarray, xs: np.ndarray, basis_mode: str) -> np.ndarray:
    mode = str(basis_mode)
    bary = np.asarray(z["barycentric"], dtype=np.float32)
    normal = np.asarray(z["normal"], dtype=np.float32)
    parent = np.asarray(z["rgb_render"], dtype=np.float32)
    camera = np.asarray(z["camera_center"], dtype=np.float32).reshape(3)
    camera = camera / max(float(np.linalg.norm(camera)), 1.0e-8)

    u = np.clip(bary[1, ys, xs], 0.0, 1.0).reshape(-1, 1)
    v = np.clip(bary[2, ys, xs], 0.0, 1.0).reshape(-1, 1)
    n = np.stack([normal[0, ys, xs], normal[1, ys, xs], normal[2, ys, xs]], axis=1)
    n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1.0e-8)
    cam = np.repeat(camera.reshape(1, 3), int(ys.size), axis=0).astype(np.float32)
    ndot = np.sum(n * cam, axis=1, keepdims=True).astype(np.float32)
    rows = [
        np.ones((int(ys.size), 1), dtype=np.float32),
        ndot,
        ndot * ndot,
        cam[:, 0:1],
        cam[:, 1:2],
        cam[:, 2:3],
    ]
    if mode in {"dir_uv_v1", "dir_uv_parent_v1"}:
        rows[1:1] = [u.astype(np.float32), v.astype(np.float32)]
    if mode == "dir_uv_parent_v1":
        rgb = np.stack([parent[0, ys, xs], parent[1, ys, xs], parent[2, ys, xs]], axis=1)
        luma = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).reshape(-1, 1)
        sat = (np.max(rgb, axis=1) - np.min(rgb, axis=1)).reshape(-1, 1)
        rows.extend([np.clip(luma, 0.0, 1.0).astype(np.float32), np.clip(sat, 0.0, 1.0).astype(np.float32)])
    out = np.concatenate(rows, axis=1).astype(np.float32)
    expected = _basis_dim(mode)
    if int(out.shape[1]) != expected:
        raise RuntimeError(f"basis dim mismatch: got {out.shape[1]}, expected {expected}")
    return out


def _luma_gradient_weight_map(chw: np.ndarray, *, strength: float, percentile: float, max_weight: float) -> tuple[np.ndarray | None, dict[str, float]]:
    if float(strength) <= 0.0:
        return None, {
            "enabled": False,
            "strength": float(strength),
            "percentile": float(percentile),
            "max_weight": float(max_weight),
        }
    arr = np.asarray(chw, dtype=np.float32)[:3]
    luma = 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
    gx = np.zeros_like(luma, dtype=np.float32)
    gy = np.zeros_like(luma, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (luma[:, 2:] - luma[:, :-2])
    gx[:, 0] = luma[:, 1] - luma[:, 0] if luma.shape[1] > 1 else 0.0
    gx[:, -1] = luma[:, -1] - luma[:, -2] if luma.shape[1] > 1 else 0.0
    gy[1:-1, :] = 0.5 * (luma[2:, :] - luma[:-2, :])
    gy[0, :] = luma[1, :] - luma[0, :] if luma.shape[0] > 1 else 0.0
    gy[-1, :] = luma[-1, :] - luma[-2, :] if luma.shape[0] > 1 else 0.0
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    positive = mag[mag > 0.0]
    scale = float(np.percentile(positive.astype(np.float64), float(np.clip(percentile, 1.0, 99.9)))) if positive.size else 1.0
    scale = max(scale, 1.0e-8)
    weight = 1.0 + float(strength) * np.clip(mag / scale, 0.0, float(max_weight))
    weight = np.nan_to_num(weight, nan=1.0, posinf=1.0 + float(strength) * float(max_weight), neginf=1.0).astype(np.float32)
    return weight, {
        "enabled": True,
        "strength": float(strength),
        "percentile": float(percentile),
        "max_weight": float(max_weight),
        "scale": float(scale),
        "mean_weight": float(np.mean(weight)),
        "p90_weight": float(np.percentile(weight.astype(np.float64), 90.0)),
        "max_observed_weight": float(np.max(weight)),
    }


def _fit_lowrank_texture(
    fit_paths: list[Path],
    candidate_faces: np.ndarray,
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    grid: int,
    basis_mode: str,
    ridge_count: float,
    solve_chunk_slots: int,
    fit_gradient_weight: float,
    fit_gradient_weight_percentile: float,
    fit_gradient_weight_max: float,
    teacher_residual_target_mode: str,
    teacher_residual_target_luma_mix: float,
    teacher_residual_target_edge_boost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    bins = int(grid) * int(grid)
    basis_dim = _basis_dim(str(basis_mode))
    slot_count = int(candidate_faces.size) * bins
    ata = np.zeros((slot_count, basis_dim, basis_dim), dtype=np.float32)
    atb = np.zeros((slot_count, basis_dim, 3), dtype=np.float32)
    feature_sum = np.zeros((slot_count, basis_dim), dtype=np.float32)
    feature_sumsq = np.zeros((slot_count, basis_dim), dtype=np.float32)
    counts = np.zeros(slot_count, dtype=np.float32)
    weight_sums = np.zeros(slot_count, dtype=np.float32)
    active_pixels = 0
    used_pixels = 0
    gradient_weight_stats: list[dict[str, float]] = []
    residual_target_stats: list[dict[str, Any]] = []
    for path in tqdm(fit_paths, desc="fit low-rank UV residual texture"):
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
            faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
            face_idx, ok = _face_indices(faces, candidate_faces)
            if not np.any(ok):
                continue
            ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
            bin_id = _bin_ids(z, ys, xs, int(grid))
            slots = face_idx.astype(np.int64) * bins + bin_id.astype(np.int64)
            phi = _basis_rows(z, ys, xs, str(basis_mode)).astype(np.float32)
            residual = np.asarray(z[residual_rgb_key], dtype=np.float32)[:3, ys, xs].T.astype(np.float32)
            local_mask = np.zeros(mask.shape, dtype=bool)
            local_mask[ys, xs] = True
            residual, residual_target_summary = transform_residual_samples_for_fit(
                z,
                local_mask,
                residual,
                str(teacher_residual_target_mode),
                float(teacher_residual_target_luma_mix),
                float(teacher_residual_target_edge_boost),
            )
            residual_target_stats.append(residual_target_summary)
            grad_weight_map, grad_stats = _luma_gradient_weight_map(
                np.asarray(z[residual_rgb_key], dtype=np.float32),
                strength=float(fit_gradient_weight),
                percentile=float(fit_gradient_weight_percentile),
                max_weight=float(fit_gradient_weight_max),
            )
            gradient_weight_stats.append(grad_stats)
            if grad_weight_map is None:
                weights = np.ones((int(ys.size),), dtype=np.float32)
            else:
                weights = grad_weight_map[ys, xs].astype(np.float32)
            np.add.at(counts, slots, 1.0)
            np.add.at(weight_sums, slots, weights)
            np.add.at(feature_sum, slots, phi)
            np.add.at(feature_sumsq, slots, phi * phi)
            weighted_phi = phi * weights[:, None]
            np.add.at(ata, slots, weighted_phi[:, :, None] * phi[:, None, :])
            np.add.at(atb, slots, weighted_phi[:, :, None] * residual[:, None, :])
            used_pixels += int(ys.size)

    nonempty = np.flatnonzero(counts > 0.0)
    coeff = np.zeros((slot_count, basis_dim, 3), dtype=np.float32)
    eye = np.eye(basis_dim, dtype=np.float64)
    failures = 0
    for start in tqdm(range(0, int(nonempty.size), int(solve_chunk_slots)), desc="solve low-rank slots"):
        idx = nonempty[start : start + int(solve_chunk_slots)]
        a = ata[idx].astype(np.float64)
        b = atb[idx].astype(np.float64)
        a += eye.reshape(1, basis_dim, basis_dim) * float(ridge_count)
        try:
            solved = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            failures += int(idx.size)
            solved = np.matmul(np.linalg.pinv(a), b)
        coeff[idx] = solved.astype(np.float32)

    coeff = coeff.reshape(int(candidate_faces.size), bins, basis_dim, 3)
    counts_2d = counts.reshape(int(candidate_faces.size), bins)
    denom = np.maximum(counts[:, None], 1.0)
    feature_mean = feature_sum / denom
    feature_var = np.maximum(feature_sumsq / denom - feature_mean * feature_mean, 0.0)
    feature_std = np.sqrt(feature_var).astype(np.float32)
    feature_mean = feature_mean.reshape(int(candidate_faces.size), bins, basis_dim)
    feature_std = feature_std.reshape(int(candidate_faces.size), bins, basis_dim)
    return coeff, counts_2d.astype(np.float32), feature_mean.astype(np.float32), feature_std.astype(np.float32), {
        "fit_active_pixels": int(active_pixels),
        "fit_used_pixels": int(used_pixels),
        "basis_mode": str(basis_mode),
        "basis_dim": int(basis_dim),
        "nonempty_bins": int(np.count_nonzero(counts > 0.0)),
        "nonempty_bin_fraction": float(np.mean(counts > 0.0)),
        "mean_nonempty_count": float(np.mean(counts[counts > 0.0])) if np.any(counts > 0.0) else 0.0,
        "mean_nonempty_weight_sum": float(np.mean(weight_sums[counts > 0.0])) if np.any(counts > 0.0) else 0.0,
        "mean_sample_weight": float(np.sum(weight_sums) / max(float(np.sum(counts)), 1.0)),
        "gradient_weight": {
            "enabled": bool(float(fit_gradient_weight) > 0.0),
            "strength": float(fit_gradient_weight),
            "percentile": float(fit_gradient_weight_percentile),
            "max_weight": float(fit_gradient_weight_max),
            "mean_weight": _mean([float(s.get("mean_weight", 1.0)) for s in gradient_weight_stats if s.get("enabled")]),
            "p90_weight": _mean([float(s.get("p90_weight", 1.0)) for s in gradient_weight_stats if s.get("enabled")]),
            "max_observed_weight": max([float(s.get("max_observed_weight", 1.0)) for s in gradient_weight_stats if s.get("enabled")] or [1.0]),
        },
        "teacher_residual_target": {
            "mode": str(teacher_residual_target_mode),
            "luma_mix": float(teacher_residual_target_luma_mix),
            "edge_boost": float(teacher_residual_target_edge_boost),
            "mean_luma_mix": _mean([float(s.get("mean_luma_mix", 0.0)) for s in residual_target_stats if s.get("mode") != "raw_rgb"]),
            "energy_ratio_after_before": _mean(
                [float(s.get("energy_ratio_after_before", 1.0)) for s in residual_target_stats if s.get("mode") != "raw_rgb"]
            ),
        },
        "ridge_count": float(ridge_count),
        "solve_failures": int(failures),
    }


def _support_confidence(
    phi: np.ndarray,
    face_idx: np.ndarray,
    bin_id: np.ndarray,
    counts: np.ndarray,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
    *,
    min_bin_count: float,
    enable_support_confidence: bool,
    support_full_count: float,
    support_count_power: float,
    support_ood_free_z: float,
    support_ood_max_z: float,
    support_std_floor: float,
    support_min_confidence: float,
) -> tuple[np.ndarray, dict[str, float]]:
    n = int(phi.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float32), {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count_mean": 0.0,
            "ood_z_mean": 0.0,
        }
    if not bool(enable_support_confidence) or feature_mean is None or feature_std is None:
        return np.ones((n,), dtype=np.float32), {
            "mean": 1.0,
            "min": 1.0,
            "max": 1.0,
            "count_mean": float(np.mean(counts[face_idx, bin_id])) if n else 0.0,
            "ood_z_mean": 0.0,
        }
    local_counts = counts[face_idx, bin_id].astype(np.float32)
    full = max(float(support_full_count), float(min_bin_count), 1.0)
    count_conf = np.clip(local_counts / full, 0.0, 1.0)
    count_conf = np.power(count_conf, max(float(support_count_power), 0.0)).astype(np.float32)

    mean = feature_mean[face_idx, bin_id].astype(np.float32)
    std = feature_std[face_idx, bin_id].astype(np.float32)
    valid_dim = std > float(support_std_floor)
    z_abs = np.zeros_like(phi, dtype=np.float32)
    z_abs[valid_dim] = np.abs((phi[valid_dim] - mean[valid_dim]) / np.maximum(std[valid_dim], float(support_std_floor)))
    denom = np.maximum(np.sum(valid_dim, axis=1), 1).astype(np.float32)
    z_rms = np.sqrt(np.sum(z_abs * z_abs, axis=1) / denom).astype(np.float32)
    free_z = max(float(support_ood_free_z), 0.0)
    max_z = max(float(support_ood_max_z), free_z + 1.0e-6)
    ood_conf = np.clip((max_z - z_rms) / max(max_z - free_z, 1.0e-6), 0.0, 1.0).astype(np.float32)
    conf = np.maximum(float(support_min_confidence), count_conf * ood_conf).astype(np.float32)
    return conf, {
        "mean": float(np.mean(conf)),
        "min": float(np.min(conf)),
        "max": float(np.max(conf)),
        "count_mean": float(np.mean(local_counts)),
        "ood_z_mean": float(np.mean(z_rms)),
    }


def _predict_delta(
    z: np.lib.npyio.NpzFile,
    candidate_faces: np.ndarray,
    coeff: np.ndarray,
    counts: np.ndarray,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
    slot_reliability: np.ndarray | None = None,
    *,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    min_bin_count: float,
    grid: int,
    basis_mode: str,
    max_abs_delta: float,
    enable_support_confidence: bool = False,
    support_full_count: float = 16.0,
    support_count_power: float = 0.5,
    support_ood_free_z: float = 1.5,
    support_ood_max_z: float = 4.0,
    support_std_floor: float = 0.02,
    support_min_confidence: float = 0.0,
    enable_slot_reliability_confidence: bool = False,
    slot_reliability_power: float = 1.0,
    return_confidence: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    parent = np.asarray(z["rgb_render"], dtype=np.float32)
    delta = np.zeros_like(parent, dtype=np.float32)
    confidence_map = np.zeros(parent.shape[1:], dtype=np.float32)
    active = _valid_mask(
        z,
        candidate_faces,
        residual_l1_key=str(residual_l1_key),
        min_l1=float(min_l1),
        min_alpha=float(min_alpha),
    )
    ys, xs = np.nonzero(active)
    if ys.size == 0:
        if return_confidence:
            return delta, active, confidence_map, {"mean": 0.0, "min": 0.0, "max": 0.0, "count_mean": 0.0, "ood_z_mean": 0.0}
        return delta, active
    faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
    face_idx, ok = _face_indices(faces, candidate_faces)
    ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
    if ys.size == 0:
        if return_confidence:
            return delta, active, confidence_map, {"mean": 0.0, "min": 0.0, "max": 0.0, "count_mean": 0.0, "ood_z_mean": 0.0}
        return delta, active
    bin_id = _bin_ids(z, ys, xs, int(grid))
    good = counts[face_idx, bin_id] >= float(min_bin_count)
    ys, xs, face_idx, bin_id = ys[good], xs[good], face_idx[good], bin_id[good]
    if ys.size == 0:
        if return_confidence:
            return delta, active, confidence_map, {"mean": 0.0, "min": 0.0, "max": 0.0, "count_mean": 0.0, "ood_z_mean": 0.0}
        return delta, active
    phi = _basis_rows(z, ys, xs, str(basis_mode)).astype(np.float32)
    local = coeff[face_idx, bin_id]
    pred = np.einsum("nk,nkc->nc", phi, local, optimize=True).astype(np.float32)
    conf, conf_stats = _support_confidence(
        phi,
        face_idx,
        bin_id,
        counts,
        feature_mean,
        feature_std,
        min_bin_count=float(min_bin_count),
        enable_support_confidence=bool(enable_support_confidence),
        support_full_count=float(support_full_count),
        support_count_power=float(support_count_power),
        support_ood_free_z=float(support_ood_free_z),
        support_ood_max_z=float(support_ood_max_z),
        support_std_floor=float(support_std_floor),
        support_min_confidence=float(support_min_confidence),
    )
    if bool(enable_slot_reliability_confidence) and slot_reliability is not None:
        rel = np.clip(slot_reliability[face_idx, bin_id].astype(np.float32), 0.0, 1.0)
        conf *= np.power(rel, max(float(slot_reliability_power), 0.0)).astype(np.float32)
        conf_stats["slot_reliability_mean"] = float(np.mean(rel)) if rel.size else 0.0
    pred *= conf[:, None]
    if float(max_abs_delta) > 0.0:
        pred = np.clip(pred, -float(max_abs_delta), float(max_abs_delta))
    delta[:, ys, xs] = pred.T
    confidence_map[ys, xs] = conf
    if return_confidence:
        return delta, active, confidence_map, conf_stats
    return delta, active


def _calibrate_slot_reliability(
    fit_paths: list[Path],
    candidate_faces: np.ndarray,
    coeff: np.ndarray,
    counts: np.ndarray,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
    *,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    min_bin_count: float,
    grid: int,
    basis_mode: str,
    max_abs_delta: float,
    calibration_alpha: float,
    max_samples_per_view: int,
    tau_quantile: float,
    full_count: float,
    min_positive_fraction: float,
    enable_support_confidence: bool,
    support_full_count: float,
    support_count_power: float,
    support_ood_free_z: float,
    support_ood_max_z: float,
    support_std_floor: float,
    support_min_confidence: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    bins = int(grid) * int(grid)
    slot_count = int(candidate_faces.size) * bins
    gain_sum = np.zeros(slot_count, dtype=np.float64)
    sample_count = np.zeros(slot_count, dtype=np.float32)
    positive_count = np.zeros(slot_count, dtype=np.float32)
    used_pixels = 0
    active_pixels = 0
    for path in tqdm(fit_paths, desc="calibrate slot reliability"):
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
                take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
                ys, xs = ys[take], xs[take]
            faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
            face_idx, ok = _face_indices(faces, candidate_faces)
            if not np.any(ok):
                continue
            ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
            bin_id = _bin_ids(z, ys, xs, int(grid))
            good = counts[face_idx, bin_id] >= float(min_bin_count)
            ys, xs, face_idx, bin_id = ys[good], xs[good], face_idx[good], bin_id[good]
            if ys.size == 0:
                continue
            phi = _basis_rows(z, ys, xs, str(basis_mode)).astype(np.float32)
            local = coeff[face_idx, bin_id]
            pred = np.einsum("nk,nkc->nc", phi, local, optimize=True).astype(np.float32)
            conf, _ = _support_confidence(
                phi,
                face_idx,
                bin_id,
                counts,
                feature_mean,
                feature_std,
                min_bin_count=float(min_bin_count),
                enable_support_confidence=bool(enable_support_confidence),
                support_full_count=float(support_full_count),
                support_count_power=float(support_count_power),
                support_ood_free_z=float(support_ood_free_z),
                support_ood_max_z=float(support_ood_max_z),
                support_std_floor=float(support_std_floor),
                support_min_confidence=float(support_min_confidence),
            )
            pred *= conf[:, None]
            if float(max_abs_delta) > 0.0:
                pred = np.clip(pred, -float(max_abs_delta), float(max_abs_delta))
            parent = np.asarray(z["rgb_render"], dtype=np.float32)[:3, ys, xs].T.astype(np.float32)
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)[:3, ys, xs].T.astype(np.float32)
            adapted = np.clip(parent + float(calibration_alpha) * pred, 0.0, 1.0)
            parent_err = np.sum(np.square(parent - gt), axis=1)
            cand_err = np.sum(np.square(adapted - gt), axis=1)
            gain = (parent_err - cand_err).astype(np.float64)
            slots = face_idx.astype(np.int64) * bins + bin_id.astype(np.int64)
            np.add.at(gain_sum, slots, gain)
            np.add.at(sample_count, slots, 1.0)
            np.add.at(positive_count, slots, (gain > 0.0).astype(np.float32))
            used_pixels += int(ys.size)

    valid = sample_count > 0.0
    mean_gain = np.zeros(slot_count, dtype=np.float32)
    mean_gain[valid] = (gain_sum[valid] / np.maximum(sample_count[valid], 1.0)).astype(np.float32)
    positive_mean = mean_gain[mean_gain > 0.0]
    tau = float(np.quantile(positive_mean.astype(np.float64), float(np.clip(tau_quantile, 0.0, 1.0)))) if positive_mean.size else 1.0
    tau = max(tau, 1.0e-10)
    gain_conf = np.clip(mean_gain / tau, 0.0, 1.0)
    count_conf = np.clip(sample_count / max(float(full_count), 1.0), 0.0, 1.0)
    pos_frac = np.divide(positive_count, np.maximum(sample_count, 1.0), out=np.zeros_like(positive_count), where=sample_count > 0)
    min_pos = float(np.clip(min_positive_fraction, 0.0, 0.99))
    pos_conf = np.clip((pos_frac - min_pos) / max(1.0 - min_pos, 1.0e-6), 0.0, 1.0)
    reliability = (gain_conf * count_conf * pos_conf).astype(np.float32)
    rel_2d = reliability.reshape(int(candidate_faces.size), bins)
    usable = reliability[valid]
    return rel_2d, {
        "enabled": True,
        "fit_view_count": int(len(fit_paths)),
        "active_pixels": int(active_pixels),
        "used_pixels": int(used_pixels),
        "valid_slots": int(np.count_nonzero(valid)),
        "positive_mean_gain_slots": int(positive_mean.size),
        "tau": float(tau),
        "mean_reliability": float(np.mean(usable)) if usable.size else 0.0,
        "nonzero_reliability_fraction": float(np.mean(reliability > 0.0)),
        "positive_fraction_mean": float(np.mean(pos_frac[valid])) if np.any(valid) else 0.0,
        "calibration_alpha": float(calibration_alpha),
        "max_samples_per_view": int(max_samples_per_view),
        "tau_quantile": float(tau_quantile),
        "full_count": float(full_count),
        "min_positive_fraction": float(min_positive_fraction),
    }


def _evaluate(
    val_paths: list[Path],
    candidate_faces: np.ndarray,
    coeff: np.ndarray,
    counts: np.ndarray,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
    slot_reliability: np.ndarray | None,
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    min_bin_count: float,
    grid: int,
    basis_mode: str,
    max_abs_delta: float,
    alpha_grid: list[float],
    compute_lpips: bool,
    ssim_max_side: int,
    lpips_max_side: int,
    min_best_positive_view_fraction: float,
    min_best_ssim_positive_view_fraction: float,
    min_best_lpips_positive_view_fraction: float,
    enable_support_confidence: bool,
    support_full_count: float,
    support_count_power: float,
    support_ood_free_z: float,
    support_ood_max_z: float,
    support_std_floor: float,
    support_min_confidence: float,
    enable_slot_reliability_confidence: bool,
    slot_reliability_power: float,
    output_dir: Path,
) -> dict[str, Any]:
    lpips_model = build_lpips_model() if compute_lpips else None
    rows_by_alpha: dict[float, list[dict[str, Any]]] = {float(a): [] for a in alpha_grid}
    projection_rows: dict[float, list[dict[str, Any]]] = {float(a): [] for a in alpha_grid}
    active_projection_rows: dict[float, list[dict[str, Any]]] = {float(a): [] for a in alpha_grid}
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(val_paths, desc="eval low-rank UV residual texture"):
        with np.load(path, allow_pickle=False) as z:
            parent = np.asarray(z["rgb_render"], dtype=np.float32)
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)
            raw_delta = np.asarray(z[residual_rgb_key], dtype=np.float32)[:3]
            teacher = np.clip(parent + raw_delta, 0.0, 1.0)
            delta, active_mask = _predict_delta(
                z,
                candidate_faces,
                coeff,
                counts,
                feature_mean,
                feature_std,
                slot_reliability,
                residual_l1_key=str(residual_l1_key),
                min_l1=float(min_l1),
                min_alpha=float(min_alpha),
                min_bin_count=float(min_bin_count),
                grid=int(grid),
                basis_mode=str(basis_mode),
                max_abs_delta=float(max_abs_delta),
                enable_support_confidence=bool(enable_support_confidence),
                support_full_count=float(support_full_count),
                support_count_power=float(support_count_power),
                support_ood_free_z=float(support_ood_free_z),
                support_ood_max_z=float(support_ood_max_z),
                support_std_floor=float(support_std_floor),
                support_min_confidence=float(support_min_confidence),
                enable_slot_reliability_confidence=bool(enable_slot_reliability_confidence),
                slot_reliability_power=float(slot_reliability_power),
            )
            full_mask = np.asarray(z["face_id"], dtype=np.int64) >= 0
            if "barycentric_valid" in z:
                full_mask &= np.asarray(z["barycentric_valid"]).astype(bool)
            if "alpha" in z:
                full_mask &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
            p_psnr = _psnr(parent, gt)
            p_ssim = image_ssim_chw(parent, gt, int(ssim_max_side))
            p_lp = image_lpips_chw(parent, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
            t_parent_psnr = _psnr(parent, teacher)
            t_parent_ssim = image_ssim_chw(parent, teacher, int(ssim_max_side))
            t_parent_lp = image_lpips_chw(parent, teacher, int(lpips_max_side), lpips_model) if compute_lpips else None
            for alpha in alpha_grid:
                pred = float(alpha) * delta
                adapted = np.clip(parent + pred, 0.0, 1.0)
                c_psnr = _psnr(adapted, gt)
                c_ssim = image_ssim_chw(adapted, gt, int(ssim_max_side))
                c_lp = image_lpips_chw(adapted, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
                rows_by_alpha[float(alpha)].append(
                    {
                        "view": path.stem,
                        "parent_psnr": float(p_psnr),
                        "candidate_psnr": float(c_psnr),
                        "psnr_gain": float(c_psnr - p_psnr),
                        "parent_ssim": float(p_ssim),
                        "candidate_ssim": float(c_ssim),
                        "ssim_gain": float(c_ssim - p_ssim),
                        "parent_lpips": float(p_lp) if compute_lpips else None,
                        "candidate_lpips": float(c_lp) if compute_lpips else None,
                        "lpips_gain": float(p_lp - c_lp) if compute_lpips else None,
                    }
                )
                t_cand_psnr = _psnr(adapted, teacher)
                t_cand_ssim = image_ssim_chw(adapted, teacher, int(ssim_max_side))
                t_cand_lp = image_lpips_chw(adapted, teacher, int(lpips_max_side), lpips_model) if compute_lpips else None
                projection_rows[float(alpha)].append(
                    {
                        "view": path.stem,
                        "parent_psnr": float(t_parent_psnr),
                        "candidate_psnr": float(t_cand_psnr),
                        "psnr_gain": float(t_cand_psnr - t_parent_psnr),
                        "parent_ssim": float(t_parent_ssim),
                        "candidate_ssim": float(t_cand_ssim),
                        "ssim_gain": float(t_cand_ssim - t_parent_ssim),
                        "parent_lpips": float(t_parent_lp) if compute_lpips else None,
                        "candidate_lpips": float(t_cand_lp) if compute_lpips else None,
                        "lpips_gain": float(t_parent_lp - t_cand_lp) if compute_lpips else None,
                        **_residual_stats(pred, raw_delta, full_mask),
                    }
                )
                active_projection_rows[float(alpha)].append({"view": path.stem, **_residual_stats(pred, raw_delta, active_mask)})
    summaries: list[dict[str, Any]] = []
    projection_summaries: dict[str, Any] = {}
    for alpha, rows in rows_by_alpha.items():
        psnr_gain = [float(r["psnr_gain"]) for r in rows]
        ssim_gain = [float(r["ssim_gain"]) for r in rows]
        lpips_gain = [float(r["lpips_gain"]) for r in rows if r.get("lpips_gain") is not None]
        row = {
            "alpha": float(alpha),
            "parent_psnr": _mean([float(r["parent_psnr"]) for r in rows]),
            "candidate_psnr": _mean([float(r["candidate_psnr"]) for r in rows]),
            "psnr_gain": _mean(psnr_gain),
            "parent_ssim": _mean([float(r["parent_ssim"]) for r in rows]),
            "candidate_ssim": _mean([float(r["candidate_ssim"]) for r in rows]),
            "ssim_gain": _mean(ssim_gain),
            "positive_view_fraction": float(np.mean(np.asarray(psnr_gain) > 0.0)),
            "ssim_positive_view_fraction": float(np.mean(np.asarray(ssim_gain) > 0.0)),
            "psnr_gain_quantiles": _quantiles(psnr_gain),
            "ssim_gain_quantiles": _quantiles(ssim_gain),
        }
        if compute_lpips:
            row.update(
                {
                    "parent_lpips": _mean([float(r["parent_lpips"]) for r in rows]),
                    "candidate_lpips": _mean([float(r["candidate_lpips"]) for r in rows]),
                    "lpips_gain": _mean(lpips_gain),
                    "lpips_positive_view_fraction": float(np.mean(np.asarray(lpips_gain) > 0.0)),
                    "lpips_gain_quantiles": _quantiles(lpips_gain),
                }
            )
        summaries.append(row)
        proj_rows = projection_rows[alpha]
        proj_img = {
            "parent_psnr": _mean([float(r["parent_psnr"]) for r in proj_rows]),
            "candidate_psnr": _mean([float(r["candidate_psnr"]) for r in proj_rows]),
            "psnr_gain": _mean([float(r["psnr_gain"]) for r in proj_rows]),
            "parent_ssim": _mean([float(r["parent_ssim"]) for r in proj_rows]),
            "candidate_ssim": _mean([float(r["candidate_ssim"]) for r in proj_rows]),
            "ssim_gain": _mean([float(r["ssim_gain"]) for r in proj_rows]),
            "parent_lpips": _mean([float(r["parent_lpips"]) for r in proj_rows if r.get("parent_lpips") is not None]),
            "candidate_lpips": _mean([float(r["candidate_lpips"]) for r in proj_rows if r.get("candidate_lpips") is not None]),
            "lpips_gain": _mean([float(r["lpips_gain"]) for r in proj_rows if r.get("lpips_gain") is not None]),
        }
        projection_summaries[str(alpha)] = {
            "image_summary": proj_img,
            "full_residual_summary": _summarize_residual_rows(proj_rows),
            "active_residual_summary": _summarize_residual_rows(active_projection_rows[alpha]),
        }
    best = max(
        summaries,
        key=lambda r: float(r.get("psnr_gain", 0.0))
        + 20.0 * float(r.get("ssim_gain", 0.0))
        + 20.0 * float(r.get("lpips_gain", 0.0)),
    )
    best_all_axis = None
    for row in summaries:
        if (
            float(row.get("psnr_gain", 0.0)) > 0.0
            and float(row.get("ssim_gain", 0.0)) > 0.0
            and (not compute_lpips or float(row.get("lpips_gain", 0.0)) > 0.0)
            and float(row.get("positive_view_fraction", 0.0)) >= float(min_best_positive_view_fraction)
            and float(row.get("ssim_positive_view_fraction", 0.0)) >= float(min_best_ssim_positive_view_fraction)
            and (
                not compute_lpips
                or float(row.get("lpips_positive_view_fraction", 0.0)) >= float(min_best_lpips_positive_view_fraction)
            )
        ):
            cand = dict(row)
            cand["balanced_score"] = (
                float(row.get("psnr_gain", 0.0))
                + 20.0 * float(row.get("ssim_gain", 0.0))
                + 20.0 * float(row.get("lpips_gain", 0.0))
            )
            if best_all_axis is None or float(cand["balanced_score"]) > float(best_all_axis["balanced_score"]):
                best_all_axis = cand
    best_alpha = float((best_all_axis or best)["alpha"])
    render_dir = output_dir / "policy_val_best"
    render_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(val_paths, desc="write low-rank policy-val renders"):
        with np.load(path, allow_pickle=False) as z:
            parent = np.asarray(z["rgb_render"], dtype=np.float32)
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)
            delta, _ = _predict_delta(
                z,
                candidate_faces,
                coeff,
                counts,
                feature_mean,
                feature_std,
                slot_reliability,
                residual_l1_key=str(residual_l1_key),
                min_l1=float(min_l1),
                min_alpha=float(min_alpha),
                min_bin_count=float(min_bin_count),
                grid=int(grid),
                basis_mode=str(basis_mode),
                max_abs_delta=float(max_abs_delta),
                enable_support_confidence=bool(enable_support_confidence),
                support_full_count=float(support_full_count),
                support_count_power=float(support_count_power),
                support_ood_free_z=float(support_ood_free_z),
                support_ood_max_z=float(support_ood_max_z),
                support_std_floor=float(support_std_floor),
                support_min_confidence=float(support_min_confidence),
                enable_slot_reliability_confidence=bool(enable_slot_reliability_confidence),
                slot_reliability_power=float(slot_reliability_power),
            )
            save_image_chw(render_dir / f"{path.stem}.png", np.clip(parent + best_alpha * delta, 0.0, 1.0))
            save_image_chw(render_dir / f"{path.stem}_gt.png", gt)
    return {
        "best": best,
        "best_all_axis": best_all_axis,
        "rows": summaries,
        "projection_by_alpha": projection_summaries,
        "best_alpha_for_render": best_alpha,
        "render_dir": str(render_dir),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    best = payload["policy_val"]["best"]
    best_all = payload["policy_val"].get("best_all_axis")
    alpha = str(float((best_all or best)["alpha"]))
    proj = payload["policy_val"]["projection_by_alpha"][alpha]
    lines = [
        "# v212 Low-Rank Surface Residual Texture Audit",
        "",
        f"- policy-val all-axis pass: `{payload['policy_val_all_axis_pass']}`",
        f"- selected faces: `{payload['candidate_face_summary']['selected_faces']}`",
        f"- selected score coverage: `{payload['candidate_face_summary'].get('selected_score_coverage', 0.0):.6f}`",
        f"- grid: `{payload['fit']['grid']}`",
        f"- basis mode: `{payload['fit']['basis_mode']}`",
        f"- basis dim: `{payload['fit']['basis_dim']}`",
        f"- gradient-aware fit weight: `{payload['fit'].get('gradient_weight', {})}`",
        f"- teacher residual target: `{payload['fit'].get('teacher_residual_target', {})}`",
        f"- best all-axis: `{best_all}`",
        "",
        "## Projection At Selected Alpha",
        "",
        "| scope | PSNR gain vs teacher | SSIM gain vs teacher | LPIPS gain vs teacher | cosine | energy retention | changed fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            "| full valid | {psnr:+.6f} | {ssim:+.6f} | {lpips:+.6f} | {cos:.6f} | {ret:.6f} | {chg:.6f} |"
        ).format(
            psnr=float(proj["image_summary"]["psnr_gain"]),
            ssim=float(proj["image_summary"]["ssim_gain"]),
            lpips=float(proj["image_summary"].get("lpips_gain", 0.0)),
            cos=float(proj["full_residual_summary"]["cosine"]),
            ret=float(proj["full_residual_summary"]["energy_retention"]),
            chg=float(proj["full_residual_summary"]["changed_fraction"]),
        ),
        (
            "| selected active | {psnr:+.6f} | {ssim:+.6f} | {lpips:+.6f} | {cos:.6f} | {ret:.6f} | {chg:.6f} |"
        ).format(
            psnr=float(proj["image_summary"]["psnr_gain"]),
            ssim=float(proj["image_summary"]["ssim_gain"]),
            lpips=float(proj["image_summary"].get("lpips_gain", 0.0)),
            cos=float(proj["active_residual_summary"]["cosine"]),
            ret=float(proj["active_residual_summary"]["energy_retention"]),
            chg=float(proj["active_residual_summary"]["changed_fraction"]),
        ),
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['output_json']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- renders: `{payload['policy_val']['render_dir']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit a low-rank view-conditioned per-face UV teacher-residual texture.")
    parser.add_argument("--fit_evidence_dir", default=DEFAULT_EVIDENCE)
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    parser.add_argument("--min_l1", type=float, default=0.0005)
    parser.add_argument("--min_alpha", type=float, default=0.03)
    parser.add_argument("--max_candidate_faces", type=int, default=131072)
    parser.add_argument("--candidate_target_energy_coverage", type=float, default=0.90)
    parser.add_argument("--max_candidate_face_samples_per_view", type=int, default=65536)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--basis_mode", default="dir_uv_v1", choices=["dir_v1", "dir_uv_v1", "dir_uv_parent_v1"])
    parser.add_argument("--ridge_count", type=float, default=8.0)
    parser.add_argument("--min_bin_count", type=float, default=3.0)
    parser.add_argument("--solve_chunk_slots", type=int, default=65536)
    parser.add_argument("--max_abs_delta", type=float, default=0.25)
    parser.add_argument("--fit_gradient_weight", type=float, default=0.0)
    parser.add_argument("--fit_gradient_weight_percentile", type=float, default=95.0)
    parser.add_argument("--fit_gradient_weight_max", type=float, default=4.0)
    parser.add_argument("--teacher_residual_target_mode", default="raw_rgb", choices=["raw_rgb", "luma_only", "edge_luma_mix"])
    parser.add_argument("--teacher_residual_target_luma_mix", type=float, default=0.75)
    parser.add_argument("--teacher_residual_target_edge_boost", type=float, default=0.25)
    parser.add_argument("--alpha_grid", default="0,0.03125,0.0625,0.125,0.25,0.5,0.75,1")
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--policy_val_ssim_max_side", type=int, default=512)
    parser.add_argument("--policy_val_lpips_max_side", type=int, default=256)
    parser.add_argument("--min_best_positive_view_fraction", type=float, default=1.0)
    parser.add_argument("--min_best_ssim_positive_view_fraction", type=float, default=1.0)
    parser.add_argument("--min_best_lpips_positive_view_fraction", type=float, default=1.0)
    parser.add_argument("--enable_support_confidence", action="store_true")
    parser.add_argument("--support_full_count", type=float, default=16.0)
    parser.add_argument("--support_count_power", type=float, default=0.5)
    parser.add_argument("--support_ood_free_z", type=float, default=1.5)
    parser.add_argument("--support_ood_max_z", type=float, default=4.0)
    parser.add_argument("--support_std_floor", type=float, default=0.02)
    parser.add_argument("--support_min_confidence", type=float, default=0.0)
    parser.add_argument("--enable_slot_reliability_confidence", action="store_true")
    parser.add_argument("--slot_reliability_alpha", type=float, default=1.0)
    parser.add_argument("--slot_reliability_max_samples_per_view", type=int, default=131072)
    parser.add_argument("--slot_reliability_tau_quantile", type=float, default=0.75)
    parser.add_argument("--slot_reliability_full_count", type=float, default=8.0)
    parser.add_argument("--slot_reliability_min_positive_fraction", type=float, default=0.5)
    parser.add_argument("--slot_reliability_power", type=float, default=1.0)
    parser.add_argument("--output_dir", default="/tmp/peilincai_spcarnet_v212_lowrank_residual_texture")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet-v212-lowrank-residual-texture")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--seed", type=int, default=212)
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
    coeff, counts, feature_mean, feature_std, fit_summary = _fit_lowrank_texture(
        fit_paths,
        candidate_faces,
        residual_rgb_key=str(args.residual_rgb_key),
        residual_l1_key=str(args.residual_l1_key),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        grid=int(args.grid),
        basis_mode=str(args.basis_mode),
        ridge_count=float(args.ridge_count),
        solve_chunk_slots=int(args.solve_chunk_slots),
        fit_gradient_weight=float(args.fit_gradient_weight),
        fit_gradient_weight_percentile=float(args.fit_gradient_weight_percentile),
        fit_gradient_weight_max=float(args.fit_gradient_weight_max),
        teacher_residual_target_mode=str(args.teacher_residual_target_mode),
        teacher_residual_target_luma_mix=float(args.teacher_residual_target_luma_mix),
        teacher_residual_target_edge_boost=float(args.teacher_residual_target_edge_boost),
    )
    slot_reliability = None
    reliability_summary: dict[str, Any] = {"enabled": False}
    if bool(args.enable_slot_reliability_confidence):
        slot_reliability, reliability_summary = _calibrate_slot_reliability(
            fit_paths,
            candidate_faces,
            coeff,
            counts,
            feature_mean,
            feature_std,
            residual_l1_key=str(args.residual_l1_key),
            min_l1=float(args.min_l1),
            min_alpha=float(args.min_alpha),
            min_bin_count=float(args.min_bin_count),
            grid=int(args.grid),
            basis_mode=str(args.basis_mode),
            max_abs_delta=float(args.max_abs_delta),
            calibration_alpha=float(args.slot_reliability_alpha),
            max_samples_per_view=int(args.slot_reliability_max_samples_per_view),
            tau_quantile=float(args.slot_reliability_tau_quantile),
            full_count=float(args.slot_reliability_full_count),
            min_positive_fraction=float(args.slot_reliability_min_positive_fraction),
            enable_support_confidence=bool(args.enable_support_confidence),
            support_full_count=float(args.support_full_count),
            support_count_power=float(args.support_count_power),
            support_ood_free_z=float(args.support_ood_free_z),
            support_ood_max_z=float(args.support_ood_max_z),
            support_std_floor=float(args.support_std_floor),
            support_min_confidence=float(args.support_min_confidence),
            seed=int(args.seed) + 1009,
        )
    alpha_grid = sorted({float(x) for x in str(args.alpha_grid).split(",") if x.strip()})
    policy_val = _evaluate(
        val_paths,
        candidate_faces,
        coeff,
        counts,
        feature_mean,
        feature_std,
        slot_reliability,
        residual_rgb_key=str(args.residual_rgb_key),
        residual_l1_key=str(args.residual_l1_key),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        min_bin_count=float(args.min_bin_count),
        grid=int(args.grid),
        basis_mode=str(args.basis_mode),
        max_abs_delta=float(args.max_abs_delta),
        alpha_grid=alpha_grid,
        compute_lpips=bool(args.compute_lpips),
        ssim_max_side=int(args.policy_val_ssim_max_side),
        lpips_max_side=int(args.policy_val_lpips_max_side),
        min_best_positive_view_fraction=float(args.min_best_positive_view_fraction),
        min_best_ssim_positive_view_fraction=float(args.min_best_ssim_positive_view_fraction),
        min_best_lpips_positive_view_fraction=float(args.min_best_lpips_positive_view_fraction),
        enable_support_confidence=bool(args.enable_support_confidence),
        support_full_count=float(args.support_full_count),
        support_count_power=float(args.support_count_power),
        support_ood_free_z=float(args.support_ood_free_z),
        support_ood_max_z=float(args.support_ood_max_z),
        support_std_floor=float(args.support_std_floor),
        support_min_confidence=float(args.support_min_confidence),
        enable_slot_reliability_confidence=bool(args.enable_slot_reliability_confidence),
        slot_reliability_power=float(args.slot_reliability_power),
        output_dir=output_dir,
    )
    all_axis = policy_val.get("best_all_axis") is not None
    checkpoint = {
        "schema": "spcarnet_lowrank_uv_residual_texture_checkpoint_v1",
        "candidate_faces": candidate_faces,
        "coeff": coeff.astype(np.float16),
        "counts": counts.astype(np.float16),
        "feature_mean": feature_mean.astype(np.float16),
        "feature_std": feature_std.astype(np.float16),
        "args_json": json.dumps(vars(args), sort_keys=True),
    }
    if slot_reliability is not None:
        checkpoint["slot_reliability"] = slot_reliability.astype(np.float16)
    checkpoint_path = output_dir / "v212_lowrank_uv_residual_texture.npz"
    np.savez_compressed(checkpoint_path, **checkpoint)
    payload: dict[str, Any] = {
        "schema": "spcarnet_lowrank_uv_residual_texture_audit_v1",
        "created_at": "2026-06-29",
        "fit_evidence_dir": str(args.fit_evidence_dir),
        "fit_views": int(len(fit_paths)),
        "policy_val_views": int(len(val_paths)),
        "candidate_face_summary": face_summary,
        "fit": {
            **fit_summary,
            "grid": int(args.grid),
            "bins_per_face": int(args.grid) * int(args.grid),
            "min_bin_count": float(args.min_bin_count),
            "max_abs_delta": float(args.max_abs_delta),
            "min_best_positive_view_fraction": float(args.min_best_positive_view_fraction),
            "min_best_ssim_positive_view_fraction": float(args.min_best_ssim_positive_view_fraction),
            "min_best_lpips_positive_view_fraction": float(args.min_best_lpips_positive_view_fraction),
            "enable_support_confidence": bool(args.enable_support_confidence),
            "support_full_count": float(args.support_full_count),
            "support_count_power": float(args.support_count_power),
            "support_ood_free_z": float(args.support_ood_free_z),
            "support_ood_max_z": float(args.support_ood_max_z),
            "support_std_floor": float(args.support_std_floor),
            "support_min_confidence": float(args.support_min_confidence),
            "enable_slot_reliability_confidence": bool(args.enable_slot_reliability_confidence),
            "slot_reliability_alpha": float(args.slot_reliability_alpha),
            "slot_reliability_max_samples_per_view": int(args.slot_reliability_max_samples_per_view),
            "slot_reliability_tau_quantile": float(args.slot_reliability_tau_quantile),
            "slot_reliability_full_count": float(args.slot_reliability_full_count),
            "slot_reliability_min_positive_fraction": float(args.slot_reliability_min_positive_fraction),
            "slot_reliability_power": float(args.slot_reliability_power),
            "slot_reliability_summary": reliability_summary,
        },
        "policy_val_all_axis_pass": bool(all_axis),
        "policy_val": policy_val,
        "uses_train_fit_teacher": True,
        "uses_policy_val_gt": True,
        "uses_target_or_test_gt": False,
        "checkpoint": str(checkpoint_path),
        "output_json": str(output_dir / "v212_lowrank_uv_residual_texture_audit.json"),
    }
    _write_json(output_dir / "v212_lowrank_uv_residual_texture_audit.json", payload)
    _write_md(output_dir / "v212_lowrank_uv_residual_texture_audit.md", payload)
    if wandb_run is not None:
        best = policy_val["best"]
        best_all = policy_val.get("best_all_axis") or {}
        selected_alpha = str(float((best_all or best)["alpha"]))
        proj = policy_val["projection_by_alpha"][selected_alpha]
        wandb_run.log(
            {
                "policy_val/all_axis_pass": int(all_axis),
                "policy_val/best_alpha": float(best["alpha"]),
                "policy_val/best_psnr_gain": float(best.get("psnr_gain", 0.0)),
                "policy_val/best_ssim_gain": float(best.get("ssim_gain", 0.0)),
                "policy_val/best_lpips_gain": float(best.get("lpips_gain", 0.0)),
                "projection/full_cosine": float(proj["full_residual_summary"]["cosine"]),
                "projection/full_energy_retention": float(proj["full_residual_summary"]["energy_retention"]),
                "projection/active_cosine": float(proj["active_residual_summary"]["cosine"]),
                "projection/active_energy_retention": float(proj["active_residual_summary"]["energy_retention"]),
            }
        )
        wandb_run.finish()
    print(
        json.dumps(
            {
                "output_json": str(output_dir / "v212_lowrank_uv_residual_texture_audit.json"),
                "output_md": str(output_dir / "v212_lowrank_uv_residual_texture_audit.md"),
                "policy_val_all_axis_pass": bool(all_axis),
                "best": policy_val["best"],
                "best_all_axis": policy_val.get("best_all_axis"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
