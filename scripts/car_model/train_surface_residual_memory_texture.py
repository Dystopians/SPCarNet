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
    _load_feature_rows,
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


FORBIDDEN_TARGET_APPLY_KEYS = {
    "rgb_gt",
    "residual_rgb",
    "residual_l1",
    "teacher_residual_rgb",
    "teacher_residual_l1",
    "teacher_residual_rgb_raw",
    "teacher_better_mask",
    "teacher_gain_l1",
    "teacher_parent_delta_l1",
}


def _parse_alpha_grid(value: str) -> list[float]:
    out = [float(x) for x in str(value).replace(";", ",").split(",") if str(x).strip()]
    if not out:
        raise ValueError("alpha grid is empty")
    return out


def _verify_no_gt(evidence_dir: Path) -> dict[str, Any]:
    paths = evidence_views(Path(evidence_dir))
    bad: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for idx, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as z:
            keys = sorted(str(k) for k in z.files)
        present = sorted(set(keys) & FORBIDDEN_TARGET_APPLY_KEYS)
        if idx < 4:
            samples.append({"path": str(path), "keys": keys})
        if present:
            bad.append({"path": str(path), "forbidden_keys": present})
    return {
        "mode": "strict_surface_memory_target_no_gt_preflight",
        "evidence_dir": str(evidence_dir),
        "view_count": int(len(paths)),
        "forbidden_keys": sorted(FORBIDDEN_TARGET_APPLY_KEYS),
        "bad_view_count": int(len(bad)),
        "bad_views": bad[:64],
        "sample_keys": samples,
        "target_gt_visible_to_apply": any("rgb_gt" in set(row["forbidden_keys"]) for row in bad),
        "target_residual_visible_to_apply": any(bool(set(row["forbidden_keys"]) - {"rgb_gt"}) for row in bad),
        "passed": not bad and bool(paths),
    }


def _score_samples(
    z: np.lib.npyio.NpzFile,
    ys: np.ndarray,
    xs: np.ndarray,
    residual: np.ndarray,
    *,
    residual_l1_key: str,
    mode: str,
) -> np.ndarray:
    if str(mode) == "l1" and residual_l1_key in z:
        score = np.asarray(z[residual_l1_key], dtype=np.float32)[ys, xs]
        return np.clip(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None).astype(np.float32)
    if str(mode) == "luma_gradient":
        luma = 0.299 * residual[:, 0] + 0.587 * residual[:, 1] + 0.114 * residual[:, 2]
        score = np.abs(luma).astype(np.float32)
        return np.clip(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    score = np.sqrt(np.sum(np.square(residual.astype(np.float32)), axis=1)).astype(np.float32)
    return np.clip(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def _collect_memory_samples(
    fit_paths: list[Path],
    candidate_faces: np.ndarray,
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    grid: int,
    feature_mode: str,
    max_samples_per_view: int,
    score_mode: str,
    teacher_residual_target_mode: str,
    teacher_residual_target_luma_mix: float,
    teacher_residual_target_edge_boost: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    bins = int(grid) * int(grid)
    slot_chunks: list[np.ndarray] = []
    feature_chunks: list[np.ndarray] = []
    residual_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    active_pixels = 0
    used_pixels = 0
    residual_target_stats: list[dict[str, Any]] = []

    for path in tqdm(fit_paths, desc="collect surface residual memory samples"):
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
                if residual_l1_key in z:
                    score_map = np.asarray(z[residual_l1_key], dtype=np.float32)[ys, xs]
                    prob = np.clip(np.nan_to_num(score_map, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
                    if float(np.sum(prob)) > 0.0:
                        prob = prob.astype(np.float64)
                        prob /= float(np.sum(prob))
                        take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False, p=prob)
                    else:
                        take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
                else:
                    take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
                ys, xs = ys[take], xs[take]
            faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
            face_idx, ok = _face_indices(faces, candidate_faces)
            if not np.any(ok):
                continue
            ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
            bin_id = _bin_ids(z, ys, xs, int(grid)).astype(np.int64)
            slots = face_idx.astype(np.int64) * bins + bin_id
            features = _load_feature_rows(z, ys, xs, feature_mode=str(feature_mode)).astype(np.float32)
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
            score = _score_samples(z, ys, xs, residual, residual_l1_key=str(residual_l1_key), mode=str(score_mode))
            keep = np.isfinite(score) & (score > 0.0)
            if not np.any(keep):
                continue
            slot_chunks.append(slots[keep].astype(np.int64))
            feature_chunks.append(features[keep].astype(np.float16))
            residual_chunks.append(np.clip(residual[keep], -1.0, 1.0).astype(np.float16))
            score_chunks.append(score[keep].astype(np.float32))
            used_pixels += int(np.count_nonzero(keep))

    if not slot_chunks:
        raise RuntimeError("no usable residual-memory samples collected")
    slots_all = np.concatenate(slot_chunks).astype(np.int64)
    features_all = np.concatenate(feature_chunks).astype(np.float16)
    residual_all = np.concatenate(residual_chunks).astype(np.float16)
    score_all = np.concatenate(score_chunks).astype(np.float32)
    return slots_all, features_all, residual_all, score_all, {
        "fit_active_pixels": int(active_pixels),
        "fit_used_pixels": int(used_pixels),
        "raw_sample_count": int(slots_all.size),
        "score_mode": str(score_mode),
        "teacher_residual_target": {
            "mode": str(teacher_residual_target_mode),
            "luma_mix": float(teacher_residual_target_luma_mix),
            "edge_boost": float(teacher_residual_target_edge_boost),
            "mean_luma_mix": _mean([float(s.get("mean_luma_mix", 0.0)) for s in residual_target_stats if s.get("mode") != "raw_rgb"]),
            "energy_ratio_after_before": _mean(
                [float(s.get("energy_ratio_after_before", 1.0)) for s in residual_target_stats if s.get("mode") != "raw_rgb"]
            ),
        },
    }


def _select_prototypes(
    slots: np.ndarray,
    features: np.ndarray,
    residuals: np.ndarray,
    scores: np.ndarray,
    *,
    slot_count: int,
    prototypes_per_slot: int,
    feature_std_floor: float,
) -> dict[str, Any]:
    order = np.lexsort((-scores.astype(np.float64), slots.astype(np.int64)))
    sorted_slots = slots[order]
    keep_rank = np.empty(sorted_slots.shape, dtype=np.int16)
    if sorted_slots.size == 0:
        raise RuntimeError("empty prototype sort")
    first = np.r_[True, sorted_slots[1:] != sorted_slots[:-1]]
    starts = np.flatnonzero(first)
    lengths = np.diff(np.r_[starts, sorted_slots.size])
    keep_rank[:] = -1
    for start, length in zip(starts, lengths, strict=False):
        n = min(int(length), int(prototypes_per_slot))
        keep_rank[start : start + n] = np.arange(n, dtype=np.int16)
    keep = keep_rank >= 0
    proto_slots = sorted_slots[keep].astype(np.int64)
    proto_features = features[order][keep].astype(np.float16)
    proto_residuals = residuals[order][keep].astype(np.float16)
    proto_scores = scores[order][keep].astype(np.float32)
    offsets = np.zeros(int(slot_count) + 1, dtype=np.int64)
    np.add.at(offsets, proto_slots + 1, 1)
    offsets = np.cumsum(offsets, dtype=np.int64)
    feat32 = proto_features.astype(np.float32)
    feature_mean = np.mean(feat32, axis=0).astype(np.float32)
    feature_std = np.std(feat32, axis=0).astype(np.float32)
    feature_std = np.maximum(feature_std, float(feature_std_floor)).astype(np.float32)
    slot_proto_counts = np.diff(offsets)
    return {
        "slot_ids": proto_slots,
        "features": proto_features,
        "residuals": proto_residuals,
        "scores": proto_scores,
        "offsets": offsets,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "summary": {
            "prototype_count": int(proto_slots.size),
            "nonempty_slots": int(np.count_nonzero(slot_proto_counts > 0)),
            "slot_count": int(slot_count),
            "nonempty_slot_fraction": float(np.mean(slot_proto_counts > 0)),
            "mean_prototypes_per_nonempty_slot": float(np.mean(slot_proto_counts[slot_proto_counts > 0])) if np.any(slot_proto_counts > 0) else 0.0,
            "max_prototypes_per_slot": int(np.max(slot_proto_counts)) if slot_proto_counts.size else 0,
            "feature_dim": int(proto_features.shape[1]),
            "feature_std_floor": float(feature_std_floor),
            "score_quantiles": _quantiles(proto_scores.astype(float).tolist()),
        },
    }


def _predict_memory_delta(
    z: np.lib.npyio.NpzFile,
    candidate_faces: np.ndarray,
    memory: dict[str, Any],
    *,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    grid: int,
    feature_mode: str,
    max_abs_delta: float,
    distance_sigma: float,
    confidence_kernel_full: float,
    confidence_count_full: float,
    confidence_count_power: float,
    confidence_min: float,
    enable_confidence: bool,
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
        stats = {"mean": 0.0, "min": 0.0, "max": 0.0, "count_mean": 0.0, "distance_mean": 0.0}
        return (delta, active, confidence_map, stats) if return_confidence else (delta, active)

    faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
    face_idx, ok = _face_indices(faces, candidate_faces)
    ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
    if ys.size == 0:
        stats = {"mean": 0.0, "min": 0.0, "max": 0.0, "count_mean": 0.0, "distance_mean": 0.0}
        return (delta, active, confidence_map, stats) if return_confidence else (delta, active)

    bins = int(grid) * int(grid)
    bin_id = _bin_ids(z, ys, xs, int(grid)).astype(np.int64)
    slots = face_idx.astype(np.int64) * bins + bin_id
    offsets = np.asarray(memory["offsets"], dtype=np.int64)
    starts = offsets[slots]
    ends = offsets[slots + 1]
    counts = (ends - starts).astype(np.int64)
    has = counts > 0
    ys, xs, starts, counts = ys[has], xs[has], starts[has], counts[has]
    if ys.size == 0:
        stats = {"mean": 0.0, "min": 0.0, "max": 0.0, "count_mean": 0.0, "distance_mean": 0.0}
        return (delta, active, confidence_map, stats) if return_confidence else (delta, active)

    features = _load_feature_rows(z, ys, xs, feature_mode=str(feature_mode)).astype(np.float32)
    feature_mean = np.asarray(memory["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(memory["feature_std"], dtype=np.float32)
    norm_features = (features - feature_mean.reshape(1, -1)) / feature_std.reshape(1, -1)
    proto_features = np.asarray(memory["features"], dtype=np.float32)
    proto_residuals = np.asarray(memory["residuals"], dtype=np.float32)
    proto_norm = (proto_features - feature_mean.reshape(1, -1)) / feature_std.reshape(1, -1)
    max_proto = int(memory["prototypes_per_slot"])
    sigma = max(float(distance_sigma), 1.0e-6)
    pred = np.zeros((int(ys.size), 3), dtype=np.float32)
    weight_sum = np.zeros((int(ys.size),), dtype=np.float32)
    best_dist = np.full((int(ys.size),), np.inf, dtype=np.float32)
    for rank in range(max_proto):
        valid = counts > rank
        if not np.any(valid):
            continue
        idx = starts[valid] + int(rank)
        diff = norm_features[valid] - proto_norm[idx]
        dist = np.sqrt(np.mean(diff * diff, axis=1)).astype(np.float32)
        kernel = np.exp(-0.5 * np.square(dist / sigma)).astype(np.float32)
        pred[valid] += proto_residuals[idx] * kernel[:, None]
        weight_sum[valid] += kernel
        best_dist[valid] = np.minimum(best_dist[valid], dist)
    good = weight_sum > 1.0e-8
    pred[good] /= weight_sum[good, None]
    pred[~good] = 0.0
    if bool(enable_confidence):
        kernel_conf = np.clip(weight_sum / max(float(confidence_kernel_full), 1.0e-6), 0.0, 1.0)
        count_conf = np.power(np.clip(counts.astype(np.float32) / max(float(confidence_count_full), 1.0), 0.0, 1.0), max(float(confidence_count_power), 0.0))
        conf = np.maximum(float(confidence_min), kernel_conf * count_conf).astype(np.float32)
    else:
        conf = np.ones((int(ys.size),), dtype=np.float32)
    pred *= conf[:, None]
    if float(max_abs_delta) > 0.0:
        pred = np.clip(pred, -float(max_abs_delta), float(max_abs_delta))
    delta[:, ys, xs] = pred.T
    confidence_map[ys, xs] = conf
    stats = {
        "mean": float(np.mean(conf)) if conf.size else 0.0,
        "min": float(np.min(conf)) if conf.size else 0.0,
        "max": float(np.max(conf)) if conf.size else 0.0,
        "count_mean": float(np.mean(counts)) if counts.size else 0.0,
        "distance_mean": float(np.mean(best_dist[np.isfinite(best_dist)])) if np.any(np.isfinite(best_dist)) else 0.0,
        "kernel_weight_mean": float(np.mean(weight_sum)) if weight_sum.size else 0.0,
    }
    return (delta, active, confidence_map, stats) if return_confidence else (delta, active)


def _summarize_rows(rows: list[dict[str, Any]], *, compute_lpips: bool) -> dict[str, Any]:
    psnr_gain = [float(r["psnr_gain"]) for r in rows]
    ssim_gain = [float(r["ssim_gain"]) for r in rows]
    def lower_cvar(values: list[float], fraction: float = 0.20) -> float:
        if not values:
            return 0.0
        arr = np.sort(np.asarray(values, dtype=np.float64))
        n = max(1, int(math.ceil(float(fraction) * float(arr.size))))
        return float(np.mean(arr[:n]))

    out = {
        "parent_psnr": _mean([float(r["parent_psnr"]) for r in rows]),
        "candidate_psnr": _mean([float(r["candidate_psnr"]) for r in rows]),
        "psnr_gain": _mean(psnr_gain),
        "psnr_min_view_gain": float(np.min(np.asarray(psnr_gain, dtype=np.float64))) if psnr_gain else 0.0,
        "psnr_cvar20_view_gain": lower_cvar(psnr_gain),
        "parent_ssim": _mean([float(r["parent_ssim"]) for r in rows]),
        "candidate_ssim": _mean([float(r["candidate_ssim"]) for r in rows]),
        "ssim_gain": _mean(ssim_gain),
        "ssim_min_view_gain": float(np.min(np.asarray(ssim_gain, dtype=np.float64))) if ssim_gain else 0.0,
        "ssim_cvar20_view_gain": lower_cvar(ssim_gain),
        "positive_view_fraction": float(np.mean(np.asarray(psnr_gain) > 0.0)) if rows else 0.0,
        "ssim_positive_view_fraction": float(np.mean(np.asarray(ssim_gain) > 0.0)) if rows else 0.0,
        "psnr_gain_quantiles": _quantiles(psnr_gain),
        "ssim_gain_quantiles": _quantiles(ssim_gain),
    }
    if compute_lpips:
        lpips_gain = [float(r["lpips_gain"]) for r in rows if r.get("lpips_gain") is not None]
        out.update(
            {
                "parent_lpips": _mean([float(r["parent_lpips"]) for r in rows if r.get("parent_lpips") is not None]),
                "candidate_lpips": _mean([float(r["candidate_lpips"]) for r in rows if r.get("candidate_lpips") is not None]),
                "lpips_gain": _mean(lpips_gain),
                "lpips_min_view_gain": float(np.min(np.asarray(lpips_gain, dtype=np.float64))) if lpips_gain else 0.0,
                "lpips_cvar20_view_gain": lower_cvar(lpips_gain),
                "lpips_positive_view_fraction": float(np.mean(np.asarray(lpips_gain) > 0.0)) if lpips_gain else 0.0,
                "lpips_gain_quantiles": _quantiles(lpips_gain),
            }
        )
    return out


def _evaluate_policy_val(
    val_paths: list[Path],
    candidate_faces: np.ndarray,
    memory: dict[str, Any],
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    grid: int,
    feature_mode: str,
    max_abs_delta: float,
    alpha_grid: list[float],
    compute_lpips: bool,
    ssim_max_side: int,
    lpips_max_side: int,
    min_best_positive_view_fraction: float,
    min_best_ssim_positive_view_fraction: float,
    min_best_lpips_positive_view_fraction: float,
    min_best_psnr_min_view_gain: float,
    min_best_ssim_min_view_gain: float,
    min_best_lpips_min_view_gain: float,
    min_best_psnr_cvar20_view_gain: float,
    min_best_ssim_cvar20_view_gain: float,
    min_best_lpips_cvar20_view_gain: float,
    distance_sigma: float,
    confidence_kernel_full: float,
    confidence_count_full: float,
    confidence_count_power: float,
    confidence_min: float,
    enable_confidence: bool,
    output_dir: Path,
) -> dict[str, Any]:
    lpips_model = build_lpips_model() if compute_lpips else None
    rows_by_alpha: dict[float, list[dict[str, Any]]] = {float(a): [] for a in alpha_grid}
    projection_rows: dict[float, list[dict[str, Any]]] = {float(a): [] for a in alpha_grid}
    active_projection_rows: dict[float, list[dict[str, Any]]] = {float(a): [] for a in alpha_grid}
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(val_paths, desc="eval surface residual memory policy-val"):
        with np.load(path, allow_pickle=False) as z:
            parent = np.asarray(z["rgb_render"], dtype=np.float32)
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)
            raw_delta = np.asarray(z[residual_rgb_key], dtype=np.float32)[:3]
            teacher = np.clip(parent + raw_delta, 0.0, 1.0)
            delta, active_mask = _predict_memory_delta(
                z,
                candidate_faces,
                memory,
                residual_l1_key=str(residual_l1_key),
                min_l1=float(min_l1),
                min_alpha=float(min_alpha),
                grid=int(grid),
                feature_mode=str(feature_mode),
                max_abs_delta=float(max_abs_delta),
                distance_sigma=float(distance_sigma),
                confidence_kernel_full=float(confidence_kernel_full),
                confidence_count_full=float(confidence_count_full),
                confidence_count_power=float(confidence_count_power),
                confidence_min=float(confidence_min),
                enable_confidence=bool(enable_confidence),
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
        row = {"alpha": float(alpha), **_summarize_rows(rows, compute_lpips=compute_lpips)}
        summaries.append(row)
        proj_rows = projection_rows[alpha]
        proj_img = _summarize_rows(proj_rows, compute_lpips=compute_lpips)
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
            and float(row.get("psnr_min_view_gain", 0.0)) >= float(min_best_psnr_min_view_gain)
            and float(row.get("ssim_min_view_gain", 0.0)) >= float(min_best_ssim_min_view_gain)
            and (not compute_lpips or float(row.get("lpips_min_view_gain", 0.0)) >= float(min_best_lpips_min_view_gain))
            and float(row.get("psnr_cvar20_view_gain", 0.0)) >= float(min_best_psnr_cvar20_view_gain)
            and float(row.get("ssim_cvar20_view_gain", 0.0)) >= float(min_best_ssim_cvar20_view_gain)
            and (
                not compute_lpips
                or float(row.get("lpips_cvar20_view_gain", 0.0)) >= float(min_best_lpips_cvar20_view_gain)
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
    for path in tqdm(val_paths, desc="write memory policy-val renders"):
        with np.load(path, allow_pickle=False) as z:
            parent = np.asarray(z["rgb_render"], dtype=np.float32)
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)
            delta, _ = _predict_memory_delta(
                z,
                candidate_faces,
                memory,
                residual_l1_key=str(residual_l1_key),
                min_l1=float(min_l1),
                min_alpha=float(min_alpha),
                grid=int(grid),
                feature_mode=str(feature_mode),
                max_abs_delta=float(max_abs_delta),
                distance_sigma=float(distance_sigma),
                confidence_kernel_full=float(confidence_kernel_full),
                confidence_count_full=float(confidence_count_full),
                confidence_count_power=float(confidence_count_power),
                confidence_min=float(confidence_min),
                enable_confidence=bool(enable_confidence),
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


def _evaluate_target(
    target_evidence_dir: Path,
    eval_gt_evidence_dir: Path,
    candidate_faces: np.ndarray,
    memory: dict[str, Any],
    *,
    residual_l1_key: str,
    min_alpha: float,
    grid: int,
    feature_mode: str,
    max_abs_delta: float,
    alpha: float,
    compute_lpips: bool,
    ssim_max_side: int,
    lpips_max_side: int,
    distance_sigma: float,
    confidence_kernel_full: float,
    confidence_count_full: float,
    confidence_count_power: float,
    confidence_min: float,
    enable_confidence: bool,
    output_dir: Path,
) -> dict[str, Any]:
    no_gt = _verify_no_gt(Path(target_evidence_dir))
    if not bool(no_gt["passed"]):
        return {
            "accepted": False,
            "reject_reason": "target_evidence_no_gt_preflight_failed",
            "no_gt_preflight": no_gt,
            "uses_target_or_test_gt_for_apply": False,
        }
    renders_dir = output_dir / "target_exact_renders"
    gt_dir = output_dir / "target_exact_gt"
    renders_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    eval_index = {p.stem: p for p in evidence_views(eval_gt_evidence_dir)}
    lpips_model = build_lpips_model() if compute_lpips else None
    rows: list[dict[str, Any]] = []
    changed_pixels = 0
    active_pixels = 0
    effective_pixels = 0
    total_pixels = 0
    confidence_rows: list[dict[str, float]] = []
    for path in tqdm(evidence_views(target_evidence_dir), desc="apply surface residual memory target"):
        with np.load(path, allow_pickle=False) as z:
            parent = np.asarray(z["rgb_render"], dtype=np.float32)
            delta, active, conf_map, conf_stats = _predict_memory_delta(
                z,
                candidate_faces,
                memory,
                residual_l1_key=str(residual_l1_key),
                min_l1=0.0,
                min_alpha=float(min_alpha),
                grid=int(grid),
                feature_mode=str(feature_mode),
                max_abs_delta=float(max_abs_delta),
                distance_sigma=float(distance_sigma),
                confidence_kernel_full=float(confidence_kernel_full),
                confidence_count_full=float(confidence_count_full),
                confidence_count_power=float(confidence_count_power),
                confidence_min=float(confidence_min),
                enable_confidence=bool(enable_confidence),
                return_confidence=True,
            )
            adapted = np.clip(parent + float(alpha) * delta, 0.0, 1.0)
        changed = np.any(np.abs(adapted - parent) > (0.5 / 255.0), axis=0)
        changed_pixels += int(np.count_nonzero(changed))
        active_pixels += int(np.count_nonzero(active))
        effective_pixels += int(np.count_nonzero(conf_map > 0.0))
        total_pixels += int(changed.size)
        confidence_rows.append(conf_stats)
        save_image_chw(renders_dir / f"{path.stem}.png", adapted)
        row: dict[str, Any] = {
            "view": path.stem,
            "changed_fraction": float(np.mean(changed)),
            "active_fraction": float(np.mean(active)),
            "effective_confidence_fraction": float(np.mean(conf_map > 0.0)),
            **{f"confidence_{k}": float(v) for k, v in conf_stats.items()},
        }
        if path.stem in eval_index:
            with np.load(eval_index[path.stem], allow_pickle=False) as gt_z:
                gt = np.asarray(gt_z["rgb_gt"], dtype=np.float32)
            save_image_chw(gt_dir / f"{path.stem}.png", gt)
            p_psnr = _psnr(parent, gt)
            c_psnr = _psnr(adapted, gt)
            p_ssim = image_ssim_chw(parent, gt, int(ssim_max_side))
            c_ssim = image_ssim_chw(adapted, gt, int(ssim_max_side))
            row.update(
                {
                    "parent_psnr": float(p_psnr),
                    "candidate_psnr": float(c_psnr),
                    "psnr_gain": float(c_psnr - p_psnr),
                    "parent_ssim": float(p_ssim),
                    "candidate_ssim": float(c_ssim),
                    "ssim_gain": float(c_ssim - p_ssim),
                }
            )
            if compute_lpips:
                p_lp = image_lpips_chw(parent, gt, int(lpips_max_side), lpips_model)
                c_lp = image_lpips_chw(adapted, gt, int(lpips_max_side), lpips_model)
                row.update({"parent_lpips": float(p_lp), "candidate_lpips": float(c_lp), "lpips_gain": float(p_lp - c_lp)})
        rows.append(row)

    metric_rows = [row for row in rows if "candidate_psnr" in row]
    metrics = _summarize_rows(metric_rows, compute_lpips=compute_lpips) if metric_rows else {"view_count": 0}
    metrics["view_count"] = int(len(metric_rows))
    gate_ref = {"psnr": 20.304358, "ssim": 0.557770, "lpips": 0.329222}
    gate_pass = bool(
        metric_rows
        and float(metrics.get("candidate_psnr", -math.inf)) > gate_ref["psnr"]
        and float(metrics.get("candidate_ssim", -math.inf)) > gate_ref["ssim"]
        and float(metrics.get("candidate_lpips", math.inf)) < gate_ref["lpips"]
    )
    return {
        "accepted": True,
        "target_evidence_dir": str(target_evidence_dir),
        "eval_gt_evidence_dir": str(eval_gt_evidence_dir),
        "renders_dir": str(renders_dir),
        "gt_dir": str(gt_dir),
        "selected_alpha": float(alpha),
        "no_gt_preflight": no_gt,
        "target_apply": {
            "view_count": int(len(rows)),
            "changed_pixels": int(changed_pixels),
            "active_pixels": int(active_pixels),
            "effective_confidence_pixels": int(effective_pixels),
            "total_pixels": int(total_pixels),
            "changed_fraction": float(changed_pixels / max(1, total_pixels)),
            "active_fraction": float(active_pixels / max(1, total_pixels)),
            "effective_confidence_fraction": float(effective_pixels / max(1, total_pixels)),
            "confidence_mean": _mean([float(x.get("mean", 0.0)) for x in confidence_rows]),
            "confidence_distance_mean": _mean([float(x.get("distance_mean", 0.0)) for x in confidence_rows]),
            "confidence_count_mean": _mean([float(x.get("count_mean", 0.0)) for x in confidence_rows]),
        },
        "metrics": metrics,
        "per_view": rows,
        "phasej_flowers_reference": gate_ref,
        "phasej_flowers_gate_pass": bool(gate_pass),
        "uses_target_or_test_gt_for_apply": False,
        "uses_target_or_test_gt_for_eval_only": bool(metric_rows),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    best_all = payload["policy_val"].get("best_all_axis")
    best = payload["policy_val"].get("best")
    selected = best_all or best
    exact = payload.get("target_exact") or {}
    lines = [
        "# v223 Surface Residual Memory Texture Audit",
        "",
        f"- policy-val all-axis pass: `{payload['policy_val_all_axis_pass']}`",
        f"- selected faces: `{payload['candidate_face_summary']['selected_faces']}`",
        f"- selected score coverage: `{payload['candidate_face_summary'].get('selected_score_coverage', 0.0):.6f}`",
        f"- grid: `{payload['fit']['grid']}`",
        f"- prototypes per slot: `{payload['fit']['prototypes_per_slot']}`",
        f"- feature mode: `{payload['fit']['feature_mode']}`",
        f"- prototype count: `{payload['fit']['prototype_summary']['prototype_count']}`",
        f"- nonempty slots: `{payload['fit']['prototype_summary']['nonempty_slots']}`",
        f"- teacher residual target: `{payload['fit']['teacher_residual_target']}`",
        f"- best all-axis: `{best_all}`",
        "",
        "## Selected Policy-Val",
        "",
        "| alpha | PSNR gain | SSIM gain | LPIPS gain | PSNR+/SSIM+/LPIPS+ frac |",
        "|---:|---:|---:|---:|---:|",
        (
            "| {alpha:.4f} | {psnr:+.6f} | {ssim:+.6f} | {lpips:+.6f} | {pf:.2f}/{sf:.2f}/{lf:.2f} |"
        ).format(
            alpha=float(selected.get("alpha", 0.0)),
            psnr=float(selected.get("psnr_gain", 0.0)),
            ssim=float(selected.get("ssim_gain", 0.0)),
            lpips=float(selected.get("lpips_gain", 0.0)),
            pf=float(selected.get("positive_view_fraction", 0.0)),
            sf=float(selected.get("ssim_positive_view_fraction", 0.0)),
            lf=float(selected.get("lpips_positive_view_fraction", 0.0)),
        ),
        "",
        "## Selected Tail Certificate",
        "",
        "| metric | min-view gain | CVaR20 gain |",
        "|---|---:|---:|",
        f"| PSNR | {float(selected.get('psnr_min_view_gain', 0.0)):+.6f} | {float(selected.get('psnr_cvar20_view_gain', 0.0)):+.6f} |",
        f"| SSIM | {float(selected.get('ssim_min_view_gain', 0.0)):+.6f} | {float(selected.get('ssim_cvar20_view_gain', 0.0)):+.6f} |",
        f"| LPIPS | {float(selected.get('lpips_min_view_gain', 0.0)):+.6f} | {float(selected.get('lpips_cvar20_view_gain', 0.0)):+.6f} |",
        "",
    ]
    alpha_key = str(float(selected.get("alpha", 0.0)))
    projection = payload["policy_val"].get("projection_by_alpha", {}).get(alpha_key)
    if projection:
        lines += [
            "## Projection At Selected Alpha",
            "",
            "| scope | PSNR gain vs teacher | SSIM gain vs teacher | LPIPS gain vs teacher | cosine | energy retention | changed fraction |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                "| full valid | {psnr:+.6f} | {ssim:+.6f} | {lpips:+.6f} | {cos:.6f} | {ret:.6f} | {chg:.6f} |"
            ).format(
                psnr=float(projection["image_summary"]["psnr_gain"]),
                ssim=float(projection["image_summary"]["ssim_gain"]),
                lpips=float(projection["image_summary"].get("lpips_gain", 0.0)),
                cos=float(projection["full_residual_summary"]["cosine"]),
                ret=float(projection["full_residual_summary"]["energy_retention"]),
                chg=float(projection["full_residual_summary"]["changed_fraction"]),
            ),
            (
                "| selected active | {psnr:+.6f} | {ssim:+.6f} | {lpips:+.6f} | {cos:.6f} | {ret:.6f} | {chg:.6f} |"
            ).format(
                psnr=float(projection["image_summary"]["psnr_gain"]),
                ssim=float(projection["image_summary"]["ssim_gain"]),
                lpips=float(projection["image_summary"].get("lpips_gain", 0.0)),
                cos=float(projection["active_residual_summary"]["cosine"]),
                ret=float(projection["active_residual_summary"]["energy_retention"]),
                chg=float(projection["active_residual_summary"]["changed_fraction"]),
            ),
            "",
        ]
    if exact:
        m = exact.get("metrics", {})
        lines += [
            "## Flowers Target Exact",
            "",
            f"- accepted: `{exact.get('accepted')}`",
            f"- no-GT preflight passed: `{(exact.get('no_gt_preflight') or {}).get('passed')}`",
            f"- phase-j gate pass: `{exact.get('phasej_flowers_gate_pass')}`",
            "",
            "| row | PSNR | SSIM | LPIPS |",
            "|---|---:|---:|---:|",
            f"| parent | {m.get('parent_psnr', 0.0):.6f} | {m.get('parent_ssim', 0.0):.6f} | {m.get('parent_lpips', 0.0):.6f} |",
            f"| candidate | {m.get('candidate_psnr', 0.0):.6f} | {m.get('candidate_ssim', 0.0):.6f} | {m.get('candidate_lpips', 0.0):.6f} |",
            f"| gain | {m.get('psnr_gain', 0.0):+.6f} | {m.get('ssim_gain', 0.0):+.6f} | {m.get('lpips_gain', 0.0):+.6f} |",
            "",
        ]
    lines += [
        "## Artifacts",
        "",
        f"- JSON: `{payload['output_json']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- policy-val renders: `{payload['policy_val']['render_dir']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit a surface-attached teacher residual memory texture.")
    parser.add_argument("--fit_evidence_dir", default=DEFAULT_EVIDENCE)
    parser.add_argument("--target_evidence_dir", default="")
    parser.add_argument("--eval_gt_evidence_dir", default="")
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    parser.add_argument("--min_l1", type=float, default=0.0005)
    parser.add_argument("--min_alpha", type=float, default=0.03)
    parser.add_argument("--max_candidate_faces", type=int, default=524288)
    parser.add_argument("--candidate_target_energy_coverage", type=float, default=0.97)
    parser.add_argument("--max_candidate_face_samples_per_view", type=int, default=65536)
    parser.add_argument("--max_memory_samples_per_view", type=int, default=131072)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--feature_mode", default="basic", choices=["basic", "fourier_v1"])
    parser.add_argument("--prototypes_per_slot", type=int, default=4)
    parser.add_argument("--score_mode", default="energy", choices=["energy", "l1", "luma_gradient"])
    parser.add_argument("--feature_std_floor", type=float, default=0.05)
    parser.add_argument("--teacher_residual_target_mode", default="raw_rgb", choices=["raw_rgb", "luma_only", "edge_luma_mix"])
    parser.add_argument("--teacher_residual_target_luma_mix", type=float, default=0.75)
    parser.add_argument("--teacher_residual_target_edge_boost", type=float, default=0.25)
    parser.add_argument("--max_abs_delta", type=float, default=0.25)
    parser.add_argument("--distance_sigma", type=float, default=2.0)
    parser.add_argument("--enable_confidence", action="store_true")
    parser.add_argument("--confidence_kernel_full", type=float, default=1.0)
    parser.add_argument("--confidence_count_full", type=float, default=4.0)
    parser.add_argument("--confidence_count_power", type=float, default=0.5)
    parser.add_argument("--confidence_min", type=float, default=0.0)
    parser.add_argument("--alpha_grid", default="0,0.03125,0.0625,0.125,0.25,0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--policy_val_ssim_max_side", type=int, default=512)
    parser.add_argument("--policy_val_lpips_max_side", type=int, default=256)
    parser.add_argument("--min_best_positive_view_fraction", type=float, default=1.0)
    parser.add_argument("--min_best_ssim_positive_view_fraction", type=float, default=0.75)
    parser.add_argument("--min_best_lpips_positive_view_fraction", type=float, default=0.50)
    parser.add_argument("--min_best_psnr_min_view_gain", type=float, default=0.0)
    parser.add_argument("--min_best_ssim_min_view_gain", type=float, default=-1.0e-7)
    parser.add_argument("--min_best_lpips_min_view_gain", type=float, default=-1.0e-7)
    parser.add_argument("--min_best_psnr_cvar20_view_gain", type=float, default=0.0)
    parser.add_argument("--min_best_ssim_cvar20_view_gain", type=float, default=0.0)
    parser.add_argument("--min_best_lpips_cvar20_view_gain", type=float, default=0.0)
    parser.add_argument("--run_target_exact_if_policy_pass", action="store_true")
    parser.add_argument("--force_target_exact", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet-surface-memory-texture")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--seed", type=int, default=223)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = evidence_views(Path(args.fit_evidence_dir))
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
    if candidate_faces.size <= 0:
        raise SystemExit("no candidate faces selected")

    slots, features, residuals, scores, collect_summary = _collect_memory_samples(
        fit_paths,
        candidate_faces,
        residual_rgb_key=str(args.residual_rgb_key),
        residual_l1_key=str(args.residual_l1_key),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        grid=int(args.grid),
        feature_mode=str(args.feature_mode),
        max_samples_per_view=int(args.max_memory_samples_per_view),
        score_mode=str(args.score_mode),
        teacher_residual_target_mode=str(args.teacher_residual_target_mode),
        teacher_residual_target_luma_mix=float(args.teacher_residual_target_luma_mix),
        teacher_residual_target_edge_boost=float(args.teacher_residual_target_edge_boost),
        seed=int(args.seed) + 17,
    )
    slot_count = int(candidate_faces.size) * int(args.grid) * int(args.grid)
    proto = _select_prototypes(
        slots,
        features,
        residuals,
        scores,
        slot_count=slot_count,
        prototypes_per_slot=int(args.prototypes_per_slot),
        feature_std_floor=float(args.feature_std_floor),
    )
    memory = {
        "slot_ids": proto["slot_ids"],
        "features": proto["features"],
        "residuals": proto["residuals"],
        "scores": proto["scores"],
        "offsets": proto["offsets"],
        "feature_mean": proto["feature_mean"],
        "feature_std": proto["feature_std"],
        "prototypes_per_slot": int(args.prototypes_per_slot),
    }

    policy_val = _evaluate_policy_val(
        val_paths,
        candidate_faces,
        memory,
        residual_rgb_key=str(args.residual_rgb_key),
        residual_l1_key=str(args.residual_l1_key),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        grid=int(args.grid),
        feature_mode=str(args.feature_mode),
        max_abs_delta=float(args.max_abs_delta),
        alpha_grid=_parse_alpha_grid(str(args.alpha_grid)),
        compute_lpips=bool(args.compute_lpips),
        ssim_max_side=int(args.policy_val_ssim_max_side),
        lpips_max_side=int(args.policy_val_lpips_max_side),
        min_best_positive_view_fraction=float(args.min_best_positive_view_fraction),
        min_best_ssim_positive_view_fraction=float(args.min_best_ssim_positive_view_fraction),
        min_best_lpips_positive_view_fraction=float(args.min_best_lpips_positive_view_fraction),
        min_best_psnr_min_view_gain=float(args.min_best_psnr_min_view_gain),
        min_best_ssim_min_view_gain=float(args.min_best_ssim_min_view_gain),
        min_best_lpips_min_view_gain=float(args.min_best_lpips_min_view_gain),
        min_best_psnr_cvar20_view_gain=float(args.min_best_psnr_cvar20_view_gain),
        min_best_ssim_cvar20_view_gain=float(args.min_best_ssim_cvar20_view_gain),
        min_best_lpips_cvar20_view_gain=float(args.min_best_lpips_cvar20_view_gain),
        distance_sigma=float(args.distance_sigma),
        confidence_kernel_full=float(args.confidence_kernel_full),
        confidence_count_full=float(args.confidence_count_full),
        confidence_count_power=float(args.confidence_count_power),
        confidence_min=float(args.confidence_min),
        enable_confidence=bool(args.enable_confidence),
        output_dir=output_dir,
    )
    policy_pass = policy_val.get("best_all_axis") is not None
    selected_alpha = float((policy_val.get("best_all_axis") or policy_val["best"])["alpha"])

    checkpoint = output_dir / "v223_surface_residual_memory_texture.npz"
    np.savez_compressed(
        checkpoint,
        candidate_faces=candidate_faces.astype(np.int64),
        slot_ids=memory["slot_ids"].astype(np.int64),
        offsets=memory["offsets"].astype(np.int64),
        features=memory["features"].astype(np.float16),
        residuals=memory["residuals"].astype(np.float16),
        scores=memory["scores"].astype(np.float32),
        feature_mean=memory["feature_mean"].astype(np.float32),
        feature_std=memory["feature_std"].astype(np.float32),
        args_json=json.dumps(vars(args), sort_keys=True),
    )

    target_exact = None
    can_run_exact = (
        (bool(args.run_target_exact_if_policy_pass) and bool(policy_pass)) or bool(args.force_target_exact)
    ) and bool(str(args.target_evidence_dir)) and bool(str(args.eval_gt_evidence_dir))
    if can_run_exact:
        target_exact = _evaluate_target(
            Path(args.target_evidence_dir),
            Path(args.eval_gt_evidence_dir),
            candidate_faces,
            memory,
            residual_l1_key=str(args.residual_l1_key),
            min_alpha=float(args.min_alpha),
            grid=int(args.grid),
            feature_mode=str(args.feature_mode),
            max_abs_delta=float(args.max_abs_delta),
            alpha=float(selected_alpha),
            compute_lpips=bool(args.compute_lpips),
            ssim_max_side=int(args.policy_val_ssim_max_side),
            lpips_max_side=int(args.policy_val_lpips_max_side),
            distance_sigma=float(args.distance_sigma),
            confidence_kernel_full=float(args.confidence_kernel_full),
            confidence_count_full=float(args.confidence_count_full),
            confidence_count_power=float(args.confidence_count_power),
            confidence_min=float(args.confidence_min),
            enable_confidence=bool(args.enable_confidence),
            output_dir=output_dir,
        )

    payload: dict[str, Any] = {
        "schema": "spcarnet_surface_residual_memory_texture_v1",
        "fit_evidence_dir": str(args.fit_evidence_dir),
        "fit_views": int(len(fit_paths)),
        "policy_val_views": int(len(val_paths)),
        "candidate_face_summary": face_summary,
        "fit": {
            "grid": int(args.grid),
            "bins_per_face": int(args.grid) * int(args.grid),
            "feature_mode": str(args.feature_mode),
            "prototypes_per_slot": int(args.prototypes_per_slot),
            "max_abs_delta": float(args.max_abs_delta),
            "distance_sigma": float(args.distance_sigma),
            "enable_confidence": bool(args.enable_confidence),
            "confidence_kernel_full": float(args.confidence_kernel_full),
            "confidence_count_full": float(args.confidence_count_full),
            "confidence_count_power": float(args.confidence_count_power),
            "confidence_min": float(args.confidence_min),
            "prototype_summary": proto["summary"],
            **collect_summary,
        },
        "uses_train_fit_teacher": True,
        "uses_policy_val_gt": True,
        "uses_target_or_test_gt": False,
        "policy_val_all_axis_pass": bool(policy_pass),
        "policy_val": policy_val,
        "target_exact": target_exact,
        "checkpoint": str(checkpoint),
        "output_json": str(output_dir / "v223_surface_residual_memory_texture_audit.json"),
    }
    _write_json(output_dir / "v223_surface_residual_memory_texture_audit.json", payload)
    _write_md(output_dir / "v223_surface_residual_memory_texture_audit.md", payload)

    if bool(args.enable_wandb):
        try:
            import wandb

            run = wandb.init(
                project=str(args.wandb_project),
                name=str(args.wandb_run_name or output_dir.name),
                config=vars(args),
                dir=str(output_dir),
            )
            best = policy_val["best"]
            best_all = policy_val.get("best_all_axis") or {}
            log_payload = {
                "policy_val/all_axis_pass": float(policy_pass),
                "policy_val/best_alpha": float(best.get("alpha", 0.0)),
                "policy_val/best_psnr_gain": float(best.get("psnr_gain", 0.0)),
                "policy_val/best_ssim_gain": float(best.get("ssim_gain", 0.0)),
                "policy_val/best_lpips_gain": float(best.get("lpips_gain", 0.0)),
                "policy_val/best_all_psnr_gain": float(best_all.get("psnr_gain", 0.0)),
                "fit/prototype_count": float(proto["summary"]["prototype_count"]),
                "fit/nonempty_slot_fraction": float(proto["summary"]["nonempty_slot_fraction"]),
            }
            if target_exact and target_exact.get("accepted"):
                m = target_exact.get("metrics", {})
                log_payload.update(
                    {
                        "exact/candidate_psnr": float(m.get("candidate_psnr", 0.0)),
                        "exact/candidate_ssim": float(m.get("candidate_ssim", 0.0)),
                        "exact/candidate_lpips": float(m.get("candidate_lpips", 0.0)),
                        "exact/psnr_gain": float(m.get("psnr_gain", 0.0)),
                        "exact/phasej_gate_pass": float(bool(target_exact.get("phasej_flowers_gate_pass"))),
                    }
                )
            run.log(log_payload)
            run.finish()
        except Exception as exc:  # pragma: no cover
            print(f"[wandb] disabled after init failure: {type(exc).__name__}: {exc}", flush=True)

    print(
        json.dumps(
            {
                "output_json": payload["output_json"],
                "output_md": str(output_dir / "v223_surface_residual_memory_texture_audit.md"),
                "checkpoint": str(checkpoint),
                "policy_val_all_axis_pass": bool(policy_pass),
                "best": policy_val["best"],
                "best_all_axis": policy_val.get("best_all_axis"),
                "target_exact": target_exact,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
