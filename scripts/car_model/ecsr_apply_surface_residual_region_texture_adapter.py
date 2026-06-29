#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.loss_utils import ssim  # noqa: E402


IGNORE_COPY_NAMES = {
    "train",
    "test",
    "results.json",
    "per_view.json",
    "test_results.json",
    "test_per_view.json",
    "policy_val_results.json",
    "policy_val_per_view.json",
    "geometry_eval_colmap",
}


@dataclass
class FaceAtlas:
    texture: np.ndarray
    counts: np.ndarray
    variance: np.ndarray
    sign_consistency: np.ndarray
    mean_rgb: np.ndarray
    samples: int
    view_basis_mode: str = "none"
    view_basis_coefficients: np.ndarray | None = None
    view_basis_support: np.ndarray | None = None
    view_basis_feature_mean: np.ndarray | None = None
    view_basis_feature_std: np.ndarray | None = None
    view_basis_ood_mode: str = "none"
    view_basis_ood_max_z: float = 0.0
    view_basis_ood_min_std: float = 1.0e-3
    teacher_basis_mode: str = "none"
    teacher_basis_coefficients: np.ndarray | None = None
    teacher_basis_feature_mean: np.ndarray | None = None
    teacher_basis_feature_std: np.ndarray | None = None
    teacher_basis_ood_max_z: float = 0.0
    teacher_basis_ood_min_std: float = 1.0e-3
    teacher_basis_apply_mode: str = "replace_supported"
    teacher_basis_blend: float = 1.0
    teacher_texture_basis: np.ndarray | None = None
    teacher_texture_support: np.ndarray | None = None
    teacher_texture_energy: np.ndarray | None = None
    expert_textures: np.ndarray | None = None
    expert_counts: np.ndarray | None = None
    expert_variance: np.ndarray | None = None
    expert_sign_consistency: np.ndarray | None = None
    expert_samples: np.ndarray | None = None
    expert_centers: np.ndarray | None = None
    expert_feature_mode: str = "none"
    expert_min_bin_samples: int = 1
    expert_fallback_mode: str = "global"


def parse_alpha_grid(text: str) -> list[float]:
    values: list[float] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
    if 0.0 not in values:
        values.append(0.0)
    return sorted(set(values))


def refine_alpha_grid_for_policy_val_ssim(
    base_alpha_grid: list[float],
    *,
    enabled: bool,
    steps: int,
    min_alpha: float,
) -> tuple[list[float], dict[str, Any]]:
    base = sorted(set(float(x) for x in base_alpha_grid))
    if not bool(enabled):
        return base, {"enabled": False, "alpha_grid": base}
    generated: set[float] = set(base)
    inserted: list[float] = []
    min_alpha = max(0.0, float(min_alpha))
    steps = max(0, int(steps))
    for alpha in [float(x) for x in base if float(x) > 0.0]:
        refined = float(alpha)
        for _ in range(steps):
            refined *= 0.5
            if refined <= 0.0:
                break
            if refined < min_alpha:
                continue
            rounded = round(float(refined), 12)
            if rounded not in generated:
                generated.add(rounded)
                inserted.append(rounded)
    combined = sorted(generated)
    return combined, {
        "enabled": True,
        "base_alpha_grid": base,
        "steps": int(steps),
        "min_alpha": float(min_alpha),
        "inserted_alpha_count": int(len(inserted)),
        "inserted_alpha_grid": sorted(inserted),
        "alpha_grid": combined,
    }


def augment_alpha_grid_with_midpoints(
    base_alpha_grid: list[float],
    *,
    enabled: bool,
) -> tuple[list[float], dict[str, Any]]:
    base = sorted(set(float(x) for x in base_alpha_grid))
    if not bool(enabled):
        return base, {"enabled": False, "alpha_grid": base}
    generated: set[float] = set(base)
    inserted: list[float] = []
    for lo, hi in zip(base[:-1], base[1:]):
        lo = float(lo)
        hi = float(hi)
        if hi <= 0.0 or hi <= lo:
            continue
        midpoint = round(float(0.5 * (lo + hi)), 12)
        if midpoint <= 0.0 or midpoint in generated:
            continue
        generated.add(midpoint)
        inserted.append(midpoint)
    combined = sorted(generated)
    return combined, {
        "enabled": True,
        "base_alpha_grid": base,
        "inserted_alpha_count": int(len(inserted)),
        "inserted_alpha_grid": sorted(inserted),
        "alpha_grid": combined,
    }


def parse_int_candidates(text: str, default_value: int) -> list[int]:
    values: list[int] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"candidate values must be positive, got {value}")
        values.append(value)
    if not values:
        values = [int(default_value)]
    return sorted(set(values))


def parse_float_candidates(text: str) -> list[float]:
    values: list[float] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return sorted(set(values))


def clip_delta_rgb(delta: np.ndarray, max_abs_delta_rgb: float) -> np.ndarray:
    """Apply the same RGB residual cap used by target rendering."""
    cap = float(max_abs_delta_rgb)
    if cap > 0.0:
        return np.clip(delta, -cap, cap)
    return delta


_LUMA_WEIGHTS = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)


def _luma_gradient_magnitude(rgb_chw: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_chw, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        return np.zeros((0, 0), dtype=np.float32)
    luma = (
        _LUMA_WEIGHTS[0] * rgb[0]
        + _LUMA_WEIGHTS[1] * rgb[1]
        + _LUMA_WEIGHTS[2] * rgb[2]
    )
    grad_x = np.zeros_like(luma, dtype=np.float32)
    grad_y = np.zeros_like(luma, dtype=np.float32)
    grad_x[:, 1:] = np.abs(luma[:, 1:] - luma[:, :-1])
    grad_y[1:, :] = np.abs(luma[1:, :] - luma[:-1, :])
    return np.maximum(grad_x, grad_y).astype(np.float32)


def transform_residual_samples_for_fit(
    z: np.lib.npyio.NpzFile,
    mask: np.ndarray,
    residual_samples: np.ndarray,
    mode: str,
    luma_mix: float,
    edge_boost: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode_s = str(mode or "raw_rgb")
    if mode_s == "raw_rgb":
        return residual_samples, {"mode": "raw_rgb", "sample_count": int(residual_samples.shape[0])}
    samples = np.asarray(residual_samples, dtype=np.float32)
    if samples.size == 0:
        return samples, {"mode": mode_s, "sample_count": 0}
    luma = samples @ _LUMA_WEIGHTS.reshape(3, 1)
    luma_rgb = np.repeat(luma.astype(np.float32), 3, axis=1)
    if mode_s == "luma_only":
        transformed = luma_rgb
        mix_values = np.ones((samples.shape[0],), dtype=np.float32)
    elif mode_s == "edge_luma_mix":
        base_mix = float(np.clip(float(luma_mix), 0.0, 1.0))
        edge_extra = float(np.clip(float(edge_boost), 0.0, 1.0))
        edge_weight = np.zeros((samples.shape[0],), dtype=np.float32)
        if "rgb_render" in z:
            grad = _luma_gradient_magnitude(np.asarray(z["rgb_render"], dtype=np.float32))
            if grad.size:
                ys, xs = np.nonzero(mask)
                values = grad[ys, xs].astype(np.float32)
                hi = float(np.quantile(values.astype(np.float64), 0.95)) if values.size else 0.0
                if hi > 1.0e-8:
                    edge_weight = np.clip(values / hi, 0.0, 1.0).astype(np.float32)
        mix_values = np.clip(base_mix + edge_extra * edge_weight, 0.0, 1.0).astype(np.float32)
        transformed = (1.0 - mix_values[:, None]) * samples + mix_values[:, None] * luma_rgb
    else:
        raise ValueError(f"unsupported teacher residual target mode: {mode_s}")
    before_energy = float(np.mean(np.sum(samples * samples, axis=1))) if samples.size else 0.0
    after_energy = float(np.mean(np.sum(transformed * transformed, axis=1))) if transformed.size else 0.0
    return transformed.astype(np.float32), {
        "mode": mode_s,
        "sample_count": int(samples.shape[0]),
        "mean_luma_mix": float(np.mean(mix_values)) if mix_values.size else 0.0,
        "min_luma_mix": float(np.min(mix_values)) if mix_values.size else 0.0,
        "max_luma_mix": float(np.max(mix_values)) if mix_values.size else 0.0,
        "mean_rgb_energy_before": before_energy,
        "mean_rgb_energy_after": after_energy,
        "energy_ratio_after_before": float(after_energy / before_energy) if before_energy > 1.0e-12 else 0.0,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def face_set_sha1(faces: set[int]) -> str:
    digest = hashlib.sha1()
    count = 0
    for face in sorted(int(face) for face in faces):
        digest.update(str(face).encode("ascii"))
        digest.update(b",")
        count += 1
    digest.update(f"n={count}".encode("ascii"))
    return digest.hexdigest()


def candidate_spec_audit_row(
    spec: dict[str, Any],
    candidate_index: int,
    candidate_count: int,
) -> dict[str, Any]:
    return {
        "candidate_index": int(candidate_index),
        "candidate_count": int(candidate_count),
        "fill_mode": str(spec.get("fill_mode", "")),
        "texture_size": int(spec.get("texture_size", 0)),
        "teacher_distilled_low_rank_texture_rank": int(
            spec.get("teacher_distilled_low_rank_texture_rank", 0)
        ),
        "surface_multiscale_prior_blend": float(spec.get("surface_multiscale_prior_blend", 0.0)),
        "max_abs_delta_rgb": float(spec.get("max_abs_delta_rgb", 0.0)),
        "support_mode": str(spec.get("support_mode", "")),
        "support_added_faces": int(spec.get("support_added_faces", 0)),
        "support_candidate_faces": int(spec.get("support_candidate_faces", 0)),
        "support_faces_sha1": str(spec.get("support_faces_sha1", "")),
    }


def policy_candidate_spec_key(spec: dict[str, Any]) -> tuple[str, int, int, float, float, str]:
    return (
        str(spec.get("fill_mode", "")),
        int(spec.get("texture_size", 0)),
        int(spec.get("teacher_distilled_low_rank_texture_rank", 0)),
        round(float(spec.get("surface_multiscale_prior_blend", 0.0)), 8),
        round(float(spec.get("max_abs_delta_rgb", 0.0)), 8),
        str(spec.get("support_faces_sha1", "")),
    )


def evidence_views(evidence_dir: Path) -> list[Path]:
    views_dir = evidence_dir / "views"
    if views_dir.is_dir():
        return sorted(views_dir.glob("*.npz"))
    return sorted(evidence_dir.glob("*.npz"))


def plan_adaptive_texture_size_ladder(
    view_paths: list[Path],
    candidate_faces: set[int],
    base_texture_sizes: list[int],
    *,
    residual_l1_key: str,
    policy_val_stride: int,
    min_l1: float,
    min_alpha: float,
    max_samples_per_view: int,
    max_size: int,
    min_fit_samples_per_face: float,
    min_samples_per_current_bin: float,
    min_mean_l1: float,
    support_modes: list[str] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Plan a texture-size ladder from train-fit residual density only."""
    base_sizes = sorted(set(int(x) for x in base_texture_sizes if int(x) > 0))
    if not base_sizes:
        base_sizes = [16]
    candidate_faces_set = set(int(face) for face in candidate_faces)
    support_mode_names = sorted(set(str(mode) for mode in (support_modes or []) if str(mode)))
    summary: dict[str, Any] = {
        "enabled": True,
        "mode": "train_fit_residual_density_ladder",
        "capacity_candidate_generation_scope": "train_fit_residual_density_plus_gt_free_support_union",
        "final_texture_size_selection_scope": "train_policy_val_over_train_generated_candidates",
        "base_texture_size_candidates": list(base_sizes),
        "planned_texture_size_candidates": list(base_sizes),
        "max_size": int(max_size),
        "input_view_count": int(len(view_paths)),
        "policy_val_stride": int(policy_val_stride),
        "fit_split_rule": "skip views where view_index % policy_val_stride == 0 when stride > 1",
        "residual_l1_key": str(residual_l1_key),
        "uses_policy_val_gt": False,
        "uses_target_or_test_gt": False,
        "uses_target_geometry_or_visibility": bool(
            any("target" in mode or "coview" in mode for mode in support_mode_names)
        ),
        "support_modes": support_mode_names,
        "support_union_faces_sha1": face_set_sha1(candidate_faces_set),
        "candidate_faces": int(len(candidate_faces_set)),
        "added_texture_size_candidates": [],
        "rejected_texture_size_candidates": [],
        "reason": "",
    }
    max_size = int(max_size)
    if max_size <= max(base_sizes):
        summary["reason"] = "max_size_not_larger_than_existing_candidates"
        return base_sizes, summary
    if not view_paths or not candidate_faces_set:
        summary["reason"] = "no_fit_views_or_candidate_faces"
        return base_sizes, summary

    current_size = max(base_sizes)
    rng = np.random.default_rng(59)
    stride = max(0, int(policy_val_stride))
    total_samples = 0
    total_l1 = 0.0
    fit_view_count = 0
    skipped_policy_val_views = 0
    active_faces: set[int] = set()
    occupied_keys: set[int] = set()
    bins_per_face = int(current_size) * int(current_size)

    for view_index, path in enumerate(tqdm(view_paths, desc="adaptive texture-size ladder")):
        if stride > 1 and view_index % stride == 0:
            skipped_policy_val_views += 1
            continue
        fit_view_count += 1
        with np.load(path) as z:
            if residual_l1_key not in z or "face_id" not in z or "barycentric" not in z:
                continue
            mask = _valid_sample_mask(z, candidate_faces_set, residual_l1_key, min_l1, min_alpha)
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
            if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
                take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
                local_mask = np.zeros_like(mask, dtype=bool)
                local_mask[ys[take], xs[take]] = True
                mask = local_mask
            face_ids = np.asarray(z["face_id"], dtype=np.int64)[mask]
            if face_ids.size == 0:
                continue
            l1 = np.asarray(z[residual_l1_key], dtype=np.float32)[mask]
            bin_mask = mask
            total_samples += int(face_ids.size)
            total_l1 += float(np.sum(l1.astype(np.float64)))
            active_faces.update(int(face) for face in np.unique(face_ids) if int(face) >= 0)
            bary = np.asarray(z["barycentric"], dtype=np.float32)
            ubin, vbin = _uv_bins(bary, bin_mask, int(current_size))
            bin_face_ids = np.asarray(z["face_id"], dtype=np.int64)[bin_mask]
            keys = bin_face_ids.astype(np.int64) * int(bins_per_face) + (
                vbin.astype(np.int64) * int(current_size) + ubin.astype(np.int64)
            )
            occupied_keys.update(int(key) for key in np.unique(keys))

    active_face_count = int(len(active_faces))
    mean_l1 = float(total_l1 / max(1, total_samples))
    samples_per_active_face = float(total_samples / max(1, active_face_count))
    samples_per_current_bin = float(total_samples / max(1, active_face_count * bins_per_face))
    occupied_bin_fraction = float(len(occupied_keys) / max(1, active_face_count * bins_per_face))
    summary.update(
        {
            "fit_view_count": int(fit_view_count),
            "skipped_policy_val_views": int(skipped_policy_val_views),
            "total_fit_samples": int(total_samples),
            "active_face_count": int(active_face_count),
            "current_texture_size": int(current_size),
            "current_bins_per_face": int(bins_per_face),
            "occupied_bin_count": int(len(occupied_keys)),
            "occupied_bin_fraction": float(occupied_bin_fraction),
            "samples_per_active_face": float(samples_per_active_face),
            "samples_per_current_bin": float(samples_per_current_bin),
            "mean_residual_l1": float(mean_l1),
            "min_fit_samples_per_face": float(min_fit_samples_per_face),
            "min_samples_per_current_bin": float(min_samples_per_current_bin),
            "min_mean_l1": float(min_mean_l1),
        }
    )
    reasons = []
    if total_samples <= 0 or active_face_count <= 0:
        reasons.append("no_active_train_fit_samples")
    if samples_per_active_face < float(min_fit_samples_per_face):
        reasons.append("insufficient_samples_per_active_face")
    if samples_per_current_bin < float(min_samples_per_current_bin):
        reasons.append("insufficient_samples_per_current_bin")
    if mean_l1 < float(min_mean_l1):
        reasons.append("mean_residual_l1_below_threshold")

    planned = list(base_sizes)
    added: list[int] = []
    rejected_sizes: list[dict[str, Any]] = []
    if not reasons:
        next_size = current_size * 2
        while next_size <= max_size:
            projected_bins_per_face = int(next_size) * int(next_size)
            projected_samples_per_bin = float(
                total_samples / max(1, active_face_count * projected_bins_per_face)
            )
            decision = {
                "texture_size": int(next_size),
                "projected_bins_per_face": int(projected_bins_per_face),
                "projected_samples_per_bin": float(projected_samples_per_bin),
                "min_samples_per_bin": float(min_samples_per_current_bin),
            }
            if projected_samples_per_bin < float(min_samples_per_current_bin):
                decision["accepted"] = False
                decision["reason"] = "projected_samples_per_bin_below_threshold"
                rejected_sizes.append(decision)
                break
            decision["accepted"] = True
            decision["reason"] = "projected_density_passed"
            if next_size not in planned:
                planned.append(int(next_size))
                added.append(int(next_size))
            next_size *= 2
        summary["reason"] = (
            "expanded_high_density_train_fit_residual_capacity"
            if added
            else "not_expanded_no_projected_size_met_density"
        )
    else:
        summary["reason"] = "not_expanded_" + ",".join(reasons)
    summary["added_texture_size_candidates"] = list(added)
    summary["rejected_texture_size_candidates"] = rejected_sizes
    summary["planned_texture_size_candidates"] = sorted(set(planned))
    return sorted(set(planned)), summary


def load_carrier_faces(path: Path, max_carriers: int, max_faces_per_carrier: int, max_faces: int) -> tuple[set[int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    carriers = list(payload.get("carriers") or [])
    if max_carriers > 0:
        carriers = carriers[: int(max_carriers)]
    faces: list[int] = []
    carrier_rows = []
    for carrier in carriers:
        raw = carrier.get("face_ids")
        if raw is None:
            raw = [row.get("face_id") for row in carrier.get("faces", []) if "face_id" in row]
        raw_faces = [int(x) for x in raw if int(x) >= 0]
        if max_faces_per_carrier > 0:
            raw_faces = raw_faces[: int(max_faces_per_carrier)]
        carrier_rows.append(
            {
                "carrier_id": carrier.get("carrier_id", len(carrier_rows)),
                "input_faces": int(len(raw_faces)),
                "views": int(carrier.get("view_count", len(carrier.get("views", []) or []))),
                "pixels": int(carrier.get("pixels", 0)),
                "score": float(carrier.get("score", 0.0)),
            }
        )
        faces.extend(raw_faces)
        if max_faces > 0 and len(set(faces)) >= int(max_faces):
            break
    unique = []
    seen = set()
    for face in faces:
        if face not in seen:
            unique.append(face)
            seen.add(face)
        if max_faces > 0 and len(unique) >= int(max_faces):
            break
    return set(unique), {"carrier_count_used": len(carrier_rows), "carrier_preview": carrier_rows[:20]}


def rank_fit_residual_extra_faces(
    view_paths: list[Path],
    base_faces: set[int],
    residual_l1_key: str,
    policy_val_stride: int,
    min_l1: float,
    min_alpha: float,
    min_face_samples: int,
    min_mean_l1: float,
    max_samples_per_view: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank high-residual train-fit faces without looking at policy-val or test GT."""
    face_counts: dict[int, int] = {}
    face_l1_sums: dict[int, float] = {}
    rng = np.random.default_rng(23)
    stride = max(0, int(policy_val_stride))
    fit_view_count = 0
    skipped_policy_val_views = 0
    for view_index, path in enumerate(tqdm(view_paths, desc="expand support faces")):
        if stride > 1 and view_index % stride == 0:
            skipped_policy_val_views += 1
            continue
        fit_view_count += 1
        z = np.load(path)
        if residual_l1_key not in z:
            continue
        mask = _valid_sample_mask(z, set(), residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            local_mask = np.zeros_like(mask, dtype=bool)
            local_mask[ys, xs] = True
            mask = local_mask
        face_ids = np.asarray(z["face_id"], dtype=np.int64)[mask]
        l1 = np.asarray(z[residual_l1_key], dtype=np.float32)[mask]
        nonnegative = face_ids >= 0
        face_ids = face_ids[nonnegative]
        l1 = l1[nonnegative]
        if face_ids.size == 0:
            continue
        max_face_id = int(np.max(face_ids))
        view_counts = np.bincount(face_ids, minlength=max_face_id + 1)
        view_l1_sums = np.bincount(face_ids, weights=l1.astype(np.float64), minlength=max_face_id + 1)
        for face_int in np.nonzero(view_counts)[0]:
            if int(face_int) in base_faces:
                continue
            count = int(view_counts[int(face_int)])
            face_counts[int(face_int)] = face_counts.get(int(face_int), 0) + count
            face_l1_sums[int(face_int)] = face_l1_sums.get(int(face_int), 0.0) + float(view_l1_sums[int(face_int)])

    rows: list[dict[str, Any]] = []
    for face, count in face_counts.items():
        mean_l1 = face_l1_sums[face] / max(1, int(count))
        if int(count) < int(min_face_samples):
            continue
        if float(mean_l1) < float(min_mean_l1):
            continue
        # Prefer faces that are both consistently observed and visibly wrong.
        score = float(mean_l1) * math.log1p(float(count))
        rows.append(
            {
                "face_id": int(face),
                "samples": int(count),
                "mean_l1": float(mean_l1),
                "score": float(score),
            }
        )
    rows.sort(key=lambda row: (float(row["score"]), float(row["mean_l1"]), int(row["samples"])), reverse=True)
    return rows, {
        "enabled": True,
        "mode": "fit_residual_topk",
        "base_faces": int(len(base_faces)),
        "eligible_extra_faces": int(len(rows)),
        "fit_view_count": int(fit_view_count),
        "skipped_policy_val_views": int(skipped_policy_val_views),
        "min_face_samples": int(min_face_samples),
        "min_mean_l1": float(min_mean_l1),
        "rank_preview": rows[:20],
    }


def rank_target_footprint_residual_debt_faces(
    fit_view_paths: list[Path],
    target_view_paths: list[Path],
    base_faces: set[int],
    residual_rgb_key: str,
    residual_l1_key: str,
    policy_val_stride: int,
    min_l1: float,
    min_alpha: float,
    min_face_samples: int,
    min_mean_l1: float,
    max_samples_per_view: int,
    texture_size: int,
    target_footprint_max_views: int,
    target_footprint_match_level: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank support faces by train residual debt weighted by GT-free target footprint."""
    match_level = str(target_footprint_match_level)
    if match_level not in {"bin", "face"}:
        raise ValueError(f"unsupported target_footprint_match_level: {target_footprint_match_level}")
    bins_per_face = int(texture_size) * int(texture_size)
    bin_counts: dict[int, int] = {}
    bin_view_counts: dict[int, int] = {}
    bin_l1_sums: dict[int, float] = {}
    bin_l1_sq_sums: dict[int, float] = {}
    bin_rgb_signed_sums: dict[int, np.ndarray] = {}
    bin_rgb_abs_sums: dict[int, float] = {}
    rng = np.random.default_rng(29)
    stride = max(0, int(policy_val_stride))
    fit_view_count = 0
    skipped_policy_val_views = 0
    for view_index, path in enumerate(tqdm(fit_view_paths, desc="target-debt support faces")):
        if stride > 1 and view_index % stride == 0:
            skipped_policy_val_views += 1
            continue
        fit_view_count += 1
        z = np.load(path)
        if residual_l1_key not in z or "face_id" not in z or "barycentric" not in z:
            continue
        mask = _valid_sample_mask(z, set(), residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            local_mask = np.zeros_like(mask, dtype=bool)
            local_mask[ys, xs] = True
            mask = local_mask
        face_ids = np.asarray(z["face_id"], dtype=np.int64)[mask]
        l1 = np.asarray(z[residual_l1_key], dtype=np.float32)[mask].astype(np.float64)
        nonnegative = face_ids >= 0
        if not bool(np.any(nonnegative)):
            continue
        face_ids = face_ids[nonnegative]
        l1 = l1[nonnegative]
        bary = np.asarray(z["barycentric"], dtype=np.float32)
        ubin, vbin = _uv_bins(bary, mask, int(texture_size))
        ubin = ubin[nonnegative]
        vbin = vbin[nonnegative]
        bin_ids = vbin.astype(np.int64) * int(texture_size) + ubin.astype(np.int64)
        keys = face_ids.astype(np.int64) * int(bins_per_face) + bin_ids.astype(np.int64)
        rgb_samples = None
        if residual_rgb_key in z:
            rgb = np.asarray(z[residual_rgb_key], dtype=np.float32)
            rgb_samples = np.stack([rgb[0][mask], rgb[1][mask], rgb[2][mask]], axis=1).astype(np.float64)
            rgb_samples = rgb_samples[nonnegative]
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        grouped_counts = np.bincount(inverse, minlength=int(unique_keys.size))
        grouped_l1_sums = np.bincount(inverse, weights=l1, minlength=int(unique_keys.size))
        grouped_l1_sq_sums = np.bincount(inverse, weights=l1 * l1, minlength=int(unique_keys.size))
        grouped_rgb_sums = None
        grouped_rgb_abs_sums = None
        if rgb_samples is not None:
            grouped_rgb_sums = [
                np.bincount(inverse, weights=rgb_samples[:, channel], minlength=int(unique_keys.size))
                for channel in range(3)
            ]
            grouped_rgb_abs_sums = np.bincount(
                inverse,
                weights=np.sum(np.abs(rgb_samples), axis=1),
                minlength=int(unique_keys.size),
            )
        for idx, key_value in enumerate(unique_keys):
            key = int(key_value)
            count = int(grouped_counts[idx])
            if count <= 0:
                continue
            bin_counts[key] = bin_counts.get(key, 0) + count
            bin_view_counts[key] = bin_view_counts.get(key, 0) + 1
            bin_l1_sums[key] = bin_l1_sums.get(key, 0.0) + float(grouped_l1_sums[idx])
            bin_l1_sq_sums[key] = bin_l1_sq_sums.get(key, 0.0) + float(grouped_l1_sq_sums[idx])
            if grouped_rgb_sums is not None and grouped_rgb_abs_sums is not None:
                if key not in bin_rgb_signed_sums:
                    bin_rgb_signed_sums[key] = np.zeros(3, dtype=np.float64)
                bin_rgb_signed_sums[key] += np.asarray(
                    [float(grouped_rgb_sums[channel][idx]) for channel in range(3)],
                    dtype=np.float64,
                )
                bin_rgb_abs_sums[key] = bin_rgb_abs_sums.get(key, 0.0) + float(grouped_rgb_abs_sums[idx])

    train_candidate_faces: set[int] = set()
    bin_debt_rows: dict[int, dict[str, Any]] = {}
    for key, count in bin_counts.items():
        if int(count) < int(min_face_samples):
            continue
        face = int(key // int(bins_per_face))
        if face in base_faces:
            continue
        mean_l1 = float(bin_l1_sums[key] / max(1, int(count)))
        if mean_l1 < float(min_mean_l1):
            continue
        mean_sq = float(bin_l1_sq_sums[key] / max(1, int(count)))
        variance = max(0.0, mean_sq - mean_l1 * mean_l1)
        if key in bin_rgb_signed_sums:
            sign_consistency = float(
                np.clip(
                    np.sum(np.abs(bin_rgb_signed_sums[key])) / max(float(bin_rgb_abs_sums.get(key, 0.0)), 1.0e-12),
                    0.0,
                    1.0,
                )
            )
        else:
            sign_consistency = 1.0
        view_factor = math.sqrt(float(max(1, int(bin_view_counts.get(key, 0)))))
        debt = (
            mean_l1
            * math.log1p(float(count))
            * view_factor
            * max(0.05, float(sign_consistency))
            / (math.sqrt(float(variance)) + 1.0e-3)
        )
        train_candidate_faces.add(face)
        bin_debt_rows[int(key)] = {
            "face_id": int(face),
            "bin_id": int(key % int(bins_per_face)),
            "samples": int(count),
            "views": int(bin_view_counts.get(key, 0)),
            "mean_l1": float(mean_l1),
            "variance": float(variance),
            "sign_consistency": float(sign_consistency),
            "debt": float(debt),
        }

    target_samples_by_bin, target_views_by_bin, target_summary = build_target_bin_footprint_stats(
        list(target_view_paths),
        candidate_faces=set(train_candidate_faces),
        texture_size=int(texture_size),
        min_alpha=float(min_alpha),
        max_views=int(target_footprint_max_views),
    )
    target_views_examined = int(target_summary.get("views_examined", 0) or len(target_view_paths))

    if match_level == "face":
        face_target_pixels: dict[int, int] = {}
        face_target_views: dict[int, int] = {}
        for key, pixels in target_samples_by_bin.items():
            face = int(int(key) // int(bins_per_face))
            face_target_pixels[face] = face_target_pixels.get(face, 0) + int(pixels)
            face_target_views[face] = max(face_target_views.get(face, 0), int(target_views_by_bin.get(int(key), 0)))

        face_train_debt: dict[int, float] = {}
        face_train_samples: dict[int, int] = {}
        face_train_l1_weighted: dict[int, float] = {}
        face_top_bin: dict[int, dict[str, Any]] = {}
        for debt_row in bin_debt_rows.values():
            face = int(debt_row["face_id"])
            samples = int(debt_row["samples"])
            debt = float(debt_row["debt"])
            face_train_debt[face] = face_train_debt.get(face, 0.0) + debt
            face_train_samples[face] = face_train_samples.get(face, 0) + samples
            face_train_l1_weighted[face] = face_train_l1_weighted.get(face, 0.0) + float(debt_row["mean_l1"]) * samples
            top_bin = dict(face_top_bin.get(face, {}))
            if not top_bin or debt > float(top_bin.get("debt", -1.0)):
                face_top_bin[face] = dict(debt_row)

        rows: list[dict[str, Any]] = []
        for face, train_debt in face_train_debt.items():
            target_pixels = int(face_target_pixels.get(int(face), 0))
            if target_pixels <= 0:
                continue
            target_views = int(face_target_views.get(int(face), 0))
            target_view_fraction = float(target_views / max(1, int(target_views_examined)))
            score = float(train_debt) * math.log1p(float(target_pixels)) * max(0.05, float(target_view_fraction))
            samples = int(face_train_samples.get(int(face), 0))
            top_bin = dict(face_top_bin.get(int(face), {}))
            top_bin["target_pixels"] = int(target_pixels)
            top_bin["target_views"] = int(target_views)
            top_bin["target_view_fraction"] = float(target_view_fraction)
            top_bin["score"] = float(score)
            rows.append(
                {
                    "face_id": int(face),
                    "samples": int(samples),
                    "mean_l1": float(face_train_l1_weighted.get(int(face), 0.0) / max(1, samples)),
                    "target_pixels": int(target_pixels),
                    "target_views": int(target_views),
                    "target_view_fraction": float(target_view_fraction),
                    "score": float(score),
                    "top_bin": top_bin,
                }
            )
        rows.sort(
            key=lambda row: (
                float(row["score"]),
                int(row["target_pixels"]),
                float(row["mean_l1"]),
                int(row["samples"]),
            ),
            reverse=True,
        )
        return rows, {
            "enabled": True,
            "mode": "target_footprint_residual_debt",
            "match_level": str(match_level),
            "base_faces": int(len(base_faces)),
            "eligible_extra_faces": int(len(rows)),
            "train_debt_bins": int(len(bin_debt_rows)),
            "train_debt_faces": int(len(train_candidate_faces)),
            "fit_view_count": int(fit_view_count),
            "skipped_policy_val_views": int(skipped_policy_val_views),
            "target_footprint": dict(target_summary),
            "target_covered_faces": int(len(face_target_pixels)),
            "min_face_samples": int(min_face_samples),
            "min_mean_l1": float(min_mean_l1),
            "texture_size": int(texture_size),
            "rank_preview": rows[:20],
        }

    face_scores: dict[int, float] = {}
    face_target_pixels: dict[int, int] = {}
    face_target_views: dict[int, int] = {}
    face_train_samples: dict[int, int] = {}
    face_train_l1_weighted: dict[int, float] = {}
    face_top_bin: dict[int, dict[str, Any]] = {}
    for key, debt_row in bin_debt_rows.items():
        target_pixels = int(target_samples_by_bin.get(int(key), 0))
        if target_pixels <= 0:
            continue
        target_views = int(target_views_by_bin.get(int(key), 0))
        target_view_fraction = float(target_views / max(1, int(target_views_examined)))
        bin_score = (
            float(debt_row["debt"])
            * math.log1p(float(target_pixels))
            * max(0.05, float(target_view_fraction))
        )
        face = int(debt_row["face_id"])
        face_scores[face] = face_scores.get(face, 0.0) + float(bin_score)
        face_target_pixels[face] = face_target_pixels.get(face, 0) + int(target_pixels)
        face_target_views[face] = max(face_target_views.get(face, 0), int(target_views))
        samples = int(debt_row["samples"])
        face_train_samples[face] = face_train_samples.get(face, 0) + samples
        face_train_l1_weighted[face] = face_train_l1_weighted.get(face, 0.0) + float(debt_row["mean_l1"]) * samples
        top_bin = dict(face_top_bin.get(face, {}))
        if not top_bin or float(bin_score) > float(top_bin.get("score", -1.0)):
            top_bin = dict(debt_row)
            top_bin["target_pixels"] = int(target_pixels)
            top_bin["target_views"] = int(target_views)
            top_bin["target_view_fraction"] = float(target_view_fraction)
            top_bin["score"] = float(bin_score)
            face_top_bin[face] = top_bin

    rows: list[dict[str, Any]] = []
    for face, score in face_scores.items():
        samples = int(face_train_samples.get(face, 0))
        rows.append(
            {
                "face_id": int(face),
                "samples": int(samples),
                "mean_l1": float(face_train_l1_weighted.get(face, 0.0) / max(1, samples)),
                "target_pixels": int(face_target_pixels.get(face, 0)),
                "target_views": int(face_target_views.get(face, 0)),
                "target_view_fraction": float(face_target_views.get(face, 0) / max(1, int(target_views_examined))),
                "score": float(score),
                "top_bin": dict(face_top_bin.get(face, {})),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["score"]),
            int(row["target_pixels"]),
            float(row["mean_l1"]),
            int(row["samples"]),
        ),
        reverse=True,
    )
    return rows, {
        "enabled": True,
        "mode": "target_footprint_residual_debt",
        "match_level": str(match_level),
        "base_faces": int(len(base_faces)),
        "eligible_extra_faces": int(len(rows)),
        "train_debt_bins": int(len(bin_debt_rows)),
        "train_debt_faces": int(len(train_candidate_faces)),
        "fit_view_count": int(fit_view_count),
        "skipped_policy_val_views": int(skipped_policy_val_views),
        "target_footprint": dict(target_summary),
        "min_face_samples": int(min_face_samples),
        "min_mean_l1": float(min_mean_l1),
        "texture_size": int(texture_size),
        "rank_preview": rows[:20],
    }


def expanded_candidate_faces_from_ranked_rows(
    base_faces: set[int],
    ranked_rows: list[dict[str, Any]],
    rank_summary: dict[str, Any],
    max_extra_faces: int,
) -> tuple[set[int], dict[str, Any]]:
    mode = str(rank_summary.get("mode", "fit_residual_topk"))
    if int(max_extra_faces) <= 0:
        return set(base_faces), {
            "enabled": False,
            "reason": "max_extra_faces <= 0",
            "mode": mode,
            "base_faces": int(len(base_faces)),
            "added_faces": 0,
            "candidate_faces_after_expansion": int(len(base_faces)),
            "eligible_extra_faces": int(rank_summary.get("eligible_extra_faces", len(ranked_rows))),
            "max_extra_faces": int(max_extra_faces),
            "preview": [],
        }
    selected = ranked_rows[: int(max_extra_faces)]
    expanded = set(base_faces)
    expanded.update(int(row["face_id"]) for row in selected)
    summary = {
        "enabled": True,
        "mode": mode,
        "base_faces": int(len(base_faces)),
        "added_faces": int(len(selected)),
        "candidate_faces_after_expansion": int(len(expanded)),
        "eligible_extra_faces": int(rank_summary.get("eligible_extra_faces", len(ranked_rows))),
        "train_debt_bins": int(rank_summary.get("train_debt_bins", 0)),
        "train_debt_faces": int(rank_summary.get("train_debt_faces", 0)),
        "fit_view_count": int(rank_summary.get("fit_view_count", 0)),
        "skipped_policy_val_views": int(rank_summary.get("skipped_policy_val_views", 0)),
        "target_footprint": dict(rank_summary.get("target_footprint", {})),
        "min_face_samples": int(rank_summary.get("min_face_samples", 0)),
        "min_mean_l1": float(rank_summary.get("min_mean_l1", 0.0)),
        "max_extra_faces": int(max_extra_faces),
        "preview": selected[:20],
    }
    for key in ("match_level", "target_covered_faces", "texture_size"):
        if key in rank_summary:
            summary[key] = rank_summary[key]
    return expanded, summary


def expand_candidate_faces_from_fit_residuals(
    view_paths: list[Path],
    base_faces: set[int],
    residual_l1_key: str,
    policy_val_stride: int,
    min_l1: float,
    min_alpha: float,
    min_face_samples: int,
    min_mean_l1: float,
    max_extra_faces: int,
    max_samples_per_view: int,
) -> tuple[set[int], dict[str, Any]]:
    """Add high-residual train-fit faces without looking at policy-val or test GT."""
    if int(max_extra_faces) <= 0:
        return set(base_faces), {
            "enabled": False,
            "reason": "max_extra_faces <= 0",
            "base_faces": int(len(base_faces)),
            "added_faces": 0,
        }
    ranked_rows, rank_summary = rank_fit_residual_extra_faces(
        view_paths,
        base_faces=base_faces,
        residual_l1_key=residual_l1_key,
        policy_val_stride=policy_val_stride,
        min_l1=min_l1,
        min_alpha=min_alpha,
        min_face_samples=min_face_samples,
        min_mean_l1=min_mean_l1,
        max_samples_per_view=max_samples_per_view,
    )
    return expanded_candidate_faces_from_ranked_rows(
        base_faces=base_faces,
        ranked_rows=ranked_rows,
        rank_summary=rank_summary,
        max_extra_faces=max_extra_faces,
    )


def copy_model_shell(source_model: Path, output_model: Path, force: bool) -> None:
    if output_model.exists():
        if not force:
            return
        shutil.rmtree(output_model)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in IGNORE_COPY_NAMES}

    def copy_or_link_large_model_file(src: str, dst: str) -> str:
        src_path = Path(src)
        dst_path = Path(dst)
        size = src_path.stat().st_size
        if src_path.suffix in {".pt", ".pth"} and size >= 256 * 1024 * 1024:
            dst_path.symlink_to(src_path.resolve())
            return str(dst_path)
        try:
            return shutil.copy2(src, dst)
        except OSError as exc:
            if exc.errno != 122 or src_path.suffix not in {".pt", ".pth"}:
                raise
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            dst_path.symlink_to(src_path.resolve())
            return str(dst_path)

    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_model, output_model, ignore=ignore, copy_function=copy_or_link_large_model_file)


def _valid_sample_mask(
    z: np.lib.npyio.NpzFile,
    candidate_faces: set[int],
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
) -> np.ndarray:
    face_id = np.asarray(z["face_id"], dtype=np.int64)
    if candidate_faces:
        mask = np.isin(face_id, np.fromiter(candidate_faces, dtype=np.int64))
    else:
        mask = face_id >= 0
    mask &= face_id >= 0
    if "barycentric_valid" in z:
        mask &= np.asarray(z["barycentric_valid"]).astype(bool)
    if residual_l1_key in z:
        mask &= np.asarray(z[residual_l1_key], dtype=np.float32) >= float(min_l1)
    if "alpha" in z:
        mask &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
    bary = np.asarray(z["barycentric"], dtype=np.float32)
    mask &= np.all(np.isfinite(bary), axis=0)
    mask &= np.all(bary >= -0.05, axis=0)
    mask &= np.all(bary <= 1.05, axis=0)
    return mask


def _uv_bins(bary: np.ndarray, mask: np.ndarray, texture_size: int) -> tuple[np.ndarray, np.ndarray]:
    # Use two barycentric coordinates as a local triangle chart.
    u = np.clip(bary[1][mask], 0.0, 0.999999)
    v = np.clip(bary[2][mask], 0.0, 0.999999)
    size = int(texture_size)
    return (u * size).astype(np.int32), (v * size).astype(np.int32)


def build_target_bin_footprint_stats(
    target_views: list[Path],
    candidate_faces: set[int],
    texture_size: int,
    min_alpha: float,
    max_views: int = 0,
) -> tuple[dict[int, int], dict[int, int], dict[str, Any]]:
    """Count GT-free target-view coverage for face/UV bins.

    This uses target geometry/evidence only: face ids, barycentric coordinates,
    optional barycentric validity, and alpha. It never reads target RGB GT.
    """
    disabled = {
        "enabled": False,
        "mode": "target_footprint_bin_certificate",
        "reason": "",
    }
    if not target_views or not candidate_faces:
        disabled["reason"] = "no_target_views_or_candidate_faces"
        return {}, {}, disabled

    views = list(target_views)
    if int(max_views) > 0:
        views = views[: int(max_views)]

    samples_by_key: dict[int, int] = {}
    view_count_by_key: dict[int, int] = {}
    bins_per_face = int(texture_size) * int(texture_size)
    total_valid_pixels = 0
    views_examined = 0
    views_with_target_coverage = 0
    candidate_faces_arr = np.fromiter(candidate_faces, dtype=np.int64)

    for path in tqdm(views, desc="target footprint bin certificate"):
        z = np.load(path)
        if "face_id" not in z or "barycentric" not in z:
            continue
        face_id = np.asarray(z["face_id"], dtype=np.int64)
        mask = face_id >= 0
        if candidate_faces_arr.size > 0:
            mask &= np.isin(face_id, candidate_faces_arr)
        if "barycentric_valid" in z:
            mask &= np.asarray(z["barycentric"]).shape[0] >= 3
            mask &= np.asarray(z["barycentric_valid"]).astype(bool)
        if "alpha" in z:
            mask &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
        bary = np.asarray(z["barycentric"], dtype=np.float32)
        if bary.ndim != 3 or bary.shape[0] < 3:
            continue
        views_examined += 1
        mask &= np.all(np.isfinite(bary), axis=0)
        mask &= np.all(bary >= -0.05, axis=0)
        mask &= np.all(bary <= 1.05, axis=0)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        views_with_target_coverage += 1
        ubin, vbin = _uv_bins(bary, mask, int(texture_size))
        face_samples = face_id[mask]
        bin_ids = vbin.astype(np.int64) * int(texture_size) + ubin.astype(np.int64)
        keys = face_samples.astype(np.int64) * int(bins_per_face) + bin_ids.astype(np.int64)
        unique_keys, counts = np.unique(keys, return_counts=True)
        for key, count in zip(unique_keys, counts, strict=False):
            key_i = int(key)
            samples_by_key[key_i] = samples_by_key.get(key_i, 0) + int(count)
            view_count_by_key[key_i] = view_count_by_key.get(key_i, 0) + 1
        total_valid_pixels += int(ys.size)

    if not samples_by_key:
        disabled["reason"] = "no_target_covered_bins"
        disabled["target_views_requested"] = int(len(target_views))
        disabled["target_views_used"] = int(views_examined)
        disabled["target_views_examined"] = int(views_examined)
        disabled["views_with_target_coverage"] = int(views_with_target_coverage)
        disabled["total_valid_pixels"] = int(total_valid_pixels)
        return {}, {}, disabled

    ranked = sorted(samples_by_key, key=lambda key: (samples_by_key[key], view_count_by_key[key]), reverse=True)
    summary = {
        "enabled": True,
        "mode": "target_footprint_bin_certificate",
        "texture_size": int(texture_size),
        "target_views_requested": int(len(target_views)),
        "target_views_used": int(views_examined),
        "target_views_examined": int(views_examined),
        "views_with_target_coverage": int(views_with_target_coverage),
        "max_views": int(max_views),
        "candidate_faces": int(len(candidate_faces)),
        "covered_bin_count": int(len(samples_by_key)),
        "total_valid_pixels": int(total_valid_pixels),
        "top_bins": [
            {
                "face": int(key // bins_per_face),
                "bin": int(key % bins_per_face),
                "target_pixels": int(samples_by_key[key]),
                "target_views": int(view_count_by_key[key]),
            }
            for key in ranked[:128]
        ],
    }
    return samples_by_key, view_count_by_key, summary


def geometry_face_mask(z: np.lib.npyio.NpzFile, min_alpha: float) -> np.ndarray:
    face_id = np.asarray(z["face_id"], dtype=np.int64)
    mask = face_id >= 0
    if "barycentric_valid" in z:
        mask &= np.asarray(z["barycentric_valid"]).astype(bool)
    if "alpha" in z:
        mask &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
    if "barycentric" in z:
        bary = np.asarray(z["barycentric"], dtype=np.float32)
        if bary.ndim == 3 and bary.shape[0] >= 3:
            mask &= np.all(np.isfinite(bary), axis=0)
            mask &= np.all(bary >= -0.05, axis=0)
            mask &= np.all(bary <= 1.05, axis=0)
    return mask


def build_face_footprint_stats(
    view_paths: list[Path],
    *,
    min_alpha: float,
    max_views: int = 0,
) -> tuple[dict[int, int], dict[int, int], dict[str, Any]]:
    views = list(view_paths)
    if int(max_views) > 0:
        views = views[: int(max_views)]
    samples_by_face: dict[int, int] = {}
    views_by_face: dict[int, int] = {}
    views_examined = 0
    total_valid_pixels = 0
    for path in views:
        z = np.load(path)
        if "face_id" not in z:
            continue
        face_id = np.asarray(z["face_id"], dtype=np.int64)
        mask = geometry_face_mask(z, float(min_alpha))
        views_examined += 1
        if not bool(np.any(mask)):
            continue
        faces, counts = np.unique(face_id[mask], return_counts=True)
        for face, count in zip(faces, counts, strict=False):
            face_i = int(face)
            if face_i < 0:
                continue
            samples_by_face[face_i] = samples_by_face.get(face_i, 0) + int(count)
            views_by_face[face_i] = views_by_face.get(face_i, 0) + 1
            total_valid_pixels += int(count)
    top_faces = sorted(samples_by_face, key=lambda face: samples_by_face[face], reverse=True)[:128]
    return samples_by_face, views_by_face, {
        "enabled": True,
        "mode": "face_footprint_stats",
        "views_requested": int(len(view_paths)),
        "views_examined": int(views_examined),
        "max_views": int(max_views),
        "covered_faces": int(len(samples_by_face)),
        "total_valid_pixels": int(total_valid_pixels),
        "top_faces": [
            {
                "face": int(face),
                "pixels": int(samples_by_face[face]),
                "views": int(views_by_face.get(face, 0)),
            }
            for face in top_faces
        ],
    }


def build_coview_face_residual_transfer_plan(
    fit_view_paths: list[Path],
    target_view_paths: list[Path],
    *,
    base_faces: set[int],
    residual_rgb_key: str,
    residual_l1_key: str,
    policy_val_stride: int,
    min_l1: float,
    min_alpha: float,
    max_samples_per_view: int,
    neighbor_stride: int,
    min_source_samples: int,
    min_source_mean_l1: float,
    min_edge_count: int,
    min_target_pixels: int,
    min_policy_val_pixels: int,
    max_faces: int,
    max_views: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Plan train-only residual transfer through image-space co-visible face adjacency."""
    stride = max(0, int(policy_val_stride))
    neighbor_stride = max(1, int(neighbor_stride))
    rng = np.random.default_rng(37)
    fit_paths: list[Path] = []
    policy_val_paths: list[Path] = []
    source_counts: dict[int, int] = {}
    source_l1_sums: dict[int, float] = {}
    source_rgb_sums: dict[int, np.ndarray] = {}
    adjacency: dict[int, dict[int, int]] = {}

    def add_edge(a: int, b: int, count: int) -> None:
        if a < 0 or b < 0 or a == b or count <= 0:
            return
        adjacency.setdefault(int(a), {})[int(b)] = adjacency.setdefault(int(a), {}).get(int(b), 0) + int(count)
        adjacency.setdefault(int(b), {})[int(a)] = adjacency.setdefault(int(b), {}).get(int(a), 0) + int(count)

    for view_index, path in enumerate(tqdm(fit_view_paths, desc="coview transfer graph")):
        if stride > 1 and view_index % stride == 0:
            policy_val_paths.append(path)
            continue
        fit_paths.append(path)
        z = np.load(path)
        if "face_id" not in z:
            continue
        face_id = np.asarray(z["face_id"], dtype=np.int64)
        geom_mask = geometry_face_mask(z, float(min_alpha))
        sampled_faces = face_id[::neighbor_stride, ::neighbor_stride]
        sampled_mask = geom_mask[::neighbor_stride, ::neighbor_stride]
        for lhs, rhs in (
            (sampled_faces[:, :-1], sampled_faces[:, 1:]),
            (sampled_faces[:-1, :], sampled_faces[1:, :]),
        ):
            if lhs.size == 0 or rhs.size == 0:
                continue
            if lhs.shape[0] == sampled_mask.shape[0]:
                mask_a = sampled_mask[:, :-1]
                mask_b = sampled_mask[:, 1:]
            else:
                mask_a = sampled_mask[:-1, :]
                mask_b = sampled_mask[1:, :]
            pair_mask = mask_a & mask_b & (lhs >= 0) & (rhs >= 0) & (lhs != rhs)
            if not bool(np.any(pair_mask)):
                continue
            a = lhs[pair_mask].astype(np.int64)
            b = rhs[pair_mask].astype(np.int64)
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
            pairs = np.stack([lo, hi], axis=1)
            unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
            for pair, count in zip(unique_pairs, counts, strict=False):
                add_edge(int(pair[0]), int(pair[1]), int(count))

        if residual_l1_key not in z or residual_rgb_key not in z:
            continue
        residual_mask = _valid_sample_mask(z, set(), residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(residual_mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            local_mask = np.zeros_like(residual_mask, dtype=bool)
            local_mask[ys[take], xs[take]] = True
            residual_mask = local_mask
        faces = face_id[residual_mask]
        l1 = np.asarray(z[residual_l1_key], dtype=np.float32)[residual_mask].astype(np.float64)
        residual = np.asarray(z[residual_rgb_key], dtype=np.float32)
        rgb = np.stack(
            [residual[0][residual_mask], residual[1][residual_mask], residual[2][residual_mask]],
            axis=1,
        ).astype(np.float64)
        valid = faces >= 0
        faces = faces[valid]
        l1 = l1[valid]
        rgb = rgb[valid]
        if faces.size == 0:
            continue
        unique_faces, inverse = np.unique(faces, return_inverse=True)
        counts = np.bincount(inverse, minlength=int(unique_faces.size))
        l1_sums = np.bincount(inverse, weights=l1, minlength=int(unique_faces.size))
        rgb_sums = [
            np.bincount(inverse, weights=rgb[:, channel], minlength=int(unique_faces.size))
            for channel in range(3)
        ]
        for idx, face in enumerate(unique_faces):
            face_i = int(face)
            count = int(counts[idx])
            if count <= 0:
                continue
            source_counts[face_i] = source_counts.get(face_i, 0) + count
            source_l1_sums[face_i] = source_l1_sums.get(face_i, 0.0) + float(l1_sums[idx])
            if face_i not in source_rgb_sums:
                source_rgb_sums[face_i] = np.zeros(3, dtype=np.float64)
            source_rgb_sums[face_i] += np.asarray(
                [float(rgb_sums[channel][idx]) for channel in range(3)],
                dtype=np.float64,
            )

    target_pixels, target_views, target_summary = build_face_footprint_stats(
        list(target_view_paths),
        min_alpha=float(min_alpha),
        max_views=int(max_views),
    )
    policy_pixels, policy_views, policy_summary = build_face_footprint_stats(
        policy_val_paths,
        min_alpha=float(min_alpha),
        max_views=int(max_views),
    )

    source_rows: dict[int, dict[str, Any]] = {}
    for face, count in source_counts.items():
        if int(count) < int(min_source_samples):
            continue
        mean_l1 = float(source_l1_sums.get(face, 0.0) / max(1, int(count)))
        if mean_l1 < float(min_source_mean_l1):
            continue
        mean_rgb = source_rgb_sums[face] / max(1, int(count))
        source_rows[int(face)] = {
            "face_id": int(face),
            "samples": int(count),
            "mean_l1": float(mean_l1),
            "mean_rgb": [float(x) for x in mean_rgb.tolist()],
        }

    best_by_dest: dict[int, dict[str, Any]] = {}
    for source_face, source in source_rows.items():
        for dest_face, edge_count in adjacency.get(int(source_face), {}).items():
            if int(edge_count) < int(min_edge_count):
                continue
            if int(target_pixels.get(int(dest_face), 0)) < int(min_target_pixels):
                continue
            if int(policy_pixels.get(int(dest_face), 0)) < int(min_policy_val_pixels):
                continue
            source_mean_l1 = float(source["mean_l1"])
            score = (
                source_mean_l1
                * math.log1p(float(source["samples"]))
                * math.log1p(float(edge_count))
                * math.log1p(float(target_pixels.get(int(dest_face), 0)))
                * math.log1p(float(policy_pixels.get(int(dest_face), 0)))
            )
            row = {
                "face_id": int(dest_face),
                "source_face_id": int(source_face),
                "score": float(score),
                "edge_count": int(edge_count),
                "source_samples": int(source["samples"]),
                "source_mean_l1": float(source_mean_l1),
                "source_mean_rgb": list(source["mean_rgb"]),
                "target_pixels": int(target_pixels.get(int(dest_face), 0)),
                "target_views": int(target_views.get(int(dest_face), 0)),
                "policy_val_pixels": int(policy_pixels.get(int(dest_face), 0)),
                "policy_val_views": int(policy_views.get(int(dest_face), 0)),
                "already_base_face": bool(int(dest_face) in base_faces),
            }
            previous = best_by_dest.get(int(dest_face))
            if previous is None or float(score) > float(previous.get("score", -1.0)):
                best_by_dest[int(dest_face)] = row
    rows = sorted(
        best_by_dest.values(),
        key=lambda row: (
            float(row["score"]),
            int(row["target_pixels"]),
            int(row["policy_val_pixels"]),
            float(row["source_mean_l1"]),
        ),
        reverse=True,
    )
    if int(max_faces) > 0:
        rows = rows[: int(max_faces)]
    return rows, {
        "enabled": True,
        "mode": "coview_face_residual_transfer",
        "fit_view_count": int(len(fit_paths)),
        "policy_val_view_count": int(len(policy_val_paths)),
        "source_face_count": int(len(source_rows)),
        "adjacency_source_face_count": int(len(adjacency)),
        "eligible_transfer_faces": int(len(best_by_dest)),
        "selected_transfer_faces": int(len(rows)),
        "base_faces": int(len(base_faces)),
        "min_source_samples": int(min_source_samples),
        "min_source_mean_l1": float(min_source_mean_l1),
        "min_edge_count": int(min_edge_count),
        "min_target_pixels": int(min_target_pixels),
        "min_policy_val_pixels": int(min_policy_val_pixels),
        "neighbor_stride": int(neighbor_stride),
        "target_footprint": dict(target_summary),
        "policy_val_footprint": dict(policy_summary),
        "rank_preview": rows[:64],
    }


def apply_coview_face_residual_transfer(
    atlas: dict[int, FaceAtlas],
    transfer_rows: list[dict[str, Any]],
    *,
    texture_size: int,
    residual_scale: float,
    max_abs_delta_rgb: float,
    synthetic_count: int,
    existing_atlas_mode: str,
    blend_max_direct_bin_count: int,
) -> dict[str, Any]:
    applied_rows: list[dict[str, Any]] = []
    skipped_existing = 0
    skipped_zero = 0
    skipped_no_blend_bins = 0
    overwritten_existing = 0
    blended_existing = 0
    created_new = 0
    size = int(texture_size)
    mode = str(existing_atlas_mode)
    max_blend_count = int(blend_max_direct_bin_count)
    if mode not in {"skip", "overwrite", "blend"}:
        raise ValueError(f"unknown coview existing atlas mode: {mode}")
    for row in transfer_rows:
        face = int(row.get("face_id", -1))
        if face < 0:
            continue
        has_existing = face in atlas
        if has_existing and mode == "skip":
            skipped_existing += 1
            continue
        rgb = np.asarray(row.get("source_mean_rgb", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
        delta = clip_delta_rgb(rgb * float(residual_scale), float(max_abs_delta_rgb)).astype(np.float32)
        if float(np.max(np.abs(delta))) <= 0.0:
            skipped_zero += 1
            continue
        count_value = max(0, int(synthetic_count))
        synthetic_texture = np.repeat(delta.reshape(1, 1, 3), size, axis=0)
        synthetic_texture = np.repeat(synthetic_texture, size, axis=1).astype(np.float32)
        synthetic_counts = np.full((size, size), count_value, dtype=np.int32)
        if has_existing and mode == "blend":
            existing = atlas[face]
            texture = np.asarray(existing.texture, dtype=np.float32)
            counts = np.asarray(existing.counts, dtype=np.int32)
            variance = np.asarray(existing.variance, dtype=np.float32)
            sign_consistency = np.asarray(existing.sign_consistency, dtype=np.float32)
            if texture.shape[:2] != (size, size):
                skipped_existing += 1
                continue
            eligible_bins = np.ones((size, size), dtype=bool)
            if max_blend_count >= 0:
                eligible_bins = counts <= max_blend_count
            if not bool(np.any(eligible_bins)):
                skipped_no_blend_bins += 1
                continue
            blend_weight = float(count_value) / (np.asarray(counts, dtype=np.float32) + float(count_value) + 1.0e-6)
            blend_weight = np.clip(blend_weight, 0.0, 1.0).astype(np.float32)
            blend_weight = np.where(eligible_bins, blend_weight, 0.0).astype(np.float32)
            texture = ((1.0 - blend_weight[..., None]) * texture) + (blend_weight[..., None] * synthetic_texture)
            counts = np.asarray(counts + np.where(eligible_bins, synthetic_counts, 0), dtype=np.int32)
            mean_rgb = np.mean(texture.reshape(-1, 3), axis=0).astype(np.float32)
            atlas[face] = FaceAtlas(
                texture=texture.astype(np.float32),
                counts=counts,
                variance=variance,
                sign_consistency=sign_consistency,
                mean_rgb=mean_rgb,
                samples=int(existing.samples) + int(count_value * size * size),
                view_basis_mode=str(existing.view_basis_mode),
                view_basis_coefficients=existing.view_basis_coefficients,
                view_basis_support=existing.view_basis_support,
                view_basis_feature_mean=existing.view_basis_feature_mean,
                view_basis_feature_std=existing.view_basis_feature_std,
                view_basis_ood_mode=str(existing.view_basis_ood_mode),
                view_basis_ood_max_z=float(existing.view_basis_ood_max_z),
                view_basis_ood_min_std=float(existing.view_basis_ood_min_std),
                teacher_basis_mode=str(existing.teacher_basis_mode),
                teacher_basis_coefficients=existing.teacher_basis_coefficients,
                teacher_basis_feature_mean=existing.teacher_basis_feature_mean,
                teacher_basis_feature_std=existing.teacher_basis_feature_std,
                teacher_basis_ood_max_z=float(existing.teacher_basis_ood_max_z),
                teacher_basis_ood_min_std=float(existing.teacher_basis_ood_min_std),
                teacher_basis_apply_mode=str(existing.teacher_basis_apply_mode),
                teacher_basis_blend=float(existing.teacher_basis_blend),
            )
            blended_existing += 1
        else:
            atlas[face] = FaceAtlas(
                texture=synthetic_texture,
                counts=synthetic_counts,
                variance=np.zeros((size, size, 3), dtype=np.float32),
                sign_consistency=np.ones((size, size, 3), dtype=np.float32),
                mean_rgb=delta.astype(np.float32),
                samples=int(count_value * size * size),
                view_basis_mode="none",
                teacher_basis_mode="none",
            )
            if has_existing:
                overwritten_existing += 1
            else:
                created_new += 1
        applied = dict(row)
        applied["applied_mean_rgb"] = [float(x) for x in delta.tolist()]
        applied["synthetic_count"] = int(count_value)
        applied["existing_atlas_mode"] = str(mode)
        applied["had_existing_atlas"] = bool(has_existing)
        applied_rows.append(applied)
    return {
        "enabled": True,
        "mode": "coview_face_residual_transfer",
        "requested_rows": int(len(transfer_rows)),
        "applied_faces": int(len(applied_rows)),
        "created_new_faces": int(created_new),
        "overwritten_existing_atlas_faces": int(overwritten_existing),
        "blended_existing_atlas_faces": int(blended_existing),
        "skipped_existing_atlas_faces": int(skipped_existing),
        "skipped_zero_delta_faces": int(skipped_zero),
        "skipped_no_blend_bins_faces": int(skipped_no_blend_bins),
        "residual_scale": float(residual_scale),
        "synthetic_count": int(synthetic_count),
        "existing_atlas_mode": str(mode),
        "blend_max_direct_bin_count": int(max_blend_count),
        "skip_existing_atlas": bool(mode == "skip"),
        "applied_preview": applied_rows[:64],
    }


def _normalized_camera_center(z: np.lib.npyio.NpzFile) -> np.ndarray | None:
    if "camera_center" not in z:
        return None
    center = np.asarray(z["camera_center"], dtype=np.float32).reshape(-1)
    if center.size < 3 or not bool(np.all(np.isfinite(center[:3]))):
        return None
    direction = center[:3].astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm > 1.0e-8:
        direction = direction / norm
    else:
        direction = np.zeros((3,), dtype=np.float32)
    return direction.astype(np.float32)


def _view_cluster_feature_for_npz(
    z: np.lib.npyio.NpzFile,
    mode: str,
) -> np.ndarray | None:
    mode = str(mode)
    if mode == "none":
        return None
    if mode == "camera_center":
        return _normalized_camera_center(z)
    raise ValueError(f"unsupported view-cluster feature mode: {mode}")


def _view_cluster_feature_for_path(path: Path, mode: str) -> np.ndarray | None:
    try:
        with np.load(path) as z:
            return _view_cluster_feature_for_npz(z, mode)
    except Exception:
        return None


def _fit_view_cluster_profile(
    view_paths: list[Path],
    *,
    expert_count: int,
    feature_mode: str,
    min_views: int,
    iterations: int = 16,
) -> dict[str, Any]:
    feature_mode_s = str(feature_mode)
    k = max(1, int(expert_count))
    disabled = {
        "enabled": False,
        "feature_mode": feature_mode_s,
        "requested_expert_count": int(expert_count),
        "expert_count": 1,
        "min_views": int(min_views),
        "fit_view_count": int(len(view_paths)),
        "feature_view_count": 0,
        "reason": "",
    }
    if k <= 1 or feature_mode_s == "none":
        disabled["reason"] = "expert_count_le_1_or_feature_disabled"
        return disabled
    features: list[np.ndarray] = []
    used_paths: list[str] = []
    for path in view_paths:
        feature = _view_cluster_feature_for_path(path, feature_mode_s)
        if feature is None:
            continue
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        if feature.size != 3 or not bool(np.all(np.isfinite(feature))):
            continue
        norm = float(np.linalg.norm(feature))
        if norm > 1.0e-8:
            feature = feature / norm
        features.append(feature.astype(np.float32))
        used_paths.append(str(path))
    if len(features) < max(k, int(min_views)):
        disabled["reason"] = "insufficient_views_with_cluster_features"
        disabled["feature_view_count"] = int(len(features))
        return disabled
    x = np.stack(features, axis=0).astype(np.float32)
    centers: list[np.ndarray] = [x[0]]
    while len(centers) < k:
        center_arr = np.stack(centers, axis=0).astype(np.float32)
        sims = x @ center_arr.T
        farthest = int(np.argmin(np.max(sims, axis=1)))
        centers.append(x[farthest])
    center_arr = np.stack(centers, axis=0).astype(np.float32)
    assignments = np.zeros((int(x.shape[0]),), dtype=np.int64)
    for _ in range(max(1, int(iterations))):
        assignments = np.argmax(x @ center_arr.T, axis=1).astype(np.int64)
        next_centers = np.array(center_arr, copy=True)
        for idx in range(k):
            local = x[assignments == idx]
            if local.size == 0:
                continue
            center = np.mean(local, axis=0).astype(np.float32)
            norm = float(np.linalg.norm(center))
            next_centers[idx] = center / norm if norm > 1.0e-8 else center_arr[idx]
        if np.allclose(next_centers, center_arr, atol=1.0e-6):
            center_arr = next_centers
            break
        center_arr = next_centers
    assignments = np.argmax(x @ center_arr.T, axis=1).astype(np.int64)
    counts = np.bincount(assignments, minlength=k).astype(np.int64)
    valid_clusters = counts >= max(1, int(min_views))
    return {
        "enabled": True,
        "feature_mode": feature_mode_s,
        "requested_expert_count": int(expert_count),
        "expert_count": int(k),
        "min_views": int(min_views),
        "fit_view_count": int(len(view_paths)),
        "feature_view_count": int(len(features)),
        "centers": center_arr.astype(np.float32),
        "cluster_view_counts": [int(x) for x in counts.tolist()],
        "valid_clusters": [bool(x) for x in valid_clusters.tolist()],
        "valid_cluster_count": int(np.sum(valid_clusters)),
        "used_path_preview": used_paths[:16],
    }


def _assign_view_cluster_for_npz(
    z: np.lib.npyio.NpzFile,
    *,
    centers: np.ndarray | None,
    feature_mode: str,
) -> int | None:
    if centers is None:
        return None
    center_arr = np.asarray(centers, dtype=np.float32)
    if center_arr.ndim != 2 or center_arr.shape[0] <= 0:
        return None
    feature = _view_cluster_feature_for_npz(z, feature_mode)
    if feature is None:
        return None
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    if feature.size != center_arr.shape[1] or not bool(np.all(np.isfinite(feature))):
        return None
    norm = float(np.linalg.norm(feature))
    if norm > 1.0e-8:
        feature = feature / norm
    sims = center_arr @ feature.astype(np.float32)
    return int(np.argmax(sims))


def _view_cluster_profile_from_atlas(
    atlas: dict[int, FaceAtlas],
) -> tuple[np.ndarray | None, str]:
    for face_atlas in atlas.values():
        if (
            face_atlas.expert_centers is not None
            and str(face_atlas.expert_feature_mode) != "none"
        ):
            centers = np.asarray(face_atlas.expert_centers, dtype=np.float32)
            if centers.ndim == 2 and centers.shape[0] > 0:
                return centers, str(face_atlas.expert_feature_mode)
    return None, "none"


def _view_condition_feature_dim(mode: str) -> int:
    mode = str(mode)
    if mode == "none":
        return 0
    if mode == "camera_center_linear":
        return 4
    if mode == "normal_camera_linear":
        return 8
    raise ValueError(f"unsupported view-conditioned basis mode: {mode}")


def _view_condition_features_for_mask(
    z: np.lib.npyio.NpzFile,
    mode: str,
    mask: np.ndarray,
) -> np.ndarray | None:
    mode = str(mode)
    if mode == "none":
        return None
    direction = _normalized_camera_center(z)
    if direction is None:
        return None
    sample_count = int(np.count_nonzero(mask))
    if sample_count <= 0:
        return None
    if mode == "camera_center_linear":
        feature = np.concatenate([np.ones((1,), dtype=np.float32), direction], axis=0)
        return np.repeat(feature[None, :], sample_count, axis=0)
    if mode == "normal_camera_linear":
        if "normal" not in z:
            return None
        normal = np.asarray(z["normal"], dtype=np.float32)
        if normal.shape[0] != 3:
            return None
        normal_samples = np.stack([normal[0][mask], normal[1][mask], normal[2][mask]], axis=1)
        normal_samples = np.nan_to_num(normal_samples, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        normal_norm = np.linalg.norm(normal_samples, axis=1, keepdims=True)
        normal_samples = normal_samples / np.maximum(normal_norm, 1.0e-8)
        camera_features = np.repeat(direction[None, :], sample_count, axis=0)
        normal_dot_camera = np.sum(normal_samples * camera_features, axis=1, keepdims=True)
        return np.concatenate(
            [
                np.ones((sample_count, 1), dtype=np.float32),
                camera_features.astype(np.float32),
                normal_samples.astype(np.float32),
                normal_dot_camera.astype(np.float32),
            ],
            axis=1,
        )
    raise ValueError(f"unsupported view-conditioned basis mode: {mode}")


def _view_confidence_feature_for_npz(z: np.lib.npyio.NpzFile) -> np.ndarray | None:
    direction = _normalized_camera_center(z)
    if direction is None:
        return None
    return direction.astype(np.float32)


def _policy_val_rows_for_alpha(policy_val: dict[str, Any], alpha: float) -> list[dict[str, Any]]:
    per_view = policy_val.get("per_view_by_alpha", {})
    if not isinstance(per_view, dict):
        return []
    target_alpha = float(alpha)
    best_rows: list[dict[str, Any]] = []
    best_delta = float("inf")
    for key, rows in per_view.items():
        try:
            key_alpha = float(key)
        except (TypeError, ValueError):
            continue
        delta = abs(key_alpha - target_alpha)
        if delta < best_delta:
            best_delta = delta
            best_rows = list(rows or [])
    if math.isfinite(best_delta) and best_delta <= 1.0e-8:
        return best_rows
    return []


def _metric_passes_optional_threshold(value: Any, threshold: float) -> bool:
    threshold = float(threshold)
    if threshold <= -0.999:
        return True
    if value is None:
        return False
    try:
        return float(value) >= threshold
    except (TypeError, ValueError):
        return False


def sanitize_view_confidence_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict(profile or {}).items()
        if key not in {"positive_features", "negative_features"}
    }


def sanitize_view_alpha_cap_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict(profile or {}).items()
        if key not in {"anchor_features", "anchor_alpha_caps", "anchor_weights"}
    }


def build_policy_val_view_confidence_profile(
    val_views: list[Path],
    policy_val: dict[str, Any],
    alpha: float,
    *,
    enabled: bool,
    min_relative_gain: float,
    min_ssim_gain: float,
    min_l1_gain: float,
    min_lpips_gain: float,
    kernel_sigma: float,
    min_confidence: float,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "enabled": False,
        "mode": "policy_val_view_consistency_confidence",
        "decision": "not_requested",
    }
    if not bool(enabled):
        return profile
    profile.update(
        {
            "decision": "disabled",
            "alpha": float(alpha),
            "min_relative_gain": float(min_relative_gain),
            "min_ssim_gain": float(min_ssim_gain),
            "min_l1_gain": float(min_l1_gain),
            "min_lpips_gain": float(min_lpips_gain),
            "kernel_sigma": float(kernel_sigma),
            "min_confidence": float(min_confidence),
            "uses_policy_val_gt": True,
            "uses_target_gt": False,
            "feature": "normalized_camera_center",
        }
    )
    if not bool(policy_val.get("enabled", False)):
        profile["reason"] = "policy_val_disabled"
        return profile
    if float(alpha) <= 0.0:
        profile["reason"] = "zero_or_negative_alpha"
        return profile
    rows = _policy_val_rows_for_alpha(policy_val, float(alpha))
    if not rows:
        profile["reason"] = "no_policy_val_rows_for_selected_alpha"
        return profile
    view_by_stem = {path.stem: path for path in val_views}
    positive_features: list[np.ndarray] = []
    positive_weights: list[float] = []
    negative_features: list[np.ndarray] = []
    negative_weights: list[float] = []
    selected_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    missing_feature_count = 0
    for row in rows:
        view_name = str(row.get("view", ""))
        view_path = view_by_stem.get(view_name)
        if view_path is None:
            rejected = {"view": view_name, "reason": "view_npz_not_found"}
            rejected_rows.append(rejected)
            continue
        try:
            z = np.load(view_path)
            feature = _view_confidence_feature_for_npz(z)
        except Exception as exc:  # pragma: no cover - audit path for corrupt evidence.
            feature = None
            rejected_rows.append({"view": view_name, "reason": f"feature_load_failed:{type(exc).__name__}"})
        if feature is None:
            missing_feature_count += 1
            continue
        reasons: list[str] = []
        rel_gain = float(row.get("relative_gain", 0.0) or 0.0)
        if rel_gain < float(min_relative_gain):
            reasons.append(
                f"relative_gain {rel_gain:.8g} < min_relative_gain {float(min_relative_gain):.8g}"
            )
        if not _metric_passes_optional_threshold(row.get("ssim_gain"), float(min_ssim_gain)):
            reasons.append("ssim_gain_below_threshold_or_missing")
        if not _metric_passes_optional_threshold(row.get("image_l1_gain"), float(min_l1_gain)):
            reasons.append("image_l1_gain_below_threshold_or_missing")
        if not _metric_passes_optional_threshold(row.get("lpips_gain"), float(min_lpips_gain)):
            reasons.append("lpips_gain_below_threshold_or_missing")
        if reasons:
            negative_features.append(feature.astype(np.float32))
            negative_weight = max(1.0e-4, -rel_gain)
            if row.get("image_l1_gain") is not None:
                negative_weight += max(-float(row.get("image_l1_gain", 0.0) or 0.0) * 10.0, 0.0)
            if row.get("ssim_gain") is not None:
                negative_weight += max(-float(row.get("ssim_gain", 0.0) or 0.0) * 100.0, 0.0)
            if row.get("lpips_gain") is not None:
                negative_weight += max(-float(row.get("lpips_gain", 0.0) or 0.0), 0.0)
            negative_weights.append(float(negative_weight))
            rejected_rows.append(
                {
                    "view": view_name,
                    "relative_gain": rel_gain,
                    "ssim_gain": row.get("ssim_gain"),
                    "image_l1_gain": row.get("image_l1_gain"),
                    "lpips_gain": row.get("lpips_gain"),
                    "weight": float(negative_weight),
                    "feature": [float(x) for x in feature.tolist()],
                    "reasons": reasons,
                }
            )
            continue
        positive_features.append(feature.astype(np.float32))
        weight = max(rel_gain, 1.0e-4)
        if row.get("image_l1_gain") is not None:
            weight += max(float(row.get("image_l1_gain", 0.0) or 0.0) * 10.0, 0.0)
        if row.get("ssim_gain") is not None:
            weight += max(float(row.get("ssim_gain", 0.0) or 0.0) * 100.0, 0.0)
        if row.get("lpips_gain") is not None:
            weight += max(float(row.get("lpips_gain", 0.0) or 0.0), 0.0)
        positive_weights.append(float(weight))
        selected_rows.append(
            {
                "view": view_name,
                "relative_gain": rel_gain,
                "ssim_gain": row.get("ssim_gain"),
                "image_l1_gain": row.get("image_l1_gain"),
                "lpips_gain": row.get("lpips_gain"),
                "weight": float(weight),
                "feature": [float(x) for x in feature.tolist()],
            }
        )
    if not positive_features:
        profile.update(
            {
                "reason": "no_policy_val_views_pass_view_confidence_thresholds",
                "policy_val_view_count": int(len(rows)),
                "missing_feature_count": int(missing_feature_count),
                "rejected_view_count": int(len(rejected_rows)),
                "rejected_preview": rejected_rows[:32],
            }
        )
        return profile
    sigma = float(kernel_sigma)
    if sigma <= 0.0:
        sigma = 0.35
    min_conf = float(np.clip(min_confidence, 0.0, 1.0))
    weight_arr = np.asarray(positive_weights, dtype=np.float32)
    weight_arr = weight_arr / max(float(np.max(weight_arr)), 1.0e-8)
    negative_weight_arr = np.asarray(negative_weights, dtype=np.float32)
    if negative_weight_arr.size:
        negative_weight_arr = negative_weight_arr / max(float(np.max(negative_weight_arr)), 1.0e-8)
    profile.update(
        {
            "enabled": True,
            "decision": "keep_view_confidence_profile",
            "synthesis_mode": "positive_negative_kernel_confidence",
            "policy_val_view_count": int(len(rows)),
            "positive_view_count": int(len(positive_features)),
            "positive_view_fraction": float(len(positive_features) / max(1, len(rows))),
            "negative_view_count": int(len(negative_features)),
            "negative_view_fraction": float(len(negative_features) / max(1, len(rows))),
            "missing_feature_count": int(missing_feature_count),
            "rejected_view_count": int(len(rejected_rows)),
            "positive_features": [[float(x) for x in feature.tolist()] for feature in positive_features],
            "positive_weights": [float(x) for x in weight_arr.tolist()],
            "negative_features": [[float(x) for x in feature.tolist()] for feature in negative_features],
            "negative_weights": [float(x) for x in negative_weight_arr.tolist()],
            "selected_rows": selected_rows[:64],
            "rejected_preview": rejected_rows[:32],
            "kernel_sigma": float(sigma),
            "min_confidence": float(min_conf),
        }
    )
    return profile


def build_policy_val_view_alpha_cap_profile(
    val_views: list[Path],
    policy_val: dict[str, Any],
    *,
    enabled: bool,
    selection_mode: str,
    min_relative_gain: float,
    min_ssim_gain: float,
    min_l1_gain: float,
    min_lpips_gain: float,
    kernel_sigma: float,
    min_confidence: float,
    fallback_alpha: float,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "enabled": False,
        "mode": "policy_val_view_alpha_cap",
        "decision": "not_requested",
    }
    if not bool(enabled):
        return profile
    selection_mode_s = str(selection_mode)
    if selection_mode_s not in {"smallest_safe", "best_safe"}:
        selection_mode_s = "best_safe"
    fallback_alpha_f = max(0.0, float(fallback_alpha))
    profile.update(
        {
            "decision": "disabled",
            "selection_mode": selection_mode_s,
            "min_relative_gain": float(min_relative_gain),
            "min_ssim_gain": float(min_ssim_gain),
            "min_l1_gain": float(min_l1_gain),
            "min_lpips_gain": float(min_lpips_gain),
            "kernel_sigma": float(kernel_sigma),
            "min_confidence": float(min_confidence),
            "fallback_alpha": float(fallback_alpha_f),
            "uses_policy_val_gt": True,
            "uses_target_or_test_gt": False,
            "certification_independent": False,
            "apply_alpha_mode": "cap_global_alpha_by_camera_policy_val_alpha",
            "feature": "normalized_camera_center",
            "cap_interpolation_mode": "nearest_rbf_anchor",
        }
    )
    if not bool(policy_val.get("enabled", False)):
        profile["reason"] = "policy_val_disabled"
        return profile
    per_view = policy_val.get("per_view_by_alpha", {})
    if not isinstance(per_view, dict) or not per_view:
        profile["reason"] = "no_policy_val_per_view_rows"
        return profile
    rows_by_view: dict[str, list[dict[str, Any]]] = {}
    for alpha_key, rows in per_view.items():
        try:
            alpha_f = float(alpha_key)
        except (TypeError, ValueError):
            continue
        for row in rows or []:
            view_name = str(row.get("view", ""))
            if not view_name:
                continue
            enriched = dict(row)
            enriched["alpha"] = float(alpha_f)
            rows_by_view.setdefault(view_name, []).append(enriched)
    view_by_stem = {path.stem: path for path in val_views}
    anchor_features: list[np.ndarray] = []
    anchor_alpha_caps: list[float] = []
    anchor_weights: list[float] = []
    anchor_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    missing_feature_count = 0
    for view_name, view_path in view_by_stem.items():
        rows = rows_by_view.get(view_name, [])
        try:
            z = np.load(view_path)
            feature = _view_confidence_feature_for_npz(z)
        except Exception as exc:  # pragma: no cover - audit path for corrupt evidence.
            feature = None
            rejected_rows.append({"view": view_name, "reason": f"feature_load_failed:{type(exc).__name__}"})
        if feature is None:
            missing_feature_count += 1
            continue
        safe_rows: list[dict[str, Any]] = []
        unsafe_preview: list[dict[str, Any]] = []
        for row in rows:
            alpha_f = float(row.get("alpha", 0.0) or 0.0)
            if alpha_f <= 0.0:
                continue
            reasons: list[str] = []
            rel_gain = float(row.get("relative_gain", 0.0) or 0.0)
            if rel_gain < float(min_relative_gain):
                reasons.append(
                    f"relative_gain {rel_gain:.8g} < min_relative_gain {float(min_relative_gain):.8g}"
                )
            if not _metric_passes_optional_threshold(row.get("ssim_gain"), float(min_ssim_gain)):
                reasons.append("ssim_gain_below_threshold_or_missing")
            if not _metric_passes_optional_threshold(row.get("image_l1_gain"), float(min_l1_gain)):
                reasons.append("image_l1_gain_below_threshold_or_missing")
            if not _metric_passes_optional_threshold(row.get("lpips_gain"), float(min_lpips_gain)):
                reasons.append("lpips_gain_below_threshold_or_missing")
            score = rel_gain
            if row.get("image_l1_gain") is not None:
                score += max(float(row.get("image_l1_gain", 0.0) or 0.0) * 10.0, 0.0)
            if row.get("ssim_gain") is not None:
                score += max(float(row.get("ssim_gain", 0.0) or 0.0) * 100.0, 0.0)
            if row.get("lpips_gain") is not None:
                score += max(float(row.get("lpips_gain", 0.0) or 0.0), 0.0)
            compact = {
                "alpha": float(alpha_f),
                "relative_gain": rel_gain,
                "ssim_gain": row.get("ssim_gain"),
                "image_l1_gain": row.get("image_l1_gain"),
                "lpips_gain": row.get("lpips_gain"),
                "score": float(score),
            }
            if reasons:
                if len(unsafe_preview) < 8:
                    unsafe = dict(compact)
                    unsafe["reasons"] = reasons
                    unsafe_preview.append(unsafe)
                continue
            safe_rows.append(compact)
        if safe_rows:
            if selection_mode_s == "smallest_safe":
                selected = sorted(safe_rows, key=lambda item: (float(item["alpha"]), -float(item["score"])))[0]
            else:
                selected = sorted(safe_rows, key=lambda item: (float(item["score"]), float(item["alpha"])))[-1]
            alpha_cap = max(0.0, float(selected["alpha"]))
            weight = max(float(selected.get("score", 0.0) or 0.0), 1.0e-4)
            decision = "selected_safe_alpha"
        else:
            selected = {
                "alpha": float(fallback_alpha_f),
                "relative_gain": 0.0,
                "ssim_gain": None,
                "image_l1_gain": None,
                "lpips_gain": None,
                "score": 0.0,
            }
            alpha_cap = float(fallback_alpha_f)
            weight = 1.0e-4 if alpha_cap > 0.0 else 1.0
            decision = "fallback_alpha_no_safe_nonzero_policy_val_alpha"
        anchor_features.append(feature.astype(np.float32))
        anchor_alpha_caps.append(float(alpha_cap))
        anchor_weights.append(float(weight))
        anchor_rows.append(
            {
                "view": view_name,
                "alpha_cap": float(alpha_cap),
                "decision": decision,
                "selected": selected,
                "safe_alpha_count": int(len(safe_rows)),
                "unsafe_preview": unsafe_preview,
                "feature": [float(x) for x in feature.tolist()],
            }
        )
    if not anchor_features:
        profile.update(
            {
                "reason": "no_policy_val_views_with_camera_features",
                "policy_val_view_count": int(len(view_by_stem)),
                "missing_feature_count": int(missing_feature_count),
                "rejected_preview": rejected_rows[:32],
            }
        )
        return profile
    sigma = float(kernel_sigma)
    if sigma <= 0.0:
        sigma = 0.35
    min_conf = float(np.clip(min_confidence, 0.0, 1.0))
    weight_arr = np.asarray(anchor_weights, dtype=np.float32)
    weight_arr = weight_arr / max(float(np.max(weight_arr)), 1.0e-8)
    cap_arr = np.asarray(anchor_alpha_caps, dtype=np.float32)
    positive_caps = cap_arr[cap_arr > 0.0]
    profile.update(
        {
            "enabled": True,
            "decision": "keep_view_alpha_cap_profile",
            "policy_val_view_count": int(len(view_by_stem)),
            "anchor_view_count": int(len(anchor_features)),
            "selected_view_count": int(np.count_nonzero(cap_arr > 0.0)),
            "fallback_view_count": int(np.count_nonzero(cap_arr <= 0.0)),
            "selected_positive_view_fraction": float(np.mean(cap_arr > 0.0)) if cap_arr.size else 0.0,
            "missing_feature_count": int(missing_feature_count),
            "rejected_preview": rejected_rows[:32],
            "anchor_features": [[float(x) for x in feature.tolist()] for feature in anchor_features],
            "anchor_alpha_caps": [float(x) for x in cap_arr.tolist()],
            "anchor_weights": [float(x) for x in weight_arr.tolist()],
            "anchor_rows": anchor_rows[:64],
            "alpha_cap_min": float(np.min(cap_arr)) if cap_arr.size else 0.0,
            "alpha_cap_mean": float(np.mean(cap_arr)) if cap_arr.size else 0.0,
            "alpha_cap_max": float(np.max(cap_arr)) if cap_arr.size else 0.0,
            "positive_alpha_cap_min": float(np.min(positive_caps)) if positive_caps.size else 0.0,
            "positive_alpha_cap_mean": float(np.mean(positive_caps)) if positive_caps.size else 0.0,
            "positive_alpha_cap_max": float(np.max(positive_caps)) if positive_caps.size else 0.0,
            "kernel_sigma": float(sigma),
            "min_confidence": float(min_conf),
        }
    )
    return profile


def view_alpha_cap_for_npz(
    z: np.lib.npyio.NpzFile,
    profile: dict[str, Any] | None,
) -> float:
    if not profile or not bool(profile.get("enabled", False)):
        return float("inf")
    feature = _view_confidence_feature_for_npz(z)
    if feature is None:
        return max(0.0, float(profile.get("fallback_alpha", 0.0) or 0.0))
    anchors = np.asarray(profile.get("anchor_features", []), dtype=np.float32)
    if anchors.ndim != 2 or anchors.shape[0] <= 0 or anchors.shape[1] != int(feature.size):
        return max(0.0, float(profile.get("fallback_alpha", 0.0) or 0.0))
    alpha_caps = np.asarray(profile.get("anchor_alpha_caps", []), dtype=np.float32)
    if alpha_caps.shape[0] != anchors.shape[0]:
        return max(0.0, float(profile.get("fallback_alpha", 0.0) or 0.0))
    weights = np.asarray(profile.get("anchor_weights", []), dtype=np.float32)
    if weights.shape[0] != anchors.shape[0]:
        weights = np.ones((anchors.shape[0],), dtype=np.float32)
    sigma = max(float(profile.get("kernel_sigma", 0.35) or 0.35), 1.0e-6)
    dist2 = np.sum((anchors - feature[None, :]) ** 2, axis=1)
    scores = np.exp(-dist2 / (2.0 * sigma * sigma)) * np.clip(weights, 0.0, 1.0)
    if scores.size <= 0:
        return max(0.0, float(profile.get("fallback_alpha", 0.0) or 0.0))
    best_idx = int(np.argmax(scores))
    if float(scores[best_idx]) < float(profile.get("min_confidence", 0.0) or 0.0):
        return max(0.0, float(profile.get("fallback_alpha", 0.0) or 0.0))
    return max(0.0, float(alpha_caps[best_idx]))


def view_confidence_for_npz(
    z: np.lib.npyio.NpzFile,
    profile: dict[str, Any] | None,
) -> float:
    if not profile or not bool(profile.get("enabled", False)):
        return 1.0
    feature = _view_confidence_feature_for_npz(z)
    if feature is None:
        return 0.0
    positive = np.asarray(profile.get("positive_features", []), dtype=np.float32)
    if positive.ndim != 2 or positive.shape[0] <= 0 or positive.shape[1] != int(feature.size):
        return 0.0
    weights = np.asarray(profile.get("positive_weights", []), dtype=np.float32)
    if weights.shape[0] != positive.shape[0]:
        weights = np.ones((positive.shape[0],), dtype=np.float32)
    sigma = max(float(profile.get("kernel_sigma", 0.35) or 0.35), 1.0e-6)
    dist2 = np.sum((positive - feature[None, :]) ** 2, axis=1)
    positive_score = float(np.max(np.exp(-dist2 / (2.0 * sigma * sigma)) * np.clip(weights, 0.0, 1.0)))
    negative = np.asarray(profile.get("negative_features", []), dtype=np.float32)
    negative_score = 0.0
    if negative.ndim == 2 and negative.shape[0] > 0 and negative.shape[1] == int(feature.size):
        negative_weights = np.asarray(profile.get("negative_weights", []), dtype=np.float32)
        if negative_weights.shape[0] != negative.shape[0]:
            negative_weights = np.ones((negative.shape[0],), dtype=np.float32)
        negative_dist2 = np.sum((negative - feature[None, :]) ** 2, axis=1)
        negative_score = float(
            np.max(np.exp(-negative_dist2 / (2.0 * sigma * sigma)) * np.clip(negative_weights, 0.0, 1.0))
        )
    if negative_score > 0.0:
        confidence = positive_score * (positive_score / (positive_score + negative_score + 1.0e-8))
    else:
        confidence = positive_score
    if confidence < float(profile.get("min_confidence", 0.0) or 0.0):
        return 0.0
    return float(np.clip(confidence, 0.0, 1.0))


def _teacher_distilled_basis_feature_dim(mode: str) -> int:
    mode = str(mode)
    if mode == "none":
        return 0
    if mode == "face_uv_normal_camera_ridge":
        # [1, camera3, normal3, normal_dot_camera, u, v, u^2, v^2, u*v]
        return 13
    if mode == "face_uv_patch_mixture_ridge":
        # face_uv_normal_camera_ridge plus a 3x3 local UV RBF mixture and its
        # normal-view interaction. This keeps the teacher field per-face, but
        # gives it local surface capacity instead of one global smooth face fit.
        return 31
    if mode in {"low_rank_view_texture_k4", "low_rank_view_texture"}:
        # View-weight features for a rank-4 texture basis:
        # [1, camera_x, camera_y, camera_z, normal_dot_camera].
        return 5
    if mode in {"low_rank_view_texture_rich_k4", "low_rank_view_texture_rich"}:
        # Rich view-weight features for a low-rank texture basis:
        # [1, camera3, normal3, normal_dot_camera, u, v, u^2, v^2, u*v,
        #  parent_rgb3, inverse_depth, alpha].
        return 18
    if mode == "surface_feature_rff_ridge":
        # Rich base features plus deterministic UV Fourier features and
        # first-frequency normal-view interactions: 18 + 24 + 8.
        return 50
    raise ValueError(f"unsupported teacher-distilled basis mode: {mode}")


def _is_low_rank_teacher_texture_mode(mode: str) -> bool:
    return str(mode) in {
        "low_rank_view_texture_k4",
        "low_rank_view_texture",
        "low_rank_view_texture_rich_k4",
        "low_rank_view_texture_rich",
    }


def _low_rank_teacher_texture_requested_rank(mode: str, requested_rank: int | None = None) -> int:
    if not _is_low_rank_teacher_texture_mode(mode):
        return 0
    if requested_rank is None:
        return 4
    return max(1, int(requested_rank))


def _low_rank_teacher_texture_min_bin_samples(mode: str, feature_dim: int) -> int:
    if str(mode) in {"low_rank_view_texture_rich_k4", "low_rank_view_texture_rich"}:
        # The rich feature vector is ridge-regularized and deliberately allowed
        # to use sparse bins; requiring one sample per feature zeroes too much of
        # the target-visible surface before policy-val can judge the model.
        return 4
    return max(int(feature_dim), 4)


def _surface_feature_rff_tail(
    u: np.ndarray,
    v: np.ndarray,
    normal_dot_camera: np.ndarray,
) -> np.ndarray:
    features: list[np.ndarray] = []
    first_frequency: list[np.ndarray] = []
    for freq in (1.0, 2.0, 4.0):
        angle = np.float32(2.0 * math.pi * float(freq))
        for value in (u, v, u + v, u - v):
            sin_value = np.sin(angle * value).astype(np.float32)
            cos_value = np.cos(angle * value).astype(np.float32)
            features.extend([sin_value, cos_value])
            if freq == 1.0:
                first_frequency.extend([sin_value, cos_value])
    interaction = [normal_dot_camera.astype(np.float32) * item for item in first_frequency]
    return np.concatenate([*features, *interaction], axis=1).astype(np.float32)


def _teacher_distilled_basis_features_for_mask(
    z: np.lib.npyio.NpzFile,
    mode: str,
    mask: np.ndarray,
) -> np.ndarray | None:
    mode = str(mode)
    if mode == "none":
        return None
    if mode not in {
        "face_uv_normal_camera_ridge",
        "face_uv_patch_mixture_ridge",
        "surface_feature_rff_ridge",
        "low_rank_view_texture_k4",
        "low_rank_view_texture",
        "low_rank_view_texture_rich_k4",
        "low_rank_view_texture_rich",
    }:
        raise ValueError(f"unsupported teacher-distilled basis mode: {mode}")
    if "normal" not in z or "barycentric" not in z:
        return None
    direction = _normalized_camera_center(z)
    if direction is None:
        return None
    sample_count = int(np.count_nonzero(mask))
    if sample_count <= 0:
        return None
    normal = np.asarray(z["normal"], dtype=np.float32)
    bary = np.asarray(z["barycentric"], dtype=np.float32)
    if normal.shape[0] != 3 or bary.shape[0] < 3:
        return None
    normal_samples = np.stack([normal[0][mask], normal[1][mask], normal[2][mask]], axis=1)
    normal_samples = np.nan_to_num(normal_samples, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    normal_norm = np.linalg.norm(normal_samples, axis=1, keepdims=True)
    normal_samples = normal_samples / np.maximum(normal_norm, 1.0e-8)
    camera_features = np.repeat(direction[None, :], sample_count, axis=0).astype(np.float32)
    normal_dot_camera = np.sum(normal_samples * camera_features, axis=1, keepdims=True).astype(np.float32)
    u = np.clip(bary[1][mask], 0.0, 1.0).astype(np.float32)[:, None]
    v = np.clip(bary[2][mask], 0.0, 1.0).astype(np.float32)[:, None]
    if mode in {"low_rank_view_texture_k4", "low_rank_view_texture"}:
        return np.concatenate(
            [
                np.ones((sample_count, 1), dtype=np.float32),
                camera_features,
                normal_dot_camera,
            ],
            axis=1,
        )
    if mode in {"low_rank_view_texture_rich_k4", "low_rank_view_texture_rich", "surface_feature_rff_ridge"}:
        if "rgb_render" in z:
            render = np.asarray(z["rgb_render"], dtype=np.float32)
            if render.shape[0] >= 3:
                parent_rgb = np.stack(
                    [render[0][mask], render[1][mask], render[2][mask]],
                    axis=1,
                ).astype(np.float32)
            else:
                parent_rgb = np.zeros((sample_count, 3), dtype=np.float32)
        else:
            parent_rgb = np.zeros((sample_count, 3), dtype=np.float32)
        parent_rgb = np.nan_to_num(parent_rgb, nan=0.0, posinf=1.0, neginf=0.0)
        parent_rgb = np.clip(parent_rgb, 0.0, 1.0).astype(np.float32)
        if "depth" in z:
            depth = np.asarray(z["depth"], dtype=np.float32)[mask].reshape(-1, 1)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            inverse_depth = (1.0 / (1.0 + np.maximum(depth, 0.0))).astype(np.float32)
        else:
            inverse_depth = np.zeros((sample_count, 1), dtype=np.float32)
        if "alpha" in z:
            alpha = np.asarray(z["alpha"], dtype=np.float32)[mask].reshape(-1, 1)
            alpha = np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
            alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        else:
            alpha = np.ones((sample_count, 1), dtype=np.float32)
        return np.concatenate(
            [
                np.ones((sample_count, 1), dtype=np.float32),
                camera_features,
                normal_samples,
                normal_dot_camera,
                u,
                v,
                u * u,
                v * v,
                u * v,
                parent_rgb,
                inverse_depth,
                alpha,
            ],
            axis=1,
        ) if mode != "surface_feature_rff_ridge" else np.concatenate(
            [
                np.ones((sample_count, 1), dtype=np.float32),
                camera_features,
                normal_samples,
                normal_dot_camera,
                u,
                v,
                u * u,
                v * v,
                u * v,
                parent_rgb,
                inverse_depth,
                alpha,
                _surface_feature_rff_tail(u, v, normal_dot_camera),
            ],
            axis=1,
        )
    base_features = [
        np.ones((sample_count, 1), dtype=np.float32),
        camera_features,
        normal_samples,
        normal_dot_camera,
        u,
        v,
        u * u,
        v * v,
        u * v,
    ]
    if mode == "face_uv_normal_camera_ridge":
        return np.concatenate(base_features, axis=1)

    centers = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    uu = u.astype(np.float32)
    vv = v.astype(np.float32)
    patch_weights: list[np.ndarray] = []
    sigma = 0.38
    denom = 2.0 * sigma * sigma
    for cy in centers:
        for cx in centers:
            dist2 = (uu - float(cx)) * (uu - float(cx)) + (vv - float(cy)) * (vv - float(cy))
            patch_weights.append(np.exp(-dist2 / denom).astype(np.float32))
    patch = np.concatenate(patch_weights, axis=1)
    patch /= np.maximum(np.sum(patch, axis=1, keepdims=True), 1.0e-8)
    patch_view = patch * normal_dot_camera.astype(np.float32)
    return np.concatenate([*base_features, patch, patch_view], axis=1)


def _teacher_distilled_basis_features_from_uv_camera_normal(
    mode: str,
    u_values: np.ndarray,
    v_values: np.ndarray,
    camera_features: np.ndarray,
    normal_vector: np.ndarray,
) -> np.ndarray | None:
    """Build teacher-basis rows for synthetic train-only target-impact bins."""
    mode = str(mode)
    if mode == "none":
        return None
    if mode not in {
        "face_uv_normal_camera_ridge",
        "face_uv_patch_mixture_ridge",
        "surface_feature_rff_ridge",
        "low_rank_view_texture_k4",
        "low_rank_view_texture",
        "low_rank_view_texture_rich_k4",
        "low_rank_view_texture_rich",
    }:
        raise ValueError(f"unsupported teacher-distilled basis mode: {mode}")
    u_arr = np.asarray(u_values, dtype=np.float32).reshape(-1, 1)
    v_arr = np.asarray(v_values, dtype=np.float32).reshape(-1, 1)
    sample_count = int(u_arr.shape[0])
    if sample_count <= 0 or v_arr.shape[0] != sample_count:
        return None
    camera_arr = np.asarray(camera_features, dtype=np.float32)
    if camera_arr.ndim == 1:
        camera_arr = np.repeat(camera_arr.reshape(1, 3), sample_count, axis=0)
    if camera_arr.shape != (sample_count, 3):
        return None
    camera_norm = np.linalg.norm(camera_arr, axis=1, keepdims=True)
    camera_arr = camera_arr / np.maximum(camera_norm, 1.0e-8)
    normal_arr = np.asarray(normal_vector, dtype=np.float32).reshape(1, 3)
    normal_arr = np.repeat(normal_arr, sample_count, axis=0)
    normal_arr = np.nan_to_num(normal_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    normal_norm = np.linalg.norm(normal_arr, axis=1, keepdims=True)
    normal_arr = normal_arr / np.maximum(normal_norm, 1.0e-8)
    normal_dot_camera = np.sum(normal_arr * camera_arr, axis=1, keepdims=True).astype(np.float32)
    u = np.clip(u_arr, 0.0, 1.0).astype(np.float32)
    v = np.clip(v_arr, 0.0, 1.0).astype(np.float32)
    if mode in {"low_rank_view_texture_k4", "low_rank_view_texture"}:
        return np.concatenate(
            [
                np.ones((sample_count, 1), dtype=np.float32),
                camera_arr.astype(np.float32),
                normal_dot_camera,
            ],
            axis=1,
        )
    if mode in {"low_rank_view_texture_rich_k4", "low_rank_view_texture_rich", "surface_feature_rff_ridge"}:
        parent_rgb = np.zeros((sample_count, 3), dtype=np.float32)
        inverse_depth = np.zeros((sample_count, 1), dtype=np.float32)
        alpha = np.ones((sample_count, 1), dtype=np.float32)
        rich_features = np.concatenate(
            [
                np.ones((sample_count, 1), dtype=np.float32),
                camera_arr.astype(np.float32),
                normal_arr.astype(np.float32),
                normal_dot_camera,
                u,
                v,
                u * u,
                v * v,
                u * v,
                parent_rgb,
                inverse_depth,
                alpha,
            ],
            axis=1,
        )
        if mode == "surface_feature_rff_ridge":
            return np.concatenate(
                [
                    rich_features,
                    _surface_feature_rff_tail(u, v, normal_dot_camera),
                ],
                axis=1,
            )
        return rich_features
    base_features = [
        np.ones((sample_count, 1), dtype=np.float32),
        camera_arr.astype(np.float32),
        normal_arr.astype(np.float32),
        normal_dot_camera,
        u,
        v,
        u * u,
        v * v,
        u * v,
    ]
    if mode == "face_uv_normal_camera_ridge":
        return np.concatenate(base_features, axis=1)

    centers = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    patch_weights: list[np.ndarray] = []
    sigma = 0.38
    denom = 2.0 * sigma * sigma
    for cy in centers:
        for cx in centers:
            dist2 = (u - float(cx)) * (u - float(cx)) + (v - float(cy)) * (v - float(cy))
            patch_weights.append(np.exp(-dist2 / denom).astype(np.float32))
    patch = np.concatenate(patch_weights, axis=1)
    patch /= np.maximum(np.sum(patch, axis=1, keepdims=True), 1.0e-8)
    patch_view = patch * normal_dot_camera.astype(np.float32)
    return np.concatenate([*base_features, patch, patch_view], axis=1)


def disable_view_conditioned_basis(atlas: dict[int, FaceAtlas]) -> dict[int, FaceAtlas]:
    return {
        int(face): FaceAtlas(
            texture=face_atlas.texture,
            counts=face_atlas.counts,
            variance=face_atlas.variance,
            sign_consistency=face_atlas.sign_consistency,
            mean_rgb=face_atlas.mean_rgb,
            samples=int(face_atlas.samples),
            view_basis_mode="none",
            view_basis_coefficients=None,
            view_basis_support=None,
            view_basis_feature_mean=None,
            view_basis_feature_std=None,
            view_basis_ood_mode="none",
            view_basis_ood_max_z=0.0,
            view_basis_ood_min_std=1.0e-3,
            teacher_basis_mode=face_atlas.teacher_basis_mode,
            teacher_basis_coefficients=face_atlas.teacher_basis_coefficients,
            teacher_basis_feature_mean=face_atlas.teacher_basis_feature_mean,
            teacher_basis_feature_std=face_atlas.teacher_basis_feature_std,
            teacher_basis_ood_max_z=float(face_atlas.teacher_basis_ood_max_z),
            teacher_basis_ood_min_std=float(face_atlas.teacher_basis_ood_min_std),
            teacher_basis_apply_mode=str(face_atlas.teacher_basis_apply_mode),
            teacher_basis_blend=float(face_atlas.teacher_basis_blend),
            teacher_texture_basis=face_atlas.teacher_texture_basis,
            teacher_texture_support=face_atlas.teacher_texture_support,
            teacher_texture_energy=face_atlas.teacher_texture_energy,
            expert_textures=face_atlas.expert_textures,
            expert_counts=face_atlas.expert_counts,
            expert_variance=face_atlas.expert_variance,
            expert_sign_consistency=face_atlas.expert_sign_consistency,
            expert_samples=face_atlas.expert_samples,
            expert_centers=face_atlas.expert_centers,
            expert_feature_mode=str(face_atlas.expert_feature_mode),
            expert_min_bin_samples=int(face_atlas.expert_min_bin_samples),
            expert_fallback_mode=str(face_atlas.expert_fallback_mode),
        )
        for face, face_atlas in atlas.items()
    }


def disable_teacher_distilled_basis(atlas: dict[int, FaceAtlas]) -> dict[int, FaceAtlas]:
    return {
        int(face): FaceAtlas(
            texture=face_atlas.texture,
            counts=face_atlas.counts,
            variance=face_atlas.variance,
            sign_consistency=face_atlas.sign_consistency,
            mean_rgb=face_atlas.mean_rgb,
            samples=int(face_atlas.samples),
            view_basis_mode=face_atlas.view_basis_mode,
            view_basis_coefficients=face_atlas.view_basis_coefficients,
            view_basis_support=face_atlas.view_basis_support,
            view_basis_feature_mean=face_atlas.view_basis_feature_mean,
            view_basis_feature_std=face_atlas.view_basis_feature_std,
            view_basis_ood_mode=str(face_atlas.view_basis_ood_mode),
            view_basis_ood_max_z=float(face_atlas.view_basis_ood_max_z),
            view_basis_ood_min_std=float(face_atlas.view_basis_ood_min_std),
            teacher_basis_mode="none",
            teacher_basis_coefficients=None,
            teacher_basis_feature_mean=None,
            teacher_basis_feature_std=None,
            teacher_basis_ood_max_z=0.0,
            teacher_basis_ood_min_std=1.0e-3,
            teacher_basis_apply_mode="replace_supported",
            teacher_basis_blend=1.0,
            teacher_texture_basis=None,
            teacher_texture_support=None,
            teacher_texture_energy=None,
            expert_textures=face_atlas.expert_textures,
            expert_counts=face_atlas.expert_counts,
            expert_variance=face_atlas.expert_variance,
            expert_sign_consistency=face_atlas.expert_sign_consistency,
            expert_samples=face_atlas.expert_samples,
            expert_centers=face_atlas.expert_centers,
            expert_feature_mode=str(face_atlas.expert_feature_mode),
            expert_min_bin_samples=int(face_atlas.expert_min_bin_samples),
            expert_fallback_mode=str(face_atlas.expert_fallback_mode),
        )
        for face, face_atlas in atlas.items()
    }


def _copy_array_or_none(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value).copy()


def clone_face_atlas(face_atlas: FaceAtlas) -> FaceAtlas:
    return FaceAtlas(
        texture=np.asarray(face_atlas.texture).copy(),
        counts=np.asarray(face_atlas.counts).copy(),
        variance=np.asarray(face_atlas.variance).copy(),
        sign_consistency=np.asarray(face_atlas.sign_consistency).copy(),
        mean_rgb=np.asarray(face_atlas.mean_rgb).copy(),
        samples=int(face_atlas.samples),
        view_basis_mode=str(face_atlas.view_basis_mode),
        view_basis_coefficients=_copy_array_or_none(face_atlas.view_basis_coefficients),
        view_basis_support=_copy_array_or_none(face_atlas.view_basis_support),
        view_basis_feature_mean=_copy_array_or_none(face_atlas.view_basis_feature_mean),
        view_basis_feature_std=_copy_array_or_none(face_atlas.view_basis_feature_std),
        view_basis_ood_mode=str(face_atlas.view_basis_ood_mode),
        view_basis_ood_max_z=float(face_atlas.view_basis_ood_max_z),
        view_basis_ood_min_std=float(face_atlas.view_basis_ood_min_std),
        teacher_basis_mode=str(face_atlas.teacher_basis_mode),
        teacher_basis_coefficients=_copy_array_or_none(face_atlas.teacher_basis_coefficients),
        teacher_basis_feature_mean=_copy_array_or_none(face_atlas.teacher_basis_feature_mean),
        teacher_basis_feature_std=_copy_array_or_none(face_atlas.teacher_basis_feature_std),
        teacher_basis_ood_max_z=float(face_atlas.teacher_basis_ood_max_z),
        teacher_basis_ood_min_std=float(face_atlas.teacher_basis_ood_min_std),
        teacher_basis_apply_mode=str(face_atlas.teacher_basis_apply_mode),
        teacher_basis_blend=float(face_atlas.teacher_basis_blend),
        teacher_texture_basis=_copy_array_or_none(face_atlas.teacher_texture_basis),
        teacher_texture_support=_copy_array_or_none(face_atlas.teacher_texture_support),
        teacher_texture_energy=_copy_array_or_none(face_atlas.teacher_texture_energy),
        expert_textures=_copy_array_or_none(face_atlas.expert_textures),
        expert_counts=_copy_array_or_none(face_atlas.expert_counts),
        expert_variance=_copy_array_or_none(face_atlas.expert_variance),
        expert_sign_consistency=_copy_array_or_none(face_atlas.expert_sign_consistency),
        expert_samples=_copy_array_or_none(face_atlas.expert_samples),
        expert_centers=_copy_array_or_none(face_atlas.expert_centers),
        expert_feature_mode=str(face_atlas.expert_feature_mode),
        expert_min_bin_samples=int(face_atlas.expert_min_bin_samples),
        expert_fallback_mode=str(face_atlas.expert_fallback_mode),
    )


def clone_atlas(atlas: dict[int, FaceAtlas]) -> dict[int, FaceAtlas]:
    return {int(face): clone_face_atlas(face_atlas) for face, face_atlas in atlas.items()}


def _copy_face_atlas_bin(dst: FaceAtlas, src: FaceAtlas, v: int, u: int) -> None:
    dst.texture[v, u] = src.texture[v, u]
    dst.counts[v, u] = src.counts[v, u]
    dst.variance[v, u] = src.variance[v, u]
    dst.sign_consistency[v, u] = src.sign_consistency[v, u]
    if (
        dst.view_basis_coefficients is not None
        and src.view_basis_coefficients is not None
        and dst.view_basis_coefficients.shape == src.view_basis_coefficients.shape
    ):
        dst.view_basis_coefficients[v, u] = src.view_basis_coefficients[v, u]
    if (
        dst.view_basis_support is not None
        and src.view_basis_support is not None
        and dst.view_basis_support.shape == src.view_basis_support.shape
    ):
        dst.view_basis_support[v, u] = src.view_basis_support[v, u]
    if (
        dst.view_basis_feature_mean is not None
        and src.view_basis_feature_mean is not None
        and dst.view_basis_feature_mean.shape == src.view_basis_feature_mean.shape
    ):
        dst.view_basis_feature_mean[v, u] = src.view_basis_feature_mean[v, u]
    if (
        dst.view_basis_feature_std is not None
        and src.view_basis_feature_std is not None
        and dst.view_basis_feature_std.shape == src.view_basis_feature_std.shape
    ):
        dst.view_basis_feature_std[v, u] = src.view_basis_feature_std[v, u]
    if (
        dst.expert_textures is not None
        and src.expert_textures is not None
        and dst.expert_textures.shape == src.expert_textures.shape
    ):
        dst.expert_textures[:, v, u] = src.expert_textures[:, v, u]
    if (
        dst.expert_counts is not None
        and src.expert_counts is not None
        and dst.expert_counts.shape == src.expert_counts.shape
    ):
        dst.expert_counts[:, v, u] = src.expert_counts[:, v, u]
    if (
        dst.expert_variance is not None
        and src.expert_variance is not None
        and dst.expert_variance.shape == src.expert_variance.shape
    ):
        dst.expert_variance[:, v, u] = src.expert_variance[:, v, u]
    if (
        dst.expert_sign_consistency is not None
        and src.expert_sign_consistency is not None
        and dst.expert_sign_consistency.shape == src.expert_sign_consistency.shape
    ):
        dst.expert_sign_consistency[:, v, u] = src.expert_sign_consistency[:, v, u]


def _blend_face_atlas_bin(dst: FaceAtlas, src: FaceAtlas, v: int, u: int, weight: float) -> None:
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0.0:
        return
    if weight >= 1.0:
        _copy_face_atlas_bin(dst, src, v, u)
        return
    dst.texture[v, u] = (1.0 - weight) * dst.texture[v, u] + weight * src.texture[v, u]
    dst.counts[v, u] = np.maximum(dst.counts[v, u], src.counts[v, u])
    dst.variance[v, u] = (1.0 - weight) * dst.variance[v, u] + weight * src.variance[v, u]
    dst.sign_consistency[v, u] = np.clip(
        (1.0 - weight) * dst.sign_consistency[v, u] + weight * src.sign_consistency[v, u],
        0.0,
        1.0,
    )
    if (
        dst.expert_textures is not None
        and src.expert_textures is not None
        and dst.expert_textures.shape == src.expert_textures.shape
    ):
        dst.expert_textures[:, v, u] = (
            (1.0 - weight) * dst.expert_textures[:, v, u]
            + weight * src.expert_textures[:, v, u]
        )
    if (
        dst.expert_counts is not None
        and src.expert_counts is not None
        and dst.expert_counts.shape == src.expert_counts.shape
    ):
        dst.expert_counts[:, v, u] = np.maximum(dst.expert_counts[:, v, u], src.expert_counts[:, v, u])
    if (
        dst.expert_variance is not None
        and src.expert_variance is not None
        and dst.expert_variance.shape == src.expert_variance.shape
    ):
        dst.expert_variance[:, v, u] = (
            (1.0 - weight) * dst.expert_variance[:, v, u]
            + weight * src.expert_variance[:, v, u]
        )
    if (
        dst.expert_sign_consistency is not None
        and src.expert_sign_consistency is not None
        and dst.expert_sign_consistency.shape == src.expert_sign_consistency.shape
    ):
        dst.expert_sign_consistency[:, v, u] = np.clip(
            (1.0 - weight) * dst.expert_sign_consistency[:, v, u]
            + weight * src.expert_sign_consistency[:, v, u],
            0.0,
            1.0,
        )



def _local_alpha_for_samples(
    samples: np.ndarray,
    profile: dict[str, Any] | None,
    face_ids: np.ndarray | None = None,
    bin_ids: np.ndarray | None = None,
    view_cluster_id: int | None = None,
) -> np.ndarray:
    if not profile or not bool(profile.get("enabled", False)):
        return np.ones((int(samples.shape[0]),), dtype=np.float32)
    mode = str(profile.get("mode", ""))
    if mode == "policy_val_face_alpha":
        alpha = np.full(
            (int(samples.shape[0]),),
            float(profile.get("fallback_alpha", 1.0)),
            dtype=np.float32,
        )
        if face_ids is None:
            return np.clip(alpha, 0.0, float(profile.get("max_alpha", np.max(alpha) if alpha.size else 1.0)))
        face_alpha = profile.get("face_alphas", {}) or {}
        if not isinstance(face_alpha, dict):
            return np.clip(alpha, 0.0, float(profile.get("max_alpha", np.max(alpha) if alpha.size else 1.0)))
        local_faces = np.asarray(face_ids, dtype=np.int64)
        for face in np.unique(local_faces):
            value = face_alpha.get(str(int(face)))
            if value is None:
                continue
            alpha[local_faces == int(face)] = float(value)
        return np.clip(alpha, 0.0, float(profile.get("max_alpha", np.max(alpha) if alpha.size else 1.0)))
    if mode == "policy_val_bin_alpha":
        alpha = np.full(
            (int(samples.shape[0]),),
            float(profile.get("fallback_alpha", 1.0)),
            dtype=np.float32,
        )
        if face_ids is None or bin_ids is None:
            return np.clip(alpha, 0.0, float(profile.get("max_alpha", np.max(alpha) if alpha.size else 1.0)))
        raw_by_face = profile.get("bin_alphas_by_face", {}) or {}
        if not isinstance(raw_by_face, dict):
            return np.clip(alpha, 0.0, float(profile.get("max_alpha", np.max(alpha) if alpha.size else 1.0)))
        local_faces = np.asarray(face_ids, dtype=np.int64)
        local_bins = np.asarray(bin_ids, dtype=np.int64)
        for face in np.unique(local_faces):
            face_key = str(int(face))
            face_profile = raw_by_face.get(face_key)
            if not isinstance(face_profile, dict):
                continue
            fm = local_faces == int(face)
            for bin_id in np.unique(local_bins[fm]):
                value = face_profile.get(str(int(bin_id)))
                if value is None:
                    continue
                alpha[fm & (local_bins == int(bin_id))] = float(value)
        return np.clip(alpha, 0.0, float(profile.get("max_alpha", np.max(alpha) if alpha.size else 1.0)))
    if mode == "policy_val_bin_rgb_alpha":
        fallback = np.asarray(profile.get("fallback_alpha", [1.0, 1.0, 1.0]), dtype=np.float32)
        if fallback.size == 1:
            fallback = np.repeat(fallback, 3)
        if fallback.size != 3:
            fallback = np.ones((3,), dtype=np.float32)
        alpha = np.repeat(fallback[None, :], int(samples.shape[0]), axis=0).astype(np.float32)
        if face_ids is None or bin_ids is None:
            return np.clip(alpha, 0.0, float(profile.get("max_alpha", np.max(alpha) if alpha.size else 1.0)))
        raw_by_face = profile.get("bin_rgb_alphas_by_face", {}) or {}
        if not isinstance(raw_by_face, dict):
            return np.clip(alpha, 0.0, float(profile.get("max_alpha", np.max(alpha) if alpha.size else 1.0)))
        local_faces = np.asarray(face_ids, dtype=np.int64)
        local_bins = np.asarray(bin_ids, dtype=np.int64)
        for face in np.unique(local_faces):
            face_key = str(int(face))
            face_profile = raw_by_face.get(face_key)
            if not isinstance(face_profile, dict):
                continue
            fm = local_faces == int(face)
            for bin_id in np.unique(local_bins[fm]):
                value = face_profile.get(str(int(bin_id)))
                if value is None:
                    continue
                value_arr = np.asarray(value, dtype=np.float32)
                if value_arr.size == 1:
                    value_arr = np.repeat(value_arr, 3)
                if value_arr.size != 3:
                    continue
                alpha[fm & (local_bins == int(bin_id))] = value_arr.reshape(3)
        return np.clip(alpha, 0.0, float(profile.get("max_alpha", np.max(alpha) if alpha.size else 1.0)))
    if mode == "policy_val_hybrid_bin_source_alpha":
        baseline_profile = profile.get("baseline_profile", {"enabled": False})
        prior_profile = profile.get("prior_profile", {"enabled": False})
        baseline_alpha = _local_alpha_for_samples(
            samples,
            baseline_profile,
            face_ids=face_ids,
            bin_ids=bin_ids,
            view_cluster_id=view_cluster_id,
        )
        prior_alpha = _local_alpha_for_samples(
            samples,
            prior_profile,
            face_ids=face_ids,
            bin_ids=bin_ids,
            view_cluster_id=view_cluster_id,
        )
        sample_count = int(samples.shape[0])
        valid_shapes = {(sample_count,), (sample_count, 3)}
        if tuple(baseline_alpha.shape) not in valid_shapes or tuple(prior_alpha.shape) not in valid_shapes:
            raise ValueError(
                "hybrid local-alpha profiles must return shape (N,) or (N, 3); "
                f"got baseline={tuple(baseline_alpha.shape)} prior={tuple(prior_alpha.shape)}"
            )
        if baseline_alpha.ndim != prior_alpha.ndim:
            if baseline_alpha.ndim == 1:
                baseline_alpha = np.repeat(baseline_alpha[:, None], 3, axis=1)
            if prior_alpha.ndim == 1:
                prior_alpha = np.repeat(prior_alpha[:, None], 3, axis=1)
        alpha = np.array(baseline_alpha, copy=True)
        if face_ids is None or bin_ids is None:
            return np.clip(
                alpha,
                0.0,
                float(profile.get("max_alpha", np.max(alpha) if alpha.size else 1.0)),
            )
        raw_by_face = profile.get("prior_bins_by_face", {}) or {}
        if not isinstance(raw_by_face, dict):
            return np.clip(
                alpha,
                0.0,
                float(profile.get("max_alpha", np.max(alpha) if alpha.size else 1.0)),
            )
        local_faces = np.asarray(face_ids, dtype=np.int64)
        local_bins = np.asarray(bin_ids, dtype=np.int64)
        use_prior = np.zeros((int(local_faces.size),), dtype=bool)
        for face in np.unique(local_faces):
            raw_bins = raw_by_face.get(str(int(face)))
            if not raw_bins:
                continue
            allowed_bins = np.asarray([int(x) for x in raw_bins], dtype=np.int64)
            fm = local_faces == int(face)
            use_prior[fm] = np.isin(local_bins[fm], allowed_bins)
        if bool(np.any(use_prior)):
            alpha[use_prior] = prior_alpha[use_prior]
        return np.clip(
            alpha,
            0.0,
            float(profile.get("max_alpha", np.max(alpha) if alpha.size else 1.0)),
        )
    if mode == "policy_val_bin_uncertainty_shrink":
        alpha = np.full(
            (int(samples.shape[0]),),
            float(profile.get("fallback_shrink", 0.0)),
            dtype=np.float32,
        )
        if face_ids is None or bin_ids is None:
            return np.clip(alpha, 0.0, float(profile.get("max_shrink", np.max(alpha) if alpha.size else 1.0)))
        raw_by_face = profile.get("bin_shrinks_by_face", {}) or {}
        if bool(profile.get("view_cluster_local_shrink", False)):
            cluster_profiles = profile.get("cluster_bin_shrinks_by_cluster_face", {}) or {}
            cluster_key = str(int(view_cluster_id)) if view_cluster_id is not None else ""
            raw_by_face = cluster_profiles.get(cluster_key, {}) if isinstance(cluster_profiles, dict) else {}
            if not raw_by_face and bool(profile.get("view_cluster_local_global_fallback", False)):
                raw_by_face = profile.get("bin_shrinks_by_face", {}) or {}
        if not isinstance(raw_by_face, dict):
            return np.clip(alpha, 0.0, float(profile.get("max_shrink", np.max(alpha) if alpha.size else 1.0)))
        local_faces = np.asarray(face_ids, dtype=np.int64)
        local_bins = np.asarray(bin_ids, dtype=np.int64)
        for face in np.unique(local_faces):
            face_key = str(int(face))
            face_profile = raw_by_face.get(face_key)
            if not isinstance(face_profile, dict):
                continue
            fm = local_faces == int(face)
            for bin_id in np.unique(local_bins[fm]):
                value = face_profile.get(str(int(bin_id)))
                if value is None:
                    continue
                alpha[fm & (local_bins == int(bin_id))] = float(value)
        return np.clip(alpha, 0.0, float(profile.get("max_shrink", np.max(alpha) if alpha.size else 1.0)))
    if mode != "policy_val_residual_norm_buckets":
        return np.ones((int(samples.shape[0]),), dtype=np.float32)
    edges = np.asarray(profile.get("bucket_edges", []), dtype=np.float32)
    alphas = np.asarray(profile.get("bucket_alphas", []), dtype=np.float32)
    if alphas.size != edges.size + 1:
        return np.ones((int(samples.shape[0]),), dtype=np.float32)
    norm_mode = str(profile.get("norm_mode", "l2"))
    if norm_mode == "mean_abs":
        norms = np.mean(np.abs(samples.astype(np.float32)), axis=1)
    else:
        norms = np.linalg.norm(samples.astype(np.float32), axis=1)
    bucket = np.searchsorted(edges, norms, side="right")
    bucket = np.clip(bucket, 0, alphas.size - 1)
    return np.clip(alphas[bucket], 0.0, float(profile.get("max_alpha", np.max(alphas) if alphas.size else 1.0)))


def _as_rgb_chw(image: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3:
        return None
    if arr.shape[0] == 3:
        return arr
    if arr.shape[-1] == 3:
        return np.moveaxis(arr, -1, 0)
    return None


def _luminance_gradient_magnitude_chw(image: np.ndarray) -> np.ndarray | None:
    rgb = _as_rgb_chw(image)
    if rgb is None:
        return None
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    dx = np.zeros_like(lum, dtype=np.float32)
    dy = np.zeros_like(lum, dtype=np.float32)
    dx[:, :-1] = lum[:, 1:] - lum[:, :-1]
    dy[:-1, :] = lum[1:, :] - lum[:-1, :]
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def _image_linear_generator_feature_names(feature_mode: str) -> list[str]:
    mode = str(feature_mode)
    names = ["bias", "base_r", "base_g", "base_b", "base_l1"]
    if mode in {"base_rgb", "base_rgb_bary_view"}:
        names.extend(["render_r", "render_g", "render_b", "render_luma"])
    if mode == "base_rgb_bary_view":
        names.extend(["bary0", "bary1", "bary2", "view_x", "view_y", "view_z"])
    if mode not in {"base", "base_rgb", "base_rgb_bary_view"}:
        raise ValueError(f"unsupported image-linear generator feature mode: {mode}")
    return names


def _image_linear_generator_features_for_samples(
    z: np.lib.npyio.NpzFile,
    *,
    base_pixels: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    bary: np.ndarray,
    feature_mode: str,
    feature_cache: dict[str, Any] | None = None,
) -> np.ndarray:
    mode = str(feature_mode)
    _image_linear_generator_feature_names(mode)
    cache = feature_cache if feature_cache is not None else {}
    base = np.asarray(base_pixels, dtype=np.float32).reshape(-1, 3)
    n = int(base.shape[0])
    columns: list[np.ndarray] = [
        np.ones((n, 1), dtype=np.float32),
        base,
        np.mean(np.abs(base), axis=1, keepdims=True).astype(np.float32),
    ]
    if mode in {"base_rgb", "base_rgb_bary_view"}:
        render = cache.get("rgb_render_chw")
        if "rgb_render_chw" not in cache:
            render = _as_rgb_chw(np.asarray(z["rgb_render"])) if "rgb_render" in z else None
            if render is not None and render.dtype != np.float32:
                render = render.astype(np.float32, copy=False)
            cache["rgb_render_chw"] = render
        if render is None:
            rgb = np.zeros((n, 3), dtype=np.float32)
        else:
            rgb = np.stack(
                [render[0][ys, xs], render[1][ys, xs], render[2][ys, xs]],
                axis=1,
            ).astype(np.float32)
        render_luma = cache.get("rgb_render_luma")
        if "rgb_render_luma" not in cache:
            if render is None:
                render_luma = None
            else:
                render_luma = (0.299 * render[0] + 0.587 * render[1] + 0.114 * render[2]).astype(np.float32)
            cache["rgb_render_luma"] = render_luma
        if render_luma is None:
            luma = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]).astype(np.float32)
        else:
            luma = render_luma[ys, xs].reshape(-1, 1).astype(np.float32)
        columns.extend([rgb, luma])
    if mode == "base_rgb_bary_view":
        bary_samples = np.stack(
            [bary[0][ys, xs], bary[1][ys, xs], bary[2][ys, xs]],
            axis=1,
        ).astype(np.float32)
        view = cache.get("normalized_camera_center")
        if "normalized_camera_center" not in cache:
            view = _normalized_camera_center(z)
            cache["normalized_camera_center"] = view
        if view is None:
            view = np.zeros((3,), dtype=np.float32)
        view_arr = np.repeat(np.asarray(view, dtype=np.float32).reshape(1, 3), n, axis=0)
        columns.extend([bary_samples, view_arr])
    return np.concatenate(columns, axis=1).astype(np.float32)


def _apply_image_linear_generator_to_samples(
    z: np.lib.npyio.NpzFile,
    profile: dict[str, Any] | None,
    *,
    base_pixels: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    bary: np.ndarray,
    current_alpha: float = 1.0,
    feature_cache: dict[str, Any] | None = None,
) -> np.ndarray:
    if not profile or not bool(profile.get("enabled", False)):
        return base_pixels
    if str(profile.get("mode", "")) != "policy_val_image_linear_generator":
        return base_pixels
    weights = np.asarray(profile.get("weights", []), dtype=np.float32)
    if weights.ndim != 2 or weights.shape[1] != 3:
        return base_pixels
    expert_index: int | None = None
    reliability_group_index: int | None = None
    expert_weights = np.asarray(profile.get("expert_weights", []), dtype=np.float32)
    expert_enabled = np.asarray(profile.get("expert_enabled", []), dtype=bool)
    expert_centers = np.asarray(profile.get("expert_centers", []), dtype=np.float32)
    expert_feature_mode = str(profile.get("expert_feature_mode", "none"))
    if (
        expert_weights.ndim == 3
        and expert_weights.shape[1:] == weights.shape
        and expert_enabled.ndim == 1
        and expert_enabled.shape[0] == expert_weights.shape[0]
        and expert_centers.ndim == 2
        and expert_centers.shape[0] == expert_weights.shape[0]
        and expert_feature_mode != "none"
    ):
        assigned_expert_index = _assign_view_cluster_for_npz(
            z,
            centers=expert_centers,
            feature_mode=expert_feature_mode,
        )
        if assigned_expert_index is not None and 0 <= int(assigned_expert_index) < int(expert_weights.shape[0]):
            reliability_group_index = int(assigned_expert_index)
            if bool(expert_enabled[int(assigned_expert_index)]):
                expert_index = int(assigned_expert_index)
                weights = expert_weights[int(expert_index)]
    feature_mode = str(profile.get("feature_mode", "base"))
    features = _image_linear_generator_features_for_samples(
        z,
        base_pixels=base_pixels,
        ys=ys,
        xs=xs,
        bary=bary,
        feature_mode=feature_mode,
        feature_cache=feature_cache,
    )
    if features.shape[1] != weights.shape[0]:
        return base_pixels
    generated = features @ weights
    output_cap = float(profile.get("generator_output_cap", 0.0) or 0.0)
    if output_cap > 0.0:
        generated = np.clip(generated, -output_cap, output_cap)
    face_reliability_profile = profile.get("face_reliability_profile")
    if isinstance(face_reliability_profile, dict) and bool(face_reliability_profile.get("enabled", False)):
        reliability_mode = str(face_reliability_profile.get("mode", "none"))
        if reliability_mode in {"global", "view_cluster"} and "face_id" in z:
            group_id = -1
            if reliability_mode == "view_cluster":
                group_id = int(reliability_group_index) if reliability_group_index is not None else -1
            fallback_multiplier = float(face_reliability_profile.get("fallback_multiplier", 0.0) or 0.0)
            fallback_multiplier = float(np.clip(fallback_multiplier, 0.0, 1.0))
            multipliers_by_face: dict[int, float] = {}
            selected_alpha_by_face: dict[int, float] = {}
            entries_by_group = face_reliability_profile.get("entries_by_group", {})
            if isinstance(entries_by_group, dict):
                raw_group_entries = entries_by_group.get(str(int(group_id)), {})
                if isinstance(raw_group_entries, dict):
                    for face_key, value in raw_group_entries.items():
                        try:
                            face_i = int(face_key)
                        except (TypeError, ValueError):
                            continue
                        if isinstance(value, dict):
                            try:
                                multipliers_by_face[face_i] = float(np.clip(float(value.get("multiplier", 0.0)), 0.0, 1.0))
                            except (TypeError, ValueError):
                                continue
                            try:
                                selected_alpha = float(value.get("selected_alpha", float("nan")))
                            except (TypeError, ValueError):
                                selected_alpha = float("nan")
                            if math.isfinite(selected_alpha) and selected_alpha >= 0.0:
                                selected_alpha_by_face[face_i] = float(selected_alpha)
                        else:
                            try:
                                multipliers_by_face[face_i] = float(np.clip(float(value), 0.0, 1.0))
                            except (TypeError, ValueError):
                                continue
            if not multipliers_by_face:
                entries = face_reliability_profile.get("entries", [])
                if isinstance(entries, list):
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        try:
                            entry_group = int(entry.get("group_id", -1))
                            entry_face = int(entry.get("face_id"))
                            entry_multiplier = float(entry.get("multiplier", 0.0))
                        except (TypeError, ValueError):
                            continue
                        if entry_group == int(group_id):
                            multipliers_by_face[entry_face] = float(np.clip(entry_multiplier, 0.0, 1.0))
                            try:
                                selected_alpha = float(entry.get("selected_alpha", float("nan")))
                            except (TypeError, ValueError):
                                selected_alpha = float("nan")
                            if math.isfinite(selected_alpha) and selected_alpha >= 0.0:
                                selected_alpha_by_face[entry_face] = float(selected_alpha)
            if multipliers_by_face or fallback_multiplier < 1.0:
                face_img = np.asarray(z["face_id"], dtype=np.int64)
                face_samples = face_img[ys, xs].reshape(-1)
                multipliers = np.full((int(face_samples.size),), fallback_multiplier, dtype=np.float32)
                alpha_denominator = max(float(current_alpha), 1.0e-12)
                for face in np.unique(face_samples):
                    face_i = int(face)
                    if face_i in multipliers_by_face:
                        multiplier = float(multipliers_by_face[face_i])
                        selected_alpha = selected_alpha_by_face.get(face_i)
                        if selected_alpha is not None:
                            multiplier *= float(np.clip(float(selected_alpha) / alpha_denominator, 0.0, 1.0))
                        multipliers[face_samples == face_i] = float(np.clip(multiplier, 0.0, 1.0))
                generated = generated * multipliers.reshape(-1, 1)
    return generated.astype(np.float32)


def make_parent_edge_apply_profile(
    *,
    enabled: bool,
    weight: float,
    edge_tau: float,
    min_multiplier: float,
) -> dict[str, Any]:
    enabled = bool(enabled) and float(weight) > 0.0
    return {
        "enabled": bool(enabled),
        "mode": "gt_free_parent_edge_apply_shrink",
        "uses_target_or_test_gt": False,
        "uses_parent_render": bool(enabled),
        "weight": float(weight),
        "edge_tau": float(edge_tau),
        "min_multiplier": float(min_multiplier),
    }


def _profile_max_alpha(profile: dict[str, Any] | None) -> float:
    if not profile or not bool(profile.get("enabled", False)):
        return 1.0
    for key in ("max_alpha", "max_shrink"):
        if key in profile:
            try:
                return float(profile.get(key))
            except (TypeError, ValueError):
                return 1.0
    return 1.0


def _profile_effective_max_alpha(profile: dict[str, Any] | None) -> float:
    if not profile or not bool(profile.get("enabled", False)):
        return float("inf")
    return _profile_max_alpha(profile)


def _hybrid_prior_source_mask(
    profile: dict[str, Any] | None,
    face_ids: np.ndarray,
    bin_ids: np.ndarray,
) -> np.ndarray:
    local_faces = np.asarray(face_ids, dtype=np.int64)
    local_bins = np.asarray(bin_ids, dtype=np.int64)
    use_prior = np.zeros((int(local_faces.size),), dtype=bool)
    raw_by_face = dict((profile or {}).get("prior_bins_by_face", {}) or {})
    for face in np.unique(local_faces):
        raw_bins = raw_by_face.get(str(int(face)))
        if not raw_bins:
            continue
        allowed_bins = np.asarray([int(x) for x in raw_bins], dtype=np.int64)
        fm = local_faces == int(face)
        use_prior[fm] = np.isin(local_bins[fm], allowed_bins)
    return use_prior


def hybrid_bin_source_local_alpha_profile(
    baseline_profile: dict[str, Any] | None,
    prior_profile: dict[str, Any] | None,
    hybrid_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_profile = dict(baseline_profile or {"enabled": False})
    prior_profile = dict(prior_profile or {"enabled": False})
    prior_bins_by_face = dict((hybrid_profile or {}).get("allowed_bins_by_face", {}) or {})
    prior_bin_count = int(sum(len(v) for v in prior_bins_by_face.values()))
    if prior_bin_count <= 0:
        return baseline_profile
    if not bool(baseline_profile.get("enabled", False)) and not bool(prior_profile.get("enabled", False)):
        return {"enabled": False, "mode": "policy_val_hybrid_bin_source_alpha", "reason": "both_profiles_disabled"}
    return {
        "enabled": True,
        "mode": "policy_val_hybrid_bin_source_alpha",
        "baseline_profile": baseline_profile,
        "prior_profile": prior_profile,
        "prior_bins_by_face": prior_bins_by_face,
        "prior_bin_count": int(prior_bin_count),
        "baseline_mode": str(baseline_profile.get("mode", "disabled")),
        "prior_mode": str(prior_profile.get("mode", "disabled")),
        "max_alpha": float(max(_profile_max_alpha(baseline_profile), _profile_max_alpha(prior_profile))),
    }


def compatible_hybrid_guard_profile(
    baseline_profile: dict[str, Any] | None,
    prior_profile: dict[str, Any] | None,
    disabled_profile: dict[str, Any],
) -> dict[str, Any] | None:
    baseline_profile = dict(baseline_profile or disabled_profile)
    prior_profile = dict(prior_profile or disabled_profile)
    baseline_enabled = bool(baseline_profile.get("enabled", False))
    prior_enabled = bool(prior_profile.get("enabled", False))
    if not baseline_enabled and not prior_enabled:
        return dict(disabled_profile)
    if json_safe(baseline_profile) == json_safe(prior_profile):
        return dict(baseline_profile)
    return None


def _allowed_faces_from_gain_guard(profile: dict[str, Any] | None) -> set[int] | None:
    if not profile or not bool(profile.get("enabled", False)):
        return None
    if str(profile.get("mode", "")) != "policy_val_face_gain_guard":
        return None
    allowed = profile.get("allowed_faces", [])
    if not isinstance(allowed, (list, tuple, set)):
        return set()
    return {int(face) for face in allowed}


def _allowed_bins_from_uncertainty_guard(profile: dict[str, Any] | None) -> dict[int, set[int]] | None:
    if not profile or not bool(profile.get("enabled", False)):
        return None
    if str(profile.get("mode", "")) not in {
        "policy_val_bin_uncertainty_guard",
        "policy_val_sparse_residual_materialization",
        "policy_val_sparse_residual_materialization_and_bin_uncertainty_guard",
    }:
        return None
    raw = profile.get("allowed_bins_by_face", {}) or {}
    if not isinstance(raw, dict):
        return {}
    allowed: dict[int, set[int]] = {}
    for face, bins in raw.items():
        try:
            face_i = int(face)
        except (TypeError, ValueError):
            continue
        if not isinstance(bins, (list, tuple, set)):
            continue
        allowed[face_i] = {int(bin_id) for bin_id in bins}
    return allowed


def _lowpass_texture(texture: np.ndarray, counts: np.ndarray, passes: int, neighbor_min_count: int) -> np.ndarray:
    out = np.asarray(texture, dtype=np.float32).copy()
    support = np.asarray(counts, dtype=np.int64) >= int(neighbor_min_count)
    if int(passes) <= 0 or not bool(np.any(support)):
        return out
    h, w = support.shape
    for _ in range(int(passes)):
        src = out.copy()
        dst = out.copy()
        for y in range(h):
            y0 = max(0, y - 1)
            y1 = min(h, y + 2)
            for x in range(w):
                if not support[y, x]:
                    continue
                x0 = max(0, x - 1)
                x1 = min(w, x + 2)
                local_support = support[y0:y1, x0:x1]
                if not bool(np.any(local_support)):
                    continue
                local_weights = counts[y0:y1, x0:x1].astype(np.float32) * local_support.astype(np.float32)
                weight_sum = float(local_weights.sum())
                if weight_sum <= 0.0:
                    continue
                local_tex = src[y0:y1, x0:x1]
                dst[y, x] = np.sum(local_tex * local_weights[..., None], axis=(0, 1)) / weight_sum
        out = dst
    return out


def _nearest_observed_fill_texture(
    texture: np.ndarray,
    counts: np.ndarray,
    face_mean: np.ndarray,
    max_steps: int,
    decay: float,
) -> np.ndarray:
    out = np.asarray(texture, dtype=np.float32).copy()
    filled = np.asarray(counts, dtype=np.int64) > 0
    if not bool(np.any(filled)):
        out[...] = np.asarray(face_mean, dtype=np.float32)
        return out
    h, w = filled.shape
    decay = float(decay)
    for _ in range(max(0, int(max_steps))):
        if bool(np.all(filled)):
            break
        src = out.copy()
        src_filled = filled.copy()
        dst = out.copy()
        newly = np.zeros_like(filled, dtype=bool)
        for y in range(h):
            y0 = max(0, y - 1)
            y1 = min(h, y + 2)
            for x in range(w):
                if src_filled[y, x]:
                    continue
                x0 = max(0, x - 1)
                x1 = min(w, x + 2)
                local = src_filled[y0:y1, x0:x1]
                if not bool(np.any(local)):
                    continue
                dst[y, x] = np.mean(src[y0:y1, x0:x1][local], axis=0) * decay
                newly[y, x] = True
        if not bool(np.any(newly)):
            break
        out = dst
        filled |= newly
    if not bool(np.all(filled)):
        out[~filled] = np.asarray(face_mean, dtype=np.float32)
    return out


def _count_pyramid_prior_texture(
    texture: np.ndarray,
    sum_grid: np.ndarray,
    counts: np.ndarray,
    face_mean: np.ndarray,
    block_sizes: list[int],
    min_bin_samples: int,
    count_tau: float,
    blend: float,
    variance: np.ndarray | None = None,
    sign_consistency: np.ndarray | None = None,
    gate_mode: str = "none",
    min_prior_weight: float = 0.0,
    min_direct_samples: int = 1,
    min_sign_consistency: float = 0.0,
    max_mean_variance: float = -1.0,
    min_cosine: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend low-support residual bins with same-face coarse residual blocks."""
    base = np.asarray(texture, dtype=np.float32)
    out = base.copy()
    count_grid = np.asarray(counts, dtype=np.int64)
    h, w = count_grid.shape
    sizes = sorted({int(size) for size in block_sizes if int(size) > 1})
    summary = {
        "enabled": bool(sizes),
        "block_sizes": [int(size) for size in sizes],
        "total_bins": int(h * w),
        "low_support_bins": 0,
        "blended_bins": 0,
        "mean_blend_weight": 0.0,
        "max_blend_weight": 0.0,
        "gate_mode": str(gate_mode),
        "min_prior_weight": float(min_prior_weight),
        "min_direct_samples": int(min_direct_samples),
        "min_sign_consistency": float(min_sign_consistency),
        "max_mean_variance": float(max_mean_variance),
        "min_cosine": float(min_cosine),
        "gate_rejected_bins": 0,
        "empty_rejected_bins": 0,
        "prior_weight_rejected_bins": 0,
        "sign_rejected_bins": 0,
        "variance_rejected_bins": 0,
        "cosine_rejected_bins": 0,
    }
    if not sizes:
        return out, summary

    prior = np.repeat(np.asarray(face_mean, dtype=np.float32).reshape(1, 1, 3), h, axis=0)
    prior = np.repeat(prior, w, axis=1)
    prior_weight = np.zeros((h, w), dtype=np.float32)
    tau = max(0.0, float(count_tau))
    for block_size in sizes:
        scale_bonus = 1.0 / float(block_size)
        for y0 in range(0, h, block_size):
            y1 = min(h, y0 + block_size)
            for x0 in range(0, w, block_size):
                x1 = min(w, x0 + block_size)
                block_count = int(np.sum(count_grid[y0:y1, x0:x1]))
                if block_count <= 0:
                    continue
                block_sum = np.sum(np.asarray(sum_grid[y0:y1, x0:x1], dtype=np.float64), axis=(0, 1))
                block_mean = (block_sum / float(block_count)).astype(np.float32)
                count_conf = (
                    float(block_count) / (float(block_count) + tau)
                    if tau > 0.0
                    else 1.0
                )
                weight = float(np.clip(count_conf * scale_bonus, 0.0, 1.0))
                local = prior_weight[y0:y1, x0:x1]
                replace = weight > local
                if not bool(np.any(replace)):
                    continue
                patch = prior[y0:y1, x0:x1]
                patch[replace] = block_mean
                prior[y0:y1, x0:x1] = patch
                local[replace] = weight
                prior_weight[y0:y1, x0:x1] = local

    min_samples = max(1, int(min_bin_samples))
    low_support = count_grid < min_samples
    if not bool(np.any(low_support)):
        return out, summary
    direct_conf = np.ones((h, w), dtype=np.float32)
    direct_conf[low_support] = count_grid[low_support].astype(np.float32) / float(min_samples)
    gate_mode_name = str(gate_mode)
    if gate_mode_name not in {"none", "evidence_consistent"}:
        raise ValueError(f"unsupported surface multiscale prior gate mode: {gate_mode}")
    blend_weight = float(np.clip(blend, 0.0, 1.0)) * (1.0 - np.clip(direct_conf, 0.0, 1.0))
    blend_weight *= (prior_weight > 0.0).astype(np.float32)
    active = low_support & (blend_weight > 0.0)
    if gate_mode_name == "evidence_consistent" and bool(np.any(active)):
        pre_gate = active.copy()
        observed = count_grid >= max(1, int(min_direct_samples))
        active &= observed
        summary["empty_rejected_bins"] = int(np.sum(pre_gate & ~observed))

        prior_weight_ok = prior_weight >= max(0.0, float(min_prior_weight))
        active &= prior_weight_ok
        summary["prior_weight_rejected_bins"] = int(np.sum(pre_gate & ~prior_weight_ok))

        if sign_consistency is not None and float(min_sign_consistency) > 0.0:
            sign_mean = np.mean(np.asarray(sign_consistency, dtype=np.float32), axis=2)
            sign_ok = sign_mean >= float(min_sign_consistency)
            active &= sign_ok
            summary["sign_rejected_bins"] = int(np.sum(pre_gate & ~sign_ok))

        if variance is not None and float(max_mean_variance) >= 0.0:
            var_mean = np.mean(np.asarray(variance, dtype=np.float32), axis=2)
            var_ok = var_mean <= float(max_mean_variance)
            active &= var_ok
            summary["variance_rejected_bins"] = int(np.sum(pre_gate & ~var_ok))

        if float(min_cosine) > -1.0:
            base_vec = base.astype(np.float32)
            prior_vec = prior.astype(np.float32)
            numerator = np.sum(base_vec * prior_vec, axis=2)
            denom = np.linalg.norm(base_vec, axis=2) * np.linalg.norm(prior_vec, axis=2)
            cosine = np.full((h, w), -1.0, dtype=np.float32)
            valid_cosine = denom > 1.0e-8
            cosine[valid_cosine] = numerator[valid_cosine] / denom[valid_cosine]
            cosine_ok = cosine >= float(min_cosine)
            active &= cosine_ok
            summary["cosine_rejected_bins"] = int(np.sum(pre_gate & ~cosine_ok))
        summary["gate_rejected_bins"] = int(np.sum(pre_gate & ~active))
    if bool(np.any(active)):
        out[active] = (
            (1.0 - blend_weight[active, None]) * base[active]
            + blend_weight[active, None] * prior[active]
        ).astype(np.float32)
    summary.update(
        {
            "low_support_bins": int(np.sum(low_support)),
            "blended_bins": int(np.sum(active)),
            "mean_blend_weight": float(np.mean(blend_weight[active])) if bool(np.any(active)) else 0.0,
            "max_blend_weight": float(np.max(blend_weight[active])) if bool(np.any(active)) else 0.0,
        }
    )
    return out, summary


def _local_patch_prior_texture(
    texture: np.ndarray,
    sum_grid: np.ndarray,
    counts: np.ndarray,
    face_mean: np.ndarray,
    block_sizes: list[int],
    min_bin_samples: int,
    count_tau: float,
    blend: float,
    variance: np.ndarray | None = None,
    sign_consistency: np.ndarray | None = None,
    gate_mode: str = "none",
    min_prior_weight: float = 0.0,
    min_direct_samples: int = 1,
    min_sign_consistency: float = 0.0,
    max_mean_variance: float = -1.0,
    min_cosine: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend low-support bins with same-face local patch residual estimates."""
    base = np.asarray(texture, dtype=np.float32)
    out = base.copy()
    count_grid = np.asarray(counts, dtype=np.int64)
    h, w = count_grid.shape
    patch_radii = sorted({int(radius) for radius in block_sizes if int(radius) > 0})
    summary = {
        "enabled": bool(patch_radii),
        "block_sizes": [int(radius) for radius in patch_radii],
        "patch_radii": [int(radius) for radius in patch_radii],
        "total_bins": int(h * w),
        "low_support_bins": 0,
        "blended_bins": 0,
        "mean_blend_weight": 0.0,
        "max_blend_weight": 0.0,
        "gate_mode": str(gate_mode),
        "min_prior_weight": float(min_prior_weight),
        "min_direct_samples": int(min_direct_samples),
        "min_sign_consistency": float(min_sign_consistency),
        "max_mean_variance": float(max_mean_variance),
        "min_cosine": float(min_cosine),
        "gate_rejected_bins": 0,
        "empty_rejected_bins": 0,
        "prior_weight_rejected_bins": 0,
        "sign_rejected_bins": 0,
        "variance_rejected_bins": 0,
        "cosine_rejected_bins": 0,
    }
    if not patch_radii:
        return out, summary

    sum_grid_f64 = np.asarray(sum_grid, dtype=np.float64)
    prior = np.repeat(np.asarray(face_mean, dtype=np.float32).reshape(1, 1, 3), h, axis=0)
    prior = np.repeat(prior, w, axis=1)
    prior_weight = np.zeros((h, w), dtype=np.float32)
    tau = max(0.0, float(count_tau))
    for radius in patch_radii:
        scale_bonus = 1.0 / math.sqrt(float(2 * radius + 1))
        for y in range(h):
            y0 = max(0, y - radius)
            y1 = min(h, y + radius + 1)
            for x in range(w):
                x0 = max(0, x - radius)
                x1 = min(w, x + radius + 1)
                patch_count = int(np.sum(count_grid[y0:y1, x0:x1]))
                if patch_count <= 0:
                    continue
                patch_sum = np.sum(sum_grid_f64[y0:y1, x0:x1], axis=(0, 1))
                patch_mean = (patch_sum / float(patch_count)).astype(np.float32)
                count_conf = (
                    float(patch_count) / (float(patch_count) + tau)
                    if tau > 0.0
                    else 1.0
                )
                weight = float(np.clip(count_conf * scale_bonus, 0.0, 1.0))
                if weight <= float(prior_weight[y, x]):
                    continue
                prior[y, x] = patch_mean
                prior_weight[y, x] = weight

    min_samples = max(1, int(min_bin_samples))
    low_support = count_grid < min_samples
    if not bool(np.any(low_support)):
        return out, summary
    direct_conf = np.ones((h, w), dtype=np.float32)
    direct_conf[low_support] = count_grid[low_support].astype(np.float32) / float(min_samples)
    gate_mode_name = str(gate_mode)
    if gate_mode_name not in {"none", "evidence_consistent"}:
        raise ValueError(f"unsupported surface multiscale prior gate mode: {gate_mode}")
    blend_weight = float(np.clip(blend, 0.0, 1.0)) * (1.0 - np.clip(direct_conf, 0.0, 1.0))
    blend_weight *= prior_weight
    active = low_support & (blend_weight > 0.0)
    if gate_mode_name == "evidence_consistent" and bool(np.any(active)):
        pre_gate = active.copy()
        observed = count_grid >= max(1, int(min_direct_samples))
        active &= observed
        summary["empty_rejected_bins"] = int(np.sum(pre_gate & ~observed))

        prior_weight_ok = prior_weight >= max(0.0, float(min_prior_weight))
        active &= prior_weight_ok
        summary["prior_weight_rejected_bins"] = int(np.sum(pre_gate & ~prior_weight_ok))

        if sign_consistency is not None and float(min_sign_consistency) > 0.0:
            sign_mean = np.mean(np.asarray(sign_consistency, dtype=np.float32), axis=2)
            sign_ok = sign_mean >= float(min_sign_consistency)
            active &= sign_ok
            summary["sign_rejected_bins"] = int(np.sum(pre_gate & ~sign_ok))

        if variance is not None and float(max_mean_variance) >= 0.0:
            var_mean = np.mean(np.asarray(variance, dtype=np.float32), axis=2)
            var_ok = var_mean <= float(max_mean_variance)
            active &= var_ok
            summary["variance_rejected_bins"] = int(np.sum(pre_gate & ~var_ok))

        if float(min_cosine) > -1.0:
            base_vec = base.astype(np.float32)
            prior_vec = prior.astype(np.float32)
            numerator = np.sum(base_vec * prior_vec, axis=2)
            denom = np.linalg.norm(base_vec, axis=2) * np.linalg.norm(prior_vec, axis=2)
            cosine = np.full((h, w), -1.0, dtype=np.float32)
            valid_cosine = denom > 1.0e-8
            cosine[valid_cosine] = numerator[valid_cosine] / denom[valid_cosine]
            cosine_ok = cosine >= float(min_cosine)
            active &= cosine_ok
            summary["cosine_rejected_bins"] = int(np.sum(pre_gate & ~cosine_ok))
        summary["gate_rejected_bins"] = int(np.sum(pre_gate & ~active))
    if bool(np.any(active)):
        out[active] = (
            (1.0 - blend_weight[active, None]) * base[active]
            + blend_weight[active, None] * prior[active]
        ).astype(np.float32)
    summary.update(
        {
            "low_support_bins": int(np.sum(low_support)),
            "blended_bins": int(np.sum(active)),
            "mean_blend_weight": float(np.mean(blend_weight[active])) if bool(np.any(active)) else 0.0,
            "max_blend_weight": float(np.max(blend_weight[active])) if bool(np.any(active)) else 0.0,
        }
    )
    return out, summary


def fit_atlas(
    view_paths: list[Path],
    candidate_faces: set[int],
    residual_rgb_key: str,
    residual_l1_key: str,
    texture_size: int,
    policy_val_stride: int,
    min_l1: float,
    min_alpha: float,
    max_samples_per_view: int,
    fill_empty_with_face_mean: bool,
    atlas_empty_bin_fill_mode: str,
    atlas_nearest_fill_max_steps: int,
    atlas_nearest_fill_decay: float,
    atlas_lowpass_passes: int,
    atlas_lowpass_neighbor_min_count: int,
    surface_multiscale_prior_mode: str,
    surface_multiscale_prior_block_sizes: list[int],
    surface_multiscale_prior_min_bin_samples: int,
    surface_multiscale_prior_count_tau: float,
    surface_multiscale_prior_blend: float,
    surface_multiscale_prior_gate_mode: str,
    surface_multiscale_prior_min_prior_weight: float,
    surface_multiscale_prior_min_direct_samples: int,
    surface_multiscale_prior_min_sign_consistency: float,
    surface_multiscale_prior_max_mean_variance: float,
    surface_multiscale_prior_min_cosine: float,
    view_conditioned_basis_mode: str,
    view_conditioned_basis_min_bin_samples: int,
    view_conditioned_basis_ridge: float,
    view_conditioned_basis_ood_mode: str,
    view_conditioned_basis_ood_max_z: float,
    view_conditioned_basis_ood_min_std: float,
    view_cluster_expert_count: int,
    view_cluster_feature_mode: str,
    view_cluster_min_views: int,
    view_cluster_min_bin_samples: int,
    view_cluster_fallback_mode: str,
    teacher_residual_target_mode: str,
    teacher_residual_target_luma_mix: float,
    teacher_residual_target_edge_boost: float,
    teacher_distilled_basis_mode: str,
    teacher_distilled_basis_min_face_samples: int,
    teacher_distilled_basis_ridge: float,
    teacher_distilled_basis_ood_max_z: float,
    teacher_distilled_basis_ood_min_std: float,
    teacher_distilled_basis_apply_mode: str,
    teacher_distilled_basis_blend: float,
    teacher_distilled_low_rank_texture_rank: int,
    enable_adaptive_low_support_teacher_basis: bool,
    adaptive_teacher_basis_min_face_samples_floor: int,
    adaptive_teacher_basis_support_quantile: float,
    adaptive_teacher_basis_low_support_ridge_scale: float,
) -> tuple[dict[int, FaceAtlas], dict[str, Any], list[Path], list[Path]]:
    sums: dict[int, np.ndarray] = {}
    sq_sums: dict[int, np.ndarray] = {}
    sign_sums: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}
    mean_sums: dict[int, np.ndarray] = {}
    mean_counts: dict[int, int] = {}
    view_xtx_sums: dict[int, np.ndarray] = {}
    view_xty_sums: dict[int, np.ndarray] = {}
    view_feature_counts: dict[int, np.ndarray] = {}
    view_feature_sums: dict[int, np.ndarray] = {}
    view_feature_sq_sums: dict[int, np.ndarray] = {}
    expert_sums: dict[int, np.ndarray] = {}
    expert_sq_sums: dict[int, np.ndarray] = {}
    expert_sign_sums: dict[int, np.ndarray] = {}
    expert_counts: dict[int, np.ndarray] = {}
    expert_sample_counts: dict[int, np.ndarray] = {}
    teacher_basis_xtx_sums: dict[int, np.ndarray] = {}
    teacher_basis_xty_sums: dict[int, np.ndarray] = {}
    teacher_basis_feature_sums: dict[int, np.ndarray] = {}
    teacher_basis_feature_sq_sums: dict[int, np.ndarray] = {}
    teacher_basis_counts: dict[int, int] = {}
    teacher_texture_xtx_sums: dict[int, np.ndarray] = {}
    teacher_texture_xty_sums: dict[int, np.ndarray] = {}
    teacher_texture_counts: dict[int, np.ndarray] = {}
    size = int(texture_size)
    fit_views: list[Path] = []
    val_views: list[Path] = []
    total_fit_samples = 0
    total_view_basis_samples = 0
    view_basis_views = 0
    multiscale_prior_mode = str(surface_multiscale_prior_mode)
    if multiscale_prior_mode not in {"none", "count_pyramid", "local_patch"}:
        raise ValueError(f"unsupported surface multiscale prior mode: {multiscale_prior_mode}")
    multiscale_prior_faces = 0
    multiscale_prior_low_support_bins = 0
    multiscale_prior_blended_bins = 0
    multiscale_prior_total_bins = 0
    multiscale_prior_blend_weight_sum = 0.0
    multiscale_prior_max_blend_weight = 0.0
    multiscale_prior_gate_rejected_bins = 0
    multiscale_prior_empty_rejected_bins = 0
    multiscale_prior_prior_weight_rejected_bins = 0
    multiscale_prior_sign_rejected_bins = 0
    multiscale_prior_variance_rejected_bins = 0
    multiscale_prior_cosine_rejected_bins = 0
    view_basis_mode = str(view_conditioned_basis_mode)
    if view_basis_mode not in {"none", "camera_center_linear", "normal_camera_linear"}:
        raise ValueError(f"unsupported view-conditioned basis mode: {view_basis_mode}")
    view_basis_ood_mode = str(view_conditioned_basis_ood_mode)
    if view_basis_ood_mode not in {"none", "diag_z"}:
        raise ValueError(f"unsupported view-conditioned basis OOD mode: {view_basis_ood_mode}")
    view_basis_feature_dim = _view_condition_feature_dim(view_basis_mode)
    view_cluster_feature_mode_s = str(view_cluster_feature_mode)
    if view_cluster_feature_mode_s not in {"none", "camera_center"}:
        raise ValueError(f"unsupported view-cluster feature mode: {view_cluster_feature_mode_s}")
    view_cluster_fallback_mode_s = str(view_cluster_fallback_mode)
    if view_cluster_fallback_mode_s not in {"global"}:
        raise ValueError(f"unsupported view-cluster fallback mode: {view_cluster_fallback_mode_s}")
    teacher_basis_mode = str(teacher_distilled_basis_mode)
    teacher_basis_feature_dim = _teacher_distilled_basis_feature_dim(teacher_basis_mode)
    teacher_basis_apply_mode = str(teacher_distilled_basis_apply_mode)
    if teacher_basis_apply_mode not in {"replace_supported", "blend", "fill_empty_only"}:
        raise ValueError(f"unsupported teacher-distilled basis apply mode: {teacher_basis_apply_mode}")
    teacher_basis_views = 0
    teacher_basis_samples = 0
    teacher_basis_base_min_face_samples = max(1, int(teacher_distilled_basis_min_face_samples))
    residual_target_mode = str(teacher_residual_target_mode or "raw_rgb")
    if residual_target_mode not in {"raw_rgb", "luma_only", "edge_luma_mix"}:
        raise ValueError(f"unsupported teacher residual target mode: {residual_target_mode}")
    residual_target_sample_count = 0
    residual_target_mix_sum = 0.0
    residual_target_mix_min = float("inf")
    residual_target_mix_max = 0.0
    residual_target_energy_before_sum = 0.0
    residual_target_energy_after_sum = 0.0
    rng = np.random.default_rng(7)

    stride = max(0, int(policy_val_stride))
    planned_fit_view_paths = [
        path
        for view_index, path in enumerate(view_paths)
        if not (stride > 1 and view_index % stride == 0)
    ]
    view_cluster_profile = _fit_view_cluster_profile(
        planned_fit_view_paths,
        expert_count=int(view_cluster_expert_count),
        feature_mode=view_cluster_feature_mode_s,
        min_views=int(view_cluster_min_views),
    )
    view_cluster_enabled = bool(view_cluster_profile.get("enabled", False))
    view_cluster_centers = (
        np.asarray(view_cluster_profile.get("centers"), dtype=np.float32)
        if view_cluster_enabled
        else None
    )
    view_cluster_count = int(view_cluster_profile.get("expert_count", 1)) if view_cluster_enabled else 1
    view_cluster_valid = (
        np.asarray(view_cluster_profile.get("valid_clusters", []), dtype=bool)
        if view_cluster_enabled
        else np.zeros((0,), dtype=bool)
    )
    view_cluster_views_with_features = 0
    view_cluster_samples = 0
    for view_index, path in enumerate(tqdm(view_paths, desc="fit atlas")):
        if stride > 1 and view_index % stride == 0:
            val_views.append(path)
            continue
        fit_views.append(path)
        z = np.load(path)
        if residual_rgb_key not in z:
            raise KeyError(f"{path} missing {residual_rgb_key}")
        mask = _valid_sample_mask(z, candidate_faces, residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            local_mask = np.zeros_like(mask, dtype=bool)
            local_mask[ys, xs] = True
            mask = local_mask
        face_ids = np.asarray(z["face_id"], dtype=np.int64)[mask]
        residual = np.asarray(z[residual_rgb_key], dtype=np.float32)
        residual_samples = np.stack([residual[0][mask], residual[1][mask], residual[2][mask]], axis=1)
        residual_samples, residual_target_summary = transform_residual_samples_for_fit(
            z,
            mask,
            residual_samples,
            residual_target_mode,
            float(teacher_residual_target_luma_mix),
            float(teacher_residual_target_edge_boost),
        )
        transformed_count = int(residual_target_summary.get("sample_count", residual_samples.shape[0]) or 0)
        if transformed_count > 0:
            residual_target_sample_count += int(transformed_count)
            residual_target_mix_sum += (
                float(residual_target_summary.get("mean_luma_mix", 0.0) or 0.0)
                * float(transformed_count)
            )
            residual_target_mix_min = min(
                residual_target_mix_min,
                float(residual_target_summary.get("min_luma_mix", 0.0) or 0.0),
            )
            residual_target_mix_max = max(
                residual_target_mix_max,
                float(residual_target_summary.get("max_luma_mix", 0.0) or 0.0),
            )
            residual_target_energy_before_sum += (
                float(residual_target_summary.get("mean_rgb_energy_before", 0.0) or 0.0)
                * float(transformed_count)
            )
            residual_target_energy_after_sum += (
                float(residual_target_summary.get("mean_rgb_energy_after", 0.0) or 0.0)
                * float(transformed_count)
            )
        ubin, vbin = _uv_bins(np.asarray(z["barycentric"], dtype=np.float32), mask, size)
        total_fit_samples += int(face_ids.size)
        view_features = _view_condition_features_for_mask(z, view_basis_mode, mask)
        if view_features is not None:
            view_basis_views += 1
            total_view_basis_samples += int(face_ids.size)
        view_cluster_index = None
        if view_cluster_enabled:
            view_cluster_index = _assign_view_cluster_for_npz(
                z,
                centers=view_cluster_centers,
                feature_mode=view_cluster_feature_mode_s,
            )
            if (
                view_cluster_index is not None
                and 0 <= int(view_cluster_index) < int(view_cluster_count)
                and (
                    view_cluster_valid.size <= int(view_cluster_index)
                    or bool(view_cluster_valid[int(view_cluster_index)])
                )
            ):
                view_cluster_views_with_features += 1
                view_cluster_samples += int(face_ids.size)
            else:
                view_cluster_index = None
        teacher_basis_features = _teacher_distilled_basis_features_for_mask(z, teacher_basis_mode, mask)
        if teacher_basis_features is not None:
            teacher_basis_views += 1
            teacher_basis_samples += int(face_ids.size)

        for face in np.unique(face_ids):
            face = int(face)
            fm = face_ids == face
            if face not in sums:
                sums[face] = np.zeros((size, size, 3), dtype=np.float64)
                sq_sums[face] = np.zeros((size, size, 3), dtype=np.float64)
                sign_sums[face] = np.zeros((size, size, 3), dtype=np.float64)
                counts[face] = np.zeros((size, size), dtype=np.int64)
                mean_sums[face] = np.zeros((3,), dtype=np.float64)
                mean_counts[face] = 0
                if view_basis_feature_dim > 0:
                    view_xtx_sums[face] = np.zeros(
                        (size, size, view_basis_feature_dim, view_basis_feature_dim),
                        dtype=np.float64,
                    )
                    view_xty_sums[face] = np.zeros((size, size, view_basis_feature_dim, 3), dtype=np.float64)
                    view_feature_counts[face] = np.zeros((size, size), dtype=np.int64)
                    view_feature_sums[face] = np.zeros((size, size, view_basis_feature_dim), dtype=np.float64)
                    view_feature_sq_sums[face] = np.zeros((size, size, view_basis_feature_dim), dtype=np.float64)
                if view_cluster_enabled:
                    expert_sums[face] = np.zeros((view_cluster_count, size, size, 3), dtype=np.float64)
                    expert_sq_sums[face] = np.zeros((view_cluster_count, size, size, 3), dtype=np.float64)
                    expert_sign_sums[face] = np.zeros((view_cluster_count, size, size, 3), dtype=np.float64)
                    expert_counts[face] = np.zeros((view_cluster_count, size, size), dtype=np.int64)
                    expert_sample_counts[face] = np.zeros((view_cluster_count,), dtype=np.int64)
                if teacher_basis_feature_dim > 0:
                    teacher_basis_xtx_sums[face] = np.zeros(
                        (teacher_basis_feature_dim, teacher_basis_feature_dim),
                        dtype=np.float64,
                    )
                    teacher_basis_xty_sums[face] = np.zeros((teacher_basis_feature_dim, 3), dtype=np.float64)
                    teacher_basis_feature_sums[face] = np.zeros((teacher_basis_feature_dim,), dtype=np.float64)
                    teacher_basis_feature_sq_sums[face] = np.zeros((teacher_basis_feature_dim,), dtype=np.float64)
                    teacher_basis_counts[face] = 0
                    if _is_low_rank_teacher_texture_mode(teacher_basis_mode):
                        teacher_texture_xtx_sums[face] = np.zeros(
                            (size, size, teacher_basis_feature_dim, teacher_basis_feature_dim),
                            dtype=np.float64,
                        )
                        teacher_texture_xty_sums[face] = np.zeros(
                            (size, size, teacher_basis_feature_dim, 3),
                            dtype=np.float64,
                        )
                        teacher_texture_counts[face] = np.zeros((size, size), dtype=np.int64)
            np.add.at(sums[face], (vbin[fm], ubin[fm]), residual_samples[fm].astype(np.float64))
            np.add.at(sq_sums[face], (vbin[fm], ubin[fm]), np.square(residual_samples[fm]).astype(np.float64))
            np.add.at(sign_sums[face], (vbin[fm], ubin[fm]), np.sign(residual_samples[fm]).astype(np.float64))
            np.add.at(counts[face], (vbin[fm], ubin[fm]), 1)
            mean_sums[face] += residual_samples[fm].sum(axis=0)
            mean_counts[face] += int(fm.sum())
            if view_cluster_index is not None and face in expert_sums:
                ci = int(view_cluster_index)
                np.add.at(expert_sums[face][ci], (vbin[fm], ubin[fm]), residual_samples[fm].astype(np.float64))
                np.add.at(
                    expert_sq_sums[face][ci],
                    (vbin[fm], ubin[fm]),
                    np.square(residual_samples[fm]).astype(np.float64),
                )
                np.add.at(
                    expert_sign_sums[face][ci],
                    (vbin[fm], ubin[fm]),
                    np.sign(residual_samples[fm]).astype(np.float64),
                )
                np.add.at(expert_counts[face][ci], (vbin[fm], ubin[fm]), 1)
                expert_sample_counts[face][ci] += int(fm.sum())
            if view_features is not None and face in view_xtx_sums:
                face_features = view_features[fm].astype(np.float64)
                np.add.at(view_feature_counts[face], (vbin[fm], ubin[fm]), 1)
                for i in range(view_basis_feature_dim):
                    np.add.at(
                        view_feature_sums[face][..., i],
                        (vbin[fm], ubin[fm]),
                        face_features[:, i],
                    )
                    np.add.at(
                        view_feature_sq_sums[face][..., i],
                        (vbin[fm], ubin[fm]),
                        face_features[:, i] * face_features[:, i],
                    )
                for i in range(view_basis_feature_dim):
                    for j in range(view_basis_feature_dim):
                        np.add.at(
                            view_xtx_sums[face][..., i, j],
                            (vbin[fm], ubin[fm]),
                            face_features[:, i] * face_features[:, j],
                        )
                    for c in range(3):
                        np.add.at(
                            view_xty_sums[face][..., i, c],
                            (vbin[fm], ubin[fm]),
                            face_features[:, i] * residual_samples[fm, c].astype(np.float64),
                        )
            if teacher_basis_features is not None and face in teacher_basis_xtx_sums:
                face_features = teacher_basis_features[fm].astype(np.float64)
                face_targets = residual_samples[fm].astype(np.float64)
                teacher_basis_feature_sums[face] += np.sum(face_features, axis=0)
                teacher_basis_feature_sq_sums[face] += np.sum(face_features * face_features, axis=0)
                teacher_basis_counts[face] = teacher_basis_counts.get(face, 0) + int(face_features.shape[0])
                if _is_low_rank_teacher_texture_mode(teacher_basis_mode) and face in teacher_texture_xtx_sums:
                    np.add.at(teacher_texture_counts[face], (vbin[fm], ubin[fm]), 1)
                    for i in range(teacher_basis_feature_dim):
                        for j in range(teacher_basis_feature_dim):
                            np.add.at(
                                teacher_texture_xtx_sums[face][..., i, j],
                                (vbin[fm], ubin[fm]),
                                face_features[:, i] * face_features[:, j],
                            )
                        for c in range(3):
                            np.add.at(
                                teacher_texture_xty_sums[face][..., i, c],
                                (vbin[fm], ubin[fm]),
                                face_features[:, i] * face_targets[:, c],
                            )
                else:
                    teacher_basis_xtx_sums[face] += face_features.T @ face_features
                    teacher_basis_xty_sums[face] += face_features.T @ face_targets

    atlas: dict[int, FaceAtlas] = {}
    view_cluster_expert_faces = 0
    view_cluster_expert_supported_bins = 0
    view_cluster_expert_total_bins = 0
    view_basis_supported_bins_total = 0
    view_basis_bins_total = 0
    teacher_basis_supported_faces = 0
    teacher_basis_candidate_faces = 0
    teacher_basis_base_threshold_supported_faces = 0
    teacher_basis_low_support_supported_faces = 0
    teacher_basis_ridge_multiplier_sum = 0.0
    teacher_basis_ridge_multiplier_max = 1.0
    teacher_texture_supported_faces = 0
    teacher_texture_supported_bins = 0
    teacher_texture_total_bins = 0
    teacher_texture_energy_sum = 0.0
    teacher_texture_energy_count = 0
    teacher_texture_rank_sum = 0
    teacher_texture_rank_max = 0
    teacher_basis_effective_min_face_samples = int(teacher_basis_base_min_face_samples)
    teacher_basis_requested_adaptive_floor = max(1, int(adaptive_teacher_basis_min_face_samples_floor))
    teacher_basis_effective_adaptive_floor = min(
        int(teacher_basis_base_min_face_samples),
        int(teacher_basis_requested_adaptive_floor),
    )
    adaptive_teacher_basis_count_stats: dict[str, Any] = {
        "counted_faces": 0,
        "min": 0,
        "p25": 0.0,
        "median": 0.0,
        "p75": 0.0,
        "max": 0,
    }
    adaptive_teacher_basis_summary: dict[str, Any] = {
        "enabled": bool(enable_adaptive_low_support_teacher_basis),
        "base_min_face_samples": int(teacher_basis_base_min_face_samples),
        "effective_min_face_samples": int(teacher_basis_effective_min_face_samples),
        "floor": int(teacher_basis_effective_adaptive_floor),
        "requested_floor": int(teacher_basis_requested_adaptive_floor),
        "support_quantile": float(adaptive_teacher_basis_support_quantile),
        "low_support_ridge_scale": float(adaptive_teacher_basis_low_support_ridge_scale),
        "max_low_support_ridge_multiplier": 16.0,
        "reason": "not_requested",
        "support_count_stats": dict(adaptive_teacher_basis_count_stats),
        "threshold_candidate_faces_at_base_min": 0,
        "threshold_candidate_faces_at_effective_min": 0,
        "threshold_newly_candidate_faces": 0,
        "supported_faces_at_base_min": 0,
        "supported_faces_at_effective_min": 0,
        "newly_supported_faces": 0,
        "low_support_supported_faces": 0,
        "mean_ridge_multiplier": 0.0,
        "max_ridge_multiplier": 1.0,
    }
    if teacher_basis_feature_dim > 0 and teacher_basis_counts:
        teacher_count_values = np.asarray(
            [int(count) for count in teacher_basis_counts.values() if int(count) > 0],
            dtype=np.int64,
        )
        if teacher_count_values.size:
            q25, median, q75 = np.quantile(teacher_count_values.astype(np.float64), [0.25, 0.5, 0.75])
            adaptive_teacher_basis_count_stats = {
                "counted_faces": int(teacher_count_values.size),
                "min": int(np.min(teacher_count_values)),
                "p25": float(q25),
                "median": float(median),
                "p75": float(q75),
                "max": int(np.max(teacher_count_values)),
            }
            base_supported = int(np.sum(teacher_count_values >= int(teacher_basis_base_min_face_samples)))
            teacher_basis_effective_min_face_samples = int(teacher_basis_base_min_face_samples)
            adaptive_reason = "disabled"
            if bool(enable_adaptive_low_support_teacher_basis):
                support_quantile = float(np.clip(float(adaptive_teacher_basis_support_quantile), 0.0, 1.0))
                quantile_count = int(math.ceil(float(np.quantile(teacher_count_values.astype(np.float64), support_quantile))))
                teacher_basis_effective_min_face_samples = int(
                    min(
                        int(teacher_basis_base_min_face_samples),
                        max(1, int(teacher_basis_effective_adaptive_floor), quantile_count),
                    )
                )
                adaptive_reason = (
                    "lowered_by_fit_support_quantile"
                    if teacher_basis_effective_min_face_samples < int(teacher_basis_base_min_face_samples)
                    else "support_distribution_kept_base_threshold"
                )
            effective_supported = int(np.sum(teacher_count_values >= int(teacher_basis_effective_min_face_samples)))
            adaptive_teacher_basis_summary = {
                "enabled": bool(enable_adaptive_low_support_teacher_basis),
                "base_min_face_samples": int(teacher_basis_base_min_face_samples),
                "effective_min_face_samples": int(teacher_basis_effective_min_face_samples),
                "floor": int(teacher_basis_effective_adaptive_floor),
                "requested_floor": int(teacher_basis_requested_adaptive_floor),
                "support_quantile": float(adaptive_teacher_basis_support_quantile),
                "low_support_ridge_scale": float(adaptive_teacher_basis_low_support_ridge_scale),
                "max_low_support_ridge_multiplier": 16.0,
                "reason": str(adaptive_reason if bool(enable_adaptive_low_support_teacher_basis) else "not_requested"),
                "support_count_stats": dict(adaptive_teacher_basis_count_stats),
                "threshold_candidate_faces_at_base_min": int(base_supported),
                "threshold_candidate_faces_at_effective_min": int(effective_supported),
                "threshold_newly_candidate_faces": int(max(0, effective_supported - base_supported)),
                "supported_faces_at_base_min": 0,
                "supported_faces_at_effective_min": 0,
                "newly_supported_faces": 0,
                "low_support_supported_faces": 0,
                "mean_ridge_multiplier": 0.0,
                "max_ridge_multiplier": 1.0,
            }
        elif bool(enable_adaptive_low_support_teacher_basis):
            adaptive_teacher_basis_summary["reason"] = "no_positive_teacher_basis_counts"
    elif bool(enable_adaptive_low_support_teacher_basis):
        adaptive_teacher_basis_summary["reason"] = "teacher_basis_disabled_or_no_candidates"
    for face, sum_grid in sums.items():
        count_grid = counts[face]
        mean_rgb = mean_sums[face] / max(1, int(mean_counts[face]))
        texture = np.zeros_like(sum_grid, dtype=np.float32)
        variance = np.zeros_like(sum_grid, dtype=np.float32)
        sign_consistency = np.zeros_like(sum_grid, dtype=np.float32)
        nonzero = count_grid > 0
        texture[nonzero] = (sum_grid[nonzero] / count_grid[nonzero, None]).astype(np.float32)
        mean_sq = np.zeros_like(sum_grid, dtype=np.float64)
        mean_sq[nonzero] = sq_sums[face][nonzero] / count_grid[nonzero, None]
        variance[nonzero] = np.maximum(mean_sq[nonzero] - np.square(texture[nonzero].astype(np.float64)), 0.0).astype(np.float32)
        sign_consistency[nonzero] = (
            np.abs(sign_sums[face][nonzero]) / np.maximum(count_grid[nonzero, None].astype(np.float64), 1.0)
        ).astype(np.float32)
        fill_mode = str(atlas_empty_bin_fill_mode)
        if fill_mode == "face_mean" and fill_empty_with_face_mean:
            texture[~nonzero] = mean_rgb.astype(np.float32)
        elif fill_mode == "nearest_observed" and fill_empty_with_face_mean:
            texture = _nearest_observed_fill_texture(
                texture,
                count_grid,
                mean_rgb.astype(np.float32),
                max_steps=int(atlas_nearest_fill_max_steps),
                decay=float(atlas_nearest_fill_decay),
            )
        if multiscale_prior_mode in {"count_pyramid", "local_patch"}:
            prior_builder = (
                _count_pyramid_prior_texture
                if multiscale_prior_mode == "count_pyramid"
                else _local_patch_prior_texture
            )
            texture, ms_summary = prior_builder(
                texture,
                sum_grid=sum_grid,
                counts=count_grid,
                face_mean=mean_rgb.astype(np.float32),
                block_sizes=surface_multiscale_prior_block_sizes,
                min_bin_samples=int(surface_multiscale_prior_min_bin_samples),
                count_tau=float(surface_multiscale_prior_count_tau),
                blend=float(surface_multiscale_prior_blend),
                variance=variance,
                sign_consistency=sign_consistency,
                gate_mode=str(surface_multiscale_prior_gate_mode),
                min_prior_weight=float(surface_multiscale_prior_min_prior_weight),
                min_direct_samples=int(surface_multiscale_prior_min_direct_samples),
                min_sign_consistency=float(surface_multiscale_prior_min_sign_consistency),
                max_mean_variance=float(surface_multiscale_prior_max_mean_variance),
                min_cosine=float(surface_multiscale_prior_min_cosine),
            )
            multiscale_prior_faces += int(ms_summary.get("enabled", False))
            multiscale_prior_low_support_bins += int(ms_summary.get("low_support_bins", 0))
            blended_bins = int(ms_summary.get("blended_bins", 0))
            multiscale_prior_blended_bins += blended_bins
            multiscale_prior_total_bins += int(ms_summary.get("total_bins", 0))
            multiscale_prior_blend_weight_sum += float(ms_summary.get("mean_blend_weight", 0.0)) * blended_bins
            multiscale_prior_max_blend_weight = max(
                multiscale_prior_max_blend_weight,
                float(ms_summary.get("max_blend_weight", 0.0)),
            )
            multiscale_prior_gate_rejected_bins += int(ms_summary.get("gate_rejected_bins", 0))
            multiscale_prior_empty_rejected_bins += int(ms_summary.get("empty_rejected_bins", 0))
            multiscale_prior_prior_weight_rejected_bins += int(ms_summary.get("prior_weight_rejected_bins", 0))
            multiscale_prior_sign_rejected_bins += int(ms_summary.get("sign_rejected_bins", 0))
            multiscale_prior_variance_rejected_bins += int(ms_summary.get("variance_rejected_bins", 0))
            multiscale_prior_cosine_rejected_bins += int(ms_summary.get("cosine_rejected_bins", 0))
        texture = _lowpass_texture(
            texture,
            count_grid,
            passes=int(atlas_lowpass_passes),
            neighbor_min_count=max(1, int(atlas_lowpass_neighbor_min_count)),
        )
        view_basis_coefficients = None
        view_basis_support = None
        view_basis_feature_mean = None
        view_basis_feature_std = None
        expert_textures = None
        expert_count_grid = None
        expert_variance_grid = None
        expert_sign_grid = None
        expert_samples = None
        if view_cluster_enabled and face in expert_sums:
            min_expert_bin_samples = max(1, int(view_cluster_min_bin_samples))
            expert_count_grid = expert_counts[face].astype(np.int32)
            expert_textures = np.repeat(texture[None, ...], view_cluster_count, axis=0).astype(np.float32)
            expert_variance_grid = np.repeat(variance[None, ...], view_cluster_count, axis=0).astype(np.float32)
            expert_sign_grid = np.repeat(sign_consistency[None, ...], view_cluster_count, axis=0).astype(np.float32)
            expert_samples = expert_sample_counts[face].astype(np.int64)
            for expert_idx in range(view_cluster_count):
                local_counts = expert_counts[face][expert_idx]
                local_nonzero = local_counts > 0
                if not bool(np.any(local_nonzero)):
                    continue
                local_mean = np.zeros_like(texture, dtype=np.float32)
                local_mean[local_nonzero] = (
                    expert_sums[face][expert_idx][local_nonzero] / local_counts[local_nonzero, None]
                ).astype(np.float32)
                local_mean_sq = np.zeros_like(texture, dtype=np.float64)
                local_mean_sq[local_nonzero] = (
                    expert_sq_sums[face][expert_idx][local_nonzero]
                    / local_counts[local_nonzero, None]
                )
                local_variance = np.zeros_like(texture, dtype=np.float32)
                local_variance[local_nonzero] = np.maximum(
                    local_mean_sq[local_nonzero] - np.square(local_mean[local_nonzero].astype(np.float64)),
                    0.0,
                ).astype(np.float32)
                local_sign = np.zeros_like(texture, dtype=np.float32)
                local_sign[local_nonzero] = (
                    np.abs(expert_sign_sums[face][expert_idx][local_nonzero])
                    / np.maximum(local_counts[local_nonzero, None].astype(np.float64), 1.0)
                ).astype(np.float32)
                supported = local_counts >= int(min_expert_bin_samples)
                if bool(np.any(supported)):
                    expert_textures[expert_idx][supported] = local_mean[supported]
                    expert_variance_grid[expert_idx][supported] = local_variance[supported]
                    expert_sign_grid[expert_idx][supported] = local_sign[supported]
                    view_cluster_expert_supported_bins += int(np.sum(supported))
            view_cluster_expert_faces += 1
            view_cluster_expert_total_bins += int(view_cluster_count * size * size)
        if view_basis_feature_dim > 0 and face in view_xtx_sums:
            feature_count_grid = view_feature_counts[face]
            support = feature_count_grid >= max(1, int(view_conditioned_basis_min_bin_samples))
            coeffs = np.zeros((size, size, view_basis_feature_dim, 3), dtype=np.float32)
            feature_mean = np.zeros((size, size, view_basis_feature_dim), dtype=np.float32)
            feature_std = np.ones((size, size, view_basis_feature_dim), dtype=np.float32)
            feature_count = np.maximum(feature_count_grid[..., None].astype(np.float64), 1.0)
            feature_mean = (view_feature_sums[face] / feature_count).astype(np.float32)
            feature_var = view_feature_sq_sums[face] / feature_count - np.square(feature_mean.astype(np.float64))
            feature_std = np.sqrt(np.maximum(feature_var, 0.0)).astype(np.float32)
            ridge = max(0.0, float(view_conditioned_basis_ridge))
            eye = np.eye(view_basis_feature_dim, dtype=np.float64)
            ys, xs = np.nonzero(support)
            for y, x in zip(ys, xs, strict=False):
                xtx = view_xtx_sums[face][int(y), int(x)] + ridge * eye
                xty = view_xty_sums[face][int(y), int(x)]
                try:
                    coeffs[int(y), int(x)] = np.linalg.solve(xtx, xty).astype(np.float32)
                except np.linalg.LinAlgError:
                    support[int(y), int(x)] = False
            view_basis_coefficients = coeffs
            view_basis_support = support.astype(bool)
            view_basis_feature_mean = feature_mean
            view_basis_feature_std = feature_std
            view_basis_supported_bins_total += int(np.sum(view_basis_support))
            view_basis_bins_total += int(view_basis_support.size)
        teacher_basis_coefficients = None
        teacher_basis_feature_mean = None
        teacher_basis_feature_std = None
        teacher_texture_basis = None
        teacher_texture_support = None
        teacher_texture_energy = None
        if teacher_basis_feature_dim > 0 and face in teacher_basis_xtx_sums:
            teacher_basis_candidate_faces += 1
            teacher_count = int(teacher_basis_counts.get(face, 0))
            if teacher_count >= max(1, int(teacher_basis_effective_min_face_samples)):
                ridge = max(0.0, float(teacher_distilled_basis_ridge))
                ridge_multiplier = 1.0
                if bool(enable_adaptive_low_support_teacher_basis) and teacher_count < int(teacher_basis_base_min_face_samples):
                    ratio = float(teacher_basis_base_min_face_samples) / max(1.0, float(teacher_count))
                    ridge_multiplier = min(
                        16.0,
                        1.0 + max(0.0, float(adaptive_teacher_basis_low_support_ridge_scale)) * max(0.0, ratio - 1.0),
                    )
                    ridge *= float(ridge_multiplier)
                try:
                    denom = max(1, int(teacher_count))
                    teacher_basis_feature_mean = (
                        teacher_basis_feature_sums[face] / float(denom)
                    ).astype(np.float32)
                    feature_var = (
                        teacher_basis_feature_sq_sums[face] / float(denom)
                        - np.square(teacher_basis_feature_mean.astype(np.float64))
                    )
                    teacher_basis_feature_std = np.sqrt(np.maximum(feature_var, 0.0)).astype(np.float32)
                    if (
                        _is_low_rank_teacher_texture_mode(teacher_basis_mode)
                        and face in teacher_texture_xtx_sums
                        and face in teacher_texture_xty_sums
                        and face in teacher_texture_counts
                    ):
                        eye = np.eye(teacher_basis_feature_dim, dtype=np.float64)
                        support = teacher_texture_counts[face] >= _low_rank_teacher_texture_min_bin_samples(
                            teacher_basis_mode,
                            teacher_basis_feature_dim,
                        )
                        coeff_field = np.zeros((size, size, teacher_basis_feature_dim, 3), dtype=np.float32)
                        ys_lr, xs_lr = np.nonzero(support)
                        for y, x in zip(ys_lr, xs_lr, strict=False):
                            xtx = teacher_texture_xtx_sums[face][int(y), int(x)] + ridge * eye
                            xty = teacher_texture_xty_sums[face][int(y), int(x)]
                            try:
                                coeff_field[int(y), int(x)] = np.linalg.solve(xtx, xty).astype(np.float32)
                            except np.linalg.LinAlgError:
                                support[int(y), int(x)] = False
                        ys_lr, xs_lr = np.nonzero(support)
                        if ys_lr.size <= 0:
                            raise np.linalg.LinAlgError("low_rank_view_texture_k4_no_supported_bins")
                        matrix = coeff_field[ys_lr, xs_lr].transpose(1, 0, 2).reshape(
                            teacher_basis_feature_dim,
                            int(ys_lr.size) * 3,
                        )
                        u_svd, singular_values, vt_svd = np.linalg.svd(
                            matrix.astype(np.float64),
                            full_matrices=False,
                        )
                        requested_rank = _low_rank_teacher_texture_requested_rank(
                            teacher_basis_mode,
                            int(teacher_distilled_low_rank_texture_rank),
                        )
                        rank = min(
                            int(requested_rank),
                            int(teacher_basis_feature_dim),
                            int(singular_values.size),
                        )
                        if rank <= 0:
                            raise np.linalg.LinAlgError("low_rank_view_texture_k4_empty_svd")
                        sqrt_s = np.sqrt(np.maximum(singular_values[:rank], 0.0))
                        teacher_basis_coefficients = (u_svd[:, :rank] * sqrt_s[None, :]).astype(np.float32)
                        basis_flat = (sqrt_s[:, None] * vt_svd[:rank]).astype(np.float32)
                        teacher_texture_basis = np.zeros((rank, size, size, 3), dtype=np.float32)
                        basis_values = basis_flat.reshape(rank, int(ys_lr.size), 3)
                        for idx, (y, x) in enumerate(zip(ys_lr, xs_lr, strict=False)):
                            teacher_texture_basis[:, int(y), int(x), :] = basis_values[:, idx, :]
                        teacher_texture_support = support.astype(bool)
                        total_energy = float(np.sum(singular_values * singular_values))
                        if total_energy > 0.0:
                            cumulative = np.cumsum(singular_values[:rank] * singular_values[:rank]) / total_energy
                            teacher_texture_energy = cumulative.astype(np.float32)
                            teacher_texture_energy_sum += float(cumulative[-1])
                            teacher_texture_energy_count += 1
                        else:
                            teacher_texture_energy = np.zeros((rank,), dtype=np.float32)
                        teacher_texture_supported_faces += 1
                        teacher_texture_supported_bins += int(np.sum(teacher_texture_support))
                        teacher_texture_total_bins += int(teacher_texture_support.size)
                        teacher_texture_rank_sum += int(rank)
                        teacher_texture_rank_max = max(int(teacher_texture_rank_max), int(rank))
                    else:
                        eye = np.eye(teacher_basis_feature_dim, dtype=np.float64)
                        xtx = teacher_basis_xtx_sums[face] + ridge * eye
                        xty = teacher_basis_xty_sums[face]
                        teacher_basis_coefficients = np.linalg.solve(xtx, xty).astype(np.float32)
                    teacher_basis_supported_faces += 1
                    if teacher_count < int(teacher_basis_base_min_face_samples):
                        teacher_basis_low_support_supported_faces += 1
                    else:
                        teacher_basis_base_threshold_supported_faces += 1
                    teacher_basis_ridge_multiplier_sum += float(ridge_multiplier)
                    teacher_basis_ridge_multiplier_max = max(
                        float(teacher_basis_ridge_multiplier_max),
                        float(ridge_multiplier),
                    )
                except np.linalg.LinAlgError:
                    teacher_basis_coefficients = None
                    teacher_texture_basis = None
                    teacher_texture_support = None
                    teacher_texture_energy = None
        atlas[face] = FaceAtlas(
            texture=texture.astype(np.float32),
            counts=count_grid.astype(np.int32),
            variance=variance.astype(np.float32),
            sign_consistency=sign_consistency.astype(np.float32),
            mean_rgb=mean_rgb.astype(np.float32),
            samples=int(mean_counts[face]),
            view_basis_mode=view_basis_mode if view_basis_coefficients is not None else "none",
            view_basis_coefficients=view_basis_coefficients,
            view_basis_support=view_basis_support,
            view_basis_feature_mean=view_basis_feature_mean,
            view_basis_feature_std=view_basis_feature_std,
            view_basis_ood_mode=view_basis_ood_mode if view_basis_coefficients is not None else "none",
            view_basis_ood_max_z=float(view_conditioned_basis_ood_max_z),
            view_basis_ood_min_std=float(view_conditioned_basis_ood_min_std),
            teacher_basis_mode=teacher_basis_mode if teacher_basis_coefficients is not None else "none",
            teacher_basis_coefficients=teacher_basis_coefficients,
            teacher_basis_feature_mean=teacher_basis_feature_mean,
            teacher_basis_feature_std=teacher_basis_feature_std,
            teacher_basis_ood_max_z=float(teacher_distilled_basis_ood_max_z),
            teacher_basis_ood_min_std=float(teacher_distilled_basis_ood_min_std),
            teacher_basis_apply_mode=str(teacher_basis_apply_mode),
            teacher_basis_blend=float(teacher_distilled_basis_blend),
            teacher_texture_basis=teacher_texture_basis,
            teacher_texture_support=teacher_texture_support,
            teacher_texture_energy=teacher_texture_energy,
            expert_textures=expert_textures,
            expert_counts=expert_count_grid,
            expert_variance=expert_variance_grid,
            expert_sign_consistency=expert_sign_grid,
            expert_samples=expert_samples,
            expert_centers=view_cluster_centers.copy() if view_cluster_enabled and view_cluster_centers is not None else None,
            expert_feature_mode=view_cluster_feature_mode_s if view_cluster_enabled else "none",
            expert_min_bin_samples=max(1, int(view_cluster_min_bin_samples)),
            expert_fallback_mode=view_cluster_fallback_mode_s,
        )

    residual_target_before = (
        float(residual_target_energy_before_sum / max(1, residual_target_sample_count))
        if residual_target_sample_count > 0
        else 0.0
    )
    residual_target_after = (
        float(residual_target_energy_after_sum / max(1, residual_target_sample_count))
        if residual_target_sample_count > 0
        else 0.0
    )
    summary = {
        "input_views": int(len(view_paths)),
        "fit_views": int(len(fit_views)),
        "policy_val_views": int(len(val_views)),
        "candidate_faces": int(len(candidate_faces)),
        "atlas_faces": int(len(atlas)),
        "fit_samples": int(total_fit_samples),
        "teacher_residual_target": {
            "mode": str(residual_target_mode),
            "luma_mix": float(teacher_residual_target_luma_mix),
            "edge_boost": float(teacher_residual_target_edge_boost),
            "sample_count": int(residual_target_sample_count),
            "mean_luma_mix": (
                float(residual_target_mix_sum / max(1, residual_target_sample_count))
                if residual_target_sample_count > 0
                else 0.0
            ),
            "min_luma_mix": (
                float(residual_target_mix_min)
                if residual_target_sample_count > 0 and math.isfinite(residual_target_mix_min)
                else 0.0
            ),
            "max_luma_mix": float(residual_target_mix_max) if residual_target_sample_count > 0 else 0.0,
            "mean_rgb_energy_before": residual_target_before,
            "mean_rgb_energy_after": residual_target_after,
            "energy_ratio_after_before": (
                float(residual_target_after / residual_target_before)
                if residual_target_before > 1.0e-12
                else 0.0
            ),
        },
        "texture_size": int(size),
        "fill_empty_with_face_mean": bool(fill_empty_with_face_mean),
        "atlas_empty_bin_fill_mode": str(atlas_empty_bin_fill_mode),
        "atlas_nearest_fill_max_steps": int(atlas_nearest_fill_max_steps),
        "atlas_nearest_fill_decay": float(atlas_nearest_fill_decay),
        "atlas_lowpass_passes": int(atlas_lowpass_passes),
        "atlas_lowpass_neighbor_min_count": int(atlas_lowpass_neighbor_min_count),
        "surface_multiscale_prior": {
            "mode": str(multiscale_prior_mode),
            "block_sizes": [int(x) for x in surface_multiscale_prior_block_sizes],
            "min_bin_samples": int(surface_multiscale_prior_min_bin_samples),
            "count_tau": float(surface_multiscale_prior_count_tau),
            "blend": float(surface_multiscale_prior_blend),
            "gate_mode": str(surface_multiscale_prior_gate_mode),
            "min_prior_weight": float(surface_multiscale_prior_min_prior_weight),
            "min_direct_samples": int(surface_multiscale_prior_min_direct_samples),
            "min_sign_consistency": float(surface_multiscale_prior_min_sign_consistency),
            "max_mean_variance": float(surface_multiscale_prior_max_mean_variance),
            "min_cosine": float(surface_multiscale_prior_min_cosine),
            "faces": int(multiscale_prior_faces),
            "total_bins": int(multiscale_prior_total_bins),
            "low_support_bins": int(multiscale_prior_low_support_bins),
            "blended_bins": int(multiscale_prior_blended_bins),
            "blended_bin_fraction": float(multiscale_prior_blended_bins / max(1, multiscale_prior_total_bins)),
            "mean_blend_weight": float(
                multiscale_prior_blend_weight_sum / max(1, multiscale_prior_blended_bins)
            ),
            "max_blend_weight": float(multiscale_prior_max_blend_weight),
            "gate_rejected_bins": int(multiscale_prior_gate_rejected_bins),
            "empty_rejected_bins": int(multiscale_prior_empty_rejected_bins),
            "prior_weight_rejected_bins": int(multiscale_prior_prior_weight_rejected_bins),
            "sign_rejected_bins": int(multiscale_prior_sign_rejected_bins),
            "variance_rejected_bins": int(multiscale_prior_variance_rejected_bins),
            "cosine_rejected_bins": int(multiscale_prior_cosine_rejected_bins),
        },
        "view_conditioned_basis": {
            "mode": str(view_basis_mode),
            "feature_dim": int(view_basis_feature_dim),
            "fit_views_with_features": int(view_basis_views),
            "fit_samples_with_features": int(total_view_basis_samples),
            "min_bin_samples": int(view_conditioned_basis_min_bin_samples),
            "ridge": float(view_conditioned_basis_ridge),
            "ood_mode": str(view_basis_ood_mode),
            "ood_max_z": float(view_conditioned_basis_ood_max_z),
            "ood_min_std": float(view_conditioned_basis_ood_min_std),
            "supported_bins": int(view_basis_supported_bins_total),
            "total_bins": int(view_basis_bins_total),
            "supported_bin_fraction": float(view_basis_supported_bins_total / max(1, view_basis_bins_total)),
        },
        "view_cluster_experts": {
            "enabled": bool(view_cluster_enabled),
            "feature_mode": str(view_cluster_feature_mode_s),
            "requested_expert_count": int(view_cluster_expert_count),
            "expert_count": int(view_cluster_count),
            "min_views": int(view_cluster_min_views),
            "min_bin_samples": int(view_cluster_min_bin_samples),
            "fallback_mode": str(view_cluster_fallback_mode_s),
            "fit_views_with_features": int(view_cluster_views_with_features),
            "fit_samples_with_features": int(view_cluster_samples),
            "faces": int(view_cluster_expert_faces),
            "supported_bins": int(view_cluster_expert_supported_bins),
            "total_bins": int(view_cluster_expert_total_bins),
            "supported_bin_fraction": float(view_cluster_expert_supported_bins / max(1, view_cluster_expert_total_bins)),
            "profile": {
                key: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value in view_cluster_profile.items()
                if key != "centers"
            }
            | {
                "centers": (
                    view_cluster_centers.astype(float).tolist()
                    if view_cluster_enabled and view_cluster_centers is not None
                    else []
                )
            },
        },
        "teacher_distilled_basis": {
            "mode": str(teacher_basis_mode),
            "feature_dim": int(teacher_basis_feature_dim),
            "fit_views_with_features": int(teacher_basis_views),
            "fit_samples_with_features": int(teacher_basis_samples),
            "min_face_samples": int(teacher_basis_base_min_face_samples),
            "effective_min_face_samples": int(teacher_basis_effective_min_face_samples),
            "ridge": float(teacher_distilled_basis_ridge),
            "ood_max_z": float(teacher_distilled_basis_ood_max_z),
            "ood_min_std": float(teacher_distilled_basis_ood_min_std),
            "apply_mode": str(teacher_basis_apply_mode),
            "blend": float(teacher_distilled_basis_blend),
            "candidate_faces": int(teacher_basis_candidate_faces),
            "supported_faces": int(teacher_basis_supported_faces),
            "supported_face_fraction": float(teacher_basis_supported_faces / max(1, teacher_basis_candidate_faces)),
            "low_rank_texture": {
                "enabled": bool(_is_low_rank_teacher_texture_mode(teacher_basis_mode)),
                "requested_rank": int(
                    _low_rank_teacher_texture_requested_rank(
                        teacher_basis_mode,
                        int(teacher_distilled_low_rank_texture_rank),
                    )
                ),
                "effective_rank_cap": int(
                    min(
                        _low_rank_teacher_texture_requested_rank(
                            teacher_basis_mode,
                            int(teacher_distilled_low_rank_texture_rank),
                        ),
                        int(teacher_basis_feature_dim),
                    )
                    if _is_low_rank_teacher_texture_mode(teacher_basis_mode)
                    else 0
                ),
                "min_bin_samples": int(
                    _low_rank_teacher_texture_min_bin_samples(
                        teacher_basis_mode,
                        teacher_basis_feature_dim,
                    )
                    if _is_low_rank_teacher_texture_mode(teacher_basis_mode)
                    else 0
                ),
                "rank": int(teacher_texture_rank_max),
                "mean_rank": float(teacher_texture_rank_sum / max(1, teacher_texture_supported_faces)),
                "supported_faces": int(teacher_texture_supported_faces),
                "supported_bins": int(teacher_texture_supported_bins),
                "total_bins": int(teacher_texture_total_bins),
                "supported_bin_fraction": float(teacher_texture_supported_bins / max(1, teacher_texture_total_bins)),
                "mean_retained_energy": float(teacher_texture_energy_sum / max(1, teacher_texture_energy_count)),
            },
            "adaptive_low_support": dict(
                adaptive_teacher_basis_summary
                | {
                    "low_support_supported_faces": int(teacher_basis_low_support_supported_faces),
                    "supported_faces_at_base_min": int(teacher_basis_base_threshold_supported_faces),
                    "supported_faces_at_effective_min": int(teacher_basis_supported_faces),
                    "newly_supported_faces": int(teacher_basis_low_support_supported_faces),
                    "mean_ridge_multiplier": float(
                        teacher_basis_ridge_multiplier_sum / max(1, teacher_basis_supported_faces)
                    ),
                    "max_ridge_multiplier": float(teacher_basis_ridge_multiplier_max),
                }
            ),
        },
    }
    return atlas, summary, fit_views, val_views


def predict_delta_for_npz(
    z: np.lib.npyio.NpzFile,
    atlas: dict[int, FaceAtlas],
    alpha: float,
    min_alpha: float,
    min_atlas_bin_count: int = 0,
    min_atlas_face_samples: int = 0,
    max_atlas_bin_rgb_variance: float = -1.0,
    min_atlas_bin_sign_consistency: float = 0.0,
    atlas_confidence_mode: str = "none",
    atlas_confidence_count_scale: float = 0.0,
    atlas_confidence_empty_bin: float = 1.0,
    atlas_confidence_variance_scale: float = -1.0,
    atlas_confidence_sign_power: float = 0.0,
    atlas_confidence_face_sample_scale: float = 0.0,
    min_atlas_confidence: float = 0.0,
    local_alpha_profile: dict[str, Any] | None = None,
    face_gain_guard_profile: dict[str, Any] | None = None,
    bin_uncertainty_guard_profile: dict[str, Any] | None = None,
    parent_edge_apply_profile: dict[str, Any] | None = None,
    view_confidence_profile: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    face_id = np.asarray(z["face_id"], dtype=np.int64)
    h, w = face_id.shape
    delta = np.zeros((3, h, w), dtype=np.float32)
    valid = face_id >= 0
    view_confidence = view_confidence_for_npz(z, view_confidence_profile)
    if view_confidence <= 0.0:
        return delta, np.zeros_like(valid, dtype=bool)
    view_alpha_cap = view_alpha_cap_for_npz(
        z,
        dict(local_alpha_profile.get("view_alpha_cap_profile", {}))
        if isinstance(local_alpha_profile, dict)
        else None,
    )
    if "barycentric" not in z:
        return delta, np.zeros_like(valid, dtype=bool)
    if "barycentric_valid" in z:
        valid &= np.asarray(z["barycentric_valid"]).astype(bool)
    if "alpha" in z:
        valid &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
    bary = np.asarray(z["barycentric"], dtype=np.float32)
    valid &= np.all(np.isfinite(bary), axis=0)
    valid &= np.all(bary >= -0.05, axis=0)
    valid &= np.all(bary <= 1.05, axis=0)
    if not bool(valid.any()):
        return delta, valid
    atlas_faces = np.fromiter(atlas.keys(), dtype=np.int64)
    valid &= np.isin(face_id, atlas_faces)
    allowed_faces = _allowed_faces_from_gain_guard(face_gain_guard_profile)
    if allowed_faces is not None:
        if not allowed_faces:
            return delta, np.zeros_like(valid, dtype=bool)
        allowed_face_ids = np.fromiter(allowed_faces, dtype=np.int64)
        valid &= np.isin(face_id, allowed_face_ids)
    if not bool(valid.any()):
        return delta, valid
    ubin, vbin = _uv_bins(bary, valid, next(iter(atlas.values())).texture.shape[0])
    ys, xs = np.nonzero(valid)
    allowed_bins = _allowed_bins_from_uncertainty_guard(bin_uncertainty_guard_profile)
    if allowed_bins is not None:
        if not allowed_bins:
            return delta, np.zeros_like(valid, dtype=bool)
        local_faces = face_id[valid]
        local_bin_ids = (vbin.astype(np.int64) * int(next(iter(atlas.values())).texture.shape[0])) + ubin.astype(np.int64)
        bin_keep = np.zeros((int(local_faces.size),), dtype=bool)
        for face in np.unique(local_faces):
            allowed_for_face = allowed_bins.get(int(face))
            if not allowed_for_face:
                continue
            fm = local_faces == int(face)
            bin_keep[fm] = np.isin(local_bin_ids[fm], np.fromiter(allowed_for_face, dtype=np.int64))
        if not bool(np.any(bin_keep)):
            return delta, np.zeros_like(valid, dtype=bool)
        filtered = np.zeros_like(valid, dtype=bool)
        filtered[ys[bin_keep], xs[bin_keep]] = True
        valid = filtered
        ubin = ubin[bin_keep]
        vbin = vbin[bin_keep]
        ys = ys[bin_keep]
        xs = xs[bin_keep]
    out_pixels = np.zeros((int(ys.size), 3), dtype=np.float32)
    support_valid = np.zeros((int(ys.size),), dtype=bool)
    faces = face_id[valid]
    view_feature_cache: dict[str, np.ndarray | None] = {}
    teacher_basis_feature_cache: dict[str, np.ndarray | None] = {}
    image_linear_feature_cache: dict[str, Any] | None = (
        {}
        if (
            local_alpha_profile
            and bool(local_alpha_profile.get("enabled", False))
            and str(local_alpha_profile.get("mode", "")) == "policy_val_image_linear_generator"
        )
        else None
    )
    local_alpha_view_cluster_index = None
    if local_alpha_profile and bool(local_alpha_profile.get("view_cluster_local_shrink", False)):
        cluster_centers, cluster_feature_mode = _view_cluster_profile_from_atlas(atlas)
        if cluster_centers is not None and str(cluster_feature_mode) != "none":
            local_alpha_view_cluster_index = _assign_view_cluster_for_npz(
                z,
                centers=cluster_centers,
                feature_mode=str(cluster_feature_mode),
            )
    for face in np.unique(faces):
        fm = faces == face
        face_atlas = atlas[int(face)]
        if int(face_atlas.samples) < int(min_atlas_face_samples):
            continue
        if int(min_atlas_bin_count) > 0:
            bin_counts = face_atlas.counts[vbin[fm], ubin[fm]]
            bin_valid = bin_counts >= int(min_atlas_bin_count)
            if float(max_atlas_bin_rgb_variance) >= 0.0:
                bin_variance = face_atlas.variance[vbin[fm], ubin[fm]]
                bin_valid &= np.mean(bin_variance, axis=1) <= float(max_atlas_bin_rgb_variance)
            if float(min_atlas_bin_sign_consistency) > 0.0:
                bin_sign_consistency = face_atlas.sign_consistency[vbin[fm], ubin[fm]]
                bin_valid &= np.mean(bin_sign_consistency, axis=1) >= float(min_atlas_bin_sign_consistency)
            if not bool(np.any(bin_valid)):
                continue
            face_valid_indices = np.nonzero(fm)[0][bin_valid]
        else:
            face_valid_indices = np.nonzero(fm)[0]
        tex = face_atlas.texture
        confidence = np.ones((int(face_valid_indices.size),), dtype=np.float32)
        if str(atlas_confidence_mode) != "none":
            local_counts = face_atlas.counts[vbin[face_valid_indices], ubin[face_valid_indices]].astype(np.float32)
            observed = local_counts > 0.0
            if float(atlas_confidence_count_scale) > 0.0:
                count_conf = 1.0 - np.exp(-local_counts / float(atlas_confidence_count_scale))
                count_conf[~observed] = float(atlas_confidence_empty_bin)
                confidence *= np.clip(count_conf, 0.0, 1.0)
            if float(atlas_confidence_variance_scale) > 0.0:
                local_variance = np.mean(face_atlas.variance[vbin[face_valid_indices], ubin[face_valid_indices]], axis=1)
                variance_conf = np.ones_like(confidence)
                variance_conf[observed] = 1.0 / (
                    1.0 + local_variance[observed] / float(atlas_confidence_variance_scale)
                )
                confidence *= np.clip(variance_conf, 0.0, 1.0)
            if float(atlas_confidence_sign_power) > 0.0:
                local_sign = np.mean(face_atlas.sign_consistency[vbin[face_valid_indices], ubin[face_valid_indices]], axis=1)
                sign_conf = np.ones_like(confidence)
                sign_conf[observed] = np.power(
                    np.clip(local_sign[observed], 0.0, 1.0),
                    float(atlas_confidence_sign_power),
                )
                confidence *= np.clip(sign_conf, 0.0, 1.0)
            if float(atlas_confidence_face_sample_scale) > 0.0:
                face_conf = min(1.0, float(face_atlas.samples) / float(atlas_confidence_face_sample_scale))
                confidence *= float(face_conf)
        confidence = np.clip(confidence, 0.0, 1.0)
        confident = confidence > float(min_atlas_confidence)
        if not bool(np.any(confident)):
            continue
        confident_indices = face_valid_indices[confident]
        base_pixels = tex[vbin[confident_indices], ubin[confident_indices]] * confidence[confident, None]
        if (
            face_atlas.expert_textures is not None
            and face_atlas.expert_counts is not None
            and face_atlas.expert_centers is not None
            and str(face_atlas.expert_feature_mode) != "none"
        ):
            expert_index = _assign_view_cluster_for_npz(
                z,
                centers=face_atlas.expert_centers,
                feature_mode=str(face_atlas.expert_feature_mode),
            )
            if (
                expert_index is not None
                and 0 <= int(expert_index) < int(face_atlas.expert_textures.shape[0])
                and int(expert_index) < int(face_atlas.expert_counts.shape[0])
            ):
                local_expert_counts = face_atlas.expert_counts[int(expert_index)][
                    vbin[confident_indices],
                    ubin[confident_indices],
                ]
                expert_supported = local_expert_counts >= max(1, int(face_atlas.expert_min_bin_samples))
                if bool(np.any(expert_supported)):
                    expert_pixels = face_atlas.expert_textures[int(expert_index)][
                        vbin[confident_indices][expert_supported],
                        ubin[confident_indices][expert_supported],
                    ]
                    base_pixels[expert_supported] = (
                        expert_pixels.astype(np.float32)
                        * confidence[confident][expert_supported, None]
                    )
        if (
            face_atlas.view_basis_coefficients is not None
            and face_atlas.view_basis_support is not None
            and str(face_atlas.view_basis_mode) != "none"
        ):
            view_basis_mode = str(face_atlas.view_basis_mode)
            if view_basis_mode not in view_feature_cache:
                view_feature_cache[view_basis_mode] = _view_condition_features_for_mask(z, view_basis_mode, valid)
            view_features = view_feature_cache[view_basis_mode]
            if view_features is not None:
                basis_support = face_atlas.view_basis_support[vbin[confident_indices], ubin[confident_indices]]
                if bool(np.any(basis_support)):
                    basis_feature_samples = view_features[confident_indices][basis_support].astype(np.float32)
                    if (
                        str(face_atlas.view_basis_ood_mode) == "diag_z"
                        and face_atlas.view_basis_feature_mean is not None
                        and face_atlas.view_basis_feature_std is not None
                        and float(face_atlas.view_basis_ood_max_z) > 0.0
                    ):
                        local_feature_mean = face_atlas.view_basis_feature_mean[
                            vbin[confident_indices][basis_support],
                            ubin[confident_indices][basis_support],
                        ].astype(np.float32)
                        local_feature_std = face_atlas.view_basis_feature_std[
                            vbin[confident_indices][basis_support],
                            ubin[confident_indices][basis_support],
                        ].astype(np.float32)
                        std = np.maximum(local_feature_std, float(face_atlas.view_basis_ood_min_std))
                        zscore = np.abs(basis_feature_samples - local_feature_mean) / std
                        if zscore.shape[1] > 1:
                            zscore = zscore[:, 1:]
                        in_distribution = np.max(zscore, axis=1) <= float(face_atlas.view_basis_ood_max_z)
                        basis_support = np.zeros_like(basis_support, dtype=bool)
                        if bool(np.any(in_distribution)):
                            basis_indices = np.nonzero(
                                face_atlas.view_basis_support[
                                    vbin[confident_indices],
                                    ubin[confident_indices],
                                ]
                            )[0][in_distribution]
                            basis_support[basis_indices] = True
                            basis_feature_samples = basis_feature_samples[in_distribution]
                    if bool(np.any(basis_support)):
                        basis_coeffs = face_atlas.view_basis_coefficients[
                            vbin[confident_indices][basis_support],
                            ubin[confident_indices][basis_support],
                        ]
                        basis_pixels = np.einsum(
                            "nd,ndc->nc",
                            basis_feature_samples.astype(np.float32),
                            basis_coeffs.astype(np.float32),
                        )
                        local_confidence = confidence[confident][basis_support]
                        base_pixels[basis_support] = basis_pixels * local_confidence[:, None]
        if (
            face_atlas.teacher_basis_coefficients is not None
            and str(face_atlas.teacher_basis_mode) != "none"
        ):
            teacher_basis_mode = str(face_atlas.teacher_basis_mode)
            if teacher_basis_mode not in teacher_basis_feature_cache:
                teacher_basis_feature_cache[teacher_basis_mode] = _teacher_distilled_basis_features_for_mask(
                    z,
                    teacher_basis_mode,
                    valid,
                )
            teacher_features = teacher_basis_feature_cache[teacher_basis_mode]
            if teacher_features is not None:
                feature_samples = teacher_features[confident_indices].astype(np.float32)
                teacher_support = np.ones((int(feature_samples.shape[0]),), dtype=bool)
                if (
                    face_atlas.teacher_basis_feature_mean is not None
                    and face_atlas.teacher_basis_feature_std is not None
                    and float(face_atlas.teacher_basis_ood_max_z) > 0.0
                ):
                    std = np.maximum(
                        face_atlas.teacher_basis_feature_std.astype(np.float32),
                        float(face_atlas.teacher_basis_ood_min_std),
                    )
                    zscore = np.abs(feature_samples - face_atlas.teacher_basis_feature_mean.astype(np.float32)) / std
                    if zscore.shape[1] > 1:
                        zscore = zscore[:, 1:]
                    teacher_support &= np.max(zscore, axis=1) <= float(face_atlas.teacher_basis_ood_max_z)
                apply_mode = str(face_atlas.teacher_basis_apply_mode)
                if apply_mode == "fill_empty_only":
                    local_counts = face_atlas.counts[vbin[confident_indices], ubin[confident_indices]]
                    teacher_support &= local_counts <= 0
                if bool(np.any(teacher_support)):
                    teacher_pixels = None
                    if (
                        _is_low_rank_teacher_texture_mode(teacher_basis_mode)
                        and face_atlas.teacher_texture_basis is not None
                        and face_atlas.teacher_texture_support is not None
                    ):
                        texture_support = face_atlas.teacher_texture_support[
                            vbin[confident_indices],
                            ubin[confident_indices],
                        ].astype(bool)
                        teacher_support &= texture_support
                    if (
                        bool(np.any(teacher_support))
                        and
                        _is_low_rank_teacher_texture_mode(teacher_basis_mode)
                        and face_atlas.teacher_texture_basis is not None
                    ):
                        weights = np.matmul(
                            feature_samples[teacher_support].astype(np.float32),
                            face_atlas.teacher_basis_coefficients.astype(np.float32),
                        )
                        basis = face_atlas.teacher_texture_basis[
                            :,
                            vbin[confident_indices][teacher_support],
                            ubin[confident_indices][teacher_support],
                            :,
                        ].transpose(1, 0, 2)
                        teacher_pixels = np.einsum("nk,nkc->nc", weights, basis.astype(np.float32))
                    elif bool(np.any(teacher_support)):
                        teacher_pixels = np.einsum(
                            "nd,dc->nc",
                            feature_samples[teacher_support].astype(np.float32),
                            face_atlas.teacher_basis_coefficients.astype(np.float32),
                        )
                    if teacher_pixels is not None:
                        local_confidence = confidence[confident][teacher_support]
                        teacher_pixels = teacher_pixels * local_confidence[:, None]
                        if apply_mode == "blend":
                            blend = float(np.clip(face_atlas.teacher_basis_blend, 0.0, 1.0))
                            base_pixels[teacher_support] = (
                                (1.0 - blend) * base_pixels[teacher_support]
                                + blend * teacher_pixels
                            )
                        else:
                            base_pixels[teacher_support] = teacher_pixels
        if (
            local_alpha_profile
            and bool(local_alpha_profile.get("enabled", False))
            and str(local_alpha_profile.get("mode", "")) == "policy_val_image_linear_generator"
        ):
            base_pixels = _apply_image_linear_generator_to_samples(
                z,
                local_alpha_profile,
                base_pixels=base_pixels,
                ys=ys[confident_indices],
                xs=xs[confident_indices],
                bary=bary,
                current_alpha=float(alpha),
                feature_cache=image_linear_feature_cache,
            )
        local_faces = faces[confident_indices]
        local_bin_ids = (
            vbin[confident_indices].astype(np.int64) * int(tex.shape[0])
        ) + ubin[confident_indices].astype(np.int64)
        local_alpha = _local_alpha_for_samples(
            base_pixels,
            local_alpha_profile,
            face_ids=local_faces,
            bin_ids=local_bin_ids,
            view_cluster_id=local_alpha_view_cluster_index,
        )
        if local_alpha_profile and bool(local_alpha_profile.get("enabled", False)):
            mode = str(local_alpha_profile.get("mode", ""))
            if mode == "policy_val_hybrid_bin_source_alpha" and float(alpha) > 0.0:
                use_prior = _hybrid_prior_source_mask(
                    local_alpha_profile,
                    np.asarray(local_faces, dtype=np.int64),
                    np.asarray(local_bin_ids, dtype=np.int64),
                )
                baseline_cap = _profile_effective_max_alpha(
                    dict(local_alpha_profile.get("baseline_profile", {"enabled": False}))
                )
                prior_cap = _profile_effective_max_alpha(
                    dict(local_alpha_profile.get("prior_profile", {"enabled": False}))
                )
                cap_by_sample = np.full((int(local_bin_ids.size),), baseline_cap, dtype=np.float32)
                cap_by_sample[use_prior] = float(prior_cap)
                finite_cap = np.isfinite(cap_by_sample)
                if bool(np.any(finite_cap)):
                    if local_alpha.ndim == 1:
                        capped = np.array(local_alpha, copy=True)
                        capped[finite_cap] = np.minimum(
                            capped[finite_cap],
                            cap_by_sample[finite_cap] / float(alpha),
                        )
                        local_alpha = capped
                    else:
                        capped = np.array(local_alpha, copy=True)
                        capped[finite_cap] = np.minimum(
                            capped[finite_cap],
                            (cap_by_sample[finite_cap] / float(alpha))[:, None],
                        )
                        local_alpha = capped
            else:
                max_effective_alpha = float(local_alpha_profile.get("max_alpha", np.inf))
                if math.isfinite(max_effective_alpha) and float(alpha) > 0.0:
                    local_alpha = np.minimum(local_alpha, max_effective_alpha / float(alpha))
        if local_alpha.ndim == 1:
            out_pixels[confident_indices] = base_pixels * local_alpha[:, None]
        else:
            out_pixels[confident_indices] = base_pixels * local_alpha
        support_valid[confident_indices] = True
    if int(min_atlas_bin_count) > 0 or int(min_atlas_face_samples) > 0:
        final_valid = np.zeros_like(valid, dtype=bool)
        if bool(np.any(support_valid)):
            final_valid[ys[support_valid], xs[support_valid]] = True
        valid = final_valid
    else:
        final_valid = np.zeros_like(valid, dtype=bool)
        if bool(np.any(support_valid)):
            final_valid[ys[support_valid], xs[support_valid]] = True
        valid = final_valid
    parent_edge_multiplier = None
    if parent_edge_apply_profile and bool(parent_edge_apply_profile.get("enabled", False)):
        render_chw = _as_rgb_chw(np.asarray(z["rgb_render"], dtype=np.float32)) if "rgb_render" in z else None
        edge = _luminance_gradient_magnitude_chw(render_chw) if render_chw is not None else None
        if edge is not None:
            edge_tau = max(float(parent_edge_apply_profile.get("edge_tau", 0.05)), 1.0e-12)
            weight = max(0.0, float(parent_edge_apply_profile.get("weight", 0.0)))
            min_multiplier = float(np.clip(parent_edge_apply_profile.get("min_multiplier", 0.0), 0.0, 1.0))
            raw = 1.0 / (1.0 + weight * (edge / edge_tau))
            parent_edge_multiplier = np.clip(raw, min_multiplier, 1.0).astype(np.float32)
    for c in range(3):
        channel = delta[c]
        if bool(np.any(support_valid)):
            values = float(alpha) * out_pixels[support_valid, c]
            if math.isfinite(view_alpha_cap) and float(alpha) > 0.0:
                values = values * float(np.clip(view_alpha_cap / float(alpha), 0.0, 1.0))
            if parent_edge_multiplier is not None:
                values = values * parent_edge_multiplier[ys[support_valid], xs[support_valid]]
            if view_confidence < 1.0:
                values = values * float(view_confidence)
            channel[ys[support_valid], xs[support_valid]] = values
    return delta, valid


def evaluate_policy_val(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    residual_rgb_key: str,
    residual_l1_key: str,
    alpha_grid: list[float],
    min_l1: float,
    min_alpha: float,
    max_abs_delta_rgb: float,
    max_samples_per_view: int,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    enable_policy_val_image_ssim: bool,
    policy_val_ssim_max_size: int,
    enable_policy_val_image_l1: bool,
    policy_val_l1_max_size: int,
    enable_policy_val_image_lpips: bool,
    policy_val_lpips_max_size: int,
    local_alpha_profile: dict[str, Any] | None = None,
    face_gain_guard_profile: dict[str, Any] | None = None,
    bin_uncertainty_guard_profile: dict[str, Any] | None = None,
    parent_edge_apply_profile: dict[str, Any] | None = None,
    view_confidence_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not val_views:
        return {"enabled": False, "reason": "no_policy_val_views"}
    rng = np.random.default_rng(11)
    residual_chunks: list[np.ndarray] = []
    pred_by_alpha = {float(alpha): [] for alpha in alpha_grid}
    per_view_by_alpha: dict[float, list[dict[str, Any]]] = {float(alpha): [] for alpha in alpha_grid}
    ssim_gains_by_alpha: dict[float, list[float]] = {float(alpha): [] for alpha in alpha_grid}
    ssim_before_by_alpha: dict[float, list[float]] = {float(alpha): [] for alpha in alpha_grid}
    ssim_after_by_alpha: dict[float, list[float]] = {float(alpha): [] for alpha in alpha_grid}
    l1_gains_by_alpha: dict[float, list[float]] = {float(alpha): [] for alpha in alpha_grid}
    l1_before_by_alpha: dict[float, list[float]] = {float(alpha): [] for alpha in alpha_grid}
    l1_after_by_alpha: dict[float, list[float]] = {float(alpha): [] for alpha in alpha_grid}
    lpips_gains_by_alpha: dict[float, list[float]] = {float(alpha): [] for alpha in alpha_grid}
    lpips_before_by_alpha: dict[float, list[float]] = {float(alpha): [] for alpha in alpha_grid}
    lpips_after_by_alpha: dict[float, list[float]] = {float(alpha): [] for alpha in alpha_grid}
    lpips_model = build_lpips_model() if bool(enable_policy_val_image_lpips) else None
    selective_view_confidence = bool(view_confidence_profile and view_confidence_profile.get("enabled", False))
    view_alpha_cap_profile = (
        dict(local_alpha_profile.get("view_alpha_cap_profile", {}))
        if isinstance(local_alpha_profile, dict)
        else {}
    )
    selective_view_alpha_cap = bool(view_alpha_cap_profile and view_alpha_cap_profile.get("enabled", False))
    selective_view_policy = bool(selective_view_confidence or selective_view_alpha_cap)

    def metric_active_mask(metric_rows: list[dict[str, Any]]) -> np.ndarray:
        if not metric_rows:
            return np.zeros((0,), dtype=bool)
        confidences = np.asarray(
            [float(row.get("view_confidence", 1.0) or 0.0) for row in metric_rows],
            dtype=np.float64,
        )
        active = confidences > 0.0
        if selective_view_alpha_cap:
            caps = np.asarray(
                [
                    (
                        float(row.get("view_alpha_cap", 0.0) or 0.0)
                        if row.get("view_alpha_cap") is not None
                        else float("inf")
                    )
                    for row in metric_rows
                ],
                dtype=np.float64,
            )
            active &= np.isinf(caps) | (caps > 0.0)
        return active

    face_count = set()
    for path in tqdm(val_views, desc="policy-val atlas"):
        z = np.load(path)
        if residual_rgb_key not in z:
            continue
        mask = _valid_sample_mask(z, set(atlas.keys()), residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            mask = np.zeros_like(mask, dtype=bool)
            mask[ys, xs] = True
        face_count.update(int(x) for x in np.unique(np.asarray(z["face_id"], dtype=np.int64)[mask]))
        residual = np.asarray(z[residual_rgb_key], dtype=np.float32)
        target = np.stack([residual[0][mask], residual[1][mask], residual[2][mask]], axis=1)
        residual_chunks.append(target)
        view_mse_before = float(np.mean(np.sum(target * target, axis=1)))
        view_ssim_before = None
        if bool(enable_policy_val_image_ssim) and "rgb_render" in z and "rgb_gt" in z:
            view_ssim_before = image_ssim_chw(
                np.asarray(z["rgb_render"], dtype=np.float32),
                np.asarray(z["rgb_gt"], dtype=np.float32),
                int(policy_val_ssim_max_size),
            )
        view_l1_before = None
        if bool(enable_policy_val_image_l1) and "rgb_render" in z and "rgb_gt" in z:
            view_l1_before = image_l1_chw(
                np.asarray(z["rgb_render"], dtype=np.float32),
                np.asarray(z["rgb_gt"], dtype=np.float32),
                int(policy_val_l1_max_size),
            )
        view_lpips_before = None
        if bool(enable_policy_val_image_lpips) and "rgb_render" in z and "rgb_gt" in z:
            view_lpips_before = image_lpips_chw(
                np.asarray(z["rgb_render"], dtype=np.float32),
                np.asarray(z["rgb_gt"], dtype=np.float32),
                int(policy_val_lpips_max_size),
                lpips_model,
            )
        view_confidence_scalar = (
            view_confidence_for_npz(z, view_confidence_profile)
            if selective_view_confidence
            else 1.0
        )
        view_alpha_cap_scalar = (
            view_alpha_cap_for_npz(z, view_alpha_cap_profile)
            if selective_view_alpha_cap
            else float("inf")
        )
        for alpha in alpha_grid:
            pred, _valid = predict_delta_for_npz(
                z,
                atlas,
                float(alpha),
                min_alpha,
                min_atlas_bin_count=int(min_atlas_bin_count),
                min_atlas_face_samples=int(min_atlas_face_samples),
                max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
                min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
                atlas_confidence_mode=str(atlas_confidence_mode),
                atlas_confidence_count_scale=float(atlas_confidence_count_scale),
                atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
                atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
                atlas_confidence_sign_power=float(atlas_confidence_sign_power),
                atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
                min_atlas_confidence=float(min_atlas_confidence),
                local_alpha_profile=local_alpha_profile,
                face_gain_guard_profile=face_gain_guard_profile,
                bin_uncertainty_guard_profile=bin_uncertainty_guard_profile,
                parent_edge_apply_profile=parent_edge_apply_profile,
                view_confidence_profile=view_confidence_profile,
            )
            pred = clip_delta_rgb(pred, float(max_abs_delta_rgb))
            pred_samples = np.stack([pred[0][mask], pred[1][mask], pred[2][mask]], axis=1)
            pred_by_alpha[float(alpha)].append(pred_samples)
            view_err = target - pred_samples
            view_mse_after = float(np.mean(np.sum(view_err * view_err, axis=1)))
            view_rel_gain = (view_mse_before - view_mse_after) / max(view_mse_before, 1.0e-12)
            view_ssim_after = None
            view_ssim_gain = None
            view_l1_after = None
            view_l1_gain = None
            view_lpips_after = None
            view_lpips_gain = None
            adapted = None
            if view_ssim_before is not None or view_l1_before is not None or view_lpips_before is not None:
                adapted = np.clip(np.asarray(z["rgb_render"], dtype=np.float32) + pred, 0.0, 1.0)
            if view_ssim_before is not None and adapted is not None:
                view_ssim_after = image_ssim_chw(
                    adapted,
                    np.asarray(z["rgb_gt"], dtype=np.float32),
                    int(policy_val_ssim_max_size),
                )
                view_ssim_gain = float(view_ssim_after - view_ssim_before)
                ssim_gains_by_alpha[float(alpha)].append(view_ssim_gain)
                ssim_before_by_alpha[float(alpha)].append(float(view_ssim_before))
                ssim_after_by_alpha[float(alpha)].append(float(view_ssim_after))
            if view_l1_before is not None and adapted is not None:
                view_l1_after = image_l1_chw(
                    adapted,
                    np.asarray(z["rgb_gt"], dtype=np.float32),
                    int(policy_val_l1_max_size),
                )
                view_l1_gain = float(view_l1_before - view_l1_after)
                l1_gains_by_alpha[float(alpha)].append(view_l1_gain)
                l1_before_by_alpha[float(alpha)].append(float(view_l1_before))
                l1_after_by_alpha[float(alpha)].append(float(view_l1_after))
            if view_lpips_before is not None and adapted is not None:
                view_lpips_after = image_lpips_chw(
                    adapted,
                    np.asarray(z["rgb_gt"], dtype=np.float32),
                    int(policy_val_lpips_max_size),
                    lpips_model,
                )
                view_lpips_gain = float(view_lpips_before - view_lpips_after)
                lpips_gains_by_alpha[float(alpha)].append(view_lpips_gain)
                lpips_before_by_alpha[float(alpha)].append(float(view_lpips_before))
                lpips_after_by_alpha[float(alpha)].append(float(view_lpips_after))
            per_view_by_alpha[float(alpha)].append(
                {
                    "view": path.stem,
                    "samples": int(target.shape[0]),
                    "mse_before": view_mse_before,
                    "mse_after": view_mse_after,
                    "relative_gain": float(view_rel_gain),
                    "ssim_before": view_ssim_before,
                    "ssim_after": view_ssim_after,
                    "ssim_gain": view_ssim_gain,
                    "image_l1_before": view_l1_before,
                    "image_l1_after": view_l1_after,
                    "image_l1_gain": view_l1_gain,
                    "lpips_before": view_lpips_before,
                    "lpips_after": view_lpips_after,
                    "lpips_gain": view_lpips_gain,
                    "view_confidence": float(view_confidence_scalar),
                    "view_confidence_active": bool(view_confidence_scalar > 0.0),
                    "view_alpha_cap": (
                        None if not math.isfinite(view_alpha_cap_scalar) else float(view_alpha_cap_scalar)
                    ),
                    "view_alpha_cap_active": bool(
                        (not selective_view_alpha_cap)
                        or (math.isfinite(view_alpha_cap_scalar) and float(view_alpha_cap_scalar) > 0.0)
                    ),
                }
            )
    if not residual_chunks:
        return {"enabled": False, "reason": "no_policy_val_samples"}
    target_all = np.concatenate(residual_chunks, axis=0)
    mse_before = float(np.mean(np.sum(target_all * target_all, axis=1)))
    rows = []
    best = None
    for alpha, chunks in pred_by_alpha.items():
        pred_all = np.concatenate(chunks, axis=0) if chunks else np.zeros_like(target_all)
        err = target_all - pred_all
        mse_after = float(np.mean(np.sum(err * err, axis=1)))
        mae_after = float(np.mean(np.abs(err)))
        rel_gain = (mse_before - mse_after) / max(mse_before, 1.0e-12)
        view_rows = per_view_by_alpha.get(float(alpha), [])
        view_gains = np.asarray([float(row["relative_gain"]) for row in view_rows], dtype=np.float64)
        if view_gains.size:
            sorted_gains = np.sort(view_gains)
            cvar_count = max(1, int(math.ceil(0.20 * float(sorted_gains.size))))
            view_stats = {
                "view_count": int(view_gains.size),
                "positive_view_fraction": float(np.mean(view_gains > 0.0)),
                "min_view_relative_gain": float(np.min(view_gains)),
                "p10_view_relative_gain": float(np.quantile(view_gains, 0.10)),
                "cvar20_view_relative_gain": float(np.mean(sorted_gains[:cvar_count])),
            }
            if selective_view_policy:
                view_confidences = np.asarray(
                    [float(row.get("view_confidence", 1.0) or 0.0) for row in view_rows],
                    dtype=np.float64,
                )
                view_alpha_caps = np.asarray(
                    [
                        (
                            float(row.get("view_alpha_cap", 0.0) or 0.0)
                            if row.get("view_alpha_cap") is not None
                            else float("inf")
                        )
                        for row in view_rows
                    ],
                    dtype=np.float64,
                )
                alpha_cap_active = np.isinf(view_alpha_caps) | (view_alpha_caps > 0.0)
                active = metric_active_mask(view_rows)
                eps = 1.0e-12
                view_stats.update(
                    {
                        "view_confidence_selective": bool(selective_view_confidence),
                        "view_alpha_cap_selective": bool(selective_view_alpha_cap),
                        "view_confidence_active_view_fraction": float(np.mean(active)),
                        "view_confidence_mean": float(np.mean(view_confidences)),
                        "view_alpha_cap_active_view_fraction": float(np.mean(alpha_cap_active)),
                        "view_alpha_cap_mean": (
                            float(np.mean(view_alpha_caps[np.isfinite(view_alpha_caps)]))
                            if bool(np.any(np.isfinite(view_alpha_caps)))
                            else 0.0
                        ),
                        "nonnegative_view_fraction": float(np.mean(view_gains >= -eps)),
                        "active_positive_view_fraction": (
                            float(np.mean(view_gains[active] > 0.0)) if bool(np.any(active)) else 0.0
                        ),
                    }
                )
        else:
            view_stats = {
                "view_count": 0,
                "positive_view_fraction": 0.0,
                "min_view_relative_gain": 0.0,
                "p10_view_relative_gain": 0.0,
                "cvar20_view_relative_gain": 0.0,
            }
            if selective_view_policy:
                view_stats.update(
                    {
                        "view_confidence_selective": bool(selective_view_confidence),
                        "view_alpha_cap_selective": bool(selective_view_alpha_cap),
                        "view_confidence_active_view_fraction": 0.0,
                        "view_confidence_mean": 0.0,
                        "view_alpha_cap_active_view_fraction": 0.0,
                        "view_alpha_cap_mean": 0.0,
                        "nonnegative_view_fraction": 0.0,
                        "active_positive_view_fraction": 0.0,
                    }
                )
        ssim_gains = np.asarray(ssim_gains_by_alpha.get(float(alpha), []), dtype=np.float64)
        if ssim_gains.size:
            sorted_ssim_gains = np.sort(ssim_gains)
            cvar_count = max(1, int(math.ceil(0.20 * float(sorted_ssim_gains.size))))
            ssim_stats = {
                "ssim_view_count": int(ssim_gains.size),
                "ssim_before": float(np.mean(ssim_before_by_alpha[float(alpha)])),
                "ssim_after": float(np.mean(ssim_after_by_alpha[float(alpha)])),
                "ssim_gain": float(np.mean(ssim_gains)),
                "ssim_positive_view_fraction": float(np.mean(ssim_gains > 0.0)),
                "ssim_min_view_gain": float(np.min(ssim_gains)),
                "ssim_cvar20_view_gain": float(np.mean(sorted_ssim_gains[:cvar_count])),
            }
            if selective_view_policy:
                metric_rows = [row for row in view_rows if row.get("ssim_gain") is not None]
                metric_values = np.asarray([float(row["ssim_gain"]) for row in metric_rows], dtype=np.float64)
                metric_conf = np.asarray(
                    [float(row.get("view_confidence", 1.0) or 0.0) for row in metric_rows],
                    dtype=np.float64,
                )
                active = metric_active_mask(metric_rows)
                eps = 1.0e-12
                ssim_stats.update(
                    {
                        "ssim_nonnegative_view_fraction": float(np.mean(metric_values >= -eps)),
                        "ssim_active_positive_view_fraction": (
                            float(np.mean(metric_values[active] > 0.0)) if bool(np.any(active)) else 0.0
                        ),
                    }
                )
        else:
            ssim_stats = {
                "ssim_view_count": 0,
                "ssim_before": 0.0,
                "ssim_after": 0.0,
                "ssim_gain": 0.0,
                "ssim_positive_view_fraction": 0.0,
                "ssim_min_view_gain": 0.0,
                "ssim_cvar20_view_gain": 0.0,
            }
            if selective_view_policy:
                ssim_stats.update(
                    {
                        "ssim_nonnegative_view_fraction": 0.0,
                        "ssim_active_positive_view_fraction": 0.0,
                    }
                )
        l1_gains = np.asarray(l1_gains_by_alpha.get(float(alpha), []), dtype=np.float64)
        if l1_gains.size:
            sorted_l1_gains = np.sort(l1_gains)
            cvar_count = max(1, int(math.ceil(0.20 * float(sorted_l1_gains.size))))
            l1_stats = {
                "image_l1_view_count": int(l1_gains.size),
                "image_l1_before": float(np.mean(l1_before_by_alpha[float(alpha)])),
                "image_l1_after": float(np.mean(l1_after_by_alpha[float(alpha)])),
                "image_l1_gain": float(np.mean(l1_gains)),
                "image_l1_positive_view_fraction": float(np.mean(l1_gains > 0.0)),
                "image_l1_min_view_gain": float(np.min(l1_gains)),
                "image_l1_cvar20_view_gain": float(np.mean(sorted_l1_gains[:cvar_count])),
            }
            if selective_view_policy:
                metric_rows = [row for row in view_rows if row.get("image_l1_gain") is not None]
                metric_values = np.asarray([float(row["image_l1_gain"]) for row in metric_rows], dtype=np.float64)
                metric_conf = np.asarray(
                    [float(row.get("view_confidence", 1.0) or 0.0) for row in metric_rows],
                    dtype=np.float64,
                )
                active = metric_active_mask(metric_rows)
                eps = 1.0e-12
                l1_stats.update(
                    {
                        "image_l1_nonnegative_view_fraction": float(np.mean(metric_values >= -eps)),
                        "image_l1_active_positive_view_fraction": (
                            float(np.mean(metric_values[active] > 0.0)) if bool(np.any(active)) else 0.0
                        ),
                    }
                )
        else:
            l1_stats = {
                "image_l1_view_count": 0,
                "image_l1_before": 0.0,
                "image_l1_after": 0.0,
                "image_l1_gain": 0.0,
                "image_l1_positive_view_fraction": 0.0,
                "image_l1_min_view_gain": 0.0,
                "image_l1_cvar20_view_gain": 0.0,
            }
            if selective_view_policy:
                l1_stats.update(
                    {
                        "image_l1_nonnegative_view_fraction": 0.0,
                        "image_l1_active_positive_view_fraction": 0.0,
                    }
                )
        lpips_gains = np.asarray(lpips_gains_by_alpha.get(float(alpha), []), dtype=np.float64)
        if lpips_gains.size:
            sorted_lpips_gains = np.sort(lpips_gains)
            cvar_count = max(1, int(math.ceil(0.20 * float(sorted_lpips_gains.size))))
            lpips_stats = {
                "lpips_view_count": int(lpips_gains.size),
                "lpips_before": float(np.mean(lpips_before_by_alpha[float(alpha)])),
                "lpips_after": float(np.mean(lpips_after_by_alpha[float(alpha)])),
                "lpips_gain": float(np.mean(lpips_gains)),
                "lpips_positive_view_fraction": float(np.mean(lpips_gains > 0.0)),
                "lpips_min_view_gain": float(np.min(lpips_gains)),
                "lpips_cvar20_view_gain": float(np.mean(sorted_lpips_gains[:cvar_count])),
            }
            if selective_view_policy:
                metric_rows = [row for row in view_rows if row.get("lpips_gain") is not None]
                metric_values = np.asarray([float(row["lpips_gain"]) for row in metric_rows], dtype=np.float64)
                metric_conf = np.asarray(
                    [float(row.get("view_confidence", 1.0) or 0.0) for row in metric_rows],
                    dtype=np.float64,
                )
                active = metric_active_mask(metric_rows)
                eps = 1.0e-12
                lpips_stats.update(
                    {
                        "lpips_nonnegative_view_fraction": float(np.mean(metric_values >= -eps)),
                        "lpips_active_positive_view_fraction": (
                            float(np.mean(metric_values[active] > 0.0)) if bool(np.any(active)) else 0.0
                        ),
                    }
                )
        else:
            lpips_stats = {
                "lpips_view_count": 0,
                "lpips_before": 0.0,
                "lpips_after": 0.0,
                "lpips_gain": 0.0,
                "lpips_positive_view_fraction": 0.0,
                "lpips_min_view_gain": 0.0,
                "lpips_cvar20_view_gain": 0.0,
            }
            if selective_view_policy:
                lpips_stats.update(
                    {
                        "lpips_nonnegative_view_fraction": 0.0,
                        "lpips_active_positive_view_fraction": 0.0,
                    }
                )
        row = {
            "alpha": float(alpha),
            "mse_before": mse_before,
            "mse_after": mse_after,
            "relative_gain": float(rel_gain),
            "mae_after": mae_after,
            **view_stats,
            **ssim_stats,
            **l1_stats,
            **lpips_stats,
        }
        rows.append(row)
        if best is None or row["relative_gain"] > best["relative_gain"]:
            best = row
    return {
        "enabled": True,
        "samples": int(target_all.shape[0]),
        "unique_faces": int(len(face_count)),
        "mse_before": mse_before,
        "rows": rows,
        "per_view_by_alpha": {str(alpha): rows for alpha, rows in per_view_by_alpha.items()},
        "best": best or {},
    }


def annotate_sparse_materialization_selective_policy_val(
    policy_payload: dict[str, Any],
    sparse_profile: dict[str, Any] | None,
    *,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    """Mark sparse materialization policy-val rows as selective non-regressive rows.

    Sparse materialization intentionally edits only a small face/bin footprint.
    Views with no affected footprint should be treated as zero-gain safety
    evidence, not as failures to produce strictly positive gain.
    """
    if not isinstance(policy_payload, dict) or not bool(policy_payload.get("enabled", False)):
        return policy_payload
    if not isinstance(sparse_profile, dict) or not bool(sparse_profile.get("enabled", False)):
        return policy_payload
    if int(sparse_profile.get("allowed_bin_count", 0) or 0) <= 0:
        return policy_payload
    payload = dict(policy_payload)
    per_view_by_alpha = payload.get("per_view_by_alpha", {}) or {}
    if not isinstance(per_view_by_alpha, dict):
        return payload

    def rows_for_alpha(alpha: float) -> list[dict[str, Any]]:
        exact_key = str(float(alpha))
        rows = per_view_by_alpha.get(exact_key)
        if rows is not None:
            return list(rows or [])
        best_rows: list[dict[str, Any]] = []
        best_delta = float("inf")
        for key, value in per_view_by_alpha.items():
            try:
                key_alpha = float(key)
            except (TypeError, ValueError):
                continue
            delta = abs(key_alpha - float(alpha))
            if delta < best_delta:
                best_delta = delta
                best_rows = list(value or [])
        return best_rows if best_delta <= 1.0e-8 else []

    annotated_rows: list[dict[str, Any]] = []
    for row_in in payload.get("rows", []) or []:
        row = dict(row_in)
        row["sparse_materialization_selective"] = True
        row["sparse_materialization_allowed_bin_count"] = int(
            sparse_profile.get("allowed_bin_count", 0) or 0
        )
        row["sparse_materialization_allowed_sample_fraction"] = float(
            sparse_profile.get("allowed_sample_fraction", 0.0) or 0.0
        )
        view_rows = rows_for_alpha(float(row.get("alpha", 0.0)))
        rel = np.asarray([float(x.get("relative_gain", 0.0)) for x in view_rows], dtype=np.float64)
        if rel.size:
            row["nonnegative_view_fraction"] = float(np.mean(rel >= -float(eps)))
        for metric_key, output_key in (
            ("ssim_gain", "ssim_nonnegative_view_fraction"),
            ("image_l1_gain", "image_l1_nonnegative_view_fraction"),
            ("lpips_gain", "lpips_nonnegative_view_fraction"),
        ):
            vals = [
                float(x[metric_key])
                for x in view_rows
                if x.get(metric_key) is not None
            ]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                row[output_key] = float(np.mean(arr >= -float(eps)))
        annotated_rows.append(row)
    payload["rows"] = annotated_rows
    best_alpha = float((payload.get("best") or {}).get("alpha", float("nan")))
    if math.isfinite(best_alpha):
        for row in annotated_rows:
            if abs(float(row.get("alpha", 0.0)) - best_alpha) <= 1.0e-8:
                payload["best"] = dict(row)
                break
    payload["sparse_materialization_selective"] = {
        "enabled": True,
        "allowed_bin_count": int(sparse_profile.get("allowed_bin_count", 0) or 0),
        "allowed_sample_fraction": float(sparse_profile.get("allowed_sample_fraction", 0.0) or 0.0),
        "risk_semantics": "nonnegative_views_are_safe_for_sparse_noop_footprint",
    }
    return payload


def build_policy_val_prior_bin_gain_hybrid_atlas(
    val_views: list[Path],
    baseline_atlas: dict[int, FaceAtlas],
    prior_atlas: dict[int, FaceAtlas],
    residual_rgb_key: str,
    residual_l1_key: str,
    baseline_alpha: float,
    prior_alpha: float,
    min_l1: float,
    min_alpha: float,
    max_abs_delta_rgb: float,
    max_samples_per_view: int,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    baseline_local_alpha_profile: dict[str, Any] | None,
    prior_local_alpha_profile: dict[str, Any] | None,
    baseline_face_gain_guard_profile: dict[str, Any] | None,
    prior_face_gain_guard_profile: dict[str, Any] | None,
    baseline_bin_uncertainty_guard_profile: dict[str, Any] | None,
    prior_bin_uncertainty_guard_profile: dict[str, Any] | None,
    min_bin_samples: int,
    min_views: int,
    min_abs_gain: float,
    min_relative_gain: float,
    min_positive_view_fraction: float,
    enable_l1_proxy_gate: bool = False,
    min_l1_abs_gain: float = 0.0,
    min_l1_relative_gain: float = -1.0,
    min_l1_positive_view_fraction: float = 0.0,
    min_l1_min_view_gain: float = -1.0,
    min_l1_cvar20_view_gain: float = -1.0,
    max_profile_bins: int = 0,
    target_footprint_views: list[Path] | None = None,
    enable_target_footprint_bin_certificate: bool = False,
    target_footprint_min_bin_pixels: int = 1,
    target_footprint_min_views: int = 1,
    target_footprint_min_view_fraction: float = 0.0,
    target_footprint_max_views: int = 0,
    enable_target_footprint_tail_risk_certificate: bool = False,
    target_footprint_tail_risk_only_target_bins: bool = True,
    target_footprint_tail_risk_min_positive_view_fraction: float = 1.0,
    target_footprint_tail_risk_min_min_view_gain: float = 0.0,
    target_footprint_tail_risk_min_cvar20_view_gain: float = 0.0,
    source_mixture_enabled: bool = False,
    source_mixture_ridge_mode: str = "absolute",
    source_mixture_ridge: float = 1.0e-2,
    source_mixture_min_weight: float = 1.0e-4,
) -> tuple[dict[int, FaceAtlas], dict[str, Any]]:
    disabled = {
        "enabled": False,
        "mode": "policy_val_prior_bin_gain_hybrid",
        "reason": "",
    }
    if not val_views or not baseline_atlas or not prior_atlas:
        disabled["reason"] = "missing_policy_val_views_or_atlas"
        return clone_atlas(baseline_atlas), disabled
    texture_size = int(next(iter(baseline_atlas.values())).texture.shape[0])
    prior_texture_size = int(next(iter(prior_atlas.values())).texture.shape[0])
    if texture_size != prior_texture_size:
        disabled["reason"] = "texture_size_mismatch"
        disabled["baseline_texture_size"] = int(texture_size)
        disabled["prior_texture_size"] = int(prior_texture_size)
        return clone_atlas(baseline_atlas), disabled
    common_faces = set(baseline_atlas.keys()) & set(prior_atlas.keys())
    if not common_faces:
        disabled["reason"] = "no_common_faces"
        return clone_atlas(baseline_atlas), disabled
    target_samples_by_bin: dict[int, int] = {}
    target_views_by_bin: dict[int, int] = {}
    target_footprint_summary: dict[str, Any] = {
        "enabled": False,
        "mode": "target_footprint_bin_certificate",
        "reason": "not_requested",
    }
    if bool(enable_target_footprint_bin_certificate):
        target_samples_by_bin, target_views_by_bin, target_footprint_summary = build_target_bin_footprint_stats(
            list(target_footprint_views or []),
            candidate_faces=set(common_faces),
            texture_size=int(texture_size),
            min_alpha=float(min_alpha),
            max_views=int(target_footprint_max_views),
        )
        target_footprint_summary["min_bin_pixels"] = int(target_footprint_min_bin_pixels)
        target_footprint_summary["min_views"] = int(target_footprint_min_views)
        target_footprint_summary["min_view_fraction"] = float(target_footprint_min_view_fraction)
        if not bool(target_footprint_summary.get("enabled", False)):
            disabled["reason"] = "target_footprint_certificate_unavailable"
            disabled["target_footprint_bin_certificate"] = dict(target_footprint_summary)
            return clone_atlas(baseline_atlas), disabled

    rng = np.random.default_rng(83)
    bins_per_face = int(texture_size * texture_size)
    baseline_after_by_bin: dict[int, float] = {}
    prior_after_by_bin: dict[int, float] = {}
    source_mixture_num_by_bin: dict[int, float] = {}
    source_mixture_den_by_bin: dict[int, float] = {}
    source_mixture_view_components_by_bin: dict[int, list[tuple[float, float, float]]] = {}
    samples_by_bin: dict[int, int] = {}
    view_count_by_bin: dict[int, int] = {}
    positive_view_count_by_bin: dict[int, int] = {}
    view_gain_by_bin: dict[int, list[float]] = {}
    baseline_l1_after_by_bin: dict[int, float] = {}
    prior_l1_after_by_bin: dict[int, float] = {}
    l1_positive_view_count_by_bin: dict[int, int] = {}
    l1_view_gain_by_bin: dict[int, list[float]] = {}
    total_active_samples = 0
    views_used = 0
    comparison_alpha = float(min(max(float(baseline_alpha), 0.0), max(float(prior_alpha), 0.0)))
    if comparison_alpha <= 0.0:
        disabled["reason"] = "nonpositive_comparison_alpha"
        disabled["baseline_alpha"] = float(baseline_alpha)
        disabled["prior_alpha"] = float(prior_alpha)
        return clone_atlas(baseline_atlas), disabled

    for path in tqdm(val_views, desc="policy-val prior bin-gain hybrid"):
        z = np.load(path)
        if residual_rgb_key not in z:
            continue
        mask = _valid_sample_mask(z, common_faces, residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            mask = np.zeros_like(mask, dtype=bool)
            mask[ys, xs] = True
        residual = np.asarray(z[residual_rgb_key], dtype=np.float32)
        target = np.stack([residual[0][mask], residual[1][mask], residual[2][mask]], axis=1).astype(np.float64)
        baseline_pred, _baseline_valid = predict_delta_for_npz(
            z,
            baseline_atlas,
            comparison_alpha,
            min_alpha,
            min_atlas_bin_count=int(min_atlas_bin_count),
            min_atlas_face_samples=int(min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(atlas_confidence_mode),
            atlas_confidence_count_scale=float(atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(min_atlas_confidence),
            local_alpha_profile=baseline_local_alpha_profile,
            face_gain_guard_profile=baseline_face_gain_guard_profile,
            bin_uncertainty_guard_profile=baseline_bin_uncertainty_guard_profile,
        )
        prior_pred, _prior_valid = predict_delta_for_npz(
            z,
            prior_atlas,
            comparison_alpha,
            min_alpha,
            min_atlas_bin_count=int(min_atlas_bin_count),
            min_atlas_face_samples=int(min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(atlas_confidence_mode),
            atlas_confidence_count_scale=float(atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(min_atlas_confidence),
            local_alpha_profile=prior_local_alpha_profile,
            face_gain_guard_profile=prior_face_gain_guard_profile,
            bin_uncertainty_guard_profile=prior_bin_uncertainty_guard_profile,
        )
        baseline_pred = clip_delta_rgb(baseline_pred, float(max_abs_delta_rgb))
        prior_pred = clip_delta_rgb(prior_pred, float(max_abs_delta_rgb))
        baseline_samples = np.stack(
            [baseline_pred[0][mask], baseline_pred[1][mask], baseline_pred[2][mask]],
            axis=1,
        ).astype(np.float64)
        prior_samples = np.stack(
            [prior_pred[0][mask], prior_pred[1][mask], prior_pred[2][mask]],
            axis=1,
        ).astype(np.float64)
        changed = np.linalg.norm(prior_samples - baseline_samples, axis=1) > 1.0e-12
        if not bool(np.any(changed)):
            continue
        bary = np.asarray(z["barycentric"], dtype=np.float32)
        ubin, vbin = _uv_bins(bary, mask, texture_size)
        face_samples = np.asarray(z["face_id"], dtype=np.int64)[mask]
        bin_ids = vbin.astype(np.int64) * int(texture_size) + ubin.astype(np.int64)
        keys = face_samples.astype(np.int64) * int(bins_per_face) + bin_ids.astype(np.int64)
        target = target[changed]
        baseline_samples = baseline_samples[changed]
        prior_samples = prior_samples[changed]
        keys = keys[changed]
        baseline_after = np.sum((target - baseline_samples) ** 2, axis=1)
        prior_after = np.sum((target - prior_samples) ** 2, axis=1)
        source_mixture_delta = prior_samples - baseline_samples
        source_mixture_residual = target - baseline_samples
        source_mixture_num = np.sum(source_mixture_residual * source_mixture_delta, axis=1)
        source_mixture_den = np.sum(source_mixture_delta * source_mixture_delta, axis=1)
        baseline_l1_after = np.mean(np.abs(target - baseline_samples), axis=1)
        prior_l1_after = np.mean(np.abs(target - prior_samples), axis=1)
        total_active_samples += int(target.shape[0])
        views_used += 1
        for key in np.unique(keys):
            bm = keys == int(key)
            baseline_sum = float(np.sum(baseline_after[bm]))
            prior_sum = float(np.sum(prior_after[bm]))
            view_gain = float(baseline_sum - prior_sum)
            source_mixture_num_sum = float(np.sum(source_mixture_num[bm]))
            source_mixture_den_sum = float(np.sum(source_mixture_den[bm]))
            baseline_l1_sum = float(np.sum(baseline_l1_after[bm]))
            prior_l1_sum = float(np.sum(prior_l1_after[bm]))
            l1_view_gain = float(baseline_l1_sum - prior_l1_sum)
            baseline_after_by_bin[int(key)] = baseline_after_by_bin.get(int(key), 0.0) + baseline_sum
            prior_after_by_bin[int(key)] = prior_after_by_bin.get(int(key), 0.0) + prior_sum
            source_mixture_num_by_bin[int(key)] = (
                source_mixture_num_by_bin.get(int(key), 0.0) + source_mixture_num_sum
            )
            source_mixture_den_by_bin[int(key)] = (
                source_mixture_den_by_bin.get(int(key), 0.0) + source_mixture_den_sum
            )
            source_mixture_view_components_by_bin.setdefault(int(key), []).append(
                (baseline_sum, source_mixture_num_sum, source_mixture_den_sum)
            )
            baseline_l1_after_by_bin[int(key)] = baseline_l1_after_by_bin.get(int(key), 0.0) + baseline_l1_sum
            prior_l1_after_by_bin[int(key)] = prior_l1_after_by_bin.get(int(key), 0.0) + prior_l1_sum
            samples_by_bin[int(key)] = samples_by_bin.get(int(key), 0) + int(np.sum(bm))
            view_count_by_bin[int(key)] = view_count_by_bin.get(int(key), 0) + 1
            view_gain_by_bin.setdefault(int(key), []).append(view_gain)
            l1_view_gain_by_bin.setdefault(int(key), []).append(l1_view_gain)
            if baseline_sum > prior_sum:
                positive_view_count_by_bin[int(key)] = positive_view_count_by_bin.get(int(key), 0) + 1
            if baseline_l1_sum > prior_l1_sum:
                l1_positive_view_count_by_bin[int(key)] = l1_positive_view_count_by_bin.get(int(key), 0) + 1

    if not baseline_after_by_bin:
        disabled["reason"] = "no_changed_policy_val_bins"
        disabled["policy_val_views_used"] = int(views_used)
        disabled["active_samples"] = int(total_active_samples)
        return clone_atlas(baseline_atlas), disabled

    rows: list[dict[str, Any]] = []
    allowed_keys: list[int] = []
    source_mixture_weight_by_bin: dict[int, float] = {}
    source_mixture_ridge_mode = str(source_mixture_ridge_mode)
    source_mixture_positive_den = np.asarray(
        [float(v) for v in source_mixture_den_by_bin.values() if float(v) > 0.0],
        dtype=np.float64,
    )
    source_mixture_den_reference = (
        float(np.median(source_mixture_positive_den)) if source_mixture_positive_den.size else 0.0
    )
    source_mixture_ridge_terms: list[float] = []
    for key in sorted(baseline_after_by_bin):
        baseline_after = float(baseline_after_by_bin[key])
        hard_prior_after = float(prior_after_by_bin.get(key, 0.0))
        prior_after = hard_prior_after
        source_mixture_num_total = float(source_mixture_num_by_bin.get(key, 0.0))
        source_mixture_den_total = float(source_mixture_den_by_bin.get(key, 0.0))
        source_mixture_weight = 1.0
        source_mixture_ridge_term = 0.0
        if bool(source_mixture_enabled):
            if source_mixture_ridge_mode == "adaptive_den":
                source_mixture_ridge_term = max(float(source_mixture_ridge), 0.0) * max(
                    float(source_mixture_den_total),
                    float(source_mixture_den_reference),
                )
            else:
                source_mixture_ridge_term = max(float(source_mixture_ridge), 0.0)
            source_mixture_ridge_terms.append(float(source_mixture_ridge_term))
            denominator = source_mixture_den_total + source_mixture_ridge_term
            if denominator > 0.0:
                source_mixture_weight = float(np.clip(source_mixture_num_total / denominator, 0.0, 1.0))
            else:
                source_mixture_weight = 0.0
            prior_after = float(
                baseline_after
                - 2.0 * source_mixture_weight * source_mixture_num_total
                + source_mixture_weight * source_mixture_weight * source_mixture_den_total
            )
        source_mixture_weight_by_bin[int(key)] = float(source_mixture_weight)
        samples = int(samples_by_bin.get(key, 0))
        views = int(view_count_by_bin.get(key, 0))
        positive_views = int(positive_view_count_by_bin.get(key, 0))
        abs_gain = baseline_after - prior_after
        relative_gain = abs_gain / max(baseline_after, 1.0e-12)
        positive_fraction = float(positive_views / max(1, views))
        baseline_l1_after = float(baseline_l1_after_by_bin.get(key, 0.0))
        prior_l1_after = float(prior_l1_after_by_bin.get(key, 0.0))
        l1_abs_gain = baseline_l1_after - prior_l1_after
        l1_relative_gain = l1_abs_gain / max(baseline_l1_after, 1.0e-12)
        l1_positive_views = int(l1_positive_view_count_by_bin.get(key, 0))
        l1_positive_fraction = float(l1_positive_views / max(1, views))
        view_gains = np.asarray(view_gain_by_bin.get(key, []), dtype=np.float64)
        if view_gains.size:
            sorted_view_gains = np.sort(view_gains)
            cvar_count = max(1, int(math.ceil(0.20 * float(sorted_view_gains.size))))
            mean_view_gain = float(np.mean(view_gains))
            min_view_gain = float(np.min(view_gains))
            cvar20_view_gain = float(np.mean(sorted_view_gains[:cvar_count]))
        else:
            mean_view_gain = 0.0
            min_view_gain = 0.0
            cvar20_view_gain = 0.0
        if bool(source_mixture_enabled):
            source_components = source_mixture_view_components_by_bin.get(key, [])
            if source_components:
                source_view_gains = np.asarray(
                    [
                        2.0 * source_mixture_weight * float(num_sum)
                        - source_mixture_weight * source_mixture_weight * float(den_sum)
                        for _baseline_sum, num_sum, den_sum in source_components
                    ],
                    dtype=np.float64,
                )
                source_sorted_view_gains = np.sort(source_view_gains)
                source_cvar_count = max(1, int(math.ceil(0.20 * float(source_sorted_view_gains.size))))
                mean_view_gain = float(np.mean(source_view_gains))
                min_view_gain = float(np.min(source_view_gains))
                cvar20_view_gain = float(np.mean(source_sorted_view_gains[:source_cvar_count]))
                positive_views = int(np.sum(source_view_gains > 0.0))
                positive_fraction = float(positive_views / max(1, views))
            else:
                mean_view_gain = 0.0
                min_view_gain = 0.0
                cvar20_view_gain = 0.0
                positive_views = 0
                positive_fraction = 0.0
        l1_view_gains = np.asarray(l1_view_gain_by_bin.get(key, []), dtype=np.float64)
        if l1_view_gains.size:
            sorted_l1_view_gains = np.sort(l1_view_gains)
            l1_cvar_count = max(1, int(math.ceil(0.20 * float(sorted_l1_view_gains.size))))
            l1_mean_view_gain = float(np.mean(l1_view_gains))
            l1_min_view_gain = float(np.min(l1_view_gains))
            l1_cvar20_view_gain = float(np.mean(sorted_l1_view_gains[:l1_cvar_count]))
        else:
            l1_mean_view_gain = 0.0
            l1_min_view_gain = 0.0
            l1_cvar20_view_gain = 0.0
        target_pixels = int(target_samples_by_bin.get(key, 0))
        target_views = int(target_views_by_bin.get(key, 0))
        target_view_denominator = int(
            (target_footprint_summary or {}).get(
                "target_views_examined",
                (target_footprint_summary or {}).get("target_views_used", 0),
            )
            or 0
        )
        target_view_fraction = float(
            target_views / max(1, target_view_denominator)
        )
        target_footprint_keep = bool(
            not bool(enable_target_footprint_bin_certificate)
            or (
                target_pixels >= int(target_footprint_min_bin_pixels)
                and target_views >= int(target_footprint_min_views)
                and target_view_fraction >= float(target_footprint_min_view_fraction)
            )
        )
        target_tail_risk_applies = bool(enable_target_footprint_tail_risk_certificate) and (
            not bool(target_footprint_tail_risk_only_target_bins) or target_pixels > 0
        )
        target_tail_risk_keep = bool(
            not target_tail_risk_applies
            or (
                positive_fraction >= float(target_footprint_tail_risk_min_positive_view_fraction)
                and min_view_gain >= float(target_footprint_tail_risk_min_min_view_gain)
                and cvar20_view_gain >= float(target_footprint_tail_risk_min_cvar20_view_gain)
            )
        )
        l1_proxy_keep = bool(
            not bool(enable_l1_proxy_gate)
            or (
                l1_abs_gain >= float(min_l1_abs_gain)
                and l1_relative_gain >= float(min_l1_relative_gain)
                and l1_positive_fraction >= float(min_l1_positive_view_fraction)
                and l1_min_view_gain >= float(min_l1_min_view_gain)
                and l1_cvar20_view_gain >= float(min_l1_cvar20_view_gain)
                and prior_l1_after < baseline_l1_after
            )
        )
        keep = bool(
            samples >= int(min_bin_samples)
            and views >= int(min_views)
            and abs_gain >= float(min_abs_gain)
            and relative_gain >= float(min_relative_gain)
            and positive_fraction >= float(min_positive_view_fraction)
            and l1_proxy_keep
            and target_footprint_keep
            and target_tail_risk_keep
            and prior_after < baseline_after
            and (
                not bool(source_mixture_enabled)
                or source_mixture_weight >= float(source_mixture_min_weight)
            )
        )
        if keep:
            allowed_keys.append(int(key))
        rows.append(
            {
                "face": int(key // bins_per_face),
                "bin": int(key % bins_per_face),
                "samples": int(samples),
                "views": int(views),
                "positive_views": int(positive_views),
                "positive_view_fraction": float(positive_fraction),
                "l1_positive_views": int(l1_positive_views),
                "l1_positive_view_fraction": float(l1_positive_fraction),
                "mean_view_gain": float(mean_view_gain),
                "min_view_gain": float(min_view_gain),
                "cvar20_view_gain": float(cvar20_view_gain),
                "l1_mean_view_gain": float(l1_mean_view_gain),
                "l1_min_view_gain": float(l1_min_view_gain),
                "l1_cvar20_view_gain": float(l1_cvar20_view_gain),
                "baseline_after": float(baseline_after),
                "prior_after": float(prior_after),
                "hard_prior_after": float(hard_prior_after),
                "abs_gain": float(abs_gain),
                "relative_gain": float(relative_gain),
                "source_mixture_enabled": bool(source_mixture_enabled),
                "source_mixture_weight": float(source_mixture_weight),
                "source_mixture_numerator": float(source_mixture_num_total),
                "source_mixture_denominator": float(source_mixture_den_total),
                "source_mixture_ridge_term": float(source_mixture_ridge_term),
                "baseline_l1_after": float(baseline_l1_after),
                "prior_l1_after": float(prior_l1_after),
                "l1_abs_gain": float(l1_abs_gain),
                "l1_relative_gain": float(l1_relative_gain),
                "target_pixels": int(target_pixels),
                "target_views": int(target_views),
                "target_view_fraction": float(target_view_fraction),
                "target_footprint_keep": bool(target_footprint_keep),
                "target_tail_risk_applies": bool(target_tail_risk_applies),
                "target_tail_risk_keep": bool(target_tail_risk_keep),
                "l1_proxy_keep": bool(l1_proxy_keep),
                "keep": bool(keep),
            }
        )

    allowed_truncated = False
    if int(max_profile_bins) > 0 and len(allowed_keys) > int(max_profile_bins):
        rank = {
            int(row["face"]) * int(bins_per_face) + int(row["bin"]): (
                float(row["relative_gain"]),
                int(row["samples"]),
            )
            for row in rows
            if bool(row.get("keep", False))
        }
        allowed_keys = sorted(allowed_keys, key=lambda key: rank.get(int(key), (0.0, 0)), reverse=True)[
            : int(max_profile_bins)
        ]
        allowed_truncated = True

    kept_rows_pre_trunc = [row for row in rows if bool(row.get("keep", False))]
    kept_rows_pre_trunc = sorted(
        kept_rows_pre_trunc,
        key=lambda row: (float(row.get("relative_gain", 0.0)), int(row.get("samples", 0))),
        reverse=True,
    )
    allowed_key_set = {int(key) for key in allowed_keys}
    kept_rows = [
        row
        for row in kept_rows_pre_trunc
        if int(row["face"]) * int(bins_per_face) + int(row["bin"]) in allowed_key_set
    ]
    target_footprint_profile = {
        **dict(target_footprint_summary),
        "candidate_bin_count": int(len(rows)),
        "candidate_bins_with_target_footprint": int(
            sum(1 for row in rows if int(row.get("target_pixels", 0)) > 0)
        ),
        "pre_trunc_allowed_bins_with_target_footprint": int(
            sum(1 for row in kept_rows_pre_trunc if int(row.get("target_pixels", 0)) > 0)
        ),
        "allowed_bins_with_target_footprint": int(
            sum(1 for row in kept_rows if int(row.get("target_pixels", 0)) > 0)
        ),
    }
    target_tail_risk_profile = {
        "enabled": bool(enable_target_footprint_tail_risk_certificate),
        "mode": "target_footprint_tail_risk_certificate",
        "only_target_bins": bool(target_footprint_tail_risk_only_target_bins),
        "min_positive_view_fraction": float(
            target_footprint_tail_risk_min_positive_view_fraction
        ),
        "min_min_view_gain": float(target_footprint_tail_risk_min_min_view_gain),
        "min_cvar20_view_gain": float(target_footprint_tail_risk_min_cvar20_view_gain),
        "candidate_bin_count": int(len(rows)),
        "applied_bin_count": int(
            sum(1 for row in rows if bool(row.get("target_tail_risk_applies", False)))
        ),
        "pre_trunc_keep_bin_count": int(
            sum(1 for row in kept_rows_pre_trunc if bool(row.get("target_tail_risk_keep", False)))
        ),
        "allowed_keep_bin_count": int(
            sum(1 for row in kept_rows if bool(row.get("target_tail_risk_keep", False)))
        ),
        "rejected_bin_count": int(
            sum(
                1
                for row in rows
                if bool(row.get("target_tail_risk_applies", False))
                and not bool(row.get("target_tail_risk_keep", False))
            )
        ),
        "candidate_bins_with_target_footprint": int(
            sum(1 for row in rows if int(row.get("target_pixels", 0)) > 0)
        ),
        "rejected_bins_with_target_footprint": int(
            sum(
                1
                for row in rows
                if int(row.get("target_pixels", 0)) > 0
                and not bool(row.get("target_tail_risk_keep", False))
            )
        ),
    }
    l1_proxy_gate_profile = {
        "enabled": bool(enable_l1_proxy_gate),
        "mode": "prior_bin_gain_l1_proxy_gate",
        "min_l1_abs_gain": float(min_l1_abs_gain),
        "min_l1_relative_gain": float(min_l1_relative_gain),
        "min_l1_positive_view_fraction": float(min_l1_positive_view_fraction),
        "min_l1_min_view_gain": float(min_l1_min_view_gain),
        "min_l1_cvar20_view_gain": float(min_l1_cvar20_view_gain),
        "candidate_bin_count": int(len(rows)),
        "pre_trunc_keep_bin_count": int(
            sum(1 for row in kept_rows_pre_trunc if bool(row.get("l1_proxy_keep", False)))
        ),
        "allowed_keep_bin_count": int(
            sum(1 for row in kept_rows if bool(row.get("l1_proxy_keep", False)))
        ),
        "rejected_bin_count": int(
            sum(1 for row in rows if not bool(row.get("l1_proxy_keep", False)))
        ),
    }

    hybrid_atlas = clone_atlas(baseline_atlas)
    copied = 0
    for key in allowed_keys:
        face = int(key // bins_per_face)
        bin_id = int(key % bins_per_face)
        if face not in hybrid_atlas or face not in prior_atlas:
            continue
        v = int(bin_id // texture_size)
        u = int(bin_id % texture_size)
        if bool(source_mixture_enabled):
            _blend_face_atlas_bin(
                hybrid_atlas[face],
                prior_atlas[face],
                v,
                u,
                source_mixture_weight_by_bin.get(int(key), 0.0),
            )
        else:
            _copy_face_atlas_bin(hybrid_atlas[face], prior_atlas[face], v, u)
        copied += 1

    if copied <= 0:
        disabled["reason"] = "no_bins_passed_hybrid_gate"
        disabled["candidate_bin_count"] = int(len(rows))
        disabled["policy_val_views_used"] = int(views_used)
        disabled["active_samples"] = int(total_active_samples)
        disabled["pre_trunc_allowed_bin_count"] = int(len(kept_rows_pre_trunc))
        disabled["allowed_bin_count"] = 0
        disabled["target_footprint_bin_certificate"] = dict(target_footprint_profile)
        disabled["target_footprint_tail_risk_certificate"] = dict(target_tail_risk_profile)
        disabled["l1_proxy_gate"] = dict(l1_proxy_gate_profile)
        disabled["top_bins"] = kept_rows_pre_trunc[:128]
        return clone_atlas(baseline_atlas), disabled

    allowed_bins_by_face: dict[str, list[int]] = {}
    for key in sorted(allowed_key_set):
        face = int(key // bins_per_face)
        bin_id = int(key % bins_per_face)
        allowed_bins_by_face.setdefault(str(face), []).append(int(bin_id))
    profile = {
        "enabled": True,
        "mode": "policy_val_prior_bin_gain_hybrid",
        "source_mixture_enabled": bool(source_mixture_enabled),
        "source_mixture_ridge_mode": str(source_mixture_ridge_mode),
        "source_mixture_ridge": float(source_mixture_ridge),
        "source_mixture_min_weight": float(source_mixture_min_weight),
        "source_mixture_den_reference": float(source_mixture_den_reference),
        "source_mixture_ridge_term_mean": float(np.mean(source_mixture_ridge_terms))
        if source_mixture_ridge_terms
        else 0.0,
        "source_mixture_ridge_term_min": float(np.min(source_mixture_ridge_terms))
        if source_mixture_ridge_terms
        else 0.0,
        "source_mixture_ridge_term_max": float(np.max(source_mixture_ridge_terms))
        if source_mixture_ridge_terms
        else 0.0,
        "source_mixture_weight_mean": float(
            np.mean([float(row.get("source_mixture_weight", 1.0)) for row in kept_rows])
            if kept_rows
            else 0.0
        ),
        "source_mixture_weight_min": float(
            np.min([float(row.get("source_mixture_weight", 1.0)) for row in kept_rows])
            if kept_rows
            else 0.0
        ),
        "source_mixture_weight_max": float(
            np.max([float(row.get("source_mixture_weight", 1.0)) for row in kept_rows])
            if kept_rows
            else 0.0
        ),
        "texture_size": int(texture_size),
        "comparison_alpha": float(comparison_alpha),
        "baseline_alpha": float(baseline_alpha),
        "prior_alpha": float(prior_alpha),
        "min_bin_samples": int(min_bin_samples),
        "min_views": int(min_views),
        "min_abs_gain": float(min_abs_gain),
        "min_relative_gain": float(min_relative_gain),
        "min_positive_view_fraction": float(min_positive_view_fraction),
        "l1_proxy_gate": dict(l1_proxy_gate_profile),
        "max_profile_bins": int(max_profile_bins),
        "candidate_bin_count": int(len(rows)),
        "pre_trunc_allowed_bin_count": int(len(kept_rows_pre_trunc)),
        "allowed_bin_count": int(copied),
        "allowed_bin_fraction": float(copied / max(1, len(rows))),
        "allowed_truncated": bool(allowed_truncated),
        "allowed_bins_by_face": allowed_bins_by_face,
        "policy_val_views_used": int(views_used),
        "active_samples": int(total_active_samples),
        "target_footprint_bin_certificate": dict(target_footprint_profile),
        "target_footprint_tail_risk_certificate": dict(target_tail_risk_profile),
        "top_bins": kept_rows[:128],
    }
    return hybrid_atlas, profile


def build_policy_val_face_gain_guard_profile(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    residual_rgb_key: str,
    residual_l1_key: str,
    alpha: float,
    min_l1: float,
    min_alpha: float,
    max_abs_delta_rgb: float,
    max_samples_per_view: int,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    local_alpha_profile: dict[str, Any] | None,
    min_face_samples: int,
    min_relative_gain: float,
    min_positive_view_fraction: float,
) -> dict[str, Any]:
    """Build a train-policy-val face allowlist for residual application."""
    disabled = {
        "enabled": False,
        "mode": "policy_val_face_gain_guard",
        "reason": "",
    }
    if not val_views or not atlas or float(alpha) <= 0.0:
        disabled["reason"] = "no_policy_val_views_or_empty_atlas_or_zero_alpha"
        return disabled
    rng = np.random.default_rng(59)
    before_by_face: dict[int, float] = {}
    after_by_face: dict[int, float] = {}
    samples_by_face: dict[int, int] = {}
    view_count_by_face: dict[int, int] = {}
    positive_view_count_by_face: dict[int, int] = {}
    total_active_samples = 0
    view_count = 0
    for path in tqdm(val_views, desc="face gain guard"):
        z = np.load(path)
        if residual_rgb_key not in z:
            continue
        mask = _valid_sample_mask(z, set(atlas.keys()), residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            mask = np.zeros_like(mask, dtype=bool)
            mask[ys, xs] = True
        target_rgb = np.asarray(z[residual_rgb_key], dtype=np.float32)
        target = np.stack([target_rgb[0][mask], target_rgb[1][mask], target_rgb[2][mask]], axis=1)
        face_samples = np.asarray(z["face_id"], dtype=np.int64)[mask]
        pred, _valid = predict_delta_for_npz(
            z,
            atlas,
            float(alpha),
            min_alpha,
            min_atlas_bin_count=int(min_atlas_bin_count),
            min_atlas_face_samples=int(min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(atlas_confidence_mode),
            atlas_confidence_count_scale=float(atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(min_atlas_confidence),
            local_alpha_profile=local_alpha_profile,
        )
        pred = clip_delta_rgb(pred, float(max_abs_delta_rgb))
        pred_samples = np.stack([pred[0][mask], pred[1][mask], pred[2][mask]], axis=1).astype(np.float32)
        active = np.linalg.norm(pred_samples, axis=1) > 1.0e-12
        if not bool(np.any(active)):
            continue
        target = target.astype(np.float64)[active]
        pred_samples = pred_samples.astype(np.float64)[active]
        face_samples = face_samples[active]
        before = np.sum(target * target, axis=1)
        err = target - pred_samples
        after = np.sum(err * err, axis=1)
        total_active_samples += int(target.shape[0])
        view_count += 1
        for face in np.unique(face_samples):
            face = int(face)
            fm = face_samples == face
            face_before = float(np.sum(before[fm]))
            face_after = float(np.sum(after[fm]))
            before_by_face[face] = before_by_face.get(face, 0.0) + face_before
            after_by_face[face] = after_by_face.get(face, 0.0) + face_after
            samples_by_face[face] = samples_by_face.get(face, 0) + int(np.sum(fm))
            view_count_by_face[face] = view_count_by_face.get(face, 0) + 1
            gain = (face_before - face_after) / max(face_before, 1.0e-12)
            if gain > 0.0:
                positive_view_count_by_face[face] = positive_view_count_by_face.get(face, 0) + 1

    if not before_by_face:
        disabled["reason"] = "no_active_policy_val_predictions"
        disabled["policy_val_views_used"] = int(view_count)
        disabled["samples"] = int(total_active_samples)
        return disabled

    rows: list[dict[str, Any]] = []
    allowed_faces: list[int] = []
    allowed_samples = 0
    for face in sorted(before_by_face):
        before = float(before_by_face[face])
        after = float(after_by_face.get(face, 0.0))
        samples = int(samples_by_face.get(face, 0))
        views = int(view_count_by_face.get(face, 0))
        positive_views = int(positive_view_count_by_face.get(face, 0))
        relative_gain = (before - after) / max(before, 1.0e-12)
        positive_fraction = float(positive_views / max(1, views))
        keep = bool(
            samples >= int(min_face_samples)
            and relative_gain >= float(min_relative_gain)
            and positive_fraction >= float(min_positive_view_fraction)
        )
        if keep:
            allowed_faces.append(int(face))
            allowed_samples += int(samples)
        rows.append(
            {
                "face_id": int(face),
                "samples": int(samples),
                "view_count": int(views),
                "positive_view_count": int(positive_views),
                "positive_view_fraction": float(positive_fraction),
                "mse_before_sum": float(before),
                "mse_after_sum": float(after),
                "relative_gain": float(relative_gain),
                "keep": bool(keep),
            }
        )
    rows_by_gain = sorted(rows, key=lambda row: float(row["relative_gain"]))
    rows_by_samples = sorted(rows, key=lambda row: int(row["samples"]), reverse=True)
    return {
        "enabled": True,
        "mode": "policy_val_face_gain_guard",
        "alpha": float(alpha),
        "policy_val_views_used": int(view_count),
        "samples": int(total_active_samples),
        "min_face_samples": int(min_face_samples),
        "min_relative_gain": float(min_relative_gain),
        "min_positive_view_fraction": float(min_positive_view_fraction),
        "candidate_face_count": int(len(rows)),
        "allowed_face_count": int(len(allowed_faces)),
        "rejected_face_count": int(len(rows) - len(allowed_faces)),
        "allowed_sample_fraction": float(allowed_samples / max(1, total_active_samples)),
        "allowed_faces": [int(face) for face in sorted(allowed_faces)],
        "worst_faces": rows_by_gain[:32],
        "best_sampled_faces": rows_by_samples[:32],
    }


def build_policy_val_bin_uncertainty_guard_profile(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    residual_rgb_key: str,
    residual_l1_key: str,
    alpha: float,
    min_l1: float,
    min_alpha: float,
    max_abs_delta_rgb: float,
    max_samples_per_view: int,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    local_alpha_profile: dict[str, Any] | None,
    face_gain_guard_profile: dict[str, Any] | None,
    min_bin_samples: int,
    min_bin_views: int = 1,
    min_relative_gain: float = 0.0,
    min_view_relative_gain: float = -float("inf"),
    min_positive_view_fraction: float = 0.5,
    max_mean_variance: float = -1.0,
    min_mean_sign_consistency: float = 0.0,
    adaptive_frontier_on_empty: bool = False,
    adaptive_frontier_min_positive_view_fraction: float = 0.55,
    adaptive_frontier_min_risk_adjusted_gain: float = 0.0,
    adaptive_frontier_min_sample_quantile: float = 0.75,
    target_footprint_views: list[Path] | None = None,
    enable_target_visible_expansion: bool = False,
    target_visible_min_pixels: int = 1,
    target_visible_min_views: int = 1,
    target_visible_min_policy_samples: int = 1,
    target_visible_min_positive_view_fraction: float = -1.0,
    target_visible_max_extra_bins: int = 0,
    target_visible_max_views: int = 0,
    enable_train_only_target_impact_residual_basis: bool = False,
    target_impact_min_pixels: int = 1,
    target_impact_min_views: int = 1,
    target_impact_min_policy_samples: int = 0,
    target_impact_max_extra_bins: int = 0,
    target_impact_max_views: int = 0,
    enable_target_connected_region_growth: bool = False,
    target_connected_radius: int = 1,
    target_connected_min_pixels: int = 1,
    target_connected_min_views: int = 1,
    target_connected_min_policy_samples: int = 1,
    target_connected_min_positive_view_fraction: float = 0.5,
    target_connected_max_negative_relative_gain: float = 0.02,
    target_connected_max_negative_min_view_gain: float = 0.05,
    target_connected_max_extra_bins: int = 0,
    target_connected_max_views: int = 0,
) -> dict[str, Any]:
    """Build a train-policy-val face/UV-bin allowlist with uncertainty filters."""
    disabled = {
        "enabled": False,
        "mode": "policy_val_bin_uncertainty_guard",
        "reason": "",
    }
    if not val_views or not atlas or float(alpha) <= 0.0:
        disabled["reason"] = "no_policy_val_views_or_empty_atlas_or_zero_alpha"
        return disabled
    rng = np.random.default_rng(62)
    before_by_key: dict[tuple[int, int], float] = {}
    after_by_key: dict[tuple[int, int], float] = {}
    samples_by_key: dict[tuple[int, int], int] = {}
    view_count_by_key: dict[tuple[int, int], int] = {}
    positive_view_count_by_key: dict[tuple[int, int], int] = {}
    min_view_gain_by_key: dict[tuple[int, int], float] = {}
    variance_sum_by_key: dict[tuple[int, int], float] = {}
    sign_sum_by_key: dict[tuple[int, int], float] = {}
    total_active_samples = 0
    view_count = 0
    texture_size = int(next(iter(atlas.values())).texture.shape[0])
    for path in tqdm(val_views, desc="bin uncertainty guard"):
        z = np.load(path)
        if residual_rgb_key not in z or "barycentric" not in z:
            continue
        mask = _valid_sample_mask(z, set(atlas.keys()), residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            mask = np.zeros_like(mask, dtype=bool)
            mask[ys, xs] = True
        bary = np.asarray(z["barycentric"], dtype=np.float32)
        uv_u, uv_v = _uv_bins(bary, mask, texture_size)
        face_samples = np.asarray(z["face_id"], dtype=np.int64)[mask]
        target_rgb = np.asarray(z[residual_rgb_key], dtype=np.float32)
        target = np.stack([target_rgb[0][mask], target_rgb[1][mask], target_rgb[2][mask]], axis=1)
        pred, _valid = predict_delta_for_npz(
            z,
            atlas,
            float(alpha),
            min_alpha,
            min_atlas_bin_count=int(min_atlas_bin_count),
            min_atlas_face_samples=int(min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(atlas_confidence_mode),
            atlas_confidence_count_scale=float(atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(min_atlas_confidence),
            local_alpha_profile=local_alpha_profile,
            face_gain_guard_profile=face_gain_guard_profile,
        )
        pred = clip_delta_rgb(pred, float(max_abs_delta_rgb))
        pred_samples = np.stack([pred[0][mask], pred[1][mask], pred[2][mask]], axis=1).astype(np.float32)
        active = np.linalg.norm(pred_samples, axis=1) > 1.0e-12
        if not bool(np.any(active)):
            continue
        target = target.astype(np.float64)[active]
        pred_samples = pred_samples.astype(np.float64)[active]
        face_samples = face_samples[active]
        uv_u = uv_u[active]
        uv_v = uv_v[active]
        before = np.sum(target * target, axis=1)
        err = target - pred_samples
        after = np.sum(err * err, axis=1)
        total_active_samples += int(target.shape[0])
        view_count += 1
        bin_ids = (uv_v.astype(np.int64) * texture_size) + uv_u.astype(np.int64)
        for face in np.unique(face_samples):
            face_i = int(face)
            if face_i not in atlas:
                continue
            face_mask = face_samples == face_i
            face_atlas = atlas[face_i]
            for bin_id in np.unique(bin_ids[face_mask]):
                key = (face_i, int(bin_id))
                bm = face_mask & (bin_ids == int(bin_id))
                local_before = float(np.sum(before[bm]))
                local_after = float(np.sum(after[bm]))
                local_count = int(np.sum(bm))
                before_by_key[key] = before_by_key.get(key, 0.0) + local_before
                after_by_key[key] = after_by_key.get(key, 0.0) + local_after
                samples_by_key[key] = samples_by_key.get(key, 0) + local_count
                view_count_by_key[key] = view_count_by_key.get(key, 0) + 1
                gain = (local_before - local_after) / max(local_before, 1.0e-12)
                min_view_gain_by_key[key] = min(float(min_view_gain_by_key.get(key, gain)), float(gain))
                if gain > 0.0:
                    positive_view_count_by_key[key] = positive_view_count_by_key.get(key, 0) + 1
                ybin = int(bin_id) // texture_size
                xbin = int(bin_id) % texture_size
                local_variance = float(np.mean(face_atlas.variance[ybin, xbin]))
                local_sign = float(np.mean(face_atlas.sign_consistency[ybin, xbin]))
                variance_sum_by_key[key] = variance_sum_by_key.get(key, 0.0) + local_variance * local_count
                sign_sum_by_key[key] = sign_sum_by_key.get(key, 0.0) + local_sign * local_count

    if not before_by_key:
        disabled["reason"] = "no_active_policy_val_predictions"
        disabled["policy_val_views_used"] = int(view_count)
        disabled["samples"] = int(total_active_samples)
        return disabled

    rows: list[dict[str, Any]] = []
    allowed_bins_by_face: dict[str, list[int]] = {}
    allowed_samples = 0
    for key in sorted(before_by_key):
        face, bin_id = key
        before = float(before_by_key[key])
        after = float(after_by_key.get(key, 0.0))
        samples = int(samples_by_key.get(key, 0))
        views = int(view_count_by_key.get(key, 0))
        positive_views = int(positive_view_count_by_key.get(key, 0))
        min_view_gain = float(min_view_gain_by_key.get(key, 0.0))
        relative_gain = (before - after) / max(before, 1.0e-12)
        positive_fraction = float(positive_views / max(1, views))
        mean_variance = float(variance_sum_by_key.get(key, 0.0) / max(1, samples))
        mean_sign = float(sign_sum_by_key.get(key, 0.0) / max(1, samples))
        variance_ok = float(max_mean_variance) < 0.0 or mean_variance <= float(max_mean_variance)
        sign_ok = float(min_mean_sign_consistency) <= 0.0 or mean_sign >= float(min_mean_sign_consistency)
        min_view_ok = min_view_gain >= float(min_view_relative_gain)
        sample_ok = samples >= int(min_bin_samples)
        view_ok = views >= int(min_bin_views)
        relative_ok = relative_gain >= float(min_relative_gain)
        positive_ok = positive_fraction >= float(min_positive_view_fraction)
        risk_adjusted_gain = relative_gain + min(0.0, min_view_gain)
        keep = bool(
            sample_ok
            and view_ok
            and relative_ok
            and min_view_ok
            and positive_ok
            and variance_ok
            and sign_ok
        )
        if keep:
            allowed_bins_by_face.setdefault(str(int(face)), []).append(int(bin_id))
            allowed_samples += int(samples)
        rows.append(
            {
                "face_id": int(face),
                "bin_id": int(bin_id),
                "u": int(bin_id) % texture_size,
                "v": int(bin_id) // texture_size,
                "samples": int(samples),
                "view_count": int(views),
                "positive_view_count": int(positive_views),
                "positive_view_fraction": float(positive_fraction),
                "min_view_relative_gain": float(min_view_gain),
                "mse_before_sum": float(before),
                "mse_after_sum": float(after),
                "relative_gain": float(relative_gain),
                "risk_adjusted_relative_gain": float(risk_adjusted_gain),
                "mean_variance": float(mean_variance),
                "mean_sign_consistency": float(mean_sign),
                "sample_ok": bool(sample_ok),
                "view_ok": bool(view_ok),
                "relative_ok": bool(relative_ok),
                "positive_fraction_ok": bool(positive_ok),
                "variance_ok": bool(variance_ok),
                "sign_ok": bool(sign_ok),
                "min_view_ok": bool(min_view_ok),
                "keep": bool(keep),
            }
        )
    adaptive_frontier: dict[str, Any] = {
        "enabled": bool(adaptive_frontier_on_empty),
        "activated": False,
        "reason": "strict_certificate_nonempty" if allowed_bins_by_face else "not_requested",
        "min_positive_view_fraction": float(adaptive_frontier_min_positive_view_fraction),
        "min_risk_adjusted_gain": float(adaptive_frontier_min_risk_adjusted_gain),
        "min_sample_quantile": float(adaptive_frontier_min_sample_quantile),
        "requested_min_bin_samples": int(min_bin_samples),
        "effective_min_bin_samples": int(min_bin_samples),
    }
    if bool(adaptive_frontier_on_empty) and not allowed_bins_by_face:
        allowed_bins_by_face = {}
        allowed_samples = 0
        frontier_positive_fraction = min(
            float(min_positive_view_fraction),
            max(0.0, float(adaptive_frontier_min_positive_view_fraction)),
        )
        core_rows = [
            row
            for row in rows
            if (
                int(row["view_count"]) >= int(min_bin_views)
                and float(row["relative_gain"]) >= float(min_relative_gain)
                and float(row["risk_adjusted_relative_gain"]) >= float(adaptive_frontier_min_risk_adjusted_gain)
                and float(row["positive_view_fraction"]) >= float(frontier_positive_fraction)
                and bool(row["variance_ok"])
                and bool(row["sign_ok"])
            )
        ]
        effective_min_bin_samples = int(min_bin_samples)
        sample_quantiles: dict[str, float] = {}
        if core_rows:
            core_samples = np.asarray([int(row["samples"]) for row in core_rows], dtype=np.float64)
            q = float(np.clip(float(adaptive_frontier_min_sample_quantile), 0.0, 1.0))
            effective_min_bin_samples = min(
                int(min_bin_samples),
                max(1, int(np.floor(float(np.quantile(core_samples, q))))),
            )
            sample_quantiles = {
                "0.25": float(np.quantile(core_samples, 0.25)),
                "0.50": float(np.quantile(core_samples, 0.50)),
                "0.75": float(np.quantile(core_samples, 0.75)),
                "0.90": float(np.quantile(core_samples, 0.90)),
                "1.00": float(np.max(core_samples)),
            }
        for row in rows:
            frontier_sample_ok = int(row["samples"]) >= int(effective_min_bin_samples)
            frontier_keep = bool(
                frontier_sample_ok
                and int(row["view_count"]) >= int(min_bin_views)
                and float(row["relative_gain"]) >= float(min_relative_gain)
                and float(row["risk_adjusted_relative_gain"]) >= float(adaptive_frontier_min_risk_adjusted_gain)
                and float(row["positive_view_fraction"]) >= float(frontier_positive_fraction)
                and bool(row["variance_ok"])
                and bool(row["sign_ok"])
            )
            row["frontier_sample_ok"] = bool(frontier_sample_ok)
            row["frontier_keep"] = bool(frontier_keep)
            if frontier_keep:
                allowed_bins_by_face.setdefault(str(int(row["face_id"])), []).append(int(row["bin_id"]))
                allowed_samples += int(row["samples"])
        adaptive_frontier["activated"] = True
        adaptive_frontier["reason"] = (
            "frontier_selected_bins" if allowed_bins_by_face else "frontier_empty"
        )
        adaptive_frontier["effective_min_positive_view_fraction"] = float(frontier_positive_fraction)
        adaptive_frontier["core_candidate_count"] = int(len(core_rows))
        adaptive_frontier["sample_quantiles"] = dict(sample_quantiles)
        adaptive_frontier["effective_min_bin_samples"] = int(effective_min_bin_samples)
        adaptive_frontier["selected_bin_count"] = int(sum(len(v) for v in allowed_bins_by_face.values()))
    else:
        for row in rows:
            row["frontier_sample_ok"] = False
            row["frontier_keep"] = False

    target_samples_by_bin: dict[int, int] = {}
    target_views_by_bin: dict[int, int] = {}
    target_summary: dict[str, Any] = {}
    bins_per_face = int(texture_size) * int(texture_size)
    target_visible_expansion: dict[str, Any] = {
        "enabled": bool(enable_target_visible_expansion),
        "mode": "sparse_materialization_target_visible_expansion",
        "uses_target_or_test_gt": False,
        "reason": "not_requested",
        "min_pixels": int(target_visible_min_pixels),
        "min_views": int(target_visible_min_views),
        "min_policy_samples": int(target_visible_min_policy_samples),
        "min_positive_view_fraction": float(target_visible_min_positive_view_fraction),
        "max_extra_bins": int(target_visible_max_extra_bins),
        "max_views": int(target_visible_max_views),
        "original_allowed_bin_count": int(sum(len(v) for v in allowed_bins_by_face.values())),
        "original_allowed_face_count": int(len(allowed_bins_by_face)),
    }
    for row in rows:
        row["target_visible_pixels"] = 0
        row["target_visible_view_count"] = 0
        row["target_visible_expansion_candidate"] = False
        row["target_visible_expansion_added"] = False
    seed_allowed_keys_for_connected = {
        (int(face), int(bin_id))
        for face, bins in allowed_bins_by_face.items()
        for bin_id in bins
    }
    if bool(enable_target_visible_expansion):
        footprint_views = list(target_footprint_views or [])
        if not footprint_views:
            target_visible_expansion["reason"] = "no_target_footprint_views"
        else:
            candidate_faces_for_target = {int(row["face_id"]) for row in rows}
            target_samples_by_bin, target_views_by_bin, target_summary = build_target_bin_footprint_stats(
                footprint_views,
                candidate_faces_for_target,
                int(texture_size),
                float(min_alpha),
                max_views=int(target_visible_max_views),
            )
            target_visible_expansion["target_footprint"] = dict(target_summary)
            min_target_positive_fraction = (
                float(target_visible_min_positive_view_fraction)
                if float(target_visible_min_positive_view_fraction) >= 0.0
                else float(adaptive_frontier_min_positive_view_fraction)
            )
            target_visible_expansion["effective_min_positive_view_fraction"] = float(
                min_target_positive_fraction
            )
            allowed_keys = {
                (int(face), int(bin_id))
                for face, bins in allowed_bins_by_face.items()
                for bin_id in bins
            }
            expansion_candidates: list[dict[str, Any]] = []
            for row in rows:
                face_i = int(row["face_id"])
                bin_i = int(row["bin_id"])
                target_key = int(face_i) * bins_per_face + int(bin_i)
                target_pixels = int(target_samples_by_bin.get(target_key, 0))
                target_views = int(target_views_by_bin.get(target_key, 0))
                row["target_visible_pixels"] = int(target_pixels)
                row["target_visible_view_count"] = int(target_views)
                core_ok = bool(
                    (face_i, bin_i) not in allowed_keys
                    and int(row["samples"]) >= int(target_visible_min_policy_samples)
                    and int(row["view_count"]) >= int(min_bin_views)
                    and float(row["relative_gain"]) >= float(min_relative_gain)
                    and float(row["risk_adjusted_relative_gain"])
                    >= float(adaptive_frontier_min_risk_adjusted_gain)
                    and float(row["positive_view_fraction"]) >= float(min_target_positive_fraction)
                    and bool(row["min_view_ok"])
                    and bool(row["variance_ok"])
                    and bool(row["sign_ok"])
                    and target_pixels >= int(target_visible_min_pixels)
                    and target_views >= int(target_visible_min_views)
                )
                row["target_visible_expansion_candidate"] = bool(core_ok)
                if core_ok:
                    expansion_candidates.append(row)
            expansion_candidates.sort(
                key=lambda row: (
                    int(row["target_visible_pixels"]),
                    int(row["target_visible_view_count"]),
                    float(row["risk_adjusted_relative_gain"]),
                    float(row["relative_gain"]),
                    int(row["samples"]),
                ),
                reverse=True,
            )
            if int(target_visible_max_extra_bins) > 0:
                selected_expansion_rows = expansion_candidates[: int(target_visible_max_extra_bins)]
            else:
                selected_expansion_rows = expansion_candidates
            added_samples = 0
            added_target_pixels = 0
            added_target_views_total = 0
            for row in selected_expansion_rows:
                face_i = int(row["face_id"])
                bin_i = int(row["bin_id"])
                if (face_i, bin_i) in allowed_keys:
                    continue
                allowed_bins_by_face.setdefault(str(face_i), []).append(int(bin_i))
                allowed_keys.add((face_i, bin_i))
                allowed_samples += int(row["samples"])
                added_samples += int(row["samples"])
                added_target_pixels += int(row["target_visible_pixels"])
                added_target_views_total += int(row["target_visible_view_count"])
                row["target_visible_expansion_added"] = True
            target_visible_expansion.update(
                {
                    "enabled": True,
                    "reason": "expanded" if selected_expansion_rows else "no_eligible_target_visible_bins",
                    "candidate_bin_count": int(len(expansion_candidates)),
                    "added_bin_count": int(
                        sum(1 for row in rows if bool(row.get("target_visible_expansion_added", False)))
                    ),
                    "added_sample_count": int(added_samples),
                    "added_target_pixels": int(added_target_pixels),
                    "added_target_view_hits": int(added_target_views_total),
                    "final_allowed_bin_count": int(sum(len(v) for v in allowed_bins_by_face.values())),
                    "final_allowed_face_count": int(len(allowed_bins_by_face)),
                    "top_added_bins": [
                        {
                            "face_id": int(row["face_id"]),
                            "bin_id": int(row["bin_id"]),
                            "samples": int(row["samples"]),
                            "view_count": int(row["view_count"]),
                            "positive_view_fraction": float(row["positive_view_fraction"]),
                            "min_view_relative_gain": float(row["min_view_relative_gain"]),
                            "relative_gain": float(row["relative_gain"]),
                            "risk_adjusted_relative_gain": float(row["risk_adjusted_relative_gain"]),
                            "target_pixels": int(row["target_visible_pixels"]),
                            "target_views": int(row["target_visible_view_count"]),
                        }
                        for row in selected_expansion_rows[:128]
                        if bool(row.get("target_visible_expansion_added", False))
                    ],
                    "top_candidate_bins": [
                        {
                            "face_id": int(row["face_id"]),
                            "bin_id": int(row["bin_id"]),
                            "samples": int(row["samples"]),
                            "view_count": int(row["view_count"]),
                            "positive_view_fraction": float(row["positive_view_fraction"]),
                            "min_view_relative_gain": float(row["min_view_relative_gain"]),
                            "relative_gain": float(row["relative_gain"]),
                            "risk_adjusted_relative_gain": float(row["risk_adjusted_relative_gain"]),
                            "target_pixels": int(row["target_visible_pixels"]),
                            "target_views": int(row["target_visible_view_count"]),
                        }
                        for row in expansion_candidates[:128]
                    ],
                }
            )
    target_impact_residual_basis: dict[str, Any] = {
        "enabled": bool(enable_train_only_target_impact_residual_basis),
        "mode": "train_only_target_impact_residual_basis",
        "uses_policy_val_gt": True,
        "uses_target_or_test_gt": False,
        "reason": "not_requested",
        "min_pixels": int(target_impact_min_pixels),
        "min_views": int(target_impact_min_views),
        "min_policy_samples": int(target_impact_min_policy_samples),
        "max_extra_bins": int(target_impact_max_extra_bins),
        "max_views": int(target_impact_max_views),
        "original_allowed_bin_count": int(sum(len(v) for v in allowed_bins_by_face.values())),
        "original_allowed_face_count": int(len(allowed_bins_by_face)),
    }
    if bool(enable_train_only_target_impact_residual_basis):
        footprint_views = list(target_footprint_views or [])
        if not footprint_views:
            target_impact_residual_basis["reason"] = "no_target_footprint_views"
        else:
            candidate_faces_for_target = {int(row["face_id"]) for row in rows}
            if not candidate_faces_for_target:
                target_impact_residual_basis["reason"] = "no_candidate_faces"
            else:
                impact_target_samples_by_bin, impact_target_views_by_bin, impact_target_summary = (
                    build_target_bin_footprint_stats(
                        footprint_views,
                        candidate_faces_for_target,
                        int(texture_size),
                        float(min_alpha),
                        max_views=int(target_impact_max_views),
                    )
                )
                target_impact_residual_basis["target_footprint"] = dict(impact_target_summary)
                row_by_key = {(int(row["face_id"]), int(row["bin_id"])): row for row in rows}
                allowed_keys = {
                    (int(face), int(bin_id))
                    for face, bins in allowed_bins_by_face.items()
                    for bin_id in bins
                }
                impact_candidates: list[dict[str, Any]] = []
                for packed_key, target_pixels in impact_target_samples_by_bin.items():
                    face_i = int(packed_key) // int(bins_per_face)
                    bin_i = int(packed_key) % int(bins_per_face)
                    if (face_i, bin_i) in allowed_keys:
                        continue
                    if face_i not in candidate_faces_for_target:
                        continue
                    target_pixels_i = int(target_pixels)
                    target_views_i = int(impact_target_views_by_bin.get(int(packed_key), 0))
                    if target_pixels_i < int(target_impact_min_pixels):
                        continue
                    if target_views_i < int(target_impact_min_views):
                        continue
                    row = row_by_key.get((face_i, bin_i))
                    policy_samples = int(row.get("samples", 0)) if row else 0
                    if policy_samples < int(target_impact_min_policy_samples):
                        continue
                    relative_gain = float(row.get("relative_gain", 0.0)) if row else 0.0
                    impact_score = (
                        float(target_pixels_i)
                        * math.log1p(float(target_views_i))
                        * (1.0 + max(0.0, float(relative_gain)))
                    )
                    impact_candidates.append(
                        {
                            "face_id": int(face_i),
                            "bin_id": int(bin_i),
                            "target_pixels": int(target_pixels_i),
                            "target_views": int(target_views_i),
                            "policy_samples": int(policy_samples),
                            "has_policy_row": bool(row is not None),
                            "relative_gain": float(relative_gain),
                            "positive_view_fraction": (
                                float(row.get("positive_view_fraction", 0.0)) if row else 0.0
                            ),
                            "min_view_relative_gain": (
                                float(row.get("min_view_relative_gain", 0.0)) if row else 0.0
                            ),
                            "score": float(impact_score),
                        }
                    )
                impact_candidates.sort(
                    key=lambda row: (
                        float(row["score"]),
                        int(row["target_pixels"]),
                        int(row["target_views"]),
                        int(row["policy_samples"]),
                    ),
                    reverse=True,
                )
                selected_impact_rows = (
                    impact_candidates[: int(target_impact_max_extra_bins)]
                    if int(target_impact_max_extra_bins) > 0
                    else impact_candidates
                )
                added_target_pixels = 0
                added_target_views_total = 0
                added_samples = 0
                added_policy_rows = 0
                added_without_policy_rows = 0
                added_bins_by_face: dict[str, list[int]] = {}
                added_policy_bins_by_face: dict[str, list[int]] = {}
                added_no_policy_bins_by_face: dict[str, list[int]] = {}
                actually_added_rows: list[dict[str, Any]] = []
                for row in selected_impact_rows:
                    face_i = int(row["face_id"])
                    bin_i = int(row["bin_id"])
                    if (face_i, bin_i) in allowed_keys:
                        continue
                    allowed_bins_by_face.setdefault(str(face_i), []).append(int(bin_i))
                    allowed_keys.add((face_i, bin_i))
                    added_target_pixels += int(row["target_pixels"])
                    added_target_views_total += int(row["target_views"])
                    if bool(row.get("has_policy_row", False)):
                        added_policy_rows += 1
                        allowed_samples += int(row.get("policy_samples", 0))
                        added_samples += int(row.get("policy_samples", 0))
                        added_policy_bins_by_face.setdefault(str(face_i), []).append(int(bin_i))
                    else:
                        added_without_policy_rows += 1
                        added_no_policy_bins_by_face.setdefault(str(face_i), []).append(int(bin_i))
                    added_bins_by_face.setdefault(str(face_i), []).append(int(bin_i))
                    actually_added_rows.append(row)
                for bins_by_face in (
                    added_bins_by_face,
                    added_policy_bins_by_face,
                    added_no_policy_bins_by_face,
                ):
                    for bins in bins_by_face.values():
                        bins.sort()
                target_impact_residual_basis.update(
                    {
                        "reason": "expanded" if actually_added_rows else "no_eligible_target_impact_bins",
                        "candidate_bin_count": int(len(impact_candidates)),
                        "added_bin_count": int(len(actually_added_rows)),
                        "added_sample_count": int(added_samples),
                        "added_policy_row_bin_count": int(added_policy_rows),
                        "added_without_policy_row_bin_count": int(added_without_policy_rows),
                        "added_target_pixels": int(added_target_pixels),
                        "added_target_view_hits": int(added_target_views_total),
                        "final_allowed_bin_count": int(sum(len(v) for v in allowed_bins_by_face.values())),
                        "final_allowed_face_count": int(len(allowed_bins_by_face)),
                        "added_bins_by_face": added_bins_by_face,
                        "added_policy_bins_by_face": added_policy_bins_by_face,
                        "added_no_policy_bins_by_face": added_no_policy_bins_by_face,
                        "top_added_bins": actually_added_rows[:128],
                        "top_candidate_bins": impact_candidates[:128],
                    }
                )
    target_connected_region_growth: dict[str, Any] = {
        "enabled": bool(enable_target_connected_region_growth),
        "mode": "sparse_materialization_target_visible_connected_region_growth",
        "uses_target_or_test_gt": False,
        "reason": "not_requested",
        "radius": int(target_connected_radius),
        "min_pixels": int(target_connected_min_pixels),
        "min_views": int(target_connected_min_views),
        "min_policy_samples": int(target_connected_min_policy_samples),
        "min_positive_view_fraction": float(target_connected_min_positive_view_fraction),
        "max_negative_relative_gain": float(target_connected_max_negative_relative_gain),
        "max_negative_min_view_gain": float(target_connected_max_negative_min_view_gain),
        "max_extra_bins": int(target_connected_max_extra_bins),
        "max_views": int(target_connected_max_views),
        "seed_allowed_bin_count": int(sum(len(v) for v in allowed_bins_by_face.values())),
        "seed_allowed_face_count": int(len(allowed_bins_by_face)),
    }
    for row in rows:
        row["target_connected_candidate"] = False
        row["target_connected_added"] = False
        row["target_connected_seed_face_id"] = None
        row["target_connected_seed_bin_id"] = None
        row["target_connected_seed_distance"] = 0
        row["target_connected_score"] = 0.0
    if bool(enable_target_connected_region_growth):
        footprint_views = list(target_footprint_views or [])
        if int(target_connected_radius) <= 0:
            target_connected_region_growth["reason"] = "nonpositive_radius"
        elif not footprint_views:
            target_connected_region_growth["reason"] = "no_target_footprint_views"
        elif not allowed_bins_by_face:
            target_connected_region_growth["reason"] = "no_seed_allowed_bins"
        else:
            candidate_faces_for_target = {int(row["face_id"]) for row in rows}
            connected_target_samples_by_bin, connected_target_views_by_bin, connected_target_summary = (
                build_target_bin_footprint_stats(
                    footprint_views,
                    candidate_faces_for_target,
                    int(texture_size),
                    float(min_alpha),
                    max_views=int(target_connected_max_views),
                )
            )
            target_connected_region_growth["target_footprint"] = dict(connected_target_summary)
            row_by_key = {(int(row["face_id"]), int(row["bin_id"])): row for row in rows}
            allowed_keys = {
                (int(face), int(bin_id))
                for face, bins in allowed_bins_by_face.items()
                for bin_id in bins
            }
            seed_bins_by_face: dict[int, list[int]] = {}
            for face, bin_id in seed_allowed_keys_for_connected:
                seed_bins_by_face.setdefault(int(face), []).append(int(bin_id))
            target_connected_region_growth["seed_allowed_bin_count"] = int(
                len(seed_allowed_keys_for_connected)
            )
            target_connected_region_growth["seed_allowed_face_count"] = int(len(seed_bins_by_face))
            connected_candidates: list[dict[str, Any]] = []
            max_negative_relative = max(0.0, float(target_connected_max_negative_relative_gain))
            max_negative_min_view = max(0.0, float(target_connected_max_negative_min_view_gain))
            min_positive_fraction_connected = float(
                np.clip(float(target_connected_min_positive_view_fraction), 0.0, 1.0)
            )
            for row in rows:
                face_i = int(row["face_id"])
                bin_i = int(row["bin_id"])
                if (face_i, bin_i) in allowed_keys:
                    continue
                target_key = int(face_i) * bins_per_face + int(bin_i)
                target_pixels = int(connected_target_samples_by_bin.get(target_key, 0))
                target_views = int(connected_target_views_by_bin.get(target_key, 0))
                row["target_visible_pixels"] = int(target_pixels)
                row["target_visible_view_count"] = int(target_views)
                if target_pixels < int(target_connected_min_pixels):
                    continue
                if target_views < int(target_connected_min_views):
                    continue
                if int(row["samples"]) < int(target_connected_min_policy_samples):
                    continue
                if int(row["view_count"]) < int(min_bin_views):
                    continue
                if float(row["positive_view_fraction"]) < float(min_positive_fraction_connected):
                    continue
                if float(row["relative_gain"]) < -float(max_negative_relative):
                    continue
                if float(row["min_view_relative_gain"]) < -float(max_negative_min_view):
                    continue
                if not bool(row["variance_ok"]) or not bool(row["sign_ok"]):
                    continue
                seed_bins = seed_bins_by_face.get(face_i, [])
                if not seed_bins:
                    continue
                u_i = int(row["u"])
                v_i = int(row["v"])
                best_seed: tuple[float, int, int] | None = None
                for seed_bin in seed_bins:
                    seed_u = int(seed_bin) % int(texture_size)
                    seed_v = int(seed_bin) // int(texture_size)
                    distance = max(abs(int(seed_u) - u_i), abs(int(seed_v) - v_i))
                    if distance <= 0 or distance > int(target_connected_radius):
                        continue
                    seed_row = row_by_key.get((face_i, int(seed_bin)), {})
                    seed_gain = max(0.0, float(seed_row.get("relative_gain", 0.0)))
                    seed_pixels = float(seed_row.get("target_visible_pixels", 0))
                    seed_score = (
                        (1.0 + seed_gain)
                        * (1.0 + np.log1p(max(0.0, seed_pixels)))
                        / float(distance + 1)
                    )
                    if best_seed is None or float(seed_score) > float(best_seed[0]):
                        best_seed = (float(seed_score), int(seed_bin), int(distance))
                if best_seed is None:
                    continue
                support_score = np.log1p(max(0, int(row["samples"]))) * max(
                    0.05,
                    float(row["positive_view_fraction"]),
                )
                gain_score = 1.0 + max(0.0, float(row["relative_gain"]))
                target_score = np.log1p(max(0, target_pixels)) * max(1, target_views)
                score = float(best_seed[0]) * float(support_score) * float(gain_score) * float(target_score)
                row["target_connected_candidate"] = True
                row["target_connected_seed_face_id"] = int(face_i)
                row["target_connected_seed_bin_id"] = int(best_seed[1])
                row["target_connected_seed_distance"] = int(best_seed[2])
                row["target_connected_score"] = float(score)
                connected_candidates.append(row)
            connected_candidates.sort(
                key=lambda row: (
                    float(row.get("target_connected_score", 0.0)),
                    int(row.get("target_visible_pixels", 0)),
                    float(row.get("positive_view_fraction", 0.0)),
                    float(row.get("relative_gain", 0.0)),
                    int(row.get("samples", 0)),
                ),
                reverse=True,
            )
            if int(target_connected_max_extra_bins) > 0:
                selected_connected_rows = connected_candidates[: int(target_connected_max_extra_bins)]
            else:
                selected_connected_rows = connected_candidates
            added_samples = 0
            added_target_pixels = 0
            added_target_views_total = 0
            for row in selected_connected_rows:
                face_i = int(row["face_id"])
                bin_i = int(row["bin_id"])
                if (face_i, bin_i) in allowed_keys:
                    continue
                allowed_bins_by_face.setdefault(str(face_i), []).append(int(bin_i))
                allowed_keys.add((face_i, bin_i))
                allowed_samples += int(row["samples"])
                added_samples += int(row["samples"])
                added_target_pixels += int(row["target_visible_pixels"])
                added_target_views_total += int(row["target_visible_view_count"])
                row["target_connected_added"] = True
            target_connected_region_growth.update(
                {
                    "enabled": True,
                    "reason": "expanded" if selected_connected_rows else "no_eligible_connected_bins",
                    "candidate_bin_count": int(len(connected_candidates)),
                    "added_bin_count": int(
                        sum(1 for row in rows if bool(row.get("target_connected_added", False)))
                    ),
                    "added_sample_count": int(added_samples),
                    "added_target_pixels": int(added_target_pixels),
                    "added_target_view_hits": int(added_target_views_total),
                    "final_allowed_bin_count": int(sum(len(v) for v in allowed_bins_by_face.values())),
                    "final_allowed_face_count": int(len(allowed_bins_by_face)),
                    "top_added_bins": [
                        {
                            "face_id": int(row["face_id"]),
                            "bin_id": int(row["bin_id"]),
                            "u": int(row["u"]),
                            "v": int(row["v"]),
                            "seed_bin_id": int(row.get("target_connected_seed_bin_id", -1)),
                            "seed_distance": int(row.get("target_connected_seed_distance", 0)),
                            "samples": int(row["samples"]),
                            "view_count": int(row["view_count"]),
                            "positive_view_fraction": float(row["positive_view_fraction"]),
                            "min_view_relative_gain": float(row["min_view_relative_gain"]),
                            "relative_gain": float(row["relative_gain"]),
                            "target_pixels": int(row["target_visible_pixels"]),
                            "target_views": int(row["target_visible_view_count"]),
                            "score": float(row.get("target_connected_score", 0.0)),
                        }
                        for row in selected_connected_rows[:128]
                        if bool(row.get("target_connected_added", False))
                    ],
                    "top_candidate_bins": [
                        {
                            "face_id": int(row["face_id"]),
                            "bin_id": int(row["bin_id"]),
                            "u": int(row["u"]),
                            "v": int(row["v"]),
                            "seed_bin_id": int(row.get("target_connected_seed_bin_id", -1)),
                            "seed_distance": int(row.get("target_connected_seed_distance", 0)),
                            "samples": int(row["samples"]),
                            "view_count": int(row["view_count"]),
                            "positive_view_fraction": float(row["positive_view_fraction"]),
                            "min_view_relative_gain": float(row["min_view_relative_gain"]),
                            "relative_gain": float(row["relative_gain"]),
                            "target_pixels": int(row["target_visible_pixels"]),
                            "target_views": int(row["target_visible_view_count"]),
                            "score": float(row.get("target_connected_score", 0.0)),
                        }
                        for row in connected_candidates[:128]
                    ],
                }
            )
    for bins in allowed_bins_by_face.values():
        bins.sort()
    rows_by_gain = sorted(rows, key=lambda row: float(row["relative_gain"]))
    rows_by_samples = sorted(rows, key=lambda row: int(row["samples"]), reverse=True)
    allowed_bin_count = int(sum(len(v) for v in allowed_bins_by_face.values()))
    target_visible_expansion.setdefault("final_allowed_bin_count", int(allowed_bin_count))
    target_visible_expansion.setdefault("final_allowed_face_count", int(len(allowed_bins_by_face)))
    target_visible_expansion.setdefault("candidate_bin_count", 0)
    target_visible_expansion.setdefault("added_bin_count", 0)
    target_visible_expansion.setdefault("added_sample_count", 0)
    target_visible_expansion.setdefault("added_target_pixels", 0)
    target_impact_residual_basis.setdefault("final_allowed_bin_count", int(allowed_bin_count))
    target_impact_residual_basis.setdefault("final_allowed_face_count", int(len(allowed_bins_by_face)))
    target_impact_residual_basis.setdefault("candidate_bin_count", 0)
    target_impact_residual_basis.setdefault("added_bin_count", 0)
    target_impact_residual_basis.setdefault("added_sample_count", 0)
    target_impact_residual_basis.setdefault("added_policy_row_bin_count", 0)
    target_impact_residual_basis.setdefault("added_without_policy_row_bin_count", 0)
    target_impact_residual_basis.setdefault("added_target_pixels", 0)
    target_impact_residual_basis.setdefault("added_bins_by_face", {})
    target_impact_residual_basis.setdefault("added_policy_bins_by_face", {})
    target_impact_residual_basis.setdefault("added_no_policy_bins_by_face", {})
    target_connected_region_growth.setdefault("final_allowed_bin_count", int(allowed_bin_count))
    target_connected_region_growth.setdefault("final_allowed_face_count", int(len(allowed_bins_by_face)))
    target_connected_region_growth.setdefault("candidate_bin_count", 0)
    target_connected_region_growth.setdefault("added_bin_count", 0)
    target_connected_region_growth.setdefault("added_sample_count", 0)
    target_connected_region_growth.setdefault("added_target_pixels", 0)
    target_visible_expansion["target_impact_residual_basis"] = dict(target_impact_residual_basis)
    target_visible_expansion["connected_region_growth"] = dict(target_connected_region_growth)
    failure_counts = {
        "sample": int(sum(1 for row in rows if not bool(row.get("sample_ok", False)))),
        "view": int(sum(1 for row in rows if not bool(row.get("view_ok", False)))),
        "relative": int(sum(1 for row in rows if not bool(row.get("relative_ok", False)))),
        "min_view": int(sum(1 for row in rows if not bool(row.get("min_view_ok", False)))),
        "positive_fraction": int(sum(1 for row in rows if not bool(row.get("positive_fraction_ok", False)))),
        "variance": int(sum(1 for row in rows if not bool(row.get("variance_ok", False)))),
        "sign": int(sum(1 for row in rows if not bool(row.get("sign_ok", False)))),
    }
    return {
        "enabled": True,
        "mode": "policy_val_bin_uncertainty_guard",
        "alpha": float(alpha),
        "policy_val_views_used": int(view_count),
        "samples": int(total_active_samples),
        "texture_size": int(texture_size),
        "min_bin_samples": int(min_bin_samples),
        "min_bin_views": int(min_bin_views),
        "min_relative_gain": float(min_relative_gain),
        "min_view_relative_gain": float(min_view_relative_gain),
        "min_positive_view_fraction": float(min_positive_view_fraction),
        "max_mean_variance": float(max_mean_variance),
        "min_mean_sign_consistency": float(min_mean_sign_consistency),
        "adaptive_frontier": dict(adaptive_frontier),
        "target_visible_expansion": dict(target_visible_expansion),
        "target_impact_residual_basis": dict(target_impact_residual_basis),
        "target_connected_region_growth": dict(target_connected_region_growth),
        "strict_failure_counts": dict(failure_counts),
        "candidate_bin_count": int(len(rows)),
        "allowed_bin_count": int(allowed_bin_count),
        "rejected_bin_count": int(len(rows) - allowed_bin_count),
        "allowed_face_count": int(len(allowed_bins_by_face)),
        "allowed_sample_fraction": float(allowed_samples / max(1, total_active_samples)),
        "allowed_bins_by_face": allowed_bins_by_face,
        "worst_bins": rows_by_gain[:32],
        "best_gain_bins": list(reversed(rows_by_gain[-32:])),
        "best_sampled_bins": rows_by_samples[:32],
    }


def select_sparse_materialization_seed(
    policy_payload: dict[str, Any],
    *,
    min_relative_gain: float,
) -> dict[str, Any]:
    if not isinstance(policy_payload, dict) or not bool(policy_payload.get("enabled", False)):
        return {}
    best = policy_payload.get("best", {})
    if isinstance(best, dict):
        alpha = float(best.get("alpha", 0.0))
        if alpha > 0.0 and float(best.get("relative_gain", -1.0)) >= float(min_relative_gain):
            return dict(best)
    rows = [row for row in policy_payload.get("rows", []) if isinstance(row, dict)]
    candidates = []
    for row in rows:
        alpha = float(row.get("alpha", 0.0))
        if alpha <= 0.0:
            continue
        if float(row.get("relative_gain", -1.0)) < float(min_relative_gain):
            continue
        candidates.append(row)
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda row: (
            float(row.get("relative_gain", -1.0)),
            float(row.get("ssim_gain", 0.0)),
            float(row.get("image_l1_gain", 0.0)),
            -float(row.get("alpha", 0.0)),
        ),
    )


def intersect_bin_guard_profiles(
    sparse_profile: dict[str, Any],
    bin_guard_profile: dict[str, Any],
    *,
    empty_intersection_policy: str = "reject",
) -> dict[str, Any]:
    sparse_bins = _allowed_bins_from_uncertainty_guard(sparse_profile)
    guard_bins = _allowed_bins_from_uncertainty_guard(bin_guard_profile)
    if sparse_bins is None:
        return dict(bin_guard_profile)
    if guard_bins is None:
        return dict(sparse_profile)
    allowed_bins_by_face: dict[str, list[int]] = {}
    for face, sparse_allowed in sparse_bins.items():
        guard_allowed = guard_bins.get(int(face))
        if not guard_allowed:
            continue
        intersection = sorted(int(bin_id) for bin_id in sparse_allowed.intersection(guard_allowed))
        if intersection:
            allowed_bins_by_face[str(int(face))] = intersection
    allowed_count = int(sum(len(bins) for bins in allowed_bins_by_face.values()))
    merged = dict(bin_guard_profile)
    merged["enabled"] = allowed_count > 0
    merged["mode"] = "policy_val_sparse_residual_materialization_and_bin_uncertainty_guard"
    merged["allowed_bins_by_face"] = allowed_bins_by_face
    merged["allowed_bin_count"] = int(allowed_count)
    merged["allowed_face_count"] = int(len(allowed_bins_by_face))
    merged["sparse_materialization_intersection"] = {
        "enabled": bool(sparse_profile.get("enabled", False)),
        "sparse_allowed_bin_count": int(sparse_profile.get("allowed_bin_count", 0) or 0),
        "bin_guard_allowed_bin_count": int(bin_guard_profile.get("allowed_bin_count", 0) or 0),
        "intersection_allowed_bin_count": int(allowed_count),
        "empty_intersection_policy": str(empty_intersection_policy),
    }
    if allowed_count <= 0:
        sparse_allowed_count = int(sparse_profile.get("allowed_bin_count", 0) or 0)
        sparse_post_accepted = bool(sparse_profile.get("post_materialization_accepted", False))
        if (
            str(empty_intersection_policy) == "sparse_if_post_accepted"
            and bool(sparse_profile.get("enabled", False))
            and sparse_allowed_count > 0
            and sparse_post_accepted
        ):
            bridged = dict(sparse_profile)
            bridged["enabled"] = True
            bridged["mode"] = "policy_val_sparse_residual_materialization_and_bin_uncertainty_guard"
            bridged["decision"] = "bridge_sparse_after_empty_bin_guard_intersection"
            bridged["reason"] = "sparse_materialization_post_gate_accepted_bin_guard_empty"
            bridged["bin_uncertainty_guard_profile"] = dict(bin_guard_profile)
            bridged["sparse_materialization_intersection"] = dict(
                merged["sparse_materialization_intersection"]
            )
            bridged["sparse_materialization_intersection"].update(
                {
                    "bridge_activated": True,
                    "bridge_allowed_bin_count": int(sparse_allowed_count),
                    "bridge_reason": "bin_guard_empty_but_sparse_post_gate_accepted",
                }
            )
            return bridged
        merged["decision"] = "disabled_no_sparse_bin_guard_intersection"
        merged["reason"] = "sparse_materialization_and_bin_uncertainty_guard_intersection_empty"
    return merged


def policy_val_sparse_materialization_profile(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    policy_payload: dict[str, Any],
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_abs_delta_rgb: float,
    max_samples_per_view: int,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    local_alpha_profile: dict[str, Any] | None,
    face_gain_guard_profile: dict[str, Any] | None,
    seed_min_relative_gain: float,
    min_bin_samples: int,
    min_bin_views: int,
    min_relative_gain: float,
    min_view_relative_gain: float,
    min_positive_view_fraction: float,
    max_mean_variance: float,
    min_mean_sign_consistency: float,
    adaptive_frontier_on_empty: bool,
    adaptive_frontier_min_positive_view_fraction: float,
    adaptive_frontier_min_risk_adjusted_gain: float,
    adaptive_frontier_min_sample_quantile: float,
    target_footprint_views: list[Path] | None = None,
    enable_target_visible_expansion: bool = False,
    target_visible_min_pixels: int = 1,
    target_visible_min_views: int = 1,
    target_visible_min_policy_samples: int = 1,
    target_visible_min_positive_view_fraction: float = -1.0,
    target_visible_max_extra_bins: int = 0,
    target_visible_max_views: int = 0,
    enable_train_only_target_impact_residual_basis: bool = False,
    target_impact_min_pixels: int = 1,
    target_impact_min_views: int = 1,
    target_impact_min_policy_samples: int = 0,
    target_impact_max_extra_bins: int = 0,
    target_impact_max_views: int = 0,
    enable_target_connected_region_growth: bool = False,
    target_connected_radius: int = 1,
    target_connected_min_pixels: int = 1,
    target_connected_min_views: int = 1,
    target_connected_min_policy_samples: int = 1,
    target_connected_min_positive_view_fraction: float = 0.5,
    target_connected_max_negative_relative_gain: float = 0.02,
    target_connected_max_negative_min_view_gain: float = 0.05,
    target_connected_max_extra_bins: int = 0,
    target_connected_max_views: int = 0,
) -> dict[str, Any]:
    seed = select_sparse_materialization_seed(
        policy_payload,
        min_relative_gain=float(seed_min_relative_gain),
    )
    if not seed:
        return {
            "enabled": False,
            "mode": "policy_val_sparse_residual_materialization",
            "reason": "no_positive_nonzero_policy_val_seed",
            "seed_min_relative_gain": float(seed_min_relative_gain),
        }
    profile = build_policy_val_bin_uncertainty_guard_profile(
        val_views,
        atlas,
        residual_rgb_key=str(residual_rgb_key),
        residual_l1_key=str(residual_l1_key),
        alpha=float(seed.get("alpha", 0.0)),
        min_l1=float(min_l1),
        min_alpha=float(min_alpha),
        max_abs_delta_rgb=float(max_abs_delta_rgb),
        max_samples_per_view=int(max_samples_per_view),
        min_atlas_bin_count=int(min_atlas_bin_count),
        min_atlas_face_samples=int(min_atlas_face_samples),
        max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
        min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
        atlas_confidence_mode=str(atlas_confidence_mode),
        atlas_confidence_count_scale=float(atlas_confidence_count_scale),
        atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
        atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
        atlas_confidence_sign_power=float(atlas_confidence_sign_power),
        atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
        min_atlas_confidence=float(min_atlas_confidence),
        local_alpha_profile=local_alpha_profile,
        face_gain_guard_profile=face_gain_guard_profile,
        min_bin_samples=int(min_bin_samples),
        min_bin_views=int(min_bin_views),
        min_relative_gain=float(min_relative_gain),
        min_view_relative_gain=float(min_view_relative_gain),
        min_positive_view_fraction=float(min_positive_view_fraction),
        max_mean_variance=float(max_mean_variance),
        min_mean_sign_consistency=float(min_mean_sign_consistency),
        adaptive_frontier_on_empty=bool(adaptive_frontier_on_empty),
        adaptive_frontier_min_positive_view_fraction=float(adaptive_frontier_min_positive_view_fraction),
        adaptive_frontier_min_risk_adjusted_gain=float(adaptive_frontier_min_risk_adjusted_gain),
        adaptive_frontier_min_sample_quantile=float(adaptive_frontier_min_sample_quantile),
        target_footprint_views=target_footprint_views,
        enable_target_visible_expansion=bool(enable_target_visible_expansion),
        target_visible_min_pixels=int(target_visible_min_pixels),
        target_visible_min_views=int(target_visible_min_views),
        target_visible_min_policy_samples=int(target_visible_min_policy_samples),
        target_visible_min_positive_view_fraction=float(target_visible_min_positive_view_fraction),
        target_visible_max_extra_bins=int(target_visible_max_extra_bins),
        target_visible_max_views=int(target_visible_max_views),
        enable_train_only_target_impact_residual_basis=bool(
            enable_train_only_target_impact_residual_basis
        ),
        target_impact_min_pixels=int(target_impact_min_pixels),
        target_impact_min_views=int(target_impact_min_views),
        target_impact_min_policy_samples=int(target_impact_min_policy_samples),
        target_impact_max_extra_bins=int(target_impact_max_extra_bins),
        target_impact_max_views=int(target_impact_max_views),
        enable_target_connected_region_growth=bool(enable_target_connected_region_growth),
        target_connected_radius=int(target_connected_radius),
        target_connected_min_pixels=int(target_connected_min_pixels),
        target_connected_min_views=int(target_connected_min_views),
        target_connected_min_policy_samples=int(target_connected_min_policy_samples),
        target_connected_min_positive_view_fraction=float(target_connected_min_positive_view_fraction),
        target_connected_max_negative_relative_gain=float(target_connected_max_negative_relative_gain),
        target_connected_max_negative_min_view_gain=float(target_connected_max_negative_min_view_gain),
        target_connected_max_extra_bins=int(target_connected_max_extra_bins),
        target_connected_max_views=int(target_connected_max_views),
    )
    profile["mode"] = "policy_val_sparse_residual_materialization"
    profile["compatible_predict_mode"] = "policy_val_bin_uncertainty_guard"
    profile["seed_policy_val_row"] = dict(seed)
    profile["seed_alpha"] = float(seed.get("alpha", 0.0))
    profile["seed_min_relative_gain"] = float(seed_min_relative_gain)
    return profile


def apply_target_impact_carrier_fill(
    atlas: dict[int, FaceAtlas],
    sparse_profile: dict[str, Any] | None,
    *,
    mode: str,
    blend: float,
    min_face_samples: int,
    min_carrier_norm: float,
    max_abs_delta_rgb: float,
    synthetic_count: int,
) -> dict[str, Any]:
    """Fill target-impact-expanded bins with a train-only face residual carrier."""
    mode_s = str(mode)
    summary: dict[str, Any] = {
        "enabled": mode_s != "off",
        "mode": "target_impact_train_only_carrier_fill",
        "fill_mode": mode_s,
        "uses_policy_val_gt": False,
        "uses_train_fit_gt": True,
        "uses_target_or_test_gt": False,
        "blend": float(blend),
        "min_face_samples": int(min_face_samples),
        "min_carrier_norm": float(min_carrier_norm),
        "max_abs_delta_rgb": float(max_abs_delta_rgb),
        "synthetic_count": int(synthetic_count),
        "eligible_bin_count": 0,
        "filled_bin_count": 0,
        "skipped_missing_face_count": 0,
        "skipped_low_face_support_count": 0,
        "skipped_low_carrier_norm_count": 0,
        "mean_carrier_norm": 0.0,
        "max_carrier_norm": 0.0,
        "filled_bins_by_face": {},
        "top_filled_bins": [],
    }
    if mode_s == "off":
        summary["reason"] = "not_requested"
        return summary
    if mode_s not in {"no_policy_rows", "all_added"}:
        raise ValueError(f"unsupported target-impact carrier fill mode: {mode_s}")
    profile = dict(sparse_profile or {})
    impact = dict(profile.get("target_impact_residual_basis") or {})
    if not impact or not bool(impact.get("enabled", False)):
        summary["reason"] = "target_impact_residual_basis_disabled"
        return summary
    key_name = (
        "added_no_policy_bins_by_face"
        if mode_s == "no_policy_rows"
        else "added_bins_by_face"
    )
    raw_bins_by_face = impact.get(key_name, {}) or {}
    if not isinstance(raw_bins_by_face, dict) or not raw_bins_by_face:
        summary["reason"] = f"no_{key_name}"
        return summary
    blend_f = float(np.clip(float(blend), 0.0, 1.0))
    max_abs = max(0.0, float(max_abs_delta_rgb))
    min_norm = max(0.0, float(min_carrier_norm))
    min_samples = max(0, int(min_face_samples))
    synthetic_count_i = max(0, int(synthetic_count))
    filled_rows: list[dict[str, Any]] = []
    filled_bins_by_face: dict[str, list[int]] = {}
    carrier_norms: list[float] = []
    eligible = 0
    skipped_missing_face = 0
    skipped_low_support = 0
    skipped_low_norm = 0
    for face_key, raw_bins in raw_bins_by_face.items():
        try:
            face = int(face_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(raw_bins, (list, tuple, set)):
            continue
        unique_bins = sorted({int(bin_id) for bin_id in raw_bins})
        eligible += int(len(unique_bins))
        face_atlas = atlas.get(int(face))
        if face_atlas is None:
            skipped_missing_face += int(len(unique_bins))
            continue
        if int(face_atlas.samples) < int(min_samples):
            skipped_low_support += int(len(unique_bins))
            continue
        carrier = clip_delta_rgb(
            np.asarray(face_atlas.mean_rgb, dtype=np.float32).reshape(1, 3),
            max_abs,
        ).reshape(3)
        carrier_norm = float(np.linalg.norm(carrier.astype(np.float32)))
        if carrier_norm < min_norm:
            skipped_low_norm += int(len(unique_bins))
            continue
        texture_size = int(face_atlas.texture.shape[0])
        for bin_id in unique_bins:
            if int(bin_id) < 0 or int(bin_id) >= texture_size * texture_size:
                continue
            v = int(bin_id) // texture_size
            u = int(bin_id) % texture_size
            old_value = np.asarray(face_atlas.texture[v, u], dtype=np.float32)
            new_value = ((1.0 - blend_f) * old_value + blend_f * carrier).astype(np.float32)
            new_value = clip_delta_rgb(new_value.reshape(1, 3), max_abs).reshape(3)
            face_atlas.texture[v, u] = new_value
            if synthetic_count_i > 0:
                face_atlas.counts[v, u] = max(int(face_atlas.counts[v, u]), synthetic_count_i)
            face_atlas.sign_consistency[v, u] = np.maximum(
                face_atlas.sign_consistency[v, u],
                (np.abs(np.sign(new_value)) > 0).astype(np.float32),
            )
            filled_bins_by_face.setdefault(str(face), []).append(int(bin_id))
            carrier_norms.append(float(carrier_norm))
            filled_rows.append(
                {
                    "face_id": int(face),
                    "bin_id": int(bin_id),
                    "u": int(u),
                    "v": int(v),
                    "face_samples": int(face_atlas.samples),
                    "carrier_norm": float(carrier_norm),
                    "old_norm": float(np.linalg.norm(old_value.astype(np.float32))),
                    "new_norm": float(np.linalg.norm(new_value.astype(np.float32))),
                }
            )
    summary.update(
        {
            "reason": "filled" if filled_rows else "no_bins_filled",
            "eligible_bin_count": int(eligible),
            "filled_bin_count": int(len(filled_rows)),
            "skipped_missing_face_count": int(skipped_missing_face),
            "skipped_low_face_support_count": int(skipped_low_support),
            "skipped_low_carrier_norm_count": int(skipped_low_norm),
            "mean_carrier_norm": float(np.mean(carrier_norms)) if carrier_norms else 0.0,
            "max_carrier_norm": float(np.max(carrier_norms)) if carrier_norms else 0.0,
            "filled_bins_by_face": filled_bins_by_face,
            "top_filled_bins": sorted(
                filled_rows,
                key=lambda row: (float(row["carrier_norm"]), int(row["face_samples"])),
                reverse=True,
            )[:128],
        }
    )
    return summary


def apply_target_impact_multisample_residual_fill(
    atlas: dict[int, FaceAtlas],
    train_fit_views: list[Path],
    sparse_profile: dict[str, Any] | None,
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_abs_delta_rgb: float,
    max_samples_per_view: int,
    mode: str,
    radius: int,
    min_samples: int,
    max_samples_per_bin: int,
    max_views: int,
    blend: float,
    kernel_sigma: float,
    min_norm: float,
    synthetic_count: int,
) -> dict[str, Any]:
    """Materialize target-impact bins from nearby train-fit residual samples."""
    mode_s = str(mode)
    summary: dict[str, Any] = {
        "enabled": mode_s != "off",
        "mode": "target_impact_train_only_multisample_residual_fill",
        "fill_mode": mode_s,
        "uses_policy_val_gt": False,
        "uses_train_fit_gt": True,
        "uses_target_or_test_gt": False,
        "radius": int(radius),
        "min_samples": int(min_samples),
        "max_samples_per_bin": int(max_samples_per_bin),
        "max_views": int(max_views),
        "blend": float(blend),
        "kernel_sigma": float(kernel_sigma),
        "min_norm": float(min_norm),
        "max_abs_delta_rgb": float(max_abs_delta_rgb),
        "synthetic_count": int(synthetic_count),
        "eligible_bin_count": 0,
        "filled_bin_count": 0,
        "train_fit_view_count": 0,
        "train_fit_views_used": 0,
        "sample_event_count": 0,
        "skipped_missing_face_count": 0,
        "skipped_low_sample_count": 0,
        "skipped_low_norm_count": 0,
        "filled_bins_by_face": {},
        "top_filled_bins": [],
    }
    if mode_s == "off":
        summary["reason"] = "not_requested"
        return summary
    if mode_s not in {"no_policy_rows", "all_added"}:
        raise ValueError(f"unsupported target-impact multisample fill mode: {mode_s}")
    if not train_fit_views:
        summary["reason"] = "no_train_fit_views"
        return summary
    profile = dict(sparse_profile or {})
    impact = dict(profile.get("target_impact_residual_basis") or {})
    if not impact or not bool(impact.get("enabled", False)):
        summary["reason"] = "target_impact_residual_basis_disabled"
        return summary
    key_name = (
        "added_no_policy_bins_by_face"
        if mode_s == "no_policy_rows"
        else "added_bins_by_face"
    )
    raw_bins_by_face = impact.get(key_name, {}) or {}
    if not isinstance(raw_bins_by_face, dict) or not raw_bins_by_face:
        summary["reason"] = f"no_{key_name}"
        return summary

    target_bins_by_face: dict[int, list[int]] = {}
    skipped_missing_face = 0
    for face_key, raw_bins in raw_bins_by_face.items():
        try:
            face = int(face_key)
        except (TypeError, ValueError):
            continue
        face_atlas = atlas.get(face)
        if face_atlas is None:
            try:
                skipped_missing_face += len(raw_bins)
            except TypeError:
                skipped_missing_face += 1
            continue
        if not isinstance(raw_bins, (list, tuple, set)):
            continue
        texture_size = int(face_atlas.texture.shape[0])
        clean_bins: list[int] = []
        for raw_bin in raw_bins:
            try:
                bin_id = int(raw_bin)
            except (TypeError, ValueError):
                continue
            if 0 <= bin_id < texture_size * texture_size:
                clean_bins.append(bin_id)
        if clean_bins:
            target_bins_by_face[face] = sorted(set(clean_bins))
    eligible_keys = {
        (int(face), int(bin_id))
        for face, bins in target_bins_by_face.items()
        for bin_id in bins
    }
    summary["eligible_bin_count"] = int(len(eligible_keys))
    summary["skipped_missing_face_count"] = int(skipped_missing_face)
    if not eligible_keys:
        summary["reason"] = "no_valid_target_impact_bins"
        return summary

    radius_i = max(0, int(radius))
    sigma = max(1.0e-6, float(kernel_sigma))
    blend_f = float(np.clip(float(blend), 0.0, 1.0))
    min_samples_i = max(1, int(min_samples))
    max_samples_i = max(0, int(max_samples_per_bin))
    synthetic_count_i = max(0, int(synthetic_count))
    max_abs = max(0.0, float(max_abs_delta_rgb))
    min_norm_f = max(0.0, float(min_norm))
    first_atlas = next(iter(atlas.values()))
    texture_size = int(first_atlas.texture.shape[0])
    neighbor_maps: dict[int, dict[int, list[tuple[int, float]]]] = {}
    for face, target_bins in target_bins_by_face.items():
        face_atlas = atlas.get(face)
        if face_atlas is None:
            continue
        size = int(face_atlas.texture.shape[0])
        face_neighbors: dict[int, list[tuple[int, float]]] = {}
        for target_bin in target_bins:
            target_u = int(target_bin) % size
            target_v = int(target_bin) // size
            for dv in range(-radius_i, radius_i + 1):
                source_v = target_v + dv
                if source_v < 0 or source_v >= size:
                    continue
                for du in range(-radius_i, radius_i + 1):
                    source_u = target_u + du
                    if source_u < 0 or source_u >= size:
                        continue
                    dist2 = float(du * du + dv * dv)
                    if dist2 > float(radius_i * radius_i):
                        continue
                    weight = float(math.exp(-0.5 * dist2 / (sigma * sigma)))
                    source_bin = int(source_v * size + source_u)
                    face_neighbors.setdefault(source_bin, []).append((int(target_bin), weight))
        neighbor_maps[int(face)] = face_neighbors

    sum_by_key = {key: np.zeros((3,), dtype=np.float64) for key in eligible_keys}
    sq_by_key = {key: np.zeros((3,), dtype=np.float64) for key in eligible_keys}
    sign_by_key = {key: np.zeros((3,), dtype=np.float64) for key in eligible_keys}
    weight_by_key = {key: 0.0 for key in eligible_keys}
    count_by_key = {key: 0 for key in eligible_keys}
    rng = np.random.default_rng(17)
    view_paths = list(train_fit_views)
    if int(max_views) > 0:
        view_paths = view_paths[: int(max_views)]
    summary["train_fit_view_count"] = int(len(train_fit_views))
    summary["train_fit_views_used"] = int(len(view_paths))
    candidate_faces = set(int(face) for face in target_bins_by_face)
    sample_events = 0
    for path in tqdm(view_paths, desc="target-impact multisample fill"):
        z = np.load(path)
        if residual_rgb_key not in z or "barycentric" not in z or "face_id" not in z:
            continue
        mask = _valid_sample_mask(
            z,
            candidate_faces,
            residual_l1_key=residual_l1_key,
            min_l1=float(min_l1),
            min_alpha=float(min_alpha),
        )
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if int(max_samples_per_view) > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            local_mask = np.zeros_like(mask, dtype=bool)
            local_mask[ys, xs] = True
            mask = local_mask
        face_ids = np.asarray(z["face_id"], dtype=np.int64)[mask]
        residual = np.asarray(z[residual_rgb_key], dtype=np.float32)
        residual_samples = np.stack(
            [residual[0][mask], residual[1][mask], residual[2][mask]],
            axis=1,
        ).astype(np.float32)
        ubin, vbin = _uv_bins(np.asarray(z["barycentric"], dtype=np.float32), mask, texture_size)
        source_bins = (vbin.astype(np.int64) * int(texture_size) + ubin.astype(np.int64)).astype(np.int64)
        for face_raw, source_bin_raw, residual_rgb in zip(face_ids, source_bins, residual_samples, strict=False):
            face = int(face_raw)
            face_neighbors = neighbor_maps.get(face)
            if not face_neighbors:
                continue
            target_rows = face_neighbors.get(int(source_bin_raw))
            if not target_rows:
                continue
            for target_bin, weight in target_rows:
                key = (face, int(target_bin))
                if max_samples_i > 0 and int(count_by_key[key]) >= max_samples_i:
                    continue
                w = float(weight)
                rgb64 = residual_rgb.astype(np.float64)
                sum_by_key[key] += rgb64 * w
                sq_by_key[key] += np.square(rgb64) * w
                sign_by_key[key] += np.sign(rgb64) * w
                weight_by_key[key] += w
                count_by_key[key] += 1
                sample_events += 1

    filled_rows: list[dict[str, Any]] = []
    filled_bins_by_face: dict[str, list[int]] = {}
    skipped_low_sample = 0
    skipped_low_norm = 0
    for key in sorted(eligible_keys):
        face, bin_id = key
        count = int(count_by_key[key])
        if count < min_samples_i or float(weight_by_key[key]) <= 0.0:
            skipped_low_sample += 1
            continue
        estimate = (sum_by_key[key] / max(float(weight_by_key[key]), 1.0e-12)).astype(np.float32)
        estimate = clip_delta_rgb(estimate.reshape(1, 3), max_abs).reshape(3)
        estimate_norm = float(np.linalg.norm(estimate.astype(np.float32)))
        if estimate_norm < min_norm_f:
            skipped_low_norm += 1
            continue
        face_atlas = atlas.get(face)
        if face_atlas is None:
            continue
        size = int(face_atlas.texture.shape[0])
        v = int(bin_id) // size
        u = int(bin_id) % size
        old_value = np.asarray(face_atlas.texture[v, u], dtype=np.float32)
        new_value = ((1.0 - blend_f) * old_value + blend_f * estimate).astype(np.float32)
        new_value = clip_delta_rgb(new_value.reshape(1, 3), max_abs).reshape(3)
        face_atlas.texture[v, u] = new_value
        effective_count = max(synthetic_count_i, count)
        if effective_count > 0:
            face_atlas.counts[v, u] = max(int(face_atlas.counts[v, u]), int(effective_count))
        mean_sq = sq_by_key[key] / max(float(weight_by_key[key]), 1.0e-12)
        face_atlas.variance[v, u] = np.maximum(mean_sq - np.square(estimate.astype(np.float64)), 0.0).astype(np.float32)
        face_atlas.sign_consistency[v, u] = np.maximum(
            face_atlas.sign_consistency[v, u],
            np.clip(np.abs(sign_by_key[key]) / max(float(weight_by_key[key]), 1.0e-12), 0.0, 1.0).astype(np.float32),
        )
        filled_bins_by_face.setdefault(str(face), []).append(int(bin_id))
        filled_rows.append(
            {
                "face_id": int(face),
                "bin_id": int(bin_id),
                "u": int(u),
                "v": int(v),
                "sample_count": int(count),
                "weight_sum": float(weight_by_key[key]),
                "estimate_norm": float(estimate_norm),
                "old_norm": float(np.linalg.norm(old_value.astype(np.float32))),
                "new_norm": float(np.linalg.norm(new_value.astype(np.float32))),
            }
        )
    for bins in filled_bins_by_face.values():
        bins.sort()
    summary.update(
        {
            "reason": "filled" if filled_rows else "no_bins_filled",
            "filled_bin_count": int(len(filled_rows)),
            "sample_event_count": int(sample_events),
            "skipped_low_sample_count": int(skipped_low_sample),
            "skipped_low_norm_count": int(skipped_low_norm),
            "filled_bins_by_face": filled_bins_by_face,
            "top_filled_bins": sorted(
                filled_rows,
                key=lambda row: (int(row["sample_count"]), float(row["estimate_norm"])),
                reverse=True,
            )[:128],
        }
    )
    return summary


def apply_target_impact_affine_residual_fill(
    atlas: dict[int, FaceAtlas],
    train_fit_views: list[Path],
    sparse_profile: dict[str, Any] | None,
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_abs_delta_rgb: float,
    max_samples_per_view: int,
    mode: str,
    feature_mode: str,
    min_samples: int,
    max_samples_per_face: int,
    max_views: int,
    blend: float,
    ridge: float,
    max_condition: float,
    min_norm: float,
    synthetic_count: int,
) -> dict[str, Any]:
    """Fit face-local train-only residual fields and materialize target-impact bins."""
    mode_s = str(mode)
    feature_mode_s = str(feature_mode)
    summary: dict[str, Any] = {
        "enabled": mode_s != "off",
        "mode": "target_impact_train_only_affine_residual_fill",
        "fill_mode": mode_s,
        "feature_mode": feature_mode_s,
        "uses_policy_val_gt": False,
        "uses_train_fit_gt": True,
        "uses_target_or_test_gt": False,
        "min_samples": int(min_samples),
        "max_samples_per_face": int(max_samples_per_face),
        "max_views": int(max_views),
        "blend": float(blend),
        "ridge": float(ridge),
        "max_condition": float(max_condition),
        "min_norm": float(min_norm),
        "max_abs_delta_rgb": float(max_abs_delta_rgb),
        "synthetic_count": int(synthetic_count),
        "eligible_bin_count": 0,
        "filled_bin_count": 0,
        "train_fit_view_count": 0,
        "train_fit_views_used": 0,
        "sample_event_count": 0,
        "fit_face_count": 0,
        "skipped_missing_face_count": 0,
        "skipped_low_sample_face_count": 0,
        "skipped_bad_condition_face_count": 0,
        "skipped_low_norm_count": 0,
        "filled_bins_by_face": {},
        "top_filled_bins": [],
    }
    if mode_s == "off":
        summary["reason"] = "not_requested"
        return summary
    if mode_s not in {"no_policy_rows", "all_added"}:
        raise ValueError(f"unsupported target-impact affine fill mode: {mode_s}")
    if feature_mode_s not in {"face_uv_normal_camera_ridge", "face_uv_patch_mixture_ridge"}:
        raise ValueError(f"unsupported target-impact affine feature mode: {feature_mode_s}")
    if not train_fit_views:
        summary["reason"] = "no_train_fit_views"
        return summary
    profile = dict(sparse_profile or {})
    impact = dict(profile.get("target_impact_residual_basis") or {})
    if not impact or not bool(impact.get("enabled", False)):
        summary["reason"] = "target_impact_residual_basis_disabled"
        return summary
    key_name = "added_no_policy_bins_by_face" if mode_s == "no_policy_rows" else "added_bins_by_face"
    raw_bins_by_face = impact.get(key_name, {}) or {}
    if not isinstance(raw_bins_by_face, dict) or not raw_bins_by_face:
        summary["reason"] = f"no_{key_name}"
        return summary

    target_bins_by_face: dict[int, list[int]] = {}
    skipped_missing_face = 0
    for face_key, raw_bins in raw_bins_by_face.items():
        try:
            face = int(face_key)
        except (TypeError, ValueError):
            continue
        face_atlas = atlas.get(face)
        if face_atlas is None:
            try:
                skipped_missing_face += len(raw_bins)
            except TypeError:
                skipped_missing_face += 1
            continue
        if not isinstance(raw_bins, (list, tuple, set)):
            continue
        size = int(face_atlas.texture.shape[0])
        clean_bins = []
        for raw_bin in raw_bins:
            try:
                bin_id = int(raw_bin)
            except (TypeError, ValueError):
                continue
            if 0 <= bin_id < size * size:
                clean_bins.append(bin_id)
        if clean_bins:
            target_bins_by_face[face] = sorted(set(clean_bins))
    eligible_keys = {
        (int(face), int(bin_id))
        for face, bins in target_bins_by_face.items()
        for bin_id in bins
    }
    summary["eligible_bin_count"] = int(len(eligible_keys))
    summary["skipped_missing_face_count"] = int(skipped_missing_face)
    if not eligible_keys:
        summary["reason"] = "no_valid_target_impact_bins"
        return summary

    feature_dim = _teacher_distilled_basis_feature_dim(feature_mode_s)
    min_samples_i = max(int(min_samples), feature_dim + 1)
    max_samples_face_i = max(0, int(max_samples_per_face))
    max_abs = max(0.0, float(max_abs_delta_rgb))
    blend_f = float(np.clip(float(blend), 0.0, 1.0))
    ridge_f = max(float(ridge), 1.0e-12)
    max_condition_f = max(float(max_condition), 1.0)
    min_norm_f = max(0.0, float(min_norm))
    synthetic_count_i = max(0, int(synthetic_count))
    first_atlas = next(iter(atlas.values()))
    texture_size = int(first_atlas.texture.shape[0])

    face_xtx: dict[int, np.ndarray] = {}
    face_xty: dict[int, np.ndarray] = {}
    face_samples: dict[int, int] = {}
    face_normal_sum: dict[int, np.ndarray] = {}
    face_normal_count: dict[int, int] = {}
    face_camera_dirs: dict[int, list[np.ndarray]] = {}
    candidate_faces = set(int(face) for face in target_bins_by_face)
    rng = np.random.default_rng(23)
    view_paths = list(train_fit_views)
    if int(max_views) > 0:
        view_paths = view_paths[: int(max_views)]
    summary["train_fit_view_count"] = int(len(train_fit_views))
    summary["train_fit_views_used"] = int(len(view_paths))

    sample_events = 0
    for path in tqdm(view_paths, desc="target-impact affine fill fit"):
        z = np.load(path)
        if residual_rgb_key not in z or "normal" not in z or "barycentric" not in z or "face_id" not in z:
            continue
        mask = _valid_sample_mask(
            z,
            candidate_faces,
            residual_l1_key=residual_l1_key,
            min_l1=float(min_l1),
            min_alpha=float(min_alpha),
        )
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if int(max_samples_per_view) > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            local_mask = np.zeros_like(mask, dtype=bool)
            local_mask[ys, xs] = True
            mask = local_mask
        features = _teacher_distilled_basis_features_for_mask(z, feature_mode_s, mask)
        if features is None or features.shape[1] != feature_dim:
            continue
        face_ids = np.asarray(z["face_id"], dtype=np.int64)[mask]
        residual = np.asarray(z[residual_rgb_key], dtype=np.float32)
        residual_samples = np.stack(
            [residual[0][mask], residual[1][mask], residual[2][mask]],
            axis=1,
        ).astype(np.float32)
        normal = np.asarray(z["normal"], dtype=np.float32)
        normal_samples = np.stack([normal[0][mask], normal[1][mask], normal[2][mask]], axis=1)
        normal_samples = np.nan_to_num(normal_samples, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        normal_samples /= np.maximum(np.linalg.norm(normal_samples, axis=1, keepdims=True), 1.0e-8)
        direction = _normalized_camera_center(z)
        for face_raw in np.unique(face_ids):
            face = int(face_raw)
            if face not in target_bins_by_face:
                continue
            idx = np.flatnonzero(face_ids == face)
            if idx.size == 0:
                continue
            if max_samples_face_i > 0:
                remaining = int(max_samples_face_i) - int(face_samples.get(face, 0))
                if remaining <= 0:
                    continue
                if idx.size > remaining:
                    idx = idx[:remaining]
            x = np.asarray(features[idx], dtype=np.float64)
            y = np.asarray(residual_samples[idx], dtype=np.float64)
            face_xtx.setdefault(face, np.zeros((feature_dim, feature_dim), dtype=np.float64))
            face_xty.setdefault(face, np.zeros((feature_dim, 3), dtype=np.float64))
            face_xtx[face] += x.T @ x
            face_xty[face] += x.T @ y
            face_samples[face] = int(face_samples.get(face, 0)) + int(idx.size)
            face_normal_sum[face] = face_normal_sum.get(face, np.zeros((3,), dtype=np.float64)) + np.sum(
                np.asarray(normal_samples[idx], dtype=np.float64),
                axis=0,
            )
            face_normal_count[face] = int(face_normal_count.get(face, 0)) + int(idx.size)
            if direction is not None:
                dirs = face_camera_dirs.setdefault(face, [])
                if int(max_views) <= 0 or len(dirs) < int(max_views):
                    dirs.append(np.asarray(direction, dtype=np.float32))
            sample_events += int(idx.size)

    filled_rows: list[dict[str, Any]] = []
    filled_bins_by_face: dict[str, list[int]] = {}
    fit_faces = 0
    skipped_low_sample_face = 0
    skipped_bad_condition_face = 0
    skipped_low_norm = 0
    fit_stats: list[dict[str, Any]] = []
    identity = np.eye(feature_dim, dtype=np.float64)
    for face, target_bins in sorted(target_bins_by_face.items()):
        samples = int(face_samples.get(face, 0))
        if samples < min_samples_i:
            skipped_low_sample_face += 1
            continue
        mat = face_xtx[face] + ridge_f * identity
        try:
            condition = float(np.linalg.cond(mat))
        except np.linalg.LinAlgError:
            condition = float("inf")
        if not np.isfinite(condition) or condition > max_condition_f:
            skipped_bad_condition_face += 1
            continue
        try:
            coeff = np.linalg.solve(mat, face_xty[face])
        except np.linalg.LinAlgError:
            skipped_bad_condition_face += 1
            continue
        normal_count = max(int(face_normal_count.get(face, 0)), 1)
        normal_vec = (face_normal_sum[face] / float(normal_count)).astype(np.float32)
        normal_norm = float(np.linalg.norm(normal_vec))
        if normal_norm <= 1.0e-8:
            skipped_bad_condition_face += 1
            continue
        normal_vec = normal_vec / normal_norm
        dirs = face_camera_dirs.get(face, [])
        if not dirs:
            skipped_bad_condition_face += 1
            continue
        camera_arr = np.asarray(dirs, dtype=np.float32)
        face_atlas = atlas.get(face)
        if face_atlas is None:
            continue
        size = int(face_atlas.texture.shape[0])
        fit_faces += 1
        fit_stats.append({"face_id": int(face), "samples": int(samples), "condition": condition})
        for bin_id in target_bins:
            v = int(bin_id) // size
            u = int(bin_id) % size
            u_center = np.full((camera_arr.shape[0],), (float(u) + 0.5) / float(size), dtype=np.float32)
            v_center = np.full((camera_arr.shape[0],), (float(v) + 0.5) / float(size), dtype=np.float32)
            pred_features = _teacher_distilled_basis_features_from_uv_camera_normal(
                feature_mode_s,
                u_center,
                v_center,
                camera_arr,
                normal_vec,
            )
            if pred_features is None:
                continue
            preds = np.asarray(pred_features, dtype=np.float64) @ coeff
            estimate = np.mean(preds, axis=0).astype(np.float32)
            estimate = clip_delta_rgb(estimate.reshape(1, 3), max_abs).reshape(3)
            estimate_norm = float(np.linalg.norm(estimate.astype(np.float32)))
            if estimate_norm < min_norm_f:
                skipped_low_norm += 1
                continue
            old_value = np.asarray(face_atlas.texture[v, u], dtype=np.float32)
            new_value = ((1.0 - blend_f) * old_value + blend_f * estimate).astype(np.float32)
            new_value = clip_delta_rgb(new_value.reshape(1, 3), max_abs).reshape(3)
            face_atlas.texture[v, u] = new_value
            effective_count = max(synthetic_count_i, int(samples))
            if effective_count > 0:
                face_atlas.counts[v, u] = max(int(face_atlas.counts[v, u]), int(effective_count))
            pred_var = np.var(preds, axis=0).astype(np.float32) if preds.shape[0] > 1 else np.zeros((3,), dtype=np.float32)
            face_atlas.variance[v, u] = np.maximum(face_atlas.variance[v, u], pred_var)
            sign_consistency = np.abs(np.mean(np.sign(preds), axis=0)).astype(np.float32)
            face_atlas.sign_consistency[v, u] = np.maximum(
                face_atlas.sign_consistency[v, u],
                np.clip(sign_consistency, 0.0, 1.0),
            )
            filled_bins_by_face.setdefault(str(face), []).append(int(bin_id))
            filled_rows.append(
                {
                    "face_id": int(face),
                    "bin_id": int(bin_id),
                    "u": int(u),
                    "v": int(v),
                    "face_samples": int(samples),
                    "prediction_view_count": int(camera_arr.shape[0]),
                    "condition": condition,
                    "estimate_norm": float(estimate_norm),
                    "old_norm": float(np.linalg.norm(old_value.astype(np.float32))),
                    "new_norm": float(np.linalg.norm(new_value.astype(np.float32))),
                }
            )
    for bins in filled_bins_by_face.values():
        bins.sort()
    summary.update(
        {
            "reason": "filled" if filled_rows else "no_bins_filled",
            "filled_bin_count": int(len(filled_rows)),
            "train_fit_views_used": int(len(view_paths)),
            "sample_event_count": int(sample_events),
            "fit_face_count": int(fit_faces),
            "skipped_low_sample_face_count": int(skipped_low_sample_face),
            "skipped_bad_condition_face_count": int(skipped_bad_condition_face),
            "skipped_low_norm_count": int(skipped_low_norm),
            "filled_bins_by_face": filled_bins_by_face,
            "fit_stats": sorted(fit_stats, key=lambda row: int(row["samples"]), reverse=True)[:128],
            "top_filled_bins": sorted(
                filled_rows,
                key=lambda row: (int(row["face_samples"]), float(row["estimate_norm"])),
                reverse=True,
            )[:128],
        }
    )
    return summary


def calibrated_bin_uncertainty_shrink_profile_from_policy_val(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    base_alpha_grid: list[float],
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_samples_per_view: int,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    min_bin_samples: int,
    min_bin_views: int,
    min_relative_gain: float,
    min_positive_view_fraction: float,
    max_mean_variance: float,
    min_mean_sign_consistency: float,
    count_tau: float,
    gain_tau: float,
    variance_scale: float,
    sign_power: float,
    min_shrink: float,
    max_shrink: float,
    fallback_shrink: float,
    policy_mode: str,
    max_profile_bins: int,
    enable_structure_aware_shrink: bool = False,
    structure_shrink_l1_weight: float = 0.0,
    structure_shrink_gradient_weight: float = 0.0,
    structure_shrink_edge_weight: float = 0.0,
    structure_shrink_risk_tau: float = 0.002,
    structure_shrink_max_penalty: float = 1.0,
    enable_view_cluster_local_shrink: bool = False,
    view_cluster_local_global_fallback: bool = False,
    enable_image_l1_bin_certificate: bool = False,
    image_l1_certificate_mode: str = "and",
    image_l1_certificate_min_relative_gain: float = 0.0,
    image_l1_certificate_min_positive_view_fraction: float = 0.55,
    image_l1_certificate_gain_tau: float = 0.01,
    image_l1_certificate_max_abs_delta_rgb: float = -1.0,
    image_l1_certificate_pool_radius: int = 0,
    enable_image_l1_region_expansion: bool = False,
    image_l1_region_expansion_radius: int = 1,
    image_l1_region_expansion_max_bins_per_seed: int = 8,
    image_l1_region_expansion_min_neighbor_samples: int = 1,
    image_l1_region_expansion_min_neighbor_views: int = 1,
    image_l1_region_expansion_max_negative_relative_gain: float = 0.02,
    image_l1_region_expansion_max_negative_image_l1_gain: float = 0.02,
    image_l1_region_expansion_shrink_decay: float = 0.5,
) -> tuple[list[float], dict[str, Any]]:
    """Build a train-policy-val uncertainty-aware residual shrink field.

    This is not another scalar alpha search.  It estimates where the surface
    atlas residual is trustworthy, then attenuates unreliable face/UV bins by
    sample count, policy-val gain, positive-view coverage, residual variance,
    and sign consistency.  The normal policy-val risk gate still selects the
    global alpha after this local shrink field is attached.  v67 used
    sparse_positive mode: unknown bins fall back to the configured shrink.
    v68 can use keep_with_downweight mode: unknown bins keep the fallback
    residual strength, while bins with explicit negative/weak evidence are
    locally attenuated. positive_consensus is stricter: unknown bins are zero
    and only bins with enough multi-view positive policy-val evidence are
    materialized.
    """
    base = sorted(set(float(x) for x in base_alpha_grid))
    disabled = {
        "enabled": False,
        "mode": "policy_val_bin_uncertainty_shrink",
        "reason": "",
        "alpha_grid": base,
    }
    if not val_views or not atlas:
        disabled["reason"] = "no_policy_val_views_or_empty_atlas"
        return base, disabled
    rng = np.random.default_rng(67)
    before_by_key: dict[tuple[int, ...], float] = {}
    after_by_key: dict[tuple[int, ...], float] = {}
    samples_by_key: dict[tuple[int, ...], int] = {}
    view_count_by_key: dict[tuple[int, ...], int] = {}
    positive_view_count_by_key: dict[tuple[int, ...], int] = {}
    variance_sum_by_key: dict[tuple[int, ...], float] = {}
    sign_sum_by_key: dict[tuple[int, ...], float] = {}
    structure_l1_bad_sum_by_key: dict[tuple[int, ...], float] = {}
    structure_gradient_bad_sum_by_key: dict[tuple[int, ...], float] = {}
    structure_edge_sum_by_key: dict[tuple[int, ...], float] = {}
    structure_bad_view_count_by_key: dict[tuple[int, ...], int] = {}
    image_l1_before_by_key: dict[tuple[int, ...], float] = {}
    image_l1_after_by_key: dict[tuple[int, ...], float] = {}
    image_l1_samples_by_key: dict[tuple[int, ...], int] = {}
    image_l1_view_count_by_key: dict[tuple[int, ...], int] = {}
    image_l1_positive_view_count_by_key: dict[tuple[int, ...], int] = {}
    total_active_samples = 0
    view_count = 0
    structure_view_count = 0
    structure_missing_view_count = 0
    structure_enabled = bool(enable_structure_aware_shrink) and (
        float(structure_shrink_l1_weight) > 0.0
        or float(structure_shrink_gradient_weight) > 0.0
        or float(structure_shrink_edge_weight) > 0.0
    )
    image_l1_certificate_mode_s = str(image_l1_certificate_mode)
    if image_l1_certificate_mode_s not in {"and", "or", "replace"}:
        image_l1_certificate_mode_s = "and"
    image_l1_certificate_enabled = bool(enable_image_l1_bin_certificate)
    image_l1_certificate_pool_radius_i = max(0, int(image_l1_certificate_pool_radius))
    image_l1_region_expansion_enabled = bool(enable_image_l1_region_expansion) and image_l1_certificate_enabled
    image_l1_region_expansion_radius_i = max(0, int(image_l1_region_expansion_radius))
    image_l1_region_expansion_max_bins_per_seed_i = max(0, int(image_l1_region_expansion_max_bins_per_seed))
    image_l1_region_expansion_min_neighbor_samples_i = max(1, int(image_l1_region_expansion_min_neighbor_samples))
    image_l1_region_expansion_min_neighbor_views_i = max(1, int(image_l1_region_expansion_min_neighbor_views))
    image_l1_region_expansion_max_negative_relative_gain_f = max(
        0.0,
        float(image_l1_region_expansion_max_negative_relative_gain),
    )
    image_l1_region_expansion_max_negative_image_l1_gain_f = max(
        0.0,
        float(image_l1_region_expansion_max_negative_image_l1_gain),
    )
    image_l1_region_expansion_shrink_decay_f = float(
        np.clip(float(image_l1_region_expansion_shrink_decay), 0.0, 1.0)
    )
    image_l1_certificate_view_count = 0
    image_l1_certificate_missing_view_count = 0
    texture_size = int(next(iter(atlas.values())).texture.shape[0])
    cluster_centers, cluster_feature_mode = _view_cluster_profile_from_atlas(atlas)
    view_cluster_local_enabled = bool(enable_view_cluster_local_shrink) and (
        cluster_centers is not None and str(cluster_feature_mode) != "none"
    )
    view_cluster_policy_views_used = 0
    view_cluster_missing_policy_views = 0
    view_cluster_counts: dict[int, int] = {}
    for path in tqdm(val_views, desc="calibrate bin uncertainty shrink"):
        z = np.load(path)
        if residual_rgb_key not in z or "barycentric" not in z:
            continue
        view_cluster_index = None
        if view_cluster_local_enabled:
            view_cluster_index = _assign_view_cluster_for_npz(
                z,
                centers=cluster_centers,
                feature_mode=str(cluster_feature_mode),
            )
            if view_cluster_index is None:
                view_cluster_missing_policy_views += 1
                continue
            view_cluster_policy_views_used += 1
            view_cluster_counts[int(view_cluster_index)] = view_cluster_counts.get(int(view_cluster_index), 0) + 1
        mask = _valid_sample_mask(z, set(atlas.keys()), residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            mask = np.zeros_like(mask, dtype=bool)
            mask[ys, xs] = True
        bary = np.asarray(z["barycentric"], dtype=np.float32)
        uv_u, uv_v = _uv_bins(bary, mask, texture_size)
        face_samples = np.asarray(z["face_id"], dtype=np.int64)[mask]
        target_rgb = np.asarray(z[residual_rgb_key], dtype=np.float32)
        target = np.stack([target_rgb[0][mask], target_rgb[1][mask], target_rgb[2][mask]], axis=1)
        pred, _valid = predict_delta_for_npz(
            z,
            atlas,
            1.0,
            min_alpha,
            min_atlas_bin_count=int(min_atlas_bin_count),
            min_atlas_face_samples=int(min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(atlas_confidence_mode),
            atlas_confidence_count_scale=float(atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(min_atlas_confidence),
        )
        pred_samples = np.stack([pred[0][mask], pred[1][mask], pred[2][mask]], axis=1).astype(np.float32)
        active = np.linalg.norm(pred_samples, axis=1) > 1.0e-12
        if not bool(np.any(active)):
            continue
        sample_ys = ys[active]
        sample_xs = xs[active]
        target = target.astype(np.float64)[active]
        pred_samples = pred_samples.astype(np.float64)[active]
        face_samples = face_samples[active]
        uv_u = uv_u[active]
        uv_v = uv_v[active]
        before = np.sum(target * target, axis=1)
        err = target - pred_samples
        after = np.sum(err * err, axis=1)
        image_l1_before_samples = None
        image_l1_after_samples = None
        if image_l1_certificate_enabled:
            render_chw = _as_rgb_chw(np.asarray(z["rgb_render"], dtype=np.float32)) if "rgb_render" in z else None
            gt_chw = _as_rgb_chw(np.asarray(z["rgb_gt"], dtype=np.float32)) if "rgb_gt" in z else None
            if render_chw is not None and gt_chw is not None and tuple(render_chw.shape) == tuple(gt_chw.shape):
                pred_for_image_l1 = (
                    clip_delta_rgb(pred, float(image_l1_certificate_max_abs_delta_rgb))
                    if float(image_l1_certificate_max_abs_delta_rgb) >= 0.0
                    else pred
                )
                adapted = np.clip(render_chw + pred_for_image_l1.astype(np.float32), 0.0, 1.0)
                image_l1_before_map = np.mean(np.abs(render_chw - gt_chw), axis=0)
                image_l1_after_map = np.mean(np.abs(adapted - gt_chw), axis=0)
                image_l1_before_samples = image_l1_before_map[sample_ys, sample_xs].astype(np.float64)
                image_l1_after_samples = image_l1_after_map[sample_ys, sample_xs].astype(np.float64)
                image_l1_certificate_view_count += 1
            else:
                image_l1_certificate_missing_view_count += 1
        l1_bad_samples = None
        gradient_bad_samples = None
        edge_samples = None
        if structure_enabled:
            render_chw = _as_rgb_chw(np.asarray(z["rgb_render"], dtype=np.float32)) if "rgb_render" in z else None
            gt_chw = _as_rgb_chw(np.asarray(z["rgb_gt"], dtype=np.float32)) if "rgb_gt" in z else None
            if render_chw is not None and gt_chw is not None and tuple(render_chw.shape) == tuple(gt_chw.shape):
                adapted = np.clip(render_chw + pred.astype(np.float32), 0.0, 1.0)
                l1_before_map = np.mean(np.abs(render_chw - gt_chw), axis=0)
                l1_after_map = np.mean(np.abs(adapted - gt_chw), axis=0)
                l1_bad_map = np.maximum(l1_after_map - l1_before_map, 0.0).astype(np.float32)
                render_grad = _luminance_gradient_magnitude_chw(render_chw)
                gt_grad = _luminance_gradient_magnitude_chw(gt_chw)
                adapted_grad = _luminance_gradient_magnitude_chw(adapted)
                if render_grad is not None and gt_grad is not None and adapted_grad is not None:
                    gradient_bad_map = np.maximum(
                        np.abs(adapted_grad - gt_grad) - np.abs(render_grad - gt_grad),
                        0.0,
                    ).astype(np.float32)
                    edge_map = np.maximum(render_grad, gt_grad).astype(np.float32)
                    l1_bad_samples = l1_bad_map[sample_ys, sample_xs].astype(np.float64)
                    gradient_bad_samples = gradient_bad_map[sample_ys, sample_xs].astype(np.float64)
                    edge_samples = edge_map[sample_ys, sample_xs].astype(np.float64)
                    structure_view_count += 1
                else:
                    structure_missing_view_count += 1
            else:
                structure_missing_view_count += 1
        total_active_samples += int(target.shape[0])
        view_count += 1
        bin_ids = (uv_v.astype(np.int64) * texture_size) + uv_u.astype(np.int64)
        for face in np.unique(face_samples):
            face_i = int(face)
            if face_i not in atlas:
                continue
            face_mask = face_samples == face_i
            face_atlas = atlas[face_i]
            for bin_id in np.unique(bin_ids[face_mask]):
                key = (
                    (int(view_cluster_index), face_i, int(bin_id))
                    if view_cluster_local_enabled
                    else (face_i, int(bin_id))
                )
                bm = face_mask & (bin_ids == int(bin_id))
                local_before = float(np.sum(before[bm]))
                local_after = float(np.sum(after[bm]))
                local_count = int(np.sum(bm))
                before_by_key[key] = before_by_key.get(key, 0.0) + local_before
                after_by_key[key] = after_by_key.get(key, 0.0) + local_after
                samples_by_key[key] = samples_by_key.get(key, 0) + local_count
                view_count_by_key[key] = view_count_by_key.get(key, 0) + 1
                gain = (local_before - local_after) / max(local_before, 1.0e-12)
                if gain > 0.0:
                    positive_view_count_by_key[key] = positive_view_count_by_key.get(key, 0) + 1
                ybin = int(bin_id) // texture_size
                xbin = int(bin_id) % texture_size
                local_variance = float(np.mean(face_atlas.variance[ybin, xbin]))
                local_sign = float(np.mean(face_atlas.sign_consistency[ybin, xbin]))
                variance_sum_by_key[key] = variance_sum_by_key.get(key, 0.0) + local_variance * local_count
                sign_sum_by_key[key] = sign_sum_by_key.get(key, 0.0) + local_sign * local_count
                if l1_bad_samples is not None and gradient_bad_samples is not None and edge_samples is not None:
                    structure_l1 = float(np.sum(l1_bad_samples[bm]))
                    structure_gradient = float(np.sum(gradient_bad_samples[bm]))
                    structure_edge = float(np.sum(edge_samples[bm]))
                    structure_l1_bad_sum_by_key[key] = structure_l1_bad_sum_by_key.get(key, 0.0) + structure_l1
                    structure_gradient_bad_sum_by_key[key] = (
                        structure_gradient_bad_sum_by_key.get(key, 0.0) + structure_gradient
                    )
                    structure_edge_sum_by_key[key] = structure_edge_sum_by_key.get(key, 0.0) + structure_edge
                    if (
                        structure_l1 * max(0.0, float(structure_shrink_l1_weight))
                        + structure_gradient * max(0.0, float(structure_shrink_gradient_weight))
                        + structure_edge * max(0.0, float(structure_shrink_edge_weight))
                    ) > 0.0:
                        structure_bad_view_count_by_key[key] = structure_bad_view_count_by_key.get(key, 0) + 1
                if image_l1_before_samples is not None and image_l1_after_samples is not None:
                    if image_l1_certificate_pool_radius_i > 0:
                        center_u = int(bin_id) % texture_size
                        center_v = int(bin_id) // texture_size
                        patch_mask = face_mask & (
                            np.abs(uv_u.astype(np.int64) - center_u) <= image_l1_certificate_pool_radius_i
                        ) & (
                            np.abs(uv_v.astype(np.int64) - center_v) <= image_l1_certificate_pool_radius_i
                        )
                    else:
                        patch_mask = bm
                    image_l1_before = float(np.sum(image_l1_before_samples[patch_mask]))
                    image_l1_after = float(np.sum(image_l1_after_samples[patch_mask]))
                    image_l1_count = int(np.sum(patch_mask))
                    image_l1_before_by_key[key] = image_l1_before_by_key.get(key, 0.0) + image_l1_before
                    image_l1_after_by_key[key] = image_l1_after_by_key.get(key, 0.0) + image_l1_after
                    image_l1_samples_by_key[key] = image_l1_samples_by_key.get(key, 0) + image_l1_count
                    image_l1_view_count_by_key[key] = image_l1_view_count_by_key.get(key, 0) + 1
                    if image_l1_before - image_l1_after > 0.0:
                        image_l1_positive_view_count_by_key[key] = (
                            image_l1_positive_view_count_by_key.get(key, 0) + 1
                        )

    if not before_by_key:
        disabled["reason"] = "no_active_policy_val_predictions"
        disabled["policy_val_views_used"] = int(view_count)
        disabled["samples"] = int(total_active_samples)
        return base, disabled

    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    fallback_bin_count = 0
    min_shrink_f = float(min_shrink)
    max_shrink_f = float(max_shrink)
    policy_mode_s = str(policy_mode)
    requested_min_bin_views = max(1, int(min_bin_views))
    effective_min_bin_views = max(2, requested_min_bin_views) if policy_mode_s == "positive_consensus" else requested_min_bin_views
    fallback_shrink_f = float(np.clip(fallback_shrink, min_shrink_f, max_shrink_f))
    if policy_mode_s == "positive_consensus":
        fallback_shrink_f = 0.0
    for key in sorted(before_by_key):
        if view_cluster_local_enabled:
            cluster_id, face, bin_id = key
        else:
            cluster_id = None
            face, bin_id = key
        before = float(before_by_key[key])
        after = float(after_by_key.get(key, 0.0))
        samples = int(samples_by_key.get(key, 0))
        views = int(view_count_by_key.get(key, 0))
        positive_views = int(positive_view_count_by_key.get(key, 0))
        relative_gain = (before - after) / max(before, 1.0e-12)
        positive_fraction = float(positive_views / max(1, views))
        mean_variance = float(variance_sum_by_key.get(key, 0.0) / max(1, samples))
        mean_sign = float(sign_sum_by_key.get(key, 0.0) / max(1, samples))
        mean_structure_l1_bad = float(structure_l1_bad_sum_by_key.get(key, 0.0) / max(1, samples))
        mean_structure_gradient_bad = float(structure_gradient_bad_sum_by_key.get(key, 0.0) / max(1, samples))
        mean_structure_edge = float(structure_edge_sum_by_key.get(key, 0.0) / max(1, samples))
        image_l1_before = float(image_l1_before_by_key.get(key, 0.0))
        image_l1_after = float(image_l1_after_by_key.get(key, 0.0))
        image_l1_samples = int(image_l1_samples_by_key.get(key, 0))
        image_l1_views = int(image_l1_view_count_by_key.get(key, 0))
        image_l1_positive_views = int(image_l1_positive_view_count_by_key.get(key, 0))
        image_l1_relative_gain = (
            (image_l1_before - image_l1_after) / max(image_l1_before, 1.0e-12)
            if image_l1_samples > 0
            else 0.0
        )
        image_l1_positive_fraction = float(image_l1_positive_views / max(1, image_l1_views))
        image_l1_view_support_ok = image_l1_views >= int(effective_min_bin_views)
        image_l1_evidence_ok = bool(
            image_l1_certificate_enabled
            and image_l1_samples >= int(min_bin_samples)
            and image_l1_view_support_ok
            and image_l1_relative_gain >= float(image_l1_certificate_min_relative_gain)
            and image_l1_positive_fraction >= float(image_l1_certificate_min_positive_view_fraction)
        )
        structure_raw_risk = float(
            max(0.0, float(structure_shrink_l1_weight)) * mean_structure_l1_bad
            + max(0.0, float(structure_shrink_gradient_weight)) * mean_structure_gradient_bad
            + max(0.0, float(structure_shrink_edge_weight)) * mean_structure_edge
        )
        if structure_enabled:
            if float(structure_shrink_risk_tau) > 0.0:
                structure_risk_conf = structure_raw_risk / (structure_raw_risk + float(structure_shrink_risk_tau))
            else:
                structure_risk_conf = structure_raw_risk
            structure_risk_conf = float(
                np.clip(structure_risk_conf, 0.0, max(0.0, min(1.0, float(structure_shrink_max_penalty))))
            )
        else:
            structure_risk_conf = 0.0
        variance_ok = float(max_mean_variance) < 0.0 or mean_variance <= float(max_mean_variance)
        sign_ok = float(min_mean_sign_consistency) <= 0.0 or mean_sign >= float(min_mean_sign_consistency)
        view_support_ok = views >= int(effective_min_bin_views)
        evidence_ok = bool(
            samples >= int(min_bin_samples)
            and view_support_ok
            and relative_gain >= float(min_relative_gain)
            and positive_fraction >= float(min_positive_view_fraction)
            and variance_ok
            and sign_ok
        )
        residual_evidence_ok = bool(evidence_ok)
        if image_l1_certificate_enabled:
            if image_l1_certificate_mode_s == "replace":
                evidence_ok = bool(image_l1_evidence_ok and variance_ok and sign_ok)
            elif image_l1_certificate_mode_s == "or":
                evidence_ok = bool((residual_evidence_ok or image_l1_evidence_ok) and variance_ok and sign_ok)
            else:
                evidence_ok = bool(residual_evidence_ok and image_l1_evidence_ok)
        count_conf = (
            float(samples) / (float(samples) + float(count_tau))
            if float(count_tau) > 0.0
            else 1.0
        )
        gain_excess = max(0.0, relative_gain - float(min_relative_gain))
        gain_conf = (
            gain_excess / (gain_excess + float(gain_tau))
            if float(gain_tau) > 0.0
            else (1.0 if gain_excess > 0.0 or relative_gain >= float(min_relative_gain) else 0.0)
        )
        image_l1_gain_excess = max(0.0, image_l1_relative_gain - float(image_l1_certificate_min_relative_gain))
        image_l1_gain_conf = (
            image_l1_gain_excess / (image_l1_gain_excess + float(image_l1_certificate_gain_tau))
            if float(image_l1_certificate_gain_tau) > 0.0
            else (
                1.0
                if image_l1_gain_excess > 0.0
                or image_l1_relative_gain >= float(image_l1_certificate_min_relative_gain)
                else 0.0
            )
        )
        if image_l1_certificate_enabled and image_l1_certificate_mode_s == "replace":
            gain_conf_for_shrink = image_l1_gain_conf
            positive_fraction_for_shrink = image_l1_positive_fraction
        elif image_l1_certificate_enabled and image_l1_certificate_mode_s == "or":
            gain_conf_for_shrink = max(gain_conf, image_l1_gain_conf)
            positive_fraction_for_shrink = max(positive_fraction, image_l1_positive_fraction)
        else:
            gain_conf_for_shrink = gain_conf
            positive_fraction_for_shrink = positive_fraction
        variance_conf = (
            1.0 / (1.0 + max(0.0, mean_variance) / float(variance_scale))
            if float(variance_scale) > 0.0
            else 1.0
        )
        sign_conf = (
            float(np.clip(mean_sign, 0.0, 1.0)) ** float(sign_power)
            if float(sign_power) > 0.0
            else 1.0
        )
        positive_shrink = float(
            np.clip(
                max_shrink_f
                * count_conf
                * gain_conf_for_shrink
                * positive_fraction_for_shrink
                * variance_conf
                * sign_conf,
                min_shrink_f,
                max_shrink_f,
            )
        )
        if policy_mode_s == "keep_with_downweight":
            positive_deficit = max(0.0, float(min_positive_view_fraction) - positive_fraction) / max(
                float(min_positive_view_fraction), 1.0e-12
            )
            negative_gain = max(0.0, float(min_relative_gain) - relative_gain)
            negative_gain_conf = (
                negative_gain / (negative_gain + float(gain_tau))
                if float(gain_tau) > 0.0
                else (1.0 if negative_gain > 0.0 else 0.0)
            )
            variance_penalty = 0.0
            if float(max_mean_variance) >= 0.0 and mean_variance > float(max_mean_variance):
                variance_penalty = (mean_variance - float(max_mean_variance)) / (
                    mean_variance + float(max_mean_variance) + 1.0e-12
                )
            sign_penalty = 0.0
            if float(min_mean_sign_consistency) > 0.0 and mean_sign < float(min_mean_sign_consistency):
                sign_penalty = (float(min_mean_sign_consistency) - mean_sign) / max(
                    float(min_mean_sign_consistency), 1.0e-12
                )
            risk_conf = float(
                np.clip(
                    count_conf
                    * max(
                        negative_gain_conf,
                        positive_deficit,
                        variance_penalty,
                        sign_penalty,
                        structure_risk_conf,
                    ),
                    0.0,
                    1.0,
                )
            )
            shrink = float(
                np.clip(
                    fallback_shrink_f - risk_conf * (fallback_shrink_f - min_shrink_f),
                    min_shrink_f,
                    max_shrink_f,
                )
            )
            if evidence_ok:
                if structure_enabled and structure_risk_conf > 0.0:
                    structure_limited_fallback = fallback_shrink_f - structure_risk_conf * (
                        fallback_shrink_f - min_shrink_f
                    )
                    mse_supported_shrink = float(max(shrink, positive_shrink))
                    shrink = float(
                        np.clip(
                            min(mse_supported_shrink, structure_limited_fallback),
                            min_shrink_f,
                            max_shrink_f,
                        )
                    )
                else:
                    shrink = float(np.clip(max(shrink, fallback_shrink_f, positive_shrink), min_shrink_f, max_shrink_f))
            profile_row = abs(shrink - fallback_shrink_f) > 1.0e-8
        elif policy_mode_s == "positive_consensus":
            risk_conf = 0.0
            negative_gain_conf = 0.0
            positive_deficit = 0.0
            variance_penalty = 0.0
            sign_penalty = 0.0
            shrink = float(positive_shrink if evidence_ok else 0.0)
            if structure_enabled and evidence_ok and structure_risk_conf > 0.0:
                shrink = float(np.clip(shrink * (1.0 - structure_risk_conf), min_shrink_f, max_shrink_f))
            profile_row = bool(evidence_ok and abs(shrink) > 1.0e-8)
        else:
            risk_conf = 0.0
            negative_gain_conf = 0.0
            positive_deficit = 0.0
            variance_penalty = 0.0
            sign_penalty = 0.0
            shrink = float(positive_shrink if evidence_ok else fallback_shrink_f)
            if structure_enabled and evidence_ok and structure_risk_conf > 0.0:
                structure_limited_fallback = fallback_shrink_f - structure_risk_conf * (
                    fallback_shrink_f - min_shrink_f
                )
                shrink = float(np.clip(min(shrink, structure_limited_fallback), min_shrink_f, max_shrink_f))
            profile_row = bool(evidence_ok)
        row = {
            "view_cluster_index": int(cluster_id) if cluster_id is not None else None,
            "face_id": int(face),
            "bin_id": int(bin_id),
            "u": int(bin_id) % texture_size,
            "v": int(bin_id) // texture_size,
            "samples": int(samples),
            "view_count": int(views),
            "positive_view_count": int(positive_views),
            "positive_view_fraction": float(positive_fraction),
            "view_support_ok": bool(view_support_ok),
            "mse_before_sum": float(before),
            "mse_after_sum": float(after),
            "relative_gain": float(relative_gain),
            "residual_evidence_ok": bool(residual_evidence_ok),
            "image_l1_certificate_enabled": bool(image_l1_certificate_enabled),
            "image_l1_certificate_mode": str(image_l1_certificate_mode_s),
            "image_l1_view_count": int(image_l1_views),
            "image_l1_positive_view_count": int(image_l1_positive_views),
            "image_l1_positive_view_fraction": float(image_l1_positive_fraction),
            "image_l1_view_support_ok": bool(image_l1_view_support_ok),
            "image_l1_before_sum": float(image_l1_before),
            "image_l1_after_sum": float(image_l1_after),
            "image_l1_relative_gain": float(image_l1_relative_gain),
            "image_l1_evidence_ok": bool(image_l1_evidence_ok),
            "image_l1_gain_confidence": float(image_l1_gain_conf),
            "image_l1_region_expanded": False,
            "image_l1_region_seed_face_id": None,
            "image_l1_region_seed_bin_id": None,
            "image_l1_region_seed_view_cluster_index": None,
            "image_l1_region_seed_distance": 0,
            "image_l1_region_expansion_score": 0.0,
            "mean_variance": float(mean_variance),
            "mean_sign_consistency": float(mean_sign),
            "variance_ok": bool(variance_ok),
            "sign_ok": bool(sign_ok),
            "evidence_ok": bool(evidence_ok),
            "count_confidence": float(count_conf),
            "gain_confidence": float(gain_conf),
            "variance_confidence": float(variance_conf),
            "sign_confidence": float(sign_conf),
            "positive_shrink": float(positive_shrink),
            "risk_confidence": float(risk_conf),
            "structure_risk_confidence": float(structure_risk_conf),
            "structure_raw_risk": float(structure_raw_risk),
            "mean_structure_l1_bad": float(mean_structure_l1_bad),
            "mean_structure_gradient_bad": float(mean_structure_gradient_bad),
            "mean_structure_edge": float(mean_structure_edge),
            "structure_bad_view_count": int(structure_bad_view_count_by_key.get(key, 0)),
            "negative_gain_confidence": float(negative_gain_conf),
            "positive_view_deficit": float(positive_deficit),
            "variance_penalty": float(variance_penalty),
            "sign_penalty": float(sign_penalty),
            "shrink": float(shrink),
        }
        rows.append(row)
        if profile_row:
            selected_rows.append(row)
        else:
            fallback_bin_count += 1

    def _profile_row_key(row: dict[str, Any]) -> tuple[int | None, int, int]:
        cluster_value = row.get("view_cluster_index")
        cluster_key = None if cluster_value is None else int(cluster_value)
        return (cluster_key, int(row["face_id"]), int(row["bin_id"]))

    region_expansion_summary: dict[str, Any] = {
        "enabled": bool(image_l1_region_expansion_enabled),
        "uses_policy_val_gt": bool(image_l1_region_expansion_enabled),
        "uses_target_or_test_gt": False,
        "radius": int(image_l1_region_expansion_radius_i),
        "max_bins_per_seed": int(image_l1_region_expansion_max_bins_per_seed_i),
        "min_neighbor_samples": int(image_l1_region_expansion_min_neighbor_samples_i),
        "min_neighbor_views": int(image_l1_region_expansion_min_neighbor_views_i),
        "max_negative_relative_gain": float(image_l1_region_expansion_max_negative_relative_gain_f),
        "max_negative_image_l1_gain": float(image_l1_region_expansion_max_negative_image_l1_gain_f),
        "shrink_decay": float(image_l1_region_expansion_shrink_decay_f),
        "seed_bin_count": 0,
        "candidate_neighbor_count": 0,
        "expanded_bin_count_pre_trunc": 0,
        "selected_expanded_bin_count": 0,
        "mean_expanded_shrink_pre_trunc": 0.0,
        "top_expanded_bins": [],
    }
    if (
        bool(image_l1_region_expansion_enabled)
        and int(image_l1_region_expansion_radius_i) > 0
        and selected_rows
    ):
        seed_rows = [
            row
            for row in selected_rows
            if bool(row.get("image_l1_evidence_ok", False))
            and float(row.get("shrink", 0.0)) > 1.0e-8
        ]
        selected_key_set = {_profile_row_key(row) for row in selected_rows}
        candidate_rows_by_scope: dict[tuple[int | None, int], list[dict[str, Any]]] = {}
        for row in rows:
            row_key = _profile_row_key(row)
            if row_key in selected_key_set:
                continue
            if int(row.get("samples", 0)) < int(image_l1_region_expansion_min_neighbor_samples_i):
                continue
            if int(row.get("view_count", 0)) < int(image_l1_region_expansion_min_neighbor_views_i):
                continue
            if not bool(row.get("variance_ok", False)) or not bool(row.get("sign_ok", False)):
                continue
            if float(row.get("relative_gain", 0.0)) < -float(image_l1_region_expansion_max_negative_relative_gain_f):
                continue
            if (
                int(row.get("image_l1_view_count", 0)) > 0
                and float(row.get("image_l1_relative_gain", 0.0))
                < -float(image_l1_region_expansion_max_negative_image_l1_gain_f)
            ):
                continue
            scope = (row_key[0], row_key[1])
            candidate_rows_by_scope.setdefault(scope, []).append(row)

        expanded_by_key: dict[tuple[int | None, int, int], dict[str, Any]] = {}
        for seed in seed_rows:
            seed_key = _profile_row_key(seed)
            seed_scope = (seed_key[0], seed_key[1])
            seed_u = int(seed.get("u", 0))
            seed_v = int(seed.get("v", 0))
            scored_neighbors: list[tuple[float, dict[str, Any], int, float, float]] = []
            for candidate in candidate_rows_by_scope.get(seed_scope, []):
                candidate_key = _profile_row_key(candidate)
                if candidate_key in selected_key_set:
                    continue
                du = abs(int(candidate.get("u", 0)) - seed_u)
                dv = abs(int(candidate.get("v", 0)) - seed_v)
                distance = max(int(du), int(dv))
                if distance <= 0 or distance > int(image_l1_region_expansion_radius_i):
                    continue
                support_conf = min(
                    1.0,
                    float(candidate.get("samples", 0)) / max(1.0, float(min_bin_samples)),
                )
                support_conf *= min(
                    1.0,
                    float(candidate.get("view_count", 0)) / max(1.0, float(effective_min_bin_views)),
                )
                residual_risk = (
                    max(0.0, -float(candidate.get("relative_gain", 0.0)))
                    / max(float(image_l1_region_expansion_max_negative_relative_gain_f), 1.0e-12)
                    if float(image_l1_region_expansion_max_negative_relative_gain_f) > 0.0
                    else (1.0 if float(candidate.get("relative_gain", 0.0)) < 0.0 else 0.0)
                )
                image_risk = 0.0
                if int(candidate.get("image_l1_view_count", 0)) > 0:
                    image_risk = (
                        max(0.0, -float(candidate.get("image_l1_relative_gain", 0.0)))
                        / max(float(image_l1_region_expansion_max_negative_image_l1_gain_f), 1.0e-12)
                        if float(image_l1_region_expansion_max_negative_image_l1_gain_f) > 0.0
                        else (1.0 if float(candidate.get("image_l1_relative_gain", 0.0)) < 0.0 else 0.0)
                    )
                nonnegative_conf = float(np.clip(1.0 - max(residual_risk, image_risk), 0.0, 1.0))
                distance_conf = float(image_l1_region_expansion_shrink_decay_f) ** float(distance)
                score = (
                    max(0.0, float(seed.get("image_l1_relative_gain", 0.0)))
                    * support_conf
                    * nonnegative_conf
                    * distance_conf
                )
                if score <= 0.0:
                    continue
                scored_neighbors.append(
                    (
                        float(score),
                        candidate,
                        int(distance),
                        float(support_conf),
                        float(nonnegative_conf),
                    )
                )
            scored_neighbors.sort(
                key=lambda item: (
                    float(item[0]),
                    int(item[1].get("samples", 0)),
                    float(item[1].get("image_l1_relative_gain", 0.0)),
                ),
                reverse=True,
            )
            if int(image_l1_region_expansion_max_bins_per_seed_i) > 0:
                scored_neighbors = scored_neighbors[: int(image_l1_region_expansion_max_bins_per_seed_i)]
            for score, candidate, distance, support_conf, nonnegative_conf in scored_neighbors:
                candidate_key = _profile_row_key(candidate)
                inherited_shrink = float(seed.get("shrink", 0.0)) * support_conf * nonnegative_conf
                inherited_shrink *= float(image_l1_region_expansion_shrink_decay_f) ** float(distance)
                inherited_shrink = float(np.clip(inherited_shrink, min_shrink_f, max_shrink_f))
                if inherited_shrink <= 1.0e-8:
                    continue
                current = expanded_by_key.get(candidate_key)
                if current is not None and float(current.get("image_l1_region_expansion_score", 0.0)) >= float(score):
                    continue
                expanded = dict(candidate)
                expanded.update(
                    {
                        "evidence_ok": True,
                        "image_l1_region_expanded": True,
                        "image_l1_region_seed_face_id": int(seed.get("face_id", -1)),
                        "image_l1_region_seed_bin_id": int(seed.get("bin_id", -1)),
                        "image_l1_region_seed_view_cluster_index": seed.get("view_cluster_index"),
                        "image_l1_region_seed_distance": int(distance),
                        "image_l1_region_seed_shrink": float(seed.get("shrink", 0.0)),
                        "image_l1_region_expansion_support_confidence": float(support_conf),
                        "image_l1_region_expansion_nonnegative_confidence": float(nonnegative_conf),
                        "image_l1_region_expansion_score": float(score),
                        "shrink": float(inherited_shrink),
                    }
                )
                expanded_by_key[candidate_key] = expanded

        expanded_rows = sorted(
            expanded_by_key.values(),
            key=lambda row: (
                float(row.get("image_l1_region_expansion_score", 0.0)),
                float(row.get("shrink", 0.0)),
                int(row.get("samples", 0)),
            ),
            reverse=True,
        )
        if expanded_rows:
            selected_rows.extend(expanded_rows)
            fallback_bin_count = max(0, int(fallback_bin_count) - int(len(expanded_rows)))
        region_expansion_summary.update(
            {
                "seed_bin_count": int(len(seed_rows)),
                "candidate_neighbor_count": int(
                    sum(len(value) for value in candidate_rows_by_scope.values())
                ),
                "expanded_bin_count_pre_trunc": int(len(expanded_rows)),
                "mean_expanded_shrink_pre_trunc": float(
                    np.mean([float(row.get("shrink", 0.0)) for row in expanded_rows])
                )
                if expanded_rows
                else 0.0,
                "top_expanded_bins": expanded_rows[:32],
            }
        )

    if policy_mode_s == "keep_with_downweight":
        selected_rows = sorted(
            selected_rows,
            key=lambda row: (
                abs(float(row["shrink"]) - fallback_shrink_f) * max(1, int(row["samples"])),
                float(row["risk_confidence"]),
                int(row["samples"]),
            ),
            reverse=True,
        )
    else:
        selected_rows = sorted(
            selected_rows,
            key=lambda row: (
                float(row["shrink"]),
                float(row["relative_gain"]),
                int(row["samples"]),
            ),
            reverse=True,
        )
    if int(max_profile_bins) > 0 and len(selected_rows) > int(max_profile_bins):
        fallback_bin_count += len(selected_rows) - int(max_profile_bins)
        selected_rows = selected_rows[: int(max_profile_bins)]
    if bool(region_expansion_summary.get("enabled", False)):
        selected_expanded_rows = [
            row for row in selected_rows if bool(row.get("image_l1_region_expanded", False))
        ]
        region_expansion_summary["selected_expanded_bin_count"] = int(len(selected_expanded_rows))
        region_expansion_summary["mean_selected_expanded_shrink"] = (
            float(np.mean([float(row.get("shrink", 0.0)) for row in selected_expanded_rows]))
            if selected_expanded_rows
            else 0.0
        )

    bin_shrinks_by_face: dict[str, dict[str, float]] = {}
    cluster_bin_shrinks_by_cluster_face: dict[str, dict[str, dict[str, float]]] = {}
    for row in selected_rows:
        face_key = str(int(row["face_id"]))
        bin_key = str(int(row["bin_id"]))
        if view_cluster_local_enabled:
            cluster_key = str(int(row["view_cluster_index"]))
            cluster_bin_shrinks_by_cluster_face.setdefault(cluster_key, {}).setdefault(face_key, {})[
                bin_key
            ] = float(row["shrink"])
        else:
            bin_shrinks_by_face.setdefault(face_key, {})[bin_key] = float(row["shrink"])
    if view_cluster_local_enabled and bool(view_cluster_local_global_fallback):
        for cluster_profile in cluster_bin_shrinks_by_cluster_face.values():
            for face_key, face_profile in cluster_profile.items():
                bin_shrinks_by_face.setdefault(face_key, {}).update(face_profile)
    shrinks = [float(row["shrink"]) for row in selected_rows]
    selected_cluster_count = len(cluster_bin_shrinks_by_cluster_face)
    profile = {
        "enabled": True,
        "mode": "policy_val_bin_uncertainty_shrink",
        "uncertainty_shrink_policy_mode": policy_mode_s,
        "alpha_grid": base,
        "policy_val_views_used": int(view_count),
        "samples": int(total_active_samples),
        "texture_size": int(texture_size),
        "min_bin_samples": int(min_bin_samples),
        "min_bin_views": int(effective_min_bin_views),
        "requested_min_bin_views": int(requested_min_bin_views),
        "min_relative_gain": float(min_relative_gain),
        "min_positive_view_fraction": float(min_positive_view_fraction),
        "max_mean_variance": float(max_mean_variance),
        "min_mean_sign_consistency": float(min_mean_sign_consistency),
        "count_tau": float(count_tau),
        "gain_tau": float(gain_tau),
        "variance_scale": float(variance_scale),
        "sign_power": float(sign_power),
        "min_shrink": float(min_shrink_f),
        "max_shrink": float(max_shrink_f),
        "fallback_shrink": float(fallback_shrink_f),
        "view_cluster_local_shrink": bool(view_cluster_local_enabled),
        "requested_view_cluster_local_shrink": bool(enable_view_cluster_local_shrink),
        "view_cluster_local_global_fallback": bool(view_cluster_local_global_fallback),
        "view_cluster_feature_mode": str(cluster_feature_mode) if view_cluster_local_enabled else "none",
        "view_cluster_policy_val_views_used": int(view_cluster_policy_views_used),
        "view_cluster_policy_val_views_missing_feature": int(view_cluster_missing_policy_views),
        "view_cluster_policy_val_view_counts": {
            str(int(key)): int(value) for key, value in sorted(view_cluster_counts.items())
        },
        "view_cluster_selected_cluster_count": int(selected_cluster_count),
        "structure_aware_shrink": {
            "enabled": bool(structure_enabled),
            "uses_policy_val_gt": bool(structure_enabled),
            "uses_target_or_test_gt": False,
            "policy_val_views_with_structure": int(structure_view_count),
            "policy_val_views_missing_structure": int(structure_missing_view_count),
            "l1_weight": float(structure_shrink_l1_weight),
            "gradient_weight": float(structure_shrink_gradient_weight),
            "edge_weight": float(structure_shrink_edge_weight),
            "risk_tau": float(structure_shrink_risk_tau),
            "max_penalty": float(structure_shrink_max_penalty),
            "mean_selected_structure_risk_confidence": float(
                np.mean([float(row["structure_risk_confidence"]) for row in selected_rows])
            )
            if selected_rows
            else 0.0,
            "structure_downweighted_bin_count": int(
                sum(
                    1
                    for row in selected_rows
                    if float(row["structure_risk_confidence"]) > 1.0e-8
                    and float(row["shrink"]) < fallback_shrink_f - 1.0e-8
                )
            ),
        },
        "image_l1_bin_certificate": {
            "enabled": bool(image_l1_certificate_enabled),
            "mode": str(image_l1_certificate_mode_s),
            "uses_policy_val_gt": bool(image_l1_certificate_enabled),
            "uses_target_or_test_gt": False,
            "policy_val_views_with_image_l1": int(image_l1_certificate_view_count),
            "policy_val_views_missing_image_l1": int(image_l1_certificate_missing_view_count),
            "min_relative_gain": float(image_l1_certificate_min_relative_gain),
            "min_positive_view_fraction": float(image_l1_certificate_min_positive_view_fraction),
            "gain_tau": float(image_l1_certificate_gain_tau),
            "max_abs_delta_rgb": float(image_l1_certificate_max_abs_delta_rgb),
            "pool_radius": int(image_l1_certificate_pool_radius_i),
            "region_expansion": dict(region_expansion_summary),
            "candidate_evidence_ok_count": int(
                sum(1 for row in rows if bool(row.get("image_l1_evidence_ok", False)))
            ),
            "selected_evidence_ok_count": int(
                sum(1 for row in selected_rows if bool(row.get("image_l1_evidence_ok", False)))
            ),
            "mean_selected_relative_gain": float(
                np.mean([float(row.get("image_l1_relative_gain", 0.0)) for row in selected_rows])
            )
            if selected_rows
            else 0.0,
            "mean_selected_positive_view_fraction": float(
                np.mean([float(row.get("image_l1_positive_view_fraction", 0.0)) for row in selected_rows])
            )
            if selected_rows
            else 0.0,
            "top_candidate_bins": sorted(
                [
                    row
                    for row in rows
                    if int(row.get("image_l1_view_count", 0) or 0) > 0
                ],
                key=lambda row: (
                    float(row.get("image_l1_relative_gain", 0.0)),
                    float(row.get("image_l1_positive_view_fraction", 0.0)),
                    int(row.get("image_l1_view_count", 0)),
                    int(row.get("samples", 0)),
                ),
                reverse=True,
            )[:32],
        },
        "candidate_bin_count": int(len(rows)),
        "bin_uncertainty_shrink_count": int(len(selected_rows)),
        "fallback_bin_count": int(fallback_bin_count),
        "selected_face_count": int(
            len(
                {
                    str(int(row["face_id"]))
                    for row in selected_rows
                }
            )
        ),
        "mean_selected_shrink": float(np.mean(shrinks)) if shrinks else 0.0,
        "min_selected_shrink": float(np.min(shrinks)) if shrinks else 0.0,
        "max_selected_shrink": float(np.max(shrinks)) if shrinks else 0.0,
        "mean_profile_abs_delta_from_fallback": float(
            np.mean([abs(float(row["shrink"]) - fallback_shrink_f) for row in selected_rows])
        )
        if selected_rows
        else 0.0,
        "downweighted_bin_count": int(
            sum(1 for row in selected_rows if float(row["shrink"]) < fallback_shrink_f - 1.0e-8)
        ),
        "upweighted_bin_count": int(
            sum(1 for row in selected_rows if float(row["shrink"]) > fallback_shrink_f + 1.0e-8)
        ),
        "bin_uncertainty_shrink_preview": selected_rows[:32],
        "worst_bins": sorted(rows, key=lambda row: float(row["relative_gain"]))[:32],
        "bin_shrinks_by_face": bin_shrinks_by_face,
        "cluster_bin_shrinks_by_cluster_face": cluster_bin_shrinks_by_cluster_face,
    }
    return base, profile


def calibrated_alpha_grid_from_policy_val(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    base_alpha_grid: list[float],
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_samples_per_view: int,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    calibration_max_alpha: float,
    calibration_multipliers: list[float],
    min_denominator: float,
) -> tuple[list[float], dict[str, Any]]:
    """Add train-policy-val closed-form alpha candidates without using held-out GT.

    The atlas prediction is linear in alpha.  For each policy-val residual sample,
    compute the least-squares scalar alpha that best maps atlas-predicted residuals
    to the teacher residual, then test a small multiplier set around that estimate
    through the normal risk gates.  Existing user-provided alpha grid values are
    always preserved.
    """
    base = sorted(set(float(x) for x in base_alpha_grid))
    if not val_views or not atlas:
        return base, {
            "enabled": True,
            "reason": "no_policy_val_views_or_empty_atlas",
            "base_alpha_grid": base,
            "calibrated_alpha": 0.0,
            "alpha_grid": base,
        }
    rng = np.random.default_rng(53)
    numerator = 0.0
    denominator = 0.0
    sample_count = 0
    view_count = 0
    for path in tqdm(val_views, desc="calibrate alpha"):
        z = np.load(path)
        if residual_rgb_key not in z:
            continue
        mask = _valid_sample_mask(z, set(atlas.keys()), residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            mask = np.zeros_like(mask, dtype=bool)
            mask[ys, xs] = True
        target_rgb = np.asarray(z[residual_rgb_key], dtype=np.float32)
        target = np.stack([target_rgb[0][mask], target_rgb[1][mask], target_rgb[2][mask]], axis=1)
        pred, _valid = predict_delta_for_npz(
            z,
            atlas,
            1.0,
            min_alpha,
            min_atlas_bin_count=int(min_atlas_bin_count),
            min_atlas_face_samples=int(min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(atlas_confidence_mode),
            atlas_confidence_count_scale=float(atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(min_atlas_confidence),
        )
        pred_samples = np.stack([pred[0][mask], pred[1][mask], pred[2][mask]], axis=1)
        numerator += float(np.sum(pred_samples.astype(np.float64) * target.astype(np.float64)))
        denominator += float(np.sum(np.square(pred_samples.astype(np.float64))))
        sample_count += int(target.shape[0])
        view_count += 1

    if denominator <= float(min_denominator):
        return base, {
            "enabled": True,
            "reason": "small_denominator",
            "base_alpha_grid": base,
            "policy_val_views_used": int(view_count),
            "samples": int(sample_count),
            "numerator": float(numerator),
            "denominator": float(denominator),
            "calibrated_alpha": 0.0,
            "alpha_grid": base,
        }
    raw_alpha = float(numerator / denominator)
    clipped_alpha = float(np.clip(raw_alpha, 0.0, float(calibration_max_alpha)))
    generated: list[float] = []
    for multiplier in calibration_multipliers:
        value = clipped_alpha * float(multiplier)
        if value > 0.0:
            generated.append(float(np.clip(value, 0.0, float(calibration_max_alpha))))
    combined = sorted(set(round(float(x), 8) for x in [*base, *generated, 0.0]))
    return combined, {
        "enabled": True,
        "mode": "policy_val_least_squares_scalar",
        "base_alpha_grid": base,
        "multipliers": [float(x) for x in calibration_multipliers],
        "policy_val_views_used": int(view_count),
        "samples": int(sample_count),
        "numerator": float(numerator),
        "denominator": float(denominator),
        "raw_alpha": float(raw_alpha),
        "calibration_max_alpha": float(calibration_max_alpha),
        "calibrated_alpha": float(clipped_alpha),
        "generated_alpha_candidates": sorted(set(round(float(x), 8) for x in generated)),
        "alpha_grid": combined,
    }


def calibrated_local_alpha_profile_from_policy_val(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    base_alpha_grid: list[float],
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_samples_per_view: int,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    max_alpha: float,
    min_alpha_value: float,
    bucket_quantiles: list[float],
    bucket_edges: list[float],
    multiplier_grid: list[float],
    min_bucket_samples: int,
    norm_mode: str,
    min_denominator: float,
) -> tuple[list[float], dict[str, Any]]:
    """Fit a train-policy-val local alpha profile without held-out GT.

    v53 showed that a single scene-level alpha can improve residual MSE while
    hurting SSIM.  This profile estimates an absolute alpha per residual-norm
    bucket, then the existing scalar alpha grid is reused as a global multiplier
    selected by the normal policy-val risk gate.
    """
    base = sorted(set(float(x) for x in base_alpha_grid))
    disabled = {
        "enabled": False,
        "mode": "policy_val_residual_norm_buckets",
        "reason": "",
        "alpha_grid": base,
    }
    if not val_views or not atlas:
        disabled["reason"] = "no_policy_val_views_or_empty_atlas"
        return base, disabled
    rng = np.random.default_rng(54)
    pred_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    norm_chunks: list[np.ndarray] = []
    view_count = 0
    for path in tqdm(val_views, desc="calibrate local alpha"):
        z = np.load(path)
        if residual_rgb_key not in z:
            continue
        mask = _valid_sample_mask(z, set(atlas.keys()), residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            mask = np.zeros_like(mask, dtype=bool)
            mask[ys, xs] = True
        target_rgb = np.asarray(z[residual_rgb_key], dtype=np.float32)
        target = np.stack([target_rgb[0][mask], target_rgb[1][mask], target_rgb[2][mask]], axis=1)
        pred, _valid = predict_delta_for_npz(
            z,
            atlas,
            1.0,
            min_alpha,
            min_atlas_bin_count=int(min_atlas_bin_count),
            min_atlas_face_samples=int(min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(atlas_confidence_mode),
            atlas_confidence_count_scale=float(atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(min_atlas_confidence),
        )
        pred_samples = np.stack([pred[0][mask], pred[1][mask], pred[2][mask]], axis=1).astype(np.float32)
        if str(norm_mode) == "mean_abs":
            norms = np.mean(np.abs(pred_samples), axis=1)
        else:
            norms = np.linalg.norm(pred_samples, axis=1)
        nonzero = norms > 0.0
        if not bool(np.any(nonzero)):
            continue
        pred_chunks.append(pred_samples[nonzero])
        target_chunks.append(target.astype(np.float32)[nonzero])
        norm_chunks.append(norms[nonzero].astype(np.float32))
        view_count += 1
    if not pred_chunks:
        disabled["reason"] = "no_nonzero_policy_val_predictions"
        return base, disabled
    pred_all = np.concatenate(pred_chunks, axis=0).astype(np.float64)
    target_all = np.concatenate(target_chunks, axis=0).astype(np.float64)
    norms_all = np.concatenate(norm_chunks, axis=0).astype(np.float64)
    denominator_all = float(np.sum(np.square(pred_all)))
    numerator_all = float(np.sum(pred_all * target_all))
    if denominator_all <= float(min_denominator):
        disabled["reason"] = "small_denominator"
        disabled["samples"] = int(pred_all.shape[0])
        disabled["denominator"] = float(denominator_all)
        return base, disabled
    fallback_raw_alpha = float(numerator_all / denominator_all)
    fallback_alpha = float(np.clip(fallback_raw_alpha, float(min_alpha_value), float(max_alpha)))
    if bucket_edges:
        edges = sorted(set(float(x) for x in bucket_edges if float(x) > 0.0))
    else:
        valid_quantiles = [float(q) for q in bucket_quantiles if 0.0 < float(q) < 1.0]
        edges = [float(x) for x in np.quantile(norms_all, valid_quantiles)] if valid_quantiles else []
        edges = sorted(set(round(float(x), 10) for x in edges if float(x) > 0.0))
    bucket_count = len(edges) + 1
    bucket_indices = np.searchsorted(np.asarray(edges, dtype=np.float64), norms_all, side="right")
    bucket_rows: list[dict[str, Any]] = []
    bucket_alphas: list[float] = []
    for bucket_id in range(bucket_count):
        mask = bucket_indices == bucket_id
        count = int(np.sum(mask))
        if count <= 0:
            raw_alpha = fallback_raw_alpha
            clipped_alpha = fallback_alpha
            denominator = 0.0
            numerator = 0.0
            reason = "empty_bucket_fallback"
        else:
            pred_bucket = pred_all[mask]
            target_bucket = target_all[mask]
            denominator = float(np.sum(np.square(pred_bucket)))
            numerator = float(np.sum(pred_bucket * target_bucket))
            if count < int(min_bucket_samples) or denominator <= float(min_denominator):
                raw_alpha = fallback_raw_alpha
                clipped_alpha = fallback_alpha
                reason = "small_bucket_fallback"
            else:
                raw_alpha = float(numerator / denominator)
                clipped_alpha = float(np.clip(raw_alpha, float(min_alpha_value), float(max_alpha)))
                reason = "fit"
        bucket_alphas.append(float(clipped_alpha))
        low = 0.0 if bucket_id == 0 else float(edges[bucket_id - 1])
        high = None if bucket_id == bucket_count - 1 else float(edges[bucket_id])
        bucket_rows.append(
            {
                "bucket": int(bucket_id),
                "norm_low": float(low),
                "norm_high": None if high is None else float(high),
                "samples": int(count),
                "numerator": float(numerator),
                "denominator": float(denominator),
                "raw_alpha": float(raw_alpha),
                "alpha": float(clipped_alpha),
                "reason": str(reason),
            }
        )
    generated = [float(x) for x in multiplier_grid if float(x) >= 0.0]
    if 1.0 not in generated:
        generated.append(1.0)
    combined = sorted(set(round(float(x), 8) for x in [*base, *generated, 0.0]))
    profile = {
        "enabled": True,
        "mode": "policy_val_residual_norm_buckets",
        "norm_mode": str(norm_mode),
        "policy_val_views_used": int(view_count),
        "samples": int(pred_all.shape[0]),
        "bucket_edges": [float(x) for x in edges],
        "bucket_alphas": [float(x) for x in bucket_alphas],
        "bucket_rows": bucket_rows,
        "fallback_raw_alpha": float(fallback_raw_alpha),
        "fallback_alpha": float(fallback_alpha),
        "max_alpha": float(max_alpha),
        "min_alpha": float(min_alpha_value),
        "min_bucket_samples": int(min_bucket_samples),
        "multiplier_grid": sorted(set(float(x) for x in generated)),
        "alpha_grid": combined,
    }
    return combined, profile


def calibrated_face_alpha_profile_from_policy_val(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    base_alpha_grid: list[float],
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_samples_per_view: int,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    max_alpha: float,
    min_alpha_value: float,
    multiplier_grid: list[float],
    min_face_samples: int,
    min_denominator: float,
    shrink_count_tau: float,
    shrink_denominator_tau: float,
    shrink_prior: str,
) -> tuple[list[float], dict[str, Any]]:
    """Fit a train-policy-val alpha per surface face without held-out GT.

    This is a lower-variance surface-local step than per-bin fitting: each face
    receives an LS alpha when policy-val support is sufficient, otherwise it
    falls back to the global policy-val LS alpha.  The existing scalar alpha grid
    remains a global multiplier selected by the normal train-only risk gates.
    """
    base = sorted(set(float(x) for x in base_alpha_grid))
    disabled = {
        "enabled": False,
        "mode": "policy_val_face_alpha",
        "reason": "",
        "alpha_grid": base,
    }
    if not val_views or not atlas:
        disabled["reason"] = "no_policy_val_views_or_empty_atlas"
        return base, disabled
    rng = np.random.default_rng(55)
    numerator_by_face: dict[int, float] = {}
    denominator_by_face: dict[int, float] = {}
    samples_by_face: dict[int, int] = {}
    numerator_all = 0.0
    denominator_all = 0.0
    sample_count = 0
    view_count = 0
    for path in tqdm(val_views, desc="calibrate face alpha"):
        z = np.load(path)
        if residual_rgb_key not in z:
            continue
        mask = _valid_sample_mask(z, set(atlas.keys()), residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            mask = np.zeros_like(mask, dtype=bool)
            mask[ys, xs] = True
        target_rgb = np.asarray(z[residual_rgb_key], dtype=np.float32)
        target = np.stack([target_rgb[0][mask], target_rgb[1][mask], target_rgb[2][mask]], axis=1)
        face_samples = np.asarray(z["face_id"], dtype=np.int64)[mask]
        pred, _valid = predict_delta_for_npz(
            z,
            atlas,
            1.0,
            min_alpha,
            min_atlas_bin_count=int(min_atlas_bin_count),
            min_atlas_face_samples=int(min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(atlas_confidence_mode),
            atlas_confidence_count_scale=float(atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(min_atlas_confidence),
        )
        pred_samples = np.stack([pred[0][mask], pred[1][mask], pred[2][mask]], axis=1).astype(np.float32)
        nonzero = np.linalg.norm(pred_samples, axis=1) > 0.0
        if not bool(np.any(nonzero)):
            continue
        pred_samples = pred_samples[nonzero].astype(np.float64)
        target = target.astype(np.float64)[nonzero]
        face_samples = face_samples[nonzero]
        numerator_all += float(np.sum(pred_samples * target))
        denominator_all += float(np.sum(np.square(pred_samples)))
        sample_count += int(pred_samples.shape[0])
        view_count += 1
        for face in np.unique(face_samples):
            fm = face_samples == int(face)
            face_pred = pred_samples[fm]
            face_target = target[fm]
            numerator_by_face[int(face)] = numerator_by_face.get(int(face), 0.0) + float(
                np.sum(face_pred * face_target)
            )
            denominator_by_face[int(face)] = denominator_by_face.get(int(face), 0.0) + float(
                np.sum(np.square(face_pred))
            )
            samples_by_face[int(face)] = samples_by_face.get(int(face), 0) + int(np.sum(fm))
    if denominator_all <= float(min_denominator):
        disabled["reason"] = "small_denominator"
        disabled["samples"] = int(sample_count)
        disabled["denominator"] = float(denominator_all)
        return base, disabled
    fallback_raw_alpha = float(numerator_all / denominator_all)
    fallback_alpha = float(np.clip(fallback_raw_alpha, float(min_alpha_value), float(max_alpha)))
    face_alphas: dict[str, float] = {}
    fitted_rows: list[dict[str, Any]] = []
    fallback_face_count = 0
    shrink_enabled = float(shrink_count_tau) > 0.0 or float(shrink_denominator_tau) > 0.0
    prior_mode = str(shrink_prior)
    if prior_mode not in {"fallback", "zero"}:
        raise ValueError(f"unsupported face alpha shrink prior: {prior_mode}")
    for face, denominator in denominator_by_face.items():
        count = int(samples_by_face.get(face, 0))
        numerator = float(numerator_by_face.get(face, 0.0))
        if count < int(min_face_samples) or float(denominator) <= float(min_denominator):
            fallback_face_count += 1
            continue
        raw_alpha = float(numerator / float(denominator))
        clipped_alpha = float(np.clip(raw_alpha, float(min_alpha_value), float(max_alpha)))
        reliability = 1.0
        if float(shrink_count_tau) > 0.0:
            reliability *= float(count) / (float(count) + float(shrink_count_tau))
        if float(shrink_denominator_tau) > 0.0:
            reliability *= float(denominator) / (float(denominator) + float(shrink_denominator_tau))
        reliability = float(np.clip(reliability, 0.0, 1.0))
        prior_alpha = fallback_alpha if prior_mode == "fallback" else 0.0
        alpha = (
            float(np.clip(prior_alpha + reliability * (clipped_alpha - prior_alpha), float(min_alpha_value), float(max_alpha)))
            if shrink_enabled
            else clipped_alpha
        )
        face_alphas[str(int(face))] = float(alpha)
        fitted_rows.append(
            {
                "face_id": int(face),
                "samples": int(count),
                "numerator": float(numerator),
                "denominator": float(denominator),
                "raw_alpha": float(raw_alpha),
                "pre_shrink_alpha": float(clipped_alpha),
                "shrink_reliability": float(reliability),
                "alpha": float(alpha),
            }
        )
    fitted_rows.sort(key=lambda row: int(row["samples"]), reverse=True)
    generated = [float(x) for x in multiplier_grid if float(x) >= 0.0]
    if 1.0 not in generated:
        generated.append(1.0)
    combined = sorted(set(round(float(x), 8) for x in [*base, *generated, 0.0]))
    profile = {
        "enabled": True,
        "mode": "policy_val_face_alpha",
        "policy_val_views_used": int(view_count),
        "samples": int(sample_count),
        "face_alpha_count": int(len(face_alphas)),
        "fallback_face_count": int(fallback_face_count),
        "fallback_raw_alpha": float(fallback_raw_alpha),
        "fallback_alpha": float(fallback_alpha),
        "max_alpha": float(max_alpha),
        "min_alpha": float(min_alpha_value),
        "min_face_samples": int(min_face_samples),
        "shrink": {
            "enabled": bool(shrink_enabled),
            "count_tau": float(shrink_count_tau),
            "denominator_tau": float(shrink_denominator_tau),
            "prior": str(prior_mode),
        },
        "multiplier_grid": sorted(set(float(x) for x in generated)),
        "alpha_grid": combined,
        "face_alpha_preview": fitted_rows[:20],
        "face_alphas": face_alphas,
    }
    return combined, profile


def calibrated_bin_alpha_profile_from_policy_val(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    base_alpha_grid: list[float],
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_samples_per_view: int,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    max_alpha: float,
    min_alpha_value: float,
    multiplier_grid: list[float],
    min_bin_samples: int,
    min_denominator: float,
    min_positive_view_fraction: float,
    shrink_count_tau: float,
    shrink_denominator_tau: float,
    shrink_prior: str,
    max_profile_bins: int,
) -> tuple[list[float], dict[str, Any]]:
    """Fit a train-policy-val alpha per face/UV bin without held-out GT.

    v61/v62 showed that shrinking the apply mask alone is too brittle.  This
    profile keeps the same train-only interface but changes the operation from
    binary acceptance to magnitude calibration: every sufficiently supported
    face/UV bin receives a reliability-shrunk least-squares residual alpha, and
    the rest fall back to a global policy-val alpha.  The scalar alpha grid is
    still a global multiplier selected by the normal policy-val risk gates.
    """
    base = sorted(set(float(x) for x in base_alpha_grid))
    disabled = {
        "enabled": False,
        "mode": "policy_val_bin_alpha",
        "reason": "",
        "alpha_grid": base,
    }
    if not val_views or not atlas:
        disabled["reason"] = "no_policy_val_views_or_empty_atlas"
        return base, disabled
    rng = np.random.default_rng(63)
    texture_size = int(next(iter(atlas.values())).texture.shape[0])
    numerator_by_key: dict[tuple[int, int], float] = {}
    denominator_by_key: dict[tuple[int, int], float] = {}
    before_by_key: dict[tuple[int, int], float] = {}
    samples_by_key: dict[tuple[int, int], int] = {}
    view_stats_by_key: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    numerator_all = 0.0
    denominator_all = 0.0
    sample_count = 0
    view_count = 0
    for path in tqdm(val_views, desc="calibrate bin alpha"):
        z = np.load(path)
        if residual_rgb_key not in z or "barycentric" not in z:
            continue
        mask = _valid_sample_mask(z, set(atlas.keys()), residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            mask = np.zeros_like(mask, dtype=bool)
            mask[ys, xs] = True
        target_rgb = np.asarray(z[residual_rgb_key], dtype=np.float32)
        target = np.stack([target_rgb[0][mask], target_rgb[1][mask], target_rgb[2][mask]], axis=1)
        pred, _valid = predict_delta_for_npz(
            z,
            atlas,
            1.0,
            min_alpha,
            min_atlas_bin_count=int(min_atlas_bin_count),
            min_atlas_face_samples=int(min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(atlas_confidence_mode),
            atlas_confidence_count_scale=float(atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(min_atlas_confidence),
        )
        pred_samples = np.stack([pred[0][mask], pred[1][mask], pred[2][mask]], axis=1).astype(np.float32)
        nonzero = np.linalg.norm(pred_samples, axis=1) > 0.0
        if not bool(np.any(nonzero)):
            continue
        target = target.astype(np.float64)[nonzero]
        pred_samples = pred_samples.astype(np.float64)[nonzero]
        face_samples = np.asarray(z["face_id"], dtype=np.int64)[mask][nonzero]
        uv_u, uv_v = _uv_bins(np.asarray(z["barycentric"], dtype=np.float32), mask, texture_size)
        bin_samples = ((uv_v.astype(np.int64) * texture_size) + uv_u.astype(np.int64))[nonzero]
        sample_count += int(pred_samples.shape[0])
        view_count += 1
        numerator_all += float(np.sum(pred_samples * target))
        denominator_all += float(np.sum(np.square(pred_samples)))
        before_all = np.sum(target * target, axis=1)
        numerator_sample = np.sum(pred_samples * target, axis=1)
        denominator_sample = np.sum(pred_samples * pred_samples, axis=1)
        for face in np.unique(face_samples):
            face_i = int(face)
            fm = face_samples == face_i
            for bin_id in np.unique(bin_samples[fm]):
                key = (face_i, int(bin_id))
                bm = fm & (bin_samples == int(bin_id))
                numerator = float(np.sum(numerator_sample[bm]))
                denominator = float(np.sum(denominator_sample[bm]))
                before = float(np.sum(before_all[bm]))
                count = int(np.sum(bm))
                numerator_by_key[key] = numerator_by_key.get(key, 0.0) + numerator
                denominator_by_key[key] = denominator_by_key.get(key, 0.0) + denominator
                before_by_key[key] = before_by_key.get(key, 0.0) + before
                samples_by_key[key] = samples_by_key.get(key, 0) + count
                view_stats_by_key.setdefault(key, []).append((numerator, denominator, before))
    if denominator_all <= float(min_denominator):
        disabled["reason"] = "small_denominator"
        disabled["samples"] = int(sample_count)
        disabled["denominator"] = float(denominator_all)
        return base, disabled
    fallback_raw_alpha = float(numerator_all / denominator_all)
    fallback_alpha = float(np.clip(fallback_raw_alpha, float(min_alpha_value), float(max_alpha)))
    prior_mode = str(shrink_prior)
    if prior_mode not in {"fallback", "zero"}:
        raise ValueError(f"unsupported bin alpha shrink prior: {prior_mode}")
    shrink_enabled = float(shrink_count_tau) > 0.0 or float(shrink_denominator_tau) > 0.0
    candidate_rows: list[dict[str, Any]] = []
    fallback_bin_count = 0
    for key, denominator in denominator_by_key.items():
        face_i, bin_id = key
        count = int(samples_by_key.get(key, 0))
        numerator = float(numerator_by_key.get(key, 0.0))
        before = float(before_by_key.get(key, 0.0))
        if count < int(min_bin_samples) or float(denominator) <= float(min_denominator):
            fallback_bin_count += 1
            continue
        raw_alpha = float(numerator / float(denominator))
        clipped_alpha = float(np.clip(raw_alpha, float(min_alpha_value), float(max_alpha)))
        reliability = 1.0
        if float(shrink_count_tau) > 0.0:
            reliability *= float(count) / (float(count) + float(shrink_count_tau))
        if float(shrink_denominator_tau) > 0.0:
            reliability *= float(denominator) / (float(denominator) + float(shrink_denominator_tau))
        reliability = float(np.clip(reliability, 0.0, 1.0))
        prior_alpha = fallback_alpha if prior_mode == "fallback" else 0.0
        alpha = (
            float(np.clip(prior_alpha + reliability * (clipped_alpha - prior_alpha), float(min_alpha_value), float(max_alpha)))
            if shrink_enabled
            else clipped_alpha
        )
        view_rows = view_stats_by_key.get(key, [])
        positive_views = 0
        for view_numerator, view_denominator, view_before in view_rows:
            view_after = float(view_before) - 2.0 * alpha * float(view_numerator) + alpha * alpha * float(view_denominator)
            view_gain = (float(view_before) - view_after) / max(float(view_before), 1.0e-12)
            if view_gain > 0.0:
                positive_views += 1
        positive_fraction = float(positive_views / max(1, len(view_rows)))
        if positive_fraction < float(min_positive_view_fraction):
            fallback_bin_count += 1
            continue
        after = before - 2.0 * alpha * numerator + alpha * alpha * float(denominator)
        relative_gain = (before - after) / max(before, 1.0e-12)
        candidate_rows.append(
            {
                "face_id": int(face_i),
                "bin_id": int(bin_id),
                "u": int(bin_id) % texture_size,
                "v": int(bin_id) // texture_size,
                "samples": int(count),
                "view_count": int(len(view_rows)),
                "positive_view_count": int(positive_views),
                "positive_view_fraction": float(positive_fraction),
                "numerator": float(numerator),
                "denominator": float(denominator),
                "mse_before_sum": float(before),
                "raw_alpha": float(raw_alpha),
                "pre_shrink_alpha": float(clipped_alpha),
                "shrink_reliability": float(reliability),
                "alpha": float(alpha),
                "relative_gain": float(relative_gain),
            }
        )
    candidate_rows.sort(
        key=lambda row: (
            float(row["relative_gain"]),
            float(row["positive_view_fraction"]),
            int(row["samples"]),
            float(row["denominator"]),
        ),
        reverse=True,
    )
    if int(max_profile_bins) > 0:
        selected_rows = candidate_rows[: int(max_profile_bins)]
    else:
        selected_rows = candidate_rows
    bin_alphas_by_face: dict[str, dict[str, float]] = {}
    for row in selected_rows:
        bin_alphas_by_face.setdefault(str(int(row["face_id"])), {})[str(int(row["bin_id"]))] = float(row["alpha"])
    generated = [float(x) for x in multiplier_grid if float(x) >= 0.0]
    if 1.0 not in generated:
        generated.append(1.0)
    combined = sorted(set(round(float(x), 8) for x in [*base, *generated, 0.0]))
    profile = {
        "enabled": True,
        "mode": "policy_val_bin_alpha",
        "policy_val_views_used": int(view_count),
        "samples": int(sample_count),
        "texture_size": int(texture_size),
        "candidate_bin_count": int(len(candidate_rows)),
        "bin_alpha_count": int(len(selected_rows)),
        "fallback_bin_count": int(fallback_bin_count + max(0, len(candidate_rows) - len(selected_rows))),
        "fallback_raw_alpha": float(fallback_raw_alpha),
        "fallback_alpha": float(fallback_alpha),
        "max_alpha": float(max_alpha),
        "min_alpha": float(min_alpha_value),
        "min_bin_samples": int(min_bin_samples),
        "min_positive_view_fraction": float(min_positive_view_fraction),
        "max_profile_bins": int(max_profile_bins),
        "shrink": {
            "enabled": bool(shrink_enabled),
            "count_tau": float(shrink_count_tau),
            "denominator_tau": float(shrink_denominator_tau),
            "prior": str(prior_mode),
        },
        "multiplier_grid": sorted(set(float(x) for x in generated)),
        "alpha_grid": combined,
        "bin_alpha_preview": selected_rows[:32],
        "bin_alphas_by_face": bin_alphas_by_face,
    }
    return combined, profile


def calibrated_image_l1_bin_alpha_profile_from_policy_val(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    base_alpha_grid: list[float],
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_samples_per_view: int,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    local_alpha_grid: list[float],
    max_alpha: float,
    min_bin_samples: int,
    min_relative_gain: float,
    min_positive_view_fraction: float,
    count_tau: float,
    fallback_mode: str,
    max_profile_bins: int,
) -> tuple[list[float], dict[str, Any]]:
    """Fit per-bin alpha by directly minimizing held-out train image L1.

    The older bin-alpha calibration optimizes residual-vector MSE.  This profile
    instead evaluates the actual clipped image-space adaptation
    ``clip(rgb_render + alpha * predicted_delta)`` on policy-val views and stores
    only bins that improve image L1.  Target/test GT is never read by this step.
    """
    base = sorted(set(float(x) for x in base_alpha_grid))
    disabled = {
        "enabled": False,
        "mode": "policy_val_bin_alpha",
        "optimizer": "policy_val_image_l1_grid",
        "reason": "",
        "alpha_grid": base,
    }
    if not val_views or not atlas:
        disabled["reason"] = "no_policy_val_views_or_empty_atlas"
        return base, disabled

    max_alpha_f = max(0.0, float(max_alpha))
    local_grid = sorted(
        set(
            round(float(x), 8)
            for x in local_alpha_grid
            if 0.0 <= float(x) <= max_alpha_f
        )
    )
    if 0.0 not in local_grid:
        local_grid.insert(0, 0.0)
    if not local_grid:
        disabled["reason"] = "empty_local_alpha_grid"
        return base, disabled
    fallback_mode_s = str(fallback_mode)
    if fallback_mode_s not in {"zero", "global_best"}:
        raise ValueError(f"unsupported image-L1 bin alpha fallback mode: {fallback_mode_s}")

    rng = np.random.default_rng(141)
    texture_size = int(next(iter(atlas.values())).texture.shape[0])
    alpha_values = np.asarray(local_grid, dtype=np.float32)
    before_by_key: dict[tuple[int, int], float] = {}
    after_by_key: dict[tuple[int, int], np.ndarray] = {}
    samples_by_key: dict[tuple[int, int], int] = {}
    view_stats_by_key: dict[tuple[int, int], list[tuple[float, np.ndarray]]] = {}
    before_all = 0.0
    after_all = np.zeros((int(alpha_values.size),), dtype=np.float64)
    sample_count = 0
    view_count = 0
    skipped_missing_image_gt = 0
    skipped_no_valid_delta = 0

    for path in tqdm(val_views, desc="calibrate image-L1 bin alpha"):
        z = np.load(path)
        if "rgb_render" not in z or "rgb_gt" not in z or "barycentric" not in z:
            skipped_missing_image_gt += 1
            continue
        mask = _valid_sample_mask(z, set(atlas.keys()), residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            mask = np.zeros_like(mask, dtype=bool)
            mask[ys, xs] = True
        pred, _valid = predict_delta_for_npz(
            z,
            atlas,
            1.0,
            min_alpha,
            min_atlas_bin_count=int(min_atlas_bin_count),
            min_atlas_face_samples=int(min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(atlas_confidence_mode),
            atlas_confidence_count_scale=float(atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(min_atlas_confidence),
        )
        pred_samples = np.stack([pred[0][mask], pred[1][mask], pred[2][mask]], axis=1).astype(np.float32)
        nonzero = np.linalg.norm(pred_samples, axis=1) > 0.0
        if not bool(np.any(nonzero)):
            skipped_no_valid_delta += 1
            continue
        render = np.asarray(z["rgb_render"], dtype=np.float32)
        gt = np.asarray(z["rgb_gt"], dtype=np.float32)
        render_samples = np.stack([render[0][mask], render[1][mask], render[2][mask]], axis=1).astype(np.float32)
        gt_samples = np.stack([gt[0][mask], gt[1][mask], gt[2][mask]], axis=1).astype(np.float32)
        render_samples = render_samples[nonzero]
        gt_samples = gt_samples[nonzero]
        pred_samples = pred_samples[nonzero]
        before_sample = np.sum(np.abs(render_samples - gt_samples), axis=1).astype(np.float64)
        adapted = np.clip(
            render_samples[None, :, :] + alpha_values[:, None, None] * pred_samples[None, :, :],
            0.0,
            1.0,
        )
        after_sample_by_alpha = np.sum(np.abs(adapted - gt_samples[None, :, :]), axis=2).astype(np.float64)
        face_samples = np.asarray(z["face_id"], dtype=np.int64)[mask][nonzero]
        uv_u, uv_v = _uv_bins(np.asarray(z["barycentric"], dtype=np.float32), mask, texture_size)
        bin_samples = ((uv_v.astype(np.int64) * texture_size) + uv_u.astype(np.int64))[nonzero]

        sample_count += int(pred_samples.shape[0])
        view_count += 1
        before_all += float(np.sum(before_sample))
        after_all += np.sum(after_sample_by_alpha, axis=1)

        for face in np.unique(face_samples):
            face_i = int(face)
            fm = face_samples == face_i
            for bin_id in np.unique(bin_samples[fm]):
                key = (face_i, int(bin_id))
                bm = fm & (bin_samples == int(bin_id))
                before = float(np.sum(before_sample[bm]))
                after_vec = np.sum(after_sample_by_alpha[:, bm], axis=1).astype(np.float64)
                count = int(np.sum(bm))
                before_by_key[key] = before_by_key.get(key, 0.0) + before
                if key not in after_by_key:
                    after_by_key[key] = np.zeros_like(after_vec, dtype=np.float64)
                after_by_key[key] += after_vec
                samples_by_key[key] = samples_by_key.get(key, 0) + count
                view_stats_by_key.setdefault(key, []).append((before, after_vec))

    if sample_count <= 0 or before_all <= 1.0e-12:
        disabled["reason"] = "no_policy_val_image_l1_samples"
        disabled["samples"] = int(sample_count)
        disabled["policy_val_views_used"] = int(view_count)
        disabled["policy_val_views_missing_image_gt"] = int(skipped_missing_image_gt)
        disabled["policy_val_views_without_valid_delta"] = int(skipped_no_valid_delta)
        return base, disabled

    global_best_index = int(np.argmin(after_all))
    global_best_alpha = float(alpha_values[global_best_index])
    global_best_after = float(after_all[global_best_index])
    global_relative_gain = float((before_all - global_best_after) / max(before_all, 1.0e-12))
    fallback_alpha = 0.0 if fallback_mode_s == "zero" else global_best_alpha

    candidate_rows: list[dict[str, Any]] = []
    rejected_low_support = 0
    rejected_nonpositive = 0
    rejected_view_consistency = 0
    for key, before in before_by_key.items():
        face_i, bin_id = key
        count = int(samples_by_key.get(key, 0))
        if count < int(min_bin_samples):
            rejected_low_support += 1
            continue
        after_vec = after_by_key[key]
        best_index = int(np.argmin(after_vec))
        alpha = float(alpha_values[best_index])
        after = float(after_vec[best_index])
        relative_gain = float((float(before) - after) / max(float(before), 1.0e-12))
        if alpha <= 0.0 or relative_gain < float(min_relative_gain) or relative_gain <= 0.0:
            rejected_nonpositive += 1
            continue
        view_rows = view_stats_by_key.get(key, [])
        positive_views = 0
        view_gain_sum = 0.0
        for view_before, view_after_vec in view_rows:
            view_after = float(view_after_vec[best_index])
            view_gain = float((float(view_before) - view_after) / max(float(view_before), 1.0e-12))
            view_gain_sum += view_gain
            if view_gain > 0.0:
                positive_views += 1
        positive_fraction = float(positive_views / max(1, len(view_rows)))
        if positive_fraction < float(min_positive_view_fraction):
            rejected_view_consistency += 1
            continue
        reliability = 1.0
        if float(count_tau) > 0.0:
            reliability = float(count) / (float(count) + float(count_tau))
        score = float(relative_gain * reliability)
        candidate_rows.append(
            {
                "face_id": int(face_i),
                "bin_id": int(bin_id),
                "u": int(bin_id) % texture_size,
                "v": int(bin_id) // texture_size,
                "samples": int(count),
                "view_count": int(len(view_rows)),
                "positive_view_count": int(positive_views),
                "positive_view_fraction": float(positive_fraction),
                "mean_view_relative_gain": float(view_gain_sum / max(1, len(view_rows))),
                "image_l1_before_sum": float(before),
                "image_l1_after_sum": float(after),
                "alpha": float(alpha),
                "relative_gain": float(relative_gain),
                "score": float(score),
                "count_reliability": float(reliability),
            }
        )

    candidate_rows.sort(
        key=lambda row: (
            float(row["score"]),
            float(row["relative_gain"]),
            float(row["positive_view_fraction"]),
            int(row["samples"]),
        ),
        reverse=True,
    )
    selected_rows = candidate_rows[: int(max_profile_bins)] if int(max_profile_bins) > 0 else candidate_rows
    bin_alphas_by_face: dict[str, dict[str, float]] = {}
    for row in selected_rows:
        bin_alphas_by_face.setdefault(str(int(row["face_id"])), {})[str(int(row["bin_id"]))] = float(row["alpha"])

    combined = sorted(set(round(float(x), 8) for x in [*base, 0.0, 0.5, 1.0]))
    selected_gain_sum = float(sum(float(row["relative_gain"]) for row in selected_rows))
    profile = {
        "enabled": True,
        "mode": "policy_val_bin_alpha",
        "optimizer": "policy_val_image_l1_grid",
        "uses_policy_val_gt": True,
        "uses_target_or_test_gt": False,
        "policy_val_views_used": int(view_count),
        "policy_val_views_missing_image_gt": int(skipped_missing_image_gt),
        "policy_val_views_without_valid_delta": int(skipped_no_valid_delta),
        "samples": int(sample_count),
        "texture_size": int(texture_size),
        "candidate_bin_count": int(len(candidate_rows)),
        "bin_alpha_count": int(len(selected_rows)),
        "fallback_bin_count": int(
            max(0, len(before_by_key) - len(candidate_rows))
            + max(0, len(candidate_rows) - len(selected_rows))
        ),
        "fallback_alpha": float(fallback_alpha),
        "fallback_mode": str(fallback_mode_s),
        "max_alpha": float(max_alpha_f),
        "min_alpha": 0.0,
        "min_bin_samples": int(min_bin_samples),
        "min_relative_gain": float(min_relative_gain),
        "min_positive_view_fraction": float(min_positive_view_fraction),
        "count_tau": float(count_tau),
        "max_profile_bins": int(max_profile_bins),
        "local_alpha_grid": [float(x) for x in alpha_values.tolist()],
        "global_best_alpha": float(global_best_alpha),
        "global_image_l1_before_sum": float(before_all),
        "global_image_l1_after_sum": float(global_best_after),
        "global_relative_gain": float(global_relative_gain),
        "mean_selected_relative_gain": (
            float(selected_gain_sum / max(1, len(selected_rows))) if selected_rows else 0.0
        ),
        "rejected_low_support_bins": int(rejected_low_support),
        "rejected_nonpositive_bins": int(rejected_nonpositive),
        "rejected_view_consistency_bins": int(rejected_view_consistency),
        "alpha_grid": combined,
        "bin_alpha_preview": selected_rows[:32],
        "bin_alphas_by_face": bin_alphas_by_face,
    }
    return combined, profile


def calibrated_image_linear_generator_profile_from_policy_val(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    base_alpha_grid: list[float],
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_samples_per_view: int,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    feature_mode: str,
    ridge: float,
    train_max_samples_per_view: int,
    max_train_samples: int,
    generator_output_cap: float,
    alpha_grid: list[float],
    require_base_valid: bool,
    loss_mode: str,
    irls_iterations: int,
    huber_delta: float,
    training_sample_policy: str,
    min_descent_margin: float,
    min_training_samples: int,
    expert_mode: str = "none",
    expert_min_training_samples: int = 2048,
    expert_shrink_tau: float = 8192.0,
    face_reliability_mode: str = "none",
    face_reliability_min_face_samples: int = 256,
    face_reliability_min_relative_gain: float = 0.0,
    face_reliability_min_positive_view_fraction: float = 0.5,
    face_reliability_fallback_multiplier: float = 0.0,
) -> tuple[list[float], dict[str, Any]]:
    """Fit a policy-val ridge generator for image-space residuals.

    The fitted model predicts ``rgb_gt - rgb_render`` from target-available
    features: atlas residual, parent render color, barycentric coordinates, and
    optional camera direction.  It can be applied to target/test views without
    reading their GT; the normal policy-val risk gate still decides whether any
    nonzero global alpha is safe.
    """
    base = sorted(set(float(x) for x in base_alpha_grid))
    feature_mode_s = str(feature_mode)
    feature_names = _image_linear_generator_feature_names(feature_mode_s)
    disabled = {
        "enabled": False,
        "mode": "policy_val_image_linear_generator",
        "optimizer": "ridge_image_residual",
        "reason": "",
        "uses_policy_val_gt": True,
        "uses_target_or_test_gt": False,
        "feature_mode": feature_mode_s,
        "feature_names": list(feature_names),
        "ridge": float(ridge),
        "loss_mode": str(loss_mode),
        "training_sample_policy": str(training_sample_policy),
        "expert_mode": str(expert_mode),
        "face_reliability_mode": str(face_reliability_mode),
        "alpha_grid": base,
    }
    if not val_views or not atlas:
        disabled["reason"] = "no_policy_val_views_or_empty_atlas"
        return base, disabled
    if float(ridge) < 0.0:
        raise ValueError("image linear generator ridge must be >= 0")
    loss_mode_s = str(loss_mode)
    if loss_mode_s not in {"mse", "huber_irls", "l1_irls"}:
        raise ValueError(f"unsupported image linear generator loss mode: {loss_mode_s}")
    training_policy_s = str(training_sample_policy)
    if training_policy_s not in {"all", "base_l1_descent", "view_balanced", "view_balanced_base_l1_descent"}:
        raise ValueError(f"unsupported image linear generator training sample policy: {training_policy_s}")
    descent_filter_enabled = training_policy_s in {"base_l1_descent", "view_balanced_base_l1_descent"}
    view_balance_enabled = training_policy_s in {"view_balanced", "view_balanced_base_l1_descent"}
    expert_mode_s = str(expert_mode)
    if expert_mode_s not in {"none", "view_cluster"}:
        raise ValueError(f"unsupported image linear generator expert mode: {expert_mode_s}")
    face_reliability_mode_s = str(face_reliability_mode)
    if face_reliability_mode_s not in {"none", "global", "view_cluster"}:
        raise ValueError(f"unsupported image linear generator face reliability mode: {face_reliability_mode_s}")
    if face_reliability_mode_s == "view_cluster" and expert_mode_s != "view_cluster":
        raise ValueError("view-cluster face reliability requires view-cluster image-linear generator experts")
    face_reliability_min_face_samples_i = max(1, int(face_reliability_min_face_samples))
    face_reliability_min_relative_gain_f = float(face_reliability_min_relative_gain)
    face_reliability_min_positive_view_fraction_f = float(
        np.clip(face_reliability_min_positive_view_fraction, 0.0, 1.0)
    )
    face_reliability_fallback_multiplier_f = float(
        np.clip(face_reliability_fallback_multiplier, 0.0, 1.0)
    )
    expert_min_training_samples_i = max(1, int(expert_min_training_samples))
    expert_shrink_tau_f = max(0.0, float(expert_shrink_tau))
    expert_centers: np.ndarray | None = None
    expert_feature_mode = "none"
    if expert_mode_s == "view_cluster":
        expert_centers, expert_feature_mode = _view_cluster_profile_from_atlas(atlas)
        if expert_centers is None or str(expert_feature_mode) == "none":
            expert_mode_s = "none"

    requested_grid = sorted(set(round(float(x), 8) for x in alpha_grid if float(x) >= 0.0))
    combined = sorted(set(round(float(x), 8) for x in [*base, *requested_grid, 0.0]))
    rng = np.random.default_rng(144)
    feature_dim = int(len(feature_names))
    x_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    base_chunks: list[np.ndarray] = []
    face_id_chunks: list[np.ndarray] = []
    view_id_chunks: list[np.ndarray] = []
    cluster_id_chunks: list[np.ndarray] = []
    sample_count = 0
    raw_sample_count = 0
    rejected_by_training_policy = 0
    view_count = 0
    skipped_missing_image_gt = 0
    skipped_missing_base_support = 0
    capped_by_global_sample_limit = False
    per_view_cap = int(train_max_samples_per_view)
    global_cap = int(max_train_samples)

    for path in tqdm(val_views, desc="fit image-linear residual generator"):
        if global_cap > 0 and sample_count >= global_cap:
            capped_by_global_sample_limit = True
            break
        with np.load(path) as z:
            if "rgb_render" not in z or "rgb_gt" not in z or "barycentric" not in z:
                skipped_missing_image_gt += 1
                continue
            render = _as_rgb_chw(np.asarray(z["rgb_render"], dtype=np.float32))
            gt = _as_rgb_chw(np.asarray(z["rgb_gt"], dtype=np.float32))
            if render is None or gt is None:
                skipped_missing_image_gt += 1
                continue
            mask = _valid_sample_mask(z, set(atlas.keys()), residual_l1_key, min_l1, min_alpha)
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                skipped_missing_base_support += 1
                continue
            if per_view_cap > 0 and ys.size > per_view_cap:
                take = rng.choice(ys.size, size=per_view_cap, replace=False)
                ys = ys[take]
                xs = xs[take]
                mask = np.zeros_like(mask, dtype=bool)
                mask[ys, xs] = True
            pred, valid = predict_delta_for_npz(
                z,
                atlas,
                1.0,
                min_alpha,
                min_atlas_bin_count=int(min_atlas_bin_count),
                min_atlas_face_samples=int(min_atlas_face_samples),
                max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
                min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
                atlas_confidence_mode=str(atlas_confidence_mode),
                atlas_confidence_count_scale=float(atlas_confidence_count_scale),
                atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
                atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
                atlas_confidence_sign_power=float(atlas_confidence_sign_power),
                atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
                min_atlas_confidence=float(min_atlas_confidence),
            )
            if bool(require_base_valid):
                mask = mask & np.asarray(valid, dtype=bool)
                ys, xs = np.nonzero(mask)
            if ys.size == 0:
                skipped_missing_base_support += 1
                continue
            if global_cap > 0 and sample_count + int(ys.size) > global_cap:
                remaining = max(0, global_cap - sample_count)
                if remaining <= 0:
                    capped_by_global_sample_limit = True
                    break
                take = rng.choice(ys.size, size=remaining, replace=False)
                ys = ys[take]
                xs = xs[take]
                mask = np.zeros_like(mask, dtype=bool)
                mask[ys, xs] = True
                capped_by_global_sample_limit = True
            base_pixels = np.stack([pred[0][mask], pred[1][mask], pred[2][mask]], axis=1).astype(np.float32)
            face_samples = np.asarray(z["face_id"], dtype=np.int64)[mask].reshape(-1)
            bary = np.asarray(z["barycentric"], dtype=np.float32)
            feature_cache: dict[str, Any] = {}
            features = _image_linear_generator_features_for_samples(
                z,
                base_pixels=base_pixels,
                ys=ys,
                xs=xs,
                bary=bary,
                feature_mode=feature_mode_s,
                feature_cache=feature_cache,
            )
            target = np.stack(
                [
                    gt[0][ys, xs] - render[0][ys, xs],
                    gt[1][ys, xs] - render[1][ys, xs],
                    gt[2][ys, xs] - render[2][ys, xs],
                ],
                axis=1,
            ).astype(np.float32)
            if features.shape[0] == 0 or features.shape[1] != feature_dim:
                continue
            raw_sample_count += int(features.shape[0])
            if descent_filter_enabled:
                zero_l1_sample = np.sum(np.abs(target), axis=1)
                base_l1_sample = np.sum(np.abs(target - base_pixels), axis=1)
                keep = base_l1_sample + float(min_descent_margin) < zero_l1_sample
                rejected_by_training_policy += int(np.sum(~keep))
                if not bool(np.any(keep)):
                    continue
                features = features[keep]
                target = target[keep]
                base_pixels = base_pixels[keep]
                face_samples = face_samples[keep]
            current_view_id = int(view_count)
            cluster_id = -1
            if expert_mode_s == "view_cluster" and expert_centers is not None and str(expert_feature_mode) != "none":
                assigned = _assign_view_cluster_for_npz(
                    z,
                    centers=expert_centers,
                    feature_mode=str(expert_feature_mode),
                )
                cluster_id = int(assigned) if assigned is not None else -1
            x_chunks.append(features.astype(np.float32))
            target_chunks.append(target.astype(np.float32))
            base_chunks.append(base_pixels.astype(np.float32))
            face_id_chunks.append(face_samples.astype(np.int64))
            view_id_chunks.append(
                np.full((int(features.shape[0]),), current_view_id, dtype=np.int32)
            )
            cluster_id_chunks.append(
                np.full((int(features.shape[0]),), int(cluster_id), dtype=np.int32)
            )
            sample_count += int(features.shape[0])
            view_count += 1

    if sample_count <= 0 or not x_chunks:
        disabled["reason"] = "no_policy_val_generator_samples"
        disabled["samples"] = int(sample_count)
        disabled["raw_samples_before_training_policy"] = int(raw_sample_count)
        disabled["rejected_by_training_policy"] = int(rejected_by_training_policy)
        disabled["policy_val_views_used"] = int(view_count)
        disabled["policy_val_views_missing_image_gt"] = int(skipped_missing_image_gt)
        disabled["policy_val_views_without_base_support"] = int(skipped_missing_base_support)
        return combined, disabled
    if sample_count < int(min_training_samples):
        disabled["reason"] = "insufficient_policy_val_generator_samples_after_training_policy"
        disabled["samples"] = int(sample_count)
        disabled["raw_samples_before_training_policy"] = int(raw_sample_count)
        disabled["rejected_by_training_policy"] = int(rejected_by_training_policy)
        disabled["min_training_samples"] = int(min_training_samples)
        disabled["policy_val_views_used"] = int(view_count)
        return combined, disabled

    x_eval = np.concatenate(x_chunks, axis=0).astype(np.float32)
    target_eval = np.concatenate(target_chunks, axis=0).astype(np.float32)
    base_eval = np.concatenate(base_chunks, axis=0).astype(np.float32)
    face_ids_eval = np.concatenate(face_id_chunks, axis=0).astype(np.int64)
    view_ids_eval = np.concatenate(view_id_chunks, axis=0).astype(np.int32)
    cluster_ids_eval = np.concatenate(cluster_id_chunks, axis=0).astype(np.int32)
    x64 = x_eval.astype(np.float64)
    y64 = target_eval.astype(np.float64)
    reg = float(ridge) * np.eye(feature_dim, dtype=np.float64)
    if feature_dim > 0:
        reg[0, 0] = 0.0

    iterations = 1 if loss_mode_s == "mse" else max(1, int(irls_iterations))
    robust_delta = max(float(huber_delta), 1.0e-8)

    def solve_generator_weights(
        x_train64: np.ndarray,
        y_train64: np.ndarray,
        base_weights_1d: np.ndarray,
    ) -> tuple[np.ndarray, str, list[dict[str, float]]]:
        def solve_weighted(sample_weights: np.ndarray) -> tuple[np.ndarray, str]:
            weights_1d = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
            weights_1d = np.maximum(weights_1d, 1.0e-8)
            xtw = x_train64.T * weights_1d.reshape(1, -1)
            lhs = xtw @ x_train64 + reg
            rhs = xtw @ y_train64
            try:
                return np.linalg.solve(lhs, rhs), "solve"
            except np.linalg.LinAlgError:
                return np.linalg.pinv(lhs) @ rhs, "pinv"

        local_base_weights = np.asarray(base_weights_1d, dtype=np.float64).reshape(-1)
        if local_base_weights.size != int(x_train64.shape[0]):
            local_base_weights = np.ones((int(x_train64.shape[0]),), dtype=np.float64)
        local_base_weights = np.maximum(local_base_weights, 1.0e-8)
        local_base_weights = local_base_weights / max(float(np.mean(local_base_weights)), 1.0e-12)
        sample_weights = local_base_weights.astype(np.float64, copy=True)
        local_solver = "solve"
        local_history: list[dict[str, float]] = []
        local_weights = np.zeros((feature_dim, 3), dtype=np.float64)
        for iteration in range(iterations):
            local_weights, local_solver = solve_weighted(sample_weights)
            generated_iter = x_train64 @ local_weights
            residual = y_train64 - generated_iter
            sample_error = np.sqrt(np.mean(np.square(residual), axis=1))
            local_history.append(
                {
                    "iteration": float(iteration),
                    "mean_sample_error": float(np.mean(sample_error)),
                    "median_sample_error": float(np.median(sample_error)),
                    "mean_weight": float(np.mean(sample_weights)),
                    "min_weight": float(np.min(sample_weights)),
                    "max_weight": float(np.max(sample_weights)),
                }
            )
            if loss_mode_s == "mse":
                break
            if loss_mode_s == "huber_irls":
                robust_weights = np.minimum(1.0, robust_delta / np.maximum(sample_error, robust_delta))
            else:
                robust_weights = 1.0 / np.maximum(sample_error, robust_delta)
                robust_weights = robust_weights / max(float(np.mean(robust_weights)), 1.0e-8)
                robust_weights = np.clip(robust_weights, 0.05, 20.0)
            sample_weights = (local_base_weights * robust_weights).astype(np.float64)
            sample_weights = sample_weights / max(float(np.mean(sample_weights)), 1.0e-12)
        return local_weights, local_solver, local_history

    base_sample_weights = np.ones((int(x64.shape[0]),), dtype=np.float64)
    view_weight_profile: dict[str, Any] = {
        "enabled": bool(view_balance_enabled),
        "view_count": int(view_count),
        "sample_count": int(sample_count),
    }
    if view_balance_enabled and view_ids_eval.size > 0:
        view_counts = np.bincount(view_ids_eval, minlength=max(1, int(view_count))).astype(np.float64)
        nonzero = view_counts > 0.0
        per_view_weight = np.zeros_like(view_counts, dtype=np.float64)
        per_view_weight[nonzero] = 1.0 / view_counts[nonzero]
        base_sample_weights = per_view_weight[view_ids_eval]
        base_sample_weights = base_sample_weights / max(float(np.mean(base_sample_weights)), 1.0e-12)
        view_weight_profile.update(
            {
                "view_sample_counts": [int(x) for x in view_counts.astype(np.int64).tolist()],
                "min_sample_weight": float(np.min(base_sample_weights)),
                "max_sample_weight": float(np.max(base_sample_weights)),
                "mean_sample_weight": float(np.mean(base_sample_weights)),
            }
        )
    output_cap = float(generator_output_cap)
    weights, solver, irls_history = solve_generator_weights(x64, y64, base_sample_weights)
    generated = x64 @ weights
    expert_profile: dict[str, Any] = {
        "enabled": False,
        "mode": str(expert_mode_s),
        "feature_mode": str(expert_feature_mode),
        "expert_count": 0,
        "expert_min_training_samples": int(expert_min_training_samples_i),
        "expert_shrink_tau": float(expert_shrink_tau_f),
        "rows": [],
    }
    expert_weights_arr: np.ndarray | None = None
    expert_enabled_arr: np.ndarray | None = None
    if expert_mode_s == "view_cluster" and expert_centers is not None and str(expert_feature_mode) != "none":
        expert_count = int(np.asarray(expert_centers).shape[0])
        expert_weights_arr = np.repeat(weights[None, ...], expert_count, axis=0).astype(np.float64)
        expert_enabled_arr = np.zeros((expert_count,), dtype=bool)
        expert_rows: list[dict[str, Any]] = []
        for cluster_id in range(expert_count):
            cluster_mask = cluster_ids_eval == int(cluster_id)
            cluster_samples = int(np.sum(cluster_mask))
            row: dict[str, Any] = {
                "cluster_id": int(cluster_id),
                "samples": int(cluster_samples),
                "enabled": False,
                "reason": "",
            }
            if cluster_samples < int(expert_min_training_samples_i):
                row["reason"] = "insufficient_cluster_samples"
                expert_rows.append(row)
                continue
            local_weights, local_solver, local_history = solve_generator_weights(
                x64[cluster_mask],
                y64[cluster_mask],
                base_sample_weights[cluster_mask],
            )
            shrink = 1.0
            if expert_shrink_tau_f > 0.0:
                shrink = float(cluster_samples / max(cluster_samples + expert_shrink_tau_f, 1.0e-12))
            shrink = float(np.clip(shrink, 0.0, 1.0))
            routed_weights = ((1.0 - shrink) * weights + shrink * local_weights).astype(np.float64)
            expert_weights_arr[int(cluster_id)] = routed_weights
            expert_enabled_arr[int(cluster_id)] = True
            global_pred = x64[cluster_mask] @ weights
            local_pred = x64[cluster_mask] @ local_weights
            routed_pred = x64[cluster_mask] @ routed_weights
            row.update(
                {
                    "enabled": True,
                    "solver": str(local_solver),
                    "shrink_to_expert": float(shrink),
                    "global_mse": float(np.mean(np.square(y64[cluster_mask] - global_pred))),
                    "expert_raw_mse": float(np.mean(np.square(y64[cluster_mask] - local_pred))),
                    "expert_shrunk_mse": float(np.mean(np.square(y64[cluster_mask] - routed_pred))),
                    "global_l1": float(np.mean(np.abs(y64[cluster_mask] - global_pred))),
                    "expert_raw_l1": float(np.mean(np.abs(y64[cluster_mask] - local_pred))),
                    "expert_shrunk_l1": float(np.mean(np.abs(y64[cluster_mask] - routed_pred))),
                    "irls_history": local_history,
                }
            )
            generated[cluster_mask] = routed_pred
            expert_rows.append(row)
        expert_profile = {
            "enabled": bool(np.any(expert_enabled_arr)),
            "mode": "view_cluster",
            "feature_mode": str(expert_feature_mode),
            "expert_count": int(expert_count),
            "enabled_expert_count": int(np.sum(expert_enabled_arr)),
            "expert_min_training_samples": int(expert_min_training_samples_i),
            "expert_shrink_tau": float(expert_shrink_tau_f),
            "rows": expert_rows,
        }
    if output_cap > 0.0:
        generated = np.clip(generated, -output_cap, output_cap)
    generated = generated.astype(np.float32)
    face_reliability_profile: dict[str, Any] = {
        "enabled": False,
        "mode": str(face_reliability_mode_s),
        "reason": "not_requested" if face_reliability_mode_s == "none" else "",
        "uses_policy_val_gt": bool(face_reliability_mode_s != "none"),
        "uses_target_or_test_gt": False,
        "min_face_samples": int(face_reliability_min_face_samples_i),
        "min_relative_gain": float(face_reliability_min_relative_gain_f),
        "min_positive_view_fraction": float(face_reliability_min_positive_view_fraction_f),
        "fallback_multiplier": float(face_reliability_fallback_multiplier_f),
        "entries": [],
        "groups": [],
    }
    generated_before_face_reliability = generated.astype(np.float32, copy=True)
    if face_reliability_mode_s != "none":
        if face_reliability_mode_s == "view_cluster" and not (
            expert_profile.get("enabled", False) and np.any(cluster_ids_eval >= 0)
        ):
            face_reliability_profile["reason"] = "view_cluster_reliability_requested_without_enabled_experts"
        else:
            group_ids_eval = (
                cluster_ids_eval.astype(np.int32)
                if face_reliability_mode_s == "view_cluster"
                else np.full_like(view_ids_eval, -1, dtype=np.int32)
            )
            alpha_eval_grid = sorted(
                set(
                    float(x)
                    for x in combined
                    if float(x) > 0.0 and math.isfinite(float(x))
                )
            )
            if not alpha_eval_grid:
                alpha_eval_grid = [1.0]
            before_by_key: dict[tuple[int, int], float] = {}
            after_by_alpha_key: dict[tuple[float, int, int], float] = {}
            samples_by_key: dict[tuple[int, int], int] = {}
            before_by_view_key: dict[tuple[int, int, int], float] = {}
            after_by_alpha_view_key: dict[tuple[float, int, int, int], float] = {}
            before_samples = np.sum(np.square(target_eval.astype(np.float64)), axis=1)
            after_samples_by_alpha = {
                float(alpha): np.sum(
                    np.square(target_eval.astype(np.float64) - float(alpha) * generated.astype(np.float64)),
                    axis=1,
                )
                for alpha in alpha_eval_grid
            }
            for idx in range(int(face_ids_eval.shape[0])):
                group_id = int(group_ids_eval[idx])
                face = int(face_ids_eval[idx])
                view_id = int(view_ids_eval[idx])
                key = (group_id, face)
                view_key = (group_id, face, view_id)
                before_val = float(before_samples[idx])
                before_by_key[key] = before_by_key.get(key, 0.0) + before_val
                samples_by_key[key] = samples_by_key.get(key, 0) + 1
                before_by_view_key[view_key] = before_by_view_key.get(view_key, 0.0) + before_val
                for alpha, after_samples in after_samples_by_alpha.items():
                    after_val = float(after_samples[idx])
                    after_by_alpha_key[(float(alpha), group_id, face)] = (
                        after_by_alpha_key.get((float(alpha), group_id, face), 0.0) + after_val
                    )
                    after_by_alpha_view_key[(float(alpha), group_id, face, view_id)] = (
                        after_by_alpha_view_key.get((float(alpha), group_id, face, view_id), 0.0) + after_val
                    )

            rows: list[dict[str, Any]] = []
            entries: list[dict[str, Any]] = []
            entries_by_group: dict[str, dict[str, float]] = {}
            groups: dict[int, dict[str, Any]] = {}
            kept_sample_count = 0
            for key in sorted(before_by_key):
                group_id, face = int(key[0]), int(key[1])
                before_val = float(before_by_key[key])
                samples = int(samples_by_key.get(key, 0))
                best_alpha = float(alpha_eval_grid[0])
                best_after = float(after_by_alpha_key.get((best_alpha, group_id, face), float("inf")))
                best_relative_gain = (before_val - best_after) / max(before_val, 1.0e-12)
                for alpha in alpha_eval_grid[1:]:
                    candidate_after = float(after_by_alpha_key.get((float(alpha), group_id, face), float("inf")))
                    candidate_gain = (before_val - candidate_after) / max(before_val, 1.0e-12)
                    if candidate_gain > best_relative_gain:
                        best_alpha = float(alpha)
                        best_after = candidate_after
                        best_relative_gain = candidate_gain
                views = 0
                positive_views = 0
                for (view_group_id, view_face, view_id), view_before in before_by_view_key.items():
                    if int(view_group_id) != int(group_id) or int(view_face) != int(face):
                        continue
                    views += 1
                    view_after = float(after_by_alpha_view_key.get((best_alpha, group_id, face, int(view_id)), 0.0))
                    view_gain = (float(view_before) - view_after) / max(float(view_before), 1.0e-12)
                    if view_gain > 0.0:
                        positive_views += 1
                relative_gain = float(best_relative_gain)
                positive_fraction = float(positive_views / max(1, views))
                keep = bool(
                    samples >= int(face_reliability_min_face_samples_i)
                    and relative_gain >= float(face_reliability_min_relative_gain_f)
                    and positive_fraction >= float(face_reliability_min_positive_view_fraction_f)
                )
                multiplier = 1.0 if keep else float(face_reliability_fallback_multiplier_f)
                if keep:
                    kept_sample_count += int(samples)
                row = {
                    "group_id": int(group_id),
                    "face_id": int(face),
                    "samples": int(samples),
                    "view_count": int(views),
                    "positive_view_count": int(positive_views),
                    "positive_view_fraction": float(positive_fraction),
                    "mse_before_sum": float(before_val),
                    "mse_after_sum": float(best_after),
                    "relative_gain": float(relative_gain),
                    "selected_alpha": float(best_alpha),
                    "multiplier": float(multiplier),
                    "keep": bool(keep),
                }
                rows.append(row)
                group = groups.setdefault(
                    int(group_id),
                    {
                        "group_id": int(group_id),
                        "candidate_face_count": 0,
                        "kept_face_count": 0,
                        "candidate_sample_count": 0,
                        "kept_sample_count": 0,
                    },
                )
                group["candidate_face_count"] = int(group["candidate_face_count"]) + 1
                group["candidate_sample_count"] = int(group["candidate_sample_count"]) + int(samples)
                if keep:
                    group["kept_face_count"] = int(group["kept_face_count"]) + 1
                    group["kept_sample_count"] = int(group["kept_sample_count"]) + int(samples)
                if multiplier != float(face_reliability_fallback_multiplier_f):
                    entries.append(
                        {
                            "group_id": int(group_id),
                            "face_id": int(face),
                            "multiplier": float(multiplier),
                            "selected_alpha": float(best_alpha),
                        }
                    )
                    entries_by_group.setdefault(str(int(group_id)), {})[str(int(face))] = {
                        "multiplier": float(multiplier),
                        "selected_alpha": float(best_alpha),
                    }
            if entries or float(face_reliability_fallback_multiplier_f) < 1.0:
                multiplier_by_key = {
                    (int(entry["group_id"]), int(entry["face_id"])): float(entry["multiplier"])
                    for entry in entries
                }
                sample_multipliers = np.full(
                    (int(face_ids_eval.shape[0]),),
                    float(face_reliability_fallback_multiplier_f),
                    dtype=np.float32,
                )
                for idx in range(int(face_ids_eval.shape[0])):
                    key = (int(group_ids_eval[idx]), int(face_ids_eval[idx]))
                    if key in multiplier_by_key:
                        sample_multipliers[idx] = float(multiplier_by_key[key])
                generated = (generated * sample_multipliers.reshape(-1, 1)).astype(np.float32)
            group_rows: list[dict[str, Any]] = []
            for group in sorted(groups.values(), key=lambda row: int(row["group_id"])):
                candidate_samples = int(group.get("candidate_sample_count", 0))
                kept_samples = int(group.get("kept_sample_count", 0))
                candidate_faces = int(group.get("candidate_face_count", 0))
                kept_faces = int(group.get("kept_face_count", 0))
                group_rows.append(
                    dict(group)
                    | {
                        "kept_sample_fraction": float(kept_samples / max(1, candidate_samples)),
                        "kept_face_fraction": float(kept_faces / max(1, candidate_faces)),
                    }
                )
            rows_by_gain = sorted(rows, key=lambda row: float(row["relative_gain"]))
            rows_by_samples = sorted(rows, key=lambda row: int(row["samples"]), reverse=True)
            face_reliability_profile = {
                "enabled": True,
                "mode": str(face_reliability_mode_s),
                "uses_policy_val_gt": True,
                "uses_target_or_test_gt": False,
                "min_face_samples": int(face_reliability_min_face_samples_i),
                "min_relative_gain": float(face_reliability_min_relative_gain_f),
                "min_positive_view_fraction": float(face_reliability_min_positive_view_fraction_f),
                "fallback_multiplier": float(face_reliability_fallback_multiplier_f),
                "certification_independent": False,
                "certification_note": (
                    "face reliability is fit on train policy-val evidence and must be checked by a held-out "
                    "run before being used as a final certified claim"
                ),
                "alpha_selection_mode": "per_face_group_best_alpha_grid",
                "apply_alpha_mode": "cap_generator_by_selected_alpha_over_current_alpha",
                "alpha_eval_grid": [float(x) for x in alpha_eval_grid],
                "candidate_face_group_count": int(len(rows)),
                "kept_face_group_count": int(len(entries)),
                "kept_sample_fraction": float(kept_sample_count / max(1, int(face_ids_eval.shape[0]))),
                "entries": entries,
                "entries_by_group": entries_by_group,
                "groups": group_rows,
                "worst_rows": rows_by_gain[:32],
                "best_sampled_rows": rows_by_samples[:32],
            }
    zero_mse = float(np.mean(np.square(target_eval)))
    base_mse = float(np.mean(np.square(target_eval - base_eval)))
    generator_mse = float(np.mean(np.square(target_eval - generated)))
    zero_l1 = float(np.mean(np.abs(target_eval)))
    base_l1 = float(np.mean(np.abs(target_eval - base_eval)))
    generator_l1 = float(np.mean(np.abs(target_eval - generated)))
    per_view_rows: list[dict[str, Any]] = []
    for view_id in sorted(set(int(x) for x in view_ids_eval.tolist())):
        mask = view_ids_eval == int(view_id)
        if not bool(np.any(mask)):
            continue
        view_base_mse = float(np.mean(np.square(target_eval[mask] - base_eval[mask])))
        view_generator_mse = float(np.mean(np.square(target_eval[mask] - generated[mask])))
        view_base_l1 = float(np.mean(np.abs(target_eval[mask] - base_eval[mask])))
        view_generator_l1 = float(np.mean(np.abs(target_eval[mask] - generated[mask])))
        per_view_rows.append(
            {
                "view_id": int(view_id),
                "samples": int(np.sum(mask)),
                "base_mse": float(view_base_mse),
                "generator_mse": float(view_generator_mse),
                "generator_relative_gain_vs_base_mse": float(
                    (view_base_mse - view_generator_mse) / max(view_base_mse, 1.0e-12)
                ),
                "base_l1": float(view_base_l1),
                "generator_l1": float(view_generator_l1),
                "generator_relative_gain_vs_base_l1": float(
                    (view_base_l1 - view_generator_l1) / max(view_base_l1, 1.0e-12)
                ),
            }
        )
    per_view_mse_gain_fraction = float(
        np.mean([float(row["generator_relative_gain_vs_base_mse"]) > 0.0 for row in per_view_rows])
    ) if per_view_rows else 0.0
    per_view_l1_gain_fraction = float(
        np.mean([float(row["generator_relative_gain_vs_base_l1"]) > 0.0 for row in per_view_rows])
    ) if per_view_rows else 0.0
    profile = {
        "enabled": True,
        "mode": "policy_val_image_linear_generator",
        "optimizer": "ridge_image_residual",
        "uses_policy_val_gt": True,
        "uses_target_or_test_gt": False,
        "feature_mode": feature_mode_s,
        "feature_names": list(feature_names),
        "weights": weights.astype(float).tolist(),
        "expert_mode": str(expert_mode_s),
        "expert_profile": dict(expert_profile),
        "expert_weights": (
            expert_weights_arr.astype(float).tolist()
            if expert_weights_arr is not None and bool((expert_profile or {}).get("enabled", False))
            else []
        ),
        "expert_enabled": (
            expert_enabled_arr.astype(bool).tolist()
            if expert_enabled_arr is not None and bool((expert_profile or {}).get("enabled", False))
            else []
        ),
        "expert_centers": (
            np.asarray(expert_centers, dtype=np.float32).astype(float).tolist()
            if expert_centers is not None and bool((expert_profile or {}).get("enabled", False))
            else []
        ),
        "expert_feature_mode": str(expert_feature_mode if bool((expert_profile or {}).get("enabled", False)) else "none"),
        "face_reliability_mode": str(face_reliability_mode_s),
        "face_reliability_profile": dict(face_reliability_profile),
        "ridge": float(ridge),
        "loss_mode": str(loss_mode_s),
        "irls_iterations": int(iterations),
        "huber_delta": float(robust_delta),
        "irls_history": irls_history,
        "training_sample_policy": str(training_policy_s),
        "min_descent_margin": float(min_descent_margin),
        "min_training_samples": int(min_training_samples),
        "solver": str(solver),
        "generator_output_cap": float(output_cap),
        "require_base_valid": bool(require_base_valid),
        "policy_val_views_used": int(view_count),
        "policy_val_views_missing_image_gt": int(skipped_missing_image_gt),
        "policy_val_views_without_base_support": int(skipped_missing_base_support),
        "samples": int(sample_count),
        "raw_samples_before_training_policy": int(raw_sample_count),
        "rejected_by_training_policy": int(rejected_by_training_policy),
        "training_policy_keep_fraction": float(sample_count / max(1, raw_sample_count)),
        "view_balanced_training": dict(view_weight_profile),
        "max_samples_per_view": int(max_samples_per_view),
        "train_max_samples_per_view": int(train_max_samples_per_view),
        "max_train_samples": int(max_train_samples),
        "capped_by_global_sample_limit": bool(capped_by_global_sample_limit),
        "zero_mse": float(zero_mse),
        "base_mse": float(base_mse),
        "generator_pre_face_reliability_mse": float(
            np.mean(np.square(target_eval - generated_before_face_reliability))
        ),
        "generator_mse": float(generator_mse),
        "generator_relative_gain_vs_base_mse": float((base_mse - generator_mse) / max(base_mse, 1.0e-12)),
        "generator_relative_gain_vs_zero_mse": float((zero_mse - generator_mse) / max(zero_mse, 1.0e-12)),
        "zero_l1": float(zero_l1),
        "base_l1": float(base_l1),
        "generator_pre_face_reliability_l1": float(
            np.mean(np.abs(target_eval - generated_before_face_reliability))
        ),
        "generator_l1": float(generator_l1),
        "generator_relative_gain_vs_base_l1": float((base_l1 - generator_l1) / max(base_l1, 1.0e-12)),
        "generator_relative_gain_vs_zero_l1": float((zero_l1 - generator_l1) / max(zero_l1, 1.0e-12)),
        "per_view_training_mse_gain_fraction": float(per_view_mse_gain_fraction),
        "per_view_training_l1_gain_fraction": float(per_view_l1_gain_fraction),
        "per_view_training_rows_preview": per_view_rows[:32],
        "alpha_grid": combined,
    }
    return combined, profile


def calibrated_bin_rgb_alpha_profile_from_policy_val(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    base_alpha_grid: list[float],
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_samples_per_view: int,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    max_alpha: float,
    min_alpha_value: float,
    multiplier_grid: list[float],
    min_bin_samples: int,
    min_denominator: float,
    min_positive_view_fraction: float,
    shrink_count_tau: float,
    shrink_denominator_tau: float,
    shrink_prior: str,
    max_profile_bins: int,
) -> tuple[list[float], dict[str, Any]]:
    """Fit per-channel train-policy-val alpha for each face/UV bin.

    v64's scalar bin alpha fixes residual magnitude but cannot correct bins
    whose RGB residual direction is biased. This keeps the same train-only
    evidence contract and learns an RGB shrink vector per reliable face/bin.
    Weak bins fall back to a global RGB shrink vector; the normal policy-val
    risk gate still chooses the global multiplier.
    """
    base = sorted(set(float(x) for x in base_alpha_grid))
    disabled = {
        "enabled": False,
        "mode": "policy_val_bin_rgb_alpha",
        "reason": "",
        "alpha_grid": base,
    }
    if not val_views or not atlas:
        disabled["reason"] = "no_policy_val_views_or_empty_atlas"
        return base, disabled
    rng = np.random.default_rng(66)
    texture_size = int(next(iter(atlas.values())).texture.shape[0])
    numerator_by_key: dict[tuple[int, int], np.ndarray] = {}
    denominator_by_key: dict[tuple[int, int], np.ndarray] = {}
    before_by_key: dict[tuple[int, int], float] = {}
    samples_by_key: dict[tuple[int, int], int] = {}
    view_stats_by_key: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, float]]] = {}
    numerator_all = np.zeros((3,), dtype=np.float64)
    denominator_all = np.zeros((3,), dtype=np.float64)
    sample_count = 0
    view_count = 0
    for path in tqdm(val_views, desc="calibrate bin RGB alpha"):
        z = np.load(path)
        if residual_rgb_key not in z or "barycentric" not in z:
            continue
        mask = _valid_sample_mask(z, set(atlas.keys()), residual_l1_key, min_l1, min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if max_samples_per_view > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            mask = np.zeros_like(mask, dtype=bool)
            mask[ys, xs] = True
        target_rgb = np.asarray(z[residual_rgb_key], dtype=np.float32)
        target = np.stack([target_rgb[0][mask], target_rgb[1][mask], target_rgb[2][mask]], axis=1)
        pred, _valid = predict_delta_for_npz(
            z,
            atlas,
            1.0,
            min_alpha,
            min_atlas_bin_count=int(min_atlas_bin_count),
            min_atlas_face_samples=int(min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(atlas_confidence_mode),
            atlas_confidence_count_scale=float(atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(min_atlas_confidence),
        )
        pred_samples = np.stack([pred[0][mask], pred[1][mask], pred[2][mask]], axis=1).astype(np.float32)
        nonzero = np.linalg.norm(pred_samples, axis=1) > 0.0
        if not bool(np.any(nonzero)):
            continue
        target = target.astype(np.float64)[nonzero]
        pred_samples = pred_samples.astype(np.float64)[nonzero]
        face_samples = np.asarray(z["face_id"], dtype=np.int64)[mask][nonzero]
        uv_u, uv_v = _uv_bins(np.asarray(z["barycentric"], dtype=np.float32), mask, texture_size)
        bin_samples = ((uv_v.astype(np.int64) * texture_size) + uv_u.astype(np.int64))[nonzero]
        sample_count += int(pred_samples.shape[0])
        view_count += 1
        numerator_all += np.sum(pred_samples * target, axis=0)
        denominator_all += np.sum(pred_samples * pred_samples, axis=0)
        before_sample = np.sum(target * target, axis=1)
        numerator_sample = pred_samples * target
        denominator_sample = pred_samples * pred_samples
        for face in np.unique(face_samples):
            face_i = int(face)
            fm = face_samples == face_i
            for bin_id in np.unique(bin_samples[fm]):
                key = (face_i, int(bin_id))
                bm = fm & (bin_samples == int(bin_id))
                numerator = np.sum(numerator_sample[bm], axis=0)
                denominator = np.sum(denominator_sample[bm], axis=0)
                before = float(np.sum(before_sample[bm]))
                count = int(np.sum(bm))
                if key not in numerator_by_key:
                    numerator_by_key[key] = np.zeros((3,), dtype=np.float64)
                    denominator_by_key[key] = np.zeros((3,), dtype=np.float64)
                numerator_by_key[key] += numerator
                denominator_by_key[key] += denominator
                before_by_key[key] = before_by_key.get(key, 0.0) + before
                samples_by_key[key] = samples_by_key.get(key, 0) + count
                view_stats_by_key.setdefault(key, []).append((numerator, denominator, before))
    if float(np.sum(denominator_all)) <= float(min_denominator):
        disabled["reason"] = "small_denominator"
        disabled["samples"] = int(sample_count)
        disabled["denominator"] = float(np.sum(denominator_all))
        return base, disabled
    fallback_raw_alpha = np.zeros((3,), dtype=np.float64)
    fallback_channel_valid = denominator_all > float(min_denominator)
    fallback_raw_alpha[fallback_channel_valid] = (
        numerator_all[fallback_channel_valid] / denominator_all[fallback_channel_valid]
    )
    fallback_alpha = np.clip(fallback_raw_alpha, float(min_alpha_value), float(max_alpha)).astype(np.float64)
    prior_mode = str(shrink_prior)
    if prior_mode not in {"fallback", "zero"}:
        raise ValueError(f"unsupported bin RGB alpha shrink prior: {prior_mode}")
    shrink_enabled = float(shrink_count_tau) > 0.0 or float(shrink_denominator_tau) > 0.0
    candidate_rows: list[dict[str, Any]] = []
    fallback_bin_count = 0
    for key, denominator in denominator_by_key.items():
        face_i, bin_id = key
        count = int(samples_by_key.get(key, 0))
        numerator = numerator_by_key.get(key, np.zeros((3,), dtype=np.float64))
        before = float(before_by_key.get(key, 0.0))
        denominator_sum = float(np.sum(denominator))
        if count < int(min_bin_samples) or denominator_sum <= float(min_denominator):
            fallback_bin_count += 1
            continue
        prior_mode = str(shrink_prior)
        prior_alpha = fallback_alpha if prior_mode == "fallback" else np.zeros((3,), dtype=np.float64)
        raw_alpha = np.array(prior_alpha, dtype=np.float64)
        channel_valid = denominator > float(min_denominator)
        raw_alpha[channel_valid] = numerator[channel_valid] / denominator[channel_valid]
        clipped_alpha = np.clip(raw_alpha, float(min_alpha_value), float(max_alpha)).astype(np.float64)
        reliability = 1.0
        if float(shrink_count_tau) > 0.0:
            reliability *= float(count) / (float(count) + float(shrink_count_tau))
        if float(shrink_denominator_tau) > 0.0:
            reliability *= denominator_sum / (denominator_sum + float(shrink_denominator_tau))
        reliability = float(np.clip(reliability, 0.0, 1.0))
        alpha = (
            np.clip(prior_alpha + reliability * (clipped_alpha - prior_alpha), float(min_alpha_value), float(max_alpha))
            if shrink_enabled
            else clipped_alpha
        )
        view_rows = view_stats_by_key.get(key, [])
        positive_views = 0
        for view_numerator, view_denominator, view_before in view_rows:
            view_after = float(view_before) - 2.0 * float(np.sum(alpha * view_numerator)) + float(
                np.sum(alpha * alpha * view_denominator)
            )
            view_gain = (float(view_before) - view_after) / max(float(view_before), 1.0e-12)
            if view_gain > 0.0:
                positive_views += 1
        positive_fraction = float(positive_views / max(1, len(view_rows)))
        if positive_fraction < float(min_positive_view_fraction):
            fallback_bin_count += 1
            continue
        after = before - 2.0 * float(np.sum(alpha * numerator)) + float(np.sum(alpha * alpha * denominator))
        relative_gain = (before - after) / max(before, 1.0e-12)
        candidate_rows.append(
            {
                "face_id": int(face_i),
                "bin_id": int(bin_id),
                "u": int(bin_id) % texture_size,
                "v": int(bin_id) // texture_size,
                "samples": int(count),
                "view_count": int(len(view_rows)),
                "positive_view_count": int(positive_views),
                "positive_view_fraction": float(positive_fraction),
                "denominator_sum": float(denominator_sum),
                "mse_before_sum": float(before),
                "raw_alpha": [float(x) for x in raw_alpha],
                "pre_shrink_alpha": [float(x) for x in clipped_alpha],
                "shrink_reliability": float(reliability),
                "alpha": [float(x) for x in alpha],
                "relative_gain": float(relative_gain),
            }
        )
    candidate_rows.sort(
        key=lambda row: (
            float(row["relative_gain"]),
            float(row["positive_view_fraction"]),
            int(row["samples"]),
            float(row["denominator_sum"]),
        ),
        reverse=True,
    )
    if int(max_profile_bins) > 0:
        selected_rows = candidate_rows[: int(max_profile_bins)]
    else:
        selected_rows = candidate_rows
    bin_rgb_alphas_by_face: dict[str, dict[str, list[float]]] = {}
    for row in selected_rows:
        bin_rgb_alphas_by_face.setdefault(str(int(row["face_id"])), {})[str(int(row["bin_id"]))] = [
            float(x) for x in row["alpha"]
        ]
    generated = [float(x) for x in multiplier_grid if float(x) >= 0.0]
    if 1.0 not in generated:
        generated.append(1.0)
    combined = sorted(set(round(float(x), 8) for x in [*base, *generated, 0.0]))
    profile = {
        "enabled": True,
        "mode": "policy_val_bin_rgb_alpha",
        "policy_val_views_used": int(view_count),
        "samples": int(sample_count),
        "texture_size": int(texture_size),
        "candidate_bin_count": int(len(candidate_rows)),
        "bin_rgb_alpha_count": int(len(selected_rows)),
        "fallback_bin_count": int(fallback_bin_count + max(0, len(candidate_rows) - len(selected_rows))),
        "fallback_raw_alpha": [float(x) for x in fallback_raw_alpha],
        "fallback_alpha": [float(x) for x in fallback_alpha],
        "max_alpha": float(max_alpha),
        "min_alpha": float(min_alpha_value),
        "min_bin_samples": int(min_bin_samples),
        "min_positive_view_fraction": float(min_positive_view_fraction),
        "max_profile_bins": int(max_profile_bins),
        "shrink": {
            "enabled": bool(shrink_enabled),
            "count_tau": float(shrink_count_tau),
            "denominator_tau": float(shrink_denominator_tau),
            "prior": str(prior_mode),
        },
        "multiplier_grid": sorted(set(float(x) for x in generated)),
        "alpha_grid": combined,
        "bin_rgb_alpha_preview": selected_rows[:32],
        "bin_rgb_alphas_by_face": bin_rgb_alphas_by_face,
    }
    return combined, profile


def save_image_chw(path: Path, image: np.ndarray) -> None:
    arr = np.clip(np.moveaxis(image, 0, -1), 0.0, 1.0)
    arr_u8 = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8).save(path)


def png_quantized_change_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before_hwc = np.clip(np.moveaxis(np.asarray(before, dtype=np.float32), 0, -1), 0.0, 1.0)
    after_hwc = np.clip(np.moveaxis(np.asarray(after, dtype=np.float32), 0, -1), 0.0, 1.0)
    before_u8 = np.clip(np.round(before_hwc * 255.0), 0, 255).astype(np.uint8)
    after_u8 = np.clip(np.round(after_hwc * 255.0), 0, 255).astype(np.uint8)
    return np.any(before_u8 != after_u8, axis=-1)


def image_ssim_chw(render: np.ndarray, gt: np.ndarray, max_size: int) -> float:
    render_t = torch.from_numpy(np.asarray(render, dtype=np.float32)).unsqueeze(0)
    gt_t = torch.from_numpy(np.asarray(gt, dtype=np.float32)).unsqueeze(0)
    if torch.cuda.is_available():
        render_t = render_t.cuda()
        gt_t = gt_t.cuda()
    if int(max_size) > 0:
        _, _, h, w = render_t.shape
        scale = min(1.0, float(max_size) / float(max(h, w)))
        if scale < 1.0:
            size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
            render_t = F.interpolate(render_t, size=size, mode="bilinear", align_corners=False)
            gt_t = F.interpolate(gt_t, size=size, mode="bilinear", align_corners=False)
    return float(ssim(render_t, gt_t).detach().cpu().item())


def image_l1_chw(render: np.ndarray, gt: np.ndarray, max_size: int) -> float:
    render_t = torch.from_numpy(np.asarray(render, dtype=np.float32)).unsqueeze(0)
    gt_t = torch.from_numpy(np.asarray(gt, dtype=np.float32)).unsqueeze(0)
    if torch.cuda.is_available():
        render_t = render_t.cuda()
        gt_t = gt_t.cuda()
    if int(max_size) > 0:
        _, _, h, w = render_t.shape
        scale = min(1.0, float(max_size) / float(max(h, w)))
        if scale < 1.0:
            size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
            render_t = F.interpolate(render_t, size=size, mode="bilinear", align_corners=False)
            gt_t = F.interpolate(gt_t, size=size, mode="bilinear", align_corners=False)
    return float(torch.mean(torch.abs(render_t - gt_t)).detach().cpu().item())


def build_lpips_model():
    from lpipsPyTorch.modules.lpips import LPIPS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LPIPS("vgg").to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def image_lpips_chw(render: np.ndarray, gt: np.ndarray, max_size: int, lpips_model) -> float:
    if lpips_model is None:
        raise RuntimeError("LPIPS policy-val gate requested but LPIPS model is not initialized")
    render_t = torch.from_numpy(np.asarray(render, dtype=np.float32)).unsqueeze(0)
    gt_t = torch.from_numpy(np.asarray(gt, dtype=np.float32)).unsqueeze(0)
    device = next(lpips_model.parameters()).device
    render_t = render_t.to(device)
    gt_t = gt_t.to(device)
    if int(max_size) > 0:
        _, _, h, w = render_t.shape
        scale = min(1.0, float(max_size) / float(max(h, w)))
        if scale < 1.0:
            size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
            render_t = F.interpolate(render_t, size=size, mode="bilinear", align_corners=False)
            gt_t = F.interpolate(gt_t, size=size, mode="bilinear", align_corners=False)
    with torch.no_grad():
        return float(lpips_model(render_t, gt_t).detach().mean().cpu().item())


def apply_to_target(
    target_views: list[Path],
    atlas: dict[int, FaceAtlas],
    output_model: Path,
    split: str,
    method_name: str,
    alpha: float,
    min_alpha: float,
    max_abs_delta_rgb: float,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    local_alpha_profile: dict[str, Any] | None = None,
    face_gain_guard_profile: dict[str, Any] | None = None,
    bin_uncertainty_guard_profile: dict[str, Any] | None = None,
    parent_edge_apply_profile: dict[str, Any] | None = None,
    view_confidence_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    method_dir = output_model / split / method_name
    render_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    if render_dir.exists():
        shutil.rmtree(render_dir)
    if gt_dir.exists():
        shutil.rmtree(gt_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    changed_pixels = 0
    png_quantized_changed_pixels = 0
    total_pixels = 0
    written = 0
    for path in tqdm(target_views, desc=f"apply atlas {split}"):
        z = np.load(path)
        if "rgb_render" not in z:
            raise KeyError(f"{path} missing rgb_render")
        rgb = np.asarray(z["rgb_render"], dtype=np.float32)
        delta, valid = predict_delta_for_npz(
            z,
            atlas,
            alpha,
            min_alpha,
            min_atlas_bin_count=int(min_atlas_bin_count),
            min_atlas_face_samples=int(min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(atlas_confidence_mode),
            atlas_confidence_count_scale=float(atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(min_atlas_confidence),
            local_alpha_profile=local_alpha_profile,
            face_gain_guard_profile=face_gain_guard_profile,
            bin_uncertainty_guard_profile=bin_uncertainty_guard_profile,
            parent_edge_apply_profile=parent_edge_apply_profile,
            view_confidence_profile=view_confidence_profile,
        )
        if float(max_abs_delta_rgb) > 0.0:
            delta = np.clip(delta, -float(max_abs_delta_rgb), float(max_abs_delta_rgb))
        changed_mask = np.any(np.abs(delta) > 1.0e-8, axis=0)
        adapted = np.clip(rgb + delta, 0.0, 1.0)
        png_changed_mask = png_quantized_change_mask(rgb, adapted)
        name = f"{path.stem}.png"
        save_image_chw(render_dir / name, adapted)
        if "rgb_gt" in z:
            save_image_chw(gt_dir / name, np.asarray(z["rgb_gt"], dtype=np.float32))
        else:
            save_image_chw(gt_dir / name, rgb)
        changed_pixels += int(changed_mask.sum())
        png_quantized_changed_pixels += int(png_changed_mask.sum())
        total_pixels += int(valid.size)
        written += 1
    return {
        "split": split,
        "method_name": method_name,
        "written_views": int(written),
        "changed_pixels": int(changed_pixels),
        "png_quantized_changed_pixels": int(png_quantized_changed_pixels),
        "total_pixels": int(total_pixels),
        "changed_fraction": float(changed_pixels / max(1, total_pixels)),
        "png_quantized_changed_fraction": float(png_quantized_changed_pixels / max(1, total_pixels)),
        "render_dir": str(render_dir),
        "gt_dir": str(gt_dir),
    }


def evaluate_target_support_profile(
    target_views: list[Path],
    atlas: dict[int, FaceAtlas],
    alpha: float,
    min_alpha: float,
    max_abs_delta_rgb: float,
    min_atlas_bin_count: int,
    min_atlas_face_samples: int,
    max_atlas_bin_rgb_variance: float,
    min_atlas_bin_sign_consistency: float,
    atlas_confidence_mode: str,
    atlas_confidence_count_scale: float,
    atlas_confidence_empty_bin: float,
    atlas_confidence_variance_scale: float,
    atlas_confidence_sign_power: float,
    atlas_confidence_face_sample_scale: float,
    min_atlas_confidence: float,
    local_alpha_profile: dict[str, Any] | None = None,
    face_gain_guard_profile: dict[str, Any] | None = None,
    bin_uncertainty_guard_profile: dict[str, Any] | None = None,
    parent_edge_apply_profile: dict[str, Any] | None = None,
    view_confidence_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    disabled = {
        "enabled": False,
        "mode": "target_support_candidate_selection",
        "reason": "",
    }
    if not target_views or not atlas or float(alpha) <= 0.0:
        disabled["reason"] = "no_target_views_or_empty_atlas_or_zero_alpha"
        return disabled
    changed_pixels = 0
    png_quantized_changed_pixels = 0
    valid_pixels = 0
    total_pixels = 0
    abs_delta_sum = 0.0
    active_abs_delta_sum = 0.0
    active_channel_values = 0
    view_rows: list[dict[str, Any]] = []
    for path in tqdm(target_views, desc="target support profile"):
        z = np.load(path)
        if "rgb_render" not in z:
            raise KeyError(f"{path} missing rgb_render")
        delta, valid = predict_delta_for_npz(
            z,
            atlas,
            alpha,
            min_alpha,
            min_atlas_bin_count=int(min_atlas_bin_count),
            min_atlas_face_samples=int(min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(atlas_confidence_mode),
            atlas_confidence_count_scale=float(atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(min_atlas_confidence),
            local_alpha_profile=local_alpha_profile,
            face_gain_guard_profile=face_gain_guard_profile,
            bin_uncertainty_guard_profile=bin_uncertainty_guard_profile,
            parent_edge_apply_profile=parent_edge_apply_profile,
            view_confidence_profile=view_confidence_profile,
        )
        if float(max_abs_delta_rgb) > 0.0:
            delta = np.clip(delta, -float(max_abs_delta_rgb), float(max_abs_delta_rgb))
        changed_mask = np.any(np.abs(delta) > 1.0e-8, axis=0)
        adapted = np.clip(np.asarray(z["rgb_render"], dtype=np.float32) + delta, 0.0, 1.0)
        png_changed_mask = png_quantized_change_mask(np.asarray(z["rgb_render"], dtype=np.float32), adapted)
        changed = int(changed_mask.sum())
        png_changed = int(png_changed_mask.sum())
        valid_count = int(np.asarray(valid, dtype=bool).sum())
        total = int(valid.size)
        abs_delta = float(np.sum(np.abs(delta)))
        active_abs_delta = float(np.sum(np.abs(delta[:, changed_mask]))) if changed > 0 else 0.0
        changed_pixels += changed
        png_quantized_changed_pixels += png_changed
        valid_pixels += valid_count
        total_pixels += total
        abs_delta_sum += abs_delta
        active_abs_delta_sum += active_abs_delta
        active_channel_values += int(changed * 3)
        view_rows.append(
            {
                "view": str(path.name),
                "changed_pixels": int(changed),
                "png_quantized_changed_pixels": int(png_changed),
                "valid_pixels": int(valid_count),
                "total_pixels": int(total),
                "changed_fraction": float(changed / max(1, total)),
                "png_quantized_changed_fraction": float(png_changed / max(1, total)),
                "valid_fraction": float(valid_count / max(1, total)),
                "mean_abs_delta": float(abs_delta / max(1, total * 3)),
                "active_mean_abs_delta": float(active_abs_delta / max(1, changed * 3)),
            }
        )
    changed_fracs = sorted(float(row["changed_fraction"]) for row in view_rows)
    png_changed_fracs = sorted(float(row["png_quantized_changed_fraction"]) for row in view_rows)
    valid_fracs = sorted(float(row["valid_fraction"]) for row in view_rows)

    def lower_cvar(values: list[float], fraction: float = 0.2) -> float:
        if not values:
            return 0.0
        count = max(1, int(math.ceil(float(len(values)) * float(fraction))))
        return float(np.mean(values[:count]))

    return {
        "enabled": True,
        "mode": "target_support_candidate_selection",
        "view_count": int(len(view_rows)),
        "changed_pixels": int(changed_pixels),
        "png_quantized_changed_pixels": int(png_quantized_changed_pixels),
        "valid_pixels": int(valid_pixels),
        "total_pixels": int(total_pixels),
        "changed_fraction": float(changed_pixels / max(1, total_pixels)),
        "png_quantized_changed_fraction": float(png_quantized_changed_pixels / max(1, total_pixels)),
        "valid_fraction": float(valid_pixels / max(1, total_pixels)),
        "min_view_changed_fraction": float(changed_fracs[0] if changed_fracs else 0.0),
        "cvar20_view_changed_fraction": lower_cvar(changed_fracs, 0.2),
        "min_view_png_quantized_changed_fraction": float(png_changed_fracs[0] if png_changed_fracs else 0.0),
        "cvar20_view_png_quantized_changed_fraction": lower_cvar(png_changed_fracs, 0.2),
        "min_view_valid_fraction": float(valid_fracs[0] if valid_fracs else 0.0),
        "cvar20_view_valid_fraction": lower_cvar(valid_fracs, 0.2),
        "mean_abs_delta": float(abs_delta_sum / max(1, total_pixels * 3)),
        "active_mean_abs_delta": float(active_abs_delta_sum / max(1, active_channel_values)),
        "per_view": view_rows,
    }


def evaluate_target_face_support_proxy(
    target_views: list[Path],
    candidate_faces: set[int],
    min_alpha: float,
    max_views: int = 0,
) -> dict[str, Any]:
    """Cheap GT-free proxy for target-visible coverage of a support face set."""
    disabled = {
        "enabled": False,
        "mode": "target_support_prerank_face_proxy",
        "reason": "",
    }
    if not target_views or not candidate_faces:
        disabled["reason"] = "no_target_views_or_empty_faces"
        return disabled
    selected_views = list(target_views)
    if int(max_views) > 0 and len(selected_views) > int(max_views):
        indices = np.linspace(0, len(selected_views) - 1, num=int(max_views), dtype=np.int64)
        selected_views = [selected_views[int(idx)] for idx in sorted(set(int(x) for x in indices))]
    face_array = np.fromiter((int(face) for face in candidate_faces), dtype=np.int64)
    covered_pixels = 0
    valid_pixels = 0
    total_pixels = 0
    view_rows: list[dict[str, Any]] = []
    for path in tqdm(selected_views, desc="target support pre-rank"):
        z = np.load(path)
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
        covered = valid & np.isin(face_id, face_array)
        covered_count = int(np.count_nonzero(covered))
        valid_count = int(np.count_nonzero(valid))
        total_count = int(face_id.size)
        covered_pixels += covered_count
        valid_pixels += valid_count
        total_pixels += total_count
        view_rows.append(
            {
                "view": str(path.name),
                "covered_pixels": int(covered_count),
                "valid_pixels": int(valid_count),
                "total_pixels": int(total_count),
                "coverage_fraction": float(covered_count / max(1, total_count)),
                "valid_fraction": float(valid_count / max(1, total_count)),
                "covered_valid_fraction": float(covered_count / max(1, valid_count)),
            }
        )
    coverage_fracs = sorted(float(row["coverage_fraction"]) for row in view_rows)
    valid_fracs = sorted(float(row["valid_fraction"]) for row in view_rows)
    covered_valid_fracs = sorted(float(row["covered_valid_fraction"]) for row in view_rows)

    def lower_cvar(values: list[float], fraction: float = 0.2) -> float:
        if not values:
            return 0.0
        count = max(1, int(math.ceil(float(len(values)) * float(fraction))))
        return float(np.mean(values[:count]))

    return {
        "enabled": True,
        "mode": "target_support_prerank_face_proxy",
        "view_count": int(len(view_rows)),
        "candidate_faces": int(len(candidate_faces)),
        "covered_pixels": int(covered_pixels),
        "valid_pixels": int(valid_pixels),
        "total_pixels": int(total_pixels),
        "coverage_fraction": float(covered_pixels / max(1, total_pixels)),
        "valid_fraction": float(valid_pixels / max(1, total_pixels)),
        "covered_valid_fraction": float(covered_pixels / max(1, valid_pixels)),
        "min_view_coverage_fraction": float(coverage_fracs[0] if coverage_fracs else 0.0),
        "cvar20_view_coverage_fraction": lower_cvar(coverage_fracs, 0.2),
        "min_view_valid_fraction": float(valid_fracs[0] if valid_fracs else 0.0),
        "cvar20_view_valid_fraction": lower_cvar(valid_fracs, 0.2),
        "min_view_covered_valid_fraction": float(
            covered_valid_fracs[0] if covered_valid_fracs else 0.0
        ),
        "cvar20_view_covered_valid_fraction": lower_cvar(covered_valid_fracs, 0.2),
        "per_view": view_rows,
    }


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _link_or_copy_dir(src: Path, dst: Path) -> bool:
    _remove_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst, target_is_directory=True)
        return True
    except OSError:
        shutil.copytree(src, dst)
        return False


def write_noop_fallback_output(
    source_model: Path,
    output_model: Path,
    split: str,
    base_method_name: str,
    method_name: str,
    target_views: list[Path],
    fallback_source: str,
) -> dict[str, Any]:
    method_dir = output_model / split / method_name
    if method_dir.exists() or method_dir.is_symlink():
        _remove_path(method_dir)
    method_dir.mkdir(parents=True, exist_ok=True)
    render_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    base_method_dir = source_model / split / base_method_name
    render_source = base_method_dir / "renders"
    gt_source = base_method_dir / "gt"
    used_symlink = False
    copied_from_source = False
    written = 0
    total_pixels = 0
    if str(fallback_source) == "source_model" and render_source.is_dir() and gt_source.is_dir():
        used_symlink = _link_or_copy_dir(render_source, render_dir)
        used_symlink = _link_or_copy_dir(gt_source, gt_dir) and used_symlink
        copied_from_source = True
        written = len([p for p in render_source.iterdir() if p.is_file()])
    else:
        render_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        for path in tqdm(target_views, desc=f"write no-op fallback {split}"):
            z = np.load(path)
            if "rgb_render" not in z:
                raise KeyError(f"{path} missing rgb_render for no-op fallback")
            name = f"{path.stem}.png"
            save_image_chw(render_dir / name, np.asarray(z["rgb_render"], dtype=np.float32))
            if "rgb_gt" in z:
                save_image_chw(gt_dir / name, np.asarray(z["rgb_gt"], dtype=np.float32))
            else:
                save_image_chw(gt_dir / name, np.asarray(z["rgb_render"], dtype=np.float32))
            total_pixels += int(np.asarray(z["rgb_render"]).shape[-1] * np.asarray(z["rgb_render"]).shape[-2])
            written += 1

    results_source = source_model / "results.json"
    if str(fallback_source) == "source_model" and results_source.is_file():
        try:
            source_results = json.loads(results_source.read_text(encoding="utf-8"))
            if base_method_name in source_results:
                existing = {}
                results_path = output_model / "results.json"
                if results_path.is_file():
                    existing = json.loads(results_path.read_text(encoding="utf-8"))
                existing[method_name] = source_results[base_method_name]
                results_path.write_text(json.dumps(existing, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    per_view_source = source_model / "per_view.json"
    if str(fallback_source) == "source_model" and per_view_source.is_file():
        try:
            source_per_view = json.loads(per_view_source.read_text(encoding="utf-8"))
            if base_method_name in source_per_view:
                existing = {}
                per_view_path = output_model / "per_view.json"
                if per_view_path.is_file():
                    existing = json.loads(per_view_path.read_text(encoding="utf-8"))
                existing[method_name] = source_per_view[base_method_name]
                per_view_path.write_text(json.dumps(existing, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    return {
        "split": split,
        "method_name": method_name,
        "written_views": int(written),
        "changed_pixels": 0,
        "total_pixels": int(total_pixels),
        "changed_fraction": 0.0,
        "render_dir": str(render_dir),
        "gt_dir": str(gt_dir),
        "fallback_noop": True,
        "fallback_source": str(fallback_source),
        "fallback_source_method": str(base_method_dir),
        "fallback_copied_from_source": bool(copied_from_source),
        "fallback_symlinked": bool(used_symlink),
    }


def save_atlas_npz(path: Path, atlas: dict[int, FaceAtlas]) -> None:
    faces = np.array(sorted(atlas.keys()), dtype=np.int64)
    textures = np.stack([atlas[int(face)].texture for face in faces], axis=0) if faces.size else np.zeros((0, 1, 1, 3), dtype=np.float32)
    counts = np.stack([atlas[int(face)].counts for face in faces], axis=0) if faces.size else np.zeros((0, 1, 1), dtype=np.int32)
    variances = np.stack([atlas[int(face)].variance for face in faces], axis=0) if faces.size else np.zeros((0, 1, 1, 3), dtype=np.float32)
    sign_consistency = np.stack([atlas[int(face)].sign_consistency for face in faces], axis=0) if faces.size else np.zeros((0, 1, 1, 3), dtype=np.float32)
    means = np.stack([atlas[int(face)].mean_rgb for face in faces], axis=0) if faces.size else np.zeros((0, 3), dtype=np.float32)
    samples = np.array([atlas[int(face)].samples for face in faces], dtype=np.int64)
    view_basis_enabled = bool(
        faces.size
        and any(atlas[int(face)].view_basis_coefficients is not None for face in faces)
    )
    payload: dict[str, Any] = {
        "face_id": faces,
        "texture": textures,
        "counts": counts,
        "variance": variances,
        "sign_consistency": sign_consistency,
        "mean_rgb": means,
        "samples": samples,
    }
    if view_basis_enabled:
        first = next(atlas[int(face)].view_basis_coefficients for face in faces if atlas[int(face)].view_basis_coefficients is not None)
        assert first is not None
        h, w, dim, _channels = first.shape
        coeffs = []
        supports = []
        modes = []
        feature_means = []
        feature_stds = []
        ood_modes = []
        ood_max_z = []
        ood_min_std = []
        for face in faces:
            face_atlas = atlas[int(face)]
            if face_atlas.view_basis_coefficients is None:
                coeffs.append(np.zeros((h, w, dim, 3), dtype=np.float32))
                supports.append(np.zeros((h, w), dtype=bool))
                modes.append("none")
                feature_means.append(np.zeros((h, w, dim), dtype=np.float32))
                feature_stds.append(np.ones((h, w, dim), dtype=np.float32))
            else:
                coeffs.append(face_atlas.view_basis_coefficients.astype(np.float32))
                supports.append(
                    np.asarray(face_atlas.view_basis_support, dtype=bool)
                    if face_atlas.view_basis_support is not None
                    else np.zeros((h, w), dtype=bool)
                )
                modes.append(str(face_atlas.view_basis_mode))
                feature_means.append(
                    face_atlas.view_basis_feature_mean.astype(np.float32)
                    if face_atlas.view_basis_feature_mean is not None
                    else np.zeros((h, w, dim), dtype=np.float32)
                )
                feature_stds.append(
                    face_atlas.view_basis_feature_std.astype(np.float32)
                    if face_atlas.view_basis_feature_std is not None
                    else np.ones((h, w, dim), dtype=np.float32)
                )
            ood_modes.append(str(face_atlas.view_basis_ood_mode))
            ood_max_z.append(float(face_atlas.view_basis_ood_max_z))
            ood_min_std.append(float(face_atlas.view_basis_ood_min_std))
        payload["view_basis_coefficients"] = np.stack(coeffs, axis=0)
        payload["view_basis_support"] = np.stack(supports, axis=0).astype(np.uint8)
        payload["view_basis_mode"] = np.asarray(modes)
        payload["view_basis_feature_mean"] = np.stack(feature_means, axis=0)
        payload["view_basis_feature_std"] = np.stack(feature_stds, axis=0)
        payload["view_basis_ood_mode"] = np.asarray(ood_modes)
        payload["view_basis_ood_max_z"] = np.asarray(ood_max_z, dtype=np.float32)
        payload["view_basis_ood_min_std"] = np.asarray(ood_min_std, dtype=np.float32)
    expert_enabled = bool(
        faces.size
        and any(atlas[int(face)].expert_textures is not None for face in faces)
    )
    if expert_enabled:
        first_expert = next(
            atlas[int(face)].expert_textures
            for face in faces
            if atlas[int(face)].expert_textures is not None
        )
        assert first_expert is not None
        expert_count, h, w, _channels = first_expert.shape
        expert_textures = []
        expert_counts = []
        expert_variance = []
        expert_sign = []
        expert_samples = []
        expert_modes = []
        expert_min_bins = []
        expert_fallback_modes = []
        centers = []
        for face in faces:
            face_atlas = atlas[int(face)]
            if face_atlas.expert_textures is None:
                expert_textures.append(
                    np.repeat(face_atlas.texture[None, ...], expert_count, axis=0).astype(np.float32)
                )
                expert_counts.append(np.zeros((expert_count, h, w), dtype=np.int32))
                expert_variance.append(
                    np.repeat(face_atlas.variance[None, ...], expert_count, axis=0).astype(np.float32)
                )
                expert_sign.append(
                    np.repeat(face_atlas.sign_consistency[None, ...], expert_count, axis=0).astype(np.float32)
                )
                expert_samples.append(np.zeros((expert_count,), dtype=np.int64))
                expert_modes.append("none")
                centers.append(np.zeros((expert_count, 3), dtype=np.float32))
                expert_min_bins.append(1)
                expert_fallback_modes.append("global")
            else:
                expert_textures.append(face_atlas.expert_textures.astype(np.float32))
                expert_counts.append(
                    face_atlas.expert_counts.astype(np.int32)
                    if face_atlas.expert_counts is not None
                    else np.zeros((expert_count, h, w), dtype=np.int32)
                )
                expert_variance.append(
                    face_atlas.expert_variance.astype(np.float32)
                    if face_atlas.expert_variance is not None
                    else np.repeat(face_atlas.variance[None, ...], expert_count, axis=0).astype(np.float32)
                )
                expert_sign.append(
                    face_atlas.expert_sign_consistency.astype(np.float32)
                    if face_atlas.expert_sign_consistency is not None
                    else np.repeat(face_atlas.sign_consistency[None, ...], expert_count, axis=0).astype(np.float32)
                )
                expert_samples.append(
                    face_atlas.expert_samples.astype(np.int64)
                    if face_atlas.expert_samples is not None
                    else np.zeros((expert_count,), dtype=np.int64)
                )
                expert_modes.append(str(face_atlas.expert_feature_mode))
                centers.append(
                    face_atlas.expert_centers.astype(np.float32)
                    if face_atlas.expert_centers is not None
                    else np.zeros((expert_count, 3), dtype=np.float32)
                )
                expert_min_bins.append(int(face_atlas.expert_min_bin_samples))
                expert_fallback_modes.append(str(face_atlas.expert_fallback_mode))
        payload["expert_texture"] = np.stack(expert_textures, axis=0)
        payload["expert_counts"] = np.stack(expert_counts, axis=0)
        payload["expert_variance"] = np.stack(expert_variance, axis=0)
        payload["expert_sign_consistency"] = np.stack(expert_sign, axis=0)
        payload["expert_samples"] = np.stack(expert_samples, axis=0)
        payload["expert_feature_mode"] = np.asarray(expert_modes)
        payload["expert_centers"] = np.stack(centers, axis=0)
        payload["expert_min_bin_samples"] = np.asarray(expert_min_bins, dtype=np.int32)
        payload["expert_fallback_mode"] = np.asarray(expert_fallback_modes)
    teacher_basis_enabled = bool(
        faces.size
        and any(atlas[int(face)].teacher_basis_coefficients is not None for face in faces)
    )
    if teacher_basis_enabled:
        teacher_shapes = [
            atlas[int(face)].teacher_basis_coefficients.shape
            for face in faces
            if atlas[int(face)].teacher_basis_coefficients is not None
        ]
        teacher_dim = max(int(shape[0]) for shape in teacher_shapes)
        teacher_channels = max(int(shape[1]) for shape in teacher_shapes)
        teacher_coeffs = []
        teacher_output_channels = []
        teacher_modes = []
        teacher_feature_means = []
        teacher_feature_stds = []
        teacher_ood_max_z = []
        teacher_ood_min_std = []
        teacher_apply_modes = []
        teacher_blends = []
        for face in faces:
            face_atlas = atlas[int(face)]
            if face_atlas.teacher_basis_coefficients is None:
                teacher_coeffs.append(np.zeros((teacher_dim, teacher_channels), dtype=np.float32))
                teacher_output_channels.append(0)
                teacher_modes.append("none")
                teacher_feature_means.append(np.zeros((teacher_dim,), dtype=np.float32))
                teacher_feature_stds.append(np.ones((teacher_dim,), dtype=np.float32))
            else:
                local_coeffs = face_atlas.teacher_basis_coefficients.astype(np.float32)
                coeffs_padded = np.zeros((teacher_dim, teacher_channels), dtype=np.float32)
                local_dim = min(teacher_dim, int(local_coeffs.shape[0]))
                local_channels = min(teacher_channels, int(local_coeffs.shape[1]))
                coeffs_padded[:local_dim, :local_channels] = local_coeffs[:local_dim, :local_channels]
                teacher_coeffs.append(coeffs_padded)
                teacher_output_channels.append(int(local_coeffs.shape[1]))
                teacher_modes.append(str(face_atlas.teacher_basis_mode))
                local_mean = (
                    face_atlas.teacher_basis_feature_mean.astype(np.float32)
                    if face_atlas.teacher_basis_feature_mean is not None
                    else np.zeros((0,), dtype=np.float32)
                )
                mean_padded = np.zeros((teacher_dim,), dtype=np.float32)
                mean_count = min(teacher_dim, int(local_mean.shape[0]))
                if mean_count > 0:
                    mean_padded[:mean_count] = local_mean[:mean_count]
                teacher_feature_means.append(mean_padded)
                local_std = (
                    face_atlas.teacher_basis_feature_std.astype(np.float32)
                    if face_atlas.teacher_basis_feature_std is not None
                    else np.ones((0,), dtype=np.float32)
                )
                std_padded = np.ones((teacher_dim,), dtype=np.float32)
                std_count = min(teacher_dim, int(local_std.shape[0]))
                if std_count > 0:
                    std_padded[:std_count] = local_std[:std_count]
                teacher_feature_stds.append(std_padded)
            teacher_ood_max_z.append(float(face_atlas.teacher_basis_ood_max_z))
            teacher_ood_min_std.append(float(face_atlas.teacher_basis_ood_min_std))
            teacher_apply_modes.append(str(face_atlas.teacher_basis_apply_mode))
            teacher_blends.append(float(face_atlas.teacher_basis_blend))
        payload["teacher_basis_coefficients"] = np.stack(teacher_coeffs, axis=0)
        payload["teacher_basis_output_channels"] = np.asarray(teacher_output_channels, dtype=np.int32)
        payload["teacher_basis_mode"] = np.asarray(teacher_modes)
        payload["teacher_basis_feature_mean"] = np.stack(teacher_feature_means, axis=0)
        payload["teacher_basis_feature_std"] = np.stack(teacher_feature_stds, axis=0)
        payload["teacher_basis_ood_max_z"] = np.asarray(teacher_ood_max_z, dtype=np.float32)
        payload["teacher_basis_ood_min_std"] = np.asarray(teacher_ood_min_std, dtype=np.float32)
        payload["teacher_basis_apply_mode"] = np.asarray(teacher_apply_modes)
        payload["teacher_basis_blend"] = np.asarray(teacher_blends, dtype=np.float32)
    teacher_texture_enabled = bool(
        faces.size
        and any(atlas[int(face)].teacher_texture_basis is not None for face in faces)
    )
    if teacher_texture_enabled:
        first_texture_basis = next(
            atlas[int(face)].teacher_texture_basis
            for face in faces
            if atlas[int(face)].teacher_texture_basis is not None
        )
        assert first_texture_basis is not None
        _first_rank, h, w, _channels = first_texture_basis.shape
        rank = max(
            int(atlas[int(face)].teacher_texture_basis.shape[0])
            for face in faces
            if atlas[int(face)].teacher_texture_basis is not None
        )
        texture_basis_rows = []
        texture_support_rows = []
        texture_rank_rows = []
        texture_energy_rows = []
        for face in faces:
            face_atlas = atlas[int(face)]
            if face_atlas.teacher_texture_basis is None:
                texture_basis_rows.append(np.zeros((rank, h, w, 3), dtype=np.float32))
                texture_support_rows.append(np.zeros((h, w), dtype=np.uint8))
                texture_rank_rows.append(0)
                texture_energy_rows.append(np.zeros((rank,), dtype=np.float32))
            else:
                local_basis = face_atlas.teacher_texture_basis.astype(np.float32)
                local_rank = min(rank, int(local_basis.shape[0]))
                padded = np.zeros((rank, h, w, 3), dtype=np.float32)
                copy_h = min(h, int(local_basis.shape[1]))
                copy_w = min(w, int(local_basis.shape[2]))
                if local_rank > 0 and copy_h > 0 and copy_w > 0:
                    padded[:local_rank, :copy_h, :copy_w, :] = local_basis[
                        :local_rank,
                        :copy_h,
                        :copy_w,
                        :,
                    ]
                local_basis = padded
                texture_basis_rows.append(local_basis)
                support_padded = np.zeros((h, w), dtype=np.uint8)
                if face_atlas.teacher_texture_support is not None:
                    local_support = face_atlas.teacher_texture_support.astype(np.uint8)
                    support_h = min(h, int(local_support.shape[0]))
                    support_w = min(w, int(local_support.shape[1]))
                    if support_h > 0 and support_w > 0:
                        support_padded[:support_h, :support_w] = local_support[:support_h, :support_w]
                texture_support_rows.append(support_padded)
                texture_rank_rows.append(int(local_rank))
                local_energy = (
                    face_atlas.teacher_texture_energy.astype(np.float32)
                    if face_atlas.teacher_texture_energy is not None
                    else np.zeros((0,), dtype=np.float32)
                )
                energy_padded = np.zeros((rank,), dtype=np.float32)
                local_energy_count = min(rank, int(local_energy.shape[0]))
                if local_energy_count > 0:
                    energy_padded[:local_energy_count] = local_energy[:local_energy_count]
                texture_energy_rows.append(energy_padded)
        payload["teacher_texture_basis"] = np.stack(texture_basis_rows, axis=0)
        payload["teacher_texture_support"] = np.stack(texture_support_rows, axis=0).astype(np.uint8)
        payload["teacher_texture_rank"] = np.asarray(texture_rank_rows, dtype=np.int32)
        payload["teacher_texture_energy"] = np.stack(texture_energy_rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def write_report(path: Path, audit: dict[str, Any]) -> None:
    val = audit.get("policy_val", {})
    target = audit.get("target_apply", {})
    risk = audit.get("policy_val_risk_gate", {})
    support = audit.get("carrier_summary", {}).get("support_expansion", {})
    support_prerank = audit.get("carrier_summary", {}).get("target_support_prerank", {})
    local_alpha = audit.get("local_alpha_profile", {})
    face_gain = audit.get("face_gain_guard_profile", {})
    bin_uncertainty = audit.get("bin_uncertainty_guard_profile", {})
    view_confidence = audit.get("view_confidence_profile", {})
    view_alpha_cap = audit.get("view_alpha_cap_profile", {})
    multiscale_prior = audit.get("fit_summary", {}).get("surface_multiscale_prior", {})
    teacher_basis = audit.get("fit_summary", {}).get("teacher_distilled_basis", {})
    policy_candidate_control = audit.get("fit_summary", {}).get("policy_candidate_control", {})
    adaptive_ladder = policy_candidate_control.get("adaptive_texture_size_ladder", {})
    fallback_alpha_value = local_alpha.get("fallback_alpha", 0.0)
    if isinstance(fallback_alpha_value, (list, tuple)):
        fallback_alpha_text = json.dumps([float(x) for x in fallback_alpha_value])
    else:
        fallback_alpha_text = f"{float(fallback_alpha_value or 0.0):.6f}"
    lines = [
        "# Surface Residual Region Texture Adapter Audit",
        "",
        f"- accepted: `{audit.get('accepted', False)}`",
        f"- source model: `{audit.get('source_model', '')}`",
        f"- fit evidence: `{audit.get('fit_evidence_dir', '')}`",
        f"- target evidence: `{audit.get('target_evidence_dir', '')}`",
        f"- region carrier: `{audit.get('region_carrier_json', '')}`",
        f"- support expansion mode: `{support.get('mode', 'none')}`",
        f"- support expansion base faces: `{support.get('base_faces', 0)}`",
        f"- support expansion added faces: `{support.get('added_faces', 0)}`",
        f"- candidate faces after expansion: `{support.get('candidate_faces_after_expansion', 0)}`",
        f"- target-support pre-rank enabled: `{support_prerank.get('enabled', False)}`",
        f"- target-support pre-rank retained support candidates: `{support_prerank.get('retained_support_candidate_count', 0)}`",
        f"- atlas faces: `{audit.get('fit_summary', {}).get('atlas_faces', 0)}`",
        f"- fit samples: `{audit.get('fit_summary', {}).get('fit_samples', 0)}`",
        f"- selected support mode: `{audit.get('fit_summary', {}).get('selected_support_mode', '')}`",
        f"- selected support added faces: `{audit.get('fit_summary', {}).get('selected_support_added_faces', 0)}`",
        f"- selected texture size: `{audit.get('fit_summary', {}).get('selected_texture_size', audit.get('fit_summary', {}).get('texture_size', 0))}`",
        f"- selected teacher low-rank texture rank: `{audit.get('fit_summary', {}).get('selected_teacher_distilled_low_rank_texture_rank', 0)}`",
        f"- teacher low-rank texture rank candidates: `{audit.get('fit_summary', {}).get('teacher_distilled_low_rank_texture_rank_candidates', [])}`",
        f"- selected fill mode: `{audit.get('fit_summary', {}).get('selected_atlas_empty_bin_fill_mode', audit.get('fit_summary', {}).get('atlas_empty_bin_fill_mode', ''))}`",
        f"- selected max abs delta RGB: `{float(audit.get('fit_summary', {}).get('selected_max_abs_delta_rgb', audit.get('fit_summary', {}).get('requested_max_abs_delta_rgb', 0.0)) or 0.0):.6f}`",
        f"- max abs delta RGB candidates: `{audit.get('fit_summary', {}).get('max_abs_delta_rgb_candidates', [audit.get('fit_summary', {}).get('requested_max_abs_delta_rgb', 0.0)])}`",
        f"- policy candidate dominance pruning: `{policy_candidate_control.get('dominance_pruning_enabled', False)}`",
        f"- policy candidates planned before pruning: `{policy_candidate_control.get('planned_candidate_count_before_pruning', 0)}`",
        f"- policy candidates planned after pruning: `{policy_candidate_control.get('planned_candidate_count_after_pruning', 0)}`",
        f"- policy candidates executed: `{policy_candidate_control.get('executed_candidate_count', 0)}`",
        f"- policy candidate early-stop mode: `{policy_candidate_control.get('early_stop_effective_mode', 'none')}`",
        f"- policy candidate early-stop skipped: `{policy_candidate_control.get('early_stop_skipped_count', 0)}`",
        f"- adaptive texture-size ladder enabled: `{adaptive_ladder.get('enabled', False)}`",
        f"- adaptive texture-size ladder reason: `{adaptive_ladder.get('reason', 'not_requested')}`",
        f"- adaptive texture-size ladder base candidates: `{adaptive_ladder.get('base_texture_size_candidates', [])}`",
        f"- adaptive texture-size ladder planned candidates: `{adaptive_ladder.get('planned_texture_size_candidates', [])}`",
        f"- adaptive texture-size ladder added candidates: `{adaptive_ladder.get('added_texture_size_candidates', [])}`",
        f"- adaptive texture-size ladder rejected candidates: `{adaptive_ladder.get('rejected_texture_size_candidates', [])}`",
        f"- adaptive texture-size ladder fit samples: `{adaptive_ladder.get('total_fit_samples', 0)}`",
        f"- adaptive texture-size ladder skipped policy-val views: `{adaptive_ladder.get('skipped_policy_val_views', 0)}`",
        f"- adaptive texture-size ladder support modes: `{adaptive_ladder.get('support_modes', [])}`",
        f"- adaptive texture-size ladder support union sha1: `{adaptive_ladder.get('support_union_faces_sha1', '')}`",
        f"- adaptive texture-size ladder final selection scope: `{adaptive_ladder.get('final_texture_size_selection_scope', '')}`",
        f"- surface multiscale prior mode: `{multiscale_prior.get('mode', 'none')}`",
        f"- surface multiscale prior selected blend: `{float(audit.get('fit_summary', {}).get('selected_surface_multiscale_prior_blend', multiscale_prior.get('blend', 0.0)) or 0.0):.6f}`",
        f"- surface multiscale prior blend candidates: `{audit.get('fit_summary', {}).get('surface_multiscale_prior_blend_candidates', [multiscale_prior.get('blend', 0.0)])}`",
        f"- surface multiscale prior gate mode: `{multiscale_prior.get('gate_mode', 'none')}`",
        f"- surface multiscale prior block sizes: `{multiscale_prior.get('block_sizes', [])}`",
        f"- surface multiscale prior blended bins: `{multiscale_prior.get('blended_bins', 0)}`",
        f"- surface multiscale prior blended-bin fraction: `{float(multiscale_prior.get('blended_bin_fraction', 0.0) or 0.0):.6f}`",
        f"- surface multiscale prior mean blend weight: `{float(multiscale_prior.get('mean_blend_weight', 0.0) or 0.0):.6f}`",
        f"- surface multiscale prior gate rejected bins: `{int(multiscale_prior.get('gate_rejected_bins', 0) or 0)}`",
        f"- surface multiscale prior empty-bin rejects: `{int(multiscale_prior.get('empty_rejected_bins', 0) or 0)}`",
        f"- surface multiscale prior sign rejects: `{int(multiscale_prior.get('sign_rejected_bins', 0) or 0)}`",
        f"- surface multiscale prior variance rejects: `{int(multiscale_prior.get('variance_rejected_bins', 0) or 0)}`",
        f"- surface multiscale prior cosine rejects: `{int(multiscale_prior.get('cosine_rejected_bins', 0) or 0)}`",
        f"- view-conditioned basis mode: `{audit.get('fit_summary', {}).get('view_conditioned_basis', {}).get('mode', 'none')}`",
        f"- view-conditioned basis effective mode: `{audit.get('fit_summary', {}).get('view_conditioned_basis', {}).get('effective_mode', audit.get('fit_summary', {}).get('view_conditioned_basis', {}).get('mode', 'none'))}`",
        f"- view-conditioned basis guard decision: `{audit.get('fit_summary', {}).get('view_conditioned_basis', {}).get('guard', {}).get('decision', 'not_requested')}`",
        f"- view-conditioned basis supported bins: `{audit.get('fit_summary', {}).get('view_conditioned_basis', {}).get('supported_bins', 0)}`",
        f"- view-conditioned basis supported-bin fraction: `{audit.get('fit_summary', {}).get('view_conditioned_basis', {}).get('supported_bin_fraction', 0.0):.6f}`",
        f"- view-conditioned basis OOD mode: `{audit.get('fit_summary', {}).get('view_conditioned_basis', {}).get('ood_mode', 'none')}`",
        f"- view-conditioned basis OOD max-z: `{audit.get('fit_summary', {}).get('view_conditioned_basis', {}).get('ood_max_z', 0.0)}`",
        f"- view-conditioned basis OOD min-std: `{audit.get('fit_summary', {}).get('view_conditioned_basis', {}).get('ood_min_std', 0.0)}`",
        f"- view-cluster experts enabled: `{audit.get('fit_summary', {}).get('view_cluster_experts', {}).get('enabled', False)}`",
        f"- view-cluster expert count: `{audit.get('fit_summary', {}).get('view_cluster_experts', {}).get('expert_count', 1)}`",
        f"- view-cluster expert supported bins: `{audit.get('fit_summary', {}).get('view_cluster_experts', {}).get('supported_bins', 0)}`",
        f"- view-cluster expert supported-bin fraction: `{audit.get('fit_summary', {}).get('view_cluster_experts', {}).get('supported_bin_fraction', 0.0):.6f}`",
        f"- teacher-distilled basis mode: `{teacher_basis.get('mode', 'none')}`",
        f"- teacher-distilled basis effective mode: `{teacher_basis.get('effective_mode', teacher_basis.get('mode', 'none'))}`",
        f"- teacher-distilled basis guard decision: `{teacher_basis.get('guard', {}).get('decision', 'not_requested')}`",
        f"- teacher-distilled basis requested min-face samples: `{teacher_basis.get('min_face_samples', 0)}`",
        f"- teacher-distilled basis effective min-face samples: `{teacher_basis.get('effective_min_face_samples', teacher_basis.get('min_face_samples', 0))}`",
        f"- adaptive low-support teacher basis enabled: `{teacher_basis.get('adaptive_low_support', {}).get('enabled', False)}`",
        f"- adaptive low-support teacher basis reason: `{teacher_basis.get('adaptive_low_support', {}).get('reason', 'not_requested')}`",
        f"- adaptive low-support teacher basis newly supported faces: `{teacher_basis.get('adaptive_low_support', {}).get('newly_supported_faces', 0)}`",
        f"- adaptive low-support teacher basis low-support solved faces: `{teacher_basis.get('adaptive_low_support', {}).get('low_support_supported_faces', 0)}`",
        f"- adaptive low-support teacher basis max ridge multiplier: `{float(teacher_basis.get('adaptive_low_support', {}).get('max_ridge_multiplier', 1.0) or 1.0):.6f}`",
        f"- teacher-distilled basis supported faces: `{teacher_basis.get('supported_faces', 0)}`",
        f"- teacher-distilled basis supported-face fraction: `{teacher_basis.get('supported_face_fraction', 0.0):.6f}`",
        f"- teacher-distilled basis apply mode: `{teacher_basis.get('apply_mode', '')}`",
        f"- teacher-distilled basis blend: `{teacher_basis.get('blend', 0.0)}`",
        f"- policy-val enabled: `{val.get('enabled', False)}`",
        f"- policy-val samples: `{val.get('samples', 0)}`",
        f"- selected alpha: `{audit.get('selected_alpha', 0.0)}`",
        f"- local alpha calibration: `{audit.get('fit_summary', {}).get('local_alpha_calibration', {}).get('enabled', False)}`",
        f"- local alpha mode: `{local_alpha.get('mode', 'disabled')}`",
        f"- local alpha optimizer: `{local_alpha.get('optimizer', 'n/a')}`",
        f"- local alpha fallback alpha: `{fallback_alpha_text}`",
        f"- local alpha fallback mode: `{local_alpha.get('fallback_mode', 'n/a')}`",
        f"- local alpha face count: `{local_alpha.get('face_alpha_count', 0)}`",
        f"- local alpha bin count: `{local_alpha.get('bin_alpha_count', 0)}`",
        f"- local alpha bin RGB count: `{local_alpha.get('bin_rgb_alpha_count', 0)}`",
        f"- local alpha image-L1 optimizer uses target/test GT: `{local_alpha.get('uses_target_or_test_gt', 'n/a')}`",
        f"- local alpha image-L1 optimizer global rel gain: `{float(local_alpha.get('global_relative_gain', 0.0) or 0.0):.9f}`",
        f"- local alpha image-L1 optimizer mean selected rel gain: `{float(local_alpha.get('mean_selected_relative_gain', 0.0) or 0.0):.9f}`",
        f"- local alpha uncertainty-shrink bin count: `{local_alpha.get('bin_uncertainty_shrink_count', 0)}`",
        f"- local alpha uncertainty-shrink policy mode: `{local_alpha.get('uncertainty_shrink_policy_mode', 'n/a')}`",
        f"- local alpha view-cluster local shrink: `{local_alpha.get('view_cluster_local_shrink', False)}`",
        f"- local alpha view-cluster selected clusters: `{local_alpha.get('view_cluster_selected_cluster_count', 0)}`",
        f"- local alpha view-cluster policy-val view counts: `{local_alpha.get('view_cluster_policy_val_view_counts', {})}`",
        f"- local alpha image-L1 bin certificate enabled: `{local_alpha.get('image_l1_bin_certificate', {}).get('enabled', False)}`",
        f"- local alpha image-L1 bin certificate mode: `{local_alpha.get('image_l1_bin_certificate', {}).get('mode', 'n/a')}`",
        f"- local alpha image-L1 bin certificate pool radius: `{local_alpha.get('image_l1_bin_certificate', {}).get('pool_radius', 0)}`",
        f"- local alpha image-L1 bin certificate candidate ok bins: `{local_alpha.get('image_l1_bin_certificate', {}).get('candidate_evidence_ok_count', 0)}`",
        f"- local alpha image-L1 bin certificate selected ok bins: `{local_alpha.get('image_l1_bin_certificate', {}).get('selected_evidence_ok_count', 0)}`",
        f"- local alpha image-L1 bin certificate mean selected rel gain: `{float(local_alpha.get('image_l1_bin_certificate', {}).get('mean_selected_relative_gain', 0.0) or 0.0):.9f}`",
        f"- local alpha image-L1 region expansion enabled: `{local_alpha.get('image_l1_bin_certificate', {}).get('region_expansion', {}).get('enabled', False)}`",
        f"- local alpha image-L1 region expansion seeds: `{local_alpha.get('image_l1_bin_certificate', {}).get('region_expansion', {}).get('seed_bin_count', 0)}`",
        f"- local alpha image-L1 region expansion pre-trunc bins: `{local_alpha.get('image_l1_bin_certificate', {}).get('region_expansion', {}).get('expanded_bin_count_pre_trunc', 0)}`",
        f"- local alpha image-L1 region expansion selected bins: `{local_alpha.get('image_l1_bin_certificate', {}).get('region_expansion', {}).get('selected_expanded_bin_count', 0)}`",
        f"- local alpha uncertainty-shrink min bin views: `{local_alpha.get('min_bin_views', 0)}`",
        f"- local alpha uncertainty-shrink mean: `{float(local_alpha.get('mean_selected_shrink', 0.0) or 0.0):.6f}`",
        f"- local alpha uncertainty-shrink downweighted bins: `{local_alpha.get('downweighted_bin_count', 0)}`",
        f"- local alpha uncertainty-shrink upweighted bins: `{local_alpha.get('upweighted_bin_count', 0)}`",
        f"- local alpha candidate bins: `{local_alpha.get('candidate_bin_count', 0)}`",
        f"- local alpha fallback bins: `{local_alpha.get('fallback_bin_count', 0)}`",
        f"- face gain guard enabled: `{face_gain.get('enabled', False)}`",
        f"- face gain guard decision: `{face_gain.get('decision', 'not_requested')}`",
        f"- face gain guard allowed faces: `{face_gain.get('allowed_face_count', 0)}`",
        f"- face gain guard rejected faces: `{face_gain.get('rejected_face_count', 0)}`",
        f"- face gain guard allowed sample fraction: `{face_gain.get('allowed_sample_fraction', 0.0):.6f}`",
        f"- bin uncertainty guard enabled: `{bin_uncertainty.get('enabled', False)}`",
        f"- bin uncertainty guard decision: `{bin_uncertainty.get('decision', 'not_requested')}`",
        f"- bin uncertainty guard allowed bins: `{bin_uncertainty.get('allowed_bin_count', 0)}`",
        f"- bin uncertainty guard rejected bins: `{bin_uncertainty.get('rejected_bin_count', 0)}`",
        f"- bin uncertainty guard allowed faces: `{bin_uncertainty.get('allowed_face_count', 0)}`",
        f"- bin uncertainty guard allowed sample fraction: `{float(bin_uncertainty.get('allowed_sample_fraction', 0.0) or 0.0):.6f}`",
        f"- train-only target-impact residual basis enabled: `{bin_uncertainty.get('target_impact_residual_basis', {}).get('enabled', False)}`",
        f"- train-only target-impact residual basis candidates: `{bin_uncertainty.get('target_impact_residual_basis', {}).get('candidate_bin_count', 0)}`",
        f"- train-only target-impact residual basis added bins: `{bin_uncertainty.get('target_impact_residual_basis', {}).get('added_bin_count', 0)}`",
        f"- train-only target-impact residual basis added no-policy-row bins: `{bin_uncertainty.get('target_impact_residual_basis', {}).get('added_without_policy_row_bin_count', 0)}`",
        f"- train-only target-impact residual basis added target pixels: `{bin_uncertainty.get('target_impact_residual_basis', {}).get('added_target_pixels', 0)}`",
        f"- sparse target-connected growth enabled: `{bin_uncertainty.get('target_connected_region_growth', {}).get('enabled', False)}`",
        f"- sparse target-connected growth seeds: `{bin_uncertainty.get('target_connected_region_growth', {}).get('seed_allowed_bin_count', 0)}`",
        f"- sparse target-connected growth candidates: `{bin_uncertainty.get('target_connected_region_growth', {}).get('candidate_bin_count', 0)}`",
        f"- sparse target-connected growth added bins: `{bin_uncertainty.get('target_connected_region_growth', {}).get('added_bin_count', 0)}`",
        f"- sparse target-connected growth added target pixels: `{bin_uncertainty.get('target_connected_region_growth', {}).get('added_target_pixels', 0)}`",
        f"- view consistency confidence enabled: `{view_confidence.get('enabled', False)}`",
        f"- view consistency confidence decision: `{view_confidence.get('decision', 'not_requested')}`",
        f"- view consistency confidence positive views: `{view_confidence.get('positive_view_count', 0)}`",
        f"- view consistency confidence positive-view fraction: `{float(view_confidence.get('positive_view_fraction', 0.0) or 0.0):.6f}`",
        f"- view consistency confidence kernel sigma: `{float(view_confidence.get('kernel_sigma', 0.0) or 0.0):.6f}`",
        f"- view consistency confidence min confidence: `{float(view_confidence.get('min_confidence', 0.0) or 0.0):.6f}`",
        f"- view consistency confidence post accepted: `{view_confidence.get('post_confidence_accepted', False)}`",
        f"- view alpha cap enabled: `{view_alpha_cap.get('enabled', False)}`",
        f"- view alpha cap decision: `{view_alpha_cap.get('decision', 'not_requested')}`",
        f"- view alpha cap selected views: `{view_alpha_cap.get('selected_view_count', 0)}`",
        f"- view alpha cap fallback views: `{view_alpha_cap.get('fallback_view_count', 0)}`",
        f"- view alpha cap post accepted: `{view_alpha_cap.get('post_cap_accepted', False)}`",
        f"- view alpha cap post selected alpha: `{float(view_alpha_cap.get('post_cap_selected_alpha', 0.0) or 0.0):.6f}`",
        f"- policy-val relative gain: `{val.get('best', {}).get('relative_gain', 0.0):.6f}`",
        f"- policy-val positive-view fraction: `{val.get('best', {}).get('positive_view_fraction', 0.0):.6f}`",
        f"- policy-val CVaR20 view relative gain: `{val.get('best', {}).get('cvar20_view_relative_gain', 0.0):.6f}`",
        f"- policy-val min-view relative gain: `{val.get('best', {}).get('min_view_relative_gain', 0.0):.6f}`",
        f"- policy-val image SSIM gain: `{val.get('best', {}).get('ssim_gain', 0.0):.9f}`",
        f"- policy-val image SSIM positive-view fraction: `{val.get('best', {}).get('ssim_positive_view_fraction', 0.0):.6f}`",
        f"- policy-val image SSIM min-view gain: `{val.get('best', {}).get('ssim_min_view_gain', 0.0):.9f}`",
        f"- policy-val image L1 gain: `{val.get('best', {}).get('image_l1_gain', 0.0):.9f}`",
        f"- policy-val image L1 positive-view fraction: `{val.get('best', {}).get('image_l1_positive_view_fraction', 0.0):.6f}`",
        f"- policy-val image L1 min-view gain: `{val.get('best', {}).get('image_l1_min_view_gain', 0.0):.9f}`",
        f"- policy-val image LPIPS gain: `{val.get('best', {}).get('lpips_gain', 0.0):.9f}`",
        f"- policy-val image LPIPS positive-view fraction: `{val.get('best', {}).get('lpips_positive_view_fraction', 0.0):.6f}`",
        f"- policy-val image LPIPS min-view gain: `{val.get('best', {}).get('lpips_min_view_gain', 0.0):.9f}`",
        f"- policy-val risk gate: `{risk.get('passed', True)}`",
        f"- target written views: `{target.get('written_views', 0)}`",
        f"- target changed fraction: `{target.get('changed_fraction', 0.0):.6f}`",
        f"- effective policy: `{audit.get('effective_policy', '')}`",
        f"- target coverage gate: `{audit.get('target_coverage_gate', {}).get('passed', True)}`",
        f"- reject reason: `{audit.get('reject_reason', '')}`",
        "",
        "## Alpha Rows",
        "",
        "| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | LPIPS gain | LPIPS pos frac | LPIPS min gain | mse before | mse after |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in val.get("rows", []) or []:
        lines.append(
            f"| {float(row.get('alpha', 0.0)):.4f} | {float(row.get('relative_gain', 0.0)):.6f} | "
            f"{float(row.get('positive_view_fraction', 0.0)):.6f} | "
            f"{float(row.get('cvar20_view_relative_gain', 0.0)):.6f} | "
            f"{float(row.get('min_view_relative_gain', 0.0)):.6f} | "
            f"{float(row.get('ssim_gain', 0.0)):.9f} | "
            f"{float(row.get('ssim_positive_view_fraction', 0.0)):.6f} | "
            f"{float(row.get('ssim_min_view_gain', 0.0)):.9f} | "
            f"{float(row.get('image_l1_gain', 0.0)):.9f} | "
            f"{float(row.get('image_l1_positive_view_fraction', 0.0)):.6f} | "
            f"{float(row.get('image_l1_min_view_gain', 0.0)):.9f} | "
            f"{float(row.get('lpips_gain', 0.0)):.9f} | "
            f"{float(row.get('lpips_positive_view_fraction', 0.0)):.6f} | "
            f"{float(row.get('lpips_min_view_gain', 0.0)):.9f} | "
            f"{float(row.get('mse_before', 0.0)):.8f} | {float(row.get('mse_after', 0.0)):.8f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def policy_val_risk_reasons(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    selective_view_policy = bool(
        row.get("view_confidence_selective", False) or row.get("view_alpha_cap_selective", False)
        or row.get("sparse_materialization_selective", False)
    )
    pos_frac_key = "nonnegative_view_fraction" if selective_view_policy else "positive_view_fraction"
    pos_frac = float(row.get(pos_frac_key, row.get("positive_view_fraction", 0.0)) or 0.0)
    min_pos_frac = float(args.min_policy_val_positive_view_fraction)
    if min_pos_frac > 0.0 and pos_frac < min_pos_frac:
        reasons.append(
            f"{pos_frac_key} {pos_frac:.6f} < min_policy_val_positive_view_fraction {min_pos_frac:.6f}"
        )
    cvar20 = float(row.get("cvar20_view_relative_gain", 0.0))
    min_cvar20 = float(args.min_policy_val_cvar20_relative_gain)
    if min_cvar20 > -1.0 and cvar20 < min_cvar20:
        reasons.append(
            f"cvar20_view_relative_gain {cvar20:.6f} < min_policy_val_cvar20_relative_gain {min_cvar20:.6f}"
        )
    min_view_gain = float(row.get("min_view_relative_gain", 0.0))
    min_allowed_view_gain = float(args.min_policy_val_min_view_relative_gain)
    if min_allowed_view_gain > -1.0 and min_view_gain < min_allowed_view_gain:
        reasons.append(
            f"min_view_relative_gain {min_view_gain:.6f} < min_policy_val_min_view_relative_gain {min_allowed_view_gain:.6f}"
        )
    if bool(getattr(args, "enable_policy_val_image_ssim_gate", False)):
        ssim_mean_gain = float(row.get("ssim_gain", 0.0))
        min_ssim_mean_gain = float(args.min_policy_val_ssim_mean_gain)
        if min_ssim_mean_gain > -1.0 and ssim_mean_gain < min_ssim_mean_gain:
            reasons.append(
                f"ssim_gain {ssim_mean_gain:.9f} < min_policy_val_ssim_mean_gain {min_ssim_mean_gain:.9f}"
            )
        ssim_pos_frac_key = (
            "ssim_nonnegative_view_fraction" if selective_view_policy else "ssim_positive_view_fraction"
        )
        ssim_pos_frac = float(row.get(ssim_pos_frac_key, row.get("ssim_positive_view_fraction", 0.0)) or 0.0)
        min_ssim_pos_frac = float(args.min_policy_val_ssim_positive_view_fraction)
        if min_ssim_pos_frac > 0.0 and ssim_pos_frac < min_ssim_pos_frac:
            reasons.append(
                f"{ssim_pos_frac_key} {ssim_pos_frac:.6f} < "
                f"min_policy_val_ssim_positive_view_fraction {min_ssim_pos_frac:.6f}"
            )
        ssim_min_gain = float(row.get("ssim_min_view_gain", 0.0))
        min_ssim_min_gain = float(args.min_policy_val_ssim_min_view_gain)
        if min_ssim_min_gain > -1.0 and ssim_min_gain < min_ssim_min_gain:
            reasons.append(
                f"ssim_min_view_gain {ssim_min_gain:.9f} < "
                f"min_policy_val_ssim_min_view_gain {min_ssim_min_gain:.9f}"
            )
    if bool(getattr(args, "enable_policy_val_image_l1_gate", False)):
        l1_mean_gain = float(row.get("image_l1_gain", 0.0))
        min_l1_mean_gain = float(args.min_policy_val_l1_mean_gain)
        if min_l1_mean_gain > -1.0 and l1_mean_gain < min_l1_mean_gain:
            reasons.append(
                f"image_l1_gain {l1_mean_gain:.9f} < min_policy_val_l1_mean_gain {min_l1_mean_gain:.9f}"
            )
        l1_pos_frac_key = (
            "image_l1_nonnegative_view_fraction"
            if selective_view_policy
            else "image_l1_positive_view_fraction"
        )
        l1_pos_frac = float(row.get(l1_pos_frac_key, row.get("image_l1_positive_view_fraction", 0.0)) or 0.0)
        min_l1_pos_frac = float(args.min_policy_val_l1_positive_view_fraction)
        if min_l1_pos_frac > 0.0 and l1_pos_frac < min_l1_pos_frac:
            reasons.append(
                f"{l1_pos_frac_key} {l1_pos_frac:.6f} < "
                f"min_policy_val_l1_positive_view_fraction {min_l1_pos_frac:.6f}"
            )
        l1_min_gain = float(row.get("image_l1_min_view_gain", 0.0))
        min_l1_min_gain = float(args.min_policy_val_l1_min_view_gain)
        if min_l1_min_gain > -1.0 and l1_min_gain < min_l1_min_gain:
            reasons.append(
                f"image_l1_min_view_gain {l1_min_gain:.9f} < "
                f"min_policy_val_l1_min_view_gain {min_l1_min_gain:.9f}"
            )
        l1_cvar20 = float(row.get("image_l1_cvar20_view_gain", 0.0))
        min_l1_cvar20 = float(args.min_policy_val_l1_cvar20_view_gain)
        if min_l1_cvar20 > -1.0 and l1_cvar20 < min_l1_cvar20:
            reasons.append(
                f"image_l1_cvar20_view_gain {l1_cvar20:.9f} < "
                f"min_policy_val_l1_cvar20_view_gain {min_l1_cvar20:.9f}"
            )
    if bool(getattr(args, "enable_policy_val_image_lpips_gate", False)):
        lpips_mean_gain = float(row.get("lpips_gain", 0.0))
        min_lpips_mean_gain = float(args.min_policy_val_lpips_mean_gain)
        if min_lpips_mean_gain > -1.0 and lpips_mean_gain < min_lpips_mean_gain:
            reasons.append(
                f"lpips_gain {lpips_mean_gain:.9f} < min_policy_val_lpips_mean_gain {min_lpips_mean_gain:.9f}"
            )
        lpips_pos_frac_key = (
            "lpips_nonnegative_view_fraction" if selective_view_policy else "lpips_positive_view_fraction"
        )
        lpips_pos_frac = float(row.get(lpips_pos_frac_key, row.get("lpips_positive_view_fraction", 0.0)) or 0.0)
        min_lpips_pos_frac = float(args.min_policy_val_lpips_positive_view_fraction)
        if min_lpips_pos_frac > 0.0 and lpips_pos_frac < min_lpips_pos_frac:
            reasons.append(
                f"{lpips_pos_frac_key} {lpips_pos_frac:.6f} < "
                f"min_policy_val_lpips_positive_view_fraction {min_lpips_pos_frac:.6f}"
            )
        lpips_min_gain = float(row.get("lpips_min_view_gain", 0.0))
        min_lpips_min_gain = float(args.min_policy_val_lpips_min_view_gain)
        if min_lpips_min_gain > -1.0 and lpips_min_gain < min_lpips_min_gain:
            reasons.append(
                f"lpips_min_view_gain {lpips_min_gain:.9f} < "
                f"min_policy_val_lpips_min_view_gain {min_lpips_min_gain:.9f}"
            )
        lpips_cvar20 = float(row.get("lpips_cvar20_view_gain", 0.0))
        min_lpips_cvar20 = float(args.min_policy_val_lpips_cvar20_view_gain)
        if min_lpips_cvar20 > -1.0 and lpips_cvar20 < min_lpips_cvar20:
            reasons.append(
                f"lpips_cvar20_view_gain {lpips_cvar20:.9f} < "
                f"min_policy_val_lpips_cvar20_view_gain {min_lpips_cvar20:.9f}"
            )
    if bool(getattr(args, "enable_policy_val_effective_margin_gate", False)):
        effective_relative = float(row.get("relative_gain", 0.0))
        min_effective_relative = float(args.min_policy_val_effective_relative_gain)
        if min_effective_relative > -1.0 and effective_relative < min_effective_relative:
            reasons.append(
                f"effective_relative_gain {effective_relative:.9f} < "
                f"min_policy_val_effective_relative_gain {min_effective_relative:.9f}"
            )
        effective_ssim = float(row.get("ssim_gain", 0.0))
        min_effective_ssim = float(args.min_policy_val_effective_ssim_gain)
        if min_effective_ssim > -1.0 and effective_ssim < min_effective_ssim:
            reasons.append(
                f"effective_ssim_gain {effective_ssim:.9f} < "
                f"min_policy_val_effective_ssim_gain {min_effective_ssim:.9f}"
            )
        effective_l1 = float(row.get("image_l1_gain", 0.0))
        min_effective_l1 = float(args.min_policy_val_effective_l1_gain)
        if min_effective_l1 > -1.0 and effective_l1 < min_effective_l1:
            reasons.append(
                f"effective_image_l1_gain {effective_l1:.9f} < "
                f"min_policy_val_effective_l1_gain {min_effective_l1:.9f}"
            )
        effective_ssim_cvar20 = float(row.get("ssim_cvar20_view_gain", 0.0))
        min_effective_ssim_cvar20 = float(args.min_policy_val_effective_ssim_cvar20_gain)
        if min_effective_ssim_cvar20 > -1.0 and effective_ssim_cvar20 < min_effective_ssim_cvar20:
            reasons.append(
                f"effective_ssim_cvar20_view_gain {effective_ssim_cvar20:.9f} < "
                f"min_policy_val_effective_ssim_cvar20_gain {min_effective_ssim_cvar20:.9f}"
            )
        effective_l1_cvar20 = float(row.get("image_l1_cvar20_view_gain", 0.0))
        min_effective_l1_cvar20 = float(args.min_policy_val_effective_l1_cvar20_gain)
        if min_effective_l1_cvar20 > -1.0 and effective_l1_cvar20 < min_effective_l1_cvar20:
            reasons.append(
                f"effective_image_l1_cvar20_view_gain {effective_l1_cvar20:.9f} < "
                f"min_policy_val_effective_l1_cvar20_gain {min_effective_l1_cvar20:.9f}"
            )
        effective_lpips = float(row.get("lpips_gain", 0.0))
        min_effective_lpips = float(args.min_policy_val_effective_lpips_gain)
        if min_effective_lpips > -1.0 and effective_lpips < min_effective_lpips:
            reasons.append(
                f"effective_lpips_gain {effective_lpips:.9f} < "
                f"min_policy_val_effective_lpips_gain {min_effective_lpips:.9f}"
            )
        effective_lpips_cvar20 = float(row.get("lpips_cvar20_view_gain", 0.0))
        min_effective_lpips_cvar20 = float(args.min_policy_val_effective_lpips_cvar20_gain)
        if min_effective_lpips_cvar20 > -1.0 and effective_lpips_cvar20 < min_effective_lpips_cvar20:
            reasons.append(
                f"effective_lpips_cvar20_view_gain {effective_lpips_cvar20:.9f} < "
                f"min_policy_val_effective_lpips_cvar20_gain {min_effective_lpips_cvar20:.9f}"
            )
    return reasons


def select_policy_val_frontier_row(
    safe_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    best_by_relative = max(safe_rows, key=lambda row: float(row.get("relative_gain", -1.0)))
    if not bool(getattr(args, "enable_policy_val_alpha_frontier_selection", False)):
        return best_by_relative, {
            "frontier_enabled": False,
            "frontier_reason": "disabled",
            "best_by_relative_alpha": float(best_by_relative.get("alpha", 0.0)),
        }

    best_relative_gain = float(best_by_relative.get("relative_gain", 0.0))
    best_ssim_gain = max(float(row.get("ssim_gain", 0.0)) for row in safe_rows)
    best_l1_gain = max(float(row.get("image_l1_gain", 0.0)) for row in safe_rows)
    relative_threshold = float(args.min_policy_val_relative_gain)
    if best_relative_gain > relative_threshold:
        relative_threshold = max(
            relative_threshold,
            float(args.policy_val_alpha_frontier_min_relative_fraction) * best_relative_gain,
        )
    ssim_threshold = -math.inf
    if best_ssim_gain > 0.0:
        ssim_threshold = float(args.policy_val_alpha_frontier_min_ssim_fraction) * best_ssim_gain
    l1_threshold = -math.inf
    if best_l1_gain > 0.0:
        l1_threshold = float(args.policy_val_alpha_frontier_min_l1_fraction) * best_l1_gain

    eligible = [
        dict(row)
        for row in safe_rows
        if float(row.get("relative_gain", -1.0)) >= relative_threshold
        and float(row.get("ssim_gain", 0.0)) >= ssim_threshold
        and float(row.get("image_l1_gain", 0.0)) >= l1_threshold
    ]
    if not eligible:
        return best_by_relative, {
            "frontier_enabled": True,
            "frontier_reason": "no_frontier_eligible_rows",
            "best_by_relative_alpha": float(best_by_relative.get("alpha", 0.0)),
            "selected_alpha": float(best_by_relative.get("alpha", 0.0)),
            "eligible_alpha_count": 0,
            "thresholds": {
                "relative_gain": float(relative_threshold),
                "ssim_gain": None if not math.isfinite(ssim_threshold) else float(ssim_threshold),
                "image_l1_gain": None if not math.isfinite(l1_threshold) else float(l1_threshold),
            },
        }

    mode = str(getattr(args, "policy_val_alpha_frontier_mode", "smallest_effective"))
    if mode in {"knee", "tail_knee"}:
        best_axis_gains = {
            "relative_gain": float(best_relative_gain),
            "ssim_gain": float(best_ssim_gain),
            "image_l1_gain": float(best_l1_gain),
        }
        active_axes = [axis for axis, best_gain in best_axis_gains.items() if float(best_gain) > 0.0]
        tail_axes = [
            "min_view_relative_gain",
            "cvar20_view_relative_gain",
            "ssim_min_view_gain",
            "ssim_cvar20_view_gain",
            "image_l1_min_view_gain",
            "image_l1_cvar20_view_gain",
        ]

        def normalized_score(row: dict[str, Any]) -> float:
            if not active_axes:
                return float(row.get("relative_gain", 0.0))
            score_terms = []
            for axis in active_axes:
                best_gain = max(float(best_axis_gains[axis]), 1.0e-12)
                score_terms.append(max(0.0, min(1.0, float(row.get(axis, 0.0)) / best_gain)))
            return float(sum(score_terms) / max(len(score_terms), 1))

        sorted_safe = sorted(safe_rows, key=lambda row: float(row.get("alpha", 0.0)))
        knee_profiles: list[dict[str, Any]] = []
        prev_alpha = 0.0
        prev_score = 0.0
        for row in sorted_safe:
            alpha = float(row.get("alpha", 0.0))
            score_value = normalized_score(row)
            delta_alpha = max(alpha - prev_alpha, 1.0e-12)
            marginal_slope = max(0.0, score_value - prev_score) / delta_alpha
            knee_profiles.append(
                {
                    "alpha": alpha,
                    "score": float(score_value),
                    "marginal_slope": float(marginal_slope),
                    "relative_gain": float(row.get("relative_gain", 0.0)),
                    "ssim_gain": float(row.get("ssim_gain", 0.0)),
                    "image_l1_gain": float(row.get("image_l1_gain", 0.0)),
                }
            )
            prev_alpha = alpha
            prev_score = score_value

        min_score = float(getattr(args, "policy_val_alpha_frontier_knee_min_score_fraction", 0.55))
        slope_drop = float(getattr(args, "policy_val_alpha_frontier_knee_slope_drop_fraction", 0.85))
        tail_min_score = float(
            getattr(args, "policy_val_alpha_frontier_tail_knee_min_score_fraction", 0.70)
        )
        tail_min_regression_count = int(
            getattr(args, "policy_val_alpha_frontier_tail_knee_min_regression_count", 3)
        )
        tail_eps = float(getattr(args, "policy_val_alpha_frontier_tail_knee_eps", 1.0e-10))
        selected = None
        frontier_reason = "knee_not_found_fallback_to_smallest_effective"
        best_prior_slope = 0.0
        for idx, row in enumerate(sorted_safe[:-1]):
            profile = knee_profiles[idx]
            best_prior_slope = max(best_prior_slope, float(profile["marginal_slope"]))
            next_slope = float(knee_profiles[idx + 1]["marginal_slope"])
            slope_ratio = next_slope / max(best_prior_slope, 1.0e-12)
            profile["next_marginal_slope"] = float(next_slope)
            profile["next_slope_ratio"] = float(slope_ratio)
            next_row = sorted_safe[idx + 1]
            regressed_tail_axes = [
                axis
                for axis in tail_axes
                if float(next_row.get(axis, 0.0)) < float(row.get(axis, 0.0)) - tail_eps
            ]
            profile["next_tail_regressed_axes"] = regressed_tail_axes
            profile["next_tail_regression_count"] = int(len(regressed_tail_axes))

        if mode == "tail_knee":
            for idx, row in enumerate(sorted_safe[:-1]):
                profile = knee_profiles[idx]
                if (
                    float(profile["score"]) >= tail_min_score
                    and int(profile.get("next_tail_regression_count", 0)) >= tail_min_regression_count
                ):
                    selected = row
                    frontier_reason = "selected_tail_knee_before_robust_tail_regression"
                    break
        if selected is None:
            for idx, row in enumerate(sorted_safe[:-1]):
                profile = knee_profiles[idx]
                slope_ratio = float(profile.get("next_slope_ratio", 1.0))
                if float(profile["score"]) >= min_score and slope_ratio <= slope_drop:
                    selected = row
                    frontier_reason = "selected_knee_before_diminishing_returns"
                    break
        if selected is None:
            selected = min(
                eligible,
                key=lambda row: (
                    float(row.get("alpha", 0.0)),
                    -float(row.get("relative_gain", 0.0)),
                    -float(row.get("ssim_gain", 0.0)),
                    -float(row.get("image_l1_gain", 0.0)),
                ),
            )
    elif mode == "best_score":
        max_alpha = max(float(row.get("alpha", 0.0)) for row in safe_rows)
        max_alpha = max(max_alpha, 1.0e-12)

        def score(row: dict[str, Any]) -> tuple[float, float]:
            rel_norm = float(row.get("relative_gain", 0.0)) / max(best_relative_gain, 1.0e-12)
            ssim_norm = (
                float(row.get("ssim_gain", 0.0)) / max(best_ssim_gain, 1.0e-12)
                if best_ssim_gain > 0.0
                else 0.0
            )
            l1_norm = (
                float(row.get("image_l1_gain", 0.0)) / max(best_l1_gain, 1.0e-12)
                if best_l1_gain > 0.0
                else 0.0
            )
            alpha_penalty = float(args.policy_val_alpha_frontier_alpha_penalty) * (
                float(row.get("alpha", 0.0)) / max_alpha
            )
            return (rel_norm + ssim_norm + l1_norm - alpha_penalty, -float(row.get("alpha", 0.0)))

        selected = max(eligible, key=score)
    elif mode == "smallest_effective":
        selected = min(
            eligible,
            key=lambda row: (
                float(row.get("alpha", 0.0)),
                -float(row.get("relative_gain", 0.0)),
                -float(row.get("ssim_gain", 0.0)),
                -float(row.get("image_l1_gain", 0.0)),
            ),
        )
    else:
        raise ValueError(f"unsupported policy_val_alpha_frontier_mode: {mode}")

    frontier_preview = sorted(
        eligible,
        key=lambda row: float(row.get("alpha", 0.0)),
    )[:16]
    selection_payload = {
        "frontier_enabled": True,
        "frontier_reason": (
            frontier_reason
            if mode in {"knee", "tail_knee"}
            else "selected_smallest_effective_alpha"
            if mode == "smallest_effective"
            else "selected_best_score"
        ),
        "frontier_mode": mode,
        "best_by_relative_alpha": float(best_by_relative.get("alpha", 0.0)),
        "best_by_relative_gain": float(best_relative_gain),
        "selected_alpha": float(selected.get("alpha", 0.0)),
        "selected_relative_gain": float(selected.get("relative_gain", 0.0)),
        "eligible_alpha_count": int(len(eligible)),
        "thresholds": {
            "relative_gain": float(relative_threshold),
            "ssim_gain": None if not math.isfinite(ssim_threshold) else float(ssim_threshold),
            "image_l1_gain": None if not math.isfinite(l1_threshold) else float(l1_threshold),
        },
        "best_axis_gains": {
            "relative_gain": float(best_relative_gain),
            "ssim_gain": float(best_ssim_gain),
            "image_l1_gain": float(best_l1_gain),
        },
        "frontier_preview": frontier_preview,
    }
    if mode in {"knee", "tail_knee"}:
        selection_payload["knee"] = {
            "min_score_fraction": float(args.policy_val_alpha_frontier_knee_min_score_fraction),
            "slope_drop_fraction": float(args.policy_val_alpha_frontier_knee_slope_drop_fraction),
            "tail_min_score_fraction": float(args.policy_val_alpha_frontier_tail_knee_min_score_fraction),
            "tail_min_regression_count": int(args.policy_val_alpha_frontier_tail_knee_min_regression_count),
            "tail_eps": float(args.policy_val_alpha_frontier_tail_knee_eps),
            "active_axes": list(active_axes),
            "tail_axes": list(tail_axes),
            "profiles": knee_profiles[:16],
        }
    return selected, selection_payload


def select_policy_val_payload_by_risk_gate(
    policy_payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], float, list[str], bool]:
    policy_payload = dict(policy_payload)
    best_row = dict(policy_payload.get("best") or {})
    selected_alpha_override: float | None = None
    if bool(args.select_alpha_by_risk_gate) and bool(policy_payload.get("enabled", False)):
        safe_rows = []
        for row in policy_payload.get("rows", []) or []:
            if float(row.get("alpha", 0.0)) <= 0.0:
                continue
            if float(row.get("relative_gain", -1.0)) < float(args.min_policy_val_relative_gain):
                continue
            if policy_val_risk_reasons(dict(row), args):
                continue
            safe_rows.append(dict(row))
        if safe_rows:
            best_row, frontier_selection = select_policy_val_frontier_row(safe_rows, args)
            policy_payload["best"] = best_row
            selection_mode = "risk_gate"
            refinement = policy_payload.get("ssim_alpha_refinement") or {}
            midpoint = policy_payload.get("alpha_midpoint_refinement") or {}
            if bool(refinement.get("enabled", False)) and float(best_row.get("alpha", 0.0)) in {
                float(x) for x in refinement.get("inserted_alpha_grid", []) or []
            }:
                selection_mode = "risk_gate_ssim_alpha_refined"
            if bool(midpoint.get("enabled", False)) and float(best_row.get("alpha", 0.0)) in {
                float(x) for x in midpoint.get("inserted_alpha_grid", []) or []
            }:
                selection_mode = "risk_gate_alpha_midpoint"
            if bool(frontier_selection.get("frontier_enabled", False)):
                selection_mode = f"{selection_mode}_frontier"
            policy_payload["selection"] = {
                "mode": selection_mode,
                "safe_alpha_count": int(len(safe_rows)),
                "selected_alpha": float(best_row.get("alpha", 0.0)),
                "selected_from_refinement": selection_mode == "risk_gate_ssim_alpha_refined",
                "selected_from_midpoint": bool(
                    midpoint.get("enabled", False)
                    and float(best_row.get("alpha", 0.0)) in {float(x) for x in midpoint.get("inserted_alpha_grid", []) or []}
                ),
                "alpha_frontier": dict(frontier_selection),
            }
        else:
            selected_alpha_override = 0.0
            policy_payload["selection"] = {
                "mode": "risk_gate",
                "safe_alpha_count": 0,
                "selected_alpha": 0.0,
                "selected_from_refinement": False,
                "selected_from_midpoint": False,
                "alpha_frontier": {
                    "frontier_enabled": bool(getattr(args, "enable_policy_val_alpha_frontier_selection", False)),
                    "frontier_reason": "no_safe_rows",
                },
            }
    selected_alpha = (
        float(selected_alpha_override)
        if selected_alpha_override is not None
        else float(best_row.get("alpha", 0.0))
    )
    risk_reasons = (
        policy_val_risk_reasons(best_row, args)
        if bool(policy_payload.get("enabled", False)) and best_row
        else []
    )
    accepted_payload = bool(
        policy_payload.get("enabled", False)
        and int(policy_payload.get("samples", 0)) >= int(args.min_policy_val_samples)
        and float(best_row.get("relative_gain", -1.0)) >= float(args.min_policy_val_relative_gain)
        and selected_alpha > 0.0
        and not risk_reasons
    )
    return policy_payload, best_row, float(selected_alpha), risk_reasons, bool(accepted_payload)


def select_policy_val_preacceptance_repair_seed(
    policy_payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not bool(policy_payload.get("enabled", False)):
        return None, []
    candidates: list[tuple[dict[str, Any], list[str]]] = []
    for row_in in policy_payload.get("rows", []) or []:
        row = dict(row_in)
        if float(row.get("alpha", 0.0)) <= 0.0:
            continue
        if float(row.get("relative_gain", -1.0)) < float(args.min_policy_val_relative_gain):
            continue
        if bool(getattr(args, "enable_policy_val_image_ssim_gate", False)):
            if float(row.get("ssim_gain", 0.0)) < float(args.min_policy_val_ssim_mean_gain):
                continue
            if float(row.get("ssim_positive_view_fraction", 0.0)) < float(
                args.min_policy_val_ssim_positive_view_fraction
            ):
                continue
            if float(row.get("ssim_min_view_gain", 0.0)) < float(args.min_policy_val_ssim_min_view_gain):
                continue
        if bool(getattr(args, "enable_policy_val_image_l1_gate", False)):
            if float(row.get("image_l1_gain", 0.0)) < float(args.min_policy_val_l1_mean_gain):
                continue
            if float(row.get("image_l1_positive_view_fraction", 0.0)) < float(
                args.min_policy_val_l1_positive_view_fraction
            ):
                continue
            if float(row.get("image_l1_min_view_gain", 0.0)) < float(args.min_policy_val_l1_min_view_gain):
                continue
        reasons = policy_val_risk_reasons(row, args)
        if not reasons:
            continue
        allowed_tail_reason = all(
            reason.startswith("min_view_relative_gain ")
            or reason.startswith("cvar20_view_relative_gain ")
            or reason.startswith("positive_view_fraction ")
            for reason in reasons
        )
        if allowed_tail_reason:
            candidates.append((row, reasons))
    if not candidates:
        return None, []
    row, reasons = max(candidates, key=lambda item: float(item[0].get("relative_gain", -1.0)))
    return dict(row), list(reasons)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit/apply a train-evidence surface residual texture atlas.")
    parser.add_argument("--source_model", required=True)
    parser.add_argument("--fit_evidence_dir", required=True)
    parser.add_argument("--target_evidence_dir", required=True)
    parser.add_argument("--region_carrier_json", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--target_split", choices=("train", "test"), default="test")
    parser.add_argument("--base_method_name", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument("--method_name", default="ours_26000_teacher_region_texture_adapter")
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    parser.add_argument(
        "--teacher_residual_target_mode",
        choices=("raw_rgb", "luma_only", "edge_luma_mix"),
        default="raw_rgb",
        help=(
            "Train-fit-only transform applied to teacher RGB residual samples before fitting "
            "the surface atlas/decoder. edge_luma_mix suppresses chroma residual on parent "
            "luma edges to target SSIM/LPIPS-oriented structure instead of raw RGB MSE."
        ),
    )
    parser.add_argument("--teacher_residual_target_luma_mix", type=float, default=0.75)
    parser.add_argument("--teacher_residual_target_edge_boost", type=float, default=0.25)
    parser.add_argument("--texture_size", type=int, default=16)
    parser.add_argument(
        "--texture_size_candidates",
        default="",
        help=(
            "Optional comma-separated texture sizes for train-only capacity auto-policy. "
            "If empty, keeps the fixed --texture_size legacy behavior."
        ),
    )
    parser.add_argument(
        "--enable_adaptive_texture_size_ladder",
        action="store_true",
        help=(
            "Auto-augment texture-size candidates from train-fit residual density. "
            "This uses only fit split evidence and never target/test GT."
        ),
    )
    parser.add_argument("--adaptive_texture_size_ladder_max_size", type=int, default=32)
    parser.add_argument("--adaptive_texture_size_ladder_min_fit_samples_per_face", type=float, default=512.0)
    parser.add_argument("--adaptive_texture_size_ladder_min_samples_per_current_bin", type=float, default=2.0)
    parser.add_argument("--adaptive_texture_size_ladder_min_mean_l1", type=float, default=0.002)
    parser.add_argument("--max_carriers", type=int, default=64)
    parser.add_argument("--max_faces_per_carrier", type=int, default=128)
    parser.add_argument("--max_faces", type=int, default=4096)
    parser.add_argument(
        "--support_expansion_mode",
        choices=("none", "fit_residual_topk", "target_footprint_residual_debt"),
        default="none",
        help=(
            "Optional support expansion. fit_residual_topk adds high-residual faces from atlas fit "
            "views while excluding policy-val views; target_footprint_residual_debt additionally "
            "weights train residual debt by GT-free target geometry footprint."
        ),
    )
    parser.add_argument(
        "--support_expansion_max_extra_faces",
        type=int,
        default=0,
        help="Maximum non-carrier faces to add when support expansion is enabled.",
    )
    parser.add_argument(
        "--support_expansion_max_extra_faces_candidates",
        default="",
        help=(
            "Optional comma-separated max-extra-face ladder. Each top-K support footprint is evaluated "
            "by the same train policy-val gate. Empty preserves --support_expansion_max_extra_faces."
        ),
    )
    parser.add_argument(
        "--support_expansion_min_face_samples",
        type=int,
        default=64,
        help="Minimum fit-view samples required for an expanded face.",
    )
    parser.add_argument(
        "--support_expansion_min_mean_l1",
        type=float,
        default=0.0,
        help="Minimum mean residual L1 required for an expanded face.",
    )
    parser.add_argument(
        "--target_footprint_residual_debt_match_level",
        choices=("bin", "face"),
        default="bin",
        help=(
            "For target_footprint_residual_debt expansion, require target footprint to hit the same "
            "face/UV bin as train residual debt, or allow train-certified debt to transfer within the same face."
        ),
    )
    parser.add_argument(
        "--enable_coview_face_residual_transfer",
        action="store_true",
        help=(
            "Add a train-only co-visible face residual-transfer candidate. Source residuals are learned "
            "from fit views, destination faces must be target-visible and policy-val-visible, and selection "
            "still goes through the policy-val certificate."
        ),
    )
    parser.add_argument("--coview_transfer_max_faces", type=int, default=0)
    parser.add_argument("--coview_transfer_neighbor_stride", type=int, default=8)
    parser.add_argument("--coview_transfer_min_source_samples", type=int, default=64)
    parser.add_argument("--coview_transfer_min_source_mean_l1", type=float, default=0.0)
    parser.add_argument("--coview_transfer_min_edge_count", type=int, default=8)
    parser.add_argument("--coview_transfer_min_target_pixels", type=int, default=128)
    parser.add_argument("--coview_transfer_min_policy_val_pixels", type=int, default=128)
    parser.add_argument("--coview_transfer_max_views", type=int, default=0)
    parser.add_argument("--coview_transfer_residual_scale", type=float, default=0.25)
    parser.add_argument("--coview_transfer_synthetic_count", type=int, default=1)
    parser.add_argument(
        "--coview_transfer_existing_atlas_mode",
        choices=("skip", "overwrite", "blend"),
        default="skip",
        help=(
            "How coview transfer handles faces already fitted by the train atlas. "
            "skip preserves v118 behavior, overwrite replaces the atlas with a synthetic residual, "
            "and blend injects the transfer residual as pseudo-count evidence."
        ),
    )
    parser.add_argument(
        "--coview_transfer_blend_max_direct_bin_count",
        type=int,
        default=-1,
        help=(
            "For blend mode, only inject transfer pseudo-counts into atlas bins whose direct fit count is "
            "<= this value. -1 blends all bins and preserves the original v119 behavior."
        ),
    )
    parser.add_argument("--coview_transfer_overwrite_existing_atlas", action="store_true")
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--alpha_grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--min_l1", type=float, default=0.0)
    parser.add_argument("--min_alpha", type=float, default=0.03)
    parser.add_argument(
        "--min_atlas_bin_count",
        type=int,
        default=0,
        help="Only apply/policy-evaluate atlas residuals on UV bins with at least this many fit samples.",
    )
    parser.add_argument(
        "--min_atlas_face_samples",
        type=int,
        default=0,
        help="Only apply/policy-evaluate atlas residuals on faces with at least this many fit samples.",
    )
    parser.add_argument(
        "--max_atlas_bin_rgb_variance",
        type=float,
        default=-1.0,
        help="Optional bin-certification gate: require mean per-channel residual variance in a UV bin to be <= this value.",
    )
    parser.add_argument(
        "--min_atlas_bin_sign_consistency",
        type=float,
        default=0.0,
        help="Optional bin-certification gate: require mean RGB residual sign consistency in a UV bin to be >= this value.",
    )
    parser.add_argument(
        "--atlas_confidence_mode",
        choices=("none", "count_var_sign"),
        default="none",
        help="Optional continuous residual confidence. Default keeps legacy hard-gated atlas behavior.",
    )
    parser.add_argument(
        "--atlas_confidence_count_scale",
        type=float,
        default=0.0,
        help="If >0, multiply residuals by 1-exp(-bin_count/scale); empty bins use --atlas_confidence_empty_bin.",
    )
    parser.add_argument(
        "--atlas_confidence_empty_bin",
        type=float,
        default=1.0,
        help="Confidence assigned to face-mean-filled bins with zero direct fit samples when confidence mode is enabled.",
    )
    parser.add_argument(
        "--atlas_confidence_variance_scale",
        type=float,
        default=-1.0,
        help="If >0, down-weight observed bins with high residual RGB variance using 1/(1+variance/scale).",
    )
    parser.add_argument(
        "--atlas_confidence_sign_power",
        type=float,
        default=0.0,
        help="If >0, down-weight observed bins by mean sign-consistency raised to this power.",
    )
    parser.add_argument(
        "--atlas_confidence_face_sample_scale",
        type=float,
        default=0.0,
        help="If >0, down-weight all residuals on faces with fewer fit samples than this scale.",
    )
    parser.add_argument(
        "--min_atlas_confidence",
        type=float,
        default=0.0,
        help="Treat pixels at or below this continuous confidence as unchanged support.",
    )
    parser.add_argument(
        "--atlas_lowpass_passes",
        type=int,
        default=0,
        help="Number of count-weighted 3x3 low-pass passes to apply to atlas texture after fitting.",
    )
    parser.add_argument(
        "--atlas_lowpass_neighbor_min_count",
        type=int,
        default=1,
        help="Minimum neighbor bin count used for atlas low-pass smoothing.",
    )
    parser.add_argument(
        "--surface_multiscale_prior_mode",
        choices=("none", "count_pyramid", "local_patch"),
        default="none",
        help=(
            "Optional surface residual prior. count_pyramid keeps direct high-support bins "
            "and blends low-support bins with same-face coarse residual blocks; local_patch "
            "uses same-face local UV patch residual estimates."
        ),
    )
    parser.add_argument(
        "--surface_multiscale_prior_block_sizes",
        default="2,4,8",
        help=(
            "Comma-separated UV block sizes for count_pyramid, or local patch radii for "
            "--surface_multiscale_prior_mode local_patch."
        ),
    )
    parser.add_argument(
        "--surface_multiscale_prior_min_bin_samples",
        type=int,
        default=8,
        help="Bins below this direct fit-sample count are blended with the count-pyramid prior.",
    )
    parser.add_argument(
        "--surface_multiscale_prior_count_tau",
        type=float,
        default=32.0,
        help="Count confidence tau for coarse residual blocks in the v69 prior.",
    )
    parser.add_argument(
        "--surface_multiscale_prior_blend",
        type=float,
        default=1.0,
        help="Maximum blend weight from direct low-support residuals toward the multiscale prior.",
    )
    parser.add_argument(
        "--surface_multiscale_prior_blend_candidates",
        default="",
        help=(
            "Optional comma-separated blend ladder for policy-val selection. When it includes 0, "
            "nonzero multiscale-prior candidates must be non-regressive versus the best zero-blend anchor."
        ),
    )
    parser.add_argument(
        "--surface_multiscale_prior_gate_mode",
        choices=("none", "evidence_consistent"),
        default="none",
        help=(
            "Optional gate for surface priors. evidence_consistent only blends low-support "
            "bins that have direct samples and agree with the prior under train-evidence "
            "sign/variance/cosine checks."
        ),
    )
    parser.add_argument(
        "--surface_multiscale_prior_min_prior_weight",
        type=float,
        default=0.0,
        help="Minimum coarse-prior confidence required by evidence_consistent gate.",
    )
    parser.add_argument(
        "--surface_multiscale_prior_min_direct_samples",
        type=int,
        default=1,
        help="Minimum direct bin samples required by evidence_consistent gate.",
    )
    parser.add_argument(
        "--surface_multiscale_prior_min_sign_consistency",
        type=float,
        default=0.0,
        help="Minimum mean direct residual sign consistency required by evidence_consistent gate.",
    )
    parser.add_argument(
        "--surface_multiscale_prior_max_mean_variance",
        type=float,
        default=-1.0,
        help="Maximum mean direct residual variance required by evidence_consistent gate; <0 disables.",
    )
    parser.add_argument(
        "--surface_multiscale_prior_min_cosine",
        type=float,
        default=0.0,
        help="Minimum cosine agreement between direct low-support residual and coarse prior.",
    )
    parser.add_argument(
        "--enable_policy_val_prior_bin_gain_hybrid",
        action="store_true",
        help=(
            "Add a train-policy-val bin-gain hybrid candidate: start from the zero-blend atlas and "
            "copy only nonzero-prior bins that improve policy-val residual error."
        ),
    )
    parser.add_argument(
        "--prior_bin_gain_hybrid_min_bin_samples",
        type=int,
        default=4,
        help="Minimum policy-val samples required for a prior bin to be copied into the hybrid atlas.",
    )
    parser.add_argument(
        "--prior_bin_gain_hybrid_min_views",
        type=int,
        default=1,
        help="Minimum policy-val views required for a prior bin to be copied into the hybrid atlas.",
    )
    parser.add_argument(
        "--prior_bin_gain_hybrid_min_abs_gain",
        type=float,
        default=0.0,
        help="Minimum absolute policy-val residual-error reduction required for a prior bin.",
    )
    parser.add_argument(
        "--prior_bin_gain_hybrid_min_relative_gain",
        type=float,
        default=0.0,
        help="Minimum relative policy-val residual-error gain required for a prior bin.",
    )
    parser.add_argument(
        "--prior_bin_gain_hybrid_min_positive_view_fraction",
        type=float,
        default=0.5,
        help="Minimum fraction of policy-val views where the prior bin beats the zero-blend bin.",
    )
    parser.add_argument(
        "--prior_bin_gain_hybrid_max_profile_bins",
        type=int,
        default=0,
        help="If >0, keep only the top policy-val-gain hybrid bins after gating.",
    )
    parser.add_argument(
        "--enable_policy_val_source_mixture",
        action="store_true",
        help=(
            "For prior-bin hybrid candidates, fit a policy-val per-bin continuous source weight "
            "instead of hard-copying source bins into the baseline atlas."
        ),
    )
    parser.add_argument(
        "--source_mixture_ridge",
        type=float,
        default=1.0e-2,
        help="Ridge regularization for policy-val per-bin source-mixture weights.",
    )
    parser.add_argument(
        "--source_mixture_ridge_mode",
        choices=("absolute", "adaptive_den"),
        default="absolute",
        help=(
            "How to scale source-mixture ridge. 'absolute' preserves earlier behavior; "
            "'adaptive_den' scales ridge by max(local denominator, median positive denominator)."
        ),
    )
    parser.add_argument(
        "--source_mixture_min_weight",
        type=float,
        default=1.0e-4,
        help="Reject source-mixture bins whose fitted source weight is below this threshold.",
    )
    parser.add_argument(
        "--enable_prior_bin_gain_hybrid_l1_proxy_gate",
        action="store_true",
        help=(
            "Require copied hybrid bins to improve a local RGB-L1 policy-val proxy in addition "
            "to residual MSE. This aligns bin selection with image-L1/SSIM guardrails."
        ),
    )
    parser.add_argument("--prior_bin_gain_hybrid_min_l1_abs_gain", type=float, default=0.0)
    parser.add_argument("--prior_bin_gain_hybrid_min_l1_relative_gain", type=float, default=-1.0)
    parser.add_argument("--prior_bin_gain_hybrid_min_l1_positive_view_fraction", type=float, default=0.0)
    parser.add_argument("--prior_bin_gain_hybrid_min_l1_min_view_gain", type=float, default=-1.0)
    parser.add_argument("--prior_bin_gain_hybrid_min_l1_cvar20_view_gain", type=float, default=-1.0)
    parser.add_argument(
        "--view_conditioned_basis_mode",
        choices=("none", "camera_center_linear", "normal_camera_linear"),
        default="none",
        help=(
            "Optional persistent view-conditioned residual basis. camera_center_linear fits "
            "against [1, normalized camera_center]; normal_camera_linear additionally uses "
            "per-pixel normal and normal-dot-camera features. Fitting uses train fit views "
            "only; unsupported bins fall back to the legacy mean atlas."
        ),
    )
    parser.add_argument(
        "--view_conditioned_basis_guard_mode",
        choices=("none", "policy_val_nonregressive"),
        default="none",
        help=(
            "Optional train-policy-val guard for view-conditioned basis. policy_val_nonregressive "
            "compares the basis atlas to the same mean atlas with basis disabled and falls back "
            "unless the basis is non-regressive on the policy metrics used by the atlas selector."
        ),
    )
    parser.add_argument(
        "--view_conditioned_basis_min_bin_samples",
        type=int,
        default=16,
        help="Minimum fit samples in a face/UV bin before view-conditioned coefficients are used.",
    )
    parser.add_argument(
        "--view_conditioned_basis_ridge",
        type=float,
        default=1.0e-3,
        help="Ridge value for the per-bin view-conditioned least-squares fit.",
    )
    parser.add_argument(
        "--view_conditioned_basis_ood_mode",
        choices=("none", "diag_z"),
        default="none",
        help=(
            "Optional per-bin view-feature OOD guard for view-conditioned basis. diag_z "
            "falls back to the legacy mean atlas when the target view feature is outside "
            "the fit-view diagonal z-score envelope for that face/UV bin."
        ),
    )
    parser.add_argument(
        "--view_conditioned_basis_ood_max_z",
        type=float,
        default=2.5,
        help="Max per-feature diagonal z-score allowed by --view_conditioned_basis_ood_mode diag_z.",
    )
    parser.add_argument(
        "--view_conditioned_basis_ood_min_std",
        type=float,
        default=5.0e-2,
        help="Minimum feature standard deviation used by the diagonal z-score OOD guard.",
    )
    parser.add_argument(
        "--view_cluster_expert_count",
        type=int,
        default=1,
        help=(
            "Optional v135 GT-free view-clustered residual experts. Values >1 fit "
            "separate local surface residual atlases for camera-center clusters."
        ),
    )
    parser.add_argument(
        "--view_cluster_feature_mode",
        choices=("none", "camera_center"),
        default="camera_center",
        help="Target-safe feature used to route a view to a residual expert.",
    )
    parser.add_argument(
        "--view_cluster_min_views",
        type=int,
        default=2,
        help="Minimum fit views required for a view cluster to become active.",
    )
    parser.add_argument(
        "--view_cluster_min_bin_samples",
        type=int,
        default=4,
        help="Minimum per-expert face/UV-bin samples before replacing the global residual.",
    )
    parser.add_argument(
        "--view_cluster_fallback_mode",
        choices=("global",),
        default="global",
        help="Fallback for bins unsupported by the routed expert.",
    )
    parser.add_argument(
        "--teacher_distilled_basis_mode",
        choices=(
            "none",
            "face_uv_normal_camera_ridge",
            "face_uv_patch_mixture_ridge",
            "surface_feature_rff_ridge",
            "low_rank_view_texture_k4",
            "low_rank_view_texture",
            "low_rank_view_texture_rich_k4",
            "low_rank_view_texture_rich",
        ),
        default="none",
        help=(
            "Optional v65 teacher-distilled shared residual field. "
            "face_uv_normal_camera_ridge fits one ridge residual model per face using "
            "Phase-J teacher residuals and features [camera, normal, normal-dot-camera, UV polynomial]. "
            "face_uv_patch_mixture_ridge adds a 3x3 local UV RBF mixture and normal-view interactions. "
            "surface_feature_rff_ridge fits a train-only per-face Fourier/RFF surface feature decoder. "
            "low_rank_view_texture_k4 fits four view/UV-aware mixture weights per face as a compact "
            "v169 teacher residual texture. low_rank_view_texture_rich_k4 keeps the same rank-4 "
            "surface factorization but adds UV, normal, parent RGB, depth, and alpha context. "
            "The low_rank_view_texture and low_rank_view_texture_rich aliases expose configurable rank."
        ),
    )
    parser.add_argument(
        "--teacher_distilled_basis_guard_mode",
        choices=("none", "policy_val_nonregressive"),
        default="none",
        help=(
            "Optional train-policy-val guard for the v65 shared teacher basis. "
            "policy_val_nonregressive compares against the same atlas with the shared basis disabled."
        ),
    )
    parser.add_argument("--teacher_distilled_basis_min_face_samples", type=int, default=1024)
    parser.add_argument("--teacher_distilled_basis_ridge", type=float, default=1.0e-2)
    parser.add_argument("--teacher_distilled_basis_ood_max_z", type=float, default=3.0)
    parser.add_argument("--teacher_distilled_basis_ood_min_std", type=float, default=5.0e-2)
    parser.add_argument(
        "--teacher_distilled_basis_apply_mode",
        choices=("replace_supported", "blend", "fill_empty_only"),
        default="blend",
    )
    parser.add_argument("--teacher_distilled_basis_blend", type=float, default=0.5)
    parser.add_argument(
        "--teacher_distilled_low_rank_texture_rank",
        type=int,
        default=4,
        help="Requested rank for low_rank_view_texture* teacher residual texture modes.",
    )
    parser.add_argument(
        "--teacher_distilled_low_rank_texture_rank_candidates",
        default="",
        help=(
            "Optional comma-separated rank ladder for low-rank teacher residual textures. "
            "Selection uses train policy-val only; empty keeps --teacher_distilled_low_rank_texture_rank."
        ),
    )
    parser.add_argument(
        "--enable_adaptive_low_support_teacher_basis",
        action="store_true",
        help=(
            "Lower the teacher-distilled per-face solve threshold from the requested ceiling "
            "using only fit-evidence face support statistics, then increase ridge on newly "
            "enabled low-support faces. This expands representation capacity without reading target/test GT."
        ),
    )
    parser.add_argument(
        "--adaptive_teacher_basis_min_face_samples_floor",
        type=int,
        default=128,
        help="Minimum allowed effective face-sample threshold for adaptive low-support teacher basis.",
    )
    parser.add_argument(
        "--adaptive_teacher_basis_support_quantile",
        type=float,
        default=0.25,
        help="Fit-evidence support quantile used to choose the adaptive teacher-basis threshold.",
    )
    parser.add_argument(
        "--adaptive_teacher_basis_low_support_ridge_scale",
        type=float,
        default=0.5,
        help="Extra ridge multiplier scale for faces enabled below the requested teacher-basis threshold.",
    )
    parser.add_argument(
        "--atlas_empty_bin_fill_mode",
        choices=("zero", "face_mean", "nearest_observed", "auto_policy"),
        default="face_mean",
        help=(
            "How to fill UV atlas bins with no direct fit samples. The default "
            "keeps legacy face-mean expansion; nearest_observed iteratively diffuses "
            "neighboring observed residuals without changing the observed-bin counts. "
            "auto_policy fits both face_mean and nearest_observed atlases, then selects "
            "the accepted fill mode with the stronger train policy-val evidence."
        ),
    )
    parser.add_argument(
        "--atlas_nearest_fill_max_steps",
        type=int,
        default=32,
        help="Maximum dilation steps for --atlas_empty_bin_fill_mode nearest_observed.",
    )
    parser.add_argument(
        "--atlas_nearest_fill_decay",
        type=float,
        default=0.92,
        help="Per-step residual decay for nearest-observed empty-bin filling.",
    )
    parser.add_argument("--max_samples_per_view", type=int, default=240000)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=0.12)
    parser.add_argument(
        "--max_abs_delta_rgb_candidates",
        default="",
        help=(
            "Optional comma-separated residual-cap ladder for policy-val selection. "
            "Each candidate uses the same cap for train policy-val, target support profiling, and target apply."
        ),
    )
    parser.add_argument("--min_policy_val_samples", type=int, default=1024)
    parser.add_argument("--min_policy_val_relative_gain", type=float, default=0.0)
    parser.add_argument(
        "--min_policy_val_positive_view_fraction",
        type=float,
        default=0.0,
        help="Optional robust gate: require the selected alpha to improve at least this fraction of policy-val views.",
    )
    parser.add_argument(
        "--min_policy_val_cvar20_relative_gain",
        type=float,
        default=-1.0,
        help="Optional robust gate: require the selected alpha's worst-20-percent view relative gain to exceed this value.",
    )
    parser.add_argument(
        "--min_policy_val_min_view_relative_gain",
        type=float,
        default=-1.0,
        help="Optional robust gate: require every policy-val view relative gain to exceed this value.",
    )
    parser.add_argument(
        "--select_alpha_by_risk_gate",
        action="store_true",
        help="Select the best alpha that satisfies aggregate and robust policy-val gates instead of the mean-best alpha.",
    )
    parser.add_argument(
        "--enable_policy_val_ssim_alpha_refinement",
        action="store_true",
        help=(
            "Augment the policy-val alpha line search with deterministic half-step low-alpha candidates. "
            "This keeps the same SSIM/L1 risk gates and only changes the certified candidate set."
        ),
    )
    parser.add_argument(
        "--enable_policy_val_alpha_midpoint_refinement",
        action="store_true",
        help=(
            "Augment the policy-val alpha line search with deterministic midpoints between adjacent "
            "candidate alphas. This makes intermediate strengths such as 0.1875 train-policy-selectable "
            "without hand-writing them into --alpha_grid."
        ),
    )
    parser.add_argument(
        "--enable_policy_val_alpha_frontier_selection",
        action="store_true",
        help=(
            "After normal policy-val risk gates, select the smallest alpha that preserves a configured "
            "fraction of the best train-policy-val relative/SSIM/L1 gains instead of always selecting "
            "the max-relative-gain alpha."
        ),
    )
    parser.add_argument(
        "--policy_val_alpha_frontier_mode",
        choices=("smallest_effective", "best_score", "knee", "tail_knee"),
        default="smallest_effective",
        help="Frontier selection mode used by --enable_policy_val_alpha_frontier_selection.",
    )
    parser.add_argument("--policy_val_alpha_frontier_min_relative_fraction", type=float, default=0.75)
    parser.add_argument("--policy_val_alpha_frontier_min_ssim_fraction", type=float, default=0.75)
    parser.add_argument("--policy_val_alpha_frontier_min_l1_fraction", type=float, default=0.75)
    parser.add_argument(
        "--policy_val_alpha_frontier_alpha_penalty",
        type=float,
        default=0.25,
        help="Alpha-size penalty used only by --policy_val_alpha_frontier_mode best_score.",
    )
    parser.add_argument(
        "--policy_val_alpha_frontier_knee_min_score_fraction",
        type=float,
        default=0.55,
        help=(
            "For --policy_val_alpha_frontier_mode knee, require the selected alpha to reach this "
            "normalized policy-val gain score before it can be considered a conservative knee."
        ),
    )
    parser.add_argument(
        "--policy_val_alpha_frontier_knee_slope_drop_fraction",
        type=float,
        default=0.85,
        help=(
            "For --policy_val_alpha_frontier_mode knee, stop before the next alpha when its marginal "
            "normalized gain slope drops below this fraction of the best prior slope."
        ),
    )
    parser.add_argument(
        "--policy_val_alpha_frontier_tail_knee_min_score_fraction",
        type=float,
        default=0.70,
        help=(
            "For --policy_val_alpha_frontier_mode tail_knee, require this normalized aggregate "
            "policy-val score before a robust-tail regression can stop alpha growth."
        ),
    )
    parser.add_argument(
        "--policy_val_alpha_frontier_tail_knee_min_regression_count",
        type=int,
        default=3,
        help=(
            "For --policy_val_alpha_frontier_mode tail_knee, stop before the next alpha when at least "
            "this many robust tail metrics regress."
        ),
    )
    parser.add_argument(
        "--policy_val_alpha_frontier_tail_knee_eps",
        type=float,
        default=1.0e-10,
        help="Tolerance used when detecting robust tail regression for tail_knee mode.",
    )
    parser.add_argument(
        "--enable_preacceptance_policy_val_guard_repair",
        action="store_true",
        help=(
            "If a candidate has positive mean SSIM/L1 gains but fails only robust policy-val tail gates, "
            "fit train-policy-val face/bin guards before final risk-gate selection."
        ),
    )
    parser.add_argument(
        "--policy_val_ssim_alpha_refinement_steps",
        type=int,
        default=7,
        help="Number of repeated half-step refinements inserted below each positive alpha candidate.",
    )
    parser.add_argument(
        "--policy_val_ssim_alpha_refinement_min_alpha",
        type=float,
        default=0.001,
        help="Smallest positive refined alpha considered by the SSIM-safe policy-val line search.",
    )
    parser.add_argument(
        "--enable_policy_val_alpha_calibration",
        action="store_true",
        help=(
            "Augment --alpha_grid with a closed-form least-squares alpha estimated only from "
            "train policy-val residual samples, then still select through the normal risk gates."
        ),
    )
    parser.add_argument(
        "--alpha_calibration_max_alpha",
        type=float,
        default=0.5,
        help="Maximum alpha allowed for policy-val calibrated alpha candidates.",
    )
    parser.add_argument(
        "--alpha_calibration_multipliers",
        default="0.5,0.75,1.0,1.25",
        help="Comma-separated multipliers around the calibrated alpha candidate.",
    )
    parser.add_argument(
        "--alpha_calibration_min_denominator",
        type=float,
        default=1.0e-12,
        help="Minimum least-squares denominator required to add calibrated alpha candidates.",
    )
    parser.add_argument(
        "--enable_policy_val_local_alpha_calibration",
        action="store_true",
        help=(
            "Fit a train policy-val residual-norm bucket alpha profile. The learned per-bucket "
            "alpha is applied locally, while the scalar alpha grid becomes a global multiplier "
            "selected by the normal risk gates."
        ),
    )
    parser.add_argument("--local_alpha_calibration_max_alpha", type=float, default=0.5)
    parser.add_argument("--local_alpha_calibration_min_alpha", type=float, default=0.0)
    parser.add_argument(
        "--local_alpha_calibration_bucket_quantiles",
        default="0.25,0.5,0.75,0.9",
        help="Comma-separated policy-val residual-norm quantiles used as bucket edges.",
    )
    parser.add_argument(
        "--local_alpha_calibration_bucket_edges",
        default="",
        help="Optional absolute residual-norm bucket edges. If set, overrides quantiles.",
    )
    parser.add_argument(
        "--local_alpha_calibration_multipliers",
        default="0.5,0.75,1.0,1.25",
        help="Scalar multiplier candidates added to --alpha_grid when local alpha calibration is enabled.",
    )
    parser.add_argument("--local_alpha_calibration_min_bucket_samples", type=int, default=1024)
    parser.add_argument(
        "--local_alpha_calibration_norm_mode",
        choices=("l2", "mean_abs"),
        default="l2",
        help="Residual prediction norm used to assign policy-val samples to local alpha buckets.",
    )
    parser.add_argument(
        "--local_alpha_calibration_min_denominator",
        type=float,
        default=1.0e-12,
        help="Minimum least-squares denominator required for local alpha calibration.",
    )
    parser.add_argument(
        "--enable_policy_val_face_alpha_calibration",
        action="store_true",
        help=(
            "Fit a train policy-val least-squares alpha per surface face. Low-support faces "
            "fall back to a global policy-val alpha, and the scalar alpha grid remains a "
            "global multiplier selected by the normal risk gates."
        ),
    )
    parser.add_argument("--face_alpha_calibration_max_alpha", type=float, default=0.5)
    parser.add_argument("--face_alpha_calibration_min_alpha", type=float, default=0.0)
    parser.add_argument("--face_alpha_calibration_multipliers", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--face_alpha_calibration_min_face_samples", type=int, default=256)
    parser.add_argument(
        "--face_alpha_calibration_min_denominator",
        type=float,
        default=1.0e-12,
        help="Minimum least-squares denominator required for face alpha calibration.",
    )
    parser.add_argument(
        "--face_alpha_calibration_shrink_count_tau",
        type=float,
        default=0.0,
        help=(
            "If >0, shrink each fitted face alpha by count/(count+tau). "
            "This is a train-policy-val reliability prior and defaults to legacy no shrink."
        ),
    )
    parser.add_argument(
        "--face_alpha_calibration_shrink_denominator_tau",
        type=float,
        default=0.0,
        help=(
            "If >0, additionally shrink each fitted face alpha by denom/(denom+tau), "
            "where denom is the policy-val LS denominator."
        ),
    )
    parser.add_argument(
        "--face_alpha_calibration_shrink_prior",
        choices=("fallback", "zero"),
        default="fallback",
        help="Prior alpha used by face-alpha reliability shrink. Defaults to legacy fallback-alpha prior.",
    )
    parser.add_argument(
        "--enable_policy_val_bin_alpha_calibration",
        action="store_true",
        help=(
            "Fit a train policy-val least-squares alpha per face/UV bin. This calibrates "
            "residual magnitude instead of only masking bins; low-support bins fall back "
            "to a global policy-val alpha, and the scalar alpha grid remains a global multiplier."
        ),
    )
    parser.add_argument("--bin_alpha_calibration_max_alpha", type=float, default=0.5)
    parser.add_argument("--bin_alpha_calibration_min_alpha", type=float, default=0.0)
    parser.add_argument("--bin_alpha_calibration_multipliers", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--bin_alpha_calibration_min_bin_samples", type=int, default=64)
    parser.add_argument(
        "--bin_alpha_calibration_min_denominator",
        type=float,
        default=1.0e-12,
        help="Minimum least-squares denominator required for bin alpha calibration.",
    )
    parser.add_argument(
        "--bin_alpha_calibration_min_positive_view_fraction",
        type=float,
        default=0.5,
        help="Minimum fraction of policy-val views improved by a fitted bin alpha before it is stored.",
    )
    parser.add_argument(
        "--bin_alpha_calibration_shrink_count_tau",
        type=float,
        default=128.0,
        help="If >0, shrink each fitted bin alpha by count/(count+tau).",
    )
    parser.add_argument(
        "--bin_alpha_calibration_shrink_denominator_tau",
        type=float,
        default=0.0,
        help=(
            "If >0, additionally shrink each fitted bin alpha by denom/(denom+tau), "
            "where denom is the policy-val LS denominator."
        ),
    )
    parser.add_argument(
        "--bin_alpha_calibration_shrink_prior",
        choices=("fallback", "zero"),
        default="fallback",
        help="Prior alpha used by bin-alpha reliability shrink.",
    )
    parser.add_argument(
        "--bin_alpha_calibration_max_profile_bins",
        type=int,
        default=8192,
        help="Maximum number of calibrated face/UV bins stored in the profile. <=0 stores all fitted bins.",
    )
    parser.add_argument(
        "--enable_policy_val_bin_rgb_alpha_calibration",
        action="store_true",
        help=(
            "Fit a train policy-val per-channel RGB alpha per face/UV bin. This is a v66 "
            "extension of scalar bin alpha: reliable bins learn a three-channel residual "
            "shrink vector, weak bins fall back to a global RGB shrink vector, and the "
            "scalar alpha grid remains a global multiplier."
        ),
    )
    parser.add_argument("--bin_rgb_alpha_calibration_max_alpha", type=float, default=0.5)
    parser.add_argument("--bin_rgb_alpha_calibration_min_alpha", type=float, default=0.0)
    parser.add_argument("--bin_rgb_alpha_calibration_multipliers", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--bin_rgb_alpha_calibration_min_bin_samples", type=int, default=64)
    parser.add_argument(
        "--bin_rgb_alpha_calibration_min_denominator",
        type=float,
        default=1.0e-12,
        help="Minimum summed least-squares denominator required for bin RGB alpha calibration.",
    )
    parser.add_argument(
        "--bin_rgb_alpha_calibration_min_positive_view_fraction",
        type=float,
        default=0.5,
        help="Minimum fraction of policy-val views improved by a fitted RGB bin alpha before it is stored.",
    )
    parser.add_argument("--bin_rgb_alpha_calibration_shrink_count_tau", type=float, default=128.0)
    parser.add_argument("--bin_rgb_alpha_calibration_shrink_denominator_tau", type=float, default=0.0)
    parser.add_argument(
        "--bin_rgb_alpha_calibration_shrink_prior",
        choices=("fallback", "zero"),
        default="fallback",
        help="Prior alpha used by bin-RGB-alpha reliability shrink.",
    )
    parser.add_argument(
        "--bin_rgb_alpha_calibration_max_profile_bins",
        type=int,
        default=8192,
        help="Maximum number of calibrated RGB face/UV bins stored in the profile. <=0 stores all fitted bins.",
    )
    parser.add_argument(
        "--enable_policy_val_bin_uncertainty_shrink",
        action="store_true",
        help=(
            "Build a train-policy-val uncertainty-aware local shrink profile. "
            "sparse_positive is the v67 behavior; keep_with_downweight is the v68 "
            "behavior that keeps fallback residual strength and only stores explicit "
            "downweights/upweights for evidenced bins. positive_consensus is a stricter "
            "view-consistent mode that defaults unknown bins to zero and materializes "
            "only multi-view positive train-policy-val bins."
        ),
    )
    parser.add_argument(
        "--bin_uncertainty_shrink_policy_mode",
        choices=("sparse_positive", "keep_with_downweight", "positive_consensus"),
        default="sparse_positive",
    )
    parser.add_argument("--bin_uncertainty_shrink_min_bin_samples", type=int, default=64)
    parser.add_argument("--bin_uncertainty_shrink_min_bin_views", type=int, default=1)
    parser.add_argument("--bin_uncertainty_shrink_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--bin_uncertainty_shrink_min_positive_view_fraction", type=float, default=0.75)
    parser.add_argument("--bin_uncertainty_shrink_max_mean_variance", type=float, default=-1.0)
    parser.add_argument("--bin_uncertainty_shrink_min_mean_sign_consistency", type=float, default=0.0)
    parser.add_argument("--bin_uncertainty_shrink_count_tau", type=float, default=128.0)
    parser.add_argument("--bin_uncertainty_shrink_gain_tau", type=float, default=0.01)
    parser.add_argument("--bin_uncertainty_shrink_variance_scale", type=float, default=0.004)
    parser.add_argument("--bin_uncertainty_shrink_sign_power", type=float, default=0.5)
    parser.add_argument("--bin_uncertainty_shrink_min_shrink", type=float, default=0.0)
    parser.add_argument("--bin_uncertainty_shrink_max_shrink", type=float, default=1.0)
    parser.add_argument("--bin_uncertainty_shrink_fallback_shrink", type=float, default=0.0)
    parser.add_argument(
        "--bin_uncertainty_shrink_max_profile_bins",
        type=int,
        default=8192,
        help="Maximum number of uncertainty-shrink face/UV bins stored in the profile. <=0 stores all fitted bins.",
    )
    parser.add_argument(
        "--enable_view_cluster_local_shrink",
        "--enable_policy_val_cluster_local_shrink",
        dest="enable_view_cluster_local_shrink",
        action="store_true",
        help=(
            "When view-cluster experts are enabled, calibrate uncertainty shrink per "
            "train-policy-val view cluster instead of globally per face/UV bin."
        ),
    )
    parser.add_argument(
        "--view_cluster_local_shrink_global_fallback",
        action="store_true",
        help=(
            "Also expose selected cluster-local bins through the legacy global shrink "
            "table. Disabled by default so unknown clusters remain explicit no-op."
        ),
    )
    parser.add_argument(
        "--enable_policy_val_image_l1_bin_certificate",
        action="store_true",
        help=(
            "Use local train-policy-val image-L1 before/after evidence as a bin-level certificate "
            "for uncertainty shrink. This can replace or augment the residual-MSE proxy while "
            "still keeping target/test GT out of apply-time decisions."
        ),
    )
    parser.add_argument(
        "--image_l1_bin_certificate_mode",
        choices=("and", "or", "replace"),
        default="and",
        help=(
            "How the image-L1 bin certificate combines with residual-MSE evidence: "
            "and=require both, or=allow either, replace=use image-L1 as the bin acceptance signal."
        ),
    )
    parser.add_argument("--image_l1_bin_certificate_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--image_l1_bin_certificate_min_positive_view_fraction", type=float, default=0.55)
    parser.add_argument("--image_l1_bin_certificate_gain_tau", type=float, default=0.01)
    parser.add_argument(
        "--image_l1_bin_certificate_pool_radius",
        type=int,
        default=0,
        help=(
            "UV-bin radius used to pool same-face policy-val image-L1 evidence for each certified bin. "
            "0 is exact-bin evidence; >0 is a patch-level certificate for sparse outdoor coverage."
        ),
    )
    parser.add_argument(
        "--enable_policy_val_image_l1_region_expansion",
        action="store_true",
        help=(
            "Expand image-L1-certified seed bins to same-face neighboring bins with policy-val support and no "
            "strong negative evidence. This increases coverage without target/test GT."
        ),
    )
    parser.add_argument("--image_l1_region_expansion_radius", type=int, default=1)
    parser.add_argument("--image_l1_region_expansion_max_bins_per_seed", type=int, default=8)
    parser.add_argument("--image_l1_region_expansion_min_neighbor_samples", type=int, default=1)
    parser.add_argument("--image_l1_region_expansion_min_neighbor_views", type=int, default=1)
    parser.add_argument("--image_l1_region_expansion_max_negative_relative_gain", type=float, default=0.02)
    parser.add_argument("--image_l1_region_expansion_max_negative_image_l1_gain", type=float, default=0.02)
    parser.add_argument("--image_l1_region_expansion_shrink_decay", type=float, default=0.5)
    parser.add_argument(
        "--enable_policy_val_image_l1_bin_alpha_optimization",
        action="store_true",
        help=(
            "Fit a per-face/UV-bin local alpha by directly minimizing policy-val image L1 "
            "of clip(rgb_render + alpha * residual_delta). This is GT-free at target/test apply time."
        ),
    )
    parser.add_argument(
        "--image_l1_bin_alpha_grid",
        default="0,0.0625,0.125,0.25,0.5,0.75,1.0",
        help="Local alpha candidates for policy-val image-L1 bin optimization.",
    )
    parser.add_argument("--image_l1_bin_alpha_max_alpha", type=float, default=1.0)
    parser.add_argument("--image_l1_bin_alpha_min_bin_samples", type=int, default=8)
    parser.add_argument("--image_l1_bin_alpha_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--image_l1_bin_alpha_min_positive_view_fraction", type=float, default=0.55)
    parser.add_argument("--image_l1_bin_alpha_count_tau", type=float, default=64.0)
    parser.add_argument(
        "--image_l1_bin_alpha_fallback_mode",
        choices=("zero", "global_best"),
        default="zero",
        help="Fallback local alpha for bins not certified by image-L1 optimization.",
    )
    parser.add_argument(
        "--image_l1_bin_alpha_max_profile_bins",
        type=int,
        default=8192,
        help="Maximum image-L1 optimized bins stored in the local alpha profile. <=0 stores all bins.",
    )
    parser.add_argument(
        "--enable_policy_val_image_linear_residual_generator",
        action="store_true",
        help=(
            "Fit a policy-val ridge linear generator for image-space residuals and apply it target/test-GT-free "
            "through the normal certified residual texture path."
        ),
    )
    parser.add_argument(
        "--image_linear_generator_feature_mode",
        choices=("base", "base_rgb", "base_rgb_bary_view"),
        default="base_rgb",
    )
    parser.add_argument("--image_linear_generator_ridge", type=float, default=1.0e-2)
    parser.add_argument("--image_linear_generator_train_max_samples_per_view", type=int, default=100000)
    parser.add_argument("--image_linear_generator_max_train_samples", type=int, default=1000000)
    parser.add_argument("--image_linear_generator_output_cap", type=float, default=0.12)
    parser.add_argument(
        "--image_linear_generator_loss_mode",
        choices=("mse", "huber_irls", "l1_irls"),
        default="mse",
        help="Policy-val generator objective. v144 used mse; v145 can use robust IRLS losses.",
    )
    parser.add_argument("--image_linear_generator_irls_iterations", type=int, default=4)
    parser.add_argument("--image_linear_generator_huber_delta", type=float, default=0.02)
    parser.add_argument(
        "--image_linear_generator_training_sample_policy",
        choices=("all", "base_l1_descent", "view_balanced", "view_balanced_base_l1_descent"),
        default="all",
        help=(
            "Samples used to train the image-linear generator. base_l1_descent keeps only policy-val pixels "
            "where the base atlas residual improves image L1 over no residual; view_balanced gives each "
            "policy-val view equal regression weight to improve cross-view gain coverage."
        ),
    )
    parser.add_argument("--image_linear_generator_min_descent_margin", type=float, default=0.0)
    parser.add_argument("--image_linear_generator_min_training_samples", type=int, default=512)
    parser.add_argument(
        "--image_linear_generator_alpha_grid",
        default="0,0.03125,0.0625,0.125,0.25,0.5,0.75,1.0",
        help="Global alpha candidates evaluated after fitting the policy-val image-linear residual generator.",
    )
    parser.add_argument(
        "--image_linear_generator_expert_mode",
        choices=("none", "view_cluster"),
        default="none",
        help="Optional mixture-of-experts routing for the image-linear generator.",
    )
    parser.add_argument(
        "--image_linear_generator_expert_min_training_samples",
        type=int,
        default=2048,
        help="Minimum policy-val samples required to fit one image-linear expert.",
    )
    parser.add_argument(
        "--image_linear_generator_expert_shrink_tau",
        type=float,
        default=8192.0,
        help="Sample-count shrinkage tau for blending each expert back to the global generator.",
    )
    parser.add_argument(
        "--image_linear_generator_face_reliability_mode",
        choices=("none", "global", "view_cluster"),
        default="none",
        help=(
            "Use train-policy-val face reliability to suppress image-linear generator outputs before the "
            "final policy-val gate. view_cluster computes a separate face allowlist for each routed expert."
        ),
    )
    parser.add_argument("--image_linear_generator_face_reliability_min_face_samples", type=int, default=256)
    parser.add_argument("--image_linear_generator_face_reliability_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--image_linear_generator_face_reliability_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument("--image_linear_generator_face_reliability_fallback_multiplier", type=float, default=0.0)
    parser.add_argument(
        "--image_linear_generator_allow_unvalidated_base_pixels",
        action="store_true",
        help=(
            "Allow fitting/applying generator samples before the normal atlas-valid mask. Disabled by default; "
            "the default keeps v144 inside the same certified surface support as the parent residual atlas."
        ),
    )
    parser.add_argument(
        "--enable_policy_val_structure_aware_shrink",
        action="store_true",
        help=(
            "Augment policy-val bin uncertainty shrink with local structure-risk evidence from train-policy-val "
            "views. Uses only fit/policy-val rgb_render/rgb_gt, never target/test GT, to downweight bins that "
            "increase local L1 or gradient-structure error."
        ),
    )
    parser.add_argument("--structure_shrink_l1_weight", type=float, default=0.0)
    parser.add_argument("--structure_shrink_gradient_weight", type=float, default=0.0)
    parser.add_argument("--structure_shrink_edge_weight", type=float, default=0.0)
    parser.add_argument("--structure_shrink_risk_tau", type=float, default=0.002)
    parser.add_argument(
        "--structure_shrink_max_penalty",
        type=float,
        default=1.0,
        help="Maximum normalized structure risk injected into keep-with-downweight local shrink.",
    )
    parser.add_argument(
        "--enable_parent_edge_apply_shrink",
        action="store_true",
        help=(
            "Apply an additional GT-free view-conditioned confidence based only on parent rgb_render edge "
            "strength. This downweights residuals on high-gradient target/policy-val pixels without reading "
            "target/test GT."
        ),
    )
    parser.add_argument("--parent_edge_apply_shrink_weight", type=float, default=0.0)
    parser.add_argument("--parent_edge_apply_shrink_tau", type=float, default=0.05)
    parser.add_argument("--parent_edge_apply_shrink_min_multiplier", type=float, default=0.25)
    parser.add_argument(
        "--enable_policy_val_view_consistency_confidence",
        action="store_true",
        help=(
            "Fit a train-policy-val camera-direction confidence profile from views where the residual "
            "candidate actually improves held-out train views. At apply time this is target-GT-free: "
            "it only compares the target camera direction to the certified positive policy-val views."
        ),
    )
    parser.add_argument("--view_confidence_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--view_confidence_min_ssim_gain", type=float, default=-1.0)
    parser.add_argument("--view_confidence_min_l1_gain", type=float, default=-1.0)
    parser.add_argument("--view_confidence_min_lpips_gain", type=float, default=-1.0)
    parser.add_argument("--view_confidence_kernel_sigma", type=float, default=0.35)
    parser.add_argument("--view_confidence_min_confidence", type=float, default=0.05)
    parser.add_argument(
        "--enable_policy_val_view_alpha_cap",
        action="store_true",
        help=(
            "Fit a train-policy-val camera-direction alpha-cap profile. Target/test apply stays GT-free: "
            "each view receives at most the nearest certified policy-val residual strength."
        ),
    )
    parser.add_argument(
        "--view_alpha_cap_selection_mode",
        choices=("smallest_safe", "best_safe"),
        default="smallest_safe",
    )
    parser.add_argument("--view_alpha_cap_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--view_alpha_cap_min_ssim_gain", type=float, default=-1.0e-5)
    parser.add_argument("--view_alpha_cap_min_l1_gain", type=float, default=-1.0e-6)
    parser.add_argument("--view_alpha_cap_min_lpips_gain", type=float, default=-1.0)
    parser.add_argument("--view_alpha_cap_kernel_sigma", type=float, default=0.35)
    parser.add_argument("--view_alpha_cap_min_confidence", type=float, default=0.05)
    parser.add_argument("--view_alpha_cap_fallback_alpha", type=float, default=0.0)
    parser.add_argument(
        "--view_alpha_cap_seed_stage",
        choices=("pre_guard", "post_view_confidence"),
        default="pre_guard",
        help=(
            "Choose which train-policy-val evidence seeds the view alpha-cap profile. pre_guard freezes "
            "the candidate after basis guards but before sparse/face/bin/view-confidence guards can replace "
            "it with a no-op; final acceptance still uses the fully guarded policy-val evaluation."
        ),
    )
    parser.add_argument(
        "--enable_policy_val_face_gain_guard",
        action="store_true",
        help=(
            "After global train policy-val selection, build a face-level no-regression allowlist "
            "from policy-val residual gains and apply residuals only on allowed faces."
        ),
    )
    parser.add_argument("--face_gain_guard_min_face_samples", type=int, default=256)
    parser.add_argument("--face_gain_guard_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--face_gain_guard_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument(
        "--enable_policy_val_bin_uncertainty_guard",
        action="store_true",
        help=(
            "After train policy-val selection, build a face/UV-bin no-regression allowlist "
            "with variance/sign uncertainty filters and apply residuals only on allowed bins."
        ),
    )
    parser.add_argument(
        "--enable_policy_val_sparse_residual_materialization",
        action="store_true",
        help=(
            "Before final risk-gate selection, sparsify residual materialization at face/UV-bin level "
            "using only train policy-val evidence. Unstable bins are not rendered on target views."
        ),
    )
    parser.add_argument(
        "--enable_policy_val_sparse_materialization_frontier",
        action="store_true",
        help=(
            "If the strict sparse materialization certificate selects no bins, use a policy-val "
            "risk-adjusted local frontier instead of falling back to a no-op candidate."
        ),
    )
    parser.add_argument(
        "--sparse_materialization_seed_min_relative_gain",
        type=float,
        default=0.0,
        help="Minimum nonzero policy-val relative gain required to seed sparse residual materialization.",
    )
    parser.add_argument("--sparse_materialization_min_bin_samples", type=int, default=16)
    parser.add_argument(
        "--sparse_materialization_min_bin_views",
        type=int,
        default=1,
        help="Minimum number of train policy-val views that must observe a face/UV bin before sparse materialization can keep it.",
    )
    parser.add_argument("--sparse_materialization_min_relative_gain", type=float, default=0.0)
    parser.add_argument(
        "--sparse_materialization_min_view_relative_gain",
        type=float,
        default=-float("inf"),
        help="Minimum per-view relative gain required for a sparse materialization bin; use near zero for no-tail-regression local certificates.",
    )
    parser.add_argument("--sparse_materialization_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument("--sparse_materialization_frontier_min_positive_view_fraction", type=float, default=0.55)
    parser.add_argument("--sparse_materialization_frontier_min_risk_adjusted_gain", type=float, default=0.0)
    parser.add_argument("--sparse_materialization_frontier_min_sample_quantile", type=float, default=0.75)
    parser.add_argument(
        "--sparse_materialization_max_mean_variance",
        type=float,
        default=-1.0,
        help="If >=0, reject sparse materialization bins above this mean residual RGB variance.",
    )
    parser.add_argument(
        "--sparse_materialization_min_mean_sign_consistency",
        type=float,
        default=0.0,
        help="If >0, reject sparse materialization bins below this mean sign-consistency.",
    )
    parser.add_argument(
        "--enable_sparse_materialization_target_visible_expansion",
        action="store_true",
        help=(
            "After train-policy-val sparse certification, add extra locally safe bins that are visible "
            "on GT-free target evidence. This expands target footprint without reading target/test GT."
        ),
    )
    parser.add_argument("--sparse_materialization_target_visible_min_pixels", type=int, default=1)
    parser.add_argument("--sparse_materialization_target_visible_min_views", type=int, default=1)
    parser.add_argument("--sparse_materialization_target_visible_min_policy_samples", type=int, default=1)
    parser.add_argument(
        "--sparse_materialization_target_visible_min_positive_view_fraction",
        type=float,
        default=-1.0,
        help=(
            "Minimum train-policy-val positive-view fraction for target-visible sparse expansion; "
            "<0 reuses sparse_materialization_frontier_min_positive_view_fraction."
        ),
    )
    parser.add_argument(
        "--sparse_materialization_target_visible_max_extra_bins",
        type=int,
        default=0,
        help="Maximum extra target-visible bins to add after sparse certification; 0 means no cap.",
    )
    parser.add_argument(
        "--enable_train_only_target_impact_residual_basis",
        action="store_true",
        help=(
            "Allow sparse materialization to cover high-footprint target-visible bins using only "
            "train/policy-val residual evidence, even when a target-visible bin has no policy-val row. "
            "This records added-without-policy-row bins separately and never reads target/test GT."
        ),
    )
    parser.add_argument("--target_impact_min_pixels", type=int, default=1)
    parser.add_argument("--target_impact_min_views", type=int, default=1)
    parser.add_argument(
        "--target_impact_min_policy_samples",
        type=int,
        default=0,
        help="Minimum policy-val samples required for target-impact basis bins; 0 permits atlas-only bins.",
    )
    parser.add_argument(
        "--target_impact_max_extra_bins",
        type=int,
        default=0,
        help="Maximum extra target-impact bins to add after sparse certification; 0 means no cap.",
    )
    parser.add_argument(
        "--target_impact_max_views",
        type=int,
        default=0,
        help="Optional max target footprint views used by target-impact basis; 0 uses all available views.",
    )
    parser.add_argument(
        "--target_impact_carrier_fill_mode",
        choices=("off", "no_policy_rows", "all_added"),
        default="off",
        help=(
            "After target-impact sparse expansion, fill selected bins with a train-fit face residual carrier. "
            "This is target/test-GT-free and is intended to increase residual capacity, not footprint alone."
        ),
    )
    parser.add_argument("--target_impact_carrier_fill_blend", type=float, default=0.5)
    parser.add_argument("--target_impact_carrier_fill_min_face_samples", type=int, default=128)
    parser.add_argument("--target_impact_carrier_fill_min_norm", type=float, default=1.0e-4)
    parser.add_argument("--target_impact_carrier_fill_synthetic_count", type=int, default=1)
    parser.add_argument(
        "--target_impact_multisample_fill_mode",
        choices=("off", "no_policy_rows", "all_added"),
        default="off",
        help=(
            "After target-impact sparse expansion, fill selected bins from nearby train-fit residual samples. "
            "This reads no target/test RGB GT and is re-evaluated by the policy-val gate."
        ),
    )
    parser.add_argument("--target_impact_multisample_fill_radius", type=int, default=1)
    parser.add_argument("--target_impact_multisample_fill_min_samples", type=int, default=4)
    parser.add_argument("--target_impact_multisample_fill_max_samples_per_bin", type=int, default=128)
    parser.add_argument("--target_impact_multisample_fill_max_views", type=int, default=0)
    parser.add_argument("--target_impact_multisample_fill_blend", type=float, default=1.0)
    parser.add_argument("--target_impact_multisample_fill_kernel_sigma", type=float, default=1.0)
    parser.add_argument("--target_impact_multisample_fill_min_norm", type=float, default=1.0e-4)
    parser.add_argument("--target_impact_multisample_fill_synthetic_count", type=int, default=2)
    parser.add_argument(
        "--target_impact_affine_fill_mode",
        choices=("off", "no_policy_rows", "all_added"),
        default="off",
        help=(
            "After target-impact sparse expansion, fit train-only face-local residual fields "
            "and materialize selected target-impact bins before policy-val recertification."
        ),
    )
    parser.add_argument(
        "--target_impact_affine_fill_feature_mode",
        choices=("face_uv_normal_camera_ridge", "face_uv_patch_mixture_ridge"),
        default="face_uv_patch_mixture_ridge",
    )
    parser.add_argument("--target_impact_affine_fill_min_samples", type=int, default=64)
    parser.add_argument("--target_impact_affine_fill_max_samples_per_face", type=int, default=4096)
    parser.add_argument("--target_impact_affine_fill_max_views", type=int, default=0)
    parser.add_argument("--target_impact_affine_fill_blend", type=float, default=1.0)
    parser.add_argument("--target_impact_affine_fill_ridge", type=float, default=1.0e-2)
    parser.add_argument("--target_impact_affine_fill_max_condition", type=float, default=1.0e7)
    parser.add_argument("--target_impact_affine_fill_min_norm", type=float, default=1.0e-4)
    parser.add_argument("--target_impact_affine_fill_synthetic_count", type=int, default=4)
    parser.add_argument(
        "--enable_sparse_materialization_target_connected_region_growth",
        action="store_true",
        help=(
            "Grow extra target-visible bins only when they are same-face UV neighbors of "
            "pre-expansion sparse-certified bins and policy-val evidence has no strong regression."
        ),
    )
    parser.add_argument("--sparse_materialization_target_connected_radius", type=int, default=1)
    parser.add_argument("--sparse_materialization_target_connected_min_pixels", type=int, default=1)
    parser.add_argument("--sparse_materialization_target_connected_min_views", type=int, default=1)
    parser.add_argument("--sparse_materialization_target_connected_min_policy_samples", type=int, default=1)
    parser.add_argument(
        "--sparse_materialization_target_connected_min_positive_view_fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--sparse_materialization_target_connected_max_negative_relative_gain",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--sparse_materialization_target_connected_max_negative_min_view_gain",
        type=float,
        default=0.05,
    )
    parser.add_argument("--sparse_materialization_target_connected_max_extra_bins", type=int, default=0)
    parser.add_argument("--bin_uncertainty_guard_min_bin_samples", type=int, default=64)
    parser.add_argument("--bin_uncertainty_guard_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--bin_uncertainty_guard_min_positive_view_fraction", type=float, default=0.75)
    parser.add_argument(
        "--bin_uncertainty_guard_empty_intersection_policy",
        choices=["reject", "sparse_if_post_accepted"],
        default="reject",
        help=(
            "Policy when sparse materialization bins pass their post gate but the later bin-uncertainty "
            "guard has an empty intersection. The bridge option keeps sparse-certified bins and records "
            "the fallback explicitly instead of forcing a no-op."
        ),
    )
    parser.add_argument(
        "--bin_uncertainty_guard_max_mean_variance",
        type=float,
        default=-1.0,
        help="If >=0, reject policy-val allowed bins whose atlas residual RGB variance exceeds this mean.",
    )
    parser.add_argument(
        "--bin_uncertainty_guard_min_mean_sign_consistency",
        type=float,
        default=0.0,
        help="If >0, reject policy-val allowed bins whose mean sign-consistency is below this value.",
    )
    parser.add_argument(
        "--enable_policy_val_image_ssim_gate",
        action="store_true",
        help="Compute train policy-val image-level SSIM rows and allow SSIM-aware risk gates.",
    )
    parser.add_argument("--policy_val_ssim_max_size", type=int, default=512)
    parser.add_argument("--min_policy_val_ssim_mean_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_ssim_positive_view_fraction", type=float, default=0.0)
    parser.add_argument("--min_policy_val_ssim_min_view_gain", type=float, default=-1.0)
    parser.add_argument(
        "--enable_policy_val_image_l1_gate",
        action="store_true",
        help=(
            "Compute a train policy-val full-image L1 proxy and allow L1-aware risk gates. "
            "The gain is L1_before - L1_after, so positive means lower image error."
        ),
    )
    parser.add_argument("--policy_val_l1_max_size", type=int, default=512)
    parser.add_argument("--min_policy_val_l1_mean_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_l1_positive_view_fraction", type=float, default=0.0)
    parser.add_argument("--min_policy_val_l1_min_view_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_l1_cvar20_view_gain", type=float, default=-1.0)
    parser.add_argument(
        "--enable_policy_val_image_lpips_gate",
        action="store_true",
        help=(
            "Compute train policy-val full-image LPIPS rows and allow LPIPS-aware risk gates. "
            "The gain is LPIPS_before - LPIPS_after, so positive means lower perceptual distance."
        ),
    )
    parser.add_argument("--policy_val_lpips_max_size", type=int, default=512)
    parser.add_argument("--min_policy_val_lpips_mean_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_lpips_positive_view_fraction", type=float, default=0.0)
    parser.add_argument("--min_policy_val_lpips_min_view_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_lpips_cvar20_view_gain", type=float, default=-1.0)
    parser.add_argument(
        "--enable_policy_val_effective_margin_gate",
        action="store_true",
        help=(
            "Require selected train policy-val improvements to exceed explicit effect-size margins, not only "
            "non-negative/noisy gains. This is intended to reject tiny policy-val wins that are unlikely to "
            "survive target/test generalization."
        ),
    )
    parser.add_argument("--min_policy_val_effective_relative_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_effective_ssim_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_effective_l1_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_effective_ssim_cvar20_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_effective_l1_cvar20_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_effective_lpips_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_effective_lpips_cvar20_gain", type=float, default=-1.0)
    parser.add_argument(
        "--min_target_changed_fraction",
        type=float,
        default=0.0,
        help="Optional final gate: reject accepted atlases that modify too little target support.",
    )
    parser.add_argument(
        "--enable_target_support_candidate_selection",
        action="store_true",
        help=(
            "Before auto-policy candidate selection, estimate target-visible support for each train-policy-safe "
            "candidate without reading target GT, then rank non-regressive survivors by target support first."
        ),
    )
    parser.add_argument(
        "--enable_target_visible_energy_score",
        action="store_true",
        help=(
            "When target-support candidate selection is enabled, rank train-policy-safe candidates by "
            "GT-free target-visible residual energy before raw changed-area coverage. This favors candidates "
            "that are both visible on the target trajectory and large enough to affect rendered appearance."
        ),
    )
    parser.add_argument(
        "--target_support_prerank_top_k",
        type=int,
        default=0,
        help=(
            "If >0, pre-rank support candidate sets by GT-free target face coverage and keep only top-K "
            "before expensive atlas refit/policy-val. This only affects support-set ladders."
        ),
    )
    parser.add_argument(
        "--target_support_prerank_max_views",
        type=int,
        default=0,
        help="Optional maximum number of target views used by the cheap support pre-rank proxy; 0 uses all.",
    )
    parser.add_argument(
        "--enable_policy_candidate_dominance_pruning",
        action="store_true",
        help=(
            "Before expensive atlas fitting, prune strictly equivalent policy candidates. "
            "Equivalence requires the same support face set, fill mode, texture size, prior blend, and RGB cap."
        ),
    )
    parser.add_argument(
        "--policy_candidate_early_stop_mode",
        choices=("none", "first_accepted"),
        default="none",
        help=(
            "Optional candidate-loop early stop. first_accepted stops after the first policy-val accepted "
            "candidate; it is disabled when target-support ranking or prior-bin hybrid search is requested."
        ),
    )
    parser.add_argument(
        "--enable_target_footprint_bin_certificate",
        action="store_true",
        help=(
            "When building a policy-val prior-bin hybrid atlas, also require each copied face/UV bin to have "
            "GT-free target-view footprint support. This guards against policy-val-positive bins that never "
            "act on the target split."
        ),
    )
    parser.add_argument("--target_footprint_min_bin_pixels", type=int, default=1)
    parser.add_argument("--target_footprint_min_views", type=int, default=1)
    parser.add_argument("--target_footprint_min_view_fraction", type=float, default=0.0)
    parser.add_argument(
        "--target_footprint_max_views",
        type=int,
        default=0,
        help="Optional maximum target views used for bin-footprint certification; 0 uses all target views.",
    )
    parser.add_argument(
        "--enable_target_footprint_tail_risk_certificate",
        action="store_true",
        help=(
            "When building a policy-val prior-bin hybrid atlas, reject target-covered bins whose "
            "policy-val per-view residual-error gain has unsafe tail behavior."
        ),
    )
    parser.add_argument(
        "--target_footprint_tail_risk_all_bins",
        action="store_true",
        help=(
            "Apply the target-footprint tail-risk certificate to every policy-val candidate bin, "
            "not only bins covered by target footprint evidence."
        ),
    )
    parser.add_argument(
        "--target_footprint_tail_risk_min_positive_view_fraction",
        type=float,
        default=1.0,
        help="Minimum per-bin policy-val positive-view fraction required by the tail-risk certificate.",
    )
    parser.add_argument(
        "--target_footprint_tail_risk_min_min_view_gain",
        type=float,
        default=0.0,
        help="Minimum per-bin worst policy-val view residual-error gain required by the tail-risk certificate.",
    )
    parser.add_argument(
        "--target_footprint_tail_risk_min_cvar20_view_gain",
        type=float,
        default=0.0,
        help="Minimum per-bin bottom-20%% policy-val view residual-error gain required by the tail-risk certificate.",
    )
    parser.add_argument(
        "--write_noop_on_reject",
        action="store_true",
        help=(
            "When policy/coverage gates reject the candidate, still materialize a fair no-op method "
            "by exporting target evidence RGB renders unless the legacy source_model mode is requested."
        ),
    )
    parser.add_argument(
        "--noop_fallback_source",
        choices=["target_evidence", "source_model"],
        default="target_evidence",
        help=(
            "Where rejected no-op renders come from. target_evidence matches the atlas/no-op protocol; "
            "source_model is retained only for legacy compatibility."
        ),
    )
    parser.add_argument("--fill_empty_with_face_mean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if int(args.texture_size) <= 0:
        parser.error("--texture_size must be > 0")
    if int(args.adaptive_texture_size_ladder_max_size) <= 0:
        parser.error("--adaptive_texture_size_ladder_max_size must be > 0")
    if float(args.adaptive_texture_size_ladder_min_fit_samples_per_face) < 0.0:
        parser.error("--adaptive_texture_size_ladder_min_fit_samples_per_face must be >= 0")
    if float(args.adaptive_texture_size_ladder_min_samples_per_current_bin) < 0.0:
        parser.error("--adaptive_texture_size_ladder_min_samples_per_current_bin must be >= 0")
    if float(args.adaptive_texture_size_ladder_min_mean_l1) < 0.0:
        parser.error("--adaptive_texture_size_ladder_min_mean_l1 must be >= 0")
    if float(args.atlas_nearest_fill_decay) < 0.0:
        parser.error("--atlas_nearest_fill_decay must be >= 0")
    try:
        surface_multiscale_prior_block_sizes = parse_int_candidates(
            str(args.surface_multiscale_prior_block_sizes),
            2,
        )
    except ValueError as exc:
        parser.error(f"invalid --surface_multiscale_prior_block_sizes: {exc}")
    if int(args.surface_multiscale_prior_min_bin_samples) <= 0:
        parser.error("--surface_multiscale_prior_min_bin_samples must be > 0")
    if float(args.surface_multiscale_prior_count_tau) < 0.0:
        parser.error("--surface_multiscale_prior_count_tau must be >= 0")
    if not 0.0 <= float(args.surface_multiscale_prior_blend) <= 1.0:
        parser.error("--surface_multiscale_prior_blend must be in [0, 1]")
    try:
        surface_multiscale_prior_blend_candidates = (
            parse_float_candidates(str(args.surface_multiscale_prior_blend_candidates))
            if str(args.surface_multiscale_prior_blend_candidates).strip()
            else [float(args.surface_multiscale_prior_blend)]
        )
    except ValueError as exc:
        parser.error(f"invalid --surface_multiscale_prior_blend_candidates: {exc}")
    if not surface_multiscale_prior_blend_candidates:
        surface_multiscale_prior_blend_candidates = [float(args.surface_multiscale_prior_blend)]
    surface_multiscale_prior_blend_candidates = sorted(
        set(round(float(x), 8) for x in surface_multiscale_prior_blend_candidates)
    )
    if str(args.surface_multiscale_prior_mode) == "none":
        surface_multiscale_prior_blend_candidates = [float(args.surface_multiscale_prior_blend)]
    for blend_candidate in surface_multiscale_prior_blend_candidates:
        if not 0.0 <= float(blend_candidate) <= 1.0:
            parser.error("--surface_multiscale_prior_blend_candidates values must be in [0, 1]")
    if float(args.max_abs_delta_rgb) < 0.0:
        parser.error("--max_abs_delta_rgb must be >= 0")
    try:
        max_abs_delta_rgb_candidates = (
            parse_float_candidates(str(args.max_abs_delta_rgb_candidates))
            if str(args.max_abs_delta_rgb_candidates).strip()
            else [float(args.max_abs_delta_rgb)]
        )
    except ValueError as exc:
        parser.error(f"invalid --max_abs_delta_rgb_candidates: {exc}")
    if not max_abs_delta_rgb_candidates:
        max_abs_delta_rgb_candidates = [float(args.max_abs_delta_rgb)]
    max_abs_delta_rgb_candidates = sorted(
        set(round(float(x), 8) for x in max_abs_delta_rgb_candidates)
    )
    for cap_candidate in max_abs_delta_rgb_candidates:
        if float(cap_candidate) < 0.0:
            parser.error("--max_abs_delta_rgb_candidates values must be >= 0")
    if float(args.surface_multiscale_prior_min_prior_weight) < 0.0:
        parser.error("--surface_multiscale_prior_min_prior_weight must be >= 0")
    if int(args.surface_multiscale_prior_min_direct_samples) <= 0:
        parser.error("--surface_multiscale_prior_min_direct_samples must be > 0")
    if not 0.0 <= float(args.surface_multiscale_prior_min_sign_consistency) <= 1.0:
        parser.error("--surface_multiscale_prior_min_sign_consistency must be in [0, 1]")
    if float(args.surface_multiscale_prior_min_cosine) < -1.0 or float(args.surface_multiscale_prior_min_cosine) > 1.0:
        parser.error("--surface_multiscale_prior_min_cosine must be in [-1, 1]")
    if int(args.prior_bin_gain_hybrid_min_bin_samples) <= 0:
        parser.error("--prior_bin_gain_hybrid_min_bin_samples must be > 0")
    if int(args.prior_bin_gain_hybrid_min_views) <= 0:
        parser.error("--prior_bin_gain_hybrid_min_views must be > 0")
    if float(args.prior_bin_gain_hybrid_min_abs_gain) < 0.0:
        parser.error("--prior_bin_gain_hybrid_min_abs_gain must be >= 0")
    if not 0.0 <= float(args.prior_bin_gain_hybrid_min_positive_view_fraction) <= 1.0:
        parser.error("--prior_bin_gain_hybrid_min_positive_view_fraction must be in [0, 1]")
    if int(args.prior_bin_gain_hybrid_max_profile_bins) < 0:
        parser.error("--prior_bin_gain_hybrid_max_profile_bins must be >= 0")
    if bool(args.enable_policy_val_source_mixture) and not bool(args.enable_policy_val_prior_bin_gain_hybrid):
        parser.error("--enable_policy_val_source_mixture requires --enable_policy_val_prior_bin_gain_hybrid")
    if float(args.source_mixture_ridge) < 0.0:
        parser.error("--source_mixture_ridge must be >= 0")
    if not 0.0 <= float(args.source_mixture_min_weight) <= 1.0:
        parser.error("--source_mixture_min_weight must be in [0, 1]")
    if int(args.policy_val_ssim_alpha_refinement_steps) < 0:
        parser.error("--policy_val_ssim_alpha_refinement_steps must be >= 0")
    if float(args.policy_val_ssim_alpha_refinement_min_alpha) < 0.0:
        parser.error("--policy_val_ssim_alpha_refinement_min_alpha must be >= 0")
    if float(args.prior_bin_gain_hybrid_min_l1_abs_gain) < 0.0:
        parser.error("--prior_bin_gain_hybrid_min_l1_abs_gain must be >= 0")
    if not 0.0 <= float(args.prior_bin_gain_hybrid_min_l1_positive_view_fraction) <= 1.0:
        parser.error("--prior_bin_gain_hybrid_min_l1_positive_view_fraction must be in [0, 1]")
    if int(args.view_conditioned_basis_min_bin_samples) <= 0:
        parser.error("--view_conditioned_basis_min_bin_samples must be > 0")
    if float(args.view_conditioned_basis_ridge) < 0.0:
        parser.error("--view_conditioned_basis_ridge must be >= 0")
    if str(args.view_conditioned_basis_ood_mode) != "none":
        if float(args.view_conditioned_basis_ood_max_z) <= 0.0:
            parser.error("--view_conditioned_basis_ood_max_z must be > 0 when OOD mode is enabled")
        if float(args.view_conditioned_basis_ood_min_std) <= 0.0:
            parser.error("--view_conditioned_basis_ood_min_std must be > 0 when OOD mode is enabled")
    if int(args.view_cluster_expert_count) < 1:
        parser.error("--view_cluster_expert_count must be >= 1")
    if int(args.view_cluster_min_views) < 1:
        parser.error("--view_cluster_min_views must be >= 1")
    if int(args.view_cluster_min_bin_samples) < 1:
        parser.error("--view_cluster_min_bin_samples must be >= 1")
    if bool(args.enable_view_cluster_local_shrink) and int(args.view_cluster_expert_count) <= 1:
        parser.error("--enable_view_cluster_local_shrink requires --view_cluster_expert_count > 1")
    if int(args.teacher_distilled_basis_min_face_samples) <= 0:
        parser.error("--teacher_distilled_basis_min_face_samples must be > 0")
    if int(args.adaptive_teacher_basis_min_face_samples_floor) <= 0:
        parser.error("--adaptive_teacher_basis_min_face_samples_floor must be > 0")
    if not 0.0 <= float(args.adaptive_teacher_basis_support_quantile) <= 1.0:
        parser.error("--adaptive_teacher_basis_support_quantile must be in [0, 1]")
    if float(args.adaptive_teacher_basis_low_support_ridge_scale) < 0.0:
        parser.error("--adaptive_teacher_basis_low_support_ridge_scale must be >= 0")
    if float(args.teacher_distilled_basis_ridge) < 0.0:
        parser.error("--teacher_distilled_basis_ridge must be >= 0")
    if float(args.teacher_distilled_basis_ood_max_z) <= 0.0:
        parser.error("--teacher_distilled_basis_ood_max_z must be > 0")
    if float(args.teacher_distilled_basis_ood_min_std) <= 0.0:
        parser.error("--teacher_distilled_basis_ood_min_std must be > 0")
    if not 0.0 <= float(args.teacher_distilled_basis_blend) <= 1.0:
        parser.error("--teacher_distilled_basis_blend must be in [0, 1]")
    if int(args.teacher_distilled_low_rank_texture_rank) <= 0:
        parser.error("--teacher_distilled_low_rank_texture_rank must be > 0")
    try:
        teacher_low_rank_texture_rank_candidates = (
            parse_int_candidates(
                str(args.teacher_distilled_low_rank_texture_rank_candidates),
                int(args.teacher_distilled_low_rank_texture_rank),
            )
            if _is_low_rank_teacher_texture_mode(str(args.teacher_distilled_basis_mode))
            else [0]
        )
    except ValueError as exc:
        parser.error(f"invalid --teacher_distilled_low_rank_texture_rank_candidates: {exc}")
    if int(args.face_gain_guard_min_face_samples) <= 0:
        parser.error("--face_gain_guard_min_face_samples must be > 0")
    if not 0.0 <= float(args.face_gain_guard_min_positive_view_fraction) <= 1.0:
        parser.error("--face_gain_guard_min_positive_view_fraction must be in [0, 1]")
    if int(args.bin_uncertainty_guard_min_bin_samples) <= 0:
        parser.error("--bin_uncertainty_guard_min_bin_samples must be > 0")
    if not 0.0 <= float(args.bin_uncertainty_guard_min_positive_view_fraction) <= 1.0:
        parser.error("--bin_uncertainty_guard_min_positive_view_fraction must be in [0, 1]")
    if not 0.0 <= float(args.bin_uncertainty_guard_min_mean_sign_consistency) <= 1.0:
        parser.error("--bin_uncertainty_guard_min_mean_sign_consistency must be in [0, 1]")
    if int(args.bin_alpha_calibration_min_bin_samples) <= 0:
        parser.error("--bin_alpha_calibration_min_bin_samples must be > 0")
    if not 0.0 <= float(args.bin_alpha_calibration_min_positive_view_fraction) <= 1.0:
        parser.error("--bin_alpha_calibration_min_positive_view_fraction must be in [0, 1]")
    if float(args.bin_alpha_calibration_max_alpha) < float(args.bin_alpha_calibration_min_alpha):
        parser.error("--bin_alpha_calibration_max_alpha must be >= --bin_alpha_calibration_min_alpha")
    if bool(args.enable_policy_val_bin_alpha_calibration) and bool(
        args.enable_policy_val_bin_rgb_alpha_calibration
    ):
        parser.error("enable either scalar bin alpha or RGB bin alpha calibration, not both")
    if int(args.bin_uncertainty_shrink_min_bin_samples) <= 0:
        parser.error("--bin_uncertainty_shrink_min_bin_samples must be > 0")
    if int(args.bin_uncertainty_shrink_min_bin_views) <= 0:
        parser.error("--bin_uncertainty_shrink_min_bin_views must be > 0")
    if not 0.0 <= float(args.bin_uncertainty_shrink_min_positive_view_fraction) <= 1.0:
        parser.error("--bin_uncertainty_shrink_min_positive_view_fraction must be in [0, 1]")
    if not 0.0 <= float(args.bin_uncertainty_shrink_min_mean_sign_consistency) <= 1.0:
        parser.error("--bin_uncertainty_shrink_min_mean_sign_consistency must be in [0, 1]")
    if float(args.bin_uncertainty_shrink_count_tau) < 0.0:
        parser.error("--bin_uncertainty_shrink_count_tau must be >= 0")
    if float(args.bin_uncertainty_shrink_gain_tau) < 0.0:
        parser.error("--bin_uncertainty_shrink_gain_tau must be >= 0")
    if float(args.bin_uncertainty_shrink_variance_scale) < 0.0:
        parser.error("--bin_uncertainty_shrink_variance_scale must be >= 0")
    if float(args.bin_uncertainty_shrink_sign_power) < 0.0:
        parser.error("--bin_uncertainty_shrink_sign_power must be >= 0")
    if float(args.bin_uncertainty_shrink_max_shrink) < float(args.bin_uncertainty_shrink_min_shrink):
        parser.error("--bin_uncertainty_shrink_max_shrink must be >= --bin_uncertainty_shrink_min_shrink")
    if bool(args.enable_policy_val_image_l1_bin_certificate) and not bool(
        args.enable_policy_val_bin_uncertainty_shrink
    ):
        parser.error("--enable_policy_val_image_l1_bin_certificate requires --enable_policy_val_bin_uncertainty_shrink")
    if float(args.image_l1_bin_certificate_min_relative_gain) < 0.0:
        parser.error("--image_l1_bin_certificate_min_relative_gain must be >= 0")
    if not 0.0 <= float(args.image_l1_bin_certificate_min_positive_view_fraction) <= 1.0:
        parser.error("--image_l1_bin_certificate_min_positive_view_fraction must be in [0, 1]")
    if float(args.image_l1_bin_certificate_gain_tau) < 0.0:
        parser.error("--image_l1_bin_certificate_gain_tau must be >= 0")
    if int(args.image_l1_bin_certificate_pool_radius) < 0:
        parser.error("--image_l1_bin_certificate_pool_radius must be >= 0")
    if bool(args.enable_policy_val_image_l1_region_expansion) and not bool(
        args.enable_policy_val_image_l1_bin_certificate
    ):
        parser.error("--enable_policy_val_image_l1_region_expansion requires --enable_policy_val_image_l1_bin_certificate")
    if int(args.image_l1_region_expansion_radius) < 0:
        parser.error("--image_l1_region_expansion_radius must be >= 0")
    if int(args.image_l1_region_expansion_max_bins_per_seed) < 0:
        parser.error("--image_l1_region_expansion_max_bins_per_seed must be >= 0")
    if int(args.image_l1_region_expansion_min_neighbor_samples) < 1:
        parser.error("--image_l1_region_expansion_min_neighbor_samples must be >= 1")
    if int(args.image_l1_region_expansion_min_neighbor_views) < 1:
        parser.error("--image_l1_region_expansion_min_neighbor_views must be >= 1")
    if float(args.image_l1_region_expansion_max_negative_relative_gain) < 0.0:
        parser.error("--image_l1_region_expansion_max_negative_relative_gain must be >= 0")
    if float(args.image_l1_region_expansion_max_negative_image_l1_gain) < 0.0:
        parser.error("--image_l1_region_expansion_max_negative_image_l1_gain must be >= 0")
    if not 0.0 <= float(args.image_l1_region_expansion_shrink_decay) <= 1.0:
        parser.error("--image_l1_region_expansion_shrink_decay must be in [0, 1]")
    if float(args.image_l1_bin_alpha_max_alpha) < 0.0:
        parser.error("--image_l1_bin_alpha_max_alpha must be >= 0")
    if int(args.image_l1_bin_alpha_min_bin_samples) < 1:
        parser.error("--image_l1_bin_alpha_min_bin_samples must be >= 1")
    if float(args.image_l1_bin_alpha_min_relative_gain) < 0.0:
        parser.error("--image_l1_bin_alpha_min_relative_gain must be >= 0")
    if not 0.0 <= float(args.image_l1_bin_alpha_min_positive_view_fraction) <= 1.0:
        parser.error("--image_l1_bin_alpha_min_positive_view_fraction must be in [0, 1]")
    if float(args.image_l1_bin_alpha_count_tau) < 0.0:
        parser.error("--image_l1_bin_alpha_count_tau must be >= 0")
    if int(args.image_l1_bin_alpha_max_profile_bins) < 0:
        parser.error("--image_l1_bin_alpha_max_profile_bins must be >= 0")
    if float(args.image_linear_generator_ridge) < 0.0:
        parser.error("--image_linear_generator_ridge must be >= 0")
    if int(args.image_linear_generator_train_max_samples_per_view) < 1:
        parser.error("--image_linear_generator_train_max_samples_per_view must be >= 1")
    if int(args.image_linear_generator_max_train_samples) < 1:
        parser.error("--image_linear_generator_max_train_samples must be >= 1")
    if float(args.image_linear_generator_output_cap) < 0.0:
        parser.error("--image_linear_generator_output_cap must be >= 0")
    if int(args.image_linear_generator_irls_iterations) < 1:
        parser.error("--image_linear_generator_irls_iterations must be >= 1")
    if float(args.image_linear_generator_huber_delta) <= 0.0:
        parser.error("--image_linear_generator_huber_delta must be > 0")
    if float(args.image_linear_generator_min_descent_margin) < 0.0:
        parser.error("--image_linear_generator_min_descent_margin must be >= 0")
    if int(args.image_linear_generator_min_training_samples) < 1:
        parser.error("--image_linear_generator_min_training_samples must be >= 1")
    if int(args.image_linear_generator_expert_min_training_samples) < 1:
        parser.error("--image_linear_generator_expert_min_training_samples must be >= 1")
    if float(args.image_linear_generator_expert_shrink_tau) < 0.0:
        parser.error("--image_linear_generator_expert_shrink_tau must be >= 0")
    if int(args.image_linear_generator_face_reliability_min_face_samples) < 1:
        parser.error("--image_linear_generator_face_reliability_min_face_samples must be >= 1")
    if not 0.0 <= float(args.image_linear_generator_face_reliability_min_positive_view_fraction) <= 1.0:
        parser.error("--image_linear_generator_face_reliability_min_positive_view_fraction must be in [0, 1]")
    if not 0.0 <= float(args.image_linear_generator_face_reliability_fallback_multiplier) <= 1.0:
        parser.error("--image_linear_generator_face_reliability_fallback_multiplier must be in [0, 1]")
    if bool(args.enable_policy_val_structure_aware_shrink) and not bool(args.enable_policy_val_bin_uncertainty_shrink):
        parser.error("--enable_policy_val_structure_aware_shrink requires --enable_policy_val_bin_uncertainty_shrink")
    if int(args.sparse_materialization_min_bin_samples) < 1:
        parser.error("--sparse_materialization_min_bin_samples must be >= 1")
    if int(args.sparse_materialization_min_bin_views) < 1:
        parser.error("--sparse_materialization_min_bin_views must be >= 1")
    if float(args.sparse_materialization_min_relative_gain) < 0.0:
        parser.error("--sparse_materialization_min_relative_gain must be >= 0")
    if not 0.0 <= float(args.sparse_materialization_min_positive_view_fraction) <= 1.0:
        parser.error("--sparse_materialization_min_positive_view_fraction must be in [0, 1]")
    if not 0.0 <= float(args.sparse_materialization_frontier_min_positive_view_fraction) <= 1.0:
        parser.error("--sparse_materialization_frontier_min_positive_view_fraction must be in [0, 1]")
    if not 0.0 <= float(args.sparse_materialization_frontier_min_sample_quantile) <= 1.0:
        parser.error("--sparse_materialization_frontier_min_sample_quantile must be in [0, 1]")
    if float(args.sparse_materialization_min_mean_sign_consistency) < 0.0:
        parser.error("--sparse_materialization_min_mean_sign_consistency must be >= 0")
    if int(args.sparse_materialization_target_visible_min_pixels) < 1:
        parser.error("--sparse_materialization_target_visible_min_pixels must be >= 1")
    if int(args.sparse_materialization_target_visible_min_views) < 1:
        parser.error("--sparse_materialization_target_visible_min_views must be >= 1")
    if int(args.sparse_materialization_target_visible_min_policy_samples) < 1:
        parser.error("--sparse_materialization_target_visible_min_policy_samples must be >= 1")
    if int(args.sparse_materialization_target_visible_max_extra_bins) < 0:
        parser.error("--sparse_materialization_target_visible_max_extra_bins must be >= 0")
    if (
        float(args.sparse_materialization_target_visible_min_positive_view_fraction) >= 0.0
        and not 0.0 <= float(args.sparse_materialization_target_visible_min_positive_view_fraction) <= 1.0
    ):
        parser.error("--sparse_materialization_target_visible_min_positive_view_fraction must be <0 or in [0, 1]")
    if int(args.target_impact_min_pixels) < 1:
        parser.error("--target_impact_min_pixels must be >= 1")
    if int(args.target_impact_min_views) < 1:
        parser.error("--target_impact_min_views must be >= 1")
    if int(args.target_impact_min_policy_samples) < 0:
        parser.error("--target_impact_min_policy_samples must be >= 0")
    if int(args.target_impact_max_extra_bins) < 0:
        parser.error("--target_impact_max_extra_bins must be >= 0")
    if int(args.target_impact_max_views) < 0:
        parser.error("--target_impact_max_views must be >= 0")
    if not 0.0 <= float(args.target_impact_carrier_fill_blend) <= 1.0:
        parser.error("--target_impact_carrier_fill_blend must be in [0, 1]")
    if int(args.target_impact_carrier_fill_min_face_samples) < 0:
        parser.error("--target_impact_carrier_fill_min_face_samples must be >= 0")
    if float(args.target_impact_carrier_fill_min_norm) < 0.0:
        parser.error("--target_impact_carrier_fill_min_norm must be >= 0")
    if int(args.target_impact_carrier_fill_synthetic_count) < 0:
        parser.error("--target_impact_carrier_fill_synthetic_count must be >= 0")
    if (
        str(args.target_impact_carrier_fill_mode) != "off"
        and not bool(args.enable_train_only_target_impact_residual_basis)
    ):
        parser.error("--target_impact_carrier_fill_mode requires --enable_train_only_target_impact_residual_basis")
    if int(args.target_impact_multisample_fill_radius) < 0:
        parser.error("--target_impact_multisample_fill_radius must be >= 0")
    if int(args.target_impact_multisample_fill_min_samples) < 1:
        parser.error("--target_impact_multisample_fill_min_samples must be >= 1")
    if int(args.target_impact_multisample_fill_max_samples_per_bin) < 0:
        parser.error("--target_impact_multisample_fill_max_samples_per_bin must be >= 0")
    if int(args.target_impact_multisample_fill_max_views) < 0:
        parser.error("--target_impact_multisample_fill_max_views must be >= 0")
    if not 0.0 <= float(args.target_impact_multisample_fill_blend) <= 1.0:
        parser.error("--target_impact_multisample_fill_blend must be in [0, 1]")
    if float(args.target_impact_multisample_fill_kernel_sigma) <= 0.0:
        parser.error("--target_impact_multisample_fill_kernel_sigma must be > 0")
    if float(args.target_impact_multisample_fill_min_norm) < 0.0:
        parser.error("--target_impact_multisample_fill_min_norm must be >= 0")
    if int(args.target_impact_multisample_fill_synthetic_count) < 0:
        parser.error("--target_impact_multisample_fill_synthetic_count must be >= 0")
    if (
        str(args.target_impact_multisample_fill_mode) != "off"
        and not bool(args.enable_train_only_target_impact_residual_basis)
    ):
        parser.error("--target_impact_multisample_fill_mode requires --enable_train_only_target_impact_residual_basis")
    if int(args.target_impact_affine_fill_min_samples) < 1:
        parser.error("--target_impact_affine_fill_min_samples must be >= 1")
    if int(args.target_impact_affine_fill_max_samples_per_face) < 0:
        parser.error("--target_impact_affine_fill_max_samples_per_face must be >= 0")
    if int(args.target_impact_affine_fill_max_views) < 0:
        parser.error("--target_impact_affine_fill_max_views must be >= 0")
    if not 0.0 <= float(args.target_impact_affine_fill_blend) <= 1.0:
        parser.error("--target_impact_affine_fill_blend must be in [0, 1]")
    if float(args.target_impact_affine_fill_ridge) <= 0.0:
        parser.error("--target_impact_affine_fill_ridge must be > 0")
    if float(args.target_impact_affine_fill_max_condition) <= 1.0:
        parser.error("--target_impact_affine_fill_max_condition must be > 1")
    if float(args.target_impact_affine_fill_min_norm) < 0.0:
        parser.error("--target_impact_affine_fill_min_norm must be >= 0")
    if int(args.target_impact_affine_fill_synthetic_count) < 0:
        parser.error("--target_impact_affine_fill_synthetic_count must be >= 0")
    if (
        str(args.target_impact_affine_fill_mode) != "off"
        and not bool(args.enable_train_only_target_impact_residual_basis)
    ):
        parser.error("--target_impact_affine_fill_mode requires --enable_train_only_target_impact_residual_basis")
    if int(args.sparse_materialization_target_connected_radius) < 0:
        parser.error("--sparse_materialization_target_connected_radius must be >= 0")
    if int(args.sparse_materialization_target_connected_min_pixels) < 1:
        parser.error("--sparse_materialization_target_connected_min_pixels must be >= 1")
    if int(args.sparse_materialization_target_connected_min_views) < 1:
        parser.error("--sparse_materialization_target_connected_min_views must be >= 1")
    if int(args.sparse_materialization_target_connected_min_policy_samples) < 1:
        parser.error("--sparse_materialization_target_connected_min_policy_samples must be >= 1")
    if not 0.0 <= float(args.sparse_materialization_target_connected_min_positive_view_fraction) <= 1.0:
        parser.error("--sparse_materialization_target_connected_min_positive_view_fraction must be in [0, 1]")
    if float(args.sparse_materialization_target_connected_max_negative_relative_gain) < 0.0:
        parser.error("--sparse_materialization_target_connected_max_negative_relative_gain must be >= 0")
    if float(args.sparse_materialization_target_connected_max_negative_min_view_gain) < 0.0:
        parser.error("--sparse_materialization_target_connected_max_negative_min_view_gain must be >= 0")
    if int(args.sparse_materialization_target_connected_max_extra_bins) < 0:
        parser.error("--sparse_materialization_target_connected_max_extra_bins must be >= 0")
    if float(args.structure_shrink_l1_weight) < 0.0:
        parser.error("--structure_shrink_l1_weight must be >= 0")
    if float(args.structure_shrink_gradient_weight) < 0.0:
        parser.error("--structure_shrink_gradient_weight must be >= 0")
    if float(args.structure_shrink_edge_weight) < 0.0:
        parser.error("--structure_shrink_edge_weight must be >= 0")
    if float(args.structure_shrink_risk_tau) < 0.0:
        parser.error("--structure_shrink_risk_tau must be >= 0")
    if not 0.0 <= float(args.structure_shrink_max_penalty) <= 1.0:
        parser.error("--structure_shrink_max_penalty must be in [0, 1]")
    if float(args.parent_edge_apply_shrink_weight) < 0.0:
        parser.error("--parent_edge_apply_shrink_weight must be >= 0")
    if float(args.parent_edge_apply_shrink_tau) < 0.0:
        parser.error("--parent_edge_apply_shrink_tau must be >= 0")
    if not 0.0 <= float(args.parent_edge_apply_shrink_min_multiplier) <= 1.0:
        parser.error("--parent_edge_apply_shrink_min_multiplier must be in [0, 1]")
    if float(args.view_confidence_kernel_sigma) <= 0.0:
        parser.error("--view_confidence_kernel_sigma must be > 0")
    if not 0.0 <= float(args.view_confidence_min_confidence) <= 1.0:
        parser.error("--view_confidence_min_confidence must be in [0, 1]")
    if float(args.view_alpha_cap_kernel_sigma) <= 0.0:
        parser.error("--view_alpha_cap_kernel_sigma must be > 0")
    if not 0.0 <= float(args.view_alpha_cap_min_confidence) <= 1.0:
        parser.error("--view_alpha_cap_min_confidence must be in [0, 1]")
    if float(args.view_alpha_cap_fallback_alpha) < 0.0:
        parser.error("--view_alpha_cap_fallback_alpha must be >= 0")
    for frontier_fraction_name in (
        "policy_val_alpha_frontier_min_relative_fraction",
        "policy_val_alpha_frontier_min_ssim_fraction",
        "policy_val_alpha_frontier_min_l1_fraction",
        "policy_val_alpha_frontier_knee_min_score_fraction",
        "policy_val_alpha_frontier_knee_slope_drop_fraction",
        "policy_val_alpha_frontier_tail_knee_min_score_fraction",
    ):
        frontier_fraction = float(getattr(args, frontier_fraction_name))
        if not 0.0 <= frontier_fraction <= 1.0:
            parser.error(f"--{frontier_fraction_name} must be in [0, 1]")
    if float(args.policy_val_alpha_frontier_alpha_penalty) < 0.0:
        parser.error("--policy_val_alpha_frontier_alpha_penalty must be >= 0")
    if int(args.policy_val_alpha_frontier_tail_knee_min_regression_count) < 1:
        parser.error("--policy_val_alpha_frontier_tail_knee_min_regression_count must be >= 1")
    if float(args.policy_val_alpha_frontier_tail_knee_eps) < 0.0:
        parser.error("--policy_val_alpha_frontier_tail_knee_eps must be >= 0")
    local_alpha_modes = [
        bool(args.enable_policy_val_local_alpha_calibration),
        bool(args.enable_policy_val_face_alpha_calibration),
        bool(args.enable_policy_val_bin_alpha_calibration),
        bool(args.enable_policy_val_bin_rgb_alpha_calibration),
        bool(args.enable_policy_val_bin_uncertainty_shrink),
        bool(args.enable_policy_val_image_l1_bin_alpha_optimization),
        bool(args.enable_policy_val_image_linear_residual_generator),
    ]
    if sum(int(flag) for flag in local_alpha_modes) > 1:
        parser.error(
            "enable at most one local alpha calibration mode: bucket, face, scalar bin, RGB bin, "
            "uncertainty shrink, image-L1 bin alpha, or image-linear generator"
        )
    if int(args.bin_rgb_alpha_calibration_min_bin_samples) <= 0:
        parser.error("--bin_rgb_alpha_calibration_min_bin_samples must be > 0")
    if not 0.0 <= float(args.bin_rgb_alpha_calibration_min_positive_view_fraction) <= 1.0:
        parser.error("--bin_rgb_alpha_calibration_min_positive_view_fraction must be in [0, 1]")
    if float(args.bin_rgb_alpha_calibration_max_alpha) < float(args.bin_rgb_alpha_calibration_min_alpha):
        parser.error("--bin_rgb_alpha_calibration_max_alpha must be >= --bin_rgb_alpha_calibration_min_alpha")
    if int(args.target_support_prerank_top_k) < 0:
        parser.error("--target_support_prerank_top_k must be >= 0")
    if int(args.target_support_prerank_max_views) < 0:
        parser.error("--target_support_prerank_max_views must be >= 0")
    if int(args.target_footprint_min_bin_pixels) <= 0:
        parser.error("--target_footprint_min_bin_pixels must be > 0")
    if int(args.target_footprint_min_views) <= 0:
        parser.error("--target_footprint_min_views must be > 0")
    if not 0.0 <= float(args.target_footprint_min_view_fraction) <= 1.0:
        parser.error("--target_footprint_min_view_fraction must be in [0, 1]")
    if int(args.target_footprint_max_views) < 0:
        parser.error("--target_footprint_max_views must be >= 0")
    if int(args.coview_transfer_max_faces) < 0:
        parser.error("--coview_transfer_max_faces must be >= 0")
    if int(args.coview_transfer_neighbor_stride) <= 0:
        parser.error("--coview_transfer_neighbor_stride must be > 0")
    if int(args.coview_transfer_min_source_samples) <= 0:
        parser.error("--coview_transfer_min_source_samples must be > 0")
    if float(args.coview_transfer_min_source_mean_l1) < 0.0:
        parser.error("--coview_transfer_min_source_mean_l1 must be >= 0")
    if int(args.coview_transfer_min_edge_count) <= 0:
        parser.error("--coview_transfer_min_edge_count must be > 0")
    if int(args.coview_transfer_min_target_pixels) <= 0:
        parser.error("--coview_transfer_min_target_pixels must be > 0")
    if int(args.coview_transfer_min_policy_val_pixels) <= 0:
        parser.error("--coview_transfer_min_policy_val_pixels must be > 0")
    if int(args.coview_transfer_max_views) < 0:
        parser.error("--coview_transfer_max_views must be >= 0")
    if float(args.coview_transfer_residual_scale) < 0.0:
        parser.error("--coview_transfer_residual_scale must be >= 0")
    if int(args.coview_transfer_synthetic_count) < 0:
        parser.error("--coview_transfer_synthetic_count must be >= 0")
    if not 0.0 <= float(args.target_footprint_tail_risk_min_positive_view_fraction) <= 1.0:
        parser.error("--target_footprint_tail_risk_min_positive_view_fraction must be in [0, 1]")
    if bool(args.enable_target_footprint_tail_risk_certificate) and not bool(
        args.enable_policy_val_prior_bin_gain_hybrid
    ):
        parser.error(
            "--enable_target_footprint_tail_risk_certificate requires "
            "--enable_policy_val_prior_bin_gain_hybrid"
        )
    if (
        bool(args.enable_target_footprint_tail_risk_certificate)
        and not bool(args.target_footprint_tail_risk_all_bins)
        and not bool(args.enable_target_footprint_bin_certificate)
    ):
        parser.error(
            "--enable_target_footprint_tail_risk_certificate requires "
            "--enable_target_footprint_bin_certificate unless --target_footprint_tail_risk_all_bins is set"
        )
    source_model = Path(args.source_model)
    output_model = Path(args.output_model)
    fit_evidence = Path(args.fit_evidence_dir)
    target_evidence = Path(args.target_evidence_dir)
    carrier_json = Path(args.region_carrier_json)
    parent_edge_apply_profile = make_parent_edge_apply_profile(
        enabled=bool(args.enable_parent_edge_apply_shrink),
        weight=float(args.parent_edge_apply_shrink_weight),
        edge_tau=float(args.parent_edge_apply_shrink_tau),
        min_multiplier=float(args.parent_edge_apply_shrink_min_multiplier),
    )

    copy_model_shell(source_model, output_model, force=bool(args.force))
    candidate_faces, carrier_summary = load_carrier_faces(
        carrier_json,
        max_carriers=int(args.max_carriers),
        max_faces_per_carrier=int(args.max_faces_per_carrier),
        max_faces=int(args.max_faces),
    )
    fit_views_all = evidence_views(fit_evidence)
    if not fit_views_all:
        raise FileNotFoundError(f"no fit npz views found in {fit_evidence}")
    base_candidate_faces = set(candidate_faces)
    support_expansion_summary: dict[str, Any] = {
        "enabled": False,
        "mode": str(args.support_expansion_mode),
        "base_faces": int(len(base_candidate_faces)),
        "added_faces": 0,
        "candidate_faces_after_expansion": int(len(base_candidate_faces)),
    }
    support_candidate_sets: list[dict[str, Any]] = [
        {
            "support_mode": "base_carrier",
            "faces": set(base_candidate_faces),
            "summary": {
                "enabled": False,
                "mode": "base_carrier",
                "base_faces": int(len(base_candidate_faces)),
                "added_faces": 0,
                "candidate_faces_after_expansion": int(len(base_candidate_faces)),
            },
        }
    ]
    if str(args.support_expansion_mode) in ("fit_residual_topk", "target_footprint_residual_debt"):
        try:
            support_extra_candidates = parse_int_candidates(
                str(args.support_expansion_max_extra_faces_candidates),
                int(args.support_expansion_max_extra_faces),
            )
        except ValueError as exc:
            parser.error(str(exc))
        if str(args.support_expansion_mode) == "target_footprint_residual_debt":
            target_views_for_support = evidence_views(target_evidence)
            if not target_views_for_support:
                raise FileNotFoundError(f"no target npz views found in {target_evidence}")
            probe_texture_size = int(
                parse_int_candidates(str(args.texture_size_candidates), int(args.texture_size))[0]
            )
            ranked_extra_faces, rank_summary = rank_target_footprint_residual_debt_faces(
                fit_views_all,
                target_views_for_support,
                base_faces=set(base_candidate_faces),
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                policy_val_stride=int(args.policy_val_stride),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                min_face_samples=int(args.support_expansion_min_face_samples),
                min_mean_l1=float(args.support_expansion_min_mean_l1),
                max_samples_per_view=int(args.max_samples_per_view),
                texture_size=int(probe_texture_size),
                target_footprint_max_views=int(args.target_footprint_max_views),
                target_footprint_match_level=str(args.target_footprint_residual_debt_match_level),
            )
        else:
            ranked_extra_faces, rank_summary = rank_fit_residual_extra_faces(
                fit_views_all,
                base_faces=set(base_candidate_faces),
                residual_l1_key=str(args.residual_l1_key),
                policy_val_stride=int(args.policy_val_stride),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                min_face_samples=int(args.support_expansion_min_face_samples),
                min_mean_l1=float(args.support_expansion_min_mean_l1),
                max_samples_per_view=int(args.max_samples_per_view),
            )
        expansion_candidates: list[dict[str, Any]] = []
        for max_extra_faces in support_extra_candidates:
            expanded_candidate_faces, candidate_summary = expanded_candidate_faces_from_ranked_rows(
                base_faces=set(base_candidate_faces),
                ranked_rows=ranked_extra_faces,
                rank_summary=rank_summary,
                max_extra_faces=int(max_extra_faces),
            )
            support_mode = (
                str(args.support_expansion_mode)
                if len(support_extra_candidates) == 1
                else f"{str(args.support_expansion_mode)}_{int(max_extra_faces)}"
            )
            candidate_summary["support_mode"] = str(support_mode)
            expansion_candidates.append(dict(candidate_summary))
            support_candidate_sets.append(
                {
                    "support_mode": str(support_mode),
                    "faces": set(expanded_candidate_faces),
                    "summary": dict(candidate_summary),
                }
            )
        if len(expansion_candidates) == 1:
            support_expansion_summary = dict(expansion_candidates[0])
        else:
            support_expansion_summary = {
                "enabled": True,
                "mode": f"{str(args.support_expansion_mode)}_ladder",
                "base_faces": int(len(base_candidate_faces)),
                "eligible_extra_faces": int(rank_summary.get("eligible_extra_faces", len(ranked_extra_faces))),
                "train_debt_bins": int(rank_summary.get("train_debt_bins", 0)),
                "train_debt_faces": int(rank_summary.get("train_debt_faces", 0)),
                "fit_view_count": int(rank_summary.get("fit_view_count", 0)),
                "skipped_policy_val_views": int(rank_summary.get("skipped_policy_val_views", 0)),
                "target_footprint": dict(rank_summary.get("target_footprint", {})),
                "min_face_samples": int(rank_summary.get("min_face_samples", 0)),
                "min_mean_l1": float(rank_summary.get("min_mean_l1", 0.0)),
                "max_extra_faces_candidates": [int(x) for x in support_extra_candidates],
                "candidates": expansion_candidates,
                "rank_preview": list(rank_summary.get("rank_preview", [])),
            }
    carrier_summary["support_expansion"] = support_expansion_summary
    coview_transfer_summary: dict[str, Any] = {
        "enabled": bool(args.enable_coview_face_residual_transfer),
        "mode": "coview_face_residual_transfer",
        "selected_transfer_faces": 0,
    }
    if bool(args.enable_coview_face_residual_transfer):
        target_views_for_transfer = evidence_views(target_evidence)
        if not target_views_for_transfer:
            raise FileNotFoundError(f"no target npz views found in {target_evidence}")
        transfer_rows, coview_transfer_summary = build_coview_face_residual_transfer_plan(
            fit_views_all,
            target_views_for_transfer,
            base_faces=set(base_candidate_faces),
            residual_rgb_key=str(args.residual_rgb_key),
            residual_l1_key=str(args.residual_l1_key),
            policy_val_stride=int(args.policy_val_stride),
            min_l1=float(args.min_l1),
            min_alpha=float(args.min_alpha),
            max_samples_per_view=int(args.max_samples_per_view),
            neighbor_stride=int(args.coview_transfer_neighbor_stride),
            min_source_samples=int(args.coview_transfer_min_source_samples),
            min_source_mean_l1=float(args.coview_transfer_min_source_mean_l1),
            min_edge_count=int(args.coview_transfer_min_edge_count),
            min_target_pixels=int(args.coview_transfer_min_target_pixels),
            min_policy_val_pixels=int(args.coview_transfer_min_policy_val_pixels),
            max_faces=int(args.coview_transfer_max_faces),
            max_views=int(args.coview_transfer_max_views),
        )
        transfer_faces = {int(row["face_id"]) for row in transfer_rows}
        coview_transfer_summary = dict(coview_transfer_summary)
        coview_transfer_summary["transfer_rows"] = list(transfer_rows)
        coview_transfer_summary["residual_scale"] = float(args.coview_transfer_residual_scale)
        coview_transfer_summary["synthetic_count"] = int(args.coview_transfer_synthetic_count)
        coview_transfer_summary["blend_max_direct_bin_count"] = int(args.coview_transfer_blend_max_direct_bin_count)
        coview_existing_atlas_mode = str(args.coview_transfer_existing_atlas_mode)
        if bool(args.coview_transfer_overwrite_existing_atlas):
            coview_existing_atlas_mode = "overwrite"
        coview_transfer_summary["existing_atlas_mode"] = str(coview_existing_atlas_mode)
        coview_transfer_summary["overwrite_existing_atlas"] = bool(coview_existing_atlas_mode == "overwrite")
        carrier_summary["coview_face_residual_transfer"] = dict(coview_transfer_summary)
        if transfer_rows:
            expanded_faces = set(base_candidate_faces)
            expanded_faces.update(transfer_faces)
            support_candidate_sets.append(
                {
                    "support_mode": "coview_face_residual_transfer",
                    "faces": expanded_faces,
                    "summary": {
                        "enabled": True,
                        "mode": "coview_face_residual_transfer",
                        "base_faces": int(len(base_candidate_faces)),
                        "added_faces": int(len(expanded_faces - set(base_candidate_faces))),
                        "candidate_faces_after_expansion": int(len(expanded_faces)),
                        "transfer_row_count": int(len(transfer_rows)),
                        "coview_transfer": dict(coview_transfer_summary),
                    },
                }
            )
    else:
        carrier_summary["coview_face_residual_transfer"] = dict(coview_transfer_summary)
    target_support_prerank_summary: dict[str, Any] = {
        "enabled": False,
        "mode": "target_support_prerank_face_proxy",
        "requested_top_k": int(args.target_support_prerank_top_k),
        "max_views": int(args.target_support_prerank_max_views),
    }
    if int(args.target_support_prerank_top_k) > 0 and len(support_candidate_sets) > 1:
        prerank_views = evidence_views(target_evidence)
        if not prerank_views:
            raise FileNotFoundError(f"no target npz views found in {target_evidence}")

        def support_prerank_score(candidate: dict[str, Any]) -> tuple[float, float, float, float, float]:
            profile = dict(candidate.get("target_support_prerank_profile") or {})
            return (
                float(profile.get("coverage_fraction", 0.0)),
                float(profile.get("cvar20_view_coverage_fraction", 0.0)),
                float(profile.get("min_view_coverage_fraction", 0.0)),
                float(profile.get("covered_valid_fraction", 0.0)),
                float((candidate.get("summary") or {}).get("added_faces", 0)),
            )

        profiled_support_candidates: list[dict[str, Any]] = []
        for idx, support_candidate in enumerate(support_candidate_sets):
            candidate = dict(support_candidate)
            candidate["original_index"] = int(idx)
            profile = evaluate_target_face_support_proxy(
                prerank_views,
                set(candidate.get("faces") or set()),
                min_alpha=float(args.min_alpha),
                max_views=int(args.target_support_prerank_max_views),
            )
            candidate["target_support_prerank_profile"] = dict(profile)
            candidate_summary = dict(candidate.get("summary") or {})
            candidate_summary["target_support_prerank_profile"] = dict(profile)
            candidate["summary"] = candidate_summary
            profiled_support_candidates.append(candidate)
        ranked_support_candidates = sorted(
            profiled_support_candidates,
            key=support_prerank_score,
            reverse=True,
        )
        keep_count = min(int(args.target_support_prerank_top_k), len(ranked_support_candidates))
        retained_support_candidates = ranked_support_candidates[:keep_count]
        support_candidate_sets = [
            {
                "support_mode": str(candidate.get("support_mode", "")),
                "faces": set(candidate.get("faces") or set()),
                "summary": dict(candidate.get("summary") or {}),
            }
            for candidate in retained_support_candidates
        ]
        target_support_prerank_summary = {
            "enabled": True,
            "mode": "target_support_prerank_face_proxy",
            "requested_top_k": int(args.target_support_prerank_top_k),
            "max_views": int(args.target_support_prerank_max_views),
            "input_support_candidate_count": int(len(profiled_support_candidates)),
            "retained_support_candidate_count": int(len(support_candidate_sets)),
            "retained_support_modes": [
                str(candidate.get("support_mode", "")) for candidate in retained_support_candidates
            ],
            "score_order": [
                {
                    "support_mode": str(candidate.get("support_mode", "")),
                    "support_added_faces": int((candidate.get("summary") or {}).get("added_faces", 0)),
                    "original_index": int(candidate.get("original_index", -1)),
                    "coverage_fraction": float(
                        (candidate.get("target_support_prerank_profile") or {}).get("coverage_fraction", 0.0)
                    ),
                    "cvar20_view_coverage_fraction": float(
                        (candidate.get("target_support_prerank_profile") or {}).get(
                            "cvar20_view_coverage_fraction", 0.0
                        )
                    ),
                    "min_view_coverage_fraction": float(
                        (candidate.get("target_support_prerank_profile") or {}).get(
                            "min_view_coverage_fraction", 0.0
                        )
                    ),
                    "covered_valid_fraction": float(
                        (candidate.get("target_support_prerank_profile") or {}).get(
                            "covered_valid_fraction", 0.0
                        )
                    ),
                }
                for candidate in ranked_support_candidates
            ],
        }
    elif int(args.target_support_prerank_top_k) > 0:
        target_support_prerank_summary = {
            "enabled": False,
            "mode": "target_support_prerank_face_proxy",
            "reason": "single_support_candidate",
            "requested_top_k": int(args.target_support_prerank_top_k),
            "max_views": int(args.target_support_prerank_max_views),
            "input_support_candidate_count": int(len(support_candidate_sets)),
            "retained_support_candidate_count": int(len(support_candidate_sets)),
        }
    carrier_summary["target_support_prerank"] = target_support_prerank_summary

    target_support_candidate_selection_enabled = bool(args.enable_target_support_candidate_selection)
    policy_candidate_control: dict[str, Any] = {
        "enabled": bool(
            args.enable_policy_candidate_dominance_pruning
            or str(args.policy_candidate_early_stop_mode) != "none"
            or bool(args.enable_adaptive_texture_size_ladder)
        ),
        "dominance_pruning_requested": bool(args.enable_policy_candidate_dominance_pruning),
        "dominance_pruning_enabled": bool(args.enable_policy_candidate_dominance_pruning),
        "dominance_pruning_mode": "strict_equivalent_support_and_policy_axes",
        "early_stop_requested_mode": str(args.policy_candidate_early_stop_mode),
        "early_stop_effective_mode": "none",
        "early_stop_disabled_reasons": [],
        "early_stop_triggered": False,
        "support_candidate_count_before_pruning": int(len(support_candidate_sets)),
        "support_candidate_count_after_pruning": int(len(support_candidate_sets)),
        "support_duplicate_pruned_count": 0,
        "support_duplicate_pruned": [],
        "planned_candidate_count_before_pruning": 0,
        "planned_candidate_count_after_pruning": 0,
        "spec_duplicate_pruned_count": 0,
        "spec_duplicate_pruned": [],
        "executed_candidate_count": 0,
        "early_stop_skipped_count": 0,
        "early_stop_skipped": [],
    }
    if bool(args.enable_policy_candidate_dominance_pruning):
        seen_support: dict[str, dict[str, Any]] = {}
        retained_support_candidate_sets: list[dict[str, Any]] = []
        pruned_support_rows: list[dict[str, Any]] = []
        for original_index, support_candidate in enumerate(support_candidate_sets, start=1):
            faces = set(int(face) for face in support_candidate.get("faces", set()))
            support_digest = face_set_sha1(faces)
            support_summary = dict(support_candidate.get("summary") or {})
            support_summary["support_faces_sha1"] = str(support_digest)
            support_summary["support_candidate_faces"] = int(len(faces))
            normalized_candidate = {
                "support_mode": str(support_candidate.get("support_mode", "")),
                "faces": faces,
                "summary": support_summary,
            }
            if support_digest in seen_support:
                dominated_by = dict(seen_support[support_digest])
                pruned_support_rows.append(
                    {
                        "reason": "duplicate_support_faces",
                        "original_index": int(original_index),
                        "support_mode": str(normalized_candidate["support_mode"]),
                        "support_candidate_faces": int(len(faces)),
                        "support_faces_sha1": str(support_digest),
                        "dominated_by_original_index": int(dominated_by.get("original_index", 0)),
                        "dominated_by_support_mode": str(dominated_by.get("support_mode", "")),
                    }
                )
                continue
            seen_support[support_digest] = {
                "original_index": int(original_index),
                "support_mode": str(normalized_candidate["support_mode"]),
            }
            retained_support_candidate_sets.append(normalized_candidate)
        support_candidate_sets = retained_support_candidate_sets
        policy_candidate_control["support_candidate_count_after_pruning"] = int(len(support_candidate_sets))
        policy_candidate_control["support_duplicate_pruned_count"] = int(len(pruned_support_rows))
        policy_candidate_control["support_duplicate_pruned"] = pruned_support_rows[:64]
        if pruned_support_rows:
            print(
                "[policy-candidate] dominance-pruned "
                f"{len(pruned_support_rows)} duplicate support candidate(s)",
                flush=True,
            )

    try:
        texture_size_candidates = parse_int_candidates(str(args.texture_size_candidates), int(args.texture_size))
    except ValueError as exc:
        parser.error(str(exc))
    adaptive_texture_size_ladder_summary: dict[str, Any] = {
        "enabled": False,
        "mode": "train_fit_residual_density_ladder",
        "base_texture_size_candidates": [int(x) for x in texture_size_candidates],
        "planned_texture_size_candidates": [int(x) for x in texture_size_candidates],
        "reason": "not_requested",
    }
    if bool(args.enable_adaptive_texture_size_ladder):
        support_union_faces: set[int] = set()
        for support_candidate in support_candidate_sets:
            support_union_faces.update(int(face) for face in support_candidate.get("faces", set()))
        support_modes = [
            str(support_candidate.get("support_mode", ""))
            for support_candidate in support_candidate_sets
        ]
        texture_size_candidates, adaptive_texture_size_ladder_summary = plan_adaptive_texture_size_ladder(
            fit_views_all,
            support_union_faces,
            texture_size_candidates,
            residual_l1_key=str(args.residual_l1_key),
            policy_val_stride=int(args.policy_val_stride),
            min_l1=float(args.min_l1),
            min_alpha=float(args.min_alpha),
            max_samples_per_view=int(args.max_samples_per_view),
            max_size=int(args.adaptive_texture_size_ladder_max_size),
            min_fit_samples_per_face=float(args.adaptive_texture_size_ladder_min_fit_samples_per_face),
            min_samples_per_current_bin=float(args.adaptive_texture_size_ladder_min_samples_per_current_bin),
            min_mean_l1=float(args.adaptive_texture_size_ladder_min_mean_l1),
            support_modes=support_modes,
        )
    policy_candidate_control["adaptive_texture_size_ladder"] = dict(adaptive_texture_size_ladder_summary)

    def build_policy_candidate(
        fill_mode: str,
        texture_size: int,
        teacher_low_rank_texture_rank: int,
        surface_multiscale_prior_blend: float,
        max_abs_delta_rgb_candidate: float,
        support_mode: str,
        support_faces: set[int],
        support_summary: dict[str, Any],
        candidate_index: int,
        candidate_count: int,
    ) -> dict[str, Any]:
        candidate_label = (
            f"{candidate_index}/{candidate_count} "
            f"support={support_mode} "
            f"added={int(support_summary.get('added_faces', 0))} "
            f"faces={len(support_faces)} "
            f"texture={int(texture_size)} "
            f"rank={int(teacher_low_rank_texture_rank)} "
            f"fill={fill_mode} "
            f"prior_blend={float(surface_multiscale_prior_blend):.6g} "
            f"cap={float(max_abs_delta_rgb_candidate):.6g}"
        )
        print(f"[policy-candidate] start {candidate_label}", flush=True)
        cand_atlas, cand_fit_summary, cand_fit_views, cand_val_views = fit_atlas(
            fit_views_all,
            candidate_faces=support_faces,
            residual_rgb_key=str(args.residual_rgb_key),
            residual_l1_key=str(args.residual_l1_key),
            texture_size=int(texture_size),
            policy_val_stride=int(args.policy_val_stride),
            min_l1=float(args.min_l1),
            min_alpha=float(args.min_alpha),
            max_samples_per_view=int(args.max_samples_per_view),
            fill_empty_with_face_mean=bool(args.fill_empty_with_face_mean),
            atlas_empty_bin_fill_mode=str(fill_mode),
            atlas_nearest_fill_max_steps=int(args.atlas_nearest_fill_max_steps),
            atlas_nearest_fill_decay=float(args.atlas_nearest_fill_decay),
            atlas_lowpass_passes=int(args.atlas_lowpass_passes),
            atlas_lowpass_neighbor_min_count=int(args.atlas_lowpass_neighbor_min_count),
            surface_multiscale_prior_mode=str(args.surface_multiscale_prior_mode),
            surface_multiscale_prior_block_sizes=list(surface_multiscale_prior_block_sizes),
            surface_multiscale_prior_min_bin_samples=int(args.surface_multiscale_prior_min_bin_samples),
            surface_multiscale_prior_count_tau=float(args.surface_multiscale_prior_count_tau),
            surface_multiscale_prior_blend=float(surface_multiscale_prior_blend),
            surface_multiscale_prior_gate_mode=str(args.surface_multiscale_prior_gate_mode),
            surface_multiscale_prior_min_prior_weight=float(args.surface_multiscale_prior_min_prior_weight),
            surface_multiscale_prior_min_direct_samples=int(args.surface_multiscale_prior_min_direct_samples),
            surface_multiscale_prior_min_sign_consistency=float(args.surface_multiscale_prior_min_sign_consistency),
            surface_multiscale_prior_max_mean_variance=float(args.surface_multiscale_prior_max_mean_variance),
            surface_multiscale_prior_min_cosine=float(args.surface_multiscale_prior_min_cosine),
            view_conditioned_basis_mode=str(args.view_conditioned_basis_mode),
            view_conditioned_basis_min_bin_samples=int(args.view_conditioned_basis_min_bin_samples),
            view_conditioned_basis_ridge=float(args.view_conditioned_basis_ridge),
            view_conditioned_basis_ood_mode=str(args.view_conditioned_basis_ood_mode),
            view_conditioned_basis_ood_max_z=float(args.view_conditioned_basis_ood_max_z),
            view_conditioned_basis_ood_min_std=float(args.view_conditioned_basis_ood_min_std),
            view_cluster_expert_count=int(args.view_cluster_expert_count),
            view_cluster_feature_mode=str(args.view_cluster_feature_mode),
            view_cluster_min_views=int(args.view_cluster_min_views),
            view_cluster_min_bin_samples=int(args.view_cluster_min_bin_samples),
            view_cluster_fallback_mode=str(args.view_cluster_fallback_mode),
            teacher_residual_target_mode=str(args.teacher_residual_target_mode),
            teacher_residual_target_luma_mix=float(args.teacher_residual_target_luma_mix),
            teacher_residual_target_edge_boost=float(args.teacher_residual_target_edge_boost),
            teacher_distilled_basis_mode=str(args.teacher_distilled_basis_mode),
            teacher_distilled_basis_min_face_samples=int(args.teacher_distilled_basis_min_face_samples),
            teacher_distilled_basis_ridge=float(args.teacher_distilled_basis_ridge),
            teacher_distilled_basis_ood_max_z=float(args.teacher_distilled_basis_ood_max_z),
            teacher_distilled_basis_ood_min_std=float(args.teacher_distilled_basis_ood_min_std),
            teacher_distilled_basis_apply_mode=str(args.teacher_distilled_basis_apply_mode),
            teacher_distilled_basis_blend=float(args.teacher_distilled_basis_blend),
            teacher_distilled_low_rank_texture_rank=int(teacher_low_rank_texture_rank),
            enable_adaptive_low_support_teacher_basis=bool(args.enable_adaptive_low_support_teacher_basis),
            adaptive_teacher_basis_min_face_samples_floor=int(args.adaptive_teacher_basis_min_face_samples_floor),
            adaptive_teacher_basis_support_quantile=float(args.adaptive_teacher_basis_support_quantile),
            adaptive_teacher_basis_low_support_ridge_scale=float(args.adaptive_teacher_basis_low_support_ridge_scale),
        )
        cand_fit_summary["support_mode"] = str(support_mode)
        cand_fit_summary["candidate_index"] = int(candidate_index)
        cand_fit_summary["candidate_count"] = int(candidate_count)
        cand_fit_summary["candidate_label"] = str(candidate_label)
        cand_fit_summary["teacher_distilled_low_rank_texture_rank_candidate"] = int(teacher_low_rank_texture_rank)
        cand_fit_summary["teacher_distilled_low_rank_texture_rank_candidates"] = [
            int(x) for x in teacher_low_rank_texture_rank_candidates
        ]
        cand_fit_summary["support_base_faces"] = int(support_summary.get("base_faces", len(base_candidate_faces)))
        cand_fit_summary["support_added_faces"] = int(support_summary.get("added_faces", 0))
        cand_fit_summary["support_candidate_faces"] = int(len(support_faces))
        cand_fit_summary["surface_multiscale_prior_blend_candidate"] = float(surface_multiscale_prior_blend)
        cand_fit_summary["surface_multiscale_prior_blend_candidates"] = [
            float(x) for x in surface_multiscale_prior_blend_candidates
        ]
        cand_fit_summary["max_abs_delta_rgb_candidate"] = float(max_abs_delta_rgb_candidate)
        cand_fit_summary["max_abs_delta_rgb_candidates"] = [
            float(x) for x in max_abs_delta_rgb_candidates
        ]
        coview_transfer_payload = dict(support_summary.get("coview_transfer") or {})
        if bool(coview_transfer_payload.get("enabled", False)):
            transfer_rows = list(coview_transfer_payload.get("transfer_rows", []))
            coview_application = apply_coview_face_residual_transfer(
                cand_atlas,
                transfer_rows,
                texture_size=int(texture_size),
                residual_scale=float(args.coview_transfer_residual_scale),
                max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                synthetic_count=int(args.coview_transfer_synthetic_count),
                existing_atlas_mode=str(coview_transfer_payload.get("existing_atlas_mode", "skip")),
                blend_max_direct_bin_count=int(coview_transfer_payload.get("blend_max_direct_bin_count", -1)),
            )
            cand_fit_summary["coview_face_residual_transfer"] = {
                "plan": coview_transfer_payload,
                "application": coview_application,
            }
        else:
            cand_fit_summary["coview_face_residual_transfer"] = {
                "enabled": False,
                "mode": "coview_face_residual_transfer",
                "reason": "not_requested_for_candidate",
            }
        alpha_candidates = parse_alpha_grid(args.alpha_grid)
        alpha_calibration_summary = {"enabled": False, "alpha_grid": list(alpha_candidates)}
        local_alpha_profile: dict[str, Any] = {"enabled": False, "alpha_grid": list(alpha_candidates)}
        if bool(args.enable_policy_val_alpha_calibration):
            try:
                alpha_multipliers = [float(x) for x in parse_alpha_grid(args.alpha_calibration_multipliers)]
            except ValueError as exc:
                parser.error(f"invalid --alpha_calibration_multipliers: {exc}")
            alpha_candidates, alpha_calibration_summary = calibrated_alpha_grid_from_policy_val(
                cand_val_views,
                cand_atlas,
                base_alpha_grid=alpha_candidates,
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                max_samples_per_view=int(args.max_samples_per_view),
                min_atlas_bin_count=int(args.min_atlas_bin_count),
                min_atlas_face_samples=int(args.min_atlas_face_samples),
                max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                atlas_confidence_mode=str(args.atlas_confidence_mode),
                atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                min_atlas_confidence=float(args.min_atlas_confidence),
                calibration_max_alpha=float(args.alpha_calibration_max_alpha),
                calibration_multipliers=alpha_multipliers,
                min_denominator=float(args.alpha_calibration_min_denominator),
            )
        cand_fit_summary["alpha_calibration"] = dict(alpha_calibration_summary)
        if bool(args.enable_policy_val_local_alpha_calibration):
            try:
                local_quantiles = parse_float_candidates(str(args.local_alpha_calibration_bucket_quantiles))
                local_edges = parse_float_candidates(str(args.local_alpha_calibration_bucket_edges))
                local_multipliers = parse_float_candidates(str(args.local_alpha_calibration_multipliers))
            except ValueError as exc:
                parser.error(f"invalid local alpha calibration setting: {exc}")
            alpha_candidates, local_alpha_profile = calibrated_local_alpha_profile_from_policy_val(
                cand_val_views,
                cand_atlas,
                base_alpha_grid=alpha_candidates,
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                max_samples_per_view=int(args.max_samples_per_view),
                min_atlas_bin_count=int(args.min_atlas_bin_count),
                min_atlas_face_samples=int(args.min_atlas_face_samples),
                max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                atlas_confidence_mode=str(args.atlas_confidence_mode),
                atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                min_atlas_confidence=float(args.min_atlas_confidence),
                max_alpha=float(args.local_alpha_calibration_max_alpha),
                min_alpha_value=float(args.local_alpha_calibration_min_alpha),
                bucket_quantiles=local_quantiles,
                bucket_edges=local_edges,
                multiplier_grid=local_multipliers,
                min_bucket_samples=int(args.local_alpha_calibration_min_bucket_samples),
                norm_mode=str(args.local_alpha_calibration_norm_mode),
                min_denominator=float(args.local_alpha_calibration_min_denominator),
            )
        if bool(args.enable_policy_val_face_alpha_calibration):
            try:
                face_multipliers = parse_float_candidates(str(args.face_alpha_calibration_multipliers))
            except ValueError as exc:
                parser.error(f"invalid face alpha calibration setting: {exc}")
            alpha_candidates, local_alpha_profile = calibrated_face_alpha_profile_from_policy_val(
                cand_val_views,
                cand_atlas,
                base_alpha_grid=alpha_candidates,
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                max_samples_per_view=int(args.max_samples_per_view),
                min_atlas_bin_count=int(args.min_atlas_bin_count),
                min_atlas_face_samples=int(args.min_atlas_face_samples),
                max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                atlas_confidence_mode=str(args.atlas_confidence_mode),
                atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                min_atlas_confidence=float(args.min_atlas_confidence),
                max_alpha=float(args.face_alpha_calibration_max_alpha),
                min_alpha_value=float(args.face_alpha_calibration_min_alpha),
                multiplier_grid=face_multipliers,
                min_face_samples=int(args.face_alpha_calibration_min_face_samples),
                min_denominator=float(args.face_alpha_calibration_min_denominator),
                shrink_count_tau=float(args.face_alpha_calibration_shrink_count_tau),
                shrink_denominator_tau=float(args.face_alpha_calibration_shrink_denominator_tau),
                shrink_prior=str(args.face_alpha_calibration_shrink_prior),
            )
        if bool(args.enable_policy_val_bin_alpha_calibration):
            try:
                bin_multipliers = parse_float_candidates(str(args.bin_alpha_calibration_multipliers))
            except ValueError as exc:
                parser.error(f"invalid bin alpha calibration setting: {exc}")
            alpha_candidates, local_alpha_profile = calibrated_bin_alpha_profile_from_policy_val(
                cand_val_views,
                cand_atlas,
                base_alpha_grid=alpha_candidates,
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                max_samples_per_view=int(args.max_samples_per_view),
                min_atlas_bin_count=int(args.min_atlas_bin_count),
                min_atlas_face_samples=int(args.min_atlas_face_samples),
                max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                atlas_confidence_mode=str(args.atlas_confidence_mode),
                atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                min_atlas_confidence=float(args.min_atlas_confidence),
                max_alpha=float(args.bin_alpha_calibration_max_alpha),
                min_alpha_value=float(args.bin_alpha_calibration_min_alpha),
                multiplier_grid=bin_multipliers,
                min_bin_samples=int(args.bin_alpha_calibration_min_bin_samples),
                min_denominator=float(args.bin_alpha_calibration_min_denominator),
                min_positive_view_fraction=float(args.bin_alpha_calibration_min_positive_view_fraction),
                shrink_count_tau=float(args.bin_alpha_calibration_shrink_count_tau),
                shrink_denominator_tau=float(args.bin_alpha_calibration_shrink_denominator_tau),
                shrink_prior=str(args.bin_alpha_calibration_shrink_prior),
                max_profile_bins=int(args.bin_alpha_calibration_max_profile_bins),
            )
        if bool(args.enable_policy_val_bin_rgb_alpha_calibration):
            try:
                bin_rgb_multipliers = parse_float_candidates(str(args.bin_rgb_alpha_calibration_multipliers))
            except ValueError as exc:
                parser.error(f"invalid bin RGB alpha calibration setting: {exc}")
            alpha_candidates, local_alpha_profile = calibrated_bin_rgb_alpha_profile_from_policy_val(
                cand_val_views,
                cand_atlas,
                base_alpha_grid=alpha_candidates,
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                max_samples_per_view=int(args.max_samples_per_view),
                min_atlas_bin_count=int(args.min_atlas_bin_count),
                min_atlas_face_samples=int(args.min_atlas_face_samples),
                max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                atlas_confidence_mode=str(args.atlas_confidence_mode),
                atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                min_atlas_confidence=float(args.min_atlas_confidence),
                max_alpha=float(args.bin_rgb_alpha_calibration_max_alpha),
                min_alpha_value=float(args.bin_rgb_alpha_calibration_min_alpha),
                multiplier_grid=bin_rgb_multipliers,
                min_bin_samples=int(args.bin_rgb_alpha_calibration_min_bin_samples),
                min_denominator=float(args.bin_rgb_alpha_calibration_min_denominator),
                min_positive_view_fraction=float(args.bin_rgb_alpha_calibration_min_positive_view_fraction),
                shrink_count_tau=float(args.bin_rgb_alpha_calibration_shrink_count_tau),
                shrink_denominator_tau=float(args.bin_rgb_alpha_calibration_shrink_denominator_tau),
                shrink_prior=str(args.bin_rgb_alpha_calibration_shrink_prior),
                max_profile_bins=int(args.bin_rgb_alpha_calibration_max_profile_bins),
            )
        if bool(args.enable_policy_val_bin_uncertainty_shrink):
            alpha_candidates, local_alpha_profile = calibrated_bin_uncertainty_shrink_profile_from_policy_val(
                cand_val_views,
                cand_atlas,
                base_alpha_grid=alpha_candidates,
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                max_samples_per_view=int(args.max_samples_per_view),
                min_atlas_bin_count=int(args.min_atlas_bin_count),
                min_atlas_face_samples=int(args.min_atlas_face_samples),
                max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                atlas_confidence_mode=str(args.atlas_confidence_mode),
                atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                min_atlas_confidence=float(args.min_atlas_confidence),
                min_bin_samples=int(args.bin_uncertainty_shrink_min_bin_samples),
                min_bin_views=int(args.bin_uncertainty_shrink_min_bin_views),
                min_relative_gain=float(args.bin_uncertainty_shrink_min_relative_gain),
                min_positive_view_fraction=float(args.bin_uncertainty_shrink_min_positive_view_fraction),
                max_mean_variance=float(args.bin_uncertainty_shrink_max_mean_variance),
                min_mean_sign_consistency=float(args.bin_uncertainty_shrink_min_mean_sign_consistency),
                count_tau=float(args.bin_uncertainty_shrink_count_tau),
                gain_tau=float(args.bin_uncertainty_shrink_gain_tau),
                variance_scale=float(args.bin_uncertainty_shrink_variance_scale),
                sign_power=float(args.bin_uncertainty_shrink_sign_power),
                min_shrink=float(args.bin_uncertainty_shrink_min_shrink),
                max_shrink=float(args.bin_uncertainty_shrink_max_shrink),
                fallback_shrink=float(args.bin_uncertainty_shrink_fallback_shrink),
                policy_mode=str(args.bin_uncertainty_shrink_policy_mode),
                max_profile_bins=int(args.bin_uncertainty_shrink_max_profile_bins),
                enable_structure_aware_shrink=bool(args.enable_policy_val_structure_aware_shrink),
                structure_shrink_l1_weight=float(args.structure_shrink_l1_weight),
                structure_shrink_gradient_weight=float(args.structure_shrink_gradient_weight),
                structure_shrink_edge_weight=float(args.structure_shrink_edge_weight),
                structure_shrink_risk_tau=float(args.structure_shrink_risk_tau),
                structure_shrink_max_penalty=float(args.structure_shrink_max_penalty),
                enable_view_cluster_local_shrink=bool(args.enable_view_cluster_local_shrink),
                view_cluster_local_global_fallback=bool(args.view_cluster_local_shrink_global_fallback),
                enable_image_l1_bin_certificate=bool(args.enable_policy_val_image_l1_bin_certificate),
                image_l1_certificate_mode=str(args.image_l1_bin_certificate_mode),
                image_l1_certificate_min_relative_gain=float(args.image_l1_bin_certificate_min_relative_gain),
                image_l1_certificate_min_positive_view_fraction=float(
                    args.image_l1_bin_certificate_min_positive_view_fraction
                ),
                image_l1_certificate_gain_tau=float(args.image_l1_bin_certificate_gain_tau),
                image_l1_certificate_max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                image_l1_certificate_pool_radius=int(args.image_l1_bin_certificate_pool_radius),
                enable_image_l1_region_expansion=bool(args.enable_policy_val_image_l1_region_expansion),
                image_l1_region_expansion_radius=int(args.image_l1_region_expansion_radius),
                image_l1_region_expansion_max_bins_per_seed=int(args.image_l1_region_expansion_max_bins_per_seed),
                image_l1_region_expansion_min_neighbor_samples=int(
                    args.image_l1_region_expansion_min_neighbor_samples
                ),
                image_l1_region_expansion_min_neighbor_views=int(args.image_l1_region_expansion_min_neighbor_views),
                image_l1_region_expansion_max_negative_relative_gain=float(
                    args.image_l1_region_expansion_max_negative_relative_gain
                ),
                image_l1_region_expansion_max_negative_image_l1_gain=float(
                    args.image_l1_region_expansion_max_negative_image_l1_gain
                ),
                image_l1_region_expansion_shrink_decay=float(args.image_l1_region_expansion_shrink_decay),
            )
        if bool(args.enable_policy_val_image_l1_bin_alpha_optimization):
            try:
                image_l1_bin_alpha_grid = parse_float_candidates(str(args.image_l1_bin_alpha_grid))
            except ValueError as exc:
                parser.error(f"invalid image-L1 bin alpha grid: {exc}")
            alpha_candidates, local_alpha_profile = calibrated_image_l1_bin_alpha_profile_from_policy_val(
                cand_val_views,
                cand_atlas,
                base_alpha_grid=alpha_candidates,
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                max_samples_per_view=int(args.max_samples_per_view),
                min_atlas_bin_count=int(args.min_atlas_bin_count),
                min_atlas_face_samples=int(args.min_atlas_face_samples),
                max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                atlas_confidence_mode=str(args.atlas_confidence_mode),
                atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                min_atlas_confidence=float(args.min_atlas_confidence),
                local_alpha_grid=image_l1_bin_alpha_grid,
                max_alpha=float(args.image_l1_bin_alpha_max_alpha),
                min_bin_samples=int(args.image_l1_bin_alpha_min_bin_samples),
                min_relative_gain=float(args.image_l1_bin_alpha_min_relative_gain),
                min_positive_view_fraction=float(args.image_l1_bin_alpha_min_positive_view_fraction),
                count_tau=float(args.image_l1_bin_alpha_count_tau),
                fallback_mode=str(args.image_l1_bin_alpha_fallback_mode),
                max_profile_bins=int(args.image_l1_bin_alpha_max_profile_bins),
            )
        if bool(args.enable_policy_val_image_linear_residual_generator):
            try:
                image_linear_alpha_grid = parse_float_candidates(str(args.image_linear_generator_alpha_grid))
            except ValueError as exc:
                parser.error(f"invalid image-linear generator alpha grid: {exc}")
            alpha_candidates, local_alpha_profile = calibrated_image_linear_generator_profile_from_policy_val(
                cand_val_views,
                cand_atlas,
                base_alpha_grid=alpha_candidates,
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                max_samples_per_view=int(args.max_samples_per_view),
                min_atlas_bin_count=int(args.min_atlas_bin_count),
                min_atlas_face_samples=int(args.min_atlas_face_samples),
                max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                atlas_confidence_mode=str(args.atlas_confidence_mode),
                atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                min_atlas_confidence=float(args.min_atlas_confidence),
                feature_mode=str(args.image_linear_generator_feature_mode),
                ridge=float(args.image_linear_generator_ridge),
                train_max_samples_per_view=int(args.image_linear_generator_train_max_samples_per_view),
                max_train_samples=int(args.image_linear_generator_max_train_samples),
                generator_output_cap=float(args.image_linear_generator_output_cap),
                alpha_grid=image_linear_alpha_grid,
                require_base_valid=not bool(args.image_linear_generator_allow_unvalidated_base_pixels),
                loss_mode=str(args.image_linear_generator_loss_mode),
                irls_iterations=int(args.image_linear_generator_irls_iterations),
                huber_delta=float(args.image_linear_generator_huber_delta),
                training_sample_policy=str(args.image_linear_generator_training_sample_policy),
                min_descent_margin=float(args.image_linear_generator_min_descent_margin),
                min_training_samples=int(args.image_linear_generator_min_training_samples),
                expert_mode=str(args.image_linear_generator_expert_mode),
                expert_min_training_samples=int(args.image_linear_generator_expert_min_training_samples),
                expert_shrink_tau=float(args.image_linear_generator_expert_shrink_tau),
                face_reliability_mode=str(args.image_linear_generator_face_reliability_mode),
                face_reliability_min_face_samples=int(
                    args.image_linear_generator_face_reliability_min_face_samples
                ),
                face_reliability_min_relative_gain=float(
                    args.image_linear_generator_face_reliability_min_relative_gain
                ),
                face_reliability_min_positive_view_fraction=float(
                    args.image_linear_generator_face_reliability_min_positive_view_fraction
                ),
                face_reliability_fallback_multiplier=float(
                    args.image_linear_generator_face_reliability_fallback_multiplier
                ),
            )
        alpha_candidates, ssim_alpha_refinement_summary = refine_alpha_grid_for_policy_val_ssim(
            alpha_candidates,
            enabled=bool(args.enable_policy_val_ssim_alpha_refinement),
            steps=int(args.policy_val_ssim_alpha_refinement_steps),
            min_alpha=float(args.policy_val_ssim_alpha_refinement_min_alpha),
        )
        alpha_candidates, alpha_midpoint_refinement_summary = augment_alpha_grid_with_midpoints(
            alpha_candidates,
            enabled=bool(args.enable_policy_val_alpha_midpoint_refinement),
        )
        cand_fit_summary["local_alpha_calibration"] = dict(local_alpha_profile)
        cand_fit_summary["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
        cand_fit_summary["alpha_midpoint_refinement"] = dict(alpha_midpoint_refinement_summary)
        cand_policy_val = evaluate_policy_val(
            cand_val_views,
            cand_atlas,
            residual_rgb_key=str(args.residual_rgb_key),
            residual_l1_key=str(args.residual_l1_key),
            alpha_grid=alpha_candidates,
            min_l1=float(args.min_l1),
            min_alpha=float(args.min_alpha),
            max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
            max_samples_per_view=int(args.max_samples_per_view),
            min_atlas_bin_count=int(args.min_atlas_bin_count),
            min_atlas_face_samples=int(args.min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(args.atlas_confidence_mode),
            atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(args.min_atlas_confidence),
            enable_policy_val_image_ssim=bool(args.enable_policy_val_image_ssim_gate),
            policy_val_ssim_max_size=int(args.policy_val_ssim_max_size),
            enable_policy_val_image_l1=bool(args.enable_policy_val_image_l1_gate),
            policy_val_l1_max_size=int(args.policy_val_l1_max_size),
            enable_policy_val_image_lpips=bool(args.enable_policy_val_image_lpips_gate),
            policy_val_lpips_max_size=int(args.policy_val_lpips_max_size),
            local_alpha_profile=local_alpha_profile,
            parent_edge_apply_profile=parent_edge_apply_profile,
        )
        cand_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
        cand_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
        cand_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
        cand_policy_val["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
        cand_policy_val["alpha_midpoint_refinement"] = dict(alpha_midpoint_refinement_summary)

        def select_policy_val_payload(
            policy_payload: dict[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any], float, list[str], bool]:
            return select_policy_val_payload_by_risk_gate(policy_payload, args)

        cand_policy_val, cand_best, cand_selected_alpha, cand_risk_reasons, cand_accepted = (
            select_policy_val_payload(cand_policy_val)
        )

        view_basis_guard: dict[str, Any] = {
            "enabled": False,
            "mode": str(args.view_conditioned_basis_guard_mode),
            "decision": "not_requested",
        }
        if (
            str(args.view_conditioned_basis_guard_mode) == "policy_val_nonregressive"
            and str(args.view_conditioned_basis_mode) != "none"
            and int((cand_fit_summary.get("view_conditioned_basis") or {}).get("feature_dim", 0)) > 0
        ):
            legacy_atlas = disable_view_conditioned_basis(cand_atlas)
            legacy_policy_val = evaluate_policy_val(
                cand_val_views,
                legacy_atlas,
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                alpha_grid=alpha_candidates,
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                max_samples_per_view=int(args.max_samples_per_view),
                min_atlas_bin_count=int(args.min_atlas_bin_count),
                min_atlas_face_samples=int(args.min_atlas_face_samples),
                max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                atlas_confidence_mode=str(args.atlas_confidence_mode),
                atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                min_atlas_confidence=float(args.min_atlas_confidence),
                enable_policy_val_image_ssim=bool(args.enable_policy_val_image_ssim_gate),
                policy_val_ssim_max_size=int(args.policy_val_ssim_max_size),
                enable_policy_val_image_l1=bool(args.enable_policy_val_image_l1_gate),
                policy_val_l1_max_size=int(args.policy_val_l1_max_size),
                enable_policy_val_image_lpips=bool(args.enable_policy_val_image_lpips_gate),
                policy_val_lpips_max_size=int(args.policy_val_lpips_max_size),
                local_alpha_profile=local_alpha_profile,
                parent_edge_apply_profile=parent_edge_apply_profile,
            )
            legacy_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
            legacy_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
            legacy_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
            legacy_policy_val["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
            legacy_policy_val["alpha_midpoint_refinement"] = dict(alpha_midpoint_refinement_summary)
            legacy_policy_val, legacy_best, legacy_selected_alpha, legacy_risk_reasons, legacy_accepted = (
                select_policy_val_payload(legacy_policy_val)
            )
            guarded_metrics = [
                "relative_gain",
                "ssim_gain",
                "image_l1_gain",
                "cvar20_view_relative_gain",
                "min_view_relative_gain",
                "image_l1_cvar20_view_gain",
                "image_l1_min_view_gain",
            ]
            if bool(args.enable_policy_val_image_lpips_gate):
                guarded_metrics.extend(
                    [
                        "lpips_gain",
                        "lpips_cvar20_view_gain",
                        "lpips_min_view_gain",
                    ]
                )
            guard_reasons: list[str] = []
            eps = 1.0e-12
            if not cand_accepted and legacy_accepted:
                guard_reasons.append("basis rejected by policy-val while mean-atlas baseline is accepted")
            if cand_accepted and legacy_accepted:
                for metric in guarded_metrics:
                    if float(cand_best.get(metric, -1.0)) + eps < float(legacy_best.get(metric, -1.0)):
                        guard_reasons.append(
                            f"{metric} {float(cand_best.get(metric, -1.0)):.8f} < "
                            f"legacy {float(legacy_best.get(metric, -1.0)):.8f}"
                        )
            fallback_to_mean = bool(guard_reasons and legacy_accepted)
            view_basis_guard = {
                "enabled": True,
                "mode": "policy_val_nonregressive",
                "decision": "fallback_to_mean" if fallback_to_mean else "keep_view_basis",
                "reasons": guard_reasons,
                "basis_accepted": bool(cand_accepted),
                "legacy_accepted": bool(legacy_accepted),
                "basis_selected_alpha": float(cand_selected_alpha),
                "legacy_selected_alpha": float(legacy_selected_alpha),
                "basis_best": dict(cand_best),
                "legacy_best": dict(legacy_best),
            }
            if fallback_to_mean:
                cand_atlas = legacy_atlas
                cand_policy_val = legacy_policy_val
                cand_best = legacy_best
                cand_selected_alpha = float(legacy_selected_alpha)
                cand_risk_reasons = legacy_risk_reasons
                cand_accepted = bool(legacy_accepted)
                if "view_conditioned_basis" in cand_fit_summary:
                    cand_fit_summary["view_conditioned_basis"]["effective_mode"] = "none"
                    cand_fit_summary["view_conditioned_basis"]["requested_mode"] = str(
                        args.view_conditioned_basis_mode
                    )
            elif "view_conditioned_basis" in cand_fit_summary:
                cand_fit_summary["view_conditioned_basis"]["effective_mode"] = str(args.view_conditioned_basis_mode)
                cand_fit_summary["view_conditioned_basis"]["requested_mode"] = str(args.view_conditioned_basis_mode)
        if "view_conditioned_basis" in cand_fit_summary:
            cand_fit_summary["view_conditioned_basis"]["guard"] = dict(view_basis_guard)
        teacher_basis_guard: dict[str, Any] = {
            "enabled": False,
            "mode": str(args.teacher_distilled_basis_guard_mode),
            "decision": "not_requested",
        }
        if (
            str(args.teacher_distilled_basis_guard_mode) == "policy_val_nonregressive"
            and str(args.teacher_distilled_basis_mode) != "none"
            and int((cand_fit_summary.get("teacher_distilled_basis") or {}).get("feature_dim", 0)) > 0
        ):
            legacy_atlas = disable_teacher_distilled_basis(cand_atlas)
            legacy_policy_val = evaluate_policy_val(
                cand_val_views,
                legacy_atlas,
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                alpha_grid=alpha_candidates,
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                max_samples_per_view=int(args.max_samples_per_view),
                min_atlas_bin_count=int(args.min_atlas_bin_count),
                min_atlas_face_samples=int(args.min_atlas_face_samples),
                max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                atlas_confidence_mode=str(args.atlas_confidence_mode),
                atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                min_atlas_confidence=float(args.min_atlas_confidence),
                enable_policy_val_image_ssim=bool(args.enable_policy_val_image_ssim_gate),
                policy_val_ssim_max_size=int(args.policy_val_ssim_max_size),
                enable_policy_val_image_l1=bool(args.enable_policy_val_image_l1_gate),
                policy_val_l1_max_size=int(args.policy_val_l1_max_size),
                enable_policy_val_image_lpips=bool(args.enable_policy_val_image_lpips_gate),
                policy_val_lpips_max_size=int(args.policy_val_lpips_max_size),
                local_alpha_profile=local_alpha_profile,
                parent_edge_apply_profile=parent_edge_apply_profile,
            )
            legacy_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
            legacy_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
            legacy_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
            legacy_policy_val["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
            legacy_policy_val["alpha_midpoint_refinement"] = dict(alpha_midpoint_refinement_summary)
            legacy_policy_val, legacy_best, legacy_selected_alpha, legacy_risk_reasons, legacy_accepted = (
                select_policy_val_payload(legacy_policy_val)
            )
            guarded_metrics = [
                "relative_gain",
                "ssim_gain",
                "image_l1_gain",
                "cvar20_view_relative_gain",
                "min_view_relative_gain",
                "image_l1_cvar20_view_gain",
                "image_l1_min_view_gain",
            ]
            if bool(args.enable_policy_val_image_lpips_gate):
                guarded_metrics.extend(
                    [
                        "lpips_gain",
                        "lpips_cvar20_view_gain",
                        "lpips_min_view_gain",
                    ]
                )
            guard_reasons: list[str] = []
            eps = 1.0e-12
            if not cand_accepted and legacy_accepted:
                guard_reasons.append("teacher basis rejected by policy-val while legacy atlas is accepted")
            if cand_accepted and legacy_accepted:
                for metric in guarded_metrics:
                    if float(cand_best.get(metric, -1.0)) + eps < float(legacy_best.get(metric, -1.0)):
                        guard_reasons.append(
                            f"{metric} {float(cand_best.get(metric, -1.0)):.8f} < "
                            f"legacy {float(legacy_best.get(metric, -1.0)):.8f}"
                        )
            fallback_to_legacy = bool(guard_reasons and legacy_accepted)
            teacher_basis_guard = {
                "enabled": True,
                "mode": "policy_val_nonregressive",
                "decision": "fallback_to_legacy" if fallback_to_legacy else "keep_teacher_basis",
                "reasons": guard_reasons,
                "basis_accepted": bool(cand_accepted),
                "legacy_accepted": bool(legacy_accepted),
                "basis_selected_alpha": float(cand_selected_alpha),
                "legacy_selected_alpha": float(legacy_selected_alpha),
                "basis_best": dict(cand_best),
                "legacy_best": dict(legacy_best),
            }
            if fallback_to_legacy:
                cand_atlas = legacy_atlas
                cand_policy_val = legacy_policy_val
                cand_best = legacy_best
                cand_selected_alpha = float(legacy_selected_alpha)
                cand_risk_reasons = legacy_risk_reasons
                cand_accepted = bool(legacy_accepted)
                if "teacher_distilled_basis" in cand_fit_summary:
                    cand_fit_summary["teacher_distilled_basis"]["effective_mode"] = "none"
                    cand_fit_summary["teacher_distilled_basis"]["requested_mode"] = str(
                        args.teacher_distilled_basis_mode
                    )
            elif "teacher_distilled_basis" in cand_fit_summary:
                cand_fit_summary["teacher_distilled_basis"]["effective_mode"] = str(
                    args.teacher_distilled_basis_mode
                )
                cand_fit_summary["teacher_distilled_basis"]["requested_mode"] = str(
                    args.teacher_distilled_basis_mode
                )
        if "teacher_distilled_basis" in cand_fit_summary:
            cand_fit_summary["teacher_distilled_basis"]["guard"] = dict(teacher_basis_guard)
        view_alpha_cap_pre_guard_policy_val = copy.deepcopy(cand_policy_val)
        view_alpha_cap_pre_guard_best = copy.deepcopy(cand_best)
        view_alpha_cap_pre_guard_selected_alpha = float(cand_selected_alpha)
        view_alpha_cap_pre_guard_risk_reasons = list(cand_risk_reasons)
        view_alpha_cap_pre_guard_accepted = bool(cand_accepted)
        guard_repair_seed_row: dict[str, Any] | None = None
        guard_repair_seed_reasons: list[str] = []
        guard_repair_seed_alpha = float(cand_selected_alpha)
        if bool(args.enable_preacceptance_policy_val_guard_repair) and not bool(cand_accepted):
            guard_repair_seed_row, guard_repair_seed_reasons = select_policy_val_preacceptance_repair_seed(
                cand_policy_val,
                args,
            )
            if guard_repair_seed_row:
                guard_repair_seed_alpha = float(guard_repair_seed_row.get("alpha", 0.0))
        cand_fit_summary["preacceptance_guard_repair"] = {
            "enabled": bool(args.enable_preacceptance_policy_val_guard_repair),
            "seed_found": bool(guard_repair_seed_row),
            "seed_alpha": float(guard_repair_seed_alpha if guard_repair_seed_row else 0.0),
            "seed_best": dict(guard_repair_seed_row or {}),
            "seed_risk_reasons": list(guard_repair_seed_reasons),
        }
        sparse_materialization_profile: dict[str, Any] = {
            "enabled": False,
            "mode": "policy_val_sparse_residual_materialization",
            "decision": "not_requested",
        }
        selected_sparse_materialization_profile: dict[str, Any] = {
            "enabled": False,
            "mode": "policy_val_sparse_residual_materialization",
            "decision": "not_selected",
        }
        if bool(args.enable_policy_val_sparse_residual_materialization):
            sparse_materialization_profile = policy_val_sparse_materialization_profile(
                cand_val_views,
                cand_atlas,
                cand_policy_val,
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                max_samples_per_view=int(args.max_samples_per_view),
                min_atlas_bin_count=int(args.min_atlas_bin_count),
                min_atlas_face_samples=int(args.min_atlas_face_samples),
                max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                atlas_confidence_mode=str(args.atlas_confidence_mode),
                atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                min_atlas_confidence=float(args.min_atlas_confidence),
                local_alpha_profile=local_alpha_profile,
                face_gain_guard_profile=None,
                seed_min_relative_gain=float(args.sparse_materialization_seed_min_relative_gain),
                min_bin_samples=int(args.sparse_materialization_min_bin_samples),
                min_bin_views=int(args.sparse_materialization_min_bin_views),
                min_relative_gain=float(args.sparse_materialization_min_relative_gain),
                min_view_relative_gain=float(args.sparse_materialization_min_view_relative_gain),
                min_positive_view_fraction=float(args.sparse_materialization_min_positive_view_fraction),
                max_mean_variance=float(args.sparse_materialization_max_mean_variance),
                min_mean_sign_consistency=float(args.sparse_materialization_min_mean_sign_consistency),
                adaptive_frontier_on_empty=bool(args.enable_policy_val_sparse_materialization_frontier),
                adaptive_frontier_min_positive_view_fraction=float(
                    args.sparse_materialization_frontier_min_positive_view_fraction
                ),
                adaptive_frontier_min_risk_adjusted_gain=float(
                    args.sparse_materialization_frontier_min_risk_adjusted_gain
                ),
                adaptive_frontier_min_sample_quantile=float(
                    args.sparse_materialization_frontier_min_sample_quantile
                ),
                target_footprint_views=(
                    evidence_views(target_evidence)
                    if (
                        bool(args.enable_sparse_materialization_target_visible_expansion)
                        or bool(args.enable_train_only_target_impact_residual_basis)
                        or bool(args.enable_sparse_materialization_target_connected_region_growth)
                    )
                    else None
                ),
                enable_target_visible_expansion=bool(
                    args.enable_sparse_materialization_target_visible_expansion
                ),
                target_visible_min_pixels=int(args.sparse_materialization_target_visible_min_pixels),
                target_visible_min_views=int(args.sparse_materialization_target_visible_min_views),
                target_visible_min_policy_samples=int(
                    args.sparse_materialization_target_visible_min_policy_samples
                ),
                target_visible_min_positive_view_fraction=float(
                    args.sparse_materialization_target_visible_min_positive_view_fraction
                ),
                target_visible_max_extra_bins=int(args.sparse_materialization_target_visible_max_extra_bins),
                target_visible_max_views=int(args.target_footprint_max_views),
                enable_train_only_target_impact_residual_basis=bool(
                    args.enable_train_only_target_impact_residual_basis
                ),
                target_impact_min_pixels=int(args.target_impact_min_pixels),
                target_impact_min_views=int(args.target_impact_min_views),
                target_impact_min_policy_samples=int(args.target_impact_min_policy_samples),
                target_impact_max_extra_bins=int(args.target_impact_max_extra_bins),
                target_impact_max_views=int(args.target_impact_max_views),
                enable_target_connected_region_growth=bool(
                    args.enable_sparse_materialization_target_connected_region_growth
                ),
                target_connected_radius=int(args.sparse_materialization_target_connected_radius),
                target_connected_min_pixels=int(args.sparse_materialization_target_connected_min_pixels),
                target_connected_min_views=int(args.sparse_materialization_target_connected_min_views),
                target_connected_min_policy_samples=int(
                    args.sparse_materialization_target_connected_min_policy_samples
                ),
                target_connected_min_positive_view_fraction=float(
                    args.sparse_materialization_target_connected_min_positive_view_fraction
                ),
                target_connected_max_negative_relative_gain=float(
                    args.sparse_materialization_target_connected_max_negative_relative_gain
                ),
                target_connected_max_negative_min_view_gain=float(
                    args.sparse_materialization_target_connected_max_negative_min_view_gain
                ),
                target_connected_max_extra_bins=int(
                    args.sparse_materialization_target_connected_max_extra_bins
                ),
                target_connected_max_views=int(args.target_footprint_max_views),
            )
            target_impact_carrier_fill = apply_target_impact_carrier_fill(
                cand_atlas,
                sparse_materialization_profile,
                mode=str(args.target_impact_carrier_fill_mode),
                blend=float(args.target_impact_carrier_fill_blend),
                min_face_samples=int(args.target_impact_carrier_fill_min_face_samples),
                min_carrier_norm=float(args.target_impact_carrier_fill_min_norm),
                max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                synthetic_count=int(args.target_impact_carrier_fill_synthetic_count),
            )
            sparse_materialization_profile["target_impact_carrier_fill"] = dict(
                target_impact_carrier_fill
            )
            cand_fit_summary["target_impact_carrier_fill"] = dict(target_impact_carrier_fill)
            target_impact_multisample_fill = apply_target_impact_multisample_residual_fill(
                cand_atlas,
                cand_fit_views,
                sparse_materialization_profile,
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                max_samples_per_view=int(args.max_samples_per_view),
                mode=str(args.target_impact_multisample_fill_mode),
                radius=int(args.target_impact_multisample_fill_radius),
                min_samples=int(args.target_impact_multisample_fill_min_samples),
                max_samples_per_bin=int(args.target_impact_multisample_fill_max_samples_per_bin),
                max_views=int(args.target_impact_multisample_fill_max_views),
                blend=float(args.target_impact_multisample_fill_blend),
                kernel_sigma=float(args.target_impact_multisample_fill_kernel_sigma),
                min_norm=float(args.target_impact_multisample_fill_min_norm),
                synthetic_count=int(args.target_impact_multisample_fill_synthetic_count),
            )
            sparse_materialization_profile["target_impact_multisample_fill"] = dict(
                target_impact_multisample_fill
            )
            cand_fit_summary["target_impact_multisample_fill"] = dict(target_impact_multisample_fill)
            target_impact_affine_fill = apply_target_impact_affine_residual_fill(
                cand_atlas,
                cand_fit_views,
                sparse_materialization_profile,
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                max_samples_per_view=int(args.max_samples_per_view),
                mode=str(args.target_impact_affine_fill_mode),
                feature_mode=str(args.target_impact_affine_fill_feature_mode),
                min_samples=int(args.target_impact_affine_fill_min_samples),
                max_samples_per_face=int(args.target_impact_affine_fill_max_samples_per_face),
                max_views=int(args.target_impact_affine_fill_max_views),
                blend=float(args.target_impact_affine_fill_blend),
                ridge=float(args.target_impact_affine_fill_ridge),
                max_condition=float(args.target_impact_affine_fill_max_condition),
                min_norm=float(args.target_impact_affine_fill_min_norm),
                synthetic_count=int(args.target_impact_affine_fill_synthetic_count),
            )
            sparse_materialization_profile["target_impact_affine_fill"] = dict(
                target_impact_affine_fill
            )
            cand_fit_summary["target_impact_affine_fill"] = dict(target_impact_affine_fill)
            if bool(sparse_materialization_profile.get("enabled", False)) and int(
                sparse_materialization_profile.get("allowed_bin_count", 0) or 0
            ) > 0:
                sparse_policy_val = evaluate_policy_val(
                    cand_val_views,
                    cand_atlas,
                    residual_rgb_key=str(args.residual_rgb_key),
                    residual_l1_key=str(args.residual_l1_key),
                    alpha_grid=alpha_candidates,
                    min_l1=float(args.min_l1),
                    min_alpha=float(args.min_alpha),
                    max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                    max_samples_per_view=int(args.max_samples_per_view),
                    min_atlas_bin_count=int(args.min_atlas_bin_count),
                    min_atlas_face_samples=int(args.min_atlas_face_samples),
                    max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                    min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                    atlas_confidence_mode=str(args.atlas_confidence_mode),
                    atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                    atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                    atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                    atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                    atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                    min_atlas_confidence=float(args.min_atlas_confidence),
                    enable_policy_val_image_ssim=bool(args.enable_policy_val_image_ssim_gate),
                    policy_val_ssim_max_size=int(args.policy_val_ssim_max_size),
                    enable_policy_val_image_l1=bool(args.enable_policy_val_image_l1_gate),
                    policy_val_l1_max_size=int(args.policy_val_l1_max_size),
                    enable_policy_val_image_lpips=bool(args.enable_policy_val_image_lpips_gate),
                    policy_val_lpips_max_size=int(args.policy_val_lpips_max_size),
                    local_alpha_profile=local_alpha_profile,
                    bin_uncertainty_guard_profile=sparse_materialization_profile,
                    parent_edge_apply_profile=parent_edge_apply_profile,
                )
                sparse_policy_val = annotate_sparse_materialization_selective_policy_val(
                    sparse_policy_val,
                    sparse_materialization_profile,
                )
                sparse_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
                sparse_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
                sparse_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
                sparse_policy_val["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
                sparse_policy_val["alpha_midpoint_refinement"] = dict(alpha_midpoint_refinement_summary)
                (
                    sparse_policy_val,
                    sparse_best,
                    sparse_selected_alpha,
                    sparse_risk_reasons,
                    sparse_accepted,
                ) = select_policy_val_payload(sparse_policy_val)
                sparse_materialization_profile["post_materialization_accepted"] = bool(sparse_accepted)
                sparse_materialization_profile["post_materialization_selected_alpha"] = float(sparse_selected_alpha)
                sparse_materialization_profile["post_materialization_best"] = dict(sparse_best)
                sparse_materialization_profile["post_materialization_risk_reasons"] = list(sparse_risk_reasons)
                replace_with_sparse = bool(sparse_accepted or not cand_accepted)
                sparse_materialization_profile["decision"] = (
                    "replace_candidate_policy_val" if replace_with_sparse else "original_candidate_kept"
                )
                if replace_with_sparse:
                    cand_policy_val = sparse_policy_val
                    cand_best = sparse_best
                    cand_selected_alpha = float(sparse_selected_alpha)
                    cand_risk_reasons = sparse_risk_reasons
                    cand_accepted = bool(sparse_accepted)
                    guard_repair_seed_row = None
                    guard_repair_seed_reasons = []
                    guard_repair_seed_alpha = float(cand_selected_alpha)
                    selected_sparse_materialization_profile = dict(sparse_materialization_profile)
            else:
                sparse_materialization_profile["decision"] = "disabled_no_active_allowed_bins"
        cand_fit_summary["sparse_residual_materialization"] = dict(sparse_materialization_profile)
        face_gain_guard_profile: dict[str, Any] = {
            "enabled": False,
            "mode": "policy_val_face_gain_guard",
            "decision": "not_requested",
        }
        if bool(args.enable_policy_val_face_gain_guard):
            sparse_selective_candidate = bool(
                cand_accepted
                and selected_sparse_materialization_profile.get("enabled", False)
                and cand_best.get("sparse_materialization_selective", False)
            )
            if sparse_selective_candidate:
                face_gain_guard_profile = {
                    "enabled": False,
                    "mode": "policy_val_face_gain_guard",
                    "decision": "skipped_sparse_materialization_already_bin_certified",
                    "reason": (
                        "sparse materialization uses a bin-level selective non-regression "
                        "certificate; applying a face-level positive-view guard would "
                        "downcast sparse no-op views into failures"
                    ),
                    "sparse_materialization_allowed_bin_count": int(
                        selected_sparse_materialization_profile.get("allowed_bin_count", 0) or 0
                    ),
                    "sparse_materialization_allowed_face_count": int(
                        selected_sparse_materialization_profile.get("allowed_face_count", 0) or 0
                    ),
                }
                guard_repair_seed_alpha = float(cand_selected_alpha)
            elif (cand_accepted or guard_repair_seed_row) and float(guard_repair_seed_alpha) > 0.0:
                face_gain_guard_profile = build_policy_val_face_gain_guard_profile(
                    cand_val_views,
                    cand_atlas,
                    residual_rgb_key=str(args.residual_rgb_key),
                    residual_l1_key=str(args.residual_l1_key),
                    alpha=float(guard_repair_seed_alpha),
                    min_l1=float(args.min_l1),
                    min_alpha=float(args.min_alpha),
                    max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                    max_samples_per_view=int(args.max_samples_per_view),
                    min_atlas_bin_count=int(args.min_atlas_bin_count),
                    min_atlas_face_samples=int(args.min_atlas_face_samples),
                    max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                    min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                    atlas_confidence_mode=str(args.atlas_confidence_mode),
                    atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                    atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                    atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                    atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                    atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                    min_atlas_confidence=float(args.min_atlas_confidence),
                    local_alpha_profile=local_alpha_profile,
                    min_face_samples=int(args.face_gain_guard_min_face_samples),
                    min_relative_gain=float(args.face_gain_guard_min_relative_gain),
                    min_positive_view_fraction=float(args.face_gain_guard_min_positive_view_fraction),
                )
                if bool(face_gain_guard_profile.get("enabled", False)):
                    face_gain_guard_profile["preacceptance_repair"] = bool(
                        guard_repair_seed_row and not cand_accepted
                    )
                    guarded_policy_val = evaluate_policy_val(
                        cand_val_views,
                        cand_atlas,
                        residual_rgb_key=str(args.residual_rgb_key),
                        residual_l1_key=str(args.residual_l1_key),
                        alpha_grid=alpha_candidates,
                        min_l1=float(args.min_l1),
                        min_alpha=float(args.min_alpha),
                        max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                        max_samples_per_view=int(args.max_samples_per_view),
                        min_atlas_bin_count=int(args.min_atlas_bin_count),
                        min_atlas_face_samples=int(args.min_atlas_face_samples),
                        max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                        min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                        atlas_confidence_mode=str(args.atlas_confidence_mode),
                        atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                        atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                        atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                        atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                        atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                        min_atlas_confidence=float(args.min_atlas_confidence),
                        enable_policy_val_image_ssim=bool(args.enable_policy_val_image_ssim_gate),
                        policy_val_ssim_max_size=int(args.policy_val_ssim_max_size),
                        enable_policy_val_image_l1=bool(args.enable_policy_val_image_l1_gate),
                        policy_val_l1_max_size=int(args.policy_val_l1_max_size),
                        enable_policy_val_image_lpips=bool(args.enable_policy_val_image_lpips_gate),
                        policy_val_lpips_max_size=int(args.policy_val_lpips_max_size),
                        local_alpha_profile=local_alpha_profile,
                        face_gain_guard_profile=face_gain_guard_profile,
                        parent_edge_apply_profile=parent_edge_apply_profile,
                    )
                    guarded_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
                    guarded_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
                    guarded_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
                    guarded_policy_val["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
                    guarded_policy_val["alpha_midpoint_refinement"] = dict(alpha_midpoint_refinement_summary)
                    (
                        guarded_policy_val,
                        guarded_best,
                        guarded_selected_alpha,
                        guarded_risk_reasons,
                        guarded_accepted,
                    ) = select_policy_val_payload(guarded_policy_val)
                    face_gain_guard_profile["post_guard_accepted"] = bool(guarded_accepted)
                    face_gain_guard_profile["post_guard_selected_alpha"] = float(guarded_selected_alpha)
                    face_gain_guard_profile["post_guard_best"] = dict(guarded_best)
                    face_gain_guard_profile["post_guard_risk_reasons"] = list(guarded_risk_reasons)
                    if guarded_accepted:
                        face_gain_guard_profile["decision"] = "keep_face_gain_guard"
                        cand_policy_val = guarded_policy_val
                        cand_best = guarded_best
                        cand_selected_alpha = float(guarded_selected_alpha)
                        cand_risk_reasons = guarded_risk_reasons
                        cand_accepted = True
                    else:
                        face_gain_guard_profile["decision"] = "reject_candidate_after_face_gain_guard"
                        cand_policy_val = guarded_policy_val
                        cand_best = guarded_best
                        cand_selected_alpha = float(guarded_selected_alpha)
                        cand_risk_reasons = guarded_risk_reasons
                        cand_accepted = False
                    if cand_accepted or not guard_repair_seed_row:
                        guard_repair_seed_alpha = float(cand_selected_alpha)
                else:
                    face_gain_guard_profile["decision"] = "disabled_no_active_allowed_faces"
                    if cand_accepted:
                        cand_accepted = False
                        cand_risk_reasons = [str(face_gain_guard_profile.get("reason", "face gain guard disabled"))]
            else:
                face_gain_guard_profile = {
                    "enabled": False,
                    "mode": "policy_val_face_gain_guard",
                    "decision": "skipped_candidate_not_accepted",
                    "reason": "candidate_not_accepted_or_zero_alpha",
                }
        bin_uncertainty_guard_profile: dict[str, Any] = {
            "enabled": False,
            "mode": "policy_val_bin_uncertainty_guard",
            "decision": "not_requested",
        }
        if bool(selected_sparse_materialization_profile.get("enabled", False)):
            bin_uncertainty_guard_profile = dict(selected_sparse_materialization_profile)
        if bool(args.enable_policy_val_bin_uncertainty_guard):
            if (cand_accepted or guard_repair_seed_row) and float(guard_repair_seed_alpha) > 0.0:
                raw_bin_uncertainty_guard_profile = build_policy_val_bin_uncertainty_guard_profile(
                    cand_val_views,
                    cand_atlas,
                    residual_rgb_key=str(args.residual_rgb_key),
                    residual_l1_key=str(args.residual_l1_key),
                    alpha=float(guard_repair_seed_alpha),
                    min_l1=float(args.min_l1),
                    min_alpha=float(args.min_alpha),
                    max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                    max_samples_per_view=int(args.max_samples_per_view),
                    min_atlas_bin_count=int(args.min_atlas_bin_count),
                    min_atlas_face_samples=int(args.min_atlas_face_samples),
                    max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                    min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                    atlas_confidence_mode=str(args.atlas_confidence_mode),
                    atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                    atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                    atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                    atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                    atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                    min_atlas_confidence=float(args.min_atlas_confidence),
                    local_alpha_profile=local_alpha_profile,
                    face_gain_guard_profile=face_gain_guard_profile,
                    min_bin_samples=int(args.bin_uncertainty_guard_min_bin_samples),
                    min_relative_gain=float(args.bin_uncertainty_guard_min_relative_gain),
                    min_positive_view_fraction=float(args.bin_uncertainty_guard_min_positive_view_fraction),
                    max_mean_variance=float(args.bin_uncertainty_guard_max_mean_variance),
                    min_mean_sign_consistency=float(args.bin_uncertainty_guard_min_mean_sign_consistency),
                )
                if bool(selected_sparse_materialization_profile.get("enabled", False)):
                    bin_uncertainty_guard_profile = intersect_bin_guard_profiles(
                        selected_sparse_materialization_profile,
                        raw_bin_uncertainty_guard_profile,
                        empty_intersection_policy=str(
                            args.bin_uncertainty_guard_empty_intersection_policy
                        ),
                    )
                else:
                    bin_uncertainty_guard_profile = raw_bin_uncertainty_guard_profile
                if bool(bin_uncertainty_guard_profile.get("enabled", False)):
                    bin_uncertainty_guard_profile["preacceptance_repair"] = bool(
                        guard_repair_seed_row and not cand_accepted
                    )
                    guarded_policy_val = evaluate_policy_val(
                        cand_val_views,
                        cand_atlas,
                        residual_rgb_key=str(args.residual_rgb_key),
                        residual_l1_key=str(args.residual_l1_key),
                        alpha_grid=alpha_candidates,
                        min_l1=float(args.min_l1),
                        min_alpha=float(args.min_alpha),
                        max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                        max_samples_per_view=int(args.max_samples_per_view),
                        min_atlas_bin_count=int(args.min_atlas_bin_count),
                        min_atlas_face_samples=int(args.min_atlas_face_samples),
                        max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                        min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                        atlas_confidence_mode=str(args.atlas_confidence_mode),
                        atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                        atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                        atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                        atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                        atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                        min_atlas_confidence=float(args.min_atlas_confidence),
                        enable_policy_val_image_ssim=bool(args.enable_policy_val_image_ssim_gate),
                        policy_val_ssim_max_size=int(args.policy_val_ssim_max_size),
                        enable_policy_val_image_l1=bool(args.enable_policy_val_image_l1_gate),
                        policy_val_l1_max_size=int(args.policy_val_l1_max_size),
                        enable_policy_val_image_lpips=bool(args.enable_policy_val_image_lpips_gate),
                        policy_val_lpips_max_size=int(args.policy_val_lpips_max_size),
                        local_alpha_profile=local_alpha_profile,
                        face_gain_guard_profile=face_gain_guard_profile,
                        bin_uncertainty_guard_profile=bin_uncertainty_guard_profile,
                        parent_edge_apply_profile=parent_edge_apply_profile,
                    )
                    if bool(selected_sparse_materialization_profile.get("enabled", False)) and int(
                        bin_uncertainty_guard_profile.get("allowed_bin_count", 0) or 0
                    ) > 0:
                        guarded_policy_val = annotate_sparse_materialization_selective_policy_val(
                            guarded_policy_val,
                            bin_uncertainty_guard_profile,
                        )
                        bin_uncertainty_guard_profile["sparse_selective_annotation"] = {
                            "enabled": True,
                            "reason": (
                                "preserve sparse selective non-regression semantics after "
                                "bin-guard intersection or bridge"
                            ),
                            "source_sparse_allowed_bin_count": int(
                                selected_sparse_materialization_profile.get("allowed_bin_count", 0) or 0
                            ),
                            "guard_allowed_bin_count": int(
                                bin_uncertainty_guard_profile.get("allowed_bin_count", 0) or 0
                            ),
                        }
                    guarded_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
                    guarded_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
                    guarded_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
                    guarded_policy_val["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
                    guarded_policy_val["alpha_midpoint_refinement"] = dict(alpha_midpoint_refinement_summary)
                    (
                        guarded_policy_val,
                        guarded_best,
                        guarded_selected_alpha,
                        guarded_risk_reasons,
                        guarded_accepted,
                    ) = select_policy_val_payload(guarded_policy_val)
                    bin_uncertainty_guard_profile["post_guard_accepted"] = bool(guarded_accepted)
                    bin_uncertainty_guard_profile["post_guard_selected_alpha"] = float(guarded_selected_alpha)
                    bin_uncertainty_guard_profile["post_guard_best"] = dict(guarded_best)
                    bin_uncertainty_guard_profile["post_guard_risk_reasons"] = list(guarded_risk_reasons)
                    if guarded_accepted:
                        bin_uncertainty_guard_profile["decision"] = "keep_bin_uncertainty_guard"
                        cand_policy_val = guarded_policy_val
                        cand_best = guarded_best
                        cand_selected_alpha = float(guarded_selected_alpha)
                        cand_risk_reasons = guarded_risk_reasons
                        cand_accepted = True
                    else:
                        bin_uncertainty_guard_profile["decision"] = "reject_candidate_after_bin_uncertainty_guard"
                        cand_policy_val = guarded_policy_val
                        cand_best = guarded_best
                        cand_selected_alpha = float(guarded_selected_alpha)
                        cand_risk_reasons = guarded_risk_reasons
                        cand_accepted = False
                else:
                    bin_uncertainty_guard_profile["decision"] = "disabled_no_active_allowed_bins"
                    if cand_accepted:
                        cand_accepted = False
                        cand_risk_reasons = [
                            str(bin_uncertainty_guard_profile.get("reason", "bin uncertainty guard disabled"))
                        ]
            else:
                bin_uncertainty_guard_profile = {
                    "enabled": False,
                    "mode": "policy_val_bin_uncertainty_guard",
                    "decision": "skipped_candidate_not_accepted",
                    "reason": "candidate_not_accepted_or_zero_alpha",
                }
        view_confidence_profile: dict[str, Any] = {
            "enabled": False,
            "mode": "policy_val_view_consistency_confidence",
            "decision": "not_requested",
        }
        if bool(args.enable_policy_val_view_consistency_confidence):
            confidence_seed_alpha = float(guard_repair_seed_alpha)
            confidence_seed_source = "preacceptance_guard_repair"
            if confidence_seed_alpha <= 0.0:
                confidence_seed_alpha = float(cand_selected_alpha)
                confidence_seed_source = "selected_alpha"
            if confidence_seed_alpha <= 0.0 and cand_best:
                confidence_seed_alpha = float(cand_best.get("alpha", 0.0) or 0.0)
                confidence_seed_source = "policy_val_best_alpha"
            view_confidence_profile = build_policy_val_view_confidence_profile(
                cand_val_views,
                cand_policy_val,
                confidence_seed_alpha,
                enabled=True,
                min_relative_gain=float(args.view_confidence_min_relative_gain),
                min_ssim_gain=float(args.view_confidence_min_ssim_gain),
                min_l1_gain=float(args.view_confidence_min_l1_gain),
                min_lpips_gain=float(args.view_confidence_min_lpips_gain),
                kernel_sigma=float(args.view_confidence_kernel_sigma),
                min_confidence=float(args.view_confidence_min_confidence),
            )
            view_confidence_profile["seed_source"] = str(confidence_seed_source)
            if bool(view_confidence_profile.get("enabled", False)):
                confidence_policy_val = evaluate_policy_val(
                    cand_val_views,
                    cand_atlas,
                    residual_rgb_key=str(args.residual_rgb_key),
                    residual_l1_key=str(args.residual_l1_key),
                    alpha_grid=alpha_candidates,
                    min_l1=float(args.min_l1),
                    min_alpha=float(args.min_alpha),
                    max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                    max_samples_per_view=int(args.max_samples_per_view),
                    min_atlas_bin_count=int(args.min_atlas_bin_count),
                    min_atlas_face_samples=int(args.min_atlas_face_samples),
                    max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                    min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                    atlas_confidence_mode=str(args.atlas_confidence_mode),
                    atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                    atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                    atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                    atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                    atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                    min_atlas_confidence=float(args.min_atlas_confidence),
                    enable_policy_val_image_ssim=bool(args.enable_policy_val_image_ssim_gate),
                    policy_val_ssim_max_size=int(args.policy_val_ssim_max_size),
                    enable_policy_val_image_l1=bool(args.enable_policy_val_image_l1_gate),
                    policy_val_l1_max_size=int(args.policy_val_l1_max_size),
                    enable_policy_val_image_lpips=bool(args.enable_policy_val_image_lpips_gate),
                    policy_val_lpips_max_size=int(args.policy_val_lpips_max_size),
                    local_alpha_profile=local_alpha_profile,
                    face_gain_guard_profile=face_gain_guard_profile,
                    bin_uncertainty_guard_profile=bin_uncertainty_guard_profile,
                    parent_edge_apply_profile=parent_edge_apply_profile,
                    view_confidence_profile=view_confidence_profile,
                )
                confidence_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
                confidence_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
                confidence_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
                confidence_policy_val["view_consistency_confidence"] = sanitize_view_confidence_profile(
                    view_confidence_profile
                )
                confidence_policy_val["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
                confidence_policy_val["alpha_midpoint_refinement"] = dict(alpha_midpoint_refinement_summary)
                (
                    confidence_policy_val,
                    confidence_best,
                    confidence_selected_alpha,
                    confidence_risk_reasons,
                    confidence_accepted,
                ) = select_policy_val_payload(confidence_policy_val)
                view_confidence_profile["post_confidence_accepted"] = bool(confidence_accepted)
                view_confidence_profile["post_confidence_selected_alpha"] = float(confidence_selected_alpha)
                view_confidence_profile["post_confidence_best"] = dict(confidence_best)
                view_confidence_profile["post_confidence_risk_reasons"] = list(confidence_risk_reasons)
                current_rel = float((cand_best or {}).get("relative_gain", -1.0))
                confidence_rel = float((confidence_best or {}).get("relative_gain", -1.0))
                current_cvar = float((cand_best or {}).get("cvar20_view_relative_gain", -1.0))
                confidence_cvar = float((confidence_best or {}).get("cvar20_view_relative_gain", -1.0))
                replace_with_confidence = bool(
                    confidence_accepted
                    and (
                        not cand_accepted
                        or confidence_rel >= current_rel
                        or (confidence_cvar > current_cvar and confidence_rel >= 0.90 * max(current_rel, 1.0e-12))
                    )
                )
                view_confidence_profile["decision"] = (
                    "replace_candidate_policy_val" if replace_with_confidence else "original_candidate_kept"
                )
                if replace_with_confidence:
                    cand_policy_val = confidence_policy_val
                    cand_best = confidence_best
                    cand_selected_alpha = float(confidence_selected_alpha)
                    cand_risk_reasons = confidence_risk_reasons
                    cand_accepted = bool(confidence_accepted)
                    guard_repair_seed_alpha = float(cand_selected_alpha)
                elif not cand_accepted:
                    view_confidence_profile["decision"] = "reject_candidate_after_view_confidence"
                    cand_policy_val = confidence_policy_val
                    cand_best = confidence_best
                    cand_selected_alpha = float(confidence_selected_alpha)
                    cand_risk_reasons = confidence_risk_reasons
                    cand_accepted = False
            else:
                view_confidence_profile["decision"] = "disabled_no_certified_positive_views"
        view_alpha_cap_profile: dict[str, Any] = {
            "enabled": False,
            "mode": "policy_val_view_alpha_cap",
            "decision": "not_requested",
        }
        if bool(args.enable_policy_val_view_alpha_cap):
            view_alpha_cap_seed_stage = str(args.view_alpha_cap_seed_stage)
            if view_alpha_cap_seed_stage == "pre_guard":
                view_alpha_cap_seed_policy_val = view_alpha_cap_pre_guard_policy_val
                view_alpha_cap_seed_best = view_alpha_cap_pre_guard_best
                view_alpha_cap_seed_selected_alpha = float(view_alpha_cap_pre_guard_selected_alpha)
                view_alpha_cap_seed_risk_reasons = list(view_alpha_cap_pre_guard_risk_reasons)
                view_alpha_cap_seed_accepted = bool(view_alpha_cap_pre_guard_accepted)
            else:
                view_alpha_cap_seed_policy_val = cand_policy_val
                view_alpha_cap_seed_best = cand_best
                view_alpha_cap_seed_selected_alpha = float(cand_selected_alpha)
                view_alpha_cap_seed_risk_reasons = list(cand_risk_reasons)
                view_alpha_cap_seed_accepted = bool(cand_accepted)
            view_alpha_cap_profile = build_policy_val_view_alpha_cap_profile(
                cand_val_views,
                view_alpha_cap_seed_policy_val,
                enabled=True,
                selection_mode=str(args.view_alpha_cap_selection_mode),
                min_relative_gain=float(args.view_alpha_cap_min_relative_gain),
                min_ssim_gain=float(args.view_alpha_cap_min_ssim_gain),
                min_l1_gain=float(args.view_alpha_cap_min_l1_gain),
                min_lpips_gain=float(args.view_alpha_cap_min_lpips_gain),
                kernel_sigma=float(args.view_alpha_cap_kernel_sigma),
                min_confidence=float(args.view_alpha_cap_min_confidence),
                fallback_alpha=float(args.view_alpha_cap_fallback_alpha),
            )
            view_alpha_cap_profile["seed_source"] = f"{view_alpha_cap_seed_stage}_policy_val"
            view_alpha_cap_profile["seed_stage"] = str(view_alpha_cap_seed_stage)
            view_alpha_cap_profile["seed_selected_alpha"] = float(view_alpha_cap_seed_selected_alpha)
            view_alpha_cap_profile["seed_best"] = dict(view_alpha_cap_seed_best)
            view_alpha_cap_profile["seed_risk_reasons"] = list(view_alpha_cap_seed_risk_reasons)
            view_alpha_cap_profile["seed_accepted"] = bool(view_alpha_cap_seed_accepted)
            view_alpha_cap_profile["pre_cap_selected_alpha"] = float(cand_selected_alpha)
            view_alpha_cap_profile["pre_cap_best"] = dict(cand_best)
            view_alpha_cap_profile["pre_cap_risk_reasons"] = list(cand_risk_reasons)
            if bool(view_alpha_cap_profile.get("enabled", False)):
                capped_local_alpha_profile = dict(local_alpha_profile)
                capped_local_alpha_profile["view_alpha_cap_profile"] = dict(view_alpha_cap_profile)
                capped_policy_val = evaluate_policy_val(
                    cand_val_views,
                    cand_atlas,
                    residual_rgb_key=str(args.residual_rgb_key),
                    residual_l1_key=str(args.residual_l1_key),
                    alpha_grid=alpha_candidates,
                    min_l1=float(args.min_l1),
                    min_alpha=float(args.min_alpha),
                    max_abs_delta_rgb=float(max_abs_delta_rgb_candidate),
                    max_samples_per_view=int(args.max_samples_per_view),
                    min_atlas_bin_count=int(args.min_atlas_bin_count),
                    min_atlas_face_samples=int(args.min_atlas_face_samples),
                    max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                    min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                    atlas_confidence_mode=str(args.atlas_confidence_mode),
                    atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                    atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                    atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                    atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                    atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                    min_atlas_confidence=float(args.min_atlas_confidence),
                    enable_policy_val_image_ssim=bool(args.enable_policy_val_image_ssim_gate),
                    policy_val_ssim_max_size=int(args.policy_val_ssim_max_size),
                    enable_policy_val_image_l1=bool(args.enable_policy_val_image_l1_gate),
                    policy_val_l1_max_size=int(args.policy_val_l1_max_size),
                    enable_policy_val_image_lpips=bool(args.enable_policy_val_image_lpips_gate),
                    policy_val_lpips_max_size=int(args.policy_val_lpips_max_size),
                    local_alpha_profile=capped_local_alpha_profile,
                    face_gain_guard_profile=face_gain_guard_profile,
                    bin_uncertainty_guard_profile=bin_uncertainty_guard_profile,
                    parent_edge_apply_profile=parent_edge_apply_profile,
                    view_confidence_profile=view_confidence_profile,
                )
                capped_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
                capped_policy_val["local_alpha_calibration"] = dict(capped_local_alpha_profile)
                capped_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
                capped_policy_val["view_consistency_confidence"] = sanitize_view_confidence_profile(
                    view_confidence_profile
                )
                capped_policy_val["view_alpha_cap"] = sanitize_view_alpha_cap_profile(view_alpha_cap_profile)
                capped_policy_val["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
                capped_policy_val["alpha_midpoint_refinement"] = dict(alpha_midpoint_refinement_summary)
                (
                    capped_policy_val,
                    capped_best,
                    capped_selected_alpha,
                    capped_risk_reasons,
                    capped_accepted,
                ) = select_policy_val_payload(capped_policy_val)
                view_alpha_cap_profile["post_cap_accepted"] = bool(capped_accepted)
                view_alpha_cap_profile["post_cap_selected_alpha"] = float(capped_selected_alpha)
                view_alpha_cap_profile["post_cap_best"] = dict(capped_best)
                view_alpha_cap_profile["post_cap_risk_reasons"] = list(capped_risk_reasons)
                current_rel = float((cand_best or {}).get("relative_gain", -1.0))
                capped_rel = float((capped_best or {}).get("relative_gain", -1.0))
                current_reason_count = len(cand_risk_reasons)
                capped_reason_count = len(capped_risk_reasons)
                replace_with_cap = bool(
                    capped_accepted
                    or (
                        not cand_accepted
                        and (
                            capped_reason_count < current_reason_count
                            or (
                                capped_reason_count == current_reason_count
                                and capped_rel >= current_rel
                            )
                        )
                    )
                )
                view_alpha_cap_profile["decision"] = (
                    "replace_candidate_policy_val" if replace_with_cap else "original_candidate_kept"
                )
                if replace_with_cap:
                    local_alpha_profile = capped_local_alpha_profile
                    cand_policy_val = capped_policy_val
                    cand_best = capped_best
                    cand_selected_alpha = float(capped_selected_alpha)
                    cand_risk_reasons = capped_risk_reasons
                    cand_accepted = bool(capped_accepted)
                    guard_repair_seed_alpha = float(cand_selected_alpha)
                elif not cand_accepted:
                    view_alpha_cap_profile["decision"] = "reject_candidate_after_view_alpha_cap"
            else:
                view_alpha_cap_profile["decision"] = "disabled_no_safe_policy_val_view_alpha_caps"
        cand_fit_summary["face_gain_guard"] = dict(face_gain_guard_profile)
        cand_fit_summary["bin_uncertainty_guard"] = dict(bin_uncertainty_guard_profile)
        cand_fit_summary["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
        cand_fit_summary["view_consistency_confidence"] = {
            **sanitize_view_confidence_profile(view_confidence_profile)
        }
        cand_fit_summary["view_alpha_cap"] = {
            **sanitize_view_alpha_cap_profile(view_alpha_cap_profile)
        }
        print(
            "[policy-candidate] done "
            f"{candidate_label} "
            f"accepted={bool(cand_accepted)} "
            f"alpha={float(cand_selected_alpha):.8g} "
            f"relative_gain={float((cand_best or {}).get('relative_gain', -1.0)):.8g} "
            f"ssim_gain={float((cand_best or {}).get('ssim_gain', 0.0)):.8g} "
            f"l1_gain={float((cand_best or {}).get('image_l1_gain', 0.0)):.8g}",
            flush=True,
        )
        return {
            "candidate_index": int(candidate_index),
            "candidate_count": int(candidate_count),
            "candidate_label": str(candidate_label),
            "fill_mode": str(fill_mode),
            "texture_size": int(texture_size),
            "teacher_distilled_low_rank_texture_rank": int(teacher_low_rank_texture_rank),
            "surface_multiscale_prior_blend": float(surface_multiscale_prior_blend),
            "max_abs_delta_rgb": float(max_abs_delta_rgb_candidate),
            "support_mode": str(support_mode),
            "support_summary": dict(support_summary),
            "atlas": cand_atlas,
            "fit_summary": cand_fit_summary,
            "policy_val": cand_policy_val,
            "best": cand_best,
            "selected_alpha": float(cand_selected_alpha),
            "alpha_candidates": [float(x) for x in alpha_candidates],
            "policy_val_views": list(cand_val_views),
            "local_alpha_profile": dict(local_alpha_profile),
            "face_gain_guard_profile": dict(face_gain_guard_profile),
            "bin_uncertainty_guard_profile": dict(bin_uncertainty_guard_profile),
            "view_confidence_profile": dict(view_confidence_profile),
            "view_alpha_cap_profile": dict(view_alpha_cap_profile),
            "risk_gate_reasons": cand_risk_reasons,
            "accepted": bool(cand_accepted),
        }

    requested_fill_mode = str(args.atlas_empty_bin_fill_mode)
    fill_mode_candidates = (
        ["face_mean", "nearest_observed"] if requested_fill_mode == "auto_policy" else [requested_fill_mode]
    )
    candidate_specs: list[dict[str, Any]] = []
    for support_candidate in support_candidate_sets:
        support_faces = set(int(face) for face in support_candidate["faces"])
        support_faces_digest = face_set_sha1(support_faces)
        for texture_size in texture_size_candidates:
            for teacher_low_rank_texture_rank in teacher_low_rank_texture_rank_candidates:
                for mode in fill_mode_candidates:
                    for surface_multiscale_prior_blend in surface_multiscale_prior_blend_candidates:
                        for max_abs_delta_rgb_candidate in max_abs_delta_rgb_candidates:
                            support_summary = dict(support_candidate["summary"])
                            support_summary.setdefault("support_faces_sha1", support_faces_digest)
                            candidate_specs.append(
                                {
                                    "fill_mode": str(mode),
                                    "texture_size": int(texture_size),
                                    "teacher_distilled_low_rank_texture_rank": int(teacher_low_rank_texture_rank),
                                    "surface_multiscale_prior_blend": float(surface_multiscale_prior_blend),
                                    "max_abs_delta_rgb": float(max_abs_delta_rgb_candidate),
                                    "support_mode": str(support_candidate["support_mode"]),
                                    "support_faces": set(support_faces),
                                    "support_faces_sha1": str(support_faces_digest),
                                    "support_summary": support_summary,
                                    "support_added_faces": int(support_summary.get("added_faces", 0)),
                                    "support_candidate_faces": int(len(support_faces)),
                                }
                            )
    policy_candidate_control["planned_candidate_count_before_pruning"] = int(len(candidate_specs))
    if bool(args.enable_policy_candidate_dominance_pruning):
        seen_specs: dict[tuple[str, int, float, float, str], dict[str, Any]] = {}
        retained_candidate_specs: list[dict[str, Any]] = []
        pruned_spec_rows: list[dict[str, Any]] = []
        original_candidate_count = int(len(candidate_specs))
        for original_index, spec in enumerate(candidate_specs, start=1):
            key = policy_candidate_spec_key(spec)
            if key in seen_specs:
                dominated_by = dict(seen_specs[key])
                row = candidate_spec_audit_row(spec, original_index, original_candidate_count)
                row["reason"] = "duplicate_policy_candidate_spec"
                row["dominated_by_candidate_index"] = int(dominated_by.get("candidate_index", 0))
                row["dominated_by_support_mode"] = str(dominated_by.get("support_mode", ""))
                pruned_spec_rows.append(row)
                continue
            seen_specs[key] = candidate_spec_audit_row(spec, original_index, original_candidate_count)
            retained_candidate_specs.append(spec)
        candidate_specs = retained_candidate_specs
        policy_candidate_control["spec_duplicate_pruned_count"] = int(len(pruned_spec_rows))
        policy_candidate_control["spec_duplicate_pruned"] = pruned_spec_rows[:64]
        if pruned_spec_rows:
            print(
                "[policy-candidate] dominance-pruned "
                f"{len(pruned_spec_rows)} duplicate policy candidate spec(s)",
                flush=True,
            )
    policy_candidate_control["planned_candidate_count_after_pruning"] = int(len(candidate_specs))
    print(
        "[policy-candidate] planned "
        f"{len(candidate_specs)} candidates "
        f"(before_pruning={policy_candidate_control['planned_candidate_count_before_pruning']})",
        flush=True,
    )
    early_stop_disabled_reasons: list[str] = []
    if target_support_candidate_selection_enabled:
        early_stop_disabled_reasons.append("target_support_candidate_selection_requires_full_candidate_ranking")
    if bool(args.enable_policy_val_prior_bin_gain_hybrid):
        early_stop_disabled_reasons.append("prior_bin_gain_hybrid_requires_full_accepted_candidate_pool")
    if str(args.policy_candidate_early_stop_mode) != "none":
        if not early_stop_disabled_reasons:
            policy_candidate_control["early_stop_effective_mode"] = str(args.policy_candidate_early_stop_mode)
        else:
            policy_candidate_control["early_stop_disabled_reasons"] = early_stop_disabled_reasons
    candidate_runs: list[dict[str, Any]] = []
    for candidate_index, spec in enumerate(candidate_specs, start=1):
        candidate = build_policy_candidate(
            str(spec["fill_mode"]),
            int(spec["texture_size"]),
            int(spec.get("teacher_distilled_low_rank_texture_rank", 0)),
            float(spec["surface_multiscale_prior_blend"]),
            float(spec["max_abs_delta_rgb"]),
            str(spec["support_mode"]),
            set(spec["support_faces"]),
            dict(spec["support_summary"]),
            int(candidate_index),
            int(len(candidate_specs)),
        )
        candidate_runs.append(candidate)
        if (
            str(policy_candidate_control.get("early_stop_effective_mode", "none")) == "first_accepted"
            and bool(candidate.get("accepted", False))
        ):
            skipped_rows: list[dict[str, Any]] = []
            for skipped_index, skipped_spec in enumerate(candidate_specs[candidate_index:], start=candidate_index + 1):
                row = candidate_spec_audit_row(skipped_spec, skipped_index, len(candidate_specs))
                row["reason"] = "early_stop_first_accepted"
                row["stopped_by_candidate_index"] = int(candidate_index)
                row["stopped_by_candidate_label"] = str(candidate.get("candidate_label", ""))
                skipped_rows.append(row)
            policy_candidate_control["early_stop_triggered"] = True
            policy_candidate_control["early_stop_skipped_count"] = int(len(skipped_rows))
            policy_candidate_control["early_stop_skipped"] = skipped_rows[:64]
            print(
                "[policy-candidate] early-stop first_accepted after "
                f"{candidate_index}/{len(candidate_specs)} candidate(s); "
                f"skipped={len(skipped_rows)}",
                flush=True,
            )
            break
    policy_candidate_control["executed_candidate_count"] = int(len(candidate_runs))

    target_footprint_bin_certificate_enabled = bool(args.enable_target_footprint_bin_certificate)
    target_support_views = (
        evidence_views(target_evidence)
        if (target_support_candidate_selection_enabled or target_footprint_bin_certificate_enabled)
        else []
    )

    def attach_target_support_profile(candidate: dict[str, Any]) -> None:
        if not target_support_candidate_selection_enabled:
            candidate["target_support_profile"] = {
                "enabled": False,
                "mode": "target_support_candidate_selection",
                "reason": "not_requested",
            }
            return
        if "target_support_profile" in candidate:
            return
        if not bool(candidate.get("accepted", False)) or float(candidate.get("selected_alpha", 0.0)) <= 0.0:
            candidate["target_support_profile"] = {
                "enabled": False,
                "mode": "target_support_candidate_selection",
                "reason": "candidate_not_accepted_or_zero_alpha",
            }
            return
        candidate["target_support_profile"] = evaluate_target_support_profile(
            target_support_views,
            candidate["atlas"],
            alpha=float(candidate.get("selected_alpha", 0.0)),
            min_alpha=float(args.min_alpha),
            max_abs_delta_rgb=float(candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb)),
            min_atlas_bin_count=int(args.min_atlas_bin_count),
            min_atlas_face_samples=int(args.min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(args.atlas_confidence_mode),
            atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(args.min_atlas_confidence),
            local_alpha_profile=dict(candidate.get("local_alpha_profile", {"enabled": False})),
            face_gain_guard_profile=dict(
                candidate.get("face_gain_guard_profile", {"enabled": False, "mode": "policy_val_face_gain_guard"})
            ),
            bin_uncertainty_guard_profile=dict(
                candidate.get(
                    "bin_uncertainty_guard_profile",
                    {"enabled": False, "mode": "policy_val_bin_uncertainty_guard"},
                )
            ),
            parent_edge_apply_profile=dict(parent_edge_apply_profile),
            view_confidence_profile=dict(
                candidate.get(
                    "view_confidence_profile",
                    {"enabled": False, "mode": "policy_val_view_consistency_confidence"},
                )
            ),
        )

    def select_policy_val_payload_for_candidate(
        policy_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], float, list[str], bool]:
        return select_policy_val_payload_by_risk_gate(policy_payload, args)

    def policy_metric_score(
        candidate: dict[str, Any],
    ) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float, float]:
        cand_best = dict(candidate.get("best") or {})
        return (
            float(cand_best.get("relative_gain", -1.0)),
            float(cand_best.get("ssim_gain", 0.0)),
            float(cand_best.get("image_l1_gain", 0.0)),
            float(cand_best.get("lpips_gain", 0.0)),
            float(cand_best.get("cvar20_view_relative_gain", -1.0)),
            float(cand_best.get("min_view_relative_gain", -1.0)),
            float(cand_best.get("image_l1_cvar20_view_gain", -1.0)),
            float(cand_best.get("image_l1_min_view_gain", -1.0)),
            float(cand_best.get("lpips_cvar20_view_gain", -1.0)),
            float(cand_best.get("lpips_min_view_gain", -1.0)),
            float((candidate.get("support_summary") or {}).get("added_faces", 0)),
            -float(candidate.get("texture_size", 0)),
            -float(candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb)),
        )

    def target_support_energy_score(profile: dict[str, Any]) -> float:
        changed_fraction = float(
            profile.get("png_quantized_changed_fraction", profile.get("changed_fraction", 0.0)) or 0.0
        )
        active_mean_abs_delta = float(profile.get("active_mean_abs_delta", 0.0) or 0.0)
        mean_abs_delta = float(profile.get("mean_abs_delta", 0.0) or 0.0)
        if active_mean_abs_delta > 0.0:
            return float(changed_fraction * active_mean_abs_delta)
        return float(mean_abs_delta)

    def target_support_cvar20_energy_score(profile: dict[str, Any]) -> float:
        active_mean_abs_delta = float(profile.get("active_mean_abs_delta", 0.0) or 0.0)
        mean_abs_delta = float(profile.get("mean_abs_delta", 0.0) or 0.0)
        scale = active_mean_abs_delta if active_mean_abs_delta > 0.0 else mean_abs_delta
        changed_fraction = float(
            profile.get(
                "cvar20_view_png_quantized_changed_fraction",
                profile.get("cvar20_view_changed_fraction", 0.0),
            )
            or 0.0
        )
        return float(changed_fraction * scale)

    def target_support_min_view_energy_score(profile: dict[str, Any]) -> float:
        active_mean_abs_delta = float(profile.get("active_mean_abs_delta", 0.0) or 0.0)
        mean_abs_delta = float(profile.get("mean_abs_delta", 0.0) or 0.0)
        scale = active_mean_abs_delta if active_mean_abs_delta > 0.0 else mean_abs_delta
        changed_fraction = float(
            profile.get(
                "min_view_png_quantized_changed_fraction",
                profile.get("min_view_changed_fraction", 0.0),
            )
            or 0.0
        )
        return float(changed_fraction * scale)

    def target_support_score(candidate: dict[str, Any]) -> tuple[float, ...]:
        profile = dict(candidate.get("target_support_profile") or {})
        if not bool(profile.get("enabled", False)):
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if bool(
                args.enable_target_visible_energy_score
            ) else (0.0, 0.0, 0.0, 0.0, 0.0)
        changed_fraction = float(profile.get("changed_fraction", 0.0) or 0.0)
        png_changed_fraction = float(profile.get("png_quantized_changed_fraction", changed_fraction) or 0.0)
        cvar20_changed = float(profile.get("cvar20_view_changed_fraction", 0.0) or 0.0)
        cvar20_png_changed = float(
            profile.get("cvar20_view_png_quantized_changed_fraction", cvar20_changed) or 0.0
        )
        min_changed = float(profile.get("min_view_changed_fraction", 0.0) or 0.0)
        min_png_changed = float(profile.get("min_view_png_quantized_changed_fraction", min_changed) or 0.0)
        valid_fraction = float(profile.get("valid_fraction", 0.0) or 0.0)
        mean_abs_delta = float(profile.get("mean_abs_delta", 0.0) or 0.0)
        active_mean_abs_delta = float(profile.get("active_mean_abs_delta", 0.0) or 0.0)
        if bool(args.enable_target_visible_energy_score):
            return (
                target_support_energy_score(profile),
                target_support_cvar20_energy_score(profile),
                target_support_min_view_energy_score(profile),
                png_changed_fraction,
                cvar20_png_changed,
                min_png_changed,
                changed_fraction,
                cvar20_changed,
                min_changed,
                valid_fraction,
                mean_abs_delta,
            )
        return (changed_fraction, cvar20_changed, min_changed, valid_fraction, mean_abs_delta)

    def target_support_score_dict(candidate: dict[str, Any]) -> dict[str, float]:
        profile = dict(candidate.get("target_support_profile") or {})
        return {
            "energy_score": target_support_energy_score(profile),
            "cvar20_view_energy_score": target_support_cvar20_energy_score(profile),
            "min_view_energy_score": target_support_min_view_energy_score(profile),
            "changed_fraction": float(profile.get("changed_fraction", 0.0) or 0.0),
            "png_quantized_changed_fraction": float(
                profile.get("png_quantized_changed_fraction", profile.get("changed_fraction", 0.0)) or 0.0
            ),
            "cvar20_view_changed_fraction": float(
                profile.get("cvar20_view_changed_fraction", 0.0) or 0.0
            ),
            "cvar20_view_png_quantized_changed_fraction": float(
                profile.get(
                    "cvar20_view_png_quantized_changed_fraction",
                    profile.get("cvar20_view_changed_fraction", 0.0),
                )
                or 0.0
            ),
            "min_view_changed_fraction": float(profile.get("min_view_changed_fraction", 0.0) or 0.0),
            "min_view_png_quantized_changed_fraction": float(
                profile.get(
                    "min_view_png_quantized_changed_fraction",
                    profile.get("min_view_changed_fraction", 0.0),
                )
                or 0.0
            ),
            "valid_fraction": float(profile.get("valid_fraction", 0.0) or 0.0),
            "mean_abs_delta": float(profile.get("mean_abs_delta", 0.0) or 0.0),
            "active_mean_abs_delta": float(profile.get("active_mean_abs_delta", 0.0) or 0.0),
        }

    def target_support_certificate(candidate: dict[str, Any]) -> dict[str, Any]:
        profile = dict(candidate.get("target_support_profile") or {})
        min_changed = float(args.min_target_changed_fraction)
        changed_threshold = min_changed if min_changed > 0.0 else 1.0e-12
        changed = float(profile.get("changed_fraction", 0.0) or 0.0)
        valid = float(profile.get("valid_fraction", 0.0) or 0.0)
        enabled = bool(target_support_candidate_selection_enabled and profile.get("enabled", False))
        reasons: list[str] = []
        if not target_support_candidate_selection_enabled:
            reasons.append("target_support_candidate_selection_not_enabled")
        elif not enabled:
            reasons.append(str(profile.get("reason", "target_support_profile_disabled")))
        if changed < changed_threshold:
            reasons.append(
                f"changed_fraction {changed:.8f} < threshold {changed_threshold:.8f}"
            )
        if valid <= 0.0:
            reasons.append("valid_fraction <= 0")
        return {
            "enabled": bool(enabled),
            "passed": bool(enabled and not reasons),
            "changed_fraction": float(changed),
            "changed_fraction_threshold": float(changed_threshold),
            "valid_fraction": float(valid),
            "reasons": reasons,
        }

    def policy_candidate_score(candidate: dict[str, Any]) -> tuple[float, ...]:
        if target_support_candidate_selection_enabled:
            return target_support_score(candidate) + policy_metric_score(candidate)
        return policy_metric_score(candidate)

    policy_auto_enabled = (
        requested_fill_mode == "auto_policy"
        or len(texture_size_candidates) > 1
        or len(teacher_low_rank_texture_rank_candidates) > 1
        or len(support_candidate_sets) > 1
        or len(surface_multiscale_prior_blend_candidates) > 1
        or len(max_abs_delta_rgb_candidates) > 1
    )
    if policy_auto_enabled:
        accepted_candidates = [candidate for candidate in candidate_runs if bool(candidate.get("accepted", False))]
        if target_support_candidate_selection_enabled:
            if not target_support_views:
                raise FileNotFoundError(f"no target npz views found in {target_evidence}")
            for candidate in accepted_candidates:
                attach_target_support_profile(candidate)
            for candidate in candidate_runs:
                if not bool(candidate.get("accepted", False)):
                    attach_target_support_profile(candidate)
        selectable = accepted_candidates if accepted_candidates else candidate_runs
        baseline_candidate = next(
            (
                candidate
                for candidate in accepted_candidates
                if abs(float(candidate.get("surface_multiscale_prior_blend", 0.0))) <= 1.0e-12
                and abs(
                    float(candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb))
                    - float(args.max_abs_delta_rgb)
                )
                <= 1.0e-12
            ),
            None,
        )
        if baseline_candidate is not None:
            zero_blend_candidates = [
                candidate
                for candidate in accepted_candidates
                if abs(float(candidate.get("surface_multiscale_prior_blend", 0.0))) <= 1.0e-12
                and abs(
                    float(candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb))
                    - float(args.max_abs_delta_rgb)
                )
                <= 1.0e-12
            ]
            baseline_candidate = max(zero_blend_candidates, key=policy_candidate_score)
        if baseline_candidate is None:
            zero_blend_candidates = [
                candidate
                for candidate in accepted_candidates
                if abs(float(candidate.get("surface_multiscale_prior_blend", 0.0))) <= 1.0e-12
            ]
            if zero_blend_candidates:
                baseline_candidate = max(zero_blend_candidates, key=policy_candidate_score)
        if baseline_candidate is None:
            baseline_candidate = next(
                (
                    candidate
                    for candidate in accepted_candidates
                    if str(candidate.get("fill_mode", "")) == "face_mean"
                    and int(candidate.get("texture_size", 0)) == int(args.texture_size)
                    and str(candidate.get("support_mode", "")) == "base_carrier"
                ),
                None,
            )
        if baseline_candidate is None:
            baseline_candidate = next(
                (
                    candidate
                    for candidate in accepted_candidates
                    if str(candidate.get("fill_mode", "")) == "face_mean"
                ),
                None,
            )
        if bool(args.enable_policy_val_prior_bin_gain_hybrid) and baseline_candidate is not None:
            hybrid_candidates: list[dict[str, Any]] = []
            base_texture_size = int(baseline_candidate.get("texture_size", 0))
            base_support_mode = str(baseline_candidate.get("support_mode", ""))
            base_support_added = int((baseline_candidate.get("support_summary") or {}).get("added_faces", 0))
            base_cap = float(baseline_candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb))
            base_fill_mode = str(baseline_candidate.get("fill_mode", ""))
            for source_candidate in list(accepted_candidates):
                if source_candidate is baseline_candidate:
                    continue
                if float(source_candidate.get("surface_multiscale_prior_blend", 0.0)) <= 1.0e-12:
                    continue
                if int(source_candidate.get("texture_size", 0)) != base_texture_size:
                    continue
                if str(source_candidate.get("support_mode", "")) != base_support_mode:
                    continue
                if int((source_candidate.get("support_summary") or {}).get("added_faces", 0)) != base_support_added:
                    continue
                if abs(float(source_candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb)) - base_cap) > 1.0e-12:
                    continue
                if str(source_candidate.get("fill_mode", "")) != base_fill_mode:
                    continue
                policy_val_views = list(source_candidate.get("policy_val_views") or baseline_candidate.get("policy_val_views") or [])
                alpha_candidates = [
                    float(x)
                    for x in (
                        source_candidate.get("alpha_candidates")
                        or baseline_candidate.get("alpha_candidates")
                        or parse_alpha_grid(args.alpha_grid)
                    )
                ]
                hybrid_label = (
                    "hybrid "
                    f"baseline={baseline_candidate.get('candidate_label', 'unknown')} "
                    f"source={source_candidate.get('candidate_label', 'unknown')}"
                )
                print(f"[policy-candidate] start {hybrid_label}", flush=True)
                hybrid_atlas, hybrid_profile = build_policy_val_prior_bin_gain_hybrid_atlas(
                    policy_val_views,
                    baseline_candidate["atlas"],
                    source_candidate["atlas"],
                    residual_rgb_key=str(args.residual_rgb_key),
                    residual_l1_key=str(args.residual_l1_key),
                    baseline_alpha=float(baseline_candidate.get("selected_alpha", 0.0)),
                    prior_alpha=float(source_candidate.get("selected_alpha", 0.0)),
                    min_l1=float(args.min_l1),
                    min_alpha=float(args.min_alpha),
                    max_abs_delta_rgb=base_cap,
                    max_samples_per_view=int(args.max_samples_per_view),
                    min_atlas_bin_count=int(args.min_atlas_bin_count),
                    min_atlas_face_samples=int(args.min_atlas_face_samples),
                    max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                    min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                    atlas_confidence_mode=str(args.atlas_confidence_mode),
                    atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                    atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                    atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                    atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                    atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                    min_atlas_confidence=float(args.min_atlas_confidence),
                    baseline_local_alpha_profile=dict(
                        baseline_candidate.get("local_alpha_profile", {"enabled": False})
                    ),
                    prior_local_alpha_profile=dict(
                        source_candidate.get("local_alpha_profile", {"enabled": False})
                    ),
                    baseline_face_gain_guard_profile=dict(
                        baseline_candidate.get(
                            "face_gain_guard_profile",
                            {"enabled": False, "mode": "policy_val_face_gain_guard"},
                        )
                    ),
                    prior_face_gain_guard_profile=dict(
                        source_candidate.get(
                            "face_gain_guard_profile",
                            {"enabled": False, "mode": "policy_val_face_gain_guard"},
                        )
                    ),
                    baseline_bin_uncertainty_guard_profile=dict(
                        baseline_candidate.get(
                            "bin_uncertainty_guard_profile",
                            {"enabled": False, "mode": "policy_val_bin_uncertainty_guard"},
                        )
                    ),
                    prior_bin_uncertainty_guard_profile=dict(
                        source_candidate.get(
                            "bin_uncertainty_guard_profile",
                            {"enabled": False, "mode": "policy_val_bin_uncertainty_guard"},
                        )
                    ),
                    min_bin_samples=int(args.prior_bin_gain_hybrid_min_bin_samples),
                    min_views=int(args.prior_bin_gain_hybrid_min_views),
                    min_abs_gain=float(args.prior_bin_gain_hybrid_min_abs_gain),
                    min_relative_gain=float(args.prior_bin_gain_hybrid_min_relative_gain),
                    min_positive_view_fraction=float(args.prior_bin_gain_hybrid_min_positive_view_fraction),
                    enable_l1_proxy_gate=bool(args.enable_prior_bin_gain_hybrid_l1_proxy_gate),
                    min_l1_abs_gain=float(args.prior_bin_gain_hybrid_min_l1_abs_gain),
                    min_l1_relative_gain=float(args.prior_bin_gain_hybrid_min_l1_relative_gain),
                    min_l1_positive_view_fraction=float(
                        args.prior_bin_gain_hybrid_min_l1_positive_view_fraction
                    ),
                    min_l1_min_view_gain=float(args.prior_bin_gain_hybrid_min_l1_min_view_gain),
                    min_l1_cvar20_view_gain=float(args.prior_bin_gain_hybrid_min_l1_cvar20_view_gain),
                    max_profile_bins=int(args.prior_bin_gain_hybrid_max_profile_bins),
                    target_footprint_views=list(target_support_views),
                    enable_target_footprint_bin_certificate=bool(
                        args.enable_target_footprint_bin_certificate
                    ),
                    target_footprint_min_bin_pixels=int(args.target_footprint_min_bin_pixels),
                    target_footprint_min_views=int(args.target_footprint_min_views),
                    target_footprint_min_view_fraction=float(args.target_footprint_min_view_fraction),
                    target_footprint_max_views=int(args.target_footprint_max_views),
                    enable_target_footprint_tail_risk_certificate=bool(
                        args.enable_target_footprint_tail_risk_certificate
                    ),
                    target_footprint_tail_risk_only_target_bins=not bool(
                        args.target_footprint_tail_risk_all_bins
                    ),
                    target_footprint_tail_risk_min_positive_view_fraction=float(
                        args.target_footprint_tail_risk_min_positive_view_fraction
                    ),
                    target_footprint_tail_risk_min_min_view_gain=float(
                        args.target_footprint_tail_risk_min_min_view_gain
                    ),
                    target_footprint_tail_risk_min_cvar20_view_gain=float(
                        args.target_footprint_tail_risk_min_cvar20_view_gain
                    ),
                    source_mixture_enabled=bool(args.enable_policy_val_source_mixture),
                    source_mixture_ridge_mode=str(args.source_mixture_ridge_mode),
                    source_mixture_ridge=float(args.source_mixture_ridge),
                    source_mixture_min_weight=float(args.source_mixture_min_weight),
                )
                if not bool(hybrid_profile.get("enabled", False)):
                    continue
                hybrid_local_alpha_profile = hybrid_bin_source_local_alpha_profile(
                    dict(baseline_candidate.get("local_alpha_profile", {"enabled": False})),
                    dict(source_candidate.get("local_alpha_profile", {"enabled": False})),
                    dict(hybrid_profile),
                )
                disabled_face_gain_guard_profile = {
                    "enabled": False,
                    "mode": "policy_val_face_gain_guard",
                }
                disabled_bin_uncertainty_guard_profile = {
                    "enabled": False,
                    "mode": "policy_val_bin_uncertainty_guard",
                }
                hybrid_face_gain_guard_profile = compatible_hybrid_guard_profile(
                    dict(
                        baseline_candidate.get(
                            "face_gain_guard_profile",
                            disabled_face_gain_guard_profile,
                        )
                    ),
                    dict(
                        source_candidate.get(
                            "face_gain_guard_profile",
                            disabled_face_gain_guard_profile,
                        )
                    ),
                    disabled_face_gain_guard_profile,
                )
                hybrid_bin_uncertainty_guard_profile = compatible_hybrid_guard_profile(
                    source_candidate.get(
                        "bin_uncertainty_guard_profile",
                        disabled_bin_uncertainty_guard_profile,
                    ),
                    baseline_candidate.get(
                        "bin_uncertainty_guard_profile",
                        disabled_bin_uncertainty_guard_profile,
                    ),
                    disabled_bin_uncertainty_guard_profile,
                )
                if (
                    hybrid_face_gain_guard_profile is None
                    or hybrid_bin_uncertainty_guard_profile is None
                ):
                    continue
                hybrid_policy_val = evaluate_policy_val(
                    policy_val_views,
                    hybrid_atlas,
                    residual_rgb_key=str(args.residual_rgb_key),
                    residual_l1_key=str(args.residual_l1_key),
                    alpha_grid=alpha_candidates,
                    min_l1=float(args.min_l1),
                    min_alpha=float(args.min_alpha),
                    max_abs_delta_rgb=base_cap,
                    max_samples_per_view=int(args.max_samples_per_view),
                    min_atlas_bin_count=int(args.min_atlas_bin_count),
                    min_atlas_face_samples=int(args.min_atlas_face_samples),
                    max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                    min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                    atlas_confidence_mode=str(args.atlas_confidence_mode),
                    atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
                    atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
                    atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
                    atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
                    atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
                    min_atlas_confidence=float(args.min_atlas_confidence),
                    enable_policy_val_image_ssim=bool(args.enable_policy_val_image_ssim_gate),
                    policy_val_ssim_max_size=int(args.policy_val_ssim_max_size),
                    enable_policy_val_image_l1=bool(args.enable_policy_val_image_l1_gate),
                    policy_val_l1_max_size=int(args.policy_val_l1_max_size),
                    enable_policy_val_image_lpips=bool(args.enable_policy_val_image_lpips_gate),
                    policy_val_lpips_max_size=int(args.policy_val_lpips_max_size),
                    local_alpha_profile=hybrid_local_alpha_profile,
                    face_gain_guard_profile=hybrid_face_gain_guard_profile,
                    bin_uncertainty_guard_profile=hybrid_bin_uncertainty_guard_profile,
                    parent_edge_apply_profile=parent_edge_apply_profile,
                )
                hybrid_policy_val["alpha_calibration"] = dict(
                    (source_candidate.get("policy_val") or {}).get("alpha_calibration", {})
                )
                hybrid_policy_val["local_alpha_calibration"] = dict(hybrid_local_alpha_profile)
                hybrid_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
                hybrid_policy_val["ssim_alpha_refinement"] = dict(
                    (source_candidate.get("policy_val") or {}).get("ssim_alpha_refinement", {})
                )
                hybrid_policy_val["alpha_midpoint_refinement"] = dict(
                    (source_candidate.get("policy_val") or {}).get("alpha_midpoint_refinement", {})
                )
                (
                    hybrid_policy_val,
                    hybrid_best,
                    hybrid_selected_alpha,
                    hybrid_risk_reasons,
                    hybrid_accepted,
                ) = select_policy_val_payload_for_candidate(hybrid_policy_val)
                hybrid_fit_summary = dict(source_candidate.get("fit_summary", {}))
                hybrid_fit_summary["candidate_label"] = str(hybrid_label)
                hybrid_fit_summary["ssim_alpha_refinement"] = dict(
                    (source_candidate.get("fit_summary") or {}).get("ssim_alpha_refinement", {})
                )
                hybrid_fit_summary["alpha_midpoint_refinement"] = dict(
                    (source_candidate.get("fit_summary") or {}).get("alpha_midpoint_refinement", {})
                )
                hybrid_fit_summary["policy_val_prior_bin_gain_hybrid"] = dict(hybrid_profile)
                hybrid_fit_summary["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
                hybrid_fit_summary["policy_val_prior_bin_gain_hybrid_source"] = {
                    "baseline_surface_multiscale_prior_blend": float(
                        baseline_candidate.get("surface_multiscale_prior_blend", 0.0)
                    ),
                    "source_surface_multiscale_prior_blend": float(
                        source_candidate.get("surface_multiscale_prior_blend", 0.0)
                    ),
                    "baseline_best": dict(baseline_candidate.get("best") or {}),
                    "source_best": dict(source_candidate.get("best") or {}),
                }
                hybrid_candidate = {
                    "candidate_index": int(source_candidate.get("candidate_index", 0)),
                    "candidate_count": int(source_candidate.get("candidate_count", 0)),
                    "candidate_label": str(hybrid_label),
                    "fill_mode": base_fill_mode,
                    "texture_size": int(base_texture_size),
                    "surface_multiscale_prior_blend": float(
                        source_candidate.get("surface_multiscale_prior_blend", 0.0)
                    ),
                    "max_abs_delta_rgb": float(base_cap),
                    "support_mode": base_support_mode,
                    "support_summary": dict(baseline_candidate.get("support_summary", {})),
                    "atlas": hybrid_atlas,
                    "fit_summary": hybrid_fit_summary,
                    "policy_val": hybrid_policy_val,
                    "best": hybrid_best,
                    "selected_alpha": float(hybrid_selected_alpha),
                    "alpha_candidates": [float(x) for x in alpha_candidates],
                    "policy_val_views": list(policy_val_views),
                    "local_alpha_profile": dict(hybrid_local_alpha_profile),
                    "face_gain_guard_profile": dict(hybrid_face_gain_guard_profile),
                    "bin_uncertainty_guard_profile": dict(hybrid_bin_uncertainty_guard_profile),
                    "risk_gate_reasons": list(hybrid_risk_reasons),
                    "accepted": bool(hybrid_accepted),
                    "policy_val_prior_bin_gain_hybrid": True,
                }
                print(
                    "[policy-candidate] done "
                    f"{hybrid_label} "
                    f"accepted={bool(hybrid_accepted)} "
                    f"alpha={float(hybrid_selected_alpha):.8g} "
                    f"relative_gain={float((hybrid_best or {}).get('relative_gain', -1.0)):.8g} "
                    f"ssim_gain={float((hybrid_best or {}).get('ssim_gain', 0.0)):.8g} "
                    f"l1_gain={float((hybrid_best or {}).get('image_l1_gain', 0.0)):.8g}",
                    flush=True,
                )
                hybrid_candidates.append(hybrid_candidate)
            for hybrid_candidate in hybrid_candidates:
                candidate_runs.append(hybrid_candidate)
                if bool(hybrid_candidate.get("accepted", False)):
                    accepted_candidates.append(hybrid_candidate)
                if target_support_candidate_selection_enabled:
                    attach_target_support_profile(hybrid_candidate)
        if baseline_candidate is not None:
            # Alternative fill/capacity candidates are allowed to replace the
            # calibrated face-mean atlas only when train policy-val evidence is
            # non-regressive on both mean and tail axes. This keeps the decision
            # train-only while preventing a tiny MSE gain from overriding the
            # SSIM/tail guard.
            eps = 1.0e-12
            base_best = dict(baseline_candidate.get("best") or {})
            guarded_candidates = [baseline_candidate]
            for candidate in accepted_candidates:
                if candidate is baseline_candidate:
                    continue
                cand_best = dict(candidate.get("best") or {})
                if float(cand_best.get("relative_gain", -1.0)) + eps < float(
                    base_best.get("relative_gain", -1.0)
                ):
                    continue
                if float(cand_best.get("ssim_gain", 0.0)) + eps < float(base_best.get("ssim_gain", 0.0)):
                    continue
                if float(cand_best.get("image_l1_gain", 0.0)) + eps < float(
                    base_best.get("image_l1_gain", 0.0)
                ):
                    continue
                if float(cand_best.get("lpips_gain", 0.0)) + eps < float(base_best.get("lpips_gain", 0.0)):
                    continue
                if float(cand_best.get("cvar20_view_relative_gain", -1.0)) + eps < float(
                    base_best.get("cvar20_view_relative_gain", -1.0)
                ):
                    continue
                if float(cand_best.get("min_view_relative_gain", -1.0)) + eps < float(
                    base_best.get("min_view_relative_gain", -1.0)
                ):
                    continue
                if float(cand_best.get("image_l1_cvar20_view_gain", -1.0)) + eps < float(
                    base_best.get("image_l1_cvar20_view_gain", -1.0)
                ):
                    continue
                if float(cand_best.get("image_l1_min_view_gain", -1.0)) + eps < float(
                    base_best.get("image_l1_min_view_gain", -1.0)
                ):
                    continue
                if float(cand_best.get("lpips_cvar20_view_gain", -1.0)) + eps < float(
                    base_best.get("lpips_cvar20_view_gain", -1.0)
                ):
                    continue
                if float(cand_best.get("lpips_min_view_gain", -1.0)) + eps < float(
                    base_best.get("lpips_min_view_gain", -1.0)
                ):
                    continue
                guarded_candidates.append(candidate)
            selectable = guarded_candidates
        selected_candidate = max(selectable, key=policy_candidate_score)
        ranked_candidates = sorted(selectable, key=policy_candidate_score, reverse=True)
        ranked_all_candidates = sorted(candidate_runs, key=policy_candidate_score, reverse=True)
        target_support_enabled_candidates = [
            candidate
            for candidate in selectable
            if bool((candidate.get("target_support_profile") or {}).get("enabled", False))
        ]
        best_target_support_candidate = (
            max(target_support_enabled_candidates, key=target_support_score)
            if target_support_enabled_candidates
            else None
        )
        global_target_support_enabled_candidates = [
            candidate
            for candidate in candidate_runs
            if bool((candidate.get("target_support_profile") or {}).get("enabled", False))
        ]
        global_best_target_support_candidate = (
            max(global_target_support_enabled_candidates, key=target_support_score)
            if global_target_support_enabled_candidates
            else None
        )
        fill_mode_selection = {
            "mode": "auto_policy",
            "selected_fill_mode": str(selected_candidate.get("fill_mode", "")),
            "selected_texture_size": int(selected_candidate.get("texture_size", 0)),
            "selected_teacher_distilled_low_rank_texture_rank": int(
                selected_candidate.get("teacher_distilled_low_rank_texture_rank", 0)
            ),
            "selected_surface_multiscale_prior_blend": float(
                selected_candidate.get("surface_multiscale_prior_blend", 0.0)
            ),
            "selected_max_abs_delta_rgb": float(
                selected_candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb)
            ),
            "selected_support_mode": str(selected_candidate.get("support_mode", "")),
            "accepted_candidate_count": int(len(accepted_candidates)),
            "guard": (
                "zero_blend_or_base_face_mean_nonregressive_relative_ssim_l1_cvar_min_view"
                + ("_target_support_lexicographic" if target_support_candidate_selection_enabled else "")
            ),
            "score_order_scope": "eligible_after_nonreg_guard",
            "selectable_candidate_count": int(len(selectable)),
            "all_candidate_count": int(len(candidate_runs)),
            "texture_size_candidates": [int(x) for x in texture_size_candidates],
            "teacher_distilled_low_rank_texture_rank_candidates": [
                int(x) for x in teacher_low_rank_texture_rank_candidates
            ],
            "surface_multiscale_prior_blend_candidates": [
                float(x) for x in surface_multiscale_prior_blend_candidates
            ],
            "max_abs_delta_rgb_candidates": [
                float(x) for x in max_abs_delta_rgb_candidates
            ],
            "score_order": [
                {
                    "fill_mode": str(candidate.get("fill_mode", "")),
                    "candidate_label": str(candidate.get("candidate_label", "")),
                    "texture_size": int(candidate.get("texture_size", 0)),
                    "teacher_distilled_low_rank_texture_rank": int(
                        candidate.get("teacher_distilled_low_rank_texture_rank", 0)
                    ),
                    "surface_multiscale_prior_blend": float(
                        candidate.get("surface_multiscale_prior_blend", 0.0)
                    ),
                    "max_abs_delta_rgb": float(
                        candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb)
                    ),
                    "support_mode": str(candidate.get("support_mode", "")),
                    "support_added_faces": int((candidate.get("support_summary") or {}).get("added_faces", 0)),
                    "policy_val_prior_bin_gain_hybrid": bool(
                        candidate.get("policy_val_prior_bin_gain_hybrid", False)
                    ),
                    "prior_bin_gain_hybrid_allowed_bins": int(
                        (
                            (candidate.get("fit_summary") or {})
                            .get("policy_val_prior_bin_gain_hybrid", {})
                            .get("allowed_bin_count", 0)
                        )
                        or 0
                    ),
                    "accepted": bool(candidate.get("accepted", False)),
                    "selected_alpha": float(candidate.get("selected_alpha", 0.0)),
                    "relative_gain": float((candidate.get("best") or {}).get("relative_gain", -1.0)),
                    "ssim_gain": float((candidate.get("best") or {}).get("ssim_gain", 0.0)),
                    "image_l1_gain": float((candidate.get("best") or {}).get("image_l1_gain", 0.0)),
                    "image_l1_positive_view_fraction": float(
                        (candidate.get("best") or {}).get("image_l1_positive_view_fraction", 0.0)
                    ),
                    "image_l1_min_view_gain": float(
                        (candidate.get("best") or {}).get("image_l1_min_view_gain", 0.0)
                    ),
                    "image_l1_cvar20_view_gain": float(
                        (candidate.get("best") or {}).get("image_l1_cvar20_view_gain", 0.0)
                    ),
                    "cvar20_view_relative_gain": float(
                        (candidate.get("best") or {}).get("cvar20_view_relative_gain", -1.0)
                    ),
                    "min_view_relative_gain": float(
                        (candidate.get("best") or {}).get("min_view_relative_gain", -1.0)
                    ),
                    "target_support_enabled": bool(
                        (candidate.get("target_support_profile") or {}).get("enabled", False)
                    ),
                    "target_changed_fraction": float(
                        (candidate.get("target_support_profile") or {}).get("changed_fraction", 0.0)
                    ),
                    "target_cvar20_view_changed_fraction": float(
                        (candidate.get("target_support_profile") or {}).get(
                            "cvar20_view_changed_fraction", 0.0
                        )
                    ),
                    "target_min_view_changed_fraction": float(
                        (candidate.get("target_support_profile") or {}).get(
                            "min_view_changed_fraction", 0.0
                        )
                    ),
                    "target_valid_fraction": float(
                        (candidate.get("target_support_profile") or {}).get("valid_fraction", 0.0)
                    ),
                    "target_support_score": target_support_score_dict(candidate),
                    "target_support_certificate": target_support_certificate(candidate),
                    "policy_candidate_score_tuple": [
                        float(x) for x in policy_candidate_score(candidate)
                    ],
                }
                for candidate in ranked_candidates
            ],
            "global_score_order_top": [
                {
                    "fill_mode": str(candidate.get("fill_mode", "")),
                    "candidate_label": str(candidate.get("candidate_label", "")),
                    "texture_size": int(candidate.get("texture_size", 0)),
                    "teacher_distilled_low_rank_texture_rank": int(
                        candidate.get("teacher_distilled_low_rank_texture_rank", 0)
                    ),
                    "surface_multiscale_prior_blend": float(
                        candidate.get("surface_multiscale_prior_blend", 0.0)
                    ),
                    "max_abs_delta_rgb": float(
                        candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb)
                    ),
                    "support_mode": str(candidate.get("support_mode", "")),
                    "policy_val_prior_bin_gain_hybrid": bool(
                        candidate.get("policy_val_prior_bin_gain_hybrid", False)
                    ),
                    "accepted": bool(candidate.get("accepted", False)),
                    "selected_alpha": float(candidate.get("selected_alpha", 0.0)),
                    "relative_gain": float((candidate.get("best") or {}).get("relative_gain", -1.0)),
                    "ssim_gain": float((candidate.get("best") or {}).get("ssim_gain", 0.0)),
                    "image_l1_gain": float((candidate.get("best") or {}).get("image_l1_gain", 0.0)),
                    "target_support_score": target_support_score_dict(candidate),
                    "target_support_certificate": target_support_certificate(candidate),
                }
                for candidate in ranked_all_candidates[:8]
            ],
        }
    else:
        selected_candidate = candidate_runs[0]
        if target_support_candidate_selection_enabled:
            if not target_support_views:
                raise FileNotFoundError(f"no target npz views found in {target_evidence}")
            attach_target_support_profile(selected_candidate)
        fill_mode_selection = {
            "mode": "fixed",
            "selected_fill_mode": str(selected_candidate.get("fill_mode", "")),
            "selected_texture_size": int(selected_candidate.get("texture_size", 0)),
            "selected_teacher_distilled_low_rank_texture_rank": int(
                selected_candidate.get("teacher_distilled_low_rank_texture_rank", 0)
            ),
            "selected_surface_multiscale_prior_blend": float(
                selected_candidate.get("surface_multiscale_prior_blend", 0.0)
            ),
            "selected_max_abs_delta_rgb": float(
                selected_candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb)
            ),
            "selected_support_mode": str(selected_candidate.get("support_mode", "")),
            "accepted_candidate_count": int(bool(selected_candidate.get("accepted", False))),
            "teacher_distilled_low_rank_texture_rank_candidates": [
                int(x) for x in teacher_low_rank_texture_rank_candidates
            ],
            "surface_multiscale_prior_blend_candidates": [
                float(x) for x in surface_multiscale_prior_blend_candidates
            ],
            "max_abs_delta_rgb_candidates": [
                float(x) for x in max_abs_delta_rgb_candidates
            ],
        }

    atlas = selected_candidate["atlas"]
    fit_summary = dict(selected_candidate["fit_summary"])
    policy_val = dict(selected_candidate["policy_val"])
    target_support_profile = dict(
        selected_candidate.get(
            "target_support_profile",
            {
                "enabled": False,
                "mode": "target_support_candidate_selection",
                "reason": "not_requested",
            },
        )
    )
    local_alpha_profile = dict(selected_candidate.get("local_alpha_profile", {"enabled": False}))
    face_gain_guard_profile = dict(
        selected_candidate.get(
            "face_gain_guard_profile",
            {"enabled": False, "mode": "policy_val_face_gain_guard"},
        )
    )
    bin_uncertainty_guard_profile = dict(
        selected_candidate.get(
            "bin_uncertainty_guard_profile",
            {"enabled": False, "mode": "policy_val_bin_uncertainty_guard"},
        )
    )
    view_confidence_profile = dict(
        selected_candidate.get(
            "view_confidence_profile",
            {"enabled": False, "mode": "policy_val_view_consistency_confidence"},
        )
    )
    view_alpha_cap_profile = dict(
        selected_candidate.get(
            "view_alpha_cap_profile",
            (local_alpha_profile or {}).get(
                "view_alpha_cap_profile",
                {"enabled": False, "mode": "policy_val_view_alpha_cap"},
            ),
        )
    )
    best = dict(selected_candidate["best"])
    selected_alpha = float(selected_candidate["selected_alpha"])
    risk_gate_reasons = list(selected_candidate["risk_gate_reasons"])
    accepted = bool(selected_candidate["accepted"])
    policy_candidate_control["selected_candidate_index"] = int(selected_candidate.get("candidate_index", 0))
    policy_candidate_control["selected_candidate_label"] = str(selected_candidate.get("candidate_label", ""))
    policy_candidate_control["selected_candidate_was_executed"] = True
    policy_candidate_control["executed_candidate_labels"] = [
        str(candidate.get("candidate_label", "")) for candidate in candidate_runs[:64]
    ]
    policy_candidate_control["executed_candidate_count_including_hybrids"] = int(len(candidate_runs))
    policy_val["fill_mode_selection"] = fill_mode_selection
    fit_summary["policy_candidate_control"] = dict(policy_candidate_control)
    fit_summary["candidate_plan"] = [
        {
            "candidate_index": int(idx),
            "candidate_count": int(len(candidate_specs)),
            "fill_mode": str(spec.get("fill_mode", "")),
            "texture_size": int(spec.get("texture_size", 0)),
            "surface_multiscale_prior_blend": float(spec.get("surface_multiscale_prior_blend", 0.0)),
            "max_abs_delta_rgb": float(spec.get("max_abs_delta_rgb", args.max_abs_delta_rgb)),
            "support_mode": str(spec.get("support_mode", "")),
            "support_added_faces": int(spec.get("support_added_faces", 0)),
            "support_candidate_faces": int(spec.get("support_candidate_faces", 0)),
            "support_faces_sha1": str(spec.get("support_faces_sha1", "")),
        }
        for idx, spec in enumerate(candidate_specs, start=1)
    ]
    fit_summary["selected_candidate_label"] = str(selected_candidate.get("candidate_label", ""))
    fit_summary["selected_candidate_index"] = int(selected_candidate.get("candidate_index", 0))
    fit_summary["candidate_count"] = int(len(candidate_specs))
    fit_summary["requested_atlas_empty_bin_fill_mode"] = requested_fill_mode
    fit_summary["selected_atlas_empty_bin_fill_mode"] = str(selected_candidate.get("fill_mode", ""))
    fit_summary["requested_texture_size"] = int(args.texture_size)
    fit_summary["texture_size_candidates"] = [int(x) for x in texture_size_candidates]
    fit_summary["selected_texture_size"] = int(selected_candidate.get("texture_size", int(args.texture_size)))
    fit_summary["requested_teacher_distilled_low_rank_texture_rank"] = int(
        args.teacher_distilled_low_rank_texture_rank
    )
    fit_summary["teacher_distilled_low_rank_texture_rank_candidates"] = [
        int(x) for x in teacher_low_rank_texture_rank_candidates
    ]
    fit_summary["selected_teacher_distilled_low_rank_texture_rank"] = int(
        selected_candidate.get(
            "teacher_distilled_low_rank_texture_rank",
            int(args.teacher_distilled_low_rank_texture_rank)
            if _is_low_rank_teacher_texture_mode(str(args.teacher_distilled_basis_mode))
            else 0,
        )
    )
    fit_summary["requested_surface_multiscale_prior_blend"] = float(args.surface_multiscale_prior_blend)
    fit_summary["surface_multiscale_prior_blend_candidates"] = [
        float(x) for x in surface_multiscale_prior_blend_candidates
    ]
    fit_summary["selected_surface_multiscale_prior_blend"] = float(
        selected_candidate.get("surface_multiscale_prior_blend", float(args.surface_multiscale_prior_blend))
    )
    fit_summary["selected_policy_val_prior_bin_gain_hybrid"] = bool(
        selected_candidate.get("policy_val_prior_bin_gain_hybrid", False)
    )
    fit_summary["requested_policy_val_source_mixture"] = bool(args.enable_policy_val_source_mixture)
    fit_summary["source_mixture_ridge"] = float(args.source_mixture_ridge)
    fit_summary["source_mixture_min_weight"] = float(args.source_mixture_min_weight)
    selected_max_abs_delta_rgb = float(
        selected_candidate.get("max_abs_delta_rgb", float(args.max_abs_delta_rgb))
    )
    fit_summary["requested_max_abs_delta_rgb"] = float(args.max_abs_delta_rgb)
    fit_summary["max_abs_delta_rgb_candidates"] = [
        float(x) for x in max_abs_delta_rgb_candidates
    ]
    fit_summary["selected_max_abs_delta_rgb"] = float(selected_max_abs_delta_rgb)
    fit_summary["selected_support_mode"] = str(selected_candidate.get("support_mode", ""))
    fit_summary["selected_support_added_faces"] = int(
        (selected_candidate.get("support_summary") or {}).get("added_faces", 0)
    )
    fit_summary["target_support_candidate_selection"] = {
        "enabled": bool(target_support_candidate_selection_enabled),
        "target_visible_energy_score_enabled": bool(args.enable_target_visible_energy_score),
        "ranking_primary": (
            "target_visible_residual_energy"
            if bool(args.enable_target_visible_energy_score)
            else "target_changed_fraction"
        ),
        "selected_profile": dict(target_support_profile),
        "selected_certificate": target_support_certificate(selected_candidate),
        "selected_score": target_support_score_dict(selected_candidate),
        "best_profile": dict(
            (best_target_support_candidate or {}).get("target_support_profile", {})
            if policy_auto_enabled
            else {}
        ),
        "best_certificate": (
            target_support_certificate(best_target_support_candidate)
            if policy_auto_enabled and best_target_support_candidate is not None
            else {"enabled": False, "passed": False, "reasons": ["no_target_support_profiled_candidate"]}
        ),
        "best_score": (
            target_support_score_dict(best_target_support_candidate)
            if policy_auto_enabled and best_target_support_candidate is not None
            else {}
        ),
        "best_candidate": (
            {
                "fill_mode": str(best_target_support_candidate.get("fill_mode", "")),
                "texture_size": int(best_target_support_candidate.get("texture_size", 0)),
                "support_mode": str(best_target_support_candidate.get("support_mode", "")),
                "surface_multiscale_prior_blend": float(
                    best_target_support_candidate.get("surface_multiscale_prior_blend", 0.0)
                ),
                "max_abs_delta_rgb": float(
                    best_target_support_candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb)
                ),
                "accepted": bool(best_target_support_candidate.get("accepted", False)),
                "selected_alpha": float(best_target_support_candidate.get("selected_alpha", 0.0)),
            }
            if policy_auto_enabled and best_target_support_candidate is not None
            else {}
        ),
        "selected_is_best_target_support": bool(
            policy_auto_enabled
            and best_target_support_candidate is not None
            and selected_candidate is best_target_support_candidate
        ),
        "best_scope": "eligible_after_nonreg_guard" if policy_auto_enabled else "fixed",
        "global_best_profile": dict(
            (global_best_target_support_candidate or {}).get("target_support_profile", {})
            if policy_auto_enabled
            else {}
        ),
        "global_best_certificate": (
            target_support_certificate(global_best_target_support_candidate)
            if policy_auto_enabled and global_best_target_support_candidate is not None
            else {"enabled": False, "passed": False, "reasons": ["no_target_support_profiled_candidate"]}
        ),
        "global_best_score": (
            target_support_score_dict(global_best_target_support_candidate)
            if policy_auto_enabled and global_best_target_support_candidate is not None
            else {}
        ),
        "global_best_candidate": (
            {
                "fill_mode": str(global_best_target_support_candidate.get("fill_mode", "")),
                "texture_size": int(global_best_target_support_candidate.get("texture_size", 0)),
                "support_mode": str(global_best_target_support_candidate.get("support_mode", "")),
                "surface_multiscale_prior_blend": float(
                    global_best_target_support_candidate.get("surface_multiscale_prior_blend", 0.0)
                ),
                "max_abs_delta_rgb": float(
                    global_best_target_support_candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb)
                ),
                "accepted": bool(global_best_target_support_candidate.get("accepted", False)),
                "selected_alpha": float(global_best_target_support_candidate.get("selected_alpha", 0.0)),
            }
            if policy_auto_enabled and global_best_target_support_candidate is not None
            else {}
        ),
        "selected_is_global_best_target_support": bool(
            policy_auto_enabled
            and global_best_target_support_candidate is not None
            and selected_candidate is global_best_target_support_candidate
        ),
    }
    fit_summary["target_support_prerank"] = dict(target_support_prerank_summary)
    target_apply: dict[str, Any] = {}
    accepted_before_target_coverage = bool(accepted)
    reject_reason = "; ".join(risk_gate_reasons) if risk_gate_reasons and not accepted else ""
    fallback_written = False
    effective_policy = "accepted_atlas" if accepted else "rejected"
    if accepted:
        target_views = evidence_views(target_evidence)
        if not target_views:
            raise FileNotFoundError(f"no target npz views found in {target_evidence}")
        target_apply = apply_to_target(
            target_views,
            atlas,
            output_model=output_model,
            split=str(args.target_split),
            method_name=str(args.method_name),
            alpha=selected_alpha,
            min_alpha=float(args.min_alpha),
            max_abs_delta_rgb=float(selected_max_abs_delta_rgb),
            min_atlas_bin_count=int(args.min_atlas_bin_count),
            min_atlas_face_samples=int(args.min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
            atlas_confidence_mode=str(args.atlas_confidence_mode),
            atlas_confidence_count_scale=float(args.atlas_confidence_count_scale),
            atlas_confidence_empty_bin=float(args.atlas_confidence_empty_bin),
            atlas_confidence_variance_scale=float(args.atlas_confidence_variance_scale),
            atlas_confidence_sign_power=float(args.atlas_confidence_sign_power),
            atlas_confidence_face_sample_scale=float(args.atlas_confidence_face_sample_scale),
            min_atlas_confidence=float(args.min_atlas_confidence),
            local_alpha_profile=local_alpha_profile,
            face_gain_guard_profile=face_gain_guard_profile,
            bin_uncertainty_guard_profile=bin_uncertainty_guard_profile,
            parent_edge_apply_profile=parent_edge_apply_profile,
            view_confidence_profile=view_confidence_profile,
        )
        target_change_floor_enabled = bool(
            args.enable_policy_val_sparse_residual_materialization
            or args.enable_target_support_candidate_selection
            or args.enable_policy_val_view_consistency_confidence
            or args.enable_policy_val_view_alpha_cap
        )
        min_changed = float(args.min_target_changed_fraction)
        if min_changed <= 0.0 and target_change_floor_enabled:
            min_changed = 1.0e-12
        png_changed_pixels = int(target_apply.get("png_quantized_changed_pixels", target_apply.get("changed_pixels", 0)) or 0)
        if min_changed > 0.0 and float(target_apply.get("changed_fraction", 0.0)) < min_changed:
            accepted = False
            reject_reason = (
                f"target_changed_fraction {float(target_apply.get('changed_fraction', 0.0)):.8f} "
                f"< min_target_changed_fraction {min_changed:.8f}"
            )
            effective_policy = "rejected_target_coverage"
        if accepted and target_change_floor_enabled and png_changed_pixels <= 0:
            accepted = False
            reject_reason = "target_png_quantized_changed_pixels <= 0 for target-visible/sparse policy"
            effective_policy = "rejected_target_coverage"
    if not accepted and bool(args.write_noop_on_reject):
        target_views = evidence_views(target_evidence)
        if not target_views:
            raise FileNotFoundError(f"no target npz views found in {target_evidence}")
        target_apply = write_noop_fallback_output(
            source_model=source_model,
            output_model=output_model,
            split=str(args.target_split),
            base_method_name=str(args.base_method_name),
            method_name=str(args.method_name),
            target_views=target_views,
            fallback_source=str(args.noop_fallback_source),
        )
        fallback_written = True
        effective_policy = "fallback_noop"
        if not reject_reason:
            reject_reason = "policy_val_candidate_rejected; wrote no-op fallback"

    atlas_path = output_model / "surface_residual_region_texture_atlas.npz"
    save_atlas_npz(atlas_path, atlas)
    audit = {
        "accepted": bool(accepted),
        "effective_policy": str(effective_policy),
        "fallback_written": bool(fallback_written),
        "source_model": str(source_model),
        "output_model": str(output_model),
        "fit_evidence_dir": str(fit_evidence),
        "target_evidence_dir": str(target_evidence),
        "region_carrier_json": str(carrier_json),
        "base_method_name": str(args.base_method_name),
        "method_name": str(args.method_name),
        "target_split": str(args.target_split),
        "residual_rgb_key": str(args.residual_rgb_key),
        "residual_l1_key": str(args.residual_l1_key),
        "carrier_summary": carrier_summary,
        "fit_summary": fit_summary,
        "policy_val": policy_val,
        "fill_mode_candidates": [
            {
                "fill_mode": str(candidate.get("fill_mode", "")),
                "texture_size": int(candidate.get("texture_size", 0)),
                "max_abs_delta_rgb": float(candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb)),
                "support_mode": str(candidate.get("support_mode", "")),
                "support_summary": dict(candidate.get("support_summary", {})),
                "policy_val_prior_bin_gain_hybrid": bool(
                    candidate.get("policy_val_prior_bin_gain_hybrid", False)
                ),
                "accepted": bool(candidate.get("accepted", False)),
                "selected_alpha": float(candidate.get("selected_alpha", 0.0)),
                "local_alpha_calibration": dict(candidate.get("local_alpha_profile", {"enabled": False})),
                "parent_edge_apply_shrink": dict(parent_edge_apply_profile),
                "face_gain_guard": {
                    key: value
                    for key, value in dict(candidate.get("face_gain_guard_profile", {})).items()
                    if key != "allowed_faces"
                },
                "bin_uncertainty_guard": {
                    key: value
                    for key, value in dict(candidate.get("bin_uncertainty_guard_profile", {})).items()
                    if key != "allowed_bins_by_face"
                },
                "view_consistency_confidence": {
                    **sanitize_view_confidence_profile(candidate.get("view_confidence_profile", {}))
                },
                "view_alpha_cap": {
                    **sanitize_view_alpha_cap_profile(candidate.get("view_alpha_cap_profile", {}))
                },
                "risk_gate_reasons": list(candidate.get("risk_gate_reasons", [])),
                "fit_summary": dict(candidate.get("fit_summary", {})),
                "policy_val_best": dict(candidate.get("best", {})),
            }
            for candidate in candidate_runs
        ],
        "accepted_before_target_coverage": bool(accepted_before_target_coverage),
        "reject_reason": reject_reason,
        "selected_alpha": float(selected_alpha),
        "local_alpha_profile": local_alpha_profile,
        "parent_edge_apply_profile": parent_edge_apply_profile,
        "face_gain_guard_profile": face_gain_guard_profile,
        "bin_uncertainty_guard_profile": bin_uncertainty_guard_profile,
        "view_confidence_profile": {
            **sanitize_view_confidence_profile(view_confidence_profile)
        },
        "view_alpha_cap_profile": {
            **sanitize_view_alpha_cap_profile(view_alpha_cap_profile)
        },
        "policy_val_risk_gate": {
            "passed": bool(not risk_gate_reasons),
            "reasons": risk_gate_reasons,
            "min_policy_val_positive_view_fraction": float(args.min_policy_val_positive_view_fraction),
            "min_policy_val_cvar20_relative_gain": float(args.min_policy_val_cvar20_relative_gain),
            "min_policy_val_min_view_relative_gain": float(args.min_policy_val_min_view_relative_gain),
            "enable_policy_val_image_ssim_gate": bool(args.enable_policy_val_image_ssim_gate),
            "min_policy_val_ssim_mean_gain": float(args.min_policy_val_ssim_mean_gain),
            "min_policy_val_ssim_positive_view_fraction": float(args.min_policy_val_ssim_positive_view_fraction),
            "min_policy_val_ssim_min_view_gain": float(args.min_policy_val_ssim_min_view_gain),
            "enable_policy_val_image_l1_gate": bool(args.enable_policy_val_image_l1_gate),
            "min_policy_val_l1_mean_gain": float(args.min_policy_val_l1_mean_gain),
            "min_policy_val_l1_positive_view_fraction": float(args.min_policy_val_l1_positive_view_fraction),
            "min_policy_val_l1_min_view_gain": float(args.min_policy_val_l1_min_view_gain),
            "min_policy_val_l1_cvar20_view_gain": float(args.min_policy_val_l1_cvar20_view_gain),
            "enable_policy_val_image_lpips_gate": bool(args.enable_policy_val_image_lpips_gate),
            "policy_val_lpips_max_size": int(args.policy_val_lpips_max_size),
            "min_policy_val_lpips_mean_gain": float(args.min_policy_val_lpips_mean_gain),
            "min_policy_val_lpips_positive_view_fraction": float(
                args.min_policy_val_lpips_positive_view_fraction
            ),
            "min_policy_val_lpips_min_view_gain": float(args.min_policy_val_lpips_min_view_gain),
            "min_policy_val_lpips_cvar20_view_gain": float(args.min_policy_val_lpips_cvar20_view_gain),
            "enable_policy_val_effective_margin_gate": bool(
                args.enable_policy_val_effective_margin_gate
            ),
            "min_policy_val_effective_relative_gain": float(
                args.min_policy_val_effective_relative_gain
            ),
            "min_policy_val_effective_ssim_gain": float(args.min_policy_val_effective_ssim_gain),
            "min_policy_val_effective_l1_gain": float(args.min_policy_val_effective_l1_gain),
            "min_policy_val_effective_ssim_cvar20_gain": float(
                args.min_policy_val_effective_ssim_cvar20_gain
            ),
            "min_policy_val_effective_l1_cvar20_gain": float(
                args.min_policy_val_effective_l1_cvar20_gain
            ),
            "min_policy_val_effective_lpips_gain": float(args.min_policy_val_effective_lpips_gain),
            "min_policy_val_effective_lpips_cvar20_gain": float(
                args.min_policy_val_effective_lpips_cvar20_gain
            ),
            "enable_prior_bin_gain_hybrid_l1_proxy_gate": bool(
                args.enable_prior_bin_gain_hybrid_l1_proxy_gate
            ),
            "prior_bin_gain_hybrid_min_l1_abs_gain": float(
                args.prior_bin_gain_hybrid_min_l1_abs_gain
            ),
            "prior_bin_gain_hybrid_min_l1_relative_gain": float(
                args.prior_bin_gain_hybrid_min_l1_relative_gain
            ),
            "prior_bin_gain_hybrid_min_l1_positive_view_fraction": float(
                args.prior_bin_gain_hybrid_min_l1_positive_view_fraction
            ),
            "prior_bin_gain_hybrid_min_l1_min_view_gain": float(
                args.prior_bin_gain_hybrid_min_l1_min_view_gain
            ),
            "prior_bin_gain_hybrid_min_l1_cvar20_view_gain": float(
                args.prior_bin_gain_hybrid_min_l1_cvar20_view_gain
            ),
            "selected_sparse_materialization_selective": bool(
                best.get("sparse_materialization_selective", False)
            ),
            "selected_positive_view_fraction": float(best.get("positive_view_fraction", 0.0)),
            "selected_nonnegative_view_fraction": float(
                best.get("nonnegative_view_fraction", best.get("positive_view_fraction", 0.0))
            ),
            "selected_cvar20_view_relative_gain": float(best.get("cvar20_view_relative_gain", 0.0)),
            "selected_min_view_relative_gain": float(best.get("min_view_relative_gain", 0.0)),
            "selected_ssim_gain": float(best.get("ssim_gain", 0.0)),
            "selected_ssim_positive_view_fraction": float(best.get("ssim_positive_view_fraction", 0.0)),
            "selected_ssim_nonnegative_view_fraction": float(
                best.get("ssim_nonnegative_view_fraction", best.get("ssim_positive_view_fraction", 0.0))
            ),
            "selected_ssim_min_view_gain": float(best.get("ssim_min_view_gain", 0.0)),
            "selected_image_l1_gain": float(best.get("image_l1_gain", 0.0)),
            "selected_image_l1_positive_view_fraction": float(best.get("image_l1_positive_view_fraction", 0.0)),
            "selected_image_l1_nonnegative_view_fraction": float(
                best.get(
                    "image_l1_nonnegative_view_fraction",
                    best.get("image_l1_positive_view_fraction", 0.0),
                )
            ),
            "selected_image_l1_min_view_gain": float(best.get("image_l1_min_view_gain", 0.0)),
            "selected_image_l1_cvar20_view_gain": float(best.get("image_l1_cvar20_view_gain", 0.0)),
            "selected_lpips_gain": float(best.get("lpips_gain", 0.0)),
            "selected_lpips_positive_view_fraction": float(best.get("lpips_positive_view_fraction", 0.0)),
            "selected_lpips_nonnegative_view_fraction": float(
                best.get("lpips_nonnegative_view_fraction", best.get("lpips_positive_view_fraction", 0.0))
            ),
            "selected_lpips_min_view_gain": float(best.get("lpips_min_view_gain", 0.0)),
            "selected_lpips_cvar20_view_gain": float(best.get("lpips_cvar20_view_gain", 0.0)),
        },
        "target_footprint_tail_risk_gate": {
            "enabled": bool(args.enable_target_footprint_tail_risk_certificate),
            "all_bins": bool(args.target_footprint_tail_risk_all_bins),
            "min_positive_view_fraction": float(
                args.target_footprint_tail_risk_min_positive_view_fraction
            ),
            "min_min_view_gain": float(args.target_footprint_tail_risk_min_min_view_gain),
            "min_cvar20_view_gain": float(args.target_footprint_tail_risk_min_cvar20_view_gain),
            "selected_profile": dict(
                (
                    (selected_candidate.get("fit_summary") or {})
                    .get("policy_val_prior_bin_gain_hybrid", {})
                    .get("target_footprint_tail_risk_certificate", {})
                )
                if isinstance(selected_candidate, dict)
                else {}
            ),
        },
        "target_apply": target_apply,
        "target_coverage_gate": {
            "min_target_changed_fraction": float(args.min_target_changed_fraction),
            "changed_fraction": float(target_apply.get("changed_fraction", 0.0)) if target_apply else 0.0,
            "passed": bool(not reject_reason),
        },
        "atlas_path": str(atlas_path),
        "settings": vars(args),
    }
    safe_audit = json_safe(audit)
    audit_path = output_model / "surface_residual_region_texture_adapter_audit.json"
    audit_path.write_text(json.dumps(safe_audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(output_model / "surface_residual_region_texture_adapter_audit.md", safe_audit)
    print(
        json.dumps(
            {k: safe_audit[k] for k in ("accepted", "effective_policy", "selected_alpha", "target_apply")},
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if accepted or fallback_written else 2


if __name__ == "__main__":
    raise SystemExit(main())
