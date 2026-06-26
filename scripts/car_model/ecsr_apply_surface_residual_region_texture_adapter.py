#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        "surface_multiscale_prior_blend": float(spec.get("surface_multiscale_prior_blend", 0.0)),
        "max_abs_delta_rgb": float(spec.get("max_abs_delta_rgb", 0.0)),
        "support_mode": str(spec.get("support_mode", "")),
        "support_added_faces": int(spec.get("support_added_faces", 0)),
        "support_candidate_faces": int(spec.get("support_candidate_faces", 0)),
        "support_faces_sha1": str(spec.get("support_faces_sha1", "")),
    }


def policy_candidate_spec_key(spec: dict[str, Any]) -> tuple[str, int, float, float, str]:
    return (
        str(spec.get("fill_mode", "")),
        int(spec.get("texture_size", 0)),
        round(float(spec.get("surface_multiscale_prior_blend", 0.0)), 8),
        round(float(spec.get("max_abs_delta_rgb", 0.0)), 8),
        str(spec.get("support_faces_sha1", "")),
    )


def evidence_views(evidence_dir: Path) -> list[Path]:
    views_dir = evidence_dir / "views"
    if views_dir.is_dir():
        return sorted(views_dir.glob("*.npz"))
    return sorted(evidence_dir.glob("*.npz"))


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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank support faces by train residual debt weighted by GT-free target footprint."""
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
    return expanded, {
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

    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_model, output_model, ignore=ignore)


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
    raise ValueError(f"unsupported teacher-distilled basis mode: {mode}")


def _teacher_distilled_basis_features_for_mask(
    z: np.lib.npyio.NpzFile,
    mode: str,
    mask: np.ndarray,
) -> np.ndarray | None:
    mode = str(mode)
    if mode == "none":
        return None
    if mode not in {"face_uv_normal_camera_ridge", "face_uv_patch_mixture_ridge"}:
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



def _local_alpha_for_samples(
    samples: np.ndarray,
    profile: dict[str, Any] | None,
    face_ids: np.ndarray | None = None,
    bin_ids: np.ndarray | None = None,
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
        baseline_alpha = _local_alpha_for_samples(samples, baseline_profile, face_ids=face_ids, bin_ids=bin_ids)
        prior_alpha = _local_alpha_for_samples(samples, prior_profile, face_ids=face_ids, bin_ids=bin_ids)
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
    if str(profile.get("mode", "")) != "policy_val_bin_uncertainty_guard":
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
    teacher_distilled_basis_mode: str,
    teacher_distilled_basis_min_face_samples: int,
    teacher_distilled_basis_ridge: float,
    teacher_distilled_basis_ood_max_z: float,
    teacher_distilled_basis_ood_min_std: float,
    teacher_distilled_basis_apply_mode: str,
    teacher_distilled_basis_blend: float,
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
    teacher_basis_xtx_sums: dict[int, np.ndarray] = {}
    teacher_basis_xty_sums: dict[int, np.ndarray] = {}
    teacher_basis_feature_sums: dict[int, np.ndarray] = {}
    teacher_basis_feature_sq_sums: dict[int, np.ndarray] = {}
    teacher_basis_counts: dict[int, int] = {}
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
    teacher_basis_mode = str(teacher_distilled_basis_mode)
    teacher_basis_feature_dim = _teacher_distilled_basis_feature_dim(teacher_basis_mode)
    teacher_basis_apply_mode = str(teacher_distilled_basis_apply_mode)
    if teacher_basis_apply_mode not in {"replace_supported", "blend", "fill_empty_only"}:
        raise ValueError(f"unsupported teacher-distilled basis apply mode: {teacher_basis_apply_mode}")
    teacher_basis_views = 0
    teacher_basis_samples = 0
    rng = np.random.default_rng(7)

    stride = max(0, int(policy_val_stride))
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
        ubin, vbin = _uv_bins(np.asarray(z["barycentric"], dtype=np.float32), mask, size)
        total_fit_samples += int(face_ids.size)
        view_features = _view_condition_features_for_mask(z, view_basis_mode, mask)
        if view_features is not None:
            view_basis_views += 1
            total_view_basis_samples += int(face_ids.size)
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
                if teacher_basis_feature_dim > 0:
                    teacher_basis_xtx_sums[face] = np.zeros(
                        (teacher_basis_feature_dim, teacher_basis_feature_dim),
                        dtype=np.float64,
                    )
                    teacher_basis_xty_sums[face] = np.zeros((teacher_basis_feature_dim, 3), dtype=np.float64)
                    teacher_basis_feature_sums[face] = np.zeros((teacher_basis_feature_dim,), dtype=np.float64)
                    teacher_basis_feature_sq_sums[face] = np.zeros((teacher_basis_feature_dim,), dtype=np.float64)
                    teacher_basis_counts[face] = 0
            np.add.at(sums[face], (vbin[fm], ubin[fm]), residual_samples[fm].astype(np.float64))
            np.add.at(sq_sums[face], (vbin[fm], ubin[fm]), np.square(residual_samples[fm]).astype(np.float64))
            np.add.at(sign_sums[face], (vbin[fm], ubin[fm]), np.sign(residual_samples[fm]).astype(np.float64))
            np.add.at(counts[face], (vbin[fm], ubin[fm]), 1)
            mean_sums[face] += residual_samples[fm].sum(axis=0)
            mean_counts[face] += int(fm.sum())
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
                teacher_basis_xtx_sums[face] += face_features.T @ face_features
                teacher_basis_xty_sums[face] += face_features.T @ face_targets
                teacher_basis_feature_sums[face] += np.sum(face_features, axis=0)
                teacher_basis_feature_sq_sums[face] += np.sum(face_features * face_features, axis=0)
                teacher_basis_counts[face] = teacher_basis_counts.get(face, 0) + int(face_features.shape[0])

    atlas: dict[int, FaceAtlas] = {}
    view_basis_supported_bins_total = 0
    view_basis_bins_total = 0
    teacher_basis_supported_faces = 0
    teacher_basis_candidate_faces = 0
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
        if teacher_basis_feature_dim > 0 and face in teacher_basis_xtx_sums:
            teacher_basis_candidate_faces += 1
            teacher_count = int(teacher_basis_counts.get(face, 0))
            if teacher_count >= max(1, int(teacher_distilled_basis_min_face_samples)):
                ridge = max(0.0, float(teacher_distilled_basis_ridge))
                eye = np.eye(teacher_basis_feature_dim, dtype=np.float64)
                xtx = teacher_basis_xtx_sums[face] + ridge * eye
                xty = teacher_basis_xty_sums[face]
                try:
                    teacher_basis_coefficients = np.linalg.solve(xtx, xty).astype(np.float32)
                    denom = max(1, int(teacher_count))
                    teacher_basis_feature_mean = (
                        teacher_basis_feature_sums[face] / float(denom)
                    ).astype(np.float32)
                    feature_var = (
                        teacher_basis_feature_sq_sums[face] / float(denom)
                        - np.square(teacher_basis_feature_mean.astype(np.float64))
                    )
                    teacher_basis_feature_std = np.sqrt(np.maximum(feature_var, 0.0)).astype(np.float32)
                    teacher_basis_supported_faces += 1
                except np.linalg.LinAlgError:
                    teacher_basis_coefficients = None
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
        )

    summary = {
        "input_views": int(len(view_paths)),
        "fit_views": int(len(fit_views)),
        "policy_val_views": int(len(val_views)),
        "candidate_faces": int(len(candidate_faces)),
        "atlas_faces": int(len(atlas)),
        "fit_samples": int(total_fit_samples),
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
        "teacher_distilled_basis": {
            "mode": str(teacher_basis_mode),
            "feature_dim": int(teacher_basis_feature_dim),
            "fit_views_with_features": int(teacher_basis_views),
            "fit_samples_with_features": int(teacher_basis_samples),
            "min_face_samples": int(teacher_distilled_basis_min_face_samples),
            "ridge": float(teacher_distilled_basis_ridge),
            "ood_max_z": float(teacher_distilled_basis_ood_max_z),
            "ood_min_std": float(teacher_distilled_basis_ood_min_std),
            "apply_mode": str(teacher_basis_apply_mode),
            "blend": float(teacher_distilled_basis_blend),
            "candidate_faces": int(teacher_basis_candidate_faces),
            "supported_faces": int(teacher_basis_supported_faces),
            "supported_face_fraction": float(teacher_basis_supported_faces / max(1, teacher_basis_candidate_faces)),
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
) -> tuple[np.ndarray, np.ndarray]:
    face_id = np.asarray(z["face_id"], dtype=np.int64)
    h, w = face_id.shape
    delta = np.zeros((3, h, w), dtype=np.float32)
    valid = face_id >= 0
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
                    teacher_pixels = np.einsum(
                        "nd,dc->nc",
                        feature_samples[teacher_support].astype(np.float32),
                        face_atlas.teacher_basis_coefficients.astype(np.float32),
                    )
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
        local_faces = faces[confident_indices]
        local_bin_ids = (
            vbin[confident_indices].astype(np.int64) * int(tex.shape[0])
        ) + ubin[confident_indices].astype(np.int64)
        local_alpha = _local_alpha_for_samples(
            base_pixels,
            local_alpha_profile,
            face_ids=local_faces,
            bin_ids=local_bin_ids,
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
            if parent_edge_multiplier is not None:
                values = values * parent_edge_multiplier[ys[support_valid], xs[support_valid]]
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
    local_alpha_profile: dict[str, Any] | None = None,
    face_gain_guard_profile: dict[str, Any] | None = None,
    bin_uncertainty_guard_profile: dict[str, Any] | None = None,
    parent_edge_apply_profile: dict[str, Any] | None = None,
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
            adapted = None
            if view_ssim_before is not None or view_l1_before is not None:
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
        else:
            view_stats = {
                "view_count": 0,
                "positive_view_fraction": 0.0,
                "min_view_relative_gain": 0.0,
                "p10_view_relative_gain": 0.0,
                "cvar20_view_relative_gain": 0.0,
            }
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
        row = {
            "alpha": float(alpha),
            "mse_before": mse_before,
            "mse_after": mse_after,
            "relative_gain": float(rel_gain),
            "mae_after": mae_after,
            **view_stats,
            **ssim_stats,
            **l1_stats,
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
    min_relative_gain: float,
    min_positive_view_fraction: float,
    max_mean_variance: float,
    min_mean_sign_consistency: float,
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
        relative_gain = (before - after) / max(before, 1.0e-12)
        positive_fraction = float(positive_views / max(1, views))
        mean_variance = float(variance_sum_by_key.get(key, 0.0) / max(1, samples))
        mean_sign = float(sign_sum_by_key.get(key, 0.0) / max(1, samples))
        variance_ok = float(max_mean_variance) < 0.0 or mean_variance <= float(max_mean_variance)
        sign_ok = float(min_mean_sign_consistency) <= 0.0 or mean_sign >= float(min_mean_sign_consistency)
        keep = bool(
            samples >= int(min_bin_samples)
            and relative_gain >= float(min_relative_gain)
            and positive_fraction >= float(min_positive_view_fraction)
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
                "mse_before_sum": float(before),
                "mse_after_sum": float(after),
                "relative_gain": float(relative_gain),
                "mean_variance": float(mean_variance),
                "mean_sign_consistency": float(mean_sign),
                "variance_ok": bool(variance_ok),
                "sign_ok": bool(sign_ok),
                "keep": bool(keep),
            }
        )
    for bins in allowed_bins_by_face.values():
        bins.sort()
    rows_by_gain = sorted(rows, key=lambda row: float(row["relative_gain"]))
    rows_by_samples = sorted(rows, key=lambda row: int(row["samples"]), reverse=True)
    allowed_bin_count = int(sum(len(v) for v in allowed_bins_by_face.values()))
    return {
        "enabled": True,
        "mode": "policy_val_bin_uncertainty_guard",
        "alpha": float(alpha),
        "policy_val_views_used": int(view_count),
        "samples": int(total_active_samples),
        "texture_size": int(texture_size),
        "min_bin_samples": int(min_bin_samples),
        "min_relative_gain": float(min_relative_gain),
        "min_positive_view_fraction": float(min_positive_view_fraction),
        "max_mean_variance": float(max_mean_variance),
        "min_mean_sign_consistency": float(min_mean_sign_consistency),
        "candidate_bin_count": int(len(rows)),
        "allowed_bin_count": int(allowed_bin_count),
        "rejected_bin_count": int(len(rows) - allowed_bin_count),
        "allowed_face_count": int(len(allowed_bins_by_face)),
        "allowed_sample_fraction": float(allowed_samples / max(1, total_active_samples)),
        "allowed_bins_by_face": allowed_bins_by_face,
        "worst_bins": rows_by_gain[:32],
        "best_sampled_bins": rows_by_samples[:32],
    }


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
    locally attenuated.
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
    before_by_key: dict[tuple[int, int], float] = {}
    after_by_key: dict[tuple[int, int], float] = {}
    samples_by_key: dict[tuple[int, int], int] = {}
    view_count_by_key: dict[tuple[int, int], int] = {}
    positive_view_count_by_key: dict[tuple[int, int], int] = {}
    variance_sum_by_key: dict[tuple[int, int], float] = {}
    sign_sum_by_key: dict[tuple[int, int], float] = {}
    structure_l1_bad_sum_by_key: dict[tuple[int, int], float] = {}
    structure_gradient_bad_sum_by_key: dict[tuple[int, int], float] = {}
    structure_edge_sum_by_key: dict[tuple[int, int], float] = {}
    structure_bad_view_count_by_key: dict[tuple[int, int], int] = {}
    total_active_samples = 0
    view_count = 0
    structure_view_count = 0
    structure_missing_view_count = 0
    structure_enabled = bool(enable_structure_aware_shrink) and (
        float(structure_shrink_l1_weight) > 0.0
        or float(structure_shrink_gradient_weight) > 0.0
        or float(structure_shrink_edge_weight) > 0.0
    )
    texture_size = int(next(iter(atlas.values())).texture.shape[0])
    for path in tqdm(val_views, desc="calibrate bin uncertainty shrink"):
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
    fallback_shrink_f = float(np.clip(fallback_shrink, min_shrink_f, max_shrink_f))
    policy_mode_s = str(policy_mode)
    for key in sorted(before_by_key):
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
        evidence_ok = bool(
            samples >= int(min_bin_samples)
            and relative_gain >= float(min_relative_gain)
            and positive_fraction >= float(min_positive_view_fraction)
            and variance_ok
            and sign_ok
        )
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
                max_shrink_f * count_conf * gain_conf * positive_fraction * variance_conf * sign_conf,
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
            "face_id": int(face),
            "bin_id": int(bin_id),
            "u": int(bin_id) % texture_size,
            "v": int(bin_id) // texture_size,
            "samples": int(samples),
            "view_count": int(views),
            "positive_view_count": int(positive_views),
            "positive_view_fraction": float(positive_fraction),
            "mse_before_sum": float(before),
            "mse_after_sum": float(after),
            "relative_gain": float(relative_gain),
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

    bin_shrinks_by_face: dict[str, dict[str, float]] = {}
    for row in selected_rows:
        face_key = str(int(row["face_id"]))
        bin_shrinks_by_face.setdefault(face_key, {})[str(int(row["bin_id"]))] = float(row["shrink"])
    shrinks = [float(row["shrink"]) for row in selected_rows]
    profile = {
        "enabled": True,
        "mode": "policy_val_bin_uncertainty_shrink",
        "uncertainty_shrink_policy_mode": policy_mode_s,
        "alpha_grid": base,
        "policy_val_views_used": int(view_count),
        "samples": int(total_active_samples),
        "texture_size": int(texture_size),
        "min_bin_samples": int(min_bin_samples),
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
        "candidate_bin_count": int(len(rows)),
        "bin_uncertainty_shrink_count": int(len(selected_rows)),
        "fallback_bin_count": int(fallback_bin_count),
        "selected_face_count": int(len(bin_shrinks_by_face)),
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
        )
        if float(max_abs_delta_rgb) > 0.0:
            delta = np.clip(delta, -float(max_abs_delta_rgb), float(max_abs_delta_rgb))
        changed_mask = np.any(np.abs(delta) > 1.0e-8, axis=0)
        adapted = np.clip(rgb + delta, 0.0, 1.0)
        name = f"{path.stem}.png"
        save_image_chw(render_dir / name, adapted)
        if "rgb_gt" in z:
            save_image_chw(gt_dir / name, np.asarray(z["rgb_gt"], dtype=np.float32))
        else:
            save_image_chw(gt_dir / name, rgb)
        changed_pixels += int(changed_mask.sum())
        total_pixels += int(valid.size)
        written += 1
    return {
        "split": split,
        "method_name": method_name,
        "written_views": int(written),
        "changed_pixels": int(changed_pixels),
        "total_pixels": int(total_pixels),
        "changed_fraction": float(changed_pixels / max(1, total_pixels)),
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
        )
        if float(max_abs_delta_rgb) > 0.0:
            delta = np.clip(delta, -float(max_abs_delta_rgb), float(max_abs_delta_rgb))
        changed_mask = np.any(np.abs(delta) > 1.0e-8, axis=0)
        changed = int(changed_mask.sum())
        valid_count = int(np.asarray(valid, dtype=bool).sum())
        total = int(valid.size)
        abs_delta = float(np.sum(np.abs(delta)))
        active_abs_delta = float(np.sum(np.abs(delta[:, changed_mask]))) if changed > 0 else 0.0
        changed_pixels += changed
        valid_pixels += valid_count
        total_pixels += total
        abs_delta_sum += abs_delta
        active_abs_delta_sum += active_abs_delta
        active_channel_values += int(changed * 3)
        view_rows.append(
            {
                "view": str(path.name),
                "changed_pixels": int(changed),
                "valid_pixels": int(valid_count),
                "total_pixels": int(total),
                "changed_fraction": float(changed / max(1, total)),
                "valid_fraction": float(valid_count / max(1, total)),
                "mean_abs_delta": float(abs_delta / max(1, total * 3)),
                "active_mean_abs_delta": float(active_abs_delta / max(1, changed * 3)),
            }
        )
    changed_fracs = sorted(float(row["changed_fraction"]) for row in view_rows)
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
        "valid_pixels": int(valid_pixels),
        "total_pixels": int(total_pixels),
        "changed_fraction": float(changed_pixels / max(1, total_pixels)),
        "valid_fraction": float(valid_pixels / max(1, total_pixels)),
        "min_view_changed_fraction": float(changed_fracs[0] if changed_fracs else 0.0),
        "cvar20_view_changed_fraction": lower_cvar(changed_fracs, 0.2),
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
    teacher_basis_enabled = bool(
        faces.size
        and any(atlas[int(face)].teacher_basis_coefficients is not None for face in faces)
    )
    if teacher_basis_enabled:
        first_teacher = next(
            atlas[int(face)].teacher_basis_coefficients
            for face in faces
            if atlas[int(face)].teacher_basis_coefficients is not None
        )
        assert first_teacher is not None
        teacher_dim, teacher_channels = first_teacher.shape
        teacher_coeffs = []
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
                teacher_modes.append("none")
                teacher_feature_means.append(np.zeros((teacher_dim,), dtype=np.float32))
                teacher_feature_stds.append(np.ones((teacher_dim,), dtype=np.float32))
            else:
                teacher_coeffs.append(face_atlas.teacher_basis_coefficients.astype(np.float32))
                teacher_modes.append(str(face_atlas.teacher_basis_mode))
                teacher_feature_means.append(
                    face_atlas.teacher_basis_feature_mean.astype(np.float32)
                    if face_atlas.teacher_basis_feature_mean is not None
                    else np.zeros((teacher_dim,), dtype=np.float32)
                )
                teacher_feature_stds.append(
                    face_atlas.teacher_basis_feature_std.astype(np.float32)
                    if face_atlas.teacher_basis_feature_std is not None
                    else np.ones((teacher_dim,), dtype=np.float32)
                )
            teacher_ood_max_z.append(float(face_atlas.teacher_basis_ood_max_z))
            teacher_ood_min_std.append(float(face_atlas.teacher_basis_ood_min_std))
            teacher_apply_modes.append(str(face_atlas.teacher_basis_apply_mode))
            teacher_blends.append(float(face_atlas.teacher_basis_blend))
        payload["teacher_basis_coefficients"] = np.stack(teacher_coeffs, axis=0)
        payload["teacher_basis_mode"] = np.asarray(teacher_modes)
        payload["teacher_basis_feature_mean"] = np.stack(teacher_feature_means, axis=0)
        payload["teacher_basis_feature_std"] = np.stack(teacher_feature_stds, axis=0)
        payload["teacher_basis_ood_max_z"] = np.asarray(teacher_ood_max_z, dtype=np.float32)
        payload["teacher_basis_ood_min_std"] = np.asarray(teacher_ood_min_std, dtype=np.float32)
        payload["teacher_basis_apply_mode"] = np.asarray(teacher_apply_modes)
        payload["teacher_basis_blend"] = np.asarray(teacher_blends, dtype=np.float32)
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
    multiscale_prior = audit.get("fit_summary", {}).get("surface_multiscale_prior", {})
    teacher_basis = audit.get("fit_summary", {}).get("teacher_distilled_basis", {})
    policy_candidate_control = audit.get("fit_summary", {}).get("policy_candidate_control", {})
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
        f"- selected fill mode: `{audit.get('fit_summary', {}).get('selected_atlas_empty_bin_fill_mode', audit.get('fit_summary', {}).get('atlas_empty_bin_fill_mode', ''))}`",
        f"- selected max abs delta RGB: `{float(audit.get('fit_summary', {}).get('selected_max_abs_delta_rgb', audit.get('fit_summary', {}).get('requested_max_abs_delta_rgb', 0.0)) or 0.0):.6f}`",
        f"- max abs delta RGB candidates: `{audit.get('fit_summary', {}).get('max_abs_delta_rgb_candidates', [audit.get('fit_summary', {}).get('requested_max_abs_delta_rgb', 0.0)])}`",
        f"- policy candidate dominance pruning: `{policy_candidate_control.get('dominance_pruning_enabled', False)}`",
        f"- policy candidates planned before pruning: `{policy_candidate_control.get('planned_candidate_count_before_pruning', 0)}`",
        f"- policy candidates planned after pruning: `{policy_candidate_control.get('planned_candidate_count_after_pruning', 0)}`",
        f"- policy candidates executed: `{policy_candidate_control.get('executed_candidate_count', 0)}`",
        f"- policy candidate early-stop mode: `{policy_candidate_control.get('early_stop_effective_mode', 'none')}`",
        f"- policy candidate early-stop skipped: `{policy_candidate_control.get('early_stop_skipped_count', 0)}`",
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
        f"- teacher-distilled basis mode: `{teacher_basis.get('mode', 'none')}`",
        f"- teacher-distilled basis effective mode: `{teacher_basis.get('effective_mode', teacher_basis.get('mode', 'none'))}`",
        f"- teacher-distilled basis guard decision: `{teacher_basis.get('guard', {}).get('decision', 'not_requested')}`",
        f"- teacher-distilled basis supported faces: `{teacher_basis.get('supported_faces', 0)}`",
        f"- teacher-distilled basis supported-face fraction: `{teacher_basis.get('supported_face_fraction', 0.0):.6f}`",
        f"- teacher-distilled basis apply mode: `{teacher_basis.get('apply_mode', '')}`",
        f"- teacher-distilled basis blend: `{teacher_basis.get('blend', 0.0)}`",
        f"- policy-val enabled: `{val.get('enabled', False)}`",
        f"- policy-val samples: `{val.get('samples', 0)}`",
        f"- selected alpha: `{audit.get('selected_alpha', 0.0)}`",
        f"- local alpha calibration: `{audit.get('fit_summary', {}).get('local_alpha_calibration', {}).get('enabled', False)}`",
        f"- local alpha mode: `{local_alpha.get('mode', 'disabled')}`",
        f"- local alpha fallback alpha: `{fallback_alpha_text}`",
        f"- local alpha face count: `{local_alpha.get('face_alpha_count', 0)}`",
        f"- local alpha bin count: `{local_alpha.get('bin_alpha_count', 0)}`",
        f"- local alpha bin RGB count: `{local_alpha.get('bin_rgb_alpha_count', 0)}`",
        f"- local alpha uncertainty-shrink bin count: `{local_alpha.get('bin_uncertainty_shrink_count', 0)}`",
        f"- local alpha uncertainty-shrink policy mode: `{local_alpha.get('uncertainty_shrink_policy_mode', 'n/a')}`",
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
        f"- policy-val risk gate: `{risk.get('passed', True)}`",
        f"- target written views: `{target.get('written_views', 0)}`",
        f"- target changed fraction: `{target.get('changed_fraction', 0.0):.6f}`",
        f"- effective policy: `{audit.get('effective_policy', '')}`",
        f"- target coverage gate: `{audit.get('target_coverage_gate', {}).get('passed', True)}`",
        f"- reject reason: `{audit.get('reject_reason', '')}`",
        "",
        "## Alpha Rows",
        "",
        "| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"{float(row.get('mse_before', 0.0)):.8f} | {float(row.get('mse_after', 0.0)):.8f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def policy_val_risk_reasons(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    pos_frac = float(row.get("positive_view_fraction", 0.0))
    min_pos_frac = float(args.min_policy_val_positive_view_fraction)
    if min_pos_frac > 0.0 and pos_frac < min_pos_frac:
        reasons.append(
            f"positive_view_fraction {pos_frac:.6f} < min_policy_val_positive_view_fraction {min_pos_frac:.6f}"
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
        ssim_pos_frac = float(row.get("ssim_positive_view_fraction", 0.0))
        min_ssim_pos_frac = float(args.min_policy_val_ssim_positive_view_fraction)
        if min_ssim_pos_frac > 0.0 and ssim_pos_frac < min_ssim_pos_frac:
            reasons.append(
                f"ssim_positive_view_fraction {ssim_pos_frac:.6f} < "
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
        l1_pos_frac = float(row.get("image_l1_positive_view_fraction", 0.0))
        min_l1_pos_frac = float(args.min_policy_val_l1_positive_view_fraction)
        if min_l1_pos_frac > 0.0 and l1_pos_frac < min_l1_pos_frac:
            reasons.append(
                f"image_l1_positive_view_fraction {l1_pos_frac:.6f} < "
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
    return reasons


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
            best_row = max(safe_rows, key=lambda row: float(row.get("relative_gain", -1.0)))
            policy_payload["best"] = best_row
            selection_mode = "risk_gate"
            refinement = policy_payload.get("ssim_alpha_refinement") or {}
            if bool(refinement.get("enabled", False)) and float(best_row.get("alpha", 0.0)) in {
                float(x) for x in refinement.get("inserted_alpha_grid", []) or []
            }:
                selection_mode = "risk_gate_ssim_alpha_refined"
            policy_payload["selection"] = {
                "mode": selection_mode,
                "safe_alpha_count": int(len(safe_rows)),
                "selected_alpha": float(best_row.get("alpha", 0.0)),
                "selected_from_refinement": selection_mode == "risk_gate_ssim_alpha_refined",
            }
        else:
            selected_alpha_override = 0.0
            policy_payload["selection"] = {
                "mode": "risk_gate",
                "safe_alpha_count": 0,
                "selected_alpha": 0.0,
                "selected_from_refinement": False,
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
    parser.add_argument("--texture_size", type=int, default=16)
    parser.add_argument(
        "--texture_size_candidates",
        default="",
        help=(
            "Optional comma-separated texture sizes for train-only capacity auto-policy. "
            "If empty, keeps the fixed --texture_size legacy behavior."
        ),
    )
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
        "--teacher_distilled_basis_mode",
        choices=("none", "face_uv_normal_camera_ridge", "face_uv_patch_mixture_ridge"),
        default="none",
        help=(
            "Optional v65 teacher-distilled shared residual field. "
            "face_uv_normal_camera_ridge fits one ridge residual model per face using "
            "Phase-J teacher residuals and features [camera, normal, normal-dot-camera, UV polynomial]. "
            "face_uv_patch_mixture_ridge adds a 3x3 local UV RBF mixture and normal-view interactions."
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
            "downweights/upweights for evidenced bins."
        ),
    )
    parser.add_argument(
        "--bin_uncertainty_shrink_policy_mode",
        choices=("sparse_positive", "keep_with_downweight"),
        default="sparse_positive",
    )
    parser.add_argument("--bin_uncertainty_shrink_min_bin_samples", type=int, default=64)
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
    parser.add_argument("--bin_uncertainty_guard_min_bin_samples", type=int, default=64)
    parser.add_argument("--bin_uncertainty_guard_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--bin_uncertainty_guard_min_positive_view_fraction", type=float, default=0.75)
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
    if int(args.teacher_distilled_basis_min_face_samples) <= 0:
        parser.error("--teacher_distilled_basis_min_face_samples must be > 0")
    if float(args.teacher_distilled_basis_ridge) < 0.0:
        parser.error("--teacher_distilled_basis_ridge must be >= 0")
    if float(args.teacher_distilled_basis_ood_max_z) <= 0.0:
        parser.error("--teacher_distilled_basis_ood_max_z must be > 0")
    if float(args.teacher_distilled_basis_ood_min_std) <= 0.0:
        parser.error("--teacher_distilled_basis_ood_min_std must be > 0")
    if not 0.0 <= float(args.teacher_distilled_basis_blend) <= 1.0:
        parser.error("--teacher_distilled_basis_blend must be in [0, 1]")
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
    if bool(args.enable_policy_val_structure_aware_shrink) and not bool(args.enable_policy_val_bin_uncertainty_shrink):
        parser.error("--enable_policy_val_structure_aware_shrink requires --enable_policy_val_bin_uncertainty_shrink")
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
    local_alpha_modes = [
        bool(args.enable_policy_val_local_alpha_calibration),
        bool(args.enable_policy_val_face_alpha_calibration),
        bool(args.enable_policy_val_bin_alpha_calibration),
        bool(args.enable_policy_val_bin_rgb_alpha_calibration),
        bool(args.enable_policy_val_bin_uncertainty_shrink),
    ]
    if sum(int(flag) for flag in local_alpha_modes) > 1:
        parser.error(
            "enable at most one local alpha calibration mode: bucket, face, scalar bin, RGB bin, or uncertainty shrink"
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

    def build_policy_candidate(
        fill_mode: str,
        texture_size: int,
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
            f"fill={fill_mode} "
            f"prior_blend={float(surface_multiscale_prior_blend):.6g} "
            f"cap={float(max_abs_delta_rgb_candidate):.6g}"
        )
        print(f"[policy-candidate] start {candidate_label}", flush=True)
        cand_atlas, cand_fit_summary, _fit_views, cand_val_views = fit_atlas(
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
            teacher_distilled_basis_mode=str(args.teacher_distilled_basis_mode),
            teacher_distilled_basis_min_face_samples=int(args.teacher_distilled_basis_min_face_samples),
            teacher_distilled_basis_ridge=float(args.teacher_distilled_basis_ridge),
            teacher_distilled_basis_ood_max_z=float(args.teacher_distilled_basis_ood_max_z),
            teacher_distilled_basis_ood_min_std=float(args.teacher_distilled_basis_ood_min_std),
            teacher_distilled_basis_apply_mode=str(args.teacher_distilled_basis_apply_mode),
            teacher_distilled_basis_blend=float(args.teacher_distilled_basis_blend),
        )
        cand_fit_summary["support_mode"] = str(support_mode)
        cand_fit_summary["candidate_index"] = int(candidate_index)
        cand_fit_summary["candidate_count"] = int(candidate_count)
        cand_fit_summary["candidate_label"] = str(candidate_label)
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
            )
        alpha_candidates, ssim_alpha_refinement_summary = refine_alpha_grid_for_policy_val_ssim(
            alpha_candidates,
            enabled=bool(args.enable_policy_val_ssim_alpha_refinement),
            steps=int(args.policy_val_ssim_alpha_refinement_steps),
            min_alpha=float(args.policy_val_ssim_alpha_refinement_min_alpha),
        )
        cand_fit_summary["local_alpha_calibration"] = dict(local_alpha_profile)
        cand_fit_summary["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
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
            local_alpha_profile=local_alpha_profile,
            parent_edge_apply_profile=parent_edge_apply_profile,
        )
        cand_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
        cand_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
        cand_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
        cand_policy_val["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)

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
                local_alpha_profile=local_alpha_profile,
                parent_edge_apply_profile=parent_edge_apply_profile,
            )
            legacy_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
            legacy_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
            legacy_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
            legacy_policy_val["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
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
                local_alpha_profile=local_alpha_profile,
                parent_edge_apply_profile=parent_edge_apply_profile,
            )
            legacy_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
            legacy_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
            legacy_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
            legacy_policy_val["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
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
        face_gain_guard_profile: dict[str, Any] = {
            "enabled": False,
            "mode": "policy_val_face_gain_guard",
            "decision": "not_requested",
        }
        if bool(args.enable_policy_val_face_gain_guard):
            if (cand_accepted or guard_repair_seed_row) and float(guard_repair_seed_alpha) > 0.0:
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
                        local_alpha_profile=local_alpha_profile,
                        face_gain_guard_profile=face_gain_guard_profile,
                        parent_edge_apply_profile=parent_edge_apply_profile,
                    )
                    guarded_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
                    guarded_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
                    guarded_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
                    guarded_policy_val["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
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
        if bool(args.enable_policy_val_bin_uncertainty_guard):
            if (cand_accepted or guard_repair_seed_row) and float(guard_repair_seed_alpha) > 0.0:
                bin_uncertainty_guard_profile = build_policy_val_bin_uncertainty_guard_profile(
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
                        local_alpha_profile=local_alpha_profile,
                        face_gain_guard_profile=face_gain_guard_profile,
                        bin_uncertainty_guard_profile=bin_uncertainty_guard_profile,
                        parent_edge_apply_profile=parent_edge_apply_profile,
                    )
                    guarded_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
                    guarded_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
                    guarded_policy_val["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
                    guarded_policy_val["ssim_alpha_refinement"] = dict(ssim_alpha_refinement_summary)
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
        cand_fit_summary["face_gain_guard"] = dict(face_gain_guard_profile)
        cand_fit_summary["bin_uncertainty_guard"] = dict(bin_uncertainty_guard_profile)
        cand_fit_summary["parent_edge_apply_shrink"] = dict(parent_edge_apply_profile)
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
            for mode in fill_mode_candidates:
                for surface_multiscale_prior_blend in surface_multiscale_prior_blend_candidates:
                    for max_abs_delta_rgb_candidate in max_abs_delta_rgb_candidates:
                        support_summary = dict(support_candidate["summary"])
                        support_summary.setdefault("support_faces_sha1", support_faces_digest)
                        candidate_specs.append(
                            {
                                "fill_mode": str(mode),
                                "texture_size": int(texture_size),
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
        )

    def select_policy_val_payload_for_candidate(
        policy_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], float, list[str], bool]:
        return select_policy_val_payload_by_risk_gate(policy_payload, args)

    def policy_metric_score(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        cand_best = dict(candidate.get("best") or {})
        return (
            float(cand_best.get("relative_gain", -1.0)),
            float(cand_best.get("ssim_gain", 0.0)),
            float(cand_best.get("image_l1_gain", 0.0)),
            float(cand_best.get("cvar20_view_relative_gain", -1.0)),
            float(cand_best.get("min_view_relative_gain", -1.0)),
            float(cand_best.get("image_l1_cvar20_view_gain", -1.0)),
            float(cand_best.get("image_l1_min_view_gain", -1.0)),
            float((candidate.get("support_summary") or {}).get("added_faces", 0)),
            -float(candidate.get("texture_size", 0)),
            -float(candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb)),
        )

    def target_support_score(candidate: dict[str, Any]) -> tuple[float, float, float, float, float]:
        profile = dict(candidate.get("target_support_profile") or {})
        if not bool(profile.get("enabled", False)):
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        return (
            float(profile.get("changed_fraction", 0.0)),
            float(profile.get("cvar20_view_changed_fraction", 0.0)),
            float(profile.get("min_view_changed_fraction", 0.0)),
            float(profile.get("valid_fraction", 0.0)),
            float(profile.get("mean_abs_delta", 0.0)),
        )

    def target_support_score_dict(candidate: dict[str, Any]) -> dict[str, float]:
        profile = dict(candidate.get("target_support_profile") or {})
        return {
            "changed_fraction": float(profile.get("changed_fraction", 0.0) or 0.0),
            "cvar20_view_changed_fraction": float(
                profile.get("cvar20_view_changed_fraction", 0.0) or 0.0
            ),
            "min_view_changed_fraction": float(profile.get("min_view_changed_fraction", 0.0) or 0.0),
            "valid_fraction": float(profile.get("valid_fraction", 0.0) or 0.0),
            "mean_abs_delta": float(profile.get("mean_abs_delta", 0.0) or 0.0),
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
        fill_mode_selection = {
            "mode": "fixed",
            "selected_fill_mode": str(selected_candidate.get("fill_mode", "")),
            "selected_texture_size": int(selected_candidate.get("texture_size", 0)),
            "selected_surface_multiscale_prior_blend": float(
                selected_candidate.get("surface_multiscale_prior_blend", 0.0)
            ),
            "selected_max_abs_delta_rgb": float(
                selected_candidate.get("max_abs_delta_rgb", args.max_abs_delta_rgb)
            ),
            "selected_support_mode": str(selected_candidate.get("support_mode", "")),
            "accepted_candidate_count": int(bool(selected_candidate.get("accepted", False))),
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
        )
        min_changed = float(args.min_target_changed_fraction)
        if min_changed > 0.0 and float(target_apply.get("changed_fraction", 0.0)) < min_changed:
            accepted = False
            reject_reason = (
                f"target_changed_fraction {float(target_apply.get('changed_fraction', 0.0)):.8f} "
                f"< min_target_changed_fraction {min_changed:.8f}"
            )
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
            "selected_positive_view_fraction": float(best.get("positive_view_fraction", 0.0)),
            "selected_cvar20_view_relative_gain": float(best.get("cvar20_view_relative_gain", 0.0)),
            "selected_min_view_relative_gain": float(best.get("min_view_relative_gain", 0.0)),
            "selected_ssim_gain": float(best.get("ssim_gain", 0.0)),
            "selected_ssim_positive_view_fraction": float(best.get("ssim_positive_view_fraction", 0.0)),
            "selected_ssim_min_view_gain": float(best.get("ssim_min_view_gain", 0.0)),
            "selected_image_l1_gain": float(best.get("image_l1_gain", 0.0)),
            "selected_image_l1_positive_view_fraction": float(best.get("image_l1_positive_view_fraction", 0.0)),
            "selected_image_l1_min_view_gain": float(best.get("image_l1_min_view_gain", 0.0)),
            "selected_image_l1_cvar20_view_gain": float(best.get("image_l1_cvar20_view_gain", 0.0)),
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
