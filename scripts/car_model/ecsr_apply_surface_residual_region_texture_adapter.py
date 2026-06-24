#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def expanded_candidate_faces_from_ranked_rows(
    base_faces: set[int],
    ranked_rows: list[dict[str, Any]],
    rank_summary: dict[str, Any],
    max_extra_faces: int,
) -> tuple[set[int], dict[str, Any]]:
    if int(max_extra_faces) <= 0:
        return set(base_faces), {
            "enabled": False,
            "reason": "max_extra_faces <= 0",
            "mode": "fit_residual_topk",
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
        "mode": "fit_residual_topk",
        "base_faces": int(len(base_faces)),
        "added_faces": int(len(selected)),
        "candidate_faces_after_expansion": int(len(expanded)),
        "eligible_extra_faces": int(rank_summary.get("eligible_extra_faces", len(ranked_rows))),
        "fit_view_count": int(rank_summary.get("fit_view_count", 0)),
        "skipped_policy_val_views": int(rank_summary.get("skipped_policy_val_views", 0)),
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
    raise ValueError(f"unsupported teacher-distilled basis mode: {mode}")


def _teacher_distilled_basis_features_for_mask(
    z: np.lib.npyio.NpzFile,
    mode: str,
    mask: np.ndarray,
) -> np.ndarray | None:
    mode = str(mode)
    if mode == "none":
        return None
    if mode != "face_uv_normal_camera_ridge":
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
        ],
        axis=1,
    )


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
        local_alpha = _local_alpha_for_samples(
            base_pixels,
            local_alpha_profile,
            face_ids=faces[confident_indices],
            bin_ids=(vbin[confident_indices].astype(np.int64) * int(tex.shape[0]))
            + ubin[confident_indices].astype(np.int64),
        )
        if local_alpha_profile and bool(local_alpha_profile.get("enabled", False)):
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
    for c in range(3):
        channel = delta[c]
        if bool(np.any(support_valid)):
            channel[ys[support_valid], xs[support_valid]] = float(alpha) * out_pixels[support_valid, c]
    return delta, valid


def evaluate_policy_val(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    residual_rgb_key: str,
    residual_l1_key: str,
    alpha_grid: list[float],
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
    enable_policy_val_image_ssim: bool,
    policy_val_ssim_max_size: int,
    enable_policy_val_image_l1: bool,
    policy_val_l1_max_size: int,
    local_alpha_profile: dict[str, Any] | None = None,
    face_gain_guard_profile: dict[str, Any] | None = None,
    bin_uncertainty_guard_profile: dict[str, Any] | None = None,
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
            )
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
                adapted = np.clip(np.asarray(z["rgb_render"], dtype=np.float32) + np.clip(pred, -1.0, 1.0), 0.0, 1.0)
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


def build_policy_val_face_gain_guard_profile(
    val_views: list[Path],
    atlas: dict[int, FaceAtlas],
    residual_rgb_key: str,
    residual_l1_key: str,
    alpha: float,
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
    local_alpha = audit.get("local_alpha_profile", {})
    face_gain = audit.get("face_gain_guard_profile", {})
    bin_uncertainty = audit.get("bin_uncertainty_guard_profile", {})
    teacher_basis = audit.get("fit_summary", {}).get("teacher_distilled_basis", {})
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
        f"- atlas faces: `{audit.get('fit_summary', {}).get('atlas_faces', 0)}`",
        f"- fit samples: `{audit.get('fit_summary', {}).get('fit_samples', 0)}`",
        f"- selected support mode: `{audit.get('fit_summary', {}).get('selected_support_mode', '')}`",
        f"- selected support added faces: `{audit.get('fit_summary', {}).get('selected_support_added_faces', 0)}`",
        f"- selected texture size: `{audit.get('fit_summary', {}).get('selected_texture_size', audit.get('fit_summary', {}).get('texture_size', 0))}`",
        f"- selected fill mode: `{audit.get('fit_summary', {}).get('selected_atlas_empty_bin_fill_mode', audit.get('fit_summary', {}).get('atlas_empty_bin_fill_mode', ''))}`",
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
        choices=("none", "fit_residual_topk"),
        default="none",
        help=(
            "Optional train-fit-only support expansion. fit_residual_topk adds high-residual faces "
            "from atlas fit views while excluding policy-val views; target/test GT is never used."
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
        choices=("none", "face_uv_normal_camera_ridge"),
        default="none",
        help=(
            "Optional v65 teacher-distilled shared residual field. "
            "face_uv_normal_camera_ridge fits one ridge residual model per face using "
            "Phase-J teacher residuals and features [camera, normal, normal-dot-camera, UV polynomial]."
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
    local_alpha_modes = [
        bool(args.enable_policy_val_local_alpha_calibration),
        bool(args.enable_policy_val_face_alpha_calibration),
        bool(args.enable_policy_val_bin_alpha_calibration),
        bool(args.enable_policy_val_bin_rgb_alpha_calibration),
    ]
    if sum(int(flag) for flag in local_alpha_modes) > 1:
        parser.error(
            "enable at most one local alpha calibration mode: bucket, face, scalar bin, or RGB bin"
        )
    if int(args.bin_rgb_alpha_calibration_min_bin_samples) <= 0:
        parser.error("--bin_rgb_alpha_calibration_min_bin_samples must be > 0")
    if not 0.0 <= float(args.bin_rgb_alpha_calibration_min_positive_view_fraction) <= 1.0:
        parser.error("--bin_rgb_alpha_calibration_min_positive_view_fraction must be in [0, 1]")
    if float(args.bin_rgb_alpha_calibration_max_alpha) < float(args.bin_rgb_alpha_calibration_min_alpha):
        parser.error("--bin_rgb_alpha_calibration_max_alpha must be >= --bin_rgb_alpha_calibration_min_alpha")
    source_model = Path(args.source_model)
    output_model = Path(args.output_model)
    fit_evidence = Path(args.fit_evidence_dir)
    target_evidence = Path(args.target_evidence_dir)
    carrier_json = Path(args.region_carrier_json)

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
    if str(args.support_expansion_mode) == "fit_residual_topk":
        try:
            support_extra_candidates = parse_int_candidates(
                str(args.support_expansion_max_extra_faces_candidates),
                int(args.support_expansion_max_extra_faces),
            )
        except ValueError as exc:
            parser.error(str(exc))
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
                "fit_residual_topk"
                if len(support_extra_candidates) == 1
                else f"fit_residual_topk_{int(max_extra_faces)}"
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
                "mode": "fit_residual_topk_ladder",
                "base_faces": int(len(base_candidate_faces)),
                "eligible_extra_faces": int(rank_summary.get("eligible_extra_faces", len(ranked_extra_faces))),
                "fit_view_count": int(rank_summary.get("fit_view_count", 0)),
                "skipped_policy_val_views": int(rank_summary.get("skipped_policy_val_views", 0)),
                "min_face_samples": int(rank_summary.get("min_face_samples", 0)),
                "min_mean_l1": float(rank_summary.get("min_mean_l1", 0.0)),
                "max_extra_faces_candidates": [int(x) for x in support_extra_candidates],
                "candidates": expansion_candidates,
                "rank_preview": list(rank_summary.get("rank_preview", [])),
            }
    carrier_summary["support_expansion"] = support_expansion_summary

    try:
        texture_size_candidates = parse_int_candidates(str(args.texture_size_candidates), int(args.texture_size))
    except ValueError as exc:
        parser.error(str(exc))

    def build_policy_candidate(
        fill_mode: str,
        texture_size: int,
        support_mode: str,
        support_faces: set[int],
        support_summary: dict[str, Any],
    ) -> dict[str, Any]:
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
        cand_fit_summary["support_base_faces"] = int(support_summary.get("base_faces", len(base_candidate_faces)))
        cand_fit_summary["support_added_faces"] = int(support_summary.get("added_faces", 0))
        cand_fit_summary["support_candidate_faces"] = int(len(support_faces))
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
        cand_fit_summary["local_alpha_calibration"] = dict(local_alpha_profile)
        cand_policy_val = evaluate_policy_val(
            cand_val_views,
            cand_atlas,
            residual_rgb_key=str(args.residual_rgb_key),
            residual_l1_key=str(args.residual_l1_key),
            alpha_grid=alpha_candidates,
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
            enable_policy_val_image_ssim=bool(args.enable_policy_val_image_ssim_gate),
            policy_val_ssim_max_size=int(args.policy_val_ssim_max_size),
            enable_policy_val_image_l1=bool(args.enable_policy_val_image_l1_gate),
            policy_val_l1_max_size=int(args.policy_val_l1_max_size),
            local_alpha_profile=local_alpha_profile,
        )
        cand_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
        cand_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)

        def select_policy_val_payload(
            policy_payload: dict[str, Any],
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
                    policy_payload["selection"] = {
                        "mode": "risk_gate",
                        "safe_alpha_count": int(len(safe_rows)),
                        "selected_alpha": float(best_row.get("alpha", 0.0)),
                    }
                else:
                    selected_alpha_override = 0.0
                    policy_payload["selection"] = {
                        "mode": "risk_gate",
                        "safe_alpha_count": 0,
                        "selected_alpha": 0.0,
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
            )
            legacy_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
            legacy_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
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
            )
            legacy_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
            legacy_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
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
        face_gain_guard_profile: dict[str, Any] = {
            "enabled": False,
            "mode": "policy_val_face_gain_guard",
            "decision": "not_requested",
        }
        if bool(args.enable_policy_val_face_gain_guard):
            if cand_accepted and float(cand_selected_alpha) > 0.0:
                face_gain_guard_profile = build_policy_val_face_gain_guard_profile(
                    cand_val_views,
                    cand_atlas,
                    residual_rgb_key=str(args.residual_rgb_key),
                    residual_l1_key=str(args.residual_l1_key),
                    alpha=float(cand_selected_alpha),
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
                    local_alpha_profile=local_alpha_profile,
                    min_face_samples=int(args.face_gain_guard_min_face_samples),
                    min_relative_gain=float(args.face_gain_guard_min_relative_gain),
                    min_positive_view_fraction=float(args.face_gain_guard_min_positive_view_fraction),
                )
                if bool(face_gain_guard_profile.get("enabled", False)):
                    guarded_policy_val = evaluate_policy_val(
                        cand_val_views,
                        cand_atlas,
                        residual_rgb_key=str(args.residual_rgb_key),
                        residual_l1_key=str(args.residual_l1_key),
                        alpha_grid=alpha_candidates,
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
                        enable_policy_val_image_ssim=bool(args.enable_policy_val_image_ssim_gate),
                        policy_val_ssim_max_size=int(args.policy_val_ssim_max_size),
                        enable_policy_val_image_l1=bool(args.enable_policy_val_image_l1_gate),
                        policy_val_l1_max_size=int(args.policy_val_l1_max_size),
                        local_alpha_profile=local_alpha_profile,
                        face_gain_guard_profile=face_gain_guard_profile,
                    )
                    guarded_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
                    guarded_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
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
            if cand_accepted and float(cand_selected_alpha) > 0.0:
                bin_uncertainty_guard_profile = build_policy_val_bin_uncertainty_guard_profile(
                    cand_val_views,
                    cand_atlas,
                    residual_rgb_key=str(args.residual_rgb_key),
                    residual_l1_key=str(args.residual_l1_key),
                    alpha=float(cand_selected_alpha),
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
                    local_alpha_profile=local_alpha_profile,
                    face_gain_guard_profile=face_gain_guard_profile,
                    min_bin_samples=int(args.bin_uncertainty_guard_min_bin_samples),
                    min_relative_gain=float(args.bin_uncertainty_guard_min_relative_gain),
                    min_positive_view_fraction=float(args.bin_uncertainty_guard_min_positive_view_fraction),
                    max_mean_variance=float(args.bin_uncertainty_guard_max_mean_variance),
                    min_mean_sign_consistency=float(args.bin_uncertainty_guard_min_mean_sign_consistency),
                )
                if bool(bin_uncertainty_guard_profile.get("enabled", False)):
                    guarded_policy_val = evaluate_policy_val(
                        cand_val_views,
                        cand_atlas,
                        residual_rgb_key=str(args.residual_rgb_key),
                        residual_l1_key=str(args.residual_l1_key),
                        alpha_grid=alpha_candidates,
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
                        enable_policy_val_image_ssim=bool(args.enable_policy_val_image_ssim_gate),
                        policy_val_ssim_max_size=int(args.policy_val_ssim_max_size),
                        enable_policy_val_image_l1=bool(args.enable_policy_val_image_l1_gate),
                        policy_val_l1_max_size=int(args.policy_val_l1_max_size),
                        local_alpha_profile=local_alpha_profile,
                        face_gain_guard_profile=face_gain_guard_profile,
                        bin_uncertainty_guard_profile=bin_uncertainty_guard_profile,
                    )
                    guarded_policy_val["alpha_calibration"] = dict(alpha_calibration_summary)
                    guarded_policy_val["local_alpha_calibration"] = dict(local_alpha_profile)
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
        return {
            "fill_mode": str(fill_mode),
            "texture_size": int(texture_size),
            "support_mode": str(support_mode),
            "support_summary": dict(support_summary),
            "atlas": cand_atlas,
            "fit_summary": cand_fit_summary,
            "policy_val": cand_policy_val,
            "best": cand_best,
            "selected_alpha": float(cand_selected_alpha),
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
    candidate_runs = [
        build_policy_candidate(
            mode,
            texture_size,
            str(support_candidate["support_mode"]),
            set(support_candidate["faces"]),
            dict(support_candidate["summary"]),
        )
        for support_candidate in support_candidate_sets
        for texture_size in texture_size_candidates
        for mode in fill_mode_candidates
    ]

    def policy_candidate_score(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float, float]:
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
        )

    policy_auto_enabled = (
        requested_fill_mode == "auto_policy"
        or len(texture_size_candidates) > 1
        or len(support_candidate_sets) > 1
    )
    if policy_auto_enabled:
        accepted_candidates = [candidate for candidate in candidate_runs if bool(candidate.get("accepted", False))]
        selectable = accepted_candidates if accepted_candidates else candidate_runs
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
                (candidate for candidate in accepted_candidates if str(candidate.get("fill_mode", "")) == "face_mean"),
                None,
            )
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
        fill_mode_selection = {
            "mode": "auto_policy",
            "selected_fill_mode": str(selected_candidate.get("fill_mode", "")),
            "selected_texture_size": int(selected_candidate.get("texture_size", 0)),
            "selected_support_mode": str(selected_candidate.get("support_mode", "")),
            "accepted_candidate_count": int(len(accepted_candidates)),
            "guard": "baseline_basecarrier_face_mean_nonregressive_relative_ssim_l1_cvar_min_view",
            "texture_size_candidates": [int(x) for x in texture_size_candidates],
            "score_order": [
                {
                    "fill_mode": str(candidate.get("fill_mode", "")),
                    "texture_size": int(candidate.get("texture_size", 0)),
                    "support_mode": str(candidate.get("support_mode", "")),
                    "support_added_faces": int((candidate.get("support_summary") or {}).get("added_faces", 0)),
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
                }
                for candidate in sorted(candidate_runs, key=policy_candidate_score, reverse=True)
            ],
        }
    else:
        selected_candidate = candidate_runs[0]
        fill_mode_selection = {
            "mode": "fixed",
            "selected_fill_mode": str(selected_candidate.get("fill_mode", "")),
            "selected_texture_size": int(selected_candidate.get("texture_size", 0)),
            "selected_support_mode": str(selected_candidate.get("support_mode", "")),
            "accepted_candidate_count": int(bool(selected_candidate.get("accepted", False))),
        }

    atlas = selected_candidate["atlas"]
    fit_summary = dict(selected_candidate["fit_summary"])
    policy_val = dict(selected_candidate["policy_val"])
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
    policy_val["fill_mode_selection"] = fill_mode_selection
    fit_summary["requested_atlas_empty_bin_fill_mode"] = requested_fill_mode
    fit_summary["selected_atlas_empty_bin_fill_mode"] = str(selected_candidate.get("fill_mode", ""))
    fit_summary["requested_texture_size"] = int(args.texture_size)
    fit_summary["texture_size_candidates"] = [int(x) for x in texture_size_candidates]
    fit_summary["selected_texture_size"] = int(selected_candidate.get("texture_size", int(args.texture_size)))
    fit_summary["selected_support_mode"] = str(selected_candidate.get("support_mode", ""))
    fit_summary["selected_support_added_faces"] = int(
        (selected_candidate.get("support_summary") or {}).get("added_faces", 0)
    )
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
            max_abs_delta_rgb=float(args.max_abs_delta_rgb),
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
                "support_mode": str(candidate.get("support_mode", "")),
                "support_summary": dict(candidate.get("support_summary", {})),
                "accepted": bool(candidate.get("accepted", False)),
                "selected_alpha": float(candidate.get("selected_alpha", 0.0)),
                "local_alpha_calibration": dict(candidate.get("local_alpha_profile", {"enabled": False})),
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
