#!/usr/bin/env python3
"""Fit train-only residual deltas onto local 4-split surface subdivisions.

This operator is a renderer-compatible proxy for a tiny per-face texture.  It
does not add an image-space adapter.  Instead, it replaces selected high-error
triangles with four coplanar sub-triangles and stores a bounded residual color
delta on the three new midpoint vertices.  Candidate faces are accepted only
when a deterministic train-cache policy-validation split improves the local
residual reconstruction proxy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import copy_model_metadata, checkpoint_path, validate_faces
from utils.sh_utils import C0, C1


FACE_KEYS = ("importance_score", "image_size", "pixel_count")
TOPOLOGY_TENSOR_KEYS = ("triangles_points", "_triangle_indices")
ATTRIBUTE_TENSOR_KEYS = ("features_dc", "features_rest")
LUMA_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model", type=Path, required=True)
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--top_k", type=int, default=1024)
    parser.add_argument("--min_view_hits", type=int, default=2)
    parser.add_argument("--min_consistency", type=float, default=0.88)
    parser.add_argument("--min_pixel_count", type=float, default=8.0)
    parser.add_argument("--max_samples_per_face_view", type=int, default=64)
    parser.add_argument("--high_error_quantile", type=float, default=0.70)
    parser.add_argument("--min_alpha", type=float, default=0.05)
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=0.050)
    parser.add_argument("--feature_mode", choices=("dc", "sh1"), default="dc")
    parser.add_argument(
        "--materialize_mode",
        choices=("subdivision", "vertex_delta"),
        default="subdivision",
        help="subdivision adds midpoint vertices; vertex_delta keeps topology fixed and edits existing vertex attributes.",
    )
    parser.add_argument(
        "--max_abs_sh_coeff",
        type=float,
        default=0.0,
        help="Bound for SH1 coefficient deltas. 0 derives it from max_abs_delta_rgb / C1.",
    )
    parser.add_argument("--lambda_ridge", type=float, default=2e-2)
    parser.add_argument("--min_fit_samples", type=int, default=24)
    parser.add_argument("--min_val_samples", type=int, default=12)
    parser.add_argument("--min_policy_val_relative_gain", type=float, default=0.05)
    parser.add_argument(
        "--policy_val_offsets",
        default="",
        help=(
            "Comma-separated train-only validation offsets for robust per-face "
            "cross-validation. Empty preserves the historical single offset 0."
        ),
    )
    parser.add_argument(
        "--min_policy_val_offsets",
        type=int,
        default=0,
        help="Minimum passing offsets required. 0 requires every requested offset.",
    )
    parser.add_argument(
        "--min_policy_val_offset_fraction",
        type=float,
        default=1.0,
        help="Minimum fraction of requested offsets that must pass for a face.",
    )
    parser.add_argument("--min_view_gain_views", type=int, default=0)
    parser.add_argument("--min_view_gain_relative_gain", type=float, default=0.0)
    parser.add_argument("--min_view_gain_samples", type=int, default=4)
    parser.add_argument("--min_view_gain_fraction", type=float, default=0.0)
    parser.add_argument("--luma_preserve", action="store_true")
    parser.add_argument("--min_luma_relative_gain", type=float, default=0.0)
    parser.add_argument("--max_mean_luma_shift", type=float, default=0.0)
    parser.add_argument("--luma_shrink_grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--luma_shrink_selection", choices=("min", "max"), default="min")
    parser.add_argument("--structure_preserve", action="store_true")
    parser.add_argument("--structure_weight_strength", type=float, default=2.0)
    parser.add_argument("--min_structure_relative_gain", type=float, default=0.0)
    parser.add_argument("--max_structure_mean_luma_shift", type=float, default=0.0)
    parser.add_argument("--structure_shrink_grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--structure_shrink_selection", choices=("min", "max"), default="max")
    parser.add_argument("--anchor_support", action="store_true")
    parser.add_argument("--anchor_max_error_quantile", type=float, default=0.35)
    parser.add_argument("--anchor_samples_per_face_view", type=int, default=0)
    parser.add_argument("--anchor_weight", type=float, default=0.25)
    parser.add_argument("--candidate_plan_out", type=Path, default=None)
    parser.add_argument("--materialize_plan_in", type=Path, default=None)
    parser.add_argument("--materialize_plan_limit", type=int, default=0)
    parser.add_argument("--max_faces_to_apply", type=int, default=512)
    parser.add_argument("--force_apply", action="store_true")
    parser.add_argument("--no_op_on_fail", action="store_true", default=True)
    parser.add_argument(
        "--min_effective_mean_relative_gain",
        type=float,
        default=-1.0e30,
        help=(
            "Reject candidates whose final train-only proxy mean relative gain across "
            "policy offsets is below this value. The default disables the gate."
        ),
    )
    parser.add_argument(
        "--min_effective_min_relative_gain",
        type=float,
        default=-1.0e30,
        help=(
            "Reject candidates whose worst final train-only proxy relative gain across "
            "policy offsets is below this value. The default disables the gate."
        ),
    )
    parser.add_argument(
        "--min_effective_delta_abs_mean",
        type=float,
        default=0.0,
        help=(
            "Reject candidates whose mean absolute attribute delta is below this value. "
            "Use this with render gates to avoid accepting near no-op edits."
        ),
    )
    parser.add_argument(
        "--allow_no_effect_accept",
        action="store_true",
        help="Allow an audit to remain accepted even when materialization changes no tracked checkpoint tensors.",
    )
    parser.add_argument(
        "--min_materialized_attribute_delta",
        type=float,
        default=1e-9,
        help="Minimum max absolute feature change counted as a real vertex_delta materialization effect.",
    )
    parser.add_argument(
        "--vertex_delta_min_incident_support_fraction",
        type=float,
        default=0.0,
        help=(
            "For materialize_mode=vertex_delta, require this fraction of faces incident "
            "to the edited vertices to be in the train-evidence support set. 0 disables."
        ),
    )
    parser.add_argument(
        "--vertex_delta_max_incident_faces",
        type=int,
        default=0,
        help=(
            "For materialize_mode=vertex_delta, reject candidate faces touching a vertex "
            "with more than this many incident faces. 0 disables."
        ),
    )
    return parser.parse_args()


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def read_selected_faces(
    csv_path: Path,
    *,
    top_k: int,
    min_view_hits: int,
    min_consistency: float,
    min_pixel_count: float,
) -> tuple[list[int], dict[int, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            view_hits = int(_float(row, "view_hits"))
            consistency = _float(row, "residual_consistency")
            pixel_count = _float(row, "pixel_count")
            if view_hits < int(min_view_hits):
                continue
            if consistency < float(min_consistency):
                continue
            if pixel_count < float(min_pixel_count):
                continue
            rows.append(
                {
                    "face_id": int(_float(row, "face_id")),
                    "score": _float(row, "score"),
                    "pixel_count": pixel_count,
                    "view_hits": view_hits,
                    "consistency": consistency,
                    "mean_l1_error": _float(row, "mean_l1_error"),
                }
            )
    rows.sort(key=lambda r: (float(r["score"]), float(r["pixel_count"])), reverse=True)
    rows = rows[: int(top_k)]
    stats = {
        int(row["face_id"]): {
            "score": float(row["score"]),
            "pixel_count": float(row["pixel_count"]),
            "view_hits": float(row["view_hits"]),
            "consistency": float(row["consistency"]),
            "mean_l1_error": float(row["mean_l1_error"]),
        }
        for row in rows
    }
    return [int(row["face_id"]) for row in rows], stats


def parse_policy_offsets(raw: str, *, stride: int) -> list[int]:
    stride = max(int(stride), 2)
    if not str(raw).strip():
        return [0]
    offsets: list[int] = []
    for item in str(raw).replace(" ", ",").split(","):
        if not item:
            continue
        offset = int(item) % stride
        if offset not in offsets:
            offsets.append(offset)
    return offsets or [0]


def parse_float_grid(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).replace(" ", ",").split(","):
        if not item:
            continue
        value = float(item)
        if value not in values:
            values.append(value)
    return sorted(values) or [0.0]


def split_view_paths(view_paths: list[Path], stride: int, offset: int = 0) -> tuple[list[Path], list[Path]]:
    if len(view_paths) < 3:
        return view_paths, view_paths
    stride = max(int(stride), 2)
    offset = int(offset) % stride
    fit: list[Path] = []
    val: list[Path] = []
    for idx, path in enumerate(view_paths):
        if idx % stride == offset:
            val.append(path)
        else:
            fit.append(path)
    if not fit or not val:
        return view_paths, view_paths
    return fit, val


def _basis_midpoint(bary: np.ndarray) -> np.ndarray:
    u = bary[:, 0]
    v = bary[:, 1]
    w = bary[:, 2]
    basis = np.stack([4.0 * u * v, 4.0 * v * w, 4.0 * w * u], axis=1)
    return np.clip(basis, 0.0, 1.0).astype(np.float32)


def _sh1_terms(directions: np.ndarray) -> np.ndarray:
    dirs = directions.astype(np.float32)
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8)
    x = dirs[:, 0]
    y = dirs[:, 1]
    z = dirs[:, 2]
    return np.stack(
        [
            np.full_like(x, float(C0), dtype=np.float32),
            -float(C1) * y,
            float(C1) * z,
            -float(C1) * x,
        ],
        axis=1,
    ).astype(np.float32)


def _subdivision_basis(
    bary: np.ndarray,
    *,
    feature_mode: str,
    materialize_mode: str = "subdivision",
    directions: np.ndarray | None = None,
) -> np.ndarray:
    if str(materialize_mode) == "vertex_delta":
        spatial_basis = np.clip(bary, 0.0, 1.0).astype(np.float32)
    else:
        spatial_basis = _basis_midpoint(bary)
    if str(feature_mode) == "dc":
        return spatial_basis
    if directions is None:
        raise RuntimeError("SH subdivision basis requires camera directions")
    sh_terms = _sh1_terms(directions)
    return (spatial_basis[:, :, None] * sh_terms[:, None, :]).reshape(bary.shape[0], -1).astype(np.float32)


def _as_hw(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):
            arr = np.mean(arr, axis=0)
        elif arr.shape[-1] in (1, 3, 4):
            arr = np.mean(arr, axis=-1)
        else:
            arr = np.squeeze(arr)
    if arr.ndim != 2:
        return np.zeros((1, 1), dtype=np.float32)
    return arr.astype(np.float32)


def _normalize01(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    vals = arr[finite]
    lo = float(np.percentile(vals, 5.0))
    hi = float(np.percentile(vals, 95.0))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = np.clip((arr - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32)


def _gradient_magnitude_hw(value: np.ndarray) -> np.ndarray:
    arr = _as_hw(value)
    if min(arr.shape) < 2:
        return np.zeros_like(arr, dtype=np.float32)
    gy, gx = np.gradient(arr.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _structure_risk_from_npz(z: Any, residual_l1: np.ndarray) -> np.ndarray:
    risk = np.zeros_like(residual_l1, dtype=np.float32)
    parts = 0
    if "texture" in z.files:
        tex = _as_hw(z["texture"])
        if tex.shape == risk.shape:
            risk += _normalize01(tex)
            parts += 1
    if "depth" in z.files:
        depth_grad = _gradient_magnitude_hw(z["depth"])
        if depth_grad.shape == risk.shape:
            risk += _normalize01(depth_grad)
            parts += 1
    if "normal" in z.files:
        normal = np.asarray(z["normal"], dtype=np.float32)
        if normal.ndim == 3:
            if normal.shape[0] in (3, 4):
                channels = [normal[i] for i in range(min(3, normal.shape[0]))]
            elif normal.shape[-1] in (3, 4):
                channels = [normal[..., i] for i in range(min(3, normal.shape[-1]))]
            else:
                channels = []
            if channels and channels[0].shape == risk.shape:
                normal_grad = np.zeros_like(risk, dtype=np.float32)
                for channel in channels:
                    normal_grad += _gradient_magnitude_hw(channel)
                risk += _normalize01(normal_grad)
                parts += 1
    if parts <= 0:
        return np.ones_like(residual_l1, dtype=np.float32)
    return np.clip(risk / float(parts), 0.0, 1.0).astype(np.float32)


def collect_samples(
    view_paths: list[Path],
    selected_faces: list[int],
    face_stats: dict[int, dict[str, float]],
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    feature_mode: str,
    materialize_mode: str,
    high_error_quantile: float,
    min_alpha: float,
    max_samples_per_face_view: int,
    anchor_support: bool,
    anchor_max_error_quantile: float,
    anchor_samples_per_face_view: int,
    anchor_weight: float,
    structure_preserve: bool,
    structure_weight_strength: float,
) -> dict[int, dict[str, list[np.ndarray]]]:
    selected = set(int(x) for x in selected_faces)
    samples: dict[int, dict[str, list[np.ndarray]]] = {
        int(fid): {"basis": [], "target": [], "weight": [], "structure_weight": []}
        for fid in selected_faces
    }
    for view_path in view_paths:
        with np.load(view_path) as z:
            required = {"face_id", "residual_l1", "alpha", "residual_rgb", "barycentric", "barycentric_valid"}
            if str(feature_mode) == "sh1":
                required.add("camera_center")
            missing = sorted(required - set(z.files))
            if missing:
                raise RuntimeError(f"{view_path} missing required subdivision evidence fields: {missing}")
            face_id = z["face_id"].astype(np.int64)
            residual_l1 = z["residual_l1"].astype(np.float32)
            alpha = z["alpha"].astype(np.float32)
            if alpha.ndim == 3:
                alpha = np.squeeze(alpha, axis=0)
            residual_rgb = z["residual_rgb"].astype(np.float32)
            barycentric = z["barycentric"].astype(np.float32)
            bary_valid = z["barycentric_valid"].astype(bool)
            camera_center = z["camera_center"].astype(np.float32).reshape(-1)[:3] if "camera_center" in z.files else None
            structure_risk = _structure_risk_from_npz(z, residual_l1) if bool(structure_preserve) else np.zeros_like(residual_l1, dtype=np.float32)

        threshold = float(np.quantile(residual_l1.reshape(-1), float(high_error_quantile)))

        def append_from_mask(mask: np.ndarray, *, cap_per_face: int, weight_scale: float) -> None:
            if cap_per_face <= 0 or not np.any(mask):
                return
            flat_faces = face_id[mask].reshape(-1)
            if flat_faces.size == 0:
                return
            present = sorted(set(int(x) for x in np.unique(flat_faces)) & selected)
            if not present:
                return
            ys_all, xs_all = np.nonzero(mask)
            for fid in present:
                local = flat_faces == int(fid)
                idx = np.nonzero(local)[0]
                if idx.size == 0:
                    continue
                cap = min(int(cap_per_face), int(idx.size))
                if idx.size > cap:
                    idx = idx[np.linspace(0, idx.size - 1, cap, dtype=np.int64)]
                ys = ys_all[idx]
                xs = xs_all[idx]
                bary = barycentric[:, ys, xs].T.astype(np.float32)
                inside = np.all((bary >= -0.25) & (bary <= 1.25), axis=1)
                if not np.any(inside):
                    continue
                ys = ys[inside]
                xs = xs[inside]
                bary = np.clip(bary[inside], 0.0, 1.0)
                bary = bary / np.maximum(bary.sum(axis=1, keepdims=True), 1e-8)
                directions = None
                if str(feature_mode) == "sh1":
                    face_vertex_ids = faces[int(fid)]
                    tri = vertices[face_vertex_ids]
                    positions = bary @ tri
                    directions = positions - camera_center[None, :]
                basis = _subdivision_basis(
                    bary,
                    feature_mode=str(feature_mode),
                    materialize_mode=str(materialize_mode),
                    directions=directions,
                )
                target = residual_rgb[:, ys, xs].T.astype(np.float32)
                l1 = residual_l1[ys, xs].astype(np.float32)
                consistency = float(face_stats.get(int(fid), {}).get("consistency", 1.0))
                base_weight = np.maximum(l1, 1e-4).astype(np.float32)
                if float(weight_scale) > 0.0:
                    base_weight = np.maximum(base_weight, float(weight_scale)).astype(np.float32)
                weight = base_weight * max(consistency, 1e-3)
                risk = structure_risk[ys, xs].astype(np.float32)
                structure_weight = weight * (1.0 + max(float(structure_weight_strength), 0.0) * np.clip(risk, 0.0, 1.0))
                samples[int(fid)]["basis"].append(basis)
                samples[int(fid)]["target"].append(target)
                samples[int(fid)]["weight"].append(weight)
                samples[int(fid)]["structure_weight"].append(structure_weight.astype(np.float32))

        high_mask = bary_valid & (residual_l1 >= threshold) & (alpha >= float(min_alpha))
        append_from_mask(high_mask, cap_per_face=int(max_samples_per_face_view), weight_scale=0.0)

        if bool(anchor_support):
            anchor_threshold = float(np.quantile(residual_l1.reshape(-1), float(anchor_max_error_quantile)))
            anchor_mask = bary_valid & (residual_l1 <= anchor_threshold) & (alpha >= float(min_alpha))
            # Anchors represent already-good pixels. Without a minimum weight,
            # their tiny residuals would not constrain the least-squares solve.
            anchor_scale = max(float(threshold), 1e-4) * max(float(anchor_weight), 0.0)
            append_from_mask(
                anchor_mask,
                cap_per_face=int(anchor_samples_per_face_view),
                weight_scale=float(anchor_scale),
            )
    return samples


def _pack_face_samples(
    face_samples: dict[str, list[np.ndarray]],
    *,
    weight_key: str = "weight",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not face_samples["basis"]:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    weights = face_samples.get(weight_key) or face_samples.get("weight", [])
    return (
        np.concatenate(face_samples["basis"], axis=0),
        np.concatenate(face_samples["target"], axis=0),
        np.concatenate(weights, axis=0),
    )


def fit_delta(
    basis_fit: np.ndarray,
    target_fit: np.ndarray,
    weight_fit: np.ndarray,
    basis_val: np.ndarray,
    target_val: np.ndarray,
    weight_val: np.ndarray,
    *,
    strength: float,
    max_abs_delta_rgb: float,
    coeff_bounds: np.ndarray | None,
    lambda_ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    y_fit = np.clip(target_fit * float(strength), -float(max_abs_delta_rgb), float(max_abs_delta_rgb))
    y_val = np.clip(target_val * float(strength), -float(max_abs_delta_rgb), float(max_abs_delta_rgb))
    wf = weight_fit.reshape(-1, 1).astype(np.float32)
    xtw = basis_fit.T @ (basis_fit * wf)
    rhs = basis_fit.T @ (y_fit * wf)
    basis_dim = int(basis_fit.shape[1])
    xtw = xtw + np.eye(basis_dim, dtype=np.float32) * float(lambda_ridge)
    try:
        delta = np.linalg.solve(xtw, rhs).astype(np.float32)
    except np.linalg.LinAlgError:
        delta = np.zeros((basis_dim, 3), dtype=np.float32)
    if coeff_bounds is None:
        delta = np.clip(delta, -float(max_abs_delta_rgb), float(max_abs_delta_rgb))
    else:
        bounds = np.asarray(coeff_bounds, dtype=np.float32).reshape(-1, 1)
        delta = np.clip(delta, -bounds, bounds)

    wv = weight_val.reshape(-1, 1).astype(np.float32)
    pred0 = np.zeros_like(y_val)
    pred = basis_val @ delta
    denom = float(np.maximum(wv.sum(), 1e-8))
    initial = float((((pred0 - y_val) ** 2) * wv).sum() / denom)
    final = float((((pred - y_val) ** 2) * wv).sum() / denom)
    gain = float((initial - final) / max(initial, 1e-8))
    stats = {
        "fit_samples": int(basis_fit.shape[0]),
        "val_samples": int(basis_val.shape[0]),
        "initial_val_mse": initial,
        "final_val_mse": final,
        "relative_gain": gain,
        "delta_abs_mean": float(np.abs(delta).mean()) if delta.size else 0.0,
        "delta_abs_max": float(np.abs(delta).max()) if delta.size else 0.0,
    }
    return delta, stats


def evaluate_delta(
    delta: np.ndarray,
    basis_val: np.ndarray,
    target_val: np.ndarray,
    weight_val: np.ndarray,
    *,
    fit_samples: int,
    strength: float,
    max_abs_delta_rgb: float,
) -> dict[str, float]:
    y_val = np.clip(target_val * float(strength), -float(max_abs_delta_rgb), float(max_abs_delta_rgb))
    wv = weight_val.reshape(-1, 1).astype(np.float32)
    pred0 = np.zeros_like(y_val)
    pred = basis_val @ delta
    denom = float(np.maximum(wv.sum(), 1e-8))
    initial = float((((pred0 - y_val) ** 2) * wv).sum() / denom)
    final = float((((pred - y_val) ** 2) * wv).sum() / denom)
    gain = float((initial - final) / max(initial, 1e-8))
    return {
        "fit_samples": int(fit_samples),
        "val_samples": int(basis_val.shape[0]),
        "initial_val_mse": initial,
        "final_val_mse": final,
        "relative_gain": gain,
        "delta_abs_mean": float(np.abs(delta).mean()) if delta.size else 0.0,
        "delta_abs_max": float(np.abs(delta).max()) if delta.size else 0.0,
    }


def evaluate_luma_delta(
    delta: np.ndarray,
    basis_val: np.ndarray,
    target_val: np.ndarray,
    weight_val: np.ndarray,
    *,
    strength: float,
    max_abs_delta_rgb: float,
) -> dict[str, float]:
    y_val = np.clip(target_val * float(strength), -float(max_abs_delta_rgb), float(max_abs_delta_rgb))
    pred = basis_val @ delta
    target_luma = y_val @ LUMA_WEIGHTS
    pred_luma = pred @ LUMA_WEIGHTS
    w = weight_val.astype(np.float32).reshape(-1)
    denom = float(np.maximum(w.sum(), 1e-8))
    initial = float(((target_luma**2) * w).sum() / denom)
    final = float((((pred_luma - target_luma) ** 2) * w).sum() / denom)
    gain = float((initial - final) / max(initial, 1e-8))
    mean_pred = float((pred_luma * w).sum() / denom)
    mean_abs_pred = float((np.abs(pred_luma) * w).sum() / denom)
    return {
        "initial_luma_mse": initial,
        "final_luma_mse": final,
        "luma_relative_gain": gain,
        "mean_pred_luma": mean_pred,
        "mean_abs_pred_luma": mean_abs_pred,
        "max_abs_pred_luma": float(np.abs(pred_luma).max()) if pred_luma.size else 0.0,
    }


def shrink_sh1_dc_luma(delta: np.ndarray, beta: float) -> np.ndarray:
    out = delta.astype(np.float32, copy=True)
    denom = float(np.dot(LUMA_WEIGHTS, LUMA_WEIGHTS))
    for row_idx in (0, 4, 8):
        coeff = out[row_idx]
        projection = float(np.dot(coeff, LUMA_WEIGHTS) / max(denom, 1e-8)) * LUMA_WEIGHTS
        out[row_idx] = coeff - float(beta) * projection
    return out


def shrink_sh1_all_luma(delta: np.ndarray, beta: float) -> np.ndarray:
    out = delta.astype(np.float32, copy=True)
    denom = float(np.dot(LUMA_WEIGHTS, LUMA_WEIGHTS))
    for row_idx in range(int(out.shape[0])):
        coeff = out[row_idx]
        projection = float(np.dot(coeff, LUMA_WEIGHTS) / max(denom, 1e-8)) * LUMA_WEIGHTS
        out[row_idx] = coeff - float(beta) * projection
    return out


def choose_structure_preserved_delta(
    delta: np.ndarray,
    offset_sets: list[dict[str, Any]],
    *,
    fid: int,
    args: argparse.Namespace,
    required_offsets: int,
    required_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    betas = [min(max(float(x), 0.0), 1.0) for x in parse_float_grid(args.structure_shrink_grid)]
    best: tuple[tuple[int, float, float], np.ndarray, dict[str, Any]] | None = None
    accepted_choice: tuple[np.ndarray, dict[str, Any]] | None = None
    for beta in betas:
        candidate = shrink_sh1_all_luma(delta, beta)
        rows: list[dict[str, Any]] = []
        gains: list[float] = []
        for offset_set in offset_sets:
            offset = int(offset_set["offset"])
            fit_samples = offset_set["fit_samples"]
            val_samples = offset_set["val_samples"]
            xf, _, _ = _pack_face_samples(fit_samples[int(fid)], weight_key="structure_weight")
            xv, yv, wv = _pack_face_samples(val_samples[int(fid)], weight_key="structure_weight")
            fit_count = int(xf.shape[0])
            if fit_count < int(args.min_fit_samples) or xv.shape[0] < int(args.min_val_samples):
                rows.append(
                    {
                        "offset": offset,
                        "accepted": False,
                        "decision_reasons": ["insufficient_structure_samples"],
                        "fit_samples": fit_count,
                        "val_samples": int(xv.shape[0]),
                    }
                )
                continue
            rgb_proxy = evaluate_delta(
                candidate,
                xv,
                yv,
                wv,
                fit_samples=fit_count,
                strength=float(args.strength),
                max_abs_delta_rgb=float(args.max_abs_delta_rgb),
            )
            luma_proxy = evaluate_luma_delta(
                candidate,
                xv,
                yv,
                wv,
                strength=float(args.strength),
                max_abs_delta_rgb=float(args.max_abs_delta_rgb),
            )
            reasons: list[str] = []
            if float(rgb_proxy["relative_gain"]) < float(args.min_structure_relative_gain):
                reasons.append("structure_relative_gain_below_threshold")
            if float(args.max_structure_mean_luma_shift) > 0.0 and abs(float(luma_proxy["mean_pred_luma"])) > float(args.max_structure_mean_luma_shift):
                reasons.append("structure_luma_shift_exceeds_threshold")
            if not reasons:
                gains.append(float(rgb_proxy["relative_gain"]))
            rows.append(
                {
                    "offset": offset,
                    "accepted": not reasons,
                    "decision_reasons": reasons,
                    "structure_rgb_proxy": rgb_proxy,
                    "structure_luma_proxy": luma_proxy,
                }
            )
        passing_count = sum(1 for row in rows if bool(row.get("accepted", False)))
        passing_fraction = float(passing_count / max(len(offset_sets), 1))
        summary = {
            "enabled": True,
            "mode": "sh1_all_luma_structure_shrink",
            "beta": float(beta),
            "accepted": bool(passing_count >= int(required_offsets) and passing_fraction >= float(required_fraction)),
            "passing_count": int(passing_count),
            "offset_count": int(len(offset_sets)),
            "passing_fraction": passing_fraction,
            "min_structure_relative_gain": float(args.min_structure_relative_gain),
            "max_structure_mean_luma_shift": float(args.max_structure_mean_luma_shift),
            "weight_strength": float(args.structure_weight_strength),
            "rows": rows,
        }
        if bool(summary["accepted"]):
            accepted_choice = (candidate, summary)
            if str(args.structure_shrink_selection) == "min":
                return candidate, summary
            continue
        score = (int(passing_count), float(np.mean(gains)) if gains else -1.0e9, -float(beta))
        if best is None or score > best[0]:
            best = (score, candidate, summary)
    if accepted_choice is not None:
        return accepted_choice
    if best is None:
        return delta, {"enabled": True, "mode": "sh1_all_luma_structure_shrink", "accepted": False, "rows": []}
    return best[1], best[2]


def choose_luma_preserved_delta(
    delta: np.ndarray,
    offset_sets: list[dict[str, Any]],
    *,
    fid: int,
    args: argparse.Namespace,
    required_offsets: int,
    required_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    betas = [min(max(float(x), 0.0), 1.0) for x in parse_float_grid(args.luma_shrink_grid)]
    best: tuple[tuple[int, float, float], np.ndarray, dict[str, Any]] | None = None
    accepted_choice: tuple[np.ndarray, dict[str, Any]] | None = None
    for beta in betas:
        candidate = shrink_sh1_dc_luma(delta, beta)
        rows: list[dict[str, Any]] = []
        gains: list[float] = []
        for offset_set in offset_sets:
            offset = int(offset_set["offset"])
            fit_samples = offset_set["fit_samples"]
            val_samples = offset_set["val_samples"]
            xf, _, _ = _pack_face_samples(fit_samples[int(fid)])
            xv, yv, wv = _pack_face_samples(val_samples[int(fid)])
            fit_count = int(xf.shape[0])
            if fit_count < int(args.min_fit_samples) or xv.shape[0] < int(args.min_val_samples):
                rows.append(
                    {
                        "offset": offset,
                        "accepted": False,
                        "decision_reasons": ["insufficient_samples"],
                        "fit_samples": fit_count,
                        "val_samples": int(xv.shape[0]),
                    }
                )
                continue
            rgb_proxy = evaluate_delta(
                candidate,
                xv,
                yv,
                wv,
                fit_samples=fit_count,
                strength=float(args.strength),
                max_abs_delta_rgb=float(args.max_abs_delta_rgb),
            )
            luma_proxy = evaluate_luma_delta(
                candidate,
                xv,
                yv,
                wv,
                strength=float(args.strength),
                max_abs_delta_rgb=float(args.max_abs_delta_rgb),
            )
            reasons: list[str] = []
            if float(rgb_proxy["relative_gain"]) < float(args.min_policy_val_relative_gain):
                reasons.append("relative_gain_below_threshold")
            if float(luma_proxy["luma_relative_gain"]) < float(args.min_luma_relative_gain):
                reasons.append("luma_gain_below_threshold")
            if float(args.max_mean_luma_shift) > 0.0 and abs(float(luma_proxy["mean_pred_luma"])) > float(args.max_mean_luma_shift):
                reasons.append("mean_luma_shift_exceeds_threshold")
            if not reasons:
                gains.append(float(luma_proxy["luma_relative_gain"]))
            rows.append(
                {
                    "offset": offset,
                    "accepted": not reasons,
                    "decision_reasons": reasons,
                    "rgb_proxy": rgb_proxy,
                    "luma_proxy": luma_proxy,
                }
            )
        passing_count = sum(1 for row in rows if bool(row.get("accepted", False)))
        passing_fraction = float(passing_count / max(len(offset_sets), 1))
        summary = {
            "enabled": True,
            "mode": "sh1_dc_luma_shrink",
            "beta": float(beta),
            "accepted": bool(passing_count >= int(required_offsets) and passing_fraction >= float(required_fraction)),
            "passing_count": int(passing_count),
            "offset_count": int(len(offset_sets)),
            "passing_fraction": passing_fraction,
            "rows": rows,
        }
        if bool(summary["accepted"]):
            accepted_choice = (candidate, summary)
            if str(args.luma_shrink_selection) == "min":
                return candidate, summary
            continue
        score = (int(passing_count), float(np.mean(gains)) if gains else -1.0e9, -float(beta))
        if best is None or score > best[0]:
            best = (score, candidate, summary)
    if accepted_choice is not None:
        return accepted_choice
    if best is None:
        return delta, {"enabled": True, "mode": "sh1_dc_luma_shrink", "accepted": False, "rows": []}
    return best[1], best[2]


def view_gain_certificate(
    delta: np.ndarray,
    face_samples: dict[str, list[np.ndarray]],
    *,
    strength: float,
    max_abs_delta_rgb: float,
    min_samples: int,
    min_relative_gain: float,
    min_views: int,
    min_fraction: float,
) -> dict[str, Any]:
    view_rows: list[dict[str, float]] = []
    basis_views = face_samples.get("basis", [])
    target_views = face_samples.get("target", [])
    weight_views = face_samples.get("weight", [])
    for basis, target, weight in zip(basis_views, target_views, weight_views):
        if int(basis.shape[0]) < int(min_samples):
            continue
        y = np.clip(target * float(strength), -float(max_abs_delta_rgb), float(max_abs_delta_rgb))
        w = weight.reshape(-1, 1).astype(np.float32)
        denom = float(np.maximum(w.sum(), 1e-8))
        pred = basis @ delta
        initial = float(((y**2) * w).sum() / denom)
        final = float((((pred - y) ** 2) * w).sum() / denom)
        gain = float((initial - final) / max(initial, 1e-8))
        view_rows.append(
            {
                "samples": int(basis.shape[0]),
                "initial_mse": initial,
                "final_mse": final,
                "relative_gain": gain,
            }
        )
    passing = [row for row in view_rows if float(row["relative_gain"]) >= float(min_relative_gain)]
    view_count = int(len(view_rows))
    passing_count = int(len(passing))
    fraction = float(passing_count / max(view_count, 1))
    if int(min_views) <= 0:
        passed = True
    else:
        passed = passing_count >= int(min_views) and fraction >= float(min_fraction)
    return {
        "passed": bool(passed),
        "view_count": view_count,
        "passing_count": passing_count,
        "passing_fraction": fraction,
        "min_views": int(min_views),
        "min_fraction": float(min_fraction),
        "min_relative_gain": float(min_relative_gain),
        "min_samples": int(min_samples),
        "views": view_rows,
    }


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
        else:
            out[key] = value
    return out


def _tensor_change_summary(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    min_abs_delta: float,
) -> dict[str, Any]:
    before_cpu = before.detach().cpu()
    after_cpu = after.detach().cpu()
    row: dict[str, Any] = {
        "before_shape": [int(x) for x in before_cpu.shape],
        "after_shape": [int(x) for x in after_cpu.shape],
        "shape_changed": tuple(before_cpu.shape) != tuple(after_cpu.shape),
        "num_changed": 0,
        "max_abs_delta": 0.0,
        "changed": False,
    }
    if bool(row["shape_changed"]):
        row["changed"] = True
        row["num_changed"] = None
        row["max_abs_delta"] = None
        return row
    if before_cpu.numel() == 0:
        return row
    if torch.is_floating_point(before_cpu) or torch.is_floating_point(after_cpu):
        delta = (after_cpu.float() - before_cpu.float()).abs()
        max_abs = float(delta.max().item()) if delta.numel() else 0.0
        row["max_abs_delta"] = max_abs
        row["num_changed"] = int((delta > float(min_abs_delta)).sum().item())
        row["changed"] = bool(max_abs > float(min_abs_delta))
    else:
        changed = after_cpu != before_cpu
        row["num_changed"] = int(changed.sum().item())
        row["max_abs_delta"] = float((after_cpu.long() - before_cpu.long()).abs().max().item()) if changed.any() else 0.0
        row["changed"] = bool(row["num_changed"])
    return row


def materialization_effect_summary(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    min_abs_attribute_delta: float,
) -> dict[str, Any]:
    topology: dict[str, Any] = {}
    attributes: dict[str, Any] = {}
    for key in TOPOLOGY_TENSOR_KEYS:
        if torch.is_tensor(before.get(key)) and torch.is_tensor(after.get(key)):
            topology[key] = _tensor_change_summary(before[key], after[key], min_abs_delta=0.0)
    for key in ATTRIBUTE_TENSOR_KEYS:
        if torch.is_tensor(before.get(key)) and torch.is_tensor(after.get(key)):
            attributes[key] = _tensor_change_summary(
                before[key],
                after[key],
                min_abs_delta=float(min_abs_attribute_delta),
            )
    topology_changed = any(bool(row.get("changed", False)) for row in topology.values())
    attribute_changed = any(bool(row.get("changed", False)) for row in attributes.values())
    max_attribute_delta = 0.0
    for row in attributes.values():
        value = row.get("max_abs_delta")
        if value is not None:
            max_attribute_delta = max(max_attribute_delta, float(value))
    return {
        "has_effect": bool(topology_changed or attribute_changed),
        "topology_changed": bool(topology_changed),
        "attribute_changed": bool(attribute_changed),
        "max_attribute_delta": float(max_attribute_delta),
        "min_abs_attribute_delta": float(min_abs_attribute_delta),
        "topology": topology,
        "attributes": attributes,
    }


def build_vertex_face_adjacency(faces: np.ndarray) -> list[set[int]]:
    if faces.size == 0:
        return []
    vertex_count = int(np.max(faces)) + 1
    vertex_faces: list[set[int]] = [set() for _ in range(max(vertex_count, 0))]
    for face_id, ids in enumerate(faces.astype(np.int64)):
        for vertex_id in ids.tolist():
            if vertex_id < 0:
                continue
            if vertex_id >= len(vertex_faces):
                vertex_faces.extend(set() for _ in range(vertex_id + 1 - len(vertex_faces)))
            vertex_faces[int(vertex_id)].add(int(face_id))
    return vertex_faces


def vertex_delta_spillover_certificate(
    *,
    face_id: int,
    faces: np.ndarray,
    vertex_faces: list[set[int]],
    support_faces: set[int],
    min_support_fraction: float,
    max_incident_faces: int,
) -> dict[str, Any]:
    enabled = float(min_support_fraction) > 0.0 or int(max_incident_faces) > 0
    if face_id < 0 or face_id >= int(faces.shape[0]):
        return {
            "enabled": bool(enabled),
            "passed": False,
            "decision_reasons": ["face_id_out_of_range"],
        }
    incident: set[int] = set()
    max_vertex_incident = 0
    for vertex_id in faces[int(face_id)].astype(np.int64).tolist():
        local = vertex_faces[int(vertex_id)] if 0 <= int(vertex_id) < len(vertex_faces) else set()
        max_vertex_incident = max(max_vertex_incident, len(local))
        incident.update(local)
    supported = incident & support_faces
    incident_count = int(len(incident))
    support_fraction = float(len(supported) / max(incident_count, 1))
    reasons: list[str] = []
    if float(min_support_fraction) > 0.0 and support_fraction < float(min_support_fraction):
        reasons.append("incident_support_fraction_below_threshold")
    if int(max_incident_faces) > 0 and max_vertex_incident > int(max_incident_faces):
        reasons.append("vertex_incident_faces_exceeds_threshold")
    return {
        "enabled": bool(enabled),
        "passed": not reasons,
        "decision_reasons": reasons,
        "incident_face_count": incident_count,
        "supported_incident_face_count": int(len(supported)),
        "support_fraction": support_fraction,
        "max_vertex_incident_faces": int(max_vertex_incident),
        "min_support_fraction": float(min_support_fraction),
        "max_incident_faces": int(max_incident_faces),
    }


def effective_proxy_certificate(proxy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    min_mean = float(args.min_effective_mean_relative_gain)
    min_min = float(args.min_effective_min_relative_gain)
    min_delta = float(args.min_effective_delta_abs_mean)
    enabled = bool(min_mean > -1.0e29 or min_min > -1.0e29 or min_delta > 0.0)
    reasons: list[str] = []
    mean_gain = float(proxy.get("mean_relative_gain", 0.0))
    min_gain = float(proxy.get("relative_gain", 0.0))
    delta_abs_mean = float(proxy.get("delta_abs_mean", 0.0))
    if min_mean > -1.0e29 and mean_gain < min_mean:
        reasons.append("effective_mean_relative_gain_below_threshold")
    if min_min > -1.0e29 and min_gain < min_min:
        reasons.append("effective_min_relative_gain_below_threshold")
    if min_delta > 0.0 and delta_abs_mean < min_delta:
        reasons.append("effective_delta_abs_mean_below_threshold")
    return {
        "enabled": enabled,
        "passed": not reasons,
        "decision_reasons": reasons,
        "mean_relative_gain": mean_gain,
        "min_relative_gain": min_gain,
        "delta_abs_mean": delta_abs_mean,
        "thresholds": {
            "min_effective_mean_relative_gain": min_mean,
            "min_effective_min_relative_gain": min_min,
            "min_effective_delta_abs_mean": min_delta,
        },
    }


def read_candidate_plan(path: Path, *, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    meta: dict[str, Any] = {}
    if isinstance(plan, list):
        rows = plan
    elif isinstance(plan, dict):
        meta = plan
        if isinstance(plan.get("candidates"), list):
            rows = plan["candidates"]
        elif isinstance(plan.get("accepted"), list):
            rows = plan["accepted"]
        elif isinstance(plan.get("accepted_preview"), list):
            rows = plan["accepted_preview"]
        else:
            rows = []
    else:
        rows = []
    rows = [row for row in rows if isinstance(row, dict)]
    if int(limit) > 0:
        rows = rows[: int(limit)]
    return rows, meta


def materialize_subdivision(
    state: dict[str, Any],
    accepted: list[dict[str, Any]],
    *,
    feature_mode: str,
) -> dict[str, Any]:
    out = clone_state(state)
    vertices = state["triangles_points"].detach().cpu().float()
    faces = state["_triangle_indices"].detach().cpu().long()
    features_dc = state["features_dc"].detach().cpu().float()
    features_rest = state["features_rest"].detach().cpu().float()
    vertex_weight = state["vertex_weight"].detach().cpu().float()

    accepted_by_face = {int(row["face_id"]): row for row in accepted}
    remove_mask = torch.zeros((faces.shape[0],), dtype=torch.bool)
    new_vertices: list[torch.Tensor] = []
    new_fdc: list[torch.Tensor] = []
    new_frest: list[torch.Tensor] = []
    new_weight: list[torch.Tensor] = []
    new_faces: list[torch.Tensor] = []
    face_source_ids: list[int] = []
    next_vertex = int(vertices.shape[0])

    for face_id, row in accepted_by_face.items():
        if face_id < 0 or face_id >= int(faces.shape[0]):
            continue
        ids = faces[face_id].long()
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= int(vertices.shape[0]):
            continue
        remove_mask[face_id] = True
        a, b, c = [int(x) for x in ids.tolist()]
        edge_pairs = [(a, b), (b, c), (c, a)]
        delta = torch.as_tensor(row.get("delta_coeff", row.get("delta_rgb")), dtype=torch.float32)
        if str(feature_mode) == "sh1":
            delta = delta.view(3, 4, 3)
        mids: list[int] = []
        for edge_idx, (u, v) in enumerate(edge_pairs):
            mids.append(next_vertex)
            next_vertex += 1
            new_vertices.append((vertices[u] + vertices[v]) * 0.5)
            if str(feature_mode) == "sh1":
                new_fdc.append((features_dc[u] + features_dc[v]) * 0.5 + delta[edge_idx, 0].view(1, 3))
                rest = (features_rest[u] + features_rest[v]) * 0.5
                rest_count = min(int(rest.shape[0]), 3)
                if rest_count > 0:
                    rest = rest.clone()
                    rest[:rest_count, :] = rest[:rest_count, :] + delta[edge_idx, 1 : 1 + rest_count, :]
                new_frest.append(rest)
            else:
                new_fdc.append((features_dc[u] + features_dc[v]) * 0.5 + delta[edge_idx].view(1, 3) / float(C0))
                new_frest.append((features_rest[u] + features_rest[v]) * 0.5)
            new_weight.append((vertex_weight[u] + vertex_weight[v]) * 0.5)
        mab, mbc, mca = mids
        new_faces.extend(
            [
                torch.tensor([a, mab, mca], dtype=torch.long),
                torch.tensor([mab, b, mbc], dtype=torch.long),
                torch.tensor([mca, mbc, c], dtype=torch.long),
                torch.tensor([mab, mbc, mca], dtype=torch.long),
            ]
        )
        face_source_ids.extend([face_id] * 4)

    keep_faces = faces[~remove_mask]
    if new_vertices:
        out["triangles_points"] = torch.cat([vertices, torch.stack(new_vertices, dim=0)], dim=0).to(
            dtype=state["triangles_points"].dtype
        )
        out["features_dc"] = torch.cat([features_dc, torch.stack(new_fdc, dim=0)], dim=0).to(
            dtype=state["features_dc"].dtype
        )
        out["features_rest"] = torch.cat([features_rest, torch.stack(new_frest, dim=0)], dim=0).to(
            dtype=state["features_rest"].dtype
        )
        out["vertex_weight"] = torch.cat([vertex_weight, torch.stack(new_weight, dim=0)], dim=0).to(
            dtype=state["vertex_weight"].dtype
        )
        out["_triangle_indices"] = torch.cat([keep_faces, torch.stack(new_faces, dim=0)], dim=0).to(
            dtype=state["_triangle_indices"].dtype
        )
    else:
        out["_triangle_indices"] = keep_faces.to(dtype=state["_triangle_indices"].dtype)

    for key in FACE_KEYS:
        value = state.get(key)
        if not torch.is_tensor(value) or value.shape[0] != faces.shape[0]:
            continue
        kept = value.detach().cpu()[~remove_mask]
        if face_source_ids:
            added = value.detach().cpu()[torch.as_tensor(face_source_ids, dtype=torch.long)].clone()
            out[key] = torch.cat([kept, added], dim=0).to(dtype=value.dtype)
        else:
            out[key] = kept.to(dtype=value.dtype)
    return out


def materialize_vertex_delta(
    state: dict[str, Any],
    accepted: list[dict[str, Any]],
    *,
    feature_mode: str,
) -> dict[str, Any]:
    out = clone_state(state)
    faces = state["_triangle_indices"].detach().cpu().long()
    features_dc = state["features_dc"].detach().cpu().float()
    features_rest = state["features_rest"].detach().cpu().float()
    dc_delta = torch.zeros_like(features_dc)
    rest_delta = torch.zeros_like(features_rest)
    counts = torch.zeros((features_dc.shape[0],), dtype=torch.float32)

    for row in accepted:
        face_id = int(row["face_id"])
        if face_id < 0 or face_id >= int(faces.shape[0]):
            continue
        ids = faces[face_id].long()
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= int(features_dc.shape[0]):
            continue
        delta = torch.as_tensor(row.get("delta_coeff", row.get("delta_rgb")), dtype=torch.float32)
        if str(feature_mode) == "sh1":
            delta = delta.view(3, 4, 3)
        else:
            delta = delta.view(3, 3)
        for local_idx, vertex_id in enumerate(int(x) for x in ids.tolist()):
            if str(feature_mode) == "sh1":
                dc_delta[vertex_id] += delta[local_idx, 0].view(1, 3)
                rest_count = min(int(features_rest.shape[1]), 3)
                if rest_count > 0:
                    rest_delta[vertex_id, :rest_count, :] += delta[local_idx, 1 : 1 + rest_count, :]
            else:
                dc_delta[vertex_id] += delta[local_idx].view(1, 3) / float(C0)
            counts[vertex_id] += 1.0

    touched = counts > 0
    if bool(touched.any()):
        scale = counts.clamp_min(1.0).view(-1, 1, 1)
        features_dc = features_dc + dc_delta / scale
        if features_rest.numel() > 0:
            features_rest = features_rest + rest_delta / scale.view(-1, 1, 1)
    out["features_dc"] = features_dc.to(dtype=state["features_dc"].dtype)
    out["features_rest"] = features_rest.to(dtype=state["features_rest"].dtype)
    return out


def materialize(
    state: dict[str, Any],
    accepted: list[dict[str, Any]],
    *,
    feature_mode: str,
    materialize_mode: str,
) -> dict[str, Any]:
    if str(materialize_mode) == "vertex_delta":
        return materialize_vertex_delta(state, accepted, feature_mode=str(feature_mode))
    return materialize_subdivision(state, accepted, feature_mode=str(feature_mode))


def filter_plan_candidates(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    materialize_mode: str | None = None,
    faces_np: np.ndarray | None = None,
    support_face_ids: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    vertex_faces: list[set[int]] = []
    mode = str(materialize_mode if materialize_mode is not None else getattr(args, "materialize_mode", ""))
    spillover_enabled = bool(
        mode == "vertex_delta"
        and faces_np is not None
        and (
            float(args.vertex_delta_min_incident_support_fraction) > 0.0
            or int(args.vertex_delta_max_incident_faces) > 0
        )
    )
    if spillover_enabled:
        vertex_faces = build_vertex_face_adjacency(faces_np)
    support = support_face_ids or set()
    for row in rows:
        candidate = dict(row)
        effective = effective_proxy_certificate(candidate.get("proxy", {}), args)
        candidate["effective_proxy"] = effective
        spillover = candidate.get("vertex_delta_spillover")
        if spillover_enabled:
            spillover = vertex_delta_spillover_certificate(
                face_id=int(candidate.get("face_id", -1)),
                faces=faces_np,
                vertex_faces=vertex_faces,
                support_faces=support,
                min_support_fraction=float(args.vertex_delta_min_incident_support_fraction),
                max_incident_faces=int(args.vertex_delta_max_incident_faces),
            )
            candidate["vertex_delta_spillover"] = spillover
        reasons: list[str] = []
        if not bool(candidate.get("policy_pass", True)) and not bool(args.force_apply):
            reasons.append("policy_pass_false")
        reasons.extend(str(item) for item in effective.get("decision_reasons", []))
        if isinstance(spillover, dict) and not bool(spillover.get("passed", True)):
            reasons.extend(str(item) for item in spillover.get("decision_reasons", []))
        if reasons:
            candidate["plan_filter_decision_reasons"] = reasons
            rejected.append(candidate)
        else:
            kept.append(candidate)
    return kept, {
        "requested_faces": int(len(rows)),
        "kept_faces": int(len(kept)),
        "rejected_faces": int(len(rejected)),
        "effective_proxy_gate": {
            "enabled": bool(
                float(args.min_effective_mean_relative_gain) > -1.0e29
                or float(args.min_effective_min_relative_gain) > -1.0e29
                or float(args.min_effective_delta_abs_mean) > 0.0
            ),
            "min_effective_mean_relative_gain": float(args.min_effective_mean_relative_gain),
            "min_effective_min_relative_gain": float(args.min_effective_min_relative_gain),
            "min_effective_delta_abs_mean": float(args.min_effective_delta_abs_mean),
        },
        "vertex_delta_generalization": {
            "enabled": bool(spillover_enabled),
            "min_incident_support_fraction": float(args.vertex_delta_min_incident_support_fraction),
            "max_incident_faces": int(args.vertex_delta_max_incident_faces),
            "support_source": "plan_rows_or_train_evidence_selected_faces",
        },
        "rejected_preview": rejected[:20],
    }


def main() -> int:
    args = parse_args()
    source_checkpoint = checkpoint_path(args.source_model, args.iteration)
    output_checkpoint = args.output_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud_state_dict.pt"
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    copy_model_metadata(args.source_model, args.output_model)
    state = torch.load(source_checkpoint, map_location="cpu")

    if args.materialize_plan_in is not None:
        accepted, plan_meta = read_candidate_plan(args.materialize_plan_in, limit=int(args.materialize_plan_limit))
        materialize_mode = str(plan_meta.get("materialize_mode", args.materialize_mode)) if isinstance(plan_meta, dict) else str(args.materialize_mode)
        plan_support: set[int] = set()
        for row in accepted:
            if not isinstance(row, dict):
                continue
            try:
                face_id = int(row.get("face_id", -1))
            except Exception:
                continue
            if face_id >= 0:
                plan_support.add(face_id)
        faces_np = state["_triangle_indices"].detach().cpu().long().numpy()
        accepted, plan_filter = filter_plan_candidates(
            accepted,
            args,
            materialize_mode=materialize_mode,
            faces_np=faces_np if str(materialize_mode) == "vertex_delta" else None,
            support_face_ids=plan_support,
        )
        out = materialize(
            state,
            accepted,
            feature_mode=str(args.feature_mode),
            materialize_mode=str(materialize_mode),
        )
        materialization_effect = materialization_effect_summary(
            state,
            out,
            min_abs_attribute_delta=float(args.min_materialized_attribute_delta),
        )
        requested_accepted_faces = int(len(accepted))
        materialization_reasons: list[str] = []
        if accepted and not bool(materialization_effect["has_effect"]) and not bool(args.allow_no_effect_accept):
            materialization_reasons.append("materialization_has_no_effect")
            out = clone_state(state)
            accepted = []
            materialization_effect = materialization_effect_summary(
                state,
                out,
                min_abs_attribute_delta=float(args.min_materialized_attribute_delta),
            )
        torch.save(out, output_checkpoint)
        degenerate, invalid = validate_faces(out["triangles_points"], out["_triangle_indices"])
        before_faces = int(state["_triangle_indices"].shape[0])
        before_vertices = int(state["triangles_points"].shape[0])
        after_faces = int(out["_triangle_indices"].shape[0])
        after_vertices = int(out["triangles_points"].shape[0])
        no_op_copy = bool(not accepted)
        policy_pass = bool(accepted) and all(bool(row.get("policy_pass", True)) for row in accepted)
        audit = {
            "operator": "surface_residual_subdivision_delta_plan_materialize",
            "test_usage": "none",
            "source_model": str(args.source_model),
            "source_checkpoint": str(source_checkpoint),
            "output_model": str(args.output_model),
            "output_checkpoint": str(output_checkpoint),
            "iteration": int(args.iteration),
            "materialize_plan_in": str(args.materialize_plan_in),
            "materialize_plan_limit": int(args.materialize_plan_limit),
            "plan_filter": plan_filter,
            "requested_accepted_faces": int(requested_accepted_faces),
            "accepted_faces": int(len(accepted)),
            "accepted": bool(accepted),
            "policy_pass": bool(policy_pass),
            "materialized": bool(accepted),
            "no_op_copy": bool(no_op_copy),
            "materialization_decision_reasons": materialization_reasons,
            "materialization_effect": materialization_effect,
            "feature_mode": str(args.feature_mode),
            "materialize_mode": str(materialize_mode),
            "basis_dim": int(12 if str(args.feature_mode) == "sh1" else 3),
            "delta_storage": "sh_coeff" if str(args.feature_mode) == "sh1" else "rgb_dc",
            "plan_source_operator": plan_meta.get("operator") if isinstance(plan_meta, dict) else None,
            "plan_source_output_model": plan_meta.get("output_model") if isinstance(plan_meta, dict) else None,
            "topology_before": {"triangles": before_faces, "vertices": before_vertices},
            "topology_after": {
                "triangles": after_faces,
                "vertices": after_vertices,
                "degenerate_face_count": int(degenerate),
                "invalid_index_count": int(invalid),
            },
            "accepted_preview": accepted[:20],
        }
        (args.output_model / "surface_residual_subdivision_delta_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# ECSR Surface Residual Subdivision Delta Plan Materialization Audit",
            "",
            f"- operator: `{audit['operator']}`",
            f"- plan: `{audit['materialize_plan_in']}`",
            f"- plan filter kept faces: `{audit['plan_filter']['kept_faces']}` / `{audit['plan_filter']['requested_faces']}`",
            f"- requested accepted faces: `{audit['requested_accepted_faces']}`",
            f"- accepted faces: `{audit['accepted_faces']}`",
            f"- feature mode: `{audit['feature_mode']}`",
            f"- no-op copy: `{str(audit['no_op_copy']).lower()}`",
            f"- triangles: `{before_faces}` -> `{after_faces}`",
            f"- vertices: `{before_vertices}` -> `{after_vertices}`",
            f"- degenerate faces: `{degenerate}`",
            f"- invalid indices: `{invalid}`",
        ]
        (args.output_model / "surface_residual_subdivision_delta_audit.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(audit, indent=2))
        return 0 if int(degenerate) == 0 and int(invalid) == 0 else 1

    vertices_np = state["triangles_points"].detach().cpu().float().numpy()
    faces_np = state["_triangle_indices"].detach().cpu().long().numpy()

    top_csv = args.evidence_dir / "top_residual_supports.csv"
    view_paths = sorted((args.evidence_dir / "per_view_npz").glob("*.npz"))
    if not view_paths:
        view_paths = sorted((args.evidence_dir / "views").glob("*.npz"))
    if not view_paths:
        view_paths = sorted(args.evidence_dir.glob("*.npz"))
    selected_faces, face_stats = read_selected_faces(
        top_csv,
        top_k=int(args.top_k),
        min_view_hits=int(args.min_view_hits),
        min_consistency=float(args.min_consistency),
        min_pixel_count=float(args.min_pixel_count),
    )
    vertex_faces = build_vertex_face_adjacency(faces_np)
    support_face_ids = set(int(face_id) for face_id in selected_faces)
    offsets = parse_policy_offsets(args.policy_val_offsets, stride=int(args.policy_val_stride))
    offset_sets: list[dict[str, Any]] = []
    for offset in offsets:
        fit_views, val_views = split_view_paths(view_paths, int(args.policy_val_stride), offset=offset)
        fit_samples = collect_samples(
            fit_views,
            selected_faces,
            face_stats,
            vertices=vertices_np,
            faces=faces_np,
            feature_mode=str(args.feature_mode),
            materialize_mode=str(args.materialize_mode),
            high_error_quantile=float(args.high_error_quantile),
            min_alpha=float(args.min_alpha),
            max_samples_per_face_view=int(args.max_samples_per_face_view),
            anchor_support=bool(args.anchor_support),
            anchor_max_error_quantile=float(args.anchor_max_error_quantile),
            anchor_samples_per_face_view=int(args.anchor_samples_per_face_view),
            anchor_weight=float(args.anchor_weight),
            structure_preserve=bool(args.structure_preserve),
            structure_weight_strength=float(args.structure_weight_strength),
        )
        val_samples = collect_samples(
            val_views,
            selected_faces,
            face_stats,
            vertices=vertices_np,
            faces=faces_np,
            feature_mode=str(args.feature_mode),
            materialize_mode=str(args.materialize_mode),
            high_error_quantile=float(args.high_error_quantile),
            min_alpha=float(args.min_alpha),
            max_samples_per_face_view=int(args.max_samples_per_face_view),
            anchor_support=bool(args.anchor_support),
            anchor_max_error_quantile=float(args.anchor_max_error_quantile),
            anchor_samples_per_face_view=int(args.anchor_samples_per_face_view),
            anchor_weight=float(args.anchor_weight),
            structure_preserve=bool(args.structure_preserve),
            structure_weight_strength=float(args.structure_weight_strength),
        )
        offset_sets.append(
            {
                "offset": int(offset),
                "fit_views": fit_views,
                "val_views": val_views,
                "fit_samples": fit_samples,
                "val_samples": val_samples,
            }
        )

    candidates: list[dict[str, Any]] = []
    required_offsets = int(args.min_policy_val_offsets)
    if required_offsets <= 0:
        required_offsets = len(offset_sets)
    required_offsets = min(max(required_offsets, 1), max(len(offset_sets), 1))
    required_fraction = float(args.min_policy_val_offset_fraction)
    if str(args.feature_mode) == "sh1":
        max_abs_sh_coeff = float(args.max_abs_sh_coeff)
        if max_abs_sh_coeff <= 0.0:
            max_abs_sh_coeff = float(args.max_abs_delta_rgb) / max(float(C1), 1e-8)
        per_midpoint_bounds = np.asarray(
            [float(args.max_abs_delta_rgb) / float(C0), max_abs_sh_coeff, max_abs_sh_coeff, max_abs_sh_coeff],
            dtype=np.float32,
        )
        coeff_bounds = np.tile(per_midpoint_bounds, 3)
    else:
        max_abs_sh_coeff = 0.0
        coeff_bounds = None
    for fid in selected_faces:
        fold_rows: list[dict[str, Any]] = []
        passing_deltas: list[np.ndarray] = []
        candidate_deltas: list[np.ndarray] = []
        for offset_set in offset_sets:
            offset = int(offset_set["offset"])
            fit_samples = offset_set["fit_samples"]
            val_samples = offset_set["val_samples"]
            xf, yf, wf = _pack_face_samples(fit_samples[int(fid)])
            xv, yv, wv = _pack_face_samples(val_samples[int(fid)])
            if xf.shape[0] < int(args.min_fit_samples) or xv.shape[0] < int(args.min_val_samples):
                fold_rows.append(
                    {
                        "offset": offset,
                        "accepted": False,
                        "decision_reasons": ["insufficient_samples"],
                        "fit_samples": int(xf.shape[0]),
                        "val_samples": int(xv.shape[0]),
                    }
                )
                continue
            delta, stats = fit_delta(
                xf,
                yf,
                wf,
                xv,
                yv,
                wv,
                strength=float(args.strength),
                max_abs_delta_rgb=float(args.max_abs_delta_rgb),
                coeff_bounds=coeff_bounds,
                lambda_ridge=float(args.lambda_ridge),
            )
            certificate = view_gain_certificate(
                delta,
                val_samples[int(fid)],
                strength=float(args.strength),
                max_abs_delta_rgb=float(args.max_abs_delta_rgb),
                min_samples=int(args.min_view_gain_samples),
                min_relative_gain=float(args.min_view_gain_relative_gain),
                min_views=int(args.min_view_gain_views),
                min_fraction=float(args.min_view_gain_fraction),
            )
            reasons: list[str] = []
            if float(stats["relative_gain"]) < float(args.min_policy_val_relative_gain):
                reasons.append("relative_gain_below_threshold")
            if not bool(certificate["passed"]):
                reasons.append("view_gain_certificate_failed")
            accepted_fold = not reasons
            candidate_deltas.append(delta)
            if accepted_fold:
                passing_deltas.append(delta)
            fold_rows.append(
                {
                    "offset": offset,
                    "accepted": bool(accepted_fold),
                    "decision_reasons": reasons,
                    "proxy": stats,
                    "view_gain_certificate": certificate,
                }
            )

        delta_pool = passing_deltas if passing_deltas else candidate_deltas
        if not delta_pool:
            continue
        delta = np.mean(np.stack(delta_pool, axis=0), axis=0).astype(np.float32)
        luma_projection: dict[str, Any] = {"enabled": False}
        if bool(args.luma_preserve) and str(args.feature_mode) == "sh1" and int(delta.shape[0]) == 12:
            delta, luma_projection = choose_luma_preserved_delta(
                delta,
                offset_sets,
                fid=int(fid),
                args=args,
                required_offsets=int(required_offsets),
                required_fraction=float(required_fraction),
            )
        structure_projection: dict[str, Any] = {"enabled": False}
        if bool(args.structure_preserve) and str(args.feature_mode) == "sh1" and int(delta.shape[0]) == 12:
            delta, structure_projection = choose_structure_preserved_delta(
                delta,
                offset_sets,
                fid=int(fid),
                args=args,
                required_offsets=int(required_offsets),
                required_fraction=float(required_fraction),
            )

        final_fold_rows: list[dict[str, Any]] = []
        for offset_set in offset_sets:
            offset = int(offset_set["offset"])
            fit_samples = offset_set["fit_samples"]
            val_samples = offset_set["val_samples"]
            xf, _, _ = _pack_face_samples(fit_samples[int(fid)])
            xv, yv, wv = _pack_face_samples(val_samples[int(fid)])
            fit_count = int(xf.shape[0])
            if fit_count < int(args.min_fit_samples) or xv.shape[0] < int(args.min_val_samples):
                final_fold_rows.append(
                    {
                        "offset": offset,
                        "accepted": False,
                        "decision_reasons": ["insufficient_samples"],
                        "fit_samples": fit_count,
                        "val_samples": int(xv.shape[0]),
                    }
                )
                continue
            final_stats = evaluate_delta(
                delta,
                xv,
                yv,
                wv,
                fit_samples=fit_count,
                strength=float(args.strength),
                max_abs_delta_rgb=float(args.max_abs_delta_rgb),
            )
            final_certificate = view_gain_certificate(
                delta,
                val_samples[int(fid)],
                strength=float(args.strength),
                max_abs_delta_rgb=float(args.max_abs_delta_rgb),
                min_samples=int(args.min_view_gain_samples),
                min_relative_gain=float(args.min_view_gain_relative_gain),
                min_views=int(args.min_view_gain_views),
                min_fraction=float(args.min_view_gain_fraction),
            )
            reasons: list[str] = []
            if float(final_stats["relative_gain"]) < float(args.min_policy_val_relative_gain):
                reasons.append("relative_gain_below_threshold")
            if not bool(final_certificate["passed"]):
                reasons.append("view_gain_certificate_failed")
            luma_stats: dict[str, float] | None = None
            if bool(args.luma_preserve) and str(args.feature_mode) == "sh1":
                luma_stats = evaluate_luma_delta(
                    delta,
                    xv,
                    yv,
                    wv,
                    strength=float(args.strength),
                    max_abs_delta_rgb=float(args.max_abs_delta_rgb),
                )
                if float(luma_stats["luma_relative_gain"]) < float(args.min_luma_relative_gain):
                    reasons.append("luma_gain_below_threshold")
                if float(args.max_mean_luma_shift) > 0.0 and abs(float(luma_stats["mean_pred_luma"])) > float(args.max_mean_luma_shift):
                    reasons.append("mean_luma_shift_exceeds_threshold")
            structure_stats: dict[str, Any] | None = None
            if bool(args.structure_preserve) and str(args.feature_mode) == "sh1":
                _, _, swv = _pack_face_samples(val_samples[int(fid)], weight_key="structure_weight")
                structure_rgb = evaluate_delta(
                    delta,
                    xv,
                    yv,
                    swv,
                    fit_samples=fit_count,
                    strength=float(args.strength),
                    max_abs_delta_rgb=float(args.max_abs_delta_rgb),
                )
                structure_luma = evaluate_luma_delta(
                    delta,
                    xv,
                    yv,
                    swv,
                    strength=float(args.strength),
                    max_abs_delta_rgb=float(args.max_abs_delta_rgb),
                )
                structure_stats = {"rgb_proxy": structure_rgb, "luma_proxy": structure_luma}
                if float(structure_rgb["relative_gain"]) < float(args.min_structure_relative_gain):
                    reasons.append("structure_relative_gain_below_threshold")
                if float(args.max_structure_mean_luma_shift) > 0.0 and abs(float(structure_luma["mean_pred_luma"])) > float(args.max_structure_mean_luma_shift):
                    reasons.append("structure_luma_shift_exceeds_threshold")
            final_fold_rows.append(
                {
                    "offset": offset,
                    "accepted": not reasons,
                    "decision_reasons": reasons,
                    "fit_proxy": next((row.get("proxy", {}) for row in fold_rows if int(row.get("offset", -1)) == offset), {}),
                    "final_proxy": final_stats,
                    "luma_proxy": luma_stats or {},
                    "structure_proxy": structure_stats or {},
                    "view_gain_certificate": final_certificate,
                }
            )

        passing_count = sum(1 for row in final_fold_rows if bool(row.get("accepted", False)))
        passing_fraction = float(passing_count / max(len(offset_sets), 1))
        policy_pass = passing_count >= required_offsets and passing_fraction >= required_fraction
        vertex_delta_spillover: dict[str, Any] = {"enabled": False, "passed": True, "decision_reasons": []}
        if str(args.materialize_mode) == "vertex_delta":
            vertex_delta_spillover = vertex_delta_spillover_certificate(
                face_id=int(fid),
                faces=faces_np,
                vertex_faces=vertex_faces,
                support_faces=support_face_ids,
                min_support_fraction=float(args.vertex_delta_min_incident_support_fraction),
                max_incident_faces=int(args.vertex_delta_max_incident_faces),
            )
            if not bool(vertex_delta_spillover.get("passed", True)):
                policy_pass = False
        finite_stats = [row["final_proxy"] for row in final_fold_rows if isinstance(row.get("final_proxy"), dict)]
        relative_gains = [float(row["relative_gain"]) for row in finite_stats]
        proxy = {
            "fit_samples": int(sum(float(row.get("fit_samples", row.get("final_proxy", {}).get("fit_samples", 0))) for row in final_fold_rows)),
            "val_samples": int(sum(float(row.get("val_samples", row.get("final_proxy", {}).get("val_samples", 0))) for row in final_fold_rows)),
            "relative_gain": float(min(relative_gains)) if relative_gains else 0.0,
            "mean_relative_gain": float(np.mean(relative_gains)) if relative_gains else 0.0,
            "passing_offsets": int(passing_count),
            "offset_count": int(len(offset_sets)),
            "passing_fraction": passing_fraction,
            "delta_abs_mean": float(np.abs(delta).mean()) if delta.size else 0.0,
            "delta_abs_max": float(np.abs(delta).max()) if delta.size else 0.0,
        }
        effective_proxy = effective_proxy_certificate(proxy, args)
        if not bool(effective_proxy.get("passed", True)):
            policy_pass = False
        certificate = {
            "passed": bool(policy_pass),
            "required_offsets": int(required_offsets),
            "required_fraction": required_fraction,
            "passing_count": int(passing_count),
            "offset_count": int(len(offset_sets)),
            "passing_fraction": passing_fraction,
            "fit_folds": fold_rows,
            "final_folds": final_fold_rows,
            "luma_preservation": luma_projection,
            "structure_preservation": structure_projection,
            "vertex_delta_spillover": vertex_delta_spillover,
            "effective_proxy": effective_proxy,
        }
        if bool(args.force_apply) or policy_pass:
            candidates.append(
                {
                    "face_id": int(fid),
                    "face_stats": face_stats.get(int(fid), {}),
                    "delta_rgb": delta.tolist() if str(args.feature_mode) == "dc" else [],
                    "delta_coeff": delta.tolist(),
                    "proxy": proxy,
                    "view_gain_certificate": certificate,
                    "luma_preservation": luma_projection,
                    "structure_preservation": structure_projection,
                    "vertex_delta_spillover": vertex_delta_spillover,
                    "effective_proxy": effective_proxy,
                    "policy_pass": bool(policy_pass),
                }
            )
    candidates.sort(
        key=lambda row: (
            float(row["proxy"]["relative_gain"]),
            float(row["face_stats"].get("score", 0.0)),
            float(row["face_stats"].get("pixel_count", 0.0)),
        ),
        reverse=True,
    )
    if args.candidate_plan_out is not None:
        args.candidate_plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_plan_out.write_text(
            json.dumps(
                {
                    "operator": "surface_residual_subdivision_delta_candidate_plan",
                    "source_model": str(args.source_model),
                    "iteration": int(args.iteration),
                    "feature_mode": str(args.feature_mode),
                    "materialize_mode": str(args.materialize_mode),
                    "policy_val_offsets": [int(x) for x in offsets],
                    "min_policy_val_offsets": int(required_offsets),
                    "min_policy_val_offset_fraction": float(required_fraction),
                    "luma_preservation": {
                        "enabled": bool(args.luma_preserve),
                        "shrink_grid": [float(x) for x in parse_float_grid(args.luma_shrink_grid)],
                        "shrink_selection": str(args.luma_shrink_selection),
                    },
                    "structure_preservation": {
                        "enabled": bool(args.structure_preserve),
                        "weight_strength": float(args.structure_weight_strength),
                        "min_structure_relative_gain": float(args.min_structure_relative_gain),
                        "max_structure_mean_luma_shift": float(args.max_structure_mean_luma_shift),
                        "shrink_grid": [float(x) for x in parse_float_grid(args.structure_shrink_grid)],
                        "shrink_selection": str(args.structure_shrink_selection),
                    },
                    "vertex_delta_generalization": {
                        "enabled": bool(
                            str(args.materialize_mode) == "vertex_delta"
                            and (
                                float(args.vertex_delta_min_incident_support_fraction) > 0.0
                                or int(args.vertex_delta_max_incident_faces) > 0
                            )
                        ),
                        "min_incident_support_fraction": float(args.vertex_delta_min_incident_support_fraction),
                        "max_incident_faces": int(args.vertex_delta_max_incident_faces),
                        "support_source": "train_evidence_selected_faces",
                    },
                    "effective_proxy_gate": {
                        "enabled": bool(
                            float(args.min_effective_mean_relative_gain) > -1.0e29
                            or float(args.min_effective_min_relative_gain) > -1.0e29
                            or float(args.min_effective_delta_abs_mean) > 0.0
                        ),
                        "min_effective_mean_relative_gain": float(args.min_effective_mean_relative_gain),
                        "min_effective_min_relative_gain": float(args.min_effective_min_relative_gain),
                        "min_effective_delta_abs_mean": float(args.min_effective_delta_abs_mean),
                    },
                    "candidate_count": int(len(candidates)),
                    "candidates": candidates,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    accepted = candidates[: int(args.max_faces_to_apply)]

    requested_accepted_faces = int(len(accepted))
    materialization_reasons: list[str] = []
    materialization_effect: dict[str, Any]
    accepted_policy_pass = bool(accepted) and all(bool(row.get("policy_pass", False)) for row in accepted)
    if not accepted and bool(args.no_op_on_fail):
        out = clone_state(state)
        accepted_flag = False
        no_op_copy = True
        materialization_effect = materialization_effect_summary(
            state,
            out,
            min_abs_attribute_delta=float(args.min_materialized_attribute_delta),
        )
    else:
        out = materialize(
            state,
            accepted,
            feature_mode=str(args.feature_mode),
            materialize_mode=str(args.materialize_mode),
        )
        materialization_effect = materialization_effect_summary(
            state,
            out,
            min_abs_attribute_delta=float(args.min_materialized_attribute_delta),
        )
        if accepted and not bool(materialization_effect["has_effect"]) and not bool(args.allow_no_effect_accept):
            materialization_reasons.append("materialization_has_no_effect")
            out = clone_state(state)
            accepted = []
            accepted_policy_pass = False
            accepted_flag = False
            no_op_copy = True
            materialization_effect = materialization_effect_summary(
                state,
                out,
                min_abs_attribute_delta=float(args.min_materialized_attribute_delta),
            )
        else:
            accepted_flag = bool(accepted)
            no_op_copy = False
    torch.save(out, output_checkpoint)
    degenerate, invalid = validate_faces(out["triangles_points"], out["_triangle_indices"])

    before_faces = int(state["_triangle_indices"].shape[0])
    before_vertices = int(state["triangles_points"].shape[0])
    after_faces = int(out["_triangle_indices"].shape[0])
    after_vertices = int(out["triangles_points"].shape[0])
    accepted_luma_betas = [
        float(row.get("luma_preservation", {}).get("beta"))
        for row in accepted
        if isinstance(row.get("luma_preservation"), dict) and row.get("luma_preservation", {}).get("beta") is not None
    ]
    accepted_structure_betas = [
        float(row.get("structure_preservation", {}).get("beta"))
        for row in accepted
        if isinstance(row.get("structure_preservation"), dict) and row.get("structure_preservation", {}).get("beta") is not None
    ]
    audit = {
        "operator": "surface_residual_subdivision_delta",
        "test_usage": "none",
        "source_model": str(args.source_model),
        "source_checkpoint": str(source_checkpoint),
        "output_model": str(args.output_model),
        "output_checkpoint": str(output_checkpoint),
        "iteration": int(args.iteration),
        "evidence_dir": str(args.evidence_dir),
        "view_counts": {
            "offsets": [
                {
                    "offset": int(row["offset"]),
                    "fit": int(len(row["fit_views"])),
                    "policy_val": int(len(row["val_views"])),
                }
                for row in offset_sets
            ]
        },
        "selected_faces": int(len(selected_faces)),
        "candidate_faces": int(len(candidates)),
        "requested_accepted_faces": int(requested_accepted_faces),
        "accepted_faces": int(len(accepted)),
        "accepted": accepted_flag,
        "policy_pass": bool(accepted_policy_pass),
        "materialized": accepted_flag,
        "no_op_copy": no_op_copy,
        "materialization_decision_reasons": materialization_reasons,
        "materialization_effect": materialization_effect,
        "force_apply": bool(args.force_apply),
        "filters": {
            "top_k": int(args.top_k),
            "min_view_hits": int(args.min_view_hits),
            "min_consistency": float(args.min_consistency),
            "min_pixel_count": float(args.min_pixel_count),
            "high_error_quantile": float(args.high_error_quantile),
            "min_alpha": float(args.min_alpha),
        },
        "strength": float(args.strength),
        "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
        "feature_mode": str(args.feature_mode),
        "materialize_mode": str(args.materialize_mode),
        "basis_dim": int(12 if str(args.feature_mode) == "sh1" else 3),
        "delta_storage": "sh_coeff" if str(args.feature_mode) == "sh1" else "rgb_dc",
        "features_rest_channels_written": int(3 if str(args.feature_mode) == "sh1" else 0),
        "max_abs_sh_coeff": float(max_abs_sh_coeff),
        "luma_preservation": {
            "enabled": bool(args.luma_preserve),
            "active": bool(args.luma_preserve and str(args.feature_mode) == "sh1"),
            "mode": "sh1_dc_luma_shrink",
            "min_luma_relative_gain": float(args.min_luma_relative_gain),
            "max_mean_luma_shift": float(args.max_mean_luma_shift),
            "shrink_grid": [float(x) for x in parse_float_grid(args.luma_shrink_grid)],
            "shrink_selection": str(args.luma_shrink_selection),
            "accepted_beta_mean": float(np.mean(accepted_luma_betas)) if accepted_luma_betas else 0.0,
            "accepted_beta_max": float(np.max(accepted_luma_betas)) if accepted_luma_betas else 0.0,
        },
        "structure_preservation": {
            "enabled": bool(args.structure_preserve),
            "active": bool(args.structure_preserve and str(args.feature_mode) == "sh1"),
            "mode": "texture_depth_normal_weighted_sh1_all_luma_shrink",
            "weight_strength": float(args.structure_weight_strength),
            "min_structure_relative_gain": float(args.min_structure_relative_gain),
            "max_structure_mean_luma_shift": float(args.max_structure_mean_luma_shift),
            "shrink_grid": [float(x) for x in parse_float_grid(args.structure_shrink_grid)],
            "shrink_selection": str(args.structure_shrink_selection),
            "accepted_beta_mean": float(np.mean(accepted_structure_betas)) if accepted_structure_betas else 0.0,
            "accepted_beta_max": float(np.max(accepted_structure_betas)) if accepted_structure_betas else 0.0,
            "fields_used": ["texture", "depth", "normal"],
        },
        "vertex_delta_generalization": {
            "enabled": bool(
                str(args.materialize_mode) == "vertex_delta"
                and (
                    float(args.vertex_delta_min_incident_support_fraction) > 0.0
                    or int(args.vertex_delta_max_incident_faces) > 0
                )
            ),
            "min_incident_support_fraction": float(args.vertex_delta_min_incident_support_fraction),
            "max_incident_faces": int(args.vertex_delta_max_incident_faces),
            "support_source": "train_evidence_selected_faces",
        },
        "effective_proxy_gate": {
            "enabled": bool(
                float(args.min_effective_mean_relative_gain) > -1.0e29
                or float(args.min_effective_min_relative_gain) > -1.0e29
                or float(args.min_effective_delta_abs_mean) > 0.0
            ),
            "min_effective_mean_relative_gain": float(args.min_effective_mean_relative_gain),
            "min_effective_min_relative_gain": float(args.min_effective_min_relative_gain),
            "min_effective_delta_abs_mean": float(args.min_effective_delta_abs_mean),
        },
        "anchor_support": {
            "enabled": bool(args.anchor_support),
            "anchor_max_error_quantile": float(args.anchor_max_error_quantile),
            "anchor_samples_per_face_view": int(args.anchor_samples_per_face_view),
            "anchor_weight": float(args.anchor_weight),
        },
        "lambda_ridge": float(args.lambda_ridge),
        "min_policy_val_relative_gain": float(args.min_policy_val_relative_gain),
        "policy_val_offsets": [int(x) for x in offsets],
        "min_policy_val_offsets": int(required_offsets),
        "min_policy_val_offset_fraction": float(required_fraction),
        "max_faces_to_apply": int(args.max_faces_to_apply),
        "view_gain_certificate": {
            "min_view_gain_views": int(args.min_view_gain_views),
            "min_view_gain_relative_gain": float(args.min_view_gain_relative_gain),
            "min_view_gain_samples": int(args.min_view_gain_samples),
            "min_view_gain_fraction": float(args.min_view_gain_fraction),
        },
        "topology_before": {"triangles": before_faces, "vertices": before_vertices},
        "topology_after": {
            "triangles": after_faces,
            "vertices": after_vertices,
            "degenerate_face_count": int(degenerate),
            "invalid_index_count": int(invalid),
        },
        "mean_proxy_relative_gain": float(np.mean([row["proxy"]["relative_gain"] for row in accepted]))
        if accepted
        else 0.0,
        "mean_delta_abs": float(np.mean([row["proxy"]["delta_abs_mean"] for row in accepted])) if accepted else 0.0,
        "accepted_preview": accepted[:20],
    }
    (args.output_model / "surface_residual_subdivision_delta_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ECSR Surface Residual Subdivision Delta Audit",
        "",
        f"- operator: `{audit['operator']}`",
        f"- selected faces: `{audit['selected_faces']}`",
        f"- candidate faces: `{audit['candidate_faces']}`",
        f"- requested accepted faces: `{audit['requested_accepted_faces']}`",
        f"- accepted faces: `{audit['accepted_faces']}`",
        f"- policy offsets: `{','.join(str(x) for x in audit['policy_val_offsets'])}`",
        f"- min passing offsets: `{audit['min_policy_val_offsets']}`",
        f"- feature mode: `{audit['feature_mode']}`",
        f"- basis dim: `{audit['basis_dim']}`",
        f"- luma preservation: `{str(audit['luma_preservation']['active']).lower()}`",
        f"- structure preservation: `{str(audit['structure_preservation']['active']).lower()}`",
        f"- vertex-delta generalization: `{str(audit['vertex_delta_generalization']['enabled']).lower()}`",
        f"- effective proxy gate: `{str(audit['effective_proxy_gate']['enabled']).lower()}`",
        f"- anchor support: `{str(audit['anchor_support']['enabled']).lower()}`",
        f"- no-op copy: `{str(audit['no_op_copy']).lower()}`",
        f"- mean proxy relative gain: `{audit['mean_proxy_relative_gain']:.6f}`",
        f"- mean delta abs: `{audit['mean_delta_abs']:.6f}`",
        f"- triangles: `{before_faces}` -> `{after_faces}`",
        f"- vertices: `{before_vertices}` -> `{after_vertices}`",
        f"- degenerate faces: `{degenerate}`",
        f"- invalid indices: `{invalid}`",
    ]
    (args.output_model / "surface_residual_subdivision_delta_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))
    return 0 if int(degenerate) == 0 and int(invalid) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
