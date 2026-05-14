#!/usr/bin/env python3
"""Fit train-certified face-local residual appearance deltas.

This operator is a representation-level successor to the shared-vertex SH
delta.  Instead of changing the SH coefficients of vertices shared by many
faces, it duplicates the three vertices of train-certified high-residual faces
and redirects only those faces to the local copies.  Geometry and triangle count
are preserved; the added local vertices carry a bounded SH residual state.

No held-out test residuals are read.  Fitting uses train-cache views and a
deterministic train policy-validation split decides which face-local deltas are
materialized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import copy_model_metadata, checkpoint_path, validate_faces
from utils.sh_utils import C0, C1, C2, C3


@dataclass
class PixelSamples:
    face_ids: np.ndarray
    barycentric: np.ndarray
    residual_rgb: np.ndarray
    weights: np.ndarray
    camera_centers: np.ndarray
    view_names: list[str]

    @property
    def count(self) -> int:
        return int(self.face_ids.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model", type=Path, required=True)
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--top_k", type=int, default=2048)
    parser.add_argument("--min_view_hits", type=int, default=2)
    parser.add_argument("--min_consistency", type=float, default=0.80)
    parser.add_argument("--min_pixel_count", type=float, default=6.0)
    parser.add_argument("--max_samples_per_face_view", type=int, default=64)
    parser.add_argument("--max_total_samples", type=int, default=320000)
    parser.add_argument("--high_error_quantile", type=float, default=0.65)
    parser.add_argument("--min_alpha", type=float, default=0.05)
    parser.add_argument("--barycentric_tolerance", type=float, default=0.35)
    parser.add_argument(
        "--uniform_barycentric",
        action="store_true",
        help="Use equal 1/3 weights when the evidence cache does not contain barycentric maps.",
    )
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.18)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=0.014)
    parser.add_argument(
        "--max_abs_sh_coeff",
        type=float,
        default=0.0,
        help="Bound for each non-DC SH coefficient delta. 0 derives it from max_abs_delta_rgb / C1.",
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Face-local residual SH degree. 1 preserves historical behavior; 3 uses the full stored SH basis.",
    )
    parser.add_argument("--lambda_mag", type=float, default=2e-2)
    parser.add_argument("--lambda_sh1_mag", type=float, default=5e-2)
    parser.add_argument("--lambda_smooth", type=float, default=8e-2)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--max_faces_to_apply", type=int, default=2048)
    parser.add_argument("--min_policy_val_relative_gain", type=float, default=0.02)
    parser.add_argument("--min_policy_val_samples", type=int, default=512)
    parser.add_argument("--min_policy_val_unique_faces", type=int, default=16)
    parser.add_argument(
        "--validation_shrink_mode",
        choices=("none", "global", "face"),
        default="none",
        help=(
            "Train-only residual amplitude calibration. 'global' fits one shrink "
            "scale on policy-val samples; 'face' fits one shrink scale per selected face."
        ),
    )
    parser.add_argument(
        "--validation_shrink_min_samples",
        type=int,
        default=8,
        help="Minimum policy-val samples required before a face gets a nonzero face shrink scale.",
    )
    parser.add_argument(
        "--crossfold_gain_certificate_folds",
        type=int,
        default=0,
        help=(
            "If >1, split train evidence views into this many interleaved folds "
            "and require each accepted face to have nonnegative proxy gain across enough folds. "
            "This is an all-train fold-consistency check, not an independent cross-fit certificate."
        ),
    )
    parser.add_argument("--crossfold_min_passing_folds", type=int, default=0)
    parser.add_argument("--crossfold_min_fold_relative_gain", type=float, default=0.0)
    parser.add_argument("--crossfold_min_fold_samples", type=int, default=4)
    parser.add_argument("--min_face_policy_val_relative_gain", type=float, default=0.0)
    parser.add_argument("--min_face_policy_val_samples", type=int, default=8)
    parser.add_argument(
        "--min_face_gain_certificate_views",
        type=int,
        default=0,
        help=(
            "If >0, require each accepted face to have predicted residual MSE gain "
            "on at least this many policy-val train views."
        ),
    )
    parser.add_argument(
        "--min_face_gain_certificate_relative_gain",
        type=float,
        default=0.0,
        help="Minimum per-view relative MSE gain for one policy-val train view to certify a face.",
    )
    parser.add_argument(
        "--min_face_gain_certificate_view_samples",
        type=int,
        default=4,
        help="Minimum samples from one policy-val train view before it can certify a face.",
    )
    parser.add_argument(
        "--min_face_gain_certificate_fraction",
        type=float,
        default=0.0,
        help="Optional minimum fraction of eligible policy-val train views that must certify a face.",
    )
    parser.add_argument(
        "--min_face_view_consensus",
        type=float,
        default=0.0,
        help=(
            "If >0, require this fraction of policy-val train views for a face "
            "to agree with the face residual direction before materializing its local vertices."
        ),
    )
    parser.add_argument(
        "--min_face_consensus_views",
        type=int,
        default=2,
        help="Minimum policy-val train views needed for the face/view consensus certificate.",
    )
    parser.add_argument(
        "--min_face_consensus_view_samples",
        type=int,
        default=4,
        help="Minimum samples from one policy-val train view before it votes in face/view consensus.",
    )
    parser.add_argument(
        "--face_consensus_min_cosine",
        type=float,
        default=0.0,
        help="Minimum cosine against the per-face residual direction for one view to count as agreeing.",
    )
    parser.add_argument(
        "--patch_cert_rings",
        type=int,
        default=0,
        help=(
            "If >0, grow accepted face seeds into connected train-evidence patches "
            "using selected-face adjacency and require a patch-level train policy-val gain."
        ),
    )
    parser.add_argument("--patch_cert_max_faces_per_seed", type=int, default=8)
    parser.add_argument("--patch_cert_min_direction_cosine", type=float, default=0.90)
    parser.add_argument("--patch_cert_min_neighbor_policy_val_samples", type=int, default=4)
    parser.add_argument("--patch_cert_min_neighbor_policy_val_relative_gain", type=float, default=-0.02)
    parser.add_argument("--patch_cert_min_policy_val_samples", type=int, default=16)
    parser.add_argument("--patch_cert_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--patch_cert_neighbor_mode", choices=("topology", "centroid", "both"), default="topology")
    parser.add_argument("--patch_cert_centroid_candidates_per_seed", type=int, default=64)
    parser.add_argument(
        "--patch_cert_crossfold_folds",
        type=int,
        default=0,
        help=(
            "If >1, require each accepted patch carrier to pass a train-only fold proxy-gain "
            "certificate. This gates the patch itself, not only the seed face."
        ),
    )
    parser.add_argument("--patch_cert_crossfold_min_passing_folds", type=int, default=0)
    parser.add_argument("--patch_cert_crossfold_min_fold_relative_gain", type=float, default=0.0)
    parser.add_argument("--patch_cert_crossfold_min_fold_samples", type=int, default=4)
    parser.add_argument(
        "--patch_cert_neighbor_crossfold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When patch-fold certification is enabled, require each neighbor to pass "
            "the same train-only fold certificate before it can enter a patch."
        ),
    )
    parser.add_argument(
        "--patch_cert_shrink",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When patch certification is enabled, fit one train-only shrink scale per accepted patch.",
    )
    parser.add_argument(
        "--strict_patchcert_carrier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require the paper-facing strict PatchCert carrier policy: patch growth, "
            "patch-fold certification, neighbor fold admission, post-shrink checks, "
            "and certified whole-carrier plan replay."
        ),
    )
    parser.add_argument(
        "--candidate_plan_out",
        type=Path,
        default=None,
        help=(
            "Write the final train-certified accepted face-local residual carrier and fitted "
            "coefficients to a JSON plan for later materialization."
        ),
    )
    parser.add_argument(
        "--materialize_plan_in",
        type=Path,
        default=None,
        help="Materialize face-local residuals from a previously written candidate plan instead of refitting.",
    )
    parser.add_argument(
        "--materialize_plan_limit",
        type=int,
        default=0,
        help="Keep only the first N rows from --materialize_plan_in after optional face-id filtering. 0 keeps all.",
    )
    parser.add_argument(
        "--materialize_plan_face_ids",
        default="",
        help="Optional comma-separated face ids to materialize from --materialize_plan_in.",
    )
    parser.add_argument(
        "--materialize_plan_scale",
        type=float,
        default=1.0,
        help="Uniform scale applied to plan coefficients during materialization. Used only with --materialize_plan_in.",
    )
    parser.add_argument(
        "--materialize_plan_alpha_json",
        type=Path,
        default=None,
        help=(
            "Optional JSON containing per-face alpha multipliers for materialized plan rows. "
            "Supported forms: {'face_alphas': {'123': 0.5}} or "
            "{'face_alphas': [{'face_id': 123, 'alpha': 0.5}]}."
        ),
    )
    parser.add_argument(
        "--materialize_allow_uncertified_plan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow materializing legacy plan rows that do not carry explicit "
            "policy/PatchCert certification. Default is strict: uncertified "
            "rows are rejected so plan replay cannot bypass the train-only gate."
        ),
    )
    parser.add_argument("--no_op_on_fail", action="store_true", default=True)
    parser.add_argument("--force_apply", action="store_true")
    parser.add_argument("--device", default="cuda")
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
                    "mean_residual_r": _float(row, "mean_residual_r"),
                    "mean_residual_g": _float(row, "mean_residual_g"),
                    "mean_residual_b": _float(row, "mean_residual_b"),
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
            "mean_residual_r": float(row["mean_residual_r"]),
            "mean_residual_g": float(row["mean_residual_g"]),
            "mean_residual_b": float(row["mean_residual_b"]),
        }
        for row in rows
    }
    return [int(row["face_id"]) for row in rows], stats


def split_view_paths(view_paths: list[Path], stride: int) -> tuple[list[Path], list[Path]]:
    if len(view_paths) < 3:
        return view_paths, view_paths
    stride = max(int(stride), 2)
    fit: list[Path] = []
    val: list[Path] = []
    for idx, path in enumerate(view_paths):
        if idx % stride == 0:
            val.append(path)
        else:
            fit.append(path)
    if not fit or not val:
        return view_paths, view_paths
    return fit, val


def collect_samples(
    view_paths: list[Path],
    selected_faces: list[int],
    face_stats: dict[int, dict[str, float]],
    *,
    high_error_quantile: float,
    min_alpha: float,
    barycentric_tolerance: float,
    max_samples_per_face_view: int,
    max_total_samples: int,
    uniform_barycentric: bool,
) -> PixelSamples:
    selected = set(int(fid) for fid in selected_faces)
    face_chunks: list[np.ndarray] = []
    bary_chunks: list[np.ndarray] = []
    residual_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    center_chunks: list[np.ndarray] = []
    sample_view_names: list[str] = []
    remaining = int(max_total_samples)
    tol = float(barycentric_tolerance)

    for view_path in view_paths:
        if remaining <= 0:
            break
        with np.load(view_path) as z:
            required = {"face_id", "residual_l1", "alpha", "residual_rgb", "camera_center"}
            if not bool(uniform_barycentric):
                required.update({"barycentric", "barycentric_valid"})
            missing = sorted(required - set(z.files))
            if missing:
                raise RuntimeError(
                    f"{view_path} missing required face-local SH1 evidence fields: {missing}. "
                    "Rebuild the cache with barycentric maps or use --uniform_barycentric."
                )
            face_id = z["face_id"].astype(np.int64)
            residual_l1 = z["residual_l1"].astype(np.float32)
            alpha = z["alpha"].astype(np.float32)
            if alpha.ndim == 3:
                alpha = np.squeeze(alpha, axis=0)
            residual_rgb = z["residual_rgb"].astype(np.float32)
            if bool(uniform_barycentric):
                barycentric = np.empty((3,) + face_id.shape, dtype=np.float32)
                bary_valid = np.ones_like(face_id, dtype=bool)
            else:
                barycentric = z["barycentric"].astype(np.float32)
                bary_valid = z["barycentric_valid"].astype(bool)
            camera_center = z["camera_center"].astype(np.float32).reshape(3)

        threshold = float(np.quantile(residual_l1.reshape(-1), float(high_error_quantile)))
        base_valid = bary_valid & (residual_l1 >= threshold) & (alpha >= float(min_alpha))
        if not np.any(base_valid):
            continue
        present = sorted(set(int(x) for x in np.unique(face_id[base_valid])) & selected)
        if not present:
            continue
        for fid in present:
            if remaining <= 0:
                break
            mask = base_valid & (face_id == int(fid))
            if not np.any(mask):
                continue
            if bool(uniform_barycentric):
                b = np.full((int(mask.sum()), 3), 1.0 / 3.0, dtype=np.float32)
            else:
                b = barycentric[:, mask].T.astype(np.float32)
            inside = np.all((b >= -tol) & (b <= 1.0 + tol), axis=1)
            if not np.any(inside):
                continue
            ys, xs = np.nonzero(mask)
            ys = ys[inside]
            xs = xs[inside]
            b = b[inside]
            b = np.clip(b, 0.0, 1.0)
            b = b / np.maximum(b.sum(axis=1, keepdims=True), 1e-8)
            n = int(b.shape[0])
            if n <= 0:
                continue
            cap = min(int(max_samples_per_face_view), remaining, n)
            if n > cap:
                take = np.linspace(0, n - 1, cap, dtype=np.int64)
                ys = ys[take]
                xs = xs[take]
                b = b[take]
                n = cap
            residual = residual_rgb[:, ys, xs].T.astype(np.float32)
            l1 = residual_l1[ys, xs].astype(np.float32)
            stat = face_stats.get(int(fid), {})
            consistency = float(stat.get("consistency", 1.0))
            weights = np.maximum(l1, 1e-4) * max(consistency, 1e-3)
            face_chunks.append(np.full((n,), int(fid), dtype=np.int64))
            bary_chunks.append(b.astype(np.float32))
            residual_chunks.append(residual.astype(np.float32))
            weight_chunks.append(weights.astype(np.float32))
            center_chunks.append(np.repeat(camera_center[None, :], n, axis=0).astype(np.float32))
            sample_view_names.extend([view_path.stem] * n)
            remaining -= n

    if not face_chunks:
        empty = np.empty((0,), dtype=np.int64)
        return PixelSamples(
            face_ids=empty,
            barycentric=np.empty((0, 3), dtype=np.float32),
            residual_rgb=np.empty((0, 3), dtype=np.float32),
            weights=np.empty((0,), dtype=np.float32),
            camera_centers=np.empty((0, 3), dtype=np.float32),
            view_names=[],
        )
    return PixelSamples(
        face_ids=np.concatenate(face_chunks),
        barycentric=np.concatenate(bary_chunks),
        residual_rgb=np.concatenate(residual_chunks),
        weights=np.concatenate(weight_chunks),
        camera_centers=np.concatenate(center_chunks),
        view_names=sample_view_names,
    )


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
        else:
            out[key] = value
    return out


def surface_edges(faces_local: torch.Tensor) -> torch.Tensor:
    if faces_local.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    edges = torch.cat([faces_local[:, [0, 1]], faces_local[:, [1, 2]], faces_local[:, [2, 0]]], dim=0)
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def localize_samples(
    faces: torch.Tensor,
    selected_faces: list[int],
    samples: PixelSamples,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_vertex_ids = faces[torch.as_tensor(selected_faces, dtype=torch.long)].long().reshape(-1)
    selected_faces_local = torch.arange(len(selected_faces) * 3, dtype=torch.long).reshape(-1, 3)
    face_to_local = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    sample_faces_local = torch.as_tensor([face_to_local[int(fid)] for fid in samples.face_ids], dtype=torch.long)
    sample_vertex_ids = selected_faces_local[sample_faces_local]
    return source_vertex_ids, selected_faces_local, sample_vertex_ids


def _sh_basis(
    vertices_local: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    bary: torch.Tensor,
    camera_centers: torch.Tensor,
    *,
    degree: int,
) -> torch.Tensor:
    degree = int(degree)
    basis_count = (degree + 1) ** 2
    if sample_vertex_ids.numel() == 0:
        return torch.empty((0, 3, basis_count), dtype=torch.float32, device=vertices_local.device)
    vpos = vertices_local[sample_vertex_ids]
    dirs = vpos - camera_centers[:, None, :]
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]
    terms = [
        torch.full_like(x, float(C0)),
        -float(C1) * y,
        float(C1) * z,
        -float(C1) * x,
    ]
    if degree >= 2:
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        yz = y * z
        xz = x * z
        terms.extend(
            [
                float(C2[0]) * xy,
                float(C2[1]) * yz,
                float(C2[2]) * (2.0 * zz - xx - yy),
                float(C2[3]) * xz,
                float(C2[4]) * (xx - yy),
            ]
        )
    if degree >= 3:
        terms.extend(
            [
                float(C3[0]) * y * (3.0 * xx - yy),
                float(C3[1]) * xy * z,
                float(C3[2]) * y * (4.0 * zz - xx - yy),
                float(C3[3]) * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
                float(C3[4]) * x * (4.0 * zz - xx - yy),
                float(C3[5]) * z * (xx - yy),
                float(C3[6]) * x * (xx - 3.0 * yy),
            ]
        )
    basis = torch.stack(terms, dim=-1)
    return basis * bary[:, :, None]


def _predict(coeff: torch.Tensor, sample_vertex_ids: torch.Tensor, weighted_basis: torch.Tensor) -> torch.Tensor:
    if sample_vertex_ids.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=coeff.device)
    sample_coeff = coeff[sample_vertex_ids]
    return (sample_coeff * weighted_basis[:, :, :, None]).sum(dim=(1, 2))


def _weighted_mse(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if sample_vertex_ids.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=coeff.device)
    pred = _predict(coeff, sample_vertex_ids, weighted_basis)
    return (((pred - target) ** 2) * weights[:, None]).sum() / (weights.sum().clamp_min(1e-8) * 3.0)


def evaluate_proxy(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, float]:
    if sample_vertex_ids.numel() == 0:
        return {
            "samples": 0,
            "mse_before": 0.0,
            "mse_after": 0.0,
            "relative_gain": 0.0,
            "mae_before": 0.0,
            "mae_after": 0.0,
        }
    zero = torch.zeros_like(coeff)
    with torch.no_grad():
        mse_before = _weighted_mse(zero, sample_vertex_ids, weighted_basis, target, weights)
        mse_after = _weighted_mse(coeff, sample_vertex_ids, weighted_basis, target, weights)
        pred_after = _predict(coeff, sample_vertex_ids, weighted_basis)
        mae_before = (target.abs() * weights[:, None]).sum() / (weights.sum().clamp_min(1e-8) * 3.0)
        mae_after = ((pred_after - target).abs() * weights[:, None]).sum() / (weights.sum().clamp_min(1e-8) * 3.0)
        gain = (mse_before - mse_after) / mse_before.clamp_min(1e-12)
    return {
        "samples": int(sample_vertex_ids.shape[0]),
        "mse_before": float(mse_before.detach().cpu().item()),
        "mse_after": float(mse_after.detach().cpu().item()),
        "relative_gain": float(gain.detach().cpu().item()),
        "mae_before": float(mae_before.detach().cpu().item()),
        "mae_after": float(mae_after.detach().cpu().item()),
    }


def evaluate_proxy_by_face(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    sample_face_ids: np.ndarray,
) -> dict[int, dict[str, float]]:
    if sample_vertex_ids.numel() == 0:
        return {}
    with torch.no_grad():
        pred = _predict(coeff, sample_vertex_ids, weighted_basis).detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    weight_np = weights.detach().cpu().numpy().reshape(-1)
    face_np = sample_face_ids.astype(np.int64, copy=False).reshape(-1)
    out: dict[int, dict[str, float]] = {}
    for fid in np.unique(face_np).tolist():
        mask = face_np == int(fid)
        if not np.any(mask):
            continue
        w = weight_np[mask].reshape(-1, 1)
        y = target_np[mask]
        p = pred[mask]
        denom = float(max(float(w.sum()) * 3.0, 1e-8))
        mse_before = float(((y**2) * w).sum() / denom)
        mse_after = float((((p - y) ** 2) * w).sum() / denom)
        mae_before = float((np.abs(y) * w).sum() / denom)
        mae_after = float((np.abs(p - y) * w).sum() / denom)
        out[int(fid)] = {
            "samples": int(mask.sum()),
            "mse_before": mse_before,
            "mse_after": mse_after,
            "relative_gain": float((mse_before - mse_after) / max(mse_before, 1e-12)),
            "mae_before": mae_before,
            "mae_after": mae_after,
        }
    return out


def face_view_consensus_report(
    samples: PixelSamples,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    min_consensus: float,
    min_views: int,
    min_view_samples: int,
    min_cosine: float,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    enabled = float(min_consensus) > 0.0
    summary: dict[str, Any] = {
        "enabled": bool(enabled),
        "min_face_view_consensus": float(min_consensus),
        "min_face_consensus_views": int(min_views),
        "min_face_consensus_view_samples": int(min_view_samples),
        "face_consensus_min_cosine": float(min_cosine),
        "faces_evaluated": 0,
        "faces_passing": 0,
    }
    if not enabled:
        return {}, summary
    if samples.count == 0:
        return {}, summary

    target_np = target.detach().cpu().numpy().astype(np.float32, copy=False)
    weight_np = weights.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    face_np = samples.face_ids.astype(np.int64, copy=False).reshape(-1)
    view_np = np.asarray(samples.view_names, dtype=object)
    if target_np.shape[0] != face_np.shape[0] or view_np.shape[0] != face_np.shape[0]:
        raise ValueError("sample face/view arrays do not match target shape")

    required_views = max(int(min_views), 1)
    required_view_samples = max(int(min_view_samples), 1)
    out: dict[int, dict[str, Any]] = {}
    for fid in np.unique(face_np).tolist():
        face_mask = face_np == int(fid)
        view_vectors: list[np.ndarray] = []
        view_names: list[str] = []
        view_sample_counts: list[int] = []
        for view_name in sorted(set(str(v) for v in view_np[face_mask].tolist())):
            view_mask = face_mask & (view_np == view_name)
            sample_count = int(view_mask.sum())
            if sample_count < required_view_samples:
                continue
            w = weight_np[view_mask].reshape(-1, 1)
            denom = max(float(w.sum()), 1e-8)
            vector = (target_np[view_mask] * w).sum(axis=0) / denom
            if float(np.linalg.norm(vector)) <= 1e-10:
                continue
            view_vectors.append(vector.astype(np.float32, copy=False))
            view_names.append(view_name)
            view_sample_counts.append(sample_count)
        if view_vectors:
            vectors = np.stack(view_vectors, axis=0)
            direction = vectors.mean(axis=0)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-10:
                norms = np.maximum(np.linalg.norm(vectors, axis=1), 1e-10)
                cosines = (vectors @ direction) / (norms * direction_norm)
                agreeing = int(np.sum(cosines >= float(min_cosine)))
            else:
                cosines = np.zeros((len(view_vectors),), dtype=np.float32)
                agreeing = 0
        else:
            direction_norm = 0.0
            cosines = np.empty((0,), dtype=np.float32)
            agreeing = 0
        view_count = int(len(view_vectors))
        consensus = float(agreeing / max(view_count, 1))
        passed = bool(view_count >= required_views and consensus >= float(min_consensus))
        out[int(fid)] = {
            "view_count": view_count,
            "agreeing_views": agreeing,
            "consensus": consensus,
            "residual_norm": float(direction_norm),
            "mean_cosine": float(np.mean(cosines)) if cosines.size else 0.0,
            "min_cosine": float(np.min(cosines)) if cosines.size else 0.0,
            "passed": passed,
            "view_names": view_names[:16],
            "view_sample_counts": view_sample_counts[:16],
        }

    summary["faces_evaluated"] = int(len(out))
    summary["faces_passing"] = int(sum(1 for row in out.values() if bool(row.get("passed", False))))
    return out, summary


def face_view_gain_certificate_report(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    samples: PixelSamples,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    min_views: int,
    min_relative_gain: float,
    min_view_samples: int,
    min_fraction: float,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    enabled = int(min_views) > 0
    summary: dict[str, Any] = {
        "enabled": bool(enabled),
        "min_face_gain_certificate_views": int(min_views),
        "min_face_gain_certificate_relative_gain": float(min_relative_gain),
        "min_face_gain_certificate_view_samples": int(min_view_samples),
        "min_face_gain_certificate_fraction": float(min_fraction),
        "faces_evaluated": 0,
        "faces_passing": 0,
        "eligible_views": 0,
        "beneficial_views": 0,
        "mean_beneficial_fraction": 0.0,
    }
    if not enabled:
        return {}, summary
    if samples.count == 0 or sample_vertex_ids.numel() == 0:
        return {}, summary

    with torch.no_grad():
        pred_np = _predict(coeff, sample_vertex_ids, weighted_basis).detach().cpu().numpy().astype(np.float32, copy=False)
    target_np = target.detach().cpu().numpy().astype(np.float32, copy=False)
    weight_np = weights.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    face_np = samples.face_ids.astype(np.int64, copy=False).reshape(-1)
    view_np = np.asarray(samples.view_names, dtype=object)
    if (
        pred_np.shape[0] != target_np.shape[0]
        or target_np.shape[0] != face_np.shape[0]
        or view_np.shape[0] != face_np.shape[0]
    ):
        raise ValueError("sample prediction/target/face/view arrays do not match")

    required_views = max(int(min_views), 1)
    required_view_samples = max(int(min_view_samples), 1)
    required_fraction = max(float(min_fraction), 0.0)
    out: dict[int, dict[str, Any]] = {}
    beneficial_fractions: list[float] = []
    for fid in np.unique(face_np).tolist():
        face_mask = face_np == int(fid)
        view_rows: list[dict[str, Any]] = []
        for view_name in sorted(set(str(v) for v in view_np[face_mask].tolist())):
            view_mask = face_mask & (view_np == view_name)
            sample_count = int(view_mask.sum())
            if sample_count < required_view_samples:
                continue
            w = weight_np[view_mask].reshape(-1, 1)
            y = target_np[view_mask]
            p = pred_np[view_mask]
            denom = max(float(w.sum()) * 3.0, 1e-8)
            mse_before = float(((y**2) * w).sum() / denom)
            mse_after = float((((p - y) ** 2) * w).sum() / denom)
            relative_gain = float((mse_before - mse_after) / max(mse_before, 1e-12))
            view_rows.append(
                {
                    "view_name": view_name,
                    "samples": sample_count,
                    "mse_before": mse_before,
                    "mse_after": mse_after,
                    "relative_gain": relative_gain,
                    "passed": bool(relative_gain >= float(min_relative_gain)),
                }
            )

        eligible = int(len(view_rows))
        beneficial = int(sum(1 for row in view_rows if bool(row["passed"])))
        fraction = float(beneficial / max(eligible, 1))
        passed = bool(eligible >= required_views and beneficial >= required_views and fraction >= required_fraction)
        gains = [float(row["relative_gain"]) for row in view_rows]
        beneficial_fractions.append(fraction)
        out[int(fid)] = {
            "eligible_view_count": eligible,
            "beneficial_view_count": beneficial,
            "beneficial_fraction": fraction,
            "min_relative_gain": float(min(gains)) if gains else 0.0,
            "mean_relative_gain": float(np.mean(gains)) if gains else 0.0,
            "max_relative_gain": float(max(gains)) if gains else 0.0,
            "passed": passed,
            "views": view_rows[:16],
        }

    summary["faces_evaluated"] = int(len(out))
    summary["faces_passing"] = int(sum(1 for row in out.values() if bool(row.get("passed", False))))
    summary["eligible_views"] = int(sum(int(row.get("eligible_view_count", 0)) for row in out.values()))
    summary["beneficial_views"] = int(sum(int(row.get("beneficial_view_count", 0)) for row in out.values()))
    summary["mean_beneficial_fraction"] = float(np.mean(beneficial_fractions)) if beneficial_fractions else 0.0
    return out, summary


def calibrate_coeff_by_policy_val(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    samples: PixelSamples,
    target: torch.Tensor,
    weights: torch.Tensor,
    selected_faces: list[int],
    *,
    mode: str,
    min_samples: int,
) -> tuple[torch.Tensor, dict[str, Any], dict[int, dict[str, Any]]]:
    mode = str(mode)
    summary: dict[str, Any] = {
        "mode": mode,
        "enabled": mode != "none",
        "min_samples": int(min_samples),
        "global_scale": 1.0,
        "faces_evaluated": 0,
        "faces_scaled": 0,
        "zero_scale_faces": 0,
        "mean_scale": 1.0,
        "min_scale": 1.0,
        "max_scale": 1.0,
    }
    if mode == "none":
        return coeff, summary, {}
    if samples.count == 0 or sample_vertex_ids.numel() == 0 or coeff.numel() == 0:
        if mode == "global":
            return coeff * 0.0, {**summary, "global_scale": 0.0, "mean_scale": 0.0, "min_scale": 0.0, "max_scale": 0.0}, {}
        return coeff * 0.0, {**summary, "mean_scale": 0.0, "min_scale": 0.0, "max_scale": 0.0}, {}

    with torch.no_grad():
        pred_np = _predict(coeff, sample_vertex_ids, weighted_basis).detach().cpu().numpy().astype(np.float32, copy=False)
    target_np = target.detach().cpu().numpy().astype(np.float32, copy=False)
    weight_np = weights.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1, 1)
    face_np = samples.face_ids.astype(np.int64, copy=False).reshape(-1)
    if pred_np.shape[0] != target_np.shape[0] or face_np.shape[0] != target_np.shape[0]:
        raise ValueError("sample prediction/target/face arrays do not match for validation shrink")

    def fit_scale(mask: np.ndarray) -> tuple[float, int, float]:
        sample_count = int(mask.sum())
        if sample_count < max(int(min_samples), 1):
            return 0.0, sample_count, 0.0
        p = pred_np[mask]
        y = target_np[mask]
        w = weight_np[mask]
        numerator = float((w * p * y).sum())
        denominator = float((w * p * p).sum())
        if denominator <= 1e-12:
            return 0.0, sample_count, 0.0
        raw_scale = numerator / denominator
        scale = float(min(max(raw_scale, 0.0), 1.0))
        return scale, sample_count, float(raw_scale)

    if mode == "global":
        scale, sample_count, raw_scale = fit_scale(np.ones((target_np.shape[0],), dtype=bool))
        out = coeff * float(scale)
        summary.update(
            {
                "global_scale": scale,
                "samples": sample_count,
                "raw_global_scale": raw_scale,
                "mean_scale": scale,
                "min_scale": scale,
                "max_scale": scale,
                "faces_evaluated": int(len(np.unique(face_np))) if face_np.size else 0,
                "faces_scaled": int(len(np.unique(face_np))) if scale < 0.999999 and face_np.size else 0,
                "zero_scale_faces": int(len(np.unique(face_np))) if scale <= 1e-8 and face_np.size else 0,
            }
        )
        return out, summary, {}

    if mode != "face":
        raise ValueError(f"unsupported validation shrink mode: {mode}")

    out = coeff.clone()
    face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    per_face: dict[int, dict[str, Any]] = {}
    scales: list[float] = []
    for fid in selected_faces:
        face_id = int(fid)
        scale, sample_count, raw_scale = fit_scale(face_np == face_id)
        scales.append(scale)
        row = face_to_selected[face_id]
        out[row * 3 : row * 3 + 3] = out[row * 3 : row * 3 + 3] * float(scale)
        per_face[face_id] = {
            "scale": float(scale),
            "raw_scale": float(raw_scale),
            "samples": int(sample_count),
            "passed_min_samples": bool(sample_count >= max(int(min_samples), 1)),
        }

    if scales:
        scale_np = np.asarray(scales, dtype=np.float32)
        summary.update(
            {
                "faces_evaluated": int(len(scales)),
                "faces_scaled": int(np.sum(scale_np < 0.999999)),
                "zero_scale_faces": int(np.sum(scale_np <= 1e-8)),
                "mean_scale": float(scale_np.mean()),
                "min_scale": float(scale_np.min()),
                "max_scale": float(scale_np.max()),
            }
        )
    return out, summary, per_face


def summarize_crossfold_face_gain(
    *,
    coeff: torch.Tensor,
    faces: torch.Tensor,
    selected_faces: list[int],
    source_vertex_ids: torch.Tensor,
    vertices: torch.Tensor,
    view_paths: list[Path],
    face_stats: dict[int, dict[str, float]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    folds = int(args.crossfold_gain_certificate_folds)
    summary: dict[str, Any] = {
        "enabled": bool(folds > 1),
        "certificate_type": "all_train_fold_consistency_not_crossfit",
        "folds": max(folds, 0),
        "min_passing_folds": int(args.crossfold_min_passing_folds),
        "min_fold_relative_gain": float(args.crossfold_min_fold_relative_gain),
        "min_fold_samples": int(args.crossfold_min_fold_samples),
        "faces_evaluated": 0,
        "faces_passing": 0,
        "fold_summaries": [],
    }
    if folds <= 1:
        return {}, summary
    required_passing = int(args.crossfold_min_passing_folds)
    if required_passing <= 0:
        required_passing = folds
    summary["min_passing_folds"] = int(required_passing)
    if not selected_faces or not view_paths or coeff.numel() == 0:
        return {}, summary

    per_face_rows: dict[int, dict[str, Any]] = {
        int(fid): {
            "passing_folds": 0,
            "eligible_folds": 0,
            "folds": [],
        }
        for fid in selected_faces
    }
    vertices_local = vertices[source_vertex_ids].float() if source_vertex_ids.numel() else torch.empty((0, 3), dtype=torch.float32)
    for fold_idx in range(folds):
        fold_paths = [path for idx, path in enumerate(view_paths) if idx % folds == fold_idx]
        fold_samples = collect_samples(
            fold_paths,
            selected_faces,
            face_stats,
            high_error_quantile=float(args.high_error_quantile),
            min_alpha=float(args.min_alpha),
            barycentric_tolerance=float(args.barycentric_tolerance),
            max_samples_per_face_view=int(args.max_samples_per_face_view),
            max_total_samples=max(int(args.max_total_samples // max(folds, 1)), 1),
            uniform_barycentric=bool(args.uniform_barycentric),
        )
        if fold_samples.count:
            _, _, fold_sample_vertex_ids = localize_samples(faces, selected_faces, fold_samples)
        else:
            fold_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)
        fold_ids, fold_basis, fold_target, fold_weights = samples_to_tensors(
            fold_samples,
            fold_sample_vertex_ids,
            vertices_local,
            strength=float(args.strength),
            max_abs_delta_rgb=float(args.max_abs_delta_rgb),
            sh_degree=int(args.sh_degree),
            device=device,
        )
        fold_proxy = evaluate_proxy(coeff, fold_ids, fold_basis, fold_target, fold_weights)
        fold_face = evaluate_proxy_by_face(coeff, fold_ids, fold_basis, fold_target, fold_weights, fold_samples.face_ids)
        fold_passing_faces = 0
        for fid in selected_faces:
            stats = fold_face.get(int(fid), {})
            samples = int(stats.get("samples", 0))
            relative_gain = float(stats.get("relative_gain", -1.0))
            eligible = samples >= int(args.crossfold_min_fold_samples)
            passed = bool(eligible and relative_gain >= float(args.crossfold_min_fold_relative_gain))
            row = per_face_rows[int(fid)]
            if eligible:
                row["eligible_folds"] += 1
            if passed:
                row["passing_folds"] += 1
                fold_passing_faces += 1
            row["folds"].append(
                {
                    "fold": int(fold_idx),
                    "samples": samples,
                    "relative_gain": relative_gain,
                    "eligible": bool(eligible),
                    "passed": bool(passed),
                }
            )
        summary["fold_summaries"].append(
            {
                "fold": int(fold_idx),
                "view_names": [p.stem for p in fold_paths],
                "samples": int(fold_samples.count),
                "proxy": fold_proxy,
                "passing_faces": int(fold_passing_faces),
            }
        )

    for fid, row in per_face_rows.items():
        gains = [float(fold["relative_gain"]) for fold in row["folds"] if bool(fold["eligible"])]
        row["passed"] = bool(int(row["passing_folds"]) >= required_passing)
        row["min_relative_gain"] = float(min(gains)) if gains else 0.0
        row["mean_relative_gain"] = float(np.mean(gains)) if gains else 0.0
    summary["faces_evaluated"] = int(len(per_face_rows))
    summary["faces_passing"] = int(sum(1 for row in per_face_rows.values() if bool(row.get("passed", False))))
    return per_face_rows, summary


def face_residual_direction(face_stats: dict[int, dict[str, float]], face_id: int) -> np.ndarray:
    stats = face_stats.get(int(face_id), {})
    vec = np.asarray(
        [
            float(stats.get("mean_residual_r", 0.0)),
            float(stats.get("mean_residual_g", 0.0)),
            float(stats.get("mean_residual_b", 0.0)),
        ],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.zeros((3,), dtype=np.float32)
    return vec / norm


def residual_direction_cosine(face_stats: dict[int, dict[str, float]], a: int, b: int) -> float:
    da = face_residual_direction(face_stats, int(a))
    db = face_residual_direction(face_stats, int(b))
    if float(np.linalg.norm(da)) <= 1e-8 or float(np.linalg.norm(db)) <= 1e-8:
        return 0.0
    return float(np.clip(float(np.dot(da, db)), -1.0, 1.0))


def selected_face_adjacency(faces: torch.Tensor, selected_faces: list[int]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {int(fid): set() for fid in selected_faces}
    vertex_to_faces: dict[int, list[int]] = {}
    for fid in selected_faces:
        face_id = int(fid)
        if face_id < 0 or face_id >= int(faces.shape[0]):
            continue
        for vertex_id in faces[face_id].detach().cpu().long().tolist():
            vertex_to_faces.setdefault(int(vertex_id), []).append(face_id)
    for incident in vertex_to_faces.values():
        if len(incident) < 2:
            continue
        for fid in incident:
            row = adjacency.setdefault(int(fid), set())
            for other in incident:
                if int(other) != int(fid):
                    row.add(int(other))
    return adjacency


def selected_face_centers(
    faces: torch.Tensor,
    vertices: torch.Tensor,
    selected_faces: list[int],
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    if not selected_faces:
        return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float32), {}
    face_ids = np.asarray([int(fid) for fid in selected_faces], dtype=np.int64)
    valid = (face_ids >= 0) & (face_ids < int(faces.shape[0]))
    face_ids = face_ids[valid]
    if face_ids.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float32), {}
    face_tensor = faces[torch.as_tensor(face_ids, dtype=torch.long)].detach().cpu().long()
    centers = vertices[face_tensor].detach().cpu().float().mean(dim=1).numpy().astype(np.float32, copy=False)
    return face_ids, centers, {int(fid): idx for idx, fid in enumerate(face_ids.tolist())}


def centroid_neighbor_candidates(
    seed: int,
    face_ids: np.ndarray,
    centers: np.ndarray,
    center_index: dict[int, int],
    max_candidates: int,
) -> list[int]:
    idx = center_index.get(int(seed))
    if idx is None or centers.shape[0] <= 1:
        return []
    delta = centers - centers[idx : idx + 1]
    dist2 = np.sum(delta * delta, axis=1)
    out: list[int] = []
    for j in np.argsort(dist2):
        fid = int(face_ids[int(j)])
        if fid == int(seed):
            continue
        out.append(fid)
        if len(out) >= int(max_candidates):
            break
    return out


def evaluate_proxy_for_faces(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    sample_face_ids: np.ndarray,
    face_ids: list[int],
) -> dict[str, float]:
    if not face_ids or sample_vertex_ids.numel() == 0:
        return evaluate_proxy(
            coeff,
            sample_vertex_ids[:0],
            weighted_basis[:0],
            target[:0],
            weights[:0],
        )
    mask = np.isin(sample_face_ids.astype(np.int64, copy=False), np.asarray(face_ids, dtype=np.int64))
    if not np.any(mask):
        return evaluate_proxy(
            coeff,
            sample_vertex_ids[:0],
            weighted_basis[:0],
            target[:0],
            weights[:0],
        )
    idx = torch.as_tensor(np.nonzero(mask)[0], dtype=torch.long, device=sample_vertex_ids.device)
    return evaluate_proxy(coeff, sample_vertex_ids[idx], weighted_basis[idx], target[idx], weights[idx])


def build_patch_crossfold_cache(
    *,
    coeff: torch.Tensor,
    faces: torch.Tensor,
    selected_faces: list[int],
    source_vertex_ids: torch.Tensor,
    vertices: torch.Tensor,
    view_paths: list[Path],
    face_stats: dict[int, dict[str, float]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    folds = int(args.patch_cert_crossfold_folds)
    summary: dict[str, Any] = {
        "enabled": bool(folds > 1),
        "certificate_type": "all_train_patch_fold_consistency_not_crossfit",
        "folds": max(folds, 0),
        "min_passing_folds": int(args.patch_cert_crossfold_min_passing_folds),
        "min_fold_relative_gain": float(args.patch_cert_crossfold_min_fold_relative_gain),
        "min_fold_samples": int(args.patch_cert_crossfold_min_fold_samples),
        "fold_summaries": [],
    }
    if folds <= 1 or not selected_faces or not view_paths or coeff.numel() == 0:
        return [], summary
    required_passing = int(args.patch_cert_crossfold_min_passing_folds)
    if required_passing <= 0:
        required_passing = folds
    summary["min_passing_folds"] = int(required_passing)
    vertices_local = vertices[source_vertex_ids].float() if source_vertex_ids.numel() else torch.empty((0, 3), dtype=torch.float32)
    cache: list[dict[str, Any]] = []
    for fold_idx in range(folds):
        fold_paths = [path for idx, path in enumerate(view_paths) if idx % folds == fold_idx]
        fold_samples = collect_samples(
            fold_paths,
            selected_faces,
            face_stats,
            high_error_quantile=float(args.high_error_quantile),
            min_alpha=float(args.min_alpha),
            barycentric_tolerance=float(args.barycentric_tolerance),
            max_samples_per_face_view=int(args.max_samples_per_face_view),
            max_total_samples=max(int(args.max_total_samples // max(folds, 1)), 1),
            uniform_barycentric=bool(args.uniform_barycentric),
        )
        if fold_samples.count:
            _, _, fold_sample_vertex_ids = localize_samples(faces, selected_faces, fold_samples)
        else:
            fold_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)
        fold_ids, fold_basis, fold_target, fold_weights = samples_to_tensors(
            fold_samples,
            fold_sample_vertex_ids,
            vertices_local,
            strength=float(args.strength),
            max_abs_delta_rgb=float(args.max_abs_delta_rgb),
            sh_degree=int(args.sh_degree),
            device=device,
        )
        fold_proxy = evaluate_proxy(coeff, fold_ids, fold_basis, fold_target, fold_weights)
        row = {
            "fold": int(fold_idx),
            "view_names": [p.stem for p in fold_paths],
            "samples": int(fold_samples.count),
            "proxy": fold_proxy,
            "ids": fold_ids,
            "basis": fold_basis,
            "target": fold_target,
            "weights": fold_weights,
            "face_ids": fold_samples.face_ids,
        }
        summary["fold_summaries"].append(
            {
                "fold": int(fold_idx),
                "view_names": row["view_names"],
                "samples": int(fold_samples.count),
                "proxy": fold_proxy,
            }
        )
        cache.append(row)
    return cache, summary


def patch_crossfold_certificate_for_faces(
    coeff: torch.Tensor,
    fold_cache: list[dict[str, Any]],
    face_ids: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    folds = int(args.patch_cert_crossfold_folds)
    enabled = bool(folds > 1)
    required_passing = int(args.patch_cert_crossfold_min_passing_folds)
    if required_passing <= 0:
        required_passing = max(folds, 0)
    result: dict[str, Any] = {
        "enabled": enabled,
        "folds": max(folds, 0),
        "min_passing_folds": int(required_passing),
        "min_fold_relative_gain": float(args.patch_cert_crossfold_min_fold_relative_gain),
        "min_fold_samples": int(args.patch_cert_crossfold_min_fold_samples),
        "passing_folds": 0,
        "eligible_folds": 0,
        "passed": not enabled,
        "fold_rows": [],
    }
    if not enabled:
        return result
    gains: list[float] = []
    for fold in fold_cache:
        proxy = evaluate_proxy_for_faces(
            coeff,
            fold["ids"],
            fold["basis"],
            fold["target"],
            fold["weights"],
            fold["face_ids"],
            face_ids,
        )
        samples = int(proxy.get("samples", 0))
        relative_gain = float(proxy.get("relative_gain", -1.0))
        eligible = samples >= int(args.patch_cert_crossfold_min_fold_samples)
        passed = bool(eligible and relative_gain >= float(args.patch_cert_crossfold_min_fold_relative_gain))
        if eligible:
            result["eligible_folds"] = int(result["eligible_folds"]) + 1
            gains.append(relative_gain)
        if passed:
            result["passing_folds"] = int(result["passing_folds"]) + 1
        result["fold_rows"].append(
            {
                "fold": int(fold.get("fold", len(result["fold_rows"]))),
                "samples": samples,
                "relative_gain": relative_gain,
                "eligible": bool(eligible),
                "passed": bool(passed),
            }
        )
    result["passed"] = bool(int(result["passing_folds"]) >= required_passing)
    result["min_relative_gain"] = float(min(gains)) if gains else 0.0
    result["mean_relative_gain"] = float(np.mean(np.asarray(gains, dtype=np.float32))) if gains else 0.0
    return result


def clone_face_coeffs(coeff: torch.Tensor, selected_faces: list[int], face_ids: list[int]) -> dict[int, torch.Tensor]:
    if not face_ids or coeff.numel() == 0:
        return {}
    face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    out: dict[int, torch.Tensor] = {}
    for fid in face_ids:
        row = face_to_selected.get(int(fid))
        if row is None:
            continue
        out[int(fid)] = coeff[row * 3 : row * 3 + 3].clone()
    return out


def restore_face_coeffs(coeff: torch.Tensor, selected_faces: list[int], snapshot: dict[int, torch.Tensor]) -> None:
    if not snapshot or coeff.numel() == 0:
        return
    face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    for fid, value in snapshot.items():
        row = face_to_selected.get(int(fid))
        if row is None:
            continue
        coeff[row * 3 : row * 3 + 3] = value.to(device=coeff.device, dtype=coeff.dtype)


def fit_patch_scale(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    sample_face_ids: np.ndarray,
    face_ids: list[int],
    min_samples: int,
) -> tuple[float, int, float]:
    if not face_ids or sample_vertex_ids.numel() == 0:
        return 0.0, 0, 0.0
    mask = np.isin(sample_face_ids.astype(np.int64, copy=False), np.asarray(face_ids, dtype=np.int64))
    sample_count = int(mask.sum())
    if sample_count < max(int(min_samples), 1):
        return 0.0, sample_count, 0.0
    idx = torch.as_tensor(np.nonzero(mask)[0], dtype=torch.long, device=sample_vertex_ids.device)
    with torch.no_grad():
        pred = _predict(coeff, sample_vertex_ids[idx], weighted_basis[idx])
        y = target[idx]
        w = weights[idx].clamp_min(1e-8).view(-1, 1)
        numerator = float((w * pred * y).sum().detach().cpu().item())
        denominator = float((w * pred * pred).sum().detach().cpu().item())
    if denominator <= 1e-12:
        return 0.0, sample_count, 0.0
    raw_scale = numerator / denominator
    return float(min(max(raw_scale, 0.0), 1.0)), sample_count, float(raw_scale)


def scale_face_coeffs(
    coeff: torch.Tensor,
    selected_faces: list[int],
    face_ids: list[int],
    scale: float,
) -> None:
    if not face_ids or coeff.numel() == 0:
        return
    face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    for fid in face_ids:
        row = face_to_selected.get(int(fid))
        if row is None:
            continue
        coeff[row * 3 : row * 3 + 3] = coeff[row * 3 : row * 3 + 3] * float(scale)


def grow_patch_certified_faces(
    *,
    coeff: torch.Tensor,
    faces: torch.Tensor,
    vertices: torch.Tensor,
    selected_faces: list[int],
    seed_faces: list[int],
    face_stats: dict[int, dict[str, float]],
    face_policy: dict[int, dict[str, float]],
    val_ids: torch.Tensor,
    val_basis: torch.Tensor,
    val_target: torch.Tensor,
    val_weights: torch.Tensor,
    val_samples: PixelSamples,
    patch_crossfold_cache: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any], dict[int, dict[str, Any]]]:
    rings = int(args.patch_cert_rings)
    patch_crossfold_enabled = bool(int(args.patch_cert_crossfold_folds) > 1)
    patch_neighbor_crossfold = bool(args.patch_cert_neighbor_crossfold) and patch_crossfold_enabled
    summary: dict[str, Any] = {
        "enabled": bool(rings > 0),
        "rings": max(rings, 0),
        "max_faces_per_seed": int(args.patch_cert_max_faces_per_seed),
        "min_direction_cosine": float(args.patch_cert_min_direction_cosine),
        "min_neighbor_policy_val_samples": int(args.patch_cert_min_neighbor_policy_val_samples),
        "min_neighbor_policy_val_relative_gain": float(args.patch_cert_min_neighbor_policy_val_relative_gain),
        "min_policy_val_samples": int(args.patch_cert_min_policy_val_samples),
        "min_relative_gain": float(args.patch_cert_min_relative_gain),
        "neighbor_mode": str(args.patch_cert_neighbor_mode),
        "centroid_candidates_per_seed": int(args.patch_cert_centroid_candidates_per_seed),
        "patch_shrink": bool(args.patch_cert_shrink),
        "patch_crossfold_enabled": patch_crossfold_enabled,
        "patch_crossfold_folds": int(args.patch_cert_crossfold_folds),
        "patch_crossfold_min_passing_folds": int(args.patch_cert_crossfold_min_passing_folds),
        "patch_crossfold_min_fold_relative_gain": float(args.patch_cert_crossfold_min_fold_relative_gain),
        "patch_crossfold_min_fold_samples": int(args.patch_cert_crossfold_min_fold_samples),
        "patch_neighbor_crossfold": patch_neighbor_crossfold,
        "seed_faces": int(len(seed_faces)),
        "accepted_faces_before": int(len(seed_faces)),
        "accepted_faces_after": int(len(seed_faces)),
        "accepted_patches": 0,
        "rejected_patches": 0,
        "rejected_patch_crossfold": 0,
        "rejected_neighbor_crossfold": 0,
        "rejected_patch_budget": 0,
        "rejected_post_shrink_policy_val": 0,
        "accepted_patch_crossfold": 0,
        "accepted_post_shrink_policy_val": 0,
        "accepted_post_shrink_patch_crossfold": 0,
        "mean_patch_size": 1.0 if seed_faces else 0.0,
        "preview": [],
    }
    if rings <= 0 or not seed_faces:
        return list(seed_faces), summary, {}

    adjacency = selected_face_adjacency(faces, selected_faces)
    centroid_face_ids, centroid_centers, centroid_index = selected_face_centers(faces, vertices, selected_faces)
    selected_set = set(int(fid) for fid in selected_faces)
    assigned: set[int] = set()
    accepted: list[int] = []
    patch_by_face: dict[int, dict[str, Any]] = {}
    patch_sizes: list[int] = []

    for seed in seed_faces:
        seed_id = int(seed)
        if seed_id in assigned:
            continue
        patch: list[int] = [seed_id]
        seen = {seed_id}
        frontier = [seed_id]
        for _ in range(rings):
            next_frontier: list[int] = []
            for fid in frontier:
                neighbors: list[int] = []
                if str(args.patch_cert_neighbor_mode) in {"topology", "both"}:
                    neighbors.extend(sorted(adjacency.get(int(fid), set())))
                if int(fid) == seed_id and str(args.patch_cert_neighbor_mode) in {"centroid", "both"}:
                    neighbors.extend(
                        centroid_neighbor_candidates(
                            seed_id,
                            centroid_face_ids,
                            centroid_centers,
                            centroid_index,
                            int(args.patch_cert_centroid_candidates_per_seed),
                        )
                    )
                deduped_neighbors = []
                seen_neighbors: set[int] = set()
                for nb in neighbors:
                    if int(nb) in seen_neighbors:
                        continue
                    seen_neighbors.add(int(nb))
                    deduped_neighbors.append(int(nb))
                for nb in deduped_neighbors:
                    nb = int(nb)
                    if nb in seen or nb in assigned or nb not in selected_set:
                        continue
                    stats = face_stats.get(nb, {})
                    if int(stats.get("view_hits", 0)) < int(args.min_view_hits):
                        continue
                    if float(stats.get("pixel_count", 0.0)) < float(args.min_pixel_count):
                        continue
                    if residual_direction_cosine(face_stats, seed_id, nb) < float(args.patch_cert_min_direction_cosine):
                        continue
                    proxy = face_policy.get(nb, {})
                    if int(proxy.get("samples", 0)) < int(args.patch_cert_min_neighbor_policy_val_samples):
                        continue
                    if float(proxy.get("relative_gain", -1.0)) < float(args.patch_cert_min_neighbor_policy_val_relative_gain):
                        continue
                    if patch_neighbor_crossfold:
                        neighbor_crossfold = patch_crossfold_certificate_for_faces(coeff, patch_crossfold_cache, [nb], args)
                        if not bool(neighbor_crossfold.get("passed", False)):
                            summary["rejected_neighbor_crossfold"] = int(summary["rejected_neighbor_crossfold"]) + 1
                            continue
                    patch.append(nb)
                    seen.add(nb)
                    next_frontier.append(nb)
                    if len(patch) >= max(int(args.patch_cert_max_faces_per_seed), 1):
                        break
                if len(patch) >= max(int(args.patch_cert_max_faces_per_seed), 1):
                    break
            frontier = next_frontier
            if not frontier or len(patch) >= max(int(args.patch_cert_max_faces_per_seed), 1):
                break

        proxy_before_shrink = evaluate_proxy_for_faces(
            coeff,
            val_ids,
            val_basis,
            val_target,
            val_weights,
            val_samples.face_ids,
            patch,
        )
        passed = (
            int(proxy_before_shrink.get("samples", 0)) >= int(args.patch_cert_min_policy_val_samples)
            and float(proxy_before_shrink.get("relative_gain", -1.0)) >= float(args.patch_cert_min_relative_gain)
        )
        patch_crossfold = patch_crossfold_certificate_for_faces(coeff, patch_crossfold_cache, patch, args)
        if patch_crossfold_enabled and not bool(patch_crossfold.get("passed", False)):
            passed = False
        if not passed:
            summary["rejected_patches"] = int(summary["rejected_patches"]) + 1
            if patch_crossfold_enabled and not bool(patch_crossfold.get("passed", False)):
                summary["rejected_patch_crossfold"] = int(summary["rejected_patch_crossfold"]) + 1
            patch = [seed_id]
            proxy_before_shrink = evaluate_proxy_for_faces(
                coeff,
                val_ids,
                val_basis,
                val_target,
                val_weights,
                val_samples.face_ids,
                patch,
            )
            patch_crossfold = patch_crossfold_certificate_for_faces(coeff, patch_crossfold_cache, patch, args)
            if patch_crossfold_enabled:
                passed = (
                    int(proxy_before_shrink.get("samples", 0)) >= int(args.patch_cert_min_policy_val_samples)
                    and float(proxy_before_shrink.get("relative_gain", -1.0)) >= float(args.patch_cert_min_relative_gain)
                    and bool(patch_crossfold.get("passed", False))
                )
            else:
                # Preserve historical PatchCert behavior when the new patch-fold
                # certificate is disabled: a rejected grown patch falls back to
                # its already face-certified seed instead of rejecting the seed.
                passed = False
            if patch_crossfold_enabled and not passed:
                if patch_crossfold_enabled and not bool(patch_crossfold.get("passed", False)):
                    summary["rejected_patch_crossfold"] = int(summary["rejected_patch_crossfold"]) + 1
                patch_record = {
                    "seed_face": seed_id,
                    "faces": [seed_id],
                    "patch_size": 1,
                    "proxy": proxy_before_shrink,
                    "proxy_before_shrink": proxy_before_shrink,
                    "passed_patch_gain": False,
                    "patch_crossfold_certificate": patch_crossfold,
                    "scale": 0.0,
                    "raw_scale": 0.0,
                    "scale_samples": int(proxy_before_shrink.get("samples", 0)),
                    "rejected": True,
                }
                if len(summary["preview"]) < 20:
                    summary["preview"].append(patch_record)
                continue

        scale = 1.0
        raw_scale = 1.0
        scale_samples = int(proxy_before_shrink.get("samples", 0))
        coeff_snapshot = clone_face_coeffs(coeff, selected_faces, patch)
        if bool(args.patch_cert_shrink) and len(patch) > 1:
            scale, scale_samples, raw_scale = fit_patch_scale(
                coeff,
                val_ids,
                val_basis,
                val_target,
                val_weights,
                val_samples.face_ids,
                patch,
                int(args.patch_cert_min_policy_val_samples),
            )
            scale_face_coeffs(coeff, selected_faces, patch, scale)
        post_shrink_patch_crossfold = patch_crossfold_certificate_for_faces(coeff, patch_crossfold_cache, patch, args)
        if patch_crossfold_enabled and not bool(post_shrink_patch_crossfold.get("passed", False)):
            restore_face_coeffs(coeff, selected_faces, coeff_snapshot)
            summary["rejected_patches"] = int(summary["rejected_patches"]) + 1
            summary["rejected_patch_crossfold"] = int(summary["rejected_patch_crossfold"]) + 1
            patch_record = {
                "seed_face": seed_id,
                "faces": [int(fid) for fid in patch],
                "patch_size": int(len(patch)),
                "proxy": proxy_before_shrink,
                "proxy_before_shrink": proxy_before_shrink,
                "passed_patch_gain": False,
                "patch_crossfold_certificate": patch_crossfold,
                "post_shrink_patch_crossfold_certificate": post_shrink_patch_crossfold,
                "scale": float(scale),
                "raw_scale": float(raw_scale),
                "scale_samples": int(scale_samples),
                "rejected": True,
                "rejected_reason": "post_shrink_patch_crossfold_failed",
            }
            if len(summary["preview"]) < 20:
                summary["preview"].append(patch_record)
            continue
        proxy_after_shrink = evaluate_proxy_for_faces(
            coeff,
            val_ids,
            val_basis,
            val_target,
            val_weights,
            val_samples.face_ids,
            patch,
        )
        post_shrink_policy_pass = (
            int(proxy_after_shrink.get("samples", 0)) >= int(args.patch_cert_min_policy_val_samples)
            and float(proxy_after_shrink.get("relative_gain", -1.0)) >= float(args.patch_cert_min_relative_gain)
        )
        if not post_shrink_policy_pass:
            restore_face_coeffs(coeff, selected_faces, coeff_snapshot)
            summary["rejected_patches"] = int(summary["rejected_patches"]) + 1
            summary["rejected_post_shrink_policy_val"] = int(summary["rejected_post_shrink_policy_val"]) + 1
            patch_record = {
                "seed_face": seed_id,
                "faces": [int(fid) for fid in patch],
                "patch_size": int(len(patch)),
                "proxy": proxy_after_shrink,
                "proxy_before_shrink": proxy_before_shrink,
                "passed_patch_gain": False,
                "patch_crossfold_certificate": patch_crossfold,
                "post_shrink_patch_crossfold_certificate": post_shrink_patch_crossfold,
                "scale": float(scale),
                "raw_scale": float(raw_scale),
                "scale_samples": int(scale_samples),
                "rejected": True,
                "rejected_reason": "post_shrink_policy_val_failed",
            }
            if len(summary["preview"]) < 20:
                summary["preview"].append(patch_record)
            continue
        budget = int(args.max_faces_to_apply)
        if budget >= 0 and len(accepted) + len(patch) > budget:
            restore_face_coeffs(coeff, selected_faces, coeff_snapshot)
            summary["rejected_patches"] = int(summary["rejected_patches"]) + 1
            summary["rejected_patch_budget"] = int(summary["rejected_patch_budget"]) + 1
            patch_record = {
                "seed_face": seed_id,
                "faces": [int(fid) for fid in patch],
                "patch_size": int(len(patch)),
                "proxy": proxy_after_shrink,
                "proxy_before_shrink": proxy_before_shrink,
                "passed_patch_gain": False,
                "patch_crossfold_certificate": patch_crossfold,
                "post_shrink_patch_crossfold_certificate": post_shrink_patch_crossfold,
                "scale": float(scale),
                "raw_scale": float(raw_scale),
                "scale_samples": int(scale_samples),
                "rejected": True,
                "rejected_reason": "patch_budget_would_split_carrier",
            }
            if len(summary["preview"]) < 20:
                summary["preview"].append(patch_record)
            continue

        assigned.update(int(fid) for fid in patch)
        accepted.extend(int(fid) for fid in patch)
        patch_sizes.append(len(patch))
        if len(patch) > 1:
            summary["accepted_patches"] = int(summary["accepted_patches"]) + 1
        if patch_crossfold_enabled and bool(patch_crossfold.get("passed", False)):
            summary["accepted_patch_crossfold"] = int(summary["accepted_patch_crossfold"]) + 1
        if post_shrink_policy_pass:
            summary["accepted_post_shrink_policy_val"] = int(summary["accepted_post_shrink_policy_val"]) + 1
        if patch_crossfold_enabled and bool(post_shrink_patch_crossfold.get("passed", False)):
            summary["accepted_post_shrink_patch_crossfold"] = int(summary["accepted_post_shrink_patch_crossfold"]) + 1
        patch_record = {
            "seed_face": seed_id,
            "faces": [int(fid) for fid in patch],
            "patch_size": int(len(patch)),
            "proxy": proxy_after_shrink,
            "proxy_before_shrink": proxy_before_shrink,
            "passed_patch_gain": bool(passed),
            "patch_crossfold_certificate": patch_crossfold,
            "post_shrink_patch_crossfold_certificate": post_shrink_patch_crossfold,
            "scale": float(scale),
            "raw_scale": float(raw_scale),
            "scale_samples": int(scale_samples),
        }
        for fid in patch:
            patch_by_face[int(fid)] = patch_record
        if len(summary["preview"]) < 20:
            summary["preview"].append(patch_record)

    if patch_sizes:
        summary["mean_patch_size"] = float(np.mean(np.asarray(patch_sizes, dtype=np.float32)))
    summary["accepted_faces_after"] = int(len(accepted))
    return accepted, summary, patch_by_face


def solve_coeff_delta(
    selected_faces_local: torch.Tensor,
    fit_sample_vertex_ids: torch.Tensor,
    fit_weighted_basis: torch.Tensor,
    fit_target: torch.Tensor,
    fit_weights: torch.Tensor,
    *,
    vertex_count: int,
    max_abs_dc_coeff: float,
    max_abs_sh_coeff: float,
    lambda_mag: float,
    lambda_sh1_mag: float,
    lambda_smooth: float,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    basis_count = int(fit_weighted_basis.shape[2]) if fit_weighted_basis.ndim == 3 else 4
    if vertex_count <= 0 or fit_sample_vertex_ids.numel() == 0:
        return torch.empty((0, basis_count, 3), dtype=torch.float32), {
            "initial_fit_mse": 0.0,
            "final_fit_mse": 0.0,
            "final_mag_loss": 0.0,
            "final_sh_mag_loss": 0.0,
            "final_smooth_loss": 0.0,
            "basis_count": int(basis_count),
        }
    fit_sample_vertex_ids = fit_sample_vertex_ids.to(device=device)
    fit_weighted_basis = fit_weighted_basis.to(device=device)
    fit_target = fit_target.to(device=device)
    fit_weights = fit_weights.to(device=device).clamp_min(1e-8)
    selected_faces_local = selected_faces_local.to(device=device)
    edges = surface_edges(selected_faces_local.detach().cpu()).to(device=device)
    bounds = torch.full((basis_count,), float(max_abs_sh_coeff), dtype=torch.float32, device=device)
    bounds[0] = float(max_abs_dc_coeff)
    bounds = bounds.view(1, basis_count, 1)
    param = torch.zeros((int(vertex_count), basis_count, 3), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([param], lr=float(lr))

    with torch.no_grad():
        zero = torch.zeros_like(param)
        initial_fit_mse = _weighted_mse(zero, fit_sample_vertex_ids, fit_weighted_basis, fit_target, fit_weights)

    final_fit_mse = initial_fit_mse
    final_mag_loss = torch.zeros((), dtype=torch.float32, device=device)
    final_sh_mag_loss = torch.zeros((), dtype=torch.float32, device=device)
    final_smooth_loss = torch.zeros((), dtype=torch.float32, device=device)
    for _ in range(int(steps)):
        coeff = bounds * torch.tanh(param)
        data_loss = _weighted_mse(coeff, fit_sample_vertex_ids, fit_weighted_basis, fit_target, fit_weights)
        mag_loss = (coeff[:, 0, :] ** 2).mean()
        sh_mag_loss = (coeff[:, 1:, :] ** 2).mean() if basis_count > 1 else torch.zeros((), dtype=torch.float32, device=device)
        if edges.numel():
            smooth_loss = ((coeff[edges[:, 0]] - coeff[edges[:, 1]]) ** 2).mean()
        else:
            smooth_loss = torch.zeros((), dtype=torch.float32, device=device)
        loss = (
            data_loss
            + float(lambda_mag) * mag_loss
            + float(lambda_sh1_mag) * sh_mag_loss
            + float(lambda_smooth) * smooth_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_fit_mse = data_loss.detach()
        final_mag_loss = mag_loss.detach()
        final_sh_mag_loss = sh_mag_loss.detach()
        final_smooth_loss = smooth_loss.detach()

    with torch.no_grad():
        coeff = (bounds * torch.tanh(param)).detach().cpu()
    return coeff, {
        "initial_fit_mse": float(initial_fit_mse.detach().cpu().item()),
        "final_fit_mse": float(final_fit_mse.detach().cpu().item()),
        "final_mag_loss": float(final_mag_loss.detach().cpu().item()),
        "final_sh_mag_loss": float(final_sh_mag_loss.detach().cpu().item()),
        "basis_count": int(basis_count),
        "final_smooth_loss": float(final_smooth_loss.detach().cpu().item()),
    }


def samples_to_tensors(
    samples: PixelSamples,
    sample_vertex_ids: torch.Tensor,
    vertices_local: torch.Tensor,
    *,
    strength: float,
    max_abs_delta_rgb: float,
    sh_degree: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = sample_vertex_ids.to(device=device)
    bary = torch.as_tensor(samples.barycentric, dtype=torch.float32, device=device)
    centers = torch.as_tensor(samples.camera_centers, dtype=torch.float32, device=device)
    vertices_local = vertices_local.to(device=device)
    weighted_basis = _sh_basis(vertices_local, ids, bary, centers, degree=int(sh_degree))
    target = torch.as_tensor(samples.residual_rgb, dtype=torch.float32, device=device)
    target = (target * float(strength)).clamp(-float(max_abs_delta_rgb), float(max_abs_delta_rgb))
    weights = torch.as_tensor(samples.weights, dtype=torch.float32, device=device)
    return ids, weighted_basis, target, weights


def materialize_facelocal(
    state: dict[str, Any],
    faces: torch.Tensor,
    selected_faces: list[int],
    source_vertex_ids: torch.Tensor,
    coeff: torch.Tensor,
    accepted_faces: list[int],
) -> dict[str, Any]:
    out = clone_state(state)
    if not accepted_faces:
        return out
    vertex_count = int(state["triangles_points"].shape[0])
    face_count = int(faces.shape[0])
    face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    accepted_local_rows = [face_to_selected[int(fid)] for fid in accepted_faces]
    local_vertex_indices: list[int] = []
    for row in accepted_local_rows:
        local_vertex_indices.extend([row * 3, row * 3 + 1, row * 3 + 2])
    local_idx = torch.as_tensor(local_vertex_indices, dtype=torch.long)
    source_idx = source_vertex_ids[local_idx].long()
    coeff_add = coeff[local_idx]

    new_faces = faces.clone()
    start = vertex_count
    for out_row, fid in enumerate(accepted_faces):
        new_faces[int(fid)] = torch.tensor([start + out_row * 3, start + out_row * 3 + 1, start + out_row * 3 + 2])

    for key, value in state.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        cpu = value.detach().cpu()
        if key == "_triangle_indices":
            out[key] = new_faces.to(dtype=value.dtype)
        elif cpu.ndim > 0 and int(cpu.shape[0]) == vertex_count:
            append = cpu[source_idx].clone()
            if key == "features_dc":
                append = append + coeff_add[:, 0:1, :].to(dtype=append.dtype)
            elif key == "features_rest":
                append = append.clone()
                if append.ndim == 3 and append.shape[1] > 0 and coeff_add.shape[1] > 1:
                    rest_count = min(int(append.shape[1]), int(coeff_add.shape[1]) - 1)
                    append[:, :rest_count, :] = append[:, :rest_count, :] + coeff_add[:, 1 : 1 + rest_count, :].to(
                        dtype=append.dtype
                    )
            out[key] = torch.cat([cpu, append], dim=0).to(dtype=value.dtype)
        elif cpu.ndim > 0 and int(cpu.shape[0]) == face_count:
            out[key] = cpu.clone().to(dtype=value.dtype)
        else:
            out[key] = cpu.clone()
    return out


def _parse_face_id_filter(raw: str) -> set[int]:
    out: set[int] = set()
    for item in str(raw or "").replace(" ", ",").split(","):
        if not item:
            continue
        out.add(int(item))
    return out


def read_candidate_plan(
    path: Path,
    *,
    limit: int = 0,
    face_ids: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta: dict[str, Any] = {}
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        meta = payload
        if isinstance(payload.get("candidates"), list):
            rows = payload["candidates"]
        elif isinstance(payload.get("accepted"), list):
            rows = payload["accepted"]
        elif isinstance(payload.get("accepted_preview"), list):
            rows = payload["accepted_preview"]
        else:
            rows = []
    else:
        rows = []
    filtered: list[dict[str, Any]] = []
    keep_ids = face_ids or set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            face_id = int(row.get("face_id", -1))
        except Exception:
            continue
        if keep_ids and face_id not in keep_ids:
            continue
        filtered.append(dict(row))
    if int(limit) > 0:
        filtered = filtered[: int(limit)]
    return filtered, meta


def read_plan_alphas(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("face_alphas", payload.get("alphas", payload)) if isinstance(payload, dict) else payload
    alphas: dict[int, float] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                face_id = int(key)
                alpha = float(value)
            except Exception:
                continue
            if math.isfinite(alpha):
                alphas[face_id] = alpha
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                face_id = int(item.get("face_id"))
                alpha = float(item.get("alpha", item.get("scale", 1.0)))
            except Exception:
                continue
            if math.isfinite(alpha):
                alphas[face_id] = alpha
    return alphas


def validate_strict_materialize_request(args: argparse.Namespace) -> None:
    errors: list[str] = []
    scale = float(args.materialize_plan_scale)
    if int(args.materialize_plan_limit) > 0:
        errors.append("materialize_plan_limit would row-slice a certified carrier")
    if str(args.materialize_plan_face_ids or "").strip():
        errors.append("materialize_plan_face_ids would subset a certified carrier")
    if not math.isfinite(scale) or abs(scale - 1.0) > 1e-12:
        errors.append("materialize_plan_scale would alter certified coefficients")
    if args.materialize_plan_alpha_json is not None:
        errors.append("materialize_plan_alpha_json would alter certified coefficients")
    if errors:
        raise ValueError(
            "Strict certified plan materialization rejected unsafe replay controls: "
            + "; ".join(errors)
            + ". Use --materialize_allow_uncertified_plan only for explicitly labeled legacy ablations."
        )


def validate_strict_plan_carrier_integrity(
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    export_policy = str(meta.get("plan_export_policy", "")) if isinstance(meta, dict) else ""
    if export_policy != "final_certified_accepted_faces_only":
        issues.append(
            {
                "scope": "plan_meta",
                "decision_reason": "plan_export_policy_not_final_certified_accepted_faces_only",
                "plan_export_policy": export_policy,
            }
        )
    if isinstance(meta, dict) and not bool(meta.get("strict_patchcert_carrier", False)):
        issues.append(
            {
                "scope": "plan_meta",
                "decision_reason": "plan_source_not_strict_patchcert_carrier",
                "strict_patchcert_carrier": bool(meta.get("strict_patchcert_carrier", False)),
            }
        )
    row_face_ids: set[int] = set()
    face_counts: dict[int, int] = {}
    for row in rows:
        try:
            face_id = int(row.get("face_id", -1))
        except Exception:
            continue
        row_face_ids.add(face_id)
        face_counts[face_id] = int(face_counts.get(face_id, 0)) + 1
    for face_id, count in sorted(face_counts.items()):
        if count > 1:
            issues.append(
                {
                    "face_id": int(face_id),
                    "decision_reason": "duplicate_face_rows",
                    "row_count": int(count),
                }
            )
    patch_faces_by_face: dict[int, set[int]] = {}
    for row in rows:
        try:
            face_id = int(row.get("face_id", -1))
        except Exception:
            face_id = -1
        patch_cert = row.get("patch_certificate")
        if not isinstance(patch_cert, dict):
            issues.append({"face_id": face_id, "decision_reason": "missing_patch_certificate"})
            continue
        faces_raw = patch_cert.get("faces")
        if not isinstance(faces_raw, list) or not faces_raw:
            issues.append({"face_id": face_id, "decision_reason": "missing_patch_faces"})
            continue
        patch_faces: set[int] = set()
        for value in faces_raw:
            try:
                patch_faces.add(int(value))
            except Exception:
                continue
        if face_id not in patch_faces:
            issues.append({"face_id": face_id, "decision_reason": "patch_certificate_face_mismatch"})
        patch_faces_by_face[face_id] = patch_faces
        missing = sorted(int(fid) for fid in patch_faces if int(fid) not in row_face_ids)
        if missing:
            issues.append(
                {
                    "face_id": face_id,
                    "decision_reason": "patch_carrier_split_by_plan_rows",
                    "missing_patch_faces": missing[:20],
                    "missing_patch_face_count": int(len(missing)),
                }
            )
    for face_id, patch_faces in sorted(patch_faces_by_face.items()):
        for member in sorted(patch_faces):
            member_patch = patch_faces_by_face.get(int(member))
            if member_patch is None:
                continue
            if member_patch != patch_faces:
                issues.append(
                    {
                        "face_id": int(face_id),
                        "decision_reason": "inconsistent_patch_certificate_faces",
                        "other_face_id": int(member),
                    }
                )
    return issues


def plan_rows_to_facelocal_coeff(
    rows: list[dict[str, Any]],
    faces: torch.Tensor,
    *,
    fallback_basis_count: int,
    alpha_by_face: dict[int, float] | None = None,
    require_certified: bool = True,
) -> tuple[list[int], torch.Tensor, list[dict[str, Any]]]:
    selected_faces: list[int] = []
    coeff_rows: list[torch.Tensor] = []
    rejected: list[dict[str, Any]] = []
    face_count = int(faces.shape[0])
    for row in rows:
        face_id = int(row.get("face_id", -1))
        coeff_raw = row.get("delta_coeff", row.get("coeff"))
        reasons: list[str] = []
        if face_id < 0 or face_id >= face_count:
            reasons.append("invalid_face_id")
        if coeff_raw is None:
            reasons.append("missing_delta_coeff")
        if require_certified:
            if not bool(row.get("policy_pass", False)):
                reasons.append("policy_pass_not_true")
            if not bool(row.get("final_certified_face", False)):
                reasons.append("final_certified_face_not_true")
            patch_cert = row.get("patch_certificate")
            if not isinstance(patch_cert, dict):
                reasons.append("missing_patch_certificate")
            else:
                if bool(patch_cert.get("rejected", False)):
                    reasons.append("patch_certificate_rejected")
                if not bool(patch_cert.get("passed_patch_gain", False)):
                    reasons.append("patch_gain_not_passed")
                for key in ("patch_crossfold_certificate", "post_shrink_patch_crossfold_certificate"):
                    cert = patch_cert.get(key)
                    if not isinstance(cert, dict):
                        reasons.append(f"{key}_missing")
                    elif not bool(cert.get("enabled", False)):
                        reasons.append(f"{key}_not_enabled")
                    elif not bool(cert.get("passed", False)):
                        reasons.append(f"{key}_not_passed")
        if reasons:
            rejected.append({"face_id": face_id, "decision_reasons": reasons})
            continue
        coeff = torch.as_tensor(coeff_raw, dtype=torch.float32)
        if coeff.ndim == 2 and coeff.shape == (3, 3):
            coeff = coeff[:, None, :]
        if coeff.ndim != 3 or int(coeff.shape[0]) != 3 or int(coeff.shape[2]) != 3:
            rejected.append(
                {
                    "face_id": face_id,
                    "decision_reasons": ["invalid_delta_coeff_shape"],
                    "shape": list(coeff.shape),
                }
            )
            continue
        alpha = float((alpha_by_face or {}).get(face_id, 1.0))
        if not math.isfinite(alpha) or alpha < 0.0:
            rejected.append(
                {
                    "face_id": face_id,
                    "decision_reasons": ["invalid_materialize_plan_alpha"],
                    "alpha": alpha,
                }
            )
            continue
        coeff = coeff * alpha
        basis_count = int(coeff.shape[1])
        if basis_count <= 0:
            basis_count = int(fallback_basis_count)
        selected_faces.append(face_id)
        coeff_rows.append(coeff[:, :basis_count, :])
    if not coeff_rows:
        basis_count = max(int(fallback_basis_count), 1)
        return selected_faces, torch.empty((0, basis_count, 3), dtype=torch.float32), rejected
    basis_count = max(int(row.shape[1]) for row in coeff_rows)
    padded: list[torch.Tensor] = []
    for coeff in coeff_rows:
        if int(coeff.shape[1]) == basis_count:
            padded.append(coeff)
            continue
        pad = torch.zeros((3, basis_count - int(coeff.shape[1]), 3), dtype=torch.float32)
        padded.append(torch.cat([coeff, pad], dim=1))
    return selected_faces, torch.cat(padded, dim=0), rejected


def write_candidate_plan(
    path: Path,
    *,
    args: argparse.Namespace,
    selected_faces: list[int],
    plan_faces: list[int],
    coeff: torch.Tensor,
    face_stats: dict[int, dict[str, float]],
    face_policy: dict[int, dict[str, float]],
    validation_shrink_by_face: dict[int, dict[str, Any]],
    face_view_gain_certificate: dict[int, dict[str, Any]],
    crossfold_face_gain: dict[int, dict[str, Any]],
    face_view_consensus: dict[int, dict[str, Any]],
    patch_cert_by_face: dict[int, dict[str, Any]],
    fit_proxy: dict[str, float],
    val_proxy: dict[str, float],
) -> None:
    face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    candidates: list[dict[str, Any]] = []
    for fid in plan_faces:
        row = face_to_selected.get(int(fid))
        if row is None:
            continue
        local = coeff[row * 3 : row * 3 + 3].detach().cpu().float()
        candidates.append(
            {
                "face_id": int(fid),
                "rank": int(len(candidates)),
                "delta_coeff": local.tolist(),
                "face_stats": face_stats.get(int(fid), {}),
                "policy_val_proxy": face_policy.get(int(fid), {}),
                "validation_shrink": validation_shrink_by_face.get(int(fid), {}),
                "face_view_gain_certificate": face_view_gain_certificate.get(int(fid), {}),
                "crossfold_face_gain_certificate": crossfold_face_gain.get(int(fid), {}),
                "face_view_consensus": face_view_consensus.get(int(fid), {}),
                "patch_certificate": patch_cert_by_face.get(int(fid), {}),
                "policy_pass": True,
                "final_certified_face": True,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "operator": "surface_residual_facelocal_sh_delta_candidate_plan",
                "test_usage": "none",
                "source_model": str(args.source_model),
                "iteration": int(args.iteration),
                "evidence_dir": str(args.evidence_dir),
                "sh_degree": int(args.sh_degree),
                "basis_count": int((int(args.sh_degree) + 1) ** 2),
                "plan_export_policy": "final_certified_accepted_faces_only",
                "strict_patchcert_carrier": bool(args.strict_patchcert_carrier),
                "strength": float(args.strength),
                "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
                "candidate_count": int(len(candidates)),
                "fit_proxy": fit_proxy,
                "policy_val_proxy": val_proxy,
                "filters": {
                    "top_k": int(args.top_k),
                    "min_view_hits": int(args.min_view_hits),
                    "min_consistency": float(args.min_consistency),
                    "min_pixel_count": float(args.min_pixel_count),
                    "high_error_quantile": float(args.high_error_quantile),
                    "min_alpha": float(args.min_alpha),
                    "uniform_barycentric": bool(args.uniform_barycentric),
                },
                "candidates": candidates,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_plan_materialize_audit(output_model: Path, audit: dict[str, Any]) -> None:
    (output_model / "surface_residual_facelocal_sh1_delta_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ECSR Face-Local Surface Residual SH Delta Plan Materialization Audit",
        "",
        f"- operator: `{audit['operator']}`",
        f"- plan: `{audit['materialize_plan_in']}`",
        f"- requested plan rows: `{audit['requested_plan_rows']}`",
        f"- alpha json: `{audit.get('materialize_plan_alpha_json', '')}`",
        f"- alpha faces: `{audit.get('materialize_plan_alpha_faces', 0)}`",
        f"- accepted faces: `{audit['accepted_faces']}`",
        f"- vertices added: `{audit['vertices_added']}`",
        f"- accepted: `{audit['accepted']}`",
        f"- no-op copy: `{audit['no_op_copy']}`",
        f"- rejected plan rows: `{len(audit['rejected_plan_rows'])}`",
        f"- triangles unchanged: `{audit['topology_triangles_unchanged']}`",
        f"- degenerate faces: `{audit['topology_after']['degenerate_face_count']}`",
        f"- invalid indices: `{audit['topology_after']['invalid_index_count']}`",
    ]
    (output_model / "surface_residual_facelocal_sh1_delta_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_audit(output_model: Path, audit: dict[str, Any]) -> None:
    (output_model / "surface_residual_facelocal_sh1_delta_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    md = [
        "# ECSR Face-Local Surface Residual SH Delta Audit",
        "",
        f"- operator: `{audit['operator']}`",
        f"- sh degree: `{audit['sh_degree']}`",
        f"- basis count: `{audit['basis_count']}`",
        f"- source model: `{audit['source_model']}`",
        f"- output model: `{audit['output_model']}`",
        f"- evidence dir: `{audit['evidence_dir']}`",
        f"- selected faces: `{audit['selected_faces']}`",
        f"- accepted faces: `{audit['accepted_faces']}`",
        f"- vertices added: `{audit['vertices_added']}`",
        f"- fit samples: `{audit['fit_proxy']['samples']}`",
        f"- policy-val samples: `{audit['policy_val_proxy']['samples']}`",
        f"- policy-val relative gain: `{audit['policy_val_proxy']['relative_gain']:.6f}`",
        f"- validation shrink enabled: `{audit['validation_shrink']['enabled']}`",
        f"- validation shrink mode: `{audit['validation_shrink']['mode']}`",
        f"- validation shrink mean scale: `{audit['validation_shrink']['mean_scale']:.6f}`",
        f"- validation shrink zero-scale faces: `{audit['validation_shrink']['zero_scale_faces']}`",
        f"- face/view gain certificate enabled: `{audit['face_view_gain_certificate']['enabled']}`",
        f"- face/view gain certificate passing faces: `{audit['face_view_gain_certificate']['faces_passing']}`",
        f"- train-fold consistency enabled: `{audit['crossfold_face_gain_certificate']['enabled']}`",
        f"- train-fold consistency type: `{audit['crossfold_face_gain_certificate']['certificate_type']}`",
        f"- train-fold consistency passing faces: `{audit['crossfold_face_gain_certificate']['faces_passing']}`",
        f"- train-fold consistency min passing folds: `{audit['crossfold_face_gain_certificate']['min_passing_folds']}`",
        f"- face/view consensus enabled: `{audit['face_view_consensus']['enabled']}`",
        f"- face/view consensus passing faces: `{audit['face_view_consensus']['faces_passing']}`",
        f"- patch certificate enabled: `{audit['patch_certificate']['enabled']}`",
        f"- patch certificate accepted patches: `{audit['patch_certificate']['accepted_patches']}`",
        f"- patch certificate accepted faces after growth: `{audit['patch_certificate']['accepted_faces_after']}`",
        f"- accepted: `{audit['accepted']}`",
        f"- no-op copy: `{audit['no_op_copy']}`",
        f"- coeff abs mean: `{audit['coeff_abs_mean']:.8f}`",
        f"- coeff abs max: `{audit['coeff_abs_max']:.8f}`",
        f"- topology triangles unchanged: `{audit['topology_triangles_unchanged']}`",
        f"- degenerate faces: `{audit['topology_after']['degenerate_face_count']}`",
        f"- invalid indices: `{audit['topology_after']['invalid_index_count']}`",
        "",
        "This is a persistent checkpoint-level face-local appearance update. It does not read held-out test residuals.",
    ]
    (output_model / "surface_residual_facelocal_sh1_delta_audit.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if bool(args.force_apply) and args.candidate_plan_out is not None:
        raise ValueError(
            "--force_apply cannot be combined with --candidate_plan_out because forced rows "
            "do not carry the strict train-only certification required for replay."
        )
    if bool(args.patch_cert_neighbor_crossfold) and int(args.patch_cert_crossfold_folds) <= 1:
        raise ValueError(
            "--patch_cert_neighbor_crossfold requires --patch_cert_crossfold_folds > 1; "
            "otherwise neighbor admission would silently skip the fold certificate."
        )
    if bool(args.strict_patchcert_carrier):
        strict_errors: list[str] = []
        if int(args.patch_cert_rings) <= 0:
            strict_errors.append("--patch_cert_rings must be > 0")
        if int(args.patch_cert_crossfold_folds) <= 1:
            strict_errors.append("--patch_cert_crossfold_folds must be > 1")
        if int(args.patch_cert_crossfold_min_passing_folds) <= 0:
            strict_errors.append("--patch_cert_crossfold_min_passing_folds must be > 0")
        if not bool(args.patch_cert_neighbor_crossfold):
            strict_errors.append("--patch_cert_neighbor_crossfold must be enabled")
        if not bool(args.patch_cert_shrink):
            strict_errors.append("--patch_cert_shrink must be enabled")
        if bool(args.force_apply):
            strict_errors.append("--force_apply is incompatible with strict PatchCert carrier mode")
        if bool(args.materialize_allow_uncertified_plan):
            strict_errors.append("--materialize_allow_uncertified_plan is incompatible with strict PatchCert carrier mode")
        if args.materialize_plan_in is not None:
            try:
                validate_strict_materialize_request(args)
            except ValueError as exc:
                strict_errors.append(str(exc))
        if strict_errors:
            raise ValueError("Strict PatchCert carrier configuration failed: " + "; ".join(strict_errors))
    source_checkpoint = checkpoint_path(args.source_model, args.iteration)
    output_checkpoint = args.output_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud_state_dict.pt"
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    copy_model_metadata(args.source_model, args.output_model)

    state = torch.load(source_checkpoint, map_location="cpu")
    faces = state["_triangle_indices"].detach().cpu().long()
    vertices = state["triangles_points"].detach().cpu().float()

    if args.materialize_plan_in is not None:
        strict_materialize = not bool(args.materialize_allow_uncertified_plan)
        if strict_materialize:
            validate_strict_materialize_request(args)
        plan_rows, plan_meta = read_candidate_plan(
            args.materialize_plan_in,
            limit=int(args.materialize_plan_limit),
            face_ids=_parse_face_id_filter(args.materialize_plan_face_ids),
        )
        plan_basis_count = int(plan_meta.get("basis_count", (int(args.sh_degree) + 1) ** 2)) if isinstance(plan_meta, dict) else int((int(args.sh_degree) + 1) ** 2)
        alpha_by_face = read_plan_alphas(args.materialize_plan_alpha_json)
        strict_plan_carrier_issues: list[dict[str, Any]] = []
        if strict_materialize:
            strict_plan_carrier_issues = validate_strict_plan_carrier_integrity(plan_rows, plan_meta)
            if strict_plan_carrier_issues:
                preview = json.dumps(strict_plan_carrier_issues[:20], indent=2)
                raise ValueError(
                    "Strict certified plan materialization rejected carrier integrity issues: "
                    + preview
                )
        accepted_faces, coeff, rejected_plan_rows = plan_rows_to_facelocal_coeff(
            plan_rows,
            faces,
            fallback_basis_count=plan_basis_count,
            alpha_by_face=alpha_by_face,
            require_certified=strict_materialize,
        )
        if strict_materialize and rejected_plan_rows:
            preview = json.dumps(rejected_plan_rows[:20], indent=2)
            raise ValueError(
                "Strict certified plan materialization rejected row-level certification failures: "
                + preview
            )
        coeff = coeff * float(args.materialize_plan_scale)
        if accepted_faces or not bool(args.no_op_on_fail):
            source_vertex_ids = faces[torch.as_tensor(accepted_faces, dtype=torch.long)].long().reshape(-1) if accepted_faces else torch.empty((0,), dtype=torch.long)
            out = materialize_facelocal(
                state,
                faces,
                accepted_faces,
                source_vertex_ids,
                coeff,
                accepted_faces,
            )
        else:
            out = clone_state(state)
        torch.save(out, output_checkpoint)
        degenerate, invalid = validate_faces(out["triangles_points"], out["_triangle_indices"])
        topology_triangles_unchanged = int(out["_triangle_indices"].shape[0]) == int(faces.shape[0])
        vertices_added = int(out["triangles_points"].shape[0]) - int(vertices.shape[0])
        coeff_abs = coeff.abs() if coeff.numel() and accepted_faces else torch.empty((0,), dtype=torch.float32)
        no_op_copy = bool(not accepted_faces)
        accepted_set = set(int(fid) for fid in accepted_faces)
        accepted_plan_rows = [row for row in plan_rows if int(row.get("face_id", -1)) in accepted_set]
        policy_pass = bool(accepted_faces) and all(bool(row.get("policy_pass", True)) for row in accepted_plan_rows)
        audit = {
            "operator": "surface_residual_facelocal_sh_delta_plan_materialize",
            "test_usage": "none",
            "source_model": str(args.source_model),
            "source_checkpoint": str(source_checkpoint),
            "output_model": str(args.output_model),
            "output_checkpoint": str(output_checkpoint),
            "iteration": int(args.iteration),
            "materialize_plan_in": str(args.materialize_plan_in),
            "materialize_plan_limit": int(args.materialize_plan_limit),
            "materialize_plan_face_ids": str(args.materialize_plan_face_ids),
            "materialize_plan_scale": float(args.materialize_plan_scale),
            "materialize_allow_uncertified_plan": bool(args.materialize_allow_uncertified_plan),
            "strict_patchcert_carrier": bool(args.strict_patchcert_carrier),
            "strict_materialize": bool(strict_materialize),
            "strict_plan_carrier_issues": strict_plan_carrier_issues[:20],
            "materialize_plan_alpha_json": str(args.materialize_plan_alpha_json) if args.materialize_plan_alpha_json else "",
            "materialize_plan_alpha_faces": int(len(alpha_by_face)),
            "plan_source_operator": plan_meta.get("operator") if isinstance(plan_meta, dict) else None,
            "plan_export_policy": plan_meta.get("plan_export_policy") if isinstance(plan_meta, dict) else None,
            "plan_source_model": plan_meta.get("source_model") if isinstance(plan_meta, dict) else None,
            "requested_plan_rows": int(len(plan_rows)),
            "rejected_plan_rows": rejected_plan_rows[:20],
            "selected_faces": int(len(accepted_faces)),
            "candidate_faces": int(len(plan_rows)),
            "accepted_faces": int(len(accepted_faces)),
            "vertices_added": int(vertices_added),
            "sh_degree": int(round((plan_basis_count**0.5) - 1)) if plan_basis_count > 0 else int(args.sh_degree),
            "basis_count": int(plan_basis_count),
            "accepted": bool(accepted_faces),
            "policy_pass": bool(policy_pass),
            "force_apply": bool(args.force_apply),
            "no_op_copy": no_op_copy,
            "coeff_abs_mean": float(coeff_abs.mean().item()) if coeff_abs.numel() else 0.0,
            "coeff_abs_max": float(coeff_abs.max().item()) if coeff_abs.numel() else 0.0,
            "topology_before": {
                "triangles": int(faces.shape[0]),
                "vertices": int(vertices.shape[0]),
            },
            "topology_after": {
                "triangles": int(out["_triangle_indices"].shape[0]),
                "vertices": int(out["triangles_points"].shape[0]),
                "degenerate_face_count": int(degenerate),
                "invalid_index_count": int(invalid),
            },
            "topology_triangles_unchanged": bool(topology_triangles_unchanged),
            "accepted_preview": [
                {
                    "face_id": int(row.get("face_id", -1)),
                    "rank": int(row.get("rank", idx)),
                    "policy_val_proxy": row.get("policy_val_proxy", {}),
                    "face_stats": row.get("face_stats", {}),
                }
                for idx, row in enumerate(plan_rows[:20])
                if int(row.get("face_id", -1)) in accepted_set
            ],
        }
        write_plan_materialize_audit(args.output_model, audit)
        print(json.dumps(audit, indent=2))
        return 0 if degenerate == 0 and invalid == 0 else 1

    selected_faces, face_stats = read_selected_faces(
        args.evidence_dir / "top_residual_supports.csv",
        top_k=int(args.top_k),
        min_view_hits=int(args.min_view_hits),
        min_consistency=float(args.min_consistency),
        min_pixel_count=float(args.min_pixel_count),
    )
    selected_faces = [fid for fid in selected_faces if 0 <= int(fid) < int(faces.shape[0])]

    view_paths = sorted((args.evidence_dir / "views").glob("*.npz"))
    if not view_paths:
        view_paths = sorted((args.evidence_dir / "per_view_npz").glob("*.npz"))
    fit_paths, val_paths = split_view_paths(view_paths, int(args.policy_val_stride))
    fit_samples = collect_samples(
        fit_paths,
        selected_faces,
        face_stats,
        high_error_quantile=float(args.high_error_quantile),
        min_alpha=float(args.min_alpha),
        barycentric_tolerance=float(args.barycentric_tolerance),
        max_samples_per_face_view=int(args.max_samples_per_face_view),
        max_total_samples=int(args.max_total_samples),
        uniform_barycentric=bool(args.uniform_barycentric),
    )
    val_samples = collect_samples(
        val_paths,
        selected_faces,
        face_stats,
        high_error_quantile=float(args.high_error_quantile),
        min_alpha=float(args.min_alpha),
        barycentric_tolerance=float(args.barycentric_tolerance),
        max_samples_per_face_view=int(args.max_samples_per_face_view),
        max_total_samples=max(int(args.max_total_samples // 2), 1),
        uniform_barycentric=bool(args.uniform_barycentric),
    )

    if selected_faces and fit_samples.count:
        source_vertex_ids, selected_faces_local, fit_sample_vertex_ids = localize_samples(faces, selected_faces, fit_samples)
        _, _, val_sample_vertex_ids = localize_samples(faces, selected_faces, val_samples) if val_samples.count else (
            source_vertex_ids,
            selected_faces_local,
            torch.empty((0, 3), dtype=torch.long),
        )
    else:
        source_vertex_ids = torch.empty((0,), dtype=torch.long)
        selected_faces_local = torch.empty((0, 3), dtype=torch.long)
        fit_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)
        val_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    vertices_local = vertices[source_vertex_ids].float() if source_vertex_ids.numel() else torch.empty((0, 3), dtype=torch.float32)
    fit_ids, fit_basis, fit_target, fit_weights = samples_to_tensors(
        fit_samples,
        fit_sample_vertex_ids,
        vertices_local,
        strength=float(args.strength),
        max_abs_delta_rgb=float(args.max_abs_delta_rgb),
        device=device,
        sh_degree=int(args.sh_degree),
    )
    val_ids, val_basis, val_target, val_weights = samples_to_tensors(
        val_samples,
        val_sample_vertex_ids,
        vertices_local,
        strength=float(args.strength),
        max_abs_delta_rgb=float(args.max_abs_delta_rgb),
        device=device,
        sh_degree=int(args.sh_degree),
    )

    max_abs_dc_coeff = float(args.max_abs_delta_rgb) / float(C0)
    max_abs_sh_coeff = float(args.max_abs_sh_coeff) if float(args.max_abs_sh_coeff) > 0 else float(args.max_abs_delta_rgb) / float(C1)
    coeff, solver = solve_coeff_delta(
        selected_faces_local,
        fit_ids,
        fit_basis,
        fit_target,
        fit_weights,
        vertex_count=int(source_vertex_ids.shape[0]),
        max_abs_dc_coeff=max_abs_dc_coeff,
        max_abs_sh_coeff=max_abs_sh_coeff,
        lambda_mag=float(args.lambda_mag),
        lambda_sh1_mag=float(args.lambda_sh1_mag),
        lambda_smooth=float(args.lambda_smooth),
        steps=int(args.steps),
        lr=float(args.lr),
        device=device,
    )
    coeff_device = coeff.to(device=device)
    coeff_device, validation_shrink_summary, validation_shrink_by_face = calibrate_coeff_by_policy_val(
        coeff_device,
        val_ids,
        val_basis,
        val_samples,
        val_target,
        val_weights,
        selected_faces,
        mode=str(args.validation_shrink_mode),
        min_samples=int(args.validation_shrink_min_samples),
    )
    coeff = coeff_device.detach().cpu()
    fit_proxy = evaluate_proxy(coeff_device, fit_ids, fit_basis, fit_target, fit_weights)
    val_proxy = evaluate_proxy(coeff_device, val_ids, val_basis, val_target, val_weights)
    face_policy = evaluate_proxy_by_face(coeff_device, val_ids, val_basis, val_target, val_weights, val_samples.face_ids)
    face_view_gain_certificate, face_view_gain_certificate_summary = face_view_gain_certificate_report(
        coeff_device,
        val_ids,
        val_basis,
        val_samples,
        val_target,
        val_weights,
        min_views=int(args.min_face_gain_certificate_views),
        min_relative_gain=float(args.min_face_gain_certificate_relative_gain),
        min_view_samples=int(args.min_face_gain_certificate_view_samples),
        min_fraction=float(args.min_face_gain_certificate_fraction),
    )
    crossfold_face_gain, crossfold_face_gain_summary = summarize_crossfold_face_gain(
        coeff=coeff_device,
        faces=faces,
        selected_faces=selected_faces,
        source_vertex_ids=source_vertex_ids,
        vertices=vertices,
        view_paths=view_paths,
        face_stats=face_stats,
        args=args,
        device=device,
    )
    patch_crossfold_cache, patch_crossfold_cache_summary = build_patch_crossfold_cache(
        coeff=coeff_device,
        faces=faces,
        selected_faces=selected_faces,
        source_vertex_ids=source_vertex_ids,
        vertices=vertices,
        view_paths=view_paths,
        face_stats=face_stats,
        args=args,
        device=device,
    )
    face_view_consensus, face_view_consensus_summary = face_view_consensus_report(
        val_samples,
        val_target,
        val_weights,
        min_consensus=float(args.min_face_view_consensus),
        min_views=int(args.min_face_consensus_views),
        min_view_samples=int(args.min_face_consensus_view_samples),
        min_cosine=float(args.face_consensus_min_cosine),
    )
    fit_unique_faces = int(np.unique(fit_samples.face_ids).size) if fit_samples.count else 0
    val_unique_faces = int(np.unique(val_samples.face_ids).size) if val_samples.count else 0

    global_policy_pass = (
        fit_samples.count > 0
        and val_samples.count >= int(args.min_policy_val_samples)
        and val_unique_faces >= int(args.min_policy_val_unique_faces)
        and float(val_proxy["relative_gain"]) >= float(args.min_policy_val_relative_gain)
    )
    face_candidates: list[int] = []
    for fid in selected_faces:
        stats = face_policy.get(int(fid), {})
        if int(stats.get("samples", 0)) < int(args.min_face_policy_val_samples):
            continue
        if float(stats.get("relative_gain", -1.0)) < float(args.min_face_policy_val_relative_gain):
            continue
        gain_certificate = face_view_gain_certificate.get(int(fid), {})
        if bool(face_view_gain_certificate_summary.get("enabled", False)) and not bool(gain_certificate.get("passed", False)):
            continue
        crossfold_certificate = crossfold_face_gain.get(int(fid), {})
        if bool(crossfold_face_gain_summary.get("enabled", False)) and not bool(crossfold_certificate.get("passed", False)):
            continue
        consensus = face_view_consensus.get(int(fid), {})
        if bool(face_view_consensus_summary.get("enabled", False)) and not bool(consensus.get("passed", False)):
            continue
        face_candidates.append(int(fid))
    face_candidates.sort(
        key=lambda fid: (
            float(face_policy.get(fid, {}).get("relative_gain", 0.0)),
            float(face_stats.get(fid, {}).get("score", 0.0)),
            float(face_stats.get(fid, {}).get("pixel_count", 0.0)),
        ),
        reverse=True,
    )
    accepted_faces = face_candidates[: max(int(args.max_faces_to_apply), 0)]
    accepted_faces, patch_cert_summary, patch_cert_by_face = grow_patch_certified_faces(
        coeff=coeff_device,
        faces=faces,
        vertices=vertices,
        selected_faces=selected_faces,
        seed_faces=accepted_faces,
        face_stats=face_stats,
        face_policy=face_policy,
        val_ids=val_ids,
        val_basis=val_basis,
        val_target=val_target,
        val_weights=val_weights,
        val_samples=val_samples,
        patch_crossfold_cache=patch_crossfold_cache,
        args=args,
    )
    coeff = coeff_device.detach().cpu()
    accepted = bool((global_policy_pass and accepted_faces) or args.force_apply)
    if bool(args.force_apply) and not accepted_faces:
        accepted_faces = selected_faces[: max(int(args.max_faces_to_apply), 0)]
        accepted = bool(accepted_faces)
    no_op_copy = bool((not accepted) and args.no_op_on_fail)

    if args.candidate_plan_out is not None:
        write_candidate_plan(
            args.candidate_plan_out,
            args=args,
            selected_faces=selected_faces,
            plan_faces=accepted_faces if accepted else [],
            coeff=coeff,
            face_stats=face_stats,
            face_policy=face_policy,
            validation_shrink_by_face=validation_shrink_by_face,
            face_view_gain_certificate=face_view_gain_certificate,
            crossfold_face_gain=crossfold_face_gain,
            face_view_consensus=face_view_consensus,
            patch_cert_by_face=patch_cert_by_face,
            fit_proxy=fit_proxy,
            val_proxy=val_proxy,
        )

    if accepted or not args.no_op_on_fail:
        out = materialize_facelocal(state, faces, selected_faces, source_vertex_ids, coeff, accepted_faces)
    else:
        out = clone_state(state)
    torch.save(out, output_checkpoint)

    degenerate, invalid = validate_faces(out["triangles_points"], out["_triangle_indices"])
    topology_triangles_unchanged = int(out["_triangle_indices"].shape[0]) == int(faces.shape[0])
    vertices_added = int(out["triangles_points"].shape[0]) - int(vertices.shape[0])
    accepted_coeff_abs = torch.empty((0,), dtype=torch.float32)
    if accepted_faces and coeff.numel():
        face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
        local_ids: list[int] = []
        for fid in accepted_faces:
            row = face_to_selected[int(fid)]
            local_ids.extend([row * 3, row * 3 + 1, row * 3 + 2])
        accepted_coeff_abs = coeff[torch.as_tensor(local_ids, dtype=torch.long)].abs()
    audit = {
        "operator": "surface_residual_facelocal_sh_delta",
        "test_usage": "none",
        "source_model": str(args.source_model),
        "source_checkpoint": str(source_checkpoint),
        "output_model": str(args.output_model),
        "output_checkpoint": str(output_checkpoint),
        "iteration": int(args.iteration),
        "sh_degree": int(args.sh_degree),
        "basis_count": int((int(args.sh_degree) + 1) ** 2),
        "evidence_dir": str(args.evidence_dir),
        "selected_faces": int(len(selected_faces)),
        "face_policy_candidates": int(len(face_candidates)),
        "accepted_faces": int(len(accepted_faces)) if accepted else 0,
        "vertices_added": int(vertices_added if accepted else 0),
        "fit_views": [p.stem for p in fit_paths],
        "policy_val_views": [p.stem for p in val_paths],
        "fit_proxy": fit_proxy,
        "policy_val_proxy": val_proxy,
        "fit_unique_faces": int(fit_unique_faces),
        "policy_val_unique_faces": int(val_unique_faces),
        "solver": solver,
        "validation_shrink": validation_shrink_summary,
        "face_view_gain_certificate": face_view_gain_certificate_summary,
        "crossfold_face_gain_certificate": crossfold_face_gain_summary,
        "face_view_consensus": face_view_consensus_summary,
        "patch_crossfold_cache": patch_crossfold_cache_summary,
        "patch_certificate": patch_cert_summary,
        "filters": {
            "top_k": int(args.top_k),
            "min_view_hits": int(args.min_view_hits),
            "min_consistency": float(args.min_consistency),
            "min_pixel_count": float(args.min_pixel_count),
            "high_error_quantile": float(args.high_error_quantile),
            "min_alpha": float(args.min_alpha),
            "barycentric_tolerance": float(args.barycentric_tolerance),
            "uniform_barycentric": bool(args.uniform_barycentric),
        },
        "strength": float(args.strength),
        "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
        "max_abs_dc_coeff": float(max_abs_dc_coeff),
        "max_abs_sh_coeff": float(max_abs_sh_coeff),
        "lambda_mag": float(args.lambda_mag),
        "lambda_sh1_mag": float(args.lambda_sh1_mag),
        "lambda_smooth": float(args.lambda_smooth),
        "max_faces_to_apply": int(args.max_faces_to_apply),
        "min_policy_val_relative_gain": float(args.min_policy_val_relative_gain),
        "min_policy_val_samples": int(args.min_policy_val_samples),
        "min_policy_val_unique_faces": int(args.min_policy_val_unique_faces),
        "validation_shrink_mode": str(args.validation_shrink_mode),
        "validation_shrink_min_samples": int(args.validation_shrink_min_samples),
        "crossfold_gain_certificate_folds": int(args.crossfold_gain_certificate_folds),
        "crossfold_min_passing_folds": int(args.crossfold_min_passing_folds),
        "crossfold_min_fold_relative_gain": float(args.crossfold_min_fold_relative_gain),
        "crossfold_min_fold_samples": int(args.crossfold_min_fold_samples),
        "min_face_policy_val_relative_gain": float(args.min_face_policy_val_relative_gain),
        "min_face_policy_val_samples": int(args.min_face_policy_val_samples),
        "min_face_gain_certificate_views": int(args.min_face_gain_certificate_views),
        "min_face_gain_certificate_relative_gain": float(args.min_face_gain_certificate_relative_gain),
        "min_face_gain_certificate_view_samples": int(args.min_face_gain_certificate_view_samples),
        "min_face_gain_certificate_fraction": float(args.min_face_gain_certificate_fraction),
        "min_face_view_consensus": float(args.min_face_view_consensus),
        "min_face_consensus_views": int(args.min_face_consensus_views),
        "min_face_consensus_view_samples": int(args.min_face_consensus_view_samples),
        "face_consensus_min_cosine": float(args.face_consensus_min_cosine),
        "patch_cert_rings": int(args.patch_cert_rings),
        "patch_cert_max_faces_per_seed": int(args.patch_cert_max_faces_per_seed),
        "patch_cert_min_direction_cosine": float(args.patch_cert_min_direction_cosine),
        "patch_cert_min_neighbor_policy_val_samples": int(args.patch_cert_min_neighbor_policy_val_samples),
        "patch_cert_min_neighbor_policy_val_relative_gain": float(args.patch_cert_min_neighbor_policy_val_relative_gain),
        "patch_cert_min_policy_val_samples": int(args.patch_cert_min_policy_val_samples),
        "patch_cert_min_relative_gain": float(args.patch_cert_min_relative_gain),
        "patch_cert_neighbor_mode": str(args.patch_cert_neighbor_mode),
        "patch_cert_centroid_candidates_per_seed": int(args.patch_cert_centroid_candidates_per_seed),
        "patch_cert_crossfold_folds": int(args.patch_cert_crossfold_folds),
        "patch_cert_crossfold_min_passing_folds": int(args.patch_cert_crossfold_min_passing_folds),
        "patch_cert_crossfold_min_fold_relative_gain": float(args.patch_cert_crossfold_min_fold_relative_gain),
        "patch_cert_crossfold_min_fold_samples": int(args.patch_cert_crossfold_min_fold_samples),
        "patch_cert_neighbor_crossfold": bool(args.patch_cert_neighbor_crossfold),
        "patch_cert_shrink": bool(args.patch_cert_shrink),
        "strict_patchcert_carrier": bool(args.strict_patchcert_carrier),
        "global_policy_pass": bool(global_policy_pass),
        "policy_pass": bool(global_policy_pass),
        "accepted": bool(accepted),
        "force_apply": bool(args.force_apply),
        "no_op_copy": no_op_copy,
        "coeff_abs_mean": float(accepted_coeff_abs.mean().item()) if accepted_coeff_abs.numel() and accepted else 0.0,
        "coeff_abs_max": float(accepted_coeff_abs.max().item()) if accepted_coeff_abs.numel() and accepted else 0.0,
        "topology_before": {
            "triangles": int(faces.shape[0]),
            "vertices": int(vertices.shape[0]),
        },
        "topology_after": {
            "triangles": int(out["_triangle_indices"].shape[0]),
            "vertices": int(out["triangles_points"].shape[0]),
            "degenerate_face_count": int(degenerate),
            "invalid_index_count": int(invalid),
        },
        "topology_triangles_unchanged": bool(topology_triangles_unchanged),
        "accepted_preview": [
            {
                "face_id": int(fid),
                "face_stats": face_stats.get(int(fid), {}),
                "policy_val_proxy": face_policy.get(int(fid), {}),
                "validation_shrink": validation_shrink_by_face.get(int(fid), {}),
                "face_view_gain_certificate": face_view_gain_certificate.get(int(fid), {}),
                "crossfold_face_gain_certificate": crossfold_face_gain.get(int(fid), {}),
                "face_view_consensus": face_view_consensus.get(int(fid), {}),
                "patch_certificate": patch_cert_by_face.get(int(fid), {}),
            }
            for fid in (accepted_faces[:20] if accepted else [])
        ],
    }
    write_audit(args.output_model, audit)
    print(json.dumps(audit, indent=2))
    return 0 if degenerate == 0 and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
