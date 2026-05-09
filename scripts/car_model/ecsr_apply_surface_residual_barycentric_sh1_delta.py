#!/usr/bin/env python3
"""Fit a train-val certified barycentric residual SH1 appearance delta.

This is the view-dependent successor to the Phase-K DC-only operator.  It uses
the same rich Surface Evidence Cache, but fits residual RGB onto persistent
vertex SH coefficients:

    residual_rgb(pixel) ~= sum_j bary_j(pixel) * SH1_delta(vertex_j, camera_dir_j)

Only train-cache views are used.  A deterministic subset of train views is held
out as policy validation; by default the checkpoint is copied unchanged if the
held-out residual proxy does not improve.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from utils.sh_utils import C0, C1


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
    parser.add_argument("--min_consistency", type=float, default=0.85)
    parser.add_argument("--min_pixel_count", type=float, default=8.0)
    parser.add_argument("--max_samples_per_face_view", type=int, default=96)
    parser.add_argument("--max_total_samples", type=int, default=260000)
    parser.add_argument("--high_error_quantile", type=float, default=0.75)
    parser.add_argument("--min_alpha", type=float, default=0.05)
    parser.add_argument("--barycentric_tolerance", type=float, default=0.35)
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.10)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=0.010)
    parser.add_argument(
        "--max_abs_sh_coeff",
        type=float,
        default=0.0,
        help="Bound for each SH1 coefficient delta. 0 derives it from max_abs_delta_rgb / C1.",
    )
    parser.add_argument("--lambda_mag", type=float, default=3e-2)
    parser.add_argument("--lambda_sh1_mag", type=float, default=6e-2)
    parser.add_argument("--lambda_smooth", type=float, default=1.2e-1)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--min_policy_val_relative_gain", type=float, default=0.02)
    parser.add_argument("--min_policy_val_samples", type=int, default=512)
    parser.add_argument("--min_policy_val_unique_faces", type=int, default=16)
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
) -> PixelSamples:
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
            required = {
                "face_id",
                "residual_l1",
                "alpha",
                "residual_rgb",
                "barycentric",
                "barycentric_valid",
                "camera_center",
            }
            missing = sorted(required - set(z.files))
            if missing:
                raise RuntimeError(
                    f"{view_path} missing required SH1 evidence fields: {missing}. "
                    "Rebuild the cache with the current ecsr_build_surface_evidence_cache.py."
                )
            face_id = z["face_id"].astype(np.int64)
            residual_l1 = z["residual_l1"].astype(np.float32)
            alpha = z["alpha"].astype(np.float32)
            if alpha.ndim == 3:
                alpha = np.squeeze(alpha, axis=0)
            residual_rgb = z["residual_rgb"].astype(np.float32)
            barycentric = z["barycentric"].astype(np.float32)
            bary_valid = z["barycentric_valid"].astype(bool)
            camera_center = z["camera_center"].astype(np.float32).reshape(3)

        threshold = float(np.quantile(residual_l1.reshape(-1), float(high_error_quantile)))
        base_valid = bary_valid & (residual_l1 >= threshold) & (alpha >= float(min_alpha))
        if not np.any(base_valid):
            continue

        for fid in selected_faces:
            if remaining <= 0:
                break
            mask = base_valid & (face_id == int(fid))
            if not np.any(mask):
                continue
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
    face_ids_tensor = torch.as_tensor(selected_faces, dtype=torch.long)
    selected_faces_global = faces[face_ids_tensor].long()
    unique_vertices, inverse = torch.unique(selected_faces_global.reshape(-1), sorted=True, return_inverse=True)
    selected_faces_local = inverse.reshape(-1, 3).long()
    face_to_local = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    sample_faces_local = torch.as_tensor([face_to_local[int(fid)] for fid in samples.face_ids], dtype=torch.long)
    sample_vertex_ids = selected_faces_local[sample_faces_local]
    return unique_vertices, selected_faces_local, sample_vertex_ids


def _sh1_basis(
    vertices_local: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    bary: torch.Tensor,
    camera_centers: torch.Tensor,
) -> torch.Tensor:
    if sample_vertex_ids.numel() == 0:
        return torch.empty((0, 3, 4), dtype=torch.float32, device=vertices_local.device)
    vpos = vertices_local[sample_vertex_ids]
    dirs = vpos - camera_centers[:, None, :]
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]
    basis = torch.stack(
        [
            torch.full_like(x, float(C0)),
            -float(C1) * y,
            float(C1) * z,
            -float(C1) * x,
        ],
        dim=-1,
    )
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
    if vertex_count <= 0 or fit_sample_vertex_ids.numel() == 0:
        return torch.empty((0, 4, 3), dtype=torch.float32), {
            "initial_fit_mse": 0.0,
            "final_fit_mse": 0.0,
            "final_mag_loss": 0.0,
            "final_sh1_mag_loss": 0.0,
            "final_smooth_loss": 0.0,
        }
    fit_sample_vertex_ids = fit_sample_vertex_ids.to(device=device)
    fit_weighted_basis = fit_weighted_basis.to(device=device)
    fit_target = fit_target.to(device=device)
    fit_weights = fit_weights.to(device=device).clamp_min(1e-8)
    selected_faces_local = selected_faces_local.to(device=device)
    edges = surface_edges(selected_faces_local.detach().cpu()).to(device=device)
    bounds = torch.tensor(
        [float(max_abs_dc_coeff), float(max_abs_sh_coeff), float(max_abs_sh_coeff), float(max_abs_sh_coeff)],
        dtype=torch.float32,
        device=device,
    ).view(1, 4, 1)
    param = torch.zeros((int(vertex_count), 4, 3), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([param], lr=float(lr))

    with torch.no_grad():
        zero = torch.zeros_like(param)
        initial_fit_mse = _weighted_mse(zero, fit_sample_vertex_ids, fit_weighted_basis, fit_target, fit_weights)

    final_fit_mse = initial_fit_mse
    final_mag_loss = torch.zeros((), dtype=torch.float32, device=device)
    final_sh1_mag_loss = torch.zeros((), dtype=torch.float32, device=device)
    final_smooth_loss = torch.zeros((), dtype=torch.float32, device=device)
    for _ in range(int(steps)):
        coeff = bounds * torch.tanh(param)
        data_loss = _weighted_mse(coeff, fit_sample_vertex_ids, fit_weighted_basis, fit_target, fit_weights)
        mag_loss = (coeff[:, 0, :] ** 2).mean()
        sh1_mag_loss = (coeff[:, 1:4, :] ** 2).mean()
        if edges.numel():
            smooth_loss = ((coeff[edges[:, 0]] - coeff[edges[:, 1]]) ** 2).mean()
        else:
            smooth_loss = torch.zeros((), dtype=torch.float32, device=device)
        loss = (
            data_loss
            + float(lambda_mag) * mag_loss
            + float(lambda_sh1_mag) * sh1_mag_loss
            + float(lambda_smooth) * smooth_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_fit_mse = data_loss.detach()
        final_mag_loss = mag_loss.detach()
        final_sh1_mag_loss = sh1_mag_loss.detach()
        final_smooth_loss = smooth_loss.detach()

    with torch.no_grad():
        coeff = (bounds * torch.tanh(param)).detach().cpu()
    return coeff, {
        "initial_fit_mse": float(initial_fit_mse.detach().cpu().item()),
        "final_fit_mse": float(final_fit_mse.detach().cpu().item()),
        "final_mag_loss": float(final_mag_loss.detach().cpu().item()),
        "final_sh1_mag_loss": float(final_sh1_mag_loss.detach().cpu().item()),
        "final_smooth_loss": float(final_smooth_loss.detach().cpu().item()),
    }


def samples_to_tensors(
    samples: PixelSamples,
    sample_vertex_ids: torch.Tensor,
    vertices_local: torch.Tensor,
    *,
    strength: float,
    max_abs_delta_rgb: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = sample_vertex_ids.to(device=device)
    bary = torch.as_tensor(samples.barycentric, dtype=torch.float32, device=device)
    centers = torch.as_tensor(samples.camera_centers, dtype=torch.float32, device=device)
    vertices_local = vertices_local.to(device=device)
    weighted_basis = _sh1_basis(vertices_local, ids, bary, centers)
    target = torch.as_tensor(samples.residual_rgb, dtype=torch.float32, device=device)
    target = (target * float(strength)).clamp(-float(max_abs_delta_rgb), float(max_abs_delta_rgb))
    weights = torch.as_tensor(samples.weights, dtype=torch.float32, device=device)
    return ids, weighted_basis, target, weights


def write_audit(output_model: Path, audit: dict[str, Any]) -> None:
    (output_model / "surface_residual_barycentric_sh1_delta_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    md = [
        "# ECSR Surface Residual Barycentric SH1 Delta Audit",
        "",
        f"- operator: `{audit['operator']}`",
        f"- source model: `{audit['source_model']}`",
        f"- output model: `{audit['output_model']}`",
        f"- evidence dir: `{audit['evidence_dir']}`",
        f"- selected faces: `{audit['selected_faces']}`",
        f"- modified vertices: `{audit['vertices_modified']}`",
        f"- fit samples: `{audit['fit_proxy']['samples']}`",
        f"- fit unique faces: `{audit['fit_unique_faces']}`",
        f"- policy-val samples: `{audit['policy_val_proxy']['samples']}`",
        f"- policy-val unique faces: `{audit['policy_val_unique_faces']}`",
        f"- policy-val relative gain: `{audit['policy_val_proxy']['relative_gain']:.6f}`",
        f"- accepted: `{audit['accepted']}`",
        f"- no-op copy: `{audit['no_op_copy']}`",
        f"- coeff abs mean: `{audit['coeff_abs_mean']:.8f}`",
        f"- coeff abs max: `{audit['coeff_abs_max']:.8f}`",
        f"- SH1 coeff abs mean: `{audit['sh1_coeff_abs_mean']:.8f}`",
        f"- topology unchanged: `{audit['topology_unchanged']}`",
        f"- degenerate faces: `{audit['topology_after']['degenerate_face_count']}`",
        f"- invalid indices: `{audit['topology_after']['invalid_index_count']}`",
        "",
        "This is a persistent checkpoint-level SH appearance update. It does not read held-out test residuals.",
    ]
    (output_model / "surface_residual_barycentric_sh1_delta_audit.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    source_checkpoint = checkpoint_path(args.source_model, args.iteration)
    output_checkpoint = args.output_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud_state_dict.pt"
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    copy_model_metadata(args.source_model, args.output_model)

    state = torch.load(source_checkpoint, map_location="cpu")
    faces = state["_triangle_indices"].detach().cpu().long()
    vertices = state["triangles_points"].detach().cpu().float()
    features_dc = state["features_dc"].detach().cpu().float()
    features_rest = state["features_rest"].detach().cpu().float()
    if features_dc.ndim != 3 or features_dc.shape[1:] != (1, 3):
        raise ValueError(f"expected features_dc shape [V,1,3], got {tuple(features_dc.shape)}")
    if features_rest.ndim != 3 or features_rest.shape[1] < 3 or features_rest.shape[2] != 3:
        raise ValueError(f"expected features_rest shape [V,>=3,3], got {tuple(features_rest.shape)}")

    selected_faces, face_stats = read_selected_faces(
        args.evidence_dir / "top_residual_supports.csv",
        top_k=int(args.top_k),
        min_view_hits=int(args.min_view_hits),
        min_consistency=float(args.min_consistency),
        min_pixel_count=float(args.min_pixel_count),
    )
    selected_faces = [fid for fid in selected_faces if 0 <= int(fid) < int(faces.shape[0])]

    view_paths = sorted((args.evidence_dir / "views").glob("*.npz"))
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
    )

    if selected_faces and fit_samples.count:
        unique_vertices, selected_faces_local, fit_sample_vertex_ids = localize_samples(faces, selected_faces, fit_samples)
        _, _, val_sample_vertex_ids = localize_samples(faces, selected_faces, val_samples) if val_samples.count else (
            unique_vertices,
            selected_faces_local,
            torch.empty((0, 3), dtype=torch.long),
        )
    else:
        unique_vertices = torch.empty((0,), dtype=torch.long)
        selected_faces_local = torch.empty((0, 3), dtype=torch.long)
        fit_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)
        val_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    vertices_local = vertices[unique_vertices].float() if unique_vertices.numel() else torch.empty((0, 3), dtype=torch.float32)
    fit_ids, fit_basis, fit_target, fit_weights = samples_to_tensors(
        fit_samples,
        fit_sample_vertex_ids,
        vertices_local,
        strength=float(args.strength),
        max_abs_delta_rgb=float(args.max_abs_delta_rgb),
        device=device,
    )
    val_ids, val_basis, val_target, val_weights = samples_to_tensors(
        val_samples,
        val_sample_vertex_ids,
        vertices_local,
        strength=float(args.strength),
        max_abs_delta_rgb=float(args.max_abs_delta_rgb),
        device=device,
    )

    max_abs_dc_coeff = float(args.max_abs_delta_rgb) / float(C0)
    max_abs_sh_coeff = float(args.max_abs_sh_coeff) if float(args.max_abs_sh_coeff) > 0 else float(args.max_abs_delta_rgb) / float(C1)
    coeff, solver = solve_coeff_delta(
        selected_faces_local,
        fit_ids,
        fit_basis,
        fit_target,
        fit_weights,
        vertex_count=int(unique_vertices.shape[0]),
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
    fit_proxy = evaluate_proxy(coeff_device, fit_ids, fit_basis, fit_target, fit_weights)
    val_proxy = evaluate_proxy(coeff_device, val_ids, val_basis, val_target, val_weights)
    fit_unique_faces = int(np.unique(fit_samples.face_ids).size) if fit_samples.count else 0
    val_unique_faces = int(np.unique(val_samples.face_ids).size) if val_samples.count else 0
    policy_pass = (
        fit_samples.count > 0
        and val_samples.count >= int(args.min_policy_val_samples)
        and val_unique_faces >= int(args.min_policy_val_unique_faces)
        and float(val_proxy["relative_gain"]) >= float(args.min_policy_val_relative_gain)
    )
    accepted = bool(policy_pass or args.force_apply)
    no_op_copy = bool((not accepted) and args.no_op_on_fail)

    out = clone_state(state)
    if accepted or not args.no_op_on_fail:
        out_features_dc = features_dc.clone()
        out_features_rest = features_rest.clone()
        if coeff.numel():
            out_features_dc[unique_vertices, 0, :] = out_features_dc[unique_vertices, 0, :] + coeff[:, 0, :]
            out_features_rest[unique_vertices, 0:3, :] = out_features_rest[unique_vertices, 0:3, :] + coeff[:, 1:4, :]
        out["features_dc"] = out_features_dc.to(dtype=state["features_dc"].dtype)
        out["features_rest"] = out_features_rest.to(dtype=state["features_rest"].dtype)
    torch.save(out, output_checkpoint)

    degenerate, invalid = validate_faces(out["triangles_points"], out["_triangle_indices"])
    topology_unchanged = (
        int(out["_triangle_indices"].shape[0]) == int(faces.shape[0])
        and int(out["triangles_points"].shape[0]) == int(vertices.shape[0])
    )
    audit = {
        "operator": "surface_residual_barycentric_sh1_delta",
        "test_usage": "none",
        "source_model": str(args.source_model),
        "source_checkpoint": str(source_checkpoint),
        "output_model": str(args.output_model),
        "output_checkpoint": str(output_checkpoint),
        "iteration": int(args.iteration),
        "evidence_dir": str(args.evidence_dir),
        "selected_faces": int(len(selected_faces)),
        "vertices_modified": int(unique_vertices.shape[0]) if accepted or not args.no_op_on_fail else 0,
        "fit_views": [p.stem for p in fit_paths],
        "policy_val_views": [p.stem for p in val_paths],
        "fit_proxy": fit_proxy,
        "policy_val_proxy": val_proxy,
        "fit_unique_faces": int(fit_unique_faces),
        "policy_val_unique_faces": int(val_unique_faces),
        "solver": solver,
        "filters": {
            "top_k": int(args.top_k),
            "min_view_hits": int(args.min_view_hits),
            "min_consistency": float(args.min_consistency),
            "min_pixel_count": float(args.min_pixel_count),
            "high_error_quantile": float(args.high_error_quantile),
            "min_alpha": float(args.min_alpha),
            "barycentric_tolerance": float(args.barycentric_tolerance),
        },
        "strength": float(args.strength),
        "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
        "max_abs_dc_coeff": float(max_abs_dc_coeff),
        "max_abs_sh_coeff": float(max_abs_sh_coeff),
        "lambda_mag": float(args.lambda_mag),
        "lambda_sh1_mag": float(args.lambda_sh1_mag),
        "lambda_smooth": float(args.lambda_smooth),
        "min_policy_val_relative_gain": float(args.min_policy_val_relative_gain),
        "min_policy_val_samples": int(args.min_policy_val_samples),
        "min_policy_val_unique_faces": int(args.min_policy_val_unique_faces),
        "policy_pass": bool(policy_pass),
        "accepted": accepted,
        "force_apply": bool(args.force_apply),
        "no_op_copy": no_op_copy,
        "coeff_abs_mean": float(coeff.abs().mean().item()) if coeff.numel() and accepted else 0.0,
        "coeff_abs_max": float(coeff.abs().max().item()) if coeff.numel() and accepted else 0.0,
        "dc_coeff_abs_mean": float(coeff[:, 0, :].abs().mean().item()) if coeff.numel() and accepted else 0.0,
        "sh1_coeff_abs_mean": float(coeff[:, 1:4, :].abs().mean().item()) if coeff.numel() and accepted else 0.0,
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
        "topology_unchanged": bool(topology_unchanged),
    }
    write_audit(args.output_model, audit)
    print(json.dumps(audit, indent=2))
    return 0 if degenerate == 0 and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
