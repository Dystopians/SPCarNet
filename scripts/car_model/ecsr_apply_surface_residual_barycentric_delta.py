#!/usr/bin/env python3
"""Fit a policy-val certified barycentric residual SH-DC delta.

This Phase-D operator consumes the rich Surface Evidence Cache. Unlike the
older per-face mean residual writers, it fits per-pixel residual RGB targets
through reconstructed barycentric coordinates:

    residual_rgb(pixel) ~= sum_j bary_j(pixel) * delta_rgb(vertex_j)

Only train-cache views are used. A deterministic subset of train views is held
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
from utils.sh_utils import C0


@dataclass
class PixelSamples:
    face_ids: np.ndarray
    barycentric: np.ndarray
    residual_rgb: np.ndarray
    weights: np.ndarray
    view_names: list[str]

    @property
    def count(self) -> int:
        return int(self.face_ids.shape[0])


def empty_pixel_samples() -> PixelSamples:
    empty = np.empty((0,), dtype=np.int64)
    return PixelSamples(
        face_ids=empty,
        barycentric=np.empty((0, 3), dtype=np.float32),
        residual_rgb=np.empty((0, 3), dtype=np.float32),
        weights=np.empty((0,), dtype=np.float32),
        view_names=[],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model", type=Path, required=True)
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--top_k", type=int, default=256)
    parser.add_argument("--min_view_hits", type=int, default=2)
    parser.add_argument("--min_consistency", type=float, default=0.90)
    parser.add_argument("--min_pixel_count", type=float, default=8.0)
    parser.add_argument("--max_samples_per_face_view", type=int, default=128)
    parser.add_argument("--max_total_samples", type=int, default=180000)
    parser.add_argument("--high_error_quantile", type=float, default=0.80)
    parser.add_argument("--min_alpha", type=float, default=0.05)
    parser.add_argument("--barycentric_tolerance", type=float, default=0.35)
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.18)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=0.018)
    parser.add_argument("--lambda_mag", type=float, default=2e-2)
    parser.add_argument("--lambda_smooth", type=float, default=8e-2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--min_policy_val_relative_gain", type=float, default=0.02)
    parser.add_argument("--min_policy_val_samples", type=int, default=512)
    parser.add_argument("--min_policy_val_unique_faces", type=int, default=16)
    parser.add_argument(
        "--policy_val_filter_faces",
        action="store_true",
        help=(
            "Run a train-holdout face-level gain pass, then refit using only "
            "faces whose individual policy-val proxy improves. This prevents "
            "a good aggregate proxy from materializing locally harmful patches."
        ),
    )
    parser.add_argument("--policy_val_face_min_samples", type=int, default=8)
    parser.add_argument("--policy_val_face_min_relative_gain", type=float, default=0.0)
    parser.add_argument(
        "--policy_val_face_max_keep",
        type=int,
        default=0,
        help="Optional cap after face-level filtering; 0 keeps every passing face.",
    )
    parser.add_argument(
        "--candidate_cluster_json",
        type=Path,
        default=Path(""),
        help=(
            "Optional Phase-B view-support graph JSON. When set, selected faces "
            "come from fixed train-evidence support clusters instead of isolated "
            "top residual faces."
        ),
    )
    parser.add_argument(
        "--candidate_cluster_csv",
        type=Path,
        default=Path(""),
        help="Optional Phase-B candidate_clusters.csv fallback for cluster-guided support selection.",
    )
    parser.add_argument(
        "--cluster_operator_types",
        default="certificate_cluster_contraction_candidate,surface_attached_attribute_recovery_candidate",
        help="Comma-separated operator_type allowlist for candidate cluster rows.",
    )
    parser.add_argument("--max_clusters", type=int, default=0)
    parser.add_argument("--cluster_min_redundancy_score", type=float, default=-1.0e30)
    parser.add_argument(
        "--cluster_expand_with_top_residual_faces",
        action="store_true",
        help=(
            "Use Phase-B clusters as anchors, then append deterministic top "
            "residual faces until --cluster_expand_target_faces is reached. "
            "This keeps the policy fixed while avoiding under-supported sparse clusters."
        ),
    )
    parser.add_argument(
        "--cluster_expand_target_faces",
        type=int,
        default=0,
        help="Target selected face count for cluster expansion. 0 falls back to --top_k.",
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


def _parse_operator_types(text: str) -> set[str]:
    return {token.strip() for token in str(text or "").split(",") if token.strip()}


def _cluster_faces_from_row(row: dict[str, Any]) -> list[int]:
    faces = row.get("faces", [])
    if isinstance(faces, str):
        return [int(token) for token in faces.replace(",", " ").split() if token.strip()]
    if isinstance(faces, list):
        return [int(face) for face in faces]
    return []


def read_candidate_cluster_faces(
    *,
    json_path: Path,
    csv_path: Path,
    operator_types: set[str],
    max_clusters: int,
    min_redundancy_score: float,
) -> tuple[list[int], list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    source = ""
    if str(json_path) and json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows = [row for row in payload.get("candidate_clusters", []) if isinstance(row, dict)]
        source = str(json_path)
    elif str(csv_path) and csv_path.is_file():
        source = str(csv_path)
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "candidate_id": row.get("candidate_id", ""),
                        "operator_type": row.get("operator_type", ""),
                        "faces": row.get("faces", ""),
                        "num_faces": int(float(row.get("num_faces", 0) or 0)),
                        "mean_redundancy_score": float(row.get("mean_redundancy_score", 0.0) or 0.0),
                        "risk_flags": row.get("risk_flags", ""),
                    }
                )
    if not rows:
        return [], [], source

    filtered: list[dict[str, Any]] = []
    for row in rows:
        operator = str(row.get("operator_type", ""))
        if operator_types and operator not in operator_types:
            continue
        score = float(row.get("mean_redundancy_score", 0.0) or 0.0)
        if score < float(min_redundancy_score):
            continue
        faces = _cluster_faces_from_row(row)
        if not faces:
            continue
        record = dict(row)
        record["faces"] = faces
        record["mean_redundancy_score"] = score
        filtered.append(record)
    filtered.sort(
        key=lambda row: (
            float(row.get("mean_redundancy_score", 0.0) or 0.0),
            int(row.get("num_faces", len(row.get("faces", []))) or len(row.get("faces", []))),
        ),
        reverse=True,
    )
    if int(max_clusters) > 0:
        filtered = filtered[: int(max_clusters)]
    ordered_faces: list[int] = []
    seen: set[int] = set()
    for row in filtered:
        for face_id in _cluster_faces_from_row(row):
            if face_id in seen:
                continue
            seen.add(face_id)
            ordered_faces.append(int(face_id))
    return ordered_faces, filtered, source


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
    selected = set(int(x) for x in selected_faces)
    face_chunks: list[np.ndarray] = []
    bary_chunks: list[np.ndarray] = []
    residual_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    sample_view_names: list[str] = []
    remaining = int(max_total_samples)
    tol = float(barycentric_tolerance)

    for view_path in view_paths:
        if remaining <= 0:
            break
        with np.load(view_path) as z:
            required = {"face_id", "residual_l1", "alpha", "residual_rgb", "barycentric", "barycentric_valid"}
            missing = sorted(required - set(z.files))
            if missing:
                raise RuntimeError(f"{view_path} missing required rich evidence fields: {missing}")
            face_id = z["face_id"].astype(np.int64)
            residual_l1 = z["residual_l1"].astype(np.float32)
            alpha = z["alpha"].astype(np.float32)
            if alpha.ndim == 3:
                alpha = np.squeeze(alpha, axis=0)
            residual_rgb = z["residual_rgb"].astype(np.float32)
            barycentric = z["barycentric"].astype(np.float32)
            bary_valid = z["barycentric_valid"].astype(bool)

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
            sample_view_names.extend([view_path.stem] * n)
            remaining -= n

    if not face_chunks:
        return empty_pixel_samples()

    return PixelSamples(
        face_ids=np.concatenate(face_chunks),
        barycentric=np.concatenate(bary_chunks),
        residual_rgb=np.concatenate(residual_chunks),
        weights=np.concatenate(weight_chunks),
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


def _weighted_mse(
    delta: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    bary: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if sample_vertex_ids.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=delta.device)
    pred = (delta[sample_vertex_ids] * bary[:, :, None]).sum(dim=1)
    return (((pred - target) ** 2) * weights[:, None]).sum() / (weights.sum().clamp_min(1e-8) * 3.0)


def evaluate_proxy(
    delta: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    bary: torch.Tensor,
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
    zero = torch.zeros_like(delta)
    with torch.no_grad():
        mse_before = _weighted_mse(zero, sample_vertex_ids, bary, target, weights)
        mse_after = _weighted_mse(delta, sample_vertex_ids, bary, target, weights)
        pred_after = (delta[sample_vertex_ids] * bary[:, :, None]).sum(dim=1)
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


def face_gain_report(
    delta: torch.Tensor,
    samples: PixelSamples,
    sample_vertex_ids: torch.Tensor,
    bary: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    min_samples: int,
    min_relative_gain: float,
    max_keep: int,
) -> tuple[list[int], dict[str, Any]]:
    if sample_vertex_ids.numel() == 0 or samples.count == 0:
        return [], {
            "enabled": True,
            "input_faces": 0,
            "kept_faces": 0,
            "rejected_faces": 0,
            "min_samples": int(min_samples),
            "min_relative_gain": float(min_relative_gain),
            "max_keep": int(max_keep),
            "face_preview": [],
        }

    face_ids = samples.face_ids.astype(np.int64)
    unique_faces = np.unique(face_ids)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for face_id in unique_faces:
            idx_np = np.nonzero(face_ids == int(face_id))[0]
            if idx_np.size <= 0:
                continue
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=delta.device)
            proxy = evaluate_proxy(
                delta,
                sample_vertex_ids.index_select(0, idx),
                bary.index_select(0, idx),
                target.index_select(0, idx),
                weights.index_select(0, idx),
            )
            keep = (
                int(proxy["samples"]) >= int(min_samples)
                and float(proxy["relative_gain"]) >= float(min_relative_gain)
            )
            rows.append(
                {
                    "face_id": int(face_id),
                    "samples": int(proxy["samples"]),
                    "relative_gain": float(proxy["relative_gain"]),
                    "mse_before": float(proxy["mse_before"]),
                    "mse_after": float(proxy["mse_after"]),
                    "keep": bool(keep),
                }
            )

    rows.sort(key=lambda row: (bool(row["keep"]), float(row["relative_gain"]), int(row["samples"])), reverse=True)
    kept = [int(row["face_id"]) for row in rows if bool(row["keep"])]
    if int(max_keep) > 0:
        kept = kept[: int(max_keep)]
        kept_set = set(kept)
        for row in rows:
            row["keep"] = int(row["face_id"]) in kept_set
    report = {
        "enabled": True,
        "input_faces": int(unique_faces.size),
        "kept_faces": int(len(kept)),
        "rejected_faces": int(unique_faces.size - len(kept)),
        "min_samples": int(min_samples),
        "min_relative_gain": float(min_relative_gain),
        "max_keep": int(max_keep),
        "face_preview": rows[:100],
    }
    return kept, report


def solve_delta(
    selected_faces_local: torch.Tensor,
    fit_sample_vertex_ids: torch.Tensor,
    fit_bary: torch.Tensor,
    fit_target: torch.Tensor,
    fit_weights: torch.Tensor,
    *,
    vertex_count: int,
    max_abs_delta_rgb: float,
    lambda_mag: float,
    lambda_smooth: float,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    if vertex_count <= 0 or fit_sample_vertex_ids.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32), {
            "initial_fit_mse": 0.0,
            "final_fit_mse": 0.0,
            "final_mag_loss": 0.0,
            "final_smooth_loss": 0.0,
        }
    fit_sample_vertex_ids = fit_sample_vertex_ids.to(device=device)
    fit_bary = fit_bary.to(device=device)
    fit_target = fit_target.to(device=device)
    fit_weights = fit_weights.to(device=device).clamp_min(1e-8)
    selected_faces_local = selected_faces_local.to(device=device)
    edges = surface_edges(selected_faces_local.detach().cpu()).to(device=device)
    param = torch.zeros((int(vertex_count), 3), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([param], lr=float(lr))

    with torch.no_grad():
        zero = torch.zeros_like(param)
        initial_fit_mse = _weighted_mse(zero, fit_sample_vertex_ids, fit_bary, fit_target, fit_weights)

    final_fit_mse = initial_fit_mse
    final_mag_loss = torch.zeros((), dtype=torch.float32, device=device)
    final_smooth_loss = torch.zeros((), dtype=torch.float32, device=device)
    for _ in range(int(steps)):
        delta = float(max_abs_delta_rgb) * torch.tanh(param)
        data_loss = _weighted_mse(delta, fit_sample_vertex_ids, fit_bary, fit_target, fit_weights)
        mag_loss = (delta**2).mean()
        if edges.numel():
            smooth_loss = ((delta[edges[:, 0]] - delta[edges[:, 1]]) ** 2).mean()
        else:
            smooth_loss = torch.zeros((), dtype=torch.float32, device=device)
        loss = data_loss + float(lambda_mag) * mag_loss + float(lambda_smooth) * smooth_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_fit_mse = data_loss.detach()
        final_mag_loss = mag_loss.detach()
        final_smooth_loss = smooth_loss.detach()

    with torch.no_grad():
        delta = (float(max_abs_delta_rgb) * torch.tanh(param)).detach().cpu()
    return delta, {
        "initial_fit_mse": float(initial_fit_mse.detach().cpu().item()),
        "final_fit_mse": float(final_fit_mse.detach().cpu().item()),
        "final_mag_loss": float(final_mag_loss.detach().cpu().item()),
        "final_smooth_loss": float(final_smooth_loss.detach().cpu().item()),
    }


def samples_to_tensors(
    samples: PixelSamples,
    sample_vertex_ids: torch.Tensor,
    *,
    strength: float,
    max_abs_delta_rgb: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bary = torch.as_tensor(samples.barycentric, dtype=torch.float32, device=device)
    target = torch.as_tensor(samples.residual_rgb, dtype=torch.float32, device=device)
    target = (target * float(strength)).clamp(-float(max_abs_delta_rgb), float(max_abs_delta_rgb))
    weights = torch.as_tensor(samples.weights, dtype=torch.float32, device=device)
    return sample_vertex_ids.to(device=device), bary, target, weights


def write_audit(output_model: Path, audit: dict[str, Any]) -> None:
    (output_model / "surface_residual_barycentric_delta_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    md = [
        "# ECSR Surface Residual Barycentric Delta Audit",
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
        f"- policy-val face filter: `{audit['policy_val_face_filter']['enabled']}`",
        f"- policy-val kept faces: `{audit['policy_val_face_filter']['kept_faces']}`",
        f"- cluster policy: `{audit['cluster_policy']['enabled']}`",
        f"- clusters used: `{audit['cluster_policy']['clusters_used']}`",
        f"- min policy-val samples: `{audit['min_policy_val_samples']}`",
        f"- min policy-val unique faces: `{audit['min_policy_val_unique_faces']}`",
        f"- accepted: `{audit['accepted']}`",
        f"- no-op copy: `{audit['no_op_copy']}`",
        f"- delta RGB abs mean: `{audit['delta_rgb_abs_mean']:.8f}`",
        f"- delta RGB abs max: `{audit['delta_rgb_abs_max']:.8f}`",
        f"- topology unchanged: `{audit['topology_unchanged']}`",
        f"- degenerate faces: `{audit['topology_after']['degenerate_face_count']}`",
        f"- invalid indices: `{audit['topology_after']['invalid_index_count']}`",
        "",
        "This is a checkpoint-level representation update. It does not edit rendered images.",
    ]
    (output_model / "surface_residual_barycentric_delta_audit.md").write_text(
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
    selected_faces, face_stats = read_selected_faces(
        args.evidence_dir / "top_residual_supports.csv",
        top_k=int(args.top_k),
        min_view_hits=int(args.min_view_hits),
        min_consistency=float(args.min_consistency),
        min_pixel_count=float(args.min_pixel_count),
    )
    top_residual_faces = list(selected_faces)
    cluster_faces, cluster_records, cluster_source = read_candidate_cluster_faces(
        json_path=args.candidate_cluster_json,
        csv_path=args.candidate_cluster_csv,
        operator_types=_parse_operator_types(args.cluster_operator_types),
        max_clusters=int(args.max_clusters),
        min_redundancy_score=float(args.cluster_min_redundancy_score),
    )
    if cluster_faces:
        if bool(args.cluster_expand_with_top_residual_faces):
            target_faces = int(args.cluster_expand_target_faces)
            if target_faces <= 0:
                target_faces = int(args.top_k)
            expanded: list[int] = []
            seen: set[int] = set()
            for face_id in list(cluster_faces) + top_residual_faces:
                if face_id in seen:
                    continue
                seen.add(face_id)
                expanded.append(int(face_id))
                if len(expanded) >= target_faces:
                    break
            selected_faces = expanded
        else:
            selected_faces = cluster_faces
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
    fit_ids, fit_bary, fit_target, fit_weights = samples_to_tensors(
        fit_samples,
        fit_sample_vertex_ids,
        strength=float(args.strength),
        max_abs_delta_rgb=float(args.max_abs_delta_rgb),
        device=device,
    )
    val_ids, val_bary, val_target, val_weights = samples_to_tensors(
        val_samples,
        val_sample_vertex_ids,
        strength=float(args.strength),
        max_abs_delta_rgb=float(args.max_abs_delta_rgb),
        device=device,
    )

    delta_rgb, solver = solve_delta(
        selected_faces_local,
        fit_ids,
        fit_bary,
        fit_target,
        fit_weights,
        vertex_count=int(unique_vertices.shape[0]),
        max_abs_delta_rgb=float(args.max_abs_delta_rgb),
        lambda_mag=float(args.lambda_mag),
        lambda_smooth=float(args.lambda_smooth),
        steps=int(args.steps),
        lr=float(args.lr),
        device=device,
    )
    face_filter_report: dict[str, Any] = {
        "enabled": bool(args.policy_val_filter_faces),
        "input_faces": int(len(selected_faces)),
        "kept_faces": int(len(selected_faces)),
        "rejected_faces": 0,
        "min_samples": int(args.policy_val_face_min_samples),
        "min_relative_gain": float(args.policy_val_face_min_relative_gain),
        "max_keep": int(args.policy_val_face_max_keep),
        "face_preview": [],
    }
    if bool(args.policy_val_filter_faces):
        delta_device_initial = delta_rgb.to(device=device)
        kept_faces, face_filter_report = face_gain_report(
            delta_device_initial,
            val_samples,
            val_ids,
            val_bary,
            val_target,
            val_weights,
            min_samples=int(args.policy_val_face_min_samples),
            min_relative_gain=float(args.policy_val_face_min_relative_gain),
            max_keep=int(args.policy_val_face_max_keep),
        )
        kept = set(int(face_id) for face_id in kept_faces)
        selected_faces = [fid for fid in selected_faces if int(fid) in kept]
        if selected_faces:
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
            if fit_samples.count:
                unique_vertices, selected_faces_local, fit_sample_vertex_ids = localize_samples(
                    faces,
                    selected_faces,
                    fit_samples,
                )
                _, _, val_sample_vertex_ids = (
                    localize_samples(faces, selected_faces, val_samples)
                    if val_samples.count
                    else (
                        unique_vertices,
                        selected_faces_local,
                        torch.empty((0, 3), dtype=torch.long),
                    )
                )
            else:
                selected_faces = []

        if not selected_faces:
            fit_samples = empty_pixel_samples()
            val_samples = empty_pixel_samples()
            unique_vertices = torch.empty((0,), dtype=torch.long)
            selected_faces_local = torch.empty((0, 3), dtype=torch.long)
            fit_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)
            val_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)

        fit_ids, fit_bary, fit_target, fit_weights = samples_to_tensors(
            fit_samples,
            fit_sample_vertex_ids,
            strength=float(args.strength),
            max_abs_delta_rgb=float(args.max_abs_delta_rgb),
            device=device,
        )
        val_ids, val_bary, val_target, val_weights = samples_to_tensors(
            val_samples,
            val_sample_vertex_ids,
            strength=float(args.strength),
            max_abs_delta_rgb=float(args.max_abs_delta_rgb),
            device=device,
        )
        delta_rgb, solver = solve_delta(
            selected_faces_local,
            fit_ids,
            fit_bary,
            fit_target,
            fit_weights,
            vertex_count=int(unique_vertices.shape[0]),
            max_abs_delta_rgb=float(args.max_abs_delta_rgb),
            lambda_mag=float(args.lambda_mag),
            lambda_smooth=float(args.lambda_smooth),
            steps=int(args.steps),
            lr=float(args.lr),
            device=device,
        )
    delta_device = delta_rgb.to(device=device)
    fit_proxy = evaluate_proxy(delta_device, fit_ids, fit_bary, fit_target, fit_weights)
    val_proxy = evaluate_proxy(delta_device, val_ids, val_bary, val_target, val_weights)
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
        if delta_rgb.numel():
            out_features_dc[unique_vertices] = out_features_dc[unique_vertices] + delta_rgb[:, None, :] / float(C0)
        out["features_dc"] = out_features_dc.to(dtype=state["features_dc"].dtype)
    torch.save(out, output_checkpoint)

    degenerate, invalid = validate_faces(out["triangles_points"], out["_triangle_indices"])
    topology_unchanged = (
        int(out["_triangle_indices"].shape[0]) == int(faces.shape[0])
        and int(out["triangles_points"].shape[0]) == int(vertices.shape[0])
    )
    audit = {
        "operator": "surface_residual_barycentric_delta",
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
        "policy_val_face_filter": face_filter_report,
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
        "cluster_policy": {
            "enabled": bool(cluster_faces),
            "source": cluster_source,
            "operator_types": sorted(_parse_operator_types(args.cluster_operator_types)),
            "max_clusters": int(args.max_clusters),
            "cluster_min_redundancy_score": float(args.cluster_min_redundancy_score),
            "expand_with_top_residual_faces": bool(args.cluster_expand_with_top_residual_faces),
            "cluster_expand_target_faces": int(args.cluster_expand_target_faces),
            "clusters_used": int(len(cluster_records)),
            "faces_from_clusters": int(len(cluster_faces)),
            "selected_faces_after_expansion": int(len(selected_faces)),
            "cluster_preview": cluster_records[:20],
        },
        "strength": float(args.strength),
        "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
        "lambda_mag": float(args.lambda_mag),
        "lambda_smooth": float(args.lambda_smooth),
        "min_policy_val_relative_gain": float(args.min_policy_val_relative_gain),
        "min_policy_val_samples": int(args.min_policy_val_samples),
        "min_policy_val_unique_faces": int(args.min_policy_val_unique_faces),
        "policy_pass": bool(policy_pass),
        "accepted": accepted,
        "force_apply": bool(args.force_apply),
        "no_op_copy": no_op_copy,
        "delta_rgb_abs_mean": float(delta_rgb.abs().mean().item()) if delta_rgb.numel() and accepted else 0.0,
        "delta_rgb_abs_max": float(delta_rgb.abs().max().item()) if delta_rgb.numel() and accepted else 0.0,
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
