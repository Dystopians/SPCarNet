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
from utils.sh_utils import C0


FACE_KEYS = ("importance_score", "image_size", "pixel_count")


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
    parser.add_argument("--lambda_ridge", type=float, default=2e-2)
    parser.add_argument("--min_fit_samples", type=int, default=24)
    parser.add_argument("--min_val_samples", type=int, default=12)
    parser.add_argument("--min_policy_val_relative_gain", type=float, default=0.05)
    parser.add_argument("--max_faces_to_apply", type=int, default=512)
    parser.add_argument("--force_apply", action="store_true")
    parser.add_argument("--no_op_on_fail", action="store_true", default=True)
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


def _basis_midpoint(bary: np.ndarray) -> np.ndarray:
    u = bary[:, 0]
    v = bary[:, 1]
    w = bary[:, 2]
    basis = np.stack([4.0 * u * v, 4.0 * v * w, 4.0 * w * u], axis=1)
    return np.clip(basis, 0.0, 1.0).astype(np.float32)


def collect_samples(
    view_paths: list[Path],
    selected_faces: list[int],
    face_stats: dict[int, dict[str, float]],
    *,
    high_error_quantile: float,
    min_alpha: float,
    max_samples_per_face_view: int,
) -> dict[int, dict[str, list[np.ndarray]]]:
    selected = set(int(x) for x in selected_faces)
    samples: dict[int, dict[str, list[np.ndarray]]] = {
        int(fid): {"basis": [], "target": [], "weight": []}
        for fid in selected_faces
    }
    for view_path in view_paths:
        with np.load(view_path) as z:
            required = {"face_id", "residual_l1", "alpha", "residual_rgb", "barycentric", "barycentric_valid"}
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

        threshold = float(np.quantile(residual_l1.reshape(-1), float(high_error_quantile)))
        valid = bary_valid & (residual_l1 >= threshold) & (alpha >= float(min_alpha))
        if not np.any(valid):
            continue
        flat_faces = face_id[valid].reshape(-1)
        if flat_faces.size == 0:
            continue
        present = sorted(set(int(x) for x in np.unique(flat_faces)) & selected)
        if not present:
            continue
        ys_all, xs_all = np.nonzero(valid)
        for fid in present:
            local = flat_faces == int(fid)
            idx = np.nonzero(local)[0]
            if idx.size == 0:
                continue
            cap = min(int(max_samples_per_face_view), int(idx.size))
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
            basis = _basis_midpoint(bary)
            target = residual_rgb[:, ys, xs].T.astype(np.float32)
            l1 = residual_l1[ys, xs].astype(np.float32)
            consistency = float(face_stats.get(int(fid), {}).get("consistency", 1.0))
            weight = np.maximum(l1, 1e-4).astype(np.float32) * max(consistency, 1e-3)
            samples[int(fid)]["basis"].append(basis)
            samples[int(fid)]["target"].append(target)
            samples[int(fid)]["weight"].append(weight)
    return samples


def _pack_face_samples(face_samples: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not face_samples["basis"]:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.concatenate(face_samples["basis"], axis=0),
        np.concatenate(face_samples["target"], axis=0),
        np.concatenate(face_samples["weight"], axis=0),
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
    lambda_ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    y_fit = np.clip(target_fit * float(strength), -float(max_abs_delta_rgb), float(max_abs_delta_rgb))
    y_val = np.clip(target_val * float(strength), -float(max_abs_delta_rgb), float(max_abs_delta_rgb))
    wf = weight_fit.reshape(-1, 1).astype(np.float32)
    xtw = basis_fit.T @ (basis_fit * wf)
    rhs = basis_fit.T @ (y_fit * wf)
    xtw = xtw + np.eye(3, dtype=np.float32) * float(lambda_ridge)
    try:
        delta = np.linalg.solve(xtw, rhs).astype(np.float32)
    except np.linalg.LinAlgError:
        delta = np.zeros((3, 3), dtype=np.float32)
    delta = np.clip(delta, -float(max_abs_delta_rgb), float(max_abs_delta_rgb))

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


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
        else:
            out[key] = value
    return out


def materialize(
    state: dict[str, Any],
    accepted: list[dict[str, Any]],
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
        delta = torch.as_tensor(row["delta_rgb"], dtype=torch.float32)
        mids: list[int] = []
        for edge_idx, (u, v) in enumerate(edge_pairs):
            mids.append(next_vertex)
            next_vertex += 1
            new_vertices.append((vertices[u] + vertices[v]) * 0.5)
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


def main() -> int:
    args = parse_args()
    source_checkpoint = checkpoint_path(args.source_model, args.iteration)
    output_checkpoint = args.output_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud_state_dict.pt"
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    copy_model_metadata(args.source_model, args.output_model)

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
    fit_views, val_views = split_view_paths(view_paths, int(args.policy_val_stride))
    fit_samples = collect_samples(
        fit_views,
        selected_faces,
        face_stats,
        high_error_quantile=float(args.high_error_quantile),
        min_alpha=float(args.min_alpha),
        max_samples_per_face_view=int(args.max_samples_per_face_view),
    )
    val_samples = collect_samples(
        val_views,
        selected_faces,
        face_stats,
        high_error_quantile=float(args.high_error_quantile),
        min_alpha=float(args.min_alpha),
        max_samples_per_face_view=int(args.max_samples_per_face_view),
    )

    candidates: list[dict[str, Any]] = []
    for fid in selected_faces:
        xf, yf, wf = _pack_face_samples(fit_samples[int(fid)])
        xv, yv, wv = _pack_face_samples(val_samples[int(fid)])
        if xf.shape[0] < int(args.min_fit_samples) or xv.shape[0] < int(args.min_val_samples):
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
            lambda_ridge=float(args.lambda_ridge),
        )
        if bool(args.force_apply) or float(stats["relative_gain"]) >= float(args.min_policy_val_relative_gain):
            candidates.append(
                {
                    "face_id": int(fid),
                    "face_stats": face_stats.get(int(fid), {}),
                    "delta_rgb": delta.tolist(),
                    "proxy": stats,
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
    accepted = candidates[: int(args.max_faces_to_apply)]

    state = torch.load(source_checkpoint, map_location="cpu")
    if not accepted and bool(args.no_op_on_fail):
        out = clone_state(state)
        accepted_flag = False
        no_op_copy = True
    else:
        out = materialize(state, accepted)
        accepted_flag = bool(accepted)
        no_op_copy = False
    torch.save(out, output_checkpoint)
    degenerate, invalid = validate_faces(out["triangles_points"], out["_triangle_indices"])

    before_faces = int(state["_triangle_indices"].shape[0])
    before_vertices = int(state["triangles_points"].shape[0])
    after_faces = int(out["_triangle_indices"].shape[0])
    after_vertices = int(out["triangles_points"].shape[0])
    audit = {
        "operator": "surface_residual_subdivision_delta",
        "test_usage": "none",
        "source_model": str(args.source_model),
        "source_checkpoint": str(source_checkpoint),
        "output_model": str(args.output_model),
        "output_checkpoint": str(output_checkpoint),
        "iteration": int(args.iteration),
        "evidence_dir": str(args.evidence_dir),
        "view_counts": {"fit": len(fit_views), "policy_val": len(val_views)},
        "selected_faces": int(len(selected_faces)),
        "candidate_faces": int(len(candidates)),
        "accepted_faces": int(len(accepted)),
        "accepted": accepted_flag,
        "no_op_copy": no_op_copy,
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
        "lambda_ridge": float(args.lambda_ridge),
        "min_policy_val_relative_gain": float(args.min_policy_val_relative_gain),
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
        f"- accepted faces: `{audit['accepted_faces']}`",
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
