#!/usr/bin/env python3
"""Solve a bounded surface residual SH-DC delta from train-only evidence.

The previous Phase-D residual writer copied each high-error face residual
directly onto its incident vertices. This variant treats the evidence as a
regularized inverse problem over the selected surface patch: face residuals are
fit by the mean of their three vertex deltas, while ridge and edge smoothness
penalties keep the update local and conservative.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model", type=Path, required=True)
    parser.add_argument("--evidence_csv", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--top_k", type=int, default=768)
    parser.add_argument("--min_view_hits", type=int, default=1)
    parser.add_argument("--min_consistency", type=float, default=0.8)
    parser.add_argument("--min_pixel_count", type=float, default=8.0)
    parser.add_argument("--strength", type=float, default=0.12)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=0.010)
    parser.add_argument("--lambda_mag", type=float, default=2e-2)
    parser.add_argument("--lambda_smooth", type=float, default=1e-1)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def read_evidence(
    path: Path,
    *,
    top_k: int,
    min_view_hits: int,
    min_consistency: float,
    min_pixel_count: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
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
                    "rank": int(_float(row, "rank")),
                    "face_id": int(_float(row, "face_id")),
                    "score": _float(row, "score"),
                    "pixel_count": pixel_count,
                    "view_hits": view_hits,
                    "consistency": consistency,
                    "mean_l1_error": _float(row, "mean_l1_error"),
                    "residual_rgb": np.asarray(
                        [
                            _float(row, "mean_residual_r"),
                            _float(row, "mean_residual_g"),
                            _float(row, "mean_residual_b"),
                        ],
                        dtype=np.float32,
                    ),
                }
            )
    rows.sort(key=lambda x: (float(x["score"]), float(x["pixel_count"])), reverse=True)
    return rows[: int(top_k)]


def selected_surface(
    faces: torch.Tensor,
    vertices: torch.Tensor,
    evidence: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]], list[int]]:
    valid_rows: list[dict[str, Any]] = []
    skipped: list[int] = []
    for row in evidence:
        face_id = int(row["face_id"])
        if face_id < 0 or face_id >= int(faces.shape[0]):
            skipped.append(face_id)
            continue
        vertex_ids = faces[face_id].long()
        if int(vertex_ids.min().item()) < 0 or int(vertex_ids.max().item()) >= int(vertices.shape[0]):
            skipped.append(face_id)
            continue
        valid_rows.append(row)

    if not valid_rows:
        empty = torch.empty((0,), dtype=torch.long)
        return empty.reshape(0, 3), empty, empty.reshape(0, 3), empty, valid_rows, skipped

    face_ids = torch.as_tensor([int(row["face_id"]) for row in valid_rows], dtype=torch.long)
    selected_faces_global = faces[face_ids].long()
    unique_vertices, inverse = torch.unique(selected_faces_global.reshape(-1), sorted=True, return_inverse=True)
    selected_faces_local = inverse.reshape(-1, 3).long()
    targets = torch.as_tensor(
        np.stack([row["residual_rgb"] for row in valid_rows], axis=0),
        dtype=torch.float32,
    )
    weights = torch.as_tensor(
        [
            float(max(row["pixel_count"], 1.0) ** 0.5) * float(max(row["consistency"], 1e-3))
            for row in valid_rows
        ],
        dtype=torch.float32,
    )
    return selected_faces_local, unique_vertices, targets, weights, valid_rows, skipped


def surface_edges(faces_local: torch.Tensor) -> torch.Tensor:
    if faces_local.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    edges = torch.cat(
        [
            faces_local[:, [0, 1]],
            faces_local[:, [1, 2]],
            faces_local[:, [2, 0]],
        ],
        dim=0,
    )
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def solve_delta(
    faces_local: torch.Tensor,
    targets_rgb: torch.Tensor,
    weights: torch.Tensor,
    *,
    strength: float,
    max_abs_delta_rgb: float,
    lambda_mag: float,
    lambda_smooth: float,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    vertex_count = int(faces_local.max().item()) + 1 if faces_local.numel() else 0
    if vertex_count == 0:
        return torch.empty((0, 3), dtype=torch.float32), {
            "initial_data_loss": 0.0,
            "final_data_loss": 0.0,
            "final_mag_loss": 0.0,
            "final_smooth_loss": 0.0,
        }

    faces_local = faces_local.to(device=device)
    target = (targets_rgb.to(device=device) * float(strength)).clamp(
        -float(max_abs_delta_rgb),
        float(max_abs_delta_rgb),
    )
    w = weights.to(device=device).clamp_min(1e-6).reshape(-1, 1)
    edges = surface_edges(faces_local.detach().cpu()).to(device=device)
    param = torch.zeros((vertex_count, 3), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([param], lr=float(lr))

    with torch.no_grad():
        pred0 = torch.zeros_like(target)
        initial_data_loss = (((pred0 - target) ** 2) * w).sum() / w.sum().clamp_min(1e-6)

    final_data_loss = initial_data_loss
    final_mag_loss = torch.zeros((), device=device)
    final_smooth_loss = torch.zeros((), device=device)
    for _ in range(int(steps)):
        delta = float(max_abs_delta_rgb) * torch.tanh(param)
        face_pred = delta[faces_local].mean(dim=1)
        data_loss = (((face_pred - target) ** 2) * w).sum() / w.sum().clamp_min(1e-6)
        mag_loss = (delta**2).mean()
        if edges.numel():
            smooth_loss = ((delta[edges[:, 0]] - delta[edges[:, 1]]) ** 2).mean()
        else:
            smooth_loss = torch.zeros((), dtype=torch.float32, device=device)
        loss = data_loss + float(lambda_mag) * mag_loss + float(lambda_smooth) * smooth_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_data_loss = data_loss.detach()
        final_mag_loss = mag_loss.detach()
        final_smooth_loss = smooth_loss.detach()

    with torch.no_grad():
        delta = (float(max_abs_delta_rgb) * torch.tanh(param)).detach().cpu()
    stats = {
        "initial_data_loss": float(initial_data_loss.detach().cpu().item()),
        "final_data_loss": float(final_data_loss.detach().cpu().item()),
        "final_mag_loss": float(final_mag_loss.detach().cpu().item()),
        "final_smooth_loss": float(final_smooth_loss.detach().cpu().item()),
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


def write_audit(output_model: Path, audit: dict[str, Any]) -> None:
    (output_model / "surface_residual_ridge_delta_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    before = audit["topology_before"]
    after = audit["topology_after"]
    unchanged = before["triangles"] == after["triangles"] and before["vertices"] == after["vertices"]
    lines = [
        "# ECSR Surface Residual Ridge Delta Audit",
        "",
        f"- operator: `{audit['operator']}`",
        f"- source model: `{audit['source_model']}`",
        f"- output model: `{audit['output_model']}`",
        f"- iteration: `{audit['iteration']}`",
        f"- evidence rows read: `{audit['evidence_rows_read']}`",
        f"- faces used: `{audit['faces_used']}`",
        f"- invalid/skipped faces: `{audit['faces_skipped']}`",
        f"- vertices modified: `{audit['vertices_modified']}`",
        f"- selected patch edges: `{audit['selected_patch_edges']}`",
        f"- strength: `{audit['strength']}`",
        f"- max abs delta RGB: `{audit['max_abs_delta_rgb']}`",
        f"- ridge lambda: `{audit['lambda_mag']}`",
        f"- smooth lambda: `{audit['lambda_smooth']}`",
        f"- initial data loss: `{audit['solver']['initial_data_loss']:.8f}`",
        f"- final data loss: `{audit['solver']['final_data_loss']:.8f}`",
        f"- delta RGB abs mean: `{audit['delta_rgb_abs_mean']:.8f}`",
        f"- delta RGB abs max: `{audit['delta_rgb_abs_max']:.8f}`",
        f"- topology unchanged: `{unchanged}`",
        f"- degenerate faces: `{after['degenerate_face_count']}`",
        f"- invalid indices: `{after['invalid_index_count']}`",
        "",
        "This operator uses only train-side Phase-A evidence. It changes SH DC",
        "coefficients on the selected surface patch and leaves topology intact.",
    ]
    (output_model / "surface_residual_ridge_delta_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_checkpoint = checkpoint_path(args.source_model, args.iteration)
    output_checkpoint = args.output_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud_state_dict.pt"
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    copy_model_metadata(args.source_model, args.output_model)

    state = torch.load(source_checkpoint, map_location="cpu")
    faces = state["_triangle_indices"].detach().cpu().long()
    vertices = state["triangles_points"].detach().cpu()
    features_dc = state["features_dc"].detach().cpu().clone().float()
    if features_dc.ndim != 3 or features_dc.shape[1] != 1 or features_dc.shape[2] != 3:
        raise ValueError(f"expected features_dc shape [V,1,3], got {tuple(features_dc.shape)}")

    evidence = read_evidence(
        args.evidence_csv,
        top_k=int(args.top_k),
        min_view_hits=int(args.min_view_hits),
        min_consistency=float(args.min_consistency),
        min_pixel_count=float(args.min_pixel_count),
    )
    faces_local, unique_vertices, targets_rgb, weights, used_rows, skipped_faces = selected_surface(
        faces,
        vertices,
        evidence,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    delta_rgb, solver_stats = solve_delta(
        faces_local,
        targets_rgb,
        weights,
        strength=float(args.strength),
        max_abs_delta_rgb=float(args.max_abs_delta_rgb),
        lambda_mag=float(args.lambda_mag),
        lambda_smooth=float(args.lambda_smooth),
        steps=int(args.steps),
        lr=float(args.lr),
        device=device,
    )

    if unique_vertices.numel():
        features_dc[unique_vertices, 0, :] += delta_rgb / float(C0)
    out_state = clone_state(state)
    out_state["features_dc"] = features_dc.to(dtype=state["features_dc"].dtype)
    torch.save(out_state, output_checkpoint)
    degenerate, invalid = validate_faces(out_state["triangles_points"], out_state["_triangle_indices"])

    patch_edges = surface_edges(faces_local)
    audit = {
        "operator": "surface_residual_ridge_dc_delta",
        "test_usage": "none",
        "source_model": str(args.source_model),
        "source_checkpoint": str(source_checkpoint),
        "output_model": str(args.output_model),
        "output_checkpoint": str(output_checkpoint),
        "iteration": int(args.iteration),
        "evidence_csv": str(args.evidence_csv),
        "evidence_rows_read": int(len(evidence)),
        "top_k_requested": int(args.top_k),
        "filters": {
            "min_view_hits": int(args.min_view_hits),
            "min_consistency": float(args.min_consistency),
            "min_pixel_count": float(args.min_pixel_count),
        },
        "faces_used": int(faces_local.shape[0]),
        "faces_skipped": int(len(skipped_faces)),
        "vertices_modified": int(unique_vertices.numel()),
        "selected_patch_edges": int(patch_edges.shape[0]),
        "strength": float(args.strength),
        "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
        "lambda_mag": float(args.lambda_mag),
        "lambda_smooth": float(args.lambda_smooth),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "device": str(device),
        "solver": solver_stats,
        "delta_rgb_abs_mean": float(delta_rgb.abs().mean().item()) if delta_rgb.numel() else 0.0,
        "delta_rgb_abs_max": float(delta_rgb.abs().max().item()) if delta_rgb.numel() else 0.0,
        "topology_before": {
            "triangles": int(faces.shape[0]),
            "vertices": int(vertices.shape[0]),
        },
        "topology_after": {
            "triangles": int(out_state["_triangle_indices"].shape[0]),
            "vertices": int(out_state["triangles_points"].shape[0]),
            "degenerate_face_count": int(degenerate),
            "invalid_index_count": int(invalid),
        },
        "used_faces_preview": [
            {
                "rank": int(row["rank"]),
                "face_id": int(row["face_id"]),
                "score": float(row["score"]),
                "pixel_count": float(row["pixel_count"]),
                "view_hits": int(row["view_hits"]),
                "consistency": float(row["consistency"]),
                "residual_rgb": [float(x) for x in row["residual_rgb"].tolist()],
            }
            for row in used_rows[:20]
        ],
    }
    write_audit(args.output_model, audit)
    print(json.dumps(audit, indent=2))
    return 0 if degenerate == 0 and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
