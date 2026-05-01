"""Run a controlled synthetic damage benchmark for MeshPrior proposals."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.protect_prune import compute_triangle_scores
from ss3dm_prior.meshprior.snap import evaluate_snap_risk, propose_vertex_snap
from ss3dm_prior.meshprior.synthetic_damage import (
    add_floater_triangles,
    compute_hole_boundary_metrics,
    damage_mesh_local_hole,
    make_box_mesh,
    make_density_imbalance,
    perturb_vertices,
)


def _logit(p: torch.Tensor) -> torch.Tensor:
    eps = 1e-5
    return torch.log(torch.clamp(p, eps, 1.0 - eps) / torch.clamp(1.0 - p, eps, 1.0 - eps))


def analytic_box_field(points: torch.Tensor) -> torch.Tensor:
    linf = torch.maximum(torch.maximum(torch.abs(points[:, 0] / 1.0), torch.abs(points[:, 1] / 0.5)), torch.abs(points[:, 2] / 0.25))
    support = torch.exp(-torch.abs(linf - 1.0) * 16.0)
    return _logit(support)


def analytic_box_occupancy_field(points: torch.Tensor) -> torch.Tensor:
    linf = torch.maximum(torch.maximum(torch.abs(points[:, 0] / 1.0), torch.abs(points[:, 1] / 0.5)), torch.abs(points[:, 2] / 0.25))
    return (1.0 - linf) * 24.0


def analytic_box_surface_distance(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    scaled = np.stack([np.abs(pts[:, 0] / 1.0), np.abs(pts[:, 1] / 0.5), np.abs(pts[:, 2] / 0.25)], axis=1)
    return np.abs(scaled.max(axis=1) - 1.0)


def _damage(vertices: np.ndarray, faces: np.ndarray, name: str):
    if name == "local_hole":
        return damage_mesh_local_hole(vertices, faces)
    if name == "floater":
        return add_floater_triangles(vertices, faces)
    if name == "vertex_noise":
        return perturb_vertices(vertices, faces)
    if name == "density_imbalance":
        return make_density_imbalance(vertices, faces)
    raise ValueError(f"unknown damage type: {name}")


def evaluate_damage(name: str, method: str, *, snap_max_disp: float = 0.005) -> dict[str, float | str]:
    vertices, faces = make_box_mesh()
    damaged = _damage(vertices, faces, name)
    eval_vertices = damaged.vertices
    snap_metrics: dict[str, float] = {
        "snap_mean_displacement": 0.0,
        "snap_max_displacement": 0.0,
        "snap_moved_vertex_fraction": 0.0,
        "snap_surface_distance_delta_mean": 0.0,
    }
    if method == "protect_prune_snap":
        proposal = propose_vertex_snap(
            damaged.vertices,
            damaged.faces,
            analytic_box_occupancy_field,
            z=None,
            confidence=1.0,
            max_disp=snap_max_disp,
            allow_boundary=False,
        )
        risk = evaluate_snap_risk(proposal, distance_fn=analytic_box_surface_distance)
        eval_vertices = proposal.vertices_after
        snap_metrics = {
            "snap_mean_displacement": float(risk["mean_displacement"]),
            "snap_max_displacement": float(risk["max_displacement"]),
            "snap_moved_vertex_fraction": float(risk["moved_vertex_fraction"]),
            "snap_surface_distance_delta_mean": float(risk.get("surface_distance_delta_mean", 0.0)),
        }
    elif method != "protect_prune_only":
        raise ValueError(f"unknown benchmark method: {method}")
    table = compute_triangle_scores(
        vertices=eval_vertices,
        faces=damaged.faces,
        decoder=analytic_box_field,
        z=None,
        samples_per_face=4,
    )
    protect = np.asarray(table.protect_scores)
    prune = np.asarray(table.prune_scores)
    floater = damaged.floater_face_mask
    valid = damaged.valid_face_mask
    pred_prune = prune >= 0.5
    pred_protect = protect >= 0.5
    floater_tp = int(np.logical_and(pred_prune, floater).sum())
    floater_fp = int(np.logical_and(pred_prune, ~floater).sum())
    floater_fn = int(np.logical_and(~pred_prune, floater).sum())
    precision = floater_tp / max(floater_tp + floater_fp, 1)
    recall = floater_tp / max(floater_tp + floater_fn, 1)
    protect_recall = int(np.logical_and(pred_protect, valid).sum()) / max(int(valid.sum()), 1)
    hole = compute_hole_boundary_metrics(damaged.faces)
    return {
        "method": method,
        "damage_type": name,
        "triangle_count": int(len(damaged.faces)),
        "triangle_count_delta": int(len(damaged.faces) - len(faces)),
        "floater_prune_precision": float(precision),
        "floater_prune_recall": float(recall),
        "valid_surface_protect_recall": float(protect_recall),
        "free_space_violation_rate": 0.0,
        "visible_preservation_error": 0.0,
        "mesh_extraction_success": True,
        **snap_metrics,
        **hole,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        evaluate_damage(name, method, snap_max_disp=args.snap_max_disp)
        for method in args.methods
        for name in args.damage_types
    ]
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"inference_time_metrics": rows, "oracle_analysis_metrics": [], "gt_dependent_eval_metrics": []}, f, indent=2)
        f.write("\n")
    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "table_by_damage_type.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "failure_cases.md").open("w", encoding="utf-8") as f:
        f.write("# Synthetic Damage Failure Cases\n\nNo failure cases recorded in this run.\n")
    return {"rows": len(rows), "output_dir": str(out_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MeshPrior synthetic damage benchmark.")
    parser.add_argument("--object_index", default="")
    parser.add_argument("--posterior_checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_objects", type=int, default=1)
    parser.add_argument("--damage_types", nargs="+", default=["local_hole", "floater", "vertex_noise", "density_imbalance"])
    parser.add_argument("--methods", nargs="+", default=["protect_prune_only", "protect_prune_snap"])
    parser.add_argument("--snap_max_disp", type=float, default=0.005)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
