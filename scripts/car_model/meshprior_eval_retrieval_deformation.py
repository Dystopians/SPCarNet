"""Evaluate retrieval-deformation MeshPrior fallback on synthetic damage."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.car_model.meshprior_run_synthetic_damage_benchmark import evaluate_damage
from ss3dm_prior.meshprior.retrieval_deformation import (
    AnchorBank,
    build_retrieval_proposals,
    compute_retrieval_triangle_scores,
    propose_retrieval_snap,
    retrieve_anchor,
    smooth_deform_anchor_to_observed,
)
from ss3dm_prior.meshprior.synthetic_damage import (
    add_floater_triangles,
    compute_hole_boundary_metrics,
    damage_mesh_local_hole,
    make_box_mesh,
    make_density_imbalance,
    perturb_vertices,
)


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


def _retrieval_row(bank: AnchorBank, damage_type: str, method: str, *, snap_max_disp: float) -> dict[str, Any]:
    clean_vertices, clean_faces = make_box_mesh()
    damaged = _damage(clean_vertices, clean_faces, damage_type)
    retrieval = retrieve_anchor(damaged.vertices, bank, query_object_id="synthetic_eval_box")
    anchor = bank.points[retrieval.anchor_index]
    deform_metrics = {"deform_mean_displacement": 0.0, "deform_max_displacement": 0.0, "deform_moved_fraction": 0.0}
    if method == "retrieval_deform":
        anchor, deform_metrics = smooth_deform_anchor_to_observed(anchor, damaged.vertices, blend=0.2, max_disp=0.01)
    elif method != "retrieval_only":
        raise ValueError(f"unknown retrieval method: {method}")

    eval_vertices = damaged.vertices
    snap_vertices, snap_metrics = propose_retrieval_snap(eval_vertices, anchor, max_disp=snap_max_disp)
    table = compute_retrieval_triangle_scores(
        vertices=snap_vertices if method == "retrieval_deform" else eval_vertices,
        faces=damaged.faces,
        anchor_points=anchor,
        retrieval=retrieval,
        samples_per_face=4,
    )
    batch = build_retrieval_proposals(table, region_id=f"{method}_{damage_type}", retrieval=retrieval)
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
    proposal_types = sorted({p.proposal_type for p in batch.proposals})
    return {
        "method": method,
        "damage_type": damage_type,
        "anchor_id": retrieval.object_id,
        "retrieval_score": retrieval.score,
        "retrieval_margin": retrieval.margin,
        "retrieval_uncertainty": retrieval.uncertainty,
        "triangle_count": int(len(damaged.faces)),
        "floater_prune_precision": float(precision),
        "floater_prune_recall": float(recall),
        "valid_surface_protect_recall": float(protect_recall),
        "free_space_violation_rate": 0.0,
        "proposal_types": ",".join(proposal_types),
        **hole,
        **snap_metrics,
        **deform_metrics,
    }


def _stage3_proxy_row(damage_type: str) -> dict[str, Any]:
    row = evaluate_damage(damage_type, "protect_prune_snap", snap_max_disp=0.005)
    return {
        **row,
        "method": "stage3_posterior_proxy",
        "damage_type": damage_type,
        "anchor_id": "",
        "retrieval_score": None,
        "retrieval_margin": None,
        "retrieval_uncertainty": None,
        "proposal_types": "protect,prune,snap",
        "deform_mean_displacement": 0.0,
        "deform_max_displacement": 0.0,
        "deform_moved_fraction": 0.0,
    }


def _recommendation(rows: list[dict[str, Any]]) -> str:
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    def avg(method: str, key: str) -> float:
        vals = [float(r[key]) for r in by_method.get(method, []) if r.get(key) is not None]
        return sum(vals) / max(len(vals), 1)

    retrieval_score = avg("retrieval_only", "floater_prune_recall") + avg("retrieval_only", "valid_surface_protect_recall")
    posterior_score = avg("stage3_posterior_proxy", "floater_prune_recall") + avg("stage3_posterior_proxy", "valid_surface_protect_recall")
    deform_protect = avg("retrieval_deform", "valid_surface_protect_recall")
    retrieval_protect = avg("retrieval_only", "valid_surface_protect_recall")
    if deform_protect + 1e-6 < retrieval_protect:
        return "KEEP_RETRIEVAL_ONLY_KILL_DEFORMATION"
    if retrieval_score > posterior_score + 0.05:
        return "PIVOT_TO_RETRIEVAL_DEFORMATION"
    return "KEEP_AS_BASELINE"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = AnchorBank.from_npz(args.anchor_bank)
    damage_types = args.damage_types
    rows: list[dict[str, Any]] = []
    for damage_type in damage_types:
        rows.append(_stage3_proxy_row(damage_type))
    for method in ["retrieval_only", "retrieval_deform"]:
        for damage_type in damage_types:
            rows.append(_retrieval_row(bank, damage_type, method, snap_max_disp=args.snap_max_disp))
    recommendation = _recommendation(rows)
    payload = {
        "anchor_bank": str(args.anchor_bank),
        "damage_types": damage_types,
        "inference_time_metrics": rows,
        "oracle_analysis_metrics": [],
        "gt_dependent_eval_metrics": [],
        "recommendation": recommendation,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    columns = sorted({k for row in rows for k in row.keys()})
    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Retrieval-Deformation Evaluation\n\n")
        f.write(f"Recommendation: `{recommendation}`\n\n")
        f.write(f"Rows: `{len(rows)}`\n")
    return {"output_dir": str(out_dir), "rows": len(rows), "recommendation": recommendation}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MeshPrior retrieval-deformation fallback.")
    parser.add_argument("--anchor_bank", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--damage_types", nargs="+", default=["local_hole", "floater", "vertex_noise", "density_imbalance"])
    parser.add_argument("--snap_max_disp", type=float, default=0.005)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
