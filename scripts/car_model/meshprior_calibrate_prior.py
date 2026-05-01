"""Run targeted MeshPrior prior-calibration experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.car_model.meshprior_run_synthetic_damage_benchmark import (
    analytic_box_field,
    analytic_box_occupancy_field,
    analytic_box_surface_distance,
)
from ss3dm_prior.meshprior.calibration import evaluate_snap_calibration_profile
from ss3dm_prior.meshprior.synthetic_damage import make_box_mesh, perturb_vertices


def run(args: argparse.Namespace) -> dict[str, object]:
    vertices, faces = make_box_mesh()
    damaged = perturb_vertices(vertices, faces, sigma=0.03, seed=0)
    rows = [
        evaluate_snap_calibration_profile(
            vertices=damaged.vertices,
            faces=damaged.faces,
            valid_face_mask=damaged.valid_face_mask,
            support_decoder=analytic_box_field,
            occupancy_decoder=analytic_box_occupancy_field,
            surface_distance_fn=analytic_box_surface_distance,
            profile_name=profile,
        )
        for profile in args.profiles
    ]
    by_profile = {str(row["profile"]): row for row in rows}
    uncal = by_profile.get("none")
    cal = by_profile.get("surface_support_v1")
    summary = {
        "damage_type": "vertex_noise",
        "rows": rows,
        "calibrated_improves_recall_vs_uncalibrated": bool(
            cal is not None
            and uncal is not None
            and float(cal["snapped_valid_surface_protect_recall"]) > float(uncal["snapped_valid_surface_protect_recall"])
        ),
        "calibrated_keeps_baseline_recall": bool(
            cal is not None
            and float(cal["snapped_valid_surface_protect_recall"]) >= float(cal["baseline_valid_surface_protect_recall"])
        ),
        "calibrated_surface_improves": bool(cal is not None and float(cal["surface_distance_delta_mean"]) > 0.0),
        "free_space_safe": bool(all(float(row["free_space_violation_delta"]) <= 0.0 for row in rows)),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MeshPrior prior calibration.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--profiles", nargs="+", default=["none", "surface_support_v1"])
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
