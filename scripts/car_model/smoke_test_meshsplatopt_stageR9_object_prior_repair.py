#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.csef_builder import write_ascii_ply
from ss3dm_prior.meshsplatopt.edit_types import MeshState
from ss3dm_prior.meshsplatopt.object_prior_repair import make_object_prior_repair_proposals, write_object_repair_outputs


def make_car_like_box_missing_side() -> MeshState:
    vertices = np.asarray(
        [
            [0, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 1],
            [2, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return MeshState(vertices, faces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/carnet/meshsplatopt/stageR9_object_prior_repair_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = make_car_like_box_missing_side()
    write_ascii_ply(out / "car_like_missing_side.ply", state.vertices, state.faces)
    confident = make_object_prior_repair_proposals(
        state,
        region_id="synthetic_vehicle",
        canonicalization_confidence=0.85,
        posterior_uncertainty=0.2,
        missing_panel_bbox=((0.0, 0.0), (2.0, 1.0)),
    )
    uncertain = make_object_prior_repair_proposals(
        state,
        region_id="uncertain_vehicle",
        canonicalization_confidence=0.3,
        posterior_uncertainty=0.8,
        missing_panel_bbox=((0.0, 0.0), (2.0, 1.0)),
    )
    write_object_repair_outputs(confident + uncertain, out / "object_repair_outputs")
    confident_types = {p.proposal_type for p in confident}
    uncertain_types = {p.proposal_type for p in uncertain}
    all_metadata_safe = all(
        p.metadata.get("prior_proposes_evidence_disposes") and p.metadata.get("requires_scene_counterfactual_validation")
        for p in confident + uncertain
    )
    checks = {
        "confident_package_has_protect": "vehicle_protect_mask" in confident_types,
        "confident_package_has_fill": "vehicle_discontinuity_fill_candidate" in confident_types,
        "uncertain_has_no_fill": "vehicle_discontinuity_fill_candidate" not in uncertain_types,
        "uncertain_limited_to_protect_or_prune": uncertain_types.issubset({"vehicle_protect_mask"}),
        "all_metadata_requires_scene_gate": all_metadata_safe,
    }
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    (out / "object_prior_repair_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# MeshSplatOpt Stage R9 Object Prior Repair Smoke", "", f"Status: `{report['status']}`", "", "## Checks", ""]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    (out / "object_prior_repair_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(f"Stage R9 object-prior smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
