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

from ss3dm_prior.meshsplatopt.edit_portfolio import PortfolioItem
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshSplatOptEditType, MeshState
from ss3dm_prior.meshsplatopt.ground_void_fill import make_ground_plane_void_fill
from ss3dm_prior.meshsplatopt.repair_state_machine import run_repair_state_machine


def make_state() -> MeshState:
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, -0.2], [0, 1, 0], [5, 5, 1], [6, 5, 1], [5, 6, 1]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64)
    return MeshState(vertices, faces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/carnet/meshsplatopt/stageR12_portfolio_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = make_state()
    fill = make_ground_plane_void_fill(state, bbox_min=(1, 0), bbox_max=(2, 1), observed_support=True, proposal_id="state_fill").edit
    prior_fill = make_ground_plane_void_fill(
        state, bbox_min=(10, 10), bbox_max=(12, 12), observed_support=False, allow_prior_only=True, proposal_id="prior_only"
    ).edit
    items = [
        PortfolioItem(
            MeshEdit("delete_floater", MeshSplatOptEditType.DELETE_TRIANGLES.value, "floater", affected_faces=[2]),
            0.8,
            0.0,
            -1.0,
            0.0,
            0.1,
        ),
        PortfolioItem(
            MeshEdit(
                "snap_dent",
                MeshSplatOptEditType.SNAP_VERTICES.value,
                "dent",
                affected_vertices=[2],
                attribute_changes={"target_positions": {"2": [1, 1, 0]}},
            ),
            0.5,
            0.0,
            0.0,
            0.0,
            0.2,
        ),
        PortfolioItem(fill, 0.9, 0.1, 2.0, 0.1, 0.2),
        PortfolioItem(MeshEdit("appearance", MeshSplatOptEditType.APPEARANCE_RESET.value, "appearance"), 0.2, 0.0, 0.0, 0.0, 0.1),
        PortfolioItem(prior_fill, 0.7, 0.1, 2.0, 0.2, 0.7, prior_only_flag=True),
    ]
    result = run_repair_state_machine(state, items, args.output_dir)
    accepted_types = {x["edit"]["edit_type"] for x in result.accepted_edits}
    rejected_reasons = [x.get("reason", "") for x in result.rejected_edits]
    checks = {
        "three_edit_classes_accepted": len(accepted_types.intersection({"DELETE_TRIANGLES", "SNAP_VERTICES", "FILL_PATCH", "APPEARANCE_RESET"})) >= 3,
        "prior_only_fill_rejected": "prior_only_rejected_by_state_machine" in rejected_reasons,
        "trace_written": Path(args.output_dir, "state_machine_trace.json").exists(),
        "audit_written": Path(args.output_dir, "final_audit.json").exists(),
    }
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "accepted_types": sorted(accepted_types)}
    Path(args.output_dir, "portfolio_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(f"Stage R12 smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
