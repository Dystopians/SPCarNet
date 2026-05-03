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

from ss3dm_prior.meshsplatopt.counterfactual_edit_gate import validate_edit_counterfactual
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshSplatOptEditType, MeshState
from ss3dm_prior.meshsplatopt.ground_void_fill import make_ground_plane_void_fill


def make_mesh() -> MeshState:
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return MeshState(vertices, faces)


def exact_equal(a: MeshState, b: MeshState) -> bool:
    return np.array_equal(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/carnet/meshsplatopt/stageR10_counterfactual_edits_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    good_state = make_mesh()
    good_fill = make_ground_plane_void_fill(
        good_state,
        bbox_min=(1.0, 0.0),
        bbox_max=(2.0, 1.0),
        observed_support=True,
        proposal_id="good_fill",
    ).edit
    good_report = validate_edit_counterfactual(good_state, good_fill, snapshot_path=out / "good_fill.npz", commit_on_accept=True)

    bad_floater_state = make_mesh()
    bad_before = bad_floater_state.copy()
    bad_floater = MeshEdit(
        edit_id="bad_floater",
        edit_type=MeshSplatOptEditType.FILL_PATCH.value,
        defect_id="bad",
        inserted_vertices=[[10, 10, 10], [11, 10, 10], [10, 11, 10]],
        inserted_faces=[[0, 1, 2]],
        evidence_summary={"boundary_loop_support": False, "free_space_risk": 0.9},
        risk_summary={"free_space_risk": 0.9},
    )
    bad_report = validate_edit_counterfactual(bad_floater_state, bad_floater, snapshot_path=out / "bad_floater.npz")

    snap_state = make_mesh()
    snap_before = snap_state.copy()
    snap = MeshEdit(
        edit_id="snap_free_space",
        edit_type=MeshSplatOptEditType.SNAP_VERTICES.value,
        defect_id="snap",
        affected_vertices=[0],
        attribute_changes={"target_positions": {"0": [0, 0, 5]}},
        risk_summary={"free_space_risk": 0.8, "snap_through_free_space": True},
    )
    snap_report = validate_edit_counterfactual(snap_state, snap, snapshot_path=out / "snap.npz")

    delete_state = make_mesh()
    delete_before = delete_state.copy()
    delete = MeshEdit(
        edit_id="delete_supported",
        edit_type=MeshSplatOptEditType.DELETE_TRIANGLES.value,
        defect_id="delete",
        affected_faces=[0],
        risk_summary={"deletes_supported_surface": True},
    )
    delete_report = validate_edit_counterfactual(delete_state, delete, snapshot_path=out / "delete.npz")

    checks = {
        "good_fill_accepted": good_report.accepted,
        "bad_floater_rejected": not bad_report.accepted and exact_equal(bad_floater_state, bad_before),
        "snap_free_space_rejected": not snap_report.accepted and exact_equal(snap_state, snap_before),
        "delete_supported_surface_rejected": not delete_report.accepted and exact_equal(delete_state, delete_before),
        "non_delete_edit_accepted": good_report.edit_type == "FILL_PATCH" and good_report.accepted,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reports": [good_report.to_dict(), bad_report.to_dict(), snap_report.to_dict(), delete_report.to_dict()],
    }
    (out / "counterfactual_edits_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# MeshSplatOpt Stage R10 Counterfactual Edits Smoke", "", f"Status: `{report['status']}`", "", "## Checks", ""]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    (out / "counterfactual_edits_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(f"Stage R10 smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
