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
from ss3dm_prior.meshsplatopt.edit_apply import apply_edit, verify_mesh_integrity
from ss3dm_prior.meshsplatopt.edit_snapshot import create_snapshot, rollback_edit
from ss3dm_prior.meshsplatopt.edit_types import MeshState
from ss3dm_prior.meshsplatopt.ground_void_fill import make_ground_plane_void_fill
from ss3dm_prior.meshsplatopt.hole_fill import find_boundary_loops, make_boundary_loop_fill, write_fill_outputs


def make_plane_with_hole(grid_n: int = 5, missing: set[tuple[int, int]] | None = None) -> MeshState:
    missing = missing or {(2, 2)}
    vertices = [(float(x), float(y), 0.0) for y in range(grid_n) for x in range(grid_n)]
    faces = []
    for y in range(grid_n - 1):
        for x in range(grid_n - 1):
            if (x, y) in missing:
                continue
            v0 = y * grid_n + x
            v1 = y * grid_n + x + 1
            v2 = (y + 1) * grid_n + x
            v3 = (y + 1) * grid_n + x + 1
            faces.extend([(v0, v1, v3), (v0, v3, v2)])
    return MeshState(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))


def boundary_edge_count(state: MeshState) -> int:
    return sum(len(loop) for loop in find_boundary_loops(state.faces))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/carnet/meshsplatopt/stageR8_giant_void_fill_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    small = make_plane_with_hole()
    loops = find_boundary_loops(small.faces)
    proposal = make_boundary_loop_fill(small, loops[0], proposal_id="small_hole_fill")
    small_before_boundary = boundary_edge_count(small)
    small_after = small.copy()
    apply_edit(small_after, proposal.edit)
    small_after_boundary = boundary_edge_count(small_after)

    giant = make_ground_plane_void_fill(
        small,
        bbox_min=(1.0, 1.0),
        bbox_max=(4.0, 4.0),
        z=0.0,
        grid_resolution=4,
        proposal_id="giant_ground_void",
        observed_support=True,
    )
    giant_after = small.copy()
    apply_edit(giant_after, giant.edit)
    giant_valid = verify_mesh_integrity(giant_after)["valid"]

    unknown_normal = make_ground_plane_void_fill(
        small,
        bbox_min=(10.0, 10.0),
        bbox_max=(14.0, 14.0),
        observed_support=False,
        allow_prior_only=False,
        proposal_id="unknown_normal",
    )
    unknown_prior = make_ground_plane_void_fill(
        small,
        bbox_min=(10.0, 10.0),
        bbox_max=(14.0, 14.0),
        observed_support=False,
        allow_prior_only=True,
        proposal_id="unknown_prior_diagnostic",
    )

    rollback_state = small.copy()
    before = rollback_state.copy()
    create_snapshot(rollback_state, out / "snapshots/fill.npz")
    apply_edit(rollback_state, proposal.edit)
    rollback_edit(rollback_state, out / "snapshots/fill.npz")
    rollback_exact = np.array_equal(rollback_state.vertices, before.vertices) and np.array_equal(rollback_state.faces, before.faces)

    degenerate = make_boundary_loop_fill(small, [0, 1, 1], proposal_id="degenerate")
    proposals = [proposal, giant, unknown_normal, unknown_prior, degenerate]
    write_ascii_ply(out / "small_hole_before.ply", small.vertices, small.faces)
    write_fill_outputs(small, proposals, out / "fill_outputs", preview_state=small_after)

    checks = {
        "small_hole_boundary_reduced": small_after_boundary < small_before_boundary,
        "giant_ground_void_valid_patch": giant.edit is not None and giant_valid and giant.certificate["expected_area_repaired"] > 0.0,
        "unknown_void_rejects_normal_mode": unknown_normal.edit is None and bool(unknown_normal.rejected_reason),
        "prior_only_diagnostic_marked": unknown_prior.edit is not None and bool(unknown_prior.certificate["prior_only_flag"]),
        "rollback_exact": rollback_exact,
        "degenerate_boundary_rejected": degenerate.edit is None and bool(degenerate.rejected_reason),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "small_before_boundary": small_before_boundary,
        "small_after_boundary": small_after_boundary,
        "checks": checks,
    }
    (out / "fill_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# MeshSplatOpt Stage R8 Giant Void Fill Smoke", "", f"Status: `{report['status']}`", "", "## Checks", ""]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    (out / "fill_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(f"Stage R8 fill smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
