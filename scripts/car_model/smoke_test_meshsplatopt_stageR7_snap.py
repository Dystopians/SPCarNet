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
from ss3dm_prior.meshsplatopt.edit_apply import apply_edit
from ss3dm_prior.meshsplatopt.edit_snapshot import create_snapshot, rollback_edit
from ss3dm_prior.meshsplatopt.edit_types import MeshState
from ss3dm_prior.meshsplatopt.snap_proposals import fit_plane, make_snap_proposals, point_plane_residual, write_snap_outputs


def make_dented_plane() -> tuple[MeshState, int]:
    vertices = []
    grid_n = 5
    dent_vid = 2 * grid_n + 2
    for y in range(grid_n):
        for x in range(grid_n):
            z = -0.4 if y * grid_n + x == dent_vid else 0.0
            vertices.append((float(x), float(y), z))
    faces = []
    for y in range(grid_n - 1):
        for x in range(grid_n - 1):
            v0 = y * grid_n + x
            v1 = y * grid_n + x + 1
            v2 = (y + 1) * grid_n + x
            v3 = (y + 1) * grid_n + x + 1
            faces.extend([(v0, v1, v3), (v0, v3, v2)])
    return MeshState(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)), dent_vid


def plane_error(state: MeshState) -> float:
    normal, d = fit_plane(state.vertices)
    return float(np.mean(np.abs(point_plane_residual(state.vertices, normal, d))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/carnet/meshsplatopt/stageR7_snap_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state, dent_vid = make_dented_plane()
    before = state.copy()
    write_ascii_ply(out / "dented_before.ply", state.vertices, state.faces)
    proposals = make_snap_proposals(state, candidate_vertices=[dent_vid], supported_vertices={dent_vid})
    best = min([p for p in proposals if not p.rejected_reason], key=lambda p: p.expected_error_after)
    before_error = plane_error(state)
    create_snapshot(state, out / "snapshots/dent_snap.npz")
    apply_edit(state, best.edit)
    after_error = plane_error(state)
    rollback_edit(state, out / "snapshots/dent_snap.npz")
    rollback_exact = np.array_equal(state.vertices, before.vertices) and np.array_equal(state.faces, before.faces)

    floater_state = MeshState(
        vertices=np.vstack([before.vertices, np.asarray([[10.0, 10.0, 2.0]], dtype=np.float64)]),
        faces=before.faces.copy(),
    )
    floater_vid = len(floater_state.vertices) - 1
    floater_proposals = make_snap_proposals(floater_state, candidate_vertices=[floater_vid], supported_vertices=set())
    floater_rejected = bool(floater_proposals and floater_proposals[0].rejected_reason)

    misaligned_state, mid = make_dented_plane()
    misaligned_state.vertices[mid, 2] = 0.25
    mis_props = make_snap_proposals(misaligned_state, candidate_vertices=[mid], supported_vertices={mid})
    mis_best = min([p for p in mis_props if not p.rejected_reason], key=lambda p: p.expected_error_after)
    mis_before = plane_error(misaligned_state)
    apply_edit(misaligned_state, mis_best.edit)
    mis_after = plane_error(misaligned_state)

    write_snap_outputs(before, proposals + floater_proposals + mis_props, out / "snap_outputs", preview_proposal=best)
    checks = {
        "dent_error_reduced": after_error < before_error,
        "floater_rejected_without_support": floater_rejected,
        "misalignment_error_reduced": mis_after < mis_before,
        "rollback_exact": rollback_exact,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "before_error": before_error,
        "after_error": after_error,
        "misalignment_before_error": mis_before,
        "misalignment_after_error": mis_after,
        "checks": checks,
    }
    (out / "snap_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# MeshSplatOpt Stage R7 Snap Smoke", "", f"Status: `{report['status']}`", "", "## Checks", ""]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    (out / "snap_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(f"Stage R7 snap smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
