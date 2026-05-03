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
from ss3dm_prior.meshsplatopt.edit_apply import apply_edit, summarize_topology_delta, verify_mesh_integrity
from ss3dm_prior.meshsplatopt.edit_snapshot import create_snapshot, rollback_edit
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshSplatOptEditType, MeshState


def make_mesh() -> MeshState:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]], dtype=np.int64)
    return MeshState(vertices=vertices, faces=faces, attributes={"source": "stageR5_smoke"})


def arrays_equal(a: MeshState, b: MeshState) -> bool:
    return np.array_equal(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces) and a.attributes == b.attributes


def run_roundtrip(state: MeshState, edit: MeshEdit, snapshot_path: Path, mesh_after_path: Path) -> dict:
    before = state.copy()
    create_snapshot(state, snapshot_path)
    after = apply_edit(state, edit)
    write_ascii_ply(mesh_after_path, after.vertices, after.faces)
    delta = summarize_topology_delta(before, after)
    integrity_after = verify_mesh_integrity(after.copy())
    rollback_edit(state, snapshot_path)
    return {
        "edit_type": edit.edit_type,
        "rollback_exact": arrays_equal(state, before),
        "integrity_after": integrity_after,
        "topology_delta": delta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="outputs/carnet/meshsplatopt/stageR5_reversible_edits_smoke",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = []

    state = make_mesh()
    write_ascii_ply(out / "before.ply", state.vertices, state.faces)
    reports.append(
        run_roundtrip(
            state,
            MeshEdit(edit_id="delete_0", edit_type=MeshSplatOptEditType.DELETE_TRIANGLES.value, defect_id="d", affected_faces=[0]),
            out / "snapshots/delete.npz",
            out / "after_delete.ply",
        )
    )
    reports.append(
        run_roundtrip(
            state,
            MeshEdit(
                edit_id="snap_0",
                edit_type=MeshSplatOptEditType.SNAP_VERTICES.value,
                defect_id="d",
                affected_vertices=[2],
                attribute_changes={"target_positions": {"2": [1.0, 1.0, 0.25]}},
            ),
            out / "snapshots/snap.npz",
            out / "after_snap.ply",
        )
    )
    reports.append(
        run_roundtrip(
            state,
            MeshEdit(
                edit_id="fill_0",
                edit_type=MeshSplatOptEditType.FILL_PATCH.value,
                defect_id="d",
                inserted_vertices=[[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [0.5, 2.5, 0.0]],
                inserted_faces=[[0, 1, 2]],
            ),
            out / "snapshots/fill.npz",
            out / "after_fill.ply",
        )
    )
    reports.append(
        run_roundtrip(
            state,
            MeshEdit(
                edit_id="collapse_0",
                edit_type=MeshSplatOptEditType.EDGE_COLLAPSE.value,
                defect_id="d",
                affected_vertices=[1, 4],
            ),
            out / "snapshots/collapse.npz",
            out / "after_collapse.ply",
        )
    )
    reports.append(
        run_roundtrip(
            state,
            MeshEdit(edit_id="split_0", edit_type=MeshSplatOptEditType.SPLIT_TRIANGLES.value, defect_id="d", affected_faces=[0]),
            out / "snapshots/split.npz",
            out / "after_split.ply",
        )
    )
    reports.append(
        run_roundtrip(
            state,
            MeshEdit(edit_id="protect_0", edit_type=MeshSplatOptEditType.PROTECT.value, defect_id="d", affected_faces=[0]),
            out / "snapshots/protect.npz",
            out / "after_protect.ply",
        )
    )
    reports.append(
        run_roundtrip(
            state,
            MeshEdit(edit_id="appearance_0", edit_type=MeshSplatOptEditType.APPEARANCE_RESET.value, defect_id="d", affected_faces=[0]),
            out / "snapshots/appearance.npz",
            out / "after_appearance.ply",
        )
    )

    invalid_index = verify_mesh_integrity(MeshState(vertices=state.vertices, faces=np.asarray([[0, 1, 99]], dtype=np.int64)))
    degenerate = verify_mesh_integrity(MeshState(vertices=state.vertices, faces=np.asarray([[0, 1, 1]], dtype=np.int64)))
    checks = {
        "all_roundtrip_exact": all(r["rollback_exact"] for r in reports),
        "all_after_integrity_valid": all(r["integrity_after"]["valid"] for r in reports),
        "invalid_index_caught": not invalid_index["valid"],
        "degenerate_face_caught": not degenerate["valid"],
        "fill_added_face": any(r["edit_type"] == "FILL_PATCH" and r["topology_delta"]["faces_delta"] == 1 for r in reports),
        "split_added_faces": any(r["edit_type"] == "SPLIT_TRIANGLES" and r["topology_delta"]["faces_delta"] == 2 for r in reports),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "roundtrips": reports,
        "invalid_index_check": invalid_index,
        "degenerate_check": degenerate,
        "checks": checks,
    }
    (out / "edit_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# MeshSplatOpt Stage R5 Reversible Edits Smoke", "", f"Status: `{status}`", "", "## Checks", ""]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    (out / "edit_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if status != "PASS":
        raise SystemExit(f"Stage R5 reversible edit smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
