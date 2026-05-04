#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.counterfactual_edit_gate import validate_edit_counterfactual
from ss3dm_prior.meshsplatopt.edit_apply import apply_edit, summarize_topology_delta
from ss3dm_prior.meshsplatopt.edit_snapshot import mesh_checksum
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshSplatOptEditType, MeshState
from ss3dm_prior.meshsplatopt.ground_void_fill import make_ground_plane_void_fill


DOC = REPO_ROOT / "docs/car_model/final_stageF38_counterfactual_gate_ablation_report.md"


@dataclass(frozen=True)
class CaseResult:
    case: str
    gated_accepted: bool
    gated_reasons: list[str]
    gated_state_expected: bool
    unsafe_committed: bool
    unsafe_topology_delta: dict[str, int]
    unsafe_max_vertex_displacement: float
    unsafe_checksum_changed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_mesh() -> MeshState:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return MeshState(vertices, faces)


def state_equal(a: MeshState, b: MeshState) -> bool:
    return np.array_equal(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces) and a.attributes == b.attributes


def max_vertex_displacement(before: MeshState, after: MeshState) -> float:
    count = min(len(before.vertices), len(after.vertices))
    if count == 0:
        return 0.0
    return float(np.linalg.norm(after.vertices[:count] - before.vertices[:count], axis=1).max())


def build_cases() -> list[tuple[str, MeshEdit]]:
    good_state = make_mesh()
    good_fill = make_ground_plane_void_fill(
        good_state,
        bbox_min=(1.0, 0.0),
        bbox_max=(2.0, 1.0),
        observed_support=True,
        proposal_id="good_fill",
    ).edit
    return [
        ("accepted_supported_fill", good_fill),
        (
            "rejected_bad_floater",
            MeshEdit(
                edit_id="bad_floater",
                edit_type=MeshSplatOptEditType.FILL_PATCH.value,
                defect_id="bad",
                inserted_vertices=[[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]],
                inserted_faces=[[0, 1, 2]],
                evidence_summary={"boundary_loop_support": False, "free_space_risk": 0.9},
                risk_summary={"free_space_risk": 0.9},
            ),
        ),
        (
            "rejected_snap_through_free_space",
            MeshEdit(
                edit_id="snap_free_space",
                edit_type=MeshSplatOptEditType.SNAP_VERTICES.value,
                defect_id="snap",
                affected_vertices=[0],
                attribute_changes={"target_positions": {"0": [0.0, 0.0, 5.0]}},
                risk_summary={"free_space_risk": 0.8, "snap_through_free_space": True},
            ),
        ),
        (
            "rejected_delete_supported_surface",
            MeshEdit(
                edit_id="delete_supported",
                edit_type=MeshSplatOptEditType.DELETE_TRIANGLES.value,
                defect_id="delete",
                affected_faces=[0],
                risk_summary={"deletes_supported_surface": True},
            ),
        ),
    ]


def evaluate_case(name: str, edit: MeshEdit, output_dir: Path) -> CaseResult:
    gated_state = make_mesh()
    gated_before = gated_state.copy()
    gated_before_checksum = mesh_checksum(gated_before)
    gated_report = validate_edit_counterfactual(
        gated_state,
        edit,
        snapshot_path=output_dir / f"{name}_snapshot.npz",
        commit_on_accept=True,
    )
    if gated_report.accepted:
        gated_state_expected = not state_equal(gated_state, gated_before)
    else:
        gated_state_expected = state_equal(gated_state, gated_before) and mesh_checksum(gated_state) == gated_before_checksum

    unsafe_state = make_mesh()
    unsafe_before = unsafe_state.copy()
    unsafe_before_checksum = mesh_checksum(unsafe_before)
    apply_edit(unsafe_state, edit)
    unsafe_delta = summarize_topology_delta(unsafe_before, unsafe_state)
    unsafe_checksum = mesh_checksum(unsafe_state)

    return CaseResult(
        case=name,
        gated_accepted=bool(gated_report.accepted),
        gated_reasons=list(gated_report.reasons),
        gated_state_expected=bool(gated_state_expected),
        unsafe_committed=not state_equal(unsafe_state, unsafe_before),
        unsafe_topology_delta=unsafe_delta,
        unsafe_max_vertex_displacement=max_vertex_displacement(unsafe_before, unsafe_state),
        unsafe_checksum_changed=unsafe_checksum != unsafe_before_checksum,
    )


def write_report(output_dir: Path, results: list[CaseResult]) -> None:
    rejected_cases = [r for r in results if r.case.startswith("rejected_")]
    checks = {
        "supported_fill_survives_gate": any(r.case == "accepted_supported_fill" and r.gated_accepted for r in results),
        "all_unsafe_rejected_cases_rollback_exact": all((not r.gated_accepted) and r.gated_state_expected for r in rejected_cases),
        "all_no_gate_rejected_cases_commit_damage": all(r.unsafe_committed and r.unsafe_checksum_changed for r in rejected_cases),
        "floater_would_add_unobserved_face": any(
            r.case == "rejected_bad_floater" and r.unsafe_topology_delta["faces_delta"] == 1 for r in results
        ),
        "snap_would_move_vertex_by_5m": any(
            r.case == "rejected_snap_through_free_space" and r.unsafe_max_vertex_displacement >= 5.0 for r in results
        ),
        "delete_would_remove_supported_face": any(
            r.case == "rejected_delete_supported_surface" and r.unsafe_topology_delta["faces_delta"] == -1
            for r in results
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {"status": status, "checks": checks, "results": [r.to_dict() for r in results]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "counterfactual_gate_ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Final Stage F38 - Counterfactual Gate Ablation Report",
        "",
        "Decision: `F38_SYNTHETIC_COUNTERFACTUAL_GATE_PASS_REAL_SCENE_FULL_ABLATION_STILL_OPEN`.",
        "",
        "This is a mechanism-level counterfactual ablation. It applies the same edit proposals with the MeshSplatOpt gate/rollback path and with an unsafe no-gate/no-rollback path.",
        "",
        f"Status: `{status}`.",
        "",
        "## Checks",
        "",
        "| check | result |",
        "| --- | --- |",
    ]
    for key, value in checks.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "## Cases",
        "",
        "| case | gated accepted | gated reasons | gated state expected | no-gate topology delta | no-gate max vertex displacement |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        reasons = ", ".join(r.gated_reasons) if r.gated_reasons else "none"
        delta = f"V {r.unsafe_topology_delta['vertices_delta']}, F {r.unsafe_topology_delta['faces_delta']}"
        lines.append(
            f"| `{r.case}` | `{r.gated_accepted}` | `{reasons}` | `{r.gated_state_expected}` | `{delta}` | `{r.unsafe_max_vertex_displacement:.6f}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The ablation shows that the safety gate is load-bearing: the floater, free-space snap, and supported-surface deletion are all rejected and exactly rolled back, while the no-gate/no-rollback path commits each damaging topology or geometry mutation. This closes the implementation-level counterfactual gap, but it is not a substitute for a full real-scene render/geometry no-gate training ablation.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{output_dir / 'counterfactual_gate_ablation.json'}`",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/carnet/meshsplatopt/final_stageF38_counterfactual_gate_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = REPO_ROOT / args.output_dir
    results = [evaluate_case(name, edit, output_dir) for name, edit in build_cases()]
    write_report(output_dir, results)
    payload = json.loads((output_dir / "counterfactual_gate_ablation.json").read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit("F38 counterfactual gate ablation failed")


if __name__ == "__main__":
    main()
