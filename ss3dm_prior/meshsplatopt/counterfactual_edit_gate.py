from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .edit_apply import apply_edit, summarize_topology_delta, verify_mesh_integrity
from .edit_snapshot import create_snapshot, rollback_edit
from .edit_types import MeshEdit, MeshSplatOptEditType, MeshState


@dataclass(frozen=True)
class CounterfactualGateReport:
    edit_id: str
    edit_type: str
    accepted: bool
    reasons: list[str]
    metrics: dict[str, Any]
    topology_delta: dict[str, int]
    rollback_performed: bool
    snapshot_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_edit_counterfactual(
    state: MeshState,
    edit: MeshEdit,
    *,
    snapshot_path: str | Path,
    commit_on_accept: bool = True,
    free_space_threshold: float = 0.35,
) -> CounterfactualGateReport:
    before = state.copy()
    create_snapshot(state, snapshot_path)
    reasons: list[str] = []
    rollback_performed = False
    metrics: dict[str, Any] = {
        "render_metrics_available": False,
        "sparse_geometry_metrics_available": False,
        "changed_pixel_metrics_available": False,
    }
    try:
        apply_edit(state, edit)
        integrity = verify_mesh_integrity(state)
        metrics["topology_valid"] = integrity["valid"]
        metrics["topology_errors"] = integrity["errors"]
        if not integrity["valid"]:
            reasons.append("topology_integrity_failed")
    except Exception as exc:
        metrics["topology_valid"] = False
        metrics["apply_error"] = str(exc)
        reasons.append("edit_apply_failed")
        rollback_edit(state, snapshot_path)
        return CounterfactualGateReport(
            edit_id=edit.edit_id,
            edit_type=edit.edit_type,
            accepted=False,
            reasons=reasons,
            metrics=metrics,
            topology_delta=summarize_topology_delta(before, state),
            rollback_performed=True,
            snapshot_path=str(snapshot_path),
        )

    risk = dict(edit.risk_summary)
    evidence = dict(edit.evidence_summary)
    free_space_risk = float(risk.get("free_space_risk", evidence.get("free_space_risk", 0.0)))
    metrics["free_space_risk"] = free_space_risk
    metrics["csef_debt_reduction"] = float(evidence.get("csef_debt_reduction", 0.0))
    prior_only = bool(risk.get("prior_only_flag", evidence.get("prior_only_flag", False)))
    metrics["prior_only_flag"] = prior_only

    edit_type = MeshSplatOptEditType(edit.edit_type)
    if free_space_risk > free_space_threshold:
        reasons.append("free_space_gate_failed")
    if prior_only and not risk.get("diagnostic_mode", False):
        reasons.append("prior_only_not_allowed_for_commit")
    if edit_type == MeshSplatOptEditType.DELETE_TRIANGLES and risk.get("deletes_supported_surface", False):
        reasons.append("delete_supported_surface_rejected")
    if edit_type == MeshSplatOptEditType.SNAP_VERTICES and risk.get("snap_through_free_space", False):
        reasons.append("snap_free_space_rejected")
    if edit_type == MeshSplatOptEditType.FILL_PATCH and evidence.get("boundary_loop_support", True) is False:
        reasons.append("fill_boundary_certificate_failed")

    accepted = not reasons
    delta = summarize_topology_delta(before, state)
    if not accepted or not commit_on_accept:
        rollback_edit(state, snapshot_path)
        rollback_performed = True
    return CounterfactualGateReport(
        edit_id=edit.edit_id,
        edit_type=edit.edit_type,
        accepted=accepted,
        reasons=reasons,
        metrics=metrics,
        topology_delta=delta,
        rollback_performed=rollback_performed,
        snapshot_path=str(snapshot_path),
    )


def write_counterfactual_report(report: CounterfactualGateReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
