from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .counterfactual_edit_gate import validate_edit_counterfactual
from .edit_portfolio import PortfolioItem, rank_portfolio
from .edit_types import MeshState


STATES = [
    "GEOMETRY_ACQUISITION",
    "DEFECT_MINING",
    "LOW_RISK_CLEANUP",
    "SNAP_REPAIR",
    "GIANT_VOID_REPAIR",
    "OBJECT_PRIOR_REPAIR",
    "APPEARANCE_RECOVERY",
    "TOPOLOGY_RETENTION",
    "VALIDATION_ROLLBACK",
    "FINAL_AUDIT",
]


@dataclass
class RepairStateMachineResult:
    accepted_edits: list[dict[str, Any]] = field(default_factory=list)
    rejected_edits: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    final_audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_repair_state_machine(
    state: MeshState,
    portfolio: list[PortfolioItem],
    output_dir: str | Path,
    *,
    allow_prior_only: bool = False,
) -> RepairStateMachineResult:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = RepairStateMachineResult()
    ranked = rank_portfolio(portfolio)
    result.trace.append({"state": "GEOMETRY_ACQUISITION", "vertices": len(state.vertices), "faces": len(state.faces)})
    result.trace.append({"state": "DEFECT_MINING", "candidate_count": len(ranked)})
    buckets = {
        "LOW_RISK_CLEANUP": {"DELETE_TRIANGLES", "EDGE_COLLAPSE", "FACE_MERGE"},
        "SNAP_REPAIR": {"SNAP_VERTICES"},
        "GIANT_VOID_REPAIR": {"FILL_PATCH", "SPLIT_TRIANGLES"},
        "APPEARANCE_RECOVERY": {"APPEARANCE_RESET"},
    }
    for state_name in STATES[2:9]:
        result.trace.append({"state": state_name, "entered": True})
        allowed = buckets.get(state_name, set())
        for item in ranked:
            if item.edit.edit_type not in allowed:
                continue
            if item.prior_only_flag and not allow_prior_only:
                result.rejected_edits.append({"edit": item.edit.to_dict(), "reason": "prior_only_rejected_by_state_machine"})
                continue
            report = validate_edit_counterfactual(
                state,
                item.edit,
                snapshot_path=out / "snapshots" / f"{item.edit.edit_id}.npz",
                commit_on_accept=True,
            )
            if report.accepted:
                result.accepted_edits.append({"edit": item.edit.to_dict(), "gate_report": report.to_dict(), "portfolio_score": item.score()})
            else:
                result.rejected_edits.append({"edit": item.edit.to_dict(), "gate_report": report.to_dict(), "portfolio_score": item.score()})
    result.trace.append({"state": "FINAL_AUDIT", "accepted": len(result.accepted_edits), "rejected": len(result.rejected_edits)})
    result.final_audit = {
        "final_vertices": len(state.vertices),
        "final_faces": len(state.faces),
        "accepted_count": len(result.accepted_edits),
        "rejected_count": len(result.rejected_edits),
    }
    write_state_machine_outputs(result, out, ranked)
    return result


def write_state_machine_outputs(result: RepairStateMachineResult, output_dir: Path, ranked: list[PortfolioItem]) -> None:
    (output_dir / "edit_portfolio.json").write_text(json.dumps([x.to_dict() for x in ranked], indent=2), encoding="utf-8")
    (output_dir / "state_machine_trace.json").write_text(json.dumps(result.trace, indent=2), encoding="utf-8")
    (output_dir / "accepted_edits.json").write_text(json.dumps(result.accepted_edits, indent=2), encoding="utf-8")
    (output_dir / "rejected_edits.json").write_text(json.dumps(result.rejected_edits, indent=2), encoding="utf-8")
    (output_dir / "final_audit.json").write_text(json.dumps(result.final_audit, indent=2), encoding="utf-8")
    lines = ["# Repair Summary", "", f"- accepted edits: `{len(result.accepted_edits)}`", f"- rejected edits: `{len(result.rejected_edits)}`"]
    (output_dir / "repair_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
