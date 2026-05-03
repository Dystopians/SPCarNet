from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .edit_types import MeshEdit, MeshSplatOptEditType, MeshState
from .ground_void_fill import make_ground_plane_void_fill
from .snap_proposals import make_snap_proposals


@dataclass(frozen=True)
class ObjectRepairProposal:
    proposal_id: str
    proposal_type: str
    edit: MeshEdit | None
    confidence: float
    uncertainty: float
    metadata: dict[str, Any]
    rejected_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["edit"] = self.edit.to_dict() if self.edit is not None else None
        return data


def make_object_prior_repair_proposals(
    state: MeshState,
    *,
    region_id: str = "vehicle_region",
    canonicalization_confidence: float,
    posterior_uncertainty: float,
    missing_panel_bbox: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> list[ObjectRepairProposal]:
    proposals: list[ObjectRepairProposal] = []
    base_meta = {
        "region_id": region_id,
        "prior_proposes_evidence_disposes": True,
        "requires_scene_counterfactual_validation": True,
        "canonicalization_confidence": float(canonicalization_confidence),
        "posterior_uncertainty": float(posterior_uncertainty),
    }
    safe_conf = float(np.clip(canonicalization_confidence * (1.0 - posterior_uncertainty), 0.0, 1.0))
    if canonicalization_confidence < 0.35 or posterior_uncertainty > 0.75:
        edit = MeshEdit(
            edit_id=f"{region_id}_protect_uncertain_edit",
            edit_type=MeshSplatOptEditType.PROTECT.value,
            defect_id=region_id,
            affected_faces=list(range(len(state.faces))),
            evidence_summary=base_meta,
            risk_summary={"aggressive_object_prior_disabled": True},
        )
        proposals.append(
            ObjectRepairProposal(
                proposal_id=f"{region_id}_protect_uncertain",
                proposal_type="vehicle_protect_mask",
                edit=edit,
                confidence=max(0.1, canonicalization_confidence),
                uncertainty=posterior_uncertainty,
                metadata={**base_meta, "aggressive_proposals_disabled": True},
            )
        )
        return proposals

    protect_edit = MeshEdit(
        edit_id=f"{region_id}_protect_edit",
        edit_type=MeshSplatOptEditType.PROTECT.value,
        defect_id=region_id,
        affected_faces=list(range(len(state.faces))),
        evidence_summary=base_meta,
    )
    proposals.append(
        ObjectRepairProposal(
            proposal_id=f"{region_id}_protect",
            proposal_type="vehicle_protect_mask",
            edit=protect_edit,
            confidence=safe_conf,
            uncertainty=posterior_uncertainty,
            metadata=base_meta,
        )
    )

    snap_candidates = make_snap_proposals(
        state,
        candidate_vertices=[int(np.argmax(np.abs(state.vertices[:, 2] - np.median(state.vertices[:, 2]))))],
        supported_vertices=set(range(len(state.vertices))),
        evidence_source="object_prior_surface_target",
    )
    for i, snap in enumerate([p for p in snap_candidates if not p.rejected_reason][:1]):
        proposals.append(
            ObjectRepairProposal(
                proposal_id=f"{region_id}_surface_snap_{i:04d}",
                proposal_type="vehicle_surface_snap_candidate",
                edit=snap.edit,
                confidence=safe_conf,
                uncertainty=max(posterior_uncertainty, snap.uncertainty),
                metadata={**base_meta, "snap_target": "object_prior_surface"},
            )
        )

    if missing_panel_bbox is not None and canonicalization_confidence >= 0.65 and posterior_uncertainty <= 0.45:
        fill = make_ground_plane_void_fill(
            state,
            bbox_min=missing_panel_bbox[0],
            bbox_max=missing_panel_bbox[1],
            z=float(np.median(state.vertices[:, 2])),
            grid_resolution=1,
            proposal_id=f"{region_id}_panel_fill",
            observed_support=True,
        )
        if fill.edit is not None:
            proposals.append(
                ObjectRepairProposal(
                    proposal_id=f"{region_id}_discontinuity_fill",
                    proposal_type="vehicle_discontinuity_fill_candidate",
                    edit=fill.edit,
                    confidence=safe_conf,
                    uncertainty=posterior_uncertainty,
                    metadata={**base_meta, "fill_target": "object_prior_missing_panel", "scene_gate_required": True},
                )
            )

    return proposals


def write_object_repair_outputs(proposals: list[ObjectRepairProposal], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "object_repair_proposals.json").write_text(json.dumps([p.to_dict() for p in proposals], indent=2), encoding="utf-8")
    with (out / "object_repair_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["proposal_id", "proposal_type", "confidence", "uncertainty", "has_edit", "rejected_reason"])
        for p in proposals:
            writer.writerow([p.proposal_id, p.proposal_type, p.confidence, p.uncertainty, p.edit is not None, p.rejected_reason])
    lines = ["# Object Prior Repair Proposal Report", "", f"- proposals: `{len(proposals)}`", ""]
    for p in proposals:
        lines.append(f"- `{p.proposal_id}` `{p.proposal_type}` confidence `{p.confidence:.3f}`")
    (out / "object_repair_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
