from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SCELocalSurgeryConfig:
    min_surface_support: float = 0.25
    max_free_space_risk: float = 0.25
    min_depth_conflict: float = 1.0
    min_render_debt: float = 1.0


def propose_sce_local_surgery(clusters: list[Mapping[str, Any]], cfg: SCELocalSurgeryConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or SCELocalSurgeryConfig()
    proposals: list[dict[str, Any]] = []
    for rank, row in enumerate(clusters):
        cluster_id = int(row.get("cluster_id", rank))
        depth_conflict = float(row.get("depth_conflict", row.get("certificate_pressure", 0.0)))
        render_debt = float(row.get("render_debt", 0.0))
        support = float(row.get("surface_support", row.get("positive_surface_evidence", 0.0)))
        free_space = float(row.get("free_space_risk", 0.0))
        prior_only = bool(row.get("prior_only_flag", False))
        hole = float(row.get("hole_score", 0.0))
        variation = float(row.get("in_triangle_depth_variation", 0.0))
        appearance = float(row.get("appearance_ghost_score", 0.0))

        action = "PROTECT"
        reason = "parent-good sparse evidence should be protected"
        topology_cost = 0
        accepted = True
        if prior_only or support < cfg.min_surface_support or free_space > cfg.max_free_space_risk:
            action = "REJECT"
            reason = "insufficient support or high free-space risk"
            accepted = False
        elif appearance > 0.0 and depth_conflict <= 0.0:
            action = "APPEARANCE_RESET"
            reason = "radiance ghost with geometry support"
        elif hole > 0.0 and render_debt >= cfg.min_render_debt:
            action = "FILL_PATCH"
            reason = "supported boundary or void debt"
            topology_cost = 1
        elif variation >= cfg.min_depth_conflict and render_debt >= cfg.min_render_debt:
            action = "SPLIT_TRIANGLES"
            reason = "large triangle under-resolves sparse depth variation"
            topology_cost = 1
        elif depth_conflict >= cfg.min_depth_conflict:
            action = "SNAP_VERTICES"
            reason = "supported local sparse-depth mismatch"

        proposals.append(
            {
                "proposal_id": f"sce_local_{rank:04d}",
                "cluster_id": cluster_id,
                "action": action,
                "accepted": bool(accepted),
                "reason": reason,
                "linked_sentinel_ids": list(row.get("linked_sentinel_ids", [])),
                "linked_csef_region_ids": list(row.get("linked_csef_region_ids", [])),
                "evidence_summary": {
                    "depth_conflict": depth_conflict,
                    "render_debt": render_debt,
                    "surface_support": support,
                    "hole_score": hole,
                    "in_triangle_depth_variation": variation,
                    "appearance_ghost_score": appearance,
                },
                "free_space_risk": free_space,
                "prior_only_flag": prior_only,
                "expected_topology_cost": topology_cost,
                "rollback_snapshot": f"rollback_cluster_{cluster_id}",
            }
        )
    return proposals


def apply_synthetic_sce_local_surgery(proposal: Mapping[str, Any], sentinel_error: float) -> dict[str, Any]:
    action = str(proposal.get("action", ""))
    if not bool(proposal.get("accepted", False)):
        return {"applied": False, "rolled_back": False, "sentinel_error_before": sentinel_error, "sentinel_error_after": sentinel_error}
    factors = {
        "SNAP_VERTICES": 0.35,
        "SPLIT_TRIANGLES": 0.45,
        "FILL_PATCH": 0.50,
        "APPEARANCE_RESET": 0.90,
        "PROTECT": 1.00,
    }
    after = float(sentinel_error) * float(factors.get(action, 1.0))
    harmful = after > float(sentinel_error)
    return {"applied": True, "rolled_back": harmful, "sentinel_error_before": float(sentinel_error), "sentinel_error_after": float(sentinel_error if harmful else after)}


def write_sce_local_surgery_outputs(proposals: list[dict[str, Any]], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sce_local_surgery_proposals.json").write_text(json.dumps({"proposals": proposals}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if proposals:
        with (out / "sce_local_surgery_proposals.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["proposal_id", "cluster_id", "action", "accepted", "reason", "free_space_risk", "prior_only_flag", "expected_topology_cost"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in proposals:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
    report = ["# SCE Local Surgery Report", "", f"- proposals: `{len(proposals)}`", ""]
    for row in proposals[:20]:
        report.append(f"- `{row['proposal_id']}` cluster `{row['cluster_id']}` action `{row['action']}` accepted `{row['accepted']}`")
    (out / "sce_local_surgery_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

