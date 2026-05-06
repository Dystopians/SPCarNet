from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


ACTION_TYPES = {
    "ROLLBACK_ONLY",
    "APPEARANCE_ONLY",
    "SNAP_LOCAL",
    "SPLIT_ALLOCATE",
    "FILL_PATCH_LOCAL",
    "DELETE_OR_COLLAPSE",
    "REJECT_UNOBSERVED",
}


@dataclass(frozen=True)
class CertificatePlannerConfig:
    high_conflict_pressure: float = 5.0
    high_render_debt: float = 1.0
    high_redundancy: float = 1.0
    min_surface_support: float = 0.25
    max_prior_only_risk: float = 0.5


def _f(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value


def _i(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except Exception:
        return default


def choose_certificate_action(row: Mapping[str, Any], cfg: CertificatePlannerConfig) -> tuple[str, list[str], str]:
    pressure = _f(row, "certificate_pressure")
    render_gain = _f(row, "render_gain")
    render_debt = _f(row, "render_debt")
    support = _f(row, "surface_support", _f(row, "positive_surface_evidence", 0.0))
    redundancy = _f(row, "redundancy")
    prior_only = _f(row, "prior_only_risk")
    gate_critical = _i(row, "gate_critical_count")
    invalid = _i(row, "candidate_invalid_count")
    bad_tradeoff = _i(row, "bad_tradeoff_count")
    hole_score = _f(row, "hole_score")
    snap_score = _f(row, "snap_score")

    certificates = ["depth_nonregression", "absrel_nonregression", "topology_audit"]
    if prior_only > float(cfg.max_prior_only_risk) and support < float(cfg.min_surface_support):
        return "REJECT_UNOBSERVED", ["no_prior_only_hallucination"], "weak evidence and high prior-only risk"
    if pressure > 0.0 and (gate_critical > 0 or bad_tradeoff > 0 or invalid > 0):
        if snap_score >= pressure and support >= float(cfg.min_surface_support):
            return "SNAP_LOCAL", certificates + ["normal_consistency"], "supported sparse-depth conflict with local snap evidence"
        return "ROLLBACK_ONLY", certificates, "sparse-depth certificate is violated"
    if render_debt >= float(cfg.high_render_debt) and support >= float(cfg.min_surface_support):
        if hole_score > 0.0:
            return "FILL_PATCH_LOCAL", ["boundary_support", "depth_nonregression", "changed_pixel_safety"], "supported render hole"
        return "SPLIT_ALLOCATE", ["surface_support", "render_improvement", "depth_nonregression"], "supported render debt needs capacity"
    if redundancy >= float(cfg.high_redundancy) and support < float(cfg.min_surface_support) and render_debt <= 0.0:
        return "DELETE_OR_COLLAPSE", ["low_positive_evidence", "changed_pixel_safety", "depth_nonregression"], "redundant low-evidence cluster"
    if render_gain > 0.0 or render_debt > 0.0:
        return "APPEARANCE_ONLY", ["geometry_frozen", "render_improvement"], "appearance can improve without geometry edit"
    return "REJECT_UNOBSERVED", ["insufficient_evidence"], "no certified benefit"


def plan_certificate_edits(ecg: Mapping[str, Any], *, cfg: CertificatePlannerConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CertificatePlannerConfig()
    rows = list(ecg.get("cluster_summary", []))
    plans = []
    for rank, row in enumerate(rows):
        action, certs, reason = choose_certificate_action(row, cfg)
        plans.append(
            {
                "rank": rank,
                "action": action,
                "target_cluster_ids": [int(row.get("cluster_id", -1))],
                "target_sparse_point_ids": [],
                "required_certificates": certs,
                "expected_risk": "high" if action in {"SNAP_LOCAL", "SPLIT_ALLOCATE", "FILL_PATCH_LOCAL", "DELETE_OR_COLLAPSE"} else "low",
                "expected_benefit": float(row.get("certificate_pressure", 0.0)) + float(row.get("render_debt", 0.0)),
                "headline_allowed": action not in {"REJECT_UNOBSERVED"},
                "touches_topology": action in {"SPLIT_ALLOCATE", "FILL_PATCH_LOCAL", "DELETE_OR_COLLAPSE"},
                "recommended_recovery_flags": {
                    "freeze_topology_updates": action not in {"SPLIT_ALLOCATE", "FILL_PATCH_LOCAL", "DELETE_OR_COLLAPSE"},
                    "enable_sparse_depth_parent_rollback_loss": action in {"ROLLBACK_ONLY", "SNAP_LOCAL", "SPLIT_ALLOCATE", "FILL_PATCH_LOCAL"},
                },
                "reason": reason,
            }
        )
    return {"config": asdict(cfg), "plans": plans, "action_types": sorted({p["action"] for p in plans})}


def write_certificate_edit_plan(plan: Mapping[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "certificate_edit_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(plan.get("plans", []))
    if rows:
        flat = []
        for row in rows:
            flat.append(
                {
                    "rank": row["rank"],
                    "action": row["action"],
                    "target_cluster_ids": ",".join(str(x) for x in row["target_cluster_ids"]),
                    "required_certificates": ",".join(row["required_certificates"]),
                    "expected_risk": row["expected_risk"],
                    "expected_benefit": f"{float(row['expected_benefit']):.9g}",
                    "headline_allowed": int(bool(row["headline_allowed"])),
                    "touches_topology": int(bool(row["touches_topology"])),
                    "reason": row["reason"],
                }
            )
        with (out / "certificate_edit_plan.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)
    report = ["# Certificate Edit Plan Report", "", f"- planned edits: `{len(rows)}`", f"- action types: `{', '.join(plan.get('action_types', []))}`", ""]
    report.append("| rank | action | clusters | risk | topology |")
    report.append("|---:|---|---|---|---:|")
    for row in rows[:20]:
        report.append(
            f"| {row['rank']} | {row['action']} | {','.join(str(x) for x in row['target_cluster_ids'])} | {row['expected_risk']} | {int(bool(row['touches_topology']))} |"
        )
    (out / "certificate_edit_plan_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

