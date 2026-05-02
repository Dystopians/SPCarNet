"""Gate parking cluster proposal metadata into a mesh-extraction action plan."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_terms(proposal: dict[str, Any]) -> dict[str, float]:
    scores = proposal.get("metadata", {}).get("scores", {})
    return {
        "support_score": float(scores.get("support_score", proposal.get("confidence", 0.0))),
        "shape_score": float(scores.get("shape_score", 0.0)),
        "ground_safe_score": float(scores.get("ground_safe_score", 0.0)),
        "uncertainty": float(scores.get("uncertainty", 1.0 - float(proposal.get("confidence", 0.0)))),
    }


def _cluster_terms(proposal: dict[str, Any]) -> dict[str, float]:
    cluster = proposal.get("metadata", {}).get("cluster", {})
    return {
        "view_count": float(cluster.get("view_count", 0.0)),
        "sparse_point_count_sum": float(cluster.get("sparse_point_count_sum", 0.0)),
        "mean_ground_overlap": float(cluster.get("mean_ground_overlap", 1.0)),
    }


def gate_one(
    proposal: dict[str, Any],
    *,
    min_candidate_support: float,
    min_candidate_views: int,
    max_candidate_uncertainty: float,
    uncertainty_action_threshold: float,
) -> dict[str, Any]:
    ptype = str(proposal.get("proposal_type", "unknown"))
    terms = _score_terms(proposal)
    cluster = _cluster_terms(proposal)
    support = terms["support_score"]
    uncertainty = terms["uncertainty"]
    views = cluster["view_count"]
    score = float(proposal.get("confidence", proposal.get("score_mean", 0.0)))
    metadata_only = bool(proposal.get("metadata", {}).get("metadata_only", False))

    reasons: list[str] = []
    decision = "rejected"
    action = "none"

    if not metadata_only:
        reasons.append("not_metadata_only")
    if views < min_candidate_views:
        reasons.append("insufficient_multiview_support")
    if support < min_candidate_support:
        reasons.append("support_below_candidate_threshold")
    if uncertainty > max_candidate_uncertainty:
        reasons.append("uncertainty_above_candidate_threshold")

    candidate_ready = (
        metadata_only
        and views >= min_candidate_views
        and support >= min_candidate_support
        and uncertainty <= max_candidate_uncertainty
    )

    if ptype == "prune":
        decision = "deferred"
        action = "wait_for_scene_mesh_evidence"
        reasons.append("prune_requires_explicit_scene_mesh_evidence")
    elif ptype == "uncertainty":
        if uncertainty >= uncertainty_action_threshold:
            decision = "diagnostic"
            action = "inspect_uncertain_region"
            reasons.append("high_uncertainty_region")
        else:
            decision = "deferred"
            action = "no_uncertainty_action"
            reasons.append("uncertainty_below_action_threshold")
    elif ptype in {"protect", "snap_candidate", "fill_candidate"} and candidate_ready:
        decision = "candidate_extract"
        action = "extract_local_mesh_patch"
        reasons.append("metadata_candidate_passed")
        if ptype == "protect":
            reasons.append("protect_is_safe_first_action")
        elif ptype == "snap_candidate":
            reasons.append("snap_still_requires_mesh_and_render_gate")
        elif ptype == "fill_candidate":
            reasons.append("fill_still_requires_mesh_and_render_gate")

    return {
        "proposal_id": str(proposal.get("proposal_id", "")),
        "proposal_type": ptype,
        "region_id": str(proposal.get("region_id", "")),
        "decision": decision,
        "action": action,
        "score": score,
        "support_score": support,
        "uncertainty": uncertainty,
        "view_count": int(views),
        "sparse_point_count_sum": int(cluster["sparse_point_count_sum"]),
        "metadata_only": metadata_only,
        "requires_mesh_extraction": bool(proposal.get("metadata", {}).get("requires_mesh_extraction", True)),
        "requires_scene_gate": bool(proposal.get("metadata", {}).get("requires_scene_gate", True)),
        "reasons": reasons,
    }


def _group_targets(results: list[dict[str, Any]], decision: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in results:
        if row["decision"] != decision:
            continue
        target = grouped.setdefault(
            row["region_id"],
            {
                "region_id": row["region_id"],
                "proposal_types": [],
                "max_score": 0.0,
                "max_support_score": 0.0,
                "min_uncertainty": 1.0,
                "view_count": row["view_count"],
                "sparse_point_count_sum": row["sparse_point_count_sum"],
                "next_action": row["action"],
            },
        )
        target["proposal_types"].append(row["proposal_type"])
        target["max_score"] = max(float(target["max_score"]), float(row["score"]))
        target["max_support_score"] = max(float(target["max_support_score"]), float(row["support_score"]))
        target["min_uncertainty"] = min(float(target["min_uncertainty"]), float(row["uncertainty"]))
    return sorted(grouped.values(), key=lambda x: (-float(x["max_support_score"]), -int(x["view_count"]), x["region_id"]))


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load(Path(args.proposals))
    proposals = list(payload.get("proposals", []))
    results = [
        gate_one(
            proposal,
            min_candidate_support=args.min_candidate_support,
            min_candidate_views=args.min_candidate_views,
            max_candidate_uncertainty=args.max_candidate_uncertainty,
            uncertainty_action_threshold=args.uncertainty_action_threshold,
        )
        for proposal in proposals
    ]
    counts = defaultdict(int)
    for row in results:
        counts[row["decision"]] += 1

    action_plan = {
        "source_proposals": str(args.proposals),
        "metadata_gate": True,
        "geometry_edited": False,
        "thresholds": {
            "min_candidate_support": args.min_candidate_support,
            "min_candidate_views": args.min_candidate_views,
            "max_candidate_uncertainty": args.max_candidate_uncertainty,
            "uncertainty_action_threshold": args.uncertainty_action_threshold,
        },
        "mesh_extraction_targets": _group_targets(results, "candidate_extract"),
        "diagnostic_targets": _group_targets(results, "diagnostic"),
        "decision_counts": dict(sorted(counts.items())),
        "results": results,
        "notes": [
            "This is a metadata gate, not the M9 before/after mesh scene gate.",
            "Candidate_extract means the region is ready for local mesh patch extraction, not geometry application.",
            "Prune proposals are deferred until explicit scene mesh evidence exists.",
        ],
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata_gate_report.json").write_text(json.dumps(action_plan, indent=2) + "\n", encoding="utf-8")
    (out / "action_plan.json").write_text(
        json.dumps(
            {
                "mesh_extraction_targets": action_plan["mesh_extraction_targets"],
                "diagnostic_targets": action_plan["diagnostic_targets"],
                "decision_counts": action_plan["decision_counts"],
                "geometry_edited": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (out / "metadata_gate_results.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "proposal_id",
            "proposal_type",
            "region_id",
            "decision",
            "action",
            "score",
            "support_score",
            "uncertainty",
            "view_count",
            "sparse_point_count_sum",
            "metadata_only",
            "requires_mesh_extraction",
            "requires_scene_gate",
            "reasons",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({**row, "reasons": ";".join(row["reasons"])})
    with (out / "metadata_gate_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Parking Metadata Gate Report\n\n")
        f.write("- geometry edited: `false`\n")
        f.write(f"- proposals evaluated: `{len(results)}`\n")
        for key, value in sorted(counts.items()):
            f.write(f"- {key}: `{value}`\n")
        f.write(f"- mesh extraction targets: `{len(action_plan['mesh_extraction_targets'])}`\n")
        f.write(f"- diagnostic targets: `{len(action_plan['diagnostic_targets'])}`\n")
    return action_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate parking metadata proposals into a mesh-extraction action plan.")
    parser.add_argument("--proposals", default="outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals/proposals.json")
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/parking_phone_tiny/metadata_gate")
    parser.add_argument("--min_candidate_support", type=float, default=0.80)
    parser.add_argument("--min_candidate_views", type=int, default=4)
    parser.add_argument("--max_candidate_uncertainty", type=float, default=0.25)
    parser.add_argument("--uncertainty_action_threshold", type=float, default=0.25)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "decision_counts": result["decision_counts"],
                "mesh_extraction_targets": len(result["mesh_extraction_targets"]),
                "diagnostic_targets": len(result["diagnostic_targets"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
