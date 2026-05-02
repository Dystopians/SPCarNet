"""Score consolidated parking clusters into MeshPrior proposal metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extent_score(extent: list[float]) -> float:
    dims = sorted(max(float(x), 1e-6) for x in extent)
    elongation = dims[2] / dims[1]
    flatness = dims[0] / dims[1]
    score = 0.0
    if 1.0 <= elongation <= 5.5:
        score += 0.55
    if 0.05 <= flatness <= 1.5:
        score += 0.45
    return score


def _proposal_scores(cluster: dict[str, Any]) -> dict[str, float]:
    confidence = float(cluster.get("confidence", 0.0))
    view_score = min(1.0, float(cluster.get("view_count", 0)) / 6.0)
    sparse_score = min(1.0, float(cluster.get("sparse_point_count_sum", 0)) / 512.0)
    ground_overlap = float(cluster.get("mean_ground_overlap", 1.0))
    ground_safe = max(0.0, 1.0 - ground_overlap)
    shape_score = _extent_score(cluster.get("bbox3d_extent", [0.0, 0.0, 0.0]))
    support = 0.35 * confidence + 0.25 * view_score + 0.25 * sparse_score + 0.15 * ground_safe
    uncertainty = max(0.0, 1.0 - support)
    protect = max(0.0, min(1.0, 0.65 * support + 0.35 * shape_score))
    prune = max(0.0, min(1.0, (1.0 - support) * (0.5 + ground_overlap)))
    snap = max(0.0, min(1.0, 0.55 * support + 0.45 * shape_score))
    fill = max(0.0, min(1.0, 0.45 * support + 0.35 * shape_score + 0.20 * min(1.0, view_score)))
    return {
        "support_score": support,
        "shape_score": shape_score,
        "ground_safe_score": ground_safe,
        "uncertainty": uncertainty,
        "protect": protect,
        "prune": prune,
        "snap_candidate": snap,
        "fill_candidate": fill,
    }


def _proposal(cluster: dict[str, Any], proposal_type: str, score: float, scores: dict[str, float]) -> dict[str, Any]:
    return {
        "proposal_id": f"{cluster['cluster_id']}_{proposal_type}",
        "proposal_type": proposal_type,
        "region_id": cluster["cluster_id"],
        "face_indices": [],
        "confidence": float(score),
        "score_mean": float(score),
        "score_max": float(score),
        "metadata": {
            "metadata_only": True,
            "requires_mesh_extraction": True,
            "requires_scene_gate": True,
            "cluster": cluster,
            "scores": scores,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = _load(Path(args.consolidated_regions))
    clusters = [c for c in source.get("clusters", []) if c.get("eligible_for_proposal", False)]
    if args.max_clusters > 0:
        clusters = clusters[: args.max_clusters]
    proposals = []
    rows = []
    for cluster in clusters:
        scores = _proposal_scores(cluster)
        for ptype in ("protect", "prune", "snap_candidate", "fill_candidate", "uncertainty"):
            score = scores["uncertainty"] if ptype == "uncertainty" else scores[ptype]
            proposals.append(_proposal(cluster, ptype, score, scores))
        rows.append(
            {
                "cluster_id": cluster["cluster_id"],
                "view_count": cluster["view_count"],
                "sparse_point_count_sum": cluster["sparse_point_count_sum"],
                "confidence": cluster["confidence"],
                **scores,
            }
        )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_consolidated_regions": str(args.consolidated_regions),
        "cluster_count": len(clusters),
        "proposal_count": len(proposals),
        "proposal_types": ["protect", "prune", "snap_candidate", "fill_candidate", "uncertainty"],
        "metadata_only": True,
        "proposals": proposals,
        "notes": [
            "Parking cluster proposals are metadata-only because no editable local scene mesh has been extracted yet.",
            "They must pass scene gates and mesh application checks before geometry can be modified.",
        ],
    }
    (out / "proposals.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (out / "proposal_scores.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["cluster_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "proposal_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Parking Cluster Proposal Report\n\n")
        f.write(f"- clusters scored: `{len(clusters)}`\n")
        f.write(f"- proposals: `{len(proposals)}`\n")
        f.write("- proposal types: `protect`, `prune`, `snap_candidate`, `fill_candidate`, `uncertainty`\n")
        f.write("- metadata_only: `true`\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score parking consolidated clusters into MeshPrior proposal metadata.")
    parser.add_argument("--consolidated_regions", default="outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions.json")
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals")
    parser.add_argument("--max_clusters", type=int, default=0)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({k: result[k] for k in ("cluster_count", "proposal_count", "proposal_types")}, indent=2))


if __name__ == "__main__":
    main()
