"""Cluster parking image ROI candidates into consolidated 3D regions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _eligible_regions(payload: dict[str, Any], *, min_sparse_points: int) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("regions", []):
        if not row.get("evidence", {}).get("eligible_for_posterior", False):
            continue
        if int(row.get("sparse_point_count", 0)) < min_sparse_points:
            continue
        center = np.asarray(row.get("centroid3d", [0.0, 0.0, 0.0]), dtype=np.float64)
        if not np.isfinite(center).all():
            continue
        rows.append(row)
    return rows


def _region_center(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(row["centroid3d"], dtype=np.float64)


def _cluster_regions(rows: list[dict[str, Any]], *, radius: float) -> list[list[int]]:
    if not rows:
        return []
    centers = np.stack([_region_center(r) for r in rows], axis=0)
    order = np.argsort([-int(r.get("sparse_point_count", 0)) for r in rows])
    assigned = np.zeros(len(rows), dtype=bool)
    clusters: list[list[int]] = []
    for idx in order:
        if assigned[idx]:
            continue
        d = np.linalg.norm(centers - centers[idx], axis=1)
        members = np.where((d <= radius) & (~assigned))[0].tolist()
        for m in members:
            assigned[m] = True
        clusters.append(members)
    return clusters


def _consolidate(rows: list[dict[str, Any]], members: list[int], cluster_id: int) -> dict[str, Any]:
    subset = [rows[i] for i in members]
    weights = np.asarray([max(1, int(r.get("sparse_point_count", 0))) for r in subset], dtype=np.float64)
    centers = np.stack([_region_center(r) for r in subset], axis=0)
    center = np.average(centers, axis=0, weights=weights)
    mins = []
    maxs = []
    for r in subset:
        mn = np.asarray(r.get("bbox3d_min", center), dtype=np.float64)
        mx = np.asarray(r.get("bbox3d_max", center), dtype=np.float64)
        if np.isfinite(mn).all() and np.isfinite(mx).all():
            mins.append(mn)
            maxs.append(mx)
    bbox_min = np.min(np.stack(mins, axis=0), axis=0) if mins else center
    bbox_max = np.max(np.stack(maxs, axis=0), axis=0) if maxs else center
    images = sorted({r["image_name"] for r in subset})
    support = int(sum(int(r.get("sparse_point_count", 0)) for r in subset))
    mean_score = float(np.mean([float(r.get("evidence", {}).get("car_likeness_score", 0.0)) for r in subset]))
    mean_ground = float(np.mean([float(r.get("ground_overlap_fraction", 0.0)) for r in subset]))
    multiview_score = min(1.0, len(images) / 3.0)
    sparse_score = min(1.0, support / 96.0)
    confidence = float(0.45 * mean_score + 0.35 * multiview_score + 0.20 * sparse_score)
    return {
        "cluster_id": f"parking_region_{cluster_id:04d}",
        "member_count": len(subset),
        "view_count": len(images),
        "image_names": images,
        "member_region_ids": [r["region_id"] for r in subset],
        "sparse_point_count_sum": support,
        "centroid3d": [float(x) for x in center.tolist()],
        "bbox3d_min": [float(x) for x in bbox_min.tolist()],
        "bbox3d_max": [float(x) for x in bbox_max.tolist()],
        "bbox3d_extent": [float(x) for x in (bbox_max - bbox_min).tolist()],
        "mean_ground_overlap": mean_ground,
        "mean_car_likeness_score": mean_score,
        "confidence": confidence,
        "eligible_for_proposal": bool(confidence >= 0.65 and len(images) >= 1),
        "notes": ["clustered from image ROI candidates; editable mesh geometry is not inferred here"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load(Path(args.image_regions))
    rows = _eligible_regions(payload, min_sparse_points=args.min_sparse_points)
    clusters = _cluster_regions(rows, radius=args.cluster_radius)
    consolidated = [_consolidate(rows, members, i) for i, members in enumerate(clusters)]
    consolidated.sort(key=lambda r: (not r["eligible_for_proposal"], -r["view_count"], -r["sparse_point_count_sum"], r["cluster_id"]))
    if args.max_clusters > 0:
        consolidated = consolidated[: args.max_clusters]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "source_image_regions": str(args.image_regions),
        "cluster_radius": float(args.cluster_radius),
        "min_sparse_points": int(args.min_sparse_points),
        "input_region_count": int(payload.get("region_count", len(payload.get("regions", [])))),
        "eligible_input_count": len(rows),
        "cluster_count": len(consolidated),
        "eligible_cluster_count": sum(1 for c in consolidated if c["eligible_for_proposal"]),
        "clusters": consolidated,
        "notes": [
            "Clusters consolidate repeated image ROI detections into coarse 3D vehicle-region candidates.",
            "They are not directly editable meshes; proposal generation must still pass scene gates.",
        ],
    }
    (out / "consolidated_regions.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (out / "consolidated_regions_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "member_count", "view_count", "sparse_point_count_sum", "confidence", "eligible", "centroid_x", "centroid_y", "centroid_z"])
        for c in consolidated:
            writer.writerow([c["cluster_id"], c["member_count"], c["view_count"], c["sparse_point_count_sum"], f"{c['confidence']:.6f}", int(c["eligible_for_proposal"]), *[f"{x:.9g}" for x in c["centroid3d"]]])
    with (out / "consolidation_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Parking Region Consolidation Report\n\n")
        f.write(f"- input regions: `{result['input_region_count']}`\n")
        f.write(f"- eligible inputs used: `{result['eligible_input_count']}`\n")
        f.write(f"- clusters: `{result['cluster_count']}`\n")
        f.write(f"- eligible clusters: `{result['eligible_cluster_count']}`\n")
        f.write(f"- cluster radius: `{result['cluster_radius']}`\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster parking image ROI candidates into 3D regions.")
    parser.add_argument("--image_regions", default="outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/image_regions.json")
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/parking_phone_tiny/region_consolidation")
    parser.add_argument("--cluster_radius", type=float, default=0.35)
    parser.add_argument("--min_sparse_points", type=int, default=8)
    parser.add_argument("--max_clusters", type=int, default=0)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({k: result[k] for k in ("eligible_input_count", "cluster_count", "eligible_cluster_count")}, indent=2))


if __name__ == "__main__":
    main()
