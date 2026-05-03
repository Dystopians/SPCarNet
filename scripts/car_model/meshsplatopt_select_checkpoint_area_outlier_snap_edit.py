#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_adapter import load_checkpoint_state
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshSplatOptEditType
from scripts.car_model.meshsplatopt_select_checkpoint_area_outlier_edit import triangle_areas_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_area_ratio_to_median", type=float, default=1000.0)
    parser.add_argument("--min_percentile", type=float, default=99.9)
    parser.add_argument("--shrink_factor", type=float, default=0.25)
    parser.add_argument("--chunk_size", type=int, default=250000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < float(args.shrink_factor) < 1.0:
        raise SystemExit("--shrink_factor must be between 0 and 1")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = load_checkpoint_state(args.checkpoint_path)
    areas = triangle_areas_from_checkpoint(payload, chunk_size=args.chunk_size)
    faces = payload["_triangle_indices"].detach().cpu().to(dtype=torch.long)
    vertices = payload["triangles_points"].detach().cpu().to(dtype=torch.float32)
    triangle_count = int(faces.shape[0])
    vertex_count = int(vertices.shape[0])
    median = float(np.median(areas)) if len(areas) else 0.0
    percentile_threshold = float(np.percentile(areas, args.min_percentile)) if len(areas) else 0.0
    ratio_threshold = median * float(args.min_area_ratio_to_median)
    threshold = max(percentile_threshold, ratio_threshold)
    candidate_ids = np.where(areas >= threshold)[0]
    status = "PASS" if len(candidate_ids) else "NO_CANDIDATE"
    selected_face = int(sorted((int(i) for i in candidate_ids), key=lambda i: float(areas[i]), reverse=True)[0]) if len(candidate_ids) else -1
    edit = None
    target_positions: dict[str, list[float]] = {}
    selected_vertices: list[int] = []
    before_area = float(areas[selected_face]) if selected_face >= 0 else 0.0
    after_area = 0.0
    max_displacement = 0.0
    if selected_face >= 0:
        face = faces[selected_face]
        tri = vertices[face]
        centroid = tri.mean(dim=0)
        snapped = centroid + float(args.shrink_factor) * (tri - centroid)
        cross = torch.cross(snapped[1] - snapped[0], snapped[2] - snapped[0], dim=0)
        after_area = float(0.5 * torch.linalg.norm(cross).item())
        displacements = torch.linalg.norm(snapped - tri, dim=1)
        max_displacement = float(displacements.max().item())
        selected_vertices = [int(x) for x in face.tolist()]
        for vid, pos in zip(selected_vertices, snapped.tolist()):
            target_positions[str(int(vid))] = [float(x) for x in pos]
        edit = MeshEdit(
            edit_id="real_checkpoint_area_outlier_snap_shrink",
            edit_type=MeshSplatOptEditType.SNAP_VERTICES.value,
            defect_id="checkpoint_area_outlier_geometry",
            affected_faces=[selected_face],
            affected_vertices=selected_vertices,
            attribute_changes={
                "target_positions": target_positions,
                "selector": "checkpoint_area_outlier_snap_shrink",
                "shrink_factor": float(args.shrink_factor),
            },
            topology_cost_delta=0.0,
            evidence_summary={
                "selector": "checkpoint_area_outlier_snap_shrink",
                "area_threshold": threshold,
                "median_area": median,
                "selected_area_before": before_area,
                "selected_area_after": after_area,
                "area_reduction": before_area - after_area,
                "selected_face": selected_face,
                "selected_vertices": selected_vertices,
                "csef_source": "checkpoint_area_statistics_not_edge_topology",
            },
            risk_summary={
                "free_space_risk": 0.0,
                "snap_through_free_space": False,
                "requires_render_backed_gate": True,
                "max_vertex_displacement": max_displacement,
            },
        )
        (out / "selected_snap_edit.json").write_text(json.dumps(edit.to_dict(), indent=2), encoding="utf-8")
    report = {
        "status": status,
        "checkpoint_path": args.checkpoint_path,
        "triangles": triangle_count,
        "vertices": vertex_count,
        "median_area": median,
        "max_area": float(np.max(areas)) if len(areas) else 0.0,
        "percentile_threshold": percentile_threshold,
        "ratio_threshold": ratio_threshold,
        "effective_threshold": threshold,
        "candidate_count": int(len(candidate_ids)),
        "selected_face": selected_face,
        "selected_vertices": selected_vertices,
        "selected_area_before": before_area,
        "selected_area_after": after_area,
        "area_reduction": before_area - after_area,
        "shrink_factor": float(args.shrink_factor),
        "max_vertex_displacement": max_displacement,
        "edit_json": str(out / "selected_snap_edit.json") if edit is not None else "",
        "note": "non-delete checkpoint-statistics SNAP proposal; render-backed gate is mandatory before acceptance",
    }
    (out / "area_outlier_snap_selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# MeshSplatOpt Checkpoint Area-Outlier Snap Selection",
        "",
        f"- status: `{status}`",
        f"- checkpoint: `{args.checkpoint_path}`",
        f"- selected face: `{selected_face}`",
        f"- selected vertices: `{selected_vertices}`",
        f"- area before: `{before_area}`",
        f"- area after: `{after_area}`",
        f"- max vertex displacement: `{max_displacement}`",
        "",
        "Render-backed validation is mandatory before accepting this non-delete edit.",
    ]
    (out / "area_outlier_snap_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit("no checkpoint area-outlier snap candidate selected")


if __name__ == "__main__":
    main()
