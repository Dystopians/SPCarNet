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

from ss3dm_prior.meshsplatopt.checkpoint_adapter import checkpoint_to_mesh_state, load_checkpoint_state
from ss3dm_prior.meshsplatopt.snap_proposals import make_snap_proposals
from scripts.car_model.meshsplatopt_select_checkpoint_area_outlier_edit import triangle_areas_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a local CSEF-style SNAP_VERTICES edit from a checkpoint.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k_faces", type=int, default=32)
    parser.add_argument("--min_area_ratio_to_median", type=float, default=100.0)
    parser.add_argument("--min_percentile", type=float, default=99.5)
    parser.add_argument("--max_displacement_fraction", type=float, default=0.02)
    parser.add_argument("--residual_threshold_fraction", type=float, default=0.002)
    parser.add_argument("--min_error_reduction", type=float, default=1e-7)
    parser.add_argument("--chunk_size", type=int, default=250000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = load_checkpoint_state(args.checkpoint_path)
    state = checkpoint_to_mesh_state(payload)
    areas = triangle_areas_from_checkpoint(payload, chunk_size=args.chunk_size)
    faces = payload["_triangle_indices"].detach().cpu().to(dtype=torch.long)
    triangle_count = int(faces.shape[0])
    vertex_count = int(payload["triangles_points"].shape[0])
    median = float(np.median(areas)) if len(areas) else 0.0
    percentile_threshold = float(np.percentile(areas, args.min_percentile)) if len(areas) else 0.0
    ratio_threshold = median * float(args.min_area_ratio_to_median)
    threshold = max(percentile_threshold, ratio_threshold)
    candidate_face_ids = np.where(areas >= threshold)[0]
    ranked_faces = sorted((int(i) for i in candidate_face_ids), key=lambda i: float(areas[i]), reverse=True)[: int(args.top_k_faces)]
    candidate_vertices = sorted({int(v) for fid in ranked_faces for v in faces[int(fid)].tolist()})
    proposals = make_snap_proposals(
        state,
        candidate_vertices=candidate_vertices,
        supported_vertices=set(candidate_vertices),
        max_displacement_fraction=float(args.max_displacement_fraction),
        residual_threshold_fraction=float(args.residual_threshold_fraction),
        evidence_source="checkpoint_area_seeded_local_plane_csef",
    )
    valid = [
        p
        for p in proposals
        if not p.rejected_reason and (p.expected_error_before - p.expected_error_after) >= float(args.min_error_reduction)
    ]
    best = max(valid, key=lambda p: (p.expected_error_before - p.expected_error_after, -p.uncertainty), default=None)
    status = "PASS" if best is not None else "NO_CANDIDATE"
    if best is not None:
        best_edit = best.edit.to_dict()
        best_edit["edit_id"] = "real_checkpoint_csef_local_snap"
        best_edit["defect_id"] = "checkpoint_local_surface_residual"
        selected_vertices = best_edit.get("affected_vertices", [])
        selected_faces = [int(fid) for fid in ranked_faces if any(int(v) in selected_vertices for v in faces[int(fid)].tolist())]
        best_edit["affected_faces"] = selected_faces[: int(args.top_k_faces)]
        best_edit["evidence_summary"]["seed_selector"] = "large_triangle_area_candidates"
        best_edit["evidence_summary"]["area_threshold"] = threshold
        best_edit["evidence_summary"]["median_area"] = median
        best_edit["evidence_summary"]["candidate_face_count"] = int(len(candidate_face_ids))
        (out / "selected_local_snap_edit.json").write_text(json.dumps(best_edit, indent=2), encoding="utf-8")
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
        "candidate_face_count": int(len(candidate_face_ids)),
        "ranked_faces": ranked_faces,
        "candidate_vertices": candidate_vertices,
        "proposal_count": int(len(proposals)),
        "valid_proposal_count": int(len(valid)),
        "selected": best.to_dict() if best is not None else None,
        "edit_json": str(out / "selected_local_snap_edit.json") if best is not None else "",
        "note": "local-plane CSEF-style checkpoint SNAP proposal; render-backed gate is mandatory before acceptance",
    }
    (out / "local_snap_selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# MeshSplatOpt Checkpoint Local Snap Selection",
        "",
        f"- status: `{status}`",
        f"- checkpoint: `{args.checkpoint_path}`",
        f"- candidate faces: `{len(candidate_face_ids)}`",
        f"- candidate vertices: `{len(candidate_vertices)}`",
        f"- proposals: `{len(proposals)}`",
        f"- valid proposals: `{len(valid)}`",
        f"- edit json: `{report['edit_json']}`",
        "",
        "Render-backed validation is mandatory before accepting this non-delete edit.",
    ]
    (out / "local_snap_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit("no checkpoint local snap candidate selected")


if __name__ == "__main__":
    main()
