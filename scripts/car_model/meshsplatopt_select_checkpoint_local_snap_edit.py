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
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshSplatOptEditType
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
    parser.add_argument("--max_selected_vertices", type=int, default=1)
    parser.add_argument("--min_selected_vertex_distance", type=float, default=0.0)
    parser.add_argument("--max_proposal_uncertainty", type=float, default=1.0)
    parser.add_argument("--exclude_boundary_vertices", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=250000)
    return parser.parse_args()


def select_portfolio(proposals, vertices: np.ndarray, *, max_selected: int, min_distance: float) -> list:
    best_by_vertex = {}
    for proposal in proposals:
        if proposal.rejected_reason:
            continue
        if not proposal.edit.affected_vertices:
            continue
        vid = int(proposal.edit.affected_vertices[0])
        reduction = float(proposal.expected_error_before - proposal.expected_error_after)
        current = best_by_vertex.get(vid)
        if current is None or reduction > float(current.expected_error_before - current.expected_error_after):
            best_by_vertex[vid] = proposal
    ranked = sorted(
        best_by_vertex.values(),
        key=lambda p: (float(p.expected_error_before - p.expected_error_after), -float(p.uncertainty)),
        reverse=True,
    )
    selected = []
    selected_vertices: list[int] = []
    for proposal in ranked:
        vid = int(proposal.edit.affected_vertices[0])
        if selected_vertices and min_distance > 0.0:
            distances = [float(np.linalg.norm(vertices[vid] - vertices[other])) for other in selected_vertices]
            if min(distances) < min_distance:
                continue
        selected.append(proposal)
        selected_vertices.append(vid)
        if len(selected) >= max(1, int(max_selected)):
            break
    return selected


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
        and float(p.uncertainty) <= float(args.max_proposal_uncertainty)
        and not (bool(args.exclude_boundary_vertices) and bool(p.edit.risk_summary.get("boundary_vertex", False)))
    ]
    selected = select_portfolio(
        valid,
        state.vertices,
        max_selected=int(args.max_selected_vertices),
        min_distance=float(args.min_selected_vertex_distance),
    )
    status = "PASS" if selected else "NO_CANDIDATE"
    if selected:
        selected_vertices = [int(p.edit.affected_vertices[0]) for p in selected]
        selected_faces = [int(fid) for fid in ranked_faces if any(int(v) in selected_vertices for v in faces[int(fid)].tolist())]
        target_positions: dict[str, list[float]] = {}
        total_before = 0.0
        total_after = 0.0
        max_uncertainty = 0.0
        for proposal in selected:
            target_positions.update(proposal.edit.attribute_changes.get("target_positions", {}))
            total_before += float(proposal.expected_error_before)
            total_after += float(proposal.expected_error_after)
            max_uncertainty = max(max_uncertainty, float(proposal.uncertainty))
        edit = MeshEdit(
            edit_id="real_checkpoint_csef_local_snap_portfolio",
            edit_type=MeshSplatOptEditType.SNAP_VERTICES.value,
            defect_id="checkpoint_local_surface_residual_portfolio",
            affected_vertices=selected_vertices,
            affected_faces=selected_faces[: int(args.top_k_faces)],
            attribute_changes={
                "target_positions": target_positions,
                "selector": "checkpoint_area_seeded_local_plane_csef_portfolio",
                "max_selected_vertices": int(args.max_selected_vertices),
                "min_selected_vertex_distance": float(args.min_selected_vertex_distance),
                "max_proposal_uncertainty": float(args.max_proposal_uncertainty),
                "exclude_boundary_vertices": bool(args.exclude_boundary_vertices),
            },
            topology_cost_delta=0.0,
            evidence_summary={
                "selector": "csef_local_plane_snap_portfolio",
                "seed_selector": "large_triangle_area_candidates",
                "area_threshold": threshold,
                "median_area": median,
                "candidate_face_count": int(len(candidate_face_ids)),
                "selected_vertex_count": int(len(selected_vertices)),
                "total_local_plane_residual_before": total_before,
                "total_local_plane_residual_after": total_after,
                "total_expected_residual_reduction": total_before - total_after,
                "max_proposal_uncertainty": float(args.max_proposal_uncertainty),
                "exclude_boundary_vertices": bool(args.exclude_boundary_vertices),
            },
            risk_summary={
                "free_space_risk": 0.0,
                "max_uncertainty": max_uncertainty,
                "requires_render_backed_gate": True,
            },
        )
        (out / "selected_local_snap_edit.json").write_text(json.dumps(edit.to_dict(), indent=2), encoding="utf-8")
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
        "max_proposal_uncertainty": float(args.max_proposal_uncertainty),
        "exclude_boundary_vertices": bool(args.exclude_boundary_vertices),
        "selected": selected[0].to_dict() if selected else None,
        "selected_proposals": [p.to_dict() for p in selected],
        "selected_vertex_count": int(len(selected)),
        "selected_vertices": [int(p.edit.affected_vertices[0]) for p in selected],
        "total_expected_residual_reduction": float(
            sum(float(p.expected_error_before - p.expected_error_after) for p in selected)
        ),
        "edit_json": str(out / "selected_local_snap_edit.json") if selected else "",
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
