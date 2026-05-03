#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_adapter import checkpoint_to_mesh_state, load_checkpoint_state
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshSplatOptEditType
from ss3dm_prior.meshsplatopt.snap_proposals import make_snap_proposals
from scripts.car_model.meshsplatopt_select_checkpoint_area_outlier_edit import triangle_areas_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select residual-aware local SNAP_VERTICES checkpoint edits.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--render_set", default="test", choices=["train", "test"])
    parser.add_argument("--iteration", type=int, default=2000)
    parser.add_argument(
        "--camera_index_offset",
        type=int,
        default=None,
        help="Offset from residual image index to cameras.json index. Defaults to auto inference.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k_faces", type=int, default=2048)
    parser.add_argument("--min_percentile", type=float, default=98.0)
    parser.add_argument("--min_area_ratio_to_median", type=float, default=20.0)
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--residual_view_quantile", type=float, default=0.75)
    parser.add_argument("--max_selected_vertices", type=int, default=16)
    parser.add_argument("--min_selected_vertex_distance", type=float, default=0.25)
    parser.add_argument("--max_displacement_fraction", type=float, default=0.01)
    parser.add_argument("--residual_threshold_fraction", type=float, default=0.001)
    parser.add_argument("--max_proposal_uncertainty", type=float, default=0.55)
    parser.add_argument("--exclude_boundary_vertices", action="store_true")
    parser.add_argument("--min_error_reduction", type=float, default=1e-7)
    parser.add_argument("--chunk_size", type=int, default=250000)
    return parser.parse_args()


def load_residual(path_render: Path, path_gt: Path) -> np.ndarray:
    render = np.asarray(Image.open(path_render).convert("RGB"), dtype=np.float32) / 255.0
    gt = np.asarray(Image.open(path_gt).convert("RGB"), dtype=np.float32) / 255.0
    if render.shape != gt.shape:
        gt = np.asarray(Image.fromarray((gt * 255).astype(np.uint8)).resize((render.shape[1], render.shape[0])), dtype=np.float32) / 255.0
    return np.mean(np.abs(render - gt), axis=2)


def load_residual_views(model_path: Path, render_set: str, iteration: int, *, max_views: int, quantile: float) -> tuple[list[dict], int]:
    root = model_path / render_set / f"ours_{iteration}"
    render_dir = root / "renders"
    gt_dir = root / "gt"
    if not render_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing render/gt directories under {root}")
    rows = []
    for render_path in sorted(render_dir.glob("*.png")):
        gt_path = gt_dir / render_path.name
        if not gt_path.exists():
            continue
        residual = load_residual(render_path, gt_path)
        rows.append(
            {
                "index": int(render_path.stem),
                "name": render_path.name,
                "mean_residual": float(np.mean(residual)),
                "residual": residual,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No paired residual images found under {root}")
    threshold = float(np.quantile([r["mean_residual"] for r in rows], float(quantile)))
    selected = [r for r in rows if float(r["mean_residual"]) >= threshold]
    selected = sorted(selected, key=lambda r: float(r["mean_residual"]), reverse=True)[: max(1, int(max_views))]
    return selected, len(rows)


def infer_camera_index_offset(render_set: str, num_render_views: int, num_cameras: int, explicit_offset: int | None) -> int:
    if explicit_offset is not None:
        return int(explicit_offset)
    if render_set == "test":
        return 0
    inferred = int(num_cameras) - int(num_render_views)
    if inferred < 0:
        raise ValueError(
            f"Cannot infer camera offset: num_render_views={num_render_views} exceeds num_cameras={num_cameras}"
        )
    return inferred


def project_vertices(vertices: np.ndarray, camera: dict, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape
    c = np.asarray(camera["position"], dtype=np.float64)
    rot_c2w = np.asarray(camera["rotation"], dtype=np.float64)
    r_wc = rot_c2w.T
    x_cam = (vertices - c.reshape(1, 3)) @ r_wc.T
    z = x_cam[:, 2]
    fx = float(camera["fx"]) * (w / float(camera["width"]))
    fy = float(camera["fy"]) * (h / float(camera["height"]))
    u = fx * (x_cam[:, 0] / np.maximum(z, 1e-9)) + 0.5 * w
    v = fy * (x_cam[:, 1] / np.maximum(z, 1e-9)) + 0.5 * h
    valid = (z > 1e-6) & (u >= 0.0) & (u < w) & (v >= 0.0) & (v < h)
    return np.stack([u, v], axis=1), valid


def sample_residual(residual: np.ndarray, uv: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full((uv.shape[0],), np.nan, dtype=np.float64)
    h, w = residual.shape
    ids = np.where(valid)[0]
    if len(ids) == 0:
        return out
    x = np.clip(np.rint(uv[ids, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(uv[ids, 1]).astype(np.int64), 0, h - 1)
    out[ids] = residual[y, x]
    return out


def select_portfolio(proposals, vertices: np.ndarray, scores: dict[int, float], *, max_selected: int, min_distance: float) -> list:
    best_by_vertex = {}
    for proposal in proposals:
        if proposal.rejected_reason or not proposal.edit.affected_vertices:
            continue
        vid = int(proposal.edit.affected_vertices[0])
        reduction = float(proposal.expected_error_before - proposal.expected_error_after)
        score = reduction * float(scores.get(vid, 0.0))
        current = best_by_vertex.get(vid)
        if current is None or score > current[0]:
            best_by_vertex[vid] = (score, proposal)
    ranked = sorted(best_by_vertex.values(), key=lambda item: item[0], reverse=True)
    selected = []
    selected_vertices: list[int] = []
    for score, proposal in ranked:
        if score <= 0.0:
            continue
        vid = int(proposal.edit.affected_vertices[0])
        if selected_vertices and min_distance > 0:
            if min(float(np.linalg.norm(vertices[vid] - vertices[other])) for other in selected_vertices) < min_distance:
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
    vertices = np.asarray(state.vertices, dtype=np.float64)
    faces = payload["_triangle_indices"].detach().cpu().to(dtype=torch.long)
    areas = triangle_areas_from_checkpoint(payload, chunk_size=args.chunk_size)
    median = float(np.median(areas)) if len(areas) else 0.0
    threshold = max(float(np.percentile(areas, args.min_percentile)), median * float(args.min_area_ratio_to_median))
    candidate_face_ids = np.where(areas >= threshold)[0]
    ranked_faces = sorted((int(i) for i in candidate_face_ids), key=lambda i: float(areas[i]), reverse=True)[: int(args.top_k_faces)]
    candidate_vertices = sorted({int(v) for fid in ranked_faces for v in faces[int(fid)].tolist()})

    cameras = json.loads((Path(args.model_path) / "cameras.json").read_text(encoding="utf-8"))
    residual_views, num_render_views = load_residual_views(
        Path(args.model_path),
        args.render_set,
        int(args.iteration),
        max_views=int(args.max_views),
        quantile=float(args.residual_view_quantile),
    )
    camera_index_offset = infer_camera_index_offset(
        args.render_set,
        num_render_views,
        len(cameras),
        args.camera_index_offset,
    )
    candidate_array = vertices[np.asarray(candidate_vertices, dtype=np.int64)]
    residual_samples: dict[int, list[float]] = {int(v): [] for v in candidate_vertices}
    for view in residual_views:
        camera_index = int(view["index"]) + int(camera_index_offset)
        if camera_index >= len(cameras):
            continue
        residual = view["residual"]
        uv, valid = project_vertices(candidate_array, cameras[camera_index], residual.shape)
        sampled = sample_residual(residual, uv, valid)
        for local_i, value in enumerate(sampled):
            if np.isfinite(value):
                residual_samples[candidate_vertices[local_i]].append(float(value))
    residual_scores = {
        vid: float(np.mean(values)) for vid, values in residual_samples.items() if len(values) > 0
    }
    scored_vertices = sorted(residual_scores, key=lambda vid: residual_scores[vid], reverse=True)
    candidate_vertices_scored = scored_vertices[: max(int(args.max_selected_vertices) * 64, int(args.max_selected_vertices))]

    proposals = make_snap_proposals(
        state,
        candidate_vertices=candidate_vertices_scored,
        supported_vertices=set(candidate_vertices_scored),
        max_displacement_fraction=float(args.max_displacement_fraction),
        residual_threshold_fraction=float(args.residual_threshold_fraction),
        evidence_source=f"{args.render_set}_render_residual_local_plane_csef",
    )
    accepted_before_risk = [
        p
        for p in proposals
        if not p.rejected_reason
        and (p.expected_error_before - p.expected_error_after) >= float(args.min_error_reduction)
    ]
    valid = [
        p
        for p in accepted_before_risk
        if float(p.uncertainty) <= float(args.max_proposal_uncertainty)
        and not (bool(args.exclude_boundary_vertices) and bool(p.edit.risk_summary.get("boundary_vertex", False)))
    ]
    selected = select_portfolio(
        valid,
        vertices,
        residual_scores,
        max_selected=int(args.max_selected_vertices),
        min_distance=float(args.min_selected_vertex_distance),
    )
    status = "PASS" if selected else "NO_CANDIDATE"
    edit_json = ""
    if selected:
        selected_vertices = [int(p.edit.affected_vertices[0]) for p in selected]
        target_positions = {}
        total_before = total_after = 0.0
        residual_mean = 0.0
        for proposal in selected:
            vid = int(proposal.edit.affected_vertices[0])
            target_positions.update(proposal.edit.attribute_changes.get("target_positions", {}))
            total_before += float(proposal.expected_error_before)
            total_after += float(proposal.expected_error_after)
            residual_mean += float(residual_scores.get(vid, 0.0))
        residual_mean /= max(len(selected), 1)
        edit = MeshEdit(
            edit_id="real_checkpoint_csef_residual_snap_portfolio",
            edit_type=MeshSplatOptEditType.SNAP_VERTICES.value,
            defect_id="render_residual_local_surface_debt",
            affected_vertices=selected_vertices,
            affected_faces=[int(fid) for fid in ranked_faces if any(int(v) in selected_vertices for v in faces[int(fid)].tolist())],
            attribute_changes={
                "target_positions": target_positions,
                "selector": "render_residual_seeded_local_plane_csef_portfolio",
                "render_set": args.render_set,
                "camera_index_offset": int(camera_index_offset),
            },
            topology_cost_delta=0.0,
            evidence_summary={
                "selector": "render_residual_seeded_local_plane_csef_portfolio",
                "render_set": args.render_set,
                "camera_index_offset": int(camera_index_offset),
                "num_render_views": int(num_render_views),
                "uses_gt_residual": True,
                "candidate_face_count": int(len(candidate_face_ids)),
                "candidate_vertex_count": int(len(candidate_vertices)),
                "scored_vertex_count": int(len(residual_scores)),
                "selected_vertex_count": int(len(selected_vertices)),
                "mean_selected_residual": residual_mean,
                "total_local_plane_residual_before": total_before,
                "total_local_plane_residual_after": total_after,
                "total_expected_residual_reduction": total_before - total_after,
            },
            risk_summary={
                "requires_render_backed_gate": True,
                "heldout_selection_risk": args.render_set == "test",
                "free_space_risk": 0.0,
            },
        )
        edit_json = str(out / "selected_residual_snap_edit.json")
        Path(edit_json).write_text(json.dumps(edit.to_dict(), indent=2), encoding="utf-8")

    compact_selected = [
        {
            "proposal_id": p.proposal_id,
            "vertex": int(p.edit.affected_vertices[0]),
            "residual_score": float(residual_scores.get(int(p.edit.affected_vertices[0]), 0.0)),
            "before": float(p.expected_error_before),
            "after": float(p.expected_error_after),
            "uncertainty": float(p.uncertainty),
        }
        for p in selected
    ]
    report = {
        "status": status,
        "checkpoint_path": args.checkpoint_path,
        "model_path": args.model_path,
        "render_set": args.render_set,
        "camera_index_offset": int(camera_index_offset),
        "camera_index_offset_mode": "explicit" if args.camera_index_offset is not None else "auto",
        "num_render_views": int(num_render_views),
        "uses_gt_residual": True,
        "triangles": int(faces.shape[0]),
        "vertices": int(vertices.shape[0]),
        "candidate_face_count": int(len(candidate_face_ids)),
        "candidate_vertex_count": int(len(candidate_vertices)),
        "scored_vertex_count": int(len(residual_scores)),
        "proposal_count": int(len(proposals)),
        "accepted_before_risk_filter_count": int(len(accepted_before_risk)),
        "valid_proposal_count": int(len(valid)),
        "rejection_reasons": dict(Counter(str(p.rejected_reason or "accepted") for p in proposals)),
        "risk_filter_rejections": {
            "uncertainty": int(
                sum(float(p.uncertainty) > float(args.max_proposal_uncertainty) for p in accepted_before_risk)
            ),
            "boundary": int(
                sum(
                    bool(args.exclude_boundary_vertices) and bool(p.edit.risk_summary.get("boundary_vertex", False))
                    for p in accepted_before_risk
                )
            ),
        },
        "selected_vertex_count": int(len(selected)),
        "selected": compact_selected,
        "edit_json": edit_json,
        "residual_views": [{"index": int(v["index"]), "mean_residual": float(v["mean_residual"])} for v in residual_views],
        "note": "Residual-aware selector. If render_set=test, this is diagnostic and must not be used as an inference-time proposal claim.",
    }
    (out / "residual_snap_selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit("no residual-aware local snap candidate selected")


if __name__ == "__main__":
    main()
