#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_adapter import checkpoint_to_mesh_state, load_checkpoint_state
from ss3dm_prior.meshsplatopt.hole_fill import find_boundary_loops, loop_area_xy, make_boundary_loop_fill
from scripts.car_model.meshsplatopt_select_checkpoint_residual_snap_edit import (
    infer_camera_index_offset,
    load_residual_views,
    project_vertices,
    sample_residual,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a checkpoint boundary-loop FILL_PATCH edit.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_loop_vertices", type=int, default=3)
    parser.add_argument("--max_loop_vertices", type=int, default=24)
    parser.add_argument("--min_area", type=float, default=1e-6)
    parser.add_argument("--max_area", type=float, default=25.0)
    parser.add_argument("--rank", choices=["smallest", "largest", "residual"], default="smallest")
    parser.add_argument("--model_path")
    parser.add_argument("--render_set", default="train", choices=["train", "test"])
    parser.add_argument("--iteration", type=int, default=2000)
    parser.add_argument("--camera_index_offset", type=int, default=None)
    parser.add_argument("--max_views", type=int, default=24)
    parser.add_argument("--residual_view_quantile", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = load_checkpoint_state(args.checkpoint_path)
    state = checkpoint_to_mesh_state(payload)
    vertices = np.asarray(state.vertices, dtype=np.float64)
    loops = find_boundary_loops(np.asarray(state.faces, dtype=np.int64))
    candidates = []
    for i, loop in enumerate(loops):
        if len(loop) < int(args.min_loop_vertices) or len(loop) > int(args.max_loop_vertices):
            continue
        area = loop_area_xy(vertices, loop)
        if area < float(args.min_area) or area > float(args.max_area):
            continue
        candidates.append({"loop_index": i, "loop": loop, "area_xy": float(area), "loop_vertices": len(loop)})
    residual_views = []
    camera_index_offset = None
    if args.rank == "residual":
        if not args.model_path:
            raise SystemExit("--model_path is required when --rank residual")
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
        for row in candidates:
            loop_vertices = vertices[np.asarray(row["loop"], dtype=np.int64)]
            samples = []
            for view in residual_views:
                camera_index = int(view["index"]) + int(camera_index_offset)
                if camera_index >= len(cameras):
                    continue
                uv, valid = project_vertices(loop_vertices, cameras[camera_index], view["residual"].shape)
                values = sample_residual(view["residual"], uv, valid)
                samples.extend([float(x) for x in values if np.isfinite(x)])
            row["residual_score"] = float(np.mean(samples)) if samples else 0.0
            row["rank_score"] = float(row["residual_score"] * np.sqrt(max(row["area_xy"], 1e-12)))
        candidates = sorted(candidates, key=lambda row: row["rank_score"], reverse=True)
    else:
        reverse = args.rank == "largest"
        candidates = sorted(candidates, key=lambda row: row["area_xy"], reverse=reverse)
    if not candidates:
        report = {
            "status": "NO_CANDIDATE",
            "checkpoint_path": args.checkpoint_path,
            "loop_count": len(loops),
            "candidate_count": 0,
            "filters": vars(args),
        }
        (out / "boundary_fill_selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise SystemExit("no boundary fill candidate selected")
    chosen = candidates[0]
    proposal = make_boundary_loop_fill(state, chosen["loop"], proposal_id="checkpoint_boundary_fill")
    if proposal.edit is None:
        raise SystemExit(f"selected loop rejected: {proposal.rejected_reason}")
    edit_path = out / "selected_boundary_fill_edit.json"
    edit_path.write_text(json.dumps(proposal.edit.to_dict(), indent=2), encoding="utf-8")
    report = {
        "status": "PASS",
        "checkpoint_path": args.checkpoint_path,
        "triangles": int(state.faces.shape[0]),
        "vertices": int(state.vertices.shape[0]),
        "loop_count": len(loops),
        "candidate_count": len(candidates),
        "selected": {
            "loop_index": int(chosen["loop_index"]),
            "loop_vertices": int(chosen["loop_vertices"]),
            "area_xy": float(chosen["area_xy"]),
            "residual_score": float(chosen.get("residual_score", 0.0)),
            "rank_score": float(chosen.get("rank_score", chosen["area_xy"])),
            "expected_added_vertices": int(len(proposal.edit.inserted_vertices)),
            "expected_added_faces": int(len(proposal.edit.inserted_faces)),
        },
        "rank": args.rank,
        "render_set": args.render_set if args.rank == "residual" else "",
        "camera_index_offset": camera_index_offset,
        "residual_views": [
            {"index": int(v["index"]), "mean_residual": float(v["mean_residual"])} for v in residual_views
        ],
        "edit_json": str(edit_path),
        "certificate": proposal.certificate,
    }
    (out / "boundary_fill_selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
