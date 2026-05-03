#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_adapter import checkpoint_to_mesh_state, load_checkpoint_state
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshSplatOptEditType
from ss3dm_prior.meshsplatopt.snap_proposals import vertex_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand a seed SNAP_VERTICES edit into a local patch SNAP_VERTICES edit.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--seed_edit_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--neighbor_hops", type=int, default=1)
    parser.add_argument("--max_radius", type=float, default=0.12)
    parser.add_argument("--falloff_radius", type=float, default=0.06)
    parser.add_argument("--min_radius_fraction", type=float, default=0.0)
    parser.add_argument("--min_neighbor_weight", type=float, default=0.15)
    parser.add_argument("--max_patch_vertices", type=int, default=256)
    parser.add_argument("--max_neighbor_displacement_fraction", type=float, default=0.75)
    return parser.parse_args()


def k_hop_neighbors(neighbors: list[set[int]], seed: int, hops: int) -> set[int]:
    out = {int(seed)}
    frontier = {int(seed)}
    for _ in range(max(0, int(hops))):
        nxt: set[int] = set()
        for vid in frontier:
            nxt.update(neighbors[int(vid)])
        nxt.difference_update(out)
        out.update(nxt)
        frontier = nxt
        if not frontier:
            break
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = load_checkpoint_state(args.checkpoint_path)
    state = checkpoint_to_mesh_state(payload)
    vertices = np.asarray(state.vertices, dtype=np.float64)
    faces = np.asarray(state.faces, dtype=np.int64)
    seed_edit = json.loads(Path(args.seed_edit_json).read_text(encoding="utf-8"))
    seed_targets = seed_edit.get("attribute_changes", {}).get("target_positions", {})
    if not seed_targets:
        raise SystemExit("seed edit has no target_positions")

    neighbors = vertex_neighbors(faces, len(vertices))
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    max_radius = max(float(args.max_radius), bbox_diag * float(args.min_radius_fraction), 1e-8)
    falloff_radius = max(float(args.falloff_radius), 1e-8)
    max_neighbor_fraction = float(args.max_neighbor_displacement_fraction)

    accum: dict[int, list[np.ndarray]] = defaultdict(list)
    seed_rows = []
    for seed_text, target in seed_targets.items():
        seed = int(seed_text)
        if seed < 0 or seed >= len(vertices):
            continue
        seed_target = np.asarray(target, dtype=np.float64)
        displacement = seed_target - vertices[seed]
        disp_norm = float(np.linalg.norm(displacement))
        if disp_norm <= 1e-12:
            continue
        patch = k_hop_neighbors(neighbors, seed, int(args.neighbor_hops))
        patch = sorted(
            (vid for vid in patch if float(np.linalg.norm(vertices[vid] - vertices[seed])) <= max_radius),
            key=lambda vid: float(np.linalg.norm(vertices[vid] - vertices[seed])),
        )
        for vid in patch:
            dist = float(np.linalg.norm(vertices[vid] - vertices[seed]))
            if vid == seed:
                weight = 1.0
            else:
                weight = float(np.exp(-0.5 * (dist / falloff_radius) ** 2))
                weight = max(weight, float(args.min_neighbor_weight))
                weight = min(weight, max_neighbor_fraction)
            accum[int(vid)].append(vertices[vid] + displacement * weight)
        seed_rows.append({"seed": seed, "seed_displacement": disp_norm, "patch_vertices": len(patch)})

    if not accum:
        raise SystemExit("no patch vertices selected")
    ranked_vertices = sorted(
        accum,
        key=lambda vid: max(float(np.linalg.norm(target - vertices[vid])) for target in accum[vid]),
        reverse=True,
    )[: int(args.max_patch_vertices)]
    target_positions = {
        str(int(vid)): [float(x) for x in np.mean(np.stack(accum[int(vid)], axis=0), axis=0)]
        for vid in ranked_vertices
    }
    affected = set(int(v) for v in ranked_vertices)
    affected_faces = [
        int(fid)
        for fid, face in enumerate(faces)
        if int(face[0]) in affected or int(face[1]) in affected or int(face[2]) in affected
    ]
    seed_evidence = seed_edit.get("evidence_summary", {})
    seed_risk = seed_edit.get("risk_summary", {})
    edit = MeshEdit(
        edit_id="residual_snap_patch_expansion",
        edit_type=MeshSplatOptEditType.SNAP_VERTICES.value,
        defect_id=seed_edit.get("defect_id", "render_residual_patch_debt"),
        affected_vertices=[int(v) for v in ranked_vertices],
        affected_faces=affected_faces,
        attribute_changes={
            "target_positions": target_positions,
            "selector": "residual_snap_patch_expansion",
            "seed_edit": str(args.seed_edit_json),
            "neighbor_hops": int(args.neighbor_hops),
            "max_radius": float(max_radius),
            "falloff_radius": float(falloff_radius),
        },
        topology_cost_delta=0.0,
        evidence_summary={
            **seed_evidence,
            "selector": "residual_snap_patch_expansion",
            "seed_vertex_count": int(len(seed_targets)),
            "patch_vertex_count": int(len(ranked_vertices)),
            "patch_face_count": int(len(affected_faces)),
            "seed_rows": seed_rows,
        },
        risk_summary={
            **seed_risk,
            "requires_render_backed_gate": True,
            "patch_expansion": True,
            "max_neighbor_displacement_fraction": max_neighbor_fraction,
        },
    )
    edit_path = out / "expanded_patch_snap_edit.json"
    edit_path.write_text(json.dumps(edit.to_dict(), indent=2), encoding="utf-8")
    displacement_norms = [
        float(np.linalg.norm(np.asarray(target_positions[str(vid)], dtype=np.float64) - vertices[int(vid)]))
        for vid in ranked_vertices
    ]
    report = {
        "status": "PASS",
        "checkpoint_path": args.checkpoint_path,
        "seed_edit_json": args.seed_edit_json,
        "edit_json": str(edit_path),
        "triangles": int(faces.shape[0]),
        "vertices": int(vertices.shape[0]),
        "seed_vertex_count": int(len(seed_targets)),
        "patch_vertex_count": int(len(ranked_vertices)),
        "patch_face_count": int(len(affected_faces)),
        "neighbor_hops": int(args.neighbor_hops),
        "max_radius": float(max_radius),
        "falloff_radius": float(falloff_radius),
        "max_displacement": float(max(displacement_norms)),
        "mean_displacement": float(np.mean(displacement_norms)),
    }
    (out / "expanded_patch_snap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
