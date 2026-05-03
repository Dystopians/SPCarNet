#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib.path import Path as MplPath
from scipy.spatial import Delaunay

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshSplatOptEditType


def _ordered_loop_from_fan(edit: MeshEdit, old_vertex_count: int) -> list[int]:
    loop: list[int] = []
    for face in edit.inserted_faces:
        old = [int(v) for v in face if int(v) < old_vertex_count]
        if len(old) != 2:
            continue
        if not loop:
            loop.extend(old)
            continue
        if loop[-1] == old[0]:
            loop.append(old[1])
        elif loop[-1] == old[1]:
            loop.append(old[0])
        elif loop[0] == old[1]:
            loop.insert(0, old[0])
        elif loop[0] == old[0]:
            loop.insert(0, old[1])
    if len(loop) > 1 and loop[0] == loop[-1]:
        loop = loop[:-1]
    dedup: list[int] = []
    for vid in loop:
        if vid not in dedup:
            dedup.append(vid)
    if len(dedup) < 3:
        raise ValueError("Could not recover an ordered boundary loop from the fan fill edit")
    return dedup


def _plane_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    u = vh[0]
    v = vh[1]
    n = np.cross(u, v)
    n_norm = np.linalg.norm(n)
    if n_norm <= 1e-12:
        raise ValueError("Degenerate boundary loop plane")
    n = n / n_norm
    return center, u, v


def _polygon_area_xy(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _interior_grid(poly: np.ndarray, spacing: float) -> np.ndarray:
    lo = poly.min(axis=0)
    hi = poly.max(axis=0)
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    xs = np.arange(lo[0] + 0.5 * spacing, hi[0], spacing)
    ys = np.arange(lo[1] + 0.5 * spacing, hi[1], spacing)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    grid = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
    return grid[MplPath(poly).contains_points(grid)]


def expand_fill_to_grid(
    checkpoint_path: Path,
    edit_path: Path,
    *,
    output_dir: Path,
    target_added_vertices: int,
    max_added_faces: int,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    vertices = payload["triangles_points"].detach().cpu().numpy().astype(np.float64)
    old_vertex_count = int(vertices.shape[0])
    edit = MeshEdit(**json.loads(edit_path.read_text(encoding="utf-8")))
    if MeshSplatOptEditType(edit.edit_type) != MeshSplatOptEditType.FILL_PATCH:
        raise ValueError(f"Expected FILL_PATCH edit, got {edit.edit_type}")

    loop = _ordered_loop_from_fan(edit, old_vertex_count)
    loop_points = vertices[np.asarray(loop, dtype=np.int64)]
    center, u, v = _plane_basis(loop_points)
    loop_2d = np.stack([(loop_points - center) @ u, (loop_points - center) @ v], axis=1)
    area = _polygon_area_xy(loop_2d)
    target_added_vertices = max(1, int(target_added_vertices))
    spacing = math.sqrt(max(area, 1e-12) / float(target_added_vertices))
    interior_2d = _interior_grid(loop_2d, spacing)
    if int(interior_2d.shape[0]) == 0:
        interior_2d = np.asarray([[0.0, 0.0]], dtype=np.float64)
    interior_3d = center[None, :] + interior_2d[:, 0:1] * u[None, :] + interior_2d[:, 1:2] * v[None, :]

    all_2d = np.concatenate([loop_2d, interior_2d], axis=0)
    tri = Delaunay(all_2d)
    path = MplPath(loop_2d)
    faces: list[list[int]] = []
    for simplex in tri.simplices:
        pts = all_2d[np.asarray(simplex)]
        tri_area = _polygon_area_xy(pts)
        if tri_area <= 1e-10:
            continue
        if not bool(path.contains_point(pts.mean(axis=0), radius=1e-9)):
            continue
        global_face: list[int] = []
        for idx in simplex.tolist():
            if idx < len(loop):
                global_face.append(int(loop[idx]))
            else:
                global_face.append(old_vertex_count + int(idx - len(loop)))
        faces.append(global_face)
    if max_added_faces > 0 and len(faces) > int(max_added_faces):
        faces = faces[: int(max_added_faces)]
    if len(faces) == 0:
        raise ValueError("Grid fill triangulation produced no valid faces")

    expanded = MeshEdit(
        edit_id=f"{edit.edit_id}_grid_fill",
        edit_type=MeshSplatOptEditType.FILL_PATCH.value,
        defect_id=edit.defect_id,
        affected_vertices=list(loop),
        affected_faces=list(edit.affected_faces),
        inserted_vertices=interior_3d.astype(float).tolist(),
        inserted_faces=faces,
        deleted_vertices=list(edit.deleted_vertices),
        deleted_faces=list(edit.deleted_faces),
        attribute_changes={
            **dict(edit.attribute_changes),
            "faces_are_global_indices": True,
            "fill_mode": "plane_grid_delaunay",
            "fill_grid_spacing": float(spacing),
            "face_field_init": str(edit.attribute_changes.get("face_field_init", "nearest")),
            "face_field_init_scale": float(edit.attribute_changes.get("face_field_init_scale", 0.5)),
        },
        topology_cost_delta=float(len(faces)),
        evidence_summary={
            **dict(edit.evidence_summary),
            "expected_topology_cost": int(len(faces)),
            "expected_added_vertices": int(interior_3d.shape[0]),
            "expected_added_faces": int(len(faces)),
            "expected_area_repaired": float(area),
            "fill_mode": "plane_grid_delaunay",
        },
        risk_summary=dict(edit.risk_summary),
        rollback_snapshot_path=edit.rollback_snapshot_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    edit_out = output_dir / "selected_boundary_grid_fill_edit.json"
    edit_out.write_text(json.dumps(expanded.to_dict(), indent=2), encoding="utf-8")
    report = {
        "status": "PASS",
        "source_edit": str(edit_path),
        "checkpoint_path": str(checkpoint_path),
        "output_edit": str(edit_out),
        "loop_vertices": int(len(loop)),
        "added_vertices": int(interior_3d.shape[0]),
        "added_faces": int(len(faces)),
        "area_2d": float(area),
        "spacing": float(spacing),
        "max_added_faces": int(max_added_faces),
    }
    (output_dir / "boundary_grid_fill_expansion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Boundary Grid Fill Expansion",
        "",
        f"- status: `{report['status']}`",
        f"- loop vertices: `{report['loop_vertices']}`",
        f"- added vertices: `{report['added_vertices']}`",
        f"- added faces: `{report['added_faces']}`",
        f"- 2D area: `{report['area_2d']:.6f}`",
        f"- grid spacing: `{report['spacing']:.6f}`",
    ]
    (output_dir / "boundary_grid_fill_expansion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand a fan FILL_PATCH edit into a denser plane-grid fill.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--edit_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_added_vertices", type=int, default=48)
    parser.add_argument("--max_added_faces", type=int, default=192)
    args = parser.parse_args()
    report = expand_fill_to_grid(
        checkpoint_path=Path(args.checkpoint_path),
        edit_path=Path(args.edit_json),
        output_dir=Path(args.output_dir),
        target_added_vertices=args.target_added_vertices,
        max_added_faces=args.max_added_faces,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
