#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import (  # noqa: E402
    FACE_KEYS,
    VERTEX_KEYS,
    checkpoint_path,
    copy_model_metadata,
    validate_faces,
)


def _simplify_open3d(vertices: np.ndarray, faces: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    simplified = mesh.simplify_quadric_decimation(int(target_faces))
    simplified.remove_duplicated_triangles()
    simplified.remove_degenerate_triangles()
    simplified.remove_unreferenced_vertices()
    out_vertices = np.asarray(simplified.vertices, dtype=np.float32)
    out_faces = np.asarray(simplified.triangles, dtype=np.int64)
    if out_faces.size == 0:
        raise RuntimeError("Open3D QEM returned an empty mesh")
    return out_vertices, out_faces


def _simplify_fast(vertices: np.ndarray, faces: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    import fast_simplification

    out_vertices, out_faces = fast_simplification.simplify(
        vertices.astype(np.float64, copy=False),
        faces.astype(np.int64, copy=False),
        target_count=int(target_faces),
    )
    out_vertices = np.asarray(out_vertices, dtype=np.float32)
    out_faces = np.asarray(out_faces, dtype=np.int64)
    if out_faces.size == 0:
        raise RuntimeError("fast_simplification returned an empty mesh")
    return out_vertices, out_faces


def _simplify(vertices: np.ndarray, faces: np.ndarray, target_faces: int, backend: str) -> tuple[np.ndarray, np.ndarray]:
    if backend == "open3d":
        return _simplify_open3d(vertices, faces, target_faces)
    if backend == "fast_simplification":
        return _simplify_fast(vertices, faces, target_faces)
    raise ValueError(f"Unknown backend: {backend}")


def _nearest_indices(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    _, idx = cKDTree(reference.astype(np.float64)).query(query.astype(np.float64), k=1, workers=-1)
    return np.asarray(idx, dtype=np.int64)


def _transfer_state(state: dict[str, Any], vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    src_vertices = state["triangles_points"].detach().cpu().numpy()
    src_faces = state["_triangle_indices"].detach().cpu().long().numpy()
    src_centroids = src_vertices[src_faces].mean(axis=1)
    dst_centroids = vertices[faces].mean(axis=1)
    vertex_nn = _nearest_indices(vertices, src_vertices)
    face_nn = _nearest_indices(dst_centroids, src_centroids)

    out: dict[str, Any] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        cpu = value.detach().cpu()
        if key == "triangles_points":
            out[key] = torch.as_tensor(vertices, dtype=cpu.dtype)
        elif key == "_triangle_indices":
            out[key] = torch.as_tensor(faces, dtype=cpu.dtype)
        elif key in VERTEX_KEYS and cpu.shape[0] == src_vertices.shape[0]:
            out[key] = cpu[torch.as_tensor(vertex_nn, dtype=torch.long)].clone()
        elif key in FACE_KEYS and cpu.shape[0] == src_faces.shape[0]:
            out[key] = cpu[torch.as_tensor(face_nn, dtype=torch.long)].clone()
        else:
            out[key] = cpu.clone()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Open3D QEM decimation to a Mesh Splatting checkpoint.")
    parser.add_argument("--source_model", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--target_faces", type=int, default=0)
    parser.add_argument("--target_fraction", type=float, default=0.5)
    parser.add_argument("--backend", choices=["open3d", "fast_simplification"], default="open3d")
    args = parser.parse_args()

    src_model = Path(args.source_model)
    out_model = Path(args.output_model)
    src_checkpoint = checkpoint_path(src_model, args.iteration)
    out_checkpoint = out_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud_state_dict.pt"
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    copy_model_metadata(src_model, out_model)

    state = torch.load(src_checkpoint, map_location="cpu")
    vertices = state["triangles_points"].detach().cpu().numpy()
    faces = state["_triangle_indices"].detach().cpu().long().numpy()
    target_faces = int(args.target_faces) if args.target_faces > 0 else int(round(faces.shape[0] * args.target_fraction))
    target_faces = max(1, min(int(faces.shape[0] - 1), target_faces))
    qem_vertices, qem_faces = _simplify(vertices, faces, target_faces, args.backend)
    out_state = _transfer_state(state, qem_vertices, qem_faces)
    degenerate, invalid = validate_faces(out_state["triangles_points"], out_state["_triangle_indices"])
    torch.save(out_state, out_checkpoint)

    audit = {
        "source_model": str(src_model),
        "source_checkpoint": str(src_checkpoint),
        "output_model": str(out_model),
        "output_checkpoint": str(out_checkpoint),
        "iteration": int(args.iteration),
        "method": f"{args.backend}_quadric_decimation",
        "backend": args.backend,
        "target_faces": int(target_faces),
        "pre_triangles": int(faces.shape[0]),
        "post_triangles": int(qem_faces.shape[0]),
        "pre_vertices": int(vertices.shape[0]),
        "post_vertices": int(qem_vertices.shape[0]),
        "removed_triangles": int(faces.shape[0] - qem_faces.shape[0]),
        "removed_fraction": float((faces.shape[0] - qem_faces.shape[0]) / max(faces.shape[0], 1)),
        "degenerate_face_count": int(degenerate),
        "invalid_index_count": int(invalid),
        "attribute_transfer": "nearest_source_vertex_for_vertex_tensors_nearest_source_centroid_for_face_tensors",
    }
    (out_model / "topology_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if degenerate == 0 and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
