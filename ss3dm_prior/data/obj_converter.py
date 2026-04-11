"""Convert large town OBJ meshes into compact binary caches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import numpy as np
import trimesh

from ss3dm_prior.utils.io import dump_json


@dataclass
class ConvertedMesh:
    vertices: np.ndarray
    faces: np.ndarray
    face_centroids: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    bbox: dict[str, Any]
    meta: dict[str, Any]


def _load_obj_mesh(obj_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(obj_path, process=False, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geometries = [geometry for geometry in loaded.geometry.values() if len(geometry.faces) > 0]
        if not geometries:
            raise ValueError(f"No mesh geometry found in scene: {obj_path}")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh from {obj_path}, got {type(mesh)}")
    return mesh


def compute_face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    face_vertices = vertices[faces]
    edge_a = face_vertices[:, 1] - face_vertices[:, 0]
    edge_b = face_vertices[:, 2] - face_vertices[:, 0]
    cross = np.cross(edge_a, edge_b)
    double_area = np.linalg.norm(cross, axis=1)
    face_areas = (0.5 * double_area).astype(np.float32)
    safe_norm = np.where(double_area > 0.0, double_area, 1.0)
    face_normals = (cross / safe_norm[:, None]).astype(np.float32)
    face_centroids = face_vertices.mean(axis=1).astype(np.float32)
    return face_centroids, face_normals, face_areas


def convert_obj_to_arrays(
    *,
    obj_path: str | Path,
    town_id: str,
    conversion_command: str,
    vertex_dtype: str = "float32",
    face_dtype: str = "int32",
) -> ConvertedMesh:
    obj_path = Path(obj_path).expanduser().resolve()
    started = time.time()
    mesh = _load_obj_mesh(obj_path)

    vertices = np.asarray(mesh.vertices, dtype=np.dtype(vertex_dtype))
    faces_dtype = np.int32 if face_dtype == "int32" else np.int64
    faces = np.asarray(mesh.faces, dtype=faces_dtype)
    face_centroids, face_normals, face_areas = compute_face_geometry(vertices, faces)

    bbox = {
        "min": vertices.min(axis=0).astype(float).tolist(),
        "max": vertices.max(axis=0).astype(float).tolist(),
        "extent": (vertices.max(axis=0) - vertices.min(axis=0)).astype(float).tolist(),
    }
    meta = {
        "town_id": town_id,
        "source_obj_path": str(obj_path),
        "num_vertices": int(len(vertices)),
        "num_faces": int(len(faces)),
        "vertex_dtype": str(vertices.dtype),
        "face_dtype": str(faces.dtype),
        "conversion_command": conversion_command,
        "conversion_time_sec": float(time.time() - started),
        "converted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return ConvertedMesh(
        vertices=vertices,
        faces=faces,
        face_centroids=face_centroids,
        face_normals=face_normals,
        face_areas=face_areas,
        bbox=bbox,
        meta=meta,
    )


def save_converted_mesh(output_dir: str | Path, converted: ConvertedMesh) -> dict[str, str]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vertices_path = out_dir / "vertices.npy"
    faces_path = out_dir / "faces.npy"
    centroids_path = out_dir / "face_centroids.npy"
    normals_path = out_dir / "face_normals.npy"
    areas_path = out_dir / "face_areas.npy"
    bbox_path = out_dir / "bbox.json"
    meta_path = out_dir / "mesh_meta.json"

    np.save(vertices_path, converted.vertices, allow_pickle=False)
    np.save(faces_path, converted.faces, allow_pickle=False)
    np.save(centroids_path, converted.face_centroids, allow_pickle=False)
    np.save(normals_path, converted.face_normals, allow_pickle=False)
    np.save(areas_path, converted.face_areas, allow_pickle=False)
    dump_json(bbox_path, converted.bbox, indent=2)
    dump_json(meta_path, converted.meta, indent=2)

    return {
        "vertices": str(vertices_path),
        "faces": str(faces_path),
        "face_centroids": str(centroids_path),
        "face_normals": str(normals_path),
        "face_areas": str(areas_path),
        "bbox": str(bbox_path),
        "mesh_meta": str(meta_path),
    }


def convert_obj_to_cache(
    *,
    obj_path: str | Path,
    out_dir: str | Path,
    town_id: str,
    conversion_command: str,
    vertex_dtype: str = "float32",
    face_dtype: str = "int32",
) -> ConvertedMesh:
    converted = convert_obj_to_arrays(
        obj_path=obj_path,
        town_id=town_id,
        conversion_command=conversion_command,
        vertex_dtype=vertex_dtype,
        face_dtype=face_dtype,
    )
    save_converted_mesh(out_dir, converted)
    return converted
