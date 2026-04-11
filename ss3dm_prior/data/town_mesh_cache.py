"""Read and query binary town mesh caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ss3dm_prior.utils.io import load_json


@dataclass
class TownMeshCache:
    cache_dir: Path
    vertices: np.ndarray
    faces: np.ndarray
    face_centroids: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    bbox: dict[str, Any]
    mesh_meta: dict[str, Any]
    spatial_index_hint: str = "face_centroids_linear_scan"

    def query_faces_in_radius(
        self,
        center: np.ndarray | list[float] | tuple[float, float, float],
        radius: float,
        margin: float = 0.0,
        coordinate_scale: float = 1.0,
    ) -> np.ndarray:
        center = np.asarray(center, dtype=np.float32).reshape(1, 3)
        effective_radius = float(radius) + float(margin)
        scaled_centroids = self.face_centroids * float(coordinate_scale)
        squared_dist = np.sum((scaled_centroids - center) ** 2, axis=1)
        return squared_dist <= effective_radius**2

    def build_local_mesh_from_face_mask(
        self,
        face_mask: np.ndarray,
        coordinate_scale: float = 1.0,
    ) -> dict[str, np.ndarray]:
        face_mask = np.asarray(face_mask, dtype=bool)
        selected_faces = self.faces[face_mask]
        if len(selected_faces) == 0:
            return {
                "vertices": np.zeros((0, 3), dtype=self.vertices.dtype),
                "faces": np.zeros((0, 3), dtype=self.faces.dtype),
                "face_centroids": np.zeros((0, 3), dtype=self.face_centroids.dtype),
                "face_normals": np.zeros((0, 3), dtype=self.face_normals.dtype),
                "face_areas": np.zeros((0,), dtype=self.face_areas.dtype),
                "vertex_indices": np.zeros((0,), dtype=self.faces.dtype),
                "face_indices": np.zeros((0,), dtype=np.int64),
            }

        unique_vertex_indices, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
        local_faces = inverse.reshape(-1, 3).astype(self.faces.dtype)
        face_indices = np.nonzero(face_mask)[0].astype(np.int64)
        scale = float(coordinate_scale)
        return {
            "vertices": (self.vertices[unique_vertex_indices] * scale).astype(self.vertices.dtype),
            "faces": local_faces,
            "face_centroids": (self.face_centroids[face_indices] * scale).astype(self.face_centroids.dtype),
            "face_normals": self.face_normals[face_indices],
            "face_areas": (self.face_areas[face_indices] * (scale**2)).astype(self.face_areas.dtype),
            "vertex_indices": unique_vertex_indices.astype(self.faces.dtype),
            "face_indices": face_indices,
        }


def load_town_mesh_cache(cache_dir: str | Path, mmap: bool = True) -> TownMeshCache:
    cache_dir = Path(cache_dir).expanduser().resolve()
    mmap_mode = "r" if mmap else None
    return TownMeshCache(
        cache_dir=cache_dir,
        vertices=np.load(cache_dir / "vertices.npy", mmap_mode=mmap_mode),
        faces=np.load(cache_dir / "faces.npy", mmap_mode=mmap_mode),
        face_centroids=np.load(cache_dir / "face_centroids.npy", mmap_mode=mmap_mode),
        face_normals=np.load(cache_dir / "face_normals.npy", mmap_mode=mmap_mode),
        face_areas=np.load(cache_dir / "face_areas.npy", mmap_mode=mmap_mode),
        bbox=load_json(cache_dir / "bbox.json"),
        mesh_meta=load_json(cache_dir / "mesh_meta.json"),
    )
