"""Synthetic mesh damage utilities for MeshPrior benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DamagedMesh:
    vertices: np.ndarray
    faces: np.ndarray
    valid_face_mask: np.ndarray
    floater_face_mask: np.ndarray
    removed_face_indices: list[int]
    damage_type: str


def make_box_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-1, -0.5, -0.25],
            [1, -0.5, -0.25],
            [1, 0.5, -0.25],
            [-1, 0.5, -0.25],
            [-1, -0.5, 0.25],
            [1, -0.5, 0.25],
            [1, 0.5, 0.25],
            [-1, 0.5, 0.25],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def damage_mesh_local_hole(vertices: np.ndarray, faces: np.ndarray, remove_count: int = 2) -> DamagedMesh:
    remove = list(range(min(remove_count, len(faces))))
    keep = np.ones(len(faces), dtype=bool)
    keep[remove] = False
    damaged_faces = faces[keep]
    return DamagedMesh(
        vertices=vertices.copy(),
        faces=damaged_faces.copy(),
        valid_face_mask=np.ones(len(damaged_faces), dtype=bool),
        floater_face_mask=np.zeros(len(damaged_faces), dtype=bool),
        removed_face_indices=remove,
        damage_type="local_hole",
    )


def add_floater_triangles(vertices: np.ndarray, faces: np.ndarray, offset: float = 4.0) -> DamagedMesh:
    floater_vertices = np.asarray([[offset, offset, offset], [offset + 0.5, offset, offset], [offset, offset + 0.5, offset]], dtype=np.float32)
    out_vertices = np.concatenate([vertices, floater_vertices], axis=0)
    floater_face = np.asarray([[len(vertices), len(vertices) + 1, len(vertices) + 2]], dtype=np.int64)
    out_faces = np.concatenate([faces, floater_face], axis=0)
    floater_mask = np.zeros(len(out_faces), dtype=bool)
    floater_mask[-1] = True
    return DamagedMesh(
        vertices=out_vertices,
        faces=out_faces,
        valid_face_mask=~floater_mask,
        floater_face_mask=floater_mask,
        removed_face_indices=[],
        damage_type="floater",
    )


def perturb_vertices(vertices: np.ndarray, faces: np.ndarray, sigma: float = 0.03, seed: int = 0) -> DamagedMesh:
    rng = np.random.default_rng(seed)
    out_vertices = vertices + rng.normal(0.0, sigma, size=vertices.shape).astype(np.float32)
    return DamagedMesh(
        vertices=out_vertices,
        faces=faces.copy(),
        valid_face_mask=np.ones(len(faces), dtype=bool),
        floater_face_mask=np.zeros(len(faces), dtype=bool),
        removed_face_indices=[],
        damage_type="vertex_noise",
    )


def make_density_imbalance(vertices: np.ndarray, faces: np.ndarray) -> DamagedMesh:
    out_faces = np.concatenate([faces, faces[: min(2, len(faces))]], axis=0)
    return DamagedMesh(
        vertices=vertices.copy(),
        faces=out_faces.copy(),
        valid_face_mask=np.ones(len(out_faces), dtype=bool),
        floater_face_mask=np.zeros(len(out_faces), dtype=bool),
        removed_face_indices=[],
        damage_type="density_imbalance",
    )


def compute_hole_boundary_metrics(faces: np.ndarray) -> dict[str, float]:
    from collections import defaultdict

    counts: dict[tuple[int, int], int] = defaultdict(int)
    for face in faces:
        for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            counts[tuple(sorted((int(u), int(v))))] += 1
    boundary = sum(1 for c in counts.values() if c == 1)
    total = max(len(counts), 1)
    return {"boundary_edge_count": float(boundary), "hole_boundary_score": float(boundary / total)}
