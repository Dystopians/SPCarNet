"""Guarded hole-fill proposal utilities for MeshPrior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class BoundaryLoop:
    vertex_indices: list[int]
    edge_indices: list[tuple[int, int]]
    closed: bool
    metadata: dict


@dataclass
class FillProposal:
    vertices_before: np.ndarray
    faces_before: np.ndarray
    vertices_after: np.ndarray
    faces_after: np.ndarray
    added_vertex_indices: list[int]
    added_face_indices: list[int]
    loop: BoundaryLoop
    confidence: float
    accepted: bool
    metadata: dict


def _as_mesh(mesh: tuple[np.ndarray, np.ndarray] | object) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, tuple):
        return np.asarray(mesh[0], dtype=np.float32), np.asarray(mesh[1], dtype=np.int64)
    return np.asarray(getattr(mesh, "vertices"), dtype=np.float32), np.asarray(getattr(mesh, "faces"), dtype=np.int64)


def _edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, dtype=np.int64):
        for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(u), int(v))))
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def find_boundary_loops(mesh: tuple[np.ndarray, np.ndarray] | object) -> list[BoundaryLoop]:
    """Return ordered boundary loops from one-face edges."""
    vertices, faces = _as_mesh(mesh)
    del vertices
    boundary_edges = [edge for edge, count in _edge_counts(faces).items() if count == 1]
    adjacency: dict[int, list[int]] = {}
    for u, v in boundary_edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    unused = {tuple(sorted(edge)) for edge in boundary_edges}
    loops: list[BoundaryLoop] = []
    while unused:
        start_edge = min(unused)
        start, nxt = start_edge
        loop = [start, nxt]
        unused.remove(start_edge)
        prev, cur = start, nxt
        closed = False
        while True:
            candidates = [v for v in adjacency.get(cur, []) if v != prev and tuple(sorted((cur, v))) in unused]
            if not candidates:
                if start in adjacency.get(cur, []):
                    closed = True
                break
            nxt = min(candidates)
            unused.remove(tuple(sorted((cur, nxt))))
            if nxt == start:
                closed = True
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        edge_indices = [tuple(sorted((loop[i], loop[(i + 1) % len(loop)]))) for i in range(len(loop))] if closed else []
        loops.append(
            BoundaryLoop(
                vertex_indices=[int(v) for v in loop],
                edge_indices=[(int(u), int(v)) for u, v in edge_indices],
                closed=closed,
                metadata={
                    "length": int(len(loop)),
                    "nonmanifold_boundary_vertices": int(sum(len(v) != 2 for v in adjacency.values())),
                },
            )
        )
    return loops


def _call_decoder(decoder: Callable[..., torch.Tensor], z: torch.Tensor | None, points: torch.Tensor) -> torch.Tensor:
    if z is None:
        return decoder(points).reshape(-1)
    zz = z
    if zz.ndim == 1:
        zz = zz.unsqueeze(0)
    return decoder(points.unsqueeze(0), zz).reshape(-1)


@torch.no_grad()
def extract_local_field_patch(
    decoder: Callable[..., torch.Tensor],
    z: torch.Tensor | None,
    local_bbox: tuple[np.ndarray, np.ndarray],
    resolution: int = 12,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    """Sample decoder logits on a local regular grid."""
    lo, hi = (np.asarray(local_bbox[0], dtype=np.float32), np.asarray(local_bbox[1], dtype=np.float32))
    axes = [np.linspace(float(lo[i]), float(hi[i]), int(resolution), dtype=np.float32) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    pts = torch.as_tensor(grid.reshape(-1, 3), dtype=torch.float32, device=torch.device(device))
    zz = z.to(torch.device(device)) if z is not None else None
    logits = _call_decoder(decoder, zz, pts).detach().cpu().numpy().reshape(grid.shape[:-1])
    return {"points": grid, "logits": logits, "support": 1.0 / (1.0 + np.exp(-logits))}


def score_hole_candidates(
    mesh: tuple[np.ndarray, np.ndarray] | object,
    boundary_loops: list[BoundaryLoop],
    region_evidence: dict | None = None,
) -> list[dict[str, float | int | bool]]:
    """Score boundary loops with conservative geometry and evidence checks."""
    vertices, _ = _as_mesh(mesh)
    evidence = region_evidence or {}
    mesh_extent = np.ptp(vertices, axis=0)
    mesh_diag = float(np.linalg.norm(mesh_extent) + 1e-12)
    out: list[dict[str, float | int | bool]] = []
    for idx, loop in enumerate(boundary_loops):
        pts = vertices[np.asarray(loop.vertex_indices, dtype=np.int64)] if loop.vertex_indices else np.zeros((0, 3), dtype=np.float32)
        loop_extent = np.ptp(pts, axis=0) if len(pts) else np.zeros(3, dtype=np.float32)
        loop_diag = float(np.linalg.norm(loop_extent))
        size_score = 1.0 - min(loop_diag / max(mesh_diag, 1e-12), 1.0)
        field_support = float(evidence.get("field_support", 1.0))
        uncertainty = float(evidence.get("uncertainty", 0.0))
        accepted = bool(loop.closed and len(loop.vertex_indices) >= 3 and field_support >= 0.45 and uncertainty < 0.75)
        out.append(
            {
                "loop_index": int(idx),
                "vertex_count": int(len(loop.vertex_indices)),
                "loop_diag": float(loop_diag),
                "mesh_diag": float(mesh_diag),
                "size_score": float(size_score),
                "field_support": field_support,
                "uncertainty": uncertainty,
                "accepted": accepted,
                "score": float(size_score * field_support * (1.0 - min(max(uncertainty, 0.0), 1.0))),
            }
        )
    return out


def clip_patch_to_hole_boundary(
    patch_vertices: np.ndarray,
    patch_faces: np.ndarray,
    boundary_vertices: np.ndarray,
    *,
    padding: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep patch vertices inside the padded boundary bbox."""
    verts = np.asarray(patch_vertices, dtype=np.float32)
    faces = np.asarray(patch_faces, dtype=np.int64)
    boundary = np.asarray(boundary_vertices, dtype=np.float32)
    lo = boundary.min(axis=0) - float(padding)
    hi = boundary.max(axis=0) + float(padding)
    inside = np.all((verts >= lo) & (verts <= hi), axis=1)
    keep_faces = np.all(inside[faces], axis=1) if len(faces) else np.zeros(0, dtype=bool)
    return verts, faces[keep_faces]


def _connected_component_count(vertices: np.ndarray, faces: np.ndarray) -> int:
    del vertices
    if len(faces) == 0:
        return 0
    parent = list(range(len(faces)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    owners: dict[int, int] = {}
    for fi, face in enumerate(faces):
        for v in face:
            v = int(v)
            if v in owners:
                union(fi, owners[v])
            else:
                owners[v] = fi
    return len({find(i) for i in range(len(faces))})


def build_fill_proposal(
    mesh: tuple[np.ndarray, np.ndarray] | object,
    boundary_loop: BoundaryLoop,
    decoder: Callable[..., torch.Tensor] | None = None,
    z: torch.Tensor | None = None,
    *,
    min_support: float = 0.45,
    uncertainty: float = 0.0,
    device: str | torch.device = "cpu",
) -> FillProposal:
    """Build a guarded fan-cap fill proposal for one boundary loop."""
    vertices, faces = _as_mesh(mesh)
    loop_ids = np.asarray(boundary_loop.vertex_indices, dtype=np.int64)
    loop_vertices = vertices[loop_ids]
    centroid = loop_vertices.mean(axis=0).astype(np.float32)
    support = 1.0
    if decoder is not None:
        pts = torch.as_tensor(np.concatenate([loop_vertices, centroid[None]], axis=0), dtype=torch.float32, device=torch.device(device))
        zz = z.to(torch.device(device)) if z is not None else None
        logits = _call_decoder(decoder, zz, pts)
        support = float(torch.sigmoid(logits).mean().detach().cpu().item())

    accepted = bool(boundary_loop.closed and len(loop_ids) >= 3 and support >= float(min_support) and float(uncertainty) < 0.75)
    new_vertices = np.concatenate([vertices, centroid[None]], axis=0)
    center_id = len(vertices)
    patch_faces = np.asarray([[int(loop_ids[i]), int(loop_ids[(i + 1) % len(loop_ids)]), center_id] for i in range(len(loop_ids))], dtype=np.int64)
    _, patch_faces = clip_patch_to_hole_boundary(new_vertices, patch_faces, loop_vertices, padding=1e-4)
    if not accepted:
        patch_faces = np.zeros((0, 3), dtype=np.int64)
        new_vertices = vertices.copy()
        added_vertices: list[int] = []
    else:
        added_vertices = [int(center_id)]
    new_faces = np.concatenate([faces, patch_faces], axis=0) if len(patch_faces) else faces.copy()
    added_faces = list(range(len(faces), len(new_faces)))
    return FillProposal(
        vertices_before=vertices,
        faces_before=faces,
        vertices_after=new_vertices,
        faces_after=new_faces,
        added_vertex_indices=added_vertices,
        added_face_indices=[int(x) for x in added_faces],
        loop=boundary_loop,
        confidence=float(support),
        accepted=bool(accepted and len(patch_faces) > 0),
        metadata={"field_support": float(support), "uncertainty": float(uncertainty), "patch_face_count": int(len(patch_faces))},
    )


def evaluate_fill_risk(
    proposal: FillProposal,
    *,
    free_space_violation_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float | bool]:
    before_boundary = sum(1 for c in _edge_counts(proposal.faces_before).values() if c == 1)
    after_boundary = sum(1 for c in _edge_counts(proposal.faces_after).values() if c == 1)
    before_components = _connected_component_count(proposal.vertices_before, proposal.faces_before)
    after_components = _connected_component_count(proposal.vertices_after, proposal.faces_after)
    out: dict[str, float | bool] = {
        "accepted": bool(proposal.accepted),
        "added_vertex_count": float(len(proposal.added_vertex_indices)),
        "added_face_count": float(len(proposal.added_face_indices)),
        "boundary_edge_count_before": float(before_boundary),
        "boundary_edge_count_after": float(after_boundary),
        "boundary_edge_delta": float(before_boundary - after_boundary),
        "component_count_before": float(before_components),
        "component_count_after": float(after_components),
        "component_count_delta": float(after_components - before_components),
        "free_space_violation_delta": 0.0,
    }
    if free_space_violation_fn is not None and proposal.added_vertex_indices:
        before = np.asarray(free_space_violation_fn(proposal.vertices_before), dtype=np.float64)
        after = np.asarray(free_space_violation_fn(proposal.vertices_after), dtype=np.float64)
        out["free_space_violation_delta"] = float(after.mean() - before.mean())
    return out
