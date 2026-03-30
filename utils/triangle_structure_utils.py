from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class TriangleStructureCache:
    num_triangles: int
    topology_signature: Tuple[int, int, int]
    # Edge table
    unique_edges: torch.Tensor  # [E, 2] int64
    edge_counts: torch.Tensor  # [E] int64
    # For each edge occurrence from original 3*T edge list:
    edge_owner_tri: torch.Tensor  # [3T] int64
    edge_unique_id_per_occurrence: torch.Tensor  # [3T] int64
    # 1-ring neighbors for each triangle (python lists of triangle ids)
    tri_neighbors: List[List[int]]
    # Per-triangle edge-level tags
    boundary_edge_count: torch.Tensor  # [T] int32
    nonmanifold_edge_count: torch.Tensor  # [T] int32


@dataclass
class TriangleStructureMetrics:
    boundary_edge_count: torch.Tensor  # [T] int32
    is_boundary_triangle: torch.Tensor  # [T] bool
    nonmanifold_edge_count: torch.Tensor  # [T] int32
    is_nonmanifold_triangle: torch.Tensor  # [T] bool
    mean_abs_dihedral_rad: torch.Tensor  # [T] float32
    mean_abs_dihedral_deg: torch.Tensor  # [T] float32
    coplanar_neighbor_fraction: torch.Tensor  # [T] float32
    qem_like: torch.Tensor  # [T] float32
    flatness_score: torch.Tensor  # [T] float32, larger => flatter/redundant


def _topology_signature(triangle_indices: torch.Tensor) -> Tuple[int, int, int]:
    # Lightweight signature for cache validity.
    t = triangle_indices.detach()
    num_tri = int(t.shape[0])
    num_vals = int(t.numel())
    checksum = int(torch.sum(t.to(torch.int64)).item()) if num_vals > 0 else 0
    return (num_tri, num_vals, checksum)


def _build_cache(triangle_indices: torch.Tensor) -> TriangleStructureCache:
    tri = triangle_indices.to(torch.int64).contiguous()
    device = tri.device
    t = int(tri.shape[0])

    if t == 0:
        empty_i64 = torch.zeros((0,), dtype=torch.int64, device=device)
        empty_e = torch.zeros((0, 2), dtype=torch.int64, device=device)
        empty_i32 = torch.zeros((0,), dtype=torch.int32, device=device)
        return TriangleStructureCache(
            num_triangles=0,
            topology_signature=_topology_signature(tri),
            unique_edges=empty_e,
            edge_counts=empty_i64,
            edge_owner_tri=empty_i64,
            edge_unique_id_per_occurrence=empty_i64,
            tri_neighbors=[],
            boundary_edge_count=empty_i32,
            nonmanifold_edge_count=empty_i32,
        )

    e01 = tri[:, [0, 1]]
    e12 = tri[:, [1, 2]]
    e20 = tri[:, [2, 0]]
    edges = torch.cat([e01, e12, e20], dim=0)  # [3T,2]
    edges = torch.sort(edges, dim=1).values
    edge_owner_tri = torch.arange(t, device=device, dtype=torch.int64).repeat(3)

    unique_edges, inverse, counts = torch.unique(
        edges, dim=0, return_inverse=True, return_counts=True
    )

    boundary_occ = (counts[inverse] == 1).to(torch.int32)
    nonmanifold_occ = (counts[inverse] > 2).to(torch.int32)
    boundary_edge_count = torch.zeros((t,), dtype=torch.int32, device=device)
    nonmanifold_edge_count = torch.zeros((t,), dtype=torch.int32, device=device)
    boundary_edge_count.index_add_(0, edge_owner_tri, boundary_occ)
    nonmanifold_edge_count.index_add_(0, edge_owner_tri, nonmanifold_occ)

    tri_neighbors: List[List[int]] = [[] for _ in range(t)]
    # Group edge occurrences by unique edge id. Most edges have <=2 owners.
    order = torch.argsort(inverse)
    inv_sorted = inverse[order].detach().cpu()
    tri_sorted = edge_owner_tri[order].detach().cpu()
    edge_sorted = torch.arange(order.shape[0], dtype=torch.int64)

    start = 0
    n_occ = int(inv_sorted.numel())
    while start < n_occ:
        eid = int(inv_sorted[start].item())
        end = start + 1
        while end < n_occ and int(inv_sorted[end].item()) == eid:
            end += 1
        owners = tri_sorted[start:end].tolist()
        if len(owners) >= 2:
            # Build all neighbor relations for non-manifold edges too.
            for i in range(len(owners)):
                oi = int(owners[i])
                for j in range(len(owners)):
                    if i == j:
                        continue
                    oj = int(owners[j])
                    tri_neighbors[oi].append(oj)
        start = end

    # Deduplicate neighbors.
    tri_neighbors = [sorted(set(nbrs)) for nbrs in tri_neighbors]

    return TriangleStructureCache(
        num_triangles=t,
        topology_signature=_topology_signature(tri),
        unique_edges=unique_edges,
        edge_counts=counts,
        edge_owner_tri=edge_owner_tri,
        edge_unique_id_per_occurrence=inverse,
        tri_neighbors=tri_neighbors,
        boundary_edge_count=boundary_edge_count,
        nonmanifold_edge_count=nonmanifold_edge_count,
    )


def get_or_build_structure_cache(
    triangle_indices: torch.Tensor, cache: Optional[TriangleStructureCache] = None
) -> TriangleStructureCache:
    sig = _topology_signature(triangle_indices)
    if cache is not None and cache.topology_signature == sig:
        return cache
    return _build_cache(triangle_indices)


def _triangle_planes_and_areas(
    vertices: torch.Tensor, triangle_indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Returns:
    # - plane coefficients pi=[nx, ny, nz, d], n is unit normal [T,4]
    # - centroids [T,3]
    # - areas [T]
    tri = triangle_indices.to(torch.int64)
    pts = vertices[tri]  # [T,3,3]
    ab = pts[:, 1] - pts[:, 0]
    ac = pts[:, 2] - pts[:, 0]
    cross = torch.cross(ab, ac, dim=1)
    area = 0.5 * torch.linalg.norm(cross, dim=1)
    n = cross / torch.clamp(torch.linalg.norm(cross, dim=1, keepdim=True), min=1e-12)
    c = pts.mean(dim=1)
    d = -torch.sum(n * c, dim=1, keepdim=True)
    pi = torch.cat([n, d], dim=1)
    return pi, c, area


def compute_triangle_structure_metrics(
    vertices: torch.Tensor,
    triangle_indices: torch.Tensor,
    cache: Optional[TriangleStructureCache] = None,
    coplanar_angle_threshold_deg: float = 8.0,
    qem_weight_mode: str = "area",
    flatness_alpha: float = 0.5,
) -> Tuple[TriangleStructureMetrics, TriangleStructureCache]:
    """
    Compute local structure signals used by PRISM (without changing topology).

    Definitions:
    - mean_abs_dihedral: mean of abs(dihedral) over shared-edge neighbors.
    - coplanar_neighbor_fraction: fraction of 1-ring neighbors with |dihedral| <= threshold.
    - qem_like: local QEM-inspired residual at triangle centroid:
        For each triangle i, ring R(i)=self U 1-ring.
        pi_j = [n_j, d_j], x_i=[c_i,1],
        qem_like_i = sum_j w_j * (pi_j dot x_i)^2.
    - flatness_score: higher => flatter / more locally redundant approximation:
        flatness = 1 - [alpha * qem_norm + (1-alpha) * dihedral_norm],
        where qem_norm=q/(1+q), dihedral_norm=mean_abs_dihedral/pi.
    """
    tri = triangle_indices.to(torch.int64).contiguous()
    device = tri.device
    t = int(tri.shape[0])

    cache = get_or_build_structure_cache(triangle_indices=tri, cache=cache)

    if t == 0:
        zf = torch.zeros((0,), dtype=torch.float32, device=device)
        zi32 = torch.zeros((0,), dtype=torch.int32, device=device)
        zb = torch.zeros((0,), dtype=torch.bool, device=device)
        metrics = TriangleStructureMetrics(
            boundary_edge_count=zi32,
            is_boundary_triangle=zb,
            nonmanifold_edge_count=zi32,
            is_nonmanifold_triangle=zb,
            mean_abs_dihedral_rad=zf,
            mean_abs_dihedral_deg=zf,
            coplanar_neighbor_fraction=zf,
            qem_like=zf,
            flatness_score=zf,
        )
        return metrics, cache

    pi, centroids, areas = _triangle_planes_and_areas(vertices=vertices, triangle_indices=tri)
    normals = pi[:, :3]

    mean_abs_dihedral_rad = torch.zeros((t,), dtype=torch.float32, device=device)
    dihedral_count = torch.zeros((t,), dtype=torch.float32, device=device)

    # Compute dihedral over manifold edges (count==2).
    order = torch.argsort(cache.edge_unique_id_per_occurrence)
    inv_sorted = cache.edge_unique_id_per_occurrence[order].detach().cpu()
    tri_sorted = cache.edge_owner_tri[order].detach().cpu()
    start = 0
    n_occ = int(inv_sorted.numel())
    while start < n_occ:
        eid = int(inv_sorted[start].item())
        end = start + 1
        while end < n_occ and int(inv_sorted[end].item()) == eid:
            end += 1
        owners = tri_sorted[start:end].tolist()
        if len(owners) == 2:
            a = int(owners[0])
            b = int(owners[1])
            cos_ab = torch.clamp(torch.abs(torch.dot(normals[a], normals[b])), 0.0, 1.0)
            angle = torch.arccos(cos_ab)
            mean_abs_dihedral_rad[a] += angle
            mean_abs_dihedral_rad[b] += angle
            dihedral_count[a] += 1.0
            dihedral_count[b] += 1.0
        start = end

    mean_abs_dihedral_rad = mean_abs_dihedral_rad / torch.clamp(dihedral_count, min=1.0)
    mean_abs_dihedral_deg = mean_abs_dihedral_rad * (180.0 / torch.pi)

    coplanar_neighbor_fraction = torch.zeros((t,), dtype=torch.float32, device=device)
    cos_thr = float(torch.cos(torch.tensor(coplanar_angle_threshold_deg * torch.pi / 180.0)).item())
    for i in range(t):
        nbrs = cache.tri_neighbors[i]
        if len(nbrs) == 0:
            coplanar_neighbor_fraction[i] = 0.0
            continue
        nbr_t = torch.tensor(nbrs, dtype=torch.int64, device=device)
        cos_vals = torch.abs(torch.sum(normals[nbr_t] * normals[i].unsqueeze(0), dim=1))
        coplanar_neighbor_fraction[i] = torch.mean((cos_vals >= cos_thr).to(torch.float32))

    x_h = torch.cat(
        [centroids, torch.ones((t, 1), dtype=centroids.dtype, device=device)], dim=1
    )  # [T,4]
    if qem_weight_mode == "uniform":
        weights = torch.ones_like(areas)
    else:
        weights = torch.clamp(areas, min=1e-8)

    qem_like = torch.zeros((t,), dtype=torch.float32, device=device)
    for i in range(t):
        ring = [i] + cache.tri_neighbors[i]
        ring_t = torch.tensor(ring, dtype=torch.int64, device=device)
        # x^T (sum w p p^T) x == sum w (p.x)^2
        dot_vals = torch.sum(pi[ring_t] * x_h[i].unsqueeze(0), dim=1)
        qem_like[i] = torch.sum(weights[ring_t] * dot_vals * dot_vals)

    qem_norm = qem_like / (1.0 + qem_like)
    dihedral_norm = mean_abs_dihedral_rad / torch.pi
    alpha = float(max(0.0, min(flatness_alpha, 1.0)))
    flatness_score = 1.0 - (alpha * qem_norm + (1.0 - alpha) * dihedral_norm)
    flatness_score = torch.clamp(flatness_score, 0.0, 1.0)

    is_boundary_triangle = cache.boundary_edge_count > 0
    is_nonmanifold_triangle = cache.nonmanifold_edge_count > 0

    metrics = TriangleStructureMetrics(
        boundary_edge_count=cache.boundary_edge_count,
        is_boundary_triangle=is_boundary_triangle,
        nonmanifold_edge_count=cache.nonmanifold_edge_count,
        is_nonmanifold_triangle=is_nonmanifold_triangle,
        mean_abs_dihedral_rad=mean_abs_dihedral_rad,
        mean_abs_dihedral_deg=mean_abs_dihedral_deg,
        coplanar_neighbor_fraction=coplanar_neighbor_fraction,
        qem_like=qem_like,
        flatness_score=flatness_score,
    )
    return metrics, cache


def debug_print_triangle_structure(
    metrics: TriangleStructureMetrics,
    triangle_ids: Optional[torch.Tensor] = None,
    max_print: int = 10,
) -> None:
    """
    Debug helper that prints key structural signals for selected triangles.
    """
    t = int(metrics.flatness_score.numel())
    if t == 0:
        print("[TriStruct] empty mesh")
        return

    if triangle_ids is None:
        # Print top-flat triangles by default.
        k = min(int(max_print), t)
        _, ids = torch.topk(metrics.flatness_score, k=k, largest=True, sorted=True)
        triangle_ids = ids
    else:
        triangle_ids = triangle_ids[: max_print]

    print("[TriStruct] triangle diagnostics")
    for tid in triangle_ids.tolist():
        print(
            "  tri={} boundary={} nonmanifold={} dihedral_deg={:.4f} coplanar_frac={:.4f} qem_like={:.6e} flatness={:.4f}".format(
                int(tid),
                int(metrics.boundary_edge_count[tid].item()),
                int(metrics.nonmanifold_edge_count[tid].item()),
                float(metrics.mean_abs_dihedral_deg[tid].item()),
                float(metrics.coplanar_neighbor_fraction[tid].item()),
                float(metrics.qem_like[tid].item()),
                float(metrics.flatness_score[tid].item()),
            )
        )
