from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from .csef_types import CSEFBuildResult, CSEFRegion, CSEFSample


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = Path(path)
    try:
        import trimesh

        loaded = trimesh.load(mesh_path, process=False)
        if isinstance(loaded, trimesh.Scene):
            meshes = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError(f"No mesh geometry in scene: {mesh_path}")
            loaded = trimesh.util.concatenate(meshes)
        return np.asarray(loaded.vertices, dtype=np.float64), np.asarray(loaded.faces, dtype=np.int64)
    except Exception:
        return _load_ascii_ply(mesh_path)


def _load_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"Unsupported mesh format without trimesh: {path}")
    vertex_count = 0
    face_count = 0
    header_end = None
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
        if len(parts) == 3 and parts[:2] == ["element", "face"]:
            face_count = int(parts[2])
        if line.strip() == "end_header":
            header_end = i + 1
            break
    if header_end is None:
        raise ValueError(f"PLY has no header end: {path}")
    vertices = []
    for line in lines[header_end : header_end + vertex_count]:
        x, y, z = line.split()[:3]
        vertices.append((float(x), float(y), float(z)))
    faces = []
    start = header_end + vertex_count
    for line in lines[start : start + face_count]:
        parts = line.split()
        if int(parts[0]) != 3:
            continue
        faces.append((int(parts[1]), int(parts[2]), int(parts[3])))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def write_ascii_ply(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in vertices:
            f.write(f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def triangle_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,))
    tri = vertices[faces]
    centroids = tri.mean(axis=1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    areas = 0.5 * norm
    normals = np.zeros_like(cross)
    valid = norm > 1e-12
    normals[valid] = cross[valid] / norm[valid, None]
    return centroids, normals, areas


def edge_ownership(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            owners[tuple(sorted((int(a), int(b))))].append(fi)
    return owners


def connected_components(faces: np.ndarray, edge_to_faces: dict[tuple[int, int], list[int]]) -> list[list[int]]:
    neighbors: list[set[int]] = [set() for _ in range(len(faces))]
    for owners in edge_to_faces.values():
        if len(owners) < 2:
            continue
        for a in owners:
            for b in owners:
                if a != b:
                    neighbors[a].add(b)
    seen = np.zeros((len(faces),), dtype=bool)
    components: list[list[int]] = []
    for start in range(len(faces)):
        if seen[start]:
            continue
        queue: deque[int] = deque([start])
        seen[start] = True
        comp = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in neighbors[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    queue.append(nxt)
        components.append(comp)
    return components


def _normalize(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < eps:
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def build_csef(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    scene_model: str = "unknown",
    scene_source: str = "mesh",
    mesh_path: str = "",
    external_evidence_available: bool = False,
) -> tuple[CSEFBuildResult, list[CSEFSample]]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    centroids, normals, areas = triangle_geometry(vertices, faces)
    edge_to_faces = edge_ownership(faces)
    components = connected_components(faces, edge_to_faces)
    face_to_component = np.zeros((len(faces),), dtype=np.int64)
    for ci, comp in enumerate(components):
        face_to_component[np.asarray(comp, dtype=np.int64)] = ci

    boundary_counts = np.zeros((len(faces),), dtype=np.float64)
    for owners in edge_to_faces.values():
        if len(owners) == 1:
            boundary_counts[owners[0]] += 1.0
    boundary_score = boundary_counts / 3.0
    area_norm = _normalize(areas)
    comp_sizes = np.asarray([len(comp) for comp in components], dtype=np.float64)
    max_comp = float(max(comp_sizes.max(), 1.0)) if len(comp_sizes) else 1.0
    samples: list[CSEFSample] = []
    regions: list[CSEFRegion] = []

    for ci, comp in enumerate(components):
        comp_arr = np.asarray(comp, dtype=np.int64)
        comp_size_ratio = len(comp) / max_comp
        is_small_component = comp_size_ratio < 0.25 and len(components) > 1
        region_id = f"component_{ci:04d}"
        comp_boundary = boundary_score[comp_arr]
        comp_area = areas[comp_arr]
        comp_centroids = centroids[comp_arr]
        bbox_min = comp_centroids.min(axis=0) if len(comp_centroids) else np.zeros(3)
        bbox_max = comp_centroids.max(axis=0) if len(comp_centroids) else np.zeros(3)
        defect_candidates: list[str] = []
        if is_small_component:
            defect_candidates.append("FLOATER_COMPONENT")
        if float(np.mean(comp_boundary)) > 0.15:
            defect_candidates.append("SMALL_BOUNDARY_HOLE")
        if float(np.max(comp_boundary)) > 0.6 and float(np.sum(comp_area)) > float(np.median(areas) * 4.0 if len(areas) else 0.0):
            defect_candidates.append("GIANT_GROUND_VOID_CANDIDATE")
        if not defect_candidates:
            defect_candidates.append("SUPPORTED_SURFACE")

        boundary_loop_ids = [f"{region_id}_boundary"] if float(np.max(comp_boundary, initial=0.0)) > 0.0 else []
        region_stats = {
            "face_count": float(len(comp)),
            "area_sum": float(np.sum(comp_area)),
            "mean_boundary_edge_score": float(np.mean(comp_boundary)) if len(comp_boundary) else 0.0,
            "max_boundary_edge_score": float(np.max(comp_boundary)) if len(comp_boundary) else 0.0,
            "component_size_ratio": float(comp_size_ratio),
        }
        regions.append(
            CSEFRegion(
                region_id=region_id,
                defect_type_candidates=defect_candidates,
                bbox={"min": bbox_min.tolist(), "max": bbox_max.tolist()},
                boundary_loop_ids=boundary_loop_ids,
                mesh_face_indices=[int(x) for x in comp],
                image_evidence_refs=[],
                sparse_point_refs=[],
                summary_stats=region_stats,
            )
        )

    for fi in range(len(faces)):
        ci = int(face_to_component[fi])
        comp_ratio = len(components[ci]) / max_comp
        is_small_component = comp_ratio < 0.25 and len(components) > 1
        positive = float(np.clip(0.65 + 0.25 * comp_ratio - 0.55 * boundary_score[fi], 0.0, 1.0))
        if external_evidence_available:
            positive = float(np.clip(positive + 0.15, 0.0, 1.0))
        debt = float(np.clip(0.10 + 0.95 * boundary_score[fi] + 0.10 * area_norm[fi], 0.0, 1.0))
        if is_small_component:
            positive = float(min(positive, 0.2))
            debt = float(max(debt, 0.55))
        uncertainty = float(np.clip((0.55 if not external_evidence_available else 0.25) + 0.35 * boundary_score[fi], 0.0, 1.0))
        if is_small_component:
            uncertainty = float(max(uncertainty, 0.9))
        notes = []
        if boundary_score[fi] > 0:
            notes.append("boundary_edge_supported_debt")
        if is_small_component:
            notes.append("small_component_low_support")
        samples.append(
            CSEFSample(
                sample_id=f"face_{fi:06d}",
                position=tuple(float(x) for x in centroids[fi]),
                normal=tuple(float(x) for x in normals[fi]),
                region_id=f"component_{ci:04d}",
                positive_surface_evidence=positive,
                negative_free_space_evidence=0.0,
                explanation_debt=debt,
                prior_support=0.0,
                topology_cost=float(areas[fi]),
                uncertainty=uncertainty,
                evidence_sources=["mesh_topology", "boundary_edges", "connected_components"],
                notes=notes,
            )
        )

    summary: dict[str, Any] = {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "region_count": int(len(regions)),
        "mean_explanation_debt": float(np.mean([s.explanation_debt for s in samples])) if samples else 0.0,
        "max_explanation_debt": float(np.max([s.explanation_debt for s in samples])) if samples else 0.0,
        "mean_uncertainty": float(np.mean([s.uncertainty for s in samples])) if samples else 0.0,
        "boundary_face_count": int(np.sum(boundary_score > 0.0)),
        "external_evidence_available": bool(external_evidence_available),
    }
    result = CSEFBuildResult(
        scene_model=scene_model,
        scene_source=scene_source,
        mesh_path=mesh_path,
        regions=regions,
        global_summary=summary,
    )
    return result, samples


def write_csef_outputs(result: CSEFBuildResult, samples: list[CSEFSample], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    positions = np.asarray([s.position for s in samples], dtype=np.float64)
    normals = np.asarray([s.normal for s in samples], dtype=np.float64)
    scalars = {
        "positive_surface_evidence": np.asarray([s.positive_surface_evidence for s in samples], dtype=np.float64),
        "negative_free_space_evidence": np.asarray([s.negative_free_space_evidence for s in samples], dtype=np.float64),
        "explanation_debt": np.asarray([s.explanation_debt for s in samples], dtype=np.float64),
        "prior_support": np.asarray([s.prior_support for s in samples], dtype=np.float64),
        "topology_cost": np.asarray([s.topology_cost for s in samples], dtype=np.float64),
        "uncertainty": np.asarray([s.uncertainty for s in samples], dtype=np.float64),
    }
    np.savez(out / "csef_samples.npz", positions=positions, normals=normals, **scalars)
    (out / "csef_regions.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    with (out / "csef_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in result.global_summary.items():
            writer.writerow([key, value])
    report = [
        "# CSEF Build Report",
        "",
        f"- scene model: `{result.scene_model}`",
        f"- scene source: `{result.scene_source}`",
        f"- mesh path: `{result.mesh_path}`",
        f"- samples: `{len(samples)}`",
        f"- regions: `{len(result.regions)}`",
        "",
        "## Global Summary",
        "",
    ]
    for key, value in result.global_summary.items():
        report.append(f"- `{key}`: `{value}`")
    report.extend(["", "## Regions", ""])
    for region in result.regions:
        report.append(
            f"- `{region.region_id}`: faces `{len(region.mesh_face_indices)}`, "
            f"candidates `{', '.join(region.defect_type_candidates)}`"
        )
    (out / "csef_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
