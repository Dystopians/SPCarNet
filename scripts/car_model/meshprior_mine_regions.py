"""Mine candidate object regions from a scene mesh.

This is intentionally conservative. It can run as a dry-run without a mesh or
segmentation artifacts and still writes the expected output contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.region_types import (
    ObjectCanonicalization,
    RegionEvidence,
    RegionMiningResult,
    SceneMeshRegion,
)


def _as_float_list(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(x, dtype=np.float64).reshape(-1)]


def discover_mesh_path(scene_model: str | Path) -> Path | None:
    path = Path(scene_model)
    if path.is_file() and path.suffix.lower() == ".ply":
        return path
    if not path.exists() or not path.is_dir():
        return None
    candidates = []
    preferred_tokens = ("fuse_post", "fuse", "mesh", "point_cloud")
    for ply in path.rglob("*.ply"):
        score = 100
        lower = ply.name.lower()
        for i, token in enumerate(preferred_tokens):
            if token in lower:
                score = min(score, i)
        candidates.append((score, len(ply.parts), ply))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1], str(t[2])))
    return candidates[0][2]


def discover_segmentation_artifacts(scene_model: str | Path, scene_source: str | Path) -> list[str]:
    roots = [Path(scene_model), Path(scene_source)]
    patterns = ("*mask*.json", "*mask*.png", "*segment*.json", "*segmentation*.json")
    found: list[str] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for pattern in patterns:
            found.extend(str(p) for p in root.rglob(pattern) if p.is_file())
    return sorted(set(found))[:128]


def load_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import trimesh

        mesh = trimesh.load(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate([g for g in mesh.dump() if isinstance(g, trimesh.Trimesh)])
        return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
    except Exception:
        return _load_ascii_ply(path)


def _load_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        header = []
        for line in f:
            header.append(line.rstrip("\n"))
            if line.strip() == "end_header":
                break
        vertex_count = 0
        face_count = 0
        for line in header:
            parts = line.split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
            if len(parts) == 3 and parts[:2] == ["element", "face"]:
                face_count = int(parts[2])
        vertices = []
        for _ in range(vertex_count):
            parts = f.readline().split()
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
        faces = []
        for _ in range(face_count):
            parts = f.readline().split()
            n = int(parts[0])
            if n != 3:
                continue
            faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.zeros((0,), dtype=np.float64)
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def connected_face_components(faces: np.ndarray) -> list[list[int]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(faces):
        for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_to_faces[tuple(sorted((int(u), int(v))))].append(fi)
    neighbors: list[list[int]] = [[] for _ in range(len(faces))]
    for owners in edge_to_faces.values():
        if len(owners) < 2:
            continue
        for a in owners:
            for b in owners:
                if a != b:
                    neighbors[a].append(b)
    seen = np.zeros((len(faces),), dtype=bool)
    components: list[list[int]] = []
    for start in range(len(faces)):
        if seen[start]:
            continue
        q: deque[int] = deque([start])
        seen[start] = True
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in neighbors[cur]:
                if not seen[nb]:
                    seen[nb] = True
                    q.append(nb)
        components.append(comp)
    return components


def boundary_edge_count(faces: np.ndarray, face_indices: Iterable[int]) -> int:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for fi in face_indices:
        face = faces[int(fi)]
        for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            counts[tuple(sorted((int(u), int(v))))] += 1
    return sum(1 for c in counts.values() if c == 1)


def car_likeness_from_bbox(extent: np.ndarray, surface_area: float, boundary_edges: int) -> tuple[float, list[str]]:
    notes: list[str] = []
    extent = np.maximum(np.asarray(extent, dtype=np.float64), 1e-9)
    dims = np.sort(extent)
    height = dims[0]
    width = dims[1]
    length = dims[2]
    elongation = length / max(width, 1e-9)
    flatness = height / max(width, 1e-9)
    compact_area = surface_area / max(length * width + length * height + width * height, 1e-9)
    score = 0.0
    if 1.1 <= elongation <= 4.5:
        score += 0.35
    else:
        notes.append(f"bbox_elongation_out_of_range={elongation:.3f}")
    if 0.15 <= flatness <= 1.4:
        score += 0.30
    else:
        notes.append(f"bbox_flatness_out_of_range={flatness:.3f}")
    if compact_area > 0.05:
        score += 0.20
    else:
        notes.append(f"surface_area_low={surface_area:.6f}")
    if boundary_edges < 2000:
        score += 0.15
    score = float(max(0.0, min(1.0, score)))
    return score, notes


def mine_regions_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    mesh_path: Path,
    segmentation_available: bool,
    min_triangles: int = 1,
    eligibility_threshold: float = 0.35,
) -> list[SceneMeshRegion]:
    areas = triangle_areas(vertices, faces)
    components = connected_face_components(faces)
    regions: list[SceneMeshRegion] = []
    for component_id, face_ids in enumerate(components):
        if len(face_ids) < min_triangles:
            continue
        comp_faces = faces[np.asarray(face_ids, dtype=np.int64)]
        unique_vertices = np.unique(comp_faces.reshape(-1))
        pts = vertices[unique_vertices]
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        extent = bbox_max - bbox_min
        centroid = pts.mean(axis=0)
        surface_area = float(areas[np.asarray(face_ids, dtype=np.int64)].sum())
        b_edges = boundary_edge_count(faces, face_ids)
        all_edges = max(3 * len(face_ids), 1)
        hole_score = float(b_edges / all_edges)
        density = float(len(unique_vertices) / max(surface_area, 1e-9))
        car_score, notes = car_likeness_from_bbox(extent, surface_area, b_edges)
        ground_reject = 1.0 if extent.min() < 0.05 * max(extent.max(), 1e-9) and extent.max() > 1.0 else 0.0
        enough_faces_for_prior = len(face_ids) >= 4
        if not enough_faces_for_prior:
            notes.append("too_few_triangles_for_posterior")
        eligible = bool(car_score >= eligibility_threshold and ground_reject < 0.5 and enough_faces_for_prior)
        evidence = RegionEvidence(
            segmentation_available=segmentation_available,
            segmentation_score=0.0,
            geometry_score=car_score,
            ground_rejection_score=float(ground_reject),
            observed_support_score=0.0,
            car_likeness_score=car_score,
            eligible_for_posterior=eligible,
            notes=notes,
        )
        canonicalization = ObjectCanonicalization(
            mode="bbox_center_unit_extent",
            center=_as_float_list(centroid),
            scale=float(max(np.linalg.norm(extent) * 0.5, 1e-9)),
            confidence=float(car_score),
            notes=["front_axis_unknown; M3 must refine orientation before posterior inference"],
        )
        regions.append(
            SceneMeshRegion(
                region_id=f"region_{component_id:04d}",
                source_mesh_path=str(mesh_path),
                component_id=component_id,
                face_indices=[int(x) for x in face_ids],
                triangle_count=int(len(face_ids)),
                vertex_count=int(len(unique_vertices)),
                bbox_min=_as_float_list(bbox_min),
                bbox_max=_as_float_list(bbox_max),
                bbox_extent=_as_float_list(extent),
                centroid=_as_float_list(centroid),
                surface_area=surface_area,
                vertex_density=density,
                boundary_edge_count=int(b_edges),
                connected_components=1,
                approximate_hole_boundary_score=hole_score,
                evidence=evidence,
                canonicalization=canonicalization,
            )
        )
    regions.sort(key=lambda r: (not r.evidence.eligible_for_posterior, -r.triangle_count, r.region_id))
    return regions


def write_outputs(result: RegionMiningResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "regions.json").open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
        f.write("\n")
    with (output_dir / "regions_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "region_id",
                "component_id",
                "triangle_count",
                "vertex_count",
                "surface_area",
                "bbox_extent_x",
                "bbox_extent_y",
                "bbox_extent_z",
                "boundary_edge_count",
                "hole_boundary_score",
                "car_likeness_score",
                "eligible_for_posterior",
            ]
        )
        for r in result.regions:
            writer.writerow(
                [
                    r.region_id,
                    r.component_id,
                    r.triangle_count,
                    r.vertex_count,
                    f"{r.surface_area:.9g}",
                    *[f"{v:.9g}" for v in r.bbox_extent],
                    r.boundary_edge_count,
                    f"{r.approximate_hole_boundary_score:.9g}",
                    f"{r.evidence.car_likeness_score:.9g}",
                    int(r.evidence.eligible_for_posterior),
                ]
            )
    eligible = sum(1 for r in result.regions if r.evidence.eligible_for_posterior)
    with (output_dir / "region_mining_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Region Mining Report\n\n")
        f.write(f"- mode: `{result.mode}`\n")
        f.write(f"- scene_model: `{result.scene_model}`\n")
        f.write(f"- scene_source: `{result.scene_source}`\n")
        f.write(f"- mesh_path: `{result.mesh_path}`\n")
        f.write(f"- regions: `{len(result.regions)}`\n")
        f.write(f"- eligible_for_posterior: `{eligible}`\n")
        f.write(f"- segmentation_artifacts: `{len(result.segmentation_artifacts)}`\n")
        if result.notes:
            f.write("\n## Notes\n\n")
            for note in result.notes:
                f.write(f"- {note}\n")


def run_region_mining(args: argparse.Namespace) -> RegionMiningResult:
    mesh_path = discover_mesh_path(args.scene_model)
    segmentation = discover_segmentation_artifacts(args.scene_model, args.scene_source)
    notes: list[str] = []
    regions: list[SceneMeshRegion] = []
    if not segmentation:
        notes.append("No segmentation artifacts found; used geometry-only dry heuristic.")
    if mesh_path is None:
        notes.append("No PLY mesh found. Dry-run emitted an empty region set.")
    else:
        vertices, faces = load_ply_mesh(mesh_path)
        if vertices.size == 0 or faces.size == 0:
            notes.append(f"Mesh `{mesh_path}` has no usable triangular geometry.")
        else:
            regions = mine_regions_from_mesh(
                vertices,
                faces,
                mesh_path=mesh_path,
                segmentation_available=bool(segmentation),
                min_triangles=max(1, int(args.min_triangles)),
                eligibility_threshold=float(args.eligibility_threshold),
            )
            notes.append(f"Loaded mesh with {len(vertices)} vertices and {len(faces)} faces.")
    result = RegionMiningResult(
        scene_model=str(args.scene_model),
        scene_source=str(args.scene_source),
        mode=str(args.mode),
        mesh_path=str(mesh_path) if mesh_path is not None else None,
        regions=regions,
        segmentation_artifacts=segmentation,
        notes=notes,
    )
    write_outputs(result, Path(args.output_dir))
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine MeshPrior scene/object regions.")
    parser.add_argument("--scene_model", required=True)
    parser.add_argument("--scene_source", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["dry_run"], default="dry_run")
    parser.add_argument("--min_triangles", type=int, default=1)
    parser.add_argument("--eligibility_threshold", type=float, default=0.35)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    result = run_region_mining(args)
    print(
        json.dumps(
            {
                "regions": len(result.regions),
                "eligible_for_posterior": sum(1 for r in result.regions if r.evidence.eligible_for_posterior),
                "mesh_path": result.mesh_path,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
