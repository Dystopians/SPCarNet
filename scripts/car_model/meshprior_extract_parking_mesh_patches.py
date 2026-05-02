"""Extract local mesh patches for gated parking MeshPrior targets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_triangle_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    state = torch.load(path, map_location="cpu")
    vertices = state.get("triangles_points")
    faces = state.get("_triangle_indices")
    if vertices is None or faces is None:
        raise KeyError("checkpoint must contain triangles_points and _triangle_indices")
    return np.asarray(vertices.detach().cpu(), dtype=np.float32), np.asarray(faces.detach().cpu(), dtype=np.int64)


def _expanded_bbox(cluster: dict[str, Any], expansion: float) -> tuple[np.ndarray, np.ndarray]:
    bbox_min = np.asarray(cluster["bbox3d_min"], dtype=np.float32)
    bbox_max = np.asarray(cluster["bbox3d_max"], dtype=np.float32)
    extent = np.maximum(bbox_max - bbox_min, 1e-6)
    return bbox_min - extent * expansion, bbox_max + extent * expansion


def _compact_patch(vertices: np.ndarray, faces: np.ndarray, face_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_faces = faces[face_ids]
    vertex_ids = np.unique(selected_faces.reshape(-1))
    remap = {int(v): i for i, v in enumerate(vertex_ids.tolist())}
    compact_faces = np.asarray([[remap[int(v)] for v in face] for face in selected_faces], dtype=np.int64)
    compact_vertices = vertices[vertex_ids]
    return compact_vertices, compact_faces, vertex_ids.astype(np.int64)


def _write_npz(
    path: Path,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    original_vertex_indices: np.ndarray,
    original_face_indices: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        original_vertex_indices=np.asarray(original_vertex_indices, dtype=np.int64),
        original_face_indices=np.asarray(original_face_indices, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    action_plan = _load_json(Path(args.action_plan))
    consolidated = _load_json(Path(args.consolidated_regions))
    clusters = {str(c["cluster_id"]): c for c in consolidated.get("clusters", [])}
    vertices, faces = _load_triangle_state(Path(args.triangle_state))
    centroids = vertices[faces].mean(axis=1)
    out = Path(args.output_dir)
    patch_dir = out / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for target in action_plan.get("mesh_extraction_targets", []):
        region_id = str(target["region_id"])
        cluster = clusters[region_id]
        bbox_min, bbox_max = _expanded_bbox(cluster, args.bbox_expansion)
        mask = np.all((centroids >= bbox_min) & (centroids <= bbox_max), axis=1)
        face_ids = np.nonzero(mask)[0].astype(np.int64)
        if args.max_faces_per_patch > 0 and len(face_ids) > args.max_faces_per_patch:
            center = np.asarray(cluster["centroid3d"], dtype=np.float32)
            order = np.argsort(np.linalg.norm(centroids[face_ids] - center[None, :], axis=1))
            face_ids = face_ids[order[: args.max_faces_per_patch]]
        patch_vertices, patch_faces, vertex_ids = _compact_patch(vertices, faces, face_ids) if len(face_ids) else (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
        patch_path = patch_dir / f"{region_id}.npz"
        metadata = {
            "region_id": region_id,
            "proposal_types": list(target.get("proposal_types", [])),
            "source_triangle_state": str(args.triangle_state),
            "bbox_expansion": float(args.bbox_expansion),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "geometry_edited": False,
        }
        _write_npz(
            patch_path,
            vertices=patch_vertices,
            faces=patch_faces,
            original_vertex_indices=vertex_ids,
            original_face_indices=face_ids,
            metadata=metadata,
        )
        rows.append(
            {
                "region_id": region_id,
                "proposal_types": ",".join(target.get("proposal_types", [])),
                "patch_path": str(patch_path),
                "face_count": int(len(face_ids)),
                "vertex_count": int(len(vertex_ids)),
                "bbox_expansion": float(args.bbox_expansion),
                "view_count": int(target.get("view_count", 0)),
                "sparse_point_count_sum": int(target.get("sparse_point_count_sum", 0)),
            }
        )

    summary = {
        "source_action_plan": str(args.action_plan),
        "source_triangle_state": str(args.triangle_state),
        "patch_count": len(rows),
        "nonempty_patch_count": sum(1 for r in rows if int(r["face_count"]) > 0),
        "total_patch_faces": sum(int(r["face_count"]) for r in rows),
        "min_patch_faces": min((int(r["face_count"]) for r in rows), default=0),
        "max_patch_faces": max((int(r["face_count"]) for r in rows), default=0),
        "geometry_edited": False,
        "patches": rows,
        "notes": [
            "Patches are copied from the trained triangle checkpoint and retain original face/vertex indices.",
            "No source model geometry is modified.",
            "These patches enable later before/after scene gates with rollback.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "mesh_patch_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out / "mesh_patch_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "region_id",
            "proposal_types",
            "patch_path",
            "face_count",
            "vertex_count",
            "bbox_expansion",
            "view_count",
            "sparse_point_count_sum",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "mesh_patch_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Parking Mesh Patch Extraction Report\n\n")
        f.write("- geometry edited: `false`\n")
        f.write(f"- patches: `{summary['patch_count']}`\n")
        f.write(f"- nonempty patches: `{summary['nonempty_patch_count']}`\n")
        f.write(f"- total patch faces: `{summary['total_patch_faces']}`\n")
        f.write(f"- patch face range: `{summary['min_patch_faces']}` - `{summary['max_patch_faces']}`\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract local triangle patches for gated parking regions.")
    parser.add_argument("--action_plan", default="outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/action_plan.json")
    parser.add_argument("--consolidated_regions", default="outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions.json")
    parser.add_argument(
        "--triangle_state",
        default="outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model/point_cloud/iteration_200/point_cloud_state_dict.pt",
    )
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/parking_phone_tiny/mesh_patches")
    parser.add_argument("--bbox_expansion", type=float, default=0.5)
    parser.add_argument("--max_faces_per_patch", type=int, default=0)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "patch_count": result["patch_count"],
                "nonempty_patch_count": result["nonempty_patch_count"],
                "total_patch_faces": result["total_patch_faces"],
                "face_range": [result["min_patch_faces"], result["max_patch_faces"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
