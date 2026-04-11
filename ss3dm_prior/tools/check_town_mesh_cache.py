"""Validate one binary town mesh cache directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ss3dm_prior.data.town_mesh_cache import load_town_mesh_cache


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check binary town mesh cache files and print a summary."
    )
    parser.add_argument("--cache_dir", required=True, help="Path to one town cache directory.")
    return parser


def main() -> int:
    args = make_argparser().parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache = load_town_mesh_cache(cache_dir, mmap=True)

    warnings: list[str] = []
    if cache.vertices.ndim != 2 or cache.vertices.shape[1] != 3:
        warnings.append(f"unexpected vertices shape: {cache.vertices.shape}")
    if cache.faces.ndim != 2 or cache.faces.shape[1] != 3:
        warnings.append(f"unexpected faces shape: {cache.faces.shape}")
    if len(cache.faces) != len(cache.face_centroids):
        warnings.append("face_centroids length mismatch")
    if len(cache.faces) != len(cache.face_normals):
        warnings.append("face_normals length mismatch")
    if len(cache.faces) != len(cache.face_areas):
        warnings.append("face_areas length mismatch")
    if np.any(cache.faces < 0) or np.any(cache.faces >= len(cache.vertices)):
        warnings.append("face indices out of range")

    bbox_min = np.asarray(cache.bbox["min"], dtype=np.float32)
    bbox_max = np.asarray(cache.bbox["max"], dtype=np.float32)
    computed_min = np.asarray(cache.vertices.min(axis=0), dtype=np.float32)
    computed_max = np.asarray(cache.vertices.max(axis=0), dtype=np.float32)
    if not np.allclose(bbox_min, computed_min, atol=1e-4):
        warnings.append("bbox min mismatch")
    if not np.allclose(bbox_max, computed_max, atol=1e-4):
        warnings.append("bbox max mismatch")

    print(f"town_id: {cache.mesh_meta.get('town_id')}")
    print(f"cache_dir: {cache_dir}")
    print(f"num_vertices: {len(cache.vertices)}")
    print(f"num_faces: {len(cache.faces)}")
    print(f"vertex_dtype: {cache.vertices.dtype}")
    print(f"face_dtype: {cache.faces.dtype}")
    print(f"bbox_min: {computed_min.astype(float).tolist()}")
    print(f"bbox_max: {computed_max.astype(float).tolist()}")
    print(f"spatial_index_hint: {cache.spatial_index_hint}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("warnings: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
