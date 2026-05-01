#!/usr/bin/env python3
"""SP-CarNet Stage 1 object index builder.

Scans the existing whole-car patch cache and emits an object-level index that
groups patches by car/object identity. For the current MeshFleet v5 cache each
object maps to exactly one NPZ patch (verified during Stage 1 audit). The
builder still implements the multi-patch aggregation code path so the wrapper
is future-proof against schema variants where a single car spans multiple
patches.

See ``docs/car_model/spcarnet_stage1_object_cache_design.md`` for the contract.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "outputs/ss3dm_prior_car/meshfleet_car_cache_v5"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/carnet/spcarnet/object_index_v1.json"


def _load_split_yaml(path: Path) -> dict[str, str]:
    with path.open() as f:
        spec = yaml.safe_load(f)
    town_to_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for town in spec.get(f"{split}_towns", []) or []:
            town_to_split[town] = split
    return town_to_split


def _load_manifest(path: Path) -> dict[str, str | None]:
    if not path.is_file():
        return {}
    with path.open() as f:
        manifest = json.load(f)
    records = manifest.get("records", []) or []
    out: dict[str, str | None] = {}
    for rec in records:
        car_id = rec.get("car_id")
        if not car_id:
            continue
        local_path = rec.get("local_path")
        out[car_id] = local_path
    return out


def _read_patch_index(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _identity_transform() -> dict[str, Any]:
    return {
        "type": "identity",
        "center": [0.0, 0.0, 0.0],
        "scale": 1.0,
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def _pca_transform(npz_path: Path) -> dict[str, Any]:
    """Fallback PCA frame.

    Caveats documented in the design doc: eigenvector signs flip on near-symmetric
    cars and there is no robust front-vs-back convention. ``identity`` remains the
    recommended default.
    """
    data = np.load(npz_path)
    points = data["clean_points"].astype(np.float64)
    center = points.mean(axis=0)
    centred = points - center
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    rotation = eigvecs[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0
    projected = centred @ rotation
    sign_x = 1.0 if projected[:, 0].mean() >= 0 else -1.0
    rotation[:, 0] *= sign_x
    scale = float(2.0 * np.std(centred @ rotation[:, 0]))
    if scale <= 0.0 or not np.isfinite(scale):
        scale = 1.0
    return {
        "type": "pca",
        "center": [float(c) for c in center],
        "scale": scale,
        "rotation": [[float(v) for v in row] for row in rotation.tolist()],
    }


def _build_canonical_transform(mode: str, npz_path: Path) -> dict[str, Any]:
    if mode == "identity":
        return _identity_transform()
    if mode == "pca":
        return _pca_transform(npz_path)
    if mode == "mesh-axis":
        raise SystemExit(
            "[build_spcarnet_object_index] --canonicalization mesh-axis is deferred "
            "to Stage 2; Stage 1 supports identity (default) or pca only."
        )
    raise SystemExit(f"[build_spcarnet_object_index] unknown canonicalization: {mode}")


def _resolve_npz_path(record: dict[str, Any]) -> Path:
    """Patch records store ``patch_file`` as a directory (legacy) or a file."""
    raw = Path(record["patch_file"])
    if raw.is_file():
        return raw
    candidate = raw / f"{record['patch_id']}.npz"
    if candidate.is_file():
        return candidate
    return raw  # let downstream code surface the FileNotFoundError if any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch_cache_dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--split_config", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--canonicalization", choices=("identity", "pca", "mesh-axis"), default="identity")
    parser.add_argument("--limit", type=int, default=0, help="Cap to first N objects (smoke runs); 0 = full.")
    args = parser.parse_args(argv)

    cache_dir = Path(args.patch_cache_dir).resolve()
    split_yaml = Path(args.split_config) if args.split_config else cache_dir / "split_meshfleet_car.yaml"
    manifest_path = Path(args.manifest) if args.manifest else cache_dir / "source_mesh_manifest.json"
    patch_index_path = cache_dir / "patch_index.jsonl"
    output_path = Path(args.output).resolve()

    if not patch_index_path.is_file():
        raise SystemExit(f"patch_index.jsonl not found under {cache_dir}")

    started = time.time()
    print(f"[build] cache={cache_dir}", file=sys.stderr)
    print(f"[build] split_yaml={split_yaml}", file=sys.stderr)
    print(f"[build] manifest={manifest_path}", file=sys.stderr)

    town_to_split = _load_split_yaml(split_yaml) if split_yaml.is_file() else {}
    car_to_glb = _load_manifest(manifest_path)
    records = _read_patch_index(patch_index_path)
    print(f"[build] patch_records={len(records)}", file=sys.stderr)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[rec["sequence_id"]].append(rec)

    object_ids = sorted(grouped.keys())
    if args.limit and args.limit > 0:
        object_ids = object_ids[: args.limit]

    objects: list[dict[str, Any]] = []
    n_pseudo = 0
    cache_format_versions: set[int] = set()
    pseudo_log_path = output_path.with_name("pseudo_object_warnings.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pseudo_handle = pseudo_log_path.open("w") if False else None  # opened lazily

    for object_id in object_ids:
        patches = grouped[object_id]
        npz_paths = [_resolve_npz_path(p) for p in patches]
        primary = npz_paths[0]
        town_id = patches[0].get("town_id", "")
        split = town_to_split.get(town_id, "unknown")
        cache_format_versions.add(int(patches[0].get("patch_cache_format_version", -1)))
        canonical = _build_canonical_transform(args.canonicalization, primary)
        n_clean = int(patches[0].get("num_clean_points", 0))

        is_pseudo = len(patches) > 1
        if is_pseudo:
            n_pseudo += 1
            if pseudo_handle is None:
                pseudo_handle = pseudo_log_path.open("w")
            pseudo_handle.write(
                json.dumps({"object_id": object_id, "patch_count": len(patches), "town_id": town_id}) + "\n"
            )
            canonical["type"] = "pseudo-aggregated"

        objects.append(
            {
                "object_id": object_id,
                "split": split,
                "town_id": town_id,
                "patch_files": [str(p) for p in npz_paths],
                "patch_ids": [p["patch_id"] for p in patches],
                "canonical_transform": canonical,
                "source_mesh_path": car_to_glb.get(object_id),
                "n_clean_points": n_clean,
                "patch_format_version": int(patches[0].get("patch_cache_format_version", -1)),
            }
        )

    if pseudo_handle is not None:
        pseudo_handle.close()

    counts_per_object = [len(grouped[oid]) for oid in object_ids]
    splits_count = defaultdict(int)
    for obj in objects:
        splits_count[obj["split"]] += 1

    stats = {
        "n_objects": len(objects),
        "n_patches": int(sum(counts_per_object)),
        "patches_per_object": {
            "min": int(min(counts_per_object)) if counts_per_object else 0,
            "median": float(statistics.median(counts_per_object)) if counts_per_object else 0.0,
            "mean": float(statistics.mean(counts_per_object)) if counts_per_object else 0.0,
            "max": int(max(counts_per_object)) if counts_per_object else 0,
        },
        "splits": dict(splits_count),
        "scanner_pose_available": False,
        "symmetry_persisted": False,
        "occupancy_queries_available": True,
        "free_space_queries_available": True,
        "cache_format_versions": sorted(cache_format_versions),
        "n_pseudo_objects": n_pseudo,
        "canonicalization": args.canonicalization,
    }

    document = {
        "version": 1,
        "source_patch_index": str(patch_index_path),
        "source_split_yaml": str(split_yaml),
        "source_manifest": str(manifest_path),
        "stats": stats,
        "objects": objects,
    }

    with output_path.open("w") as f:
        json.dump(document, f, indent=2)

    elapsed = time.time() - started
    print(f"[build] wrote {output_path}", file=sys.stderr)
    print(
        f"[build] objects={stats['n_objects']} pseudo={stats['n_pseudo_objects']} "
        f"splits={stats['splits']} elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    if n_pseudo:
        print(f"[build] pseudo-object warnings -> {pseudo_log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
