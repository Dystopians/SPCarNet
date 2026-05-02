#!/usr/bin/env python3
"""Audit local M25 datasets for MeshSplatting compatibility."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _count_files(path: Path, suffixes: set[str] | None = None, limit: int | None = None) -> int:
    if not path.is_dir():
        return 0
    count = 0
    for entry in path.rglob("*"):
        if entry.is_file() and (suffixes is None or entry.suffix in suffixes):
            count += 1
            if limit is not None and count >= limit:
                return count
    return count


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _human_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _has_colmap_model(scene: Path) -> bool:
    sparse = scene / "sparse" / "0"
    return (
        ((sparse / "cameras.bin").is_file() and (sparse / "images.bin").is_file())
        or ((sparse / "cameras.txt").is_file() and (sparse / "images.txt").is_file())
    ) and ((sparse / "points3D.bin").is_file() or (sparse / "points3D.txt").is_file())


def _image_dirs(scene: Path) -> list[dict[str, Any]]:
    out = []
    for child in sorted(scene.iterdir()) if scene.is_dir() else []:
        if child.is_dir() and child.name.startswith("images"):
            out.append({"name": child.name, "images": _count_files(child, IMAGE_EXTS)})
    return out


def audit_scene(scene: Path, dataset: str) -> dict[str, Any]:
    image_dirs = _image_dirs(scene)
    has_colmap = _has_colmap_model(scene)
    has_sparse = (scene / "sparse" / "0").is_dir()
    trainable = bool(has_colmap and any(item["images"] > 0 for item in image_dirs))
    status = "trainable_colmap" if trainable else "needs_conversion"
    if not image_dirs:
        status = "downloaded_non_scene_or_archive"
    return {
        "dataset": dataset,
        "scene": scene.name,
        "path": str(scene),
        "size": _human_bytes(_dir_size_bytes(scene)),
        "image_dirs": image_dirs,
        "has_sparse_0": has_sparse,
        "has_colmap_model": has_colmap,
        "trainable_with_current_loader": trainable,
        "status": status,
    }


def discover_scenes(dataset_root: Path, dataset: str) -> list[Path]:
    if not dataset_root.is_dir():
        return []
    direct_scenes = [
        child
        for child in sorted(dataset_root.iterdir())
        if child.is_dir() and (_image_dirs(child) or (child / "sparse" / "0").is_dir())
    ]
    if direct_scenes:
        return direct_scenes
    return [dataset_root]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", default="/data/peilincai/mesh_datasets")
    parser.add_argument("--output", default="outputs/carnet/meshprior/stage25_multidataset/dataset_audit.json")
    args = parser.parse_args()

    base = Path(args.base_dir)
    dataset_roots = {
        "mipnerf360": base / "mipnerf360",
        "tanks_and_temples": base / "tanks_and_temples",
        "tanks_and_temples_colmap": base / "tanks_and_temples_colmap",
        "eth3d": base / "eth3d",
        "eth3d_colmap": base / "eth3d_colmap",
    }

    rows = []
    for dataset, root in dataset_roots.items():
        for scene in discover_scenes(root, dataset):
            rows.append(audit_scene(scene, dataset))

    summary = {
        "base_dir": str(base),
        "datasets": {name: str(path) for name, path in dataset_roots.items()},
        "num_entries": len(rows),
        "num_trainable_colmap": sum(1 for row in rows if row["trainable_with_current_loader"]),
        "entries": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
