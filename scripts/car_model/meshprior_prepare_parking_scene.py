"""Prepare a parking COLMAP scene view for MeshPrior experiments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import read_images_binary


def _count_files(path: Path, suffixes: tuple[str, ...] | None = None) -> int:
    if not path.is_dir():
        return 0
    total = 0
    for item in path.iterdir():
        if item.is_file() and (suffixes is None or item.suffix.lower() in suffixes):
            total += 1
    return total


def _safe_symlink(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst)


def audit_scene(scene_root: Path) -> dict:
    images_dir = scene_root / "images"
    sparse_dir = scene_root / "sparse/0"
    colmap_images = read_images_binary(sparse_dir / "images.bin")
    colmap_names = {img.name for img in colmap_images.values()}
    image_names = {p.name for p in images_dir.iterdir() if p.is_file()}
    split_path = sparse_dir / "split_outoftrain_v1.json"
    return {
        "scene_root": str(scene_root),
        "images_dir": str(images_dir),
        "sparse_dir": str(sparse_dir),
        "image_count": len(image_names),
        "colmap_image_count": len(colmap_names),
        "missing_image_files": sorted(colmap_names - image_names),
        "extra_image_files": sorted(image_names - colmap_names),
        "has_cameras_bin": (sparse_dir / "cameras.bin").is_file(),
        "has_images_bin": (sparse_dir / "images.bin").is_file(),
        "has_points3d_bin": (sparse_dir / "points3D.bin").is_file(),
        "has_points3d_ply": (sparse_dir / "points3D.ply").is_file(),
        "has_split_outoftrain": split_path.is_file(),
        "split_outoftrain": str(split_path) if split_path.is_file() else "",
        "segmentation_dense_count": _count_files(scene_root.parent / "SegmentationClass_dense_for_training", (".png",)),
        "ground_mask_count": _count_files(scene_root / "ground_masks", (".png",)),
    }


def run(args: argparse.Namespace) -> dict:
    scene_root = Path(args.scene_root).resolve()
    view = Path(args.output_view).resolve()
    audit = audit_scene(scene_root)
    if audit["missing_image_files"]:
        raise RuntimeError(f"COLMAP references missing image files: {audit['missing_image_files'][:5]}")
    _safe_symlink(scene_root / "images", view / "images", overwrite=args.overwrite)
    _safe_symlink(scene_root / "sparse/0", view / "sparse/0", overwrite=args.overwrite)
    if (scene_root / "ground_masks").is_dir():
        _safe_symlink(scene_root / "ground_masks", view / "ground_masks", overwrite=args.overwrite)
    dense_seg = scene_root.parent / "SegmentationClass_dense_for_training"
    if dense_seg.is_dir():
        _safe_symlink(dense_seg, view / "segmentation_dense", overwrite=args.overwrite)
    if (scene_root / "normals").is_dir():
        _safe_symlink(scene_root / "normals", view / "normals", overwrite=args.overwrite)
    audit["dataset_view"] = str(view)
    audit["status"] = "PASS"
    view.mkdir(parents=True, exist_ok=True)
    (view / "meshprior_scene_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    with (view / "README_meshprior_scene.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Parking Scene View\n\n")
        f.write(f"source: `{scene_root}`\n\n")
        f.write(f"images: `{audit['image_count']}`\n\n")
        f.write(f"colmap images: `{audit['colmap_image_count']}`\n\n")
        f.write(f"split_outoftrain: `{audit['has_split_outoftrain']}`\n")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare parking scene symlink view.")
    parser.add_argument("--scene_root", default=str(REPO_ROOT.parent / "parking_phone_tiny_anonymized/colmap_undistorted_fix"))
    parser.add_argument("--output_view", default=str(REPO_ROOT / "outputs/carnet/meshprior/parking_phone_tiny/dataset_view"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
