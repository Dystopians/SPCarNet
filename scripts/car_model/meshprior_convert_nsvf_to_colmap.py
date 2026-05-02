#!/usr/bin/env python3
"""Convert an NSVF-style posed image scene into this repo's COLMAP text layout."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import rotmat2qvec


def _read_intrinsics(path: Path) -> tuple[float, float, float, float]:
    mat = np.loadtxt(path, dtype=np.float64)
    if mat.shape == (4, 4):
        return float(mat[0, 0]), float(mat[1, 1]), float(mat[0, 2]), float(mat[1, 2])
    if mat.shape == (3, 3):
        return float(mat[0, 0]), float(mat[1, 1]), float(mat[0, 2]), float(mat[1, 2])
    raise ValueError(f"Unsupported intrinsics shape {mat.shape}: {path}")


def _read_bbox(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if vals.size < 6:
        return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    return vals[:3], vals[3:6]


def _link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _write_points3d(path: Path, bbox_min: np.ndarray, bbox_max: np.ndarray, count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(bbox_min, bbox_max, size=(count, 3))
    rgbs = rng.integers(64, 224, size=(count, 3))
    with path.open("w", encoding="utf-8") as f:
        f.write("# Synthetic bootstrap point cloud for posed NSVF scene.\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for idx, (xyz, rgb) in enumerate(zip(pts, rgbs), start=1):
            f.write(
                f"{idx} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 1.0\n"
            )


def convert_scene(scene: Path, output: Path, point_count: int, copy_images: bool, seed: int) -> None:
    rgb_dir = scene / "rgb"
    pose_dir = scene / "pose"
    if not rgb_dir.is_dir() or not pose_dir.is_dir():
        raise FileNotFoundError(f"Expected rgb/ and pose/ under {scene}")

    output.mkdir(parents=True, exist_ok=True)
    images_out = output / "images"
    sparse_out = output / "sparse" / "0"
    images_out.mkdir(exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    matched = [(rgb, pose_dir / f"{rgb.stem}.txt") for rgb in rgb_files if (pose_dir / f"{rgb.stem}.txt").is_file()]
    if not matched:
        raise ValueError(f"No rgb/pose pairs found in {scene}")

    with Image.open(matched[0][0]) as im:
        width, height = im.size
    fx, fy, cx, cy = _read_intrinsics(scene / "intrinsics.txt")

    with (sparse_out / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n")

    with (sparse_out / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(matched)}, mean observations per image: 0\n")
        for image_id, (rgb, pose_path) in enumerate(matched, start=1):
            c2w = np.loadtxt(pose_path, dtype=np.float64)
            if c2w.shape != (4, 4):
                raise ValueError(f"Unsupported pose shape {c2w.shape}: {pose_path}")
            w2c = np.linalg.inv(c2w)
            qvec = rotmat2qvec(w2c[:3, :3])
            tvec = w2c[:3, 3]
            _link_or_copy(rgb, images_out / rgb.name, copy_images=copy_images)
            vals = [image_id, *qvec.tolist(), *tvec.tolist(), 1, rgb.name]
            f.write(" ".join(str(v) for v in vals) + "\n")
            f.write("\n")

    bbox_min, bbox_max = _read_bbox(scene / "bbox.txt")
    _write_points3d(sparse_out / "points3D.txt", bbox_min, bbox_max, point_count, seed)

    print(
        f"converted scene={scene} output={output} images={len(matched)} "
        f"points={point_count} copy_images={copy_images}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--point_count", type=int, default=100000)
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument("--seed", type=int, default=25)
    args = parser.parse_args()
    convert_scene(Path(args.scene), Path(args.output), args.point_count, args.copy_images, args.seed)


if __name__ == "__main__":
    main()
