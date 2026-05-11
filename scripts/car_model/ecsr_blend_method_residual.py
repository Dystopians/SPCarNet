#!/usr/bin/env python3
"""Blend a rendered method residual back toward a base render directory.

This is a diagnostic utility: it does not choose a paper policy by itself.  It
is useful for replaying fixed train-policy residual strengths without rerunning
the expensive surface-signal construction.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--base_method", required=True)
    parser.add_argument("--source_method", required=True)
    parser.add_argument("--output_method", required=True)
    parser.add_argument("--scale", type=float, required=True)
    return parser.parse_args()


def image_to_np(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_np(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def main() -> int:
    args = parse_args()
    base_dir = args.model_path / args.split / args.base_method
    source_dir = args.model_path / args.split / args.source_method
    out_dir = args.model_path / args.split / args.output_method
    out_render_dir = out_dir / "renders"
    out_render_dir.mkdir(parents=True, exist_ok=True)

    gt_src = base_dir / "gt"
    gt_dst = out_dir / "gt"
    gt_dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(gt_src.iterdir()):
        if path.is_file():
            target = gt_dst / path.name
            if not target.exists():
                shutil.copy2(path, target)

    base_renders = {path.name: path for path in (base_dir / "renders").iterdir() if path.is_file()}
    source_renders = {path.name: path for path in (source_dir / "renders").iterdir() if path.is_file()}
    names = sorted(set(base_renders) & set(source_renders))
    if not names:
        raise RuntimeError("no overlapping render files")
    for name in tqdm(names, desc=f"Blend residual scale={args.scale:g}"):
        base = image_to_np(base_renders[name])
        source = image_to_np(source_renders[name])
        blended = base + float(args.scale) * (source - base)
        save_np(out_render_dir / name, blended)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
