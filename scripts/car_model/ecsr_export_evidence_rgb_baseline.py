#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.ecsr_apply_surface_residual_region_texture_adapter import evidence_views, save_image_chw  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export cached evidence rgb_render/rgb_gt as a metrics.py baseline.")
    parser.add_argument("--evidence_dir", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--method_name", default="evidence_rgb_render_baseline")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    views = evidence_views(Path(args.evidence_dir))
    if not views:
        raise FileNotFoundError(f"no npz views found in {args.evidence_dir}")
    method_dir = Path(args.output_model) / str(args.split) / str(args.method_name)
    render_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    if method_dir.exists():
        if not bool(args.force):
            raise FileExistsError(method_dir)
        shutil.rmtree(method_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    for path in views:
        z = np.load(path)
        if "rgb_render" not in z or "rgb_gt" not in z:
            raise KeyError(f"{path} must contain rgb_render and rgb_gt")
        name = f"{path.stem}.png"
        save_image_chw(render_dir / name, np.asarray(z["rgb_render"], dtype=np.float32))
        save_image_chw(gt_dir / name, np.asarray(z["rgb_gt"], dtype=np.float32))
    print(f"wrote {len(views)} views to {method_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
