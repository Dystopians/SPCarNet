#!/usr/bin/env python
"""GEMS Stage Two S-GEO: export SS3DM GT depth for the g1/g2 gt_depth branch.

Sibling of tools/gems_train/ss3dm_ingest.py (LEDGER GOAL #R-05 gap g1: the
converter writes empty POINTS2D lines, so the COLMAP-sparse g1 branch is
structurally inactive on SS3DM; PROTOCOL 4.3 g1's toy branch instead consumes
per-TEST-VIEW GT depth .npy at the render resolution via gt.gt_depth_dir).

For every TEST view of a converted ss3dm_<town> dataset (split.json 'test'
stems, e.g. 'front_left_00000008'):
  1. load the raw uint16 depth PNG
     <sequence_root>/depth_gts/camera_<TOKEN_UPPER>/<frame:08d>.png;
  2. decode meters: d = raw * 1000 / 65535 (CARLA far plane 1000 m; the
     manifest's measured depth_png_scale_uint16_per_m is ~65.50 vs the exact
     encoder scale 65.535 = LiDAR-vs-depth measurement noise, <0.1%);
  3. mark far-plane/sky saturation invalid: raw >= 65534 -> 0.0 (observed sky
     value is exactly 65534 ~ 999.98 m; g1/g2 treat depth <= 0 as invalid);
  4. downsample 1920x1080 -> the -r 2 render resolution 960x540 by nearest
     sampling at target pixel centers, d_small = d[1::2, 1::2] (PIL-NEAREST
     convention: src = floor((dst+0.5)*2) = 2*dst+1); depth values are
     invariant under the diag(1,-1,1) world mirror (projection-invariant), so
     the raw per-camera z-depth IS the converted-frame z-depth;
  5. save float32 .npy as <dataset>/gt/depth/<stem>.npy (geometry_metrics.
     _find_gt_depth_file matches '<image_stem>.npy' first).

Writes <dataset>/gt/depth/export_manifest.json with the frozen conventions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image

DEFAULT_DATA_ROOT = "/data/peilincai/mesh_datasets/SS3DM"
DEFAULT_DS_ROOT = "/data/peilincai/gems_stage1/datasets"
TOWNS = ["Town01", "Town02", "Town03", "Town06"]

DEPTH_M_PER_UNIT = 1000.0 / 65535.0   # exact uint16 encoder scale (far 1000 m)
SATURATION_RAW = 65534                # observed sky/far-plane value; >= -> invalid
RENDER_DOWNSCALE = 2                  # scenes.py resolution=2 (-r 2)


def stem_to_source(stem: str):
    """'front_left_00000008' -> ('camera_FRONT_LEFT', 8)."""
    token, frame = stem.rsplit("_", 1)
    return "camera_" + token.upper(), int(frame)


def export_town(town: str, data_root: str, ds_root: str, seq: str) -> dict:
    town_l = town.lower()
    ds_dir = os.path.join(ds_root, f"ss3dm_{town_l}")
    seq_root = os.path.join(data_root, "DATA", town, seq)
    with open(os.path.join(ds_dir, "split.json")) as f:
        split = json.load(f)
    out_dir = os.path.join(ds_dir, "gt", "depth")
    os.makedirs(out_dir, exist_ok=True)

    n_written = 0
    shapes = set()
    invalid_fracs = []
    for stem in sorted(split["test"]):
        cam_dir, frame = stem_to_source(stem)
        png = os.path.join(seq_root, "depth_gts", cam_dir, f"{frame:08d}.png")
        raw = np.asarray(Image.open(png))
        if raw.dtype != np.uint16:
            raise RuntimeError(f"{png}: expected uint16, got {raw.dtype}")
        H, W = raw.shape
        if (H % RENDER_DOWNSCALE) or (W % RENDER_DOWNSCALE):
            raise RuntimeError(f"{png}: {W}x{H} not divisible by {RENDER_DOWNSCALE}")
        depth_m = raw.astype(np.float64) * DEPTH_M_PER_UNIT
        depth_m[raw >= SATURATION_RAW] = 0.0  # sky/far-plane -> invalid
        # nearest at target pixel centers (PIL-NEAREST): src = 2*dst + 1
        small = depth_m[1::RENDER_DOWNSCALE, 1::RENDER_DOWNSCALE]
        # loadCam target for resolution in [1,2,4,8]: round(orig / r)
        expect = (round(H / RENDER_DOWNSCALE), round(W / RENDER_DOWNSCALE))
        if small.shape != expect:
            raise RuntimeError(f"{stem}: downsample {small.shape} != {expect}")
        np.save(os.path.join(out_dir, f"{stem}.npy"), small.astype(np.float32))
        shapes.add(small.shape)
        invalid_fracs.append(float((small == 0.0).mean()))
        n_written += 1

    manifest = {
        "exporter": "tools/gems_train/ss3dm_export_gt_depth.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "town": town,
        "sequence_root": seq_root,
        "views": "split.json 'test' stems (whole-frame holdout, 3 front cams)",
        "n_views_written": n_written,
        "shape_hw": sorted(list(s) for s in shapes),
        "units": "meters (float32), converted-frame z-depth (mirror-invariant)",
        "decode": f"raw_uint16 * 1000/65535; raw >= {SATURATION_RAW} "
                  "(sky/far-plane saturation) -> 0.0 (invalid)",
        "downsample": "nearest at -r 2 target pixel centers: d[1::2, 1::2] "
                      "(PIL-NEAREST convention, 1920x1080 -> 960x540)",
        "mean_invalid_frac": float(np.mean(invalid_fracs)),
        "consumer": "tools/gems/geometry_metrics.py g1 (_g1_toy) + g2 via "
                    "SceneSpec.gt['gt_depth_dir']",
    }
    with open(os.path.join(out_dir, "export_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[{town}] wrote {n_written} test-view depth npy -> {out_dir} "
          f"(shapes {sorted(shapes)}, mean invalid {np.mean(invalid_fracs):.3f})")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--towns", default=",".join(TOWNS))
    ap.add_argument("--seq", default="150_streetsurf")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--ds-root", default=DEFAULT_DS_ROOT)
    args = ap.parse_args()
    for town in [t.strip() for t in args.towns.split(",") if t.strip()]:
        if town not in TOWNS:
            raise SystemExit(f"unknown town {town}")
        export_town(town, args.data_root, args.ds_root, args.seq)


if __name__ == "__main__":
    main()
