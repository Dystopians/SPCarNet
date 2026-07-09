#!/usr/bin/env python
"""GEMS Stage-4 L5: cache-compression variant builder (GOAL #E-06).

Builds a compressed variant of an existing evidence cache along ONE of the
pre-registered axes, re-runs the frozen train-LOO (K, alpha) calibration on
the variant's actual bytes, and writes a fresh manifest (sizes measured,
lossless-compressed size re-measured). The fusion net is retrained afterwards
by tools/ecr/train_fusion.py (frozen recipe) so the whole stack rides the
compressed evidence honestly.

Pre-registered axes (prompt §3 L5; no other knobs):
  jpeg95 | jpeg85 | jpeg70   — re-encode renders/ + gt/ as JPEG quality q
  halfres                    — downsample renders/gt/depths by 2 (bilinear;
                               depths area-mean); cameras unchanged (the
                               warp scales intrinsics from stored W/H)
  ksubset50                  — keep 50% of train views by greedy
                               farthest-point coverage on (center, direction)

Usage:
    python -m tools.ecr.l5_compress --src <cache> --out <cache_variant> \
        --variant jpeg85 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VARIANTS = ("jpeg95", "jpeg85", "jpeg70", "halfres", "ksubset50")


def greedy_coverage_subset(cameras, names, keep_n):
    """Farthest-point selection on (center, 2*direction) — train-side only."""
    import numpy as np
    feats = []
    for name in names:
        cam = cameras[name]
        c = np.asarray(cam.camera_center, dtype=np.float64)
        d = np.asarray(cam.view_direction().numpy(), dtype=np.float64)
        scale = max(np.linalg.norm(c), 1.0)
        feats.append(np.concatenate([c / scale, 2.0 * d]))
    feats = np.stack(feats)
    chosen = [0]
    dist = np.linalg.norm(feats - feats[0], axis=1)
    while len(chosen) < keep_n:
        idx = int(dist.argmax())
        chosen.append(idx)
        dist = np.minimum(dist, np.linalg.norm(feats - feats[idx], axis=1))
    return sorted(set(chosen))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    import numpy as np
    from PIL import Image
    from utils.evidence_lumigraph_adapter import (FrameRecord,
                                                  load_camera_index,
                                                  save_camera_index)
    from tools.ecr.build_cache import (dir_file_sizes, git_commit,
                                       measure_compressed_mb)
    from tools.ecr.transport_l2 import calibrate_k_alpha

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    if str(manifest["transport"].get("fuse")) not in ("multiband", "single"):
        raise SystemExit("run l5_compress on a PRE-net cache (single/multiband "
                         "manifest); the net is retrained afterwards")
    names = list(manifest["train_views"])
    cameras = {c.image_name: c for c in
               load_camera_index(src / "camera_index.json")}
    ext = manifest.get("image_ext", {})
    src_rext = str(ext.get("renders", "png"))
    src_gext = str(ext.get("gt", "png"))

    t0 = time.time()
    (out / "renders").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "depths").mkdir(parents=True, exist_ok=True)

    kept = names
    if args.variant.startswith("jpeg"):
        q = int(args.variant[4:])
        new_ext = {"renders": "jpg", "gt": "jpg"}
        for name in names:
            for sub, se in (("renders", src_rext), ("gt", src_gext)):
                dst = out / sub / f"{name}.jpg"
                if not dst.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    Image.open(src / sub / f"{name}.{se}").convert("RGB") \
                        .save(dst, quality=q)
            d = out / "depths" / f"{name}.npy"
            if not d.exists():
                d.parent.mkdir(parents=True, exist_ok=True)
                os.link(src / "depths" / f"{name}.npy", d)
    elif args.variant == "halfres":
        new_ext = {"renders": "png", "gt": "png"}
        for name in names:
            for sub, se in (("renders", src_rext), ("gt", src_gext)):
                dst = out / sub / f"{name}.png"
                if not dst.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    img = Image.open(src / sub / f"{name}.{se}").convert("RGB")
                    img.resize((img.width // 2, img.height // 2),
                               Image.BILINEAR).save(dst)
            dst = out / "depths" / f"{name}.npy"
            if not dst.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dep = np.load(src / "depths" / f"{name}.npy")
                h2, w2 = dep.shape[0] // 2, dep.shape[1] // 2
                dep2 = dep[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).mean(axis=(1, 3))
                np.save(dst, dep2.astype(np.float32))
    elif args.variant == "ksubset50":
        new_ext = {"renders": src_rext, "gt": src_gext}
        keep_idx = greedy_coverage_subset(cameras, names,
                                          max(len(names) // 2, 2))
        kept = [names[i] for i in keep_idx]
        for name in kept:
            for sub, se in (("renders", src_rext), ("gt", src_gext),
                            ("depths", "npy")):
                fn = f"{name}.{se}" if sub != "depths" else f"{name}.npy"
                dst = out / sub / fn
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.link(src / sub / fn, dst)
    else:
        raise SystemExit(f"unknown variant {args.variant}")

    save_camera_index([cameras[n] for n in kept], out / "camera_index.json")

    # frozen re-calibration on the variant's actual bytes
    rext, gext = new_ext["renders"], new_ext["gt"]
    frames = [FrameRecord(idx=i, name=n,
                          render_path=out / "renders" / f"{n}.{rext}",
                          gt_path=out / "gt" / f"{n}.{gext}",
                          depth_path=out / "depths" / f"{n}.npy",
                          camera=cameras[n])
              for i, n in enumerate(kept)]
    transport = dict(manifest["transport"])
    calib = calibrate_k_alpha(
        frames, k_grid=[2, 4, 8],
        alpha_grid=[0.0, 0.125, 0.25, 0.5, 0.75, 1.0],
        calib_stride=16, calib_max_views=16,
        residual_clip=transport["residual_clip"],
        depth_abs_tol=transport["depth_abs_tol"],
        depth_rel_tol=transport["depth_rel_tol"],
        direction_weight=transport["direction_weight"],
        bands=int(transport.get("bands", 4)),
        min_confidence=transport["min_confidence"], device="cuda")
    transport["fuse"] = "multiband"
    transport["bands"] = int(transport.get("bands", 4))
    transport["k"] = int(calib["k"])

    sizes = dir_file_sizes(out)
    raw_mb = sum(sizes.values()) / (1024.0 * 1024.0)
    comp_mb, method = measure_compressed_mb(out)
    manifest_out = dict(manifest)
    manifest_out.update({
        "variant": args.variant,
        "variant_src": str(src),
        "train_views": kept,
        "n_train_views": len(kept),
        "image_ext": new_ext,
        "transport": transport,
        "alpha": {"alpha": float(calib["alpha"]), "k": int(calib["k"]),
                  "source": "train_loo_k_alpha", "rows": calib["rows"],
                  "loo_psnr_gain": float(calib["psnr_gain"])},
        "sizes": {"cache_mb_raw": raw_mb, "cache_mb_compressed": comp_mb,
                  "compression_method": method, "n_files": len(sizes),
                  "files": sizes},
    })
    manifest_out["provenance"] = {
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_wallclock_sec": time.time() - t0,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest_out, indent=1) + "\n", encoding="utf-8")
    print(f"[l5_compress] {args.variant}: {len(kept)} views, "
          f"{raw_mb:.1f} MB raw / {comp_mb:.1f} MB compressed; "
          f"K={calib['k']} alpha={calib['alpha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
