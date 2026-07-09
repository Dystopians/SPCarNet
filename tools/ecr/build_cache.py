#!/usr/bin/env python
"""GEMS Stage-4 — evidence cache builder (PROTOCOL v1.2.0 §4E).

Builds the per-(checkpoint, scene) evidence cache that is PART OF THE SHIPPED
ARTIFACT: for every TRAIN view, the base model's render (PNG), a copy of the
train GT image at the training resolution (PNG), the median surf_depth (npy
float32), plus the camera index and a manifest that FREEZES the transport
config — including the train-only leave-one-out alpha calibration (production
Phase-J procedure, tools/gems_train/teacher_factory.py::ADAPTER_CONFIG).

D4: consumes checkpoint + TRAIN cameras/images only. Test views are never
rendered, read, or listed (the manifest's train_views set is what the --ecr
audit cross-checks against the scene's test split).

Usage:
    python -m tools.ecr.build_cache --checkpoint <point_cloud_state_dict.pt> \
        --scene <name> --out <cache_dir> [--gpu N] [--alpha -1.0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Frozen production Phase-J transport config (= the archived apply-script
# defaults, re-frozen in tools/gems_train/teacher_factory.py::ADAPTER_CONFIG).
TRANSPORT_CONFIG = {
    "mode": "residual",
    "k": 4,
    "residual_clip": 0.25,
    "min_confidence": 1e-4,
    "depth_abs_tol": 0.02,
    "depth_rel_tol": 0.03,
    "direction_weight": 0.35,
    "evidence_max_side": 0,
    "edge_gate": False,
    "edge_gate_quantile": -1.0,
    "edge_gate_min": 0.0,
    "edge_gate_dilate": 0,
    "local_trust_gate": False,
    "local_trust_min_supports": 2,
    "local_trust_max_residual_std": -1.0,
    "local_trust_min_agreement": 0.0,
    "local_trust_agreement_scale": 0.04,
    "local_trust_confidence_quantile": -1.0,
    "local_trust_min_confidence": 0.0,
    "local_trust_mode": "hard",
    "local_trust_min_weight": 0.0,
}
ALPHA_CALIBRATION = {
    "alpha_grid": [0.0, 0.125, 0.25, 0.5, 0.75, 1.0],
    "calib_stride": 16,
    "calib_max_views": 16,
    "calib_sampler": "stride_first",
    "policy_objective": "psnr",
}


def checkpoint_fingerprint(path: str) -> dict:
    """Identical convention to run_eval.py (sha256 of first 16 MiB + size)."""
    if os.path.isdir(path):
        path = os.path.join(path, "point_cloud_state_dict.pt")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(16 * 1024 * 1024))
    return {
        "path": os.path.abspath(path),
        "sha256_first16mb": h.hexdigest(),
        "file_size_bytes": os.path.getsize(path),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"unavailable ({exc})"


def dir_file_sizes(root: Path) -> dict[str, int]:
    sizes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            sizes[str(path.relative_to(root))] = path.stat().st_size
    return sizes


def measure_compressed_mb(root: Path) -> tuple[float, str]:
    """Lossless-compressed size of the whole cache (measured, then deleted)."""
    zstd = shutil.which("zstd")
    with tempfile.NamedTemporaryFile(suffix=".tar.cmp", delete=False,
                                     dir=str(root.parent)) as tmp:
        tmp_path = tmp.name
    try:
        if zstd:
            cmd = (f"tar -C {root} -cf - . | zstd -9 -T0 -q -f -o {tmp_path}")
            method = "tar+zstd-9"
        else:
            cmd = f"tar -C {root} -czf {tmp_path} ."
            method = "tar+gzip"
        subprocess.run(["bash", "-c", cmd], check=True)
        return os.path.getsize(tmp_path) / (1024.0 * 1024.0), method
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out", required=True, help="cache directory")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=-1.0,
                        help="fixed alpha; < 0 -> train-only LOO calibration "
                             "(production default)")
    parser.add_argument("--fuse", choices=("single", "multiband"),
                        default="single",
                        help="transport fuse: 'single' = frozen PJ-2026 "
                             "(default); 'multiband' = L2 Laplacian fusion "
                             "with joint train-LOO (K, alpha) calibration")
    parser.add_argument("--k-grid", default="2,4,8",
                        help="pre-registered K grid for --fuse multiband")
    parser.add_argument("--bands", type=int, default=4,
                        help="Laplacian bands for --fuse multiband (frozen)")
    parser.add_argument("--skip-compress", action="store_true",
                        help="skip the compressed-size measurement")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    from tools.storage_preflight import volume_report
    preflight = volume_report(args.out, 50.0)
    if not preflight["ok"]:
        print(f"[preflight ABORT] {json.dumps(preflight)}", file=sys.stderr)
        return 1

    import numpy as np
    import torch
    import torchvision

    from tools.gems.scenes import SCENES
    from tools.gems.eval_context import build_eval_context
    from tools.ecr.renderer import camera_record_from_pose
    from utils.evidence_lumigraph_adapter import (
        FrameRecord,
        calibrate_alpha,
        save_camera_index,
    )

    def camera_record_from_view(idx, view):
        return camera_record_from_pose(idx, {
            "image_name": str(getattr(view, "image_name", f"{idx:05d}")),
            "width": int(view.image_width),
            "height": int(view.image_height),
            "fovx": float(view.FoVx),
            "fovy": float(view.FoVy),
            "camera_center": [float(x) for x in
                              view.camera_center.detach().cpu().tolist()],
            "world_view_transform":
                view.world_view_transform.detach().cpu().tolist(),
        })

    if args.scene not in SCENES:
        raise SystemExit(f"unknown scene '{args.scene}'; registered: {sorted(SCENES)}")
    spec = SCENES[args.scene]

    out = Path(args.out).resolve()
    render_dir, gt_dir, depth_dir = out / "renders", out / "gt", out / "depths"
    for d in (render_dir, gt_dir, depth_dir):
        d.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"[build_cache] scene={spec.name} checkpoint={args.checkpoint}")
    ctx = build_eval_context(args.checkpoint, spec)
    train_cams = ctx.train_cams
    test_names = sorted(str(cam.image_name) for cam in ctx.test_cams)
    print(f"[build_cache] {len(train_cams)} train views "
          f"({len(test_names)} test views EXCLUDED by construction)")

    names: list[str] = []
    camera_records = []
    with torch.no_grad():
        for idx, cam in enumerate(train_cams):
            name = str(cam.image_name)
            if name in names:
                raise RuntimeError(f"duplicate train view name: {name}")
            names.append(name)
            render_path = render_dir / f"{name}.png"
            depth_path = depth_dir / f"{name}.npy"
            gt_path = gt_dir / f"{name}.png"
            for p in (render_path, depth_path, gt_path):
                p.parent.mkdir(parents=True, exist_ok=True)
            if not (render_path.is_file() and depth_path.is_file()
                    and gt_path.is_file()):
                pkg = ctx.render_view(cam)
                torchvision.utils.save_image(pkg["render"], render_path)
                np.save(depth_path, pkg["surf_depth"][0].detach().float()
                        .cpu().numpy().astype(np.float32))
                torchvision.utils.save_image(
                    cam.original_image[:3].clamp(0.0, 1.0), gt_path)
                del pkg
            camera_records.append(camera_record_from_view(idx, cam))
            if idx % 25 == 0:
                torch.cuda.empty_cache()
                print(f"[build_cache] {idx + 1}/{len(train_cams)}", flush=True)
    save_camera_index(camera_records, out / "camera_index.json")
    for name in names:
        if name in test_names:
            raise RuntimeError(f"train view name collides with a TEST view: {name}")

    # ---- train-only leave-one-out alpha calibration (production Phase-J) ----
    frames = [FrameRecord(idx=i, name=n,
                          render_path=render_dir / f"{n}.png",
                          gt_path=gt_dir / f"{n}.png",
                          depth_path=depth_dir / f"{n}.npy",
                          camera=camera_records[i])
              for i, n in enumerate(names)]
    transport_config = dict(TRANSPORT_CONFIG)
    if args.fuse == "multiband":
        from tools.ecr.transport_l2 import calibrate_k_alpha
        transport_config["fuse"] = "multiband"
        transport_config["bands"] = int(args.bands)
        k_grid = [int(x) for x in str(args.k_grid).split(",") if x.strip()]
        print(f"[build_cache] L2 joint (K, alpha) train-LOO calibration, "
              f"K grid {k_grid}...")
        calib = calibrate_k_alpha(
            frames,
            k_grid=k_grid,
            alpha_grid=ALPHA_CALIBRATION["alpha_grid"],
            calib_stride=ALPHA_CALIBRATION["calib_stride"],
            calib_max_views=ALPHA_CALIBRATION["calib_max_views"],
            residual_clip=transport_config["residual_clip"],
            depth_abs_tol=transport_config["depth_abs_tol"],
            depth_rel_tol=transport_config["depth_rel_tol"],
            direction_weight=transport_config["direction_weight"],
            bands=int(args.bands),
            min_confidence=transport_config["min_confidence"],
            device="cuda",
        )
        transport_config["k"] = int(calib["k"])
        alpha_block = {
            "alpha": float(calib["alpha"]),
            "k": int(calib["k"]),
            "source": "train_loo_k_alpha",
            "loo_psnr_gain": float(calib["psnr_gain"]),
            "rows": calib["rows"],
            "calibration_views": calib["calibration_views"],
            "k_grid": calib["k_grid"],
            "alpha_grid": calib["alpha_grid"],
        }
        print(f"[build_cache] calibrated K = {calib['k']}, "
              f"alpha = {calib['alpha']} (LOO gain {calib['psnr_gain']:.3f} dB)")
    elif args.alpha >= 0.0:
        alpha_block = {"alpha": float(args.alpha), "source": "cli_fixed"}
    else:
        print("[build_cache] calibrating alpha (train-only leave-one-out)...")
        calib = calibrate_alpha(
            frames,
            alpha_grid=ALPHA_CALIBRATION["alpha_grid"],
            k=TRANSPORT_CONFIG["k"],
            mode=TRANSPORT_CONFIG["mode"],
            calib_stride=ALPHA_CALIBRATION["calib_stride"],
            calib_max_views=ALPHA_CALIBRATION["calib_max_views"],
            residual_clip=TRANSPORT_CONFIG["residual_clip"],
            depth_abs_tol=TRANSPORT_CONFIG["depth_abs_tol"],
            depth_rel_tol=TRANSPORT_CONFIG["depth_rel_tol"],
            direction_weight=TRANSPORT_CONFIG["direction_weight"],
            policy_objective=ALPHA_CALIBRATION["policy_objective"],
            calib_sampler=ALPHA_CALIBRATION["calib_sampler"],
            device="cuda",
        )
        alpha_block = {
            "alpha": float(calib["alpha"]),
            "source": "train_loo_calibration",
            "rows": calib["rows"],
            "calibration_views": calib["calibration_views"],
            **{k: ALPHA_CALIBRATION[k] for k in ALPHA_CALIBRATION},
        }
        print(f"[build_cache] calibrated alpha = {alpha_block['alpha']}")

    sizes = dir_file_sizes(out)
    cache_mb_raw = sum(sizes.values()) / (1024.0 * 1024.0)
    if args.skip_compress:
        cache_mb_compressed, method = -1.0, "skipped"
    else:
        print("[build_cache] measuring lossless-compressed size...")
        cache_mb_compressed, method = measure_compressed_mb(out)

    manifest = {
        "format": "gems-ecr-cache-v1",
        "protocol_version": "1.2.0",
        "scene": spec.name,
        "checkpoint": checkpoint_fingerprint(args.checkpoint),
        "train_views": names,
        "n_train_views": len(names),
        "transport": transport_config,
        "alpha": alpha_block,
        "sizes": {
            "cache_mb_raw": cache_mb_raw,
            "cache_mb_compressed": cache_mb_compressed,
            "compression_method": method,
            "n_files": len(sizes),
            "files": sizes,
        },
        "provenance": {
            "git_commit": git_commit(),
            "command": " ".join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "build_wallclock_sec": time.time() - t_start,
            "storage_preflight": preflight,
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n",
                             encoding="utf-8")
    print(f"[build_cache] wrote {manifest_path} "
          f"(raw {cache_mb_raw:.1f} MB, compressed {cache_mb_compressed:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
