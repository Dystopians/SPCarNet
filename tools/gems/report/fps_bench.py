#!/usr/bin/env python
"""E4-EFF second-resolution render FPS bench (bench-only, NON-protocol res).

MATRIX E4 requires render FPS at >=2 resolutions. The protocol-resolution FPS
is already in every metrics.json (run_eval.py cost block). This script re-runs
the EXACT run_eval FPS loop (3 warmup renders, median of 3 full test-set
passes, pure forward renders, no image I/O) on existing checkpoints with the
test cameras rebuilt at resolution_scale=2.0 -> 0.5x linear protocol
resolution. Numbers produced here are labeled 'bench-only, non-protocol
resolution' and are used ONLY in the T4 efficiency table's second-resolution
column — never for quality claims.

Bench set (from RESULTS/aggregate/all_rows.json, canonical rows):
  S-REND: B0 + B5 at B50/B25/B12.5 (9 scenes)
  S-GEO : B0 + B5@B50 (4 towns)
  S-DEV : B0 + B5@B50 (toy_parking, courtyard)

Usage:
    python tools/gems/report/fps_bench.py [--gpu 4] [--limit N] [--only DIR...]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ALL_ROWS = os.path.join(REPO_ROOT, "RESULTS", "aggregate", "all_rows.json")
OUT_DEFAULT = os.path.join(REPO_ROOT, "RESULTS", "aggregate", "fps_bench_halfres.json")
RESOLUTION_SCALE = 2.0  # 0.5x linear protocol resolution


def select_targets(rows):
    targets = []
    for r in rows:
        if not r.get("canonical"):
            continue
        m, s, b = r["method"], r["suite"], r["budget_label"]
        want = (
            (m == "B0")
            or (m == "B5" and s == "S-REND" and b in ("B50", "B25", "B12.5"))
            or (m == "B5" and s in ("S-GEO", "S-DEV") and b == "B50")
        )
        if want:
            targets.append(r)
    return sorted(targets, key=lambda r: (r["suite"], r["scene"], r["budget_label"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="bench only these eval_dir names")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    import torch
    from tools.gems.scenes import SCENES
    from tools.gems.eval_context import build_eval_context
    from utils.camera_utils import cameraList_from_camInfos

    with open(ALL_ROWS) as f:
        corpus = json.load(f)
    targets = select_targets(corpus["rows"])
    if args.only:
        targets = [t for t in targets if t["eval_dir"] in set(args.only)]
    if args.limit:
        targets = targets[: args.limit]
    print(f"[fps_bench] {len(targets)} checkpoints to bench "
          f"(resolution_scale={RESOLUTION_SCALE})")

    results = []
    for i, r in enumerate(targets):
        ckpt = r["provenance"]["checkpoint"]["path"]
        rec = {
            "eval_dir": r["eval_dir"],
            "scene": r["scene"],
            "suite": r["suite"],
            "method": r["method"],
            "budget_label": r["budget_label"],
            "checkpoint": ckpt,
            "checkpoint_sha256_first16mb": r["provenance"]["checkpoint"].get(
                "sha256_first16mb"),
            "resolution_scale": RESOLUTION_SCALE,
            "label": "bench-only, non-protocol resolution "
                     "(0.5x linear protocol res)",
        }
        if not os.path.isfile(ckpt):
            rec["skipped"] = "checkpoint missing on disk"
            results.append(rec)
            print(f"  [{i+1}/{len(targets)}] {r['eval_dir']}: MISSING ckpt")
            continue
        t0 = time.time()
        spec = SCENES[r["scene"]]
        ctx = build_eval_context(ckpt, spec)
        # rebuild the test cameras at half linear resolution — same loader,
        # same split, only resolution_scale differs from the protocol mouth.
        half_cams = cameraList_from_camInfos(
            ctx.scene_info.test_cameras, RESOLUTION_SCALE, ctx._cam_args)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for cam in half_cams[:3]:
                ctx.render_view(cam)
            torch.cuda.synchronize()
            pass_seconds = []
            for _ in range(3):
                t1 = time.perf_counter()
                for cam in half_cams:
                    ctx.render_view(cam)
                torch.cuda.synchronize()
                pass_seconds.append(time.perf_counter() - t1)
        fps = len(half_cams) / statistics.median(pass_seconds)
        rec.update({
            "n_test_views": len(half_cams),
            "bench_width": int(half_cams[0].image_width),
            "bench_height": int(half_cams[0].image_height),
            "render_fps_halfres": fps,
            "fps_pass_seconds": pass_seconds,
            "peak_vram_mb_halfres": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
            "n_triangles": int(ctx.triangles._triangle_indices.shape[0]),
            "wallclock_sec": time.time() - t0,
        })
        results.append(rec)
        print(f"  [{i+1}/{len(targets)}] {r['eval_dir']}: "
              f"{rec['bench_width']}x{rec['bench_height']} "
              f"fps={fps:.1f} ({rec['wallclock_sec']:.0f}s)")
        del ctx, half_cams
        torch.cuda.empty_cache()

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": os.path.abspath(__file__),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "resolution_scale": RESOLUTION_SCALE,
        "label": "bench-only, non-protocol resolution (0.5x linear protocol res)",
        "fps_loop": "identical to run_eval.py: 3 warmup renders, median of 3 "
                    "full test-set forward passes, no image I/O",
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # merge with existing partial results if --only/--limit was used
    if (args.only or args.limit) and os.path.exists(args.out):
        with open(args.out) as f:
            old = json.load(f)
        done = {r["eval_dir"] for r in results}
        payload["results"] = [r for r in old.get("results", [])
                              if r["eval_dir"] not in done] + results
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[fps_bench] wrote {args.out} ({len(payload['results'])} records)")


if __name__ == "__main__":
    main()
