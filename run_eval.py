#!/usr/bin/env python
"""GEMS Stage One — THE single evaluation mouth (PROTOCOL.md v1.1.0, D5).

Usage:
    python run_eval.py --checkpoint <point_cloud_state_dict.pt> \
        --scene <toy_parking|garden|courtyard> --out <dir> \
        [--gpu N] [--skip-geometry] [--skip-downstream]

Writes <out>/metrics.json (per-view arrays + means, cost metrics, geometry,
downstream, provenance) and <out>/panels/ per PROTOCOL 4.5.

Rendering metrics reproduce metrics.py conventions exactly: renders are
quantized to 8-bit like torchvision.utils.save_image before PSNR/SSIM/LPIPS,
so numbers match the legacy render.py + metrics.py path bit-for-bit.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_VERSION = "1.1.0"


def parse_args():
    parser = argparse.ArgumentParser(description="GEMS Stage One evaluation mouth")
    parser.add_argument("--checkpoint", required=True,
                        help="path to point_cloud_state_dict.pt (or its directory)")
    parser.add_argument("--scene", required=True,
                        help="scene name from tools/gems/scenes.py registry")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--skip-downstream", action="store_true")
    return parser.parse_args()


def checkpoint_fingerprint(path: str) -> dict:
    """sha256 of the first 16 MiB + file size (cheap, stable identity)."""
    if os.path.isdir(path):
        path = os.path.join(path, "point_cloud_state_dict.pt")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(16 * 1024 * 1024))
    size = os.path.getsize(path)
    return {
        "path": os.path.abspath(path),
        "sha256_first16mb": h.hexdigest(),
        "file_size_bytes": size,
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception as exc:  # metrics must still be writable outside a repo
        return f"unavailable ({exc})"


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    # --- storage preflight (warn-only: eval outputs are MB-scale) ---
    from tools.storage_preflight import volume_report
    preflight = volume_report(args.out, 50.0)
    if not preflight["ok"]:
        print(f"[preflight WARNING] {json.dumps(preflight)}", file=sys.stderr)

    # heavy imports after CUDA_VISIBLE_DEVICES is fixed
    import torch
    from tools.gems.scenes import SCENES
    from tools.gems.eval_context import build_eval_context
    from tools.gems.panels import write_panels
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    from lpipsPyTorch import lpips

    if args.scene not in SCENES:
        raise SystemExit(f"unknown scene '{args.scene}'; registered: {sorted(SCENES)}")
    spec = SCENES[args.scene]

    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    print(f"[run_eval] scene={spec.name} checkpoint={args.checkpoint}")
    ctx = build_eval_context(args.checkpoint, spec)
    ctx.out_dir = os.path.abspath(args.out)
    test_cams = ctx.test_cams
    print(f"[run_eval] {len(test_cams)} test views "
          f"({len(ctx.scene_info.train_cameras)} train cams registered)")

    def quantize_like_png(img: torch.Tensor) -> torch.Tensor:
        """Exact torchvision.utils.save_image 8-bit round trip
        (mul(255).add_(0.5).clamp_(0,255).to(uint8), read back as /255)."""
        return img.mul(255).add(0.5).clamp_(0, 255).to(torch.uint8).float().div(255.0)

    # --- 4.1 rendering metrics (per view, metrics.py conventions) ---
    view_names, psnrs, ssims, lpipss = [], [], [], []
    with torch.no_grad():
        for cam in test_cams:
            pkg = ctx.render_view(cam)
            render = quantize_like_png(pkg["render"]).unsqueeze(0)[:, :3]
            gt = quantize_like_png(
                cam.original_image[:3].to(render.device).clamp(0.0, 1.0)
            ).unsqueeze(0)
            psnrs.append(psnr(render, gt).mean().item())
            ssims.append(ssim(render, gt).item())
            lpipss.append(lpips(render, gt, net_type="vgg").item())
            view_names.append(cam.image_name)
            print(f"  view {cam.image_name}: psnr={psnrs[-1]:.4f} "
                  f"ssim={ssims[-1]:.4f} lpips={lpipss[-1]:.4f}")
    # float32 accumulation, exactly like metrics.py's torch.tensor(list).mean()
    mean_psnr = torch.tensor(psnrs).mean().item()
    mean_ssim = torch.tensor(ssims).mean().item()
    mean_lpips = torch.tensor(lpipss).mean().item()
    print(f"[run_eval] PSNR {mean_psnr:.4f}  SSIM {mean_ssim:.5f}  LPIPS {mean_lpips:.5f}")

    # --- 4.2 cost metrics ---
    ckpt = checkpoint_fingerprint(args.checkpoint)
    n_triangles = int(ctx.triangles._triangle_indices.shape[0])
    n_vertices = int(ctx.triangles.vertices.shape[0])
    disk_mb = ckpt["file_size_bytes"] / (1024.0 * 1024.0)

    # render_fps: pure forward renders, no image I/O; 3 warmup renders
    # excluded; median over 3 repeats of the full test pass. Peak VRAM is
    # measured over this pure render pass (reset before).
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for cam in test_cams[:3]:
            ctx.render_view(cam)
        torch.cuda.synchronize()
        pass_seconds = []
        for _ in range(3):
            t0 = time.perf_counter()
            for cam in test_cams:
                ctx.render_view(cam)
            torch.cuda.synchronize()
            pass_seconds.append(time.perf_counter() - t0)
    render_fps = len(test_cams) / statistics.median(pass_seconds)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    print(f"[run_eval] fps={render_fps:.2f} peak_vram={peak_vram_mb:.0f} MB")

    cost = {
        "n_triangles": n_triangles,
        "n_vertices": n_vertices,
        "disk_mb": disk_mb,
        "peak_vram_mb": peak_vram_mb,
        "render_fps": render_fps,
        "fps_pass_seconds": pass_seconds,
        "n_test_views": len(test_cams),
    }

    # --- 4.3 geometry metrics (lazy: module may not have landed yet) ---
    if args.skip_geometry:
        geometry = {"skipped": "--skip-geometry"}
    else:
        try:
            from tools.gems.geometry_metrics import compute_geometry_metrics
        except ImportError as exc:
            geometry = {"skipped": f"geometry module unavailable: {exc}"}
        else:
            geometry = compute_geometry_metrics(ctx)

    # --- 4.4 downstream proxy (lazy, needs GT surface + frozen ROI) ---
    if args.skip_downstream:
        downstream = {"skipped": "--skip-downstream"}
    elif spec.roi is None:
        downstream = {"skipped": "no ROI frozen in scenes.py for this scene"}
    elif not (spec.gt.get("mesh_path") or spec.gt.get("scan_paths")):
        downstream = {"skipped": "no GT surface asset (mesh_path/scan_paths) declared"}
    elif spec.units_per_meter != 1.0:
        downstream = {"skipped": "downstream constants are metric; scene "
                                 f"units_per_meter={spec.units_per_meter} != 1.0"}
    else:
        try:
            from tools.gems.downstream_metrics import compute_downstream_metrics
        except ImportError as exc:
            downstream = {"skipped": f"downstream module unavailable: {exc}"}
        else:
            import numpy as np
            # PROTOCOL 1.1.0 §4.3: d1/d2 surface = ALL checkpoint triangles
            # (opaque_mask is definitionally all-True; see eval_context).
            opaque = ctx.opaque_mask()
            verts_np = ctx.vertices().cpu().numpy()
            faces_np = ctx.faces()[opaque].cpu().numpy()
            # Adapt SceneSpec.gt to the downstream contract: it accepts
            # 'mesh_path' (.obj) or 'scan_points' [N,3]; scan .ply files
            # (courtyard) are loaded here by the caller.
            if spec.gt.get("mesh_path") and os.path.isfile(spec.gt["mesh_path"]):
                gt_arg = {"mesh_path": spec.gt["mesh_path"]}
            else:
                import trimesh
                clouds = [np.asarray(trimesh.load(p, process=False).vertices,
                                     dtype=np.float64)
                          for p in spec.gt["scan_paths"]]
                gt_arg = {"scan_points": np.concatenate(clouds, axis=0)}
            downstream = compute_downstream_metrics(
                verts_np, faces_np, gt_arg, spec.roi, seed=0
            )
            # Persist the per-sample arrays (they feed the paired bootstrap)
            # as npz and keep only JSON-safe scalars in metrics.json.
            ds_dir = os.path.join(args.out, "downstream")
            os.makedirs(ds_dir, exist_ok=True)
            for sub_name, sub in downstream.items():
                arrays = {k: v for k, v in sub.items() if isinstance(v, np.ndarray)}
                if arrays:
                    npz_path = os.path.join(ds_dir, f"{sub_name}_per_sample.npz")
                    np.savez_compressed(npz_path, **arrays)
                    for k in arrays:
                        del sub[k]
                    sub["per_sample_npz"] = npz_path

    # --- 4.5 panels ---
    floater_tri_ids = None
    if isinstance(geometry, dict):
        g3 = geometry.get("g3", {})
        if isinstance(g3, dict):
            ids_path = g3.get("floater_tri_ids_npz") or g3.get("floater_triangle_ids_npz")
            if ids_path and os.path.exists(ids_path):
                import numpy as np
                with np.load(ids_path) as z:
                    key = "floater_tri_ids" if "floater_tri_ids" in z.files else z.files[0]
                    floater_tri_ids = z[key]
    panel_paths = write_panels(ctx, args.out, n_views=6,
                               floater_tri_ids=floater_tri_ids)

    metrics = {
        "protocol_version": PROTOCOL_VERSION,
        "scene": spec.name,
        "checkpoint": ckpt,
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "storage_preflight": preflight,
        "rendering": {
            "per_view": {
                "image_names": view_names,
                "psnr": psnrs,
                "ssim": ssims,
                "lpips": lpipss,
            },
            "mean": {"psnr": mean_psnr, "ssim": mean_ssim, "lpips": mean_lpips},
        },
        "cost": cost,
        "geometry": geometry,
        "downstream": downstream,
        "panels": [os.path.relpath(p, args.out) for p in panel_paths],
        "eval_wallclock_sec": time.time() - t_start,
    }
    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"[run_eval] wrote {metrics_path}")


if __name__ == "__main__":
    main()
