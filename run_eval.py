#!/usr/bin/env python
"""GEMS — THE single evaluation mouth (PROTOCOL.md v1.2.0, D5).

Usage:
    python run_eval.py --checkpoint <point_cloud_state_dict.pt> \
        --scene <toy_parking|garden|courtyard|...> --out <dir> \
        [--gpu N] [--skip-geometry] [--skip-downstream] \
        [--renderer base|ecr] [--ecr-cache <cache_dir>]

Writes <out>/metrics.json (per-view arrays + means, cost metrics, geometry,
downstream, provenance) and <out>/panels/ per PROTOCOL 4.5.

Rendering metrics reproduce metrics.py conventions exactly: renders are
quantized to 8-bit like torchvision.utils.save_image before PSNR/SSIM/LPIPS,
so numbers match the legacy render.py + metrics.py path bit-for-bit.

Renderer modes (PROTOCOL 1.2.0, one mouth, identical metric code):
  base (default) — the render path used for every pre-Stage-4 row; behavior
                   and numbers unchanged. Never loads any tools/ecr or
                   evidence-transport module (audited).
  ecr            — Stage-4 evidence-cached rendering: base render + frozen
                   Phase-J transport from the evidence cache built by
                   tools/ecr/build_cache.py (--ecr-cache required). Test-view
                   GT is used ONLY for metric computation, never by the
                   transport (audited by tools/audit_test_path.py --ecr).
                   Adds cost columns: cache_mb_raw, cache_mb_compressed,
                   transport_ms_per_frame, end_to_end_fps.
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
PROTOCOL_VERSION = "1.2.0"


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
    parser.add_argument("--renderer", choices=("base", "ecr"), default="base",
                        help="render path: 'base' (default, pre-Stage-4 "
                             "behavior, unchanged) or 'ecr' (Stage-4 "
                             "evidence-cached rendering; needs --ecr-cache)")
    parser.add_argument("--ecr-cache", default=None,
                        help="evidence cache dir from tools/ecr/build_cache.py "
                             "(required with --renderer ecr)")
    return parser.parse_args()


def pose_primitives(cam) -> dict:
    """Plain pose dict for the ECR renderer. Deliberately contains ONLY
    geometry — never the camera's image tensor — so no GT-bearing object
    crosses the D4 boundary into tools/ecr (PROTOCOL 1.2.0)."""
    return {
        "image_name": str(cam.image_name),
        "width": int(cam.image_width),
        "height": int(cam.image_height),
        "fovx": float(cam.FoVx),
        "fovy": float(cam.FoVy),
        "camera_center": [float(x) for x in
                          cam.camera_center.detach().cpu().tolist()],
        "world_view_transform":
            cam.world_view_transform.detach().cpu().tolist(),
    }


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

    # --- Stage-4 ECR renderer (PROTOCOL 1.2.0 §4E) ---
    ecr = None
    if args.renderer == "ecr":
        if not args.ecr_cache:
            raise SystemExit("--renderer ecr requires --ecr-cache")
        # Lazy, mode-gated import: this line never executes in base mode.
        # The Stage-2 purity audit verifies DYNAMICALLY that no tools.ecr /
        # evidence-transport module is loaded in base mode (PROTOCOL 1.2.0
        # changelog); the --ecr audit mode covers this path.
        from tools.ecr.renderer import EcrRenderer
        ecr = EcrRenderer(args.ecr_cache)
        ckpt_fp = checkpoint_fingerprint(args.checkpoint)
        cache_fp = ecr.checkpoint_fingerprint()
        if cache_fp.get("sha256_first16mb") != ckpt_fp["sha256_first16mb"]:
            raise SystemExit(
                "evidence cache / checkpoint mismatch: cache built for "
                f"{cache_fp.get('path')} ({cache_fp.get('sha256_first16mb')}), "
                f"evaluating {ckpt_fp['path']} ({ckpt_fp['sha256_first16mb']})")
        overlap = {str(c.image_name) for c in test_cams}.intersection(
            ecr.manifest["train_views"])
        if overlap:
            raise SystemExit(
                f"evidence cache lists TEST view names (D4 violation): "
                f"{sorted(overlap)[:5]}")
        print(f"[run_eval] ecr: cache={args.ecr_cache} "
              f"alpha={ecr.manifest['alpha']['alpha']} "
              f"config_hash={ecr.config_hash[:12]} "
              f"({ecr.manifest['n_train_views']} train views)")

    def quantize_like_png(img: torch.Tensor) -> torch.Tensor:
        """Exact torchvision.utils.save_image 8-bit round trip
        (mul(255).add_(0.5).clamp_(0,255).to(uint8), read back as /255)."""
        return img.mul(255).add(0.5).clamp_(0, 255).to(torch.uint8).float().div(255.0)

    # --- 4.1 rendering metrics (per view, metrics.py conventions) ---
    view_names, psnrs, ssims, lpipss = [], [], [], []
    ecr_view_infos = []
    with torch.no_grad():
        for cam in test_cams:
            pkg = ctx.render_view(cam)
            out_img = quantize_like_png(pkg["render"])
            if ecr is not None:
                # transport inputs: 8-bit-quantized base render (parity with
                # the archived PNG path) + median surf_depth; the camera's
                # image tensor never crosses into the transport.
                adapted, info = ecr.adapt(
                    str(cam.image_name), pose_primitives(cam),
                    out_img, pkg["surf_depth"][0])
                ecr_view_infos.append({
                    "image_name": str(cam.image_name),
                    "kwargs_hash": info["kwargs_hash"],
                    "transport_seconds": info["transport_seconds"],
                    "covered_fraction": info.get("covered_fraction"),
                    "mean_confidence": info.get("mean_confidence"),
                    "support_names": info.get("support_names"),
                })
                out_img = quantize_like_png(adapted)
            render = out_img.unsqueeze(0)[:, :3]
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

    if ecr is not None:
        # --- 4E cost columns: transport ms/frame + end-to-end FPS ---
        # Steady state (loader warm from the metric pass); 3 timed passes of
        # base render + transport, median, mirroring the render_fps rule.
        e2e_pass_seconds, transport_secs = [], []
        with torch.no_grad():
            for _ in range(3):
                t0 = time.perf_counter()
                for cam in test_cams:
                    pkg = ctx.render_view(cam)
                    base_q = quantize_like_png(pkg["render"])
                    _, info = ecr.adapt(
                        str(cam.image_name), pose_primitives(cam),
                        base_q, pkg["surf_depth"][0])
                    transport_secs.append(info["transport_seconds"])
                torch.cuda.synchronize()
                e2e_pass_seconds.append(time.perf_counter() - t0)
        cost["transport_ms_per_frame"] = 1000.0 * statistics.median(transport_secs)
        cost["end_to_end_fps"] = len(test_cams) / statistics.median(e2e_pass_seconds)
        cost["e2e_pass_seconds"] = e2e_pass_seconds
        cost.update(ecr.cache_cost())
        cost["total_artifact_mb"] = disk_mb + max(cost["cache_mb_raw"], 0.0)
        print(f"[run_eval] ecr: transport {cost['transport_ms_per_frame']:.1f} "
              f"ms/frame, end-to-end {cost['end_to_end_fps']:.2f} fps, cache "
              f"{cost['cache_mb_raw']:.1f} MB raw / "
              f"{cost['cache_mb_compressed']:.1f} MB compressed")

    # --- 4.3 geometry metrics (lazy: module may not have landed yet) ---
    if ecr is not None:
        geometry = {"skipped": "renderer=ecr: transport alters only rendered "
                               "RGB; g-metrics are unchanged by construction "
                               "(PROTOCOL 1.2.0 §4E) — see the checkpoint's "
                               "base rows"}
    elif args.skip_geometry:
        geometry = {"skipped": "--skip-geometry"}
    else:
        try:
            from tools.gems.geometry_metrics import compute_geometry_metrics
        except ImportError as exc:
            geometry = {"skipped": f"geometry module unavailable: {exc}"}
        else:
            geometry = compute_geometry_metrics(ctx)

    # --- 4.4 downstream proxy (lazy, needs GT surface + frozen ROI) ---
    if ecr is not None:
        downstream = {"skipped": "renderer=ecr: transport alters only rendered "
                                 "RGB; d-metrics are unchanged by construction "
                                 "(PROTOCOL 1.2.0 §4E) — see the checkpoint's "
                                 "base rows"}
    elif args.skip_downstream:
        downstream = {"skipped": "--skip-downstream"}
    elif spec.roi is None:
        downstream = {"skipped": "no ROI frozen in scenes.py for this scene"}
    elif spec.roi.get("z_band") is None:
        downstream = {"skipped": "ROI frozen without z_band (up-axis derivation "
                                 "pending, PROTOCOL 4.4); d1/d2 gated"}
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
            # with finite vertices (rasterizer culls non-finite faces; the
            # excluded count is a model-quality diagnostic).
            finite = ctx.finite_faces_mask()
            verts_np = ctx.vertices().cpu().numpy()
            faces_np = ctx.faces()[finite].cpu().numpy()
            n_nonfinite_faces = int((~finite).sum().item())
            # Adapt SceneSpec.gt to the downstream contract: it accepts
            # 'mesh_path' (.obj) or 'scan_points' [N,3]; scan .ply files
            # (courtyard) are loaded here by the caller.
            if spec.gt.get("mesh_path") and os.path.isfile(spec.gt["mesh_path"]):
                gt_arg = {"mesh_path": spec.gt["mesh_path"]}
                if spec.gt.get("mesh_transform") is not None:
                    # 4x4 into the trainer frame (e.g. SS3DM cm->m + mirror);
                    # applied by downstream_metrics._build_gt_occupancy.
                    gt_arg["mesh_transform"] = spec.gt["mesh_transform"]
            else:
                import trimesh
                transforms = spec.gt.get("scan_transforms")
                clouds = []
                for i, p in enumerate(spec.gt["scan_paths"]):
                    v = np.asarray(trimesh.load(p, process=False).vertices,
                                   dtype=np.float64)
                    if transforms is not None:
                        M = np.asarray(transforms[i], dtype=np.float64)
                        v = v @ M[:3, :3].T + M[:3, 3]
                    clouds.append(v)
                gt_arg = {"scan_points": np.concatenate(clouds, axis=0)}
            downstream = compute_downstream_metrics(
                verts_np, faces_np, gt_arg, spec.roi, seed=0
            )
            # Persist the per-sample arrays (they feed the paired bootstrap)
            # as npz and keep only JSON-safe scalars in metrics.json.
            ds_dir = os.path.join(args.out, "downstream")
            os.makedirs(ds_dir, exist_ok=True)
            for sub_name, sub in downstream.items():
                if not isinstance(sub, dict):
                    continue
                arrays = {k: v for k, v in sub.items() if isinstance(v, np.ndarray)}
                if arrays:
                    npz_path = os.path.join(ds_dir, f"{sub_name}_per_sample.npz")
                    np.savez_compressed(npz_path, **arrays)
                    for k in arrays:
                        del sub[k]
                    sub["per_sample_npz"] = npz_path
            downstream["n_nonfinite_faces_excluded"] = n_nonfinite_faces

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

    ecr_block = None
    if ecr is not None:
        reads = ecr.read_log()
        cache_root = os.path.realpath(args.ecr_cache)
        manifest_files = {
            os.path.realpath(os.path.join(cache_root, rel))
            for rel in ecr.manifest["sizes"]["files"]
        }
        reads_report = {
            "cache_root": cache_root,
            "n_reads": len(reads),
            "n_manifest_files": len(manifest_files),
            "all_reads_in_manifest": all(r in manifest_files for r in reads),
            "reads_outside_manifest": sorted(
                r for r in reads if r not in manifest_files),
            "reads": reads,
        }
        reads_path = os.path.join(args.out, "ecr_transport_reads.json")
        with open(reads_path, "w") as f:
            json.dump(reads_report, f, indent=1)
        kwargs_hashes = {v["kwargs_hash"] for v in ecr_view_infos}
        ecr_block = {
            "cache_dir": cache_root,
            "manifest_sha256": ecr.manifest_sha256,
            "config_hash": ecr.config_hash,
            "alpha": float(ecr.manifest["alpha"]["alpha"]),
            "alpha_source": ecr.manifest["alpha"].get("source"),
            "n_train_views": int(ecr.manifest["n_train_views"]),
            "per_view": ecr_view_infos,
            "per_view_kwargs_identical": len(kwargs_hashes) == 1
                and next(iter(kwargs_hashes)) == ecr.config_hash,
            "transport_reads_json": reads_path,
            "all_reads_in_manifest": reads_report["all_reads_in_manifest"],
        }

    metrics = {
        "protocol_version": PROTOCOL_VERSION,
        "renderer": args.renderer,
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
        "ecr": ecr_block,
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
