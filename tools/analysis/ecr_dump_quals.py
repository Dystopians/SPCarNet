#!/usr/bin/env python
"""GEMS Stage-4 OFFLINE qualitative/failure-case dump (prompt §5 Tier-1).

NOT part of the evaluation mouth or the audited render path: this tool runs
AFTER the metric rows are banked, purely to produce figure assets. It uses
test-view GT only to visualize error maps next to already-reported numbers
(the transport itself still runs through the same ConfinedFrameLoader /
pose-primitive boundary as run_eval). Lives outside tools/ecr on purpose so
the --ecr audit surface is unchanged.

Per test view it writes:
  base.png final.png gt.png          — 8-bit, metric-path quantization
  err_base.png err_final.png         — mean|.-gt| x4 gain, inferno colormap
  conf.png                           — transport confidence (weight_den/4)
  count.png                          — support count map (/8)
  alpha.png [beta.png]               — fusion-net maps (learned/routed caches)
plus summary.json (per-view PSNR base/final, covered_fraction, means) to pick
best / median / failure views for the grids.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ecr-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--views", nargs="*", default=None,
                    help="subset of test view names (default: all)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    import torch
    import torchvision
    import matplotlib
    from tools.gems.scenes import SCENES
    from tools.gems.eval_context import build_eval_context
    from tools.ecr.renderer import (EcrRenderer, TARGET_GT_SENTINEL,
                                    TARGET_RENDER_PREFIX, TARGET_DEPTH_PREFIX,
                                    camera_record_from_pose)
    from tools.ecr.fusion import compute_transport_features
    from utils.evidence_lumigraph_adapter import FrameRecord
    from utils.image_utils import psnr
    from run_eval import pose_primitives
    from pathlib import Path

    spec = SCENES[args.scene]
    ctx = build_eval_context(args.checkpoint, spec)
    ecr = EcrRenderer(args.ecr_cache)
    want = set(args.views) if args.views else None

    def quantize(img):
        return img.mul(255).add(0.5).clamp_(0, 255).to(
            torch.uint8).float().div(255.0)

    def save(img, path):
        torchvision.utils.save_image(img[:3].clamp(0, 1), path)

    def cmap(plane, name, gain=1.0):
        x = (plane.squeeze() * gain).clamp(0, 1).detach().cpu().numpy()
        return torch.from_numpy(
            matplotlib.colormaps[name](x)[..., :3]).permute(2, 0, 1).float()

    # feature kwargs for the visualization planes (same frozen transport
    # geometry the renderer uses; fuse inner mode for learned/routed caches)
    tr = dict(ecr.manifest["transport"])
    feat_kwargs = {k: tr[k] for k in
                   ("k", "residual_clip", "min_confidence", "depth_abs_tol",
                    "depth_rel_tol", "direction_weight") if k in tr}
    feat_kwargs["fuse"] = tr.get("inner_fuse", "single") \
        if ecr.fuse in ("learned", "routed") else \
        ("multiband" if ecr.fuse == "multiband" else "single")
    if "bands" in tr:
        feat_kwargs["bands"] = tr["bands"]

    os.makedirs(args.out, exist_ok=True)
    summary = {"scene": args.scene, "cache": str(args.ecr_cache),
               "fuse": ecr.fuse, "views": {}}
    with torch.no_grad():
        for cam in ctx.test_cams:
            name = str(cam.image_name)
            if want is not None and name not in want:
                continue
            pkg = ctx.render_view(cam)
            base = quantize(pkg["render"])
            depth = pkg["surf_depth"][0]
            adapted, info = ecr.adapt(name, pose_primitives(cam), base, depth)
            final = quantize(adapted)
            gt = quantize(cam.original_image[:3].to(final.device)
                          .clamp(0.0, 1.0))
            p_base = psnr(base.unsqueeze(0), gt.unsqueeze(0)).mean().item()
            p_final = psnr(final.unsqueeze(0), gt.unsqueeze(0)).mean().item()

            vdir = os.path.join(args.out, name)
            os.makedirs(vdir, exist_ok=True)
            save(base, os.path.join(vdir, "base.png"))
            save(final, os.path.join(vdir, "final.png"))
            save(gt, os.path.join(vdir, "gt.png"))
            save(cmap((base - gt).abs().mean(0), "inferno", 4.0),
                 os.path.join(vdir, "err_base.png"))
            save(cmap((final - gt).abs().mean(0), "inferno", 4.0),
                 os.path.join(vdir, "err_final.png"))

            # transport maps (recomputed with the frozen geometry config)
            camera = camera_record_from_pose(idx=-1, pose=pose_primitives(cam))
            target = FrameRecord(
                idx=-1, name=name,
                render_path=Path(TARGET_RENDER_PREFIX + name),
                gt_path=Path(TARGET_GT_SENTINEL),
                depth_path=Path(TARGET_DEPTH_PREFIX + name),
                camera=camera)
            ecr.loader.register_target(name, base, depth)
            try:
                feats = compute_transport_features(
                    target, ecr.train_frames, loader=ecr.loader,
                    device=ecr.device, with_color=(ecr.fuse == "routed"),
                    **feat_kwargs)
                save(cmap(feats["weight_den"] / 4.0, "viridis"),
                     os.path.join(vdir, "conf.png"))
                save(cmap(feats["support_count"] / 8.0, "viridis"),
                     os.path.join(vdir, "count.png"))
                if ecr._fusion_net is not None:
                    from tools.ecr.fusion import (features_to_input,
                                                  features_to_input_routed)
                    x = (features_to_input_routed(feats) if ecr.fuse == "routed"
                         else features_to_input(feats)).unsqueeze(0)
                    maps = ecr._fusion_net(x)[0]
                    save(cmap(maps[0], "magma"),
                         os.path.join(vdir, "alpha.png"))
                    if maps.shape[0] > 1:
                        valid = (feats["weight_den"] > float(
                            feat_kwargs.get("min_confidence", 1e-4))).float()
                        save(cmap(maps[1:2] * valid, "magma"),
                             os.path.join(vdir, "beta.png"))
            finally:
                ecr.loader.clear_target(name)

            summary["views"][name] = {
                "psnr_base": p_base, "psnr_final": p_final,
                "dpsnr": p_final - p_base,
                "covered_fraction": info.get("covered_fraction"),
                "mean_confidence": info.get("mean_confidence"),
                "alpha_mean": info.get("alpha_mean"),
                "beta_mean": info.get("beta_mean"),
            }
            print(f"  {name}: base {p_base:.3f} -> final {p_final:.3f} "
                  f"({p_final - p_base:+.3f})")
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {args.out}/summary.json ({len(summary['views'])} views)")


if __name__ == "__main__":
    main()
