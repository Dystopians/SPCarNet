#!/usr/bin/env python
"""TOPCONF EXP-TEMP (GOAL #E-13): temporal / view-path stability of the ECR
transport. GT-FREE by construction (synthetic poses have no ground truth;
nothing here touches any GT), so there is no purity surface — but the
transport still runs through the audited EcrRenderer boundary unchanged.

Per scene: deterministic smooth camera path (Catmull-Rom on camera centers +
quaternion slerp on rotations, through the name-ordered TEST poses, fixed
frame count); per frame render base + ECR-final (metric-path quantization);
metrics = temporal roughness (mean and P95 over t of mean|I_t - I_{t-1}|)
for base and final + per-step support-set switch count; side-by-side video
(base | final | beta) for supplementary.
"""
import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
FRAMES = 120  # frozen in the pre-registration
FFMPEG = os.path.expanduser("~/.local/bin/ffmpeg")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ecr-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    return ap.parse_args()


def catmull_rom(points, n_out):
    """Uniform Catmull-Rom through control points -> n_out samples."""
    import numpy as np
    pts = np.asarray(points, dtype=np.float64)
    k = len(pts)
    padded = np.concatenate([pts[:1], pts, pts[-1:]], axis=0)
    ts = np.linspace(0, k - 1, n_out)
    out = np.empty((n_out, pts.shape[1]))
    for i, t in enumerate(ts):
        seg = min(int(t), k - 2)
        u = t - seg
        p0, p1, p2, p3 = padded[seg], padded[seg + 1], padded[seg + 2], padded[seg + 3]
        out[i] = 0.5 * ((2 * p1) + (-p0 + p2) * u +
                        (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u +
                        (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
    return out


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    import numpy as np
    import torch
    import torchvision
    import matplotlib
    from scipy.spatial.transform import Rotation, Slerp
    from tools.gems.scenes import SCENES
    from tools.gems.eval_context import build_eval_context
    from tools.ecr.renderer import (EcrRenderer, TARGET_GT_SENTINEL,
                                    TARGET_RENDER_PREFIX, TARGET_DEPTH_PREFIX,
                                    camera_record_from_pose)
    from tools.ecr.fusion import compute_transport_features, features_to_input_routed
    from utils.evidence_lumigraph_adapter import FrameRecord
    from utils.graphics_utils import getProjectionMatrix
    from scene.cameras import MiniCam
    from pathlib import Path

    ctx = build_eval_context(args.checkpoint, SCENES[args.scene])
    ecr = EcrRenderer(args.ecr_cache)
    cams = sorted(ctx.test_cams, key=lambda c: str(c.image_name))
    ref = cams[0]

    # deterministic path through the test poses (world-to-cam -> c2w)
    w2c = [c.world_view_transform.detach().cpu().numpy().T for c in cams]
    c2w = [np.linalg.inv(m) for m in w2c]
    centers = np.stack([m[:3, 3] for m in c2w])
    rots = Rotation.from_matrix(np.stack([m[:3, :3] for m in c2w]))
    key_ts = np.arange(len(cams), dtype=np.float64)
    slerp = Slerp(key_ts, rots)
    path_ts = np.linspace(0, len(cams) - 1, FRAMES)
    path_centers = catmull_rom(centers, FRAMES)
    path_rots = slerp(path_ts)

    proj = getProjectionMatrix(znear=0.01, zfar=100.0,
                               fovX=ref.FoVx, fovY=ref.FoVy).transpose(0, 1).cuda()

    def cmap(plane, gain=1.0):
        x = (plane.squeeze() * gain).clamp(0, 1).detach().cpu().numpy()
        return torch.from_numpy(
            matplotlib.colormaps["magma"](x)[..., :3]).permute(2, 0, 1).float()

    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    quant = lambda img: img.mul(255).add(0.5).clamp_(0, 255).to(
        torch.uint8).float().div(255.0)

    rough_b, rough_f, switches = [], [], []
    prev_b = prev_f = prev_support = None
    tr = dict(ecr.manifest["transport"])
    feat_kwargs = {k: tr[k] for k in
                   ("k", "residual_clip", "min_confidence", "depth_abs_tol",
                    "depth_rel_tol", "direction_weight") if k in tr}
    feat_kwargs["fuse"] = tr.get("inner_fuse", "single")
    if "bands" in tr:
        feat_kwargs["bands"] = tr["bands"]

    with torch.no_grad():
        for i in range(FRAMES):
            c2w_i = np.eye(4)
            c2w_i[:3, :3] = path_rots[i].as_matrix()
            c2w_i[:3, 3] = path_centers[i]
            w2c_i = np.linalg.inv(c2w_i)
            wvt = torch.tensor(w2c_i.T, dtype=torch.float32).cuda()
            full_proj = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
            cam = MiniCam(int(ref.image_width), int(ref.image_height),
                          float(ref.FoVy), float(ref.FoVx), 0.01, 100.0,
                          wvt, full_proj)
            pkg = ctx.render_view(cam)
            base = quant(pkg["render"][:3])
            name = f"path_{i:04d}"
            pose = {"image_name": name,
                    "width": int(ref.image_width),
                    "height": int(ref.image_height),
                    "fovx": float(ref.FoVx), "fovy": float(ref.FoVy),
                    "camera_center": [float(x) for x in
                                      cam.camera_center.detach().cpu().tolist()],
                    "world_view_transform": wvt.detach().cpu().tolist()}
            adapted, info = ecr.adapt(name, pose, base, pkg["surf_depth"][0])
            final = quant(adapted)

            # beta panel (routed caches): recompute maps at this pose
            beta_rgb = torch.zeros_like(base)
            if ecr.fuse == "routed":
                camera = camera_record_from_pose(idx=-1, pose=pose)
                target = FrameRecord(
                    idx=-1, name=name,
                    render_path=Path(TARGET_RENDER_PREFIX + name),
                    gt_path=Path(TARGET_GT_SENTINEL),
                    depth_path=Path(TARGET_DEPTH_PREFIX + name),
                    camera=camera)
                ecr.loader.register_target(name, base, pkg["surf_depth"][0])
                try:
                    feats = compute_transport_features(
                        target, ecr.train_frames, loader=ecr.loader,
                        device=ecr.device, with_color=True, **feat_kwargs)
                    maps = ecr._fusion_net(
                        features_to_input_routed(feats).unsqueeze(0))[0]
                    valid = (feats["weight_den"] > float(
                        feat_kwargs.get("min_confidence", 1e-4))).float()
                    beta_rgb = cmap(maps[1:2] * valid)
                finally:
                    ecr.loader.clear_target(name)

            if prev_b is not None:
                rough_b.append(float((base - prev_b).abs().mean()))
                rough_f.append(float((final - prev_f).abs().mean()))
                cur = set(info.get("support_names") or [])
                switches.append(len(cur.symmetric_difference(prev_support)) / 2.0)
            prev_b, prev_f = base, final
            prev_support = set(info.get("support_names") or [])

            strip = torch.cat([base, final, beta_rgb.to(base.device)], dim=2)
            torchvision.utils.save_image(
                strip, os.path.join(frames_dir, f"{i:04d}.png"))
            if i % 20 == 0:
                print(f"  frame {i}/{FRAMES}", flush=True)

    rb, rf = np.array(rough_b), np.array(rough_f)
    summary = {
        "scene": args.scene, "frames": FRAMES,
        "roughness_base_mean": float(rb.mean()),
        "roughness_final_mean": float(rf.mean()),
        "roughness_ratio_mean": float(rf.mean() / rb.mean()),
        "roughness_base_p95": float(np.percentile(rb, 95)),
        "roughness_final_p95": float(np.percentile(rf, 95)),
        "roughness_ratio_p95": float(np.percentile(rf, 95) /
                                     np.percentile(rb, 95)),
        "support_switches_per_step_mean": float(np.mean(switches)),
        "support_switches_per_step_max": float(np.max(switches)),
        "acceptance_ratio_le_1p5": bool(rf.mean() / rb.mean() <= 1.5),
    }
    with open(os.path.join(args.out, "temporal.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    video = os.path.join(args.out, f"{args.scene}_path.mp4")
    # this host's ffmpeg lacks libx264; mpeg4 is built in and adequate for
    # a supplementary preview (frames are kept as the lossless source)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", "24",
                    "-i", os.path.join(frames_dir, "%04d.png"),
                    "-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", video], check=True)
    print(json.dumps(summary, indent=1))
    print(f"wrote {video}")


if __name__ == "__main__":
    main()
