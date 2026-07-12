#!/usr/bin/env python
"""TOPCONF EXP-IBR (GOAL #E-12): job-file generation for the IBRNet cell.

Camera math lives HERE (not in the delegated adapter) and is verified by the
self-reconstruction gate before any cross-method number is computed:
render a TRAIN view from its own pose given its 10 nearest other train views
— a correct convention reconstructs it far above chance.

Conventions: our repo stores world_view_transform as the TRANSPOSED
world-to-camera matrix (row-vector convention); c2w = inv(wvt^T). Axes are
OpenCV (x right, y down, z forward) end-to-end — the same convention the
adapter passes through (determined from IBRNet's own LLFF loader).
Principal point = image center (the repo's own projection assumes centered
pp; consistent with its fov-based intrinsics).

Sources per target = top-10 train views by the SAME frozen camera score the
transport uses (select_support_frames, direction_weight 0.35) — a SUPERSET
of the evidence rights the ECR transport gets (its calibrated K is 2–8),
and IBRNet's own training regime (8–10 sources). Amendment to the #E-12
pre-registration logged BEFORE any numbers existed.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
G1 = "/data/peilincai/gems_stage1"
N_SOURCES = 10


def c2w_from_wvt(wvt_rows):
    import numpy as np
    w2c = np.asarray(wvt_rows, dtype=np.float64).T
    return np.linalg.inv(w2c)


def cam_json(width, height, fx, fy, c2w):
    return {"width": int(width), "height": int(height),
            "fx": float(fx), "fy": float(fy),
            "cx": float(width) / 2.0, "cy": float(height) / 2.0,
            "c2w": [[float(v) for v in row] for row in c2w]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--cache", required=True,
                    help="evidence cache dir (cameras + train gt + depths)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=("selftest", "testset"),
                    default="testset")
    ap.add_argument("--checkpoint", default=None,
                    help="needed for testset mode (test poses via the mouth's"
                         " own context)")
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import numpy as np
    from utils.evidence_lumigraph_adapter import (
        FrameRecord, load_camera_index, select_support_frames)

    cache = args.cache
    cams = {c.image_name: c for c in
            load_camera_index(Path(cache) / "camera_index.json")}
    manifest = json.load(open(os.path.join(cache, "manifest.json")))
    train_names = list(manifest["train_views"])
    gext = str(manifest.get("image_ext", {}).get("gt", "png"))
    frames = [FrameRecord(idx=i, name=n, render_path="", gt_path="",
                          depth_path=os.path.join(cache, "depths", f"{n}.npy"),
                          camera=cams[n])
              for i, n in enumerate(train_names)]

    def depth_range(source_frames):
        lo, hi = [], []
        for f in source_frames:
            d = np.load(f.depth_path)
            v = d[np.isfinite(d) & (d > 0)]
            if v.size:
                lo.append(np.percentile(v, 2))
                hi.append(np.percentile(v, 98))
        return [max(0.05, float(min(lo)) * 0.8), float(max(hi)) * 1.25]

    def sources_for(target_frame, exclude):
        picked = select_support_frames(target_frame, frames, k=N_SOURCES,
                                       exclude_names=exclude)
        return [f for f, _ in picked]

    os.makedirs(args.out_dir, exist_ok=True)
    jobs = []
    if args.mode == "selftest":
        # median-index train view as target; its GT is the reference image
        t = frames[len(frames) // 2]
        srcs = sources_for(t, exclude=(t.name,))
        job = {
            "target": cam_json(t.camera.width, t.camera.height,
                               t.camera.fx, t.camera.fy,
                               c2w_from_wvt(t.camera.world_view_transform)),
            "sources": [dict(
                image_path=os.path.join(cache, "gt", f"{s.name}.{gext}"),
                **cam_json(s.camera.width, s.camera.height, s.camera.fx,
                           s.camera.fy,
                           c2w_from_wvt(s.camera.world_view_transform)))
                for s in srcs],
            "depth_range": depth_range(srcs),
            "out_path": os.path.join(args.out_dir, f"selftest_{t.name}.png"),
            "_reference_gt": os.path.join(cache, "gt", f"{t.name}.{gext}"),
            "_target_name": t.name,
        }
        path = os.path.join(args.out_dir, "job_selftest.json")
        json.dump(job, open(path, "w"), indent=1)
        jobs.append(path)
    else:
        from tools.gems.scenes import SCENES
        from tools.gems.eval_context import build_eval_context
        ctx = build_eval_context(args.checkpoint, SCENES[args.scene])
        for cam in ctx.test_cams:
            name = str(cam.image_name)
            from utils.evidence_lumigraph_adapter import CameraRecord
            from utils.graphics_utils import fov2focal
            w, h = int(cam.image_width), int(cam.image_height)
            wvt = cam.world_view_transform.detach().cpu().tolist()
            trec = FrameRecord(
                idx=-1, name=name, render_path="", gt_path="", depth_path="",
                camera=CameraRecord(
                    idx=-1, image_name=name, width=w, height=h,
                    fx=float(fov2focal(float(cam.FoVx), w)),
                    fy=float(fov2focal(float(cam.FoVy), h)),
                    camera_center=tuple(
                        float(x) for x in
                        cam.camera_center.detach().cpu().tolist()),
                    world_view_transform=tuple(
                        tuple(float(v) for v in row) for row in wvt)))
            srcs = sources_for(trec, exclude=(name,))
            job = {
                "target": cam_json(w, h, trec.camera.fx, trec.camera.fy,
                                   c2w_from_wvt(wvt)),
                "sources": [dict(
                    image_path=os.path.join(cache, "gt", f"{s.name}.{gext}"),
                    **cam_json(s.camera.width, s.camera.height, s.camera.fx,
                               s.camera.fy,
                               c2w_from_wvt(s.camera.world_view_transform)))
                    for s in srcs],
                "depth_range": depth_range(srcs),
                "out_path": os.path.join(args.out_dir, f"{name}.png"),
                "_target_name": name,
            }
            path = os.path.join(args.out_dir, f"job_{name}.json")
            json.dump(job, open(path, "w"), indent=1)
            jobs.append(path)
    json.dump(jobs, open(os.path.join(args.out_dir, "jobs_index.json"), "w"),
              indent=1)
    print(f"wrote {len(jobs)} job files to {args.out_dir}")


if __name__ == "__main__":
    main()
