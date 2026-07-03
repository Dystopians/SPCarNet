#!/usr/bin/env python
"""GEMS Stage Two, MATRIX cell D-1b: SS3DM -> trainer COLMAP ingestion.

Converts one SS3DM raw sequence (CARLA export, see
docs/ss3dm_prior_data_schema.md and
/data/peilincai/mesh_datasets/SS3DM/ACQUISITION_LOG.md) into the COLMAP-text
dataset layout consumed by scene/dataset_readers.py::readColmapSceneInfo,
following the conventions established by tools/gems/build_toy_parking.py:

  <out>/images/<cam>_<frame>.jpg        (symlinks into the raw dataset)
  <out>/sparse/0/cameras.txt            PINHOLE, one entry per unique intrinsics
  <out>/sparse/0/images.txt             qvec/tvec in COLMAP world-to-cam
  <out>/sparse/0/points3D.txt + .ply    ~100k LiDAR init cloud with projected colors
  <out>/split.json                      {"train": [...], "test": [...]} stems;
                                        whole frame idx % 8 == 0 -> test
  <out>/dataset_manifest.json           counts, policy, verification numbers

Conventions verified against the raw data before writing this file:
  - scenario.pt is a raw pickle (protocol 4), NOT a torch-zip checkpoint;
    loaded via ss3dm_prior.data.scenario_loader.load_scenario (pickle fallback).
  - camera 'c2w' (150,4,4) is CAM-TO-WORLD in OpenCV axes (x right, y down,
    z forward): projecting lidar_TOP world points with inv(c2w) puts ~45% of a
    360deg sweep in front of camera_FRONT, and the projected depths match the
    GT depth PNGs within ~1% (p10-p90).
  - SCHEMA SURPRISE: every c2w rotation block has det = -1 (orthonormal but
    IMPROPER). The CARLA/UE scenario world is LEFT-HANDED, so the raw basis
    cannot be represented by a COLMAP quaternion. Fix: mirror the whole world
    with WORLD_FLIP = diag(1,-1,1) (X' = M X for LiDAR points, C' = M t and
    R' = M R for cameras). Projection is invariant under a global mirror
    ((M a)-(M b) = a-b for dot products), so pixels/depths are untouched and
    det(M R) = +1. The SAME flip must be applied to anything else brought into
    this trainer world later (LiDAR evidence rays, GT mesh at g4 time).
  - LiDAR npz rays_o/rays_d/ranges are already WORLD frame -- the RAW
    left-handed world (rays_o == sensor world position, ~1.5 m from the FRONT
    camera center); world points are rays_o + rays_d * ranges[:, None]
    (schema doc rule), then y-negated into the exported trainer world.
  - depth_gts PNGs are uint16 with value = meters * 65535/1000 (measured
    ratio ~65.48 vs LiDAR z); used only for occlusion-gating point colors.
  - metas: world_offset = [0,0,0], up_vec = '+z'; CARLA world is metric.
  - GT town OBJ meshes are in CENTIMETERS (x0.01 -> meters, matches schema
    'town_mesh_unit_scale: 0.01'); recorded in the manifest, never rescaled
    here.

Camera/stride policy (frozen for all four towns, see --cameras/--frame-stride
defaults and the manifest 'policy' block): 3 front cameras x all 150 frames
= 450 images. 900 images (6 cams) at full res would need ~22.4 GB just for
GT images on the trainer's cuda data_device; 450 at -r 2 (960x540) is
~2.8 GB. Front-3 over 6-cams-stride-2 (same count): keeps full temporal
baseline density and matches the sequence's streetsurf (front-camera) design.
Images are 1920 px wide (>1600) -> train with -r 2 instead of resizing on
disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

DEFAULT_DATA_ROOT = "/data/peilincai/mesh_datasets/SS3DM"
DEFAULT_OUT_ROOT = "/data/peilincai/gems_stage1/datasets"
TOWNS = ["Town01", "Town02", "Town03", "Town06"]
ALL_CAMERA_TOKENS = ["front_left", "front", "front_right",
                     "back_left", "back", "back_right"]
DEFAULT_CAMERAS = "front_left,front,front_right"
LIDAR_NAMES = ["lidar_TOP", "lidar_FRONT", "lidar_LEFT", "lidar_RIGHT", "lidar_REAR"]
TEST_EVERY_FRAME = 8          # whole frame idx % 8 == 0 -> test (all cams of it)
MIN_RANGE_M, MAX_RANGE_M = 0.5, 120.0
GRAY = np.array([128, 128, 128], dtype=np.uint8)

# Raw SS3DM world is LEFT-handed (all c2w rotation dets are -1). Mirror the
# world across y so COLMAP quaternions can represent the poses; apply to every
# world-frame quantity exported here (camera c2w, init-cloud points).
WORLD_FLIP = np.diag([1.0, -1.0, 1.0])
WORLD_FLIP4 = np.diag([1.0, -1.0, 1.0, 1.0])

# trainer VRAM accounting (scene/cameras.py stores original_image float32 on
# data_device='cuda'): bytes = W*H*3*4 per image.
FULLRES_MB = 1920 * 1080 * 3 * 4 / 1e6      # 24.88 MB
R2_MB = 960 * 540 * 3 * 4 / 1e6             # 6.22 MB


def cam_token_to_name(token: str) -> str:
    token = token.strip().lower()
    if token not in ALL_CAMERA_TOKENS:
        raise ValueError(f"unknown camera token '{token}' (expected {ALL_CAMERA_TOKENS})")
    return "camera_" + token.upper()


def load_sequence(seq_root: str):
    """scenario.pt via the schema-doc loader (torch first, pickle fallback)."""
    from ss3dm_prior.data.scenario_loader import load_scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sd = load_scenario(seq_root)
    if sd.raw_payload is None:
        raise RuntimeError(f"scenario.pt could not be parsed at {seq_root} "
                           f"(source_format={sd.source_format})")
    return sd


def lidar_world_points(seq_root: str, frame: int, lidar_names=LIDAR_NAMES):
    """World points from every LiDAR of one frame (schema rule, range-filtered)."""
    pts = []
    for lname in lidar_names:
        z = np.load(os.path.join(seq_root, "lidars", lname, f"{frame:08d}.npz"))
        rays_o, rays_d, ranges = z["rays_o"], z["rays_d"], z["ranges"]
        p = rays_o + rays_d * ranges[:, None]
        ok = (np.isfinite(p).all(axis=1) & np.isfinite(ranges)
              & (ranges > MIN_RANGE_M) & (ranges <= MAX_RANGE_M))
        pts.append(p[ok])
    return np.concatenate(pts, axis=0)


# ---------------------------------------------------------------------------
# Convention check: project LiDAR world points into an image
# ---------------------------------------------------------------------------
def lidar_projection_check(seq_root, scenario, cam_name, frames):
    """Verify c2w = cam-to-world (OpenCV) by projecting lidar_TOP world points.

    Criterion: projected LiDAR depths must match the GT depth PNGs with a
    TIGHT ratio spread (a wrong convention scatters pixels, destroying the
    correlation). The 'opposite reading' in-front fraction is reported as info
    only -- it is scene-sign-dependent and not a valid criterion.

    Returns (report dict, depth_png_scale) and raises if the convention or the
    depth/pixel sanity fails. All math here is in the RAW (left-handed) frame;
    projection is invariant under the global WORLD_FLIP mirror.
    """
    from PIL import Image

    cam = scenario.cameras[cam_name]
    dets = np.linalg.det(cam.c2w[:, :3, :3])
    per_frame = []
    ratios = []
    for fi in frames:
        z = np.load(os.path.join(seq_root, "lidars", "lidar_TOP", f"{fi:08d}.npz"))
        pts = z["rays_o"] + z["rays_d"] * z["ranges"][:, None]
        pts = pts[np.isfinite(pts).all(axis=1) & (z["ranges"] > MIN_RANGE_M)]
        c2w = cam.c2w[fi]
        K = cam.intr[fi]
        w2c = np.linalg.inv(c2w)
        Xc = pts @ w2c[:3, :3].T + w2c[:3, 3]
        zc = Xc[:, 2]
        front = zc > 0.1
        u = K[0, 0] * Xc[:, 0] / np.where(front, zc, 1.0) + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / np.where(front, zc, 1.0) + K[1, 2]
        H, W = int(cam.hw[fi][0]), int(cam.hw[fi][1])
        inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        # opposite reading (treat c2w as w2c): should put far fewer points in front
        Xc_opp = pts @ c2w[:3, :3].T + c2w[:3, 3]
        front_opp = (Xc_opp[:, 2] > 0.1).mean()

        dpath = os.path.join(seq_root, "depth_gts", cam_name, f"{fi:08d}.png")
        frame_ratios = np.array([])
        if os.path.exists(dpath):
            d = np.asarray(Image.open(dpath), dtype=np.float64)
            ui, vi = u[inb].astype(int), v[inb].astype(int)
            dpx = d[vi, ui]
            ok = dpx > 0
            frame_ratios = dpx[ok] / zc[inb][ok]
            ratios.append(frame_ratios)
        per_frame.append({
            "frame": int(fi), "n_points": int(len(pts)),
            "in_front_frac": round(float(front.mean()), 4),
            "in_bounds_frac": round(float(inb.mean()), 4),
            "opposite_convention_in_front_frac": round(float(front_opp), 4),
            "n_depth_compared": int(frame_ratios.size),
        })
    if not ratios:
        raise RuntimeError("no depth_gts PNGs found; cannot verify the projection convention")
    ratios = np.concatenate(ratios)
    med = float(np.median(ratios))
    p10 = float(np.percentile(ratios, 10))
    p90 = float(np.percentile(ratios, 90))
    report = {
        "camera": cam_name,
        "raw_c2w_rotation_det": {"min": round(float(dets.min()), 6),
                                 "max": round(float(dets.max()), 6),
                                 "note": "det=-1: raw world is LEFT-handed; exported "
                                         "world is mirrored by diag(1,-1,1)"},
        "per_frame": per_frame,
        "depth_png_over_lidar_z_ratio": {
            "median": round(med, 4), "p10": round(p10, 4), "p90": round(p90, 4),
            "expected_uint16_scale": 65.535,  # meters * 65535/1000
        },
        "conclusion": "c2w is cam-to-world (OpenCV axes, improper det=-1 in the raw "
                      "left-handed world); LiDAR rays are raw-world-frame",
    }
    mean_front = np.mean([f["in_front_frac"] for f in per_frame])
    if mean_front < 0.25:
        raise RuntimeError(f"c2w convention check FAILED: only {mean_front:.3f} of "
                           "360deg lidar_TOP points project in front of the camera")
    if abs(med / 65.535 - 1.0) > 0.05:
        raise RuntimeError(f"depth-png/LiDAR-z scale check FAILED: median ratio {med:.3f} "
                           f"(expected ~65.535)")
    if (p90 - p10) / med > 0.15:
        raise RuntimeError(f"depth-ratio spread check FAILED: p10={p10:.2f} p90={p90:.2f} "
                           f"median={med:.2f} (pixels likely misprojected)")
    print(f"[verify] LiDAR->image convention OK: in_front={mean_front:.3f}, depth "
          f"ratio median {med:.3f} (p10 {p10:.2f} / p90 {p90:.2f}) ~ 65.535; "
          f"raw c2w det={dets.min():.3f} (left-handed, exporting mirrored world)")
    return report, med


# ---------------------------------------------------------------------------
# COLMAP export
# ---------------------------------------------------------------------------
def build_views(scenario, cam_tokens, frames):
    """One view record per (frame, camera): name, K, w2c, image source path."""
    views = []
    for token in cam_tokens:
        cname = cam_token_to_name(token)
        cam = scenario.cameras[cname]
        K0 = cam.intr[0]
        if not np.allclose(cam.intr, cam.intr[0:1]):
            raise RuntimeError(f"{cname}: intrinsics vary across frames")
        if not np.allclose(cam.hw, cam.hw[0:1]):
            raise RuntimeError(f"{cname}: image size varies across frames")
        H, W = int(cam.hw[0][0]), int(cam.hw[0][1])
        for fi in frames:
            # Mirror the left-handed raw world into the right-handed trainer
            # world: c2w' = diag(1,-1,1,1) @ c2w (det(R') = +1).
            c2w = WORLD_FLIP4 @ cam.c2w[fi]
            w2c = np.linalg.inv(c2w)
            views.append({
                "name": f"{token}_{fi:08d}",
                "cam_name": cname, "token": token, "frame": int(fi),
                "W": W, "H": H,
                "fx": float(K0[0, 0]), "fy": float(K0[1, 1]),
                "cx": float(K0[0, 2]), "cy": float(K0[1, 2]),
                "R_w2c": w2c[:3, :3], "t_w2c": w2c[:3, 3],
                "c2w": c2w,
            })
    views.sort(key=lambda v: v["name"])
    return views


def export_colmap(views, sparse_dir, points_xyz, points_rgb):
    from scene.colmap_loader import rotmat2qvec
    from scene.dataset_readers import storePly

    os.makedirs(sparse_dir, exist_ok=True)

    # cameras.txt: one PINHOLE entry per unique intrinsics tuple.
    intr_ids: dict = {}
    for v in views:
        key = (v["W"], v["H"], round(v["fx"], 6), round(v["fy"], 6),
               round(v["cx"], 6), round(v["cy"], 6))
        if key not in intr_ids:
            intr_ids[key] = len(intr_ids) + 1
        v["camera_id"] = intr_ids[key]
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n"
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                f"# Number of cameras: {len(intr_ids)}\n")
        for (W, H, fx, fy, cx, cy), cid in intr_ids.items():
            f.write(f"{cid} PINHOLE {W} {H} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n")

    # images.txt: qvec/tvec in COLMAP world-to-cam; empty POINTS2D lines.
    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n"
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
                f"# Number of images: {len(views)}\n")
        for i, v in enumerate(views):
            q = rotmat2qvec(v["R_w2c"])
            t = v["t_w2c"]
            f.write(f"{i + 1} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} "
                    f"{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} {v['camera_id']} "
                    f"{v['name']}.jpg\n\n")

    # points3D: minimal text (like toy_parking) + the .ply the reader feeds
    # to create_from_pcd.
    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n"
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
                f"# Number of points: {points_xyz.shape[0]}\n")
        for i in range(points_xyz.shape[0]):
            x, y, z = points_xyz[i]
            r, g, b = points_rgb[i]
            f.write(f"{i + 1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 1.0\n")
    storePly(os.path.join(sparse_dir, "points3D.ply"),
             points_xyz.astype(np.float64), points_rgb.astype(np.float64))
    return len(intr_ids)


# ---------------------------------------------------------------------------
# Init cloud from LiDAR with projected colors
# ---------------------------------------------------------------------------
def build_init_cloud(seq_root, scenario, cam_tokens, frames, n_points,
                     depth_scale, rng):
    from PIL import Image

    all_pts, all_frames = [], []
    for fi in frames:
        p = lidar_world_points(seq_root, fi)
        all_pts.append(p)
        all_frames.append(np.full(p.shape[0], fi, dtype=np.int32))
    pts = np.concatenate(all_pts, axis=0)
    frame_of = np.concatenate(all_frames, axis=0)
    n_raw = pts.shape[0]
    sel = rng.choice(n_raw, size=min(n_points, n_raw), replace=False)
    pts, frame_of = pts[sel], frame_of[sel]

    # Colors: project each point into the cameras of its OWN frame; accept the
    # first camera where the point is in front, in-bounds, and depth-consistent
    # with the GT depth png (occlusion gate). Gray fallback.
    cams = {t: scenario.cameras[cam_token_to_name(t)] for t in cam_tokens}
    rgb = np.tile(GRAY, (pts.shape[0], 1))
    colored = np.zeros(pts.shape[0], dtype=bool)
    for fi in np.unique(frame_of):
        idx = np.nonzero(frame_of == fi)[0]
        P = pts[idx]
        for token in cam_tokens:
            cam = cams[token]
            cname = cam_token_to_name(token)
            need = ~colored[idx]
            if not need.any():
                break
            c2w, K = cam.c2w[fi], cam.intr[fi]
            H, W = int(cam.hw[fi][0]), int(cam.hw[fi][1])
            w2c = np.linalg.inv(c2w)
            Xc = P @ w2c[:3, :3].T + w2c[:3, 3]
            zc = Xc[:, 2]
            front = zc > 0.1
            u = K[0, 0] * Xc[:, 0] / np.where(front, zc, 1.0) + K[0, 2]
            v = K[1, 1] * Xc[:, 1] / np.where(front, zc, 1.0) + K[1, 2]
            ok = need & front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not ok.any():
                continue
            dpath = os.path.join(seq_root, "depth_gts", cname, f"{fi:08d}.png")
            img = np.asarray(Image.open(
                os.path.join(seq_root, "images", cname, f"{fi:08d}.jpg")))
            ui, vi = u[ok].astype(int), v[ok].astype(int)
            if os.path.exists(dpath):
                d_m = np.asarray(Image.open(dpath), dtype=np.float64)[vi, ui] / depth_scale
                zin = zc[ok]
                vis = np.abs(zin - d_m) < np.maximum(0.05 * d_m, 0.5)
            else:
                vis = np.ones(ui.shape[0], dtype=bool)
            gidx = idx[ok]
            rgb[gidx[vis]] = img[vi[vis], ui[vis]]
            colored[gidx[vis]] = True
    return pts, rgb.astype(np.uint8), n_raw, float(colored.mean())


# ---------------------------------------------------------------------------
# Verification (round-trip + trainer loader)
# ---------------------------------------------------------------------------
def verify_roundtrip(views, sparse_dir):
    from scene.colmap_loader import (read_extrinsics_text, read_intrinsics_text,
                                     qvec2rotmat)
    intr = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))
    for c in intr.values():
        assert c.model == "PINHOLE"
    extr = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
    assert len(extr) == len(views), f"{len(extr)} != {len(views)}"
    by_name = {v["name"] + ".jpg": v for v in views}
    max_rerr = max_terr = 0.0
    for img in extr.values():
        v = by_name[img.name]
        max_rerr = max(max_rerr, float(np.abs(qvec2rotmat(img.qvec) - v["R_w2c"]).max()))
        max_terr = max(max_terr, float(np.abs(np.asarray(img.tvec) - v["t_w2c"]).max()))
        assert img.xys.shape == (0, 2) and img.point3D_ids.shape == (0,)
        c = intr[img.camera_id]
        assert (c.width, c.height) == (v["W"], v["H"])
        assert abs(c.params[0] - v["fx"]) < 1e-6
    # raw c2w rotations are orthonormal only to ~1e-7 (float64 export noise);
    # rotmat2qvec snaps to the nearest exact rotation, so allow that floor
    # (toy_parking's 1e-9 bound applies only to exact synthetic rotations).
    assert max_rerr < 1e-6 and max_terr < 1e-9, (max_rerr, max_terr)
    print(f"[verify] COLMAP text round-trip OK: qvec/R err {max_rerr:.2e}, "
          f"tvec err {max_terr:.2e} over {len(views)} views")
    return {"max_R_err": max_rerr, "max_t_err": max_terr}


def verify_scene_info(out_dir, expected_train, expected_test):
    from scene.dataset_readers import readColmapSceneInfo
    info = readColmapSceneInfo(out_dir, "images", True, split_strategy="file",
                               split_file=os.path.join(out_dir, "split.json"))
    n_train, n_test = len(info.train_cameras), len(info.test_cameras)
    n_pts = info.point_cloud.points.shape[0]
    c0 = info.train_cameras[0]
    print(f"[verify] readColmapSceneInfo OK: train={n_train} test={n_test} "
          f"init_points={n_pts} cam0={c0.image_name} {c0.width}x{c0.height} "
          f"radius={info.nerf_normalization['radius']:.2f}")
    assert (n_train, n_test) == (expected_train, expected_test), \
        f"split mismatch: got ({n_train},{n_test}) expected ({expected_train},{expected_test})"
    return {"train": n_train, "test": n_test, "init_points": int(n_pts),
            "width": int(c0.width), "height": int(c0.height),
            "nerf_norm_radius": float(info.nerf_normalization["radius"])}


def sample_mesh_bbox(obj_path, n_chunks=4, chunk_bytes=8_000_000):
    """Cheap sampled vertex bbox of a huge ASCII OBJ (head/quarter offsets)."""
    size = os.path.getsize(obj_path)
    pts = []
    with open(obj_path, "rb") as f:
        for k in range(n_chunks):
            f.seek(size * k // n_chunks)
            f.readline()
            for ln in f.read(chunk_bytes).decode("utf-8", "ignore").splitlines():
                if ln.startswith("v "):
                    parts = ln.split()
                    if len(parts) >= 4:
                        pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    pts = np.asarray(pts)
    return pts.min(0), pts.max(0), len(pts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--town", required=True, choices=TOWNS)
    parser.add_argument("--seq", default="150_streetsurf")
    parser.add_argument("--out", default=None,
                        help=f"default: {DEFAULT_OUT_ROOT}/ss3dm_<town_lower>")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--cameras", default=DEFAULT_CAMERAS,
                        help=f"comma list from {ALL_CAMERA_TOKENS}")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--init-points", type=int, default=100_000)
    parser.add_argument("--copy-images", action="store_true",
                        help="copy jpgs instead of symlinking")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    town_l = args.town.lower()
    out_dir = args.out or os.path.join(DEFAULT_OUT_ROOT, f"ss3dm_{town_l}")
    seq_root = os.path.join(args.data_root, "DATA", args.town, args.seq)
    mesh_path = os.path.join(args.data_root, "meshes", "mesh", f"{args.town}_obj.obj")
    cam_tokens = [t.strip().lower() for t in args.cameras.split(",") if t.strip()]
    rng = np.random.default_rng(args.seed)

    # ---- 1. scenario.pt (pickle-fallback loader) ---------------------------
    scenario = load_sequence(seq_root)
    n_frames = int(scenario.num_frames)
    frames = list(range(0, n_frames, args.frame_stride))
    print(f"[load] {args.town}/{args.seq}: scenario via {scenario.source_format}, "
          f"{n_frames} frames, cameras={sorted(scenario.cameras)}")

    # ---- 2. LiDAR->image convention check -----------------------------------
    check_cam = cam_token_to_name("front" if "front" in cam_tokens else cam_tokens[0])
    check_frames = [frames[0], frames[len(frames) // 2], frames[-1]]
    proj_report, depth_scale = lidar_projection_check(
        seq_root, scenario, check_cam, check_frames)

    # ---- 3. views + images/ --------------------------------------------------
    views = build_views(scenario, cam_tokens, frames)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    for v in views:
        src = os.path.join(seq_root, "images", v["cam_name"], f"{v['frame']:08d}.jpg")
        dst = os.path.join(images_dir, v["name"] + ".jpg")
        if os.path.lexists(dst):
            os.remove(dst)
        if args.copy_images:
            import shutil
            shutil.copyfile(src, dst)
        else:
            os.symlink(src, dst)
    print(f"[images] {len(views)} {'copies' if args.copy_images else 'symlinks'} "
          f"({len(cam_tokens)} cams x {len(frames)} frames) in {images_dir}")

    # ---- 4. init cloud + COLMAP text model ----------------------------------
    pts_raw, rgb, n_raw, colored_frac = build_init_cloud(
        seq_root, scenario, cam_tokens, frames, args.init_points, depth_scale, rng)
    pts = pts_raw @ WORLD_FLIP.T   # raw left-handed world -> exported trainer world
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    n_intr = export_colmap(views, sparse_dir, pts, rgb)
    print(f"[points3D] {pts.shape[0]} pts (from {n_raw} raw LiDAR returns), "
          f"{colored_frac * 100:.1f}% colored by projection, {n_intr} unique intrinsics")

    # ---- 5. split.json (every-8th FRAME is test, all cams of it) ------------
    test_frames = sorted(f for f in frames if f % TEST_EVERY_FRAME == 0)
    split_payload = {
        "train": [v["name"] for v in views if v["frame"] % TEST_EVERY_FRAME != 0],
        "test": [v["name"] for v in views if v["frame"] % TEST_EVERY_FRAME == 0],
    }
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump(split_payload, f, indent=1)
    print(f"[split] frames: {len(frames) - len(test_frames)} train / "
          f"{len(test_frames)} test; images: {len(split_payload['train'])} train / "
          f"{len(split_payload['test'])} test")

    # ---- 6. verification ------------------------------------------------------
    rt = verify_roundtrip(views, sparse_dir)
    si = verify_scene_info(out_dir, len(split_payload["train"]), len(split_payload["test"]))

    # ---- 7. GT mesh accounting (metric-only asset; NOT rescaled here) --------
    mesh_bytes = os.path.getsize(mesh_path)
    mesh_gb = mesh_bytes / 1e9
    bb_min, bb_max, n_sample = sample_mesh_bbox(mesh_path)
    # bbox containment is checked in the RAW (unmirrored) frame: mesh<->world
    # axis conventions (incl. the y flip into the trainer world) are frozen at
    # first geometry eval, not here.
    cam_centers_raw = np.stack([v["c2w"][:3, 3] for v in views]) @ WORLD_FLIP.T
    inside = np.all((cam_centers_raw >= bb_min * 0.01 - 5.0)
                    & (cam_centers_raw <= bb_max * 0.01 + 5.0), axis=1).mean()
    # trimesh ASCII OBJ parse peak RSS is ~3-6x file size (string split + f64).
    ram_lo, ram_hi = 3 * mesh_gb, 6 * mesh_gb
    ram_flag = ram_hi > 16.0
    print(f"[gt] {mesh_path}: {mesh_gb:.2f} GB OBJ (centimeters; x0.01 -> m); "
          f"sampled bbox x0.01 min {np.round(bb_min * 0.01, 1)} max {np.round(bb_max * 0.01, 1)}; "
          f"{inside * 100:.0f}% camera centers inside (+5m); "
          f"expected trimesh RSS {ram_lo:.0f}-{ram_hi:.0f} GB"
          + (" -- >16GB RSS RISK: g4 needs a streaming/decimated variant "
             "(do NOT silently decimate)" if ram_flag else ""))

    # ---- 8. manifest -----------------------------------------------------------
    n_lidar_frames = len(frames)
    manifest = {
        "matrix_cell": "D-1b (SS3DM -> trainer ingestion, GEMS Stage 2)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "converter": "tools/gems_train/ss3dm_ingest.py",
        "seed": args.seed,
        "town": args.town,
        "sequence": args.seq,
        "sequence_root": seq_root,
        "units": "meters (CARLA metric; metas.world_offset=[0,0,0], up_vec=+z)",
        "scenario_source_format": scenario.source_format,
        "policy": {
            "cameras": cam_tokens,
            "frame_stride": args.frame_stride,
            "n_images": len(views),
            "rationale": (
                "FROZEN for all 4 towns: 3 front cameras x 150 frames = 450 images. "
                "scene/cameras.py stores each GT image float32 on data_device='cuda': "
                f"900 imgs (6 cams) @1920x1080 = {900 * FULLRES_MB / 1e3:.1f} GB (infeasible); "
                f"450 @full res = {450 * FULLRES_MB / 1e3:.1f} GB; 450 @-r 2 (960x540) = "
                f"{450 * R2_MB / 1e3:.1f} GB (OK). Front-3 chosen over 6-cams-stride-2 "
                "(same count): keeps full temporal baseline density on the driven "
                "street and matches the sequence's streetsurf front-camera design."),
            "train_resolution_note": (
                "images are 1920 px wide (>1600): train with -r 2 rather than "
                "resizing on disk; registry resolution=2"),
        },
        "images": {
            "count": len(views), "width": views[0]["W"], "height": views[0]["H"],
            "format": "jpg symlinks" if not args.copy_images else "jpg copies",
            "naming": "<camera_token>_<frame:08d>.jpg",
        },
        "intrinsics": {
            "n_unique": n_intr, "model": "PINHOLE",
            "fx": views[0]["fx"], "fy": views[0]["fy"],
            "cx": views[0]["cx"], "cy": views[0]["cy"],
            "distortion": "all-zero in scenario.pt",
        },
        "pose_convention": {
            "scenario_c2w": "cam-to-world, OpenCV axes (x right, y down, z forward); "
                            "rotation det=-1 (raw CARLA world is LEFT-handed)",
            "world_flip": "exported trainer world = diag(1,-1,1) @ raw world (y negated); "
                          "applied to camera c2w and init-cloud points; projection-invariant",
            "colmap_images_txt": "qvec/tvec world-to-cam = inv(diag(1,-1,1,1) @ c2w), "
                                 "toy_parking-style round-trip verified",
        },
        "split": {
            "rule": f"whole frame idx % {TEST_EVERY_FRAME} == 0 -> test (all cameras of "
                    "that frame; avoids near-duplicate leakage between same-frame cams)",
            "train_frames": len(frames) - len(test_frames),
            "test_frames": len(test_frames),
            "train_images": len(split_payload["train"]),
            "test_images": len(split_payload["test"]),
            "test_frame_indices": test_frames,
        },
        "init_cloud": {
            "n_points": int(pts.shape[0]), "n_raw_lidar_returns": int(n_raw),
            "range_filter_m": [MIN_RANGE_M, MAX_RANGE_M],
            "colored_by_projection_frac": round(colored_frac, 4),
            "color_occlusion_gate": "GT depth png, |z-d|<max(5%d, 0.5m); gray fallback",
        },
        "verification": {
            "lidar_projection_check": proj_report,
            "depth_png_scale_uint16_per_m": round(depth_scale, 4),
            "colmap_roundtrip": rt,
            "readColmapSceneInfo": si,
        },
        "gt": {
            "mesh_path": mesh_path,
            "mesh_bytes": mesh_bytes,
            "mesh_units": "centimeters (x0.01 -> meters; schema town_mesh_unit_scale=0.01)",
            "mesh_sampled_bbox_raw_min": [round(float(x), 2) for x in bb_min],
            "mesh_sampled_bbox_raw_max": [round(float(x), 2) for x in bb_max],
            "mesh_bbox_sample_vertices": int(n_sample),
            "camera_centers_inside_bbox_x0.01_frac": round(float(inside), 4),
            "expected_trimesh_rss_gb": [round(ram_lo, 1), round(ram_hi, 1)],
            "rss_over_16gb_risk": bool(ram_flag),
            "rss_note": ("g4 loader (trimesh, full ASCII OBJ) risks >16 GB RSS for this "
                         "town; needs a streaming/decimated variant -- do NOT silently "
                         "decimate" if ram_flag else "expected to fit in <16 GB RSS"),
            "alignment_note": "mesh->trainer-world transform (cm->m scale 0.01, axis "
                              "flips incl. the diag(1,-1,1) world mirror) is NOT applied "
                              "here; freeze it at first geometry eval",
        },
        "lidar_evidence": {
            "note": "per-frame sparse depth evidence: npz with rays_o/rays_d/ranges "
                    "(float64) in the RAW left-handed world; points = rays_o + "
                    "rays_d*ranges[:,None]; apply diag(1,-1,1) (negate y) to rays_o and "
                    "rays_d to land in this dataset's trainer/COLMAP world",
            "frame_indices": frames,
            "n_frames": n_lidar_frames,
            "lidars": {
                lname: {
                    "dir": os.path.join(seq_root, "lidars", lname),
                    "pattern": "{frame:08d}.npz",
                } for lname in LIDAR_NAMES
            },
        },
    }
    with open(os.path.join(out_dir, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[done] {args.town} -> {out_dir}")
    return manifest


if __name__ == "__main__":
    main()
