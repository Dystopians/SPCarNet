#!/usr/bin/env python
"""GEMS E3 teacher factory (LEDGER GOAL #007, M4 teacher distillation).

Builds a D4-pure, teacher-augmented COLMAP dataset from a budgeted (B50)
checkpoint using ONLY train-split evidence:

 1. SUPPORT ARTIFACTS: renders every TRAIN view (RGB + median `surf_depth`)
    into the ELA FrameLoader layout
    `<out_root>/ela_workspace/train/ours_<iter>/{renders,gt,depths,camera_index.json}`;
    `gt/` = symlinks to the REAL train images at the training resolution.
 2. PSEUDO-POSES (seed 0, train poses ONLY):
      (a) leave-k-out: ceil(kout_frac*N) evenly spaced train poses; each
          pseudo-target's support EXCLUDES its own source view, and its real
          GT is never used as supervision;
      (b) jitter: round(jitter_count_frac*N) random train poses + small
          perturbation (rotation <= 2 deg, translation <= 2% of the
          getNerfppNorm train-camera extent, look-at preserved via re-aim at
          the median rendered depth);
      (c) interpolation: round(interp_count_frac*N) SLERP/linear midpoints of
          adjacent train poses (adjacency = nearest-camera-center chaining).
 3. TEACHER RENDER: base render + depth of the B50 model at each pseudo-pose,
    then `utils.evidence_lumigraph_adapter.adapt_frame` with the production
    Phase-J config = the DEFAULTS of
    scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py
    (mode=residual, k=4, residual_clip=0.25, min_confidence=1e-4,
    depth_abs_tol=0.02, depth_rel_tol=0.03, direction_weight=0.35,
    evidence_max_side=0, NO benefit/alpha calibrators, NO edge/local-trust
    gates; alpha < 0 -> train-only leave-one-out calibration via
    `calibrate_alpha` with grid 0,0.125,0.25,0.5,0.75,1.0, stride 16,
    max 16 views, sampler stride_first, objective psnr).
 4. AUGMENTED DATASET: `<out_root>/{images,sparse/0,split.json}` — a
    trainable COLMAP-format source whose split file reproduces EXACTLY the
    original test cameras (garden: the llff8 idx%8 names; toy: split.json),
    with train = original train + all pseudo views (GT := teacher renders).

D4 purity (asserted in code, recorded in the manifest):
  - support frames and pseudo-pose sources are TRAIN views only; their name
    sets are asserted disjoint from the original test-name set;
  - every adapt_frame call is checked: used support names contain no test
    name; leave-k-out calls are checked to exclude their own source view;
  - pseudo-target FrameRecords carry a sentinel gt_path
    ("__pseudo_gt_forbidden__"), so any attempted GT read raises;
  - test images are never decoded by the factory (the COLMAP reader
    Image.open()s them lazily for the camera list, but only train pixel data
    is ever loaded); test image FILES are byte-copied into the dataset solely
    so the augmented source is loadable/evaluable later.
  - DIAGNOSTIC ONLY: for 3 leave-k-out views the real held-out GT is read to
    measure PSNR(teacher) vs PSNR(base); it never enters the dataset.

CLI (LEDGER GOAL #007 spec):
  python -m tools.gems.teacher_factory --scene <name> --checkpoint <B50 pt> \
      --out-root /data/peilincai/gems_stage1/datasets_aug/<scene>_B50_teacher \
      [--kout-frac 0.12] [--jitter-count-frac 0.5] [--interp-count-frac 0.5] \
      [--gpu N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import qvec2rotmat, rotmat2qvec, read_intrinsics_binary, read_intrinsics_text
from scene.cameras import Camera
from scene.dataset_readers import getNerfppNorm, readColmapSceneInfo
from scene.triangle_model import TriangleModel
from triangle_renderer import render as triangle_render
from tools.gems.eval_context import _camera_loader_args, _read_scene_info, _resolve_checkpoint_dir
from tools.gems.scenes import SCENES
from utils.camera_utils import cameraList_from_camInfos
from utils.evidence_lumigraph_adapter import (
    CameraRecord,
    FrameLoader,
    FrameRecord,
    adapt_frame,
    calibrate_alpha,
    load_split_frames,
    read_image_tensor,
    save_camera_index,
    save_image_tensor,
)  # noqa: E501
from utils.graphics_utils import fov2focal
from utils.image_utils import psnr as _psnr

PSEUDO_GT_SENTINEL = Path("__pseudo_gt_forbidden__")

# Production Phase-J adapter config = the DEFAULTS of
# scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py
# (no auto_policy, no benefit/alpha calibrators, no edge/local-trust gates).
ADAPTER_CONFIG = {
    "mode": "residual",
    "k": 4,
    "residual_clip": 0.25,
    "min_confidence": 1e-4,
    "depth_abs_tol": 0.02,
    "depth_rel_tol": 0.03,
    "direction_weight": 0.35,
    "evidence_max_side": 0,
    "benefit_calibrator": None,
    "alpha_calibrator": None,
    "edge_gate": False,
    "local_trust_gate": False,
    # alpha: < 0 -> train-only calibration (production default --alpha -1.0)
    "alpha_grid": [0.0, 0.125, 0.25, 0.5, 0.75, 1.0],
    "calib_stride": 16,
    "calib_max_views": 16,
    "calib_sampler": "stride_first",
    "policy_objective": "psnr",
}

JITTER_ROT_MAX_DEG = 2.0
JITTER_TRANS_MAX_FRAC = 0.02
JITTER_TRANS_MIN_FRAC = 0.005
JITTER_ROLL_MAX_DEG = 0.5


# --------------------------------------------------------------------------
# small pose helpers (numpy, deterministic)
# --------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    cos = (float(np.trace(r_a.T @ r_b)) - 1.0) * 0.5
    return math.degrees(math.acos(min(1.0, max(-1.0, cos))))


def _quat_slerp(q_a: np.ndarray, q_b: np.ndarray, t: float) -> np.ndarray:
    q_a = q_a / np.linalg.norm(q_a)
    q_b = q_b / np.linalg.norm(q_b)
    dot = float(np.dot(q_a, q_b))
    if dot < 0.0:
        q_b = -q_b
        dot = -dot
    if dot > 0.9995:
        out = q_a + t * (q_b - q_a)
        return out / np.linalg.norm(out)
    theta0 = math.acos(min(1.0, max(-1.0, dot)))
    theta = theta0 * t
    s_a = math.sin(theta0 - theta) / math.sin(theta0)
    s_b = math.sin(theta) / math.sin(theta0)
    return s_a * q_a + s_b * q_b


def _slerp_rotmat(r_a: np.ndarray, r_b: np.ndarray, t: float) -> np.ndarray:
    q = _quat_slerp(rotmat2qvec(r_a), rotmat2qvec(r_b), t)
    return qvec2rotmat(q)


def _rot_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = _normalize(axis)
    a = math.radians(angle_deg)
    kx, ky, kz = float(axis[0]), float(axis[1]), float(axis[2])
    kmat = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + math.sin(a) * kmat + (1.0 - math.cos(a)) * (kmat @ kmat)


def _cam_center(r_c2w: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    # CamInfo convention: R = R_c2w (transposed COLMAP rotation), T = w2c tvec.
    # x_cam = R^T x_world + T  =>  center = -R @ T.
    return -(r_c2w @ tvec)


def _tvec_from(r_c2w: np.ndarray, center: np.ndarray) -> np.ndarray:
    return -(r_c2w.T @ center)


def _look_at_rotation(center: np.ndarray, target: np.ndarray, up_ref: np.ndarray) -> np.ndarray:
    """c2w rotation whose +z (forward) axis looks from center at target, roll
    matched to the reference y (down in COLMAP convention) axis."""
    z = _normalize(target - center)
    x = _normalize(np.cross(up_ref, z))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


# --------------------------------------------------------------------------
# pseudo-pose specs
# --------------------------------------------------------------------------

@dataclass
class PseudoPose:
    name: str
    kind: str                      # 'leave_k_out' | 'jitter' | 'interp'
    source_names: list
    r_c2w: np.ndarray              # [3,3]
    tvec: np.ndarray               # [3]  (COLMAP w2c translation)
    camera_id: int
    fovx: float
    fovy: float
    image_width: int
    image_height: int
    params: dict = field(default_factory=dict)


def _camera_record_from_view(idx: int, view) -> CameraRecord:
    width = int(view.image_width)
    height = int(view.image_height)
    return CameraRecord(
        idx=int(idx),
        image_name=str(getattr(view, "image_name", f"{idx:05d}")),
        width=width,
        height=height,
        fx=float(fov2focal(float(view.FoVx), width)),
        fy=float(fov2focal(float(view.FoVy), height)),
        camera_center=tuple(float(x) for x in view.camera_center.detach().cpu().tolist()),
        world_view_transform=tuple(
            tuple(float(v) for v in row)
            for row in view.world_view_transform.detach().cpu().tolist()
        ),
    )


def _make_pseudo_camera(pose: PseudoPose, uid: int) -> Camera:
    dummy = torch.zeros(3, pose.image_height, pose.image_width, dtype=torch.float32)
    return Camera(
        colmap_id=int(pose.camera_id),
        R=pose.r_c2w,
        T=pose.tvec,
        FoVx=float(pose.fovx),
        FoVy=float(pose.fovy),
        depth_params=None,
        image=dummy,
        invdepthmap=None,
        gt_alpha_mask=None,
        image_name=pose.name,
        uid=int(uid),
        data_device="cpu",
    )


def _quantize_8bit(img: torch.Tensor) -> torch.Tensor:
    """PROTOCOL 4.1 convention: mul 255, add 0.5, clamp, uint8, /255."""
    return img.mul(255.0).add(0.5).clamp(0.0, 255.0).to(torch.uint8).float().div(255.0)


def _psnr_8bit(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(_psnr(_quantize_8bit(a).unsqueeze(0), _quantize_8bit(b).unsqueeze(0)).mean().item())


def _median_positive(depth: np.ndarray) -> float:
    vals = depth[np.isfinite(depth) & (depth > 1e-6)]
    return float(np.median(vals)) if vals.size else 1.0


# --------------------------------------------------------------------------
# pseudo-pose generation (train poses only, seed-deterministic)
# --------------------------------------------------------------------------

def build_pseudo_poses(train_infos, train_cams, ws_depth_dir: Path,
                       kout_frac: float, jitter_frac: float, interp_frac: float,
                       seed: int) -> list[PseudoPose]:
    n = len(train_infos)
    rng = np.random.default_rng(int(seed))
    extent = float(getNerfppNorm(train_infos)["radius"])
    centers = np.stack([_cam_center(np.asarray(c.R, dtype=np.float64),
                                    np.asarray(c.T, dtype=np.float64)) for c in train_infos])

    poses: list[PseudoPose] = []
    counter = 0

    def _next_name() -> str:
        nonlocal counter
        name = f"pseudo_{counter:05d}"
        counter += 1
        return name

    def _base_fields(info, cam) -> dict:
        return {
            "camera_id": int(info.uid),
            "fovx": float(info.FovX),
            "fovy": float(info.FovY),
            "image_width": int(cam.image_width),
            "image_height": int(cam.image_height),
        }

    # (a) leave-k-out: evenly spaced train poses; pose copied verbatim.
    k_out = max(1, math.ceil(float(kout_frac) * n))
    kout_indices = sorted({(i * n) // k_out for i in range(k_out)})
    for idx in kout_indices:
        info, cam = train_infos[idx], train_cams[idx]
        poses.append(PseudoPose(
            name=_next_name(), kind="leave_k_out",
            source_names=[str(info.image_name)],
            r_c2w=np.asarray(info.R, dtype=np.float64),
            tvec=np.asarray(info.T, dtype=np.float64),
            params={"train_index": int(idx)},
            **_base_fields(info, cam),
        ))

    # (b) jitter: random train pose + bounded perturbation, look-at preserved.
    n_jitter = int(round(float(jitter_frac) * n))
    jitter_sources = sorted(rng.choice(n, size=min(n_jitter, n), replace=n_jitter > n).tolist())
    for idx in jitter_sources:
        info, cam = train_infos[idx], train_cams[idx]
        r0 = np.asarray(info.R, dtype=np.float64)
        t0 = np.asarray(info.T, dtype=np.float64)
        c0 = _cam_center(r0, t0)
        depth_path = ws_depth_dir / f"{info.image_name}.npy"
        focus = _median_positive(np.load(depth_path).astype(np.float32))
        target = c0 + r0[:, 2] * focus
        t_dir = _normalize(rng.normal(size=3))
        t_mag = float(rng.uniform(JITTER_TRANS_MIN_FRAC, JITTER_TRANS_MAX_FRAC)) * extent
        c1 = c0 + t_dir * t_mag
        r1 = _look_at_rotation(c1, target, up_ref=r0[:, 1])
        roll = float(rng.uniform(-JITTER_ROLL_MAX_DEG, JITTER_ROLL_MAX_DEG))
        r1 = r1 @ _rot_about_axis(np.array([0.0, 0.0, 1.0]), roll)
        angle = _rotation_angle_deg(r0, r1)
        if angle > JITTER_ROT_MAX_DEG:
            # cap slightly inside the bound so float error cannot exceed it
            r1 = _slerp_rotmat(r1, r0, 1.0 - 0.999 * JITTER_ROT_MAX_DEG / angle)
            angle = _rotation_angle_deg(r0, r1)
        assert angle <= JITTER_ROT_MAX_DEG + 1e-3, f"jitter rotation {angle} > {JITTER_ROT_MAX_DEG} deg"
        assert t_mag <= JITTER_TRANS_MAX_FRAC * extent + 1e-9
        poses.append(PseudoPose(
            name=_next_name(), kind="jitter",
            source_names=[str(info.image_name)],
            r_c2w=r1, tvec=_tvec_from(r1, c1),
            params={"train_index": int(idx), "translation": t_mag,
                    "translation_frac_of_extent": t_mag / extent,
                    "rotation_deg": angle, "roll_deg": roll, "focus_depth": focus},
            **_base_fields(info, cam),
        ))

    # (c) interpolation: midpoints of adjacent train poses; adjacency =
    # nearest-camera-center chaining from the first (name-sorted) view.
    order = [0]
    visited = {0}
    while len(order) < n:
        last = order[-1]
        d = np.linalg.norm(centers - centers[last], axis=1)
        d[list(visited)] = np.inf
        nxt = int(np.argmin(d))
        order.append(nxt)
        visited.add(nxt)
    pair_candidates = [(order[i], order[i + 1]) for i in range(n - 1)]
    n_interp = int(round(float(interp_frac) * n))
    m = len(pair_candidates)
    chosen = sorted({(i * m) // n_interp for i in range(min(n_interp, m))})
    for pi in chosen:
        ia, ib = pair_candidates[pi]
        info_a, info_b = train_infos[ia], train_infos[ib]
        cam_a = train_cams[ia]
        ra = np.asarray(info_a.R, dtype=np.float64)
        rb = np.asarray(info_b.R, dtype=np.float64)
        c_mid = 0.5 * (centers[ia] + centers[ib])
        r_mid = _slerp_rotmat(ra, rb, 0.5)
        poses.append(PseudoPose(
            name=_next_name(), kind="interp",
            source_names=[str(info_a.image_name), str(info_b.image_name)],
            r_c2w=r_mid, tvec=_tvec_from(r_mid, c_mid),
            params={"train_indices": [int(ia), int(ib)],
                    "center_gap": float(np.linalg.norm(centers[ia] - centers[ib]))},
            **{**_base_fields(info_a, cam_a), "camera_id": int(info_a.uid)},
        ))
    return poses


# --------------------------------------------------------------------------
# COLMAP text model writer
# --------------------------------------------------------------------------

def _read_source_intrinsics(sparse_dir: Path) -> dict:
    bin_path = sparse_dir / "cameras.bin"
    if bin_path.is_file():
        return read_intrinsics_binary(str(bin_path))
    return read_intrinsics_text(str(sparse_dir / "cameras.txt"))


def write_colmap_text_model(out_sparse: Path, intrinsics: dict, real_infos,
                            pseudo_poses: list[PseudoPose], pseudo_ext: str = ".png") -> None:
    out_sparse.mkdir(parents=True, exist_ok=True)
    with open(out_sparse / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(intrinsics)}\n")
        for cam_id in sorted(intrinsics.keys()):
            cam = intrinsics[cam_id]
            params = " ".join(f"{float(p):.10f}" for p in cam.params)
            f.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {params}\n")

    lines = []
    image_id = 1
    for info in real_infos:
        r = np.asarray(info.R, dtype=np.float64)
        qvec = rotmat2qvec(r.T)  # back to COLMAP w2c rotation
        tvec = np.asarray(info.T, dtype=np.float64)
        name = os.path.basename(str(info.image_path))
        q = " ".join(f"{float(v):.12f}" for v in qvec)
        t = " ".join(f"{float(v):.12f}" for v in tvec)
        lines.append(f"{image_id} {q} {t} {int(info.uid)} {name}\n\n")
        image_id += 1
    for pose in pseudo_poses:
        qvec = rotmat2qvec(pose.r_c2w.T)
        q = " ".join(f"{float(v):.12f}" for v in qvec)
        t = " ".join(f"{float(v):.12f}" for v in pose.tvec)
        lines.append(f"{image_id} {q} {t} {int(pose.camera_id)} {pose.name}{pseudo_ext}\n\n")
        image_id += 1
    with open(out_sparse / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {image_id - 1}\n")
        f.writelines(lines)


# --------------------------------------------------------------------------
# main factory
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True, choices=sorted(SCENES.keys()))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--kout-frac", type=float, default=0.12)
    parser.add_argument("--jitter-count-frac", type=float, default=0.5)
    parser.add_argument("--interp-count-frac", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iteration-tag", type=int, default=40000,
                        help="method-name tag for the ELA workspace (ours_<tag>)")
    parser.add_argument("--diag-count", type=int, default=3)
    parser.add_argument("--panel-count", type=int, default=2)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    torch.cuda.set_device(int(args.gpu))
    device = torch.device("cuda", int(args.gpu))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # D6 preflight
    preflight = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "storage_preflight.py"),
         str(out_root), "--min-free-gb", str(args.min_free_gb)],
        capture_output=True, text=True)
    print(preflight.stdout)
    if preflight.returncode != 0:
        print(preflight.stderr, file=sys.stderr)
        raise SystemExit("storage preflight FAILED (D6)")

    spec = SCENES[args.scene]
    scene_info = _read_scene_info(spec)
    cam_args = _camera_loader_args(spec, data_device="cpu")

    train_infos = scene_info.train_cameras            # name-sorted by reader
    test_infos = scene_info.test_cameras
    train_names = [str(c.image_name) for c in train_infos]
    test_names = [str(c.image_name) for c in test_infos]
    test_name_set = set(test_names)
    assert not (set(train_names) & test_name_set), "train/test name overlap in source split"

    # D4 assertion 0: support frame sources are the train split only.
    d4 = {"test_names": sorted(test_names)}
    d4["support_names_intersect_test"] = sorted(set(train_names) & test_name_set)
    assert not d4["support_names_intersect_test"]

    print(f"[factory] scene={args.scene} train={len(train_infos)} test={len(test_infos)}")

    # model
    triangles = TriangleModel(3)
    triangles.scaling = 4  # training/eval-time supersampling
    triangles.load_parameters(_resolve_checkpoint_dir(args.checkpoint), device="cuda")
    n_tris = int(triangles._triangle_indices.shape[0])
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                           depth_ratio=1.0, debug=False)
    bg = torch.tensor([1.0, 1.0, 1.0] if spec.white_background else [0.0, 0.0, 0.0],
                      dtype=torch.float32, device="cuda")

    train_cams = cameraList_from_camInfos(train_infos, 1.0, cam_args)
    assert [c.image_name for c in train_cams] == train_names

    method = f"ours_{int(args.iteration_tag)}"
    ws = out_root / "ela_workspace"
    ws_train = ws / "train" / method
    ws_pseudo = ws / "pseudo" / method

    # ---------------- stage 1: support artifacts ----------------
    t0 = time.perf_counter()
    render_dir = ws_train / "renders"
    gt_dir = ws_train / "gt"
    depth_dir = ws_train / "depths"
    for d in (render_dir, gt_dir, depth_dir):
        d.mkdir(parents=True, exist_ok=True)
    camera_records = []
    with torch.no_grad():
        for idx, (info, view) in enumerate(zip(train_infos, train_cams)):
            key = str(view.image_name)
            out_png = render_dir / f"{key}.png"
            out_npy = depth_dir / f"{key}.npy"
            if not (out_png.is_file() and out_npy.is_file()):
                pkg = triangle_render(view, triangles, pipe, bg)
                torchvision.utils.save_image(pkg["render"], out_png)
                np.save(out_npy, pkg["surf_depth"][0].detach().float().cpu().numpy().astype(np.float32))
                del pkg
            # gt = REAL train image, symlinked at the training resolution
            src_img = Path(str(info.image_path)).resolve()
            gt_link = gt_dir / f"{key}{src_img.suffix}"
            if not (gt_link.is_symlink() or gt_link.is_file()):
                os.symlink(src_img, gt_link)
            camera_records.append(_camera_record_from_view(idx, view))
            if idx % 25 == 0:
                torch.cuda.empty_cache()
                print(f"[support] {idx + 1}/{len(train_cams)}", flush=True)
    save_camera_index(camera_records, ws_train / "camera_index.json")
    support_sec = time.perf_counter() - t0
    print(f"[support] done: {len(train_cams)} views in {support_sec:.1f}s")

    train_frames = load_split_frames(ws, "train", method)
    assert len(train_frames) == len(train_cams)
    assert [f.name for f in train_frames] == train_names, \
        "ELA train frame order does not match name-sorted train cameras"
    # D4: no support frame carries a test name; every support gt is a train image
    assert not ({f.name for f in train_frames} & test_name_set)
    for f in train_frames:
        assert Path(f.gt_path).resolve().name.split(".")[0] in set(train_names)
    d4["support_frame_count"] = len(train_frames)
    d4["support_frames_are_train_only"] = True

    # ---------------- stage 2: alpha calibration (train-only, production default) ----------------
    t0 = time.perf_counter()
    calibration = calibrate_alpha(
        train_frames,
        alpha_grid=list(ADAPTER_CONFIG["alpha_grid"]),
        k=int(ADAPTER_CONFIG["k"]),
        mode=str(ADAPTER_CONFIG["mode"]),
        calib_stride=int(ADAPTER_CONFIG["calib_stride"]),
        calib_max_views=int(ADAPTER_CONFIG["calib_max_views"]),
        residual_clip=float(ADAPTER_CONFIG["residual_clip"]),
        depth_abs_tol=float(ADAPTER_CONFIG["depth_abs_tol"]),
        depth_rel_tol=float(ADAPTER_CONFIG["depth_rel_tol"]),
        direction_weight=float(ADAPTER_CONFIG["direction_weight"]),
        policy_objective=str(ADAPTER_CONFIG["policy_objective"]),
        calib_sampler=str(ADAPTER_CONFIG["calib_sampler"]),
        device=device,
    )
    alpha = float(calibration["alpha"])
    calib_sec = time.perf_counter() - t0
    print(f"[calibration] alpha={alpha} ({calib_sec:.1f}s); "
          f"rows={[(r['alpha'], round(r['psnr_gain'], 4)) for r in calibration['rows']]}")

    # ---------------- stage 3: pseudo poses ----------------
    pseudo_poses = build_pseudo_poses(
        train_infos, train_cams, depth_dir,
        kout_frac=args.kout_frac, jitter_frac=args.jitter_count_frac,
        interp_frac=args.interp_count_frac, seed=args.seed)
    train_name_set = set(train_names)
    for pose in pseudo_poses:
        assert set(pose.source_names) <= train_name_set, \
            f"pseudo pose {pose.name} derives from non-train views {pose.source_names}"
        assert not (set(pose.source_names) & test_name_set)
    d4["pseudo_count"] = len(pseudo_poses)
    d4["pseudo_sources_train_only"] = True
    d4["pseudo_sources_intersect_test"] = []
    kinds = {}
    for pose in pseudo_poses:
        kinds[pose.kind] = kinds.get(pose.kind, 0) + 1
    print(f"[pseudo] {len(pseudo_poses)} poses: {kinds}")

    # ---------------- stage 4: base renders + teacher bake ----------------
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    p_render_dir = ws_pseudo / "renders"
    p_depth_dir = ws_pseudo / "depths"
    p_render_dir.mkdir(parents=True, exist_ok=True)
    p_depth_dir.mkdir(parents=True, exist_ok=True)

    stats_path = out_root / "teacher_stats.jsonl"
    done_stats = {}
    if stats_path.is_file():
        for line in stats_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done_stats[row["name"]] = row

    loader = FrameLoader(device=device)
    frames_by_name = {f.name: f for f in train_frames}
    pseudo_camera_records = []
    t0 = time.perf_counter()
    adapter_seconds = []
    with torch.no_grad():
        for pidx, pose in enumerate(pseudo_poses):
            teacher_png = images_dir / f"{pose.name}.png"
            if pose.kind == "leave_k_out":
                src = frames_by_name[pose.source_names[0]]
                base_render_path = src.render_path
                base_depth_path = src.depth_path
                cam_record = src.camera  # keeps image_name = source -> auto self-exclusion
                support = [f for f in train_frames if f.name != pose.source_names[0]]
            else:
                cam = _make_pseudo_camera(pose, uid=pidx)
                base_render_path = p_render_dir / f"{pose.name}.png"
                base_depth_path = p_depth_dir / f"{pose.name}.npy"
                if not (base_render_path.is_file() and base_depth_path.is_file()):
                    pkg = triangle_render(cam, triangles, pipe, bg)
                    torchvision.utils.save_image(pkg["render"], base_render_path)
                    np.save(base_depth_path,
                            pkg["surf_depth"][0].detach().float().cpu().numpy().astype(np.float32))
                    del pkg
                cam_record = _camera_record_from_view(pidx, cam)
                support = train_frames
                del cam
            pseudo_camera_records.append(cam_record)

            if pose.name in done_stats and teacher_png.is_file():
                prev = done_stats[pose.name]
                if prev.get("kind") != pose.kind or list(prev.get("sources", [])) != list(pose.source_names):
                    raise RuntimeError(
                        f"resume mismatch for {pose.name}: stats say "
                        f"{prev.get('kind')}/{prev.get('sources')} but current pose is "
                        f"{pose.kind}/{pose.source_names}. Clean {stats_path} and "
                        f"{images_dir}/pseudo_*.png before rerunning with different "
                        f"fracs/seed.")
                continue

            # D4: the support set never contains a test view
            assert not ({f.name for f in support} & test_name_set)
            target = FrameRecord(
                idx=pidx, name=pose.name,
                render_path=Path(base_render_path),
                gt_path=PSEUDO_GT_SENTINEL,      # target GT forbidden (D4)
                depth_path=Path(base_depth_path),
                camera=cam_record,
            )
            t_a = time.perf_counter()
            adapted, info = adapt_frame(
                target, support,
                k=int(ADAPTER_CONFIG["k"]),
                alpha=alpha,
                mode=str(ADAPTER_CONFIG["mode"]),
                residual_clip=float(ADAPTER_CONFIG["residual_clip"]),
                min_confidence=float(ADAPTER_CONFIG["min_confidence"]),
                depth_abs_tol=float(ADAPTER_CONFIG["depth_abs_tol"]),
                depth_rel_tol=float(ADAPTER_CONFIG["depth_rel_tol"]),
                direction_weight=float(ADAPTER_CONFIG["direction_weight"]),
                evidence_max_side=int(ADAPTER_CONFIG["evidence_max_side"]),
                loader=loader, device=device,
            )
            adapter_sec = time.perf_counter() - t_a
            adapter_seconds.append(adapter_sec)
            used = set(info.get("support_names", []))
            assert not (used & test_name_set), f"teacher for {pose.name} used test evidence"
            if pose.kind == "leave_k_out":
                assert pose.source_names[0] not in used, \
                    f"leave-k-out {pose.name} used its own held-out view {pose.source_names[0]}"
            base_img = loader.render(str(base_render_path))
            mean_abs_signal = float((adapted - base_img).abs().mean().item())
            save_image_tensor(adapted, teacher_png)
            row = {
                "name": pose.name, "kind": pose.kind, "sources": pose.source_names,
                "support_used": int(info.get("support_count", 0)),
                "support_names_used": sorted(used),
                "mean_signal_support_count": float(info.get("mean_signal_support_count", 0.0)),
                "mean_confidence": float(info.get("mean_confidence", 0.0)),
                "covered_fraction": float(info.get("covered_fraction", 0.0)),
                "mean_abs_signal": mean_abs_signal,
                "adapter_seconds": adapter_sec,
                "pose_params": pose.params,
            }
            done_stats[pose.name] = row
            with open(stats_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            del adapted
            if pidx % 20 == 0:
                torch.cuda.empty_cache()
                print(f"[teacher] {pidx + 1}/{len(pseudo_poses)} "
                      f"(adapter {adapter_sec:.2f}s)", flush=True)
    save_camera_index(pseudo_camera_records, ws_pseudo / "camera_index.json")
    teacher_sec = time.perf_counter() - t0
    print(f"[teacher] done: {len(pseudo_poses)} poses in {teacher_sec:.1f}s")

    # ---------------- stage 5: assemble the augmented COLMAP dataset ----------------
    t0 = time.perf_counter()
    # real images: byte-copies (train used as GT; test copied ONLY so the
    # augmented source is loadable/evaluable — never read by this factory).
    for info in list(train_infos) + list(test_infos):
        src = Path(str(info.image_path))
        dst = images_dir / src.name
        if not dst.is_file():
            shutil.copy2(src, dst)
    # sanity: teacher renders already in images_dir as pseudo_*.png
    for pose in pseudo_poses:
        assert (images_dir / f"{pose.name}.png").is_file()

    intrinsics = _read_source_intrinsics(Path(spec.source_path) / "sparse" / "0")
    write_colmap_text_model(out_root / "sparse" / "0", intrinsics,
                            list(train_infos) + list(test_infos), pseudo_poses)

    # original init ply (+ toy's trackless points3D.txt for parity)
    src_sparse = Path(spec.source_path) / "sparse" / "0"
    for fname in ("points3D.ply", "points3D.txt"):
        src_f = src_sparse / fname
        dst_f = out_root / "sparse" / "0" / fname
        if src_f.is_file() and not dst_f.is_file():
            if fname == "points3D.txt" and args.scene == "garden":
                continue  # garden has only .bin whose tracks reference old image ids
            shutil.copy2(src_f, dst_f)

    pseudo_names = [pose.name for pose in pseudo_poses]
    split_payload = {"train": train_names + pseudo_names, "test": test_names}
    (out_root / "split.json").write_text(json.dumps(split_payload, indent=1) + "\n")
    assemble_sec = time.perf_counter() - t0

    # ---------------- stage 6: verification (readColmapSceneInfo + split) ----------------
    t0 = time.perf_counter()
    aug_info = readColmapSceneInfo(str(out_root), "images", True,
                                   split_strategy="file",
                                   split_file=str(out_root / "split.json"))
    aug_test = [str(c.image_name) for c in aug_info.test_cameras]
    aug_train = [str(c.image_name) for c in aug_info.train_cameras]
    assert set(aug_test) == test_name_set and len(aug_test) == len(test_names), \
        f"augmented test set mismatch: {sorted(set(aug_test) ^ test_name_set)}"
    assert set(aug_train) == (train_name_set | set(pseudo_names)), \
        "augmented train set != original train + pseudo"
    assert len(aug_train) == len(train_names) + len(pseudo_names)
    verify_sec = time.perf_counter() - t0
    d4["aug_test_names_match_original"] = True
    d4["aug_train_names_match_train_plus_pseudo"] = True
    d4["aug_train_count"] = len(aug_train)
    d4["aug_test_count"] = len(aug_test)
    print(f"[verify] augmented dataset OK: train={len(aug_train)} "
          f"(real {len(train_names)} + pseudo {len(pseudo_names)}), test={len(aug_test)}")

    # ---------------- stage 7: leave-k-out DIAGNOSTIC + panels ----------------
    kout_poses = [p for p in pseudo_poses if p.kind == "leave_k_out"]
    n_diag = min(int(args.diag_count), len(kout_poses))
    diag_idx = sorted({(i * len(kout_poses)) // n_diag for i in range(n_diag)})
    diagnostics = []
    panels_dir = out_root / "panels_diagnostic"
    panels_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for j, ki in enumerate(diag_idx):
            pose = kout_poses[ki]
            src = frames_by_name[pose.source_names[0]]
            base = read_image_tensor(Path(src.render_path), device=device)
            teacher = read_image_tensor(images_dir / f"{pose.name}.png", device=device)
            real_gt = read_image_tensor(Path(src.gt_path), device=device)  # DIAGNOSTIC ONLY
            row = {
                "pseudo_name": pose.name,
                "held_out_view": pose.source_names[0],
                "psnr_base_vs_gt": _psnr_8bit(base, real_gt),
                "psnr_teacher_vs_gt": _psnr_8bit(teacher, real_gt),
                "label": ("DIAGNOSTIC — held-out real GT used ONLY for this measurement; "
                          "it never enters the dataset"),
            }
            row["teacher_gain_db"] = row["psnr_teacher_vs_gt"] - row["psnr_base_vs_gt"]
            diagnostics.append(row)
            if j < int(args.panel_count):
                panel = torch.cat([base, teacher, real_gt], dim=2)
                torchvision.utils.save_image(
                    panel, panels_dir / f"{pose.name}_base_teacher_gt.png")
            print(f"[diag] {pose.name} ({pose.source_names[0]}): "
                  f"base {row['psnr_base_vs_gt']:.3f} dB -> teacher "
                  f"{row['psnr_teacher_vs_gt']:.3f} dB (Δ {row['teacher_gain_db']:+.3f})")

    # ---------------- manifest ----------------
    stats_rows = [done_stats[p.name] for p in pseudo_poses if p.name in done_stats]
    adapter_mean = (sum(adapter_seconds) / len(adapter_seconds)) if adapter_seconds else None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    manifest = {
        "tool": "tools/gems/teacher_factory.py",
        "goal": "LEDGER GOAL #007 (E3, M4 teacher distillation)",
        "command": sys.argv,
        "git_commit": commit,
        "scene": args.scene,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_triangles": n_tris,
        "seed": int(args.seed),
        "out_root": str(out_root),
        "counts": {
            "train_real": len(train_names),
            "test_real": len(test_names),
            "pseudo_total": len(pseudo_poses),
            "pseudo_by_kind": kinds,
            "aug_train_total": len(aug_train),
        },
        "adapter_config": {k: v for k, v in ADAPTER_CONFIG.items()},
        "alpha": alpha,
        "alpha_source": "train_only_leave_one_out_calibration (production default --alpha -1)",
        "alpha_calibration": calibration,
        "jitter_bounds": {"rotation_max_deg": JITTER_ROT_MAX_DEG,
                          "translation_max_frac_of_extent": JITTER_TRANS_MAX_FRAC,
                          "roll_max_deg": JITTER_ROLL_MAX_DEG},
        "d4": {
            **d4,
            "target_gt_policy": "pseudo-target FrameRecords carry sentinel gt_path "
                                f"'{PSEUDO_GT_SENTINEL}'; adapt_frame never reads target GT",
            "test_images_in_dataset": "byte-copied for later eval loading only; "
                                      "never decoded/read by the factory",
        },
        "teacher_stats_summary": {
            "mean_signal_support_count": float(np.mean(
                [r["mean_signal_support_count"] for r in stats_rows])) if stats_rows else None,
            "mean_covered_fraction": float(np.mean(
                [r["covered_fraction"] for r in stats_rows])) if stats_rows else None,
            "mean_abs_signal": float(np.mean(
                [r["mean_abs_signal"] for r in stats_rows])) if stats_rows else None,
            "mean_adapter_seconds": adapter_mean,
        },
        "diagnostic_leave_k_out": diagnostics,
        "wallclock_sec": {
            "support_render": support_sec,
            "alpha_calibration": calib_sec,
            "pseudo_render_and_teacher_bake": teacher_sec,
            "assemble": assemble_sec,
            "verify_load": verify_sec,
            "total": time.perf_counter() - t_start,
        },
        "per_pose_stats_file": str(stats_path),
        "storage_preflight": json.loads(preflight.stdout) if preflight.stdout.strip() else None,
    }
    (out_root / "teacher_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[factory] manifest written: {out_root / 'teacher_manifest.json'}")
    print(f"[factory] total wall-clock {manifest['wallclock_sec']['total'] / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
