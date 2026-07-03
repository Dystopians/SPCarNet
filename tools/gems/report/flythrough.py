"""GEMS Stage 2 — E11 T2 flythrough videos (MATRIX cell E11-QUAL, T2 item).

Two ~10 s / 30 fps mp4 videos over the banked single-mouth eval rows, TWO
models side by side per frame (left = clean 30k baseline, right = B5@B50):
  (1) S-REND  garden        {garden_clean30k_v2, garden_B50_importance_ft_e1v2}
  (2) S-GEO   ss3dm_town01  {ss3dm_town01_clean30k_geo_v1, ss3dm_town01_B50_geo_v1}

Camera path (TRAIN-trajectory interpolation; no test poses, no GT pixels):
  - keyframes = every KEYFRAME_STRIDE-th TRAIN camera in image-name order
    (ss3dm scenes are first restricted to the single center-front stream
    `^front_\\d+` so the path follows one smooth drive instead of hopping
    between the 3 rig cameras);
  - camera centers   -> Catmull-Rom spline through the keyframe centers
    (endpoint keyframes duplicated as phantom control points);
  - camera rotations -> per-segment quaternion SLERP between keyframe c2w
    rotations (the teacher_factory pseudo-pose math, PROTOCOL-visible);
  - frames allocated uniformly in chord arc length along the whole path;
  - intrinsics (FoV, WxH) = the first keyframe's (asserted ~constant).

Rendering: tools.gems.eval_context training-time settings (supersampling x4)
at the frozen protocol resolution, checkpoints taken from each eval row's
banked metrics.json (same resolution rule as qual_grids.py). D4 purity: only
TRAIN poses are interpolated and no image pixels (train or test) are read for
supervision — pseudo cameras carry dummy zero images.

Encoding: composite frames are streamed from RAM straight into ffmpeg's
stdin (rawvideo pipe; NO intermediate frame files — the home/tmp filesystem
is quota-capped) with h264_nvenc, fallback mpeg4, last-resort imageio.
Output:
  RESULTS/figures/videos/<scene>_flythrough.mp4  +  manifest.json

Usage:
    python -m tools.gems.report.flythrough --gpu 4 [--scenes garden ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

EVAL_ROOT = "/data/peilincai/gems_stage1/eval"
OUT_DEFAULT = os.path.join(REPO_ROOT, "RESULTS", "figures", "videos")
FFMPEG = shutil.which("ffmpeg") or os.path.expanduser("~/.local/bin/ffmpeg")

# ---------------------------------------------------------------------------
# Frozen video registry (E11 T2): banked eval rows only, clean row FIRST
# (left pane). ss3dm train cams are restricted to the center-front stream.
# ---------------------------------------------------------------------------
VIDEOS = {
    "garden": {
        "suite": "S-REND",
        "models": [("clean 30k", "garden_clean30k_v2"),
                   ("B5 @ B50 (importance+FT)", "garden_B50_importance_ft_e1v2")],
        "train_name_filter": None,
    },
    "ss3dm_town01": {
        "suite": "S-GEO",
        "models": [("clean 30k", "ss3dm_town01_clean30k_geo_v1"),
                   ("B5 @ B50 (importance+FT)", "ss3dm_town01_B50_geo_v1")],
        "train_name_filter": r"^front_\d+$",
    },
}

KEYFRAME_STRIDE = 3
N_FRAMES = 300
FPS = 30


def _load_row(eval_row: str) -> dict:
    with open(os.path.join(EVAL_ROOT, eval_row, "metrics.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# pose interpolation (cribbed from tools/gems_train/teacher_factory.py)
# ---------------------------------------------------------------------------

def _quat_slerp(q_a, q_b, t: float):
    import numpy as np
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


def _catmull_rom(p0, p1, p2, p3, t: float):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2.0 * p1) + (-p0 + p2) * t
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)


def build_path(train_cams, name_filter: str | None, stride: int, n_frames: int):
    """Interpolated (r_c2w [3,3], center [3]) pose list through the train
    trajectory + the intrinsics dict of the first keyframe."""
    import numpy as np
    from scene.colmap_loader import qvec2rotmat, rotmat2qvec

    cams = sorted(train_cams, key=lambda c: c.image_name)
    if name_filter is not None:
        pat = re.compile(name_filter)
        cams = [c for c in cams if pat.match(c.image_name)]
    assert len(cams) >= 4, f"too few path cameras ({len(cams)})"
    keys = cams[::stride]

    centers = np.stack([-(np.asarray(c.R) @ np.asarray(c.T)) for c in keys])
    quats = [rotmat2qvec(np.asarray(c.R)) for c in keys]

    fovx = [float(c.FoVx) for c in keys]
    fovy = [float(c.FoVy) for c in keys]
    assert max(fovx) - min(fovx) < 1e-3 and max(fovy) - min(fovy) < 1e-3, \
        "keyframe intrinsics vary; constant-intrinsics path assumption broken"
    intr = {"fovx": fovx[0], "fovy": fovy[0],
            "width": int(keys[0].image_width), "height": int(keys[0].image_height)}

    # chord-length parameterization over the K-1 segments
    seg_len = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    seg_len = np.maximum(seg_len, 1e-9)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    s_samples = np.linspace(0.0, cum[-1], n_frames, endpoint=True)
    s_samples[-1] = np.nextafter(cum[-1], 0.0)  # keep last sample in-range

    poses = []
    for s in s_samples:
        i = int(np.searchsorted(cum, s, side="right") - 1)
        i = min(max(i, 0), len(seg_len) - 1)
        t = float((s - cum[i]) / seg_len[i])
        p0 = centers[max(i - 1, 0)]
        p1, p2 = centers[i], centers[i + 1]
        p3 = centers[min(i + 2, len(centers) - 1)]
        center = _catmull_rom(p0, p1, p2, p3, t)
        r_c2w = qvec2rotmat(_quat_slerp(quats[i], quats[i + 1], t))
        poses.append((r_c2w, center))

    meta = {"n_path_cams": len(cams), "n_keyframes": len(keys),
            "keyframe_stride": stride,
            "first_keyframe": keys[0].image_name,
            "last_keyframe": keys[-1].image_name,
            "path_length_scene_units": float(cum[-1]),
            "intrinsics": intr}
    return poses, intr, meta


def _make_path_camera(r_c2w, center, intr, uid: int):
    import numpy as np
    import torch
    from scene.cameras import Camera

    tvec = -(np.asarray(r_c2w).T @ np.asarray(center))
    dummy = torch.zeros(3, intr["height"], intr["width"], dtype=torch.float32)
    return Camera(colmap_id=uid, R=np.asarray(r_c2w), T=tvec,
                  FoVx=intr["fovx"], FoVy=intr["fovy"], depth_params=None,
                  image=dummy, invdepthmap=None, gt_alpha_mask=None,
                  image_name=f"fly_{uid:04d}", uid=uid, data_device="cpu")


# ---------------------------------------------------------------------------
# rendering + composition
# ---------------------------------------------------------------------------

def _render_frames(ctx, poses, intr):
    """Render every path pose with the CURRENT ctx.triangles -> uint8 stack."""
    import numpy as np
    import torch

    frames = np.empty((len(poses), intr["height"], intr["width"], 3),
                      dtype=np.uint8)
    for i, (r_c2w, center) in enumerate(poses):
        cam = _make_path_camera(r_c2w, center, intr, i)
        pkg = ctx.render_view(cam)
        rgb = pkg["render"].detach().float().clamp(0, 1).cpu().numpy()
        frames[i] = (rgb.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
        del pkg, cam
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            print(f"[flythrough]   rendered {i + 1}/{len(poses)} frames",
                  flush=True)
    return frames


def _label_font(px: int):
    from PIL import ImageFont
    try:
        import matplotlib
        ttf = os.path.join(os.path.dirname(matplotlib.__file__),
                           "mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf")
        return ImageFont.truetype(ttf, px)
    except Exception:
        return ImageFont.load_default()


def _compose_frame(left, right, labels, scene_tag, font):
    import numpy as np
    from PIL import Image, ImageDraw

    comp = np.concatenate([left, right], axis=1)
    img = Image.fromarray(comp)
    draw = ImageDraw.Draw(img, "RGBA")
    w = left.shape[1]
    for k, (x0, text) in enumerate(((0, labels[0]), (w, labels[1]))):
        bbox = draw.textbbox((x0 + 10, 8), text, font=font)
        draw.rectangle([bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4],
                       fill=(0, 0, 0, 160))
        draw.text((x0 + 10, 8), text, fill=(255, 255, 255, 255), font=font)
    bbox = draw.textbbox((10, img.height - 8), scene_tag, font=font, anchor="lb")
    draw.rectangle([bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4],
                   fill=(0, 0, 0, 160))
    draw.text((10, img.height - 8), scene_tag, fill=(255, 255, 255, 255),
              font=font, anchor="lb")
    draw.line([(w, 0), (w, img.height)], fill=(255, 255, 255, 220), width=2)
    return img


def _encode_stream(frames, out_mp4: str, fps: int) -> str:
    """Stream a uint8 [N,H,W,3] stack into ffmpeg stdin -> mp4 (no frame
    files on disk; the home/tmp filesystem is quota-capped). Returns the
    codec actually used."""
    n, h, w = frames.shape[:3]
    for codec, extra in (("h264_nvenc", ["-pix_fmt", "yuv420p", "-b:v", "8M"]),
                         ("mpeg4", ["-q:v", "3"])):
        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{w}x{h}", "-framerate", str(fps),
               "-i", "pipe:0", "-c:v", codec, *extra, out_mp4]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        try:
            for i in range(n):
                proc.stdin.write(frames[i].tobytes())
            proc.stdin.close()
            rc = proc.wait()
        except BrokenPipeError:
            rc = proc.wait()
        if rc == 0 and os.path.isfile(out_mp4) and os.path.getsize(out_mp4) > 0:
            return codec
        err = proc.stderr.read().decode(errors="replace").strip()
        print(f"[flythrough] ffmpeg {codec} failed (rc={rc}): {err[:400]}",
              flush=True)
    # last resort: imageio
    import imageio.v2 as imageio
    imageio.mimwrite(out_mp4, list(frames), fps=fps)
    return "imageio"


def build_video(scene: str, cfg: dict, out_dir: str,
                n_frames: int, fps: int, manifest: dict):
    import numpy as np
    import torch
    from scene.triangle_model import TriangleModel
    from tools.gems.eval_context import build_eval_context, _resolve_checkpoint_dir
    from tools.gems.scenes import SCENES

    spec = SCENES[scene]
    rows = cfg["models"]
    ckpts = {}
    for label, eval_row in rows:
        m = _load_row(eval_row)
        ckpt = m["checkpoint"]["path"]
        assert os.path.isfile(ckpt), f"banked checkpoint missing: {ckpt}"
        ckpts[label] = {"eval_row": eval_row, "checkpoint": ckpt,
                        "psnr_mean_banked": m["rendering"]["mean"]["psnr"]}
        print(f"[flythrough] {scene} [{label}] ckpt = {ckpt}", flush=True)

    # context on the FIRST (clean) checkpoint; path from TRAIN cams only
    print(f"[flythrough] {scene}: building eval context", flush=True)
    ctx = build_eval_context(ckpts[rows[0][0]]["checkpoint"], spec)
    poses, intr, path_meta = build_path(
        ctx.train_cams, cfg["train_name_filter"], KEYFRAME_STRIDE, n_frames)
    print(f"[flythrough] {scene}: path = {path_meta}", flush=True)

    stacks = {}
    for label, _ in rows:
        if label != rows[0][0]:
            print(f"[flythrough] {scene}: swapping model -> {label}", flush=True)
            del ctx.triangles
            torch.cuda.empty_cache()
            tri = TriangleModel(3)
            tri.scaling = 4  # training-time supersampling (eval_context)
            tri.load_parameters(
                _resolve_checkpoint_dir(ckpts[label]["checkpoint"]), device="cuda")
            ctx.triangles = tri
        print(f"[flythrough] {scene}: rendering {n_frames} frames [{label}]",
              flush=True)
        stacks[label] = _render_frames(ctx, poses, intr)
    del ctx
    torch.cuda.empty_cache()

    # side-by-side composition with label overlays (in RAM; no frame files)
    font = _label_font(max(14, intr["height"] // 30))
    (label_l, _), (label_r, _) = rows
    scene_tag = f"{scene} ({cfg['suite']}) — E11 T2 flythrough, train-trajectory interpolation"
    comp = np.empty((n_frames, intr["height"], 2 * intr["width"], 3),
                    dtype=np.uint8)
    for i in range(n_frames):
        img = _compose_frame(stacks[label_l][i], stacks[label_r][i],
                             (label_l, label_r), scene_tag, font)
        comp[i] = np.asarray(img)
    del stacks
    print(f"[flythrough] {scene}: composed {n_frames} frames in RAM",
          flush=True)

    out_mp4 = os.path.join(out_dir, f"{scene}_flythrough.mp4")
    codec = _encode_stream(comp, out_mp4, fps)
    del comp
    size = os.path.getsize(out_mp4)
    print(f"[flythrough] wrote {out_mp4} ({size / 1e6:.1f} MB, codec={codec})",
          flush=True)

    manifest["scenes"][scene] = {
        "suite": cfg["suite"],
        "output_mp4": out_mp4,
        "file_size_bytes": size,
        "codec": codec,
        "fps": fps,
        "n_frames": n_frames,
        "duration_s": n_frames / fps,
        "pane_resolution_wh": [intr["width"], intr["height"]],
        "composite_resolution_wh": [2 * intr["width"], intr["height"]],
        "layout": f"side-by-side: left={label_l}, right={label_r}",
        "path": path_meta,
        "train_name_filter": cfg["train_name_filter"],
        "models": [{"label": lbl, **ckpts[lbl]} for lbl, _ in rows],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--scenes", nargs="*", default=list(VIDEOS.keys()))
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    prior = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            prior = json.load(f).get("scenes", {})
    manifest = {
        "generator": "tools/gems/report/flythrough.py",
        "path_rule": (
            f"keyframes = every {KEYFRAME_STRIDE}rd TRAIN camera in image-name "
            "order (ss3dm: center-front stream only); centers = Catmull-Rom "
            "spline; rotations = per-segment quaternion SLERP (teacher_factory "
            "math); frames uniform in chord arc length; intrinsics = first "
            "keyframe; NO test poses, NO GT pixels (D4-pure)"),
        "render_settings": "tools.gems.eval_context training-time settings "
                           "(supersampling x4, protocol resolution)",
        "eval_root": EVAL_ROOT,
        "scenes": prior,
    }
    for scene in args.scenes:
        build_video(scene, VIDEOS[scene], args.out,
                    args.frames, args.fps, manifest)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1)
    print(f"[flythrough] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
