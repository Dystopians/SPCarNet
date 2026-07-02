"""GEMS M2 — per-triangle evidence from TRAIN-view renders (E1 importance input).

Renders ALL TRAIN views of a checkpoint at protocol resolution (scene registry
ingestion config + supersampling x4, identical to run_eval / training-time
settings) and accumulates, per triangle:

  pixels_total     supersampled pixel count owned via rend_ids (median-depth
                   owner), gated exactly like the verified g3 support pass in
                   tools/gems/geometry_metrics.py: (ids >= 0) & (ids < T) &
                   (depth_full > 0) at the supersampled resolution.
  views_seen       number of train views where the triangle owns >= 1 pixel.
  residual_sum     sum of per-pixel |render - GT| L1 (mean over RGB, native
                   resolution, nearest-upsampled to the supersampled grid)
                   over the pixels owned by the triangle.
  residual_pixels  count of supersampled pixels the residual was accumulated
                   over (same gate as pixels_total, equal by construction;
                   kept as a separate column so the two stats stay decoupled
                   if the gates ever diverge).
  max_blending_max per-triangle max over views of the renderer's
                   `max_blending` output.

Auxiliary columns stored alongside: triangle areas, and the checkpoint's own
`importance_score` / `pixel_count` (read from the raw state dict —
TriangleModel.load_parameters zeroes them on load).

D4 purity: ONLY train cameras are rendered and only train-view GT images are
read. The split is asserted (train/test name sets disjoint, every rendered
camera in the train set, non-empty test set so the split demonstrably
happened). Test cameras are never image-loaded by this module.

Importance v1 (PRE-REGISTERED, do not tune) = pixels_total.

Usage (module):
    from tools.gems.triangle_evidence import compute_triangle_evidence
    meta = compute_triangle_evidence(checkpoint=..., spec=..., out_npz=...)

Usage (CLI):
    python -m tools.gems.triangle_evidence --checkpoint <pt> \
        --scene <toy_parking|garden|courtyard> --out <npz> \
        [--max-views N] [--gpu N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# rend_ids is a float32 channel; integer ids above 2^24 are not exactly
# representable (same guard as geometry_metrics.compute_g3).
MAX_TRIANGLES_FLOAT32_IDS = 2 ** 24


def checkpoint_fingerprint(path: str) -> dict:
    """sha256 of the first 16 MiB + file size (same convention as run_eval)."""
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


class _TrainOnlyContext:
    """Duck-typed subset of tools.gems.eval_context.EvalContext that never
    image-loads test cameras (D4: the evidence pass must not read test GT).

    Exposes: .triangles, .pipe, .bg, .spec, .train_cams, .render_view(cam),
    .vertices(), .faces(), plus .train_names / .test_names (split audit).
    """

    def __init__(self, checkpoint_path: str, spec):
        import torch
        from types import SimpleNamespace
        from scene.triangle_model import TriangleModel
        from utils.camera_utils import cameraList_from_camInfos
        from tools.gems.eval_context import (
            _read_scene_info,
            _camera_loader_args,
            _resolve_checkpoint_dir,
        )

        self.spec = spec
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        scene_info = _read_scene_info(spec)

        # --- D4 split assertions (never render/read test views) ---
        self.train_names = [c.image_name for c in scene_info.train_cameras]
        self.test_names = [c.image_name for c in scene_info.test_cameras]
        train_set, test_set = set(self.train_names), set(self.test_names)
        assert len(test_set) > 0, (
            f"scene '{spec.name}': eval split produced ZERO test cameras — "
            "split is broken; refusing to treat all views as train")
        overlap = train_set & test_set
        assert not overlap, (
            f"scene '{spec.name}': train/test splits overlap: {sorted(overlap)[:5]}")

        cam_args = _camera_loader_args(spec, data_device="cpu")
        # Only TRAIN camera images are ever loaded.
        self.train_cams = cameraList_from_camInfos(
            scene_info.train_cameras, 1.0, cam_args)

        self.triangles = TriangleModel(3)
        self.triangles.scaling = 4  # protocol resolution: training-time supersampling x4
        self.triangles.load_parameters(
            _resolve_checkpoint_dir(checkpoint_path), device="cuda")

        self.pipe = SimpleNamespace(
            convert_SHs_python=False, compute_cov3D_python=False,
            depth_ratio=1.0, debug=False,
        )
        bg_color = [1.0, 1.0, 1.0] if spec.white_background else [0.0, 0.0, 0.0]
        self.bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def render_view(self, cam) -> dict:
        import torch
        from triangle_renderer import render as _triangle_render
        with torch.no_grad():
            return _triangle_render(cam, self.triangles, self.pipe, self.bg)

    def vertices(self):
        return self.triangles.vertices.detach()

    def faces(self):
        return self.triangles._triangle_indices.detach().long()


def _triangle_areas(ctx):
    import torch
    verts = ctx.vertices()
    faces = ctx.faces()
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    return 0.5 * torch.linalg.cross(b - a, c - a).norm(dim=1)


def compute_triangle_evidence(ctx_like=None, *, checkpoint=None, spec=None,
                              out_npz, max_views=None, log_every=20):
    """Accumulate per-triangle evidence over ALL TRAIN views (one render loop
    for all statistics). Returns the meta dict (also embedded in the npz).

    Either pass a ready ctx_like (duck-typed: .triangles/.pipe/.bg/.spec/
    .train_cams/.render_view/.vertices/.faces) or (checkpoint, spec) to build
    a train-only context (test camera images are never loaded).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    if ctx_like is None:
        if checkpoint is None or spec is None:
            raise ValueError("pass ctx_like or both checkpoint= and spec=")
        ctx_like = _TrainOnlyContext(checkpoint, spec)
    ctx = ctx_like

    n_triangles = int(ctx.faces().shape[0])
    assert n_triangles > 0, "checkpoint has no triangles"
    assert n_triangles < MAX_TRIANGLES_FLOAT32_IDS, (
        f"n_triangles={n_triangles} >= 2^24 exceeds rend_ids float32 precision")

    train_cams = ctx.train_cams
    # D4 guard: everything we render must be a TRAIN camera by name.
    train_name_set = set(getattr(ctx, "train_names", [c.image_name for c in train_cams]))
    test_name_set = set(getattr(ctx, "test_names", []))

    n_total = len(train_cams)
    if max_views is not None and int(max_views) < n_total:
        view_indices = np.unique(
            np.linspace(0, n_total - 1, int(max_views)).round().astype(int))
    else:
        view_indices = np.arange(n_total)

    device = "cuda"
    pixels_total = torch.zeros(n_triangles, dtype=torch.int64, device=device)
    views_seen = torch.zeros(n_triangles, dtype=torch.int32, device=device)
    residual_sum = torch.zeros(n_triangles, dtype=torch.float64, device=device)
    residual_pixels = torch.zeros(n_triangles, dtype=torch.int64, device=device)
    max_blending_max = torch.zeros(n_triangles, dtype=torch.float32, device=device)

    t0 = time.time()
    used_view_names = []
    for i, view_idx in enumerate(view_indices):
        cam = train_cams[int(view_idx)]
        name = cam.image_name
        assert name in train_name_set, (
            f"D4 violation: camera '{name}' is not in the train split")
        assert name not in test_name_set, (
            f"D4 violation: camera '{name}' is in the TEST split")
        used_view_names.append(name)

        pkg = ctx.render_view(cam)

        # --- rend_ids gating: CRIBBED from geometry_metrics._support_counts
        # (verified g3 support pass). rend_ids is supersampled and
        # UNINITIALIZED where no triangle reached the median-depth test, so it
        # must be gated by depth_full > 0 at the same resolution.
        ids = pkg["rend_ids"].detach().reshape(-1)
        depth_full = pkg.get("depth_full")
        valid = (ids >= 0) & (ids < n_triangles)
        if depth_full is not None:
            valid &= depth_full.detach().reshape(-1) > 0
        idx = ids[valid].round().long()

        counts = torch.bincount(idx, minlength=n_triangles)
        pixels_total += counts
        views_seen += (counts > 0).to(torch.int32)

        # --- residual attribution: native-res per-pixel |render - GT| L1
        # (mean over RGB), nearest-upsampled to the supersampled grid,
        # accumulated by rend_ids over the same valid gate.
        render = pkg["render"].detach().clamp(0.0, 1.0)          # [3, H0, W0]
        gt = cam.original_image[:3].to(render.device).clamp(0.0, 1.0)
        resid_native = (render - gt).abs().mean(dim=0)           # [H0, W0]
        hs, ws = pkg["depth_full"].shape[-2:] if depth_full is not None \
            else pkg["rend_ids"].shape[-2:]
        resid_up = F.interpolate(
            resid_native[None, None], size=(int(hs), int(ws)), mode="nearest"
        ).reshape(-1)
        residual_sum += torch.bincount(
            idx, weights=resid_up[valid].double(), minlength=n_triangles)
        residual_pixels += counts

        # --- renderer's per-triangle max_blending, max over views
        max_blending_max = torch.maximum(
            max_blending_max, pkg["max_blending"].detach().float())

        del pkg, ids, valid, idx, counts, resid_native, resid_up
        if (i + 1) % int(log_every) == 0 or (i + 1) == len(view_indices):
            print(f"[triangle_evidence] {i + 1}/{len(view_indices)} views "
                  f"({time.time() - t0:.1f}s)", flush=True)
        torch.cuda.empty_cache()

    areas = _triangle_areas(ctx)

    # Auxiliary columns from the RAW checkpoint state dict (load_parameters
    # zeroes importance_score/pixel_count on the model itself).
    ckpt_path = checkpoint
    if ckpt_path is None:
        ckpt_path = getattr(ctx, "checkpoint_path", None)
    ckpt_importance = np.zeros(n_triangles, dtype=np.float32)
    ckpt_pixel_count = np.zeros(n_triangles, dtype=np.int64)
    fingerprint = None
    if ckpt_path is not None:
        fingerprint = checkpoint_fingerprint(str(ckpt_path))
        state = torch.load(fingerprint["path"], map_location="cpu")
        if torch.is_tensor(state.get("importance_score")) and \
                state["importance_score"].shape[0] == n_triangles:
            ckpt_importance = state["importance_score"].detach().cpu().numpy().astype(np.float32)
        if torch.is_tensor(state.get("pixel_count")) and \
                state["pixel_count"].shape[0] == n_triangles:
            ckpt_pixel_count = state["pixel_count"].detach().cpu().numpy().astype(np.int64)
        del state

    wallclock_sec = time.time() - t0
    meta = {
        "scene": getattr(ctx.spec, "name", "?"),
        "checkpoint": fingerprint,
        "n_triangles": n_triangles,
        "n_train_views_total": n_total,
        "n_views_used": int(len(view_indices)),
        "max_views": None if max_views is None else int(max_views),
        "split": "train_only (D4-asserted)",
        "n_test_views_excluded": len(test_name_set),
        "supersampling": 4,
        "importance_v1": "pixels_total (pre-registered, untuned)",
        "residual_pixels_note": "equal to pixels_total by construction (same gate)",
        "wallclock_sec": wallclock_sec,
    }

    out_npz = os.path.abspath(out_npz)
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        pixels_total=pixels_total.cpu().numpy(),
        views_seen=views_seen.cpu().numpy(),
        residual_sum=residual_sum.cpu().numpy(),
        residual_pixels=residual_pixels.cpu().numpy(),
        max_blending_max=max_blending_max.cpu().numpy(),
        triangle_area=areas.cpu().numpy().astype(np.float32),
        ckpt_importance_score=ckpt_importance,
        ckpt_pixel_count=ckpt_pixel_count,
        view_names=np.asarray(used_view_names),
        meta_json=np.bytes_(json.dumps(meta)),
    )
    print(f"[triangle_evidence] wrote {out_npz} "
          f"({len(view_indices)} views in {wallclock_sec:.1f}s)", flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    from tools.gems.scenes import SCENES
    if args.scene not in SCENES:
        raise SystemExit(f"unknown scene '{args.scene}'; registered: {sorted(SCENES)}")
    meta = compute_triangle_evidence(
        checkpoint=args.checkpoint, spec=SCENES[args.scene],
        out_npz=args.out, max_views=args.max_views)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
