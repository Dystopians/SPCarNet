#!/usr/bin/env python3
import json
import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from triangle_renderer import TriangleModel, render
from utils.colmap_sparse_utils import extract_colmap_sparse_points
from utils.geometry_metrics_utils import depth_metrics, normal_metrics_from_abs_cos


def _safe_unit(v: np.ndarray, eps: float = 1e-10):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.clip(n, eps, None)
    return v / n


def _try_build_kdtree(points_xyz: np.ndarray):
    try:
        from scipy.spatial import cKDTree  # type: ignore

        return cKDTree(points_xyz)
    except Exception:
        return None


def evaluate_geometry(
    dataset,
    iteration: int,
    pipe,
    max_points_per_view: int,
    point_error_max: float,
    normal_knn: int,
    compute_normal: bool,
    seed: int,
):
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        triangles = TriangleModel(dataset.sh_degree)
        triangles.scaling = 4
        scene = Scene(
            args=dataset,
            triangles=triangles,
            init_opacity=None,
            set_sigma=None,
            load_iteration=iteration,
            shuffle=False,
        )
        if len(scene.getTestCameras()) == 0:
            raise RuntimeError("No test cameras found. Use --eval and provide a split with test views.")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        info_by_name = {c.image_name: c for c in scene.scene_info.test_cameras}
        colmap_points = scene.scene_info.colmap_points3d or {}
        if len(colmap_points) == 0:
            raise RuntimeError("COLMAP sparse points are unavailable; cannot evaluate geometry.")
        sparse = extract_colmap_sparse_points(colmap_points3d=colmap_points, error_max=-1.0)
        point_ids = sparse.point_ids
        point_xyz = sparse.xyz
        pid_to_xyz = sparse.pid_to_xyz
        pid_to_error = sparse.pid_to_error

        kdtree = _try_build_kdtree(point_xyz) if compute_normal else None
        normal_cache = {}

        def estimate_point_normal(pid: int):
            if pid in normal_cache:
                return normal_cache[pid]
            if kdtree is None:
                return None
            idx = np.searchsorted(point_ids, pid)
            if idx >= point_ids.shape[0] or int(point_ids[idx]) != int(pid):
                return None
            k = min(int(normal_knn) + 1, point_xyz.shape[0])
            _, nn = kdtree.query(point_xyz[idx], k=k)
            if np.isscalar(nn):
                return None
            nn = np.array(nn, dtype=np.int64)
            if nn.shape[0] < 4:
                return None
            neigh = point_xyz[nn[1:]]
            centered = neigh - neigh.mean(axis=0, keepdims=True)
            cov = centered.T @ centered / max(1, centered.shape[0] - 1)
            w, v = np.linalg.eigh(cov)
            n = v[:, np.argmin(w)]
            n = n / max(np.linalg.norm(n), 1e-10)
            normal_cache[pid] = n
            return n

        all_depth_pred = []
        all_depth_gt = []
        all_cos_abs = []
        per_view = []

        for view in tqdm(scene.getTestCameras(), desc="COLMAP geometry eval"):
            cam_info = info_by_name.get(view.image_name, None)
            if cam_info is None:
                continue
            if cam_info.colmap_xys is None or cam_info.colmap_point3D_ids is None:
                continue

            xys = np.array(cam_info.colmap_xys, dtype=np.float64)
            pids = np.array(cam_info.colmap_point3D_ids, dtype=np.int64)
            if xys.shape[0] == 0:
                continue

            valid = pids > 0
            valid &= np.array([int(pid) in pid_to_xyz for pid in pids], dtype=bool)
            if point_error_max > 0:
                valid &= np.array([pid_to_error.get(int(pid), 1e9) <= point_error_max for pid in pids], dtype=bool)
            if not np.any(valid):
                continue

            xys = xys[valid]
            pids = pids[valid]
            if max_points_per_view > 0 and xys.shape[0] > max_points_per_view:
                pick = rng.choice(xys.shape[0], size=max_points_per_view, replace=False)
                xys = xys[pick]
                pids = pids[pick]

            render_pkg = render(view, triangles, pipe, background)
            pred_depth = render_pkg["surf_depth"][0].detach().cpu().numpy()
            pred_normal = render_pkg["rend_normal"].detach().cpu().numpy().transpose(1, 2, 0)
            pred_normal = _safe_unit(pred_normal)
            h, w = pred_depth.shape

            sx = float(view.image_width) / float(cam_info.width)
            sy = float(view.image_height) / float(cam_info.height)
            px = np.clip(np.round(xys[:, 0] * sx).astype(np.int64), 0, w - 1)
            py = np.clip(np.round(xys[:, 1] * sy).astype(np.int64), 0, h - 1)

            xyz = np.stack([pid_to_xyz[int(pid)] for pid in pids], axis=0)
            # cam_info.R stores transpose(qvec2rotmat), so X_cam(row) = X_world @ R + T
            xyz_cam = xyz @ np.array(cam_info.R, dtype=np.float64) + np.array(cam_info.T, dtype=np.float64)[None, :]
            gt_depth = xyz_cam[:, 2]
            pd = pred_depth[py, px]

            keep = (gt_depth > 1e-6) & np.isfinite(gt_depth) & np.isfinite(pd) & (pd > 1e-6)
            if not np.any(keep):
                continue
            gt_depth = gt_depth[keep]
            pd = pd[keep]
            px = px[keep]
            py = py[keep]
            pids = pids[keep]

            all_depth_pred.append(pd)
            all_depth_gt.append(gt_depth)

            normal_count = 0
            if compute_normal:
                cos_vals = []
                for i, pid in enumerate(pids):
                    gn = estimate_point_normal(int(pid))
                    if gn is None:
                        continue
                    # normal sign is ambiguous; use abs cosine.
                    pn = pred_normal[py[i], px[i]]
                    c = float(abs(np.dot(pn, gn)))
                    if np.isfinite(c):
                        cos_vals.append(max(0.0, min(1.0, c)))
                if len(cos_vals) > 0:
                    cos_vals = np.array(cos_vals, dtype=np.float64)
                    all_cos_abs.append(cos_vals)
                    normal_count = int(cos_vals.shape[0])

            per_view.append(
                {
                    "image_name": view.image_name,
                    "depth_points": int(gt_depth.shape[0]),
                    "depth_mae": float(np.mean(np.abs(pd - gt_depth))),
                    "depth_abs_rel": float(np.mean(np.abs(pd - gt_depth) / np.clip(gt_depth, 1e-8, None))),
                    "normal_points": normal_count,
                }
            )

        if len(all_depth_pred) == 0:
            raise RuntimeError("No valid COLMAP correspondences were evaluated.")

        depth_pred = np.concatenate(all_depth_pred, axis=0)
        depth_gt = np.concatenate(all_depth_gt, axis=0)
        depth_stats = depth_metrics(depth_pred, depth_gt)

        normal_stats = None
        if len(all_cos_abs) > 0:
            normal_stats = normal_metrics_from_abs_cos(np.concatenate(all_cos_abs, axis=0))

        result = {
            "model_path": dataset.model_path,
            "iteration": int(scene.loaded_iter),
            "num_test_views": int(len(scene.getTestCameras())),
            "num_views_evaluated": int(len(per_view)),
            "point_error_max": float(point_error_max),
            "max_points_per_view": int(max_points_per_view),
            "depth": depth_stats,
            "normal": normal_stats,
            "normal_note": "COLMAP does not provide ground-truth normals; normals are estimated by local PCA on sparse 3D points."
            if compute_normal
            else "normal evaluation disabled",
            "per_view": per_view,
        }
        return result


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate geometric realism against COLMAP sparse geometry.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--max_points_per_view", default=3000, type=int)
    parser.add_argument("--point_error_max", default=2.0, type=float)
    parser.add_argument("--normal_knn", default=24, type=int)
    parser.add_argument("--no_normal", action="store_true")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--output", default="", type=str)
    args = get_combined_args(parser)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    result = evaluate_geometry(
        dataset=dataset,
        iteration=args.iteration,
        pipe=pipe,
        max_points_per_view=args.max_points_per_view,
        point_error_max=args.point_error_max,
        normal_knn=args.normal_knn,
        compute_normal=not args.no_normal,
        seed=args.seed,
    )

    out_path = (
        Path(args.output)
        if args.output
        else Path(dataset.model_path) / "geometry_eval_colmap" / f"iter_{result['iteration']}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[GeomEval] Saved: {out_path}")
    print("[GeomEval] Depth:", result["depth"])
    if result["normal"] is not None:
        print("[GeomEval] Normal:", result["normal"])
