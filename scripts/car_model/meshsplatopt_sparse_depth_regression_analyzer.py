#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams  # noqa: E402
from scene import Scene  # noqa: E402
from triangle_renderer import TriangleModel, render  # noqa: E402
from utils.prism_geometry_proxy import (  # noqa: E402
    GeometryProxyConfig,
    build_geometry_proxy_context,
    collect_view_sparse_depth_correspondences,
    normalize_image_key,
)
from utils.sparse_depth_regression import (  # noqa: E402
    SparseDepthRegressionConfig,
    build_sparse_depth_regression_table,
    write_sparse_depth_regression_outputs,
)


def _clone_dataset(dataset, model_path: str):
    out = copy.copy(dataset)
    out.model_path = str(model_path)
    return out


def _load_scene(dataset, model_path: str, iteration: int):
    triangles = TriangleModel(dataset.sh_degree)
    triangles.scaling = 4
    scene = Scene(
        args=_clone_dataset(dataset, model_path),
        triangles=triangles,
        init_opacity=None,
        set_sigma=None,
        load_iteration=int(iteration),
        shuffle=False,
    )
    return scene, triangles


def _select_views(scene, split: str):
    split = str(split).lower()
    if split == "train":
        return list(scene.getTrainCameras())
    if split == "test":
        return list(scene.getTestCameras())
    if split in {"all", "train_test"}:
        return list(scene.getTrainCameras()) + list(scene.getTestCameras())
    if split == "calibration":
        train = list(scene.getTrainCameras())
        return train[:: max(1, len(train) // 32)] if len(train) > 32 else train
    raise ValueError(f"unknown split: {split}")


def _sample_depth(render_pkg: dict[str, Any], px: np.ndarray, py: np.ndarray) -> np.ndarray:
    surf_depth = render_pkg.get("surf_depth", None)
    if surf_depth is None:
        return np.full(px.shape, np.nan, dtype=np.float64)
    depth = surf_depth[0].detach().cpu().numpy()
    h, w = depth.shape
    x = np.clip(px.astype(np.int64), 0, w - 1)
    y = np.clip(py.astype(np.int64), 0, h - 1)
    return depth[y, x].astype(np.float64)


def _sample_alpha(render_pkg: dict[str, Any], px: np.ndarray, py: np.ndarray) -> np.ndarray | None:
    alpha = render_pkg.get("alpha", render_pkg.get("rend_alpha", None))
    if alpha is None:
        return None
    try:
        arr = alpha.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        h, w = arr.shape
        x = np.clip(px.astype(np.int64), 0, w - 1)
        y = np.clip(py.astype(np.int64), 0, h - 1)
        return arr[y, x].astype(np.float64)
    except Exception:
        return None


def _sample_rgb_residual(render_pkg: dict[str, Any], view, px: np.ndarray, py: np.ndarray) -> np.ndarray | None:
    image = render_pkg.get("render", None)
    gt = getattr(view, "original_image", None)
    if image is None or gt is None:
        return None
    try:
        pred = torch.clamp(image.detach(), 0.0, 1.0).cpu().numpy()
        target = torch.clamp(gt[:3].detach(), 0.0, 1.0).cpu().numpy()
        h, w = pred.shape[-2], pred.shape[-1]
        x = np.clip(px.astype(np.int64), 0, w - 1)
        y = np.clip(py.astype(np.int64), 0, h - 1)
        return np.mean(np.abs(pred[:, y, x] - target[:, y, x]), axis=0).astype(np.float64)
    except Exception:
        return None


def _extend_optional(dst: dict[str, list], key: str, values: np.ndarray | None, n: int) -> None:
    if values is None:
        dst.setdefault(key, []).extend([np.nan] * n)
    else:
        dst.setdefault(key, []).extend(np.asarray(values).reshape(-1).tolist())


def run(args) -> int:
    dataset = args.dataset
    pipe = args.pipe
    parent_scene, parent_triangles = _load_scene(dataset, args.parent_model_path, args.parent_iteration)
    _, candidate_triangles = _load_scene(dataset, args.candidate_model_path, args.candidate_iteration)
    views = _select_views(parent_scene, args.split)
    if not views:
        raise RuntimeError(f"no views selected for split={args.split}; use --eval for test split")
    if len(parent_scene.scene_info.colmap_points3d or {}) == 0:
        raise RuntimeError("COLMAP sparse points are unavailable; cannot analyze sparse-depth regressions.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    proxy_cfg = GeometryProxyConfig(
        max_points_per_view=int(args.max_points_per_view),
        point_error_max=float(args.point_error_max),
        normal_knn=int(args.normal_knn),
        compute_normal=False,
        seed=int(args.seed),
        sample_mode=str(args.sample_mode),
        low_error_fraction=float(args.low_error_fraction),
    )
    cam_infos = []
    cam_infos.extend(list(getattr(parent_scene.scene_info, "train_cameras", []) or []))
    cam_infos.extend(list(getattr(parent_scene.scene_info, "test_cameras", []) or []))
    proxy_ctx = build_geometry_proxy_context(
        colmap_points3d=parent_scene.scene_info.colmap_points3d,
        cam_infos=cam_infos,
        cfg=proxy_cfg,
    )
    rng = np.random.default_rng(int(args.seed))
    rows: dict[str, list] = {
        "image_name": [],
        "image_key": [],
        "point3D_id": [],
        "px": [],
        "py": [],
        "width": [],
        "height": [],
        "gt_depth": [],
        "parent_pred_depth": [],
        "candidate_pred_depth": [],
    }
    dropped: dict[str, int] = {}
    with torch.no_grad():
        for view in views:
            parent_pkg = render(view, parent_triangles, pipe, background)
            candidate_pkg = render(view, candidate_triangles, pipe, background)
            corr = collect_view_sparse_depth_correspondences(view=view, ctx=proxy_ctx, cfg=proxy_cfg, rng=rng)
            n = int(corr.get("num_matches", 0))
            if n <= 0:
                reason = str(corr.get("reason", "unknown"))
                dropped[reason] = int(dropped.get(reason, 0)) + 1
                continue
            px = np.asarray(corr["px"], dtype=np.int64)
            py = np.asarray(corr["py"], dtype=np.int64)
            parent_depth = _sample_depth(parent_pkg, px, py)
            candidate_depth = _sample_depth(candidate_pkg, px, py)
            depth_image = parent_pkg["surf_depth"][0]
            height = int(depth_image.shape[0])
            width = int(depth_image.shape[1])
            image_name = str(getattr(view, "image_name", ""))
            rows["image_name"].extend([image_name] * n)
            rows["image_key"].extend([normalize_image_key(image_name)] * n)
            rows["point3D_id"].extend(np.asarray(corr["point3D_id"], dtype=np.int64).tolist())
            rows["px"].extend(px.tolist())
            rows["py"].extend(py.tolist())
            rows["width"].extend([width] * n)
            rows["height"].extend([height] * n)
            rows["gt_depth"].extend(np.asarray(corr["gt_depth"], dtype=np.float64).tolist())
            rows["parent_pred_depth"].extend(parent_depth.tolist())
            rows["candidate_pred_depth"].extend(candidate_depth.tolist())
            _extend_optional(rows, "parent_alpha", _sample_alpha(parent_pkg, px, py), n)
            _extend_optional(rows, "candidate_alpha", _sample_alpha(candidate_pkg, px, py), n)
            _extend_optional(rows, "parent_rgb_residual", _sample_rgb_residual(parent_pkg, view, px, py), n)
            _extend_optional(rows, "candidate_rgb_residual", _sample_rgb_residual(candidate_pkg, view, px, py), n)

    cfg = SparseDepthRegressionConfig(
        margin_abs=float(args.margin_abs),
        margin_rel=float(args.margin_rel),
        gate_top_fraction=float(args.gate_top_fraction),
        boundary_fraction=float(args.boundary_fraction),
        cluster_grid_size=int(args.cluster_grid_size),
    )
    table = build_sparse_depth_regression_table(rows, cfg)
    manifest = {
        "source_path": str(dataset.source_path),
        "images": str(dataset.images),
        "resolution": int(dataset.resolution),
        "eval": bool(dataset.eval),
        "split": str(args.split),
        "parent_model_path": str(args.parent_model_path),
        "parent_iteration": int(args.parent_iteration),
        "candidate_model_path": str(args.candidate_model_path),
        "candidate_iteration": int(args.candidate_iteration),
        "max_points_per_view": int(args.max_points_per_view),
        "point_error_max": float(args.point_error_max),
        "sample_mode": str(args.sample_mode),
        "low_error_fraction": float(args.low_error_fraction),
        "seed": int(args.seed),
        "dropped_views": dropped,
        "note": "If split=test, this output is diagnostic only and must not be used for training sentinel selection.",
    }
    payload = write_sparse_depth_regression_outputs(
        output_dir=Path(args.output_dir),
        table=table,
        cfg=cfg,
        manifest=manifest,
    )
    print(json.dumps(payload["summary"]["global"], indent=2, sort_keys=True))
    print(f"[SCE1] Saved sparse-depth regression outputs to {args.output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare parent/candidate sparse COLMAP depth correspondences.")
    model = ModelParams(parser, sentinel=False)
    pipe = PipelineParams(parser)
    parser.add_argument("--parent_model_path", required=True)
    parser.add_argument("--parent_iteration", type=int, required=True)
    parser.add_argument("--candidate_model_path", required=True)
    parser.add_argument("--candidate_iteration", type=int, required=True)
    parser.add_argument("--split", choices=("train", "test", "calibration", "all"), default="test")
    parser.add_argument("--max_points_per_view", type=int, default=500)
    parser.add_argument("--point_error_max", type=float, default=2.0)
    parser.add_argument("--normal_knn", type=int, default=24)
    parser.add_argument("--sample_mode", default="mixed_low_error")
    parser.add_argument("--low_error_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--margin_abs", type=float, default=0.0)
    parser.add_argument("--margin_rel", type=float, default=0.0)
    parser.add_argument("--gate_top_fraction", type=float, default=0.10)
    parser.add_argument("--boundary_fraction", type=float, default=0.05)
    parser.add_argument("--cluster_grid_size", type=int, default=64)
    parser.add_argument("--output_dir", required=True)
    parser.set_defaults(_model_group=model, _pipe_group=pipe)
    return parser


def main() -> int:
    parser = build_parser()
    parsed = parser.parse_args()
    parsed.dataset = parser.get_default("_model_group").extract(parsed)
    parsed.pipe = parser.get_default("_pipe_group").extract(parsed)
    return run(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
