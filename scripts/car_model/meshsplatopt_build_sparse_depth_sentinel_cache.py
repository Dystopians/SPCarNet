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
    estimate_view_sparse_observability,
    normalize_image_key,
)
from utils.sparse_depth_regression import SparseDepthRegressionConfig, build_sparse_depth_regression_table  # noqa: E402
from utils.sparse_depth_sentinel_cache import (  # noqa: E402
    SentinelCacheConfig,
    build_sparse_depth_sentinel_cache,
    write_sparse_depth_sentinel_cache,
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


def _split_views(scene, split: str):
    split = str(split).lower()
    if split == "test":
        raise RuntimeError("SCE2 sentinel cache builder refuses test split to prevent leakage.")
    train = list(scene.getTrainCameras())
    if split == "train":
        return train
    if split == "calibration":
        return train[:: max(1, len(train) // 32)] if len(train) > 32 else train
    raise ValueError(f"unsupported sentinel split: {split}")


def _load_view_risk(path: str) -> dict[str, float]:
    if not path:
        return {}
    p = Path(path)
    if p.is_dir():
        csv_path = p / "per_view_regression_summary.csv"
    elif p.suffix.lower() == ".csv":
        csv_path = p
    else:
        csv_path = p.parent / "per_view_regression_summary.csv"
    if not csv_path.is_file():
        return {}
    import csv

    out: dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = normalize_image_key(row.get("image_key", ""))
            try:
                out[key] = max(float(row.get("delta_absrel", 0.0)), 0.0) + 0.1 * max(float(row.get("delta_mae", 0.0)), 0.0)
            except Exception:
                out[key] = 0.0
    return out


def _select_views(views, proxy_ctx, proxy_cfg, args):
    if int(args.num_views) <= 0 or len(views) <= int(args.num_views):
        return list(views), [{"image_key": normalize_image_key(getattr(v, "image_name", "")), "selection_reason": "all_available"} for v in views]
    risk = _load_view_risk(str(args.regression_report))
    scored = []
    for idx, view in enumerate(views):
        key = normalize_image_key(getattr(view, "image_name", ""))
        obs = estimate_view_sparse_observability(view=view, ctx=proxy_ctx, cfg=proxy_cfg)
        observable = float(obs.get("depth_matches", 0.0))
        score = 0.0
        reason = []
        if bool(args.prefer_observable_views):
            score += min(observable, float(args.max_points_per_view)) / max(1.0, float(args.max_points_per_view))
            reason.append("observable")
        if bool(args.prefer_hard_views) and key in risk:
            score += 2.0 * float(risk[key])
            reason.append("hard_regression_report")
        scored.append((float(score), idx, view, ",".join(reason) if reason else "deterministic"))
    scored.sort(key=lambda x: (-x[0], x[1]))
    pool = scored[: max(int(args.num_views), min(len(scored), int(args.num_views) * 2))]
    pool.sort(key=lambda x: x[1])
    if len(pool) > int(args.num_views):
        pick_idx = np.linspace(0, len(pool) - 1, num=int(args.num_views))
        selected = [pool[int(round(float(i)))] for i in pick_idx]
    else:
        selected = pool
    selected_keys = {id(x[2]) for x in selected}
    manifest = []
    for score, idx, view, reason in scored:
        if id(view) in selected_keys:
            manifest.append(
                {
                    "image_key": normalize_image_key(getattr(view, "image_name", "")),
                    "score": float(score),
                    "source_index": int(idx),
                    "selection_reason": reason,
                }
            )
    return [x[2] for x in selected], manifest


def _sample_depth(render_pkg: dict[str, Any], px: np.ndarray, py: np.ndarray) -> np.ndarray:
    surf_depth = render_pkg.get("surf_depth", None)
    if surf_depth is None:
        return np.full(px.shape, np.nan, dtype=np.float64)
    depth = surf_depth[0].detach().cpu().numpy()
    h, w = depth.shape
    x = np.clip(px.astype(np.int64), 0, w - 1)
    y = np.clip(py.astype(np.int64), 0, h - 1)
    return depth[y, x].astype(np.float64)


def run(args) -> int:
    if str(args.split).lower() == "test":
        raise RuntimeError("SCE2 sentinel cache builder refuses test split to prevent leakage.")
    dataset = args.dataset
    pipe = args.pipe
    parent_scene, parent_triangles = _load_scene(dataset, args.parent_model_path, args.parent_iteration)
    candidate_triangles = None
    if args.candidate_model_path:
        _, candidate_triangles = _load_scene(dataset, args.candidate_model_path, args.candidate_iteration)
    if len(parent_scene.scene_info.colmap_points3d or {}) == 0:
        raise RuntimeError("COLMAP sparse points are unavailable; cannot build sentinel cache.")

    proxy_cfg = GeometryProxyConfig(
        max_points_per_view=int(args.max_points_per_view),
        point_error_max=float(args.point_error_max),
        normal_knn=24,
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
    views, view_manifest = _select_views(_split_views(parent_scene, args.split), proxy_ctx, proxy_cfg, args)
    if not views:
        raise RuntimeError(f"no train/calibration views selected for split={args.split}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
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
            candidate_pkg = render(view, candidate_triangles, pipe, background) if candidate_triangles is not None else None
            corr = collect_view_sparse_depth_correspondences(view=view, ctx=proxy_ctx, cfg=proxy_cfg, rng=rng)
            n = int(corr.get("num_matches", 0))
            if n <= 0:
                reason = str(corr.get("reason", "unknown"))
                dropped[reason] = int(dropped.get(reason, 0)) + 1
                continue
            px = np.asarray(corr["px"], dtype=np.int64)
            py = np.asarray(corr["py"], dtype=np.int64)
            parent_depth = _sample_depth(parent_pkg, px, py)
            candidate_depth = _sample_depth(candidate_pkg, px, py) if candidate_pkg is not None else np.full(parent_depth.shape, np.nan, dtype=np.float64)
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

    reg_table = build_sparse_depth_regression_table(rows, SparseDepthRegressionConfig(cluster_grid_size=int(args.cluster_grid_size)))
    manifest = {
        "source_path": str(dataset.source_path),
        "images": str(dataset.images),
        "resolution": int(dataset.resolution),
        "eval": bool(dataset.eval),
        "parent_model_path": str(args.parent_model_path),
        "parent_iteration": int(args.parent_iteration),
        "candidate_model_path": str(args.candidate_model_path),
        "candidate_iteration": int(args.candidate_iteration) if args.candidate_model_path else -1,
        "view_selection": view_manifest,
        "dropped_views": dropped,
        "prefer_hard_views": bool(args.prefer_hard_views),
        "prefer_observable_views": bool(args.prefer_observable_views),
        "regression_report": str(args.regression_report),
    }
    cfg = SentinelCacheConfig(
        split=str(args.split),
        seed=int(args.seed),
        max_points_per_view=int(args.max_points_per_view),
        cluster_grid_size=int(args.cluster_grid_size),
        hard_regression_weight=float(args.hard_regression_weight),
        cluster_balance=bool(args.cluster_balance),
    )
    cache = build_sparse_depth_sentinel_cache(table=reg_table, manifest=manifest, cfg=cfg)
    write_sparse_depth_sentinel_cache(output=Path(args.output), cache=cache, cfg=cfg)
    print(json.dumps(cache["manifest"], indent=2, sort_keys=True))
    print(f"[SCE2] Saved sentinel cache to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build train/calibration sparse-depth sentinel cache without test leakage.")
    model = ModelParams(parser, sentinel=False)
    pipe = PipelineParams(parser)
    parser.add_argument("--parent_model_path", required=True)
    parser.add_argument("--parent_iteration", type=int, required=True)
    parser.add_argument("--candidate_model_path", default="")
    parser.add_argument("--candidate_iteration", type=int, default=-1)
    parser.add_argument("--split", choices=("train", "calibration", "test"), default="train")
    parser.add_argument("--num_views", type=int, default=32)
    parser.add_argument("--prefer_hard_views", action="store_true")
    parser.add_argument("--prefer_observable_views", action="store_true")
    parser.add_argument("--max_points_per_view", type=int, default=500)
    parser.add_argument("--sample_mode", default="mixed_low_error")
    parser.add_argument("--low_error_fraction", type=float, default=0.5)
    parser.add_argument("--point_error_max", type=float, default=2.0)
    parser.add_argument("--regression_report", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cluster_grid_size", type=int, default=64)
    parser.add_argument("--hard_regression_weight", type=float, default=2.0)
    parser.add_argument("--cluster_balance", action="store_true")
    parser.add_argument("--output", required=True)
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
