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
from utils.prism_geometry_proxy import normalize_image_key  # noqa: E402
from utils.sentinel_parent_pareto_gate import (  # noqa: E402
    SentinelParentParetoGateConfig,
    evaluate_sentinel_parent_pareto_gate,
    write_sentinel_parent_pareto_gate_outputs,
)
from utils.sparse_depth_parent_rollback import load_sparse_depth_parent_rollback_cache  # noqa: E402
from utils.sparse_depth_regression import SparseDepthRegressionConfig, build_sparse_depth_regression_table  # noqa: E402


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


def _views_by_key(scene):
    views = list(scene.getTrainCameras()) + list(scene.getTestCameras())
    return {normalize_image_key(getattr(v, "image_name", "")): v for v in views}


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
    cache = load_sparse_depth_parent_rollback_cache(args.sentinel_cache, allow_test_cache=True)
    manifest = dict(cache.get("manifest", {}))
    split = str(manifest.get("split", ""))
    dataset = args.dataset
    pipe = args.pipe
    parent_scene, parent_triangles = _load_scene(dataset, args.parent_model_path, args.parent_iteration)
    _, candidate_triangles = _load_scene(dataset, args.candidate_model_path, args.candidate_iteration)
    view_lookup = _views_by_key(parent_scene)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
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
    missing_views = []
    with torch.no_grad():
        for key, entry in sorted(cache["by_image_key"].items(), key=lambda kv: kv[0]):
            view = view_lookup.get(str(key), None)
            if view is None:
                missing_views.append(str(key))
                continue
            px = np.asarray(entry["px"], dtype=np.int64).reshape(-1)
            py = np.asarray(entry["py"], dtype=np.int64).reshape(-1)
            parent_pkg = render(view, parent_triangles, pipe, background)
            candidate_pkg = render(view, candidate_triangles, pipe, background)
            parent_depth = _sample_depth(parent_pkg, px, py)
            candidate_depth = _sample_depth(candidate_pkg, px, py)
            depth_image = parent_pkg["surf_depth"][0]
            height = int(depth_image.shape[0])
            width = int(depth_image.shape[1])
            n = int(px.shape[0])
            image_name = str(getattr(view, "image_name", ""))
            rows["image_name"].extend([image_name] * n)
            rows["image_key"].extend([str(key)] * n)
            rows["point3D_id"].extend(np.asarray(entry["point3D_id"], dtype=np.int64).reshape(-1).tolist())
            rows["px"].extend(px.tolist())
            rows["py"].extend(py.tolist())
            rows["width"].extend([width] * n)
            rows["height"].extend([height] * n)
            rows["gt_depth"].extend(np.asarray(entry["gt_depth"], dtype=np.float64).reshape(-1).tolist())
            rows["parent_pred_depth"].extend(parent_depth.tolist())
            rows["candidate_pred_depth"].extend(candidate_depth.tolist())

    table = build_sparse_depth_regression_table(
        rows,
        SparseDepthRegressionConfig(
            margin_abs=float(args.margin_abs),
            margin_rel=float(args.margin_rel),
            gate_top_fraction=0.10,
            cluster_grid_size=int(args.cluster_grid_size),
        ),
    )
    cfg = SentinelParentParetoGateConfig(
        tolerance_absrel=float(args.tolerance_absrel),
        tolerance_mae=float(args.tolerance_mae),
        worst_view_regression_count_threshold=int(args.worst_view_regression_count_threshold),
        cluster_delta_absrel_threshold=float(args.cluster_delta_absrel_threshold),
        cluster_weight_threshold=float(args.cluster_weight_threshold),
    )
    gate = evaluate_sentinel_parent_pareto_gate(table, cfg)
    out_manifest = {
        "source_path": str(dataset.source_path),
        "split": split,
        "sentinel_cache": str(args.sentinel_cache),
        "parent_model_path": str(args.parent_model_path),
        "parent_iteration": int(args.parent_iteration),
        "candidate_model_path": str(args.candidate_model_path),
        "candidate_iteration": int(args.candidate_iteration),
        "missing_views": missing_views,
        "note": "Test split gate is report-only; train/calibration gate may be used as a pre-run diagnostic.",
    }
    write_sentinel_parent_pareto_gate_outputs(
        output_dir=Path(args.output_dir),
        table=table,
        gate=gate,
        manifest=out_manifest,
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if bool(gate["pass"]) or bool(args.allow_fail_exit_zero) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentinel-level parent-Pareto gate for sparse-depth non-regression.")
    model = ModelParams(parser, sentinel=False)
    pipe = PipelineParams(parser)
    parser.add_argument("--parent_model_path", required=True)
    parser.add_argument("--parent_iteration", type=int, required=True)
    parser.add_argument("--candidate_model_path", required=True)
    parser.add_argument("--candidate_iteration", type=int, required=True)
    parser.add_argument("--sentinel_cache", required=True)
    parser.add_argument("--tolerance_absrel", type=float, default=0.0)
    parser.add_argument("--tolerance_mae", type=float, default=0.0)
    parser.add_argument("--worst_view_regression_count_threshold", type=int, default=0)
    parser.add_argument("--cluster_delta_absrel_threshold", type=float, default=0.0)
    parser.add_argument("--cluster_weight_threshold", type=float, default=0.0)
    parser.add_argument("--margin_abs", type=float, default=0.0)
    parser.add_argument("--margin_rel", type=float, default=0.0)
    parser.add_argument("--cluster_grid_size", type=int, default=64)
    parser.add_argument("--allow_fail_exit_zero", action="store_true")
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
