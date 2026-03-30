#!/usr/bin/env python3
import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from triangle_renderer import TriangleModel, render
from utils.prism_geometry_proxy import (
    GeometryProxyConfig,
    build_geometry_proxy_context,
    evaluate_views_sparse_geometry_proxy,
)


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
        if len(scene.scene_info.colmap_points3d or {}) == 0:
            raise RuntimeError("COLMAP sparse points are unavailable; cannot evaluate geometry.")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        proxy_cfg = GeometryProxyConfig(
            max_points_per_view=int(max_points_per_view),
            point_error_max=float(point_error_max),
            normal_knn=int(normal_knn),
            compute_normal=bool(compute_normal),
            seed=int(seed),
        )
        cam_infos = []
        cam_infos.extend(list(getattr(scene.scene_info, "train_cameras", []) or []))
        cam_infos.extend(list(getattr(scene.scene_info, "test_cameras", []) or []))
        proxy_ctx = build_geometry_proxy_context(
            colmap_points3d=scene.scene_info.colmap_points3d,
            cam_infos=cam_infos,
            cfg=proxy_cfg,
        )
        proxy_res = evaluate_views_sparse_geometry_proxy(
            views=scene.getTestCameras(),
            triangles=triangles,
            render_func=render,
            pipe=pipe,
            background=background,
            ctx=proxy_ctx,
            cfg=proxy_cfg,
        )
        if proxy_res["depth"] is None:
            raise RuntimeError("No valid COLMAP correspondences were evaluated.")

        result = {
            "model_path": dataset.model_path,
            "iteration": int(scene.loaded_iter),
            "num_test_views": int(len(scene.getTestCameras())),
            "num_views_evaluated": int(proxy_res["num_depth_views_used"]),
            "point_error_max": float(point_error_max),
            "max_points_per_view": int(max_points_per_view),
            "depth": proxy_res["depth"],
            "normal": proxy_res["normal"],
            "normal_note": proxy_res["normal_note"],
            # Keep previous behavior: only include views with valid depth matches.
            "per_view": [p for p in proxy_res["per_view"] if int(p.get("depth_points", 0)) > 0],
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
