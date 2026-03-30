import json
from argparse import ArgumentParser

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from triangle_renderer import render
from utils.prism_counterfactual import (
    CalibrationConfig,
    CounterfactualGateConfig,
    build_calibration_set,
    run_counterfactual_simulation,
)
from utils.prism_geometry_proxy import GeometryProxyConfig, build_geometry_proxy_context


def _parse_candidate_ids(spec: str) -> torch.Tensor:
    values = []
    for tok in str(spec).split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            values.append(int(t))
        except ValueError:
            continue
    return torch.tensor(values, dtype=torch.int64, device="cuda")


if __name__ == "__main__":
    parser = ArgumentParser(description="Debug PRISM counterfactual simulator with candidate triangle ids")
    mp = ModelParams(parser, sentinel=False)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", type=int, default=-1, help="Use -1 to load latest model checkpoint.")
    parser.add_argument("--candidate_ids", type=str, required=True, help="Comma separated triangle ids.")
    parser.add_argument("--calib_num_buffer_views", type=int, default=8)
    parser.add_argument("--calib_num_hard_train_views", type=int, default=8)
    parser.add_argument("--calib_hard_pool_size", type=int, default=64)
    parser.add_argument("--output_json", type=str, default="")
    args = get_combined_args(parser)

    dataset = mp.extract(args)
    pipe = pp.extract(args)
    _ = op.extract(args)

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(
        args=dataset,
        triangles=triangles,
        init_opacity=None,
        set_sigma=None,
        load_iteration=args.iteration,
        shuffle=False,
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    proxy_cfg = GeometryProxyConfig(
        max_points_per_view=3000,
        point_error_max=2.0,
        normal_knn=24,
        compute_normal=True,
        seed=7,
    )
    cam_infos = []
    cam_infos.extend(list(getattr(scene.scene_info, "train_cameras", []) or []))
    cam_infos.extend(list(getattr(scene.scene_info, "test_cameras", []) or []))
    proxy_ctx = build_geometry_proxy_context(
        colmap_points3d=getattr(scene.scene_info, "colmap_points3d", None),
        cam_infos=cam_infos,
        cfg=proxy_cfg,
    )

    calib_cfg = CalibrationConfig(
        num_buffer_views=int(args.calib_num_buffer_views),
        num_hard_train_views=int(args.calib_num_hard_train_views),
        hard_view_pool_size=int(args.calib_hard_pool_size),
    )
    views = build_calibration_set(
        scene=scene,
        dataset=dataset,
        triangles=triangles,
        render_func=render,
        pipe=pipe,
        background=background,
        cfg=calib_cfg,
        proxy_ctx=proxy_ctx,
        proxy_cfg=proxy_cfg,
    )
    gate_cfg = CounterfactualGateConfig(
        min_delta_psnr_db=-0.05,
        max_delta_mae=0.002,
        max_delta_absrel=0.0008,
        max_delta_mean_angle_deg=0.3,
        max_changed_pixel_ratio=0.005,
        changed_pixel_threshold=0.02,
    )

    cand = _parse_candidate_ids(args.candidate_ids)
    decision = run_counterfactual_simulation(
        scene=scene,
        triangles=triangles,
        render_func=render,
        pipe=pipe,
        background=background,
        candidate_triangle_ids=cand,
        calibration_views=views,
        gate_cfg=gate_cfg,
        proxy_ctx=proxy_ctx,
        proxy_cfg=proxy_cfg,
    )
    payload = {
        "accept": bool(decision.accept),
        "num_candidates": int(decision.num_candidates),
        "reason": decision.reason,
        "deltas": decision.deltas,
        "baseline": decision.baseline,
        "counterfactual": decision.counterfactual,
        "num_calibration_views": int(len(views)),
    }
    print("[PRISM-CF] result:", payload)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[PRISM-CF] saved: {args.output_json}")
