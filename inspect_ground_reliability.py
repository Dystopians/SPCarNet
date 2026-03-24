import os
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, TriangleModel
from triangle_renderer import render
from utils.ground_plane_utils import GroundPlaneConfig, estimate_or_load_ground_plane
from utils.ground_association_utils import GroundAssociationConfig, GroundAssociationTracker


if __name__ == "__main__":
    parser = ArgumentParser(description="Inspect ground-aware regularization reliability before training")
    mp = ModelParams(parser, sentinel=False)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", type=int, default=-1, help="Use -1 to load latest trained mesh.")
    parser.add_argument("--max_views", type=int, default=40, help="Number of train views for association probing.")
    args = get_combined_args(parser)

    dataset = mp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(
        args=dataset,
        triangles=triangles,
        init_opacity=opt.set_weight,
        set_sigma=opt.set_sigma,
        load_iteration=args.iteration,
        shuffle=False,
    )

    plane_cfg = GroundPlaneConfig(
        source_priority=[s.strip() for s in str(opt.ground_plane_source_priority).split(",") if s.strip()],
        min_points=int(opt.ground_plane_min_points),
        ransac_iters=int(opt.ground_plane_ransac_iters),
        ransac_dist_thresh=float(opt.ground_plane_ransac_dist_thresh),
        inlier_ratio_min=float(opt.ground_plane_inlier_ratio_min),
        track_len_min=int(opt.ground_plane_track_len_min),
        obs_min=int(opt.ground_plane_obs_min),
        obs_ratio_min=float(opt.ground_plane_obs_ratio_min),
        colmap_error_max=float(opt.ground_plane_colmap_error_max),
        depth_max_samples_per_view=int(opt.ground_plane_depth_max_samples_per_view),
        depth_sample_stride=int(opt.ground_plane_depth_sample_stride),
        depth_inv_min=float(opt.ground_plane_depth_inv_min),
        mesh_sample_max=int(opt.ground_plane_mesh_sample_max),
        axis_consistency_min=float(opt.ground_plane_axis_consistency_min),
        outlier_quantile=float(opt.ground_plane_outlier_quantile),
        use_if_poor=bool(opt.ground_plane_use_if_poor),
        cache_file=str(opt.ground_plane_cache_file),
        recompute_interval=int(opt.ground_plane_recompute_interval),
        force_recompute=bool(opt.ground_plane_force_recompute),
        diag_save=bool(opt.ground_plane_diag_save),
        diag_dir=str(opt.ground_plane_diag_dir),
    )
    plane = estimate_or_load_ground_plane(scene=scene, triangles=triangles, cfg=plane_cfg, iteration=0, force_recompute=True)
    print("[InspectGround] plane:", plane)

    assoc_cfg = GroundAssociationConfig(
        min_observations=int(opt.ground_assoc_min_observations),
        min_ground_ratio=float(opt.ground_assoc_min_ground_ratio),
        min_view_consistency=float(opt.ground_assoc_min_view_consistency),
        per_view_ground_ratio=float(opt.ground_assoc_per_view_ground_ratio),
        boundary_margin=float(opt.ground_assoc_boundary_margin),
        confidence_min=float(opt.ground_assoc_confidence_min),
        use_cache=False,
        cache_file=str(opt.ground_assoc_cache_file),
        cache_every=0,
        debug_every=0,
        debug_dir=str(opt.ground_assoc_debug_dir),
        hist_bins=int(opt.ground_assoc_hist_bins),
    )
    tracker = GroundAssociationTracker(
        num_triangles=int(triangles._triangle_indices.shape[0]),
        device=triangles.vertices.device,
        model_path=scene.model_path,
        cfg=assoc_cfg,
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    views = scene.getTrainCameras()[: max(int(args.max_views), 1)]
    with torch.no_grad():
        for cam in views:
            pkg = render(cam, triangles, pipe, background)
            tracker.update_from_render(pkg, cam)

    stats = tracker.get_statistics()
    n_ground = int(stats["is_ground_mask"].sum().item())
    n_uncertain = int(stats["boundary_uncertain_mask"].sum().item())
    n_reliable = int(stats["reliable_observation_mask"].sum().item())
    print(
        "[InspectGround] coverage: reliable={} ground={} boundary_uncertain={} total={}".format(
            n_reliable,
            n_ground,
            n_uncertain,
            int(triangles._triangle_indices.shape[0]),
        )
    )
    print(
        "[InspectGround] means: support_ratio={:.5f}, view_consistency={:.5f}, confidence={:.5f}".format(
            float(stats["ground_support_ratio"].mean().item()),
            float(stats["view_consistency"].mean().item()),
            float(stats["confidence"].mean().item()),
        )
    )

    if plane.get("ok", False) and plane.get("enabled_for_loss", False) and n_ground > 64:
        print("[InspectGround] Likely reliable for ground-aware regularization.")
    else:
        print("[InspectGround] Likely unreliable; inspect masks/thresholds before training.")
