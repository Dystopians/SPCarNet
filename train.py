#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, vertex_depth_loss_hr
from triangle_renderer import render
import sys
from scene import Scene, TriangleModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, update_indoor
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False
import lpips
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils.ground_plane_utils import (
    GroundPlaneConfig,
    estimate_or_load_ground_plane,
)
from utils.ground_regularization_utils import (
    GroundRegConfig,
    aggregate_ground_regularization_losses,
    maybe_save_ground_geometry_diagnostics,
)
from utils.ground_association_utils import (
    GroundAssociationConfig,
    GroundAssociationTracker,
)
from utils.ground_debug_utils import save_ground_debug_view
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
    print("Using fused SSIM for faster training.")
except:
    FUSED_SSIM_AVAILABLE = False
try:
    from diff_triangle_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
    print("Sparse Adam optimizer available")
except:
    SPARSE_ADAM_AVAILABLE = False
from utils.render_utils import generate_path, create_videos



def training(
        dataset,   
        opt, 
        pipe,
        testing_iterations,
        saving_iterations,
        checkpoint, 
        debug_from,
        scene_name,
        load_iteration=None,
        use_sparse_adam=False,
        wandb_cfg=None,
        wandb_image_log_interval=1000,
        wandb_fixed_views=5,
        wandb_fixed_view_indices="",
        wandb_disable_fixed_views=False,
        ):
    
    first_iter = 0
    tb_writer, wandb_run = prepare_output_and_logger(dataset, wandb_cfg=wandb_cfg)

    # Load parameters, triangles and scene
    triangles = TriangleModel(dataset.sh_degree)

    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma, load_iteration=load_iteration)
    scene_ground_plane = None

    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    triangles.add_percentage = opt.add_percentage


    if scene.loaded_iter is not None:
        first_iter = max(first_iter, int(scene.loaded_iter))

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    number_of_training_views = len(viewpoint_stack)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # define the scheduler for sigma and opacity
    initial_sigma = opt.set_sigma
    final_sigma = 0.0001
    sigma_start = opt.sigma_start
    total_iters = opt.sigma_until

    init_opacity = 0.1
    final_opacity = .9999
    total_iters_opacity = opt.final_opacity_iter

    lambda_weight = opt.lambda_weight
    prune_triangles = opt.prune_triangles_threshold
    prune_size = opt.prune_size
    start_upsampling = opt.start_upsampling
    splitt_large_triangles = opt.splitt_large_triangles
    triangles.size_probs_zero = opt.size_probs_zero
    triangles.size_probs_zero_image_space = opt.size_probs_zero_image_space
    
    need_delaunay = False

    run_restricted_delaunay = opt.densify_until_iter + 1000

    depth_l1_weight = get_expon_lr_func(opt.depth_lambda_init, opt.depth_lambda_final, max_steps=opt.iterations)
    ground_assignment_state = None
    ground_association_tracker = None
    warned_low_coverage = False
    warned_plane_unstable = False
    debug_saved_count = 0

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
    ground_reg_cfg = GroundRegConfig(
        enabled=bool(opt.enable_ground_regularization),
        start_iter=int(opt.ground_reg_start_iter),
        warmup_iters=int(opt.ground_reg_warmup_iters),
        lambda_plane=float(opt.lambda_ground_plane) if bool(opt.enable_ground_plane_loss) else 0.0,
        lambda_normal=float(opt.lambda_ground_normal) if bool(opt.enable_ground_normal_loss) else 0.0,
        lambda_smoothness=float(opt.lambda_ground_smoothness) if bool(opt.enable_ground_smoothness_loss) else 0.0,
        global_scale=float(opt.ground_reg_global_scale),
        target_ratio=float(opt.ground_reg_target_ratio),
        adaptive_ema_decay=float(opt.ground_reg_adaptive_ema_decay),
        adaptive_min_scale=float(opt.ground_reg_adaptive_min_scale),
        adaptive_max_scale=float(opt.ground_reg_adaptive_max_scale),
        max_total=float(opt.ground_reg_max_total),
        smooth_start_iter=int(opt.ground_smooth_start_iter),
        huber_delta=float(opt.ground_reg_huber_delta),
        assignment_min_pixels=int(opt.ground_assign_min_pixels),
        assignment_ema_decay=float(opt.ground_assign_ema_decay),
        plane_min_ratio=float(opt.ground_plane_min_ratio),
        plane_max_abs_height=float(opt.ground_plane_max_abs_height),
        normal_min_ratio=float(opt.ground_normal_min_ratio),
        normal_max_abs_height=float(opt.ground_normal_max_abs_height),
        smooth_min_ratio=float(opt.ground_smooth_min_ratio),
        smooth_max_abs_height=float(opt.ground_smooth_max_abs_height),
        smooth_max_pairs=int(opt.ground_smooth_max_pairs),
        smooth_fallback_edges_max=int(opt.ground_smooth_fallback_edges_max),
    )
    ground_assoc_cfg = GroundAssociationConfig(
        min_observations=int(opt.ground_assoc_min_observations),
        min_ground_ratio=float(opt.ground_assoc_min_ground_ratio),
        min_view_consistency=float(opt.ground_assoc_min_view_consistency),
        per_view_ground_ratio=float(opt.ground_assoc_per_view_ground_ratio),
        boundary_margin=float(opt.ground_assoc_boundary_margin),
        confidence_min=float(opt.ground_assoc_confidence_min),
        use_cache=bool(opt.ground_assoc_use_cache),
        cache_file=str(opt.ground_assoc_cache_file),
        cache_every=int(opt.ground_assoc_cache_every),
        debug_every=int(opt.ground_assoc_debug_every),
        debug_dir=str(opt.ground_assoc_debug_dir),
        hist_bins=int(opt.ground_assoc_hist_bins),
    )

    if bool(opt.enable_ground_regularization) and bool(opt.enable_ground_mesh_assignment):
        ground_association_tracker = GroundAssociationTracker(
            num_triangles=int(triangles._triangle_indices.shape[0]),
            device=triangles.vertices.device,
            model_path=scene.model_path,
            cfg=ground_assoc_cfg,
        )
        ground_association_tracker.load_cache()

    def reset_ground_supervision_state(reason: str):
        """
        Ground association is index-based. Any topology mutation (prune/split/delaunay)
        can invalidate historical per-triangle statistics. Reset to avoid stale labels.
        """
        nonlocal ground_association_tracker, ground_assignment_state
        ground_assignment_state = None
        if ground_association_tracker is not None:
            ground_association_tracker = GroundAssociationTracker(
                num_triangles=int(triangles._triangle_indices.shape[0]),
                device=triangles.vertices.device,
                model_path=scene.model_path,
                cfg=ground_assoc_cfg,
            )
            print(
                "[GroundAssoc] reset due to topology change ({}) @iter={}, triangles={}".format(
                    reason, iteration, int(triangles._triangle_indices.shape[0])
                )
            )

    if bool(opt.enable_ground_plane_estimation or opt.enable_ground_plane_fit):
        plane_payload = estimate_or_load_ground_plane(
            scene=scene,
            triangles=triangles,
            cfg=plane_cfg,
            iteration=first_iter,
            force_recompute=False,
        )
        scene_ground_plane = plane_payload
        scene.ground_plane = plane_payload
        if plane_payload.get("ok", False):
            n = plane_payload["normal"]
            d = plane_payload["offset"]
            print(
                "[GroundPlane] ok={} enabled_for_loss={} from_cache={} inliers={} ratio={:.4f} "
                "normal=({:.5f}, {:.5f}, {:.5f}) d={:.5f} points_total={} sources={}".format(
                    plane_payload.get("ok"),
                    plane_payload.get("enabled_for_loss"),
                    plane_payload.get("from_cache", False),
                    plane_payload.get("inlier_count", -1),
                    plane_payload.get("inlier_ratio", 0.0),
                    float(n[0]), float(n[1]), float(n[2]),
                    float(d),
                    plane_payload.get("points_total", 0),
                    plane_payload.get("source_counts", {}),
                )
            )
        else:
            print(f"[GroundPlane] disabled: {plane_payload}")
            warned_plane_unstable = True

    for iteration in range(first_iter, opt.iterations + 1):
        if bool(opt.enable_ground_plane_estimation or opt.enable_ground_plane_fit):
            recompute_interval = int(opt.ground_plane_recompute_interval)
            if recompute_interval > 0 and iteration > first_iter and (iteration % recompute_interval == 0):
                plane_payload = estimate_or_load_ground_plane(
                    scene=scene,
                    triangles=triangles,
                    cfg=plane_cfg,
                    iteration=iteration,
                    force_recompute=True,
                )
                scene_ground_plane = plane_payload
                scene.ground_plane = plane_payload
                print(
                    "[GroundPlane] recompute @iter {}: ok={} enabled_for_loss={} inliers={} ratio={:.4f}".format(
                        iteration,
                        plane_payload.get("ok", False),
                        plane_payload.get("enabled_for_loss", False),
                        plane_payload.get("inlier_count", -1),
                        float(plane_payload.get("inlier_ratio", 0.0)),
                    )
                )

        if need_delaunay:
            with torch.no_grad():
                triangles.run_restricted_delaunay()
            reset_ground_supervision_state("restricted_delaunay")
            need_delaunay = False

        # Supersampling
        if iteration == start_upsampling:
            triangles.scaling = opt.upscaling_factor
        if iteration == start_upsampling + 5000:
            triangles.scaling = 4

        iter_start.record()
        triangles.update_learning_rate(iteration)

        # Sigma schedule
        if iteration < sigma_start:
            current_sigma = initial_sigma
        else:
            progress = (iteration - sigma_start) / (total_iters - sigma_start)
            progress = min(progress, 1.0)
            current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
        triangles.set_sigma(current_sigma)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if not viewpoint_stack or len(scene.getTrainCameras()) + iteration == opt.iterations:
            viewpoint_stack = scene.getTrainCameras().copy()
            if len(scene.getTrainCameras()) + iteration == opt.iterations:
                triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device="cuda") # reset to 0 to ensure that everything is deleted with an importance score of 0
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg["render"]
        assoc_stats = None
        if ground_association_tracker is not None:
            ground_association_tracker.ensure_num_triangles(int(triangles._triangle_indices.shape[0]))
            ground_association_tracker.update_from_render(render_pkg=render_pkg, viewpoint_cam=viewpoint_cam)
            ground_association_tracker.maybe_save_debug(iteration=iteration)
            if int(ground_assoc_cfg.cache_every) > 0 and (iteration % int(ground_assoc_cfg.cache_every) == 0):
                ground_association_tracker.save_cache()
            assoc_stats = ground_association_tracker.get_statistics()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if getattr(viewpoint_cam, "normal_map", None) is not None:
            gt_normal = viewpoint_cam.normal_map.cuda()
            seg_hr = gt_normal.unsqueeze(0)  # -> [1, 3, H, W]
            seg_ds_area = F.interpolate(seg_hr, size=(gt_image.shape[1], gt_image.shape[2]), mode="area")  # [1, 3, H0, W0]
            gt_normal = seg_ds_area.squeeze(0)  # -> [3, H0, W0]
        else:
            gt_normal = None

        pixel_loss = l1_loss(image, gt_image)

        image_size = render_pkg["scaling"].detach()
        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]

        importance_score = render_pkg["max_blending"].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]

        pixel_count = render_pkg["triangle_was_rendered"].detach() # Not used but could be useful. Gives per triangle, the number of pixels it covered in the current render
        mask = pixel_count > triangles.pixel_count
        triangles.pixel_count[mask] = pixel_count[mask]

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim_value)
    

        # FINAL LOSS
        loss = loss_image

        # Opacity loss
        Lweight_pure = 0.0
        lambda_weight = opt.lambda_weight if iteration < opt.start_opacity_floor else 0
        if lambda_weight > 0:
            mask_out = triangles.vertices.shape[0]
            vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]
            Lweight_pure = vertex_weights.mean()
            Lweight = lambda_weight * Lweight_pure
            loss += Lweight
        else:
            Lweight = 0

        # Vertex depth regularization
        Lvertex_depth_pure = 0.0
        lambda_vertex = opt.lambda_vertex if iteration > opt.start_vertex_opt else 0
        if lambda_vertex > 0:
            depth_down = render_pkg["surf_depth"]
            vertex_depth_out = render_pkg["vertex_depth_out"]
            image_2D = render_pkg["image_2D"]
            vertex_rendered = render_pkg["vertex_rendered"]
            Lvertex_depth_pure = vertex_depth_loss_hr(
                vertex_depth_out,
                image_2D,
                vertex_rendered,
                depth_down,
                max_diff_threshold=opt.max_diff_threshold,
            )
            Lvertex_depth = lambda_vertex * Lvertex_depth_pure
            loss += Lvertex_depth
        else:
            Lvertex_depth = 0

        # Depth loss
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and getattr(viewpoint_cam, "invdepthmap", None) is not None:
            invDepth = 1.0 / (render_pkg["expected_depth"] + 1e-6)
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
        else:
            Ll1depth = 0

        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']

        # Normal regularization (2DGS)
        Lnormal_pure = 0.0
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
        if lambda_normal > 0:
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            Lnormal_pure = normal_error.mean()
            Lnormal = lambda_normal * Lnormal_pure
            loss += Lnormal
        else:
            Lnormal = 0

        # supervised normal loss
        if gt_normal is not None:
            lambda_normals_super = opt.lambda_normals_super if iteration > opt.iteration_mesh else 0
            normal_error = (1 - (rend_normal * gt_normal).sum(dim=0))[None]
            normal_loss_super = lambda_normals_super * (normal_error).mean()
            loss += normal_loss_super
        else:
            normal_loss_super = 0

        # Ground-aware geometry regularization.
        # Intent: constrain reliably-ground triangles to remain near the dominant
        # plane, align normals with the plane normal, and keep local ground heights smooth.
        ground_reg_logs = None
        if ground_reg_cfg.enabled:
            Lground_total, ground_reg_logs, ground_assignment_state = aggregate_ground_regularization_losses(
                triangles=triangles,
                render_pkg=render_pkg,
                viewpoint_cam=viewpoint_cam,
                plane_payload=scene_ground_plane,
                ground_state=ground_assignment_state,
                association_stats=assoc_stats,
                cfg=ground_reg_cfg,
                iteration=iteration,
                image_loss_ref=loss_image.detach(),
            )
            if torch.isfinite(Lground_total):
                loss += Lground_total

            if assoc_stats is not None and (not warned_low_coverage):
                ground_count = int(assoc_stats["is_ground_mask"].sum().item())
                if ground_count < 64 and iteration > int(opt.ground_reg_start_iter):
                    print(
                        "[GroundReg][WARN] low supervision coverage: only {} triangles classified as ground at iter {}.".format(
                            ground_count, iteration
                        )
                    )
                    warned_low_coverage = True
            if (
                scene_ground_plane is not None
                and (not scene_ground_plane.get("enabled_for_loss", False))
                and (not warned_plane_unstable)
                and iteration > int(opt.ground_reg_start_iter)
            ):
                print("[GroundReg][WARN] fitted ground plane is unstable/disabled; plane-based regularization is skipped.")
                warned_plane_unstable = True

            if (
                bool(opt.debug_save_ground_visualizations)
                and getattr(viewpoint_cam, "ground_mask", None) is not None
                and debug_saved_count < int(opt.ground_debug_vis_max)
                and int(opt.ground_debug_vis_every) > 0
                and (iteration % int(opt.ground_debug_vis_every) == 0)
            ):
                dbg_dir = str(opt.ground_debug_vis_dir) if str(opt.ground_debug_vis_dir) else os.path.join(scene.model_path, "ground_debug_views")
                save_ground_debug_view(
                    out_dir=dbg_dir,
                    iteration=iteration,
                    image_name=viewpoint_cam.image_name,
                    gt_image_chw=gt_image,
                    ground_mask_hw=viewpoint_cam.ground_mask,
                    rend_ids_hw=render_pkg.get("rend_ids", None),
                    association_stats=assoc_stats,
                )
                debug_saved_count += 1

            if (
                bool(opt.debug_save_ground_visualizations)
                and assoc_stats is not None
                and scene_ground_plane is not None
                and int(opt.ground_debug_vis_every) > 0
                and (iteration % int(opt.ground_debug_vis_every) == 0)
            ):
                dbg_dir = str(opt.ground_debug_vis_dir) if str(opt.ground_debug_vis_dir) else os.path.join(scene.model_path, "ground_debug_views")
                maybe_save_ground_geometry_diagnostics(
                    triangles=triangles,
                    plane_payload=scene_ground_plane,
                    association_stats=assoc_stats,
                    out_dir=dbg_dir,
                    iteration=iteration,
                )

        loss.backward()
        iter_end.record()

        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                }
                if bool(opt.enable_ground_plane_estimation or opt.enable_ground_plane_fit) and scene_ground_plane is not None:
                    loss_dict["GP"] = "on" if scene_ground_plane.get("enabled_for_loss", False) else "off"
                if ground_reg_logs is not None:
                    loss_dict["Lg"] = f"{ground_reg_logs.get('Lground_total', 0.0):.2e}"
                    loss_dict["Gw"] = f"{ground_reg_logs.get('warmup', 0.0):.2f}"
                    loss_dict["Gs"] = f"{ground_reg_logs.get('ground_reg_adaptive_scale', 1.0):.2f}"
                if assoc_stats is not None:
                    loss_dict["Gtri"] = int(assoc_stats["is_ground_mask"].sum().item())
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer and ground_reg_logs is not None:
                tb_writer.add_scalar("ground_reg/total", ground_reg_logs.get("Lground_total", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/plane", ground_reg_logs.get("Lground_plane", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/normal", ground_reg_logs.get("Lground_normal", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/smoothness", ground_reg_logs.get("Lground_smooth", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/raw", ground_reg_logs.get("Lground_raw", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/warmup", ground_reg_logs.get("warmup", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/smooth_warmup", ground_reg_logs.get("ground_smooth_warmup", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/global_scale", ground_reg_logs.get("ground_reg_global_scale", 1.0), iteration)
                tb_writer.add_scalar("ground_reg/adaptive_scale", ground_reg_logs.get("ground_reg_adaptive_scale", 1.0), iteration)
                tb_writer.add_scalar("ground_reg/adaptive_scale_raw", ground_reg_logs.get("ground_reg_adaptive_scale_raw", 1.0), iteration)
                tb_writer.add_scalar("ground_reg/adaptive_target", ground_reg_logs.get("ground_reg_adaptive_target", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/adaptive_ema", ground_reg_logs.get("ground_reg_adaptive_ema", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/available", ground_reg_logs.get("available", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/reliable_triangles", ground_reg_logs.get("ground_triangles_reliable", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/plane_pure", ground_reg_logs.get("Lground_plane_pure", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/normal_pure", ground_reg_logs.get("Lground_normal_pure", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/smoothness_pure", ground_reg_logs.get("Lground_smooth_pure", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/plane_elements", ground_reg_logs.get("ground_plane_elements", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/normal_elements", ground_reg_logs.get("ground_normal_elements", 0.0), iteration)
                tb_writer.add_scalar("ground_reg/smooth_elements", ground_reg_logs.get("ground_smooth_elements", 0.0), iteration)
            if tb_writer and assoc_stats is not None:
                tri_ids_ground = torch.nonzero(assoc_stats["is_ground_mask"], as_tuple=True)[0]
                ground_vertices = 0
                if tri_ids_ground.numel() > 0:
                    ground_vertices = int(torch.unique(triangles._triangle_indices[tri_ids_ground]).numel())
                tb_writer.add_scalar("ground_assoc/triangles_ground", float(assoc_stats["is_ground_mask"].sum().item()), iteration)
                tb_writer.add_scalar("ground_assoc/vertices_ground", float(ground_vertices), iteration)
                tb_writer.add_scalar("ground_assoc/triangles_boundary_uncertain", float(assoc_stats["boundary_uncertain_mask"].sum().item()), iteration)
                tb_writer.add_scalar("ground_assoc/triangles_unreliable", float((~assoc_stats["reliable_observation_mask"]).sum().item()), iteration)
                tb_writer.add_scalar("ground_assoc/support_ratio_mean", float(assoc_stats["ground_support_ratio"].mean().item()), iteration)
                tb_writer.add_scalar("ground_assoc/view_consistency_mean", float(assoc_stats["view_consistency"].mean().item()), iteration)
            if wandb_run and ground_reg_logs is not None:
                wandb_run.log(
                    {
                        "ground_reg/total": ground_reg_logs.get("Lground_total", 0.0),
                        "ground_reg/plane": ground_reg_logs.get("Lground_plane", 0.0),
                        "ground_reg/normal": ground_reg_logs.get("Lground_normal", 0.0),
                        "ground_reg/smoothness": ground_reg_logs.get("Lground_smooth", 0.0),
                        "ground_reg/raw": ground_reg_logs.get("Lground_raw", 0.0),
                        "ground_reg/warmup": ground_reg_logs.get("warmup", 0.0),
                        "ground_reg/smooth_warmup": ground_reg_logs.get("ground_smooth_warmup", 0.0),
                        "ground_reg/global_scale": ground_reg_logs.get("ground_reg_global_scale", 1.0),
                        "ground_reg/adaptive_scale": ground_reg_logs.get("ground_reg_adaptive_scale", 1.0),
                        "ground_reg/adaptive_scale_raw": ground_reg_logs.get("ground_reg_adaptive_scale_raw", 1.0),
                        "ground_reg/adaptive_target": ground_reg_logs.get("ground_reg_adaptive_target", 0.0),
                        "ground_reg/adaptive_ema": ground_reg_logs.get("ground_reg_adaptive_ema", 0.0),
                        "ground_reg/available": ground_reg_logs.get("available", 0.0),
                        "ground_reg/reliable_triangles": ground_reg_logs.get("ground_triangles_reliable", 0.0),
                        "ground_reg/plane_pure": ground_reg_logs.get("Lground_plane_pure", 0.0),
                        "ground_reg/normal_pure": ground_reg_logs.get("Lground_normal_pure", 0.0),
                        "ground_reg/smoothness_pure": ground_reg_logs.get("Lground_smooth_pure", 0.0),
                        "ground_reg/plane_elements": ground_reg_logs.get("ground_plane_elements", 0.0),
                        "ground_reg/normal_elements": ground_reg_logs.get("ground_normal_elements", 0.0),
                        "ground_reg/smooth_elements": ground_reg_logs.get("ground_smooth_elements", 0.0),
                    },
                    step=iteration,
                )
            if wandb_run and assoc_stats is not None:
                tri_ids_ground = torch.nonzero(assoc_stats["is_ground_mask"], as_tuple=True)[0]
                ground_vertices = 0
                if tri_ids_ground.numel() > 0:
                    ground_vertices = int(torch.unique(triangles._triangle_indices[tri_ids_ground]).numel())
                wandb_run.log(
                    {
                        "ground_assoc/triangles_ground": float(assoc_stats["is_ground_mask"].sum().item()),
                        "ground_assoc/vertices_ground": float(ground_vertices),
                        "ground_assoc/triangles_boundary_uncertain": float(assoc_stats["boundary_uncertain_mask"].sum().item()),
                        "ground_assoc/triangles_unreliable": float((~assoc_stats["reliable_observation_mask"]).sum().item()),
                        "ground_assoc/support_ratio_mean": float(assoc_stats["ground_support_ratio"].mean().item()),
                        "ground_assoc/view_consistency_mean": float(assoc_stats["view_consistency"].mean().item()),
                    },
                    step=iteration,
                )
            
            training_report(
                tb_writer,
                wandb_run,
                scene_name,
                iteration,
                pixel_loss,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                wandb_image_log_interval=wandb_image_log_interval,
                wandb_fixed_views=wandb_fixed_views,
                wandb_fixed_view_indices=wandb_fixed_view_indices,
                wandb_disable_fixed_views=wandb_disable_fixed_views,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving model".format(iteration))
                scene.save(iteration)

            # Handle pruning operations
            if iteration % 500 == 0 and iteration < run_restricted_delaunay:
                pre_triangles = int(triangles._triangle_indices.shape[0])
                pre_vertices = int(triangles.vertices.shape[0])
                
                # Building masks to delete triangles
                triangle_vertex_weights = triangles.opacity_activation(
                    triangles.vertex_weight[triangles._triangle_indices]
                ) 
                min_weights = triangle_vertex_weights.min(dim=1).values

                mask_opacity     = (min_weights <= prune_triangles).squeeze()              # delete if too low
                mask_importance  = (triangles.importance_score <= prune_triangles).squeeze()  # delete if too low
                mask_size        = (triangles.image_size > prune_size).squeeze()                 # delete if too big

                delete_mask = mask_opacity | mask_size

                if number_of_training_views < 500: # only delete if the number of views are below 500. Otherwise, we might delete too much
                    delete_mask = delete_mask | mask_importance

                keep_mask   = ~delete_mask 

                if iteration > opt.start_pruning:
                    triangles.prune_triangles(keep_mask)
             
                # We prune vertices that are no longer used
                device = triangles.vertices.device
                used_vertex_mask = torch.zeros(triangles.vertices.shape[0], 
                                            dtype=torch.bool, 
                                            device=device)
                if triangles._triangle_indices.numel() > 0:
                    flat_indices = triangles._triangle_indices.flatten()
                    used_vertex_mask[flat_indices] = True
                
                weight_mask = (triangles.get_vertex_weight.squeeze() >= prune_triangles)
                mask_out = triangles.vertices.shape[0]
                vertex_mask = weight_mask[:mask_out] | used_vertex_mask

                triangles._prune_vertices(vertex_mask)


                triangle_vertex_weights = triangles.opacity_activation(
                    triangles.vertex_weight[triangles._triangle_indices]
                )  # [T,3]

                needs_densification = (iteration < opt.densify_until_iter and 
                                     iteration % opt.densification_interval == 0 and 
                                     iteration > opt.densify_from_iter)
                
                if needs_densification:
                    triangles.add_new_gs(iteration, cap_max=opt.max_points, splitt_large_triangles=splitt_large_triangles)
   

                if iteration > opt.start_opacity_floor:
                    start_iter = opt.start_opacity_floor
                    end_iter = total_iters_opacity  # the iteration where you want to reach final_opacity
                    a = min(1.0, max(0.0, (iteration - start_iter) / max(1, end_iter - start_iter)))
                    current_opacity = init_opacity + (final_opacity - init_opacity) * a
                    current_opacity = min(current_opacity, final_opacity)
                    triangles.update_min_weight(current_opacity)

                    prune_triangles += 0.01 
                    mask_out = triangles.vertices.shape[0]
                    triangle_vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]
                post_triangles = int(triangles._triangle_indices.shape[0])
                post_vertices = int(triangles.vertices.shape[0])
                if (post_triangles != pre_triangles) or (post_vertices != pre_vertices):
                    reset_ground_supervision_state("prune_or_densify")
            elif iteration == run_restricted_delaunay:
                need_delaunay = True
            elif iteration % 500 == 0 and iteration > run_restricted_delaunay + 1000:

                if iteration > opt.start_opacity_floor:
                    start_iter = opt.start_opacity_floor
                    end_iter = total_iters_opacity  # the iteration where you want to reach final_opacity
                    a = min(1.0, max(0.0, (iteration - start_iter) / max(1, end_iter - start_iter)))
                    current_opacity = init_opacity + (final_opacity - init_opacity) * a
                    current_opacity = min(current_opacity, final_opacity)
                    triangles.update_min_weight(current_opacity)

                    prune_triangles += 0.01 
                    mask_out = triangles.vertices.shape[0]
                    triangle_vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]
            

            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none = True)

    # cleaning of triangles that we do not need
    if ground_association_tracker is not None:
        ground_association_tracker.ensure_num_triangles(int(triangles._triangle_indices.shape[0]))
        ground_association_tracker.save_cache()
    viewpoint_stack = scene.getTrainCameras().copy()
    triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
    while viewpoint_stack:
        viewpoint_cam = viewpoint_stack.pop(0)
        render_pkg = render(viewpoint_cam, triangles, pipe, bg)

        importance_score = render_pkg["max_blending"].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]
    mask_importance  = (triangles.importance_score <= 0.5).squeeze() 
    triangles.prune_triangles(~mask_importance) # delete all the remaining triangles that do not have an influence

    device = triangles.vertices.device
    used_vertex_mask = torch.zeros(triangles.vertices.shape[0], 
                                dtype=torch.bool, 
                                device=device)
    if triangles._triangle_indices.numel() > 0:
        # Flatten indices and mark used vertices
        flat_indices = triangles._triangle_indices.flatten()
        used_vertex_mask[flat_indices] = True
    
    vertex_mask = used_vertex_mask
    triangles._prune_vertices(vertex_mask)

    scene.save(iteration)          
    if wandb_run:
        wandb_run.finish()
    print("Training is done")

def prepare_output_and_logger(args, wandb_cfg=None):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    wandb_run = None
    enable_wandb = bool(wandb_cfg.get("enable_wandb", False)) if isinstance(wandb_cfg, dict) else False
    if enable_wandb:
        if not WANDB_FOUND:
            print("WandB not available: install 'wandb' to enable online logging.")
        else:
            try:
                init_kwargs = {
                    "project": str(wandb_cfg.get("project", "mesh-splatting")),
                    "name": str(wandb_cfg.get("name", "mesh-splatting-run")),
                    "dir": args.model_path,
                    "config": {
                        "model_path": args.model_path,
                        "source_path": getattr(args, "source_path", ""),
                    },
                }
                entity = str(wandb_cfg.get("entity", "")).strip()
                if entity:
                    init_kwargs["entity"] = entity
                group = str(wandb_cfg.get("group", "")).strip()
                if group:
                    init_kwargs["group"] = group
                wandb_run = wandb.init(**init_kwargs)
                print(f"WandB enabled: project={init_kwargs['project']} run={init_kwargs['name']}")
            except Exception as e:
                print(f"WandB init failed: {e}")
                wandb_run = None
    return tb_writer, wandb_run

def _select_fixed_eval_cameras(scene: Scene, num_views: int = 5):
    test_cameras = scene.getTestCameras()
    source_cameras = test_cameras if test_cameras and len(test_cameras) > 0 else scene.getTrainCameras()
    if (not source_cameras) or len(source_cameras) == 0 or num_views <= 0:
        return []

    num_pick = min(int(num_views), len(source_cameras))
    # Fixed and deterministic subset for step-over-step comparison in WandB.
    indices = np.linspace(0, len(source_cameras) - 1, num=num_pick, dtype=int)
    return [source_cameras[int(i)] for i in indices]


def _parse_fixed_view_indices(index_spec: str):
    if not index_spec:
        return []
    values = []
    for token in str(index_spec).split(","):
        t = token.strip()
        if not t:
            continue
        try:
            values.append(int(t))
        except ValueError:
            continue
    return values


def _select_fixed_eval_cameras_with_indices(scene: Scene, num_views: int = 5, index_spec: str = ""):
    test_cameras = scene.getTestCameras()
    source_cameras = test_cameras if test_cameras and len(test_cameras) > 0 else scene.getTrainCameras()
    if (not source_cameras) or len(source_cameras) == 0 or num_views <= 0:
        return [], []

    explicit_indices = _parse_fixed_view_indices(index_spec)
    chosen_indices = []
    if explicit_indices:
        for idx in explicit_indices:
            if len(chosen_indices) >= int(num_views):
                break
            if 0 <= idx < len(source_cameras) and idx not in chosen_indices:
                chosen_indices.append(int(idx))
        if len(chosen_indices) < int(num_views):
            auto_indices = np.linspace(0, len(source_cameras) - 1, num=min(int(num_views), len(source_cameras)), dtype=int)
            for idx in auto_indices:
                i = int(idx)
                if i not in chosen_indices:
                    chosen_indices.append(i)
                if len(chosen_indices) >= int(num_views):
                    break
    else:
        auto_indices = np.linspace(0, len(source_cameras) - 1, num=min(int(num_views), len(source_cameras)), dtype=int)
        chosen_indices = [int(i) for i in auto_indices]

    cameras = [source_cameras[i] for i in chosen_indices]
    return cameras, chosen_indices


def _to_wandb_image(img_tensor, caption: str = ""):
    if not WANDB_FOUND:
        return None
    img = torch.clamp(img_tensor.detach(), 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    img_u8 = (img * 255.0).astype(np.uint8)
    return wandb.Image(img_u8, caption=caption)


def _prepare_ground_mask_hw(viewpoint, target_h: int, target_w: int, device):
    """Return binary ground mask [H,W] on device, resized if needed."""
    ground_mask = getattr(viewpoint, "ground_mask", None)
    if ground_mask is None:
        return None
    mask = ground_mask
    if not torch.is_tensor(mask):
        return None
    if mask.dim() != 2:
        return None
    mask = mask.to(device=device, dtype=torch.float32)
    if mask.shape[0] != target_h or mask.shape[1] != target_w:
        mask = F.interpolate(mask[None, None], size=(target_h, target_w), mode="nearest").squeeze(0).squeeze(0)
    return (mask > 0.5).float()


def _masked_ground_metrics(image, gt_image, ground_mask_hw):
    """
    Compute ground-only reconstruction metrics from image/gt [3,H,W] and mask [H,W].
    Returns None when mask has too few valid pixels.
    """
    if ground_mask_hw is None:
        return None
    valid = ground_mask_hw > 0.5
    valid_count = int(valid.sum().item())
    if valid_count < 16:
        return None

    mask3 = valid.unsqueeze(0).float()
    diff = torch.abs(image - gt_image)
    l1_ground = (diff * mask3).sum() / (mask3.sum() * image.shape[0] + 1e-8)

    mse_ground = (((image - gt_image) ** 2) * mask3).sum() / (mask3.sum() * image.shape[0] + 1e-8)
    psnr_ground = -10.0 * torch.log10(mse_ground + 1e-8)

    return {
        "l1_ground": float(l1_ground.item()),
        "psnr_ground": float(psnr_ground.item()),
        "ground_pixels": float(valid_count),
        "ground_ratio": float(valid.float().mean().item()),
    }


def training_report(
    tb_writer,
    wandb_run,
    scene_name,
    iteration,
    pixel_loss,
    loss,
    loss_fn,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    wandb_image_log_interval=1000,
    wandb_fixed_views=5,
    wandb_fixed_view_indices="",
    wandb_disable_fixed_views=False,
):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/pixel_loss', pixel_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    if wandb_run:
        wandb_run.log(
            {
                "train_loss/pixel_loss": float(pixel_loss.item()),
                "train_loss/total_loss": float(loss.item()),
                "train/iter_time_ms": float(elapsed),
            },
            step=iteration,
        )

    # Report test and samples of training set
    if iteration % 1000 == 0:
        torch.cuda.empty_cache()

    # Fixed reference WandB images for qualitative supervision (same 5 views every time).
    if (
        wandb_run
        and (not bool(wandb_disable_fixed_views))
        and int(wandb_image_log_interval) > 0
        and iteration % int(wandb_image_log_interval) == 0
    ):
        if not hasattr(scene, "_wandb_fixed_eval_cameras"):
            selected_cameras, selected_indices = _select_fixed_eval_cameras_with_indices(
                scene,
                int(wandb_fixed_views),
                str(wandb_fixed_view_indices),
            )
            scene._wandb_fixed_eval_cameras = selected_cameras
            scene._wandb_fixed_eval_indices = selected_indices
            selected_names = [getattr(cam, "image_name", f"cam_{i}") for i, cam in enumerate(selected_cameras)]
            print(
                "[WandB] fixed eval views: indices={} names={}".format(
                    selected_indices,
                    selected_names,
                )
            )
            try:
                wandb_run.summary["fixed_eval/indices"] = ",".join([str(i) for i in selected_indices])
                wandb_run.summary["fixed_eval/names"] = ",".join([str(n) for n in selected_names])
            except Exception:
                pass

        fixed_cameras = getattr(scene, "_wandb_fixed_eval_cameras", [])
        if fixed_cameras:
            render_images = []
            gt_images = []
            compare_images = []
            for cam_idx, camera in enumerate(fixed_cameras):
                render_img = torch.clamp(renderFunc(camera, scene.triangles, *renderArgs)["render"], 0.0, 1.0)
                gt_img = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
                diff_map = torch.abs(render_img - gt_img).mean(dim=0, keepdim=True).repeat(3, 1, 1)
                compare_img = torch.cat([gt_img, render_img, diff_map], dim=2)
                view_name = getattr(camera, "image_name", f"cam_{cam_idx}")
                wb_render = _to_wandb_image(render_img, caption=f"{cam_idx}: {view_name}")
                wb_gt = _to_wandb_image(gt_img, caption=f"{cam_idx}: {view_name}")
                wb_compare = _to_wandb_image(compare_img, caption=f"{cam_idx}: {view_name} | [GT|Render|AbsDiff]")
                if wb_render is not None:
                    render_images.append(wb_render)
                if wb_gt is not None:
                    gt_images.append(wb_gt)
                if wb_compare is not None:
                    compare_images.append(wb_compare)
            if render_images:
                wandb_run.log(
                    {
                        "fixed_eval/render": render_images,
                        "fixed_eval/gt": gt_images,
                        "fixed_eval/compare": compare_images,
                    },
                    step=iteration,
                )
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                pixel_loss_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                ground_l1_sum = 0.0
                ground_psnr_sum = 0.0
                ground_ratio_sum = 0.0
                ground_view_count = 0
                total_time = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    image = torch.clamp(renderFunc(viewpoint, scene.triangles, *renderArgs)["render"], 0.0, 1.0)
                    end_event.record()
                    torch.cuda.synchronize()
                    runtime = start_event.elapsed_time(end_event)
                    total_time += runtime

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    pixel_loss_test += loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()

                    # Ground-only metrics (masked by per-view ground segmentation).
                    ground_mask_hw = _prepare_ground_mask_hw(
                        viewpoint=viewpoint,
                        target_h=image.shape[1],
                        target_w=image.shape[2],
                        device=image.device,
                    )
                    ground_metrics = _masked_ground_metrics(image=image, gt_image=gt_image, ground_mask_hw=ground_mask_hw)
                    if ground_metrics is not None:
                        ground_l1_sum += float(ground_metrics["l1_ground"])
                        ground_psnr_sum += float(ground_metrics["psnr_ground"])
                        ground_ratio_sum += float(ground_metrics["ground_ratio"])
                        ground_view_count += 1
                psnr_test /= len(config['cameras'])
                pixel_loss_test /= len(config['cameras'])       
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])  
                total_time /= len(config['cameras'])
                fps = 1000.0 / total_time
                ground_l1_avg = (ground_l1_sum / ground_view_count) if ground_view_count > 0 else None
                ground_psnr_avg = (ground_psnr_sum / ground_view_count) if ground_view_count > 0 else None
                ground_ratio_avg = (ground_ratio_sum / ground_view_count) if ground_view_count > 0 else None
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} FPS {}".format(iteration, config['name'], pixel_loss_test, psnr_test, ssim_test, lpips_test, fps))
                if ground_view_count > 0:
                    print(
                        "[ITER {}] Ground-only {}: L1 {:.6f} PSNR {:.4f} (views={} mask_ratio={:.4f})".format(
                            iteration,
                            config['name'],
                            ground_l1_avg,
                            ground_psnr_avg,
                            ground_view_count,
                            ground_ratio_avg,
                        )
                    )

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    if ground_view_count > 0:
                        tb_writer.add_scalar(config['name'] + '/ground_only - l1_loss', float(ground_l1_avg), iteration)
                        tb_writer.add_scalar(config['name'] + '/ground_only - psnr', float(ground_psnr_avg), iteration)
                        tb_writer.add_scalar(config['name'] + '/ground_only - mask_ratio', float(ground_ratio_avg), iteration)
                if wandb_run:
                    wandb_payload = {
                        f"{config['name']}/l1": float(pixel_loss_test),
                        f"{config['name']}/psnr": float(psnr_test),
                        f"{config['name']}/ssim": float(ssim_test),
                        f"{config['name']}/lpips": float(lpips_test),
                        f"{config['name']}/fps": float(fps),
                    }
                    if ground_view_count > 0:
                        wandb_payload.update(
                            {
                                f"{config['name']}/ground_l1": float(ground_l1_avg),
                                f"{config['name']}/ground_psnr": float(ground_psnr_avg),
                                f"{config['name']}/ground_mask_ratio": float(ground_ratio_avg),
                                f"{config['name']}/ground_views_used": float(ground_view_count),
                            }
                        )
                    wandb_run.log(wandb_payload, step=iteration)

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_iteration", type=int, default=None)

    parser.add_argument("--enable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", default="mesh-splatting", type=str)
    parser.add_argument("--wandb_entity", default="", type=str)
    parser.add_argument("--wandb_group", default="", type=str)
    parser.add_argument('--wandb_name', default="Test", type=str)
    parser.add_argument("--wandb_image_log_interval", type=int, default=1000)
    parser.add_argument("--wandb_fixed_views", type=int, default=5)
    parser.add_argument("--wandb_fixed_view_indices", type=str, default="")
    parser.add_argument("--wandb_disable_fixed_views", action="store_true", default=False)
    parser.add_argument('--scene_name', default="Garden", type=str)
    parser.add_argument("--use_sparse_adam", action="store_true", default=True)
    parser.add_argument("--indoor", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    lps = lp.extract(args)
    ops = op.extract(args)
    pps = pp.extract(args)

    if args.indoor:
        ops = update_indoor(ops)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lps,
             ops,
             pps,
             args.test_iterations,
             args.save_iterations,
             args.start_checkpoint,
             args.debug_from,
             args.scene_name,
             args.load_iteration,
             use_sparse_adam=args.use_sparse_adam,
             wandb_cfg={
                 "enable_wandb": bool(args.enable_wandb),
                 "project": str(args.wandb_project),
                 "entity": str(args.wandb_entity),
                 "group": str(args.wandb_group),
                 "name": str(args.wandb_name),
             },
             wandb_image_log_interval=int(args.wandb_image_log_interval),
             wandb_fixed_views=int(args.wandb_fixed_views),
             wandb_fixed_view_indices=str(args.wandb_fixed_view_indices),
             wandb_disable_fixed_views=bool(args.wandb_disable_fixed_views),
             )
    
    # All done
    print("\nTraining complete.")