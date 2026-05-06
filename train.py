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
import copy
import torch
import numbers
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
import json
from types import SimpleNamespace
from typing import Optional
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
from utils.triangle_stats import TriangleStatsManager
from utils.triangle_structure_utils import (
    compute_triangle_structure_metrics,
)
from utils.triangle_sparse_support import SparseSupportConfig, TriangleSparseSupportEstimator
from utils.prism_scoring import (
    PrismScoreConfig,
    PrismScoreInputs,
    compute_prism_scores,
    summarize_prism_scores,
)
from utils.prism_counterfactual import (
    CalibrationConfig,
    CompactionSelectionConfig,
    CounterfactualGateConfig,
    build_calibration_set,
    counterfactual_decision_to_dict,
    run_counterfactual_simulation,
    select_prism_compaction_microbatch_ids,
    select_prism_candidate_ids,
)
from utils.prism_geometry_proxy import (
    GeometryProxyConfig,
    build_geometry_proxy_context,
    collect_view_sparse_depth_correspondences,
    normalize_image_key,
)
from utils.sparse_depth_parent_rollback import (
    compute_sparse_depth_parent_rollback_loss,
    load_sparse_depth_parent_rollback_cache,
    sparse_depth_parent_rollback_lambda,
)
from utils.prism_pipeline import (
    PrismCompactionConfig,
    PrismCompactionPhase,
    PrismPipelineConfig,
    PrismRoundController,
    PrismPhase,
)
from utils.prism_adaptive_policy import (
    AdaptiveCSEFPolicyConfig,
    decide_adaptive_csef_policy,
    update_adaptive_csef_policy_after_prune,
)
from utils.prism_validation import (
    PrismValidationConfig,
    build_prism_validation_views,
    decide_stage_best_update,
    evaluate_prism_validation_metrics,
    compare_validation_against_stage_best,
    save_validation_summary,
)
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
from scene.dataset_readers import sceneLoadTypeCallbacks



def _prepare_prism_state(opt, scene, triangles, init_iter, dataset=None, pipe=None, background=None):
    """
    PRISM-Prune neutral scaffolding.
    This function only captures config and runtime slots; it does not alter training behavior.
    """
    cfg = {
        "enabled": bool(getattr(opt, "enable_prism_pruning", False)),
        "collect_stats": bool(getattr(opt, "prism_collect_stats", False)),
        "stats_warmup_iters": int(getattr(opt, "prism_stats_warmup_iters", 0)),
        "collect_interval": int(getattr(opt, "prism_collect_interval", 0)),
        "score_recompute_interval": int(getattr(opt, "prism_score_recompute_interval", 500)),
        "max_triangles_for_heavy_metrics": int(getattr(opt, "prism_max_triangles_for_heavy_metrics", 400000)),
        "heavy_eval_budget": int(getattr(opt, "prism_heavy_eval_budget", 120000)),
        "heavy_eval_neighbor_rings": int(getattr(opt, "prism_heavy_eval_neighbor_rings", 2)),
        "force_full_heavy_eval_below": int(getattr(opt, "prism_force_full_heavy_eval_below", 400000)),
        "skip_heavy_eval_for_far_field": bool(getattr(opt, "prism_skip_heavy_eval_for_far_field", False)),
        "dead_prune_ratio": float(getattr(opt, "prism_dead_prune_ratio", 0.0)),
        "candidate_prune_ratio": float(getattr(opt, "prism_candidate_prune_ratio", 0.0)),
        "candidate_prune_ratio_per_round": float(getattr(opt, "prism_candidate_prune_ratio_per_round", 0.015)),
        "candidate_max_count_per_round": int(getattr(opt, "prism_candidate_max_count_per_round", 0)),
        "candidate_microbatch_gate": bool(getattr(opt, "prism_candidate_microbatch_gate", False)),
        "candidate_microbatch_size": int(getattr(opt, "prism_candidate_microbatch_size", 256)),
        "candidate_microbatch_max_batches": int(getattr(opt, "prism_candidate_microbatch_max_batches", 0)),
        "candidate_quality_rank": bool(getattr(opt, "prism_candidate_quality_rank", False)),
        "candidate_quality_prune_weight": float(getattr(opt, "prism_candidate_quality_prune_weight", 1.0)),
        "candidate_quality_render_penalty": float(getattr(opt, "prism_candidate_quality_render_penalty", 0.5)),
        "candidate_quality_geometry_penalty": float(getattr(opt, "prism_candidate_quality_geometry_penalty", 0.5)),
        "candidate_quality_orientation_penalty": float(getattr(opt, "prism_candidate_quality_orientation_penalty", 0.25)),
        "candidate_quality_utility_penalty": float(getattr(opt, "prism_candidate_quality_utility_penalty", 0.25)),
        "candidate_quality_uncertainty_penalty": float(getattr(opt, "prism_candidate_quality_uncertainty_penalty", 0.25)),
        "candidate_measured_impact_rank": bool(getattr(opt, "prism_candidate_measured_impact_rank", False)),
        "candidate_measured_pool_multiplier": float(getattr(opt, "prism_candidate_measured_pool_multiplier", 4.0)),
        "candidate_measured_group_size": int(getattr(opt, "prism_candidate_measured_group_size", 256)),
        "candidate_measured_max_groups": int(getattr(opt, "prism_candidate_measured_max_groups", 8)),
        "post_commit_candidate_refresh": bool(getattr(opt, "prism_post_commit_candidate_refresh", False)),
        "post_commit_refresh_min_prune_score": float(getattr(opt, "prism_post_commit_refresh_min_prune_score", 1e-6)),
        "post_commit_relaxed_max_commits": int(getattr(opt, "prism_post_commit_relaxed_max_commits", 0)),
        "post_commit_relaxed_strict_gate": bool(getattr(opt, "prism_post_commit_relaxed_strict_gate", False)),
        "post_commit_relaxed_min_delta_psnr": float(getattr(opt, "prism_post_commit_relaxed_min_delta_psnr", 0.0)),
        "post_commit_relaxed_max_delta_mae": float(getattr(opt, "prism_post_commit_relaxed_max_delta_mae", 0.0)),
        "post_commit_relaxed_max_delta_absrel": float(getattr(opt, "prism_post_commit_relaxed_max_delta_absrel", 0.0)),
        "post_commit_relaxed_max_delta_mean_angle": float(getattr(opt, "prism_post_commit_relaxed_max_delta_mean_angle", 0.0)),
        "post_commit_relaxed_max_changed_pixel_ratio": float(
            getattr(opt, "prism_post_commit_relaxed_max_changed_pixel_ratio", 0.0025)
        ),
        "adaptive_candidate_retry_on_rollback": bool(
            getattr(opt, "prism_adaptive_candidate_retry_on_rollback", False)
        ),
        "adaptive_candidate_ratio_decay": float(getattr(opt, "prism_adaptive_candidate_ratio_decay", 0.5)),
        "adaptive_candidate_min_ratio": float(getattr(opt, "prism_adaptive_candidate_min_ratio", 0.0025)),
        "adaptive_candidate_max_rollback_retries": int(
            getattr(opt, "prism_adaptive_candidate_max_rollback_retries", 3)
        ),
        "enable_adaptive_csef_policy": bool(getattr(opt, "prism_enable_adaptive_csef_policy", False)),
        "recovery_iters": int(getattr(opt, "prism_recovery_iters", 0)),
        "use_counterfactual_gate": bool(getattr(opt, "prism_use_counterfactual_gate", False)),
        "use_ground_protect": bool(getattr(opt, "prism_use_ground_protect", False)),
        "use_roi_protect": bool(getattr(opt, "prism_use_roi_protect", False)),
        "calib_num_hard_train_views": int(getattr(opt, "prism_calib_num_hard_train_views", 0)),
        "calib_num_buffer_views": int(getattr(opt, "prism_calib_num_buffer_views", 8)),
        "calib_diverse_views": bool(getattr(opt, "prism_calib_diverse_views", False)),
        "save_debug_json": bool(getattr(opt, "prism_save_debug_json", False)),
        "disable_final_cleanup_prune": bool(getattr(opt, "prism_disable_final_cleanup_prune", True)),
        "save_pre_cleanup_checkpoint": bool(getattr(opt, "prism_save_pre_cleanup_checkpoint", True)),
        "post_commit_recollect_iters": int(getattr(opt, "prism_post_commit_recollect_iters", 300)),
        "freeze_densification_after_first_commit": bool(
            getattr(opt, "prism_freeze_densification_after_first_commit", False)
        ),
        "force_recompute_scores_after_recollect": bool(
            getattr(opt, "prism_force_recompute_scores_after_recollect", True)
        ),
        "enable_compaction_stage": bool(getattr(opt, "prism_enable_compaction_stage", False)),
    }
    collect_enabled = bool(cfg["enabled"] or cfg["collect_stats"])
    manager = None
    if collect_enabled:
        manager = TriangleStatsManager(
            num_triangles=int(triangles._triangle_indices.shape[0]),
            device=triangles.vertices.device,
            init_iter=int(init_iter),
            ema_decay=0.95,
            view_hist_bins=8,
        )

    score_cfg = PrismScoreConfig(
        utility_w_vis=float(getattr(opt, "prism_utility_w_vis", 0.30)),
        utility_w_sens=float(getattr(opt, "prism_utility_w_sens", 0.25)),
        utility_w_geo=float(getattr(opt, "prism_utility_w_geo", 0.20)),
        utility_w_viewdiv=float(getattr(opt, "prism_utility_w_viewdiv", 0.15)),
        utility_w_edge=float(getattr(opt, "prism_utility_w_edge", 0.10)),
        redund_w_flat=float(getattr(opt, "prism_redund_w_flat", 0.70)),
        redund_w_coplanar=float(getattr(opt, "prism_redund_w_coplanar", 0.30)),
        norm_percentile_low=float(getattr(opt, "prism_norm_percentile_low", 5.0)),
        norm_percentile_high=float(getattr(opt, "prism_norm_percentile_high", 95.0)),
        norm_eps=float(getattr(opt, "prism_norm_eps", 1e-6)),
        thresh_protected_edge=float(getattr(opt, "prism_thresh_protected_edge", 0.60)),
        thresh_protected_geo=float(getattr(opt, "prism_thresh_protected_geo", 0.75)),
        thresh_protected_sens=float(getattr(opt, "prism_thresh_protected_sens", 0.80)),
        thresh_protected_unc=float(getattr(opt, "prism_thresh_protected_unc", 0.65)),
        thresh_dead_vis=float(getattr(opt, "prism_thresh_dead_vis", 0.02)),
        thresh_dead_sens=float(getattr(opt, "prism_thresh_dead_sens", 0.03)),
        thresh_dead_geo=float(getattr(opt, "prism_thresh_dead_geo", 0.05)),
        thresh_dead_edge=float(getattr(opt, "prism_thresh_dead_edge", 0.10)),
        thresh_suspicious_vis=float(getattr(opt, "prism_thresh_suspicious_vis", 0.05)),
        thresh_suspicious_geo=float(getattr(opt, "prism_thresh_suspicious_geo", 0.15)),
        thresh_suspicious_unc=float(getattr(opt, "prism_thresh_suspicious_unc", 0.50)),
        boundary_risk_value=float(getattr(opt, "prism_boundary_risk_value", 1.0)),
        nonmanifold_risk_value=float(getattr(opt, "prism_nonmanifold_risk_value", 1.0)),
        recent_age_iters=int(getattr(opt, "prism_recent_age_iters", 500)),
        ground_protect_bonus=float(getattr(opt, "prism_ground_protect_bonus", 1.0)),
        roi_protect_bonus=float(getattr(opt, "prism_roi_protect_bonus", 1.0)),
        use_ground_protect=bool(getattr(opt, "prism_use_ground_protect", False)),
        use_roi_protect=bool(getattr(opt, "prism_use_roi_protect", False)),
        keep_support_count_min=float(getattr(opt, "prism_keep_support_count_min", 12.0)),
        keep_plane_residual_max=float(getattr(opt, "prism_keep_plane_residual_max", 0.02)),
        keep_normal_residual_max_deg=float(getattr(opt, "prism_keep_normal_residual_max_deg", 15.0)),
        keep_orientation_dihedral_min_deg=float(getattr(opt, "prism_keep_orientation_dihedral_min_deg", 12.0)),
        keep_orientation_local_var_min=float(getattr(opt, "prism_keep_orientation_local_var_min", 0.25)),
        keep_geometry_threshold=float(getattr(opt, "prism_keep_geometry_threshold", 0.6)),
        keep_orientation_threshold=float(getattr(opt, "prism_keep_orientation_threshold", 0.6)),
        keep_render_threshold=float(getattr(opt, "prism_keep_render_threshold", 0.6)),
        keep_geometry_bonus=float(getattr(opt, "prism_keep_geometry_bonus", 1.0)),
        keep_orientation_bonus=float(getattr(opt, "prism_keep_orientation_bonus", 1.0)),
        keep_render_bonus=float(getattr(opt, "prism_keep_render_bonus", 1.0)),
        candidate_block_geometry_keep_threshold=float(
            getattr(opt, "prism_candidate_block_geometry_keep_threshold", 0.6)
        ),
        protected_dilation_rings=int(getattr(opt, "prism_protected_dilation_rings", 1)),
    )

    sparse_cfg = SparseSupportConfig(
        radius=-1.0,
        radius_factor=0.02,
        knn=32,
        min_support_points=6,
        pca_min_points=10,
        max_point_error=float(getattr(opt, "ground_plane_colmap_error_max", 2.0)),
    )
    sparse_estimator = None
    calibration_views = []
    proxy_cfg = GeometryProxyConfig(
        max_points_per_view=int(getattr(opt, "prism_proxy_max_points_per_view", 3000)),
        point_error_max=float(getattr(opt, "prism_proxy_point_error_max", 2.0)),
        normal_knn=int(getattr(opt, "prism_proxy_normal_knn", 24)),
        compute_normal=bool(int(getattr(opt, "prism_gate_min_valid_normal_matches", 64)) > 0),
        seed=7,
    )
    cam_infos = []
    cam_infos.extend(list(getattr(scene.scene_info, "train_cameras", []) or []))
    cam_infos.extend(list(getattr(scene.scene_info, "test_cameras", []) or []))
    if dataset is not None and os.path.exists(os.path.join(dataset.source_path, "sparse")):
        try:
            all_info = sceneLoadTypeCallbacks["Colmap"](
                dataset.source_path,
                dataset.images,
                False,
                split_strategy="llff",
                split_file="",
            )
            cam_infos.extend(list(getattr(all_info, "train_cameras", []) or []))
        except Exception:
            pass
    proxy_ctx = build_geometry_proxy_context(
        colmap_points3d=getattr(scene.scene_info, "colmap_points3d", None),
        cam_infos=cam_infos,
        cfg=proxy_cfg,
    )
    gate_cfg = CounterfactualGateConfig(
        min_delta_psnr_db=float(getattr(opt, "prism_gate_min_delta_psnr_db", -0.05)),
        max_delta_mae=float(getattr(opt, "prism_gate_max_delta_mae", 0.002)),
        max_delta_absrel=float(getattr(opt, "prism_gate_max_delta_absrel", 0.0008)),
        max_baseline_absrel_for_absrel_check=float(
            getattr(opt, "prism_gate_max_baseline_absrel_for_absrel_check", float("inf"))
        ),
        max_delta_mean_angle_deg=float(getattr(opt, "prism_gate_max_delta_mean_angle_deg", 0.3)),
        max_changed_pixel_ratio=float(getattr(opt, "prism_gate_max_changed_pixel_ratio", 0.005)),
        changed_pixel_threshold=float(getattr(opt, "prism_changed_pixel_threshold", 0.02)),
        min_valid_depth_matches=int(getattr(opt, "prism_gate_min_valid_depth_matches", 128)),
        min_valid_normal_matches=int(getattr(opt, "prism_gate_min_valid_normal_matches", 64)),
    )
    if collect_enabled:
        sparse_estimator = TriangleSparseSupportEstimator.from_scene(scene=scene, cfg=sparse_cfg)
        if dataset is not None and pipe is not None and background is not None:
            calib_cfg = CalibrationConfig(
                num_buffer_views=int(getattr(opt, "prism_calib_num_buffer_views", 8)),
                num_hard_train_views=int(getattr(opt, "prism_calib_num_hard_train_views", 8)),
                hard_view_pool_size=int(getattr(opt, "prism_calib_hard_pool_size", 64)),
                prefer_observable_views=bool(getattr(opt, "prism_calib_prefer_observable_views", True)),
                min_depth_matches_per_view=int(getattr(opt, "prism_calib_min_depth_matches_per_view", 24)),
                min_normal_matches_per_view=int(getattr(opt, "prism_calib_min_normal_matches_per_view", 8)),
                diverse_views=bool(getattr(opt, "prism_calib_diverse_views", False)),
                num_diverse_test_views=int(getattr(opt, "prism_calib_diverse_test_views", 0)),
                num_diverse_train_views=int(getattr(opt, "prism_calib_diverse_train_views", 0)),
            )
            calibration_views = build_calibration_set(
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
            if bool(getattr(opt, "prism_save_debug_json", False)):
                debug_dir = os.path.join(scene.model_path, "prism_debug")
                os.makedirs(debug_dir, exist_ok=True)
                manifest = getattr(calib_cfg, "_last_manifest", [])
                with open(os.path.join(debug_dir, "calibration_views.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "diverse_views": bool(getattr(opt, "prism_calib_diverse_views", False)),
                            "num_views": int(len(calibration_views)),
                            "views": manifest,
                        },
                        f,
                        indent=2,
                    )
    geom_acq_until = int(getattr(opt, "prism_geometry_acq_until_iter", -1))
    if geom_acq_until < 0:
        # Reuse existing staging landmarks: densification + restricted-delaunay settling.
        geom_acq_until = int(getattr(opt, "densify_until_iter", 0)) + 1000
    pipeline_cfg = PrismPipelineConfig(
        enabled=bool(cfg["enabled"]),
        geometry_acq_until_iter=geom_acq_until,
        stats_collection_iters=int(getattr(opt, "prism_stats_collection_iters", 500)),
        dead_rounds=int(getattr(opt, "prism_dead_rounds", 1)),
        candidate_rounds=int(getattr(opt, "prism_candidate_rounds", 3)),
        recovery_iters=int(getattr(opt, "prism_recovery_iters", 400)),
        post_commit_recollect_iters=int(getattr(opt, "prism_post_commit_recollect_iters", 300)),
        force_recompute_scores_after_recollect=bool(
            getattr(opt, "prism_force_recompute_scores_after_recollect", True)
        ),
        final_finetune_iters=int(getattr(opt, "prism_final_finetune_iters", 500)),
        topology_freeze_during_stats=bool(getattr(opt, "prism_topology_freeze_during_stats", True)),
        round_checkpoint=bool(getattr(opt, "prism_round_checkpoint", True)),
    )
    controller = PrismRoundController(
        cfg=pipeline_cfg,
        first_iter=int(init_iter),
        total_iters=int(getattr(opt, "iterations", 0)),
    )
    compaction_cfg = PrismCompactionConfig(
        enabled=bool(getattr(opt, "prism_enable_compaction_stage", False)),
        source_preference=str(getattr(opt, "prism_compaction_source_preference", "best_geometry")),
        rounds=int(getattr(opt, "prism_compaction_rounds", 2)),
        microbatch_active_ratio=float(getattr(opt, "prism_compaction_microbatch_active_ratio", 0.0035)),
        max_microbatches_per_round=int(getattr(opt, "prism_compaction_max_microbatches_per_round", 6)),
        candidate_pool_multiplier=float(getattr(opt, "prism_compaction_candidate_pool_multiplier", 6.0)),
        min_prune_count=int(getattr(opt, "prism_compaction_min_prune_count", 256)),
        roi_budget_fraction=float(getattr(opt, "prism_compaction_roi_budget_fraction", 0.10)),
        near_field_budget_fraction=float(getattr(opt, "prism_compaction_near_field_budget_fraction", 0.25)),
        roi_signal_threshold=float(getattr(opt, "prism_compaction_roi_signal_threshold", 0.05)),
        near_field_area_percentile=float(getattr(opt, "prism_compaction_near_field_area_percentile", 80.0)),
        save_checkpoints=True,
    )
    validation_cfg = PrismValidationConfig(
        interval=int(getattr(opt, "prism_validation_interval", 1000)),
        max_views=int(getattr(opt, "prism_validation_max_views", 32)),
        num_buffer_views=int(getattr(opt, "prism_validation_num_buffer_views", 16)),
        num_train_views=int(getattr(opt, "prism_validation_num_train_views", 16)),
        train_pool_size=int(getattr(opt, "prism_validation_train_pool_size", 128)),
        prefer_observable_train_views=bool(getattr(opt, "prism_validation_prefer_observable_train_views", True)),
        min_depth_matches_per_view=int(getattr(opt, "prism_validation_min_depth_matches_per_view", 24)),
        min_normal_matches_per_view=int(getattr(opt, "prism_validation_min_normal_matches_per_view", 8)),
        min_valid_depth_matches=int(getattr(opt, "prism_validation_min_valid_depth_matches", 128)),
        min_valid_normal_matches=int(getattr(opt, "prism_validation_min_valid_normal_matches", 64)),
        absrel_rel_degrade_thresh=float(getattr(opt, "prism_rollback_absrel_rel_thresh", 0.01)),
        mean_angle_degrade_thresh_deg=float(getattr(opt, "prism_rollback_mean_angle_thresh", 0.4)),
        psnr_drop_thresh_db=float(getattr(opt, "prism_rollback_psnr_drop_thresh", 0.10)),
        mae_increase_thresh=float(getattr(opt, "prism_rollback_mae_increase_thresh", 0.003)),
    )
    validation_views = []
    if collect_enabled and dataset is not None:
        validation_views = build_prism_validation_views(
            scene=scene,
            dataset=dataset,
            cfg=validation_cfg,
            proxy_ctx=proxy_ctx,
            proxy_cfg=proxy_cfg,
        )

    adaptive_policy_cfg = AdaptiveCSEFPolicyConfig(
        enabled=bool(getattr(opt, "prism_enable_adaptive_csef_policy", False)),
        min_ratio=float(getattr(opt, "prism_adaptive_policy_min_ratio", 0.006)),
        max_ratio=float(getattr(opt, "prism_adaptive_policy_max_ratio", 0.020)),
        initial_ratio=float(getattr(opt, "prism_adaptive_policy_initial_ratio", 0.012)),
        target_accept_margin=float(getattr(opt, "prism_adaptive_policy_target_accept_margin", 0.55)),
        rollback_decay=float(getattr(opt, "prism_adaptive_policy_rollback_decay", 0.55)),
        accept_growth=float(getattr(opt, "prism_adaptive_policy_accept_growth", 1.18)),
        no_candidate_decay=float(getattr(opt, "prism_adaptive_policy_no_candidate_decay", 0.75)),
        cooldown_iters=int(getattr(opt, "prism_adaptive_policy_cooldown_iters", 20)),
        max_candidate_count=int(getattr(opt, "prism_adaptive_policy_max_candidate_count", 0)),
        min_candidate_count=int(getattr(opt, "prism_adaptive_policy_min_candidate_count", 512)),
        depth_degrade_absrel=float(getattr(opt, "prism_adaptive_policy_depth_degrade_absrel", 0.004)),
        normal_degrade_deg=float(getattr(opt, "prism_adaptive_policy_normal_degrade_deg", 0.10)),
        render_degrade_psnr=float(getattr(opt, "prism_adaptive_policy_render_degrade_psnr", -0.05)),
        uncertainty_high=float(getattr(opt, "prism_adaptive_policy_uncertainty_high", 0.35)),
        geometry_keep_high=float(getattr(opt, "prism_adaptive_policy_geometry_keep_high", 0.04)),
        orientation_keep_high=float(getattr(opt, "prism_adaptive_policy_orientation_keep_high", 0.04)),
        reliable_absrel_max=float(getattr(opt, "prism_adaptive_policy_reliable_absrel_max", 2.0)),
        strict_gate_after_rejects=int(getattr(opt, "prism_adaptive_policy_strict_gate_after_rejects", 1)),
        normal_repair_penalty_boost=float(getattr(opt, "prism_adaptive_policy_normal_repair_penalty_boost", 0.8)),
        geometry_repair_penalty_boost=float(getattr(opt, "prism_adaptive_policy_geometry_repair_penalty_boost", 0.8)),
        uncertainty_penalty_boost=float(getattr(opt, "prism_adaptive_policy_uncertainty_penalty_boost", 0.6)),
        cold_start_rounds=int(getattr(opt, "prism_adaptive_policy_cold_start_rounds", 1)),
        cold_start_gate_scale=float(getattr(opt, "prism_adaptive_policy_cold_start_gate_scale", 0.70)),
        cold_start_ratio_damping=float(getattr(opt, "prism_adaptive_policy_cold_start_ratio_damping", 0.96)),
        cold_start_quality_rank=bool(getattr(opt, "prism_adaptive_policy_cold_start_quality_rank", False)),
        enable_measured_rank=bool(getattr(opt, "prism_adaptive_policy_enable_measured_rank", True)),
        enable_microbatch_gate=bool(getattr(opt, "prism_adaptive_policy_enable_microbatch_gate", True)),
        microbatch_size=int(getattr(opt, "prism_adaptive_policy_microbatch_size", 512)),
        microbatch_max_batches=int(getattr(opt, "prism_adaptive_policy_microbatch_max_batches", 0)),
    )

    return {
        "cfg": cfg,
        "collect_enabled": collect_enabled,
        "manager": manager,
        "score_cfg": score_cfg,
        "sparse_estimator": sparse_estimator,
        "calibration_views": calibration_views,
        "gate_cfg": gate_cfg,
        "structure_cache": None,
        "last_scores_summary": None,
        "last_scores": None,
        "last_counterfactual_decision": None,
        "model_path": scene.model_path,
        "last_collect_iter": -1,
        "last_score_recompute_iter": -1,
        "last_score_triangle_count": -1,
        "cached_structure_metrics": None,
        "cached_sparse_metrics": None,
        "pipe": pipe,
        "background": background,
        "controller": controller,
        "opt": opt,
        "current_phase": PrismPhase.FINAL_FINE_TUNE,
        "pruned_this_round": 0,
        "counterfactual_accept": 0,
        "rollback": 0,
        "rollback_by_validation": 0,
        "adaptive_candidate_prune_ratio": float(getattr(opt, "prism_candidate_prune_ratio_per_round", 0.015)),
        "adaptive_candidate_rollback_retries": 0,
        "last_candidate_pool_count": 0,
        "last_candidate_target_count": 0,
        "last_candidate_cap_count": 0,
        "last_candidate_selected_count": 0,
        "last_candidate_microbatch_count": 0,
        "last_candidate_microbatch_accepted_count": 0,
        "last_candidate_microbatch_rejected_count": 0,
        "last_candidate_microbatch_accepted_triangles": 0,
        "last_candidate_quality_rank_enabled": 0,
        "last_candidate_quality_score_mean": 0.0,
        "last_candidate_prune_score_mean": 0.0,
        "last_candidate_render_keep_mean": 0.0,
        "last_candidate_geometry_keep_mean": 0.0,
        "last_candidate_orientation_keep_mean": 0.0,
        "last_candidate_utility_mean": 0.0,
        "last_candidate_uncertainty_mean": 0.0,
        "last_candidate_measured_rank_enabled": 0,
        "last_candidate_measured_group_count": 0,
        "last_candidate_measured_accepted_count": 0,
        "last_candidate_measured_selected_count": 0,
        "last_candidate_measured_best_score": 0.0,
        "last_candidate_relaxed_refresh_used": 0,
        "last_candidate_relaxed_pool_count": 0,
        "last_candidate_relaxed_reject_reason": "",
        "last_candidate_relaxed_strict_gate_pass": 1,
        "last_candidate_relaxed_strict_gate_reason": "",
        "candidate_commit_count": 0,
        "relaxed_candidate_commit_count": 0,
        "relaxed_commit_records": [],
        "last_commit_relaxed_refresh_used": 0,
        "teacher_cache": {},
        "teacher_rgb_lambda": float(getattr(opt, "prism_teacher_rgb_lambda", 0.01)),
        "teacher_depth_lambda": float(getattr(opt, "prism_teacher_depth_lambda", 0.002)),
        "enable_teacher_rgb_distill": bool(getattr(opt, "prism_enable_teacher_rgb_distill", False)),
        "enable_teacher_depth_distill": bool(getattr(opt, "prism_enable_teacher_depth_distill", False)),
        "teacher_num_views": int(getattr(opt, "prism_teacher_num_views", 8)),
        "validation_cfg": validation_cfg,
        "validation_views": validation_views,
        "proxy_cfg": proxy_cfg,
        "proxy_ctx": proxy_ctx,
        "compaction_cfg": compaction_cfg,
        "adaptive_policy_cfg": adaptive_policy_cfg,
        "adaptive_csef_policy": {"ratio": float(adaptive_policy_cfg.initial_ratio)},
        "stage_best_checkpoint_dir": "",
        "last_validation_pass_checkpoint_dir": "",
        "compaction_source_checkpoint_dir": "",
        "compaction_best_geometry_checkpoint_dir": "",
        "compaction_best_speed_checkpoint_dir": "",
        "compaction_final_checkpoint_dir": "",
        "compaction_phase": PrismCompactionPhase.DISABLED,
        "stage_best_metrics": None,
        "last_validation_metrics": None,
        "last_validation_pass": 1,
        "last_validation_rules": [],
        "last_validation_deltas": {},
        "round_snapshot": None,
        "last_topology_change_iter": int(init_iter),
        "last_topology_change_reason": "init",
        "densification_frozen_after_prism_commit": 0,
        "densification_freeze_iter": -1,
        "final_cleanup_enabled": 0,
        "final_cleanup_pruned": 0,
        "pre_cleanup_checkpoint": "",
        "latest_assoc_stats": None,
    }


def _update_prism_state(prism_state, iteration, render_pkg, viewpoint_cam, triangles):
    """
    PRISM neutral update hook for per-view visibility-style stats.
    No pruning decision is performed here.
    """
    manager = prism_state.get("manager", None)
    if manager is None:
        return
    cfg = prism_state["cfg"]
    if int(iteration) < int(cfg["stats_warmup_iters"]):
        return
    interval = max(1, int(cfg["collect_interval"]))
    if int(iteration) % interval != 0:
        return
    manager.update_visibility_from_render(
        render_pkg=render_pkg,
        triangles=triangles,
        viewpoint_cam=viewpoint_cam,
        iteration=int(iteration),
    )
    prism_state["last_collect_iter"] = int(iteration)
    if bool(cfg["save_debug_json"]):
        manager.maybe_save_debug_json(
            output_dir=os.path.join(prism_state["model_path"], "prism_debug"),
            iteration=int(iteration),
        )
    return


def _update_prism_gradient_state(prism_state, triangles, iteration: int):
    """
    PRISM neutral update hook for gradient stats.
    This is called after backward; no pruning decision is performed here.
    """
    manager = prism_state.get("manager", None)
    if manager is None:
        return
    manager.update_gradient_stats(triangles=triangles, iteration=int(iteration))


def _safe_resize_signal(x: torch.Tensor, n: int, device) -> torch.Tensor:
    x = x.to(device=device, dtype=torch.float32)
    if x.numel() == n:
        return x
    if x.numel() > n:
        return x[:n]
    out = torch.zeros((n,), dtype=torch.float32, device=device)
    out[: x.numel()] = x
    return out


def _build_prism_ground_and_roi_signals(
    prism_state,
    triangles,
    render_pkg=None,
    viewpoint_cam=None,
    assoc_stats=None,
):
    n = int(triangles._triangle_indices.shape[0])
    device = triangles.vertices.device
    zero = torch.zeros((n,), dtype=torch.float32, device=device)
    ground = zero.clone()
    roi = zero.clone()
    render_keep = zero.clone()

    # 1) Ground protect from robust tracker when available.
    if isinstance(assoc_stats, dict):
        if "confidence" in assoc_stats:
            ground = torch.maximum(ground, _safe_resize_signal(assoc_stats["confidence"], n=n, device=device))
        if "ground_support_ratio" in assoc_stats:
            ground = torch.maximum(ground, _safe_resize_signal(assoc_stats["ground_support_ratio"], n=n, device=device))
        if "is_ground_mask" in assoc_stats:
            ground = torch.maximum(
                ground,
                _safe_resize_signal(assoc_stats["is_ground_mask"].to(torch.float32), n=n, device=device),
            )
        if "boundary_uncertain_mask" in assoc_stats:
            roi = torch.maximum(
                roi,
                0.5 * _safe_resize_signal(assoc_stats["boundary_uncertain_mask"].to(torch.float32), n=n, device=device),
            )

    # 2) Fallback ground/ROI from current rendered IDs + ground mask.
    if (render_pkg is not None) and (viewpoint_cam is not None):
        ids = render_pkg.get("rend_ids", None)
        if ids is not None:
            ids = ids.squeeze(0).to(device=device, dtype=torch.long)
            valid = (ids >= 0) & (ids < n)
            if torch.any(valid):
                # Ground mask projection as fallback ground signal.
                gm = getattr(viewpoint_cam, "ground_mask", None)
                if gm is not None:
                    gm_t = gm.to(device=device)
                    if gm_t.ndim == 3:
                        gm_t = gm_t[0]
                    if gm_t.ndim == 2:
                        if (gm_t.shape[0] != ids.shape[0]) or (gm_t.shape[1] != ids.shape[1]):
                            gm_t = F.interpolate(
                                gm_t[None, None].float(),
                                size=(ids.shape[0], ids.shape[1]),
                                mode="nearest",
                            ).squeeze(0).squeeze(0)
                        gm_bool = gm_t > 0.5
                        ids_valid = ids[valid]
                        ids_ground = ids[valid & gm_bool]
                        total = torch.bincount(ids_valid, minlength=n).to(torch.float32)
                        hit = torch.bincount(ids_ground, minlength=n).to(torch.float32)
                        ratio = hit / torch.clamp(total, min=1.0)
                        ground = torch.maximum(ground, ratio)
                        roi = torch.maximum(roi, ratio)

                # Near-field render keep via inverse depth per-triangle (screen-space proxy).
                surf_depth = render_pkg.get("surf_depth", None)
                if surf_depth is not None:
                    depth = surf_depth.to(device=device, dtype=torch.float32)
                    if depth.ndim == 4:
                        depth = depth[0, 0]
                    elif depth.ndim == 3:
                        depth = depth[0]
                    # Keep depth/id grids aligned before boolean indexing.
                    if (depth.shape[0] != ids.shape[0]) or (depth.shape[1] != ids.shape[1]):
                        depth = F.interpolate(
                            depth[None, None],
                            size=(ids.shape[0], ids.shape[1]),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0).squeeze(0)
                    inv = 1.0 / torch.clamp(depth, min=1e-6)
                    inv = torch.where(torch.isfinite(inv), inv, torch.zeros_like(inv))
                    inv_valid = inv[valid]
                    ids_valid = ids[valid]
                    sums = torch.bincount(ids_valid, weights=inv_valid, minlength=n).to(torch.float32)
                    cnt = torch.bincount(ids_valid, minlength=n).to(torch.float32)
                    mean_inv = sums / torch.clamp(cnt, min=1.0)
                    if torch.any(cnt > 0):
                        norm = torch.clamp(mean_inv / torch.clamp(torch.quantile(mean_inv[cnt > 0], 0.95), min=1e-6), 0.0, 1.0)
                        render_keep = torch.maximum(render_keep, norm)

    return (
        torch.clamp(ground, 0.0, 1.0),
        torch.clamp(roi, 0.0, 1.0),
        torch.clamp(render_keep, 0.0, 1.0),
    )


def _sparse_colmap_depth_lambda(
    iteration: int,
    current_phase,
    prism_enabled: bool,
    opt,
) -> float:
    if not bool(getattr(opt, "enable_sparse_colmap_depth_loss", False)):
        return 0.0
    start_iter = int(getattr(opt, "sparse_colmap_depth_start_iter", 0))
    if int(iteration) < start_iter:
        return 0.0

    phase_ok = True
    if prism_enabled:
        if current_phase == PrismPhase.RECOVERY_FINE_TUNE:
            phase_ok = bool(getattr(opt, "sparse_colmap_depth_enable_in_recovery", False))
        elif current_phase == PrismPhase.FINAL_FINE_TUNE:
            phase_ok = bool(getattr(opt, "sparse_colmap_depth_enable_in_final_finetune", False))
    if not phase_ok:
        return 0.0

    warmup_iters = max(1, int(getattr(opt, "sparse_colmap_depth_warmup_iters", 1)))
    warmup = min(1.0, max(0.0, float(int(iteration) - start_iter) / float(warmup_iters)))
    decay = 1.0
    decay_start = int(getattr(opt, "sparse_colmap_depth_decay_start_iter", -1))
    decay_end = int(getattr(opt, "sparse_colmap_depth_decay_end_iter", -1))
    decay_final = float(getattr(opt, "sparse_colmap_depth_decay_final_mult", 1.0))
    if decay_start >= 0 and decay_end > decay_start and int(iteration) >= decay_start:
        progress = min(1.0, max(0.0, float(int(iteration) - decay_start) / float(decay_end - decay_start)))
        decay = (1.0 - progress) + progress * max(0.0, decay_final)
    return float(getattr(opt, "lambda_sparse_colmap_depth", 0.0)) * warmup * decay


def _teacher_render_lambda(iteration: int, opt) -> float:
    if not bool(getattr(opt, "enable_teacher_render_loss", False)):
        return 0.0
    base = float(getattr(opt, "lambda_teacher_render", 0.0))
    if base <= 0.0:
        return 0.0
    start_iter = int(getattr(opt, "teacher_render_start_iter", 0))
    if int(iteration) < start_iter:
        return 0.0
    warmup_iters = max(1, int(getattr(opt, "teacher_render_warmup_iters", 1)))
    warmup = min(1.0, max(0.0, float(int(iteration) - start_iter) / float(warmup_iters)))
    decay = 1.0
    decay_start = int(getattr(opt, "teacher_render_decay_start_iter", -1))
    decay_end = int(getattr(opt, "teacher_render_decay_end_iter", -1))
    decay_final = float(getattr(opt, "teacher_render_decay_final_mult", 1.0))
    if decay_start >= 0 and decay_end > decay_start and int(iteration) >= decay_start:
        progress = min(1.0, max(0.0, float(int(iteration) - decay_start) / float(decay_end - decay_start)))
        decay = (1.0 - progress) + progress * max(0.0, decay_final)
    return base * warmup * decay


def _checkpoint_geometry_anchor_lambda(iteration: int, opt) -> float:
    if not bool(getattr(opt, "enable_checkpoint_geometry_anchor", False)):
        return 0.0
    base = float(getattr(opt, "lambda_checkpoint_geometry_anchor", 0.0))
    if base <= 0.0:
        return 0.0
    start_iter = int(getattr(opt, "checkpoint_geometry_anchor_start_iter", 0))
    if int(iteration) < start_iter:
        return 0.0
    warmup_iters = max(1, int(getattr(opt, "checkpoint_geometry_anchor_warmup_iters", 1)))
    warmup = min(1.0, max(0.0, float(int(iteration) - start_iter) / float(warmup_iters)))
    decay = 1.0
    decay_start = int(getattr(opt, "checkpoint_geometry_anchor_decay_start_iter", -1))
    decay_end = int(getattr(opt, "checkpoint_geometry_anchor_decay_end_iter", -1))
    decay_final = float(getattr(opt, "checkpoint_geometry_anchor_decay_final_mult", 1.0))
    if decay_start >= 0 and decay_end > decay_start and int(iteration) >= decay_start:
        progress = min(1.0, max(0.0, float(int(iteration) - decay_start) / float(decay_end - decay_start)))
        decay = (1.0 - progress) + progress * max(0.0, decay_final)
    return base * warmup * decay


def _compute_checkpoint_geometry_anchor_loss(triangles, anchor_vertices: torch.Tensor | None, lam: float, huber_delta: float):
    if anchor_vertices is None or lam <= 0.0:
        return None
    vertices = triangles.vertices
    if tuple(vertices.shape) != tuple(anchor_vertices.shape):
        return None
    anchor = anchor_vertices.to(device=vertices.device, dtype=vertices.dtype)
    delta = vertices - anchor
    per_vertex = torch.linalg.norm(delta, dim=-1)
    beta = max(float(huber_delta), 1e-8)
    loss_pure = F.smooth_l1_loss(per_vertex, torch.zeros_like(per_vertex), beta=beta, reduction="mean")
    loss_weighted = float(lam) * loss_pure
    return {
        "loss_pure": loss_pure,
        "loss_weighted": loss_weighted,
        "mean_displacement": per_vertex.detach().mean(),
        "max_displacement": per_vertex.detach().max(),
    }


def _checkpoint_render_geometry_anchor_lambda(iteration: int, opt, kind: str) -> float:
    if not bool(getattr(opt, "enable_checkpoint_render_geometry_anchor", False)):
        return 0.0
    if kind == "normal":
        base = float(getattr(opt, "lambda_checkpoint_render_normal_anchor", 0.0))
    else:
        base = float(getattr(opt, "lambda_checkpoint_render_depth_anchor", 0.0))
    if base <= 0.0:
        return 0.0
    start_iter = int(getattr(opt, "checkpoint_render_geometry_anchor_start_iter", 0))
    if int(iteration) < start_iter:
        return 0.0
    warmup_iters = max(1, int(getattr(opt, "checkpoint_render_geometry_anchor_warmup_iters", 1)))
    warmup = min(1.0, max(0.0, float(int(iteration) - start_iter) / float(warmup_iters)))
    return base * warmup


def _camera_cache_key(cam) -> str:
    return str(getattr(cam, "image_name", getattr(cam, "uid", "")))


def _build_checkpoint_render_geometry_cache(train_cameras, triangles, pipe, background) -> dict:
    cache = {}
    with torch.no_grad():
        for cam in train_cameras:
            pkg = render(cam, triangles, pipe, background)
            entry = {}
            if "surf_depth" in pkg:
                entry["surf_depth"] = pkg["surf_depth"].detach().clone()
            if "surf_normal" in pkg:
                entry["surf_normal"] = pkg["surf_normal"].detach().clone()
            if entry:
                cache[_camera_cache_key(cam)] = entry
    return cache


def _resize_like(anchor: torch.Tensor, target: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if tuple(anchor.shape[-2:]) == tuple(target.shape[-2:]):
        return anchor
    if anchor.ndim == 2:
        return F.interpolate(anchor[None, None], size=target.shape[-2:], mode="nearest").squeeze(0).squeeze(0)
    if anchor.ndim == 3:
        return F.interpolate(anchor[None], size=target.shape[-2:], mode=mode, align_corners=False).squeeze(0)
    return anchor


def _compute_checkpoint_render_geometry_anchor_loss(render_pkg, viewpoint_cam, cache: dict, depth_lam: float, normal_lam: float, huber_delta: float):
    if not cache or (depth_lam <= 0.0 and normal_lam <= 0.0):
        return None
    entry = cache.get(_camera_cache_key(viewpoint_cam), None)
    if entry is None:
        return None
    device = render_pkg["render"].device
    total = torch.tensor(0.0, device=device)
    depth_pure = torch.tensor(0.0, device=device)
    normal_pure = torch.tensor(0.0, device=device)
    if depth_lam > 0.0 and "surf_depth" in render_pkg and "surf_depth" in entry:
        pred = render_pkg["surf_depth"]
        anchor = entry["surf_depth"].to(device=pred.device, dtype=pred.dtype)
        anchor = _resize_like(anchor, pred, mode="nearest")
        mask = torch.isfinite(pred) & torch.isfinite(anchor) & (anchor > 0)
        if bool(mask.any()):
            delta = pred[mask] - anchor[mask]
            depth_pure = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=max(float(huber_delta), 1e-8), reduction="mean")
            total = total + float(depth_lam) * depth_pure
    if normal_lam > 0.0 and "surf_normal" in render_pkg and "surf_normal" in entry:
        pred_n = F.normalize(render_pkg["surf_normal"], dim=0, eps=1e-6)
        anchor_n = entry["surf_normal"].to(device=pred_n.device, dtype=pred_n.dtype)
        anchor_n = _resize_like(anchor_n, pred_n, mode="bilinear")
        anchor_n = F.normalize(anchor_n, dim=0, eps=1e-6)
        valid = torch.isfinite(pred_n).all(dim=0) & torch.isfinite(anchor_n).all(dim=0)
        if bool(valid.any()):
            cos = torch.clamp((pred_n * anchor_n).sum(dim=0), -1.0, 1.0)
            normal_pure = (1.0 - cos[valid]).mean()
            total = total + float(normal_lam) * normal_pure
    return {"loss_weighted": total, "depth_pure": depth_pure, "normal_pure": normal_pure}


def _load_teacher_render_cache(render_dir: str, train_cameras) -> dict:
    render_dir = str(render_dir or "").strip()
    if not render_dir:
        return {}
    if not os.path.isdir(render_dir):
        print(f"[TeacherRender] disabled: render_dir not found: {render_dir}")
        return {}
    cache = {}
    for idx, cam in enumerate(list(train_cameras)):
        path = os.path.join(render_dir, f"{idx:05d}.png")
        if not os.path.exists(path):
            continue
        try:
            arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            cache[str(getattr(cam, "image_name", idx))] = tensor
        except Exception as exc:
            print(f"[TeacherRender] skipped {path}: {exc}")
    print(f"[TeacherRender] loaded {len(cache)}/{len(list(train_cameras))} train renders from {render_dir}")
    return cache


def _compute_teacher_render_loss(
    viewpoint_cam,
    image,
    gt_image,
    teacher_cache: dict,
    lam: float,
    dssim_weight: float,
    mask_mode: str = "none",
    error_margin: float = 0.0,
):
    if lam <= 0.0 or not teacher_cache:
        return None
    key = str(getattr(viewpoint_cam, "image_name", ""))
    teacher = teacher_cache.get(key, None)
    if teacher is None:
        return None
    teacher = teacher.to(device=image.device, dtype=image.dtype, non_blocking=True)
    if teacher.shape[-2:] != image.shape[-2:]:
        teacher = F.interpolate(
            teacher.unsqueeze(0),
            size=(int(image.shape[-2]), int(image.shape[-1])),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    teacher = torch.clamp(teacher, 0.0, 1.0).detach()
    dssim_weight = min(1.0, max(0.0, float(dssim_weight)))
    mask = None
    mode = str(mask_mode or "none").strip().lower()
    if mode == "teacher_better":
        pred_err = torch.mean(torch.abs(image.detach() - gt_image.detach()), dim=0, keepdim=True)
        teacher_err = torch.mean(torch.abs(teacher - gt_image.detach()), dim=0, keepdim=True)
        mask = (teacher_err + float(max(0.0, error_margin)) < pred_err).to(dtype=image.dtype)
        if float(mask.mean().detach().item()) <= 1e-6:
            return None
    if mask is not None:
        teacher_l1 = (torch.abs(image - teacher) * mask).sum() / torch.clamp(mask.sum() * image.shape[0], min=1.0)
    else:
        teacher_l1 = l1_loss(image, teacher)
    if dssim_weight > 0.0:
        if FUSED_SSIM_AVAILABLE:
            teacher_ssim = fused_ssim(image.unsqueeze(0), teacher.unsqueeze(0))
        else:
            teacher_ssim = ssim(image, teacher)
        # SSIM is kept global. The counterfactual mask gates the L1 term, which
        # carries the localized corrective signal.
        pure = (1.0 - dssim_weight) * teacher_l1 + dssim_weight * (1.0 - teacher_ssim)
    else:
        teacher_ssim = None
        pure = teacher_l1
    return {
        "loss_pure": pure,
        "loss_weighted": float(lam) * pure,
        "l1": teacher_l1,
        "ssim": teacher_ssim,
        "mask_fraction": float(mask.mean().detach().item()) if mask is not None else 1.0,
    }


def _lpips_loss_lambda(iteration: int, opt) -> float:
    base = float(getattr(opt, "lambda_lpips_loss", 0.0))
    if base <= 0.0:
        return 0.0
    start_iter = int(getattr(opt, "lpips_loss_start_iter", 0))
    if int(iteration) < start_iter:
        return 0.0
    warmup_iters = max(1, int(getattr(opt, "lpips_loss_warmup_iters", 1)))
    warmup = min(1.0, max(0.0, float(int(iteration) - start_iter) / float(warmup_iters)))
    return base * warmup


def _compute_lpips_training_loss(image, gt_image, max_side: int = 512):
    max_side = max(64, int(max_side))
    pred = image
    target = gt_image
    h, w = int(image.shape[-2]), int(image.shape[-1])
    long_side = max(h, w)
    if long_side > max_side:
        scale = float(max_side) / float(long_side)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        pred = F.interpolate(pred.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        target = F.interpolate(target.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    return lpips_fn(pred, target.detach()).mean()


def _compute_sparse_colmap_depth_loss(
    viewpoint_cam,
    render_pkg,
    proxy_ctx,
    proxy_cfg: GeometryProxyConfig,
    min_matches: int,
    lam: float,
    loss_space: str = "depth",
    robust_beta: float = 0.05,
    rng: Optional[np.random.Generator] = None,
):
    if proxy_ctx is None or lam <= 0.0:
        return None
    surf_depth = render_pkg.get("surf_depth", None)
    if surf_depth is None:
        return {
            "loss_pure": None,
            "loss_weighted": None,
            "valid_matches": 0,
            "reason": "render_missing_output",
        }
    corr = collect_view_sparse_depth_correspondences(
        view=viewpoint_cam,
        ctx=proxy_ctx,
        cfg=proxy_cfg,
        rng=rng,
    )
    valid_matches = int(corr.get("num_matches", 0))
    if valid_matches < int(max(0, min_matches)):
        return {
            "loss_pure": None,
            "loss_weighted": None,
            "valid_matches": int(valid_matches),
            "reason": str(corr.get("reason", "insufficient_matches")),
        }

    pred_depth = surf_depth[0]
    device = pred_depth.device
    h, w = int(pred_depth.shape[0]), int(pred_depth.shape[1])
    px = np.clip(corr["px"], 0, w - 1).astype(np.int64, copy=False)
    py = np.clip(corr["py"], 0, h - 1).astype(np.int64, copy=False)
    px_t = torch.from_numpy(px).to(device=device, dtype=torch.long)
    py_t = torch.from_numpy(py).to(device=device, dtype=torch.long)
    gt_t = torch.from_numpy(corr["gt_depth"]).to(device=device, dtype=pred_depth.dtype)

    pred_t = pred_depth[py_t, px_t]
    # Robust sparse depth supervision.  Relative/log spaces keep far COLMAP
    # points from dominating the gradient and better match AbsRel reporting.
    space = str(loss_space or "depth").strip().lower()
    beta = float(max(1e-6, robust_beta))
    if space == "relative":
        denom = torch.clamp(gt_t.detach().abs(), min=1e-3)
        loss_pure = F.smooth_l1_loss((pred_t - gt_t) / denom, torch.zeros_like(gt_t), reduction="mean", beta=beta)
    elif space == "log":
        pred_log = torch.log(torch.clamp(pred_t, min=1e-4))
        gt_log = torch.log(torch.clamp(gt_t, min=1e-4))
        loss_pure = F.smooth_l1_loss(pred_log, gt_log, reduction="mean", beta=beta)
    elif space == "inverse":
        pred_inv = 1.0 / torch.clamp(pred_t, min=1e-4)
        gt_inv = 1.0 / torch.clamp(gt_t, min=1e-4)
        loss_pure = F.smooth_l1_loss(pred_inv, gt_inv, reduction="mean", beta=beta)
    else:
        loss_pure = F.smooth_l1_loss(pred_t, gt_t, reduction="mean", beta=beta)
    loss_weighted = float(lam) * loss_pure
    return {
        "loss_pure": loss_pure,
        "loss_weighted": loss_weighted,
        "valid_matches": int(valid_matches),
        "reason": "ok",
        "loss_space": space,
    }


def _topk_masked_ids(score_t: torch.Tensor, mask_t: torch.Tensor, k: int) -> torch.Tensor:
    device = score_t.device
    k = int(max(0, k))
    if k <= 0 or score_t.numel() == 0:
        return torch.zeros((0,), dtype=torch.int64, device=device)
    mask = mask_t.to(torch.bool)
    valid = int(mask.sum().item())
    if valid <= 0:
        return torch.zeros((0,), dtype=torch.int64, device=device)
    k = min(k, valid)
    masked = torch.where(mask, score_t.to(torch.float32), torch.full_like(score_t.to(torch.float32), -1e9))
    return torch.unique(torch.topk(masked, k=k, largest=True, sorted=False).indices.to(torch.int64))


def _build_lightweight_metric_defaults(manager, tri_count: int, device: torch.device):
    zeros_i32 = torch.zeros((tri_count,), dtype=torch.int32, device=device)
    zeros_f32 = torch.zeros((tri_count,), dtype=torch.float32, device=device)
    half = torch.full((tri_count,), 0.5, dtype=torch.float32, device=device)
    area_ema = manager.stats.projected_area_ema.to(device=device, dtype=torch.float32)
    denom = torch.clamp(torch.mean(area_ema), min=1e-6)
    geo_proxy = torch.clamp(area_ema / denom, 0.0, 1.0)
    struct_metrics = SimpleNamespace(
        boundary_edge_count=zeros_i32,
        nonmanifold_edge_count=zeros_i32,
        flatness_score=half,
        coplanar_neighbor_fraction=half,
        mean_abs_dihedral_deg=zeros_f32,
    )
    sparse_metrics = SimpleNamespace(
        support_count=zeros_i32,
        plane_residual_median=torch.full((tri_count,), 1e6, dtype=torch.float32, device=device),
        normal_angle_residual_deg=torch.full((tri_count,), 180.0, dtype=torch.float32, device=device),
        geometry_support_score_base=geo_proxy,
    )
    return struct_metrics, sparse_metrics


def _select_heavy_eval_triangle_ids(
    cfg,
    score_inputs: PrismScoreInputs,
    preliminary_scores,
    tri_count: int,
) -> torch.Tensor:
    budget = int(max(0, cfg.get("heavy_eval_budget", 0)))
    if budget <= 0 or tri_count <= 0:
        return torch.zeros((0,), dtype=torch.int64, device=preliminary_scores.prune_score_t.device)

    candidate_quota = max(1, budget // 2)
    protect_quota = max(1, budget - candidate_quota)
    far_field_ok = torch.ones((tri_count,), dtype=torch.bool, device=preliminary_scores.prune_score_t.device)
    if bool(cfg.get("skip_heavy_eval_for_far_field", False)):
        far_field_ok = ~(
            (preliminary_scores.render_keep_t <= 0.05)
            & (score_inputs.geometry_support_score_base.to(torch.float32) <= 0.10)
            & (preliminary_scores.optional_groundprotect_t <= 0.0)
            & (preliminary_scores.optional_roiprotect_t <= 0.0)
        )

    candidate_ids = _topk_masked_ids(
        score_t=preliminary_scores.prune_score_t,
        mask_t=preliminary_scores.candidate_mask & far_field_ok,
        k=candidate_quota,
    )
    protect_score = torch.stack(
        [
            preliminary_scores.geo_t,
            preliminary_scores.sens_t,
            preliminary_scores.unc_t,
            preliminary_scores.render_keep_t,
            preliminary_scores.optional_groundprotect_t,
            preliminary_scores.optional_roiprotect_t,
        ],
        dim=1,
    ).max(dim=1).values
    protect_ids = _topk_masked_ids(
        score_t=protect_score,
        mask_t=preliminary_scores.protected_mask_raw & far_field_ok,
        k=protect_quota,
    )
    seed_ids = torch.unique(torch.cat([candidate_ids, protect_ids], dim=0))
    if seed_ids.numel() == 0:
        seed_ids = _topk_masked_ids(
            score_t=protect_score + preliminary_scores.prune_score_t,
            mask_t=far_field_ok,
            k=min(budget, tri_count),
        )
    # Large-topology mode must stay local-only. Do not build a global structure
    # cache just to expand neighborhoods; that defeats the scalability fix and
    # was causing crashes around the first heavy-eval transition.
    if seed_ids.numel() > budget:
        seed_scores = preliminary_scores.prune_score_t[seed_ids]
        _, idx = torch.topk(seed_scores, k=int(budget), largest=True, sorted=False)
        seed_ids = seed_ids[idx]
    return torch.unique(seed_ids.to(torch.int64))


def _build_local_global_neighbor_map(local_neighbors, global_ids: torch.Tensor):
    out = {}
    if global_ids.numel() == 0:
        return out
    gids = [int(v) for v in global_ids.to(torch.int64).tolist()]
    for local_i, global_i in enumerate(gids):
        nbr_local = []
        if local_neighbors is not None and local_i < len(local_neighbors):
            nbr_local = local_neighbors[local_i]
        out[int(global_i)] = [int(gids[int(nid)]) for nid in nbr_local if int(nid) >= 0 and int(nid) < len(gids)]
    return out


def _update_prism_scores(
    prism_state,
    iteration,
    triangles,
    force_recompute: bool = False,
    render_pkg=None,
    viewpoint_cam=None,
    assoc_stats=None,
):
    """
    Compute PRISM score/classifier signals from collected statistics.
    No pruning decision is made here.
    """
    manager = prism_state.get("manager", None)
    if manager is None:
        return
    cfg = prism_state["cfg"]
    if int(iteration) < int(cfg["stats_warmup_iters"]):
        return
    interval = max(1, int(cfg["collect_interval"]))
    if int(iteration) % interval != 0:
        return
    tri_count = int(triangles._triangle_indices.shape[0])
    score_recompute_interval = max(1, int(cfg.get("score_recompute_interval", 500)))
    max_triangles_for_heavy = int(cfg.get("max_triangles_for_heavy_metrics", 400000))
    last_recompute_iter = int(prism_state.get("last_score_recompute_iter", -1))
    last_recompute_tri_count = int(prism_state.get("last_score_triangle_count", -1))
    have_cache = (prism_state.get("cached_structure_metrics", None) is not None) and (
        prism_state.get("cached_sparse_metrics", None) is not None
    )
    need_recompute = bool(force_recompute) or (not have_cache)
    need_recompute = need_recompute or (tri_count != last_recompute_tri_count)
    need_recompute = need_recompute or ((int(iteration) - last_recompute_iter) >= score_recompute_interval)

    if assoc_stats is None:
        assoc_stats = prism_state.get("latest_assoc_stats", None)
    ground_protect_t, roi_protect_t, render_keep_t = _build_prism_ground_and_roi_signals(
        prism_state=prism_state,
        triangles=triangles,
        render_pkg=render_pkg,
        viewpoint_cam=viewpoint_cam,
        assoc_stats=assoc_stats,
    )
    device = triangles.vertices.device
    force_full_heavy_eval_below = int(cfg.get("force_full_heavy_eval_below", max_triangles_for_heavy))
    use_two_stage_heavy = tri_count > max(0, force_full_heavy_eval_below)
    if max_triangles_for_heavy > 0:
        use_two_stage_heavy = use_two_stage_heavy or (tri_count > max_triangles_for_heavy)

    lightweight_only_mask = prism_state.get("cached_lightweight_only_mask", None)
    local_tri_neighbors = prism_state.get("cached_local_tri_neighbors", None)

    if need_recompute:
        sparse_est = prism_state.get("sparse_estimator", None)
        if sparse_est is None:
            return

        if not use_two_stage_heavy:
            struct_metrics, struct_cache = compute_triangle_structure_metrics(
                vertices=triangles.vertices,
                triangle_indices=triangles._triangle_indices,
                cache=prism_state.get("structure_cache", None),
            )
            prism_state["structure_cache"] = struct_cache
            sparse_metrics = sparse_est.compute(
                vertices=triangles.vertices,
                triangle_indices=triangles._triangle_indices,
            )
            lightweight_only_mask = torch.zeros((tri_count,), dtype=torch.bool, device=device)
            local_tri_neighbors = getattr(struct_cache, "tri_neighbors", None)
            prism_state["last_heavy_eval_info"] = {
                "mode": "full",
                "heavy_eval_triangle_count": int(tri_count),
                "heavy_eval_fraction": 1.0,
            }
        else:
            if not bool(prism_state.get("_warned_heavy_metrics_scalable", False)):
                print(
                    "[PRISM] scalable heavy-eval enabled (local submesh only): triangles={} budget={} rings={}".format(
                        tri_count,
                        int(cfg.get("heavy_eval_budget", 0)),
                        int(cfg.get("heavy_eval_neighbor_rings", 1)),
                    )
                )
                prism_state["_warned_heavy_metrics_scalable"] = True
            struct_metrics, sparse_metrics = _build_lightweight_metric_defaults(
                manager=manager,
                tri_count=tri_count,
                device=device,
            )
            cheap_inputs = PrismScoreInputs(
                vis_count_ema=manager.stats.vis_count_ema,
                grad_pos_norm_ema=manager.stats.grad_pos_norm_ema,
                grad_app_norm_ema=manager.stats.grad_app_norm_ema,
                grad_norm_var_ema=manager.stats.grad_norm_var_ema,
                view_direction_histogram=manager.stats.view_direction_histogram,
                birth_iter=manager.stats.birth_iter,
                geometry_support_score_base=sparse_metrics.geometry_support_score_base,
                boundary_edge_count=struct_metrics.boundary_edge_count,
                nonmanifold_edge_count=struct_metrics.nonmanifold_edge_count,
                flatness_score=struct_metrics.flatness_score,
                coplanar_neighbor_fraction=struct_metrics.coplanar_neighbor_fraction,
                mean_abs_dihedral_deg=struct_metrics.mean_abs_dihedral_deg,
                sparse_support_count=sparse_metrics.support_count,
                sparse_plane_residual=sparse_metrics.plane_residual_median,
                sparse_normal_residual_deg=sparse_metrics.normal_angle_residual_deg,
                render_keep_t=render_keep_t,
                tri_neighbors=None,
                ground_protect_t=ground_protect_t,
                roi_protect_t=roi_protect_t,
                lightweight_only_mask=torch.ones((tri_count,), dtype=torch.bool, device=device),
            )
            cheap_scores = compute_prism_scores(
                inputs=cheap_inputs,
                current_iter=int(iteration),
                cfg=prism_state["score_cfg"],
            )
            heavy_ids = _select_heavy_eval_triangle_ids(
                cfg=cfg,
                score_inputs=cheap_inputs,
                preliminary_scores=cheap_scores,
                tri_count=tri_count,
            )
            lightweight_only_mask = torch.ones((tri_count,), dtype=torch.bool, device=device)
            local_tri_neighbors = {}
            if heavy_ids.numel() > 0:
                lightweight_only_mask[heavy_ids] = False
                local_struct_metrics, local_struct_cache = compute_triangle_structure_metrics(
                    vertices=triangles.vertices,
                    triangle_indices=triangles._triangle_indices[heavy_ids],
                    cache=None,
                )
                sparse_subset = sparse_est.compute_subset(
                    vertices=triangles.vertices,
                    triangle_indices=triangles._triangle_indices,
                    triangle_ids=heavy_ids,
                )
                local_tri_neighbors = _build_local_global_neighbor_map(
                    local_neighbors=getattr(local_struct_cache, "tri_neighbors", None),
                    global_ids=heavy_ids,
                )
                struct_metrics.boundary_edge_count[heavy_ids] = local_struct_metrics.boundary_edge_count
                struct_metrics.nonmanifold_edge_count[heavy_ids] = local_struct_metrics.nonmanifold_edge_count
                struct_metrics.mean_abs_dihedral_deg[heavy_ids] = local_struct_metrics.mean_abs_dihedral_deg
                struct_metrics.coplanar_neighbor_fraction[heavy_ids] = local_struct_metrics.coplanar_neighbor_fraction
                struct_metrics.flatness_score[heavy_ids] = local_struct_metrics.flatness_score
                sparse_metrics.support_count[sparse_subset.triangle_ids] = sparse_subset.support_count
                sparse_metrics.plane_residual_median[sparse_subset.triangle_ids] = sparse_subset.plane_residual_median
                sparse_metrics.normal_angle_residual_deg[sparse_subset.triangle_ids] = sparse_subset.normal_angle_residual_deg
                sparse_metrics.geometry_support_score_base[sparse_subset.triangle_ids] = torch.maximum(
                    sparse_metrics.geometry_support_score_base[sparse_subset.triangle_ids],
                    sparse_subset.geometry_support_score_base,
                )
            prism_state["structure_cache"] = None
            prism_state["last_heavy_eval_info"] = {
                "mode": "two_stage",
                "heavy_eval_triangle_count": int((~lightweight_only_mask).sum().item()),
                "heavy_eval_fraction": float((~lightweight_only_mask).to(torch.float32).mean().item())
                if tri_count > 0
                else 0.0,
                "heavy_eval_budget": int(cfg.get("heavy_eval_budget", 0)),
                "heavy_eval_neighbor_rings": int(cfg.get("heavy_eval_neighbor_rings", 1)),
            }

        prism_state["cached_structure_metrics"] = struct_metrics
        prism_state["cached_sparse_metrics"] = sparse_metrics
        prism_state["cached_lightweight_only_mask"] = lightweight_only_mask
        prism_state["cached_local_tri_neighbors"] = local_tri_neighbors
        prism_state["last_score_recompute_iter"] = int(iteration)
        prism_state["last_score_triangle_count"] = int(tri_count)
    else:
        struct_metrics = prism_state.get("cached_structure_metrics", None)
        sparse_metrics = prism_state.get("cached_sparse_metrics", None)
        if (struct_metrics is None) or (sparse_metrics is None):
            return
        if lightweight_only_mask is None:
            lightweight_only_mask = torch.zeros((tri_count,), dtype=torch.bool, device=device)

    score_inputs = PrismScoreInputs(
        vis_count_ema=manager.stats.vis_count_ema,
        grad_pos_norm_ema=manager.stats.grad_pos_norm_ema,
        grad_app_norm_ema=manager.stats.grad_app_norm_ema,
        grad_norm_var_ema=manager.stats.grad_norm_var_ema,
        view_direction_histogram=manager.stats.view_direction_histogram,
        birth_iter=manager.stats.birth_iter,
        geometry_support_score_base=sparse_metrics.geometry_support_score_base,
        boundary_edge_count=struct_metrics.boundary_edge_count,
        nonmanifold_edge_count=struct_metrics.nonmanifold_edge_count,
        flatness_score=struct_metrics.flatness_score,
        coplanar_neighbor_fraction=struct_metrics.coplanar_neighbor_fraction,
        mean_abs_dihedral_deg=getattr(struct_metrics, "mean_abs_dihedral_deg", None),
        sparse_support_count=getattr(sparse_metrics, "support_count", None),
        sparse_plane_residual=getattr(sparse_metrics, "plane_residual_median", None),
        sparse_normal_residual_deg=getattr(sparse_metrics, "normal_angle_residual_deg", None),
        render_keep_t=render_keep_t,
        tri_neighbors=local_tri_neighbors,
        ground_protect_t=ground_protect_t,
        roi_protect_t=roi_protect_t,
        lightweight_only_mask=lightweight_only_mask,
    )
    score_cfg = prism_state["score_cfg"]
    scores = compute_prism_scores(
        inputs=score_inputs,
        current_iter=int(iteration),
        cfg=score_cfg,
    )
    prism_state["last_scores"] = scores
    manager.stats.triangle_state = scores.triangle_state.to(manager.stats.triangle_state.dtype)
    summary = summarize_prism_scores(scores)
    prism_state["last_scores_summary"] = summary
    heavy_info = prism_state.get("last_heavy_eval_info", {})

    if need_recompute:
        print(
            "[PRISM] score recompute iter={} mode={} heavy_eval={}/{} ({:.3f}) geom_keep_nonzero={:.3f} orient_keep_nonzero={:.3f}".format(
                int(iteration),
                str(heavy_info.get("mode", "cached")),
                int(summary.get("heavy_eval_triangle_count", 0)),
                int(tri_count),
                float(summary.get("heavy_eval_fraction", 0.0)),
                float(summary.get("geometry_keep_nonzero_fraction", 0.0)),
                float(summary.get("orientation_keep_nonzero_fraction", 0.0)),
            )
        )

    if bool(cfg["save_debug_json"]):
        out_dir = os.path.join(prism_state["model_path"], "prism_debug")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"prism_score_summary_iter_{int(iteration):06d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "meta": {
                        "force_recompute": bool(force_recompute),
                        "score_recompute_interval": int(score_recompute_interval),
                        "last_score_recompute_iter": int(prism_state.get("last_score_recompute_iter", -1)),
                        "heavy_eval": heavy_info,
                    },
                    "signals": {
                        "ground_protect_mean": float(ground_protect_t.mean().item()) if ground_protect_t.numel() > 0 else 0.0,
                        "roi_protect_mean": float(roi_protect_t.mean().item()) if roi_protect_t.numel() > 0 else 0.0,
                        "render_keep_mean": float(render_keep_t.mean().item()) if render_keep_t.numel() > 0 else 0.0,
                        "protected_raw_count": int(scores.protected_mask_raw.sum().item()),
                        "protected_dilated_count": int(scores.protected_mask_dilated.sum().item()),
                        "heavy_eval_fraction": float(summary.get("heavy_eval_fraction", 0.0)),
                        "heavy_eval_triangle_count": int(summary.get("heavy_eval_triangle_count", 0)),
                        "geometry_keep_nonzero_fraction": float(summary.get("geometry_keep_nonzero_fraction", 0.0)),
                        "orientation_keep_nonzero_fraction": float(summary.get("orientation_keep_nonzero_fraction", 0.0)),
                        "candidate_blocked_by_geometry_keep_count": int(
                            summary.get("candidate_blocked_by_geometry_keep_count", 0)
                        ),
                        "candidate_blocked_by_dilated_protect_count": int(
                            summary.get("candidate_blocked_by_dilated_protect_count", 0)
                        ),
                    },
                    "top_protected": [
                        {
                            "triangle_id": int(tid),
                            "reason": "geometry_keep"
                            if float(scores.geometry_keep_t[tid].item()) > float(score_cfg.keep_geometry_threshold)
                            else (
                                "orientation_keep"
                                if float(scores.orientation_keep_t[tid].item()) > float(score_cfg.keep_orientation_threshold)
                                else (
                                    "render_keep"
                                    if float(scores.render_keep_t[tid].item()) > float(score_cfg.keep_render_threshold)
                                    else "legacy"
                                )
                            ),
                            "geometry_keep": float(scores.geometry_keep_t[tid].item()),
                            "orientation_keep": float(scores.orientation_keep_t[tid].item()),
                            "render_keep": float(scores.render_keep_t[tid].item()),
                            "ground_protect": float(scores.optional_groundprotect_t[tid].item()),
                            "roi_protect": float(scores.optional_roiprotect_t[tid].item()),
                            "lightweight_only": bool(scores.lightweight_only_mask[tid].item()),
                        }
                        for tid in torch.nonzero(scores.protected_mask_dilated, as_tuple=True)[0][:20].tolist()
                    ],
                    "top_candidates": [
                        {
                            "triangle_id": int(tid),
                            "prune_score": float(scores.prune_score_t[tid].item()),
                            "reason": "candidate",
                            "geometry_keep": float(scores.geometry_keep_t[tid].item()),
                            "in_dilated_protected_nbr": bool(scores.candidate_blocked_by_dilated_protect[tid].item()),
                            "lightweight_only": bool(scores.lightweight_only_mask[tid].item()),
                        }
                        for tid in torch.topk(scores.prune_score_t * scores.candidate_mask.to(torch.float32), k=min(20, int(scores.prune_score_t.numel())), largest=True).indices.tolist()
                        if bool(scores.candidate_mask[tid].item())
                    ],
                    "top_blocked_candidates": [
                        {
                            "triangle_id": int(tid),
                            "prune_score": float(scores.prune_score_t[tid].item()),
                            "blocked_by_geometry_keep": bool(scores.candidate_blocked_by_geometry_keep[tid].item()),
                            "blocked_by_dilated_protect": bool(scores.candidate_blocked_by_dilated_protect[tid].item()),
                            "geometry_keep": float(scores.geometry_keep_t[tid].item()),
                            "lightweight_only": bool(scores.lightweight_only_mask[tid].item()),
                        }
                        for tid in torch.topk(scores.prune_score_t, k=min(20, int(scores.prune_score_t.numel())), largest=True).indices.tolist()
                        if (not bool(scores.candidate_mask[tid].item()))
                        and (
                            bool(scores.candidate_blocked_by_geometry_keep[tid].item())
                            or bool(scores.candidate_blocked_by_dilated_protect[tid].item())
                        )
                    ],
                },
                f,
                indent=2,
            )


def _build_prism_teacher_cache(prism_state, scene, triangles):
    if (not prism_state.get("enable_teacher_rgb_distill", False)) and (not prism_state.get("enable_teacher_depth_distill", False)):
        prism_state["teacher_cache"] = {}
        return
    views = prism_state.get("calibration_views", [])
    if len(views) == 0:
        prism_state["teacher_cache"] = {}
        return
    max_views = max(1, int(prism_state.get("teacher_num_views", 8)))
    selected = views[:max_views]
    cache = {}
    with torch.no_grad():
        for cam in selected:
            pkg = render(cam, triangles, prism_state["pipe"], prism_state["background"])
            entry = {}
            if prism_state.get("enable_teacher_rgb_distill", False):
                entry["render"] = pkg["render"].detach().clone()
            if prism_state.get("enable_teacher_depth_distill", False):
                entry["surf_depth"] = pkg["surf_depth"].detach().clone()
            cache[getattr(cam, "image_name", "")] = entry
    prism_state["teacher_cache"] = cache


def _apply_prism_teacher_distill(prism_state, viewpoint_cam, render_pkg):
    cache = prism_state.get("teacher_cache", {})
    if not cache:
        return torch.tensor(0.0, device=render_pkg["render"].device)
    key = getattr(viewpoint_cam, "image_name", "")
    if key not in cache:
        return torch.tensor(0.0, device=render_pkg["render"].device)
    entry = cache[key]
    loss = torch.tensor(0.0, device=render_pkg["render"].device)
    if prism_state.get("enable_teacher_rgb_distill", False) and ("render" in entry):
        teacher_rgb = entry["render"].to(render_pkg["render"].device)
        loss = loss + float(prism_state.get("teacher_rgb_lambda", 0.01)) * torch.mean(torch.abs(render_pkg["render"] - teacher_rgb))
    if prism_state.get("enable_teacher_depth_distill", False) and ("surf_depth" in entry):
        teacher_depth = entry["surf_depth"].to(render_pkg["surf_depth"].device)
        loss = loss + float(prism_state.get("teacher_depth_lambda", 0.002)) * torch.mean(torch.abs(render_pkg["surf_depth"] - teacher_depth))
    return loss


def _save_prism_round_checkpoint(scene, triangles, iteration: int, tag: str):
    out_dir = os.path.join(scene.model_path, "prism_round_checkpoints", f"iter_{int(iteration):06d}_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    triangles.save_parameters(out_dir)
    meta = {"iteration": int(iteration), "tag": str(tag), "triangles": int(triangles._triangle_indices.shape[0]), "vertices": int(triangles.vertices.shape[0])}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def _save_prism_round_meta(scene, iteration: int, prune_mode: str, payload: dict):
    out_dir = os.path.join(scene.model_path, "prism_round_checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"iter_{int(iteration):06d}_{str(prune_mode)}_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _sync_prism_topology_change(prism_state, triangles, iteration: int, reason: str = ""):
    manager = prism_state.get("manager", None)
    if manager is not None:
        manager.on_topology_change(
            new_num_triangles=int(triangles._triangle_indices.shape[0]),
            iteration=int(iteration),
        )
    prism_state["last_topology_change_iter"] = int(iteration)
    if reason:
        prism_state["last_topology_change_reason"] = str(reason)


def _save_final_cleanup_checkpoint(scene, triangles, iteration: int, tag: str):
    out_dir = os.path.join(scene.model_path, "prism_final_cleanup_checkpoints", f"iter_{int(iteration):06d}_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    triangles.save_parameters(out_dir)
    meta = {
        "iteration": int(iteration),
        "tag": str(tag),
        "triangles": int(triangles._triangle_indices.shape[0]),
        "vertices": int(triangles.vertices.shape[0]),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def _capture_manager_snapshot(manager):
    if manager is None:
        return None
    return {
        "num_triangles": int(manager.num_triangles),
        "ema_decay": float(manager.ema_decay),
        "view_hist_bins": int(manager.view_hist_bins),
        "last_seen_iteration": int(manager.last_seen_iteration),
        "last_global_topology_change_iter": int(manager.last_global_topology_change_iter),
        "_grad_total_mean_ema": manager._grad_total_mean_ema.detach().clone(),
        "stats": {
            "vis_count_ema": manager.stats.vis_count_ema.detach().clone(),
            "projected_area_ema": manager.stats.projected_area_ema.detach().clone(),
            "grad_pos_norm_ema": manager.stats.grad_pos_norm_ema.detach().clone(),
            "grad_app_norm_ema": manager.stats.grad_app_norm_ema.detach().clone(),
            "grad_norm_var_ema": manager.stats.grad_norm_var_ema.detach().clone(),
            "view_direction_histogram": manager.stats.view_direction_histogram.detach().clone(),
            "birth_iter": manager.stats.birth_iter.detach().clone(),
            "last_topology_change_iter": manager.stats.last_topology_change_iter.detach().clone(),
            "active_mask": manager.stats.active_mask.detach().clone(),
            "triangle_state": manager.stats.triangle_state.detach().clone(),
        },
    }


def _restore_manager_snapshot(manager, snapshot):
    if manager is None or snapshot is None:
        return
    device = manager.device
    manager.num_triangles = int(snapshot.get("num_triangles", manager.num_triangles))
    manager.ema_decay = float(snapshot.get("ema_decay", manager.ema_decay))
    manager.view_hist_bins = int(snapshot.get("view_hist_bins", manager.view_hist_bins))
    manager.last_seen_iteration = int(snapshot.get("last_seen_iteration", manager.last_seen_iteration))
    manager.last_global_topology_change_iter = int(
        snapshot.get("last_global_topology_change_iter", manager.last_global_topology_change_iter)
    )
    stats = snapshot.get("stats", {})
    manager.stats.vis_count_ema = stats["vis_count_ema"].to(device=device)
    manager.stats.projected_area_ema = stats["projected_area_ema"].to(device=device)
    manager.stats.grad_pos_norm_ema = stats["grad_pos_norm_ema"].to(device=device)
    manager.stats.grad_app_norm_ema = stats["grad_app_norm_ema"].to(device=device)
    manager.stats.grad_norm_var_ema = stats["grad_norm_var_ema"].to(device=device)
    manager.stats.view_direction_histogram = stats["view_direction_histogram"].to(device=device)
    manager.stats.birth_iter = stats["birth_iter"].to(device=device)
    manager.stats.last_topology_change_iter = stats["last_topology_change_iter"].to(device=device)
    manager.stats.active_mask = stats["active_mask"].to(device=device)
    manager.stats.triangle_state = stats["triangle_state"].to(device=device)
    manager._grad_total_mean_ema = snapshot["_grad_total_mean_ema"].to(device=device)


def _save_prism_compaction_checkpoint(scene, triangles, iteration: int, tag: str, meta: Optional[dict] = None, prism_state=None):
    out_dir = os.path.join(scene.model_path, "prism_compaction_checkpoints", f"iter_{int(iteration):06d}_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    triangles.save_parameters(out_dir)
    payload = {
        "iteration": int(iteration),
        "tag": str(tag),
        "triangles": int(triangles._triangle_indices.shape[0]),
        "vertices": int(triangles.vertices.shape[0]),
    }
    if isinstance(meta, dict):
        payload.update(meta)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if isinstance(prism_state, dict):
        manager_snapshot = _capture_manager_snapshot(prism_state.get("manager", None))
        if manager_snapshot is not None:
            torch.save(manager_snapshot, os.path.join(out_dir, "triangle_stats_snapshot.pt"))
        assoc_stats = prism_state.get("latest_assoc_stats", None)
        if assoc_stats is not None:
            torch.save(assoc_stats, os.path.join(out_dir, "assoc_stats_snapshot.pt"))
    return out_dir


def _load_saved_triangle_parameters(prism_state, triangles, checkpoint_dir: str, iteration: int, reason: str = "") -> bool:
    if not checkpoint_dir or (not os.path.exists(os.path.join(checkpoint_dir, "point_cloud_state_dict.pt"))):
        return False
    device = triangles.vertices.device
    triangles.load_parameters(checkpoint_dir, device=device)
    triangles.clear_temporary_active_mask()
    restored_manager_snapshot = False
    stats_snapshot_path = os.path.join(checkpoint_dir, "triangle_stats_snapshot.pt")
    if os.path.exists(stats_snapshot_path):
        try:
            _restore_manager_snapshot(prism_state.get("manager", None), torch.load(stats_snapshot_path, map_location=device))
            restored_manager_snapshot = True
        except Exception:
            pass
    assoc_snapshot_path = os.path.join(checkpoint_dir, "assoc_stats_snapshot.pt")
    if os.path.exists(assoc_snapshot_path):
        try:
            prism_state["latest_assoc_stats"] = torch.load(assoc_snapshot_path, map_location=device)
        except Exception:
            prism_state["latest_assoc_stats"] = None
    if restored_manager_snapshot:
        prism_state["last_topology_change_iter"] = int(iteration)
        if reason:
            prism_state["last_topology_change_reason"] = str(reason)
    else:
        _sync_prism_topology_change(
            prism_state=prism_state,
            triangles=triangles,
            iteration=int(iteration),
            reason=str(reason or "load_saved_parameters"),
        )
    return True


def _extract_geometry_key(metrics: Optional[dict]) -> tuple:
    metrics = metrics or {}
    absrel = float(metrics.get("absrel", metrics.get("AbsRel", float("nan"))))
    mean_angle = float(metrics.get("mean_angle", metrics.get("mean_normal_angle", metrics.get("MeanAngle", float("nan")))))
    depth_mae = float(metrics.get("depth_mae", metrics.get("DepthMAE", float("nan"))))
    delta = float(metrics.get("delta_1.25", metrics.get("delta_125", metrics.get("Delta1.25", float("nan")))))
    if not (np.isfinite(absrel) and np.isfinite(mean_angle) and np.isfinite(depth_mae) and np.isfinite(delta)):
        return (float("inf"), float("inf"), float("inf"), float("inf"))
    return (absrel, mean_angle, depth_mae, -delta)


def _geometry_key_is_better(lhs: Optional[dict], rhs: Optional[dict]) -> bool:
    return _extract_geometry_key(lhs) < _extract_geometry_key(rhs)


def _maybe_save_compaction_source_checkpoint(
    prism_state,
    scene,
    triangles,
    iteration: int,
    current_metrics: dict,
    pass_gate: bool,
    stage_best_update: dict,
):
    comp_cfg = prism_state.get("compaction_cfg", None)
    if comp_cfg is None or (not bool(getattr(comp_cfg, "enabled", False))):
        return
    if not bool(pass_gate):
        return
    pass_dir = _save_prism_compaction_checkpoint(
        scene=scene,
        triangles=triangles,
        iteration=int(iteration),
        tag="validation_pass",
        meta={"metrics": current_metrics, "source_kind": "validation_pass"},
        prism_state=prism_state,
    )
    prism_state["last_validation_pass_checkpoint_dir"] = str(pass_dir)
    if bool(stage_best_update.get("update", False)) or (str(stage_best_update.get("reason", "")) == "initialize_observable_stage_best"):
        best_dir = _save_prism_compaction_checkpoint(
            scene=scene,
            triangles=triangles,
            iteration=int(iteration),
            tag="best_geometry_source",
            meta={"metrics": current_metrics, "source_kind": "best_geometry"},
            prism_state=prism_state,
        )
        prism_state["stage_best_checkpoint_dir"] = str(best_dir)


def _capture_prism_round_snapshot(triangles):
    snap = {
        "vertices": triangles.vertices.detach().clone(),
        "triangle_indices": triangles._triangle_indices.detach().clone(),
        "vertex_weight": triangles.vertex_weight.detach().clone(),
        "features_dc": triangles._features_dc.detach().clone(),
        "features_rest": triangles._features_rest.detach().clone(),
        "sigma": copy.deepcopy(triangles._sigma),
        "active_sh_degree": int(triangles.active_sh_degree),
        "importance_score": triangles.importance_score.detach().clone() if torch.is_tensor(triangles.importance_score) else None,
        "image_size": triangles.image_size.detach().clone() if torch.is_tensor(triangles.image_size) else None,
        "pixel_count": triangles.pixel_count.detach().clone() if torch.is_tensor(triangles.pixel_count) else None,
        "optimizer_state": copy.deepcopy(triangles.optimizer.state_dict()) if triangles.optimizer is not None else None,
    }
    return snap


def _restore_prism_round_snapshot(prism_state, triangles, iteration: int):
    snap = prism_state.get("round_snapshot", None)
    if snap is None:
        return False

    device = triangles.vertices.device
    triangles.vertices = torch.nn.Parameter(snap["vertices"].to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True))
    triangles._triangle_indices = snap["triangle_indices"].to(device=device, dtype=torch.int32).detach().clone()
    triangles.vertex_weight = torch.nn.Parameter(snap["vertex_weight"].to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True))
    triangles._features_dc = torch.nn.Parameter(snap["features_dc"].to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True))
    triangles._features_rest = torch.nn.Parameter(snap["features_rest"].to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True))
    triangles._sigma = copy.deepcopy(snap.get("sigma", triangles._sigma))
    triangles.active_sh_degree = int(snap.get("active_sh_degree", triangles.active_sh_degree))
    triangles.importance_score = (
        snap["importance_score"].to(device=device, dtype=torch.float32).detach().clone()
        if snap.get("importance_score", None) is not None
        else torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float32, device=device)
    )
    triangles.image_size = (
        snap["image_size"].to(device=device, dtype=torch.float32).detach().clone()
        if snap.get("image_size", None) is not None
        else torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float32, device=device)
    )
    triangles.pixel_count = (
        snap["pixel_count"].to(device=device, dtype=torch.int).detach().clone()
        if snap.get("pixel_count", None) is not None
        else torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.int, device=device)
    )
    triangles.clear_temporary_active_mask()

    opt = prism_state.get("opt", None)
    if opt is None:
        return False
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    opt_state = snap.get("optimizer_state", None)
    if opt_state is not None:
        try:
            triangles.optimizer.load_state_dict(opt_state)
        except Exception:
            pass

    manager = prism_state.get("manager", None)
    if manager is not None:
        manager.on_topology_change(
            new_num_triangles=int(triangles._triangle_indices.shape[0]),
            iteration=int(iteration),
        )
    prism_state["last_topology_change_iter"] = int(iteration)
    prism_state["last_topology_change_reason"] = "rollback_restore"
    return True


def _evaluate_prism_validation(prism_state, scene, triangles, iteration: int):
    views = prism_state.get("validation_views", [])
    if len(views) == 0:
        return None
    current_metrics = evaluate_prism_validation_metrics(
        views=views,
        triangles=triangles,
        render_func=render,
        pipe=prism_state.get("pipe", None),
        background=prism_state.get("background", None),
        proxy_ctx=prism_state.get("proxy_ctx", None),
        proxy_cfg=prism_state.get("proxy_cfg", None),
        cfg=prism_state.get("validation_cfg", None),
    )
    stage_best = prism_state.get("stage_best_metrics", None)
    current_observable = bool(float(current_metrics.get("geometry_observable", 0.0)) > 0.5)
    stage_best_update = decide_stage_best_update(
        current_metrics=current_metrics,
        stage_best_metrics=stage_best,
    )
    if stage_best is None:
        if current_observable:
            stage_best = dict(current_metrics)
            pass_gate = True
            deltas = {}
            rules = []
        else:
            pass_gate = False
            deltas = {}
            rules = list(current_metrics.get("geometry_failure_reasons", ["insufficient_geometry_observability"]))
    else:
        pass_gate, deltas, rules = compare_validation_against_stage_best(
            current_metrics=current_metrics,
            stage_best_metrics=stage_best,
            cfg=prism_state["validation_cfg"],
        )
        # Geometry-first stage-best policy:
        # 1) geometry tuple decides admissibility/update priority
        # 2) image metrics can only break ties after geometry is not worse
        if pass_gate and bool(stage_best_update.get("update", False)):
            stage_best = dict(current_metrics)

    prism_state["stage_best_metrics"] = stage_best
    prism_state["last_validation_metrics"] = current_metrics
    prism_state["last_validation_pass"] = 1 if pass_gate else 0
    prism_state["last_validation_rules"] = list(rules)
    prism_state["last_validation_deltas"] = dict(deltas)
    prism_state["last_stage_best_update"] = dict(stage_best_update)
    prism_state["last_validation_eval_iter"] = int(iteration)

    out_dir = os.path.join(prism_state["model_path"], "prism_validation")
    save_validation_summary(
        out_dir=out_dir,
        iteration=int(iteration),
        phase_name=str(prism_state.get("current_phase", PrismPhase.FINAL_FINE_TUNE)),
        current_metrics=current_metrics,
        stage_best_metrics=stage_best,
        deltas=deltas,
        pass_gate=pass_gate,
        triggered_rules=rules,
        stage_best_update=stage_best_update,
    )
    debug_dir = os.path.join(prism_state["model_path"], "prism_debug")
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, f"validation_gate_iter_{int(iteration):06d}.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "iteration": int(iteration),
                "phase": str(prism_state.get("current_phase", PrismPhase.FINAL_FINE_TUNE)),
                "pass_gate": bool(pass_gate),
                "num_views": int(current_metrics.get("num_views", 0)),
                "num_depth_views_used": int(current_metrics.get("num_depth_views_used", 0)),
                "num_normal_views_used": int(current_metrics.get("num_normal_views_used", 0)),
                "total_valid_depth_matches": int(current_metrics.get("total_valid_depth_matches", 0)),
                "total_valid_normal_matches": int(current_metrics.get("total_valid_normal_matches", 0)),
                "dropped_views_reason_breakdown": current_metrics.get("dropped_views_reason_breakdown", {}),
                "geometry_failure_reasons": current_metrics.get("geometry_failure_reasons", []),
                "triggered_rules": list(rules),
                "stage_best_update": stage_best_update,
            },
            f,
            indent=2,
        )
    _maybe_save_compaction_source_checkpoint(
        prism_state=prism_state,
        scene=scene,
        triangles=triangles,
        iteration=int(iteration),
        current_metrics=current_metrics,
        pass_gate=bool(pass_gate),
        stage_best_update=stage_best_update,
    )
    return {
        "pass_gate": bool(pass_gate),
        "current_metrics": current_metrics,
        "stage_best_metrics": stage_best,
        "deltas": deltas,
        "rules": rules,
    }


def _run_prism_compaction_stage(
    prism_state,
    scene,
    triangles,
    iteration: int,
    tb_writer=None,
    wandb_run=None,
    wandb_log_state=None,
    ground_association_tracker=None,
):
    comp_cfg = prism_state.get("compaction_cfg", None)
    if comp_cfg is None or (not bool(getattr(comp_cfg, "enabled", False))):
        prism_state["compaction_phase"] = PrismCompactionPhase.DISABLED
        return {"ran": False, "reason": "disabled"}

    prism_state["compaction_phase"] = PrismCompactionPhase.SELECT_SOURCE
    pref = str(getattr(comp_cfg, "source_preference", "best_geometry")).strip().lower()
    source_dir = ""
    if pref == "best_geometry":
        source_dir = str(prism_state.get("stage_best_checkpoint_dir", "") or "")
        if not source_dir:
            source_dir = str(prism_state.get("last_validation_pass_checkpoint_dir", "") or "")
    else:
        source_dir = str(prism_state.get("last_validation_pass_checkpoint_dir", "") or "")
        if not source_dir:
            source_dir = str(prism_state.get("stage_best_checkpoint_dir", "") or "")
    if not source_dir:
        payload = {"ran": False, "reason": "no_geometry_safe_source_checkpoint"}
        out_dir = os.path.join(scene.model_path, "prism_compaction")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "compaction_summary.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        prism_state["compaction_phase"] = PrismCompactionPhase.FINALIZE
        return payload

    prism_state["compaction_phase"] = PrismCompactionPhase.RESTORE_SOURCE
    if not _load_saved_triangle_parameters(
        prism_state=prism_state,
        triangles=triangles,
        checkpoint_dir=source_dir,
        iteration=int(iteration),
        reason="compaction_source_restore",
    ):
        return {"ran": False, "reason": "failed_to_restore_source"}
    prism_state["compaction_source_checkpoint_dir"] = str(source_dir)
    prism_state["compaction_best_geometry_checkpoint_dir"] = str(source_dir)

    if ground_association_tracker is not None:
        ground_association_tracker.ensure_num_triangles(int(triangles._triangle_indices.shape[0]))

    best_geometry_metrics = prism_state.get("stage_best_metrics", None)
    best_speed_triangles = int(triangles._triangle_indices.shape[0])
    total_committed = 0
    total_rollbacks = 0
    microbatch_logs = []
    global_micro_idx = 0

    for round_idx in range(int(max(0, getattr(comp_cfg, "rounds", 0)))):
        rejected_mask = None
        round_committed = 0
        for micro_idx in range(int(max(0, getattr(comp_cfg, "max_microbatches_per_round", 0)))):
            prism_state["compaction_phase"] = PrismCompactionPhase.SCORE_REFRESH
            _update_prism_scores(
                prism_state=prism_state,
                iteration=int(iteration),
                triangles=triangles,
                force_recompute=True,
            )
            scores = prism_state.get("last_scores", None)
            manager = prism_state.get("manager", None)
            if scores is None or manager is None:
                break
            projected_area_ema = manager.stats.projected_area_ema.to(device=triangles.vertices.device, dtype=torch.float32)
            select_cfg = CompactionSelectionConfig(
                microbatch_active_ratio=float(getattr(comp_cfg, "microbatch_active_ratio", 0.0035)),
                candidate_pool_multiplier=float(getattr(comp_cfg, "candidate_pool_multiplier", 6.0)),
                min_prune_count=int(getattr(comp_cfg, "min_prune_count", 256)),
                roi_budget_fraction=float(getattr(comp_cfg, "roi_budget_fraction", 0.10)),
                near_field_budget_fraction=float(getattr(comp_cfg, "near_field_budget_fraction", 0.25)),
                roi_signal_threshold=float(getattr(comp_cfg, "roi_signal_threshold", 0.05)),
                near_field_area_percentile=float(getattr(comp_cfg, "near_field_area_percentile", 80.0)),
            )
            candidate_ids, selection_stats = select_prism_compaction_microbatch_ids(
                scores=scores,
                projected_area_ema=projected_area_ema,
                cfg=select_cfg,
                rejected_mask=rejected_mask,
            )
            if candidate_ids.numel() == 0:
                break

            prism_state["compaction_phase"] = PrismCompactionPhase.MICROBATCH_PRUNE
            decision = run_counterfactual_simulation(
                scene=scene,
                triangles=triangles,
                render_func=render,
                pipe=prism_state.get("pipe", None),
                background=prism_state.get("background", None),
                candidate_triangle_ids=candidate_ids,
                calibration_views=prism_state.get("calibration_views", []),
                gate_cfg=prism_state["gate_cfg"],
                proxy_ctx=prism_state.get("proxy_ctx", None),
                proxy_cfg=prism_state.get("proxy_cfg", None),
            )

            committed = False
            pruned_count = 0
            rollback = 0 if bool(decision.accept) else 1
            if bool(decision.accept):
                keep_mask = torch.ones(
                    (triangles._triangle_indices.shape[0],),
                    dtype=torch.bool,
                    device=triangles._triangle_indices.device,
                )
                valid = (candidate_ids >= 0) & (candidate_ids < keep_mask.numel())
                if torch.any(valid):
                    keep_mask[candidate_ids[valid]] = False
                    pruned_count = int(valid.sum().item())
                    triangles.prune_triangles(keep_mask)
                    _sync_prism_topology_change(
                        prism_state=prism_state,
                        triangles=triangles,
                        iteration=int(iteration),
                        reason="prism_compaction_commit",
                    )
                    committed = True
                    total_committed += int(pruned_count)
                    round_committed += int(pruned_count)
                    rejected_mask = None
                    if ground_association_tracker is not None:
                        ground_association_tracker.ensure_num_triangles(int(triangles._triangle_indices.shape[0]))
                    if _geometry_key_is_better(decision.counterfactual, best_geometry_metrics):
                        best_geometry_metrics = dict(decision.counterfactual)
                        prism_state["compaction_best_geometry_checkpoint_dir"] = _save_prism_compaction_checkpoint(
                            scene=scene,
                            triangles=triangles,
                            iteration=int(iteration),
                            tag="best_by_geometry",
                            meta={"round_idx": int(round_idx), "microbatch_idx": int(micro_idx), "metrics": decision.counterfactual},
                            prism_state=prism_state,
                        )
                    if int(triangles._triangle_indices.shape[0]) < int(best_speed_triangles):
                        best_speed_triangles = int(triangles._triangle_indices.shape[0])
                        prism_state["compaction_best_speed_checkpoint_dir"] = _save_prism_compaction_checkpoint(
                            scene=scene,
                            triangles=triangles,
                            iteration=int(iteration),
                            tag="best_by_speed",
                            meta={"round_idx": int(round_idx), "microbatch_idx": int(micro_idx), "triangle_count": int(best_speed_triangles)},
                            prism_state=prism_state,
                        )
            else:
                total_rollbacks += 1
                if rejected_mask is None:
                    rejected_mask = torch.zeros(
                        (scores.prune_score_t.numel(),),
                        dtype=torch.bool,
                        device=scores.prune_score_t.device,
                    )
                rejected_mask[candidate_ids] = True

            global_micro_idx += 1
            prism_state["counterfactual_accept"] = 1 if bool(decision.accept) else 0
            prism_state["rollback"] = int(rollback)
            log_payload = {
                "round_idx": int(round_idx),
                "microbatch_idx": int(micro_idx),
                "pruned_count": int(pruned_count),
                "triangle_count": int(triangles._triangle_indices.shape[0]),
                "counterfactual_accept": 1 if bool(decision.accept) else 0,
                "rollback": int(rollback),
                "selection": selection_stats,
                "decision": counterfactual_decision_to_dict(decision),
            }
            microbatch_logs.append(log_payload)

            debug_dir = os.path.join(scene.model_path, "prism_compaction")
            os.makedirs(debug_dir, exist_ok=True)
            with open(
                os.path.join(debug_dir, f"round_{int(round_idx):02d}_microbatch_{int(micro_idx):02d}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(log_payload, f, indent=2)

            step = int(iteration) + int(global_micro_idx)
            if tb_writer is not None:
                tb_writer.add_scalar("prism/compaction_round", float(round_idx), step)
                tb_writer.add_scalar("prism/compaction_microbatch_idx", float(micro_idx), step)
                tb_writer.add_scalar("prism/compaction_pruned_this_microbatch", float(pruned_count), step)
                tb_writer.add_scalar("prism/compaction_triangle_count", float(triangles._triangle_indices.shape[0]), step)
                tb_writer.add_scalar("prism/compaction_counterfactual_accept", float(1 if bool(decision.accept) else 0), step)
                tb_writer.add_scalar("prism/compaction_rollback", float(rollback), step)
            if wandb_run is not None:
                _wandb_log_filtered(
                    wandb_run,
                    {
                        "prism/compaction_round": float(round_idx),
                        "prism/compaction_microbatch_idx": float(micro_idx),
                        "prism/compaction_pruned_this_microbatch": float(pruned_count),
                        "prism/compaction_triangle_count": float(triangles._triangle_indices.shape[0]),
                        "prism/compaction_counterfactual_accept": float(1 if bool(decision.accept) else 0),
                        "prism/compaction_rollback": float(rollback),
                    },
                    step=step,
                    log_state=wandb_log_state,
                )

        if round_committed <= 0:
            break

    prism_state["compaction_phase"] = PrismCompactionPhase.FINALIZE
    final_ckpt = _save_prism_compaction_checkpoint(
        scene=scene,
        triangles=triangles,
        iteration=int(iteration),
        tag="final",
        meta={"triangle_count": int(triangles._triangle_indices.shape[0]), "total_committed": int(total_committed)},
        prism_state=prism_state,
    )
    prism_state["compaction_final_checkpoint_dir"] = str(final_ckpt)

    payload = {
        "ran": True,
        "source_checkpoint_dir": str(source_dir),
        "best_geometry_checkpoint_dir": str(prism_state.get("compaction_best_geometry_checkpoint_dir", "")),
        "best_speed_checkpoint_dir": str(prism_state.get("compaction_best_speed_checkpoint_dir", "")),
        "final_checkpoint_dir": str(final_ckpt),
        "total_pruned": int(total_committed),
        "total_rollbacks": int(total_rollbacks),
        "final_triangle_count": int(triangles._triangle_indices.shape[0]),
        "microbatches": microbatch_logs,
    }
    out_dir = os.path.join(scene.model_path, "prism_compaction")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "compaction_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def _prism_maybe_prune(prism_state, iteration, triangles, scene, render_pkg, prune_mode: str):
    """
    PRISM neutral prune hook.
    Returns False to keep legacy pruning path fully unchanged.
    """
    _ = (iteration, render_pkg)
    cfg = prism_state["cfg"]
    if not bool(cfg.get("enabled", False)):
        return {"committed": False, "pruned_count": 0, "counterfactual_accept": 0, "rollback": 0}
    scores = prism_state.get("last_scores", None)
    if scores is None:
        return {"committed": False, "pruned_count": 0, "counterfactual_accept": 0, "rollback": 0}
    prism_state["last_commit_relaxed_refresh_used"] = 0

    def _post_commit_relaxed_candidates():
        score_cfg = prism_state["score_cfg"]
        render_keep_block = scores.render_keep_t > float(score_cfg.keep_render_threshold)
        geometry_keep_block = scores.geometry_keep_t > float(score_cfg.keep_geometry_threshold)
        orientation_keep_block = scores.orientation_keep_t > float(score_cfg.keep_orientation_threshold)
        relaxed_risk_t = torch.stack(
            [
                scores.unc_t,
                scores.boundary_t,
                scores.nonmanifold_t,
                scores.optional_groundprotect_t,
                scores.optional_roiprotect_t,
                geometry_keep_block.to(torch.float32),
                orientation_keep_block.to(torch.float32),
                render_keep_block.to(torch.float32),
            ],
            dim=1,
        ).max(dim=1).values
        relaxed_score_t = torch.clamp(scores.redund_t * (1.0 - scores.utility_t) * (1.0 - relaxed_risk_t), 0.0, 1.0)
        min_score = float(cfg.get("post_commit_refresh_min_prune_score", 1e-6))
        score_ok = relaxed_score_t >= min_score if min_score <= 0.0 else relaxed_score_t > min_score
        relaxed_mask = (
            (~scores.dead_mask.to(torch.bool))
            & (scores.edge_t <= float(score_cfg.thresh_protected_edge))
            & (scores.geo_t <= float(score_cfg.thresh_protected_geo))
            & (scores.sens_t <= float(score_cfg.thresh_protected_sens))
            & (scores.unc_t <= float(score_cfg.thresh_protected_unc))
            & (~render_keep_block)
            & (scores.optional_groundprotect_t <= 0.0)
            & (scores.optional_roiprotect_t <= 0.0)
            & score_ok
        )
        return relaxed_mask, relaxed_score_t, {
            "candidate_diag_post_commit_relaxed_score_positive_count": int(
                torch.count_nonzero(relaxed_score_t > 0.0).item()
            ),
            "candidate_diag_post_commit_relaxed_score_mean": float(relaxed_score_t.to(torch.float32).mean().item())
            if relaxed_score_t.numel() > 0
            else 0.0,
            "candidate_diag_post_commit_relaxed_score_max": float(relaxed_score_t.to(torch.float32).max().item())
            if relaxed_score_t.numel() > 0
            else 0.0,
        }

    def _candidate_pool_diagnostics():
        try:
            total = int(scores.prune_score_t.numel())
            active_mask = scores.triangle_state == 0
            protected_raw = scores.protected_mask_raw.to(torch.bool)
            render_keep_block = scores.render_keep_t > float(prism_state["score_cfg"].keep_render_threshold)
            geometry_keep_block = scores.geometry_keep_t > float(prism_state["score_cfg"].keep_geometry_threshold)
            orientation_keep_block = scores.orientation_keep_t > float(prism_state["score_cfg"].keep_orientation_threshold)
            relaxed_mask, _, relaxed_score_payload = _post_commit_relaxed_candidates()
            payload = {
                "candidate_diag_total_triangles": total,
                "candidate_diag_active_state_count": int(torch.count_nonzero(active_mask).item()),
                "candidate_diag_protected_raw_count": int(torch.count_nonzero(protected_raw).item()),
                "candidate_diag_protected_dilated_count": int(torch.count_nonzero(scores.protected_mask_dilated).item()),
                "candidate_diag_dead_count": int(torch.count_nonzero(scores.dead_mask).item()),
                "candidate_diag_suspicious_count": int(torch.count_nonzero(scores.suspicious_mask).item()),
                "candidate_diag_block_edge_count": int(torch.count_nonzero(scores.edge_t > float(prism_state["score_cfg"].thresh_protected_edge)).item()),
                "candidate_diag_block_geo_count": int(torch.count_nonzero(scores.geo_t > float(prism_state["score_cfg"].thresh_protected_geo)).item()),
                "candidate_diag_block_sens_count": int(torch.count_nonzero(scores.sens_t > float(prism_state["score_cfg"].thresh_protected_sens)).item()),
                "candidate_diag_block_unc_count": int(torch.count_nonzero(scores.unc_t > float(prism_state["score_cfg"].thresh_protected_unc)).item()),
                "candidate_diag_block_recent_count": int(torch.count_nonzero(scores.recent_t > 0.0).item()),
                "candidate_diag_block_geometry_keep_count": int(torch.count_nonzero(geometry_keep_block).item()),
                "candidate_diag_block_orientation_keep_count": int(torch.count_nonzero(orientation_keep_block).item()),
                "candidate_diag_block_render_keep_count": int(torch.count_nonzero(render_keep_block).item()),
                "candidate_diag_block_candidate_geometry_keep_count": int(torch.count_nonzero(scores.candidate_blocked_by_geometry_keep).item()),
                "candidate_diag_block_candidate_dilated_count": int(torch.count_nonzero(scores.candidate_blocked_by_dilated_protect).item()),
                "candidate_relaxed_pool_count": int(torch.count_nonzero(relaxed_mask).item()),
            }
            payload.update(relaxed_score_payload)
            return payload
        except Exception:
            return {}

    policy_decision = None
    if prune_mode == "candidate" and bool(cfg.get("enable_adaptive_csef_policy", False)):
        policy_decision = decide_adaptive_csef_policy(
            cfg=prism_state.get("adaptive_policy_cfg", AdaptiveCSEFPolicyConfig(enabled=False)),
            prism_state=prism_state,
            scores_summary=prism_state.get("last_scores_summary", {}),
            iteration=int(iteration),
            total_triangles=int(scores.prune_score_t.numel()),
        )
        prism_state["last_adaptive_policy_decision"] = policy_decision.to_dict()

    dead_ratio = float(cfg.get("dead_prune_ratio", 0.0))
    cand_ratio = float(cfg.get("candidate_prune_ratio", 0.0))
    if prune_mode == "candidate":
        if policy_decision is not None and bool(policy_decision.enabled):
            cand_ratio = float(policy_decision.ratio)
            prism_state["adaptive_candidate_prune_ratio"] = float(cand_ratio)
        else:
            cand_ratio = float(prism_state.get("adaptive_candidate_prune_ratio", cfg.get("candidate_prune_ratio_per_round", cand_ratio)))
        dead_ratio = 0.0
    elif prune_mode == "dead":
        cand_ratio = 0.0
    candidate_max_count = 0
    if prune_mode == "candidate":
        candidate_max_count = int(cfg.get("candidate_max_count_per_round", 0))
        if policy_decision is not None and bool(policy_decision.enabled):
            candidate_max_count = int(policy_decision.candidate_max_count)
    measured_rank_enabled = (
        prune_mode == "candidate"
        and bool(cfg.get("candidate_measured_impact_rank", False))
        and bool(cfg.get("use_counterfactual_gate", False))
    )
    if (
        prune_mode == "candidate"
        and policy_decision is not None
        and bool(policy_decision.enabled)
        and bool(policy_decision.use_measured_rank)
        and bool(cfg.get("use_counterfactual_gate", False))
    ):
        measured_rank_enabled = True
    measured_final_cap = int(candidate_max_count)
    selection_max_count = candidate_max_count
    if measured_rank_enabled and measured_final_cap > 0:
        measured_pool_multiplier = max(1.0, float(cfg.get("candidate_measured_pool_multiplier", 4.0)))
        selection_max_count = max(measured_final_cap, int(round(measured_final_cap * measured_pool_multiplier)))

    rank_score_t = None
    quality_rank_enabled = bool(cfg.get("candidate_quality_rank", False))
    if policy_decision is not None and bool(policy_decision.enabled):
        quality_rank_enabled = bool(policy_decision.use_quality_rank)
    if prune_mode == "candidate" and quality_rank_enabled:
        render_penalty = float(cfg.get("candidate_quality_render_penalty", 0.5))
        geometry_penalty = float(cfg.get("candidate_quality_geometry_penalty", 0.5))
        orientation_penalty = float(cfg.get("candidate_quality_orientation_penalty", 0.25))
        utility_penalty = float(cfg.get("candidate_quality_utility_penalty", 0.25))
        uncertainty_penalty = float(cfg.get("candidate_quality_uncertainty_penalty", 0.25))
        if policy_decision is not None and bool(policy_decision.enabled):
            render_penalty = float(policy_decision.render_penalty)
            geometry_penalty = float(policy_decision.geometry_penalty)
            orientation_penalty = float(policy_decision.orientation_penalty)
            utility_penalty = float(policy_decision.utility_penalty)
            uncertainty_penalty = float(policy_decision.uncertainty_penalty)
        rank_score_t = (
            float(cfg.get("candidate_quality_prune_weight", 1.0)) * scores.prune_score_t.to(torch.float32)
            - render_penalty * scores.render_keep_t.to(torch.float32)
            - geometry_penalty * scores.geometry_keep_t.to(torch.float32)
            - orientation_penalty * scores.orientation_keep_t.to(torch.float32)
            - utility_penalty * scores.utility_t.to(torch.float32)
            - uncertainty_penalty * scores.unc_t.to(torch.float32)
        )
        rank_score_t = torch.where(
            scores.candidate_mask.to(torch.bool) | scores.dead_mask.to(torch.bool),
            rank_score_t,
            torch.full_like(rank_score_t, -1e9),
        )

    selection_ratio = cand_ratio
    if measured_rank_enabled:
        measured_pool_multiplier = max(1.0, float(cfg.get("candidate_measured_pool_multiplier", 4.0)))
        selection_ratio = float(cand_ratio) * measured_pool_multiplier
    candidate_ids = select_prism_candidate_ids(
        scores=scores,
        dead_prune_ratio=dead_ratio,
        candidate_prune_ratio=selection_ratio,
        candidate_max_count=selection_max_count,
        rank_score_t=rank_score_t,
    )
    raw_final_candidate_ids = None
    if measured_rank_enabled and measured_final_cap > 0 and candidate_ids.numel() > measured_final_cap:
        base_rank_scores = rank_score_t if rank_score_t is not None else scores.prune_score_t
        k_raw = min(int(measured_final_cap), int(candidate_ids.numel()))
        _, raw_order = torch.topk(base_rank_scores[candidate_ids].to(torch.float32), k=k_raw, largest=True, sorted=True)
        raw_final_candidate_ids = candidate_ids[raw_order]
    candidate_pool_count = int(torch.count_nonzero(scores.candidate_mask).item())
    target_count = int(max(0.0, cand_ratio) * int(scores.prune_score_t.numel()))
    capped_target_count = target_count
    if candidate_max_count > 0:
        capped_target_count = min(target_count, int(candidate_max_count))
    selected_count = int(candidate_ids.numel())
    candidate_diag_payload = _candidate_pool_diagnostics()
    relaxed_refresh_used = 0
    relaxed_pool_count = int(candidate_diag_payload.get("candidate_relaxed_pool_count", 0))
    relaxed_rank_score_t = None
    relaxed_reject_reason = ""
    relaxed_max_commits = int(cfg.get("post_commit_relaxed_max_commits", 0))
    relaxed_commit_count = int(prism_state.get("relaxed_candidate_commit_count", 0))
    relaxed_commit_cap_reached = bool(relaxed_max_commits > 0 and relaxed_commit_count >= relaxed_max_commits)
    if (
        prune_mode == "candidate"
        and selected_count == 0
        and bool(cfg.get("post_commit_candidate_refresh", False))
        and int(prism_state.get("candidate_commit_count", 0)) > 0
        and relaxed_pool_count > 0
        and not relaxed_commit_cap_reached
    ):
        relaxed_mask, relaxed_score_t, _ = _post_commit_relaxed_candidates()
        relaxed_ids = torch.nonzero(relaxed_mask, as_tuple=True)[0]
        if relaxed_ids.numel() > 0:
            relaxed_target = int(selection_max_count) if int(selection_max_count) > 0 else int(target_count)
            relaxed_target = min(max(1, relaxed_target), int(relaxed_ids.numel()))
            relaxed_scores = relaxed_score_t[relaxed_ids].to(torch.float32)
            _, relaxed_order = torch.topk(relaxed_scores, k=relaxed_target, largest=True, sorted=True)
            candidate_ids = relaxed_ids[relaxed_order]
            selected_count = int(candidate_ids.numel())
            relaxed_refresh_used = 1
            relaxed_rank_score_t = torch.full_like(relaxed_score_t.to(torch.float32), -1e9)
            relaxed_rank_score_t[relaxed_ids] = relaxed_score_t[relaxed_ids].to(torch.float32)
    elif (
        prune_mode == "candidate"
        and selected_count == 0
        and bool(cfg.get("post_commit_candidate_refresh", False))
        and int(prism_state.get("candidate_commit_count", 0)) > 0
        and relaxed_commit_cap_reached
    ):
        relaxed_reject_reason = "relaxed_commit_cap_reached"
    if candidate_ids.numel() > 1:
        order_score_t = relaxed_rank_score_t if relaxed_rank_score_t is not None else rank_score_t
        order_scores = order_score_t[candidate_ids] if order_score_t is not None else scores.prune_score_t[candidate_ids]
        _, order = torch.sort(order_scores, descending=True)
        candidate_ids = candidate_ids[order]
    if relaxed_rank_score_t is not None:
        rank_score_t = relaxed_rank_score_t
    prism_state["last_candidate_pool_count"] = candidate_pool_count
    prism_state["last_candidate_target_count"] = target_count
    prism_state["last_candidate_cap_count"] = capped_target_count
    prism_state["last_candidate_selected_count"] = selected_count
    prism_state["last_candidate_microbatch_count"] = 0
    prism_state["last_candidate_microbatch_accepted_count"] = 0
    prism_state["last_candidate_microbatch_rejected_count"] = 0
    prism_state["last_candidate_microbatch_accepted_triangles"] = 0
    prism_state["last_candidate_quality_rank_enabled"] = 1 if rank_score_t is not None else 0
    prism_state["last_candidate_quality_score_mean"] = 0.0
    prism_state["last_candidate_prune_score_mean"] = 0.0
    prism_state["last_candidate_render_keep_mean"] = 0.0
    prism_state["last_candidate_geometry_keep_mean"] = 0.0
    prism_state["last_candidate_orientation_keep_mean"] = 0.0
    prism_state["last_candidate_utility_mean"] = 0.0
    prism_state["last_candidate_uncertainty_mean"] = 0.0
    prism_state["last_candidate_measured_rank_enabled"] = 1 if measured_rank_enabled else 0
    prism_state["last_candidate_measured_group_count"] = 0
    prism_state["last_candidate_measured_accepted_count"] = 0
    prism_state["last_candidate_measured_selected_count"] = 0
    prism_state["last_candidate_measured_best_score"] = 0.0
    prism_state["last_candidate_relaxed_refresh_used"] = int(relaxed_refresh_used)
    prism_state["last_candidate_relaxed_pool_count"] = int(relaxed_pool_count)
    prism_state["last_candidate_relaxed_reject_reason"] = str(relaxed_reject_reason)
    prism_state["last_candidate_relaxed_strict_gate_pass"] = 1
    prism_state["last_candidate_relaxed_strict_gate_reason"] = ""

    def _candidate_quality_payload():
        return {
            "candidate_quality_rank_enabled": int(prism_state.get("last_candidate_quality_rank_enabled", 0)),
            "candidate_quality_score_mean": float(prism_state.get("last_candidate_quality_score_mean", 0.0)),
            "candidate_prune_score_mean": float(prism_state.get("last_candidate_prune_score_mean", 0.0)),
            "candidate_render_keep_mean": float(prism_state.get("last_candidate_render_keep_mean", 0.0)),
            "candidate_geometry_keep_mean": float(prism_state.get("last_candidate_geometry_keep_mean", 0.0)),
            "candidate_orientation_keep_mean": float(prism_state.get("last_candidate_orientation_keep_mean", 0.0)),
            "candidate_utility_mean": float(prism_state.get("last_candidate_utility_mean", 0.0)),
            "candidate_uncertainty_mean": float(prism_state.get("last_candidate_uncertainty_mean", 0.0)),
            "candidate_measured_rank_enabled": int(prism_state.get("last_candidate_measured_rank_enabled", 0)),
            "candidate_measured_group_count": int(prism_state.get("last_candidate_measured_group_count", 0)),
            "candidate_measured_accepted_count": int(prism_state.get("last_candidate_measured_accepted_count", 0)),
            "candidate_measured_selected_count": int(prism_state.get("last_candidate_measured_selected_count", 0)),
            "candidate_measured_best_score": float(prism_state.get("last_candidate_measured_best_score", 0.0)),
            "candidate_relaxed_refresh_used": int(prism_state.get("last_candidate_relaxed_refresh_used", 0)),
            "candidate_relaxed_pool_count": int(prism_state.get("last_candidate_relaxed_pool_count", 0)),
            "candidate_relaxed_reject_reason": str(prism_state.get("last_candidate_relaxed_reject_reason", "")),
            "candidate_relaxed_commit_count": int(prism_state.get("relaxed_candidate_commit_count", 0)),
            "candidate_relaxed_max_commits": int(cfg.get("post_commit_relaxed_max_commits", 0)),
            "candidate_relaxed_strict_gate_enabled": int(bool(cfg.get("post_commit_relaxed_strict_gate", False))),
            "candidate_relaxed_strict_gate_pass": int(prism_state.get("last_candidate_relaxed_strict_gate_pass", 1)),
            "candidate_relaxed_strict_gate_reason": str(
                prism_state.get("last_candidate_relaxed_strict_gate_reason", "")
            ),
            "adaptive_policy_enabled": int(bool(cfg.get("enable_adaptive_csef_policy", False))),
            "adaptive_policy_decision": dict(prism_state.get("last_adaptive_policy_decision", {}) or {}),
            **candidate_diag_payload,
        }

    if candidate_ids.numel() == 0:
        return {
            "committed": False,
            "pruned_count": 0,
            "counterfactual_accept": 0,
            "rollback": 0,
            "no_candidates": 1,
            "candidate_prune_ratio": float(cand_ratio),
            "candidate_pool_count": int(candidate_pool_count),
            "candidate_target_count": int(target_count),
            "candidate_cap_count": int(capped_target_count),
            "candidate_selected_count": int(selected_count),
            **_candidate_quality_payload(),
        }

    selected_rank_scores = rank_score_t[candidate_ids] if rank_score_t is not None else scores.prune_score_t[candidate_ids]
    prism_state["last_candidate_quality_score_mean"] = float(selected_rank_scores.to(torch.float32).mean().item())
    prism_state["last_candidate_prune_score_mean"] = float(scores.prune_score_t[candidate_ids].to(torch.float32).mean().item())
    prism_state["last_candidate_render_keep_mean"] = float(scores.render_keep_t[candidate_ids].to(torch.float32).mean().item())
    prism_state["last_candidate_geometry_keep_mean"] = float(scores.geometry_keep_t[candidate_ids].to(torch.float32).mean().item())
    prism_state["last_candidate_orientation_keep_mean"] = float(scores.orientation_keep_t[candidate_ids].to(torch.float32).mean().item())
    prism_state["last_candidate_utility_mean"] = float(scores.utility_t[candidate_ids].to(torch.float32).mean().item())
    prism_state["last_candidate_uncertainty_mean"] = float(scores.unc_t[candidate_ids].to(torch.float32).mean().item())

    counterfactual_accept = 0
    rollback = 0
    effective_gate_cfg = prism_state["gate_cfg"]
    if policy_decision is not None and bool(policy_decision.enabled):
        gate_scale = float(policy_decision.gate_scale)
        effective_gate_cfg = copy.copy(prism_state["gate_cfg"])
        effective_gate_cfg.min_delta_psnr_db = float(effective_gate_cfg.min_delta_psnr_db) * gate_scale
        effective_gate_cfg.max_delta_mae = float(effective_gate_cfg.max_delta_mae) * gate_scale
        effective_gate_cfg.max_delta_absrel = float(effective_gate_cfg.max_delta_absrel) * gate_scale
        effective_gate_cfg.max_delta_mean_angle_deg = float(effective_gate_cfg.max_delta_mean_angle_deg) * gate_scale
        effective_gate_cfg.max_changed_pixel_ratio = float(effective_gate_cfg.max_changed_pixel_ratio) * gate_scale

    def _finite_delta(deltas, key: str, default: float = 0.0) -> float:
        try:
            value = float((deltas or {}).get(key, default))
        except Exception:
            return float(default)
        return value if np.isfinite(value) else float(default)

    def _relaxed_strict_gate_decision(decision):
        if not int(relaxed_refresh_used):
            return True, "not_relaxed_refresh"
        if not bool(cfg.get("post_commit_relaxed_strict_gate", False)):
            return True, "strict_gate_disabled"
        deltas = dict(getattr(decision, "deltas", {}) or {})
        checks = [
            (
                "delta_psnr",
                _finite_delta(deltas, "delta_psnr"),
                ">=",
                float(cfg.get("post_commit_relaxed_min_delta_psnr", 0.0)),
            ),
            (
                "delta_mae",
                _finite_delta(deltas, "delta_mae"),
                "<=",
                float(cfg.get("post_commit_relaxed_max_delta_mae", 0.0)),
            ),
            (
                "delta_absrel",
                _finite_delta(deltas, "delta_absrel"),
                "<=",
                float(cfg.get("post_commit_relaxed_max_delta_absrel", 0.0)),
            ),
            (
                "delta_mean_angle",
                _finite_delta(deltas, "delta_mean_angle"),
                "<=",
                float(cfg.get("post_commit_relaxed_max_delta_mean_angle", 0.0)),
            ),
            (
                "changed_pixel_ratio",
                _finite_delta(deltas, "changed_pixel_ratio"),
                "<=",
                float(cfg.get("post_commit_relaxed_max_changed_pixel_ratio", 0.0025)),
            ),
        ]
        failures = []
        for key, value, op, threshold in checks:
            ok = value >= threshold if op == ">=" else value <= threshold
            if not bool(ok):
                failures.append(f"{key}{op}{threshold:g}_got_{value:g}")
        if failures:
            return False, "relaxed_strict_gate_reject:" + ",".join(failures)
        return True, "relaxed_strict_gate_pass"

    if (prune_mode == "candidate") and bool(cfg.get("use_counterfactual_gate", False)):
        debug_dir = os.path.join(prism_state["model_path"], "prism_debug")
        os.makedirs(debug_dir, exist_ok=True)
        candidate_arbitration_selected_policy = ""
        if measured_rank_enabled and measured_final_cap > 0 and candidate_ids.numel() > measured_final_cap:
            group_size = max(1, int(cfg.get("candidate_measured_group_size", 256)))
            max_groups = max(0, int(cfg.get("candidate_measured_max_groups", 8)))
            if policy_decision is not None and bool(policy_decision.enabled) and bool(policy_decision.use_measured_rank):
                group_size = max(group_size, int(policy_decision.microbatch_size))
                if max_groups > 0:
                    min_cover = int(np.ceil(float(measured_final_cap) / float(max(1, group_size))))
                    max_groups = max(max_groups, min_cover + 1)
            groups = list(torch.split(candidate_ids, group_size))
            if max_groups > 0:
                groups = groups[:max_groups]

            def _finite(v, default=0.0):
                try:
                    value = float(v)
                except Exception:
                    return float(default)
                return value if np.isfinite(value) else float(default)

            measured_logs = []
            for group_idx, group_ids in enumerate(groups):
                decision = run_counterfactual_simulation(
                    scene=scene,
                    triangles=triangles,
                    render_func=render,
                    pipe=prism_state.get("pipe", None),
                    background=prism_state.get("background", None),
                    candidate_triangle_ids=group_ids,
                    calibration_views=prism_state.get("calibration_views", []),
                    gate_cfg=effective_gate_cfg,
                    proxy_ctx=prism_state.get("proxy_ctx", None),
                    proxy_cfg=prism_state.get("proxy_cfg", None),
                )
                deltas = dict(getattr(decision, "deltas", {}) or {})
                delta_psnr = _finite(deltas.get("delta_psnr", 0.0))
                delta_mae = _finite(deltas.get("delta_mae", 0.0))
                delta_absrel = _finite(deltas.get("delta_absrel", 0.0))
                delta_normal = _finite(deltas.get("delta_mean_angle", 0.0))
                changed_pixel = _finite(deltas.get("changed_pixel_ratio", 0.0))
                measured_score = (
                    (1000.0 if bool(decision.accept) else -1000.0)
                    + 250.0 * max(0.0, -delta_absrel)
                    + 60.0 * max(0.0, -delta_mae)
                    + 2.0 * max(0.0, -delta_normal)
                    + 2.0 * max(0.0, delta_psnr)
                    - 2200.0 * max(0.0, delta_absrel)
                    - 1400.0 * max(0.0, delta_mae)
                    - 4.0 * max(0.0, delta_normal)
                    - 0.8 * max(0.0, -delta_psnr)
                    - 120.0 * max(0.0, changed_pixel)
                )
                payload = counterfactual_decision_to_dict(decision)
                payload["measured_group_index"] = int(group_idx)
                payload["measured_group_size"] = int(group_ids.numel())
                payload["measured_score"] = float(measured_score)
                with open(
                    os.path.join(debug_dir, f"counterfactual_measured_rank_iter_{int(iteration):06d}_g{group_idx:03d}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(payload, f, indent=2)
                measured_logs.append(
                    {
                        "group_idx": int(group_idx),
                        "group_ids": group_ids,
                        "accept": bool(decision.accept),
                        "score": float(measured_score),
                        "delta_absrel": float(delta_absrel),
                        "delta_mae": float(delta_mae),
                        "delta_normal": float(delta_normal),
                        "delta_psnr": float(delta_psnr),
                        "changed_pixel_ratio": float(changed_pixel),
                    }
                )

            measured_logs = sorted(
                measured_logs,
                key=lambda x: (
                    bool(x["accept"]),
                    -max(0.0, float(x["delta_absrel"])),
                    -max(0.0, float(x["delta_mae"])),
                    -max(0.0, float(x["delta_normal"])),
                    float(x["score"]),
                ),
                reverse=True,
            )
            selected_groups = []
            selected_total = 0
            for entry in measured_logs:
                if selected_total >= measured_final_cap:
                    break
                group_ids = entry["group_ids"]
                remaining = max(0, measured_final_cap - selected_total)
                if group_ids.numel() > remaining:
                    group_ids = group_ids[:remaining]
                if group_ids.numel() <= 0:
                    continue
                selected_groups.append(group_ids)
                selected_total += int(group_ids.numel())
            if selected_groups:
                # The selected groups are disjoint slices from the ranked candidate list.
                # Preserve that rank order for the downstream counterfactual microbatch gate;
                # torch.unique would sort ids and silently destroy the policy ordering.
                candidate_ids = torch.cat(selected_groups, dim=0)
                if candidate_ids.numel() > measured_final_cap:
                    candidate_ids = candidate_ids[:measured_final_cap]
                if candidate_ids.numel() < measured_final_cap:
                    selected_mask = torch.zeros(
                        int(scores.prune_score_t.numel()),
                        dtype=torch.bool,
                        device=candidate_ids.device,
                    )
                    selected_mask[candidate_ids] = True
                    remaining_ids = candidate_ids.new_zeros((0,))
                    for entry in measured_logs:
                        group_ids = entry["group_ids"]
                        group_ids = group_ids[~selected_mask[group_ids]]
                        if group_ids.numel() <= 0:
                            continue
                        remaining = max(0, measured_final_cap - int(candidate_ids.numel()) - int(remaining_ids.numel()))
                        if remaining <= 0:
                            break
                        remaining_ids = torch.cat([remaining_ids, group_ids[:remaining]], dim=0)
                        selected_mask[group_ids[:remaining]] = True
                    if remaining_ids.numel() > 0:
                        candidate_ids = torch.cat([candidate_ids, remaining_ids], dim=0)
                    if candidate_ids.numel() > measured_final_cap:
                        candidate_ids = candidate_ids[:measured_final_cap]
                selected_count = int(candidate_ids.numel())
                prism_state["last_candidate_selected_count"] = int(selected_count)
                prism_state["last_candidate_measured_selected_count"] = int(selected_count)
                if raw_final_candidate_ids is not None and raw_final_candidate_ids.numel() > 0:
                    def _policy_set_score(decision):
                        deltas = dict(getattr(decision, "deltas", {}) or {})
                        delta_psnr = _finite(deltas.get("delta_psnr", 0.0))
                        delta_mae = _finite(deltas.get("delta_mae", 0.0))
                        delta_absrel = _finite(deltas.get("delta_absrel", 0.0))
                        delta_normal = _finite(deltas.get("delta_mean_angle", 0.0))
                        changed_pixel = _finite(deltas.get("changed_pixel_ratio", 0.0))
                        return (
                            (1000.0 if bool(decision.accept) else -1000.0)
                            + 800.0 * max(0.0, -delta_absrel)
                            + 80.0 * max(0.0, -delta_mae)
                            + 3.0 * max(0.0, -delta_normal)
                            + 2.0 * max(0.0, delta_psnr)
                            - 2600.0 * max(0.0, delta_absrel)
                            - 1600.0 * max(0.0, delta_mae)
                            - 5.0 * max(0.0, delta_normal)
                            - 1.0 * max(0.0, -delta_psnr)
                            - 160.0 * max(0.0, changed_pixel)
                        )

                    measured_decision = run_counterfactual_simulation(
                        scene=scene,
                        triangles=triangles,
                        render_func=render,
                        pipe=prism_state.get("pipe", None),
                        background=prism_state.get("background", None),
                        candidate_triangle_ids=candidate_ids,
                        calibration_views=prism_state.get("calibration_views", []),
                        gate_cfg=effective_gate_cfg,
                        proxy_ctx=prism_state.get("proxy_ctx", None),
                        proxy_cfg=prism_state.get("proxy_cfg", None),
                    )
                    raw_decision = run_counterfactual_simulation(
                        scene=scene,
                        triangles=triangles,
                        render_func=render,
                        pipe=prism_state.get("pipe", None),
                        background=prism_state.get("background", None),
                        candidate_triangle_ids=raw_final_candidate_ids,
                        calibration_views=prism_state.get("calibration_views", []),
                        gate_cfg=effective_gate_cfg,
                        proxy_ctx=prism_state.get("proxy_ctx", None),
                        proxy_cfg=prism_state.get("proxy_cfg", None),
                    )
                    measured_set_score = float(_policy_set_score(measured_decision))
                    raw_set_score = float(_policy_set_score(raw_decision))
                    arbitration_payload = {
                        "iteration": int(iteration),
                        "measured_count": int(candidate_ids.numel()),
                        "raw_count": int(raw_final_candidate_ids.numel()),
                        "measured_score": measured_set_score,
                        "raw_score": raw_set_score,
                        "measured_decision": counterfactual_decision_to_dict(measured_decision),
                        "raw_decision": counterfactual_decision_to_dict(raw_decision),
                        "selected_policy": "measured",
                    }
                    if raw_set_score > measured_set_score:
                        candidate_ids = raw_final_candidate_ids
                        selected_count = int(candidate_ids.numel())
                        prism_state["last_candidate_selected_count"] = int(selected_count)
                        prism_state["last_candidate_measured_selected_count"] = int(selected_count)
                        arbitration_payload["selected_policy"] = "raw_fallback"
                    candidate_arbitration_selected_policy = str(arbitration_payload["selected_policy"])
                    with open(
                        os.path.join(debug_dir, f"counterfactual_policy_arbitration_iter_{int(iteration):06d}.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(arbitration_payload, f, indent=2)
            prism_state["last_candidate_measured_group_count"] = int(len(measured_logs))
            prism_state["last_candidate_measured_accepted_count"] = int(sum(1 for x in measured_logs if bool(x["accept"])))
            prism_state["last_candidate_measured_best_score"] = (
                float(measured_logs[0]["score"]) if len(measured_logs) > 0 else 0.0
            )
            selected_rank_scores = rank_score_t[candidate_ids] if rank_score_t is not None else scores.prune_score_t[candidate_ids]
            prism_state["last_candidate_quality_score_mean"] = float(selected_rank_scores.to(torch.float32).mean().item())
            prism_state["last_candidate_prune_score_mean"] = float(scores.prune_score_t[candidate_ids].to(torch.float32).mean().item())
            prism_state["last_candidate_render_keep_mean"] = float(scores.render_keep_t[candidate_ids].to(torch.float32).mean().item())
            prism_state["last_candidate_geometry_keep_mean"] = float(scores.geometry_keep_t[candidate_ids].to(torch.float32).mean().item())
            prism_state["last_candidate_orientation_keep_mean"] = float(scores.orientation_keep_t[candidate_ids].to(torch.float32).mean().item())
            prism_state["last_candidate_utility_mean"] = float(scores.utility_t[candidate_ids].to(torch.float32).mean().item())
            prism_state["last_candidate_uncertainty_mean"] = float(scores.unc_t[candidate_ids].to(torch.float32).mean().item())

        policy_microbatch_enabled = (
            policy_decision is not None
            and bool(policy_decision.enabled)
            and bool(policy_decision.use_microbatch_gate)
            and candidate_arbitration_selected_policy != "raw_fallback"
        )
        if bool(cfg.get("candidate_microbatch_gate", False)) or bool(policy_microbatch_enabled):
            microbatch_size = max(1, int(cfg.get("candidate_microbatch_size", 256)))
            max_batches = max(0, int(cfg.get("candidate_microbatch_max_batches", 0)))
            if policy_microbatch_enabled:
                microbatch_size = max(1, int(policy_decision.microbatch_size))
                max_batches = max(0, int(policy_decision.microbatch_max_batches))
            batches = list(torch.split(candidate_ids, microbatch_size))
            if max_batches > 0:
                batches = batches[:max_batches]
            accepted_batches = []
            rejected_count = 0
            for batch_idx, batch_ids in enumerate(batches):
                if batch_ids.numel() == 0:
                    continue
                if accepted_batches:
                    # Accepted batches are disjoint slices from candidate_ids. Keep policy order
                    # stable for cumulative counterfactual checks.
                    cumulative_ids = torch.cat(accepted_batches + [batch_ids], dim=0)
                else:
                    cumulative_ids = batch_ids
                decision = run_counterfactual_simulation(
                    scene=scene,
                    triangles=triangles,
                    render_func=render,
                    pipe=prism_state.get("pipe", None),
                    background=prism_state.get("background", None),
                    candidate_triangle_ids=cumulative_ids,
                    calibration_views=prism_state.get("calibration_views", []),
                    gate_cfg=effective_gate_cfg,
                    proxy_ctx=prism_state.get("proxy_ctx", None),
                    proxy_cfg=prism_state.get("proxy_cfg", None),
                )
                prism_state["last_counterfactual_decision"] = decision
                payload = counterfactual_decision_to_dict(decision)
                payload["microbatch_index"] = int(batch_idx)
                payload["microbatch_size"] = int(batch_ids.numel())
                payload["cumulative_candidate_count"] = int(cumulative_ids.numel())
                with open(
                    os.path.join(debug_dir, f"counterfactual_gate_iter_{int(iteration):06d}_mb{batch_idx:03d}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(payload, f, indent=2)
                if bool(decision.accept):
                    accepted_batches.append(batch_ids)
                else:
                    rejected_count += 1
            accepted_ids = (
                torch.cat(accepted_batches, dim=0)
                if accepted_batches
                else torch.zeros((0,), dtype=torch.int64, device=candidate_ids.device)
            )
            accepted_batch_count = len(accepted_batches)
            prism_state["last_candidate_microbatch_count"] = len(batches)
            prism_state["last_candidate_microbatch_accepted_count"] = accepted_batch_count
            prism_state["last_candidate_microbatch_rejected_count"] = int(rejected_count)
            prism_state["last_candidate_microbatch_accepted_triangles"] = int(accepted_ids.numel())
            if accepted_ids.numel() == 0:
                rollback = 1
                return {
                    "committed": False,
                    "pruned_count": 0,
                    "counterfactual_accept": 0,
                    "rollback": rollback,
                    "candidate_prune_ratio": float(cand_ratio),
                    "candidate_pool_count": int(candidate_pool_count),
                    "candidate_target_count": int(target_count),
                    "candidate_cap_count": int(capped_target_count),
                    "candidate_selected_count": int(selected_count),
                    "candidate_microbatch_count": int(len(batches)),
                    "candidate_microbatch_accepted_count": int(accepted_batch_count),
                    "candidate_microbatch_rejected_count": int(rejected_count),
                    "candidate_microbatch_accepted_triangles": 0,
                    **_candidate_quality_payload(),
                }
            candidate_ids = accepted_ids
            counterfactual_accept = 1
        else:
            decision = run_counterfactual_simulation(
                scene=scene,
                triangles=triangles,
                render_func=render,
                pipe=prism_state.get("pipe", None),
                background=prism_state.get("background", None),
                candidate_triangle_ids=candidate_ids,
                calibration_views=prism_state.get("calibration_views", []),
                gate_cfg=effective_gate_cfg,
                proxy_ctx=prism_state.get("proxy_ctx", None),
                proxy_cfg=prism_state.get("proxy_cfg", None),
            )
            prism_state["last_counterfactual_decision"] = decision
            with open(os.path.join(debug_dir, f"counterfactual_gate_iter_{int(iteration):06d}.json"), "w", encoding="utf-8") as f:
                json.dump(counterfactual_decision_to_dict(decision), f, indent=2)
            if not bool(decision.accept):
                rollback = 1
                return {
                    "committed": False,
                    "pruned_count": 0,
                    "counterfactual_accept": 0,
                    "rollback": rollback,
                    "candidate_prune_ratio": float(cand_ratio),
                    "candidate_pool_count": int(candidate_pool_count),
                    "candidate_target_count": int(target_count),
                    "candidate_cap_count": int(capped_target_count),
                    "candidate_selected_count": int(selected_count),
                    "candidate_microbatch_count": 0,
                    "candidate_microbatch_accepted_count": 0,
                    "candidate_microbatch_rejected_count": 0,
                    "candidate_microbatch_accepted_triangles": 0,
                    **_candidate_quality_payload(),
                }
            strict_ok, strict_reason = _relaxed_strict_gate_decision(decision)
            prism_state["last_candidate_relaxed_strict_gate_pass"] = 1 if bool(strict_ok) else 0
            prism_state["last_candidate_relaxed_strict_gate_reason"] = str(strict_reason)
            if not bool(strict_ok):
                rollback = 1
                reject_payload = counterfactual_decision_to_dict(decision)
                reject_payload["relaxed_strict_gate_pass"] = False
                reject_payload["relaxed_strict_gate_reason"] = str(strict_reason)
                with open(
                    os.path.join(debug_dir, f"counterfactual_relaxed_strict_gate_iter_{int(iteration):06d}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(reject_payload, f, indent=2)
                return {
                    "committed": False,
                    "pruned_count": 0,
                    "counterfactual_accept": 0,
                    "rollback": rollback,
                    "candidate_prune_ratio": float(cand_ratio),
                    "candidate_pool_count": int(candidate_pool_count),
                    "candidate_target_count": int(target_count),
                    "candidate_cap_count": int(capped_target_count),
                    "candidate_selected_count": int(selected_count),
                    "candidate_microbatch_count": 0,
                    "candidate_microbatch_accepted_count": 0,
                    "candidate_microbatch_rejected_count": 0,
                    "candidate_microbatch_accepted_triangles": 0,
                    **_candidate_quality_payload(),
                }
            counterfactual_accept = 1

    # Optional teacher cache: snapshot round-pre-prune render targets.
    _build_prism_teacher_cache(prism_state=prism_state, scene=scene, triangles=triangles)

    keep_mask = torch.ones(
        (triangles._triangle_indices.shape[0],),
        dtype=torch.bool,
        device=triangles._triangle_indices.device,
    )
    pre_commit_triangle_count = int(triangles._triangle_indices.shape[0])
    valid = (candidate_ids >= 0) & (candidate_ids < keep_mask.numel())
    if torch.any(valid):
        pruned_count = int(valid.sum().item())
        keep_mask[candidate_ids[valid]] = False
        triangles.prune_triangles(keep_mask)
        _sync_prism_topology_change(
            prism_state=prism_state,
            triangles=triangles,
            iteration=int(iteration),
            reason=f"prism_{str(prune_mode)}_commit",
        )
        if str(prune_mode) == "candidate":
            prism_state["candidate_commit_count"] = int(prism_state.get("candidate_commit_count", 0)) + 1
            if int(relaxed_refresh_used):
                prism_state["last_commit_relaxed_refresh_used"] = 1
                prism_state["relaxed_candidate_commit_count"] = int(
                    prism_state.get("relaxed_candidate_commit_count", 0)
                ) + 1
                records = list(prism_state.get("relaxed_commit_records", []))
                records.append(
                    {
                        "iteration": int(iteration),
                        "pre_commit_triangle_count": int(pre_commit_triangle_count),
                        "post_commit_triangle_count": int(triangles._triangle_indices.shape[0]),
                        "pruned_count": int(pruned_count),
                        "strict_gate_enabled": bool(cfg.get("post_commit_relaxed_strict_gate", False)),
                        "strict_gate_pass": int(prism_state.get("last_candidate_relaxed_strict_gate_pass", 1)),
                        "strict_gate_reason": str(prism_state.get("last_candidate_relaxed_strict_gate_reason", "")),
                    }
                )
                prism_state["relaxed_commit_records"] = records
        return {
            "committed": True,
            "pruned_count": pruned_count,
            "counterfactual_accept": int(counterfactual_accept),
            "rollback": int(rollback),
            "candidate_prune_ratio": float(cand_ratio),
            "candidate_pool_count": int(candidate_pool_count),
            "candidate_target_count": int(target_count),
            "candidate_cap_count": int(capped_target_count),
            "candidate_selected_count": int(selected_count),
            "candidate_microbatch_count": int(prism_state.get("last_candidate_microbatch_count", 0)),
            "candidate_microbatch_accepted_count": int(prism_state.get("last_candidate_microbatch_accepted_count", 0)),
            "candidate_microbatch_rejected_count": int(prism_state.get("last_candidate_microbatch_rejected_count", 0)),
            "candidate_microbatch_accepted_triangles": int(prism_state.get("last_candidate_microbatch_accepted_triangles", 0)),
            **_candidate_quality_payload(),
        }
    return {
        "committed": False,
        "pruned_count": 0,
        "counterfactual_accept": int(counterfactual_accept),
        "rollback": int(rollback),
        "candidate_prune_ratio": float(cand_ratio),
        "candidate_pool_count": int(candidate_pool_count),
        "candidate_target_count": int(target_count),
        "candidate_cap_count": int(capped_target_count),
        "candidate_selected_count": int(selected_count),
        "candidate_microbatch_count": int(prism_state.get("last_candidate_microbatch_count", 0)),
        "candidate_microbatch_accepted_count": int(prism_state.get("last_candidate_microbatch_accepted_count", 0)),
        "candidate_microbatch_rejected_count": int(prism_state.get("last_candidate_microbatch_rejected_count", 0)),
        "candidate_microbatch_accepted_triangles": int(prism_state.get("last_candidate_microbatch_accepted_triangles", 0)),
        **_candidate_quality_payload(),
    }


def training(
        dataset,   
        opt, 
        pipe,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
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
        wandb_scalar_log_interval=10,
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
    teacher_render_cache = {}
    if bool(getattr(opt, "enable_teacher_render_loss", False)):
        teacher_render_cache = _load_teacher_render_cache(
            render_dir=str(getattr(opt, "teacher_render_dir", "")),
            train_cameras=scene.getTrainCameras(),
        )
        if len(teacher_render_cache) == 0:
            print("[TeacherRender] disabled for this run: no teacher renders were loaded.")
    checkpoint_geometry_anchor_vertices = None
    if bool(getattr(opt, "enable_checkpoint_geometry_anchor", False)):
        checkpoint_geometry_anchor_vertices = triangles.vertices.detach().clone()
        print(
            "[CheckpointGeometryAnchor] enabled: "
            f"vertices={int(checkpoint_geometry_anchor_vertices.shape[0])}, "
            f"lambda={float(getattr(opt, 'lambda_checkpoint_geometry_anchor', 0.0))}"
        )
    checkpoint_render_geometry_cache = {}
    if bool(getattr(opt, "enable_checkpoint_render_geometry_anchor", False)):
        checkpoint_render_geometry_cache = _build_checkpoint_render_geometry_cache(
            train_cameras=scene.getTrainCameras(),
            triangles=triangles,
            pipe=pipe,
            background=background,
        )
        print(
            "[CheckpointRenderGeometryAnchor] enabled: "
            f"views={len(checkpoint_render_geometry_cache)}, "
            f"depth_lambda={float(getattr(opt, 'lambda_checkpoint_render_depth_anchor', 0.0))}, "
            f"normal_lambda={float(getattr(opt, 'lambda_checkpoint_render_normal_anchor', 0.0))}"
        )

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    wandb_log_state = {"last_values": {}}

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
        smooth_tri_adj_max_triangles=int(getattr(opt, "ground_smooth_tri_adj_max_triangles", 4096)),
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

    sparse_colmap_depth_proxy_ctx = None
    sparse_colmap_depth_proxy_cfg = GeometryProxyConfig(
        max_points_per_view=int(getattr(opt, "prism_proxy_max_points_per_view", 3000)),
        point_error_max=float(getattr(opt, "prism_proxy_point_error_max", 2.0)),
        normal_knn=int(getattr(opt, "prism_proxy_normal_knn", 24)),
        compute_normal=False,
        seed=7,
        sample_mode=str(getattr(opt, "sparse_colmap_depth_sample_mode", "random")),
        low_error_fraction=float(getattr(opt, "sparse_colmap_depth_low_error_fraction", 1.0)),
    )
    sparse_colmap_depth_rng = np.random.default_rng(7)
    if bool(getattr(opt, "enable_sparse_colmap_depth_loss", False)):
        colmap_points = getattr(scene.scene_info, "colmap_points3d", None)
        if colmap_points is None or len(colmap_points) == 0:
            print("[SparseCOLMAPDepth] disabled: no COLMAP sparse points in scene.")
        else:
            cam_infos = []
            cam_infos.extend(list(getattr(scene.scene_info, "train_cameras", []) or []))
            cam_infos.extend(list(getattr(scene.scene_info, "test_cameras", []) or []))
            try:
                sparse_colmap_depth_proxy_ctx = build_geometry_proxy_context(
                    colmap_points3d=colmap_points,
                    cam_infos=cam_infos,
                    cfg=sparse_colmap_depth_proxy_cfg,
                )
                print("[SparseCOLMAPDepth] enabled: context initialized.")
            except Exception as exc:
                print(f"[SparseCOLMAPDepth] disabled: failed to build context ({exc})")
                sparse_colmap_depth_proxy_ctx = None
    sparse_colmap_depth_debug = {
        "last_valid_matches": 0,
        "last_lambda": 0.0,
        "last_loss": 0.0,
        "last_reason": "disabled",
    }
    sparse_parent_rollback_cache = None
    sparse_parent_rollback_cache_split = ""
    if bool(getattr(opt, "enable_sparse_depth_parent_rollback_loss", False)):
        sparse_parent_rollback_cache = load_sparse_depth_parent_rollback_cache(
            getattr(opt, "sparse_depth_parent_rollback_cache", ""),
            allow_test_cache=bool(getattr(opt, "sparse_depth_parent_rollback_allow_test_cache", False)),
        )
        sparse_parent_rollback_cache_split = str(
            sparse_parent_rollback_cache.get("manifest", {}).get("split", "")
        )
        print(
            "[SparseParentRollback] enabled: "
            f"split={sparse_parent_rollback_cache_split}, "
            f"views={len(sparse_parent_rollback_cache.get('by_image_key', {}))}, "
            f"lambda={float(getattr(opt, 'lambda_sparse_depth_parent_rollback', 0.0))}"
        )

    prism_state = _prepare_prism_state(
        opt=opt,
        scene=scene,
        triangles=triangles,
        init_iter=first_iter,
        dataset=dataset,
        pipe=pipe,
        background=background,
    )

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
        phase_info = {
            "phase": PrismPhase.FINAL_FINE_TUNE,
            "collect_stats": True,
            "allow_topology_mutation": True,
            "should_attempt_prune": False,
            "prune_mode": None,
            "post_commit_recollect_remaining": 0,
        }
        controller = prism_state.get("controller", None)
        if controller is not None:
            phase_info = controller.step(iteration=iteration)
            prism_state["current_phase"] = phase_info["phase"]
            if bool(controller.consume_force_recompute_flag()):
                prism_state["_force_recompute_scores_after_recollect"] = True

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
            _sync_prism_topology_change(
                prism_state=prism_state,
                triangles=triangles,
                iteration=int(iteration),
                reason="restricted_delaunay",
            )
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
        prism_state["latest_assoc_stats"] = assoc_stats
        _update_prism_state(
            prism_state=prism_state,
            iteration=iteration,
            render_pkg=render_pkg,
            viewpoint_cam=viewpoint_cam,
            triangles=triangles,
        ) if bool(phase_info.get("collect_stats", True)) else None

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
        lpips_loss_pure = 0
        lpips_loss_weighted = 0
        lpips_loss_lambda = _lpips_loss_lambda(iteration=int(iteration), opt=opt)
        if lpips_loss_lambda > 0.0:
            lpips_loss_pure = _compute_lpips_training_loss(
                image=image,
                gt_image=gt_image,
                max_side=int(getattr(opt, "lpips_loss_max_side", 512)),
            )
            lpips_loss_weighted = float(lpips_loss_lambda) * lpips_loss_pure
            if torch.is_tensor(lpips_loss_weighted) and torch.isfinite(lpips_loss_weighted):
                loss += lpips_loss_weighted
        teacher_render_loss_pure = 0
        teacher_render_loss = 0
        teacher_render_l1 = 0
        teacher_render_ssim = 0
        teacher_render_mask_fraction = 0
        teacher_render_lambda = _teacher_render_lambda(iteration=int(iteration), opt=opt)
        teacher_render_res = _compute_teacher_render_loss(
            viewpoint_cam=viewpoint_cam,
            image=image,
            gt_image=gt_image,
            teacher_cache=teacher_render_cache,
            lam=float(teacher_render_lambda),
            dssim_weight=float(getattr(opt, "teacher_render_dssim", 0.2)),
            mask_mode=str(getattr(opt, "teacher_render_mask_mode", "none")),
            error_margin=float(getattr(opt, "teacher_render_error_margin", 0.0)),
        )
        if teacher_render_res is not None:
            teacher_render_loss_pure = teacher_render_res["loss_pure"]
            teacher_render_loss = teacher_render_res["loss_weighted"]
            teacher_render_l1 = teacher_render_res["l1"]
            teacher_render_ssim = teacher_render_res["ssim"] if teacher_render_res["ssim"] is not None else 0
            teacher_render_mask_fraction = float(teacher_render_res.get("mask_fraction", 1.0))
            if torch.is_tensor(teacher_render_loss) and torch.isfinite(teacher_render_loss):
                loss += teacher_render_loss

        checkpoint_geometry_anchor_loss_pure = 0
        checkpoint_geometry_anchor_loss = 0
        checkpoint_geometry_anchor_mean_disp = 0
        checkpoint_geometry_anchor_max_disp = 0
        checkpoint_geometry_anchor_lambda = _checkpoint_geometry_anchor_lambda(iteration=int(iteration), opt=opt)
        checkpoint_geometry_anchor_res = _compute_checkpoint_geometry_anchor_loss(
            triangles=triangles,
            anchor_vertices=checkpoint_geometry_anchor_vertices,
            lam=float(checkpoint_geometry_anchor_lambda),
            huber_delta=float(getattr(opt, "checkpoint_geometry_anchor_huber_delta", 0.01)),
        )
        if checkpoint_geometry_anchor_res is not None:
            checkpoint_geometry_anchor_loss_pure = checkpoint_geometry_anchor_res["loss_pure"]
            checkpoint_geometry_anchor_loss = checkpoint_geometry_anchor_res["loss_weighted"]
            checkpoint_geometry_anchor_mean_disp = checkpoint_geometry_anchor_res["mean_displacement"]
            checkpoint_geometry_anchor_max_disp = checkpoint_geometry_anchor_res["max_displacement"]
            if torch.is_tensor(checkpoint_geometry_anchor_loss) and torch.isfinite(checkpoint_geometry_anchor_loss):
                loss += checkpoint_geometry_anchor_loss

        checkpoint_render_geometry_anchor_loss = 0
        checkpoint_render_depth_anchor_loss_pure = 0
        checkpoint_render_normal_anchor_loss_pure = 0
        checkpoint_render_depth_anchor_lambda = _checkpoint_render_geometry_anchor_lambda(
            iteration=int(iteration), opt=opt, kind="depth"
        )
        checkpoint_render_normal_anchor_lambda = _checkpoint_render_geometry_anchor_lambda(
            iteration=int(iteration), opt=opt, kind="normal"
        )
        checkpoint_render_geometry_anchor_res = _compute_checkpoint_render_geometry_anchor_loss(
            render_pkg=render_pkg,
            viewpoint_cam=viewpoint_cam,
            cache=checkpoint_render_geometry_cache,
            depth_lam=float(checkpoint_render_depth_anchor_lambda),
            normal_lam=float(checkpoint_render_normal_anchor_lambda),
            huber_delta=float(getattr(opt, "checkpoint_render_geometry_anchor_huber_delta", 0.02)),
        )
        if checkpoint_render_geometry_anchor_res is not None:
            checkpoint_render_geometry_anchor_loss = checkpoint_render_geometry_anchor_res["loss_weighted"]
            checkpoint_render_depth_anchor_loss_pure = checkpoint_render_geometry_anchor_res["depth_pure"]
            checkpoint_render_normal_anchor_loss_pure = checkpoint_render_geometry_anchor_res["normal_pure"]
            if torch.is_tensor(checkpoint_render_geometry_anchor_loss) and torch.isfinite(checkpoint_render_geometry_anchor_loss):
                loss += checkpoint_render_geometry_anchor_loss

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

        # Sparse COLMAP depth supervision (train-view sparse correspondences).
        sparse_colmap_depth_loss_pure = 0
        sparse_colmap_depth_loss = 0
        sparse_colmap_depth_valid_matches = 0
        sparse_colmap_depth_lambda = 0.0
        sparse_colmap_depth_reason = "disabled"
        current_phase = phase_info.get("phase", PrismPhase.FINAL_FINE_TUNE)
        sparse_colmap_depth_lambda = _sparse_colmap_depth_lambda(
            iteration=int(iteration),
            current_phase=current_phase,
            prism_enabled=bool(prism_state.get("cfg", {}).get("enabled", False)),
            opt=opt,
        )
        if sparse_colmap_depth_proxy_ctx is not None and sparse_colmap_depth_lambda > 0.0:
            sparse_res = _compute_sparse_colmap_depth_loss(
                viewpoint_cam=viewpoint_cam,
                render_pkg=render_pkg,
                proxy_ctx=sparse_colmap_depth_proxy_ctx,
                proxy_cfg=sparse_colmap_depth_proxy_cfg,
                min_matches=int(getattr(opt, "sparse_colmap_depth_min_matches", 32)),
                lam=float(sparse_colmap_depth_lambda),
                loss_space=str(getattr(opt, "sparse_colmap_depth_loss_space", "depth")),
                robust_beta=float(getattr(opt, "sparse_colmap_depth_robust_beta", 0.05)),
                rng=sparse_colmap_depth_rng,
            )
            if sparse_res is not None:
                sparse_colmap_depth_valid_matches = int(sparse_res.get("valid_matches", 0))
                sparse_colmap_depth_reason = str(sparse_res.get("reason", "unknown"))
                sparse_colmap_depth_loss_pure = sparse_res.get("loss_pure", 0)
                sparse_colmap_depth_loss = sparse_res.get("loss_weighted", 0)
                if sparse_colmap_depth_loss is not None and torch.is_tensor(sparse_colmap_depth_loss) and torch.isfinite(sparse_colmap_depth_loss):
                    loss += sparse_colmap_depth_loss
        sparse_colmap_depth_debug["last_valid_matches"] = int(sparse_colmap_depth_valid_matches)
        sparse_colmap_depth_debug["last_lambda"] = float(sparse_colmap_depth_lambda)
        sparse_colmap_depth_debug["last_loss"] = (
            float(sparse_colmap_depth_loss.detach().item())
            if torch.is_tensor(sparse_colmap_depth_loss)
            else (float(sparse_colmap_depth_loss) if isinstance(sparse_colmap_depth_loss, (float, int)) else 0.0)
        )
        sparse_colmap_depth_debug["last_reason"] = str(sparse_colmap_depth_reason)

        # One-sided parent rollback loss on train/calibration sentinel sparse points.
        sparse_parent_rollback_loss_pure = 0
        sparse_parent_rollback_loss = 0
        sparse_parent_rollback_lambda_value = sparse_depth_parent_rollback_lambda(iteration=int(iteration), opt=opt)
        sparse_parent_rollback_active_points = 0
        sparse_parent_rollback_total_points = 0
        sparse_parent_rollback_active_fraction = 0.0
        sparse_parent_rollback_mean_violation_rel = 0.0
        sparse_parent_rollback_max_violation_rel = 0.0
        sparse_parent_rollback_mean_violation_abs = 0.0
        sparse_parent_rollback_max_violation_abs = 0.0
        sparse_parent_rollback_reason = "disabled"
        if sparse_parent_rollback_cache is not None and sparse_parent_rollback_lambda_value > 0.0:
            rollback_res = compute_sparse_depth_parent_rollback_loss(
                current_depth=render_pkg.get("surf_depth", None),
                cache_by_image_key=sparse_parent_rollback_cache.get("by_image_key", {}),
                image_key=normalize_image_key(getattr(viewpoint_cam, "image_name", "")),
                lam=float(sparse_parent_rollback_lambda_value),
                margin_abs=float(getattr(opt, "sparse_depth_parent_rollback_margin_abs", 0.0)),
                margin_rel=float(getattr(opt, "sparse_depth_parent_rollback_margin_rel", 0.0)),
                huber_delta=float(getattr(opt, "sparse_depth_parent_rollback_huber_delta", 0.05)),
                loss_space=str(getattr(opt, "sparse_depth_parent_rollback_loss_space", "combined")),
                combined_mae_beta=float(getattr(opt, "sparse_depth_parent_rollback_combined_mae_beta", 1.0)),
                max_points_per_view=int(getattr(opt, "sparse_depth_parent_rollback_max_points_per_view", 0)),
                cluster_balance=bool(getattr(opt, "sparse_depth_parent_rollback_cluster_balance", False)),
                regressed_only=bool(getattr(opt, "sparse_depth_parent_rollback_regressed_only", False)),
                cluster_top_k=int(getattr(opt, "sparse_depth_parent_rollback_cluster_top_k", 0)),
                aggregation=str(getattr(opt, "sparse_depth_parent_rollback_aggregation", "mean")),
                cvar_fraction=float(getattr(opt, "sparse_depth_parent_rollback_cvar_fraction", 0.2)),
                cvar_min_points=int(getattr(opt, "sparse_depth_parent_rollback_cvar_min_points", 16)),
                pixel_radius=int(getattr(opt, "sparse_depth_parent_rollback_pixel_radius", 0)),
                patch_reduce=str(getattr(opt, "sparse_depth_parent_rollback_patch_reduce", "center")),
                strict=bool(getattr(opt, "sparse_depth_parent_rollback_strict", False)),
            )
            sparse_parent_rollback_reason = str(rollback_res.get("reason", "unknown"))
            sparse_parent_rollback_loss_pure = rollback_res.get("loss_pure", 0)
            sparse_parent_rollback_loss = rollback_res.get("loss_weighted", 0)
            sparse_parent_rollback_active_points = int(rollback_res.get("active_points", 0))
            sparse_parent_rollback_total_points = int(rollback_res.get("total_points", 0))
            sparse_parent_rollback_active_fraction = float(rollback_res.get("active_fraction", 0.0))
            sparse_parent_rollback_mean_violation_rel = float(rollback_res.get("mean_violation_rel", 0.0))
            sparse_parent_rollback_max_violation_rel = float(rollback_res.get("max_violation_rel", 0.0))
            sparse_parent_rollback_mean_violation_abs = float(rollback_res.get("mean_violation_abs", 0.0))
            sparse_parent_rollback_max_violation_abs = float(rollback_res.get("max_violation_abs", 0.0))
            if torch.is_tensor(sparse_parent_rollback_loss) and torch.isfinite(sparse_parent_rollback_loss):
                loss += sparse_parent_rollback_loss

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

        if phase_info.get("phase", PrismPhase.FINAL_FINE_TUNE) == PrismPhase.RECOVERY_FINE_TUNE:
            loss = loss + _apply_prism_teacher_distill(
                prism_state=prism_state,
                viewpoint_cam=viewpoint_cam,
                render_pkg=render_pkg,
            )
        loss.backward()
        if bool(phase_info.get("collect_stats", True)):
            _update_prism_gradient_state(prism_state=prism_state, triangles=triangles, iteration=int(iteration))
            _update_prism_scores(
                prism_state=prism_state,
                iteration=iteration,
                triangles=triangles,
                force_recompute=False,
                render_pkg=render_pkg,
                viewpoint_cam=viewpoint_cam,
                assoc_stats=assoc_stats,
            )
            if bool(prism_state.pop("_force_recompute_scores_after_recollect", False)):
                _update_prism_scores(prism_state=prism_state, iteration=iteration, triangles=triangles, force_recompute=True)
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
            if tb_writer:
                sparse_loss_scalar = (
                    float(sparse_colmap_depth_loss.detach().item())
                    if torch.is_tensor(sparse_colmap_depth_loss)
                    else float(sparse_colmap_depth_loss) if isinstance(sparse_colmap_depth_loss, (float, int)) else 0.0
                )
                tb_writer.add_scalar("train/sparse_colmap_depth_loss", sparse_loss_scalar, iteration)
                tb_writer.add_scalar("train/sparse_colmap_depth_valid_matches", float(sparse_colmap_depth_valid_matches), iteration)
                tb_writer.add_scalar("train/sparse_colmap_depth_lambda", float(sparse_colmap_depth_lambda), iteration)
                sparse_parent_rollback_scalar = (
                    float(sparse_parent_rollback_loss.detach().item())
                    if torch.is_tensor(sparse_parent_rollback_loss)
                    else float(sparse_parent_rollback_loss)
                    if isinstance(sparse_parent_rollback_loss, (float, int))
                    else 0.0
                )
                tb_writer.add_scalar("train/sparse_parent_rollback_loss", sparse_parent_rollback_scalar, iteration)
                tb_writer.add_scalar("train/sparse_parent_rollback_active_points", float(sparse_parent_rollback_active_points), iteration)
                tb_writer.add_scalar("train/sparse_parent_rollback_lambda", float(sparse_parent_rollback_lambda_value), iteration)

            if bool(getattr(opt, "prism_save_debug_json", False)) and (int(iteration) % max(1, int(getattr(opt, "prism_collect_interval", 100))) == 0):
                out_dir = os.path.join(scene.model_path, "prism_debug")
                os.makedirs(out_dir, exist_ok=True)
                with open(
                    os.path.join(out_dir, f"sparse_depth_coverage_iter_{int(iteration):06d}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "iteration": int(iteration),
                            "phase": str(prism_state.get("current_phase", PrismPhase.FINAL_FINE_TUNE)),
                            "enabled": bool(getattr(opt, "enable_sparse_colmap_depth_loss", False)),
                            "valid_matches": int(sparse_colmap_depth_valid_matches),
                            "lambda": float(sparse_colmap_depth_lambda),
                            "loss": float(
                                sparse_colmap_depth_loss.detach().item()
                                if torch.is_tensor(sparse_colmap_depth_loss)
                                else sparse_colmap_depth_loss
                                if isinstance(sparse_colmap_depth_loss, (float, int))
                                else 0.0
                            ),
                            "reason": str(sparse_colmap_depth_reason),
                        },
                        f,
                        indent=2,
                    )

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
            should_log_wandb_scalar = bool(
                wandb_run and (int(wandb_scalar_log_interval) <= 0 or (iteration % int(wandb_scalar_log_interval) == 0))
            )
            if should_log_wandb_scalar:
                _wandb_log_filtered(
                    wandb_run=wandb_run,
                    payload={
                        "train/sparse_colmap_depth_loss": float(
                            sparse_colmap_depth_loss.detach().item()
                            if torch.is_tensor(sparse_colmap_depth_loss)
                            else sparse_colmap_depth_loss
                            if isinstance(sparse_colmap_depth_loss, (float, int))
                            else 0.0
                        ),
                        "train/sparse_colmap_depth_valid_matches": float(sparse_colmap_depth_valid_matches),
                        "train/sparse_colmap_depth_lambda": float(sparse_colmap_depth_lambda),
                        "train/sparse_parent_rollback_loss": float(
                            sparse_parent_rollback_loss.detach().item()
                            if torch.is_tensor(sparse_parent_rollback_loss)
                            else sparse_parent_rollback_loss
                            if isinstance(sparse_parent_rollback_loss, (float, int))
                            else 0.0
                        ),
                        "train/sparse_parent_rollback_active_points": float(sparse_parent_rollback_active_points),
                        "train/sparse_parent_rollback_lambda": float(sparse_parent_rollback_lambda_value),
                    },
                    step=iteration,
                    log_state=wandb_log_state,
                )
            if should_log_wandb_scalar and ground_reg_logs is not None:
                _wandb_log_filtered(
                    wandb_run=wandb_run,
                    payload={
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
                    log_state=wandb_log_state,
                )
            if should_log_wandb_scalar and assoc_stats is not None:
                tri_ids_ground = torch.nonzero(assoc_stats["is_ground_mask"], as_tuple=True)[0]
                ground_vertices = 0
                if tri_ids_ground.numel() > 0:
                    ground_vertices = int(torch.unique(triangles._triangle_indices[tri_ids_ground]).numel())
                _wandb_log_filtered(
                    wandb_run=wandb_run,
                    payload={
                        "ground_assoc/triangles_ground": float(assoc_stats["is_ground_mask"].sum().item()),
                        "ground_assoc/vertices_ground": float(ground_vertices),
                        "ground_assoc/triangles_boundary_uncertain": float(assoc_stats["boundary_uncertain_mask"].sum().item()),
                        "ground_assoc/triangles_unreliable": float((~assoc_stats["reliable_observation_mask"]).sum().item()),
                        "ground_assoc/support_ratio_mean": float(assoc_stats["ground_support_ratio"].mean().item()),
                        "ground_assoc/view_consistency_mean": float(assoc_stats["view_consistency"].mean().item()),
                    },
                    step=iteration,
                    log_state=wandb_log_state,
                )
            if tb_writer:
                tb_writer.add_scalar("prism/enabled", 1.0 if prism_state["cfg"]["enabled"] else 0.0, iteration)
                tb_writer.add_scalar("prism/collect_stats", 1.0 if prism_state["cfg"]["collect_stats"] else 0.0, iteration)
                tb_writer.add_scalar(
                    "prism/use_counterfactual_gate",
                    1.0 if prism_state["cfg"]["use_counterfactual_gate"] else 0.0,
                    iteration,
                )
                tb_writer.add_scalar("prism/current_phase", float(int(prism_state.get("current_phase", PrismPhase.FINAL_FINE_TUNE))), iteration)
                tb_writer.add_scalar("prism/active_triangle_count", float(triangles._triangle_indices.shape[0]), iteration)
                tb_writer.add_scalar("prism/pruned_this_round", float(prism_state.get("pruned_this_round", 0)), iteration)
                tb_writer.add_scalar("prism/counterfactual_accept", float(prism_state.get("counterfactual_accept", 0)), iteration)
                tb_writer.add_scalar("prism/rollback", float(prism_state.get("rollback", 0)), iteration)
                tb_writer.add_scalar("prism/rollback_by_validation", float(prism_state.get("rollback_by_validation", 0)), iteration)
                tb_writer.add_scalar("prism/adaptive_candidate_prune_ratio", float(prism_state.get("adaptive_candidate_prune_ratio", 0.0)), iteration)
                tb_writer.add_scalar("prism/adaptive_candidate_rollback_retries", float(prism_state.get("adaptive_candidate_rollback_retries", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_pool_count", float(prism_state.get("last_candidate_pool_count", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_cap_count", float(prism_state.get("last_candidate_cap_count", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_selected_count", float(prism_state.get("last_candidate_selected_count", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_microbatch_count", float(prism_state.get("last_candidate_microbatch_count", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_microbatch_accepted_count", float(prism_state.get("last_candidate_microbatch_accepted_count", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_microbatch_rejected_count", float(prism_state.get("last_candidate_microbatch_rejected_count", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_microbatch_accepted_triangles", float(prism_state.get("last_candidate_microbatch_accepted_triangles", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_quality_rank_enabled", float(prism_state.get("last_candidate_quality_rank_enabled", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_quality_score_mean", float(prism_state.get("last_candidate_quality_score_mean", 0.0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_prune_score_mean", float(prism_state.get("last_candidate_prune_score_mean", 0.0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_render_keep_mean", float(prism_state.get("last_candidate_render_keep_mean", 0.0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_geometry_keep_mean", float(prism_state.get("last_candidate_geometry_keep_mean", 0.0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_orientation_keep_mean", float(prism_state.get("last_candidate_orientation_keep_mean", 0.0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_utility_mean", float(prism_state.get("last_candidate_utility_mean", 0.0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_uncertainty_mean", float(prism_state.get("last_candidate_uncertainty_mean", 0.0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_measured_rank_enabled", float(prism_state.get("last_candidate_measured_rank_enabled", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_measured_group_count", float(prism_state.get("last_candidate_measured_group_count", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_measured_accepted_count", float(prism_state.get("last_candidate_measured_accepted_count", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_measured_selected_count", float(prism_state.get("last_candidate_measured_selected_count", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_measured_best_score", float(prism_state.get("last_candidate_measured_best_score", 0.0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_relaxed_refresh_used", float(prism_state.get("last_candidate_relaxed_refresh_used", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_relaxed_pool_count", float(prism_state.get("last_candidate_relaxed_pool_count", 0)), iteration)
                tb_writer.add_scalar("prism/relaxed_candidate_commit_count", float(prism_state.get("relaxed_candidate_commit_count", 0)), iteration)
                tb_writer.add_scalar("prism/last_candidate_relaxed_strict_gate_pass", float(prism_state.get("last_candidate_relaxed_strict_gate_pass", 1)), iteration)
                tb_writer.add_scalar("prism/validation_pass", float(prism_state.get("last_validation_pass", 1)), iteration)
                tb_writer.add_scalar(
                    "prism/post_commit_recollect_remaining",
                    float(getattr(controller, "post_commit_recollect_remaining", 0) if controller is not None else 0),
                    iteration,
                )
                tb_writer.add_scalar("prism/topology_change_iter", float(prism_state.get("last_topology_change_iter", -1)), iteration)
                tb_writer.add_scalar("prism/final_cleanup_enabled", float(prism_state.get("final_cleanup_enabled", 0)), iteration)
                tb_writer.add_scalar("prism/final_cleanup_pruned", float(prism_state.get("final_cleanup_pruned", 0)), iteration)
                tb_writer.add_scalar(
                    "prism/densification_frozen_after_commit",
                    float(prism_state.get("densification_frozen_after_prism_commit", 0)),
                    iteration,
                )
                tb_writer.add_scalar(
                    "prism/pre_cleanup_checkpoint",
                    1.0 if str(prism_state.get("pre_cleanup_checkpoint", "")) else 0.0,
                    iteration,
                )
                last_val = prism_state.get("last_validation_metrics", None)
                if isinstance(last_val, dict):
                    for k in ["psnr", "mae", "absrel", "delta_1.25", "depth_mae", "mean_angle", "abs_cos", "roi_psnr", "roi_mae", "roi_absrel", "roi_mean_angle", "roi_abs_cos"]:
                        v = float(last_val.get(k, float("nan")))
                        if np.isfinite(v):
                            tb_writer.add_scalar(f"prism/val_{k}", v, iteration)
                manager = prism_state.get("manager", None)
                if manager is not None:
                    dbg = manager.get_debug_summary(iteration=int(iteration))
                    tb_writer.add_scalar("prism/vis_count_ema_mean", dbg["vis_count_ema_mean"], iteration)
                    tb_writer.add_scalar("prism/projected_area_ema_mean", dbg["projected_area_ema_mean"], iteration)
                    tb_writer.add_scalar("prism/grad_pos_norm_ema_mean", dbg["grad_pos_norm_ema_mean"], iteration)
                    tb_writer.add_scalar("prism/grad_app_norm_ema_mean", dbg["grad_app_norm_ema_mean"], iteration)
                    tb_writer.add_scalar("prism/grad_norm_var_ema_mean", dbg["grad_norm_var_ema_mean"], iteration)
                last_scores = prism_state.get("last_scores", None)
                if last_scores is not None:
                    tb_writer.add_scalar("prism/protected_count", float(last_scores.protected_mask.sum().item()), iteration)
                    tb_writer.add_scalar("prism/protected_raw_count", float(last_scores.protected_mask_raw.sum().item()), iteration)
                    tb_writer.add_scalar("prism/protected_dilated_count", float(last_scores.protected_mask_dilated.sum().item()), iteration)
                    tb_writer.add_scalar("prism/geometry_keep_mean", float(last_scores.geometry_keep_t.mean().item()), iteration)
                    tb_writer.add_scalar("prism/orientation_keep_mean", float(last_scores.orientation_keep_t.mean().item()), iteration)
                    tb_writer.add_scalar("prism/heavy_eval_fraction", float(last_scores.heavy_eval_mask.to(torch.float32).mean().item()), iteration)
                    tb_writer.add_scalar("prism/heavy_eval_triangle_count", float(last_scores.heavy_eval_mask.sum().item()), iteration)
                    tb_writer.add_scalar("prism/geometry_keep_nonzero_fraction", float((last_scores.geometry_keep_t > 0).to(torch.float32).mean().item()), iteration)
                    tb_writer.add_scalar("prism/orientation_keep_nonzero_fraction", float((last_scores.orientation_keep_t > 0).to(torch.float32).mean().item()), iteration)
                    tb_writer.add_scalar("prism/candidate_blocked_by_geometry_keep_count", float(last_scores.candidate_blocked_by_geometry_keep.sum().item()), iteration)
                    tb_writer.add_scalar("prism/candidate_blocked_by_dilated_protect_count", float(last_scores.candidate_blocked_by_dilated_protect.sum().item()), iteration)
                    tb_writer.add_scalar("prism/dead_count", float(last_scores.dead_mask.sum().item()), iteration)
                    tb_writer.add_scalar("prism/candidate_count", float(last_scores.candidate_mask.sum().item()), iteration)
            if should_log_wandb_scalar:
                wandb_payload = {
                    "prism/enabled": 1.0 if prism_state["cfg"]["enabled"] else 0.0,
                    "prism/collect_stats": 1.0 if prism_state["cfg"]["collect_stats"] else 0.0,
                    "prism/use_counterfactual_gate": 1.0 if prism_state["cfg"]["use_counterfactual_gate"] else 0.0,
                    "prism/current_phase": float(int(prism_state.get("current_phase", PrismPhase.FINAL_FINE_TUNE))),
                    "prism/active_triangle_count": float(triangles._triangle_indices.shape[0]),
                    "prism/pruned_this_round": float(prism_state.get("pruned_this_round", 0)),
                    "prism/counterfactual_accept": float(prism_state.get("counterfactual_accept", 0)),
                    "prism/rollback": float(prism_state.get("rollback", 0)),
                    "prism/rollback_by_validation": float(prism_state.get("rollback_by_validation", 0)),
                    "prism/adaptive_candidate_prune_ratio": float(prism_state.get("adaptive_candidate_prune_ratio", 0.0)),
                    "prism/adaptive_candidate_rollback_retries": float(prism_state.get("adaptive_candidate_rollback_retries", 0)),
                    "prism/last_candidate_pool_count": float(prism_state.get("last_candidate_pool_count", 0)),
                    "prism/last_candidate_target_count": float(prism_state.get("last_candidate_target_count", 0)),
                    "prism/last_candidate_cap_count": float(prism_state.get("last_candidate_cap_count", 0)),
                    "prism/last_candidate_selected_count": float(prism_state.get("last_candidate_selected_count", 0)),
                    "prism/last_candidate_microbatch_count": float(prism_state.get("last_candidate_microbatch_count", 0)),
                    "prism/last_candidate_microbatch_accepted_count": float(prism_state.get("last_candidate_microbatch_accepted_count", 0)),
                    "prism/last_candidate_microbatch_rejected_count": float(prism_state.get("last_candidate_microbatch_rejected_count", 0)),
                    "prism/last_candidate_microbatch_accepted_triangles": float(prism_state.get("last_candidate_microbatch_accepted_triangles", 0)),
                    "prism/last_candidate_quality_rank_enabled": float(prism_state.get("last_candidate_quality_rank_enabled", 0)),
                    "prism/last_candidate_quality_score_mean": float(prism_state.get("last_candidate_quality_score_mean", 0.0)),
                    "prism/last_candidate_prune_score_mean": float(prism_state.get("last_candidate_prune_score_mean", 0.0)),
                    "prism/last_candidate_render_keep_mean": float(prism_state.get("last_candidate_render_keep_mean", 0.0)),
                    "prism/last_candidate_geometry_keep_mean": float(prism_state.get("last_candidate_geometry_keep_mean", 0.0)),
                    "prism/last_candidate_orientation_keep_mean": float(prism_state.get("last_candidate_orientation_keep_mean", 0.0)),
                    "prism/last_candidate_utility_mean": float(prism_state.get("last_candidate_utility_mean", 0.0)),
                    "prism/last_candidate_uncertainty_mean": float(prism_state.get("last_candidate_uncertainty_mean", 0.0)),
                    "prism/last_candidate_measured_rank_enabled": float(prism_state.get("last_candidate_measured_rank_enabled", 0)),
                    "prism/last_candidate_measured_group_count": float(prism_state.get("last_candidate_measured_group_count", 0)),
                    "prism/last_candidate_measured_accepted_count": float(prism_state.get("last_candidate_measured_accepted_count", 0)),
                    "prism/last_candidate_measured_selected_count": float(prism_state.get("last_candidate_measured_selected_count", 0)),
                    "prism/last_candidate_measured_best_score": float(prism_state.get("last_candidate_measured_best_score", 0.0)),
                    "prism/last_candidate_relaxed_refresh_used": float(prism_state.get("last_candidate_relaxed_refresh_used", 0)),
                    "prism/last_candidate_relaxed_pool_count": float(prism_state.get("last_candidate_relaxed_pool_count", 0)),
                    "prism/relaxed_candidate_commit_count": float(prism_state.get("relaxed_candidate_commit_count", 0)),
                    "prism/last_candidate_relaxed_strict_gate_pass": float(prism_state.get("last_candidate_relaxed_strict_gate_pass", 1)),
                    "prism/validation_pass": float(prism_state.get("last_validation_pass", 1)),
                    "prism/densification_frozen_after_commit": float(
                        prism_state.get("densification_frozen_after_prism_commit", 0)
                    ),
                    "prism/post_commit_recollect_remaining": float(
                        getattr(controller, "post_commit_recollect_remaining", 0) if controller is not None else 0
                    ),
                    "prism/topology_change_iter": float(prism_state.get("last_topology_change_iter", -1)),
                    "prism/final_cleanup_enabled": float(prism_state.get("final_cleanup_enabled", 0)),
                    "prism/final_cleanup_pruned": float(prism_state.get("final_cleanup_pruned", 0)),
                    "prism/pre_cleanup_checkpoint": 1.0 if str(prism_state.get("pre_cleanup_checkpoint", "")) else 0.0,
                }
                last_val = prism_state.get("last_validation_metrics", None)
                if isinstance(last_val, dict):
                    for k in ["psnr", "mae", "absrel", "delta_1.25", "depth_mae", "mean_angle", "abs_cos", "roi_psnr", "roi_mae", "roi_absrel", "roi_mean_angle", "roi_abs_cos"]:
                        v = float(last_val.get(k, float("nan")))
                        if np.isfinite(v):
                            wandb_payload[f"prism/val_{k}"] = v
                last_val_deltas = prism_state.get("last_validation_deltas", {})
                if isinstance(last_val_deltas, dict):
                    for k, v in last_val_deltas.items():
                        vv = float(v) if isinstance(v, (int, float, np.generic)) else float("nan")
                        if np.isfinite(vv):
                            wandb_payload[f"prism/val_delta_{k}"] = vv
                wandb_payload["prism/val_triggered_rules_count"] = float(len(prism_state.get("last_validation_rules", [])))
                manager = prism_state.get("manager", None)
                if manager is not None:
                    dbg = manager.get_debug_summary(iteration=int(iteration))
                    wandb_payload.update(
                        {
                            "prism/vis_count_ema_mean": dbg["vis_count_ema_mean"],
                            "prism/projected_area_ema_mean": dbg["projected_area_ema_mean"],
                            "prism/grad_pos_norm_ema_mean": dbg["grad_pos_norm_ema_mean"],
                            "prism/grad_app_norm_ema_mean": dbg["grad_app_norm_ema_mean"],
                            "prism/grad_norm_var_ema_mean": dbg["grad_norm_var_ema_mean"],
                        }
                    )
                last_scores = prism_state.get("last_scores", None)
                if last_scores is not None:
                    wandb_payload.update(
                        {
                            "prism/protected_count": float(last_scores.protected_mask.sum().item()),
                            "prism/protected_raw_count": float(last_scores.protected_mask_raw.sum().item()),
                            "prism/protected_dilated_count": float(last_scores.protected_mask_dilated.sum().item()),
                            "prism/geometry_keep_mean": float(last_scores.geometry_keep_t.mean().item()),
                            "prism/orientation_keep_mean": float(last_scores.orientation_keep_t.mean().item()),
                            "prism/heavy_eval_fraction": float(last_scores.heavy_eval_mask.to(torch.float32).mean().item()),
                            "prism/heavy_eval_triangle_count": float(last_scores.heavy_eval_mask.sum().item()),
                            "prism/geometry_keep_nonzero_fraction": float((last_scores.geometry_keep_t > 0).to(torch.float32).mean().item()),
                            "prism/orientation_keep_nonzero_fraction": float((last_scores.orientation_keep_t > 0).to(torch.float32).mean().item()),
                            "prism/candidate_blocked_by_geometry_keep_count": float(last_scores.candidate_blocked_by_geometry_keep.sum().item()),
                            "prism/candidate_blocked_by_dilated_protect_count": float(last_scores.candidate_blocked_by_dilated_protect.sum().item()),
                            "prism/dead_count": float(last_scores.dead_mask.sum().item()),
                            "prism/candidate_count": float(last_scores.candidate_mask.sum().item()),
                        }
                    )
                wandb_payload.update(
                    {
                        "mesh/triangle_count": float(triangles._triangle_indices.shape[0]),
                        "mesh/vertex_count": float(triangles.vertices.shape[0]),
                        "mesh/rendered_triangle_count": float(
                            (render_pkg["triangle_was_rendered"] > 0).sum().item()
                        )
                        if ("triangle_was_rendered" in render_pkg)
                        else 0.0,
                        "mesh/mean_triangle_image_size": float(render_pkg["scaling"].detach().mean().item())
                        if ("scaling" in render_pkg)
                        else 0.0,
                        "loss_components/loss_image": float(loss_image.detach().item()),
                        "loss_components/loss_weight": float(Lweight) if isinstance(Lweight, (float, int)) else float(Lweight.detach().item()),
                        "loss_components/loss_vertex_depth": float(Lvertex_depth) if isinstance(Lvertex_depth, (float, int)) else float(Lvertex_depth.detach().item()),
                        "loss_components/loss_depth_l1": float(Ll1depth) if isinstance(Ll1depth, (float, int)) else float(Ll1depth.detach().item()),
                        "loss_components/loss_normal": float(Lnormal) if isinstance(Lnormal, (float, int)) else float(Lnormal.detach().item()),
                        "loss_components/loss_normal_super": float(normal_loss_super) if isinstance(normal_loss_super, (float, int)) else float(normal_loss_super.detach().item()),
                        "loss_components/loss_teacher_render": float(teacher_render_loss) if isinstance(teacher_render_loss, (float, int)) else float(teacher_render_loss.detach().item()),
                        "loss_components/loss_teacher_render_pure": float(teacher_render_loss_pure) if isinstance(teacher_render_loss_pure, (float, int)) else float(teacher_render_loss_pure.detach().item()),
                        "loss_components/loss_checkpoint_geometry_anchor": float(checkpoint_geometry_anchor_loss) if isinstance(checkpoint_geometry_anchor_loss, (float, int)) else float(checkpoint_geometry_anchor_loss.detach().item()),
                        "loss_components/loss_checkpoint_geometry_anchor_pure": float(checkpoint_geometry_anchor_loss_pure) if isinstance(checkpoint_geometry_anchor_loss_pure, (float, int)) else float(checkpoint_geometry_anchor_loss_pure.detach().item()),
                        "loss_components/loss_checkpoint_render_geometry_anchor": float(checkpoint_render_geometry_anchor_loss) if isinstance(checkpoint_render_geometry_anchor_loss, (float, int)) else float(checkpoint_render_geometry_anchor_loss.detach().item()),
                        "loss_components/loss_checkpoint_render_depth_anchor_pure": float(checkpoint_render_depth_anchor_loss_pure) if isinstance(checkpoint_render_depth_anchor_loss_pure, (float, int)) else float(checkpoint_render_depth_anchor_loss_pure.detach().item()),
                        "loss_components/loss_checkpoint_render_normal_anchor_pure": float(checkpoint_render_normal_anchor_loss_pure) if isinstance(checkpoint_render_normal_anchor_loss_pure, (float, int)) else float(checkpoint_render_normal_anchor_loss_pure.detach().item()),
                        "loss_components/loss_sparse_parent_rollback": float(sparse_parent_rollback_loss) if isinstance(sparse_parent_rollback_loss, (float, int)) else float(sparse_parent_rollback_loss.detach().item()),
                        "loss_components/loss_sparse_parent_rollback_pure": float(sparse_parent_rollback_loss_pure) if isinstance(sparse_parent_rollback_loss_pure, (float, int)) else float(sparse_parent_rollback_loss_pure.detach().item()),
                        "loss_components/loss_lpips_train": float(lpips_loss_weighted) if isinstance(lpips_loss_weighted, (float, int)) else float(lpips_loss_weighted.detach().item()),
                        "loss_components/loss_lpips_train_pure": float(lpips_loss_pure) if isinstance(lpips_loss_pure, (float, int)) else float(lpips_loss_pure.detach().item()),
                        "lpips_train/lambda": float(lpips_loss_lambda),
                        "teacher_render/lambda": float(teacher_render_lambda),
                        "teacher_render/l1": float(teacher_render_l1) if isinstance(teacher_render_l1, (float, int)) else float(teacher_render_l1.detach().item()),
                        "teacher_render/ssim": float(teacher_render_ssim) if isinstance(teacher_render_ssim, (float, int)) else float(teacher_render_ssim.detach().item()),
                        "teacher_render/mask_fraction": float(teacher_render_mask_fraction),
                        "checkpoint_geometry_anchor/lambda": float(checkpoint_geometry_anchor_lambda),
                        "checkpoint_geometry_anchor/mean_displacement": float(checkpoint_geometry_anchor_mean_disp) if isinstance(checkpoint_geometry_anchor_mean_disp, (float, int)) else float(checkpoint_geometry_anchor_mean_disp.detach().item()),
                        "checkpoint_geometry_anchor/max_displacement": float(checkpoint_geometry_anchor_max_disp) if isinstance(checkpoint_geometry_anchor_max_disp, (float, int)) else float(checkpoint_geometry_anchor_max_disp.detach().item()),
                        "checkpoint_render_geometry_anchor/depth_lambda": float(checkpoint_render_depth_anchor_lambda),
                        "checkpoint_render_geometry_anchor/normal_lambda": float(checkpoint_render_normal_anchor_lambda),
                        "sparse_parent_rollback/lambda": float(sparse_parent_rollback_lambda_value),
                        "sparse_parent_rollback/active_points": float(sparse_parent_rollback_active_points),
                        "sparse_parent_rollback/total_points": float(sparse_parent_rollback_total_points),
                        "sparse_parent_rollback/active_fraction": float(sparse_parent_rollback_active_fraction),
                        "sparse_parent_rollback/mean_violation_rel": float(sparse_parent_rollback_mean_violation_rel),
                        "sparse_parent_rollback/max_violation_rel": float(sparse_parent_rollback_max_violation_rel),
                        "sparse_parent_rollback/mean_violation_abs": float(sparse_parent_rollback_mean_violation_abs),
                        "sparse_parent_rollback/max_violation_abs": float(sparse_parent_rollback_max_violation_abs),
                        "sparse_parent_rollback/cache_split_id": float(0.0 if sparse_parent_rollback_cache_split != "test" else 1.0),
                    }
                )
                if ground_reg_logs is not None:
                    wandb_payload["loss_components/loss_ground_total"] = float(ground_reg_logs.get("Lground_total", 0.0))
                # Optimizer group learning rates.
                if triangles.optimizer is not None:
                    lr_map = {}
                    for pg in triangles.optimizer.param_groups:
                        lr_name = str(pg.get("name", "unnamed"))
                        lr_map[f"optim/lr_{lr_name}"] = float(pg.get("lr", 0.0))
                    wandb_payload.update(lr_map)
                _wandb_log_filtered(
                    wandb_run=wandb_run,
                    payload=wandb_payload,
                    step=iteration,
                    log_state=wandb_log_state,
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
                wandb_log_state=wandb_log_state,
                wandb_scalar_log_interval=int(wandb_scalar_log_interval),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving model".format(iteration))
                scene.save(iteration)
            if iteration in checkpoint_iterations:
                checkpoint_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                print("\n[ITER {}] Saving Checkpoint: {}".format(iteration, checkpoint_path))
                torch.save((triangles.capture(), iteration), checkpoint_path)

            # Handle pruning operations
            prism_state["pruned_this_round"] = 0
            prism_state["counterfactual_accept"] = 0
            prism_state["rollback"] = 0
            prism_state["rollback_by_validation"] = 0
            if bool(phase_info.get("should_attempt_prune", False)):
                # Ensure prune decisions use freshly recomputed heavy features.
                _update_prism_scores(
                    prism_state=prism_state,
                    iteration=iteration,
                    triangles=triangles,
                    force_recompute=True,
                )
                pre_prune_triangle_count = int(triangles._triangle_indices.shape[0])
                prism_state["round_snapshot"] = _capture_prism_round_snapshot(triangles)
                if bool(getattr(opt, "prism_round_checkpoint", True)):
                    pre_ckpt_dir = _save_prism_round_checkpoint(
                        scene=scene,
                        triangles=triangles,
                        iteration=iteration,
                        tag="round_pre_{}".format(str(phase_info.get("prune_mode", "candidate"))),
                    )
                    prism_state["round_pre_ckpt_dir"] = pre_ckpt_dir
                prune_result = _prism_maybe_prune(
                    prism_state=prism_state,
                    iteration=iteration,
                    triangles=triangles,
                    scene=scene,
                    render_pkg=render_pkg,
                    prune_mode=str(phase_info.get("prune_mode", "candidate")),
                )
                prism_state["pruned_this_round"] = int(prune_result.get("pruned_count", 0))
                prism_state["counterfactual_accept"] = int(prune_result.get("counterfactual_accept", 0))
                prism_state["rollback"] = int(prune_result.get("rollback", 0))
                if bool(prism_state["cfg"].get("enable_adaptive_csef_policy", False)):
                    update_adaptive_csef_policy_after_prune(
                        prism_state=prism_state,
                        committed=bool(prune_result.get("committed", False)),
                        rollback=bool(prune_result.get("rollback", 0)),
                        no_candidates=bool(prune_result.get("no_candidates", 0)),
                        ratio=float(prune_result.get(
                            "candidate_prune_ratio",
                            prism_state.get("adaptive_candidate_prune_ratio", 0.0),
                        )),
                    )
                if controller is not None:
                    if bool(prune_result.get("no_candidates", 0)):
                        controller.report_no_candidate_retry(
                            retry_iters=int(getattr(opt, "prism_no_candidate_retry_iters", 10))
                        )
                    elif (
                        str(phase_info.get("prune_mode", "candidate")) == "candidate"
                        and bool(prism_state["cfg"].get("adaptive_candidate_retry_on_rollback", False))
                        and bool(prune_result.get("rollback", 0))
                        and not bool(prune_result.get("committed", False))
                        and int(prism_state.get("adaptive_candidate_rollback_retries", 0))
                        < int(prism_state["cfg"].get("adaptive_candidate_max_rollback_retries", 3))
                    ):
                        prism_state["adaptive_candidate_rollback_retries"] = (
                            int(prism_state.get("adaptive_candidate_rollback_retries", 0)) + 1
                        )
                        current_ratio = float(prism_state.get(
                            "adaptive_candidate_prune_ratio",
                            prism_state["cfg"].get("candidate_prune_ratio_per_round", 0.015),
                        ))
                        next_ratio = max(
                            float(prism_state["cfg"].get("adaptive_candidate_min_ratio", 0.0025)),
                            current_ratio * float(prism_state["cfg"].get("adaptive_candidate_ratio_decay", 0.5)),
                        )
                        prism_state["adaptive_candidate_prune_ratio"] = float(next_ratio)
                        controller.report_no_candidate_retry(
                            retry_iters=int(getattr(opt, "prism_no_candidate_retry_iters", 10))
                        )
                    else:
                        controller.report_prune_result(
                            prune_mode=str(phase_info.get("prune_mode", "candidate")),
                            committed=bool(prune_result.get("committed", False)),
                            pruned_count=int(prune_result.get("pruned_count", 0)),
                            counterfactual_accept=int(prune_result.get("counterfactual_accept", 0)),
                            rollback=int(prune_result.get("rollback", 0)),
                        )
                        if (
                            str(phase_info.get("prune_mode", "candidate")) == "candidate"
                            and (
                                bool(prune_result.get("committed", False))
                                or not bool(prune_result.get("rollback", 0))
                            )
                        ):
                            prism_state["adaptive_candidate_rollback_retries"] = 0
                            prism_state["adaptive_candidate_prune_ratio"] = float(
                                prism_state["cfg"].get("candidate_prune_ratio_per_round", 0.015)
                            )
                post_prune_triangle_count = int(triangles._triangle_indices.shape[0])
                _save_prism_round_meta(
                    scene=scene,
                    iteration=int(iteration),
                    prune_mode=str(phase_info.get("prune_mode", "candidate")),
                    payload={
                        "iteration": int(iteration),
                        "phase": str(phase_info.get("phase", PrismPhase.FINAL_FINE_TUNE)),
                        "prune_mode": str(phase_info.get("prune_mode", "candidate")),
                        "committed": bool(prune_result.get("committed", False)),
                        "counterfactual_accept": int(prune_result.get("counterfactual_accept", 0)),
                        "rollback": int(prune_result.get("rollback", 0)),
                        "no_candidates": int(prune_result.get("no_candidates", 0)),
                        "adaptive_candidate_retry_on_rollback": int(
                            bool(prism_state["cfg"].get("adaptive_candidate_retry_on_rollback", False))
                        ),
                        "adaptive_candidate_rollback_retries": int(
                            prism_state.get("adaptive_candidate_rollback_retries", 0)
                        ),
                        "adaptive_csef_policy_enabled": int(
                            bool(prism_state["cfg"].get("enable_adaptive_csef_policy", False))
                        ),
                        "adaptive_csef_policy": dict(prune_result.get("adaptive_policy_decision", {}) or {}),
                        "candidate_prune_ratio": float(prune_result.get(
                            "candidate_prune_ratio",
                            prism_state.get("adaptive_candidate_prune_ratio", 0.0),
                        )),
                        "candidate_pool_count": int(prune_result.get("candidate_pool_count", 0)),
                        "candidate_target_count": int(prune_result.get("candidate_target_count", 0)),
                        "candidate_cap_count": int(prune_result.get("candidate_cap_count", 0)),
                        "candidate_selected_count": int(prune_result.get("candidate_selected_count", 0)),
                        "candidate_microbatch_count": int(prune_result.get("candidate_microbatch_count", 0)),
                        "candidate_microbatch_accepted_count": int(prune_result.get("candidate_microbatch_accepted_count", 0)),
                        "candidate_microbatch_rejected_count": int(prune_result.get("candidate_microbatch_rejected_count", 0)),
                        "candidate_microbatch_accepted_triangles": int(prune_result.get("candidate_microbatch_accepted_triangles", 0)),
                        "candidate_quality_rank_enabled": int(prune_result.get("candidate_quality_rank_enabled", 0)),
                        "candidate_quality_score_mean": float(prune_result.get("candidate_quality_score_mean", 0.0)),
                        "candidate_prune_score_mean": float(prune_result.get("candidate_prune_score_mean", 0.0)),
                        "candidate_render_keep_mean": float(prune_result.get("candidate_render_keep_mean", 0.0)),
                        "candidate_geometry_keep_mean": float(prune_result.get("candidate_geometry_keep_mean", 0.0)),
                        "candidate_orientation_keep_mean": float(prune_result.get("candidate_orientation_keep_mean", 0.0)),
                        "candidate_utility_mean": float(prune_result.get("candidate_utility_mean", 0.0)),
                        "candidate_uncertainty_mean": float(prune_result.get("candidate_uncertainty_mean", 0.0)),
                        "candidate_measured_rank_enabled": int(prune_result.get("candidate_measured_rank_enabled", 0)),
                        "candidate_measured_group_count": int(prune_result.get("candidate_measured_group_count", 0)),
                        "candidate_measured_accepted_count": int(prune_result.get("candidate_measured_accepted_count", 0)),
                        "candidate_measured_selected_count": int(prune_result.get("candidate_measured_selected_count", 0)),
                        "candidate_measured_best_score": float(prune_result.get("candidate_measured_best_score", 0.0)),
                        "candidate_relaxed_refresh_used": int(prune_result.get("candidate_relaxed_refresh_used", 0)),
                        "candidate_relaxed_pool_count": int(prune_result.get("candidate_relaxed_pool_count", 0)),
                        "candidate_relaxed_reject_reason": str(prune_result.get("candidate_relaxed_reject_reason", "")),
                        "candidate_relaxed_commit_count": int(prune_result.get("candidate_relaxed_commit_count", 0)),
                        "candidate_relaxed_max_commits": int(prune_result.get("candidate_relaxed_max_commits", 0)),
                        "candidate_relaxed_strict_gate_enabled": int(
                            prune_result.get("candidate_relaxed_strict_gate_enabled", 0)
                        ),
                        "candidate_relaxed_strict_gate_pass": int(
                            prune_result.get("candidate_relaxed_strict_gate_pass", 1)
                        ),
                        "candidate_relaxed_strict_gate_reason": str(
                            prune_result.get("candidate_relaxed_strict_gate_reason", "")
                        ),
                        "candidate_diag_total_triangles": int(prune_result.get("candidate_diag_total_triangles", 0)),
                        "candidate_diag_active_state_count": int(prune_result.get("candidate_diag_active_state_count", 0)),
                        "candidate_diag_protected_raw_count": int(prune_result.get("candidate_diag_protected_raw_count", 0)),
                        "candidate_diag_protected_dilated_count": int(prune_result.get("candidate_diag_protected_dilated_count", 0)),
                        "candidate_diag_dead_count": int(prune_result.get("candidate_diag_dead_count", 0)),
                        "candidate_diag_suspicious_count": int(prune_result.get("candidate_diag_suspicious_count", 0)),
                        "candidate_diag_block_edge_count": int(prune_result.get("candidate_diag_block_edge_count", 0)),
                        "candidate_diag_block_geo_count": int(prune_result.get("candidate_diag_block_geo_count", 0)),
                        "candidate_diag_block_sens_count": int(prune_result.get("candidate_diag_block_sens_count", 0)),
                        "candidate_diag_block_unc_count": int(prune_result.get("candidate_diag_block_unc_count", 0)),
                        "candidate_diag_block_recent_count": int(prune_result.get("candidate_diag_block_recent_count", 0)),
                        "candidate_diag_block_geometry_keep_count": int(prune_result.get("candidate_diag_block_geometry_keep_count", 0)),
                        "candidate_diag_block_orientation_keep_count": int(prune_result.get("candidate_diag_block_orientation_keep_count", 0)),
                        "candidate_diag_block_render_keep_count": int(prune_result.get("candidate_diag_block_render_keep_count", 0)),
                        "candidate_diag_block_candidate_geometry_keep_count": int(prune_result.get("candidate_diag_block_candidate_geometry_keep_count", 0)),
                        "candidate_diag_block_candidate_dilated_count": int(prune_result.get("candidate_diag_block_candidate_dilated_count", 0)),
                        "candidate_diag_post_commit_relaxed_score_positive_count": int(
                            prune_result.get("candidate_diag_post_commit_relaxed_score_positive_count", 0)
                        ),
                        "candidate_diag_post_commit_relaxed_score_mean": float(
                            prune_result.get("candidate_diag_post_commit_relaxed_score_mean", 0.0)
                        ),
                        "candidate_diag_post_commit_relaxed_score_max": float(
                            prune_result.get("candidate_diag_post_commit_relaxed_score_max", 0.0)
                        ),
                        "pre_prune_triangle_count": int(pre_prune_triangle_count),
                        "post_prune_triangle_count": int(post_prune_triangle_count),
                        "recollect_iters_used": int(getattr(controller, "last_recollect_iters_used", 0) if controller is not None else 0),
                    },
                )
                if bool(prune_result.get("committed", False)):
                    if (
                        str(phase_info.get("prune_mode", "candidate")) == "candidate"
                        and bool(prism_state["cfg"].get("freeze_densification_after_first_commit", False))
                    ):
                        prism_state["densification_frozen_after_prism_commit"] = 1
                        prism_state["densification_freeze_iter"] = int(iteration)
                    if bool(getattr(opt, "prism_round_checkpoint", True)):
                        post_ckpt_dir = _save_prism_round_checkpoint(
                            scene=scene,
                            triangles=triangles,
                            iteration=iteration,
                            tag="round_post_{}".format(str(phase_info.get("prune_mode", "candidate"))),
                        )
                        prism_state["round_post_ckpt_dir"] = post_ckpt_dir
                    reset_ground_supervision_state("prism_counterfactual_commit")
                    continue
                prism_state["round_snapshot"] = None

            if (
                iteration % 500 == 0
                and iteration < run_restricted_delaunay
                and bool(phase_info.get("allow_topology_mutation", True))
                and not bool(getattr(opt, "freeze_topology_updates", False))
            ):
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

                needs_densification = (
                    iteration < opt.densify_until_iter
                    and iteration % opt.densification_interval == 0
                    and iteration > opt.densify_from_iter
                    and not bool(prism_state.get("densification_frozen_after_prism_commit", 0))
                )
                
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
                    _sync_prism_topology_change(
                        prism_state=prism_state,
                        triangles=triangles,
                        iteration=int(iteration),
                        reason="prune_or_densify",
                    )
                    reset_ground_supervision_state("prune_or_densify")
                    if wandb_run is not None and should_log_wandb_scalar:
                        _wandb_log_filtered(
                            wandb_run=wandb_run,
                            payload={
                                "mesh/triangle_count": float(post_triangles),
                                "mesh/vertex_count": float(post_vertices),
                                "mesh/post_topology_triangle_count": float(post_triangles),
                                "mesh/post_topology_vertex_count": float(post_vertices),
                                "mesh/pre_topology_triangle_count": float(pre_triangles),
                                "mesh/pre_topology_vertex_count": float(pre_vertices),
                                "prism/standard_topology_mutation": 1.0,
                            },
                            step=iteration,
                            log_state=wandb_log_state,
                        )
            elif (
                iteration == run_restricted_delaunay
                and not bool(getattr(opt, "skip_restricted_delaunay", False))
                and not bool(getattr(opt, "freeze_topology_updates", False))
            ):
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
                # Periodic dev validation bookkeeping (does not trigger rollback by itself).
                val_cfg = prism_state.get("validation_cfg", None)
                prism_enabled = bool(prism_state.get("cfg", {}).get("enabled", False))
                if prism_enabled and val_cfg is not None and int(getattr(val_cfg, "interval", 0)) > 0:
                    if int(iteration) % int(val_cfg.interval) == 0:
                        _evaluate_prism_validation(
                            prism_state=prism_state,
                            scene=scene,
                            triangles=triangles,
                            iteration=int(iteration),
                        )

                # Global rollback gate: evaluate once at the end of each recovery window.
                if controller is not None and prism_enabled:
                    current_phase = phase_info.get("phase", PrismPhase.FINAL_FINE_TUNE)
                    if (current_phase == PrismPhase.RECOVERY_FINE_TUNE) and (int(controller.recovery_remaining) == 1):
                        if int(prism_state.get("last_validation_eval_iter", -1)) == int(iteration):
                            val_result = {
                                "pass_gate": bool(prism_state.get("last_validation_pass", 1)),
                            }
                        else:
                            val_result = _evaluate_prism_validation(
                                prism_state=prism_state,
                                scene=scene,
                                triangles=triangles,
                                iteration=int(iteration),
                            )
                        if (val_result is not None) and (not bool(val_result["pass_gate"])):
                            restored = _restore_prism_round_snapshot(
                                prism_state=prism_state,
                                triangles=triangles,
                                iteration=int(iteration),
                            )
                            if restored:
                                prism_state["rollback"] = 1
                                prism_state["rollback_by_validation"] = 1
                                if int(prism_state.get("last_commit_relaxed_refresh_used", 0)):
                                    records = list(prism_state.get("relaxed_commit_records", []))
                                    for record in reversed(records):
                                        if not bool(record.get("validation_rollback", False)):
                                            record["validation_rollback"] = True
                                            record["validation_rollback_iter"] = int(iteration)
                                            record["post_rollback_triangle_count"] = int(
                                                triangles._triangle_indices.shape[0]
                                            )
                                            break
                                    prism_state["relaxed_commit_records"] = records
                                    prism_state["relaxed_candidate_commit_count"] = max(
                                        0, int(prism_state.get("relaxed_candidate_commit_count", 0)) - 1
                                    )
                                    prism_state["last_commit_relaxed_refresh_used"] = 0
                                reset_ground_supervision_state("prism_validation_rollback")
                        prism_state["round_snapshot"] = None
                if controller is not None:
                    controller.consume_recovery_step()

    compaction_result = None
    if bool(getattr(opt, "enable_prism_pruning", False)) and bool(getattr(opt, "prism_enable_compaction_stage", False)):
        compaction_result = _run_prism_compaction_stage(
            prism_state=prism_state,
            scene=scene,
            triangles=triangles,
            iteration=int(iteration),
            tb_writer=tb_writer,
            wandb_run=wandb_run,
            wandb_log_state=wandb_log_state if 'wandb_log_state' in locals() else None,
            ground_association_tracker=ground_association_tracker,
        )

    # cleaning of triangles that we do not need
    if ground_association_tracker is not None:
        ground_association_tracker.ensure_num_triangles(int(triangles._triangle_indices.shape[0]))
        ground_association_tracker.save_cache()
    prism_enabled = bool(getattr(opt, "enable_prism_pruning", False))
    disable_final_cleanup = bool(getattr(opt, "prism_disable_final_cleanup_prune", True))
    cleanup_executed = bool(prism_enabled and not disable_final_cleanup)
    cleanup_pruned = 0
    pre_cleanup_ckpt = ""
    post_cleanup_ckpt = ""
    pre_cleanup_checkpoint_forced = False
    save_pre_cleanup_requested = bool(getattr(opt, "prism_save_pre_cleanup_checkpoint", True))
    pre_triangles = int(triangles._triangle_indices.shape[0])
    pre_vertices = int(triangles.vertices.shape[0])

    if cleanup_executed:
        # In PRISM mode, always persist checkpoints around the final cleanup for auditability.
        if prism_enabled:
            pre_cleanup_checkpoint_forced = not bool(save_pre_cleanup_requested)
            pre_cleanup_ckpt = _save_final_cleanup_checkpoint(
                scene=scene,
                triangles=triangles,
                iteration=int(iteration),
                tag="pre_cleanup",
            )
        viewpoint_stack = scene.getTrainCameras().copy()
        triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        while viewpoint_stack:
            viewpoint_cam = viewpoint_stack.pop(0)
            render_pkg = render(viewpoint_cam, triangles, pipe, bg)

            importance_score = render_pkg["max_blending"].detach()
            mask = importance_score > triangles.importance_score
            triangles.importance_score[mask] = importance_score[mask]
        mask_importance = (triangles.importance_score <= 0.5).squeeze()
        cleanup_pruned = int(mask_importance.sum().item()) if torch.is_tensor(mask_importance) else 0
        triangles.prune_triangles(~mask_importance)  # delete all the remaining triangles that do not have an influence

        device = triangles.vertices.device
        used_vertex_mask = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=device)
        if triangles._triangle_indices.numel() > 0:
            # Flatten indices and mark used vertices
            flat_indices = triangles._triangle_indices.flatten()
            used_vertex_mask[flat_indices] = True

        vertex_mask = used_vertex_mask
        triangles._prune_vertices(vertex_mask)
        if prism_enabled:
            post_cleanup_ckpt = _save_final_cleanup_checkpoint(
                scene=scene,
                triangles=triangles,
                iteration=int(iteration),
                tag="post_cleanup",
            )
    elif prism_enabled and save_pre_cleanup_requested:
        pre_cleanup_ckpt = _save_final_cleanup_checkpoint(
            scene=scene,
            triangles=triangles,
            iteration=int(iteration),
            tag="pre_cleanup_skipped",
        )

    post_triangles = int(triangles._triangle_indices.shape[0])
    post_vertices = int(triangles.vertices.shape[0])
    if (post_triangles != pre_triangles) or (post_vertices != pre_vertices):
        _sync_prism_topology_change(
            prism_state=prism_state,
            triangles=triangles,
            iteration=int(iteration),
            reason="final_cleanup",
        )

    prism_state["final_cleanup_enabled"] = 1 if cleanup_executed else 0
    prism_state["final_cleanup_pruned"] = int(cleanup_pruned)
    prism_state["pre_cleanup_checkpoint"] = str(pre_cleanup_ckpt)
    final_cleanup_payload = {
        "iteration": int(iteration),
        "prism_enabled": bool(prism_enabled),
        "compaction_ran": bool(isinstance(compaction_result, dict) and compaction_result.get("ran", False)),
        "compaction_source_checkpoint_dir": str(prism_state.get("compaction_source_checkpoint_dir", "")),
        "compaction_best_geometry_checkpoint_dir": str(prism_state.get("compaction_best_geometry_checkpoint_dir", "")),
        "compaction_best_speed_checkpoint_dir": str(prism_state.get("compaction_best_speed_checkpoint_dir", "")),
        "compaction_final_checkpoint_dir": str(prism_state.get("compaction_final_checkpoint_dir", "")),
        "final_cleanup_enabled": bool(cleanup_executed),
        "final_cleanup_pruned": int(cleanup_pruned),
        "cleanup_executed": bool(cleanup_executed),
        "cleanup_pruned": int(cleanup_pruned),
        "pre_cleanup_checkpoint": str(pre_cleanup_ckpt),
        "pre_cleanup_checkpoint_forced": bool(pre_cleanup_checkpoint_forced),
        "save_pre_cleanup_requested": bool(save_pre_cleanup_requested),
        "post_cleanup_checkpoint": str(post_cleanup_ckpt),
        "pre_prune_triangle_count": int(pre_triangles),
        "post_prune_triangle_count": int(post_triangles),
        "pre_prune_vertex_count": int(pre_vertices),
        "post_prune_vertex_count": int(post_vertices),
    }
    relaxed_records = list(prism_state.get("relaxed_commit_records", []))
    active_relaxed_records = [r for r in relaxed_records if not bool(r.get("validation_rollback", False))]
    last_active_relaxed = active_relaxed_records[-1] if active_relaxed_records else None
    retained_audit = {
        "iteration": int(iteration),
        "final_triangle_count": int(post_triangles),
        "relaxed_commit_count": int(len(relaxed_records)),
        "active_relaxed_commit_count": int(len(active_relaxed_records)),
        "validation_rolled_back_relaxed_commit_count": int(len(relaxed_records) - len(active_relaxed_records)),
        "relaxed_commit_records": relaxed_records,
        "last_relaxed_post_commit_triangle_count": int(last_active_relaxed["post_commit_triangle_count"])
        if last_active_relaxed is not None
        else 0,
        "relaxed_topology_retained": bool(
            last_active_relaxed is not None and int(post_triangles) <= int(last_active_relaxed["post_commit_triangle_count"])
        ),
        "relaxed_topology_erased": bool(
            last_active_relaxed is not None and int(post_triangles) > int(last_active_relaxed["post_commit_triangle_count"])
        ),
        "triangle_delta_vs_last_relaxed_commit": int(
            post_triangles - int(last_active_relaxed["post_commit_triangle_count"])
        )
        if last_active_relaxed is not None
        else 0,
    }
    final_cleanup_payload["relaxed_retained_topology_audit"] = retained_audit
    prism_state["relaxed_topology_retained"] = 1 if bool(retained_audit["relaxed_topology_retained"]) else 0
    prism_state["relaxed_topology_erased"] = 1 if bool(retained_audit["relaxed_topology_erased"]) else 0
    debug_dir = os.path.join(scene.model_path, "prism_debug")
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, "final_cleanup_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_cleanup_payload, f, indent=2)
    with open(os.path.join(debug_dir, "relaxed_retained_topology_audit.json"), "w", encoding="utf-8") as f:
        json.dump(retained_audit, f, indent=2)
    print(
        "[PRISM][FinalCleanup] enabled={} pruned={} pre_ckpt={} post_ckpt={}".format(
            bool(cleanup_executed),
            int(cleanup_pruned),
            str(pre_cleanup_ckpt) if pre_cleanup_ckpt else "none",
            str(post_cleanup_ckpt) if post_cleanup_ckpt else "none",
        )
    )

    if wandb_run is not None:
        _wandb_log_filtered(
            wandb_run=wandb_run,
            payload={
                "prism/final_cleanup_enabled": float(prism_state.get("final_cleanup_enabled", 0)),
                "prism/final_cleanup_pruned": float(prism_state.get("final_cleanup_pruned", 0)),
                "prism/pre_cleanup_checkpoint": 1.0 if str(pre_cleanup_ckpt) else 0.0,
                "prism/final_pre_cleanup_triangle_count": float(pre_triangles),
                "prism/final_post_cleanup_triangle_count": float(post_triangles),
                "prism/final_pre_cleanup_vertex_count": float(pre_vertices),
                "prism/final_post_cleanup_vertex_count": float(post_vertices),
                "prism/relaxed_commit_count": float(len(relaxed_records)),
                "prism/active_relaxed_commit_count": float(len(active_relaxed_records)),
                "prism/validation_rolled_back_relaxed_commit_count": float(
                    len(relaxed_records) - len(active_relaxed_records)
                ),
                "prism/relaxed_topology_retained": float(prism_state.get("relaxed_topology_retained", 0)),
                "prism/relaxed_topology_erased": float(prism_state.get("relaxed_topology_erased", 0)),
                "prism/relaxed_triangle_delta_vs_last_commit": float(
                    retained_audit.get("triangle_delta_vs_last_relaxed_commit", 0)
                ),
                "prism/post_commit_recollect_remaining": float(
                    getattr(prism_state.get("controller", None), "post_commit_recollect_remaining", 0)
                    if prism_state.get("controller", None) is not None
                    else 0
                ),
                "prism/topology_change_iter": float(prism_state.get("last_topology_change_iter", -1)),
                "mesh/triangle_count": float(post_triangles),
                "mesh/vertex_count": float(post_vertices),
                "mesh/final_checkpoint_triangle_count": float(post_triangles),
                "mesh/final_checkpoint_vertex_count": float(post_vertices),
            },
            step=int(iteration),
            log_state=wandb_log_state,
        )

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
                        "split_strategy": getattr(args, "split_strategy", ""),
                        "split_file": getattr(args, "split_file", ""),
                        "iterations": getattr(args, "iterations", -1),
                        "enable_ground_regularization": bool(getattr(args, "enable_ground_regularization", False)),
                        "enable_prism_pruning": bool(getattr(args, "enable_prism_pruning", False)),
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


def _to_wandb_scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = float(value.detach().item())
    elif isinstance(value, np.generic):
        value = float(value)
    elif isinstance(value, bool):
        value = float(value)
    elif isinstance(value, numbers.Number):
        value = float(value)
    else:
        return None
    if not np.isfinite(value):
        return None
    return value


def _wandb_log_filtered(wandb_run, payload: dict, step: int, log_state: dict = None):
    if wandb_run is None or (not isinstance(payload, dict)) or len(payload) == 0:
        return
    if log_state is None:
        log_state = {}
    last_values = log_state.setdefault("last_values", {})

    clean = {}
    for k, v in payload.items():
        if isinstance(v, (list, tuple)):
            if len(v) > 0:
                clean[k] = v
            continue
        scalar = _to_wandb_scalar(v)
        if scalar is None:
            continue
        # Avoid sending unchanged scalar values every step.
        if k in last_values and abs(float(last_values[k]) - float(scalar)) < 1e-12:
            continue
        clean[k] = float(scalar)

    if len(clean) == 0:
        return
    wandb_run.log(clean, step=step)
    for k, v in clean.items():
        if isinstance(v, float):
            last_values[k] = float(v)


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
    wandb_log_state=None,
    wandb_scalar_log_interval=10,
):
    def _safe_lpips_eval(pred_img, target_img):
        # Downsample LPIPS eval inputs to avoid OOM on high-res views.
        max_side = 1024
        eval_pred = pred_img
        eval_gt = target_img
        h, w = int(pred_img.shape[-2]), int(pred_img.shape[-1])
        long_side = max(h, w)
        if long_side > max_side:
            scale = float(max_side) / float(long_side)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            eval_pred = F.interpolate(pred_img[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
            eval_gt = F.interpolate(target_img[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
        try:
            return lpips_fn(eval_pred, eval_gt).mean().double()
        except torch.OutOfMemoryError:
            print("[WARN] LPIPS eval OOM, skipping this LPIPS sample.")
            torch.cuda.empty_cache()
            return None
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print("[WARN] LPIPS eval runtime OOM, skipping this LPIPS sample.")
                torch.cuda.empty_cache()
                return None
            raise

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/pixel_loss', pixel_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    if wandb_run and (int(wandb_scalar_log_interval) <= 0 or (iteration % int(wandb_scalar_log_interval) == 0)):
        _wandb_log_filtered(
            wandb_run=wandb_run,
            payload={
                "train_loss/pixel_loss": float(pixel_loss.item()),
                "train_loss/total_loss": float(loss.item()),
                "train/iter_time_ms": float(elapsed),
            },
            step=iteration,
            log_state=wandb_log_state,
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
            with torch.no_grad():
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
    # Always keep numerical validation independent from fixed-view image logging.
    if iteration in testing_iterations:
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                pixel_loss_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                lpips_count = 0
                ground_l1_sum = 0.0
                ground_psnr_sum = 0.0
                ground_ratio_sum = 0.0
                ground_view_count = 0
                total_time = 0.0
                with torch.no_grad():
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
                        lpips_val = _safe_lpips_eval(image, gt_image)
                        if lpips_val is not None:
                            lpips_test += lpips_val
                            lpips_count += 1

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
                if lpips_count > 0:
                    lpips_test /= lpips_count
                else:
                    lpips_test = torch.tensor(float("nan"), device="cuda", dtype=torch.float64)
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
                if wandb_run and (int(wandb_scalar_log_interval) <= 0 or (iteration % int(wandb_scalar_log_interval) == 0)):
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
                    _wandb_log_filtered(
                        wandb_run=wandb_run,
                        payload=wandb_payload,
                        step=iteration,
                        log_state=wandb_log_state,
                    )

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_iteration", type=int, default=None)

    parser.add_argument("--enable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", default="mesh-splatting", type=str)
    parser.add_argument("--wandb_entity", default="", type=str)
    parser.add_argument("--wandb_group", default="", type=str)
    parser.add_argument('--wandb_name', default="Test", type=str)
    parser.add_argument("--wandb_image_log_interval", type=int, default=1000)
    parser.add_argument("--wandb_scalar_log_interval", type=int, default=10)
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
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)

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
             args.checkpoint_iterations,
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
             wandb_scalar_log_interval=int(args.wandb_scalar_log_interval),
             wandb_fixed_views=int(args.wandb_fixed_views),
             wandb_fixed_view_indices=str(args.wandb_fixed_view_indices),
             wandb_disable_fixed_views=bool(args.wandb_disable_fixed_views),
             )
    
    # All done
    print("\nTraining complete.")
