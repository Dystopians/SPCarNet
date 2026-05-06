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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        # Optional per-image binary ground masks.
        self.ground_masks = False
        self.enable_ground_masks = False
        self.ground_mask_dir = ""
        self.ground_mask_matching = "auto"
        self.ground_mask_suffix = ".png"
        self.ground_mask_missing_strategy = "empty"
        self.ground_mask_nearest_max_gap = 6
        self.ground_mask_threshold = 127
        self.ground_mask_label_value = -1
        self.ground_mask_label_rgb = ""
        self.ground_mask_debug_vis = False
        self.ground_mask_debug_dir = ""
        self.ground_mask_debug_max = 8
        # Camera split configuration for COLMAP scenes.
        # split_strategy:
        # - "llff": default every-N holdout
        # - "file": load explicit train/test split from split_file
        self.split_strategy = "llff"
        self.split_file = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 1.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.lambda_dssim = 0.2

        self.densification_interval = 500

        self.densify_from_iter = 500
        self.densify_until_iter = 10000
        self.skip_restricted_delaunay = False
        self.freeze_topology_updates = False

        self.random_background = False
        
        self.feature_lr = 0.0016 # 0.0025
        self.max_points = 4000000

        # Opacity & weight
        self.set_weight = 0.28
        self.weight_lr =  0.03
        self.lambda_weight = 1.9e-06

        # Normal loss
        self.iteration_mesh = 5000
        self.lambda_normals = 0.00005
        self.lambda_normals_super = 0.01

        self.add_percentage = 1.23

        self.set_sigma = 1.0

        # Add new triangles or vertices
        self.intervall_add_triangles = 500

        # Prune triangles and vertices
        self.prune_triangles_threshold = 0.235

        # PARAMETER SECOND STAGE
        self.lr_triangles_points_init = 0.0015

        self.start_opacity_floor = 5000

        self.start_pruning = 4000
        self.sigma_until = 30000
        self.final_opacity_iter = 24000

        self.sigma_start = 0

        self.splitt_large_triangles = 100
        self.start_upsampling = 20000
        self.upscaling_factor = 2

        self.size_probs_zero = 7.5e-05
        self.size_probs_zero_image_space = 0.0

        self.prune_size = 1400

        self.lambda_vertex = 0.00025
        self.max_diff_threshold = 0.5
        self.start_vertex_opt = 12000

        self.lamba_depth = 0.05

        self.depth_lambda_init = 0.01
        self.depth_lambda_final = 0.001
        # Sparse COLMAP depth supervision (train-view sparse correspondences).
        self.enable_sparse_colmap_depth_loss = False
        self.lambda_sparse_colmap_depth = 0.01
        self.sparse_colmap_depth_start_iter = 1000
        self.sparse_colmap_depth_warmup_iters = 3000
        self.sparse_colmap_depth_decay_start_iter = -1
        self.sparse_colmap_depth_decay_end_iter = -1
        self.sparse_colmap_depth_decay_final_mult = 1.0
        self.sparse_colmap_depth_min_matches = 32
        self.sparse_colmap_depth_loss_space = "depth"
        self.sparse_colmap_depth_robust_beta = 0.05
        self.sparse_colmap_depth_sample_mode = "random"
        self.sparse_colmap_depth_low_error_fraction = 1.0
        self.sparse_colmap_depth_enable_in_recovery = False
        self.sparse_colmap_depth_enable_in_final_finetune = False
        # Optional render-teacher distillation from a pre-rendered stronger
        # checkpoint. This is disabled by default and is intended for topology-
        # constrained recovery rows that need to regain clean-baseline appearance.
        self.enable_teacher_render_loss = False
        self.teacher_render_dir = ""
        self.lambda_teacher_render = 0.0
        self.teacher_render_dssim = 0.2
        self.teacher_render_mask_mode = "none"
        self.teacher_render_error_margin = 0.0
        self.teacher_render_start_iter = 0
        self.teacher_render_warmup_iters = 1000
        self.teacher_render_decay_start_iter = -1
        self.teacher_render_decay_end_iter = -1
        self.teacher_render_decay_final_mult = 1.0
        # Optional one-sided parent appearance rollback. Unlike teacher-render
        # distillation, this loss does not copy the parent image. It penalizes
        # only pixels where the current render is worse than the parent render
        # against the ground-truth training image, optionally aggregating the
        # worst residual tail with CVaR.
        self.enable_parent_render_rollback_loss = False
        self.parent_render_rollback_dir = ""
        self.lambda_parent_render_rollback = 0.0
        self.parent_render_rollback_start_iter = 0
        self.parent_render_rollback_warmup_iters = 1000
        self.parent_render_rollback_decay_start_iter = -1
        self.parent_render_rollback_decay_end_iter = -1
        self.parent_render_rollback_decay_final_mult = 1.0
        self.parent_render_rollback_margin_abs = 0.0
        self.parent_render_rollback_margin_rel = 0.0
        self.parent_render_rollback_huber_delta = 0.02
        self.parent_render_rollback_aggregation = "mean"
        self.parent_render_rollback_cvar_fraction = 0.1
        self.parent_render_rollback_cvar_min_pixels = 1024
        self.parent_render_rollback_patch_radius = 0
        self.parent_render_rollback_patch_reduce = "center"
        self.parent_render_rollback_error_space = "l1"
        self.parent_render_rollback_dssim_weight = 0.0
        self.parent_render_rollback_edge_weight = 0.0
        self.parent_render_rollback_ssim_window = 11
        self.parent_render_rollback_edge_guidance_weight = 0.0
        # Optional checkpoint geometry anchor for topology-frozen recovery.
        # It keeps vertices close to the loaded checkpoint while appearance
        # losses refine radiance, preventing teacher-render finetuning from
        # drifting sparse depth geometry.
        self.enable_checkpoint_geometry_anchor = False
        self.lambda_checkpoint_geometry_anchor = 0.0
        self.checkpoint_geometry_anchor_start_iter = 0
        self.checkpoint_geometry_anchor_warmup_iters = 1000
        self.checkpoint_geometry_anchor_decay_start_iter = -1
        self.checkpoint_geometry_anchor_decay_end_iter = -1
        self.checkpoint_geometry_anchor_decay_final_mult = 1.0
        self.checkpoint_geometry_anchor_huber_delta = 0.01
        self.enable_checkpoint_render_geometry_anchor = False
        self.lambda_checkpoint_render_depth_anchor = 0.0
        self.lambda_checkpoint_render_normal_anchor = 0.0
        self.checkpoint_render_geometry_anchor_start_iter = 0
        self.checkpoint_render_geometry_anchor_warmup_iters = 1000
        self.checkpoint_render_geometry_anchor_huber_delta = 0.02
        # Optional one-sided sparse-depth parent rollback loss. This consumes a
        # train/calibration sentinel cache and penalizes only current-vs-parent
        # sparse-depth regressions, never improvements over the parent.
        self.enable_sparse_depth_parent_rollback_loss = False
        self.sparse_depth_parent_rollback_cache = ""
        self.lambda_sparse_depth_parent_rollback = 0.0
        self.sparse_depth_parent_rollback_start_iter = 0
        self.sparse_depth_parent_rollback_warmup_iters = 1000
        self.sparse_depth_parent_rollback_margin_abs = 0.0
        self.sparse_depth_parent_rollback_margin_rel = 0.0
        self.sparse_depth_parent_rollback_huber_delta = 0.05
        self.sparse_depth_parent_rollback_combined_mae_beta = 1.0
        self.sparse_depth_parent_rollback_cluster_balance = False
        self.sparse_depth_parent_rollback_regressed_only = False
        self.sparse_depth_parent_rollback_cluster_top_k = 0
        self.sparse_depth_parent_rollback_max_points_per_view = 500
        self.sparse_depth_parent_rollback_loss_space = "combined"
        self.sparse_depth_parent_rollback_aggregation = "mean"
        self.sparse_depth_parent_rollback_cvar_fraction = 0.2
        self.sparse_depth_parent_rollback_cvar_min_points = 16
        self.sparse_depth_parent_rollback_pixel_radius = 0
        self.sparse_depth_parent_rollback_patch_reduce = "center"
        self.sparse_depth_parent_rollback_allow_test_cache = False
        self.sparse_depth_parent_rollback_strict = False
        self.lambda_lpips_loss = 0.0
        self.lpips_loss_start_iter = 0
        self.lpips_loss_warmup_iters = 1000
        self.lpips_loss_max_side = 512

        # Ground-plane estimation (for ground-aware regularization).
        self.enable_ground_plane_estimation = False
        self.enable_ground_plane_fit = False
        self.ground_plane_source_priority = "colmap,depth,mesh"
        self.ground_plane_cache_file = "ground_plane_cache.json"
        self.ground_plane_recompute_interval = 0
        self.ground_plane_force_recompute = False
        self.ground_plane_min_points = 200
        self.ground_plane_ransac_iters = 600
        self.ground_plane_ransac_dist_thresh = 0.04
        self.ground_plane_inlier_ratio_min = 0.35
        self.ground_plane_track_len_min = 3
        self.ground_plane_obs_min = 2
        self.ground_plane_obs_ratio_min = 0.5
        self.ground_plane_colmap_error_max = 2.0
        self.ground_plane_depth_max_samples_per_view = 4000
        self.ground_plane_depth_sample_stride = 4
        self.ground_plane_depth_inv_min = 1e-6
        self.ground_plane_mesh_sample_max = 30000
        self.ground_plane_axis_consistency_min = 0.45
        self.ground_plane_outlier_quantile = 0.95
        self.ground_plane_use_if_poor = False
        self.ground_plane_diag_every = 1000
        self.ground_plane_diag_save = False
        self.ground_plane_diag_dir = ""

        # Ground-aware geometry regularization losses.
        self.enable_ground_regularization = False
        self.enable_ground_plane_loss = False
        self.enable_ground_normal_loss = False
        self.enable_ground_smoothness_loss = False
        self.enable_ground_mesh_assignment = False
        self.debug_save_ground_visualizations = False
        self.ground_debug_vis_every = 1000
        self.ground_debug_vis_max = 16
        self.ground_debug_vis_dir = ""
        self.ground_reg_start_iter = 2000
        self.ground_reg_warmup_iters = 3000
        self.lambda_ground_plane = 0.0
        self.lambda_ground_normal = 0.0
        self.lambda_ground_smoothness = 0.0
        # Global multiplier on the full ground-regularization term.
        self.ground_reg_global_scale = 1.0
        # Optional adaptive scaling target: keep Lground around
        # target_ratio * current image loss. Set 0 to disable.
        self.ground_reg_target_ratio = 0.0
        self.ground_reg_adaptive_ema_decay = 0.95
        self.ground_reg_adaptive_min_scale = 1.0
        self.ground_reg_adaptive_max_scale = 50.0
        # Optional cap on final Lground value (0 disables cap).
        self.ground_reg_max_total = 0.0
        # If >=0, smoothness starts from this iteration (independent gate).
        self.ground_smooth_start_iter = -1
        self.ground_reg_huber_delta = 0.03
        self.ground_assign_min_pixels = 30
        self.ground_assign_ema_decay = 0.9
        self.ground_plane_min_ratio = 0.7
        self.ground_plane_max_abs_height = 0.4
        self.ground_normal_min_ratio = 0.9
        self.ground_normal_max_abs_height = 0.2
        self.ground_smooth_min_ratio = 0.8
        self.ground_smooth_max_abs_height = 0.3
        self.ground_smooth_max_pairs = 50000
        self.ground_smooth_fallback_edges_max = 80000
        # If local ground triangles exceed this, skip CPU tri-adjacency build
        # and use fallback edge smoothing path. Keep default for prior behavior.
        self.ground_smooth_tri_adj_max_triangles = 4096

        # Robust multi-view ground association (image-space -> mesh-space).
        self.ground_assoc_min_observations = 3
        self.ground_assoc_min_ground_ratio = 0.75
        self.ground_assoc_min_view_consistency = 0.6
        self.ground_assoc_per_view_ground_ratio = 0.5
        self.ground_assoc_boundary_margin = 0.1
        self.ground_assoc_confidence_min = 0.45
        self.ground_assoc_use_cache = True
        self.ground_assoc_cache_file = "ground_assoc_cache.pt"
        self.ground_assoc_cache_every = 1000
        self.ground_assoc_debug_every = 1000
        self.ground_assoc_debug_dir = ""
        self.ground_assoc_hist_bins = 80

        # PRISM-Prune scaffolding (disabled by default; neutral to legacy behavior).
        self.enable_prism_pruning = False
        self.prism_collect_stats = False
        self.prism_stats_warmup_iters = 2000
        self.prism_collect_interval = 100
        # Heavy PRISM feature recomputation (structure + sparse support) interval.
        # Keep this decoupled from light stat collection to avoid step-time spikes.
        self.prism_score_recompute_interval = 500
        # Legacy threshold above which PRISM switches from full heavy eval
        # to the scalable two-stage heavy-eval path.
        self.prism_max_triangles_for_heavy_metrics = 400000
        self.prism_heavy_eval_budget = 120000
        self.prism_heavy_eval_neighbor_rings = 2
        self.prism_force_full_heavy_eval_below = 400000
        self.prism_skip_heavy_eval_for_far_field = False
        self.prism_dead_prune_ratio = 0.0
        self.prism_candidate_prune_ratio = 0.0
        self.prism_recovery_iters = 0
        self.prism_use_counterfactual_gate = False
        self.prism_use_ground_protect = False
        self.prism_use_roi_protect = False
        self.prism_calib_num_hard_train_views = 8
        self.prism_calib_num_buffer_views = 8
        self.prism_calib_hard_pool_size = 64
        self.prism_calib_prefer_observable_views = True
        self.prism_calib_min_depth_matches_per_view = 24
        self.prism_calib_min_normal_matches_per_view = 8
        self.prism_calib_diverse_views = False
        self.prism_calib_diverse_test_views = 0
        self.prism_calib_diverse_train_views = 0
        self.prism_save_debug_json = False
        self.prism_changed_pixel_threshold = 0.02
        self.prism_gate_min_delta_psnr_db = -0.05
        self.prism_gate_max_delta_mae = 0.002
        self.prism_gate_max_delta_absrel = 0.0008
        self.prism_gate_max_baseline_absrel_for_absrel_check = float("inf")
        self.prism_gate_max_delta_mean_angle_deg = 0.3
        self.prism_gate_max_changed_pixel_ratio = 0.005
        self.prism_gate_min_valid_depth_matches = 128
        self.prism_gate_min_valid_normal_matches = 64
        self.prism_proxy_max_points_per_view = 3000
        self.prism_proxy_point_error_max = 2.0
        self.prism_proxy_normal_knn = 24
        self.prism_disable_final_cleanup_prune = True
        self.prism_save_pre_cleanup_checkpoint = True
        # PRISM pipeline state machine
        self.prism_geometry_acq_until_iter = -1
        self.prism_stats_collection_iters = 500
        self.prism_dead_rounds = 1
        self.prism_candidate_rounds = 3
        self.prism_candidate_prune_ratio_per_round = 0.015
        self.prism_candidate_max_count_per_round = 0
        self.prism_candidate_microbatch_gate = False
        self.prism_candidate_microbatch_size = 256
        self.prism_candidate_microbatch_max_batches = 0
        self.prism_candidate_quality_rank = False
        self.prism_candidate_quality_prune_weight = 1.0
        self.prism_candidate_quality_render_penalty = 0.5
        self.prism_candidate_quality_geometry_penalty = 0.5
        self.prism_candidate_quality_orientation_penalty = 0.25
        self.prism_candidate_quality_utility_penalty = 0.25
        self.prism_candidate_quality_uncertainty_penalty = 0.25
        self.prism_candidate_measured_impact_rank = False
        self.prism_candidate_measured_pool_multiplier = 4.0
        self.prism_candidate_measured_group_size = 256
        self.prism_candidate_measured_max_groups = 8
        self.prism_post_commit_candidate_refresh = False
        self.prism_post_commit_refresh_min_prune_score = 1e-6
        self.prism_post_commit_relaxed_max_commits = 0
        self.prism_post_commit_relaxed_strict_gate = False
        self.prism_post_commit_relaxed_min_delta_psnr = 0.0
        self.prism_post_commit_relaxed_max_delta_mae = 0.0
        self.prism_post_commit_relaxed_max_delta_absrel = 0.0
        self.prism_post_commit_relaxed_max_delta_mean_angle = 0.0
        self.prism_post_commit_relaxed_max_changed_pixel_ratio = 0.0025
        self.prism_no_candidate_retry_iters = 10
        self.prism_adaptive_candidate_retry_on_rollback = False
        self.prism_adaptive_candidate_ratio_decay = 0.5
        self.prism_adaptive_candidate_min_ratio = 0.0025
        self.prism_adaptive_candidate_max_rollback_retries = 3
        # Adaptive CSEF edit policy. This replaces fixed ratio sweeps with a
        # scene-state controller when enabled.
        self.prism_enable_adaptive_csef_policy = False
        self.prism_adaptive_policy_min_ratio = 0.006
        self.prism_adaptive_policy_max_ratio = 0.020
        self.prism_adaptive_policy_initial_ratio = 0.012
        self.prism_adaptive_policy_target_accept_margin = 0.55
        self.prism_adaptive_policy_rollback_decay = 0.55
        self.prism_adaptive_policy_accept_growth = 1.18
        self.prism_adaptive_policy_no_candidate_decay = 0.75
        self.prism_adaptive_policy_cooldown_iters = 20
        self.prism_adaptive_policy_max_candidate_count = 0
        self.prism_adaptive_policy_min_candidate_count = 512
        self.prism_adaptive_policy_depth_degrade_absrel = 0.004
        self.prism_adaptive_policy_normal_degrade_deg = 0.10
        self.prism_adaptive_policy_render_degrade_psnr = -0.05
        self.prism_adaptive_policy_uncertainty_high = 0.35
        self.prism_adaptive_policy_geometry_keep_high = 0.04
        self.prism_adaptive_policy_orientation_keep_high = 0.04
        self.prism_adaptive_policy_reliable_absrel_max = 2.0
        self.prism_adaptive_policy_strict_gate_after_rejects = 1
        self.prism_adaptive_policy_normal_repair_penalty_boost = 0.8
        self.prism_adaptive_policy_geometry_repair_penalty_boost = 0.8
        self.prism_adaptive_policy_uncertainty_penalty_boost = 0.6
        self.prism_adaptive_policy_cold_start_rounds = 1
        self.prism_adaptive_policy_cold_start_gate_scale = 0.70
        self.prism_adaptive_policy_cold_start_ratio_damping = 0.96
        self.prism_adaptive_policy_cold_start_quality_rank = False
        self.prism_adaptive_policy_enable_measured_rank = True
        self.prism_adaptive_policy_enable_microbatch_gate = True
        self.prism_adaptive_policy_microbatch_size = 512
        self.prism_adaptive_policy_microbatch_max_batches = 0
        self.prism_freeze_densification_after_first_commit = False
        self.prism_recovery_iters = 400
        self.prism_post_commit_recollect_iters = 300
        self.prism_force_recompute_scores_after_recollect = True
        self.prism_final_finetune_iters = 500
        self.prism_enable_compaction_stage = False
        self.prism_compaction_source_preference = "best_geometry"
        self.prism_compaction_rounds = 2
        self.prism_compaction_microbatch_active_ratio = 0.0035
        self.prism_compaction_max_microbatches_per_round = 6
        self.prism_compaction_candidate_pool_multiplier = 6.0
        self.prism_compaction_min_prune_count = 256
        self.prism_compaction_roi_budget_fraction = 0.10
        self.prism_compaction_near_field_budget_fraction = 0.25
        self.prism_compaction_roi_signal_threshold = 0.05
        self.prism_compaction_near_field_area_percentile = 80.0
        self.prism_topology_freeze_during_stats = True
        self.prism_round_checkpoint = True
        # Optional lightweight teacher distillation during recovery.
        self.prism_enable_teacher_rgb_distill = False
        self.prism_enable_teacher_depth_distill = False
        self.prism_teacher_rgb_lambda = 0.01
        self.prism_teacher_depth_lambda = 0.002
        self.prism_teacher_num_views = 8
        # PRISM global validation + rollback gate
        self.prism_validation_interval = 1000
        self.prism_validation_max_views = 32
        self.prism_validation_num_buffer_views = 16
        self.prism_validation_num_train_views = 16
        self.prism_validation_train_pool_size = 128
        self.prism_validation_prefer_observable_train_views = True
        self.prism_validation_min_depth_matches_per_view = 24
        self.prism_validation_min_normal_matches_per_view = 8
        self.prism_validation_min_valid_depth_matches = 128
        self.prism_validation_min_valid_normal_matches = 64
        self.prism_rollback_absrel_rel_thresh = 0.01
        self.prism_rollback_mean_angle_thresh = 0.4
        self.prism_rollback_psnr_drop_thresh = 0.10
        self.prism_rollback_mae_increase_thresh = 0.003

        # PRISM scoring weights (utility / redundancy).
        self.prism_utility_w_vis = 0.30
        self.prism_utility_w_sens = 0.25
        self.prism_utility_w_geo = 0.20
        self.prism_utility_w_viewdiv = 0.15
        self.prism_utility_w_edge = 0.10
        self.prism_redund_w_flat = 0.70
        self.prism_redund_w_coplanar = 0.30
        self.prism_keep_support_count_min = 12.0
        self.prism_keep_plane_residual_max = 0.02
        self.prism_keep_normal_residual_max_deg = 15.0
        self.prism_keep_orientation_dihedral_min_deg = 12.0
        self.prism_keep_orientation_local_var_min = 0.25
        self.prism_keep_geometry_threshold = 0.6
        self.prism_keep_orientation_threshold = 0.6
        self.prism_keep_render_threshold = 0.6
        self.prism_keep_geometry_bonus = 1.0
        self.prism_keep_orientation_bonus = 1.0
        self.prism_keep_render_bonus = 1.0
        self.prism_candidate_block_geometry_keep_threshold = 0.6
        self.prism_protected_dilation_rings = 1

        # PRISM robust normalization.
        self.prism_norm_percentile_low = 5.0
        self.prism_norm_percentile_high = 95.0
        self.prism_norm_eps = 1e-6

        # PRISM classifier thresholds.
        self.prism_thresh_protected_edge = 0.60
        self.prism_thresh_protected_geo = 0.75
        self.prism_thresh_protected_sens = 0.80
        self.prism_thresh_protected_unc = 0.65
        self.prism_thresh_dead_vis = 0.02
        self.prism_thresh_dead_sens = 0.03
        self.prism_thresh_dead_geo = 0.05
        self.prism_thresh_dead_edge = 0.10
        self.prism_thresh_suspicious_vis = 0.05
        self.prism_thresh_suspicious_geo = 0.15
        self.prism_thresh_suspicious_unc = 0.50

        # PRISM risk controls.
        self.prism_boundary_risk_value = 1.0
        self.prism_nonmanifold_risk_value = 1.0
        self.prism_recent_age_iters = 500
        self.prism_ground_protect_bonus = 1.0
        self.prism_roi_protect_bonus = 1.0

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def update_indoor(params):
    params.add_percentage = 1.27
    params.densify_from_iter = 1000
    params.densify_until_iter = 10000
    params.feature_lr = 0.004
    params.size_probs_zero = 0.0
    params.splitt_large_triangles = 500
    params.start_pruning = 3000
    params.weight_lr = 0.05
    params.lambda_weight = 0.0
    params.lambda_normals = 0.00001
    params.lambda_normals_super = 0.01
    params.prune_size = 1300
    params.lambda_vertex = 0.00025
    params.depth_lambda_init = 0.0
    params.depth_lambda_final = 0.0
    params.iteration_mesh = 12000
    return params
