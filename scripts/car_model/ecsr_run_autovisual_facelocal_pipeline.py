#!/usr/bin/env python3
"""Run an automatic face-local visual residual train/eval pipeline.

This script is a bounded coordinator around the existing ECSR surface-residual
tools.  It adds a scene-agnostic pipeline that:

1. builds strict train-only face-local SH residual candidate plans;
2. refits per-face materialization alphas on train evidence;
3. runs render-calibrated train-val selection over small face subsets;
4. records report-only held-out deltas without using them for selection.

The defaults are intentionally not scene-specific.  Use ``--dry_run`` to emit
the exact commands and manifests without touching model checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("PSNR", "SSIM", "LPIPS")
DEFAULT_POLICY_ROOT = "outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix"
DEFAULT_OUTPUT_ROOT = "outputs/carnet/meshsplatopt/ecsr_autovisual_facelocal_v1"
DEFAULT_DATASET_ROOT = "/data/peilincai/mesh_datasets/mipnerf360"
DEFAULT_PHASEJ_TEST_METHOD = "ours_26000_phasej_guarded_adaptedge_ela"
DEFAULT_PHASEJ_TRAINVAL_METHOD = "ours_26000_phasej_trainval_gate_rendercalib_v1"
DEFAULT_BASE_METHOD = "ours_26000_phasef_extra_compact_base"
OUTDOOR_SCENES = {"bicycle", "flowers", "garden", "stump", "treehill"}
DEFAULT_RENDER_REGION_CARRIER_TEMPLATE = (
    "outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/"
    "{scene}/render_visible_region_carriers.json"
)

REPRESENTATION_DEFAULTS: dict[str, Any] = {
    "delta_min_policy_val_samples": 512,
    "delta_min_policy_val_adaptive_sample_fraction": 0.0,
    "delta_min_policy_val_adaptive_min_samples": 0,
    "delta_min_policy_val_unique_faces": 16,
    "delta_crossfold_folds": 4,
    "delta_crossfold_min_passing_folds": 3,
    "delta_shared_residual_field": False,
    "delta_shared_residual_field_anchors": 16,
    "delta_shared_residual_field_sigma": 0.0,
    "delta_shared_residual_field_lr": 0.0,
    "delta_shared_residual_field_weight_l2": 1.0e-4,
    "delta_shared_residual_field_view_hinge_weight": 0.0,
    "delta_shared_residual_field_view_hinge_min_samples": 16,
    "delta_shared_residual_field_duplicate_smooth_weight": 0.0,
    "delta_patch_cert_cluster_basis_mode": "chart_linear",
    "delta_patch_cert_cluster_basis_steps": 240,
    "delta_patch_cert_cluster_basis_min_samples": 32,
    "delta_patch_cert_cluster_basis_max_fit_mse_regression": 0.02,
    "delta_patch_cert_cluster_basis_view_hinge_weight": 0.0,
    "delta_patch_cert_cluster_basis_view_hinge_min_samples": 16,
    "delta_patch_cert_cluster_basis_geometry_smooth_weight": 0.0,
    "delta_patch_cert_min_neighbor_policy_val_samples": 4,
    "delta_patch_cert_min_policy_val_samples": 16,
    "delta_patch_cert_min_relative_gain": 0.0,
    "delta_patch_cert_carrier_holdout_groups": 4,
    "delta_patch_cert_carrier_holdout_min_passing_groups": 3,
    "delta_patch_cert_carrier_holdout_auto_prefix_min_face_fraction": 0.0,
    "delta_patch_cert_carrier_holdout_auto_prefix_face_bonus": 0.0,
    "delta_region_core_weight": 1.0,
    "delta_region_context_weight": 1.0,
    "delta_region_outside_weight": 1.0,
    "delta_region_boundary_px": 0,
    "delta_render_region_objective": False,
    "delta_render_region_core_weight": 1.0,
    "delta_render_region_context_weight": 0.25,
    "delta_render_region_outside_penalty": 0.0,
    "delta_render_region_tail_cvar_weight": 0.0,
    "delta_render_region_tail_fraction": 0.25,
    "delta_render_region_min_view_samples": 16,
    "delta_bystander_zero_delta_weight": 0.0,
    "delta_bystander_zero_delta_include_context": True,
    "delta_bystander_zero_delta_min_samples": 0,
    "delta_witness_constraint_weight": 0.0,
    "delta_witness_constraint_tail_fraction": 0.25,
    "delta_witness_constraint_min_samples": 0,
    "delta_witness_constraint_margin": 0.0,
    "delta_witness_constraint_include_full_view": True,
    "delta_witness_constraint_include_region_view": True,
    "delta_witness_constraint_include_bystander_view": True,
    "filter_max_region_matches_per_plan_carrier": 3,
    "filter_min_regions": 1,
    "filter_min_changed_regions": 1,
    "filter_min_changed_fraction": 0.10,
    "filter_min_mean_core_balanced_delta": 0.0,
    "filter_min_mean_delta_psnr": 0.0,
    "filter_min_tail_core_balanced_delta": -1.0e-8,
    "filter_max_negative_core_balanced_fraction": 1.0,
    "filter_max_context_mse_regression": 1.0e-6,
    "filter_min_mean_crop_abs_diff": 0.0,
    "filter_min_max_crop_abs_diff": 0.0,
    "filter_tail_safe_shrink_on_tail_fail": False,
    "filter_tail_safe_shrink_min_scale": 0.5,
    "filter_tail_safe_shrink_min_raw_scale": 0.0,
    "filter_rollback_severe_tail_fail": False,
    "filter_rollback_tail_min_cvar_loss": 0.005,
    "filter_aggregate_subset": False,
    "filter_aggregate_subset_min_selected_carriers": 1,
    "filter_aggregate_subset_expected_view_count": 0,
    "filter_aggregate_subset_min_unique_views": 0,
    "filter_aggregate_subset_min_changed_unique_views": 0,
    "filter_aggregate_subset_min_view_coverage_fraction": 0.0,
    "filter_aggregate_subset_min_changed_view_coverage_fraction": 0.0,
    "filter_aggregate_subset_min_total_pixels": 0,
    "filter_aggregate_subset_min_changed_pixels": 0,
    "filter_aggregate_subset_min_changed_pixel_fraction": 0.0,
    "filter_aggregate_subset_expected_frame_pixels": 0,
    "filter_aggregate_subset_min_full_frame_changed_pixel_fraction": 0.0,
    "filter_aggregate_subset_min_area_weighted_core_balanced_delta": -1.0e30,
    "filter_aggregate_subset_min_dilution_adjusted_core_balanced_delta": -1.0e30,
    "filter_aggregate_subset_min_full_frame_visibility_adjusted_delta": -1.0e30,
    "filter_aggregate_subset_prefer_full_frame_visibility": False,
    "filter_aggregate_subset_tail_safe_shrink_carriers": False,
    "filter_aggregate_subset_tail_safe_shrink_scales": "1.0,0.85,0.75,0.6,0.5,0.35,0.2,0.1,0.05,0.035,0.02",
    "filter_aggregate_subset_tail_safe_shrink_min_scale": 0.02,
    "filter_risk_safe_shrink_on_train_risk_fail": False,
    "filter_risk_safe_shrink_min_scale": 0.25,
    "filter_drop_unmapped": False,
    "filter_require_positive_plan_proxy": True,
    "filter_region_source": "scene_prior",
    "filter_candidate_region_max_regions_per_carrier": 12,
    "filter_candidate_region_min_pixels": 16,
    "filter_candidate_region_bbox_pad": 8,
    "filter_candidate_region_min_alpha": 0.01,
    "filter_candidate_region_high_error_quantile": 0.0,
    "filter_candidate_region_max_views": 0,
    "filter_candidate_region_expand_faces": False,
    "filter_candidate_region_expand_min_face_pixels": 12,
    "filter_candidate_region_expand_min_face_views": 1,
    "filter_candidate_region_expand_max_faces_per_carrier": 32,
    "filter_candidate_region_frame_aware_ranking": False,
    "filter_candidate_region_min_frame_support_fraction": 0.0,
    "filter_candidate_region_min_residual_mass_fraction": 0.0,
    "filter_candidate_region_max_carriers": 0,
    "ela_alpha_holdout_safe_zero": False,
    "ela_alpha_risk_tail_fraction": 0.20,
    "ela_alpha_max_negative_gain_fraction": 1.0,
    "ela_alpha_min_tail_gain": -1.0e30,
    "ela_alpha_view_tail_scale_grid": "",
    "ela_alpha_view_tail_cvar_fraction": 0.25,
    "ela_alpha_view_tail_min_gain": -1.0e30,
    "ela_alpha_view_tail_max_negative_fraction": 1.0,
    "ela_alpha_view_tail_objective": "mse",
    "ela_alpha_view_tail_ssim_weight": 20.0,
    "ela_alpha_view_tail_lpips_weight": 20.0,
    "ela_alpha_view_tail_compute_lpips": False,
    "ela_alpha_view_tail_metric_max_side": 512,
    "ela_alpha_region_risk_enable": False,
    "ela_alpha_region_risk_objective_bad_only": False,
    "ela_alpha_region_risk_objective_max_balanced_delta": 0.0,
    "ela_alpha_region_risk_objective_max_delta_ssim": 0.0,
    "ela_alpha_region_risk_objective_min_delta_lpips": 0.0,
    "ela_alpha_region_risk_min_tail_gain": 0.0,
    "ela_alpha_region_risk_max_negative_fraction": 1.0,
    "ela_alpha_region_risk_min_regions": 1,
    "ela_local_trust_gate": False,
    "ela_local_trust_min_supports": 2,
    "ela_local_trust_max_residual_std": -1.0,
    "ela_local_trust_min_agreement": 0.0,
    "ela_local_trust_agreement_scale": 0.04,
    "ela_local_trust_confidence_quantile": -1.0,
    "ela_local_trust_min_confidence": 0.0,
    "ela_local_trust_mode": "hard",
    "ela_local_trust_min_weight": 0.0,
    "candidate_owned_refit": False,
    "candidate_region_expansion_closure": False,
    "candidate_region_expansion_core_priority": False,
    "candidate_region_expansion_core_min_samples": 0,
    "candidate_region_expansion_core_min_fraction": 0.0,
    "candidate_region_expansion_witness_rescue": False,
    "candidate_region_expansion_max_witnesses_per_carrier": 0,
    "candidate_region_pre_refit_risk_prune": False,
    "candidate_region_pre_refit_risk_min_changed_rows": 1,
    "candidate_region_pre_refit_risk_min_bad_rows": 1,
    "candidate_region_pre_refit_risk_max_bad_fraction": 0.5,
    "candidate_region_pre_refit_risk_balanced_margin": 1.0e-3,
    "candidate_region_pre_refit_risk_use_aux_metric_pair": True,
    "candidate_region_pre_refit_risk_ssim_margin": 1.0e-3,
    "candidate_region_pre_refit_risk_lpips_margin": 1.0e-3,
    "candidate_region_pre_refit_risk_max_removed_face_fraction": 0.5,
    "candidate_region_pre_refit_risk_shrink": False,
    "candidate_region_pre_refit_risk_shrink_min_scale": 0.25,
    "candidate_region_pre_refit_risk_shrink_severity_aware": False,
    "candidate_region_pre_refit_risk_shrink_severity_select_min": 0.5,
    "candidate_region_pre_refit_risk_shrink_severity_balanced_span": 0.05,
    "candidate_region_pre_refit_risk_shrink_tail_fraction": 0.25,
    "candidate_region_pre_refit_risk_local_suppression": False,
    "candidate_region_pre_refit_risk_local_suppression_scale": 0.02,
    "candidate_region_pre_refit_risk_local_suppression_min_bad_balanced": 0.02,
    "candidate_region_pre_refit_risk_local_suppression_positive_margin": 0.02,
    "candidate_region_pre_refit_risk_local_suppression_min_face_pixels": 12,
    "candidate_region_pre_refit_risk_local_suppression_max_faces_per_bad_row": 16,
    "filter_train_render_region_max_regions": 64,
    "filter_train_render_region_min_pixels": 128,
    "filter_train_render_region_min_crop_size": 32,
    "filter_train_render_region_context_pad": 16,
    "filter_train_render_region_tail_fraction": 0.25,
    "filter_train_render_region_skip_lpips": False,
    "plan_region_gate_min_regions": 0,
    "plan_region_gate_min_changed_regions": 0,
    "plan_region_gate_min_changed_fraction": 0.0,
    "plan_region_gate_min_core_balanced_delta": -1.0e30,
    "plan_region_gate_min_core_psnr_delta": -1.0e30,
    "plan_region_gate_min_tail_cvar_delta": -1.0e30,
    "plan_region_gate_max_context_mse_regression": 1.0e30,
    "plan_region_gate_max_negative_fraction": 1.0,
    "use_filtered_plan_for_selector": True,
    "selector_strict_replay_scales": "1.0",
    "selector_strict_adaptive_scale_policy": False,
    "selector_strict_adaptive_scale_min": 0.10,
    "selector_strict_adaptive_scale_max_extra": 5,
    "selector_strict_adaptive_scale_tail_fraction": 0.30,
    "selector_fit_plan_alphas": True,
    "selector_strict_fit_plan_alphas": False,
    "selector_balanced_ssim_weight": 20.0,
    "selector_balanced_lpips_weight": 20.0,
    "selector_enable_region_stable_promotion": False,
    "selector_region_min_trainval_balanced_delta": 0.0,
    "selector_region_min_mean_core_balanced_delta": 0.01,
    "selector_region_min_mean_delta_psnr": 0.001,
    "selector_region_min_changed_fraction": 0.50,
    "selector_region_max_negative_core_balanced_fraction": 0.35,
    "selector_region_max_context_mse_regression": 1.0e-6,
}


PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {
        "evidence_max_views": 2,
        "evidence_view_stride": 8,
        "delta_top_k": 128,
        "delta_steps": 20,
        "delta_max_total_samples": 20000,
        "delta_strength": 0.18,
        "delta_max_abs_rgb": 0.014,
        "delta_max_faces_to_apply": 16,
        "delta_patch_cert_max_faces_per_seed": 4,
        "trial_specs": "top1x0.5,score1x0.5",
        "selector_alpha_max": 1.0,
        "selector_alpha_steps": 8,
        "selector_alpha_max_total_samples": 20000,
        "selector_min_trainval_psnr_gain": 0.0,
        "selector_min_trainval_balanced_delta": 0.0,
        "selector_tail_min_trainval_balanced_delta": 0.0,
        **REPRESENTATION_DEFAULTS,
    },
    "balanced": {
        "evidence_max_views": 12,
        "evidence_view_stride": 4,
        "delta_top_k": 4096,
        "delta_steps": 800,
        "delta_max_total_samples": 320000,
        "delta_strength": 0.18,
        "delta_max_abs_rgb": 0.014,
        "delta_max_faces_to_apply": 128,
        "delta_patch_cert_max_faces_per_seed": 6,
        "trial_specs": "georisk2x0.75,patchrisk2x0.75,patchrisk4x0.5,risk4x0.5",
        "selector_alpha_max": 1.0,
        "selector_alpha_steps": 450,
        "selector_alpha_max_total_samples": 240000,
        "selector_min_trainval_psnr_gain": 0.0,
        "selector_min_trainval_balanced_delta": 0.0,
        "selector_tail_min_trainval_balanced_delta": 1.8e-5,
        **REPRESENTATION_DEFAULTS,
    },
    "visual_medium": {
        "evidence_max_views": 12,
        "evidence_view_stride": 4,
        "delta_top_k": 4096,
        "delta_steps": 800,
        "delta_max_total_samples": 320000,
        "delta_strength": 0.30,
        "delta_max_abs_rgb": 0.028,
        "delta_max_faces_to_apply": 128,
        "delta_patch_cert_max_faces_per_seed": 6,
        "trial_specs": "georisk2x0.75,patchrisk2x0.75,patchrisk4x0.5,risk4x0.5",
        "selector_alpha_max": 1.0,
        "selector_alpha_steps": 450,
        "selector_alpha_max_total_samples": 240000,
        "selector_min_trainval_psnr_gain": 2.0e-5,
        "selector_min_trainval_balanced_delta": 5.0e-5,
        "selector_tail_min_trainval_balanced_delta": 5.0e-5,
        **REPRESENTATION_DEFAULTS,
    },
    "field_smoke": {
        "evidence_max_views": 2,
        "evidence_view_stride": 8,
        "delta_top_k": 256,
        "delta_steps": 60,
        "delta_max_total_samples": 30000,
        "delta_strength": 0.30,
        "delta_max_abs_rgb": 0.030,
        "delta_max_faces_to_apply": 24,
        "delta_patch_cert_max_faces_per_seed": 6,
        "trial_specs": "top1x0.5,score1x0.5",
        "selector_alpha_max": 1.0,
        "selector_alpha_steps": 12,
        "selector_alpha_max_total_samples": 30000,
        "selector_min_trainval_psnr_gain": 0.0,
        "selector_min_trainval_balanced_delta": 0.0,
        "selector_tail_min_trainval_balanced_delta": 0.0,
        **REPRESENTATION_DEFAULTS,
        "delta_shared_residual_field": True,
        "delta_shared_residual_field_anchors": 8,
        "delta_shared_residual_field_lr": 0.018,
        "delta_shared_residual_field_weight_l2": 5.0e-5,
        "delta_shared_residual_field_view_hinge_weight": 0.04,
        "delta_shared_residual_field_view_hinge_min_samples": 8,
        "delta_shared_residual_field_duplicate_smooth_weight": 0.015,
        "delta_min_policy_val_samples": 12,
        "delta_min_policy_val_unique_faces": 1,
        "delta_crossfold_folds": 2,
        "delta_crossfold_min_passing_folds": 1,
        "delta_patch_cert_cluster_basis_mode": "field_linear",
        "delta_patch_cert_cluster_basis_steps": 80,
        "delta_patch_cert_cluster_basis_min_samples": 16,
        "delta_patch_cert_cluster_basis_max_fit_mse_regression": 0.04,
        "delta_patch_cert_cluster_basis_view_hinge_weight": 0.04,
        "delta_patch_cert_cluster_basis_view_hinge_min_samples": 8,
        "delta_patch_cert_cluster_basis_geometry_smooth_weight": 0.015,
        "delta_patch_cert_min_neighbor_policy_val_samples": 2,
        "delta_patch_cert_min_policy_val_samples": 8,
        "delta_patch_cert_carrier_holdout_groups": 2,
        "delta_patch_cert_carrier_holdout_min_passing_groups": 1,
    },
    "field_medium": {
        "evidence_max_views": 12,
        "evidence_view_stride": 4,
        "delta_top_k": 8192,
        "delta_steps": 1200,
        "delta_max_total_samples": 480000,
        "delta_strength": 0.35,
        "delta_max_abs_rgb": 0.035,
        "delta_max_faces_to_apply": 192,
        "delta_patch_cert_max_faces_per_seed": 10,
        "trial_specs": "georisk2x0.75,patchrisk2x0.75,patchrisk4x0.5,patchrisk6x0.35",
        "selector_alpha_max": 1.0,
        "selector_alpha_steps": 600,
        "selector_alpha_max_total_samples": 300000,
        "selector_min_trainval_psnr_gain": 2.0e-5,
        "selector_min_trainval_balanced_delta": 5.0e-5,
        "selector_tail_min_trainval_balanced_delta": 5.0e-5,
        **REPRESENTATION_DEFAULTS,
        "delta_shared_residual_field": True,
        "delta_shared_residual_field_anchors": 32,
        "delta_shared_residual_field_lr": 0.018,
        "delta_shared_residual_field_weight_l2": 5.0e-5,
        "delta_shared_residual_field_view_hinge_weight": 0.05,
        "delta_shared_residual_field_view_hinge_min_samples": 12,
        "delta_shared_residual_field_duplicate_smooth_weight": 0.02,
        "delta_patch_cert_cluster_basis_mode": "field_quad",
        "delta_patch_cert_cluster_basis_steps": 360,
        "delta_patch_cert_cluster_basis_min_samples": 24,
        "delta_patch_cert_cluster_basis_max_fit_mse_regression": 0.04,
        "delta_patch_cert_cluster_basis_view_hinge_weight": 0.05,
        "delta_patch_cert_cluster_basis_view_hinge_min_samples": 12,
        "delta_patch_cert_cluster_basis_geometry_smooth_weight": 0.02,
    },
    "field_region_medium": {
        "evidence_max_views": 12,
        "evidence_view_stride": 4,
        "delta_top_k": 8192,
        "delta_steps": 1200,
        "delta_max_total_samples": 480000,
        "delta_strength": 0.35,
        "delta_max_abs_rgb": 0.035,
        "delta_max_faces_to_apply": 192,
        "delta_patch_cert_max_faces_per_seed": 10,
        "trial_specs": "georisk2x0.75,patchrisk2x0.75,patchrisk4x0.5,patchrisk6x0.35",
        "selector_alpha_max": 1.0,
        "selector_alpha_steps": 600,
        "selector_alpha_max_total_samples": 300000,
        "selector_min_trainval_psnr_gain": 2.0e-5,
        "selector_min_trainval_balanced_delta": 5.0e-5,
        "selector_tail_min_trainval_balanced_delta": 5.0e-5,
        **REPRESENTATION_DEFAULTS,
        "delta_shared_residual_field": True,
        "delta_shared_residual_field_anchors": 32,
        "delta_shared_residual_field_lr": 0.018,
        "delta_shared_residual_field_weight_l2": 5.0e-5,
        "delta_shared_residual_field_view_hinge_weight": 0.05,
        "delta_shared_residual_field_view_hinge_min_samples": 12,
        "delta_shared_residual_field_duplicate_smooth_weight": 0.02,
        "delta_patch_cert_cluster_basis_mode": "field_quad",
        "delta_patch_cert_cluster_basis_steps": 360,
        "delta_patch_cert_cluster_basis_min_samples": 24,
        "delta_patch_cert_cluster_basis_max_fit_mse_regression": 0.04,
        "delta_patch_cert_cluster_basis_view_hinge_weight": 0.05,
        "delta_patch_cert_cluster_basis_view_hinge_min_samples": 12,
        "delta_patch_cert_cluster_basis_geometry_smooth_weight": 0.02,
        "delta_region_core_weight": 1.20,
        "delta_region_context_weight": 0.45,
        "delta_region_outside_weight": 0.20,
        "delta_region_boundary_px": 2,
        "delta_render_region_objective": True,
        "delta_render_region_core_weight": 1.0,
        "delta_render_region_context_weight": 0.35,
        "delta_render_region_outside_penalty": 0.02,
        "delta_render_region_tail_cvar_weight": 0.10,
        "delta_render_region_tail_fraction": 0.25,
        "delta_render_region_min_view_samples": 12,
        "filter_drop_unmapped": True,
    },
    "field_region_owned_medium": {
        "evidence_max_views": 12,
        "evidence_view_stride": 4,
        "delta_top_k": 8192,
        "delta_steps": 1200,
        "delta_max_total_samples": 480000,
        "delta_strength": 0.35,
        "delta_max_abs_rgb": 0.035,
        "delta_max_faces_to_apply": 192,
        "delta_patch_cert_max_faces_per_seed": 10,
        "trial_specs": "georisk2x0.75,patchrisk2x0.75,patchrisk4x0.5,patchrisk6x0.35",
        "selector_alpha_max": 1.0,
        "selector_alpha_steps": 600,
        "selector_alpha_max_total_samples": 300000,
        "selector_min_trainval_psnr_gain": 2.0e-5,
        "selector_min_trainval_balanced_delta": 5.0e-5,
        "selector_tail_min_trainval_balanced_delta": 5.0e-5,
        **REPRESENTATION_DEFAULTS,
        "delta_shared_residual_field": True,
        "delta_shared_residual_field_anchors": 32,
        "delta_shared_residual_field_lr": 0.018,
        "delta_shared_residual_field_weight_l2": 5.0e-5,
        "delta_shared_residual_field_view_hinge_weight": 0.05,
        "delta_shared_residual_field_view_hinge_min_samples": 12,
        "delta_shared_residual_field_duplicate_smooth_weight": 0.02,
        "delta_patch_cert_cluster_basis_mode": "field_quad",
        "delta_patch_cert_cluster_basis_steps": 360,
        "delta_patch_cert_cluster_basis_min_samples": 24,
        "delta_patch_cert_cluster_basis_max_fit_mse_regression": 0.04,
        "delta_patch_cert_cluster_basis_view_hinge_weight": 0.05,
        "delta_patch_cert_cluster_basis_view_hinge_min_samples": 12,
        "delta_patch_cert_cluster_basis_geometry_smooth_weight": 0.02,
        "delta_region_core_weight": 1.20,
        "delta_region_context_weight": 0.45,
        "delta_region_outside_weight": 0.20,
        "delta_region_boundary_px": 2,
        "delta_render_region_objective": True,
        "delta_render_region_core_weight": 1.0,
        "delta_render_region_context_weight": 0.35,
        "delta_render_region_outside_penalty": 0.02,
        "delta_render_region_tail_cvar_weight": 0.10,
        "delta_render_region_tail_fraction": 0.25,
        "delta_render_region_min_view_samples": 12,
        "filter_region_source": "candidate_owned",
        "filter_drop_unmapped": True,
        "filter_min_mean_core_balanced_delta": 0.002,
        "filter_min_mean_delta_psnr": 0.001,
        "filter_min_tail_core_balanced_delta": -0.001,
        "filter_max_negative_core_balanced_fraction": 0.25,
        "filter_train_render_region_min_pixels": 16,
        "filter_train_render_region_min_crop_size": 16,
        "selector_strict_replay_scales": "1.0,0.75,0.5,0.35,0.2",
    },
    "strict": {
        "evidence_max_views": 16,
        "evidence_view_stride": 3,
        "delta_top_k": 8192,
        "delta_steps": 1000,
        "delta_max_total_samples": 420000,
        "delta_strength": 0.18,
        "delta_max_abs_rgb": 0.014,
        "delta_max_faces_to_apply": 192,
        "delta_patch_cert_max_faces_per_seed": 8,
        "trial_specs": "georisk2x0.75,patchrisk2x0.75,patchrisk4x0.5,patchrisk6x0.35",
        "selector_alpha_max": 1.0,
        "selector_alpha_steps": 600,
        "selector_alpha_max_total_samples": 300000,
        "selector_min_trainval_psnr_gain": 2.0e-5,
        "selector_min_trainval_balanced_delta": 5.0e-5,
        "selector_tail_min_trainval_balanced_delta": 5.0e-5,
        **REPRESENTATION_DEFAULTS,
    },
}

PROFILE_DEFAULTS["field_region_owned_refit_medium"] = {
    **PROFILE_DEFAULTS["field_region_owned_medium"],
    "candidate_owned_refit": True,
    "delta_render_region_outside_penalty": 0.08,
    "delta_render_region_tail_cvar_weight": 0.25,
    "delta_render_region_context_weight": 0.20,
    "filter_min_mean_core_balanced_delta": 0.004,
    "filter_min_tail_core_balanced_delta": -0.0005,
    "filter_max_negative_core_balanced_fraction": 0.20,
    "selector_strict_adaptive_scale_policy": True,
    "selector_strict_adaptive_scale_min": 0.10,
    "selector_strict_adaptive_scale_max_extra": 5,
    "selector_strict_adaptive_scale_tail_fraction": 0.30,
}

PROFILE_DEFAULTS["field_region_owned_refit_adaptive_medium"] = {
    **PROFILE_DEFAULTS["field_region_owned_refit_medium"],
    "use_filtered_plan_for_selector": False,
}

PROFILE_DEFAULTS["field_region_owned_refit_alpha_medium"] = {
    **PROFILE_DEFAULTS["field_region_owned_refit_adaptive_medium"],
    "selector_strict_fit_plan_alphas": True,
}

PROFILE_DEFAULTS["field_region_owned_refit_risk_medium"] = {
    **PROFILE_DEFAULTS["field_region_owned_refit_medium"],
    "use_filtered_plan_for_selector": True,
    "filter_min_changed_fraction": 0.08,
    "filter_min_mean_core_balanced_delta": 0.0,
    "filter_min_mean_delta_psnr": 0.0,
    "filter_min_tail_core_balanced_delta": -2.0e-5,
    "filter_max_negative_core_balanced_fraction": 0.35,
    "filter_min_mean_crop_abs_diff": 5.0e-6,
    "filter_min_max_crop_abs_diff": 0.0039215686,
    "plan_region_gate_min_regions": 1,
    "plan_region_gate_min_changed_regions": 1,
    "plan_region_gate_min_changed_fraction": 0.05,
    "plan_region_gate_min_core_balanced_delta": 0.0,
    "plan_region_gate_min_core_psnr_delta": 0.0,
    "plan_region_gate_min_tail_cvar_delta": -2.0e-5,
    "plan_region_gate_max_context_mse_regression": 1.0e-6,
    "plan_region_gate_max_negative_fraction": 0.40,
    "selector_strict_adaptive_scale_policy": True,
    "selector_strict_adaptive_scale_min": 0.10,
    "selector_strict_adaptive_scale_max_extra": 5,
    "selector_strict_adaptive_scale_tail_fraction": 0.30,
    "selector_strict_fit_plan_alphas": False,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v1"] = {
    **PROFILE_DEFAULTS["field_region_owned_refit_medium"],
    "filter_region_source": "candidate_owned",
    "candidate_owned_refit": True,
    "use_filtered_plan_for_selector": True,
    "selector_strict_replay_scales": "1.0",
    "selector_fit_plan_alphas": False,
    "selector_strict_adaptive_scale_policy": False,
    "selector_strict_fit_plan_alphas": False,
    "filter_drop_unmapped": True,
    "filter_require_positive_plan_proxy": True,
    "filter_min_regions": 1,
    "filter_min_changed_regions": 1,
    "filter_min_changed_fraction": 0.08,
    "filter_min_mean_core_balanced_delta": 0.0,
    "filter_min_mean_delta_psnr": 0.0,
    "filter_min_tail_core_balanced_delta": -2.0e-5,
    "filter_max_negative_core_balanced_fraction": 0.35,
    "filter_max_context_mse_regression": 1.0e-6,
    "filter_min_mean_crop_abs_diff": 5.0e-6,
    "filter_min_max_crop_abs_diff": 0.0039215686,
    "plan_region_gate_min_regions": 1,
    "plan_region_gate_min_changed_regions": 1,
    "plan_region_gate_min_changed_fraction": 0.05,
    "plan_region_gate_min_core_balanced_delta": 0.0,
    "plan_region_gate_min_core_psnr_delta": 0.0,
    "plan_region_gate_min_tail_cvar_delta": -2.0e-5,
    "plan_region_gate_max_context_mse_regression": 1.0e-6,
    "plan_region_gate_max_negative_fraction": 0.40,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v2"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v1"],
    "delta_patch_cert_carrier_holdout_auto_prefix_min_face_fraction": 0.70,
    "delta_patch_cert_carrier_holdout_auto_prefix_face_bonus": 0.02,
    "filter_candidate_region_expand_faces": True,
    "filter_candidate_region_expand_min_face_pixels": 12,
    "filter_candidate_region_expand_min_face_views": 1,
    "filter_candidate_region_expand_max_faces_per_carrier": 32,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v3"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v2"],
    "delta_min_policy_val_adaptive_sample_fraction": 0.30,
    "delta_min_policy_val_adaptive_min_samples": 128,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v4"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v3"],
    "filter_tail_safe_shrink_on_tail_fail": True,
    "filter_tail_safe_shrink_min_scale": 0.50,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v5"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v4"],
    "filter_risk_safe_shrink_on_train_risk_fail": True,
    "filter_risk_safe_shrink_min_scale": 0.25,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v6"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v5"],
    "selector_strict_replay_scales": "1.0,0.85,0.75,0.6,0.5,0.35,0.2",
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v7"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v6"],
    "selector_enable_region_stable_promotion": True,
    "selector_region_min_trainval_balanced_delta": 0.0,
    "selector_region_min_mean_core_balanced_delta": 0.01,
    "selector_region_min_mean_delta_psnr": 0.001,
    "selector_region_min_changed_fraction": 0.50,
    "selector_region_max_negative_core_balanced_fraction": 0.35,
    "selector_region_max_context_mse_regression": 1.0e-6,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v8"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v7"],
    "filter_tail_safe_shrink_min_raw_scale": 0.60,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v9"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v8"],
    "candidate_region_expansion_closure": True,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v10"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v9"],
    "candidate_region_expansion_core_priority": True,
    "candidate_region_expansion_core_min_samples": 8,
    "candidate_region_expansion_core_min_fraction": 0.5,
    "candidate_region_expansion_witness_rescue": True,
    "candidate_region_expansion_max_witnesses_per_carrier": 1,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v11"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v10"],
    "filter_rollback_severe_tail_fail": True,
    "filter_rollback_tail_min_cvar_loss": 0.005,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v12"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v11"],
    "filter_aggregate_subset": True,
    "filter_aggregate_subset_min_selected_carriers": 1,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v13"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v12"],
    "filter_aggregate_subset_expected_view_count": 64,
    "filter_aggregate_subset_min_unique_views": 4,
    "filter_aggregate_subset_min_changed_unique_views": 4,
    "filter_aggregate_subset_min_changed_pixel_fraction": 0.05,
    "filter_aggregate_subset_min_area_weighted_core_balanced_delta": 0.0,
    "filter_aggregate_subset_min_dilution_adjusted_core_balanced_delta": 1.0e-5,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v14"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v13"],
    "filter_aggregate_subset_tail_safe_shrink_carriers": True,
    "filter_aggregate_subset_tail_safe_shrink_min_scale": 0.02,
    "selector_strict_replay_scales": "1.0",
    "selector_strict_adaptive_scale_policy": False,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v15"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v14"],
    "filter_aggregate_subset_min_full_frame_changed_pixel_fraction": 2.0e-5,
    "filter_aggregate_subset_min_full_frame_visibility_adjusted_delta": 5.0e-7,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v16"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v15"],
    "filter_candidate_region_frame_aware_ranking": True,
    "filter_candidate_region_min_frame_support_fraction": 1.0e-5,
    "filter_candidate_region_min_residual_mass_fraction": 1.0e-7,
    "filter_candidate_region_max_carriers": 8,
    "filter_candidate_region_max_regions_per_carrier": 16,
    "filter_candidate_region_expand_max_faces_per_carrier": 64,
    "filter_aggregate_subset_prefer_full_frame_visibility": True,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v17"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v16"],
    "ela_alpha_holdout_safe_zero": True,
    "ela_alpha_risk_tail_fraction": 0.20,
    "ela_alpha_max_negative_gain_fraction": 0.45,
    "ela_alpha_min_tail_gain": 0.0,
    "ela_alpha_region_risk_enable": True,
    "ela_alpha_region_risk_min_tail_gain": 0.0,
    "ela_alpha_region_risk_max_negative_fraction": 0.45,
    "ela_alpha_region_risk_min_regions": 1,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v18"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v17"],
    "ela_alpha_region_risk_objective_bad_only": True,
    "ela_alpha_region_risk_objective_max_balanced_delta": 0.0,
    "ela_alpha_region_risk_objective_max_delta_ssim": 0.0,
    "ela_alpha_region_risk_objective_min_delta_lpips": 0.0,
    "ela_alpha_region_risk_max_negative_fraction": 0.25,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v19"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v18"],
    "candidate_region_pre_refit_risk_prune": True,
    "candidate_region_pre_refit_risk_min_changed_rows": 1,
    "candidate_region_pre_refit_risk_min_bad_rows": 1,
    "candidate_region_pre_refit_risk_max_bad_fraction": 0.5,
    "candidate_region_pre_refit_risk_balanced_margin": 1.0e-3,
    "candidate_region_pre_refit_risk_use_aux_metric_pair": True,
    "candidate_region_pre_refit_risk_ssim_margin": 1.0e-3,
    "candidate_region_pre_refit_risk_lpips_margin": 1.0e-3,
    "candidate_region_pre_refit_risk_max_removed_face_fraction": 0.5,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v20"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v19"],
    "candidate_region_pre_refit_risk_prune": False,
    "candidate_region_pre_refit_risk_shrink": True,
    "candidate_region_pre_refit_risk_shrink_min_scale": 0.25,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v21"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v20"],
    "candidate_region_pre_refit_risk_shrink_min_scale": 0.10,
    "candidate_region_pre_refit_risk_shrink_severity_aware": True,
    "candidate_region_pre_refit_risk_shrink_severity_select_min": 0.5,
    "candidate_region_pre_refit_risk_shrink_severity_balanced_span": 0.05,
    "candidate_region_pre_refit_risk_shrink_tail_fraction": 0.25,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v22"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v21"],
    "candidate_region_pre_refit_risk_local_suppression": True,
    "candidate_region_pre_refit_risk_local_suppression_scale": 0.02,
    "candidate_region_pre_refit_risk_local_suppression_min_bad_balanced": 0.02,
    "candidate_region_pre_refit_risk_local_suppression_positive_margin": 0.02,
    "candidate_region_pre_refit_risk_local_suppression_min_face_pixels": 12,
    "candidate_region_pre_refit_risk_local_suppression_max_faces_per_bad_row": 16,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v23"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v22"],
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v24"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v23"],
    "delta_render_region_outside_penalty": 0.0,
    "delta_bystander_zero_delta_weight": 0.20,
    "delta_bystander_zero_delta_include_context": True,
    "delta_bystander_zero_delta_min_samples": 64,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v25"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v24"],
    "delta_witness_constraint_weight": 0.25,
    "delta_witness_constraint_tail_fraction": 0.25,
    "delta_witness_constraint_min_samples": 16,
    "delta_witness_constraint_margin": 0.0,
    "delta_witness_constraint_include_full_view": True,
    "delta_witness_constraint_include_region_view": True,
    "delta_witness_constraint_include_bystander_view": True,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v26"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v25"],
    "ela_local_trust_gate": True,
    "ela_local_trust_min_supports": 2,
    "ela_local_trust_max_residual_std": 0.035,
    "ela_local_trust_min_agreement": 0.45,
    "ela_local_trust_agreement_scale": 0.04,
    "ela_local_trust_confidence_quantile": 0.25,
    "ela_local_trust_min_confidence": 1.0e-4,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v27"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v26"],
    "ela_local_trust_mode": "soft",
    "ela_local_trust_min_weight": 0.02,
    "ela_local_trust_min_supports": 2,
    "ela_local_trust_max_residual_std": 0.06,
    "ela_local_trust_min_agreement": 0.20,
    "ela_local_trust_confidence_quantile": 0.10,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v28"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v27"],
    "ela_alpha_view_tail_scale_grid": "1.0,0.75,0.5,0.25,0.0",
    "ela_alpha_view_tail_cvar_fraction": 0.25,
    "ela_alpha_view_tail_min_gain": 0.0,
    "ela_alpha_view_tail_max_negative_fraction": 0.50,
}

PROFILE_DEFAULTS["field_region_render_risk_strict_v29"] = {
    **PROFILE_DEFAULTS["field_region_render_risk_strict_v28"],
    "ela_alpha_view_tail_objective": "balanced",
    "ela_alpha_view_tail_ssim_weight": 20.0,
    "ela_alpha_view_tail_lpips_weight": 20.0,
    "ela_alpha_view_tail_compute_lpips": True,
    "ela_alpha_view_tail_metric_max_side": 512,
}

PROFILE_FIELD_NAMES = tuple(next(iter(PROFILE_DEFAULTS.values())).keys())
FIXED_PROFILE_NAMES = {
    "field_region_render_risk_strict_v1",
    "field_region_render_risk_strict_v2",
    "field_region_render_risk_strict_v3",
    "field_region_render_risk_strict_v4",
    "field_region_render_risk_strict_v5",
    "field_region_render_risk_strict_v6",
    "field_region_render_risk_strict_v7",
    "field_region_render_risk_strict_v8",
    "field_region_render_risk_strict_v9",
    "field_region_render_risk_strict_v10",
    "field_region_render_risk_strict_v11",
    "field_region_render_risk_strict_v12",
    "field_region_render_risk_strict_v13",
    "field_region_render_risk_strict_v14",
    "field_region_render_risk_strict_v15",
    "field_region_render_risk_strict_v16",
    "field_region_render_risk_strict_v17",
    "field_region_render_risk_strict_v18",
    "field_region_render_risk_strict_v19",
    "field_region_render_risk_strict_v20",
    "field_region_render_risk_strict_v21",
    "field_region_render_risk_strict_v22",
    "field_region_render_risk_strict_v23",
    "field_region_render_risk_strict_v24",
    "field_region_render_risk_strict_v25",
    "field_region_render_risk_strict_v26",
    "field_region_render_risk_strict_v27",
    "field_region_render_risk_strict_v28",
    "field_region_render_risk_strict_v29",
}
PROFILE_CONTRACT_IDS = {
    "field_region_render_risk_strict_v1": "field_region_render_risk_strict_v1_fixed_train_only_no_scale_search",
    "field_region_render_risk_strict_v2": "field_region_render_risk_strict_v2_coverage_prefix_fixed_train_only_no_scale_search",
    "field_region_render_risk_strict_v3": "field_region_render_risk_strict_v3_support_aware_policy_floor_fixed_train_only_no_scale_search",
    "field_region_render_risk_strict_v4": "field_region_render_risk_strict_v4_tail_safe_shrink_fixed_train_only_no_scale_search",
    "field_region_render_risk_strict_v5": "field_region_render_risk_strict_v5_context_tail_risk_shrink_fixed_train_only_no_scale_search",
    "field_region_render_risk_strict_v6": "field_region_render_risk_strict_v6_pre_registered_trainval_shrink_ladder",
    "field_region_render_risk_strict_v7": "field_region_render_risk_strict_v7_roi_stable_trainval_promotion",
    "field_region_render_risk_strict_v8": "field_region_render_risk_strict_v8_tail_severity_gated_roi_stable_promotion",
    "field_region_render_risk_strict_v9": "field_region_render_risk_strict_v9_candidate_region_expansion_closure",
    "field_region_render_risk_strict_v10": "field_region_render_risk_strict_v10_region_core_expansion_witness",
    "field_region_render_risk_strict_v11": "field_region_render_risk_strict_v11_render_cvar_severe_tail_carrier_rollback",
    "field_region_render_risk_strict_v12": "field_region_render_risk_strict_v12_monotonic_aggregate_render_cvar_subset",
    "field_region_render_risk_strict_v13": "field_region_render_risk_strict_v13_coverage_dilution_guarded_aggregate_subset",
    "field_region_render_risk_strict_v14": "field_region_render_risk_strict_v14_coverage_dilution_guarded_per_carrier_alpha_shrink",
    "field_region_render_risk_strict_v15": "field_region_render_risk_strict_v15_full_frame_visibility_guarded_alpha_shrink",
    "field_region_render_risk_strict_v16": "field_region_render_risk_strict_v16_full_frame_visible_residual_carrier_preselection",
    "field_region_render_risk_strict_v17": "field_region_render_risk_strict_v17_region_risk_adaptive_alpha_materialization",
    "field_region_render_risk_strict_v18": "field_region_render_risk_strict_v18_objective_aware_bad_region_alpha_risk",
    "field_region_render_risk_strict_v19": "field_region_render_risk_strict_v19_objective_aware_pre_refit_carrier_risk_pruning",
    "field_region_render_risk_strict_v20": "field_region_render_risk_strict_v20_objective_aware_pre_refit_carrier_risk_shrink",
    "field_region_render_risk_strict_v21": "field_region_render_risk_strict_v21_severity_aware_pre_refit_carrier_risk_shrink",
    "field_region_render_risk_strict_v22": "field_region_render_risk_strict_v22_region_local_bad_row_suppression",
    "field_region_render_risk_strict_v23": "field_region_render_risk_strict_v23_audited_metadata_local_suppression",
    "field_region_render_risk_strict_v24": "field_region_render_risk_strict_v24_train_objective_bystander_zero_delta",
    "field_region_render_risk_strict_v25": "field_region_render_risk_strict_v25_train_objective_witness_group_cvar",
    "field_region_render_risk_strict_v26": "field_region_render_risk_strict_v26_render_layer_local_trust_reversible_residual",
    "field_region_render_risk_strict_v27": "field_region_render_risk_strict_v27_soft_local_trust_weighted_residual",
    "field_region_render_risk_strict_v28": "field_region_render_risk_strict_v28_view_tail_safe_alpha_shrink",
    "field_region_render_risk_strict_v29": "field_region_render_risk_strict_v29_balanced_lpips_view_tail_alpha_shrink",
}


@dataclass
class CommandRecord:
    stage: str
    scene: str
    command: list[str]
    log_path: str
    output_paths: dict[str, str] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""
    exit_code: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default="bicycle")
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="visual_medium")
    parser.add_argument("--policy_root", default=DEFAULT_POLICY_ROOT)
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan_template", default="")
    parser.add_argument("--filtered_plan_template", default="")
    parser.add_argument("--selector_plan_template", default="")
    parser.add_argument("--evidence_root", default="")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--outdoor_images", default="images_4")
    parser.add_argument("--indoor_images", default="images_2")
    parser.add_argument("--pipeline_label", default="autovisual_facelocal_v1")
    parser.add_argument(
        "--stages",
        default="plan,filter,selector",
        help="Comma/space separated stages from: plan, filter, selector. Summary is always written.",
    )
    parser.add_argument("--use_filtered_plan_for_selector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--wandb_project", default="mesh-splatting-ecsr")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))

    parser.add_argument("--evidence_max_views", type=int, default=None)
    parser.add_argument("--evidence_view_stride", type=int, default=None)
    parser.add_argument("--evidence_view_offset", type=int, default=0)
    parser.add_argument("--evidence_high_error_quantile", type=float, default=0.65)
    parser.add_argument("--delta_top_k", type=int, default=None)
    parser.add_argument("--delta_min_view_hits", type=int, default=2)
    parser.add_argument("--delta_min_consistency", type=float, default=0.80)
    parser.add_argument("--delta_min_pixel_count", type=float, default=6.0)
    parser.add_argument("--delta_max_samples_per_face_view", type=int, default=64)
    parser.add_argument("--delta_max_total_samples", type=int, default=None)
    parser.add_argument("--delta_high_error_quantile", type=float, default=0.65)
    parser.add_argument("--delta_strength", type=float, default=None)
    parser.add_argument("--delta_max_abs_rgb", type=float, default=None)
    parser.add_argument("--delta_sh_degree", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--delta_lambda_mag", type=float, default=0.02)
    parser.add_argument("--delta_lambda_sh1_mag", type=float, default=0.05)
    parser.add_argument("--delta_lambda_smooth", type=float, default=0.08)
    parser.add_argument("--delta_steps", type=int, default=None)
    parser.add_argument("--delta_shared_residual_field", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--delta_shared_residual_field_anchors", type=int, default=None)
    parser.add_argument("--delta_shared_residual_field_sigma", type=float, default=None)
    parser.add_argument("--delta_shared_residual_field_lr", type=float, default=None)
    parser.add_argument("--delta_shared_residual_field_weight_l2", type=float, default=None)
    parser.add_argument("--delta_shared_residual_field_view_hinge_weight", type=float, default=None)
    parser.add_argument("--delta_shared_residual_field_view_hinge_min_samples", type=int, default=None)
    parser.add_argument("--delta_shared_residual_field_duplicate_smooth_weight", type=float, default=None)
    parser.add_argument("--delta_region_core_weight", type=float, default=None)
    parser.add_argument("--delta_region_context_weight", type=float, default=None)
    parser.add_argument("--delta_region_outside_weight", type=float, default=None)
    parser.add_argument("--delta_region_boundary_px", type=int, default=None)
    parser.add_argument("--delta_render_region_objective", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--delta_render_region_core_weight", type=float, default=None)
    parser.add_argument("--delta_render_region_context_weight", type=float, default=None)
    parser.add_argument("--delta_render_region_outside_penalty", type=float, default=None)
    parser.add_argument("--delta_render_region_tail_cvar_weight", type=float, default=None)
    parser.add_argument("--delta_render_region_tail_fraction", type=float, default=None)
    parser.add_argument("--delta_render_region_min_view_samples", type=int, default=None)
    parser.add_argument("--delta_bystander_zero_delta_weight", type=float, default=None)
    parser.add_argument("--delta_bystander_zero_delta_include_context", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--delta_bystander_zero_delta_min_samples", type=int, default=None)
    parser.add_argument("--delta_witness_constraint_weight", type=float, default=None)
    parser.add_argument("--delta_witness_constraint_tail_fraction", type=float, default=None)
    parser.add_argument("--delta_witness_constraint_min_samples", type=int, default=None)
    parser.add_argument("--delta_witness_constraint_margin", type=float, default=None)
    parser.add_argument("--delta_witness_constraint_include_full_view", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--delta_witness_constraint_include_region_view", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--delta_witness_constraint_include_bystander_view", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--delta_max_faces_to_apply", type=int, default=None)
    parser.add_argument("--delta_min_policy_val_relative_gain", type=float, default=0.02)
    parser.add_argument("--delta_min_policy_val_samples", type=int, default=None)
    parser.add_argument("--delta_min_policy_val_adaptive_sample_fraction", type=float, default=None)
    parser.add_argument("--delta_min_policy_val_adaptive_min_samples", type=int, default=None)
    parser.add_argument("--delta_min_policy_val_unique_faces", type=int, default=None)
    parser.add_argument(
        "--delta_validation_shrink_mode",
        choices=("none", "global", "face", "global_gain", "face_gain"),
        default="face",
    )
    parser.add_argument("--delta_validation_gain_max_scale", type=float, default=1.0)
    parser.add_argument("--delta_min_face_policy_val_relative_gain", type=float, default=0.0)
    parser.add_argument("--delta_min_face_policy_val_samples", type=int, default=8)
    parser.add_argument("--delta_min_face_view_consensus", type=float, default=0.50)
    parser.add_argument("--delta_min_face_consensus_views", type=int, default=2)
    parser.add_argument("--delta_min_face_gain_certificate_views", type=int, default=2)
    parser.add_argument("--delta_min_face_gain_certificate_fraction", type=float, default=0.50)
    parser.add_argument("--delta_crossfold_folds", type=int, default=None)
    parser.add_argument("--delta_crossfold_min_passing_folds", type=int, default=None)
    parser.add_argument("--delta_patch_cert_rings", type=int, default=1)
    parser.add_argument("--delta_patch_cert_max_faces_per_seed", type=int, default=None)
    parser.add_argument("--delta_patch_cert_min_neighbor_policy_val_samples", type=int, default=None)
    parser.add_argument("--delta_patch_cert_min_policy_val_samples", type=int, default=None)
    parser.add_argument("--delta_patch_cert_min_relative_gain", type=float, default=None)
    parser.add_argument("--delta_patch_cert_neighbor_mode", choices=("topology", "centroid", "both"), default="both")
    parser.add_argument(
        "--delta_patch_cert_cluster_basis_mode",
        choices=("shared", "scaled", "rank2", "chart_linear", "chart_quad", "field_linear", "field_quad"),
        default=None,
    )
    parser.add_argument("--delta_patch_cert_cluster_basis_steps", type=int, default=None)
    parser.add_argument("--delta_patch_cert_cluster_basis_min_samples", type=int, default=None)
    parser.add_argument("--delta_patch_cert_cluster_basis_max_fit_mse_regression", type=float, default=None)
    parser.add_argument("--delta_patch_cert_cluster_basis_view_hinge_weight", type=float, default=None)
    parser.add_argument("--delta_patch_cert_cluster_basis_view_hinge_min_samples", type=int, default=None)
    parser.add_argument("--delta_patch_cert_cluster_basis_geometry_smooth_weight", type=float, default=None)
    parser.add_argument("--delta_patch_cert_carrier_holdout_groups", type=int, default=None)
    parser.add_argument("--delta_patch_cert_carrier_holdout_min_passing_groups", type=int, default=None)
    parser.add_argument("--delta_patch_cert_carrier_holdout_auto_prefix_min_faces", type=int, default=0)
    parser.add_argument("--delta_patch_cert_carrier_holdout_auto_prefix_min_face_fraction", type=float, default=None)
    parser.add_argument("--delta_patch_cert_carrier_holdout_auto_prefix_face_bonus", type=float, default=None)
    parser.add_argument(
        "--delta_patch_cert_carrier_holdout_auto_prefix_positive_tail_safe",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--no_seed_rescue", action="store_true")

    parser.add_argument("--trial_specs", default="")
    parser.add_argument("--selector_fit_plan_alphas", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--selector_alpha_max", type=float, default=None)
    parser.add_argument("--selector_alpha_steps", type=int, default=None)
    parser.add_argument("--selector_alpha_lr", type=float, default=0.06)
    parser.add_argument("--selector_alpha_max_total_samples", type=int, default=None)
    parser.add_argument("--selector_alpha_device", default="cuda")
    parser.add_argument("--selector_min_trainval_psnr_gain", type=float, default=None)
    parser.add_argument("--selector_min_trainval_balanced_delta", type=float, default=None)
    parser.add_argument("--selector_balanced_ssim_weight", type=float, default=None)
    parser.add_argument("--selector_balanced_lpips_weight", type=float, default=None)
    parser.add_argument("--selector_tail_min_trainval_balanced_delta", type=float, default=None)
    parser.add_argument("--selector_tail_stable_promotion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selector_tail_max_balanced_cvar_loss", type=float, default=0.0012)
    parser.add_argument("--selector_tail_min_mean_to_cvar_ratio", type=float, default=0.05)
    parser.add_argument("--selector_tail_max_lpips_positive_fraction", type=float, default=0.70)
    parser.add_argument("--selector_enable_region_stable_promotion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--selector_region_min_trainval_balanced_delta", type=float, default=None)
    parser.add_argument("--selector_region_min_mean_core_balanced_delta", type=float, default=None)
    parser.add_argument("--selector_region_min_mean_delta_psnr", type=float, default=None)
    parser.add_argument("--selector_region_min_changed_fraction", type=float, default=None)
    parser.add_argument("--selector_region_max_negative_core_balanced_fraction", type=float, default=None)
    parser.add_argument("--selector_region_max_context_mse_regression", type=float, default=None)
    parser.add_argument("--selector_strict_replay_scales", default=None)
    parser.add_argument("--selector_strict_adaptive_scale_policy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--selector_strict_adaptive_scale_min", type=float, default=None)
    parser.add_argument("--selector_strict_adaptive_scale_max_extra", type=int, default=None)
    parser.add_argument("--selector_strict_adaptive_scale_tail_fraction", type=float, default=None)
    parser.add_argument("--ela_alpha_holdout_safe_zero", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ela_alpha_risk_tail_fraction", type=float, default=None)
    parser.add_argument("--ela_alpha_max_negative_gain_fraction", type=float, default=None)
    parser.add_argument("--ela_alpha_min_tail_gain", type=float, default=None)
    parser.add_argument("--ela_alpha_view_tail_scale_grid", default=None)
    parser.add_argument("--ela_alpha_view_tail_cvar_fraction", type=float, default=None)
    parser.add_argument("--ela_alpha_view_tail_min_gain", type=float, default=None)
    parser.add_argument("--ela_alpha_view_tail_max_negative_fraction", type=float, default=None)
    parser.add_argument("--ela_alpha_view_tail_objective", choices=("mse", "balanced"), default=None)
    parser.add_argument("--ela_alpha_view_tail_ssim_weight", type=float, default=None)
    parser.add_argument("--ela_alpha_view_tail_lpips_weight", type=float, default=None)
    parser.add_argument("--ela_alpha_view_tail_compute_lpips", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ela_alpha_view_tail_metric_max_side", type=int, default=None)
    parser.add_argument("--ela_alpha_region_risk_enable", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ela_alpha_region_risk_min_tail_gain", type=float, default=None)
    parser.add_argument("--ela_alpha_region_risk_max_negative_fraction", type=float, default=None)
    parser.add_argument("--ela_alpha_region_risk_min_regions", type=int, default=None)
    parser.add_argument("--ela_local_trust_gate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ela_local_trust_min_supports", type=int, default=None)
    parser.add_argument("--ela_local_trust_max_residual_std", type=float, default=None)
    parser.add_argument("--ela_local_trust_min_agreement", type=float, default=None)
    parser.add_argument("--ela_local_trust_agreement_scale", type=float, default=None)
    parser.add_argument("--ela_local_trust_confidence_quantile", type=float, default=None)
    parser.add_argument("--ela_local_trust_min_confidence", type=float, default=None)
    parser.add_argument("--ela_local_trust_mode", choices=("hard", "soft"), default=None)
    parser.add_argument("--ela_local_trust_min_weight", type=float, default=None)
    parser.add_argument("--selector_strict_fit_plan_alphas", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--phasej_test_method", default=DEFAULT_PHASEJ_TEST_METHOD)
    parser.add_argument("--phasej_trainval_method", default=DEFAULT_PHASEJ_TRAINVAL_METHOD)
    parser.add_argument(
        "--isolate_phasej_methods",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the pipeline label to generated PhaseJ ELA method names so parallel experiments do not share render directories.",
    )
    parser.add_argument("--gate_min_psnr_gain", type=float, default=0.0)
    parser.add_argument("--gate_max_ssim_regression", type=float, default=5e-5)
    parser.add_argument("--gate_max_lpips_regression", type=float, default=1.5e-4)
    parser.add_argument("--gate_min_balanced_delta", type=float, default=0.0)

    parser.add_argument("--render_region_carrier_template", default=DEFAULT_RENDER_REGION_CARRIER_TEMPLATE)
    parser.add_argument("--filter_max_region_matches_per_plan_carrier", type=int, default=None)
    parser.add_argument("--filter_min_regions", type=int, default=None)
    parser.add_argument("--filter_min_changed_regions", type=int, default=None)
    parser.add_argument("--filter_min_changed_fraction", type=float, default=None)
    parser.add_argument("--filter_min_mean_core_balanced_delta", type=float, default=None)
    parser.add_argument("--filter_min_mean_delta_psnr", type=float, default=None)
    parser.add_argument("--filter_min_tail_core_balanced_delta", type=float, default=None)
    parser.add_argument("--filter_max_negative_core_balanced_fraction", type=float, default=None)
    parser.add_argument("--filter_max_context_mse_regression", type=float, default=None)
    parser.add_argument("--filter_min_mean_crop_abs_diff", type=float, default=None)
    parser.add_argument("--filter_min_max_crop_abs_diff", type=float, default=None)
    parser.add_argument("--filter_tail_safe_shrink_on_tail_fail", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--filter_tail_safe_shrink_min_scale", type=float, default=None)
    parser.add_argument("--filter_tail_safe_shrink_min_raw_scale", type=float, default=None)
    parser.add_argument("--filter_rollback_severe_tail_fail", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--filter_rollback_tail_min_cvar_loss", type=float, default=None)
    parser.add_argument("--filter_aggregate_subset", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--filter_aggregate_subset_min_selected_carriers", type=int, default=None)
    parser.add_argument("--filter_aggregate_subset_expected_view_count", type=int, default=None)
    parser.add_argument("--filter_aggregate_subset_min_unique_views", type=int, default=None)
    parser.add_argument("--filter_aggregate_subset_min_changed_unique_views", type=int, default=None)
    parser.add_argument("--filter_aggregate_subset_min_view_coverage_fraction", type=float, default=None)
    parser.add_argument("--filter_aggregate_subset_min_changed_view_coverage_fraction", type=float, default=None)
    parser.add_argument("--filter_aggregate_subset_min_total_pixels", type=int, default=None)
    parser.add_argument("--filter_aggregate_subset_min_changed_pixels", type=int, default=None)
    parser.add_argument("--filter_aggregate_subset_min_changed_pixel_fraction", type=float, default=None)
    parser.add_argument("--filter_aggregate_subset_expected_frame_pixels", type=int, default=None)
    parser.add_argument("--filter_aggregate_subset_min_full_frame_changed_pixel_fraction", type=float, default=None)
    parser.add_argument("--filter_aggregate_subset_min_area_weighted_core_balanced_delta", type=float, default=None)
    parser.add_argument("--filter_aggregate_subset_min_dilution_adjusted_core_balanced_delta", type=float, default=None)
    parser.add_argument("--filter_aggregate_subset_min_full_frame_visibility_adjusted_delta", type=float, default=None)
    parser.add_argument("--filter_aggregate_subset_prefer_full_frame_visibility", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--filter_aggregate_subset_tail_safe_shrink_carriers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--filter_aggregate_subset_tail_safe_shrink_scales", default=None)
    parser.add_argument("--filter_aggregate_subset_tail_safe_shrink_min_scale", type=float, default=None)
    parser.add_argument("--filter_risk_safe_shrink_on_train_risk_fail", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--filter_risk_safe_shrink_min_scale", type=float, default=None)
    parser.add_argument("--filter_drop_unmapped", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--filter_require_positive_plan_proxy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--filter_region_source", choices=("scene_prior", "candidate_owned"), default=None)
    parser.add_argument("--filter_candidate_region_max_regions_per_carrier", type=int, default=None)
    parser.add_argument("--filter_candidate_region_min_pixels", type=int, default=None)
    parser.add_argument("--filter_candidate_region_bbox_pad", type=int, default=None)
    parser.add_argument("--filter_candidate_region_min_alpha", type=float, default=None)
    parser.add_argument("--filter_candidate_region_high_error_quantile", type=float, default=None)
    parser.add_argument("--filter_candidate_region_max_views", type=int, default=None)
    parser.add_argument("--filter_candidate_region_expand_faces", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--filter_candidate_region_expand_min_face_pixels", type=int, default=None)
    parser.add_argument("--filter_candidate_region_expand_min_face_views", type=int, default=None)
    parser.add_argument("--filter_candidate_region_expand_max_faces_per_carrier", type=int, default=None)
    parser.add_argument("--filter_candidate_region_frame_aware_ranking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--filter_candidate_region_min_frame_support_fraction", type=float, default=None)
    parser.add_argument("--filter_candidate_region_min_residual_mass_fraction", type=float, default=None)
    parser.add_argument("--filter_candidate_region_max_carriers", type=int, default=None)
    parser.add_argument("--candidate_owned_refit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--candidate_region_expansion_closure", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--candidate_region_expansion_core_priority", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--candidate_region_expansion_core_min_samples", type=int, default=None)
    parser.add_argument("--candidate_region_expansion_core_min_fraction", type=float, default=None)
    parser.add_argument("--candidate_region_expansion_witness_rescue", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--candidate_region_expansion_max_witnesses_per_carrier", type=int, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_prune", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_min_changed_rows", type=int, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_min_bad_rows", type=int, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_max_bad_fraction", type=float, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_balanced_margin", type=float, default=None)
    parser.add_argument(
        "--candidate_region_pre_refit_risk_use_aux_metric_pair",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--candidate_region_pre_refit_risk_ssim_margin", type=float, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_lpips_margin", type=float, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_max_removed_face_fraction", type=float, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_shrink", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_shrink_min_scale", type=float, default=None)
    parser.add_argument(
        "--candidate_region_pre_refit_risk_shrink_severity_aware",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--candidate_region_pre_refit_risk_shrink_severity_select_min", type=float, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_shrink_severity_balanced_span", type=float, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_shrink_tail_fraction", type=float, default=None)
    parser.add_argument(
        "--candidate_region_pre_refit_risk_local_suppression",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--candidate_region_pre_refit_risk_local_suppression_scale", type=float, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_local_suppression_min_bad_balanced", type=float, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_local_suppression_positive_margin", type=float, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_local_suppression_min_face_pixels", type=int, default=None)
    parser.add_argument("--candidate_region_pre_refit_risk_local_suppression_max_faces_per_bad_row", type=int, default=None)
    parser.add_argument("--filter_train_render_region_max_regions", type=int, default=None)
    parser.add_argument("--filter_train_render_region_min_pixels", type=int, default=None)
    parser.add_argument("--filter_train_render_region_min_crop_size", type=int, default=None)
    parser.add_argument("--filter_train_render_region_context_pad", type=int, default=None)
    parser.add_argument("--filter_train_render_region_tail_fraction", type=float, default=None)
    parser.add_argument("--filter_train_render_region_skip_lpips", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ela_alpha_region_risk_objective_bad_only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ela_alpha_region_risk_objective_max_balanced_delta", type=float, default=None)
    parser.add_argument("--ela_alpha_region_risk_objective_max_delta_ssim", type=float, default=None)
    parser.add_argument("--ela_alpha_region_risk_objective_min_delta_lpips", type=float, default=None)
    parser.add_argument("--plan_region_gate_min_regions", type=int, default=None)
    parser.add_argument("--plan_region_gate_min_changed_regions", type=int, default=None)
    parser.add_argument("--plan_region_gate_min_changed_fraction", type=float, default=None)
    parser.add_argument("--plan_region_gate_min_core_balanced_delta", type=float, default=None)
    parser.add_argument("--plan_region_gate_min_core_psnr_delta", type=float, default=None)
    parser.add_argument("--plan_region_gate_min_tail_cvar_delta", type=float, default=None)
    parser.add_argument("--plan_region_gate_max_context_mse_regression", type=float, default=None)
    parser.add_argument("--plan_region_gate_max_negative_fraction", type=float, default=None)
    args = parser.parse_args()

    override_fields = explicit_profile_override_fields(sys.argv, PROFILE_FIELD_NAMES)
    setattr(args, "profile_override_fields", override_fields)
    if str(args.profile) in FIXED_PROFILE_NAMES and override_fields:
        parser.error(
            f"profile {args.profile} is fixed; remove profile-field overrides: "
            + ", ".join(override_fields)
        )

    if int(profile_value(args, "delta_crossfold_folds")) <= 1:
        parser.error("--delta_crossfold_folds must be > 1 for strict carrier plans")
    if int(profile_value(args, "delta_crossfold_min_passing_folds")) <= 0:
        parser.error("--delta_crossfold_min_passing_folds must be positive")
    if int(args.delta_patch_cert_rings) <= 0:
        parser.error("--delta_patch_cert_rings must be positive")
    if int(profile_value(args, "delta_patch_cert_carrier_holdout_min_passing_groups")) <= 0:
        parser.error("--delta_patch_cert_carrier_holdout_min_passing_groups must be positive")
    value = float(profile_value(args, "delta_min_policy_val_adaptive_sample_fraction"))
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        parser.error("--delta_min_policy_val_adaptive_sample_fraction must be in [0, 1]")
    if int(profile_value(args, "delta_min_policy_val_adaptive_min_samples")) < 0:
        parser.error("--delta_min_policy_val_adaptive_min_samples must be >= 0")
    if int(args.delta_patch_cert_carrier_holdout_auto_prefix_min_faces) < 0:
        parser.error("--delta_patch_cert_carrier_holdout_auto_prefix_min_faces must be >= 0")
    value = float(profile_value(args, "delta_patch_cert_carrier_holdout_auto_prefix_min_face_fraction"))
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        parser.error("--delta_patch_cert_carrier_holdout_auto_prefix_min_face_fraction must be in [0, 1]")
    value = float(profile_value(args, "delta_patch_cert_carrier_holdout_auto_prefix_face_bonus"))
    if not math.isfinite(value) or value < 0.0:
        parser.error("--delta_patch_cert_carrier_holdout_auto_prefix_face_bonus must be finite and >= 0")
    for name in ("delta_region_core_weight", "delta_region_context_weight", "delta_region_outside_weight"):
        value = float(profile_value(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and >= 0")
    if int(profile_value(args, "delta_region_boundary_px")) < 0:
        parser.error("--delta_region_boundary_px must be >= 0")
    for name in (
        "delta_render_region_core_weight",
        "delta_render_region_context_weight",
        "delta_render_region_outside_penalty",
        "delta_render_region_tail_cvar_weight",
    ):
        value = float(profile_value(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and >= 0")
    value = float(profile_value(args, "delta_bystander_zero_delta_weight"))
    if not math.isfinite(value) or value < 0.0:
        parser.error("--delta_bystander_zero_delta_weight must be finite and >= 0")
    value = float(profile_value(args, "delta_render_region_tail_fraction"))
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        parser.error("--delta_render_region_tail_fraction must be in (0, 1]")
    if int(profile_value(args, "delta_render_region_min_view_samples")) < 0:
        parser.error("--delta_render_region_min_view_samples must be >= 0")
    if int(profile_value(args, "delta_bystander_zero_delta_min_samples")) < 0:
        parser.error("--delta_bystander_zero_delta_min_samples must be >= 0")
    value = float(profile_value(args, "delta_witness_constraint_weight"))
    if not math.isfinite(value) or value < 0.0:
        parser.error("--delta_witness_constraint_weight must be finite and >= 0")
    value = float(profile_value(args, "delta_witness_constraint_tail_fraction"))
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        parser.error("--delta_witness_constraint_tail_fraction must be in (0, 1]")
    if int(profile_value(args, "delta_witness_constraint_min_samples")) < 0:
        parser.error("--delta_witness_constraint_min_samples must be >= 0")
    value = float(profile_value(args, "delta_witness_constraint_margin"))
    if not math.isfinite(value) or value < 0.0:
        parser.error("--delta_witness_constraint_margin must be finite and >= 0")
    if int(profile_value(args, "filter_max_region_matches_per_plan_carrier")) <= 0:
        parser.error("--filter_max_region_matches_per_plan_carrier must be positive")
    for name in ("filter_min_regions", "filter_min_changed_regions"):
        if int(profile_value(args, name)) < 0:
            parser.error(f"--{name} must be >= 0")
    if not 0.0 <= float(profile_value(args, "filter_min_changed_fraction")) <= 1.0:
        parser.error("--filter_min_changed_fraction must be in [0, 1]")
    if not 0.0 <= float(profile_value(args, "filter_max_negative_core_balanced_fraction")) <= 1.0:
        parser.error("--filter_max_negative_core_balanced_fraction must be in [0, 1]")
    for name in ("filter_min_mean_crop_abs_diff", "filter_min_max_crop_abs_diff"):
        value = float(profile_value(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and >= 0")
    value = float(profile_value(args, "filter_tail_safe_shrink_min_scale"))
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        parser.error("--filter_tail_safe_shrink_min_scale must be in (0, 1]")
    value = float(profile_value(args, "filter_tail_safe_shrink_min_raw_scale"))
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        parser.error("--filter_tail_safe_shrink_min_raw_scale must be in [0, 1]")
    value = float(profile_value(args, "filter_rollback_tail_min_cvar_loss"))
    if not math.isfinite(value) or value < 0.0:
        parser.error("--filter_rollback_tail_min_cvar_loss must be finite and >= 0")
    if int(profile_value(args, "filter_aggregate_subset_min_selected_carriers")) < 0:
        parser.error("--filter_aggregate_subset_min_selected_carriers must be >= 0")
    for name in (
        "filter_aggregate_subset_expected_view_count",
        "filter_aggregate_subset_min_unique_views",
        "filter_aggregate_subset_min_changed_unique_views",
        "filter_aggregate_subset_min_total_pixels",
        "filter_aggregate_subset_min_changed_pixels",
        "filter_aggregate_subset_expected_frame_pixels",
    ):
        if int(profile_value(args, name)) < 0:
            parser.error(f"--{name} must be >= 0")
    for name in (
        "filter_aggregate_subset_min_view_coverage_fraction",
        "filter_aggregate_subset_min_changed_view_coverage_fraction",
        "filter_aggregate_subset_min_changed_pixel_fraction",
        "filter_aggregate_subset_min_full_frame_changed_pixel_fraction",
    ):
        value = float(profile_value(args, name))
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            parser.error(f"--{name} must be in [0, 1]")
    for name in (
        "filter_aggregate_subset_min_area_weighted_core_balanced_delta",
        "filter_aggregate_subset_min_dilution_adjusted_core_balanced_delta",
        "filter_aggregate_subset_min_full_frame_visibility_adjusted_delta",
    ):
        if not math.isfinite(float(profile_value(args, name))):
            parser.error(f"--{name} must be finite")
    value = float(profile_value(args, "filter_aggregate_subset_tail_safe_shrink_min_scale"))
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        parser.error("--filter_aggregate_subset_tail_safe_shrink_min_scale must be in [0, 1]")
    for raw in str(profile_value(args, "filter_aggregate_subset_tail_safe_shrink_scales")).split(","):
        if not raw.strip():
            continue
        try:
            value = float(raw)
        except Exception:
            parser.error("--filter_aggregate_subset_tail_safe_shrink_scales must be comma-separated floats")
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            parser.error("--filter_aggregate_subset_tail_safe_shrink_scales values must be in [0, 1]")
    value = float(profile_value(args, "filter_risk_safe_shrink_min_scale"))
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        parser.error("--filter_risk_safe_shrink_min_scale must be in (0, 1]")
    if (
        not math.isfinite(float(profile_value(args, "ela_alpha_risk_tail_fraction")))
        or float(profile_value(args, "ela_alpha_risk_tail_fraction")) <= 0.0
        or float(profile_value(args, "ela_alpha_risk_tail_fraction")) > 1.0
    ):
        parser.error("--ela_alpha_risk_tail_fraction must be in (0, 1]")
    if (
        not math.isfinite(float(profile_value(args, "ela_alpha_max_negative_gain_fraction")))
        or float(profile_value(args, "ela_alpha_max_negative_gain_fraction")) < 0.0
        or float(profile_value(args, "ela_alpha_max_negative_gain_fraction")) > 1.0
    ):
        parser.error("--ela_alpha_max_negative_gain_fraction must be in [0, 1]")
    if not math.isfinite(float(profile_value(args, "ela_alpha_min_tail_gain"))):
        parser.error("--ela_alpha_min_tail_gain must be finite")
    if (
        not math.isfinite(float(profile_value(args, "ela_alpha_view_tail_cvar_fraction")))
        or float(profile_value(args, "ela_alpha_view_tail_cvar_fraction")) <= 0.0
        or float(profile_value(args, "ela_alpha_view_tail_cvar_fraction")) > 1.0
    ):
        parser.error("--ela_alpha_view_tail_cvar_fraction must be in (0, 1]")
    if not math.isfinite(float(profile_value(args, "ela_alpha_view_tail_min_gain"))):
        parser.error("--ela_alpha_view_tail_min_gain must be finite")
    if (
        not math.isfinite(float(profile_value(args, "ela_alpha_view_tail_max_negative_fraction")))
        or float(profile_value(args, "ela_alpha_view_tail_max_negative_fraction")) < 0.0
        or float(profile_value(args, "ela_alpha_view_tail_max_negative_fraction")) > 1.0
    ):
        parser.error("--ela_alpha_view_tail_max_negative_fraction must be in [0, 1]")
    if str(profile_value(args, "ela_alpha_view_tail_objective")) not in {"mse", "balanced"}:
        parser.error("--ela_alpha_view_tail_objective must be one of: mse, balanced")
    for name in ("ela_alpha_view_tail_ssim_weight", "ela_alpha_view_tail_lpips_weight"):
        if not math.isfinite(float(profile_value(args, name))):
            parser.error(f"--{name} must be finite")
    if int(profile_value(args, "ela_alpha_view_tail_metric_max_side")) < 0:
        parser.error("--ela_alpha_view_tail_metric_max_side must be >= 0")
    if (
        not math.isfinite(float(profile_value(args, "ela_alpha_region_risk_max_negative_fraction")))
        or float(profile_value(args, "ela_alpha_region_risk_max_negative_fraction")) < 0.0
        or float(profile_value(args, "ela_alpha_region_risk_max_negative_fraction")) > 1.0
    ):
        parser.error("--ela_alpha_region_risk_max_negative_fraction must be in [0, 1]")
    if not math.isfinite(float(profile_value(args, "ela_alpha_region_risk_min_tail_gain"))):
        parser.error("--ela_alpha_region_risk_min_tail_gain must be finite")
    for name in (
        "ela_alpha_region_risk_objective_max_balanced_delta",
        "ela_alpha_region_risk_objective_max_delta_ssim",
        "ela_alpha_region_risk_objective_min_delta_lpips",
    ):
        if not math.isfinite(float(profile_value(args, name))):
            parser.error(f"--{name} must be finite")
    if int(profile_value(args, "ela_alpha_region_risk_min_regions")) <= 0:
        parser.error("--ela_alpha_region_risk_min_regions must be > 0")
    if int(profile_value(args, "ela_local_trust_min_supports")) < 0:
        parser.error("--ela_local_trust_min_supports must be >= 0")
    if not math.isfinite(float(profile_value(args, "ela_local_trust_max_residual_std"))):
        parser.error("--ela_local_trust_max_residual_std must be finite; use a negative value to disable")
    if not 0.0 <= float(profile_value(args, "ela_local_trust_min_agreement")) <= 1.0:
        parser.error("--ela_local_trust_min_agreement must be in [0, 1]")
    if float(profile_value(args, "ela_local_trust_agreement_scale")) <= 0.0:
        parser.error("--ela_local_trust_agreement_scale must be > 0")
    if not -1.0 <= float(profile_value(args, "ela_local_trust_confidence_quantile")) < 1.0:
        parser.error("--ela_local_trust_confidence_quantile must be in [-1, 1)")
    if float(profile_value(args, "ela_local_trust_min_confidence")) < 0.0:
        parser.error("--ela_local_trust_min_confidence must be >= 0")
    if float(profile_value(args, "ela_local_trust_min_weight")) < 0.0:
        parser.error("--ela_local_trust_min_weight must be >= 0")
    if int(profile_value(args, "filter_train_render_region_max_regions")) <= 0:
        parser.error("--filter_train_render_region_max_regions must be > 0")
    if int(profile_value(args, "filter_train_render_region_min_pixels")) <= 0:
        parser.error("--filter_train_render_region_min_pixels must be > 0")
    if int(profile_value(args, "filter_train_render_region_min_crop_size")) <= 0:
        parser.error("--filter_train_render_region_min_crop_size must be > 0")
    if int(profile_value(args, "filter_train_render_region_context_pad")) < 0:
        parser.error("--filter_train_render_region_context_pad must be >= 0")
    if not 0.0 < float(profile_value(args, "filter_train_render_region_tail_fraction")) <= 1.0:
        parser.error("--filter_train_render_region_tail_fraction must be in (0, 1]")
    for name in ("plan_region_gate_min_regions", "plan_region_gate_min_changed_regions"):
        if int(profile_value(args, name)) < 0:
            parser.error(f"--{name} must be >= 0")
    if not 0.0 <= float(profile_value(args, "plan_region_gate_min_changed_fraction")) <= 1.0:
        parser.error("--plan_region_gate_min_changed_fraction must be in [0, 1]")
    for name in (
        "plan_region_gate_min_core_balanced_delta",
        "plan_region_gate_min_core_psnr_delta",
        "plan_region_gate_min_tail_cvar_delta",
        "plan_region_gate_max_context_mse_regression",
    ):
        if not math.isfinite(float(profile_value(args, name))):
            parser.error(f"--{name} must be finite")
    if not 0.0 <= float(profile_value(args, "plan_region_gate_max_negative_fraction")) <= 1.0:
        parser.error("--plan_region_gate_max_negative_fraction must be in [0, 1]")
    if not 0.0 < float(profile_value(args, "selector_strict_adaptive_scale_min")) <= 1.0:
        parser.error("--selector_strict_adaptive_scale_min must be in (0, 1]")
    if int(profile_value(args, "selector_strict_adaptive_scale_max_extra")) < 0:
        parser.error("--selector_strict_adaptive_scale_max_extra must be non-negative")
    if not 0.0 < float(profile_value(args, "selector_strict_adaptive_scale_tail_fraction")) <= 1.0:
        parser.error("--selector_strict_adaptive_scale_tail_fraction must be in (0, 1]")
    if float(profile_value(args, "selector_region_min_mean_core_balanced_delta")) < 0.0:
        parser.error("--selector_region_min_mean_core_balanced_delta must be non-negative")
    if float(profile_value(args, "selector_region_min_mean_delta_psnr")) < 0.0:
        parser.error("--selector_region_min_mean_delta_psnr must be non-negative")
    if not 0.0 <= float(profile_value(args, "selector_region_min_changed_fraction")) <= 1.0:
        parser.error("--selector_region_min_changed_fraction must be in [0, 1]")
    if not 0.0 <= float(profile_value(args, "selector_region_max_negative_core_balanced_fraction")) <= 1.0:
        parser.error("--selector_region_max_negative_core_balanced_fraction must be in [0, 1]")
    if float(profile_value(args, "selector_region_max_context_mse_regression")) < 0.0:
        parser.error("--selector_region_max_context_mse_regression must be non-negative")
    if str(profile_value(args, "filter_region_source")) not in {"scene_prior", "candidate_owned"}:
        parser.error("--filter_region_source must be scene_prior or candidate_owned")
    if bool(profile_value(args, "candidate_owned_refit")) and str(profile_value(args, "filter_region_source")) != "candidate_owned":
        parser.error("--candidate_owned_refit requires --filter_region_source candidate_owned")
    if int(profile_value(args, "filter_candidate_region_max_regions_per_carrier")) <= 0:
        parser.error("--filter_candidate_region_max_regions_per_carrier must be > 0")
    if int(profile_value(args, "filter_candidate_region_min_pixels")) <= 0:
        parser.error("--filter_candidate_region_min_pixels must be > 0")
    if int(profile_value(args, "filter_candidate_region_bbox_pad")) < 0:
        parser.error("--filter_candidate_region_bbox_pad must be >= 0")
    value = float(profile_value(args, "filter_candidate_region_min_alpha"))
    if not math.isfinite(value) or value < 0.0:
        parser.error("--filter_candidate_region_min_alpha must be finite and >= 0")
    value = float(profile_value(args, "filter_candidate_region_high_error_quantile"))
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        parser.error("--filter_candidate_region_high_error_quantile must be in [0, 1]")
    if int(profile_value(args, "filter_candidate_region_max_views")) < 0:
        parser.error("--filter_candidate_region_max_views must be >= 0")
    if int(profile_value(args, "filter_candidate_region_expand_min_face_pixels")) <= 0:
        parser.error("--filter_candidate_region_expand_min_face_pixels must be > 0")
    if int(profile_value(args, "filter_candidate_region_expand_min_face_views")) <= 0:
        parser.error("--filter_candidate_region_expand_min_face_views must be > 0")
    if int(profile_value(args, "filter_candidate_region_expand_max_faces_per_carrier")) < 0:
        parser.error("--filter_candidate_region_expand_max_faces_per_carrier must be >= 0")
    for name in (
        "filter_candidate_region_min_frame_support_fraction",
        "filter_candidate_region_min_residual_mass_fraction",
    ):
        value = float(profile_value(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and >= 0")
    if int(profile_value(args, "filter_candidate_region_max_carriers")) < 0:
        parser.error("--filter_candidate_region_max_carriers must be >= 0")
    if int(profile_value(args, "candidate_region_expansion_core_min_samples")) < 0:
        parser.error("--candidate_region_expansion_core_min_samples must be >= 0")
    value = float(profile_value(args, "candidate_region_expansion_core_min_fraction"))
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        parser.error("--candidate_region_expansion_core_min_fraction must be in [0, 1]")
    if int(profile_value(args, "candidate_region_expansion_max_witnesses_per_carrier")) < 0:
        parser.error("--candidate_region_expansion_max_witnesses_per_carrier must be >= 0")
    for name in (
        "candidate_region_pre_refit_risk_min_changed_rows",
        "candidate_region_pre_refit_risk_min_bad_rows",
    ):
        if int(profile_value(args, name)) <= 0:
            parser.error(f"--{name} must be > 0")
    value = float(profile_value(args, "candidate_region_pre_refit_risk_max_bad_fraction"))
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        parser.error("--candidate_region_pre_refit_risk_max_bad_fraction must be in [0, 1]")
    for name in (
        "candidate_region_pre_refit_risk_balanced_margin",
        "candidate_region_pre_refit_risk_ssim_margin",
        "candidate_region_pre_refit_risk_lpips_margin",
    ):
        value = float(profile_value(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and >= 0")
    value = float(profile_value(args, "candidate_region_pre_refit_risk_max_removed_face_fraction"))
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        parser.error("--candidate_region_pre_refit_risk_max_removed_face_fraction must be in (0, 1]")
    value = float(profile_value(args, "candidate_region_pre_refit_risk_shrink_min_scale"))
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        parser.error("--candidate_region_pre_refit_risk_shrink_min_scale must be in (0, 1]")
    value = float(profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_select_min"))
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        parser.error("--candidate_region_pre_refit_risk_shrink_severity_select_min must be in [0, 1]")
    value = float(profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_balanced_span"))
    if not math.isfinite(value) or value <= 0.0:
        parser.error("--candidate_region_pre_refit_risk_shrink_severity_balanced_span must be finite and > 0")
    value = float(profile_value(args, "candidate_region_pre_refit_risk_shrink_tail_fraction"))
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        parser.error("--candidate_region_pre_refit_risk_shrink_tail_fraction must be in (0, 1]")
    value = float(profile_value(args, "candidate_region_pre_refit_risk_local_suppression_scale"))
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        parser.error("--candidate_region_pre_refit_risk_local_suppression_scale must be in (0, 1]")
    for name in (
        "candidate_region_pre_refit_risk_local_suppression_min_bad_balanced",
        "candidate_region_pre_refit_risk_local_suppression_positive_margin",
    ):
        value = float(profile_value(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and >= 0")
    value = int(profile_value(args, "candidate_region_pre_refit_risk_local_suppression_min_face_pixels"))
    if value < 0:
        parser.error("--candidate_region_pre_refit_risk_local_suppression_min_face_pixels must be >= 0")
    value = int(profile_value(args, "candidate_region_pre_refit_risk_local_suppression_max_faces_per_bad_row"))
    if value < 1:
        parser.error("--candidate_region_pre_refit_risk_local_suppression_max_faces_per_bad_row must be >= 1")
    return args


def profile_value(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is not None and value != "":
        return value
    return PROFILE_DEFAULTS[str(args.profile)][name]


def resolved_profile_values(args: argparse.Namespace) -> dict[str, Any]:
    return {name: profile_value(args, name) for name in PROFILE_FIELD_NAMES}


def stable_json_sha256(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def explicit_profile_override_fields(argv: list[str], profile_fields: tuple[str, ...]) -> list[str]:
    fields = set(profile_fields)
    out: set[str] = set()
    for raw in argv[1:]:
        if not str(raw).startswith("--"):
            continue
        flag = str(raw).split("=", 1)[0]
        if flag.startswith("--no-"):
            name = flag[5:]
        else:
            name = flag[2:]
        normalized = name.replace("-", "_")
        if normalized in fields:
            out.add(normalized)
    return sorted(out)


def profile_contract_id(args: argparse.Namespace) -> str:
    return PROFILE_CONTRACT_IDS.get(str(args.profile), str(args.profile))


def scene_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]


def stage_set(raw: str) -> set[str]:
    stages = {item.strip() for item in str(raw).replace(",", " ").split() if item.strip()}
    allowed = {"plan", "filter", "selector"}
    unknown = stages - allowed
    if unknown:
        raise ValueError(f"unknown stages: {', '.join(sorted(unknown))}")
    return stages


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in cmd)


def safe_method_suffix(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))
    return out.strip("_") or "pipeline"


def phasej_test_method(args: argparse.Namespace) -> str:
    if not bool(getattr(args, "isolate_phasej_methods", True)):
        return str(args.phasej_test_method)
    return f"{args.phasej_test_method}_{safe_method_suffix(args.pipeline_label)}"


def phasej_trainval_method(args: argparse.Namespace) -> str:
    if not bool(getattr(args, "isolate_phasej_methods", True)):
        return str(args.phasej_trainval_method)
    return f"{args.phasej_trainval_method}_{safe_method_suffix(args.pipeline_label)}"


def output_root(args: argparse.Namespace) -> Path:
    return resolve_path(args.output_root)


def evidence_root(args: argparse.Namespace) -> Path:
    if str(args.evidence_root).strip():
        return resolve_path(args.evidence_root)
    return output_root(args) / "surface_evidence"


def plan_template(args: argparse.Namespace) -> str:
    if str(args.plan_template).strip():
        return str(resolve_path(args.plan_template))
    return str(output_root(args) / "candidate_plans" / "{scene}" / "facelocal_visual_candidate_plan.json")


def filtered_plan_template(args: argparse.Namespace) -> str:
    if str(args.filtered_plan_template).strip():
        return str(resolve_path(args.filtered_plan_template))
    return str(output_root(args) / "filtered_candidate_plans" / "{scene}" / "facelocal_visual_candidate_plan_filtered.json")


def aggregate_subset_plan_template(args: argparse.Namespace) -> str:
    filtered = Path(filtered_plan_template(args))
    return str(filtered.with_name("facelocal_visual_candidate_plan_filtered_aggregate_subset.json"))


def refit_plan_template(args: argparse.Namespace) -> str:
    return str(output_root(args) / "candidate_owned_refit_plans" / "{scene}" / "facelocal_visual_candidate_plan_refit.json")


def selector_plan_template(args: argparse.Namespace) -> str:
    if str(args.selector_plan_template).strip():
        return str(resolve_path(args.selector_plan_template))
    if bool(profile_value(args, "filter_aggregate_subset")):
        return aggregate_subset_plan_template(args)
    if bool(profile_value(args, "use_filtered_plan_for_selector")):
        return filtered_plan_template(args)
    return plan_template(args)


def format_plan_path(args: argparse.Namespace, scene: str) -> Path:
    return Path(plan_template(args).format(scene=scene))


def format_filtered_plan_path(args: argparse.Namespace, scene: str) -> Path:
    return Path(filtered_plan_template(args).format(scene=scene))


def format_aggregate_subset_plan_path(args: argparse.Namespace, scene: str) -> Path:
    return Path(aggregate_subset_plan_template(args).format(scene=scene))


def format_refit_plan_path(args: argparse.Namespace, scene: str) -> Path:
    return Path(refit_plan_template(args).format(scene=scene))


def format_selector_plan_path(args: argparse.Namespace, scene: str) -> Path:
    return Path(selector_plan_template(args).format(scene=scene))


def scene_image_dir(args: argparse.Namespace, scene: str) -> Path:
    image_dir = str(args.outdoor_images) if scene in OUTDOOR_SCENES else str(args.indoor_images)
    return Path(args.dataset_root) / scene / image_dir


def infer_expected_frame_pixels(args: argparse.Namespace, scene: str) -> int:
    configured = int(profile_value(args, "filter_aggregate_subset_expected_frame_pixels"))
    if configured > 0:
        return configured
    image_dir = scene_image_dir(args, scene)
    for suffix in ("*.png", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
        for path in sorted(image_dir.glob(suffix)):
            try:
                from PIL import Image

                with Image.open(path) as image:
                    width, height = image.size
                return int(width) * int(height)
            except Exception:
                continue
    return 0


def candidate_plan_row_count(path: Path) -> int:
    payload = load_json(path)
    if isinstance(payload, list):
        return len([row for row in payload if isinstance(row, dict)])
    if not isinstance(payload, dict):
        return 0
    for key in ("candidates", "accepted", "accepted_preview"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return len([row for row in rows if isinstance(row, dict)])
    return 0


def candidate_plan_face_ids(path: Path) -> set[int]:
    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = []
        for key in ("candidates", "accepted", "accepted_preview"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
    else:
        rows = []
    face_ids: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        _add_int_face_id(face_ids, row.get("face_id"))
    return face_ids


def use_refit_plan_for_filter(args: argparse.Namespace, scene: str) -> bool:
    if not bool(profile_value(args, "candidate_owned_refit")):
        return False
    if bool(getattr(args, "dry_run", False)):
        return True
    return candidate_plan_row_count(format_refit_plan_path(args, scene)) > 0


def format_filter_input_plan_path(args: argparse.Namespace, scene: str) -> Path:
    if use_refit_plan_for_filter(args, scene):
        return format_refit_plan_path(args, scene)
    return format_plan_path(args, scene)


def format_region_carrier_path(args: argparse.Namespace, scene: str) -> Path:
    return Path(str(resolve_path(args.render_region_carrier_template)).format(scene=scene))


def format_candidate_region_root(args: argparse.Namespace, scene: str) -> Path:
    return output_root(args) / "candidate_owned_render_regions" / scene


def format_candidate_region_carrier_path(args: argparse.Namespace, scene: str) -> Path:
    return format_candidate_region_root(args, scene) / "candidate_render_regions.json"


def format_candidate_region_objective_path(args: argparse.Namespace, scene: str) -> Path:
    return format_candidate_region_root(args, scene) / "train_render_region_objective_raw_base.json"


def format_pre_refit_risk_prune_report_path(args: argparse.Namespace, scene: str) -> Path:
    return output_root(args) / "candidate_owned_refit_plans" / scene / "pre_refit_risk_prune_report.json"


def format_pre_refit_risk_shrink_report_path(
    args: argparse.Namespace,
    scene: str,
    *,
    purpose: str = "refit",
) -> Path:
    if purpose == "refit":
        name = "pre_refit_risk_shrink_report_refit.json"
    elif purpose == "selector":
        name = "pre_refit_risk_shrink_report_selector.json"
    elif purpose == "legacy":
        name = "pre_refit_risk_shrink_report.json"
    else:
        raise ValueError(f"unknown risk-shrink report purpose: {purpose}")
    return output_root(args) / "candidate_owned_refit_plans" / scene / name


def format_pre_refit_risk_shrink_alpha_path(
    args: argparse.Namespace,
    scene: str,
    *,
    purpose: str = "refit",
) -> Path:
    if purpose == "refit":
        name = "pre_refit_risk_scale_refit.json"
    elif purpose == "selector":
        name = "selector_materialize_alpha.json"
    elif purpose == "legacy":
        name = "pre_refit_risk_shrink_materialize_alpha.json"
    else:
        raise ValueError(f"unknown risk-shrink alpha purpose: {purpose}")
    return output_root(args) / "candidate_owned_refit_plans" / scene / name


def selected_policy_model(args: argparse.Namespace, scene: str) -> Path:
    summary = load_json(resolve_path(args.policy_root) / scene / "summary.json")
    selected = summary.get("selected", {}) if isinstance(summary, dict) else {}
    model_path = selected.get("model_path") if isinstance(selected, dict) else None
    if not model_path:
        raise RuntimeError(f"missing selected model for {scene}: {resolve_path(args.policy_root) / scene / 'summary.json'}")
    model = resolve_path(model_path)
    if not model.is_dir():
        raise FileNotFoundError(model)
    return model


def command_log(root: Path, stage: str, scene: str) -> Path:
    return root / "logs" / stage / f"{scene}.log"


def replace_arg(command: list[str], flag: str, value: str) -> None:
    try:
        idx = command.index(flag)
    except ValueError as exc:
        raise ValueError(f"missing command flag {flag}") from exc
    if idx + 1 >= len(command):
        raise ValueError(f"missing value after command flag {flag}")
    command[idx + 1] = str(value)


def append_or_replace_arg(command: list[str], flag: str, value: str) -> None:
    if flag in command:
        replace_arg(command, flag, value)
    else:
        command.extend([flag, str(value)])


def _add_int_face_id(face_ids: set[int], value: Any) -> None:
    try:
        face_ids.add(int(value))
    except Exception:
        return


def _collect_face_ids_from_candidate_payload(payload: Any, face_ids: set[int]) -> None:
    if isinstance(payload, dict):
        for key in ("face_id",):
            if key in payload:
                _add_int_face_id(face_ids, payload.get(key))
        for key in ("face_ids", "carrier_faces", "faces", "expanded_face_ids", "seed_face_ids"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _collect_face_ids_from_candidate_payload(item, face_ids)
                    else:
                        _add_int_face_id(face_ids, item)
            elif isinstance(value, dict):
                _collect_face_ids_from_candidate_payload(value, face_ids)
        for key in ("candidates", "accepted", "carriers", "regions", "patch_certificate", "post_cluster_patch_certificate"):
            value = payload.get(key)
            if isinstance(value, (dict, list)):
                _collect_face_ids_from_candidate_payload(value, face_ids)
    elif isinstance(payload, list):
        for item in payload:
            _collect_face_ids_from_candidate_payload(item, face_ids)


def _candidate_region_carrier_face_map(payload: Any) -> dict[str, set[int]]:
    if not isinstance(payload, dict):
        return {}
    carriers = payload.get("carriers")
    if not isinstance(carriers, list):
        return {}
    out: dict[str, set[int]] = {}
    for carrier in carriers:
        if not isinstance(carrier, dict):
            continue
        carrier_id = str(carrier.get("carrier_id", "")).strip()
        if not carrier_id:
            continue
        face_ids: set[int] = set()
        for key in ("face_ids", "seed_face_ids", "expanded_face_ids"):
            value = carrier.get(key)
            if isinstance(value, list):
                for item in value:
                    _add_int_face_id(face_ids, item)
        if not face_ids:
            _collect_face_ids_from_candidate_payload(carrier, face_ids)
        out[carrier_id] = face_ids
    return out


def _candidate_region_carrier_face_meta(payload: Any) -> dict[str, dict[int, dict[str, Any]]]:
    if not isinstance(payload, dict):
        return {}
    carriers = payload.get("carriers")
    if not isinstance(carriers, list):
        return {}
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for carrier in carriers:
        if not isinstance(carrier, dict):
            continue
        carrier_id = str(carrier.get("carrier_id", "")).strip()
        if not carrier_id:
            continue
        meta: dict[int, dict[str, Any]] = {}
        expansion = carrier.get("expansion")
        rows = expansion.get("rows", []) if isinstance(expansion, dict) else []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    fid = int(row.get("face_id"))
                except Exception:
                    continue
                views_raw = row.get("views")
                views = {str(view) for view in views_raw} if isinstance(views_raw, list) else set()
                pixels = _float_or_none(row.get("pixels")) or 0.0
                item = meta.setdefault(fid, {"pixels": 0.0, "views": set(), "seed_face": False})
                item["pixels"] = max(float(item.get("pixels", 0.0)), float(pixels))
                item["views"] = set(item.get("views", set())) | views
                item["seed_face"] = bool(item.get("seed_face", False)) or bool(row.get("seed_face", False))
        for key in ("expanded_face_ids", "face_ids", "seed_face_ids"):
            value = carrier.get(key)
            if isinstance(value, list):
                for item in value:
                    try:
                        fid = int(item)
                    except Exception:
                        continue
                    meta.setdefault(fid, {"pixels": 0.0, "views": set(), "seed_face": key == "seed_face_ids"})
        out[carrier_id] = meta
    return out


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    return None


def _pre_refit_objective_row_is_evaluable(row: dict[str, Any]) -> bool:
    if row.get("crop_changed") is False:
        return False
    if bool(row.get("metrics_skipped_equal_crop", False)):
        return False
    return True


def _pre_refit_objective_row_is_bad(args: argparse.Namespace, row: dict[str, Any]) -> bool:
    balanced = _float_or_none(row.get("core_balanced_delta"))
    if balanced is not None and balanced < -float(profile_value(args, "candidate_region_pre_refit_risk_balanced_margin")):
        return True
    if not bool(profile_value(args, "candidate_region_pre_refit_risk_use_aux_metric_pair")):
        return False
    delta_ssim = _float_or_none(row.get("delta_core_ssim"))
    delta_lpips = _float_or_none(row.get("delta_core_lpips"))
    return (
        delta_ssim is not None
        and delta_lpips is not None
        and delta_ssim < -float(profile_value(args, "candidate_region_pre_refit_risk_ssim_margin"))
        and delta_lpips > float(profile_value(args, "candidate_region_pre_refit_risk_lpips_margin"))
    )


def _write_pre_refit_risk_prune_report(args: argparse.Namespace, scene: str, report: dict[str, Any]) -> None:
    path = format_pre_refit_risk_prune_report_path(args, scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _apply_candidate_region_pre_refit_risk_prune(
    args: argparse.Namespace,
    scene: str,
    face_ids: set[int],
) -> set[int]:
    report_path = format_pre_refit_risk_prune_report_path(args, scene)
    objective_path = format_candidate_region_objective_path(args, scene)
    carrier_path = format_candidate_region_carrier_path(args, scene)
    base_report: dict[str, Any] = {
        "enabled": bool(profile_value(args, "candidate_region_pre_refit_risk_prune")),
        "scene": scene,
        "objective_path": str(objective_path),
        "carrier_path": str(carrier_path),
        "report_path": str(report_path),
        "input_face_count": len(face_ids),
        "policy": {
            "row_evaluable": "crop_changed is not false and metrics_skipped_equal_crop is false",
            "bad_row": (
                "core_balanced_delta < -balanced_margin, optionally OR "
                "(delta_core_ssim < -ssim_margin AND delta_core_lpips > lpips_margin)"
            ),
            "min_changed_rows": int(profile_value(args, "candidate_region_pre_refit_risk_min_changed_rows")),
            "min_bad_rows": int(profile_value(args, "candidate_region_pre_refit_risk_min_bad_rows")),
            "max_bad_fraction": float(profile_value(args, "candidate_region_pre_refit_risk_max_bad_fraction")),
            "balanced_margin": float(profile_value(args, "candidate_region_pre_refit_risk_balanced_margin")),
            "use_aux_metric_pair": bool(
                profile_value(args, "candidate_region_pre_refit_risk_use_aux_metric_pair")
            ),
            "ssim_margin": float(profile_value(args, "candidate_region_pre_refit_risk_ssim_margin")),
            "lpips_margin": float(profile_value(args, "candidate_region_pre_refit_risk_lpips_margin")),
            "max_removed_face_fraction": float(
                profile_value(args, "candidate_region_pre_refit_risk_max_removed_face_fraction")
            ),
        },
    }
    if not bool(profile_value(args, "candidate_region_pre_refit_risk_prune")):
        return face_ids
    carrier_payload = load_json(carrier_path)
    objective_payload = load_json(objective_path)
    carrier_faces = _candidate_region_carrier_face_map(carrier_payload)
    rows = objective_payload.get("rows", []) if isinstance(objective_payload, dict) else []
    if not carrier_faces or not isinstance(rows, list) or not rows:
        base_report.update(
            {
                "status": "skipped_missing_inputs",
                "carrier_count": len(carrier_faces),
                "objective_row_count": len(rows) if isinstance(rows, list) else 0,
                "removed_carrier_count": 0,
                "removed_face_count": 0,
                "remaining_face_count": len(face_ids),
            }
        )
        _write_pre_refit_risk_prune_report(args, scene, base_report)
        return face_ids

    rows_by_carrier: dict[str, list[dict[str, Any]]] = {}
    missing_carrier_rows = 0
    unknown_carrier_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        carrier_id = str(row.get("carrier_id", "")).strip()
        if not carrier_id:
            missing_carrier_rows += 1
            continue
        if carrier_id not in carrier_faces:
            unknown_carrier_rows += 1
            continue
        rows_by_carrier.setdefault(carrier_id, []).append(row)

    candidates: list[dict[str, Any]] = []
    carrier_reports: list[dict[str, Any]] = []
    for carrier_id, carrier_rows in sorted(rows_by_carrier.items()):
        evaluable_rows = [row for row in carrier_rows if _pre_refit_objective_row_is_evaluable(row)]
        bad_rows = [row for row in evaluable_rows if _pre_refit_objective_row_is_bad(args, row)]
        changed_count = len(evaluable_rows)
        bad_count = len(bad_rows)
        bad_fraction = (bad_count / changed_count) if changed_count else 0.0
        balanced_values = [
            value
            for value in (_float_or_none(row.get("core_balanced_delta")) for row in evaluable_rows)
            if value is not None
        ]
        worst_balanced = min(balanced_values) if balanced_values else None
        mean_balanced = (sum(balanced_values) / len(balanced_values)) if balanced_values else None
        face_count = len(carrier_faces.get(carrier_id, set()))
        carrier_report = {
            "carrier_id": carrier_id,
            "row_count": len(carrier_rows),
            "evaluable_row_count": changed_count,
            "bad_row_count": bad_count,
            "bad_fraction": bad_fraction,
            "mean_core_balanced_delta": mean_balanced,
            "worst_core_balanced_delta": worst_balanced,
            "face_count": face_count,
            "selected_for_prune": False,
        }
        carrier_reports.append(carrier_report)
        if (
            changed_count >= int(profile_value(args, "candidate_region_pre_refit_risk_min_changed_rows"))
            and bad_count >= int(profile_value(args, "candidate_region_pre_refit_risk_min_bad_rows"))
            and bad_fraction > float(profile_value(args, "candidate_region_pre_refit_risk_max_bad_fraction"))
        ):
            score = (
                bad_fraction,
                bad_count,
                -(worst_balanced if worst_balanced is not None else 0.0),
                face_count,
            )
            candidates.append({"carrier_id": carrier_id, "score": score, "faces": carrier_faces.get(carrier_id, set())})

    max_removed_fraction = float(profile_value(args, "candidate_region_pre_refit_risk_max_removed_face_fraction"))
    max_removed_faces = max(1, int(math.floor(len(face_ids) * max_removed_fraction)))
    removed_faces: set[int] = set()
    removed_carriers: list[str] = []
    skipped_by_budget: list[str] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        candidate_faces = {fid for fid in candidate["faces"] if fid in face_ids}
        next_removed = removed_faces | candidate_faces
        if len(next_removed) >= len(face_ids) or len(next_removed) > max_removed_faces:
            skipped_by_budget.append(str(candidate["carrier_id"]))
            continue
        removed_faces = next_removed
        removed_carriers.append(str(candidate["carrier_id"]))

    for carrier_report in carrier_reports:
        carrier_report["selected_for_prune"] = carrier_report["carrier_id"] in set(removed_carriers)

    pruned = set(face_ids) - removed_faces
    report = {
        **base_report,
        "status": "applied",
        "carrier_count": len(carrier_faces),
        "objective_row_count": len([row for row in rows if isinstance(row, dict)]),
        "mapped_objective_carrier_count": len(rows_by_carrier),
        "missing_carrier_rows": missing_carrier_rows,
        "unknown_carrier_rows": unknown_carrier_rows,
        "candidate_bad_carrier_count": len(candidates),
        "removed_carrier_count": len(removed_carriers),
        "removed_carrier_ids": removed_carriers,
        "skipped_carrier_ids_by_budget": skipped_by_budget,
        "removed_face_count": len(removed_faces),
        "remaining_face_count": len(pruned),
        "removed_face_fraction": (len(removed_faces) / len(face_ids)) if face_ids else 0.0,
        "carrier_reports": carrier_reports,
    }
    _write_pre_refit_risk_prune_report(args, scene, report)
    return pruned


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tail_cvar(values: list[float], tail_fraction: float) -> float | None:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return None
    tail_count = max(1, int(math.ceil(len(finite_values) * max(0.0, min(1.0, tail_fraction)))))
    return sum(sorted(finite_values)[:tail_count]) / tail_count


def _candidate_region_pre_refit_risk_shrink_decision(
    args: argparse.Namespace,
    bad_fraction: float,
    balanced_values: list[float],
) -> tuple[float, dict[str, Any]]:
    threshold = float(profile_value(args, "candidate_region_pre_refit_risk_max_bad_fraction"))
    min_scale = float(profile_value(args, "candidate_region_pre_refit_risk_shrink_min_scale"))
    if bad_fraction <= threshold:
        fraction_severity = 0.0
    elif threshold >= 1.0:
        fraction_severity = 1.0
    else:
        fraction_severity = _clamp01((float(bad_fraction) - threshold) / max(1.0 - threshold, 1.0e-12))

    severity_aware = bool(profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_aware"))
    tail_fraction = float(profile_value(args, "candidate_region_pre_refit_risk_shrink_tail_fraction"))
    balanced_margin = float(profile_value(args, "candidate_region_pre_refit_risk_balanced_margin"))
    balanced_span = float(profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_balanced_span"))
    worst_balanced = min(balanced_values) if balanced_values else None
    tail_balanced_cvar = _tail_cvar(balanced_values, tail_fraction)
    worst_severity = (
        _clamp01((-(float(worst_balanced)) - balanced_margin) / balanced_span)
        if severity_aware and worst_balanced is not None
        else 0.0
    )
    tail_severity = (
        _clamp01((-(float(tail_balanced_cvar)) - balanced_margin) / balanced_span)
        if severity_aware and tail_balanced_cvar is not None
        else 0.0
    )
    combined_severity = max(fraction_severity, worst_severity, tail_severity)
    scale = float(max(min_scale, min(1.0, 1.0 - combined_severity * (1.0 - min_scale))))
    return scale, {
        "severity_aware": severity_aware,
        "severity": combined_severity,
        "fraction_severity": fraction_severity,
        "worst_balanced_severity": worst_severity,
        "tail_cvar_severity": tail_severity,
        "worst_core_balanced_delta": worst_balanced,
        "tail_core_balanced_cvar_delta": tail_balanced_cvar,
        "tail_fraction": tail_fraction,
        "balanced_margin": balanced_margin,
        "balanced_span": balanced_span,
    }


def _plan_materialize_alpha_path_from_payload(plan: dict[str, Any], plan_path: Path) -> Path | None:
    raw = str(plan.get("render_cvar_aggregate_subset_materialize_alpha_json", "")).strip()
    if not raw:
        nested = plan.get("render_cvar_aggregate_subset")
        if isinstance(nested, dict):
            raw = str(nested.get("materialize_alpha_json", "")).strip()
    if not raw:
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [plan_path.parent / path, ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _read_face_alphas(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw = payload.get("face_alphas", payload.get("alphas", payload)) if isinstance(payload, dict) else payload
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = []
        for row in raw:
            if isinstance(row, dict):
                items.append((row.get("face_id"), row.get("alpha", row.get("scale"))))
    else:
        items = []
    for face_id, alpha in items:
        try:
            fid = int(face_id)
            scale = float(alpha)
        except Exception:
            continue
        if fid >= 0 and math.isfinite(scale):
            out[str(fid)] = max(0.0, min(1.0, scale))
    return out


def _write_pre_refit_risk_shrink_report(
    args: argparse.Namespace,
    scene: str,
    report: dict[str, Any],
    *,
    purpose: str = "refit",
) -> None:
    path = format_pre_refit_risk_shrink_report_path(args, scene, purpose=purpose)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_candidate_region_pre_refit_risk_shrink_policy(
    args: argparse.Namespace,
    scene: str,
    plan_path: Path,
    face_scope: set[int] | None = None,
    attach_to_plan: bool = True,
    alpha_purpose: str = "refit",
) -> dict[str, Any]:
    report_path = format_pre_refit_risk_shrink_report_path(args, scene, purpose=alpha_purpose)
    alpha_path = format_pre_refit_risk_shrink_alpha_path(args, scene, purpose=alpha_purpose)
    objective_path = format_candidate_region_objective_path(args, scene)
    carrier_path = format_candidate_region_carrier_path(args, scene)
    base_report: dict[str, Any] = {
        "enabled": bool(profile_value(args, "candidate_region_pre_refit_risk_shrink")),
        "scene": scene,
        "plan_path": str(plan_path),
        "objective_path": str(objective_path),
        "carrier_path": str(carrier_path),
        "report_path": str(report_path),
        "alpha_json": str(alpha_path),
        "alpha_purpose": alpha_purpose,
        "policy": {
            "operator": "candidate_region_pre_refit_risk_carrier_shrink",
            "test_usage": "none",
            "selection_uses_test": False,
            "enforcement": (
                "per-face monotone risk scale is passed to candidate-owned facelocal refit when "
                "face_scope is explicit; the same alpha is also attachable to selector strict replay"
            ),
            "row_evaluable": "crop_changed is not false and metrics_skipped_equal_crop is false",
            "bad_row": (
                "core_balanced_delta < -balanced_margin, optionally OR "
                "(delta_core_ssim < -ssim_margin AND delta_core_lpips > lpips_margin)"
            ),
            "min_changed_rows": int(profile_value(args, "candidate_region_pre_refit_risk_min_changed_rows")),
            "min_bad_rows": int(profile_value(args, "candidate_region_pre_refit_risk_min_bad_rows")),
            "max_bad_fraction": float(profile_value(args, "candidate_region_pre_refit_risk_max_bad_fraction")),
            "balanced_margin": float(profile_value(args, "candidate_region_pre_refit_risk_balanced_margin")),
            "use_aux_metric_pair": bool(
                profile_value(args, "candidate_region_pre_refit_risk_use_aux_metric_pair")
            ),
            "ssim_margin": float(profile_value(args, "candidate_region_pre_refit_risk_ssim_margin")),
            "lpips_margin": float(profile_value(args, "candidate_region_pre_refit_risk_lpips_margin")),
            "min_scale": float(profile_value(args, "candidate_region_pre_refit_risk_shrink_min_scale")),
            "severity_aware": bool(
                profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_aware")
            ),
            "severity_select_min": float(
                profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_select_min")
            ),
            "severity_balanced_span": float(
                profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_balanced_span")
            ),
            "tail_fraction": float(profile_value(args, "candidate_region_pre_refit_risk_shrink_tail_fraction")),
            "local_suppression": bool(profile_value(args, "candidate_region_pre_refit_risk_local_suppression")),
            "locality_evidence": "carrier_expansion_view_pixel_metadata_only",
            "locality_no_fallback_without_view_pixel_match": True,
            "positive_witness_preservation": (
                "faces with train-positive witness views are preserved unless the bad row is severe"
            ),
            "local_suppression_scale": float(
                profile_value(args, "candidate_region_pre_refit_risk_local_suppression_scale")
            ),
            "local_suppression_min_bad_balanced": float(
                profile_value(args, "candidate_region_pre_refit_risk_local_suppression_min_bad_balanced")
            ),
            "local_suppression_positive_margin": float(
                profile_value(args, "candidate_region_pre_refit_risk_local_suppression_positive_margin")
            ),
            "local_suppression_min_face_pixels": int(
                profile_value(args, "candidate_region_pre_refit_risk_local_suppression_min_face_pixels")
            ),
            "local_suppression_max_faces_per_bad_row": int(
                profile_value(args, "candidate_region_pre_refit_risk_local_suppression_max_faces_per_bad_row")
            ),
        },
    }
    if not bool(profile_value(args, "candidate_region_pre_refit_risk_shrink")):
        return base_report
    if bool(getattr(args, "dry_run", False)):
        return {**base_report, "status": "dry_run"}

    carrier_payload = load_json(carrier_path)
    objective_payload = load_json(objective_path)
    plan_payload = load_json(plan_path)
    carrier_faces = _candidate_region_carrier_face_map(carrier_payload)
    carrier_face_meta = _candidate_region_carrier_face_meta(carrier_payload)
    rows = objective_payload.get("rows", []) if isinstance(objective_payload, dict) else []
    if face_scope is None:
        plan_faces = candidate_plan_face_ids(plan_path)
        face_scope_source = "plan_faces"
    else:
        plan_faces = {int(fid) for fid in face_scope}
        face_scope_source = "explicit_face_scope"
    if not carrier_faces or not isinstance(rows, list) or not rows or not plan_faces:
        report = {
            **base_report,
            "status": "skipped_missing_inputs",
            "carrier_count": len(carrier_faces),
            "objective_row_count": len(rows) if isinstance(rows, list) else 0,
            "plan_face_count": len(plan_faces),
            "face_scope_source": face_scope_source,
            "shrink_face_count": 0,
        }
        _write_pre_refit_risk_shrink_report(args, scene, report, purpose=alpha_purpose)
        return report

    rows_by_carrier: dict[str, list[dict[str, Any]]] = {}
    missing_carrier_rows = 0
    unknown_carrier_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        carrier_id = str(row.get("carrier_id", "")).strip()
        if not carrier_id:
            missing_carrier_rows += 1
            continue
        if carrier_id not in carrier_faces:
            unknown_carrier_rows += 1
            continue
        rows_by_carrier.setdefault(carrier_id, []).append(row)

    face_alphas: dict[str, float] = {}
    carrier_alphas: dict[str, float] = {}
    carrier_reports: list[dict[str, Any]] = []
    local_suppression_reports: list[dict[str, Any]] = []
    local_suppressed_faces_all: set[int] = set()
    local_preserved_faces_all: set[int] = set()
    local_suppression_enabled = bool(profile_value(args, "candidate_region_pre_refit_risk_local_suppression"))
    local_suppression_scale = float(profile_value(args, "candidate_region_pre_refit_risk_local_suppression_scale"))
    local_bad_balanced = float(profile_value(args, "candidate_region_pre_refit_risk_local_suppression_min_bad_balanced"))
    local_positive_margin = float(
        profile_value(args, "candidate_region_pre_refit_risk_local_suppression_positive_margin")
    )
    local_min_face_pixels = int(profile_value(args, "candidate_region_pre_refit_risk_local_suppression_min_face_pixels"))
    local_max_faces_per_bad_row = int(
        profile_value(args, "candidate_region_pre_refit_risk_local_suppression_max_faces_per_bad_row")
    )
    for carrier_id, carrier_rows in sorted(rows_by_carrier.items()):
        evaluable_rows = [row for row in carrier_rows if _pre_refit_objective_row_is_evaluable(row)]
        bad_rows = [row for row in evaluable_rows if _pre_refit_objective_row_is_bad(args, row)]
        changed_count = len(evaluable_rows)
        bad_count = len(bad_rows)
        bad_fraction = (bad_count / changed_count) if changed_count else 0.0
        balanced_values = [
            value
            for value in (_float_or_none(row.get("core_balanced_delta")) for row in evaluable_rows)
            if value is not None
        ]
        scale, risk_details = _candidate_region_pre_refit_risk_shrink_decision(
            args,
            bad_fraction,
            balanced_values,
        )
        fraction_selected = (
            changed_count >= int(profile_value(args, "candidate_region_pre_refit_risk_min_changed_rows"))
            and bad_count >= int(profile_value(args, "candidate_region_pre_refit_risk_min_bad_rows"))
            and bad_fraction > float(profile_value(args, "candidate_region_pre_refit_risk_max_bad_fraction"))
        )
        severity_selected = (
            bool(profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_aware"))
            and changed_count >= int(profile_value(args, "candidate_region_pre_refit_risk_min_changed_rows"))
            and bad_count >= int(profile_value(args, "candidate_region_pre_refit_risk_min_bad_rows"))
            and float(risk_details.get("severity", 0.0))
            >= float(profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_select_min"))
        )
        selected_for_shrink = fraction_selected or severity_selected
        if not selected_for_shrink:
            scale = 1.0
        if selected_for_shrink:
            carrier_alphas[carrier_id] = scale
            for face_id in carrier_faces.get(carrier_id, set()):
                if face_id not in plan_faces:
                    continue
                key = str(int(face_id))
                face_alphas[key] = min(float(face_alphas.get(key, 1.0)), float(scale))
        local_suppressed_faces: set[int] = set()
        local_preserved_faces: set[int] = set()
        local_bad_row_reports: list[dict[str, Any]] = []
        if local_suppression_enabled:
            positive_views = {
                str(row.get("view", "")).strip()
                for row in evaluable_rows
                if (_float_or_none(row.get("core_balanced_delta")) or -math.inf) >= local_positive_margin
            }
            positive_views.discard("")
            meta_by_face = carrier_face_meta.get(carrier_id, {})
            for row_index, row in enumerate(evaluable_rows):
                balanced = _float_or_none(row.get("core_balanced_delta"))
                if balanced is None or balanced > -local_bad_balanced:
                    continue
                if not _pre_refit_objective_row_is_bad(args, row):
                    continue
                row_view = str(row.get("view", "")).strip()
                skipped_reason = ""
                if not row_view:
                    skipped_reason = "missing_row_view"
                if not meta_by_face and not skipped_reason:
                    skipped_reason = "missing_carrier_face_metadata"
                candidates: list[tuple[float, int]] = []
                if not skipped_reason:
                    for face_id, meta in meta_by_face.items():
                        if face_id not in plan_faces:
                            continue
                        views = meta.get("views", set())
                        if not views or row_view not in views:
                            continue
                        pixels = float(meta.get("pixels", 0.0))
                        if pixels < local_min_face_pixels:
                            continue
                        candidates.append((pixels, int(face_id)))
                    if not candidates:
                        skipped_reason = "no_view_pixel_matched_faces"
                selected_faces: list[int] = []
                preserved_faces_for_row: list[int] = []
                severity_override_faces_for_row: list[int] = []
                severe_bad = balanced <= -max(local_bad_balanced + local_positive_margin, local_bad_balanced * 2.0)
                if not skipped_reason:
                    for _, face_id in sorted(candidates, reverse=True)[:local_max_faces_per_bad_row]:
                        meta = meta_by_face.get(face_id, {})
                        views = meta.get("views", set())
                        witness_views = sorted((set(views) & positive_views) - {row_view})
                        has_positive_witness = bool(witness_views)
                        if has_positive_witness and not severe_bad:
                            local_preserved_faces.add(face_id)
                            preserved_faces_for_row.append(face_id)
                            continue
                        if has_positive_witness and severe_bad:
                            severity_override_faces_for_row.append(face_id)
                        selected_faces.append(face_id)
                        local_suppressed_faces.add(face_id)
                local_bad_row_reports.append(
                    {
                        "row_index": row_index,
                        "view": row_view,
                        "bbox_xyxy": row.get("bbox_xyxy"),
                        "core_balanced_delta": balanced,
                        "delta_core_psnr": _float_or_none(row.get("delta_core_psnr")),
                        "delta_core_ssim": _float_or_none(row.get("delta_core_ssim")),
                        "delta_core_lpips": _float_or_none(row.get("delta_core_lpips")),
                        "candidate_face_count": len(candidates),
                        "skipped_reason": skipped_reason,
                        "severe_bad_override_positive_witness": severe_bad,
                        "positive_witness_preserved_face_count": len(preserved_faces_for_row),
                        "positive_witness_preserved_face_ids": preserved_faces_for_row,
                        "positive_witness_overridden_face_count": len(severity_override_faces_for_row),
                        "positive_witness_overridden_face_ids": severity_override_faces_for_row,
                        "selected_face_count": len(selected_faces),
                        "selected_face_ids": selected_faces,
                    }
                )
            for face_id in local_suppressed_faces:
                key = str(int(face_id))
                face_alphas[key] = min(float(face_alphas.get(key, 1.0)), local_suppression_scale)
            local_suppressed_faces_all.update(local_suppressed_faces)
            local_preserved_faces_all.update(local_preserved_faces)
            if local_bad_row_reports or local_suppressed_faces:
                local_suppression_reports.append(
                    {
                        "carrier_id": carrier_id,
                        "positive_witness_views": sorted(positive_views),
                        "bad_region_count": len(local_bad_row_reports),
                        "suppressed_face_count": len(local_suppressed_faces),
                        "suppressed_face_ids": sorted(local_suppressed_faces),
                        "positive_witness_preserved_face_count": len(local_preserved_faces),
                        "positive_witness_preserved_face_ids": sorted(local_preserved_faces),
                        "suppressed_rows": local_bad_row_reports,
                    }
                )
        carrier_reports.append(
            {
                "carrier_id": carrier_id,
                "row_count": len(carrier_rows),
                "evaluable_row_count": changed_count,
                "bad_row_count": bad_count,
                "bad_fraction": bad_fraction,
                "mean_core_balanced_delta": (sum(balanced_values) / len(balanced_values))
                if balanced_values
                else None,
                "worst_core_balanced_delta": min(balanced_values) if balanced_values else None,
                "face_count": len(carrier_faces.get(carrier_id, set())),
                "plan_face_count": len([fid for fid in carrier_faces.get(carrier_id, set()) if fid in plan_faces]),
                "selected_for_shrink": selected_for_shrink,
                "selected_by_bad_fraction": fraction_selected,
                "selected_by_severity": severity_selected,
                "shrink_scale": scale,
                "risk": risk_details,
                "local_suppressed_face_count": len(local_suppressed_faces),
                "local_positive_witness_preserved_face_count": len(local_preserved_faces),
            }
        )

    risk_face_count = len(face_alphas)
    inherited_alpha_path = (
        _plan_materialize_alpha_path_from_payload(plan_payload, plan_path)
        if isinstance(plan_payload, dict) and bool(plan_payload)
        else None
    )
    inherited_face_alphas = _read_face_alphas(inherited_alpha_path) if risk_face_count else {}
    if risk_face_count:
        for face_id, alpha in inherited_face_alphas.items():
            try:
                fid = int(face_id)
            except Exception:
                continue
            if fid not in plan_faces:
                continue
            face_alphas[str(fid)] = min(float(face_alphas.get(str(fid), 1.0)), float(alpha))
    alpha_required = bool(risk_face_count)
    report = {
        **base_report,
        "status": "applied" if alpha_required else "no_risky_plan_faces",
        "carrier_count": len(carrier_faces),
        "objective_row_count": len([row for row in rows if isinstance(row, dict)]),
        "mapped_objective_carrier_count": len(rows_by_carrier),
        "missing_carrier_rows": missing_carrier_rows,
        "unknown_carrier_rows": unknown_carrier_rows,
        "plan_face_count": len(plan_faces),
        "face_scope_source": face_scope_source,
        "shrunk_carrier_count": len(carrier_alphas),
        "shrunk_face_count": risk_face_count,
        "local_suppression_enabled": local_suppression_enabled,
        "local_suppressed_face_count": len(local_suppressed_faces_all),
        "local_positive_witness_preserved_face_count": len(local_preserved_faces_all),
        "local_suppression_reports": local_suppression_reports,
        "combined_alpha_face_count": len(face_alphas) if alpha_required else 0,
        "inherited_alpha_json": str(inherited_alpha_path) if inherited_alpha_path else "",
        "inherited_alpha_face_count": len(inherited_face_alphas),
        "carrier_alphas": carrier_alphas,
        "carrier_reports": carrier_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if alpha_required:
        alpha_path.parent.mkdir(parents=True, exist_ok=True)
        alpha_path.write_text(
            json.dumps(
                {
                    "operator": "candidate_region_pre_refit_risk_carrier_shrink",
                    "policy": "train_only_candidate_owned_render_region_per_carrier_monotone_alpha_shrink",
                    "test_usage": "none",
                    "selection_uses_test": False,
                    "scene": scene,
                    "alpha_purpose": alpha_purpose,
                    "source_objective": str(objective_path),
                    "source_carrier_json": str(carrier_path),
                    "source_plan": str(plan_path) if bool(plan_payload) else "",
                    "face_scope_source": face_scope_source,
                    "inherited_alpha_json": str(inherited_alpha_path) if inherited_alpha_path else "",
                    "min_scale": float(profile_value(args, "candidate_region_pre_refit_risk_shrink_min_scale")),
                    "severity_aware": bool(
                        profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_aware")
                    ),
                    "severity_select_min": float(
                        profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_select_min")
                    ),
                    "severity_balanced_span": float(
                        profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_balanced_span")
                    ),
                    "tail_fraction": float(profile_value(args, "candidate_region_pre_refit_risk_shrink_tail_fraction")),
                    "local_suppression_enabled": local_suppression_enabled,
                    "local_suppression_scale": local_suppression_scale,
                    "local_suppressed_face_count": len(local_suppressed_faces_all),
                    "face_alphas": face_alphas,
                    "carrier_alphas": carrier_alphas,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    report["alpha_sha256"] = file_sha256(alpha_path) if alpha_required else ""
    report["report_purpose"] = alpha_purpose
    if attach_to_plan and isinstance(plan_payload, dict) and bool(plan_payload):
        history = plan_payload.get("pre_refit_risk_shrink_history")
        if not isinstance(history, list):
            history = []
        contract = {
            "scene": scene,
            "test_usage": "none",
            "selection_uses_test": False,
            "policy": "train_only_candidate_owned_render_region_per_carrier_monotone_alpha_shrink",
            "source_objective": str(objective_path),
            "source_carrier_json": str(carrier_path),
            "materialize_alpha_json": str(alpha_path) if alpha_required else "",
            "materialize_alpha_sha256": file_sha256(alpha_path) if alpha_required else "",
            "alpha_purpose": alpha_purpose,
            "materialize_alpha_face_count": len(face_alphas) if alpha_required else 0,
            "shrunk_carrier_count": len(carrier_alphas),
            "shrunk_face_count": risk_face_count,
            "local_suppression_enabled": local_suppression_enabled,
            "local_suppressed_face_count": len(local_suppressed_faces_all),
            "inherited_alpha_json": str(inherited_alpha_path) if inherited_alpha_path else "",
            "inherited_alpha_face_count": len(inherited_face_alphas),
            "min_scale": float(profile_value(args, "candidate_region_pre_refit_risk_shrink_min_scale")),
            "severity_aware": bool(profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_aware")),
            "severity_select_min": float(
                profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_select_min")
            ),
            "severity_balanced_span": float(
                profile_value(args, "candidate_region_pre_refit_risk_shrink_severity_balanced_span")
            ),
            "tail_fraction": float(profile_value(args, "candidate_region_pre_refit_risk_shrink_tail_fraction")),
            "enforcement": (
                "selector strict replay reads render_cvar_aggregate_subset_materialize_alpha_json "
                "and passes it to --delta_facelocal_materialize_plan_alpha_json"
            ),
        }
        update = {
            "pre_refit_risk_shrink": contract,
            "pre_refit_risk_shrink_history": history + [contract],
        }
        if alpha_required:
            update["render_cvar_aggregate_subset_materialize_alpha_json"] = str(alpha_path)
        plan_payload.update(update)
        plan_path.write_text(json.dumps(plan_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def candidate_owned_refit_allowed_face_ids(args: argparse.Namespace, scene: str) -> str:
    face_ids: set[int] = set()
    for path in (format_candidate_region_carrier_path(args, scene), format_plan_path(args, scene)):
        payload = load_json(path)
        if not payload:
            continue
        _collect_face_ids_from_candidate_payload(payload, face_ids)
        if face_ids:
            break
    face_ids = _apply_candidate_region_pre_refit_risk_prune(args, scene, face_ids)
    return ",".join(str(fid) for fid in sorted(face_ids))


def run_command(record: CommandRecord, args: argparse.Namespace) -> CommandRecord:
    log_path = Path(record.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + shell_join(record.command) + "\n")
        if record.skipped:
            handle.write(f"[skipped] {record.skip_reason}\n")
            record.exit_code = 0
            return record
        if bool(args.dry_run):
            handle.write("[dry_run] skipped\n")
            record.exit_code = 0
            return record
        handle.flush()
        env = os.environ.copy()
        if int(args.gpu) >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if str(args.wandb_mode).strip():
            env["WANDB_MODE"] = str(args.wandb_mode)
        proc = subprocess.run(
            record.command,
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f"\n[exit_code] {proc.returncode}\n")
    record.exit_code = int(proc.returncode)
    if int(proc.returncode) != 0 and not bool(args.continue_on_error):
        raise RuntimeError(f"{record.stage} failed for {record.scene}; see {record.log_path}")
    return record


def plan_command(args: argparse.Namespace, scene: str) -> CommandRecord:
    root = output_root(args)
    label = str(args.pipeline_label)
    plan_path = format_plan_path(args, scene)
    plan_run_root = root / "plan_generation"
    command = [
        sys.executable,
        "scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py",
        "--policy_root",
        str(args.policy_root),
        "--dataset_root",
        str(args.dataset_root),
        "--output_root",
        str(plan_run_root),
        "--evidence_root",
        str(evidence_root(args)),
        "--scenes",
        scene,
        "--iteration",
        str(args.iteration),
        "--gpu",
        str(args.gpu),
        "--outdoor_images",
        str(args.outdoor_images),
        "--indoor_images",
        str(args.indoor_images),
        "--skip_failed_views",
        "--evidence_max_views",
        str(profile_value(args, "evidence_max_views")),
        "--evidence_view_stride",
        str(profile_value(args, "evidence_view_stride")),
        "--evidence_view_offset",
        str(args.evidence_view_offset),
        "--evidence_high_error_quantile",
        str(args.evidence_high_error_quantile),
        "--delta_operator",
        "facelocal_sh1",
        "--delta_uniform_barycentric",
        "--delta_sh_degree",
        str(args.delta_sh_degree),
        "--delta_top_k",
        str(profile_value(args, "delta_top_k")),
        "--delta_min_view_hits",
        str(args.delta_min_view_hits),
        "--delta_min_consistency",
        str(args.delta_min_consistency),
        "--delta_min_pixel_count",
        str(args.delta_min_pixel_count),
        "--delta_max_samples_per_face_view",
        str(args.delta_max_samples_per_face_view),
        "--delta_max_total_samples",
        str(profile_value(args, "delta_max_total_samples")),
        "--delta_high_error_quantile",
        str(args.delta_high_error_quantile),
        "--delta_strength",
        str(profile_value(args, "delta_strength")),
        "--delta_max_abs_rgb",
        str(profile_value(args, "delta_max_abs_rgb")),
        "--delta_lambda_mag",
        str(args.delta_lambda_mag),
        "--delta_lambda_sh1_mag",
        str(args.delta_lambda_sh1_mag),
        "--delta_lambda_smooth",
        str(args.delta_lambda_smooth),
        "--delta_steps",
        str(profile_value(args, "delta_steps")),
        "--delta_shared_residual_field_anchors",
        str(profile_value(args, "delta_shared_residual_field_anchors")),
        "--delta_shared_residual_field_sigma",
        str(profile_value(args, "delta_shared_residual_field_sigma")),
        "--delta_shared_residual_field_lr",
        str(profile_value(args, "delta_shared_residual_field_lr")),
        "--delta_shared_residual_field_weight_l2",
        str(profile_value(args, "delta_shared_residual_field_weight_l2")),
        "--delta_shared_residual_field_view_hinge_weight",
        str(profile_value(args, "delta_shared_residual_field_view_hinge_weight")),
        "--delta_shared_residual_field_view_hinge_min_samples",
        str(profile_value(args, "delta_shared_residual_field_view_hinge_min_samples")),
        "--delta_shared_residual_field_duplicate_smooth_weight",
        str(profile_value(args, "delta_shared_residual_field_duplicate_smooth_weight")),
        "--delta_min_policy_val_relative_gain",
        str(args.delta_min_policy_val_relative_gain),
        "--delta_min_policy_val_samples",
        str(profile_value(args, "delta_min_policy_val_samples")),
        "--delta_min_policy_val_adaptive_sample_fraction",
        str(profile_value(args, "delta_min_policy_val_adaptive_sample_fraction")),
        "--delta_min_policy_val_adaptive_min_samples",
        str(profile_value(args, "delta_min_policy_val_adaptive_min_samples")),
        "--delta_min_policy_val_unique_faces",
        str(profile_value(args, "delta_min_policy_val_unique_faces")),
        "--delta_validation_shrink_mode",
        str(args.delta_validation_shrink_mode),
        "--delta_validation_gain_max_scale",
        str(args.delta_validation_gain_max_scale),
        "--delta_crossfold_gain_certificate_folds",
        str(profile_value(args, "delta_crossfold_folds")),
        "--delta_crossfold_min_passing_folds",
        str(profile_value(args, "delta_crossfold_min_passing_folds")),
        "--delta_crossfold_min_fold_relative_gain",
        "0.0",
        "--delta_min_face_policy_val_relative_gain",
        str(args.delta_min_face_policy_val_relative_gain),
        "--delta_min_face_policy_val_samples",
        str(args.delta_min_face_policy_val_samples),
        "--delta_min_face_view_consensus",
        str(args.delta_min_face_view_consensus),
        "--delta_min_face_consensus_views",
        str(args.delta_min_face_consensus_views),
        "--delta_min_face_gain_certificate_views",
        str(args.delta_min_face_gain_certificate_views),
        "--delta_min_face_gain_certificate_relative_gain",
        "0.0",
        "--delta_min_face_gain_certificate_fraction",
        str(args.delta_min_face_gain_certificate_fraction),
        "--delta_patch_cert_rings",
        str(args.delta_patch_cert_rings),
        "--delta_patch_cert_max_faces_per_seed",
        str(profile_value(args, "delta_patch_cert_max_faces_per_seed")),
        "--delta_patch_cert_min_neighbor_policy_val_samples",
        str(profile_value(args, "delta_patch_cert_min_neighbor_policy_val_samples")),
        "--delta_patch_cert_min_policy_val_samples",
        str(profile_value(args, "delta_patch_cert_min_policy_val_samples")),
        "--delta_patch_cert_min_relative_gain",
        str(profile_value(args, "delta_patch_cert_min_relative_gain")),
        "--delta_patch_cert_neighbor_mode",
        str(args.delta_patch_cert_neighbor_mode),
        "--delta_patch_cert_crossfold_folds",
        str(profile_value(args, "delta_crossfold_folds")),
        "--delta_patch_cert_crossfold_min_passing_folds",
        str(profile_value(args, "delta_crossfold_min_passing_folds")),
        "--delta_patch_cert_crossfold_min_fold_relative_gain",
        "0.0",
        "--delta_patch_cert_neighbor_crossfold",
        "--delta_patch_cert_cluster_basis",
        "--delta_patch_cert_cluster_basis_mode",
        str(profile_value(args, "delta_patch_cert_cluster_basis_mode")),
        "--delta_patch_cert_cluster_basis_steps",
        str(profile_value(args, "delta_patch_cert_cluster_basis_steps")),
        "--delta_patch_cert_cluster_basis_min_samples",
        str(profile_value(args, "delta_patch_cert_cluster_basis_min_samples")),
        "--delta_patch_cert_cluster_basis_max_fit_mse_regression",
        str(profile_value(args, "delta_patch_cert_cluster_basis_max_fit_mse_regression")),
        "--delta_patch_cert_cluster_basis_view_hinge_weight",
        str(profile_value(args, "delta_patch_cert_cluster_basis_view_hinge_weight")),
        "--delta_patch_cert_cluster_basis_view_hinge_min_samples",
        str(profile_value(args, "delta_patch_cert_cluster_basis_view_hinge_min_samples")),
        "--delta_patch_cert_cluster_basis_geometry_smooth_weight",
        str(profile_value(args, "delta_patch_cert_cluster_basis_geometry_smooth_weight")),
        "--delta_patch_cert_carrier_holdout_selector",
        "--delta_patch_cert_carrier_holdout_groups",
        str(profile_value(args, "delta_patch_cert_carrier_holdout_groups")),
        "--delta_patch_cert_carrier_holdout_grouping",
        "sample_balanced",
        "--delta_patch_cert_carrier_holdout_disjoint",
        "--delta_patch_cert_carrier_holdout_min_passing_groups",
        str(profile_value(args, "delta_patch_cert_carrier_holdout_min_passing_groups")),
        "--delta_patch_cert_carrier_holdout_auto_prefix",
        "--delta_patch_cert_carrier_holdout_auto_prefix_min_faces",
        str(args.delta_patch_cert_carrier_holdout_auto_prefix_min_faces),
        "--delta_patch_cert_carrier_holdout_auto_prefix_min_face_fraction",
        str(profile_value(args, "delta_patch_cert_carrier_holdout_auto_prefix_min_face_fraction")),
        "--delta_patch_cert_carrier_holdout_auto_prefix_face_bonus",
        str(profile_value(args, "delta_patch_cert_carrier_holdout_auto_prefix_face_bonus")),
        (
            "--delta_patch_cert_carrier_holdout_auto_prefix_positive_tail_safe"
            if bool(args.delta_patch_cert_carrier_holdout_auto_prefix_positive_tail_safe)
            else "--no-delta_patch_cert_carrier_holdout_auto_prefix_positive_tail_safe"
        ),
        "--delta_strict_patchcert_carrier",
        "--delta_max_faces_to_apply",
        str(profile_value(args, "delta_max_faces_to_apply")),
        (
            "--delta_candidate_region_expansion_core_priority"
            if bool(profile_value(args, "candidate_region_expansion_core_priority"))
            else "--no-delta_candidate_region_expansion_core_priority"
        ),
        "--delta_candidate_region_expansion_core_min_samples",
        str(profile_value(args, "candidate_region_expansion_core_min_samples")),
        "--delta_candidate_region_expansion_core_min_fraction",
        str(profile_value(args, "candidate_region_expansion_core_min_fraction")),
        (
            "--delta_candidate_region_expansion_witness_rescue"
            if bool(profile_value(args, "candidate_region_expansion_witness_rescue"))
            else "--no-delta_candidate_region_expansion_witness_rescue"
        ),
        "--delta_candidate_region_expansion_max_witnesses_per_carrier",
        str(profile_value(args, "candidate_region_expansion_max_witnesses_per_carrier")),
        "--delta_facelocal_candidate_plan_out",
        str(plan_path),
        "--candidate_label",
        f"{label}_plan",
        "--candidate_base_method",
        f"ours_{args.iteration}_{label}_plan_base",
        "--candidate_test_method",
        f"ours_{args.iteration}_{label}_plan_phasej_ela",
        "--candidate_trainval_method",
        f"ours_{args.iteration}_{label}_plan_trainval_gate",
        "--phasej_test_method",
        phasej_test_method(args),
        "--phasej_trainval_method",
        phasej_trainval_method(args),
        "--train_render_region_gate_enable",
        "--train_render_region_carrier_json",
        str(args.render_region_carrier_template),
        "--train_render_region_eval_source",
        "raw_base",
        "--train_render_region_max_regions",
        str(profile_value(args, "filter_train_render_region_max_regions")),
        "--train_render_region_min_pixels",
        str(profile_value(args, "filter_train_render_region_min_pixels")),
        "--train_render_region_min_crop_size",
        str(profile_value(args, "filter_train_render_region_min_crop_size")),
        "--train_render_region_context_pad",
        str(profile_value(args, "filter_train_render_region_context_pad")),
        "--train_render_region_tail_fraction",
        str(profile_value(args, "filter_train_render_region_tail_fraction")),
        "--train_render_region_min_regions",
        str(profile_value(args, "plan_region_gate_min_regions")),
        "--train_render_region_min_changed_regions",
        str(profile_value(args, "plan_region_gate_min_changed_regions")),
        "--train_render_region_min_changed_fraction",
        str(profile_value(args, "plan_region_gate_min_changed_fraction")),
        f"--train_render_region_min_core_balanced_delta={profile_value(args, 'plan_region_gate_min_core_balanced_delta')}",
        f"--train_render_region_min_core_psnr_delta={profile_value(args, 'plan_region_gate_min_core_psnr_delta')}",
        f"--train_render_region_min_tail_cvar_delta={profile_value(args, 'plan_region_gate_min_tail_cvar_delta')}",
        "--train_render_region_max_context_mse_regression",
        str(profile_value(args, "plan_region_gate_max_context_mse_regression")),
        "--train_render_region_max_negative_fraction",
        str(profile_value(args, "plan_region_gate_max_negative_fraction")),
        "--gate_min_psnr_gain",
        str(args.gate_min_psnr_gain),
        "--gate_max_ssim_regression",
        str(args.gate_max_ssim_regression),
        "--gate_max_lpips_regression",
        str(args.gate_max_lpips_regression),
        "--gate_min_balanced_delta",
        str(args.gate_min_balanced_delta),
        "--ela_alpha_risk_tail_fraction",
        str(profile_value(args, "ela_alpha_risk_tail_fraction")),
        "--ela_alpha_max_negative_gain_fraction",
        str(profile_value(args, "ela_alpha_max_negative_gain_fraction")),
        f"--ela_alpha_min_tail_gain={profile_value(args, 'ela_alpha_min_tail_gain')}",
        "--ela_alpha_view_tail_scale_grid",
        str(profile_value(args, "ela_alpha_view_tail_scale_grid")),
        "--ela_alpha_view_tail_cvar_fraction",
        str(profile_value(args, "ela_alpha_view_tail_cvar_fraction")),
        f"--ela_alpha_view_tail_min_gain={profile_value(args, 'ela_alpha_view_tail_min_gain')}",
        "--ela_alpha_view_tail_max_negative_fraction",
        str(profile_value(args, "ela_alpha_view_tail_max_negative_fraction")),
        "--ela_alpha_view_tail_objective",
        str(profile_value(args, "ela_alpha_view_tail_objective")),
        "--ela_alpha_view_tail_ssim_weight",
        str(profile_value(args, "ela_alpha_view_tail_ssim_weight")),
        "--ela_alpha_view_tail_lpips_weight",
        str(profile_value(args, "ela_alpha_view_tail_lpips_weight")),
        "--ela_alpha_view_tail_metric_max_side",
        str(profile_value(args, "ela_alpha_view_tail_metric_max_side")),
        "--ela_alpha_region_risk_min_tail_gain",
        str(profile_value(args, "ela_alpha_region_risk_min_tail_gain")),
        "--ela_alpha_region_risk_max_negative_fraction",
        str(profile_value(args, "ela_alpha_region_risk_max_negative_fraction")),
        "--ela_alpha_region_risk_min_regions",
        str(profile_value(args, "ela_alpha_region_risk_min_regions")),
        "--ela_alpha_region_risk_objective_max_balanced_delta",
        str(profile_value(args, "ela_alpha_region_risk_objective_max_balanced_delta")),
        "--ela_alpha_region_risk_objective_max_delta_ssim",
        str(profile_value(args, "ela_alpha_region_risk_objective_max_delta_ssim")),
        "--ela_alpha_region_risk_objective_min_delta_lpips",
        str(profile_value(args, "ela_alpha_region_risk_objective_min_delta_lpips")),
        "--ela_local_trust_min_supports",
        str(profile_value(args, "ela_local_trust_min_supports")),
        "--ela_local_trust_max_residual_std",
        str(profile_value(args, "ela_local_trust_max_residual_std")),
        "--ela_local_trust_min_agreement",
        str(profile_value(args, "ela_local_trust_min_agreement")),
        "--ela_local_trust_agreement_scale",
        str(profile_value(args, "ela_local_trust_agreement_scale")),
        "--ela_local_trust_confidence_quantile",
        str(profile_value(args, "ela_local_trust_confidence_quantile")),
        "--ela_local_trust_min_confidence",
        str(profile_value(args, "ela_local_trust_min_confidence")),
        "--ela_local_trust_mode",
        str(profile_value(args, "ela_local_trust_mode")),
        "--ela_local_trust_min_weight",
        str(profile_value(args, "ela_local_trust_min_weight")),
        "--wandb_project",
        str(args.wandb_project),
        "--wandb_group",
        f"{label}_plan_generation",
        "--wandb_name",
        f"{label}_plan_{scene}",
    ]
    if bool(profile_value(args, "ela_alpha_holdout_safe_zero")):
        command.append("--ela_alpha_holdout_safe_zero")
    if bool(profile_value(args, "ela_alpha_view_tail_compute_lpips")):
        command.append("--ela_alpha_view_tail_compute_lpips")
    # The initial plan PhaseK replays Phase-J ELA before the train-render-region
    # objective JSON exists. Keep region-risk enabled for candidate-owned refit
    # and selector, where the JSON path is explicit, but do not pass a dangling
    # enable flag here.
    if bool(profile_value(args, "ela_local_trust_gate")):
        command.append("--ela_local_trust_gate")
    if bool(profile_value(args, "delta_shared_residual_field")):
        command.append("--delta_shared_residual_field")
    else:
        command.append("--no-delta_shared_residual_field")
    if bool(profile_value(args, "delta_render_region_objective")):
        command.extend(
            [
                "--delta_facelocal_region_carrier_json",
                str(args.render_region_carrier_template),
                "--delta_region_core_weight",
                str(profile_value(args, "delta_region_core_weight")),
                "--delta_region_context_weight",
                str(profile_value(args, "delta_region_context_weight")),
                "--delta_region_outside_weight",
                str(profile_value(args, "delta_region_outside_weight")),
                "--delta_region_boundary_px",
                str(profile_value(args, "delta_region_boundary_px")),
                "--delta_render_region_objective",
                "--delta_render_region_core_weight",
                str(profile_value(args, "delta_render_region_core_weight")),
                "--delta_render_region_context_weight",
                str(profile_value(args, "delta_render_region_context_weight")),
                "--delta_render_region_outside_penalty",
                str(profile_value(args, "delta_render_region_outside_penalty")),
                "--delta_render_region_tail_cvar_weight",
                str(profile_value(args, "delta_render_region_tail_cvar_weight")),
                "--delta_render_region_tail_fraction",
                str(profile_value(args, "delta_render_region_tail_fraction")),
                "--delta_render_region_min_view_samples",
                str(profile_value(args, "delta_render_region_min_view_samples")),
                "--delta_bystander_zero_delta_weight",
                str(profile_value(args, "delta_bystander_zero_delta_weight")),
                (
                    "--delta_bystander_zero_delta_include_context"
                    if bool(profile_value(args, "delta_bystander_zero_delta_include_context"))
                    else "--no-delta_bystander_zero_delta_include_context"
                ),
                "--delta_bystander_zero_delta_min_samples",
                str(profile_value(args, "delta_bystander_zero_delta_min_samples")),
                "--delta_witness_constraint_weight",
                str(profile_value(args, "delta_witness_constraint_weight")),
                "--delta_witness_constraint_tail_fraction",
                str(profile_value(args, "delta_witness_constraint_tail_fraction")),
                "--delta_witness_constraint_min_samples",
                str(profile_value(args, "delta_witness_constraint_min_samples")),
                "--delta_witness_constraint_margin",
                str(profile_value(args, "delta_witness_constraint_margin")),
                (
                    "--delta_witness_constraint_include_full_view"
                    if bool(profile_value(args, "delta_witness_constraint_include_full_view"))
                    else "--no-delta_witness_constraint_include_full_view"
                ),
                (
                    "--delta_witness_constraint_include_region_view"
                    if bool(profile_value(args, "delta_witness_constraint_include_region_view"))
                    else "--no-delta_witness_constraint_include_region_view"
                ),
                (
                    "--delta_witness_constraint_include_bystander_view"
                    if bool(profile_value(args, "delta_witness_constraint_include_bystander_view"))
                    else "--no-delta_witness_constraint_include_bystander_view"
                ),
            ]
        )
    else:
        command.append("--no-delta_render_region_objective")
    if bool(profile_value(args, "filter_train_render_region_skip_lpips")):
        command.append("--train_render_region_skip_lpips")
    else:
        command.append("--no-train_render_region_skip_lpips")
    if not bool(args.no_seed_rescue):
        command.extend(
            [
                "--delta_patch_cert_seed_rescue",
                "--delta_patch_cert_seed_rescue_min_candidates",
                "1",
                "--delta_patch_cert_seed_rescue_max_seeds",
                "16",
                "--delta_patch_cert_seed_rescue_min_aux_witnesses",
                "1",
            ]
        )
    if bool(args.force):
        command.append("--force")
    return CommandRecord(
        stage="plan",
        scene=scene,
        command=command,
        log_path=str(command_log(root, "plan", scene)),
        output_paths={
            "candidate_plan": str(plan_path),
            "render_region_carrier_json": str(format_region_carrier_path(args, scene)),
            "train_render_region_objective_raw_base": str(plan_run_root / scene / "train_render_region_objective_raw_base.json"),
            "evidence_dir": str(evidence_root(args) / scene),
            "plan_generation_root": str(plan_run_root / scene),
        },
    )


def candidate_region_command(args: argparse.Namespace, scene: str) -> CommandRecord:
    root = output_root(args)
    out_root = format_candidate_region_root(args, scene)
    carrier_path = format_candidate_region_carrier_path(args, scene)
    md_path = out_root / "candidate_render_regions.md"
    command = [
        sys.executable,
        "scripts/car_model/ecsr_build_candidate_plan_render_regions.py",
        "--scene",
        scene,
        "--candidate_plan",
        str(format_plan_path(args, scene)),
        "--evidence_dir",
        str(evidence_root(args) / scene),
        "--output_json",
        str(carrier_path),
        "--output_md",
        str(md_path),
        "--max_regions_per_carrier",
        str(profile_value(args, "filter_candidate_region_max_regions_per_carrier")),
        "--min_pixels",
        str(profile_value(args, "filter_candidate_region_min_pixels")),
        "--bbox_pad",
        str(profile_value(args, "filter_candidate_region_bbox_pad")),
        "--min_alpha",
        str(profile_value(args, "filter_candidate_region_min_alpha")),
        "--high_error_quantile",
        str(profile_value(args, "filter_candidate_region_high_error_quantile")),
        "--max_views",
        str(profile_value(args, "filter_candidate_region_max_views")),
        "--expand_min_face_pixels",
        str(profile_value(args, "filter_candidate_region_expand_min_face_pixels")),
        "--expand_min_face_views",
        str(profile_value(args, "filter_candidate_region_expand_min_face_views")),
        "--expand_max_faces_per_carrier",
        str(profile_value(args, "filter_candidate_region_expand_max_faces_per_carrier")),
        "--min_frame_support_fraction",
        str(profile_value(args, "filter_candidate_region_min_frame_support_fraction")),
        "--min_residual_mass_fraction",
        str(profile_value(args, "filter_candidate_region_min_residual_mass_fraction")),
        "--max_carriers",
        str(profile_value(args, "filter_candidate_region_max_carriers")),
    ]
    command.append(
        "--expand_faces_in_region_bbox"
        if bool(profile_value(args, "filter_candidate_region_expand_faces"))
        else "--no-expand_faces_in_region_bbox"
    )
    command.append(
        "--frame_aware_ranking"
        if bool(profile_value(args, "filter_candidate_region_frame_aware_ranking"))
        else "--no-frame_aware_ranking"
    )
    return CommandRecord(
        stage="candidate_regions",
        scene=scene,
        command=command,
        log_path=str(command_log(root, "candidate_regions", scene)),
        output_paths={
            "candidate_plan": str(format_plan_path(args, scene)),
            "evidence_dir": str(evidence_root(args) / scene),
            "candidate_region_carrier_json": str(carrier_path),
            "candidate_region_carrier_md": str(md_path),
        },
    )


def candidate_region_eval_command(args: argparse.Namespace, scene: str, *, refit: bool = False) -> CommandRecord:
    root = output_root(args)
    label = str(args.pipeline_label)
    out_root = format_candidate_region_root(args, scene)
    if refit:
        out_json = out_root / "train_render_region_objective_refit_base.json"
        out_md = out_root / "train_render_region_objective_refit_base.md"
        candidate_model = root / "candidate_owned_refit" / scene / "model"
        candidate_method = f"ours_{args.iteration}_{label}_candidate_owned_refit_base"
        stage = "candidate_region_eval_refit"
    else:
        out_json = format_candidate_region_objective_path(args, scene)
        out_md = out_root / "train_render_region_objective_raw_base.md"
        candidate_model = root / "plan_generation" / scene / "model"
        candidate_method = f"ours_{args.iteration}_{label}_plan_base"
        stage = "candidate_region_eval"
    phasej_model = selected_policy_model(args, scene)
    command = [
        sys.executable,
        "scripts/car_model/ecsr_eval_train_render_region_objective.py",
        "--scene",
        scene,
        "--carrier_json",
        str(format_candidate_region_carrier_path(args, scene)),
        "--baseline_dir",
        str(phasej_model / "train" / DEFAULT_BASE_METHOD),
        "--candidate_dir",
        str(candidate_model / "train" / candidate_method),
        "--baseline_label",
        DEFAULT_BASE_METHOD,
        "--candidate_label",
        candidate_method,
        "--output_json",
        str(out_json),
        "--output_md",
        str(out_md),
        "--max_regions",
        str(profile_value(args, "filter_train_render_region_max_regions")),
        "--min_region_pixels",
        str(profile_value(args, "filter_train_render_region_min_pixels")),
        "--min_crop_size",
        str(profile_value(args, "filter_train_render_region_min_crop_size")),
        "--context_pad",
        str(profile_value(args, "filter_train_render_region_context_pad")),
        "--tail_fraction",
        str(profile_value(args, "filter_train_render_region_tail_fraction")),
        "--ssim_weight",
        str(args.gate_ssim_weight if hasattr(args, "gate_ssim_weight") else 20.0),
        "--lpips_weight",
        str(args.gate_lpips_weight if hasattr(args, "gate_lpips_weight") else 20.0),
        "--device",
        "cuda",
    ]
    if bool(profile_value(args, "filter_train_render_region_skip_lpips")):
        command.append("--skip_lpips")
    return CommandRecord(
        stage=stage,
        scene=scene,
        command=command,
        log_path=str(command_log(root, stage, scene)),
        output_paths={
            "candidate_region_carrier_json": str(format_candidate_region_carrier_path(args, scene)),
            "candidate_region_objective": str(out_json),
            "candidate_region_objective_md": str(out_md),
            "baseline_dir": str(phasej_model / "train" / DEFAULT_BASE_METHOD),
            "candidate_dir": str(candidate_model / "train" / candidate_method),
        },
    )


def candidate_owned_refit_command(args: argparse.Namespace, scene: str) -> CommandRecord:
    root = output_root(args)
    label = str(args.pipeline_label)
    record = plan_command(args, scene)
    command = list(record.command)
    refit_root = root / "candidate_owned_refit"
    refit_plan = format_refit_plan_path(args, scene)
    candidate_method = f"ours_{args.iteration}_{label}_candidate_owned_refit_base"
    candidate_trainval_method = f"ours_{args.iteration}_{label}_candidate_owned_refit_trainval_gate"
    candidate_test_method = f"ours_{args.iteration}_{label}_candidate_owned_refit_phasej_ela"
    candidate_region_carrier = format_candidate_region_carrier_path(args, scene)
    replace_arg(command, "--output_root", str(refit_root))
    replace_arg(command, "--delta_facelocal_candidate_plan_out", str(refit_plan))
    replace_arg(command, "--candidate_label", f"{label}_candidate_owned_refit")
    replace_arg(command, "--candidate_base_method", candidate_method)
    replace_arg(command, "--candidate_test_method", candidate_test_method)
    replace_arg(command, "--candidate_trainval_method", candidate_trainval_method)
    replace_arg(command, "--train_render_region_carrier_json", str(candidate_region_carrier))
    append_or_replace_arg(command, "--delta_facelocal_region_carrier_json", str(candidate_region_carrier))
    if bool(profile_value(args, "ela_alpha_region_risk_enable")):
        if "--ela_alpha_region_risk_enable" not in command:
            command.append("--ela_alpha_region_risk_enable")
        append_or_replace_arg(command, "--ela_alpha_region_risk_json", str(format_candidate_region_objective_path(args, scene)))
    allowed_face_ids = candidate_owned_refit_allowed_face_ids(args, scene)
    if allowed_face_ids:
        append_or_replace_arg(command, "--delta_facelocal_allowed_face_ids", allowed_face_ids)
    allowed_face_set = {int(item) for item in allowed_face_ids.split(",") if item.strip()} if allowed_face_ids else set()
    risk_scale_json = ""
    if bool(profile_value(args, "candidate_region_pre_refit_risk_shrink")) and allowed_face_set:
        shrink_report = apply_candidate_region_pre_refit_risk_shrink_policy(
            args,
            scene,
            refit_plan,
            face_scope=allowed_face_set,
            attach_to_plan=False,
            alpha_purpose="refit",
        )
        alpha_json = str(shrink_report.get("alpha_json", "")).strip()
        if str(shrink_report.get("status", "")) == "applied" and alpha_json and Path(alpha_json).is_file():
            risk_scale_json = alpha_json
            append_or_replace_arg(command, "--delta_facelocal_face_risk_scale_json", risk_scale_json)
    if bool(profile_value(args, "candidate_region_expansion_closure")):
        append_or_replace_arg(
            command,
            "--delta_facelocal_candidate_region_expansion_carrier_json",
            str(candidate_region_carrier),
        )
    append_or_replace_arg(
        command,
        "--delta_candidate_region_expansion_core_min_samples",
        str(profile_value(args, "candidate_region_expansion_core_min_samples")),
    )
    append_or_replace_arg(
        command,
        "--delta_candidate_region_expansion_core_min_fraction",
        str(profile_value(args, "candidate_region_expansion_core_min_fraction")),
    )
    append_or_replace_arg(
        command,
        "--delta_candidate_region_expansion_max_witnesses_per_carrier",
        str(profile_value(args, "candidate_region_expansion_max_witnesses_per_carrier")),
    )
    command.append(
        "--delta_candidate_region_expansion_core_priority"
        if bool(profile_value(args, "candidate_region_expansion_core_priority"))
        else "--no-delta_candidate_region_expansion_core_priority"
    )
    command.append(
        "--delta_candidate_region_expansion_witness_rescue"
        if bool(profile_value(args, "candidate_region_expansion_witness_rescue"))
        else "--no-delta_candidate_region_expansion_witness_rescue"
    )
    replace_arg(command, "--wandb_group", f"{label}_candidate_owned_refit")
    replace_arg(command, "--wandb_name", f"{label}_candidate_owned_refit_{scene}")
    return CommandRecord(
        stage="candidate_owned_refit",
        scene=scene,
        command=command,
        log_path=str(command_log(root, "candidate_owned_refit", scene)),
        output_paths={
            "seed_candidate_plan": str(format_plan_path(args, scene)),
            "candidate_region_carrier_json": str(candidate_region_carrier),
            "allowed_face_ids": allowed_face_ids,
            "pre_refit_risk_prune_report": str(format_pre_refit_risk_prune_report_path(args, scene)),
            "pre_refit_risk_shrink_report": str(
                format_pre_refit_risk_shrink_report_path(args, scene, purpose="refit")
            ),
            "pre_refit_risk_shrink_report_refit": str(
                format_pre_refit_risk_shrink_report_path(args, scene, purpose="refit")
            ),
            "selector_risk_shrink_report": str(
                format_pre_refit_risk_shrink_report_path(args, scene, purpose="selector")
            ),
            "legacy_risk_shrink_report": str(
                format_pre_refit_risk_shrink_report_path(args, scene, purpose="legacy")
            ),
            "pre_refit_risk_shrink_alpha_json": str(
                format_pre_refit_risk_shrink_alpha_path(args, scene, purpose="refit")
            ),
            "selector_risk_shrink_alpha_json": str(
                format_pre_refit_risk_shrink_alpha_path(args, scene, purpose="selector")
            ),
            "pre_refit_face_risk_scale_json": risk_scale_json,
            "refit_candidate_plan": str(refit_plan),
            "evidence_dir": str(evidence_root(args) / scene),
            "refit_generation_root": str(refit_root / scene),
        },
    )


def format_candidate_owned_refit_audit_path(args: argparse.Namespace, scene: str) -> Path:
    return output_root(args) / "candidate_owned_refit" / scene / "model" / "surface_residual_facelocal_sh1_delta_audit.json"


def assert_refit_risk_scale_audit(args: argparse.Namespace, scene: str, risk_scale_json: str) -> None:
    if bool(args.dry_run) or not str(risk_scale_json).strip():
        return
    risk_path = Path(str(risk_scale_json))
    if not risk_path.is_file():
        raise RuntimeError(f"risk scale JSON missing after refit planning for {scene}: {risk_path}")
    audit_path = format_candidate_owned_refit_audit_path(args, scene)
    audit = load_json(audit_path)
    if not audit:
        raise RuntimeError(f"missing facelocal audit for risk-scale refit {scene}: {audit_path}")
    face_risk = audit.get("face_risk_scale") if isinstance(audit, dict) else None
    if not isinstance(face_risk, dict) or not bool(face_risk.get("enabled", False)):
        raise RuntimeError(f"facelocal audit did not enable face_risk_scale for {scene}: {audit_path}")
    audit_path_value = str(face_risk.get("path", "")).strip()
    if audit_path_value and Path(audit_path_value) != risk_path:
        raise RuntimeError(
            f"facelocal audit risk-scale path mismatch for {scene}: expected {risk_path}, got {audit_path_value}"
        )
    input_faces = int(face_risk.get("input_scale_faces", 0) or 0)
    matched_faces = int(face_risk.get("matched_faces", 0) or 0)
    affected_rows = int(face_risk.get("affected_coeff_rows", 0) or 0)
    if input_faces <= 0 or matched_faces <= 0 or affected_rows <= 0:
        raise RuntimeError(
            "facelocal risk-scale audit indicates no effective scaled coefficients "
            f"for {scene}: input_faces={input_faces}, matched_faces={matched_faces}, affected_rows={affected_rows}"
        )


def candidate_region_objective_for_filter(args: argparse.Namespace, scene: str) -> Path:
    if use_refit_plan_for_filter(args, scene):
        return format_candidate_region_root(args, scene) / "train_render_region_objective_refit_base.json"
    return format_candidate_region_objective_path(args, scene)


def filter_command(args: argparse.Namespace, scene: str) -> CommandRecord:
    root = output_root(args)
    plan_path = format_filter_input_plan_path(args, scene)
    filtered_path = format_filtered_plan_path(args, scene)
    summary_path = filtered_path.with_name("filter_summary.json")
    md_path = filtered_path.with_name("filter_summary.md")
    if str(profile_value(args, "filter_region_source")) == "candidate_owned":
        objective_path = candidate_region_objective_for_filter(args, scene)
        carrier_path = format_candidate_region_carrier_path(args, scene)
    else:
        objective_path = root / "plan_generation" / scene / "train_render_region_objective_raw_base.json"
        carrier_path = format_region_carrier_path(args, scene)
    command = [
        sys.executable,
        "scripts/car_model/ecsr_filter_facelocal_plan_by_render_region.py",
        "--scene",
        scene,
        "--candidate_plan",
        str(plan_path),
        "--render_region_objective",
        str(objective_path),
        "--carrier_json",
        str(carrier_path),
        "--output_plan",
        str(filtered_path),
        "--output_md",
        str(md_path),
        "--output_json",
        str(summary_path),
        "--max_region_matches_per_plan_carrier",
        str(profile_value(args, "filter_max_region_matches_per_plan_carrier")),
        "--min_regions",
        str(profile_value(args, "filter_min_regions")),
        "--min_changed_regions",
        str(profile_value(args, "filter_min_changed_regions")),
        "--min_changed_fraction",
        str(profile_value(args, "filter_min_changed_fraction")),
        f"--min_mean_core_balanced_delta={profile_value(args, 'filter_min_mean_core_balanced_delta')}",
        f"--min_mean_delta_psnr={profile_value(args, 'filter_min_mean_delta_psnr')}",
        f"--min_tail_core_balanced_delta={profile_value(args, 'filter_min_tail_core_balanced_delta')}",
        f"--max_negative_core_balanced_fraction={profile_value(args, 'filter_max_negative_core_balanced_fraction')}",
        "--max_context_mse_regression",
        str(profile_value(args, "filter_max_context_mse_regression")),
        "--min_mean_crop_abs_diff",
        str(profile_value(args, "filter_min_mean_crop_abs_diff")),
        "--min_max_crop_abs_diff",
        str(profile_value(args, "filter_min_max_crop_abs_diff")),
        "--tail_safe_shrink_min_scale",
        str(profile_value(args, "filter_tail_safe_shrink_min_scale")),
        "--tail_safe_shrink_min_raw_scale",
        str(profile_value(args, "filter_tail_safe_shrink_min_raw_scale")),
        "--rollback_tail_min_cvar_loss",
        str(profile_value(args, "filter_rollback_tail_min_cvar_loss")),
        "--risk_safe_shrink_min_scale",
        str(profile_value(args, "filter_risk_safe_shrink_min_scale")),
    ]
    command.append(
        "--tail_safe_shrink_on_tail_fail"
        if bool(profile_value(args, "filter_tail_safe_shrink_on_tail_fail"))
        else "--no-tail_safe_shrink_on_tail_fail"
    )
    command.append(
        "--rollback_severe_tail_fail"
        if bool(profile_value(args, "filter_rollback_severe_tail_fail"))
        else "--no-rollback_severe_tail_fail"
    )
    command.append(
        "--risk_safe_shrink_on_train_risk_fail"
        if bool(profile_value(args, "filter_risk_safe_shrink_on_train_risk_fail"))
        else "--no-risk_safe_shrink_on_train_risk_fail"
    )
    command.append("--drop_unmapped" if bool(profile_value(args, "filter_drop_unmapped")) else "--no-drop_unmapped")
    command.append(
        "--require_positive_plan_proxy"
        if bool(profile_value(args, "filter_require_positive_plan_proxy"))
        else "--no-require_positive_plan_proxy"
    )
    return CommandRecord(
        stage="filter",
        scene=scene,
        command=command,
        log_path=str(command_log(root, "filter", scene)),
        output_paths={
            "filter_input_plan": str(plan_path),
            "raw_candidate_plan": str(format_plan_path(args, scene)),
            "refit_candidate_plan": str(format_refit_plan_path(args, scene)) if bool(profile_value(args, "candidate_owned_refit")) else "",
            "filtered_candidate_plan": str(filtered_path),
            "filter_summary": str(summary_path),
            "filter_summary_md": str(md_path),
            "render_region_objective": str(objective_path),
            "render_region_carrier_json": str(carrier_path),
            "filter_region_source": str(profile_value(args, "filter_region_source")),
        },
    )


def aggregate_subset_command(args: argparse.Namespace, scene: str) -> CommandRecord:
    root = output_root(args)
    input_plan = format_filtered_plan_path(args, scene)
    output_plan = format_aggregate_subset_plan_path(args, scene)
    summary_path = output_plan.with_name("aggregate_subset_summary.json")
    md_path = output_plan.with_name("aggregate_subset_summary.md")
    if str(profile_value(args, "filter_region_source")) == "candidate_owned":
        objective_path = candidate_region_objective_for_filter(args, scene)
    else:
        objective_path = root / "plan_generation" / scene / "train_render_region_objective_raw_base.json"
    command = [
        sys.executable,
        "scripts/car_model/ecsr_apply_render_cvar_aggregate_subset.py",
        "--scene",
        scene,
        "--input_plan",
        str(input_plan),
        "--render_region_objective",
        str(objective_path),
        "--output_plan",
        str(output_plan),
        "--output_json",
        str(summary_path),
        "--output_md",
        str(md_path),
        "--min_regions",
        str(profile_value(args, "filter_min_regions")),
        "--min_changed_regions",
        str(profile_value(args, "filter_min_changed_regions")),
        "--min_changed_fraction",
        str(profile_value(args, "filter_min_changed_fraction")),
        f"--min_mean_core_balanced_delta={profile_value(args, 'filter_min_mean_core_balanced_delta')}",
        f"--min_mean_delta_psnr={profile_value(args, 'filter_min_mean_delta_psnr')}",
        f"--min_tail_core_balanced_delta={profile_value(args, 'filter_min_tail_core_balanced_delta')}",
        f"--max_negative_core_balanced_fraction={profile_value(args, 'filter_max_negative_core_balanced_fraction')}",
        "--max_context_mse_regression",
        str(profile_value(args, "filter_max_context_mse_regression")),
        "--tail_fraction",
        str(profile_value(args, "filter_train_render_region_tail_fraction")),
        "--min_selected_carriers",
        str(profile_value(args, "filter_aggregate_subset_min_selected_carriers")),
        "--expected_view_count",
        str(profile_value(args, "filter_aggregate_subset_expected_view_count")),
        "--min_unique_views",
        str(profile_value(args, "filter_aggregate_subset_min_unique_views")),
        "--min_changed_unique_views",
        str(profile_value(args, "filter_aggregate_subset_min_changed_unique_views")),
        "--min_view_coverage_fraction",
        str(profile_value(args, "filter_aggregate_subset_min_view_coverage_fraction")),
        "--min_changed_view_coverage_fraction",
        str(profile_value(args, "filter_aggregate_subset_min_changed_view_coverage_fraction")),
        "--min_total_pixels",
        str(profile_value(args, "filter_aggregate_subset_min_total_pixels")),
        "--min_changed_pixels",
        str(profile_value(args, "filter_aggregate_subset_min_changed_pixels")),
        "--min_changed_pixel_fraction",
        str(profile_value(args, "filter_aggregate_subset_min_changed_pixel_fraction")),
        "--expected_frame_pixels",
        str(infer_expected_frame_pixels(args, scene)),
        "--min_full_frame_changed_pixel_fraction",
        str(profile_value(args, "filter_aggregate_subset_min_full_frame_changed_pixel_fraction")),
        "--min_area_weighted_core_balanced_delta",
        str(profile_value(args, "filter_aggregate_subset_min_area_weighted_core_balanced_delta")),
        "--min_dilution_adjusted_core_balanced_delta",
        str(profile_value(args, "filter_aggregate_subset_min_dilution_adjusted_core_balanced_delta")),
        "--min_full_frame_visibility_adjusted_delta",
        str(profile_value(args, "filter_aggregate_subset_min_full_frame_visibility_adjusted_delta")),
        "--tail_safe_shrink_scales",
        str(profile_value(args, "filter_aggregate_subset_tail_safe_shrink_scales")),
        "--tail_safe_shrink_min_scale",
        str(profile_value(args, "filter_aggregate_subset_tail_safe_shrink_min_scale")),
    ]
    if bool(profile_value(args, "filter_aggregate_subset_tail_safe_shrink_carriers")):
        command.append("--tail_safe_shrink_carriers")
    command.append(
        "--prefer_full_frame_visibility"
        if bool(profile_value(args, "filter_aggregate_subset_prefer_full_frame_visibility"))
        else "--no-prefer_full_frame_visibility"
    )
    return CommandRecord(
        stage="aggregate_subset",
        scene=scene,
        command=command,
        log_path=str(command_log(root, "aggregate_subset", scene)),
        output_paths={
            "input_filtered_plan": str(input_plan),
            "aggregate_subset_plan": str(output_plan),
            "aggregate_subset_summary": str(summary_path),
            "aggregate_subset_summary_md": str(md_path),
            "aggregate_subset_alpha_json": str(output_plan.with_name("aggregate_subset_materialize_alpha.json")),
            "render_region_objective": str(objective_path),
        },
    )


def selector_command(args: argparse.Namespace, scenes: list[str]) -> CommandRecord:
    root = output_root(args)
    label = str(args.pipeline_label)
    command = [
        sys.executable,
        "scripts/car_model/ecsr_run_facelocal_coupled_selector.py",
        "--scenes",
        ",".join(scenes),
        "--gpu",
        str(args.gpu),
        "--output_root",
        str(root / "selector"),
        "--plan_template",
        selector_plan_template(args),
        "--evidence_root",
        str(evidence_root(args)),
        "--trial_specs",
        str(profile_value(args, "trial_specs")),
        "--candidate_prefix",
        label,
        "--phasej_test_method",
        phasej_test_method(args),
        "--phasej_trainval_method",
        phasej_trainval_method(args),
        "--selector_alpha_max",
        str(profile_value(args, "selector_alpha_max")),
        "--selector_alpha_steps",
        str(profile_value(args, "selector_alpha_steps")),
        "--selector_alpha_lr",
        str(args.selector_alpha_lr),
        "--selector_alpha_max_total_samples",
        str(profile_value(args, "selector_alpha_max_total_samples")),
        "--selector_alpha_device",
        str(args.selector_alpha_device),
        "--selector_min_trainval_psnr_gain",
        str(profile_value(args, "selector_min_trainval_psnr_gain")),
        "--selector_min_trainval_balanced_delta",
        str(profile_value(args, "selector_min_trainval_balanced_delta")),
        "--selector_tail_min_trainval_balanced_delta",
        str(profile_value(args, "selector_tail_min_trainval_balanced_delta")),
        "--selector_balanced_ssim_weight",
        str(profile_value(args, "selector_balanced_ssim_weight")),
        "--selector_balanced_lpips_weight",
        str(profile_value(args, "selector_balanced_lpips_weight")),
        "--selector_tail_max_balanced_cvar_loss",
        str(args.selector_tail_max_balanced_cvar_loss),
        "--selector_tail_min_mean_to_cvar_ratio",
        str(args.selector_tail_min_mean_to_cvar_ratio),
        "--selector_tail_max_lpips_positive_fraction",
        str(args.selector_tail_max_lpips_positive_fraction),
        "--selector_region_min_trainval_balanced_delta",
        str(profile_value(args, "selector_region_min_trainval_balanced_delta")),
        "--selector_region_min_mean_core_balanced_delta",
        str(profile_value(args, "selector_region_min_mean_core_balanced_delta")),
        "--selector_region_min_mean_delta_psnr",
        str(profile_value(args, "selector_region_min_mean_delta_psnr")),
        "--selector_region_min_changed_fraction",
        str(profile_value(args, "selector_region_min_changed_fraction")),
        "--selector_region_max_negative_core_balanced_fraction",
        str(profile_value(args, "selector_region_max_negative_core_balanced_fraction")),
        "--selector_region_max_context_mse_regression",
        str(profile_value(args, "selector_region_max_context_mse_regression")),
        "--selector_strict_replay_scales",
        str(profile_value(args, "selector_strict_replay_scales")),
        "--selector_strict_adaptive_scale_min",
        str(profile_value(args, "selector_strict_adaptive_scale_min")),
        "--selector_strict_adaptive_scale_max_extra",
        str(profile_value(args, "selector_strict_adaptive_scale_max_extra")),
        "--selector_strict_adaptive_scale_tail_fraction",
        str(profile_value(args, "selector_strict_adaptive_scale_tail_fraction")),
        "--gate_min_psnr_gain",
        str(args.gate_min_psnr_gain),
        "--gate_max_ssim_regression",
        str(args.gate_max_ssim_regression),
        "--gate_max_lpips_regression",
        str(args.gate_max_lpips_regression),
        "--gate_min_balanced_delta",
        str(args.gate_min_balanced_delta),
        "--ela_alpha_risk_tail_fraction",
        str(profile_value(args, "ela_alpha_risk_tail_fraction")),
        "--ela_alpha_max_negative_gain_fraction",
        str(profile_value(args, "ela_alpha_max_negative_gain_fraction")),
        f"--ela_alpha_min_tail_gain={profile_value(args, 'ela_alpha_min_tail_gain')}",
        "--ela_alpha_view_tail_scale_grid",
        str(profile_value(args, "ela_alpha_view_tail_scale_grid")),
        "--ela_alpha_view_tail_cvar_fraction",
        str(profile_value(args, "ela_alpha_view_tail_cvar_fraction")),
        f"--ela_alpha_view_tail_min_gain={profile_value(args, 'ela_alpha_view_tail_min_gain')}",
        "--ela_alpha_view_tail_max_negative_fraction",
        str(profile_value(args, "ela_alpha_view_tail_max_negative_fraction")),
        "--ela_alpha_view_tail_objective",
        str(profile_value(args, "ela_alpha_view_tail_objective")),
        "--ela_alpha_view_tail_ssim_weight",
        str(profile_value(args, "ela_alpha_view_tail_ssim_weight")),
        "--ela_alpha_view_tail_lpips_weight",
        str(profile_value(args, "ela_alpha_view_tail_lpips_weight")),
        "--ela_alpha_view_tail_metric_max_side",
        str(profile_value(args, "ela_alpha_view_tail_metric_max_side")),
        "--ela_local_trust_min_supports",
        str(profile_value(args, "ela_local_trust_min_supports")),
        "--ela_local_trust_max_residual_std",
        str(profile_value(args, "ela_local_trust_max_residual_std")),
        "--ela_local_trust_min_agreement",
        str(profile_value(args, "ela_local_trust_min_agreement")),
        "--ela_local_trust_agreement_scale",
        str(profile_value(args, "ela_local_trust_agreement_scale")),
        "--ela_local_trust_confidence_quantile",
        str(profile_value(args, "ela_local_trust_confidence_quantile")),
        "--ela_local_trust_min_confidence",
        str(profile_value(args, "ela_local_trust_min_confidence")),
        "--ela_local_trust_mode",
        str(profile_value(args, "ela_local_trust_mode")),
        "--ela_local_trust_min_weight",
        str(profile_value(args, "ela_local_trust_min_weight")),
        "--wandb_project",
        str(args.wandb_project),
        "--wandb_group",
        f"{label}_selector",
    ]
    if bool(profile_value(args, "ela_alpha_holdout_safe_zero")):
        command.append("--ela_alpha_holdout_safe_zero")
    if bool(profile_value(args, "ela_alpha_view_tail_compute_lpips")):
        command.append("--ela_alpha_view_tail_compute_lpips")
    if bool(profile_value(args, "ela_alpha_region_risk_enable")):
        command.extend(
            [
                "--ela_alpha_region_risk_enable",
                "--ela_alpha_region_risk_json_template",
                str(root / "candidate_owned_render_regions" / "{scene}" / "train_render_region_objective_refit_base.json"),
                "--ela_alpha_region_risk_min_tail_gain",
                str(profile_value(args, "ela_alpha_region_risk_min_tail_gain")),
                "--ela_alpha_region_risk_max_negative_fraction",
                str(profile_value(args, "ela_alpha_region_risk_max_negative_fraction")),
                "--ela_alpha_region_risk_min_regions",
                str(profile_value(args, "ela_alpha_region_risk_min_regions")),
                "--ela_alpha_region_risk_objective_max_balanced_delta",
                str(profile_value(args, "ela_alpha_region_risk_objective_max_balanced_delta")),
                "--ela_alpha_region_risk_objective_max_delta_ssim",
                str(profile_value(args, "ela_alpha_region_risk_objective_max_delta_ssim")),
                "--ela_alpha_region_risk_objective_min_delta_lpips",
                str(profile_value(args, "ela_alpha_region_risk_objective_min_delta_lpips")),
            ]
        )
        command.append(
            "--ela_alpha_region_risk_objective_bad_only"
            if bool(profile_value(args, "ela_alpha_region_risk_objective_bad_only"))
            else "--no-ela_alpha_region_risk_objective_bad_only"
        )
    if bool(profile_value(args, "ela_local_trust_gate")):
        command.append("--ela_local_trust_gate")
    if bool(profile_value(args, "selector_fit_plan_alphas")):
        command.append("--selector_fit_plan_alphas")
    command.append(
        "--selector_strict_adaptive_scale_policy"
        if bool(profile_value(args, "selector_strict_adaptive_scale_policy"))
        else "--no-selector_strict_adaptive_scale_policy"
    )
    command.append(
        "--selector_strict_fit_plan_alphas"
        if bool(profile_value(args, "selector_strict_fit_plan_alphas"))
        else "--no-selector_strict_fit_plan_alphas"
    )
    if bool(args.selector_tail_stable_promotion):
        command.append("--selector_enable_tail_stable_promotion")
    if bool(profile_value(args, "selector_enable_region_stable_promotion")):
        command.append("--selector_enable_region_stable_promotion")
    if bool(args.force):
        command.append("--force")
    output_paths = {
        "selector_root": str(root / "selector"),
        "selector_summary": str(root / "selector" / "coupled_selector_summary.json"),
        "selector_plan_template": selector_plan_template(args),
    }
    if bool(profile_value(args, "candidate_region_pre_refit_risk_shrink")):
        for scene in scenes:
            output_paths[f"{scene}_selector_risk_shrink_report"] = str(
                format_pre_refit_risk_shrink_report_path(args, scene, purpose="selector")
            )
            output_paths[f"{scene}_selector_risk_shrink_alpha_json"] = str(
                format_pre_refit_risk_shrink_alpha_path(args, scene, purpose="selector")
            )
    return CommandRecord(
        stage="selector",
        scene=",".join(scenes),
        command=command,
        log_path=str(command_log(root, "selector", "all_scenes")),
        output_paths=output_paths,
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_block(payload: dict[str, Any] | None) -> dict[str, float | None]:
    payload = payload or {}
    out: dict[str, float | None] = {}
    for key in METRICS:
        try:
            value = float(payload.get(key))
        except Exception:
            value = math.nan
        out[key] = value if math.isfinite(value) else None
    return out


def scene_decision_summary(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    path = output_root(args) / "selector" / scene / "coupled_selector_decision.json"
    payload = load_json(path)
    if not payload:
        return {
            "scene": scene,
            "decision_path": rel(path),
            "present": False,
            "accepted": False,
            "selected_trial": "",
            "effective_report_only_test_delta": {key: None for key in METRICS},
        }
    return {
        "scene": scene,
        "decision_path": rel(path),
        "present": True,
        "accepted": bool(payload.get("accepted", False)),
        "selected_trial": payload.get("selected_trial", ""),
        "candidate_count": int(payload.get("candidate_count", 0)),
        "selected_trainval_balanced_delta": payload.get("selected_trainval_balanced_delta"),
        "effective_report_only_test_delta": metric_block(payload.get("effective_report_only_test_delta")),
        "selection_uses_test": bool(payload.get("selection_uses_test", False)),
    }


def command_record_output_sha256s(record: CommandRecord) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, raw in record.output_paths.items():
        if not raw:
            continue
        raw_str = str(raw)
        if len(raw_str) > 512 and not any(sep in raw_str for sep in ("/", "\\")):
            continue
        if "," in raw_str and not any(sep in raw_str for sep in ("/", "\\")):
            continue
        try:
            path = Path(raw_str)
        except Exception:
            continue
        try:
            is_file = path.is_file()
        except OSError:
            # Some output metadata entries, such as comma-separated face-id
            # allowlists, are intentionally not filesystem paths.
            continue
        if is_file:
            out[key] = file_sha256(path)
    return out


def write_manifest(args: argparse.Namespace, records: list[CommandRecord], scenes: list[str]) -> None:
    root = output_root(args)
    root.mkdir(parents=True, exist_ok=True)
    resolved_args = resolved_profile_values(args)
    profile_defaults = PROFILE_DEFAULTS[str(args.profile)]
    payload = {
        "pipeline": "ecsr_autovisual_facelocal_pipeline",
        "pipeline_label": str(args.pipeline_label),
        "profile": str(args.profile),
        "profile_contract_id": profile_contract_id(args),
        "fixed_profile": str(args.profile) in FIXED_PROFILE_NAMES,
        "profile_override_policy": "forbidden" if str(args.profile) in FIXED_PROFILE_NAMES else "allowed_explicit_cli",
        "profile_override_fields": list(getattr(args, "profile_override_fields", [])),
        "profile_defaults": profile_defaults,
        "profile_defaults_sha256": stable_json_sha256(profile_defaults),
        "resolved_profile_args": resolved_args,
        "resolved_profile_args_sha256": stable_json_sha256(resolved_args),
        "dry_run": bool(args.dry_run),
        "selection_uses_test": False,
        "scenes": scenes,
        "plan_template": plan_template(args),
        "candidate_owned_refit": bool(profile_value(args, "candidate_owned_refit")),
        "refit_plan_template": refit_plan_template(args),
        "filtered_plan_template": filtered_plan_template(args),
        "aggregate_subset_plan_template": aggregate_subset_plan_template(args),
        "selector_plan_template": selector_plan_template(args),
        "render_region_carrier_template": str(args.render_region_carrier_template),
        "evidence_root": str(evidence_root(args)),
        "phasej_test_method_requested": str(args.phasej_test_method),
        "phasej_trainval_method_requested": str(args.phasej_trainval_method),
        "phasej_test_method": phasej_test_method(args),
        "phasej_trainval_method": phasej_trainval_method(args),
        "isolate_phasej_methods": bool(args.isolate_phasej_methods),
        "plan_contract": {
            "render_region_gate_enabled": True,
            "min_regions": profile_value(args, "plan_region_gate_min_regions"),
            "min_changed_regions": profile_value(args, "plan_region_gate_min_changed_regions"),
            "min_changed_fraction": profile_value(args, "plan_region_gate_min_changed_fraction"),
            "min_core_balanced_delta": profile_value(args, "plan_region_gate_min_core_balanced_delta"),
            "min_core_psnr_delta": profile_value(args, "plan_region_gate_min_core_psnr_delta"),
            "min_tail_cvar_delta": profile_value(args, "plan_region_gate_min_tail_cvar_delta"),
            "max_context_mse_regression": profile_value(args, "plan_region_gate_max_context_mse_regression"),
            "max_negative_fraction": profile_value(args, "plan_region_gate_max_negative_fraction"),
        },
        "filter_contract": {
            "region_source": profile_value(args, "filter_region_source"),
            "drop_unmapped": profile_value(args, "filter_drop_unmapped"),
            "require_positive_plan_proxy": profile_value(args, "filter_require_positive_plan_proxy"),
            "min_changed_fraction": profile_value(args, "filter_min_changed_fraction"),
            "min_mean_core_balanced_delta": profile_value(args, "filter_min_mean_core_balanced_delta"),
            "min_mean_delta_psnr": profile_value(args, "filter_min_mean_delta_psnr"),
            "min_tail_core_balanced_delta": profile_value(args, "filter_min_tail_core_balanced_delta"),
            "max_negative_core_balanced_fraction": profile_value(args, "filter_max_negative_core_balanced_fraction"),
            "max_context_mse_regression": profile_value(args, "filter_max_context_mse_regression"),
            "min_mean_crop_abs_diff": profile_value(args, "filter_min_mean_crop_abs_diff"),
            "min_max_crop_abs_diff": profile_value(args, "filter_min_max_crop_abs_diff"),
            "tail_safe_shrink_on_tail_fail": profile_value(args, "filter_tail_safe_shrink_on_tail_fail"),
            "tail_safe_shrink_min_scale": profile_value(args, "filter_tail_safe_shrink_min_scale"),
            "tail_safe_shrink_min_raw_scale": profile_value(args, "filter_tail_safe_shrink_min_raw_scale"),
            "rollback_severe_tail_fail": profile_value(args, "filter_rollback_severe_tail_fail"),
            "rollback_tail_min_cvar_loss": profile_value(args, "filter_rollback_tail_min_cvar_loss"),
            "aggregate_subset": profile_value(args, "filter_aggregate_subset"),
            "aggregate_subset_min_selected_carriers": profile_value(args, "filter_aggregate_subset_min_selected_carriers"),
            "aggregate_subset_expected_view_count": profile_value(
                args, "filter_aggregate_subset_expected_view_count"
            ),
            "aggregate_subset_min_unique_views": profile_value(args, "filter_aggregate_subset_min_unique_views"),
            "aggregate_subset_min_changed_unique_views": profile_value(
                args, "filter_aggregate_subset_min_changed_unique_views"
            ),
            "aggregate_subset_min_view_coverage_fraction": profile_value(
                args, "filter_aggregate_subset_min_view_coverage_fraction"
            ),
            "aggregate_subset_min_changed_view_coverage_fraction": profile_value(
                args, "filter_aggregate_subset_min_changed_view_coverage_fraction"
            ),
            "aggregate_subset_min_total_pixels": profile_value(args, "filter_aggregate_subset_min_total_pixels"),
            "aggregate_subset_min_changed_pixels": profile_value(
                args, "filter_aggregate_subset_min_changed_pixels"
            ),
            "aggregate_subset_min_changed_pixel_fraction": profile_value(
                args, "filter_aggregate_subset_min_changed_pixel_fraction"
            ),
            "aggregate_subset_expected_frame_pixels": profile_value(
                args, "filter_aggregate_subset_expected_frame_pixels"
            ),
            "aggregate_subset_min_full_frame_changed_pixel_fraction": profile_value(
                args, "filter_aggregate_subset_min_full_frame_changed_pixel_fraction"
            ),
            "aggregate_subset_min_area_weighted_core_balanced_delta": profile_value(
                args, "filter_aggregate_subset_min_area_weighted_core_balanced_delta"
            ),
            "aggregate_subset_min_dilution_adjusted_core_balanced_delta": profile_value(
                args, "filter_aggregate_subset_min_dilution_adjusted_core_balanced_delta"
            ),
            "aggregate_subset_min_full_frame_visibility_adjusted_delta": profile_value(
                args, "filter_aggregate_subset_min_full_frame_visibility_adjusted_delta"
            ),
            "aggregate_subset_prefer_full_frame_visibility": profile_value(
                args, "filter_aggregate_subset_prefer_full_frame_visibility"
            ),
            "aggregate_subset_tail_safe_shrink_carriers": profile_value(
                args, "filter_aggregate_subset_tail_safe_shrink_carriers"
            ),
            "aggregate_subset_tail_safe_shrink_scales": profile_value(
                args, "filter_aggregate_subset_tail_safe_shrink_scales"
            ),
            "aggregate_subset_tail_safe_shrink_min_scale": profile_value(
                args, "filter_aggregate_subset_tail_safe_shrink_min_scale"
            ),
            "risk_safe_shrink_on_train_risk_fail": profile_value(args, "filter_risk_safe_shrink_on_train_risk_fail"),
            "risk_safe_shrink_min_scale": profile_value(args, "filter_risk_safe_shrink_min_scale"),
        },
        "candidate_region_contract": {
            "expand_faces": profile_value(args, "filter_candidate_region_expand_faces"),
            "expand_min_face_pixels": profile_value(args, "filter_candidate_region_expand_min_face_pixels"),
            "expand_min_face_views": profile_value(args, "filter_candidate_region_expand_min_face_views"),
            "expand_max_faces_per_carrier": profile_value(args, "filter_candidate_region_expand_max_faces_per_carrier"),
            "frame_aware_ranking": profile_value(args, "filter_candidate_region_frame_aware_ranking"),
            "min_frame_support_fraction": profile_value(args, "filter_candidate_region_min_frame_support_fraction"),
            "min_residual_mass_fraction": profile_value(args, "filter_candidate_region_min_residual_mass_fraction"),
            "max_carriers": profile_value(args, "filter_candidate_region_max_carriers"),
            "pre_refit_risk_prune": profile_value(args, "candidate_region_pre_refit_risk_prune"),
            "pre_refit_risk_min_changed_rows": profile_value(
                args, "candidate_region_pre_refit_risk_min_changed_rows"
            ),
            "pre_refit_risk_min_bad_rows": profile_value(args, "candidate_region_pre_refit_risk_min_bad_rows"),
            "pre_refit_risk_max_bad_fraction": profile_value(
                args, "candidate_region_pre_refit_risk_max_bad_fraction"
            ),
            "pre_refit_risk_balanced_margin": profile_value(
                args, "candidate_region_pre_refit_risk_balanced_margin"
            ),
            "pre_refit_risk_use_aux_metric_pair": profile_value(
                args, "candidate_region_pre_refit_risk_use_aux_metric_pair"
            ),
            "pre_refit_risk_ssim_margin": profile_value(args, "candidate_region_pre_refit_risk_ssim_margin"),
            "pre_refit_risk_lpips_margin": profile_value(args, "candidate_region_pre_refit_risk_lpips_margin"),
            "pre_refit_risk_max_removed_face_fraction": profile_value(
                args, "candidate_region_pre_refit_risk_max_removed_face_fraction"
            ),
            "pre_refit_risk_shrink": profile_value(args, "candidate_region_pre_refit_risk_shrink"),
            "pre_refit_risk_shrink_min_scale": profile_value(
                args, "candidate_region_pre_refit_risk_shrink_min_scale"
            ),
            "pre_refit_risk_shrink_severity_aware": profile_value(
                args, "candidate_region_pre_refit_risk_shrink_severity_aware"
            ),
            "pre_refit_risk_shrink_severity_select_min": profile_value(
                args, "candidate_region_pre_refit_risk_shrink_severity_select_min"
            ),
            "pre_refit_risk_shrink_severity_balanced_span": profile_value(
                args, "candidate_region_pre_refit_risk_shrink_severity_balanced_span"
            ),
            "pre_refit_risk_shrink_tail_fraction": profile_value(
                args, "candidate_region_pre_refit_risk_shrink_tail_fraction"
            ),
            "pre_refit_risk_local_suppression": profile_value(
                args, "candidate_region_pre_refit_risk_local_suppression"
            ),
            "pre_refit_risk_local_suppression_scale": profile_value(
                args, "candidate_region_pre_refit_risk_local_suppression_scale"
            ),
            "pre_refit_risk_local_suppression_min_bad_balanced": profile_value(
                args, "candidate_region_pre_refit_risk_local_suppression_min_bad_balanced"
            ),
            "pre_refit_risk_local_suppression_positive_margin": profile_value(
                args, "candidate_region_pre_refit_risk_local_suppression_positive_margin"
            ),
        },
        "selector_contract": {
            "selector_plan_template": selector_plan_template(args),
            "fit_plan_alphas": profile_value(args, "selector_fit_plan_alphas"),
            "strict_replay_scales": profile_value(args, "selector_strict_replay_scales"),
            "strict_adaptive_scale_policy": profile_value(args, "selector_strict_adaptive_scale_policy"),
            "strict_fit_plan_alphas": profile_value(args, "selector_strict_fit_plan_alphas"),
            "min_trainval_psnr_gain": profile_value(args, "selector_min_trainval_psnr_gain"),
            "min_trainval_balanced_delta": profile_value(args, "selector_min_trainval_balanced_delta"),
            "balanced_ssim_weight": profile_value(args, "selector_balanced_ssim_weight"),
            "balanced_lpips_weight": profile_value(args, "selector_balanced_lpips_weight"),
            "tail_min_trainval_balanced_delta": profile_value(args, "selector_tail_min_trainval_balanced_delta"),
            "region_stable_promotion": profile_value(args, "selector_enable_region_stable_promotion"),
            "region_min_trainval_balanced_delta": profile_value(args, "selector_region_min_trainval_balanced_delta"),
            "region_min_mean_core_balanced_delta": profile_value(args, "selector_region_min_mean_core_balanced_delta"),
            "region_min_mean_delta_psnr": profile_value(args, "selector_region_min_mean_delta_psnr"),
            "region_min_changed_fraction": profile_value(args, "selector_region_min_changed_fraction"),
            "region_max_negative_core_balanced_fraction": profile_value(
                args, "selector_region_max_negative_core_balanced_fraction"
            ),
            "region_max_context_mse_regression": profile_value(args, "selector_region_max_context_mse_regression"),
        },
        "coverage_prefix_contract": {
            "min_faces": int(args.delta_patch_cert_carrier_holdout_auto_prefix_min_faces),
            "min_face_fraction": profile_value(args, "delta_patch_cert_carrier_holdout_auto_prefix_min_face_fraction"),
            "face_bonus": profile_value(args, "delta_patch_cert_carrier_holdout_auto_prefix_face_bonus"),
            "positive_tail_safe": bool(args.delta_patch_cert_carrier_holdout_auto_prefix_positive_tail_safe),
        },
        "support_aware_policy_contract": {
            "absolute_min_policy_val_samples": profile_value(args, "delta_min_policy_val_samples"),
            "adaptive_sample_fraction": profile_value(args, "delta_min_policy_val_adaptive_sample_fraction"),
            "adaptive_min_samples": profile_value(args, "delta_min_policy_val_adaptive_min_samples"),
            "min_policy_val_unique_faces": profile_value(args, "delta_min_policy_val_unique_faces"),
        },
        "commands": [
            {
                **asdict(record),
                "output_path_sha256s": command_record_output_sha256s(record),
            }
            for record in records
        ],
    }
    (root / "pipeline_command_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ECSR Auto-Visual Face-Local Pipeline Manifest",
        "",
        f"- label: `{args.pipeline_label}`",
        f"- profile: `{args.profile}`",
        f"- profile contract: `{profile_contract_id(args)}`",
        f"- fixed profile: `{str(str(args.profile) in FIXED_PROFILE_NAMES).lower()}`",
        f"- profile override fields: `{', '.join(getattr(args, 'profile_override_fields', [])) or 'none'}`",
        f"- resolved profile SHA-256: `{stable_json_sha256(resolved_args)}`",
        f"- delta strength: `{profile_value(args, 'delta_strength')}`",
        f"- delta max abs rgb: `{profile_value(args, 'delta_max_abs_rgb')}`",
        f"- shared residual field: `{str(bool(profile_value(args, 'delta_shared_residual_field'))).lower()}`",
        f"- shared residual field anchors: `{profile_value(args, 'delta_shared_residual_field_anchors')}`",
        f"- cluster basis mode: `{profile_value(args, 'delta_patch_cert_cluster_basis_mode')}`",
        f"- render-region objective: `{str(bool(profile_value(args, 'delta_render_region_objective'))).lower()}`",
        f"- render-region outside penalty: `{profile_value(args, 'delta_render_region_outside_penalty')}`",
        f"- render-region tail CVaR weight: `{profile_value(args, 'delta_render_region_tail_cvar_weight')}`",
        f"- bystander zero-delta weight: `{profile_value(args, 'delta_bystander_zero_delta_weight')}`",
        f"- bystander zero-delta include context: `{str(bool(profile_value(args, 'delta_bystander_zero_delta_include_context'))).lower()}`",
        f"- bystander zero-delta min samples: `{profile_value(args, 'delta_bystander_zero_delta_min_samples')}`",
        f"- witness-group CVaR weight: `{profile_value(args, 'delta_witness_constraint_weight')}`",
        f"- witness-group CVaR tail fraction: `{profile_value(args, 'delta_witness_constraint_tail_fraction')}`",
        f"- witness-group min samples: `{profile_value(args, 'delta_witness_constraint_min_samples')}`",
        f"- witness-group margin: `{profile_value(args, 'delta_witness_constraint_margin')}`",
        f"- witness groups full/region/bystander: `{str(bool(profile_value(args, 'delta_witness_constraint_include_full_view'))).lower()}` / `{str(bool(profile_value(args, 'delta_witness_constraint_include_region_view'))).lower()}` / `{str(bool(profile_value(args, 'delta_witness_constraint_include_bystander_view'))).lower()}`",
        f"- ELA local-trust gate: `{str(bool(profile_value(args, 'ela_local_trust_gate'))).lower()}`",
        f"- ELA local-trust mode/min weight: `{profile_value(args, 'ela_local_trust_mode')}` / `{profile_value(args, 'ela_local_trust_min_weight')}`",
        f"- ELA local-trust supports/std/agreement: `{profile_value(args, 'ela_local_trust_min_supports')}` / `{profile_value(args, 'ela_local_trust_max_residual_std')}` / `{profile_value(args, 'ela_local_trust_min_agreement')}`",
        f"- ELA local-trust confidence quantile/min: `{profile_value(args, 'ela_local_trust_confidence_quantile')}` / `{profile_value(args, 'ela_local_trust_min_confidence')}`",
        f"- ELA alpha view-tail scale grid: `{profile_value(args, 'ela_alpha_view_tail_scale_grid')}`",
        f"- ELA alpha view-tail cvar/min/neg: `{profile_value(args, 'ela_alpha_view_tail_cvar_fraction')}` / `{profile_value(args, 'ela_alpha_view_tail_min_gain')}` / `{profile_value(args, 'ela_alpha_view_tail_max_negative_fraction')}`",
        f"- ELA alpha view-tail objective/LPIPS: `{profile_value(args, 'ela_alpha_view_tail_objective')}` / `{str(bool(profile_value(args, 'ela_alpha_view_tail_compute_lpips'))).lower()}`",
        f"- plan region gate min changed fraction: `{profile_value(args, 'plan_region_gate_min_changed_fraction')}`",
        f"- plan region gate max negative fraction: `{profile_value(args, 'plan_region_gate_max_negative_fraction')}`",
        f"- filter region source: `{profile_value(args, 'filter_region_source')}`",
        f"- filter visible diff min mean/max: `{profile_value(args, 'filter_min_mean_crop_abs_diff')}` / `{profile_value(args, 'filter_min_max_crop_abs_diff')}`",
        f"- filter tail-safe shrink: `{str(bool(profile_value(args, 'filter_tail_safe_shrink_on_tail_fail'))).lower()}`",
        f"- filter tail-safe shrink raw floor: `{profile_value(args, 'filter_tail_safe_shrink_min_raw_scale')}`",
        f"- filter severe-tail rollback: `{str(bool(profile_value(args, 'filter_rollback_severe_tail_fail'))).lower()}`",
        f"- filter severe-tail rollback min CVaR loss: `{profile_value(args, 'filter_rollback_tail_min_cvar_loss')}`",
        f"- filter aggregate subset: `{str(bool(profile_value(args, 'filter_aggregate_subset'))).lower()}`",
        f"- filter aggregate subset min carriers: `{profile_value(args, 'filter_aggregate_subset_min_selected_carriers')}`",
        f"- filter aggregate subset changed views: `{profile_value(args, 'filter_aggregate_subset_min_changed_unique_views')}`",
        f"- filter aggregate subset changed px frac: `{profile_value(args, 'filter_aggregate_subset_min_changed_pixel_fraction')}`",
        f"- filter aggregate subset full-frame changed px frac: `{profile_value(args, 'filter_aggregate_subset_min_full_frame_changed_pixel_fraction')}`",
        f"- filter aggregate subset dilution min: `{profile_value(args, 'filter_aggregate_subset_min_dilution_adjusted_core_balanced_delta')}`",
        f"- filter aggregate subset full-frame visibility min: `{profile_value(args, 'filter_aggregate_subset_min_full_frame_visibility_adjusted_delta')}`",
        f"- filter aggregate subset prefer full-frame visibility: `{str(bool(profile_value(args, 'filter_aggregate_subset_prefer_full_frame_visibility'))).lower()}`",
        f"- filter aggregate subset carrier shrink: `{str(bool(profile_value(args, 'filter_aggregate_subset_tail_safe_shrink_carriers'))).lower()}`",
        f"- filter aggregate subset shrink scales: `{profile_value(args, 'filter_aggregate_subset_tail_safe_shrink_scales')}`",
        f"- filter aggregate subset shrink floor: `{profile_value(args, 'filter_aggregate_subset_tail_safe_shrink_min_scale')}`",
        f"- filter risk-safe shrink: `{str(bool(profile_value(args, 'filter_risk_safe_shrink_on_train_risk_fail'))).lower()}`",
        f"- candidate-region expansion: `{str(bool(profile_value(args, 'filter_candidate_region_expand_faces'))).lower()}`",
        f"- candidate-region frame-aware ranking: `{str(bool(profile_value(args, 'filter_candidate_region_frame_aware_ranking'))).lower()}`",
        f"- candidate-region frame support min: `{profile_value(args, 'filter_candidate_region_min_frame_support_fraction')}`",
        f"- candidate-region residual mass min: `{profile_value(args, 'filter_candidate_region_min_residual_mass_fraction')}`",
        f"- candidate-region max carriers: `{profile_value(args, 'filter_candidate_region_max_carriers')}`",
        f"- candidate-region pre-refit risk prune: `{str(bool(profile_value(args, 'candidate_region_pre_refit_risk_prune'))).lower()}`",
        f"- candidate-region pre-refit risk bad fraction: `>{profile_value(args, 'candidate_region_pre_refit_risk_max_bad_fraction')}`",
        f"- candidate-region pre-refit risk margins: balanced<-`{profile_value(args, 'candidate_region_pre_refit_risk_balanced_margin')}`, ssim<-`{profile_value(args, 'candidate_region_pre_refit_risk_ssim_margin')}`, lpips>`{profile_value(args, 'candidate_region_pre_refit_risk_lpips_margin')}`",
        f"- candidate-region pre-refit max removed face fraction: `{profile_value(args, 'candidate_region_pre_refit_risk_max_removed_face_fraction')}`",
        f"- candidate-region pre-refit risk shrink: `{str(bool(profile_value(args, 'candidate_region_pre_refit_risk_shrink'))).lower()}`",
        f"- candidate-region pre-refit risk shrink min scale: `{profile_value(args, 'candidate_region_pre_refit_risk_shrink_min_scale')}`",
        f"- candidate-region pre-refit severity-aware shrink: `{str(bool(profile_value(args, 'candidate_region_pre_refit_risk_shrink_severity_aware'))).lower()}`",
        f"- candidate-region pre-refit severity select min: `{profile_value(args, 'candidate_region_pre_refit_risk_shrink_severity_select_min')}`",
        f"- candidate-region pre-refit severity balanced span: `{profile_value(args, 'candidate_region_pre_refit_risk_shrink_severity_balanced_span')}`",
        f"- candidate-region pre-refit shrink tail fraction: `{profile_value(args, 'candidate_region_pre_refit_risk_shrink_tail_fraction')}`",
        f"- candidate-region pre-refit local suppression: `{str(bool(profile_value(args, 'candidate_region_pre_refit_risk_local_suppression'))).lower()}`",
        f"- candidate-region pre-refit local suppression scale: `{profile_value(args, 'candidate_region_pre_refit_risk_local_suppression_scale')}`",
        f"- candidate-region pre-refit local suppression min bad balanced: `{profile_value(args, 'candidate_region_pre_refit_risk_local_suppression_min_bad_balanced')}`",
        f"- candidate-region pre-refit local suppression positive margin: `{profile_value(args, 'candidate_region_pre_refit_risk_local_suppression_positive_margin')}`",
        f"- ELA alpha holdout-safe zero: `{str(bool(profile_value(args, 'ela_alpha_holdout_safe_zero'))).lower()}`",
        f"- ELA alpha tail fraction: `{profile_value(args, 'ela_alpha_risk_tail_fraction')}`",
        f"- ELA alpha max negative fraction: `{profile_value(args, 'ela_alpha_max_negative_gain_fraction')}`",
        f"- ELA alpha min tail gain: `{profile_value(args, 'ela_alpha_min_tail_gain')}`",
        f"- ELA alpha region-risk: `{str(bool(profile_value(args, 'ela_alpha_region_risk_enable'))).lower()}`",
        f"- ELA alpha region-risk objective bad only: `{str(bool(profile_value(args, 'ela_alpha_region_risk_objective_bad_only'))).lower()}`",
        f"- ELA alpha region-risk objective thresholds: balanced<`{profile_value(args, 'ela_alpha_region_risk_objective_max_balanced_delta')}`, ssim<`{profile_value(args, 'ela_alpha_region_risk_objective_max_delta_ssim')}`, lpips>`{profile_value(args, 'ela_alpha_region_risk_objective_min_delta_lpips')}`",
        f"- ELA alpha region-risk min tail gain: `{profile_value(args, 'ela_alpha_region_risk_min_tail_gain')}`",
        f"- ELA alpha region-risk max negative fraction: `{profile_value(args, 'ela_alpha_region_risk_max_negative_fraction')}`",
        f"- ELA alpha region-risk min regions: `{profile_value(args, 'ela_alpha_region_risk_min_regions')}`",
        f"- candidate-owned refit: `{str(bool(profile_value(args, 'candidate_owned_refit'))).lower()}`",
        f"- filter drop unmapped: `{str(bool(profile_value(args, 'filter_drop_unmapped'))).lower()}`",
        f"- isolate PhaseJ methods: `{str(bool(args.isolate_phasej_methods)).lower()}`",
        f"- PhaseJ test method: `{phasej_test_method(args)}`",
        f"- PhaseJ train-val method: `{phasej_trainval_method(args)}`",
        f"- selector alpha max: `{profile_value(args, 'selector_alpha_max')}`",
        f"- selector fit plan alphas: `{str(bool(profile_value(args, 'selector_fit_plan_alphas'))).lower()}`",
        f"- coverage-prefix min faces/fraction: `{args.delta_patch_cert_carrier_holdout_auto_prefix_min_faces}` / `{profile_value(args, 'delta_patch_cert_carrier_holdout_auto_prefix_min_face_fraction')}`",
        f"- coverage-prefix face bonus: `{profile_value(args, 'delta_patch_cert_carrier_holdout_auto_prefix_face_bonus')}`",
        f"- support-aware policy floor fraction/min: `{profile_value(args, 'delta_min_policy_val_adaptive_sample_fraction')}` / `{profile_value(args, 'delta_min_policy_val_adaptive_min_samples')}`",
        f"- dry run: `{str(bool(args.dry_run)).lower()}`",
        f"- selection uses test: `false`",
        f"- scenes: `{', '.join(scenes)}`",
        f"- plan template: `{rel(plan_template(args))}`",
        f"- refit plan template: `{rel(refit_plan_template(args))}`",
        f"- filtered plan template: `{rel(filtered_plan_template(args))}`",
        f"- aggregate subset plan template: `{rel(aggregate_subset_plan_template(args))}`",
        f"- selector plan template: `{rel(selector_plan_template(args))}`",
        f"- render-region carrier template: `{rel(args.render_region_carrier_template)}`",
        f"- evidence root: `{rel(evidence_root(args))}`",
        "",
        "| stage | scene | exit | log | command |",
        "|---|---|---:|---|---|",
    ]
    for record in records:
        exit_text = "" if record.exit_code is None else str(record.exit_code)
        lines.append(
            f"| {record.stage} | {record.scene} | {exit_text} | `{rel(record.log_path)}` | `{shell_join(record.command)}` |"
        )
    (root / "pipeline_command_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(args: argparse.Namespace, records: list[CommandRecord], scenes: list[str]) -> None:
    root = output_root(args)
    selector_payload = load_json(root / "selector" / "coupled_selector_summary.json")
    rows = [scene_decision_summary(args, scene) for scene in scenes]
    resolved_args = resolved_profile_values(args)
    profile_defaults = PROFILE_DEFAULTS[str(args.profile)]
    payload = {
        "pipeline": "ecsr_autovisual_facelocal_pipeline",
        "pipeline_label": str(args.pipeline_label),
        "profile": str(args.profile),
        "profile_contract_id": profile_contract_id(args),
        "fixed_profile": str(args.profile) in FIXED_PROFILE_NAMES,
        "profile_override_policy": "forbidden" if str(args.profile) in FIXED_PROFILE_NAMES else "allowed_explicit_cli",
        "profile_override_fields": list(getattr(args, "profile_override_fields", [])),
        "profile_defaults": profile_defaults,
        "profile_defaults_sha256": stable_json_sha256(profile_defaults),
        "resolved_profile_args": resolved_args,
        "resolved_profile_args_sha256": stable_json_sha256(resolved_args),
        "dry_run": bool(args.dry_run),
        "selection_uses_test": False,
        "command_manifest": rel(root / "pipeline_command_manifest.json"),
        "plan_template": plan_template(args),
        "candidate_owned_refit": bool(profile_value(args, "candidate_owned_refit")),
        "refit_plan_template": refit_plan_template(args),
        "filtered_plan_template": filtered_plan_template(args),
        "aggregate_subset_plan_template": aggregate_subset_plan_template(args),
        "selector_plan_template": selector_plan_template(args),
        "render_region_carrier_template": str(args.render_region_carrier_template),
        "phasej_test_method_requested": str(args.phasej_test_method),
        "phasej_trainval_method_requested": str(args.phasej_trainval_method),
        "phasej_test_method": phasej_test_method(args),
        "phasej_trainval_method": phasej_trainval_method(args),
        "isolate_phasej_methods": bool(args.isolate_phasej_methods),
        "selector_summary": rel(root / "selector" / "coupled_selector_summary.json"),
        "selector_summary_present": bool(selector_payload),
        "command_exit_codes": [
            {"stage": record.stage, "scene": record.scene, "exit_code": record.exit_code}
            for record in records
        ],
        "rows": rows,
    }
    (root / "pipeline_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# ECSR Auto-Visual Face-Local Pipeline Summary",
        "",
        "Selection is train-val render gated. Held-out deltas, when present, are report-only and are not used for promotion.",
        "",
        f"- label: `{args.pipeline_label}`",
        f"- profile: `{args.profile}`",
        f"- profile contract: `{profile_contract_id(args)}`",
        f"- fixed profile: `{str(str(args.profile) in FIXED_PROFILE_NAMES).lower()}`",
        f"- profile override fields: `{', '.join(getattr(args, 'profile_override_fields', [])) or 'none'}`",
        f"- resolved profile SHA-256: `{stable_json_sha256(resolved_args)}`",
        f"- delta strength: `{profile_value(args, 'delta_strength')}`",
        f"- delta max abs rgb: `{profile_value(args, 'delta_max_abs_rgb')}`",
        f"- selector alpha max: `{profile_value(args, 'selector_alpha_max')}`",
        f"- selector fit plan alphas: `{str(bool(profile_value(args, 'selector_fit_plan_alphas'))).lower()}`",
        f"- coverage-prefix min faces/fraction: `{args.delta_patch_cert_carrier_holdout_auto_prefix_min_faces}` / `{profile_value(args, 'delta_patch_cert_carrier_holdout_auto_prefix_min_face_fraction')}`",
        f"- coverage-prefix face bonus: `{profile_value(args, 'delta_patch_cert_carrier_holdout_auto_prefix_face_bonus')}`",
        f"- support-aware policy floor fraction/min: `{profile_value(args, 'delta_min_policy_val_adaptive_sample_fraction')}` / `{profile_value(args, 'delta_min_policy_val_adaptive_min_samples')}`",
        f"- render-region objective: `{str(bool(profile_value(args, 'delta_render_region_objective'))).lower()}`",
        f"- render-region outside penalty: `{profile_value(args, 'delta_render_region_outside_penalty')}`",
        f"- render-region tail CVaR weight: `{profile_value(args, 'delta_render_region_tail_cvar_weight')}`",
        f"- bystander zero-delta weight: `{profile_value(args, 'delta_bystander_zero_delta_weight')}`",
        f"- bystander zero-delta include context: `{str(bool(profile_value(args, 'delta_bystander_zero_delta_include_context'))).lower()}`",
        f"- bystander zero-delta min samples: `{profile_value(args, 'delta_bystander_zero_delta_min_samples')}`",
        f"- witness-group CVaR weight: `{profile_value(args, 'delta_witness_constraint_weight')}`",
        f"- witness-group CVaR tail fraction: `{profile_value(args, 'delta_witness_constraint_tail_fraction')}`",
        f"- witness-group min samples: `{profile_value(args, 'delta_witness_constraint_min_samples')}`",
        f"- witness-group margin: `{profile_value(args, 'delta_witness_constraint_margin')}`",
        f"- witness groups full/region/bystander: `{str(bool(profile_value(args, 'delta_witness_constraint_include_full_view'))).lower()}` / `{str(bool(profile_value(args, 'delta_witness_constraint_include_region_view'))).lower()}` / `{str(bool(profile_value(args, 'delta_witness_constraint_include_bystander_view'))).lower()}`",
        f"- ELA local-trust gate: `{str(bool(profile_value(args, 'ela_local_trust_gate'))).lower()}`",
        f"- ELA local-trust mode/min weight: `{profile_value(args, 'ela_local_trust_mode')}` / `{profile_value(args, 'ela_local_trust_min_weight')}`",
        f"- ELA local-trust supports/std/agreement: `{profile_value(args, 'ela_local_trust_min_supports')}` / `{profile_value(args, 'ela_local_trust_max_residual_std')}` / `{profile_value(args, 'ela_local_trust_min_agreement')}`",
        f"- ELA local-trust confidence quantile/min: `{profile_value(args, 'ela_local_trust_confidence_quantile')}` / `{profile_value(args, 'ela_local_trust_min_confidence')}`",
        f"- ELA alpha view-tail scale grid: `{profile_value(args, 'ela_alpha_view_tail_scale_grid')}`",
        f"- ELA alpha view-tail cvar/min/neg: `{profile_value(args, 'ela_alpha_view_tail_cvar_fraction')}` / `{profile_value(args, 'ela_alpha_view_tail_min_gain')}` / `{profile_value(args, 'ela_alpha_view_tail_max_negative_fraction')}`",
        f"- ELA alpha view-tail objective/LPIPS: `{profile_value(args, 'ela_alpha_view_tail_objective')}` / `{str(bool(profile_value(args, 'ela_alpha_view_tail_compute_lpips'))).lower()}`",
        f"- plan region gate min changed fraction: `{profile_value(args, 'plan_region_gate_min_changed_fraction')}`",
        f"- plan region gate max negative fraction: `{profile_value(args, 'plan_region_gate_max_negative_fraction')}`",
        f"- filter region source: `{profile_value(args, 'filter_region_source')}`",
        f"- filter visible diff min mean/max: `{profile_value(args, 'filter_min_mean_crop_abs_diff')}` / `{profile_value(args, 'filter_min_max_crop_abs_diff')}`",
        f"- filter tail-safe shrink: `{str(bool(profile_value(args, 'filter_tail_safe_shrink_on_tail_fail'))).lower()}`",
        f"- filter tail-safe shrink raw floor: `{profile_value(args, 'filter_tail_safe_shrink_min_raw_scale')}`",
        f"- filter severe-tail rollback: `{str(bool(profile_value(args, 'filter_rollback_severe_tail_fail'))).lower()}`",
        f"- filter severe-tail rollback min CVaR loss: `{profile_value(args, 'filter_rollback_tail_min_cvar_loss')}`",
        f"- filter aggregate subset: `{str(bool(profile_value(args, 'filter_aggregate_subset'))).lower()}`",
        f"- filter aggregate subset min carriers: `{profile_value(args, 'filter_aggregate_subset_min_selected_carriers')}`",
        f"- filter aggregate subset changed views: `{profile_value(args, 'filter_aggregate_subset_min_changed_unique_views')}`",
        f"- filter aggregate subset changed px frac: `{profile_value(args, 'filter_aggregate_subset_min_changed_pixel_fraction')}`",
        f"- filter aggregate subset full-frame changed px frac: `{profile_value(args, 'filter_aggregate_subset_min_full_frame_changed_pixel_fraction')}`",
        f"- filter aggregate subset dilution min: `{profile_value(args, 'filter_aggregate_subset_min_dilution_adjusted_core_balanced_delta')}`",
        f"- filter aggregate subset full-frame visibility min: `{profile_value(args, 'filter_aggregate_subset_min_full_frame_visibility_adjusted_delta')}`",
        f"- filter aggregate subset prefer full-frame visibility: `{str(bool(profile_value(args, 'filter_aggregate_subset_prefer_full_frame_visibility'))).lower()}`",
        f"- filter aggregate subset carrier shrink: `{str(bool(profile_value(args, 'filter_aggregate_subset_tail_safe_shrink_carriers'))).lower()}`",
        f"- filter aggregate subset shrink scales: `{profile_value(args, 'filter_aggregate_subset_tail_safe_shrink_scales')}`",
        f"- filter aggregate subset shrink floor: `{profile_value(args, 'filter_aggregate_subset_tail_safe_shrink_min_scale')}`",
        f"- filter risk-safe shrink: `{str(bool(profile_value(args, 'filter_risk_safe_shrink_on_train_risk_fail'))).lower()}`",
        f"- candidate-region expansion: `{str(bool(profile_value(args, 'filter_candidate_region_expand_faces'))).lower()}`",
        f"- candidate-region frame-aware ranking: `{str(bool(profile_value(args, 'filter_candidate_region_frame_aware_ranking'))).lower()}`",
        f"- candidate-region frame support min: `{profile_value(args, 'filter_candidate_region_min_frame_support_fraction')}`",
        f"- candidate-region residual mass min: `{profile_value(args, 'filter_candidate_region_min_residual_mass_fraction')}`",
        f"- candidate-region max carriers: `{profile_value(args, 'filter_candidate_region_max_carriers')}`",
        f"- candidate-region pre-refit risk prune: `{str(bool(profile_value(args, 'candidate_region_pre_refit_risk_prune'))).lower()}`",
        f"- candidate-region pre-refit risk bad fraction: `>{profile_value(args, 'candidate_region_pre_refit_risk_max_bad_fraction')}`",
        f"- candidate-region pre-refit risk margins: balanced<-`{profile_value(args, 'candidate_region_pre_refit_risk_balanced_margin')}`, ssim<-`{profile_value(args, 'candidate_region_pre_refit_risk_ssim_margin')}`, lpips>`{profile_value(args, 'candidate_region_pre_refit_risk_lpips_margin')}`",
        f"- candidate-region pre-refit max removed face fraction: `{profile_value(args, 'candidate_region_pre_refit_risk_max_removed_face_fraction')}`",
        f"- candidate-region pre-refit risk shrink: `{str(bool(profile_value(args, 'candidate_region_pre_refit_risk_shrink'))).lower()}`",
        f"- candidate-region pre-refit risk shrink min scale: `{profile_value(args, 'candidate_region_pre_refit_risk_shrink_min_scale')}`",
        f"- candidate-region pre-refit severity-aware shrink: `{str(bool(profile_value(args, 'candidate_region_pre_refit_risk_shrink_severity_aware'))).lower()}`",
        f"- candidate-region pre-refit severity select min: `{profile_value(args, 'candidate_region_pre_refit_risk_shrink_severity_select_min')}`",
        f"- candidate-region pre-refit severity balanced span: `{profile_value(args, 'candidate_region_pre_refit_risk_shrink_severity_balanced_span')}`",
        f"- candidate-region pre-refit shrink tail fraction: `{profile_value(args, 'candidate_region_pre_refit_risk_shrink_tail_fraction')}`",
        f"- candidate-region pre-refit local suppression: `{str(bool(profile_value(args, 'candidate_region_pre_refit_risk_local_suppression'))).lower()}`",
        f"- candidate-region pre-refit local suppression scale: `{profile_value(args, 'candidate_region_pre_refit_risk_local_suppression_scale')}`",
        f"- candidate-region pre-refit local suppression min bad balanced: `{profile_value(args, 'candidate_region_pre_refit_risk_local_suppression_min_bad_balanced')}`",
        f"- candidate-region pre-refit local suppression positive margin: `{profile_value(args, 'candidate_region_pre_refit_risk_local_suppression_positive_margin')}`",
        f"- ELA alpha holdout-safe zero: `{str(bool(profile_value(args, 'ela_alpha_holdout_safe_zero'))).lower()}`",
        f"- ELA alpha tail fraction: `{profile_value(args, 'ela_alpha_risk_tail_fraction')}`",
        f"- ELA alpha max negative fraction: `{profile_value(args, 'ela_alpha_max_negative_gain_fraction')}`",
        f"- ELA alpha min tail gain: `{profile_value(args, 'ela_alpha_min_tail_gain')}`",
        f"- ELA alpha region-risk: `{str(bool(profile_value(args, 'ela_alpha_region_risk_enable'))).lower()}`",
        f"- ELA alpha region-risk objective bad only: `{str(bool(profile_value(args, 'ela_alpha_region_risk_objective_bad_only'))).lower()}`",
        f"- ELA alpha region-risk objective thresholds: balanced<`{profile_value(args, 'ela_alpha_region_risk_objective_max_balanced_delta')}`, ssim<`{profile_value(args, 'ela_alpha_region_risk_objective_max_delta_ssim')}`, lpips>`{profile_value(args, 'ela_alpha_region_risk_objective_min_delta_lpips')}`",
        f"- ELA alpha region-risk min tail gain: `{profile_value(args, 'ela_alpha_region_risk_min_tail_gain')}`",
        f"- ELA alpha region-risk max negative fraction: `{profile_value(args, 'ela_alpha_region_risk_max_negative_fraction')}`",
        f"- ELA alpha region-risk min regions: `{profile_value(args, 'ela_alpha_region_risk_min_regions')}`",
        f"- candidate-owned refit: `{str(bool(profile_value(args, 'candidate_owned_refit'))).lower()}`",
        f"- isolate PhaseJ methods: `{str(bool(args.isolate_phasej_methods)).lower()}`",
        f"- PhaseJ test method: `{phasej_test_method(args)}`",
        f"- PhaseJ train-val method: `{phasej_trainval_method(args)}`",
        f"- dry run: `{str(bool(args.dry_run)).lower()}`",
        f"- command manifest: `{rel(root / 'pipeline_command_manifest.json')}`",
        f"- aggregate subset plan template: `{rel(aggregate_subset_plan_template(args))}`",
        f"- selector plan template: `{rel(selector_plan_template(args))}`",
        "",
        "| scene | decision present | accepted | selected trial | train-val balanced | effective report-only dPSNR | dSSIM | dLPIPS |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row.get("effective_report_only_test_delta", {})
        lines.append(
            f"| {row['scene']} | {str(bool(row.get('present'))).lower()} | "
            f"{str(bool(row.get('accepted'))).lower()} | {row.get('selected_trial', '')} | "
            f"{format_metric(row.get('selected_trainval_balanced_delta'))} | "
            f"{format_metric(delta.get('PSNR'))} | {format_metric(delta.get('SSIM'))} | {format_metric(delta.get('LPIPS'))} |"
        )
    (root / "pipeline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value: Any) -> str:
    try:
        out = float(value)
    except Exception:
        return "n/a"
    return f"{out:+.9f}" if math.isfinite(out) else "n/a"


def main() -> int:
    args = parse_args()
    scenes = scene_list(args.scenes)
    if not scenes:
        raise ValueError("no scenes requested")
    stages = stage_set(args.stages)
    root = output_root(args)
    records: list[CommandRecord] = []

    if "plan" in stages and not str(args.plan_template).strip():
        for scene in scenes:
            record = plan_command(args, scene)
            if format_plan_path(args, scene).is_file() and not bool(args.force):
                record.skipped = True
                record.skip_reason = "candidate plan exists; use --force to rebuild"
            records.append(run_command(record, args))
    elif "plan" in stages:
        records.append(
            CommandRecord(
                stage="plan",
                scene=",".join(scenes),
                command=[],
                log_path=str(command_log(root, "plan", "external_plan_template")),
                output_paths={"candidate_plan_template": plan_template(args)},
                skipped=True,
                skip_reason="--plan_template supplied; plan generation delegated to caller",
                exit_code=0,
            )
        )

    if "filter" in stages:
        for scene in scenes:
            if str(profile_value(args, "filter_region_source")) == "candidate_owned":
                region_record = candidate_region_command(args, scene)
                if format_candidate_region_carrier_path(args, scene).is_file() and not bool(args.force):
                    region_record.skipped = True
                    region_record.skip_reason = "candidate-owned region carriers exist; use --force to rebuild"
                records.append(run_command(region_record, args))

                eval_record = candidate_region_eval_command(args, scene)
                if format_candidate_region_objective_path(args, scene).is_file() and not bool(args.force):
                    eval_record.skipped = True
                    eval_record.skip_reason = "candidate-owned render-region objective exists; use --force to rebuild"
                records.append(run_command(eval_record, args))

                if bool(profile_value(args, "candidate_owned_refit")):
                    if (
                        bool(profile_value(args, "candidate_region_pre_refit_risk_shrink"))
                        and format_refit_plan_path(args, scene).is_file()
                        and not bool(args.force)
                    ):
                        raise RuntimeError(
                            "candidate-owned risk-shrink refit requires a fresh refit artifact; "
                            f"use --force or a fresh --output_root for {scene}"
                        )
                    refit_record = candidate_owned_refit_command(args, scene)
                    if format_refit_plan_path(args, scene).is_file() and not bool(args.force):
                        refit_record.skipped = True
                        refit_record.skip_reason = "candidate-owned refit plan exists; use --force to rebuild"
                    refit_record = run_command(refit_record, args)
                    records.append(refit_record)
                    assert_refit_risk_scale_audit(
                        args,
                        scene,
                        str(refit_record.output_paths.get("pre_refit_face_risk_scale_json", "")),
                    )

                    eval_refit_record = candidate_region_eval_command(args, scene, refit=True)
                    if candidate_region_objective_for_filter(args, scene).is_file() and not bool(args.force):
                        eval_refit_record.skipped = True
                        eval_refit_record.skip_reason = "candidate-owned refit render-region objective exists; use --force to rebuild"
                    records.append(run_command(eval_refit_record, args))

            record = filter_command(args, scene)
            if format_filtered_plan_path(args, scene).is_file() and not bool(args.force):
                record.skipped = True
                record.skip_reason = "filtered candidate plan exists; use --force to rebuild"
            records.append(run_command(record, args))

            if bool(profile_value(args, "filter_aggregate_subset")):
                aggregate_record = aggregate_subset_command(args, scene)
                if format_aggregate_subset_plan_path(args, scene).is_file() and not bool(args.force):
                    aggregate_record.skipped = True
                    aggregate_record.skip_reason = "aggregate subset candidate plan exists; use --force to rebuild"
                records.append(run_command(aggregate_record, args))

    if bool(profile_value(args, "candidate_region_pre_refit_risk_shrink")):
        for scene in scenes:
            apply_candidate_region_pre_refit_risk_shrink_policy(
                args,
                scene,
                format_selector_plan_path(args, scene),
                alpha_purpose="selector",
            )

    if "selector" in stages:
        records.append(run_command(selector_command(args, scenes), args))

    write_manifest(args, records, scenes)
    write_summary(args, records, scenes)
    print(json.dumps({"output_root": str(root), "commands": len(records), "dry_run": bool(args.dry_run)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
