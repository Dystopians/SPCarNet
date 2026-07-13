#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_gate_float(value: float) -> str:
    return f"{float(value):.12f}"


def find_v48_audit(scene: str, roots: list[Path], v48_tag: str) -> Path:
    paths: list[Path] = []
    for root in roots:
        paths.extend(root.glob(f"**/{scene}_{v48_tag}/surface_residual_region_texture_adapter_audit.json"))
    if not paths:
        raise FileNotFoundError(f"missing v48 audit for {scene}")
    return sorted(paths, key=lambda path: len(str(path)))[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one fixed v49b L1-risk fair-noop surface atlas scene.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output_root", default="/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623")
    parser.add_argument(
        "--tag",
        default="v49b_l1risk_fairnoop_autosupport_autocap_guarded_v42calib_region_texture_adapter",
    )
    parser.add_argument(
        "--v48_roots",
        default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623",
    )
    parser.add_argument(
        "--v48_tag",
        default="v48_autosupport_autocap_guarded_v42calib_region_texture_adapter",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--min_policy_val_l1_positive_view_fraction",
        type=float,
        default=1.0,
        help="Train policy-val full-image L1 positive-view fraction gate.",
    )
    parser.add_argument(
        "--min_policy_val_ssim_mean_gain",
        type=float,
        default=0.0,
        help="Train policy-val full-image SSIM mean-gain gate forwarded to the atlas adapter.",
    )
    parser.add_argument(
        "--min_policy_val_ssim_positive_view_fraction",
        type=float,
        default=0.75,
        help="Train policy-val full-image SSIM positive-view fraction gate.",
    )
    parser.add_argument(
        "--min_policy_val_ssim_min_view_gain",
        type=float,
        default=-5.0e-6,
        help="Train policy-val worst-view SSIM gain gate.",
    )
    parser.add_argument(
        "--min_policy_val_l1_mean_gain",
        type=float,
        default=0.0,
        help="Train policy-val full-image L1 mean-gain gate.",
    )
    parser.add_argument(
        "--min_policy_val_l1_min_view_gain",
        type=float,
        default=-5.0e-6,
        help="Train policy-val worst-view L1 gain gate.",
    )
    parser.add_argument(
        "--min_policy_val_l1_cvar20_view_gain",
        type=float,
        default=-5.0e-6,
        help="Train policy-val CVaR20-view L1 gain gate.",
    )
    parser.add_argument(
        "--min_target_changed_fraction",
        type=float,
        default=0.001,
        help="Optional target support coverage gate; set 0 for a train-policy-only decision.",
    )
    parser.add_argument(
        "--support_expansion_max_extra_faces_candidates",
        default="",
        help="Optional comma-separated support footprint ladder for v51-style runs.",
    )
    parser.add_argument(
        "--support_expansion_mode",
        choices=("none", "fit_residual_topk", "target_footprint_residual_debt"),
        default="fit_residual_topk",
        help="Support-expansion mode forwarded to the atlas adapter.",
    )
    parser.add_argument(
        "--support_expansion_max_extra_faces",
        type=int,
        default=2048,
        help="Base support-expansion extra-face budget forwarded to the atlas adapter.",
    )
    parser.add_argument(
        "--texture_size_candidates",
        default="8,16,24,32",
        help="Comma-separated texture-size candidates forwarded to the atlas policy.",
    )
    parser.add_argument(
        "--atlas_empty_bin_fill_mode",
        default="auto_policy",
        help="Fill policy forwarded to the atlas adapter, e.g. auto_policy, face_mean, nearest_observed.",
    )
    parser.add_argument(
        "--max_abs_delta_rgb",
        type=float,
        default=0.12,
        help="Residual RGB cap forwarded to the atlas adapter.",
    )
    parser.add_argument(
        "--max_abs_delta_rgb_candidates",
        default="",
        help="Optional comma-separated residual-cap ladder forwarded to the atlas adapter.",
    )
    parser.add_argument(
        "--surface_multiscale_prior_mode",
        choices=("none", "count_pyramid", "local_patch"),
        default="none",
        help="Optional low-support residual prior forwarded to the atlas adapter.",
    )
    parser.add_argument("--surface_multiscale_prior_block_sizes", default="2,4,8")
    parser.add_argument("--surface_multiscale_prior_min_bin_samples", type=int, default=8)
    parser.add_argument("--surface_multiscale_prior_count_tau", type=float, default=32.0)
    parser.add_argument("--surface_multiscale_prior_blend", type=float, default=1.0)
    parser.add_argument("--surface_multiscale_prior_blend_candidates", default="")
    parser.add_argument(
        "--surface_multiscale_prior_gate_mode",
        choices=("none", "evidence_consistent"),
        default="none",
    )
    parser.add_argument("--surface_multiscale_prior_min_prior_weight", type=float, default=0.0)
    parser.add_argument("--surface_multiscale_prior_min_direct_samples", type=int, default=1)
    parser.add_argument("--surface_multiscale_prior_min_sign_consistency", type=float, default=0.0)
    parser.add_argument("--surface_multiscale_prior_max_mean_variance", type=float, default=-1.0)
    parser.add_argument("--surface_multiscale_prior_min_cosine", type=float, default=0.0)
    parser.add_argument("--enable_policy_val_prior_bin_gain_hybrid", action="store_true")
    parser.add_argument("--prior_bin_gain_hybrid_min_bin_samples", type=int, default=4)
    parser.add_argument("--prior_bin_gain_hybrid_min_views", type=int, default=1)
    parser.add_argument("--prior_bin_gain_hybrid_min_abs_gain", type=float, default=0.0)
    parser.add_argument("--prior_bin_gain_hybrid_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--prior_bin_gain_hybrid_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument("--prior_bin_gain_hybrid_max_profile_bins", type=int, default=0)
    parser.add_argument("--enable_policy_val_source_mixture", action="store_true")
    parser.add_argument("--source_mixture_ridge", type=float, default=1.0e-2)
    parser.add_argument(
        "--source_mixture_ridge_mode",
        choices=("absolute", "adaptive_den"),
        default="absolute",
    )
    parser.add_argument("--source_mixture_min_weight", type=float, default=1.0e-4)
    parser.add_argument("--enable_prior_bin_gain_hybrid_l1_proxy_gate", action="store_true")
    parser.add_argument("--prior_bin_gain_hybrid_min_l1_abs_gain", type=float, default=0.0)
    parser.add_argument("--prior_bin_gain_hybrid_min_l1_relative_gain", type=float, default=-1.0)
    parser.add_argument("--prior_bin_gain_hybrid_min_l1_positive_view_fraction", type=float, default=0.0)
    parser.add_argument("--prior_bin_gain_hybrid_min_l1_min_view_gain", type=float, default=-1.0)
    parser.add_argument("--prior_bin_gain_hybrid_min_l1_cvar20_view_gain", type=float, default=-1.0)
    parser.add_argument("--enable_target_footprint_bin_certificate", action="store_true")
    parser.add_argument("--target_footprint_min_bin_pixels", type=int, default=1)
    parser.add_argument("--target_footprint_min_views", type=int, default=1)
    parser.add_argument("--target_footprint_min_view_fraction", type=float, default=0.0)
    parser.add_argument("--target_footprint_max_views", type=int, default=0)
    parser.add_argument("--enable_target_footprint_tail_risk_certificate", action="store_true")
    parser.add_argument("--target_footprint_tail_risk_all_bins", action="store_true")
    parser.add_argument("--target_footprint_tail_risk_min_positive_view_fraction", type=float, default=1.0)
    parser.add_argument("--target_footprint_tail_risk_min_min_view_gain", type=float, default=0.0)
    parser.add_argument("--target_footprint_tail_risk_min_cvar20_view_gain", type=float, default=0.0)
    parser.add_argument(
        "--view_conditioned_basis_mode",
        choices=("none", "camera_center_linear", "normal_camera_linear"),
        default="none",
        help="Optional persistent view-conditioned residual basis forwarded to the atlas adapter.",
    )
    parser.add_argument(
        "--view_conditioned_basis_guard_mode",
        choices=("none", "policy_val_nonregressive"),
        default="none",
        help="Optional train-policy-val basis fallback guard forwarded to the atlas adapter.",
    )
    parser.add_argument("--view_conditioned_basis_min_bin_samples", type=int, default=16)
    parser.add_argument("--view_conditioned_basis_ridge", type=float, default=1.0e-3)
    parser.add_argument(
        "--view_conditioned_basis_ood_mode",
        choices=("none", "diag_z"),
        default="none",
        help="Optional per-bin feature OOD fallback guard forwarded to the atlas adapter.",
    )
    parser.add_argument("--view_conditioned_basis_ood_max_z", type=float, default=2.5)
    parser.add_argument("--view_conditioned_basis_ood_min_std", type=float, default=5.0e-2)
    parser.add_argument(
        "--teacher_distilled_basis_mode",
        choices=("none", "face_uv_normal_camera_ridge", "face_uv_patch_mixture_ridge"),
        default="none",
        help="Optional shared per-face teacher-distilled residual basis forwarded to the atlas adapter.",
    )
    parser.add_argument(
        "--teacher_distilled_basis_guard_mode",
        choices=("none", "policy_val_nonregressive"),
        default="none",
        help="Optional train-policy-val fallback guard for the teacher-distilled residual basis.",
    )
    parser.add_argument("--teacher_distilled_basis_min_face_samples", type=int, default=1024)
    parser.add_argument("--teacher_distilled_basis_ridge", type=float, default=1.0e-2)
    parser.add_argument("--teacher_distilled_basis_ood_max_z", type=float, default=3.0)
    parser.add_argument("--teacher_distilled_basis_ood_min_std", type=float, default=5.0e-2)
    parser.add_argument(
        "--teacher_distilled_basis_apply_mode",
        choices=("replace_supported", "blend", "fill_empty_only"),
        default="blend",
    )
    parser.add_argument("--teacher_distilled_basis_blend", type=float, default=0.5)
    parser.add_argument("--enable_policy_val_alpha_calibration", action="store_true")
    parser.add_argument("--alpha_calibration_max_alpha", type=float, default=0.5)
    parser.add_argument("--alpha_calibration_multipliers", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--enable_policy_val_local_alpha_calibration", action="store_true")
    parser.add_argument("--local_alpha_calibration_max_alpha", type=float, default=0.5)
    parser.add_argument("--local_alpha_calibration_min_alpha", type=float, default=0.0)
    parser.add_argument("--local_alpha_calibration_bucket_quantiles", default="0.25,0.5,0.75,0.9")
    parser.add_argument("--local_alpha_calibration_bucket_edges", default="")
    parser.add_argument("--local_alpha_calibration_multipliers", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--local_alpha_calibration_min_bucket_samples", type=int, default=1024)
    parser.add_argument(
        "--local_alpha_calibration_norm_mode",
        choices=("l2", "mean_abs"),
        default="l2",
    )
    parser.add_argument("--enable_policy_val_face_alpha_calibration", action="store_true")
    parser.add_argument("--face_alpha_calibration_max_alpha", type=float, default=0.5)
    parser.add_argument("--face_alpha_calibration_min_alpha", type=float, default=0.0)
    parser.add_argument("--face_alpha_calibration_multipliers", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--face_alpha_calibration_min_face_samples", type=int, default=256)
    parser.add_argument("--face_alpha_calibration_shrink_count_tau", type=float, default=0.0)
    parser.add_argument("--face_alpha_calibration_shrink_denominator_tau", type=float, default=0.0)
    parser.add_argument(
        "--face_alpha_calibration_shrink_prior",
        choices=("fallback", "zero"),
        default="fallback",
    )
    parser.add_argument("--enable_policy_val_bin_alpha_calibration", action="store_true")
    parser.add_argument("--bin_alpha_calibration_max_alpha", type=float, default=0.5)
    parser.add_argument("--bin_alpha_calibration_min_alpha", type=float, default=0.0)
    parser.add_argument("--bin_alpha_calibration_multipliers", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--bin_alpha_calibration_min_bin_samples", type=int, default=64)
    parser.add_argument("--bin_alpha_calibration_min_denominator", type=float, default=1.0e-12)
    parser.add_argument("--bin_alpha_calibration_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument("--bin_alpha_calibration_shrink_count_tau", type=float, default=128.0)
    parser.add_argument("--bin_alpha_calibration_shrink_denominator_tau", type=float, default=0.0)
    parser.add_argument(
        "--bin_alpha_calibration_shrink_prior",
        choices=("fallback", "zero"),
        default="fallback",
    )
    parser.add_argument("--bin_alpha_calibration_max_profile_bins", type=int, default=8192)
    parser.add_argument("--enable_policy_val_bin_rgb_alpha_calibration", action="store_true")
    parser.add_argument("--bin_rgb_alpha_calibration_max_alpha", type=float, default=0.5)
    parser.add_argument("--bin_rgb_alpha_calibration_min_alpha", type=float, default=0.0)
    parser.add_argument("--bin_rgb_alpha_calibration_multipliers", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--bin_rgb_alpha_calibration_min_bin_samples", type=int, default=64)
    parser.add_argument("--bin_rgb_alpha_calibration_min_denominator", type=float, default=1.0e-12)
    parser.add_argument("--bin_rgb_alpha_calibration_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument("--bin_rgb_alpha_calibration_shrink_count_tau", type=float, default=128.0)
    parser.add_argument("--bin_rgb_alpha_calibration_shrink_denominator_tau", type=float, default=0.0)
    parser.add_argument(
        "--bin_rgb_alpha_calibration_shrink_prior",
        choices=("fallback", "zero"),
        default="fallback",
    )
    parser.add_argument("--bin_rgb_alpha_calibration_max_profile_bins", type=int, default=8192)
    parser.add_argument("--enable_policy_val_bin_uncertainty_shrink", action="store_true")
    parser.add_argument(
        "--bin_uncertainty_shrink_policy_mode",
        choices=("sparse_positive", "keep_with_downweight"),
        default="sparse_positive",
    )
    parser.add_argument("--bin_uncertainty_shrink_min_bin_samples", type=int, default=64)
    parser.add_argument("--bin_uncertainty_shrink_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--bin_uncertainty_shrink_min_positive_view_fraction", type=float, default=0.75)
    parser.add_argument("--bin_uncertainty_shrink_max_mean_variance", type=float, default=-1.0)
    parser.add_argument("--bin_uncertainty_shrink_min_mean_sign_consistency", type=float, default=0.0)
    parser.add_argument("--bin_uncertainty_shrink_count_tau", type=float, default=128.0)
    parser.add_argument("--bin_uncertainty_shrink_gain_tau", type=float, default=0.01)
    parser.add_argument("--bin_uncertainty_shrink_variance_scale", type=float, default=0.004)
    parser.add_argument("--bin_uncertainty_shrink_sign_power", type=float, default=0.5)
    parser.add_argument("--bin_uncertainty_shrink_min_shrink", type=float, default=0.0)
    parser.add_argument("--bin_uncertainty_shrink_max_shrink", type=float, default=1.0)
    parser.add_argument("--bin_uncertainty_shrink_fallback_shrink", type=float, default=0.0)
    parser.add_argument("--bin_uncertainty_shrink_max_profile_bins", type=int, default=8192)
    parser.add_argument("--enable_policy_val_face_gain_guard", action="store_true")
    parser.add_argument("--face_gain_guard_min_face_samples", type=int, default=256)
    parser.add_argument("--face_gain_guard_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--face_gain_guard_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument("--enable_policy_val_bin_uncertainty_guard", action="store_true")
    parser.add_argument("--bin_uncertainty_guard_min_bin_samples", type=int, default=64)
    parser.add_argument("--bin_uncertainty_guard_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--bin_uncertainty_guard_min_positive_view_fraction", type=float, default=0.75)
    parser.add_argument("--bin_uncertainty_guard_max_mean_variance", type=float, default=-1.0)
    parser.add_argument("--bin_uncertainty_guard_min_mean_sign_consistency", type=float, default=0.0)
    parser.add_argument("--enable_target_support_candidate_selection", action="store_true")
    parser.add_argument("--target_support_prerank_top_k", type=int, default=0)
    parser.add_argument("--target_support_prerank_max_views", type=int, default=0)
    parser.add_argument(
        "--enable_policy_candidate_dominance_pruning",
        action="store_true",
        help="Forward strict equivalent policy-candidate pruning to the atlas adapter.",
    )
    parser.add_argument(
        "--policy_candidate_early_stop_mode",
        choices=("none", "first_accepted"),
        default="none",
        help="Forward optional policy-candidate early-stop mode to the atlas adapter.",
    )
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()
    if not 0.0 <= float(args.prior_bin_gain_hybrid_min_positive_view_fraction) <= 1.0:
        parser.error("--prior_bin_gain_hybrid_min_positive_view_fraction must be in [0, 1]")
    if int(args.prior_bin_gain_hybrid_max_profile_bins) < 0:
        parser.error("--prior_bin_gain_hybrid_max_profile_bins must be >= 0")
    if bool(args.enable_policy_val_source_mixture) and not bool(args.enable_policy_val_prior_bin_gain_hybrid):
        parser.error("--enable_policy_val_source_mixture requires --enable_policy_val_prior_bin_gain_hybrid")
    if float(args.source_mixture_ridge) < 0.0:
        parser.error("--source_mixture_ridge must be >= 0")
    if not 0.0 <= float(args.source_mixture_min_weight) <= 1.0:
        parser.error("--source_mixture_min_weight must be in [0, 1]")
    if float(args.prior_bin_gain_hybrid_min_l1_abs_gain) < 0.0:
        parser.error("--prior_bin_gain_hybrid_min_l1_abs_gain must be >= 0")
    if not 0.0 <= float(args.prior_bin_gain_hybrid_min_l1_positive_view_fraction) <= 1.0:
        parser.error("--prior_bin_gain_hybrid_min_l1_positive_view_fraction must be in [0, 1]")
    if not 0.0 <= float(args.target_footprint_tail_risk_min_positive_view_fraction) <= 1.0:
        parser.error("--target_footprint_tail_risk_min_positive_view_fraction must be in [0, 1]")
    if bool(args.enable_target_footprint_tail_risk_certificate) and not bool(
        args.enable_policy_val_prior_bin_gain_hybrid
    ):
        parser.error(
            "--enable_target_footprint_tail_risk_certificate requires "
            "--enable_policy_val_prior_bin_gain_hybrid"
        )
    if (
        bool(args.enable_target_footprint_tail_risk_certificate)
        and not bool(args.target_footprint_tail_risk_all_bins)
        and not bool(args.enable_target_footprint_bin_certificate)
    ):
        parser.error(
            "--enable_target_footprint_tail_risk_certificate requires "
            "--enable_target_footprint_bin_certificate unless --target_footprint_tail_risk_all_bins is set"
        )

    scene = str(args.scene)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_model = output_root / f"{scene}_{args.tag}"
    log_path = log_dir / f"apply_metrics_{scene}.log"
    method_name = f"ours_26000_{scene}_{args.tag}"

    roots = [Path(item) for item in str(args.v48_roots).split(",") if item]
    audit = read_json(find_v48_audit(scene, roots, str(args.v48_tag)))
    apply_cmd = [
        PYTHON,
        "scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py",
        "--source_model",
        str(audit["source_model"]),
        "--fit_evidence_dir",
        str(audit["fit_evidence_dir"]),
        "--target_evidence_dir",
        str(audit["target_evidence_dir"]),
        "--region_carrier_json",
        str(audit["region_carrier_json"]),
        "--output_model",
        str(output_model),
        "--target_split",
        "test",
        "--base_method_name",
        str(audit["base_method_name"]),
        "--method_name",
        method_name,
        "--residual_rgb_key",
        "teacher_residual_rgb",
        "--residual_l1_key",
        "teacher_residual_l1",
        "--max_carriers",
        "64",
        "--max_faces_per_carrier",
        "128",
        "--max_faces",
        "4096",
        "--min_alpha",
        "0.03",
        "--max_samples_per_view",
        "240000",
        "--max_abs_delta_rgb",
        str(float(args.max_abs_delta_rgb)),
        "--texture_size",
        "16",
        "--texture_size_candidates",
        str(args.texture_size_candidates),
        "--support_expansion_mode",
        str(args.support_expansion_mode),
        "--support_expansion_max_extra_faces",
        str(int(args.support_expansion_max_extra_faces)),
    ]
    if str(args.support_expansion_max_extra_faces_candidates).strip():
        apply_cmd.extend(
            [
                "--support_expansion_max_extra_faces_candidates",
                str(args.support_expansion_max_extra_faces_candidates),
            ]
        )
    if bool(args.enable_policy_val_alpha_calibration):
        apply_cmd.extend(
            [
                "--enable_policy_val_alpha_calibration",
                "--alpha_calibration_max_alpha",
                str(float(args.alpha_calibration_max_alpha)),
                "--alpha_calibration_multipliers",
                str(args.alpha_calibration_multipliers),
            ]
        )
    if bool(args.enable_policy_val_local_alpha_calibration):
        apply_cmd.extend(
            [
                "--enable_policy_val_local_alpha_calibration",
                "--local_alpha_calibration_max_alpha",
                str(float(args.local_alpha_calibration_max_alpha)),
                "--local_alpha_calibration_min_alpha",
                str(float(args.local_alpha_calibration_min_alpha)),
                "--local_alpha_calibration_bucket_quantiles",
                str(args.local_alpha_calibration_bucket_quantiles),
                "--local_alpha_calibration_multipliers",
                str(args.local_alpha_calibration_multipliers),
                "--local_alpha_calibration_min_bucket_samples",
                str(int(args.local_alpha_calibration_min_bucket_samples)),
                "--local_alpha_calibration_norm_mode",
                str(args.local_alpha_calibration_norm_mode),
            ]
        )
        if str(args.local_alpha_calibration_bucket_edges).strip():
            apply_cmd.extend(
                [
                    "--local_alpha_calibration_bucket_edges",
                    str(args.local_alpha_calibration_bucket_edges),
                ]
            )
    if bool(args.enable_policy_val_face_alpha_calibration):
        apply_cmd.extend(
            [
                "--enable_policy_val_face_alpha_calibration",
                "--face_alpha_calibration_max_alpha",
                str(float(args.face_alpha_calibration_max_alpha)),
                "--face_alpha_calibration_min_alpha",
                str(float(args.face_alpha_calibration_min_alpha)),
                "--face_alpha_calibration_multipliers",
                str(args.face_alpha_calibration_multipliers),
                "--face_alpha_calibration_min_face_samples",
                str(int(args.face_alpha_calibration_min_face_samples)),
                "--face_alpha_calibration_shrink_count_tau",
                str(float(args.face_alpha_calibration_shrink_count_tau)),
                "--face_alpha_calibration_shrink_denominator_tau",
                str(float(args.face_alpha_calibration_shrink_denominator_tau)),
                "--face_alpha_calibration_shrink_prior",
                str(args.face_alpha_calibration_shrink_prior),
            ]
        )
    if bool(args.enable_policy_val_bin_alpha_calibration):
        apply_cmd.extend(
            [
                "--enable_policy_val_bin_alpha_calibration",
                "--bin_alpha_calibration_max_alpha",
                str(float(args.bin_alpha_calibration_max_alpha)),
                "--bin_alpha_calibration_min_alpha",
                str(float(args.bin_alpha_calibration_min_alpha)),
                "--bin_alpha_calibration_multipliers",
                str(args.bin_alpha_calibration_multipliers),
                "--bin_alpha_calibration_min_bin_samples",
                str(int(args.bin_alpha_calibration_min_bin_samples)),
                "--bin_alpha_calibration_min_denominator",
                str(float(args.bin_alpha_calibration_min_denominator)),
                "--bin_alpha_calibration_min_positive_view_fraction",
                str(float(args.bin_alpha_calibration_min_positive_view_fraction)),
                "--bin_alpha_calibration_shrink_count_tau",
                str(float(args.bin_alpha_calibration_shrink_count_tau)),
                "--bin_alpha_calibration_shrink_denominator_tau",
                str(float(args.bin_alpha_calibration_shrink_denominator_tau)),
                "--bin_alpha_calibration_shrink_prior",
                str(args.bin_alpha_calibration_shrink_prior),
                "--bin_alpha_calibration_max_profile_bins",
                str(int(args.bin_alpha_calibration_max_profile_bins)),
            ]
        )
    if bool(args.enable_policy_val_bin_rgb_alpha_calibration):
        apply_cmd.extend(
            [
                "--enable_policy_val_bin_rgb_alpha_calibration",
                "--bin_rgb_alpha_calibration_max_alpha",
                str(float(args.bin_rgb_alpha_calibration_max_alpha)),
                "--bin_rgb_alpha_calibration_min_alpha",
                str(float(args.bin_rgb_alpha_calibration_min_alpha)),
                "--bin_rgb_alpha_calibration_multipliers",
                str(args.bin_rgb_alpha_calibration_multipliers),
                "--bin_rgb_alpha_calibration_min_bin_samples",
                str(int(args.bin_rgb_alpha_calibration_min_bin_samples)),
                "--bin_rgb_alpha_calibration_min_denominator",
                str(float(args.bin_rgb_alpha_calibration_min_denominator)),
                "--bin_rgb_alpha_calibration_min_positive_view_fraction",
                str(float(args.bin_rgb_alpha_calibration_min_positive_view_fraction)),
                "--bin_rgb_alpha_calibration_shrink_count_tau",
                str(float(args.bin_rgb_alpha_calibration_shrink_count_tau)),
                "--bin_rgb_alpha_calibration_shrink_denominator_tau",
                str(float(args.bin_rgb_alpha_calibration_shrink_denominator_tau)),
                "--bin_rgb_alpha_calibration_shrink_prior",
                str(args.bin_rgb_alpha_calibration_shrink_prior),
                "--bin_rgb_alpha_calibration_max_profile_bins",
                str(int(args.bin_rgb_alpha_calibration_max_profile_bins)),
            ]
        )
    if bool(args.enable_policy_val_bin_uncertainty_shrink):
        apply_cmd.extend(
            [
                "--enable_policy_val_bin_uncertainty_shrink",
                "--bin_uncertainty_shrink_policy_mode",
                str(args.bin_uncertainty_shrink_policy_mode),
                "--bin_uncertainty_shrink_min_bin_samples",
                str(int(args.bin_uncertainty_shrink_min_bin_samples)),
                "--bin_uncertainty_shrink_min_relative_gain",
                str(float(args.bin_uncertainty_shrink_min_relative_gain)),
                "--bin_uncertainty_shrink_min_positive_view_fraction",
                str(float(args.bin_uncertainty_shrink_min_positive_view_fraction)),
                "--bin_uncertainty_shrink_max_mean_variance",
                str(float(args.bin_uncertainty_shrink_max_mean_variance)),
                "--bin_uncertainty_shrink_min_mean_sign_consistency",
                str(float(args.bin_uncertainty_shrink_min_mean_sign_consistency)),
                "--bin_uncertainty_shrink_count_tau",
                str(float(args.bin_uncertainty_shrink_count_tau)),
                "--bin_uncertainty_shrink_gain_tau",
                str(float(args.bin_uncertainty_shrink_gain_tau)),
                "--bin_uncertainty_shrink_variance_scale",
                str(float(args.bin_uncertainty_shrink_variance_scale)),
                "--bin_uncertainty_shrink_sign_power",
                str(float(args.bin_uncertainty_shrink_sign_power)),
                "--bin_uncertainty_shrink_min_shrink",
                str(float(args.bin_uncertainty_shrink_min_shrink)),
                "--bin_uncertainty_shrink_max_shrink",
                str(float(args.bin_uncertainty_shrink_max_shrink)),
                "--bin_uncertainty_shrink_fallback_shrink",
                str(float(args.bin_uncertainty_shrink_fallback_shrink)),
                "--bin_uncertainty_shrink_max_profile_bins",
                str(int(args.bin_uncertainty_shrink_max_profile_bins)),
            ]
        )
    if bool(args.enable_policy_val_face_gain_guard):
        apply_cmd.extend(
            [
                "--enable_policy_val_face_gain_guard",
                "--face_gain_guard_min_face_samples",
                str(int(args.face_gain_guard_min_face_samples)),
                "--face_gain_guard_min_relative_gain",
                str(float(args.face_gain_guard_min_relative_gain)),
                "--face_gain_guard_min_positive_view_fraction",
                str(float(args.face_gain_guard_min_positive_view_fraction)),
            ]
        )
    if bool(args.enable_policy_val_bin_uncertainty_guard):
        apply_cmd.extend(
            [
                "--enable_policy_val_bin_uncertainty_guard",
                "--bin_uncertainty_guard_min_bin_samples",
                str(int(args.bin_uncertainty_guard_min_bin_samples)),
                "--bin_uncertainty_guard_min_relative_gain",
                str(float(args.bin_uncertainty_guard_min_relative_gain)),
                "--bin_uncertainty_guard_min_positive_view_fraction",
                str(float(args.bin_uncertainty_guard_min_positive_view_fraction)),
                "--bin_uncertainty_guard_max_mean_variance",
                str(float(args.bin_uncertainty_guard_max_mean_variance)),
                "--bin_uncertainty_guard_min_mean_sign_consistency",
                str(float(args.bin_uncertainty_guard_min_mean_sign_consistency)),
            ]
        )
    if bool(args.enable_target_support_candidate_selection):
        apply_cmd.append("--enable_target_support_candidate_selection")
    if int(args.target_support_prerank_top_k) > 0:
        apply_cmd.extend(
            [
                "--target_support_prerank_top_k",
                str(int(args.target_support_prerank_top_k)),
                "--target_support_prerank_max_views",
                str(int(args.target_support_prerank_max_views)),
            ]
        )
    if bool(args.enable_policy_candidate_dominance_pruning):
        apply_cmd.append("--enable_policy_candidate_dominance_pruning")
    if str(args.policy_candidate_early_stop_mode) != "none":
        apply_cmd.extend(
            [
                "--policy_candidate_early_stop_mode",
                str(args.policy_candidate_early_stop_mode),
            ]
        )
    if bool(args.enable_policy_val_prior_bin_gain_hybrid):
        apply_cmd.extend(
            [
                "--enable_policy_val_prior_bin_gain_hybrid",
                "--prior_bin_gain_hybrid_min_bin_samples",
                str(int(args.prior_bin_gain_hybrid_min_bin_samples)),
                "--prior_bin_gain_hybrid_min_views",
                str(int(args.prior_bin_gain_hybrid_min_views)),
                "--prior_bin_gain_hybrid_min_abs_gain",
                str(float(args.prior_bin_gain_hybrid_min_abs_gain)),
                "--prior_bin_gain_hybrid_min_relative_gain",
                str(float(args.prior_bin_gain_hybrid_min_relative_gain)),
                "--prior_bin_gain_hybrid_min_positive_view_fraction",
                str(float(args.prior_bin_gain_hybrid_min_positive_view_fraction)),
                "--prior_bin_gain_hybrid_max_profile_bins",
                str(int(args.prior_bin_gain_hybrid_max_profile_bins)),
            ]
        )
        if bool(args.enable_policy_val_source_mixture):
            apply_cmd.extend(
                [
                    "--enable_policy_val_source_mixture",
                    "--source_mixture_ridge",
                    fmt_gate_float(args.source_mixture_ridge),
                    "--source_mixture_ridge_mode",
                    str(args.source_mixture_ridge_mode),
                    "--source_mixture_min_weight",
                    fmt_gate_float(args.source_mixture_min_weight),
                ]
            )
        if bool(args.enable_prior_bin_gain_hybrid_l1_proxy_gate):
            apply_cmd.extend(
                [
                    "--enable_prior_bin_gain_hybrid_l1_proxy_gate",
                    "--prior_bin_gain_hybrid_min_l1_abs_gain",
                    fmt_gate_float(float(args.prior_bin_gain_hybrid_min_l1_abs_gain)),
                    "--prior_bin_gain_hybrid_min_l1_relative_gain",
                    fmt_gate_float(float(args.prior_bin_gain_hybrid_min_l1_relative_gain)),
                    "--prior_bin_gain_hybrid_min_l1_positive_view_fraction",
                    fmt_gate_float(float(args.prior_bin_gain_hybrid_min_l1_positive_view_fraction)),
                    "--prior_bin_gain_hybrid_min_l1_min_view_gain",
                    fmt_gate_float(float(args.prior_bin_gain_hybrid_min_l1_min_view_gain)),
                    "--prior_bin_gain_hybrid_min_l1_cvar20_view_gain",
                    fmt_gate_float(float(args.prior_bin_gain_hybrid_min_l1_cvar20_view_gain)),
                ]
            )
    if bool(args.enable_target_footprint_bin_certificate):
        apply_cmd.extend(
            [
                "--enable_target_footprint_bin_certificate",
                "--target_footprint_min_bin_pixels",
                str(int(args.target_footprint_min_bin_pixels)),
                "--target_footprint_min_views",
                str(int(args.target_footprint_min_views)),
                "--target_footprint_min_view_fraction",
                str(float(args.target_footprint_min_view_fraction)),
                "--target_footprint_max_views",
                str(int(args.target_footprint_max_views)),
            ]
        )
    if bool(args.enable_target_footprint_tail_risk_certificate):
        apply_cmd.extend(
            [
                "--enable_target_footprint_tail_risk_certificate",
                "--target_footprint_tail_risk_min_positive_view_fraction",
                fmt_gate_float(args.target_footprint_tail_risk_min_positive_view_fraction),
                "--target_footprint_tail_risk_min_min_view_gain",
                fmt_gate_float(args.target_footprint_tail_risk_min_min_view_gain),
                "--target_footprint_tail_risk_min_cvar20_view_gain",
                fmt_gate_float(args.target_footprint_tail_risk_min_cvar20_view_gain),
            ]
        )
    if bool(args.target_footprint_tail_risk_all_bins):
        apply_cmd.append("--target_footprint_tail_risk_all_bins")
    apply_cmd.extend(
        [
            "--support_expansion_min_face_samples",
            "128",
            "--support_expansion_min_mean_l1",
            "0.003",
            "--policy_val_stride",
            "4",
            "--alpha_grid",
            "0,0.015625,0.03125,0.0625,0.125",
            "--min_l1",
            "0.001",
            "--min_atlas_face_samples",
            "32",
            "--atlas_confidence_mode",
            "count_var_sign",
            "--atlas_confidence_count_scale",
            "2.0",
            "--atlas_confidence_empty_bin",
            "0.5",
            "--atlas_confidence_variance_scale",
            "0.004",
            "--atlas_confidence_sign_power",
            "0.5",
            "--atlas_confidence_face_sample_scale",
            "256",
            "--min_atlas_confidence",
            "0.02",
            "--atlas_lowpass_passes",
            "1",
            "--surface_multiscale_prior_mode",
            str(args.surface_multiscale_prior_mode),
            "--surface_multiscale_prior_block_sizes",
            str(args.surface_multiscale_prior_block_sizes),
            "--surface_multiscale_prior_min_bin_samples",
            str(int(args.surface_multiscale_prior_min_bin_samples)),
            "--surface_multiscale_prior_count_tau",
            str(float(args.surface_multiscale_prior_count_tau)),
            "--surface_multiscale_prior_blend",
            str(float(args.surface_multiscale_prior_blend)),
            "--surface_multiscale_prior_gate_mode",
            str(args.surface_multiscale_prior_gate_mode),
            "--surface_multiscale_prior_min_prior_weight",
            str(float(args.surface_multiscale_prior_min_prior_weight)),
            "--surface_multiscale_prior_min_direct_samples",
            str(int(args.surface_multiscale_prior_min_direct_samples)),
            "--surface_multiscale_prior_min_sign_consistency",
            str(float(args.surface_multiscale_prior_min_sign_consistency)),
            "--surface_multiscale_prior_max_mean_variance",
            str(float(args.surface_multiscale_prior_max_mean_variance)),
            "--surface_multiscale_prior_min_cosine",
            str(float(args.surface_multiscale_prior_min_cosine)),
            "--view_conditioned_basis_mode",
            str(args.view_conditioned_basis_mode),
            "--view_conditioned_basis_guard_mode",
            str(args.view_conditioned_basis_guard_mode),
            "--view_conditioned_basis_min_bin_samples",
            str(int(args.view_conditioned_basis_min_bin_samples)),
            "--view_conditioned_basis_ridge",
            str(float(args.view_conditioned_basis_ridge)),
            "--view_conditioned_basis_ood_mode",
            str(args.view_conditioned_basis_ood_mode),
            "--view_conditioned_basis_ood_max_z",
            str(float(args.view_conditioned_basis_ood_max_z)),
            "--view_conditioned_basis_ood_min_std",
            str(float(args.view_conditioned_basis_ood_min_std)),
            "--teacher_distilled_basis_mode",
            str(args.teacher_distilled_basis_mode),
            "--teacher_distilled_basis_guard_mode",
            str(args.teacher_distilled_basis_guard_mode),
            "--teacher_distilled_basis_min_face_samples",
            str(int(args.teacher_distilled_basis_min_face_samples)),
            "--teacher_distilled_basis_ridge",
            str(float(args.teacher_distilled_basis_ridge)),
            "--teacher_distilled_basis_ood_max_z",
            str(float(args.teacher_distilled_basis_ood_max_z)),
            "--teacher_distilled_basis_ood_min_std",
            str(float(args.teacher_distilled_basis_ood_min_std)),
            "--teacher_distilled_basis_apply_mode",
            str(args.teacher_distilled_basis_apply_mode),
            "--teacher_distilled_basis_blend",
            str(float(args.teacher_distilled_basis_blend)),
            "--atlas_empty_bin_fill_mode",
            str(args.atlas_empty_bin_fill_mode),
            "--min_policy_val_samples",
            "1024",
            "--select_alpha_by_risk_gate",
            "--min_policy_val_relative_gain",
            "0.0002",
            "--min_policy_val_positive_view_fraction",
            "1.0",
            "--min_policy_val_cvar20_relative_gain",
            "0.0",
            "--min_policy_val_min_view_relative_gain",
            "0.0",
            "--enable_policy_val_image_ssim_gate",
            "--policy_val_ssim_max_size",
            "512",
            "--min_policy_val_ssim_mean_gain",
            fmt_gate_float(float(args.min_policy_val_ssim_mean_gain)),
            "--min_policy_val_ssim_positive_view_fraction",
            fmt_gate_float(float(args.min_policy_val_ssim_positive_view_fraction)),
            "--min_policy_val_ssim_min_view_gain",
            fmt_gate_float(float(args.min_policy_val_ssim_min_view_gain)),
            "--enable_policy_val_image_l1_gate",
            "--policy_val_l1_max_size",
            "512",
            "--min_policy_val_l1_mean_gain",
            fmt_gate_float(float(args.min_policy_val_l1_mean_gain)),
            "--min_policy_val_l1_positive_view_fraction",
            fmt_gate_float(float(args.min_policy_val_l1_positive_view_fraction)),
            "--min_policy_val_l1_min_view_gain",
            fmt_gate_float(float(args.min_policy_val_l1_min_view_gain)),
            "--min_policy_val_l1_cvar20_view_gain",
            fmt_gate_float(float(args.min_policy_val_l1_cvar20_view_gain)),
            "--min_target_changed_fraction",
            str(float(args.min_target_changed_fraction)),
            "--write_noop_on_reject",
            "--noop_fallback_source",
            "target_evidence",
        ]
    )
    if str(args.surface_multiscale_prior_blend_candidates).strip():
        apply_cmd.extend(
            [
                "--surface_multiscale_prior_blend_candidates",
                str(args.surface_multiscale_prior_blend_candidates),
            ]
        )
    if str(args.max_abs_delta_rgb_candidates).strip():
        apply_cmd.extend(
            [
                "--max_abs_delta_rgb_candidates",
                str(args.max_abs_delta_rgb_candidates),
            ]
        )
    if bool(args.force):
        apply_cmd.append("--force")
    metrics_cmd = [PYTHON, "metrics.py", "-m", str(output_model)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    run = None
    if str(args.wandb_project).strip():
        import wandb

        run = wandb.init(
            project=str(args.wandb_project),
            name=str(args.wandb_run_name or f"{scene}_{args.tag}"),
            group=str(args.wandb_group or "surface_atlas_scene"),
            mode=str(args.wandb_mode),
            config={
                "scene": scene,
                "tag": str(args.tag),
                "method_name": method_name,
                "output_model": str(output_model),
                "enable_policy_val_alpha_calibration": bool(args.enable_policy_val_alpha_calibration),
                "alpha_calibration_max_alpha": float(args.alpha_calibration_max_alpha),
                "alpha_calibration_multipliers": str(args.alpha_calibration_multipliers),
                "enable_policy_val_local_alpha_calibration": bool(
                    args.enable_policy_val_local_alpha_calibration
                ),
                "local_alpha_calibration_max_alpha": float(args.local_alpha_calibration_max_alpha),
                "local_alpha_calibration_min_alpha": float(args.local_alpha_calibration_min_alpha),
                "local_alpha_calibration_bucket_quantiles": str(args.local_alpha_calibration_bucket_quantiles),
                "local_alpha_calibration_bucket_edges": str(args.local_alpha_calibration_bucket_edges),
                "local_alpha_calibration_multipliers": str(args.local_alpha_calibration_multipliers),
                "local_alpha_calibration_min_bucket_samples": int(
                    args.local_alpha_calibration_min_bucket_samples
                ),
                "local_alpha_calibration_norm_mode": str(args.local_alpha_calibration_norm_mode),
                "enable_policy_val_face_alpha_calibration": bool(
                    args.enable_policy_val_face_alpha_calibration
                ),
                "face_alpha_calibration_max_alpha": float(args.face_alpha_calibration_max_alpha),
                "face_alpha_calibration_min_alpha": float(args.face_alpha_calibration_min_alpha),
                "face_alpha_calibration_multipliers": str(args.face_alpha_calibration_multipliers),
                "face_alpha_calibration_min_face_samples": int(
                    args.face_alpha_calibration_min_face_samples
                ),
                "face_alpha_calibration_shrink_count_tau": float(
                    args.face_alpha_calibration_shrink_count_tau
                ),
                "face_alpha_calibration_shrink_denominator_tau": float(
                    args.face_alpha_calibration_shrink_denominator_tau
                ),
                "face_alpha_calibration_shrink_prior": str(args.face_alpha_calibration_shrink_prior),
                "enable_policy_val_bin_alpha_calibration": bool(
                    args.enable_policy_val_bin_alpha_calibration
                ),
                "bin_alpha_calibration_max_alpha": float(args.bin_alpha_calibration_max_alpha),
                "bin_alpha_calibration_min_alpha": float(args.bin_alpha_calibration_min_alpha),
                "bin_alpha_calibration_multipliers": str(args.bin_alpha_calibration_multipliers),
                "bin_alpha_calibration_min_bin_samples": int(
                    args.bin_alpha_calibration_min_bin_samples
                ),
                "bin_alpha_calibration_min_denominator": float(
                    args.bin_alpha_calibration_min_denominator
                ),
                "bin_alpha_calibration_min_positive_view_fraction": float(
                    args.bin_alpha_calibration_min_positive_view_fraction
                ),
                "bin_alpha_calibration_shrink_count_tau": float(
                    args.bin_alpha_calibration_shrink_count_tau
                ),
                "bin_alpha_calibration_shrink_denominator_tau": float(
                    args.bin_alpha_calibration_shrink_denominator_tau
                ),
                "bin_alpha_calibration_shrink_prior": str(args.bin_alpha_calibration_shrink_prior),
                "bin_alpha_calibration_max_profile_bins": int(
                    args.bin_alpha_calibration_max_profile_bins
                ),
                "enable_policy_val_bin_rgb_alpha_calibration": bool(
                    args.enable_policy_val_bin_rgb_alpha_calibration
                ),
                "bin_rgb_alpha_calibration_max_alpha": float(args.bin_rgb_alpha_calibration_max_alpha),
                "bin_rgb_alpha_calibration_min_alpha": float(args.bin_rgb_alpha_calibration_min_alpha),
                "bin_rgb_alpha_calibration_multipliers": str(args.bin_rgb_alpha_calibration_multipliers),
                "bin_rgb_alpha_calibration_min_bin_samples": int(
                    args.bin_rgb_alpha_calibration_min_bin_samples
                ),
                "bin_rgb_alpha_calibration_min_denominator": float(
                    args.bin_rgb_alpha_calibration_min_denominator
                ),
                "bin_rgb_alpha_calibration_min_positive_view_fraction": float(
                    args.bin_rgb_alpha_calibration_min_positive_view_fraction
                ),
                "bin_rgb_alpha_calibration_shrink_count_tau": float(
                    args.bin_rgb_alpha_calibration_shrink_count_tau
                ),
                "bin_rgb_alpha_calibration_shrink_denominator_tau": float(
                    args.bin_rgb_alpha_calibration_shrink_denominator_tau
                ),
                "bin_rgb_alpha_calibration_shrink_prior": str(
                    args.bin_rgb_alpha_calibration_shrink_prior
                ),
                "bin_rgb_alpha_calibration_max_profile_bins": int(
                    args.bin_rgb_alpha_calibration_max_profile_bins
                ),
                "enable_policy_val_bin_uncertainty_shrink": bool(
                    args.enable_policy_val_bin_uncertainty_shrink
                ),
                "bin_uncertainty_shrink_policy_mode": str(
                    args.bin_uncertainty_shrink_policy_mode
                ),
                "bin_uncertainty_shrink_min_bin_samples": int(
                    args.bin_uncertainty_shrink_min_bin_samples
                ),
                "bin_uncertainty_shrink_min_relative_gain": float(
                    args.bin_uncertainty_shrink_min_relative_gain
                ),
                "bin_uncertainty_shrink_min_positive_view_fraction": float(
                    args.bin_uncertainty_shrink_min_positive_view_fraction
                ),
                "bin_uncertainty_shrink_max_mean_variance": float(
                    args.bin_uncertainty_shrink_max_mean_variance
                ),
                "bin_uncertainty_shrink_min_mean_sign_consistency": float(
                    args.bin_uncertainty_shrink_min_mean_sign_consistency
                ),
                "bin_uncertainty_shrink_count_tau": float(args.bin_uncertainty_shrink_count_tau),
                "bin_uncertainty_shrink_gain_tau": float(args.bin_uncertainty_shrink_gain_tau),
                "bin_uncertainty_shrink_variance_scale": float(
                    args.bin_uncertainty_shrink_variance_scale
                ),
                "bin_uncertainty_shrink_sign_power": float(args.bin_uncertainty_shrink_sign_power),
                "bin_uncertainty_shrink_min_shrink": float(args.bin_uncertainty_shrink_min_shrink),
                "bin_uncertainty_shrink_max_shrink": float(args.bin_uncertainty_shrink_max_shrink),
                "bin_uncertainty_shrink_fallback_shrink": float(
                    args.bin_uncertainty_shrink_fallback_shrink
                ),
                "bin_uncertainty_shrink_max_profile_bins": int(
                    args.bin_uncertainty_shrink_max_profile_bins
                ),
                "enable_policy_val_face_gain_guard": bool(args.enable_policy_val_face_gain_guard),
                "face_gain_guard_min_face_samples": int(args.face_gain_guard_min_face_samples),
                "face_gain_guard_min_relative_gain": float(args.face_gain_guard_min_relative_gain),
                "face_gain_guard_min_positive_view_fraction": float(
                    args.face_gain_guard_min_positive_view_fraction
                ),
                "enable_policy_val_bin_uncertainty_guard": bool(
                    args.enable_policy_val_bin_uncertainty_guard
                ),
                "bin_uncertainty_guard_min_bin_samples": int(
                    args.bin_uncertainty_guard_min_bin_samples
                ),
                "bin_uncertainty_guard_min_relative_gain": float(
                    args.bin_uncertainty_guard_min_relative_gain
                ),
                "bin_uncertainty_guard_min_positive_view_fraction": float(
                    args.bin_uncertainty_guard_min_positive_view_fraction
                ),
                "bin_uncertainty_guard_max_mean_variance": float(
                    args.bin_uncertainty_guard_max_mean_variance
                ),
                "bin_uncertainty_guard_min_mean_sign_consistency": float(
                    args.bin_uncertainty_guard_min_mean_sign_consistency
                ),
                "enable_target_support_candidate_selection": bool(
                    args.enable_target_support_candidate_selection
                ),
                "target_support_prerank_top_k": int(args.target_support_prerank_top_k),
                "target_support_prerank_max_views": int(args.target_support_prerank_max_views),
                "enable_policy_candidate_dominance_pruning": bool(
                    args.enable_policy_candidate_dominance_pruning
                ),
                "policy_candidate_early_stop_mode": str(args.policy_candidate_early_stop_mode),
                "support_expansion_max_extra_faces_candidates": str(
                    args.support_expansion_max_extra_faces_candidates
                ),
                "support_expansion_mode": str(args.support_expansion_mode),
                "support_expansion_max_extra_faces": int(args.support_expansion_max_extra_faces),
                "texture_size_candidates": str(args.texture_size_candidates),
                "atlas_empty_bin_fill_mode": str(args.atlas_empty_bin_fill_mode),
                "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
                "max_abs_delta_rgb_candidates": str(args.max_abs_delta_rgb_candidates),
                "min_policy_val_ssim_mean_gain": float(args.min_policy_val_ssim_mean_gain),
                "min_policy_val_ssim_positive_view_fraction": float(
                    args.min_policy_val_ssim_positive_view_fraction
                ),
                "min_policy_val_ssim_min_view_gain": float(args.min_policy_val_ssim_min_view_gain),
                "min_policy_val_l1_mean_gain": float(args.min_policy_val_l1_mean_gain),
                "min_policy_val_l1_positive_view_fraction": float(
                    args.min_policy_val_l1_positive_view_fraction
                ),
                "min_policy_val_l1_min_view_gain": float(args.min_policy_val_l1_min_view_gain),
                "min_policy_val_l1_cvar20_view_gain": float(
                    args.min_policy_val_l1_cvar20_view_gain
                ),
                "surface_multiscale_prior_mode": str(args.surface_multiscale_prior_mode),
                "surface_multiscale_prior_block_sizes": str(args.surface_multiscale_prior_block_sizes),
                "surface_multiscale_prior_min_bin_samples": int(
                    args.surface_multiscale_prior_min_bin_samples
                ),
                "surface_multiscale_prior_count_tau": float(args.surface_multiscale_prior_count_tau),
                "surface_multiscale_prior_blend": float(args.surface_multiscale_prior_blend),
                "surface_multiscale_prior_blend_candidates": str(
                    args.surface_multiscale_prior_blend_candidates
                ),
                "surface_multiscale_prior_gate_mode": str(args.surface_multiscale_prior_gate_mode),
                "surface_multiscale_prior_min_prior_weight": float(
                    args.surface_multiscale_prior_min_prior_weight
                ),
                "surface_multiscale_prior_min_direct_samples": int(
                    args.surface_multiscale_prior_min_direct_samples
                ),
                "surface_multiscale_prior_min_sign_consistency": float(
                    args.surface_multiscale_prior_min_sign_consistency
                ),
                "surface_multiscale_prior_max_mean_variance": float(
                    args.surface_multiscale_prior_max_mean_variance
                ),
                "surface_multiscale_prior_min_cosine": float(args.surface_multiscale_prior_min_cosine),
                "enable_policy_val_prior_bin_gain_hybrid": bool(
                    args.enable_policy_val_prior_bin_gain_hybrid
                ),
                "prior_bin_gain_hybrid_min_bin_samples": int(
                    args.prior_bin_gain_hybrid_min_bin_samples
                ),
                "prior_bin_gain_hybrid_min_views": int(args.prior_bin_gain_hybrid_min_views),
                "prior_bin_gain_hybrid_min_abs_gain": float(args.prior_bin_gain_hybrid_min_abs_gain),
                "prior_bin_gain_hybrid_min_relative_gain": float(
                    args.prior_bin_gain_hybrid_min_relative_gain
                ),
                "prior_bin_gain_hybrid_min_positive_view_fraction": float(
                    args.prior_bin_gain_hybrid_min_positive_view_fraction
                ),
                "prior_bin_gain_hybrid_max_profile_bins": int(
                    args.prior_bin_gain_hybrid_max_profile_bins
                ),
                "enable_policy_val_source_mixture": bool(args.enable_policy_val_source_mixture),
                "source_mixture_ridge_mode": str(args.source_mixture_ridge_mode),
                "source_mixture_ridge": float(args.source_mixture_ridge),
                "source_mixture_min_weight": float(args.source_mixture_min_weight),
                "enable_prior_bin_gain_hybrid_l1_proxy_gate": bool(
                    args.enable_prior_bin_gain_hybrid_l1_proxy_gate
                ),
                "prior_bin_gain_hybrid_min_l1_abs_gain": float(
                    args.prior_bin_gain_hybrid_min_l1_abs_gain
                ),
                "prior_bin_gain_hybrid_min_l1_relative_gain": float(
                    args.prior_bin_gain_hybrid_min_l1_relative_gain
                ),
                "prior_bin_gain_hybrid_min_l1_positive_view_fraction": float(
                    args.prior_bin_gain_hybrid_min_l1_positive_view_fraction
                ),
                "prior_bin_gain_hybrid_min_l1_min_view_gain": float(
                    args.prior_bin_gain_hybrid_min_l1_min_view_gain
                ),
                "prior_bin_gain_hybrid_min_l1_cvar20_view_gain": float(
                    args.prior_bin_gain_hybrid_min_l1_cvar20_view_gain
                ),
                "enable_target_footprint_bin_certificate": bool(
                    args.enable_target_footprint_bin_certificate
                ),
                "target_footprint_min_bin_pixels": int(args.target_footprint_min_bin_pixels),
                "target_footprint_min_views": int(args.target_footprint_min_views),
                "target_footprint_min_view_fraction": float(args.target_footprint_min_view_fraction),
                "target_footprint_max_views": int(args.target_footprint_max_views),
                "enable_target_footprint_tail_risk_certificate": bool(
                    args.enable_target_footprint_tail_risk_certificate
                ),
                "target_footprint_tail_risk_all_bins": bool(
                    args.target_footprint_tail_risk_all_bins
                ),
                "target_footprint_tail_risk_min_positive_view_fraction": float(
                    args.target_footprint_tail_risk_min_positive_view_fraction
                ),
                "target_footprint_tail_risk_min_min_view_gain": float(
                    args.target_footprint_tail_risk_min_min_view_gain
                ),
                "target_footprint_tail_risk_min_cvar20_view_gain": float(
                    args.target_footprint_tail_risk_min_cvar20_view_gain
                ),
                "view_conditioned_basis_mode": str(args.view_conditioned_basis_mode),
                "view_conditioned_basis_guard_mode": str(args.view_conditioned_basis_guard_mode),
                "view_conditioned_basis_min_bin_samples": int(
                    args.view_conditioned_basis_min_bin_samples
                ),
                "view_conditioned_basis_ridge": float(args.view_conditioned_basis_ridge),
                "view_conditioned_basis_ood_mode": str(args.view_conditioned_basis_ood_mode),
                "view_conditioned_basis_ood_max_z": float(args.view_conditioned_basis_ood_max_z),
                "view_conditioned_basis_ood_min_std": float(args.view_conditioned_basis_ood_min_std),
                "teacher_distilled_basis_mode": str(args.teacher_distilled_basis_mode),
                "teacher_distilled_basis_guard_mode": str(args.teacher_distilled_basis_guard_mode),
                "teacher_distilled_basis_min_face_samples": int(
                    args.teacher_distilled_basis_min_face_samples
                ),
                "teacher_distilled_basis_ridge": float(args.teacher_distilled_basis_ridge),
                "teacher_distilled_basis_ood_max_z": float(args.teacher_distilled_basis_ood_max_z),
                "teacher_distilled_basis_ood_min_std": float(args.teacher_distilled_basis_ood_min_std),
                "teacher_distilled_basis_apply_mode": str(args.teacher_distilled_basis_apply_mode),
                "teacher_distilled_basis_blend": float(args.teacher_distilled_basis_blend),
            },
        )
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("apply command:\n" + " ".join(apply_cmd) + "\n\n")
            log.flush()
            subprocess.run(apply_cmd, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
            if run is not None:
                run.log({"apply/returncode": 0})
            log.write("\nmetrics command:\n" + " ".join(metrics_cmd) + "\n\n")
            log.flush()
            subprocess.run(metrics_cmd, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
            if run is not None:
                run.log({"metrics/returncode": 0})
        if run is not None:
            results_path = output_model / "results.json"
            audit_path = output_model / "surface_residual_region_texture_adapter_audit.json"
            if results_path.is_file():
                results = read_json(results_path)
                method_row = results.get(method_name)
                if method_row is None and len(results) == 1:
                    method_row = next(iter(results.values()))
                if method_row is None:
                    raise KeyError(f"method {method_name!r} not found in {results_path}")
                run.log({f"metric/{key}": float(method_row[key]) for key in ("PSNR", "SSIM", "LPIPS")})
                run.save(str(results_path))
            if audit_path.is_file():
                audit_payload = read_json(audit_path)
                local_alpha_profile = audit_payload.get("local_alpha_profile") or {}
                local_alpha_mode = str(local_alpha_profile.get("mode", "disabled"))
                view_basis_payload = (audit_payload.get("fit_summary") or {}).get("view_conditioned_basis", {})
                multiscale_prior_payload = (audit_payload.get("fit_summary") or {}).get(
                    "surface_multiscale_prior",
                    {},
                )
                fit_summary_payload = audit_payload.get("fit_summary") or {}
                policy_candidate_control = fit_summary_payload.get("policy_candidate_control") or {}
                view_basis_effective_mode = str(
                    view_basis_payload.get("effective_mode", view_basis_payload.get("mode", "none"))
                )
                view_basis_guard = view_basis_payload.get("guard") or {}
                teacher_basis_payload = (audit_payload.get("fit_summary") or {}).get(
                    "teacher_distilled_basis",
                    {},
                )
                teacher_basis_effective_mode = str(
                    teacher_basis_payload.get("effective_mode", teacher_basis_payload.get("mode", "none"))
                )
                teacher_basis_guard = teacher_basis_payload.get("guard") or {}
                face_gain_guard = audit_payload.get("face_gain_guard_profile") or {}
                bin_uncertainty_guard = audit_payload.get("bin_uncertainty_guard_profile") or {}
                target_support_selection = (
                    fit_summary_payload.get("target_support_candidate_selection") or {}
                )
                target_support_profile = (
                    target_support_selection.get("selected_profile") or {}
                )
                target_support_best_profile = target_support_selection.get("best_profile") or {}
                target_support_best_score = target_support_selection.get("best_score") or {}
                target_support_selected_score = target_support_selection.get("selected_score") or {}
                target_support_selected_certificate = (
                    target_support_selection.get("selected_certificate") or {}
                )
                target_support_best_certificate = (
                    target_support_selection.get("best_certificate") or {}
                )
                target_support_prerank = fit_summary_payload.get("target_support_prerank") or {}
                target_support_prerank_rows = target_support_prerank.get("score_order") or []
                target_support_prerank_best = (
                    target_support_prerank_rows[0]
                    if isinstance(target_support_prerank_rows, list) and target_support_prerank_rows
                    else {}
                )
                prior_bin_gain_hybrid = fit_summary_payload.get("policy_val_prior_bin_gain_hybrid") or {}
                target_footprint = (
                    prior_bin_gain_hybrid.get("target_footprint_bin_certificate") or {}
                )
                target_tail_risk = (
                    prior_bin_gain_hybrid.get("target_footprint_tail_risk_certificate") or {}
                )
                candidate_footprints = []
                candidate_tail_risks = []
                for candidate in audit_payload.get("fill_mode_candidates") or []:
                    candidate_bin_gain = (
                        (candidate.get("fit_summary") or {}).get("policy_val_prior_bin_gain_hybrid") or {}
                    )
                    candidate_footprint = (
                        candidate_bin_gain.get("target_footprint_bin_certificate") or {}
                    )
                    if candidate_footprint:
                        candidate_footprints.append(candidate_footprint)
                    candidate_tail_risk = (
                        candidate_bin_gain.get("target_footprint_tail_risk_certificate") or {}
                    )
                    if candidate_tail_risk:
                        candidate_tail_risks.append(candidate_tail_risk)
                fallback_alpha_raw = local_alpha_profile.get("fallback_alpha", 0.0)
                if isinstance(fallback_alpha_raw, (list, tuple)):
                    fallback_alpha_values = [float(x) for x in fallback_alpha_raw]
                else:
                    fallback_alpha_values = [float(fallback_alpha_raw or 0.0)]
                while len(fallback_alpha_values) < 3:
                    fallback_alpha_values.append(fallback_alpha_values[-1] if fallback_alpha_values else 0.0)
                fallback_alpha_scalar = float(sum(fallback_alpha_values[:3]) / 3.0)
                run.log(
                    {
                        "policy/accepted": int(bool(audit_payload.get("accepted", False))),
                        "policy/selected_alpha": float(audit_payload.get("selected_alpha", 0.0)),
                        "policy/selected_alpha_multiplier": float(audit_payload.get("selected_alpha", 0.0)),
                        "policy/selected_max_abs_delta_rgb": float(
                            fit_summary_payload.get(
                                "selected_max_abs_delta_rgb",
                                fit_summary_payload.get("requested_max_abs_delta_rgb", 0.0),
                            )
                            or 0.0
                        ),
                        "policy/max_abs_delta_rgb_candidate_count": int(
                            len(fit_summary_payload.get("max_abs_delta_rgb_candidates", []) or [])
                        ),
                        "policy/candidate_dominance_pruning_enabled": int(
                            bool(policy_candidate_control.get("dominance_pruning_enabled", False))
                        ),
                        "policy/candidate_planned_before_pruning": int(
                            policy_candidate_control.get("planned_candidate_count_before_pruning", 0) or 0
                        ),
                        "policy/candidate_planned_after_pruning": int(
                            policy_candidate_control.get("planned_candidate_count_after_pruning", 0) or 0
                        ),
                        "policy/candidate_executed_count": int(
                            policy_candidate_control.get("executed_candidate_count", 0) or 0
                        ),
                        "policy/candidate_executed_count_including_hybrids": int(
                            policy_candidate_control.get(
                                "executed_candidate_count_including_hybrids",
                                policy_candidate_control.get("executed_candidate_count", 0),
                            )
                            or 0
                        ),
                        "policy/candidate_support_pruned_count": int(
                            policy_candidate_control.get("support_duplicate_pruned_count", 0) or 0
                        ),
                        "policy/candidate_spec_pruned_count": int(
                            policy_candidate_control.get("spec_duplicate_pruned_count", 0) or 0
                        ),
                        "policy/candidate_early_stop_triggered": int(
                            bool(policy_candidate_control.get("early_stop_triggered", False))
                        ),
                        "policy/candidate_early_stop_skipped_count": int(
                            policy_candidate_control.get("early_stop_skipped_count", 0) or 0
                        ),
                        "policy/changed_fraction": float(
                            (audit_payload.get("target_apply") or {}).get("changed_fraction", 0.0)
                        ),
                        "policy/local_alpha_enabled": int(
                            bool(local_alpha_profile.get("enabled", False))
                        ),
                        "policy/local_alpha_mode_face": int(local_alpha_mode == "policy_val_face_alpha"),
                        "policy/local_alpha_mode_bucket": int(
                            local_alpha_mode == "policy_val_residual_norm_buckets"
                        ),
                        "policy/local_alpha_mode_bin": int(local_alpha_mode == "policy_val_bin_alpha"),
                        "policy/local_alpha_mode_bin_rgb": int(
                            local_alpha_mode == "policy_val_bin_rgb_alpha"
                        ),
                        "policy/local_alpha_mode_uncertainty_shrink": int(
                            local_alpha_mode == "policy_val_bin_uncertainty_shrink"
                        ),
                        "policy/local_alpha_max": float(local_alpha_profile.get("max_alpha", 0.0) or 0.0),
                        "policy/local_alpha_max_shrink": float(
                            local_alpha_profile.get("max_shrink", 0.0) or 0.0
                        ),
                        "policy/local_alpha_fallback_alpha": fallback_alpha_scalar,
                        "policy/local_alpha_fallback_shrink": float(
                            local_alpha_profile.get("fallback_shrink", 0.0) or 0.0
                        ),
                        "policy/local_alpha_fallback_alpha_r": float(fallback_alpha_values[0]),
                        "policy/local_alpha_fallback_alpha_g": float(fallback_alpha_values[1]),
                        "policy/local_alpha_fallback_alpha_b": float(fallback_alpha_values[2]),
                        "policy/face_alpha_count": int(local_alpha_profile.get("face_alpha_count", 0) or 0),
                        "policy/fallback_face_count": int(local_alpha_profile.get("fallback_face_count", 0) or 0),
                        "policy/bin_alpha_count": int(local_alpha_profile.get("bin_alpha_count", 0) or 0),
                        "policy/bin_rgb_alpha_count": int(
                            local_alpha_profile.get("bin_rgb_alpha_count", 0) or 0
                        ),
                        "policy/bin_uncertainty_shrink_count": int(
                            local_alpha_profile.get("bin_uncertainty_shrink_count", 0) or 0
                        ),
                        "policy/bin_uncertainty_shrink_policy_keep_downweight": int(
                            local_alpha_profile.get("uncertainty_shrink_policy_mode", "")
                            == "keep_with_downweight"
                        ),
                        "policy/bin_uncertainty_shrink_mean": float(
                            local_alpha_profile.get("mean_selected_shrink", 0.0) or 0.0
                        ),
                        "policy/bin_uncertainty_shrink_min": float(
                            local_alpha_profile.get("min_selected_shrink", 0.0) or 0.0
                        ),
                        "policy/bin_uncertainty_shrink_max": float(
                            local_alpha_profile.get("max_selected_shrink", 0.0) or 0.0
                        ),
                        "policy/bin_uncertainty_shrink_downweighted_count": int(
                            local_alpha_profile.get("downweighted_bin_count", 0) or 0
                        ),
                        "policy/bin_uncertainty_shrink_upweighted_count": int(
                            local_alpha_profile.get("upweighted_bin_count", 0) or 0
                        ),
                        "policy/surface_multiscale_prior_enabled": int(
                            str(multiscale_prior_payload.get("mode", "none")) != "none"
                        ),
                        "policy/surface_multiscale_prior_selected_blend": float(
                            fit_summary_payload.get(
                                "selected_surface_multiscale_prior_blend",
                                multiscale_prior_payload.get("blend", 0.0),
                            )
                            or 0.0
                        ),
                        "policy/surface_multiscale_prior_blend_candidate_count": int(
                            len(fit_summary_payload.get("surface_multiscale_prior_blend_candidates", []) or [])
                        ),
                        "policy/surface_multiscale_prior_blended_bins": int(
                            multiscale_prior_payload.get("blended_bins", 0) or 0
                        ),
                        "policy/surface_multiscale_prior_blended_fraction": float(
                            multiscale_prior_payload.get("blended_bin_fraction", 0.0) or 0.0
                        ),
                        "policy/surface_multiscale_prior_mean_blend_weight": float(
                            multiscale_prior_payload.get("mean_blend_weight", 0.0) or 0.0
                        ),
                        "policy/surface_multiscale_prior_gate_rejected_bins": int(
                            multiscale_prior_payload.get("gate_rejected_bins", 0) or 0
                        ),
                        "policy/surface_multiscale_prior_empty_rejected_bins": int(
                            multiscale_prior_payload.get("empty_rejected_bins", 0) or 0
                        ),
                        "policy/surface_multiscale_prior_sign_rejected_bins": int(
                            multiscale_prior_payload.get("sign_rejected_bins", 0) or 0
                        ),
                        "policy/surface_multiscale_prior_variance_rejected_bins": int(
                            multiscale_prior_payload.get("variance_rejected_bins", 0) or 0
                        ),
                        "policy/surface_multiscale_prior_cosine_rejected_bins": int(
                            multiscale_prior_payload.get("cosine_rejected_bins", 0) or 0
                        ),
                        "policy/prior_bin_gain_hybrid_selected": int(
                            bool(fit_summary_payload.get("selected_policy_val_prior_bin_gain_hybrid", False))
                        ),
                        "policy/prior_bin_gain_hybrid_enabled": int(
                            bool(prior_bin_gain_hybrid.get("enabled", False))
                        ),
                        "policy/prior_bin_gain_hybrid_allowed_bins": int(
                            prior_bin_gain_hybrid.get("allowed_bin_count", 0) or 0
                        ),
                        "policy/prior_bin_gain_hybrid_candidate_bins": int(
                            prior_bin_gain_hybrid.get("candidate_bin_count", 0) or 0
                        ),
                        "policy/prior_bin_gain_hybrid_allowed_fraction": float(
                            prior_bin_gain_hybrid.get("allowed_bin_fraction", 0.0) or 0.0
                        ),
                        "policy/source_mixture_requested": int(bool(args.enable_policy_val_source_mixture)),
                        "policy/source_mixture_selected_enabled": int(
                            bool(prior_bin_gain_hybrid.get("source_mixture_enabled", False))
                        ),
                        "policy/source_mixture_weight_mean": float(
                            prior_bin_gain_hybrid.get("source_mixture_weight_mean", 0.0) or 0.0
                        ),
                        "policy/source_mixture_weight_min": float(
                            prior_bin_gain_hybrid.get("source_mixture_weight_min", 0.0) or 0.0
                        ),
                        "policy/source_mixture_weight_max": float(
                            prior_bin_gain_hybrid.get("source_mixture_weight_max", 0.0) or 0.0
                        ),
                        "policy/source_mixture_den_reference": float(
                            prior_bin_gain_hybrid.get("source_mixture_den_reference", 0.0) or 0.0
                        ),
                        "policy/source_mixture_ridge_term_mean": float(
                            prior_bin_gain_hybrid.get("source_mixture_ridge_term_mean", 0.0) or 0.0
                        ),
                        "policy/target_footprint_certificate_requested": int(
                            bool(args.enable_target_footprint_bin_certificate)
                        ),
                        "policy/target_footprint_certificate_selected_enabled": int(
                            bool(target_footprint.get("enabled", False))
                        ),
                        "policy/target_footprint_candidate_profiles": int(len(candidate_footprints)),
                        "policy/target_footprint_enabled_candidate_profiles": int(
                            sum(1 for profile in candidate_footprints if bool(profile.get("enabled", False)))
                        ),
                        "policy/target_footprint_covered_bins": int(
                            target_footprint.get("covered_bin_count", 0) or 0
                        ),
                        "policy/target_footprint_candidate_bins_with_footprint": int(
                            target_footprint.get("candidate_bins_with_target_footprint", 0) or 0
                        ),
                        "policy/target_footprint_allowed_bins_with_footprint": int(
                            target_footprint.get("allowed_bins_with_target_footprint", 0) or 0
                        ),
                        "policy/target_footprint_pre_trunc_allowed_bins_with_footprint": int(
                            target_footprint.get("pre_trunc_allowed_bins_with_target_footprint", 0) or 0
                        ),
                        "policy/target_footprint_views_examined": int(
                            target_footprint.get(
                                "target_views_examined",
                                target_footprint.get("target_views_used", 0),
                            )
                            or 0
                        ),
                        "policy/target_footprint_views_with_coverage": int(
                            target_footprint.get("views_with_target_coverage", 0) or 0
                        ),
                        "policy/target_footprint_candidate_max_allowed_bins_with_footprint": int(
                            max(
                                (
                                    int(profile.get("allowed_bins_with_target_footprint", 0) or 0)
                                    for profile in candidate_footprints
                                ),
                                default=0,
                            )
                        ),
                        "policy/target_tail_risk_requested": int(
                            bool(args.enable_target_footprint_tail_risk_certificate)
                        ),
                        "policy/target_tail_risk_selected_enabled": int(
                            bool(target_tail_risk.get("enabled", False))
                        ),
                        "policy/target_tail_risk_candidate_profiles": int(len(candidate_tail_risks)),
                        "policy/target_tail_risk_applied_bins": int(
                            target_tail_risk.get("applied_bin_count", 0) or 0
                        ),
                        "policy/target_tail_risk_rejected_bins": int(
                            target_tail_risk.get("rejected_bin_count", 0) or 0
                        ),
                        "policy/target_tail_risk_rejected_bins_with_footprint": int(
                            target_tail_risk.get("rejected_bins_with_target_footprint", 0) or 0
                        ),
                        "policy/target_tail_risk_max_rejected_bins": int(
                            max(
                                (
                                    int(profile.get("rejected_bin_count", 0) or 0)
                                    for profile in candidate_tail_risks
                                ),
                                default=0,
                            )
                        ),
                        "policy/bin_alpha_candidate_bins": int(
                            local_alpha_profile.get("candidate_bin_count", 0) or 0
                        ),
                        "policy/fallback_bin_count": int(local_alpha_profile.get("fallback_bin_count", 0) or 0),
                        "policy/view_basis_enabled": int(view_basis_effective_mode != "none"),
                        "policy/view_basis_guard_fallback": int(
                            str(view_basis_guard.get("decision", "")) == "fallback_to_mean"
                        ),
                        "policy/view_basis_supported_bins": int(
                            view_basis_payload.get("supported_bins", 0)
                            or 0
                        ),
                        "policy/view_basis_supported_bin_fraction": float(
                            view_basis_payload.get("supported_bin_fraction", 0.0)
                            or 0.0
                        ),
                        "policy/view_basis_ood_enabled": int(
                            str(view_basis_payload.get("ood_mode", "none")) != "none"
                        ),
                        "policy/view_basis_ood_max_z": float(
                            view_basis_payload.get("ood_max_z", 0.0) or 0.0
                        ),
                        "policy/view_basis_ood_min_std": float(
                            view_basis_payload.get("ood_min_std", 0.0) or 0.0
                        ),
                        "policy/teacher_basis_enabled": int(teacher_basis_effective_mode != "none"),
                        "policy/teacher_basis_guard_fallback": int(
                            str(teacher_basis_guard.get("decision", "")) == "fallback_to_legacy"
                        ),
                        "policy/teacher_basis_supported_faces": int(
                            teacher_basis_payload.get("supported_faces", 0) or 0
                        ),
                        "policy/teacher_basis_supported_face_fraction": float(
                            teacher_basis_payload.get("supported_face_fraction", 0.0) or 0.0
                        ),
                        "policy/teacher_basis_candidate_faces": int(
                            teacher_basis_payload.get("candidate_faces", 0) or 0
                        ),
                        "policy/teacher_basis_blend": float(
                            teacher_basis_payload.get("blend", 0.0) or 0.0
                        ),
                        "policy/face_gain_guard_enabled": int(bool(face_gain_guard.get("enabled", False))),
                        "policy/face_gain_guard_allowed_faces": int(
                            face_gain_guard.get("allowed_face_count", 0) or 0
                        ),
                        "policy/face_gain_guard_rejected_faces": int(
                            face_gain_guard.get("rejected_face_count", 0) or 0
                        ),
                        "policy/face_gain_guard_allowed_sample_fraction": float(
                            face_gain_guard.get("allowed_sample_fraction", 0.0) or 0.0
                        ),
                        "policy/face_gain_guard_post_accepted": int(
                            bool(face_gain_guard.get("post_guard_accepted", False))
                        ),
                        "policy/bin_uncertainty_guard_enabled": int(
                            bool(bin_uncertainty_guard.get("enabled", False))
                        ),
                        "policy/bin_uncertainty_guard_allowed_bins": int(
                            bin_uncertainty_guard.get("allowed_bin_count", 0) or 0
                        ),
                        "policy/bin_uncertainty_guard_rejected_bins": int(
                            bin_uncertainty_guard.get("rejected_bin_count", 0) or 0
                        ),
                        "policy/bin_uncertainty_guard_allowed_faces": int(
                            bin_uncertainty_guard.get("allowed_face_count", 0) or 0
                        ),
                        "policy/bin_uncertainty_guard_allowed_sample_fraction": float(
                            bin_uncertainty_guard.get("allowed_sample_fraction", 0.0) or 0.0
                        ),
                        "policy/bin_uncertainty_guard_post_accepted": int(
                            bool(bin_uncertainty_guard.get("post_guard_accepted", False))
                        ),
                        "policy/target_support_selection_enabled": int(
                            bool(target_support_selection.get("enabled", False))
                        ),
                        "policy/target_support_profile_enabled": int(
                            bool(target_support_profile.get("enabled", False))
                        ),
                        "policy/target_support_changed_fraction": float(
                            target_support_profile.get("changed_fraction", 0.0) or 0.0
                        ),
                        "policy/target_support_min_view_changed_fraction": float(
                            target_support_profile.get("min_view_changed_fraction", 0.0) or 0.0
                        ),
                        "policy/target_support_cvar20_view_changed_fraction": float(
                            target_support_profile.get("cvar20_view_changed_fraction", 0.0) or 0.0
                        ),
                        "policy/target_support_valid_fraction": float(
                            target_support_profile.get("valid_fraction", 0.0) or 0.0
                        ),
                        "policy/target_support_selected_certificate_passed": int(
                            bool(target_support_selected_certificate.get("passed", False))
                        ),
                        "policy/target_support_best_certificate_passed": int(
                            bool(target_support_best_certificate.get("passed", False))
                        ),
                        "policy/target_support_selected_is_best": int(
                            bool(target_support_selection.get("selected_is_best_target_support", False))
                        ),
                        "policy/target_support_best_changed_fraction": float(
                            target_support_best_profile.get("changed_fraction", 0.0) or 0.0
                        ),
                        "policy/target_support_best_cvar20_view_changed_fraction": float(
                            target_support_best_profile.get("cvar20_view_changed_fraction", 0.0) or 0.0
                        ),
                        "policy/target_support_best_min_view_changed_fraction": float(
                            target_support_best_profile.get("min_view_changed_fraction", 0.0) or 0.0
                        ),
                        "policy/target_support_best_valid_fraction": float(
                            target_support_best_profile.get("valid_fraction", 0.0) or 0.0
                        ),
                        "policy/target_support_delta_best_minus_selected_changed_fraction": float(
                            (target_support_best_score.get("changed_fraction", 0.0) or 0.0)
                            - (target_support_selected_score.get("changed_fraction", 0.0) or 0.0)
                        ),
                        "policy/target_support_prerank_enabled": int(
                            bool(target_support_prerank.get("enabled", False))
                        ),
                        "policy/target_support_prerank_input_candidates": int(
                            target_support_prerank.get("input_support_candidate_count", 0) or 0
                        ),
                        "policy/target_support_prerank_retained_candidates": int(
                            target_support_prerank.get("retained_support_candidate_count", 0) or 0
                        ),
                        "policy/target_support_prerank_best_coverage_fraction": float(
                            target_support_prerank_best.get("coverage_fraction", 0.0) or 0.0
                        ),
                        "policy/target_support_prerank_best_cvar20_coverage_fraction": float(
                            target_support_prerank_best.get("cvar20_view_coverage_fraction", 0.0) or 0.0
                        ),
                        "policy/target_support_prerank_best_min_coverage_fraction": float(
                            target_support_prerank_best.get("min_view_coverage_fraction", 0.0) or 0.0
                        ),
                    }
                )
                run.save(str(audit_path))
            run.save(str(log_path))
    finally:
        if run is not None:
            run.finish()
    print(json.dumps({"scene": scene, "output_model": str(output_model), "log": str(log_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
