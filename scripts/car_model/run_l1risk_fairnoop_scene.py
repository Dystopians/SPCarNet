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
        choices=("none", "fit_residual_topk"),
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
        choices=("none", "face_uv_normal_camera_ridge"),
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
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()

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
        "0.12",
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
            "0.0",
            "--min_policy_val_ssim_positive_view_fraction",
            "0.75",
            "--min_policy_val_ssim_min_view_gain",
            "-0.000005",
            "--enable_policy_val_image_l1_gate",
            "--policy_val_l1_max_size",
            "512",
            "--min_policy_val_l1_mean_gain",
            "0.0",
            "--min_policy_val_l1_positive_view_fraction",
            str(float(args.min_policy_val_l1_positive_view_fraction)),
            "--min_policy_val_l1_min_view_gain",
            "-0.000005",
            "--min_policy_val_l1_cvar20_view_gain",
            "-0.000005",
            "--min_target_changed_fraction",
            str(float(args.min_target_changed_fraction)),
            "--write_noop_on_reject",
            "--noop_fallback_source",
            "target_evidence",
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
                "support_expansion_max_extra_faces_candidates": str(
                    args.support_expansion_max_extra_faces_candidates
                ),
                "support_expansion_mode": str(args.support_expansion_mode),
                "support_expansion_max_extra_faces": int(args.support_expansion_max_extra_faces),
                "texture_size_candidates": str(args.texture_size_candidates),
                "atlas_empty_bin_fill_mode": str(args.atlas_empty_bin_fill_mode),
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
                        "policy/local_alpha_max": float(local_alpha_profile.get("max_alpha", 0.0) or 0.0),
                        "policy/local_alpha_fallback_alpha": fallback_alpha_scalar,
                        "policy/local_alpha_fallback_alpha_r": float(fallback_alpha_values[0]),
                        "policy/local_alpha_fallback_alpha_g": float(fallback_alpha_values[1]),
                        "policy/local_alpha_fallback_alpha_b": float(fallback_alpha_values[2]),
                        "policy/face_alpha_count": int(local_alpha_profile.get("face_alpha_count", 0) or 0),
                        "policy/fallback_face_count": int(local_alpha_profile.get("fallback_face_count", 0) or 0),
                        "policy/bin_alpha_count": int(local_alpha_profile.get("bin_alpha_count", 0) or 0),
                        "policy/bin_rgb_alpha_count": int(
                            local_alpha_profile.get("bin_rgb_alpha_count", 0) or 0
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
