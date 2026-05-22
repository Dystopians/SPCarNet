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
DEFAULT_RENDER_REGION_CARRIER_TEMPLATE = (
    "outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/"
    "{scene}/render_visible_region_carriers.json"
)


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
    },
}

PROFILE_FIELD_NAMES = tuple(next(iter(PROFILE_DEFAULTS.values())).keys())


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
    parser.add_argument("--use_filtered_plan_for_selector", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--delta_shared_residual_field", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--delta_shared_residual_field_anchors", type=int, default=16)
    parser.add_argument("--delta_shared_residual_field_sigma", type=float, default=0.0)
    parser.add_argument("--delta_shared_residual_field_lr", type=float, default=0.0)
    parser.add_argument("--delta_shared_residual_field_weight_l2", type=float, default=1.0e-4)
    parser.add_argument("--delta_shared_residual_field_view_hinge_weight", type=float, default=0.0)
    parser.add_argument("--delta_shared_residual_field_view_hinge_min_samples", type=int, default=16)
    parser.add_argument("--delta_shared_residual_field_duplicate_smooth_weight", type=float, default=0.0)
    parser.add_argument("--delta_max_faces_to_apply", type=int, default=None)
    parser.add_argument("--delta_min_policy_val_relative_gain", type=float, default=0.02)
    parser.add_argument("--delta_min_policy_val_samples", type=int, default=512)
    parser.add_argument("--delta_min_policy_val_unique_faces", type=int, default=16)
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
    parser.add_argument("--delta_crossfold_folds", type=int, default=4)
    parser.add_argument("--delta_crossfold_min_passing_folds", type=int, default=3)
    parser.add_argument("--delta_patch_cert_rings", type=int, default=1)
    parser.add_argument("--delta_patch_cert_max_faces_per_seed", type=int, default=None)
    parser.add_argument("--delta_patch_cert_neighbor_mode", choices=("topology", "centroid", "both"), default="both")
    parser.add_argument(
        "--delta_patch_cert_cluster_basis_mode",
        choices=("shared", "scaled", "rank2", "chart_linear", "chart_quad", "field_linear", "field_quad"),
        default="chart_linear",
    )
    parser.add_argument("--delta_patch_cert_cluster_basis_steps", type=int, default=240)
    parser.add_argument("--delta_patch_cert_cluster_basis_min_samples", type=int, default=32)
    parser.add_argument("--delta_patch_cert_cluster_basis_max_fit_mse_regression", type=float, default=0.02)
    parser.add_argument("--delta_patch_cert_cluster_basis_view_hinge_weight", type=float, default=0.0)
    parser.add_argument("--delta_patch_cert_cluster_basis_view_hinge_min_samples", type=int, default=16)
    parser.add_argument("--delta_patch_cert_cluster_basis_geometry_smooth_weight", type=float, default=0.0)
    parser.add_argument("--delta_patch_cert_carrier_holdout_groups", type=int, default=4)
    parser.add_argument("--delta_patch_cert_carrier_holdout_min_passing_groups", type=int, default=3)
    parser.add_argument("--delta_patch_cert_carrier_holdout_auto_prefix_min_faces", type=int, default=0)
    parser.add_argument("--delta_patch_cert_carrier_holdout_auto_prefix_face_bonus", type=float, default=0.0)
    parser.add_argument(
        "--delta_patch_cert_carrier_holdout_auto_prefix_positive_tail_safe",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--no_seed_rescue", action="store_true")

    parser.add_argument("--trial_specs", default="")
    parser.add_argument("--selector_alpha_max", type=float, default=None)
    parser.add_argument("--selector_alpha_steps", type=int, default=None)
    parser.add_argument("--selector_alpha_lr", type=float, default=0.06)
    parser.add_argument("--selector_alpha_max_total_samples", type=int, default=None)
    parser.add_argument("--selector_alpha_device", default="cuda")
    parser.add_argument("--selector_min_trainval_psnr_gain", type=float, default=None)
    parser.add_argument("--selector_min_trainval_balanced_delta", type=float, default=None)
    parser.add_argument("--selector_tail_min_trainval_balanced_delta", type=float, default=None)
    parser.add_argument("--selector_tail_stable_promotion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selector_tail_max_balanced_cvar_loss", type=float, default=0.0012)
    parser.add_argument("--selector_tail_min_mean_to_cvar_ratio", type=float, default=0.05)
    parser.add_argument("--selector_tail_max_lpips_positive_fraction", type=float, default=0.70)
    parser.add_argument("--phasej_test_method", default=DEFAULT_PHASEJ_TEST_METHOD)
    parser.add_argument("--phasej_trainval_method", default=DEFAULT_PHASEJ_TRAINVAL_METHOD)
    parser.add_argument("--gate_min_psnr_gain", type=float, default=0.0)
    parser.add_argument("--gate_max_ssim_regression", type=float, default=5e-5)
    parser.add_argument("--gate_max_lpips_regression", type=float, default=1.5e-4)
    parser.add_argument("--gate_min_balanced_delta", type=float, default=0.0)

    parser.add_argument("--render_region_carrier_template", default=DEFAULT_RENDER_REGION_CARRIER_TEMPLATE)
    parser.add_argument("--filter_max_region_matches_per_plan_carrier", type=int, default=3)
    parser.add_argument("--filter_min_regions", type=int, default=1)
    parser.add_argument("--filter_min_changed_regions", type=int, default=1)
    parser.add_argument("--filter_min_changed_fraction", type=float, default=0.10)
    parser.add_argument("--filter_min_mean_core_balanced_delta", type=float, default=0.0)
    parser.add_argument("--filter_min_mean_delta_psnr", type=float, default=0.0)
    parser.add_argument("--filter_min_tail_core_balanced_delta", type=float, default=-1.0e-8)
    parser.add_argument("--filter_max_context_mse_regression", type=float, default=1.0e-6)
    parser.add_argument("--filter_drop_unmapped", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--filter_require_positive_plan_proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter_train_render_region_max_regions", type=int, default=64)
    parser.add_argument("--filter_train_render_region_min_pixels", type=int, default=128)
    parser.add_argument("--filter_train_render_region_min_crop_size", type=int, default=32)
    parser.add_argument("--filter_train_render_region_context_pad", type=int, default=16)
    parser.add_argument("--filter_train_render_region_tail_fraction", type=float, default=0.25)
    parser.add_argument("--filter_train_render_region_skip_lpips", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if int(args.delta_crossfold_folds) <= 1:
        parser.error("--delta_crossfold_folds must be > 1 for strict carrier plans")
    if int(args.delta_crossfold_min_passing_folds) <= 0:
        parser.error("--delta_crossfold_min_passing_folds must be positive")
    if int(args.delta_patch_cert_rings) <= 0:
        parser.error("--delta_patch_cert_rings must be positive")
    if int(args.delta_patch_cert_carrier_holdout_min_passing_groups) <= 0:
        parser.error("--delta_patch_cert_carrier_holdout_min_passing_groups must be positive")
    if int(args.filter_max_region_matches_per_plan_carrier) <= 0:
        parser.error("--filter_max_region_matches_per_plan_carrier must be positive")
    for name in ("filter_min_regions", "filter_min_changed_regions"):
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name} must be >= 0")
    if not 0.0 <= float(args.filter_min_changed_fraction) <= 1.0:
        parser.error("--filter_min_changed_fraction must be in [0, 1]")
    if int(args.filter_train_render_region_max_regions) <= 0:
        parser.error("--filter_train_render_region_max_regions must be > 0")
    if int(args.filter_train_render_region_min_pixels) <= 0:
        parser.error("--filter_train_render_region_min_pixels must be > 0")
    if int(args.filter_train_render_region_min_crop_size) <= 0:
        parser.error("--filter_train_render_region_min_crop_size must be > 0")
    if int(args.filter_train_render_region_context_pad) < 0:
        parser.error("--filter_train_render_region_context_pad must be >= 0")
    if not 0.0 < float(args.filter_train_render_region_tail_fraction) <= 1.0:
        parser.error("--filter_train_render_region_tail_fraction must be in (0, 1]")
    return args


def profile_value(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is not None and value != "":
        return value
    return PROFILE_DEFAULTS[str(args.profile)][name]


def resolved_profile_values(args: argparse.Namespace) -> dict[str, Any]:
    return {name: profile_value(args, name) for name in PROFILE_FIELD_NAMES}


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


def selector_plan_template(args: argparse.Namespace) -> str:
    if str(args.selector_plan_template).strip():
        return str(resolve_path(args.selector_plan_template))
    if bool(args.use_filtered_plan_for_selector):
        return filtered_plan_template(args)
    return plan_template(args)


def format_plan_path(args: argparse.Namespace, scene: str) -> Path:
    return Path(plan_template(args).format(scene=scene))


def format_filtered_plan_path(args: argparse.Namespace, scene: str) -> Path:
    return Path(filtered_plan_template(args).format(scene=scene))


def format_region_carrier_path(args: argparse.Namespace, scene: str) -> Path:
    return Path(str(resolve_path(args.render_region_carrier_template)).format(scene=scene))


def command_log(root: Path, stage: str, scene: str) -> Path:
    return root / "logs" / stage / f"{scene}.log"


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
        str(args.delta_shared_residual_field_anchors),
        "--delta_shared_residual_field_sigma",
        str(args.delta_shared_residual_field_sigma),
        "--delta_shared_residual_field_lr",
        str(args.delta_shared_residual_field_lr),
        "--delta_shared_residual_field_weight_l2",
        str(args.delta_shared_residual_field_weight_l2),
        "--delta_shared_residual_field_view_hinge_weight",
        str(args.delta_shared_residual_field_view_hinge_weight),
        "--delta_shared_residual_field_view_hinge_min_samples",
        str(args.delta_shared_residual_field_view_hinge_min_samples),
        "--delta_shared_residual_field_duplicate_smooth_weight",
        str(args.delta_shared_residual_field_duplicate_smooth_weight),
        "--delta_min_policy_val_relative_gain",
        str(args.delta_min_policy_val_relative_gain),
        "--delta_min_policy_val_samples",
        str(args.delta_min_policy_val_samples),
        "--delta_min_policy_val_unique_faces",
        str(args.delta_min_policy_val_unique_faces),
        "--delta_validation_shrink_mode",
        str(args.delta_validation_shrink_mode),
        "--delta_validation_gain_max_scale",
        str(args.delta_validation_gain_max_scale),
        "--delta_crossfold_gain_certificate_folds",
        str(args.delta_crossfold_folds),
        "--delta_crossfold_min_passing_folds",
        str(args.delta_crossfold_min_passing_folds),
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
        "--delta_patch_cert_neighbor_mode",
        str(args.delta_patch_cert_neighbor_mode),
        "--delta_patch_cert_crossfold_folds",
        str(args.delta_crossfold_folds),
        "--delta_patch_cert_crossfold_min_passing_folds",
        str(args.delta_crossfold_min_passing_folds),
        "--delta_patch_cert_crossfold_min_fold_relative_gain",
        "0.0",
        "--delta_patch_cert_neighbor_crossfold",
        "--delta_patch_cert_cluster_basis",
        "--delta_patch_cert_cluster_basis_mode",
        str(args.delta_patch_cert_cluster_basis_mode),
        "--delta_patch_cert_cluster_basis_steps",
        str(args.delta_patch_cert_cluster_basis_steps),
        "--delta_patch_cert_cluster_basis_min_samples",
        str(args.delta_patch_cert_cluster_basis_min_samples),
        "--delta_patch_cert_cluster_basis_max_fit_mse_regression",
        str(args.delta_patch_cert_cluster_basis_max_fit_mse_regression),
        "--delta_patch_cert_cluster_basis_view_hinge_weight",
        str(args.delta_patch_cert_cluster_basis_view_hinge_weight),
        "--delta_patch_cert_cluster_basis_view_hinge_min_samples",
        str(args.delta_patch_cert_cluster_basis_view_hinge_min_samples),
        "--delta_patch_cert_cluster_basis_geometry_smooth_weight",
        str(args.delta_patch_cert_cluster_basis_geometry_smooth_weight),
        "--delta_patch_cert_carrier_holdout_selector",
        "--delta_patch_cert_carrier_holdout_groups",
        str(args.delta_patch_cert_carrier_holdout_groups),
        "--delta_patch_cert_carrier_holdout_grouping",
        "sample_balanced",
        "--delta_patch_cert_carrier_holdout_disjoint",
        "--delta_patch_cert_carrier_holdout_min_passing_groups",
        str(args.delta_patch_cert_carrier_holdout_min_passing_groups),
        "--delta_patch_cert_carrier_holdout_auto_prefix",
        "--delta_patch_cert_carrier_holdout_auto_prefix_min_faces",
        str(args.delta_patch_cert_carrier_holdout_auto_prefix_min_faces),
        "--delta_patch_cert_carrier_holdout_auto_prefix_face_bonus",
        str(args.delta_patch_cert_carrier_holdout_auto_prefix_face_bonus),
        (
            "--delta_patch_cert_carrier_holdout_auto_prefix_positive_tail_safe"
            if bool(args.delta_patch_cert_carrier_holdout_auto_prefix_positive_tail_safe)
            else "--no-delta_patch_cert_carrier_holdout_auto_prefix_positive_tail_safe"
        ),
        "--delta_strict_patchcert_carrier",
        "--delta_max_faces_to_apply",
        str(profile_value(args, "delta_max_faces_to_apply")),
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
        str(args.phasej_test_method),
        "--phasej_trainval_method",
        str(args.phasej_trainval_method),
        "--train_render_region_gate_enable",
        "--train_render_region_carrier_json",
        str(args.render_region_carrier_template),
        "--train_render_region_eval_source",
        "raw_base",
        "--train_render_region_max_regions",
        str(args.filter_train_render_region_max_regions),
        "--train_render_region_min_pixels",
        str(args.filter_train_render_region_min_pixels),
        "--train_render_region_min_crop_size",
        str(args.filter_train_render_region_min_crop_size),
        "--train_render_region_context_pad",
        str(args.filter_train_render_region_context_pad),
        "--train_render_region_tail_fraction",
        str(args.filter_train_render_region_tail_fraction),
        "--train_render_region_min_regions",
        "0",
        "--train_render_region_min_changed_regions",
        "0",
        "--train_render_region_min_changed_fraction",
        "0.0",
        "--train_render_region_min_core_balanced_delta=-1e30",
        "--train_render_region_min_core_psnr_delta=-1e30",
        "--train_render_region_min_tail_cvar_delta=-1e30",
        "--train_render_region_max_context_mse_regression",
        "1e30",
        "--train_render_region_max_negative_fraction",
        "1.0",
        "--gate_min_psnr_gain",
        str(args.gate_min_psnr_gain),
        "--gate_max_ssim_regression",
        str(args.gate_max_ssim_regression),
        "--gate_max_lpips_regression",
        str(args.gate_max_lpips_regression),
        "--gate_min_balanced_delta",
        str(args.gate_min_balanced_delta),
        "--wandb_project",
        str(args.wandb_project),
        "--wandb_group",
        f"{label}_plan_generation",
        "--wandb_name",
        f"{label}_plan_{scene}",
    ]
    if bool(args.delta_shared_residual_field):
        command.append("--delta_shared_residual_field")
    else:
        command.append("--no-delta_shared_residual_field")
    if bool(args.filter_train_render_region_skip_lpips):
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


def filter_command(args: argparse.Namespace, scene: str) -> CommandRecord:
    root = output_root(args)
    plan_path = format_plan_path(args, scene)
    filtered_path = format_filtered_plan_path(args, scene)
    summary_path = filtered_path.with_name("filter_summary.json")
    md_path = filtered_path.with_name("filter_summary.md")
    objective_path = root / "plan_generation" / scene / "train_render_region_objective_raw_base.json"
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
        str(format_region_carrier_path(args, scene)),
        "--output_plan",
        str(filtered_path),
        "--output_md",
        str(md_path),
        "--output_json",
        str(summary_path),
        "--max_region_matches_per_plan_carrier",
        str(args.filter_max_region_matches_per_plan_carrier),
        "--min_regions",
        str(args.filter_min_regions),
        "--min_changed_regions",
        str(args.filter_min_changed_regions),
        "--min_changed_fraction",
        str(args.filter_min_changed_fraction),
        f"--min_mean_core_balanced_delta={args.filter_min_mean_core_balanced_delta}",
        f"--min_mean_delta_psnr={args.filter_min_mean_delta_psnr}",
        f"--min_tail_core_balanced_delta={args.filter_min_tail_core_balanced_delta}",
        "--max_context_mse_regression",
        str(args.filter_max_context_mse_regression),
    ]
    command.append("--drop_unmapped" if bool(args.filter_drop_unmapped) else "--no-drop_unmapped")
    command.append(
        "--require_positive_plan_proxy"
        if bool(args.filter_require_positive_plan_proxy)
        else "--no-require_positive_plan_proxy"
    )
    return CommandRecord(
        stage="filter",
        scene=scene,
        command=command,
        log_path=str(command_log(root, "filter", scene)),
        output_paths={
            "raw_candidate_plan": str(plan_path),
            "filtered_candidate_plan": str(filtered_path),
            "filter_summary": str(summary_path),
            "filter_summary_md": str(md_path),
            "render_region_objective": str(objective_path),
            "render_region_carrier_json": str(format_region_carrier_path(args, scene)),
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
        str(args.phasej_test_method),
        "--phasej_trainval_method",
        str(args.phasej_trainval_method),
        "--selector_fit_plan_alphas",
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
        "--selector_tail_max_balanced_cvar_loss",
        str(args.selector_tail_max_balanced_cvar_loss),
        "--selector_tail_min_mean_to_cvar_ratio",
        str(args.selector_tail_min_mean_to_cvar_ratio),
        "--selector_tail_max_lpips_positive_fraction",
        str(args.selector_tail_max_lpips_positive_fraction),
        "--gate_min_psnr_gain",
        str(args.gate_min_psnr_gain),
        "--gate_max_ssim_regression",
        str(args.gate_max_ssim_regression),
        "--gate_max_lpips_regression",
        str(args.gate_max_lpips_regression),
        "--gate_min_balanced_delta",
        str(args.gate_min_balanced_delta),
        "--wandb_project",
        str(args.wandb_project),
        "--wandb_group",
        f"{label}_selector",
    ]
    if bool(args.selector_tail_stable_promotion):
        command.append("--selector_enable_tail_stable_promotion")
    if bool(args.force):
        command.append("--force")
    return CommandRecord(
        stage="selector",
        scene=",".join(scenes),
        command=command,
        log_path=str(command_log(root, "selector", "all_scenes")),
        output_paths={
            "selector_root": str(root / "selector"),
            "selector_summary": str(root / "selector" / "coupled_selector_summary.json"),
            "selector_plan_template": selector_plan_template(args),
        },
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


def write_manifest(args: argparse.Namespace, records: list[CommandRecord], scenes: list[str]) -> None:
    root = output_root(args)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": "ecsr_autovisual_facelocal_pipeline",
        "pipeline_label": str(args.pipeline_label),
        "profile": str(args.profile),
        "profile_defaults": PROFILE_DEFAULTS[str(args.profile)],
        "resolved_profile_args": resolved_profile_values(args),
        "dry_run": bool(args.dry_run),
        "selection_uses_test": False,
        "scenes": scenes,
        "plan_template": plan_template(args),
        "filtered_plan_template": filtered_plan_template(args),
        "selector_plan_template": selector_plan_template(args),
        "render_region_carrier_template": str(args.render_region_carrier_template),
        "evidence_root": str(evidence_root(args)),
        "commands": [asdict(record) for record in records],
    }
    (root / "pipeline_command_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ECSR Auto-Visual Face-Local Pipeline Manifest",
        "",
        f"- label: `{args.pipeline_label}`",
        f"- profile: `{args.profile}`",
        f"- delta strength: `{profile_value(args, 'delta_strength')}`",
        f"- delta max abs rgb: `{profile_value(args, 'delta_max_abs_rgb')}`",
        f"- selector alpha max: `{profile_value(args, 'selector_alpha_max')}`",
        f"- dry run: `{str(bool(args.dry_run)).lower()}`",
        f"- selection uses test: `false`",
        f"- scenes: `{', '.join(scenes)}`",
        f"- plan template: `{rel(plan_template(args))}`",
        f"- filtered plan template: `{rel(filtered_plan_template(args))}`",
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
    payload = {
        "pipeline": "ecsr_autovisual_facelocal_pipeline",
        "pipeline_label": str(args.pipeline_label),
        "profile": str(args.profile),
        "profile_defaults": PROFILE_DEFAULTS[str(args.profile)],
        "resolved_profile_args": resolved_profile_values(args),
        "dry_run": bool(args.dry_run),
        "selection_uses_test": False,
        "command_manifest": rel(root / "pipeline_command_manifest.json"),
        "plan_template": plan_template(args),
        "filtered_plan_template": filtered_plan_template(args),
        "selector_plan_template": selector_plan_template(args),
        "render_region_carrier_template": str(args.render_region_carrier_template),
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
        f"- delta strength: `{profile_value(args, 'delta_strength')}`",
        f"- delta max abs rgb: `{profile_value(args, 'delta_max_abs_rgb')}`",
        f"- selector alpha max: `{profile_value(args, 'selector_alpha_max')}`",
        f"- dry run: `{str(bool(args.dry_run)).lower()}`",
        f"- command manifest: `{rel(root / 'pipeline_command_manifest.json')}`",
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
            record = filter_command(args, scene)
            if format_filtered_plan_path(args, scene).is_file() and not bool(args.force):
                record.skipped = True
                record.skip_reason = "filtered candidate plan exists; use --force to rebuild"
            records.append(run_command(record, args))

    if "selector" in stages:
        records.append(run_command(selector_command(args, scenes), args))

    write_manifest(args, records, scenes)
    write_summary(args, records, scenes)
    print(json.dumps({"output_root": str(root), "commands": len(records), "dry_run": bool(args.dry_run)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
