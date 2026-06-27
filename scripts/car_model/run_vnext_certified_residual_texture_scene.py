#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.ecsr_vnext_protocol import (  # noqa: E402
    command_record,
    make_protocol_audit,
    make_run_manifest,
    path_record,
    write_json,
    write_vnext_report,
)


METHOD = "vNext_certified_residual_surface_texture"
DEFAULT_METHOD_NAME = "ours_26000_vnext_certified_residual_surface_texture"


def _python() -> str:
    return sys.executable


def _run_step(record: dict[str, Any], *, env: dict[str, str], dry_run: bool) -> dict[str, Any]:
    log_path = Path(str(record["log_path"])) if record.get("log_path") else None
    if dry_run:
        record["returncode"] = 0
        record["elapsed_sec"] = 0.0
        record["dry_run"] = True
        return record
    if log_path is None:
        raise ValueError("run step requires a log path")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {record['cmd_string']}\n\n")
        handle.flush()
        proc = subprocess.run(
            record["cmd"],
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    record["returncode"] = int(proc.returncode)
    record["elapsed_sec"] = float(time.time() - start)
    return record


def _teacher_cache_cmd(args: argparse.Namespace, teacher_cache_dir: Path) -> list[str]:
    cmd = [
        _python(),
        "scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py",
        "--base_evidence_dir",
        str(args._effective_fit_evidence_dir),
        "--teacher_render_dir",
        str(args.teacher_render_dir),
        "--out_dir",
        str(teacher_cache_dir),
        "--force",
        "--selection_mode",
        str(args.teacher_selection_mode),
        "--teacher_render_error_margin",
        str(args.teacher_render_error_margin),
        "--teacher_parent_delta_min",
        str(args.teacher_parent_delta_min),
        "--top_support_min_alpha",
        str(args.top_support_min_alpha),
        "--top_support_limit",
        str(args.top_support_limit),
    ]
    if args.parent_render_dir:
        cmd.extend(["--parent_render_dir", str(args.parent_render_dir)])
    if args.no_mask_teacher_target:
        cmd.append("--no-mask_target")
    if bool(args.reparent_allow_resize):
        cmd.append("--allow_resize")
    return cmd


def _reparent_evidence_cmd(
    args: argparse.Namespace,
    *,
    base_evidence_dir: Path,
    parent_render_dir: Path,
    out_dir: Path,
    split_label: str,
) -> list[str]:
    cmd = [
        _python(),
        "scripts/car_model/ecsr_reparent_surface_evidence_cache.py",
        "--base_evidence_dir",
        str(base_evidence_dir),
        "--parent_render_dir",
        str(parent_render_dir),
        "--out_dir",
        str(out_dir),
        "--parent_label",
        str(args.reparent_parent_label or f"{split_label}_parent"),
        "--force",
    ]
    if bool(args.reparent_allow_resize):
        cmd.append("--allow_resize")
    return cmd


def _strip_target_evidence_cmd(args: argparse.Namespace, stripped_target_evidence_dir: Path) -> list[str]:
    return [
        _python(),
        "scripts/car_model/ecsr_strip_target_evidence_for_vnext.py",
        "--target_evidence_dir",
        str(args._effective_target_evidence_dir),
        "--out_dir",
        str(stripped_target_evidence_dir),
        "--force",
    ]


def _populate_eval_gt_cmd(args: argparse.Namespace, output_model: Path, audit_path: Path) -> list[str]:
    return [
        _python(),
        "scripts/car_model/ecsr_populate_eval_gt_from_target_evidence.py",
        "--target_evidence_dir",
        str(args._effective_target_evidence_dir),
        "--output_model",
        str(output_model),
        "--split",
        str(args.target_split),
        "--method_name",
        str(args.method_name),
        "--audit_path",
        str(audit_path),
        "--force",
    ]


def _append_arg(cmd: list[str], flag: str, value: Any) -> None:
    text = str(value)
    if text.startswith("-"):
        cmd.append(f"{flag}={text}")
    else:
        cmd.extend([flag, text])


def _looks_like_negative_number(value: str) -> bool:
    if not value.startswith("-"):
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def _normalize_negative_numeric_args(cmd: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(cmd):
        token = cmd[index]
        if (
            token.startswith("--")
            and "=" not in token
            and index + 1 < len(cmd)
            and _looks_like_negative_number(str(cmd[index + 1]))
        ):
            normalized.append(f"{token}={cmd[index + 1]}")
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


def _texture_cmd(
    args: argparse.Namespace,
    fit_evidence_dir: Path,
    target_evidence_dir: Path,
    output_model: Path,
) -> list[str]:
    cmd = [
        _python(),
        "scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py",
        "--source_model",
        str(args.source_model),
        "--fit_evidence_dir",
        str(fit_evidence_dir),
        "--target_evidence_dir",
        str(target_evidence_dir),
        "--region_carrier_json",
        str(args.region_carrier_json),
        "--output_model",
        str(output_model),
        "--target_split",
        str(args.target_split),
        "--base_method_name",
        str(args.base_method_name),
        "--method_name",
        str(args.method_name),
        "--residual_rgb_key",
        "teacher_residual_rgb",
        "--residual_l1_key",
        "teacher_residual_l1",
        "--texture_size",
        str(args.texture_size),
        "--texture_size_candidates",
        str(args.texture_size_candidates),
        "--support_expansion_mode",
        str(args.support_expansion_mode),
        "--support_expansion_max_extra_faces_candidates",
        str(args.support_expansion_max_extra_faces_candidates),
        "--support_expansion_min_face_samples",
        str(args.support_expansion_min_face_samples),
        "--target_footprint_residual_debt_match_level",
        str(args.target_footprint_residual_debt_match_level),
        "--policy_val_stride",
        str(args.policy_val_stride),
        "--alpha_grid",
        str(args.alpha_grid),
        "--min_l1",
        str(args.min_l1),
        "--min_alpha",
        str(args.min_alpha),
        "--max_abs_delta_rgb",
        str(args.max_abs_delta_rgb),
        "--max_abs_delta_rgb_candidates",
        str(args.max_abs_delta_rgb_candidates),
        "--atlas_empty_bin_fill_mode",
        str(args.atlas_empty_bin_fill_mode),
        "--surface_multiscale_prior_mode",
        str(args.surface_multiscale_prior_mode),
        "--surface_multiscale_prior_blend_candidates",
        str(args.surface_multiscale_prior_blend_candidates),
        "--surface_multiscale_prior_gate_mode",
        "evidence_consistent",
        "--surface_multiscale_prior_min_direct_samples",
        str(args.surface_multiscale_prior_min_direct_samples),
        "--surface_multiscale_prior_min_sign_consistency",
        str(args.surface_multiscale_prior_min_sign_consistency),
        "--surface_multiscale_prior_min_cosine",
        str(args.surface_multiscale_prior_min_cosine),
        "--view_conditioned_basis_mode",
        str(args.view_conditioned_basis_mode),
        "--view_conditioned_basis_guard_mode",
        "policy_val_nonregressive",
        "--view_conditioned_basis_ood_mode",
        "diag_z",
        "--teacher_distilled_basis_mode",
        str(args.teacher_distilled_basis_mode),
        "--teacher_distilled_basis_guard_mode",
        "policy_val_nonregressive",
        "--teacher_distilled_basis_apply_mode",
        "blend",
        "--teacher_distilled_basis_blend",
        str(args.teacher_distilled_basis_blend),
        "--select_alpha_by_risk_gate",
        "--enable_policy_val_ssim_alpha_refinement",
        "--policy_val_ssim_alpha_refinement_steps",
        str(args.policy_val_ssim_alpha_refinement_steps),
        "--policy_val_ssim_alpha_refinement_min_alpha",
        str(args.policy_val_ssim_alpha_refinement_min_alpha),
        "--enable_preacceptance_policy_val_guard_repair",
        "--min_policy_val_samples",
        str(args.min_policy_val_samples),
        "--min_policy_val_relative_gain",
        str(args.min_policy_val_relative_gain),
        "--min_policy_val_positive_view_fraction",
        str(args.min_policy_val_positive_view_fraction),
        "--min_policy_val_cvar20_relative_gain",
        str(args.min_policy_val_cvar20_relative_gain),
        f"--min_policy_val_min_view_relative_gain={args.min_policy_val_min_view_relative_gain}",
        "--enable_policy_val_image_ssim_gate",
        f"--min_policy_val_ssim_mean_gain={args.min_policy_val_ssim_mean_gain}",
        "--min_policy_val_ssim_positive_view_fraction",
        str(args.min_policy_val_ssim_positive_view_fraction),
        f"--min_policy_val_ssim_min_view_gain={args.min_policy_val_ssim_min_view_gain}",
        "--enable_policy_val_image_l1_gate",
        "--min_policy_val_l1_mean_gain",
        str(args.min_policy_val_l1_mean_gain),
        "--min_policy_val_l1_positive_view_fraction",
        str(args.min_policy_val_l1_positive_view_fraction),
        f"--min_policy_val_l1_min_view_gain={args.min_policy_val_l1_min_view_gain}",
        "--enable_policy_val_face_gain_guard",
        "--face_gain_guard_min_positive_view_fraction",
        str(args.face_gain_guard_min_positive_view_fraction),
        "--enable_policy_val_bin_uncertainty_shrink",
        "--bin_uncertainty_shrink_policy_mode",
        str(args.bin_uncertainty_shrink_policy_mode),
        "--bin_uncertainty_shrink_min_bin_samples",
        str(args.bin_uncertainty_shrink_min_bin_samples),
        f"--bin_uncertainty_shrink_min_relative_gain={args.bin_uncertainty_shrink_min_relative_gain}",
        "--bin_uncertainty_shrink_min_positive_view_fraction",
        str(args.bin_uncertainty_shrink_min_positive_view_fraction),
        "--bin_uncertainty_shrink_fallback_shrink",
        str(args.bin_uncertainty_shrink_fallback_shrink),
        "--enable_target_support_candidate_selection",
        "--enable_policy_candidate_dominance_pruning",
        "--enable_policy_val_prior_bin_gain_hybrid",
        "--enable_prior_bin_gain_hybrid_l1_proxy_gate",
        "--enable_policy_val_source_mixture",
        "--enable_target_footprint_bin_certificate",
        "--enable_target_footprint_tail_risk_certificate",
        "--target_footprint_tail_risk_min_positive_view_fraction",
        str(args.target_footprint_tail_risk_min_positive_view_fraction),
        f"--target_footprint_tail_risk_min_min_view_gain={args.target_footprint_tail_risk_min_min_view_gain}",
        "--target_footprint_tail_risk_min_cvar20_view_gain",
        str(args.target_footprint_tail_risk_min_cvar20_view_gain),
        "--write_noop_on_reject",
        "--noop_fallback_source",
        str(args.noop_fallback_source),
        "--force",
    ]
    if bool(args.enable_adaptive_texture_size_ladder):
        cmd.extend(
            [
                "--enable_adaptive_texture_size_ladder",
                "--adaptive_texture_size_ladder_max_size",
                str(args.adaptive_texture_size_ladder_max_size),
                "--adaptive_texture_size_ladder_min_fit_samples_per_face",
                str(args.adaptive_texture_size_ladder_min_fit_samples_per_face),
                "--adaptive_texture_size_ladder_min_samples_per_current_bin",
                str(args.adaptive_texture_size_ladder_min_samples_per_current_bin),
                "--adaptive_texture_size_ladder_min_mean_l1",
                str(args.adaptive_texture_size_ladder_min_mean_l1),
            ]
        )
    if bool(args.enable_policy_val_alpha_midpoint_refinement):
        cmd.append("--enable_policy_val_alpha_midpoint_refinement")
    if bool(args.enable_policy_val_alpha_frontier_selection):
        cmd.extend(
            [
                "--enable_policy_val_alpha_frontier_selection",
                "--policy_val_alpha_frontier_mode",
                str(args.policy_val_alpha_frontier_mode),
                "--policy_val_alpha_frontier_min_relative_fraction",
                str(args.policy_val_alpha_frontier_min_relative_fraction),
                "--policy_val_alpha_frontier_min_ssim_fraction",
                str(args.policy_val_alpha_frontier_min_ssim_fraction),
                "--policy_val_alpha_frontier_min_l1_fraction",
                str(args.policy_val_alpha_frontier_min_l1_fraction),
                "--policy_val_alpha_frontier_alpha_penalty",
                str(args.policy_val_alpha_frontier_alpha_penalty),
                "--policy_val_alpha_frontier_knee_min_score_fraction",
                str(args.policy_val_alpha_frontier_knee_min_score_fraction),
                "--policy_val_alpha_frontier_knee_slope_drop_fraction",
                str(args.policy_val_alpha_frontier_knee_slope_drop_fraction),
                "--policy_val_alpha_frontier_tail_knee_min_score_fraction",
                str(args.policy_val_alpha_frontier_tail_knee_min_score_fraction),
                "--policy_val_alpha_frontier_tail_knee_min_regression_count",
                str(args.policy_val_alpha_frontier_tail_knee_min_regression_count),
                "--policy_val_alpha_frontier_tail_knee_eps",
                str(args.policy_val_alpha_frontier_tail_knee_eps),
            ]
        )
    if not bool(args.no_target_visible_energy_score):
        cmd.append("--enable_target_visible_energy_score")
    if bool(args.enable_coview_face_residual_transfer):
        cmd.extend(
            [
                "--enable_coview_face_residual_transfer",
                "--coview_transfer_max_faces",
                str(args.coview_transfer_max_faces),
                "--coview_transfer_neighbor_stride",
                str(args.coview_transfer_neighbor_stride),
                "--coview_transfer_min_source_samples",
                str(args.coview_transfer_min_source_samples),
                "--coview_transfer_min_source_mean_l1",
                str(args.coview_transfer_min_source_mean_l1),
                "--coview_transfer_min_edge_count",
                str(args.coview_transfer_min_edge_count),
                "--coview_transfer_min_target_pixels",
                str(args.coview_transfer_min_target_pixels),
                "--coview_transfer_min_policy_val_pixels",
                str(args.coview_transfer_min_policy_val_pixels),
                "--coview_transfer_max_views",
                str(args.coview_transfer_max_views),
                "--coview_transfer_residual_scale",
                str(args.coview_transfer_residual_scale),
                "--coview_transfer_synthetic_count",
                str(args.coview_transfer_synthetic_count),
                "--coview_transfer_existing_atlas_mode",
                str(args.coview_transfer_existing_atlas_mode),
                "--coview_transfer_blend_max_direct_bin_count",
                str(args.coview_transfer_blend_max_direct_bin_count),
            ]
        )
        if bool(args.coview_transfer_overwrite_existing_atlas):
            cmd.append("--coview_transfer_overwrite_existing_atlas")
    if bool(args.no_policy_val_ssim_alpha_refinement):
        cmd = [token for token in cmd if token != "--enable_policy_val_ssim_alpha_refinement"]
    if bool(args.no_preacceptance_policy_val_guard_repair):
        cmd = [token for token in cmd if token != "--enable_preacceptance_policy_val_guard_repair"]
    if bool(args.enable_policy_val_sparse_residual_materialization):
        cmd.append("--enable_policy_val_sparse_residual_materialization")
        _append_arg(cmd, "--sparse_materialization_seed_min_relative_gain", args.sparse_materialization_seed_min_relative_gain)
        _append_arg(cmd, "--sparse_materialization_min_bin_samples", args.sparse_materialization_min_bin_samples)
        _append_arg(cmd, "--sparse_materialization_min_relative_gain", args.sparse_materialization_min_relative_gain)
        _append_arg(cmd, "--sparse_materialization_min_positive_view_fraction", args.sparse_materialization_min_positive_view_fraction)
        _append_arg(cmd, "--sparse_materialization_max_mean_variance", args.sparse_materialization_max_mean_variance)
        _append_arg(cmd, "--sparse_materialization_min_mean_sign_consistency", args.sparse_materialization_min_mean_sign_consistency)
    if not bool(args.no_policy_val_bin_uncertainty_guard):
        cmd.extend(
            [
                "--enable_policy_val_bin_uncertainty_guard",
                "--bin_uncertainty_guard_min_bin_samples",
                str(args.bin_uncertainty_guard_min_bin_samples),
                "--bin_uncertainty_guard_min_positive_view_fraction",
                str(args.bin_uncertainty_guard_min_positive_view_fraction),
            ]
        )
    if bool(args.enable_policy_val_effective_margin_gate):
        cmd.append("--enable_policy_val_effective_margin_gate")
        _append_arg(cmd, "--min_policy_val_effective_relative_gain", args.min_policy_val_effective_relative_gain)
        _append_arg(cmd, "--min_policy_val_effective_ssim_gain", args.min_policy_val_effective_ssim_gain)
        _append_arg(cmd, "--min_policy_val_effective_l1_gain", args.min_policy_val_effective_l1_gain)
        _append_arg(cmd, "--min_policy_val_effective_ssim_cvar20_gain", args.min_policy_val_effective_ssim_cvar20_gain)
        _append_arg(cmd, "--min_policy_val_effective_l1_cvar20_gain", args.min_policy_val_effective_l1_cvar20_gain)
        if bool(args.enable_policy_val_image_lpips_gate):
            _append_arg(cmd, "--min_policy_val_effective_lpips_gain", args.min_policy_val_effective_lpips_gain)
            _append_arg(cmd, "--min_policy_val_effective_lpips_cvar20_gain", args.min_policy_val_effective_lpips_cvar20_gain)
    if bool(args.enable_policy_val_image_lpips_gate):
        cmd.extend(
            [
                "--enable_policy_val_image_lpips_gate",
                "--policy_val_lpips_max_size",
                str(args.policy_val_lpips_max_size),
                "--min_policy_val_lpips_mean_gain",
                str(args.min_policy_val_lpips_mean_gain),
                "--min_policy_val_lpips_positive_view_fraction",
                str(args.min_policy_val_lpips_positive_view_fraction),
                f"--min_policy_val_lpips_min_view_gain={args.min_policy_val_lpips_min_view_gain}",
            ]
        )
        _append_arg(cmd, "--min_policy_val_lpips_cvar20_view_gain", args.min_policy_val_lpips_cvar20_view_gain)
    if bool(args.no_policy_val_bin_uncertainty_shrink):
        cmd = [token for token in cmd if token != "--enable_policy_val_bin_uncertainty_shrink"]
    if bool(args.enable_policy_val_structure_aware_shrink):
        cmd.extend(
            [
                "--enable_policy_val_structure_aware_shrink",
                "--structure_shrink_l1_weight",
                str(args.structure_shrink_l1_weight),
                "--structure_shrink_gradient_weight",
                str(args.structure_shrink_gradient_weight),
                "--structure_shrink_edge_weight",
                str(args.structure_shrink_edge_weight),
                "--structure_shrink_risk_tau",
                str(args.structure_shrink_risk_tau),
                "--structure_shrink_max_penalty",
                str(args.structure_shrink_max_penalty),
            ]
        )
    if bool(args.enable_parent_edge_apply_shrink):
        cmd.extend(
            [
                "--enable_parent_edge_apply_shrink",
                "--parent_edge_apply_shrink_weight",
                str(args.parent_edge_apply_shrink_weight),
                "--parent_edge_apply_shrink_tau",
                str(args.parent_edge_apply_shrink_tau),
                "--parent_edge_apply_shrink_min_multiplier",
                str(args.parent_edge_apply_shrink_min_multiplier),
            ]
        )
    return _normalize_negative_numeric_args(cmd)


def _eval_cmd(args: argparse.Namespace, output_model: Path, results_path: Path, per_view_path: Path) -> list[str]:
    return [
        _python(),
        "scripts/car_model/evaluate_render_split_metrics.py",
        "--model_path",
        str(output_model),
        "--split",
        str(args.target_split),
        "--methods",
        str(args.method_name),
        "--output",
        str(results_path),
        "--per_view_output",
        str(per_view_path),
        "--merge_model_results",
    ]


def _maybe_wandb_log(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    if not bool(args.wandb):
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[vNext] W&B unavailable, skipping log: {exc}")
        return
    run = wandb.init(
        project=str(args.wandb_project),
        group=str(args.wandb_group) or None,
        name=str(args.wandb_name) or f"vnext-{args.scene}",
        mode=str(args.wandb_mode),
        config={
            "scene": str(args.scene),
            "method": METHOD,
            "method_name": str(args.method_name),
            "target_split": str(args.target_split),
            "protocol_audit": manifest.get("protocol_audit", {}),
            "settings": manifest.get("settings", {}),
        },
    )
    outputs = manifest.get("outputs", {}) or {}
    flat = {
        "vnext/status_complete": 1 if manifest.get("status") == "COMPLETE" else 0,
        "vnext/status_failed": 1 if manifest.get("status") == "FAILED" else 0,
        "vnext/protocol_audit_passed": 1 if (manifest.get("protocol_audit", {}) or {}).get("passed") else 0,
        "vnext/command_count": len(manifest.get("commands", []) or []),
        "vnext/error_count": len(manifest.get("errors", []) or []),
    }
    run.log(flat)
    run.summary.update(
        {
            **flat,
            "manifest_path": outputs.get("manifest_path", ""),
            "report_path": outputs.get("report_path", ""),
            "results_path": outputs.get("results_path", ""),
            "per_view_path": outputs.get("per_view_path", ""),
        }
    )
    run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one scene of vNext certified residual surface texturing.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source_model", type=Path, required=True)
    parser.add_argument("--fit_evidence_dir", type=Path, required=True)
    parser.add_argument("--target_evidence_dir", type=Path, required=True)
    parser.add_argument("--region_carrier_json", type=Path, required=True)
    parser.add_argument("--teacher_render_dir", type=Path, default=None)
    parser.add_argument("--parent_render_dir", type=Path, default=None)
    parser.add_argument(
        "--reparent_fit_parent_render_dir",
        type=Path,
        default=None,
        help=(
            "Optional stronger parent render directory used to rewrite fit evidence rgb_render/residuals "
            "before teacher-cache fitting. This is the v115 path for v106-anchored residual distillation."
        ),
    )
    parser.add_argument(
        "--reparent_target_parent_render_dir",
        type=Path,
        default=None,
        help=(
            "Optional stronger parent render directory used to rewrite target evidence rgb_render/residuals "
            "before no-GT stripping and target apply/fallback."
        ),
    )
    parser.add_argument("--reparent_parent_label", default="")
    parser.add_argument("--reparent_allow_resize", action="store_true")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--target_split", choices=("train", "test"), default="test")
    parser.add_argument("--base_method_name", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument("--method_name", default=DEFAULT_METHOD_NAME)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_teacher_cache", action="store_true")
    parser.add_argument("--skip_texture", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument(
        "--strict_no_target_gt_apply",
        action="store_true",
        help=(
            "Strip rgb_gt/residual/teacher keys from target evidence before adapter apply. "
            "Final GT images are populated only after target apply, immediately before evaluation."
        ),
    )
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "spcarnet_meshprior"))
    parser.add_argument("--wandb_group", default="vnext_certified_residual_texture")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--teacher_selection_mode", choices=("better_masked_residual", "teacher_gain", "residual_magnitude"), default="better_masked_residual")
    parser.add_argument("--teacher_render_error_margin", type=float, default=0.0)
    parser.add_argument("--teacher_parent_delta_min", type=float, default=0.005)
    parser.add_argument("--top_support_min_alpha", type=float, default=0.03)
    parser.add_argument("--top_support_limit", type=int, default=8192)
    parser.add_argument("--no_mask_teacher_target", action="store_true")
    parser.add_argument("--texture_size", type=int, default=16)
    parser.add_argument("--texture_size_candidates", default="8,16")
    parser.add_argument("--enable_adaptive_texture_size_ladder", action="store_true")
    parser.add_argument("--adaptive_texture_size_ladder_max_size", type=int, default=32)
    parser.add_argument("--adaptive_texture_size_ladder_min_fit_samples_per_face", type=float, default=512.0)
    parser.add_argument("--adaptive_texture_size_ladder_min_samples_per_current_bin", type=float, default=2.0)
    parser.add_argument("--adaptive_texture_size_ladder_min_mean_l1", type=float, default=0.002)
    parser.add_argument("--support_expansion_mode", choices=("none", "fit_residual_topk", "target_footprint_residual_debt"), default="target_footprint_residual_debt")
    parser.add_argument("--support_expansion_max_extra_faces_candidates", default="2048,4096")
    parser.add_argument("--support_expansion_min_face_samples", type=int, default=64)
    parser.add_argument("--target_footprint_residual_debt_match_level", choices=("bin", "face"), default="bin")
    parser.add_argument("--enable_coview_face_residual_transfer", action="store_true")
    parser.add_argument("--coview_transfer_max_faces", type=int, default=0)
    parser.add_argument("--coview_transfer_neighbor_stride", type=int, default=8)
    parser.add_argument("--coview_transfer_min_source_samples", type=int, default=64)
    parser.add_argument("--coview_transfer_min_source_mean_l1", type=float, default=0.0)
    parser.add_argument("--coview_transfer_min_edge_count", type=int, default=8)
    parser.add_argument("--coview_transfer_min_target_pixels", type=int, default=128)
    parser.add_argument("--coview_transfer_min_policy_val_pixels", type=int, default=128)
    parser.add_argument("--coview_transfer_max_views", type=int, default=0)
    parser.add_argument("--coview_transfer_residual_scale", type=float, default=0.25)
    parser.add_argument("--coview_transfer_synthetic_count", type=int, default=1)
    parser.add_argument("--coview_transfer_existing_atlas_mode", choices=("skip", "overwrite", "blend"), default="skip")
    parser.add_argument("--coview_transfer_blend_max_direct_bin_count", type=int, default=-1)
    parser.add_argument("--coview_transfer_overwrite_existing_atlas", action="store_true")
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--alpha_grid", default="0,0.125,0.25,0.5")
    parser.add_argument("--no_policy_val_ssim_alpha_refinement", action="store_true")
    parser.add_argument("--policy_val_ssim_alpha_refinement_steps", type=int, default=7)
    parser.add_argument("--policy_val_ssim_alpha_refinement_min_alpha", type=float, default=0.001)
    parser.add_argument("--enable_policy_val_alpha_midpoint_refinement", action="store_true")
    parser.add_argument("--enable_policy_val_alpha_frontier_selection", action="store_true")
    parser.add_argument(
        "--policy_val_alpha_frontier_mode",
        choices=("smallest_effective", "best_score", "knee", "tail_knee"),
        default="smallest_effective",
    )
    parser.add_argument("--policy_val_alpha_frontier_min_relative_fraction", type=float, default=0.75)
    parser.add_argument("--policy_val_alpha_frontier_min_ssim_fraction", type=float, default=0.75)
    parser.add_argument("--policy_val_alpha_frontier_min_l1_fraction", type=float, default=0.75)
    parser.add_argument("--policy_val_alpha_frontier_alpha_penalty", type=float, default=0.25)
    parser.add_argument("--policy_val_alpha_frontier_knee_min_score_fraction", type=float, default=0.55)
    parser.add_argument("--policy_val_alpha_frontier_knee_slope_drop_fraction", type=float, default=0.85)
    parser.add_argument("--policy_val_alpha_frontier_tail_knee_min_score_fraction", type=float, default=0.70)
    parser.add_argument("--policy_val_alpha_frontier_tail_knee_min_regression_count", type=int, default=3)
    parser.add_argument("--policy_val_alpha_frontier_tail_knee_eps", type=float, default=1.0e-10)
    parser.add_argument("--no_preacceptance_policy_val_guard_repair", action="store_true")
    parser.add_argument("--min_l1", type=float, default=0.0)
    parser.add_argument("--min_alpha", type=float, default=0.03)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=0.12)
    parser.add_argument("--max_abs_delta_rgb_candidates", default="0.08,0.12")
    parser.add_argument("--atlas_empty_bin_fill_mode", choices=("zero", "face_mean", "nearest_observed", "auto_policy"), default="auto_policy")
    parser.add_argument("--surface_multiscale_prior_mode", choices=("none", "count_pyramid", "local_patch"), default="local_patch")
    parser.add_argument("--surface_multiscale_prior_blend_candidates", default="0,0.25,0.5")
    parser.add_argument("--surface_multiscale_prior_min_direct_samples", type=int, default=1)
    parser.add_argument("--surface_multiscale_prior_min_sign_consistency", type=float, default=0.5)
    parser.add_argument("--surface_multiscale_prior_min_cosine", type=float, default=0.0)
    parser.add_argument("--view_conditioned_basis_mode", choices=("none", "camera_center_linear", "normal_camera_linear"), default="normal_camera_linear")
    parser.add_argument("--teacher_distilled_basis_mode", choices=("none", "face_uv_normal_camera_ridge", "face_uv_patch_mixture_ridge"), default="face_uv_patch_mixture_ridge")
    parser.add_argument("--teacher_distilled_basis_blend", type=float, default=0.5)
    parser.add_argument("--min_policy_val_samples", type=int, default=1024)
    parser.add_argument("--min_policy_val_relative_gain", type=float, default=0.0)
    parser.add_argument("--min_policy_val_positive_view_fraction", type=float, default=0.55)
    parser.add_argument("--min_policy_val_cvar20_relative_gain", type=float, default=0.0)
    parser.add_argument("--min_policy_val_min_view_relative_gain", type=float, default=-1.0e-6)
    parser.add_argument("--min_policy_val_ssim_mean_gain", type=float, default=-1.0e-7)
    parser.add_argument("--min_policy_val_ssim_positive_view_fraction", type=float, default=0.55)
    parser.add_argument("--min_policy_val_ssim_min_view_gain", type=float, default=-1.0e-5)
    parser.add_argument("--min_policy_val_l1_mean_gain", type=float, default=0.0)
    parser.add_argument("--min_policy_val_l1_positive_view_fraction", type=float, default=0.55)
    parser.add_argument("--min_policy_val_l1_min_view_gain", type=float, default=-1.0e-6)
    parser.add_argument("--enable_policy_val_image_lpips_gate", action="store_true")
    parser.add_argument("--policy_val_lpips_max_size", type=int, default=512)
    parser.add_argument("--min_policy_val_lpips_mean_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_lpips_positive_view_fraction", type=float, default=0.0)
    parser.add_argument("--min_policy_val_lpips_min_view_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_lpips_cvar20_view_gain", type=float, default=-1.0)
    parser.add_argument(
        "--enable_policy_val_effective_margin_gate",
        action="store_true",
        help=(
            "Require train policy-val wins to exceed explicit effect-size margins before target apply. "
            "This avoids accepting residual textures whose SSIM/L1 gains are near numerical noise."
        ),
    )
    parser.add_argument("--min_policy_val_effective_relative_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_effective_ssim_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_effective_l1_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_effective_ssim_cvar20_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_effective_l1_cvar20_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_effective_lpips_gain", type=float, default=-1.0)
    parser.add_argument("--min_policy_val_effective_lpips_cvar20_gain", type=float, default=-1.0)
    parser.add_argument("--face_gain_guard_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument("--enable_policy_val_sparse_residual_materialization", action="store_true")
    parser.add_argument("--sparse_materialization_seed_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--sparse_materialization_min_bin_samples", type=int, default=16)
    parser.add_argument("--sparse_materialization_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--sparse_materialization_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument("--sparse_materialization_max_mean_variance", type=float, default=-1.0)
    parser.add_argument("--sparse_materialization_min_mean_sign_consistency", type=float, default=0.0)
    parser.add_argument("--no_policy_val_bin_uncertainty_guard", action="store_true")
    parser.add_argument("--bin_uncertainty_guard_min_bin_samples", type=int, default=16)
    parser.add_argument("--bin_uncertainty_guard_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument("--no_policy_val_bin_uncertainty_shrink", action="store_true")
    parser.add_argument("--bin_uncertainty_shrink_policy_mode", choices=("sparse_positive", "keep_with_downweight"), default="keep_with_downweight")
    parser.add_argument("--bin_uncertainty_shrink_min_bin_samples", type=int, default=16)
    parser.add_argument("--bin_uncertainty_shrink_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--bin_uncertainty_shrink_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument("--bin_uncertainty_shrink_fallback_shrink", type=float, default=1.0)
    parser.add_argument("--enable_policy_val_structure_aware_shrink", action="store_true")
    parser.add_argument("--structure_shrink_l1_weight", type=float, default=0.0)
    parser.add_argument("--structure_shrink_gradient_weight", type=float, default=0.0)
    parser.add_argument("--structure_shrink_edge_weight", type=float, default=0.0)
    parser.add_argument("--structure_shrink_risk_tau", type=float, default=0.002)
    parser.add_argument("--structure_shrink_max_penalty", type=float, default=1.0)
    parser.add_argument("--enable_parent_edge_apply_shrink", action="store_true")
    parser.add_argument("--parent_edge_apply_shrink_weight", type=float, default=0.0)
    parser.add_argument("--parent_edge_apply_shrink_tau", type=float, default=0.05)
    parser.add_argument("--parent_edge_apply_shrink_min_multiplier", type=float, default=0.25)
    parser.add_argument(
        "--no_target_visible_energy_score",
        action="store_true",
        help=(
            "Disable the v116 target-visible residual-energy score. By default vNext ranks "
            "train-policy-safe candidates by GT-free target-visible residual energy before raw coverage."
        ),
    )
    parser.add_argument("--target_footprint_tail_risk_min_positive_view_fraction", type=float, default=0.75)
    parser.add_argument("--target_footprint_tail_risk_min_min_view_gain", type=float, default=-1.0e-8)
    parser.add_argument("--target_footprint_tail_risk_min_cvar20_view_gain", type=float, default=0.0)
    parser.add_argument(
        "--noop_fallback_source",
        choices=("source_model", "target_evidence"),
        default="target_evidence",
        help=(
            "Where rejected vNext candidates materialize the no-op parent. target_evidence keeps the same "
            "resolution/evaluation canvas as the residual adapter; source_model is available only when the "
            "source parent renders and target GT are known to share the same frame contract."
        ),
    )
    args = parser.parse_args(_normalize_negative_numeric_args(sys.argv[1:]))
    if bool(args.enable_policy_val_structure_aware_shrink) and bool(args.no_policy_val_bin_uncertainty_shrink):
        parser.error("--enable_policy_val_structure_aware_shrink requires bin uncertainty shrink to stay enabled")
    if float(args.structure_shrink_l1_weight) < 0.0:
        parser.error("--structure_shrink_l1_weight must be >= 0")
    if float(args.structure_shrink_gradient_weight) < 0.0:
        parser.error("--structure_shrink_gradient_weight must be >= 0")
    if float(args.structure_shrink_edge_weight) < 0.0:
        parser.error("--structure_shrink_edge_weight must be >= 0")
    if float(args.structure_shrink_risk_tau) < 0.0:
        parser.error("--structure_shrink_risk_tau must be >= 0")
    if not 0.0 <= float(args.structure_shrink_max_penalty) <= 1.0:
        parser.error("--structure_shrink_max_penalty must be in [0, 1]")
    if float(args.parent_edge_apply_shrink_weight) < 0.0:
        parser.error("--parent_edge_apply_shrink_weight must be >= 0")
    if float(args.parent_edge_apply_shrink_tau) < 0.0:
        parser.error("--parent_edge_apply_shrink_tau must be >= 0")
    if not 0.0 <= float(args.parent_edge_apply_shrink_min_multiplier) <= 1.0:
        parser.error("--parent_edge_apply_shrink_min_multiplier must be in [0, 1]")
    if int(args.sparse_materialization_min_bin_samples) < 1:
        parser.error("--sparse_materialization_min_bin_samples must be >= 1")
    if not 0.0 <= float(args.sparse_materialization_min_positive_view_fraction) <= 1.0:
        parser.error("--sparse_materialization_min_positive_view_fraction must be in [0, 1]")
    if float(args.sparse_materialization_min_mean_sign_consistency) < 0.0:
        parser.error("--sparse_materialization_min_mean_sign_consistency must be >= 0")
    if not bool(args.enable_policy_val_image_lpips_gate) and (
        float(args.min_policy_val_effective_lpips_gain) > -1.0
        or float(args.min_policy_val_effective_lpips_cvar20_gain) > -1.0
    ):
        parser.error("LPIPS effective thresholds require --enable_policy_val_image_lpips_gate")
    if int(args.adaptive_texture_size_ladder_max_size) <= 0:
        parser.error("--adaptive_texture_size_ladder_max_size must be > 0")
    if float(args.adaptive_texture_size_ladder_min_fit_samples_per_face) < 0.0:
        parser.error("--adaptive_texture_size_ladder_min_fit_samples_per_face must be >= 0")
    if float(args.adaptive_texture_size_ladder_min_samples_per_current_bin) < 0.0:
        parser.error("--adaptive_texture_size_ladder_min_samples_per_current_bin must be >= 0")
    if float(args.adaptive_texture_size_ladder_min_mean_l1) < 0.0:
        parser.error("--adaptive_texture_size_ladder_min_mean_l1 must be >= 0")
    for frontier_fraction_name in (
        "policy_val_alpha_frontier_min_relative_fraction",
        "policy_val_alpha_frontier_min_ssim_fraction",
        "policy_val_alpha_frontier_min_l1_fraction",
        "policy_val_alpha_frontier_knee_min_score_fraction",
        "policy_val_alpha_frontier_knee_slope_drop_fraction",
        "policy_val_alpha_frontier_tail_knee_min_score_fraction",
    ):
        frontier_fraction = float(getattr(args, frontier_fraction_name))
        if not 0.0 <= frontier_fraction <= 1.0:
            parser.error(f"--{frontier_fraction_name} must be in [0, 1]")
    if float(args.policy_val_alpha_frontier_alpha_penalty) < 0.0:
        parser.error("--policy_val_alpha_frontier_alpha_penalty must be >= 0")
    if int(args.policy_val_alpha_frontier_tail_knee_min_regression_count) < 1:
        parser.error("--policy_val_alpha_frontier_tail_knee_min_regression_count must be >= 1")
    if float(args.policy_val_alpha_frontier_tail_knee_eps) < 0.0:
        parser.error("--policy_val_alpha_frontier_tail_knee_eps must be >= 0")
    return args


def main() -> int:
    args = parse_args()
    run_root = Path(args.output_root) / str(args.scene)
    logs_dir = run_root / "logs"
    reports_dir = run_root / "reports"
    reparented_fit_evidence_dir = run_root / "fit_evidence_reparented"
    reparented_target_evidence_dir = run_root / "target_evidence_reparented"
    teacher_cache_dir = run_root / "teacher_surface_evidence"
    stripped_target_evidence_dir = run_root / "target_evidence_no_gt"
    output_model = run_root / "model"
    results_path = reports_dir / f"{args.scene}_{args.method_name}_{args.target_split}_results.json"
    per_view_path = reports_dir / f"{args.scene}_{args.method_name}_{args.target_split}_per_view.json"
    manifest_path = reports_dir / f"{args.scene}_vnext_certified_residual_texture_manifest.json"
    report_path = reports_dir / f"{args.scene}_vnext_certified_residual_texture_report.md"
    eval_gt_audit_path = reports_dir / f"{args.scene}_{args.method_name}_{args.target_split}_eval_gt_population_audit.json"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["WANDB_MODE"] = str(args.wandb_mode)
    if args.gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not args.skip_teacher_cache and args.teacher_render_dir is None:
        raise SystemExit("--teacher_render_dir is required unless --skip_teacher_cache is set")

    args._effective_fit_evidence_dir = (
        reparented_fit_evidence_dir
        if args.reparent_fit_parent_render_dir is not None
        else Path(args.fit_evidence_dir)
    )
    args._effective_target_evidence_dir = (
        reparented_target_evidence_dir
        if args.reparent_target_parent_render_dir is not None
        else Path(args.target_evidence_dir)
    )

    texture_fit_evidence_dir = Path(args._effective_fit_evidence_dir) if args.skip_teacher_cache else teacher_cache_dir
    adapter_target_evidence_dir = (
        stripped_target_evidence_dir
        if bool(args.strict_no_target_gt_apply)
        else Path(args._effective_target_evidence_dir)
    )

    commands: list[dict[str, Any]] = []
    if args.reparent_fit_parent_render_dir is not None and (
        not args.skip_texture or not args.skip_teacher_cache
    ):
        commands.append(
            command_record(
                "reparent_fit_evidence",
                _reparent_evidence_cmd(
                    args,
                    base_evidence_dir=Path(args.fit_evidence_dir),
                    parent_render_dir=Path(args.reparent_fit_parent_render_dir),
                    out_dir=reparented_fit_evidence_dir,
                    split_label="fit",
                ),
                log_path=logs_dir / "00_reparent_fit_evidence.log",
            )
        )
    if args.reparent_target_parent_render_dir is not None and not args.skip_texture:
        commands.append(
            command_record(
                "reparent_target_evidence",
                _reparent_evidence_cmd(
                    args,
                    base_evidence_dir=Path(args.target_evidence_dir),
                    parent_render_dir=Path(args.reparent_target_parent_render_dir),
                    out_dir=reparented_target_evidence_dir,
                    split_label="target",
                ),
                log_path=logs_dir / "00b_reparent_target_evidence.log",
            )
        )
    if not args.skip_teacher_cache:
        commands.append(command_record("build_teacher_surface_evidence", _teacher_cache_cmd(args, teacher_cache_dir), log_path=logs_dir / "01_teacher_cache.log"))
    if bool(args.strict_no_target_gt_apply) and not args.skip_texture:
        commands.append(
            command_record(
                "strip_target_evidence_no_gt",
                _strip_target_evidence_cmd(args, stripped_target_evidence_dir),
                log_path=logs_dir / "01b_strip_target_evidence_no_gt.log",
            )
        )
    if not args.skip_texture:
        commands.append(
            command_record(
                "apply_certified_residual_texture",
                _texture_cmd(args, texture_fit_evidence_dir, adapter_target_evidence_dir, output_model),
                log_path=logs_dir / "02_certified_texture.log",
            )
        )
    if bool(args.strict_no_target_gt_apply) and not args.skip_texture and not args.skip_eval:
        commands.append(
            command_record(
                "populate_eval_gt_from_target_evidence",
                _populate_eval_gt_cmd(args, output_model, eval_gt_audit_path),
                log_path=logs_dir / "02b_populate_eval_gt.log",
            )
        )
    if not args.skip_eval:
        commands.append(command_record("evaluate_vnext_target", _eval_cmd(args, output_model, results_path, per_view_path), log_path=logs_dir / "03_eval.log"))

    protocol_audit = make_protocol_audit(
        fit_split="train",
        policy_val_split="train",
        target_split=str(args.target_split),
        teacher_uses_gt=not bool(args.no_mask_teacher_target),
        selection_uses_test_gt=False,
        capacity_selected_on="train_policy_val_and_gt_free_target_footprint",
        thresholds_selected_on="train_policy_val",
        target_gt_visible_to_selection=False,
        target_gt_visible_to_apply=not bool(args.strict_no_target_gt_apply),
        target_gt_visible_to_eval=not bool(args.skip_eval),
        target_forbidden_keys_stripped=bool(args.strict_no_target_gt_apply),
    )
    inputs = {
        "source_model": path_record(args.source_model),
        "fit_evidence_dir": path_record(args.fit_evidence_dir),
        "target_evidence_dir": path_record(args.target_evidence_dir),
        "effective_fit_evidence_dir": path_record(args._effective_fit_evidence_dir),
        "effective_target_evidence_dir": path_record(args._effective_target_evidence_dir),
        "adapter_target_evidence_dir": path_record(adapter_target_evidence_dir),
        "reparented_fit_evidence_dir": path_record(reparented_fit_evidence_dir),
        "reparented_target_evidence_dir": path_record(reparented_target_evidence_dir),
        "reparent_fit_parent_render_dir": path_record(args.reparent_fit_parent_render_dir)
        if args.reparent_fit_parent_render_dir
        else None,
        "reparent_target_parent_render_dir": path_record(args.reparent_target_parent_render_dir)
        if args.reparent_target_parent_render_dir
        else None,
        "region_carrier_json": path_record(args.region_carrier_json, hash_file=True),
        "teacher_render_dir": path_record(args.teacher_render_dir) if args.teacher_render_dir else None,
        "parent_render_dir": path_record(args.parent_render_dir) if args.parent_render_dir else None,
        "texture_fit_evidence_dir": path_record(texture_fit_evidence_dir),
        "stripped_target_evidence_dir": path_record(stripped_target_evidence_dir),
    }
    settings = {key: value for key, value in vars(args).items() if key not in {"source_model", "fit_evidence_dir", "target_evidence_dir", "region_carrier_json", "teacher_render_dir", "parent_render_dir"}}
    manifest = make_run_manifest(
        method=METHOD,
        scene=str(args.scene),
        run_root=run_root,
        inputs=inputs,
        settings=settings,
        commands=commands,
        protocol_audit=protocol_audit,
        status="DRY_RUN" if args.dry_run else "RUNNING",
    )
    write_json(manifest_path, manifest)
    write_vnext_report(report_path, manifest)

    errors: list[dict[str, Any]] = []
    executed: list[dict[str, Any]] = []
    for command in commands:
        result = _run_step(command, env=env, dry_run=bool(args.dry_run))
        executed.append(result)
        manifest["commands"] = executed + commands[len(executed) :]
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json(manifest_path, manifest)
        write_vnext_report(report_path, manifest)
        if int(result.get("returncode") or 0) != 0:
            errors.append({"step": result.get("name"), "returncode": result.get("returncode"), "log_path": result.get("log_path")})
            break

    status = "DRY_RUN" if args.dry_run else ("FAILED" if errors else "COMPLETE")
    manifest["status"] = status
    manifest["errors"] = errors
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["outputs"] = {
        "reparented_fit_evidence_dir": str(reparented_fit_evidence_dir),
        "reparented_target_evidence_dir": str(reparented_target_evidence_dir),
        "teacher_cache_dir": str(teacher_cache_dir),
        "output_model": str(output_model),
        "results_path": str(results_path),
        "per_view_path": str(per_view_path),
        "eval_gt_audit_path": str(eval_gt_audit_path),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
    }
    write_json(manifest_path, manifest)
    write_vnext_report(report_path, manifest)
    _maybe_wandb_log(args, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
