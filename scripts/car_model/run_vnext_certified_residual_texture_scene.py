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

import numpy as np

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
TARGET_APPLY_FORBIDDEN_KEYS = {
    "rgb_gt",
    "residual_rgb",
    "residual_l1",
    "teacher_residual_rgb",
    "teacher_residual_l1",
    "teacher_residual_rgb_raw",
    "teacher_better_mask",
    "teacher_gain_l1",
    "teacher_parent_delta_l1",
}


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


def _evidence_views(evidence_dir: Path) -> list[Path]:
    views_dir = Path(evidence_dir) / "views"
    if views_dir.is_dir():
        return sorted(views_dir.glob("*.npz"))
    return sorted(Path(evidence_dir).glob("*.npz"))


def _verify_target_apply_forbidden_keys(evidence_dir: Path) -> dict[str, Any]:
    paths = _evidence_views(evidence_dir)
    audit: dict[str, Any] = {
        "enabled": True,
        "mode": "strict_no_target_gt_apply_forbidden_key_preflight",
        "evidence_dir": str(evidence_dir),
        "view_count": int(len(paths)),
        "forbidden_keys": sorted(TARGET_APPLY_FORBIDDEN_KEYS),
        "bad_view_count": 0,
        "bad_views": [],
        "sample_keys": [],
        "target_gt_visible_to_apply": False,
        "target_residual_visible_to_apply": False,
        "passed": False,
    }
    if not paths:
        audit["reason"] = "no_target_evidence_views"
        return audit
    bad_views: list[dict[str, Any]] = []
    sample_keys: list[dict[str, Any]] = []
    for idx, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as z:
            keys = sorted(str(key) for key in z.files)
        forbidden_present = sorted(set(keys) & TARGET_APPLY_FORBIDDEN_KEYS)
        if idx < 4:
            sample_keys.append({"path": str(path), "keys": keys})
        if forbidden_present:
            bad_views.append({"path": str(path), "forbidden_keys": forbidden_present})
    audit["bad_view_count"] = int(len(bad_views))
    audit["bad_views"] = bad_views[:32]
    audit["sample_keys"] = sample_keys
    audit["passed"] = not bad_views
    if bad_views:
        audit["target_gt_visible_to_apply"] = any(
            "rgb_gt" in set(row.get("forbidden_keys", [])) for row in bad_views
        )
        audit["target_residual_visible_to_apply"] = any(
            bool((set(row.get("forbidden_keys", [])) - {"rgb_gt"})) for row in bad_views
        )
    return audit


def _compute_adaptive_residual_activity_threshold(
    evidence_dir: Path,
    *,
    residual_l1_key: str,
    min_alpha: float,
    base_min_l1: float,
    quantile: float,
    floor: float,
    max_samples_per_view: int,
) -> tuple[float, dict[str, Any]]:
    summary: dict[str, Any] = {
        "enabled": True,
        "mode": "train_only_adaptive_residual_activity_threshold",
        "evidence_dir": str(evidence_dir),
        "residual_l1_key": str(residual_l1_key),
        "base_min_l1": float(base_min_l1),
        "quantile": float(quantile),
        "floor": float(floor),
        "max_samples_per_view": int(max_samples_per_view),
        "uses_target_or_test_gt": False,
    }
    paths = _evidence_views(evidence_dir)
    if not paths:
        summary["reason"] = "no_fit_evidence_views"
        return float(base_min_l1), summary
    samples: list[np.ndarray] = []
    valid_view_count = 0
    total_valid_samples = 0
    zero_sample_count = 0
    for path in paths:
        try:
            with np.load(path) as z:
                if residual_l1_key not in z or "face_id" not in z or "barycentric" not in z:
                    continue
                mask = np.asarray(z["face_id"], dtype=np.int64) >= 0
                if "barycentric_valid" in z:
                    mask &= np.asarray(z["barycentric_valid"]).astype(bool)
                if "alpha" in z:
                    mask &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
                bary = np.asarray(z["barycentric"], dtype=np.float32)
                mask &= np.all(np.isfinite(bary), axis=0)
                mask &= np.all(bary >= -0.05, axis=0)
                mask &= np.all(bary <= 1.05, axis=0)
                values = np.asarray(z[residual_l1_key], dtype=np.float32)[mask].reshape(-1)
        except Exception as exc:
            summary.setdefault("skipped_views", []).append({"path": str(path), "error": str(exc)})
            continue
        if values.size == 0:
            continue
        valid_view_count += 1
        total_valid_samples += int(values.size)
        zero_sample_count += int(np.sum(values <= 1.0e-8))
        if int(max_samples_per_view) > 0 and values.size > int(max_samples_per_view):
            take = np.linspace(0, values.size - 1, int(max_samples_per_view), dtype=np.int64)
            values = values[take]
        samples.append(values.astype(np.float32, copy=False))
    if not samples:
        summary["reason"] = "no_valid_residual_l1_samples"
        summary["fit_view_count"] = int(len(paths))
        return float(base_min_l1), summary
    merged = np.concatenate(samples, axis=0)
    q = float(np.clip(quantile, 0.0, 1.0))
    threshold = float(np.quantile(merged, q))
    threshold = max(float(base_min_l1), float(floor), threshold)
    summary.update(
        {
            "fit_view_count": int(len(paths)),
            "valid_view_count": int(valid_view_count),
            "total_valid_samples": int(total_valid_samples),
            "sampled_values": int(merged.size),
            "zero_fraction": float(zero_sample_count / max(1, total_valid_samples)),
            "selected_min_l1": float(threshold),
            "sample_quantiles": {
                "0.50": float(np.quantile(merged, 0.50)),
                "0.75": float(np.quantile(merged, 0.75)),
                "0.90": float(np.quantile(merged, 0.90)),
                "0.95": float(np.quantile(merged, 0.95)),
                "0.99": float(np.quantile(merged, 0.99)),
            },
        }
    )
    return float(threshold), summary


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
    if bool(args.teacher_cache_rewrite_rgb_render_to_parent):
        cmd.append("--rewrite_rgb_render_to_parent")
    if args.no_mask_teacher_target:
        cmd.append("--no-mask_target")
    if bool(args.reparent_allow_resize):
        cmd.append("--allow_resize")
    if str(args.teacher_cache_copy_mode) != "copy":
        cmd.extend(["--copy_mode", str(args.teacher_cache_copy_mode)])
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
    if str(args.reparent_copy_mode) != "copy":
        cmd.extend(["--copy_mode", str(args.reparent_copy_mode)])
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


def _verify_target_evidence_no_gt_cmd(target_evidence_dir: Path, audit_path: Path) -> list[str]:
    return [
        _python(),
        "scripts/car_model/ecsr_verify_target_evidence_no_gt.py",
        "--target_evidence_dir",
        str(target_evidence_dir),
        "--audit_path",
        str(audit_path),
    ]


def _populate_eval_gt_cmd(args: argparse.Namespace, output_model: Path, audit_path: Path) -> list[str]:
    return [
        _python(),
        "scripts/car_model/ecsr_populate_eval_gt_from_target_evidence.py",
        "--target_evidence_dir",
        str(args._effective_eval_gt_evidence_dir),
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


def _resolved_same_path(a: Path | None, b: Path | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return Path(a).absolute() == Path(b).absolute()


def _apply_distillation_profile(args: argparse.Namespace) -> None:
    profile = str(getattr(args, "distillation_profile", "none"))
    if profile == "none":
        args._distillation_profile_audit = {
            "enabled": False,
            "profile": "none",
        }
        return
    if profile != "teacher_to_reparented_parent":
        raise ValueError(f"unknown distillation profile: {profile}")
    if args.teacher_render_dir is None:
        raise SystemExit("--distillation_profile teacher_to_reparented_parent requires --teacher_render_dir")
    parent_render_dir = args.parent_render_dir or args.reparent_fit_parent_render_dir
    if parent_render_dir is None:
        raise SystemExit(
            "--distillation_profile teacher_to_reparented_parent requires "
            "--parent_render_dir or --reparent_fit_parent_render_dir"
        )
    if args.parent_render_dir is not None and args.reparent_fit_parent_render_dir is not None and not _resolved_same_path(
        Path(args.parent_render_dir), Path(args.reparent_fit_parent_render_dir)
    ):
        raise SystemExit(
            "--distillation_profile teacher_to_reparented_parent requires --parent_render_dir and "
            "--reparent_fit_parent_render_dir to match when both are provided"
        )
    if _resolved_same_path(Path(args.teacher_render_dir), Path(parent_render_dir)):
        raise SystemExit(
            "--distillation_profile teacher_to_reparented_parent requires distinct teacher and parent renders; "
            "using the same path would create a near-zero teacher residual target"
        )
    target_parent_defaulted = args.reparent_target_parent_render_dir is None
    if target_parent_defaulted and str(args.target_split) != "train":
        raise SystemExit(
            "--distillation_profile teacher_to_reparented_parent requires --reparent_target_parent_render_dir "
            "for non-train target splits; the fit parent render directory usually contains train frames only"
        )
    if args.parent_render_dir is None:
        args.parent_render_dir = Path(parent_render_dir)
    if args.reparent_fit_parent_render_dir is None:
        args.reparent_fit_parent_render_dir = Path(parent_render_dir)
    if target_parent_defaulted:
        args.reparent_target_parent_render_dir = Path(parent_render_dir)
    if not str(args.reparent_parent_label):
        args.reparent_parent_label = "distill_parent"
    args.strict_no_target_gt_apply = True
    args._distillation_profile_audit = {
        "enabled": True,
        "profile": profile,
        "intent": "fit teacher residuals as teacher_render - parent_render, then apply the baked residual on a GT-free target footprint",
        "teacher_render_dir": str(args.teacher_render_dir),
        "parent_render_dir": str(args.parent_render_dir),
        "reparent_fit_parent_render_dir": str(args.reparent_fit_parent_render_dir),
        "reparent_target_parent_render_dir": str(args.reparent_target_parent_render_dir),
        "target_parent_defaulted_from_fit_parent": bool(target_parent_defaulted),
        "strict_no_target_gt_apply_forced": True,
        "zero_residual_guard": "teacher_render_dir must differ from parent_render_dir",
        "target_or_test_gt_visible_to_apply": False,
    }


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
        str(getattr(args, "_effective_min_l1", args.min_l1)),
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
        "--view_cluster_expert_count",
        str(args.view_cluster_expert_count),
        "--view_cluster_feature_mode",
        str(args.view_cluster_feature_mode),
        "--view_cluster_min_views",
        str(args.view_cluster_min_views),
        "--view_cluster_min_bin_samples",
        str(args.view_cluster_min_bin_samples),
        "--view_cluster_fallback_mode",
        str(args.view_cluster_fallback_mode),
        "--teacher_distilled_basis_mode",
        str(args.teacher_distilled_basis_mode),
        "--teacher_distilled_basis_guard_mode",
        "policy_val_nonregressive",
        "--teacher_distilled_basis_apply_mode",
        "blend",
        "--teacher_distilled_basis_blend",
        str(args.teacher_distilled_basis_blend),
        "--teacher_distilled_basis_min_face_samples",
        str(args.teacher_distilled_basis_min_face_samples),
        "--teacher_distilled_basis_ridge",
        str(args.teacher_distilled_basis_ridge),
        "--teacher_distilled_low_rank_texture_rank",
        str(args.teacher_distilled_low_rank_texture_rank),
        "--teacher_distilled_low_rank_texture_rank_candidates",
        str(args.teacher_distilled_low_rank_texture_rank_candidates),
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
    ]
    if not bool(args.no_policy_val_face_gain_guard):
        cmd.extend(
            [
                "--enable_policy_val_face_gain_guard",
                "--face_gain_guard_min_positive_view_fraction",
                str(args.face_gain_guard_min_positive_view_fraction),
            ]
        )
    cmd.extend(
        [
            "--enable_policy_val_bin_uncertainty_shrink",
            "--bin_uncertainty_shrink_policy_mode",
            str(args.bin_uncertainty_shrink_policy_mode),
            "--bin_uncertainty_shrink_min_bin_samples",
            str(args.bin_uncertainty_shrink_min_bin_samples),
            "--bin_uncertainty_shrink_min_bin_views",
            str(args.bin_uncertainty_shrink_min_bin_views),
            f"--bin_uncertainty_shrink_min_relative_gain={args.bin_uncertainty_shrink_min_relative_gain}",
            "--bin_uncertainty_shrink_min_positive_view_fraction",
            str(args.bin_uncertainty_shrink_min_positive_view_fraction),
            "--bin_uncertainty_shrink_fallback_shrink",
            str(args.bin_uncertainty_shrink_fallback_shrink),
            "--write_noop_on_reject",
            "--noop_fallback_source",
            str(args.noop_fallback_source),
            "--force",
        ]
    )
    if not bool(args.no_target_support_candidate_selection):
        cmd.append("--enable_target_support_candidate_selection")
    if not bool(args.no_policy_candidate_dominance_pruning):
        cmd.append("--enable_policy_candidate_dominance_pruning")
    if not bool(args.no_policy_val_prior_bin_gain_hybrid):
        cmd.extend(
            [
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
            ]
        )
    if bool(args.enable_view_cluster_local_shrink):
        cmd.append("--enable_view_cluster_local_shrink")
    if bool(args.view_cluster_local_shrink_global_fallback):
        cmd.append("--view_cluster_local_shrink_global_fallback")
    if bool(args.enable_policy_val_image_l1_bin_certificate):
        cmd.extend(
            [
                "--enable_policy_val_image_l1_bin_certificate",
                "--image_l1_bin_certificate_mode",
                str(args.image_l1_bin_certificate_mode),
                "--image_l1_bin_certificate_min_relative_gain",
                str(args.image_l1_bin_certificate_min_relative_gain),
                "--image_l1_bin_certificate_min_positive_view_fraction",
                str(args.image_l1_bin_certificate_min_positive_view_fraction),
                "--image_l1_bin_certificate_gain_tau",
                str(args.image_l1_bin_certificate_gain_tau),
                "--image_l1_bin_certificate_pool_radius",
                str(args.image_l1_bin_certificate_pool_radius),
            ]
        )
        if bool(args.enable_policy_val_image_l1_region_expansion):
            cmd.extend(
                [
                    "--enable_policy_val_image_l1_region_expansion",
                    "--image_l1_region_expansion_radius",
                    str(args.image_l1_region_expansion_radius),
                    "--image_l1_region_expansion_max_bins_per_seed",
                    str(args.image_l1_region_expansion_max_bins_per_seed),
                    "--image_l1_region_expansion_min_neighbor_samples",
                    str(args.image_l1_region_expansion_min_neighbor_samples),
                    "--image_l1_region_expansion_min_neighbor_views",
                    str(args.image_l1_region_expansion_min_neighbor_views),
                    "--image_l1_region_expansion_max_negative_relative_gain",
                    str(args.image_l1_region_expansion_max_negative_relative_gain),
                    "--image_l1_region_expansion_max_negative_image_l1_gain",
                    str(args.image_l1_region_expansion_max_negative_image_l1_gain),
                    "--image_l1_region_expansion_shrink_decay",
                    str(args.image_l1_region_expansion_shrink_decay),
                ]
            )
    if bool(args.enable_policy_val_image_l1_bin_alpha_optimization):
        cmd.extend(
            [
                "--enable_policy_val_image_l1_bin_alpha_optimization",
                "--image_l1_bin_alpha_grid",
                str(args.image_l1_bin_alpha_grid),
                "--image_l1_bin_alpha_max_alpha",
                str(args.image_l1_bin_alpha_max_alpha),
                "--image_l1_bin_alpha_min_bin_samples",
                str(args.image_l1_bin_alpha_min_bin_samples),
                "--image_l1_bin_alpha_min_relative_gain",
                str(args.image_l1_bin_alpha_min_relative_gain),
                "--image_l1_bin_alpha_min_positive_view_fraction",
                str(args.image_l1_bin_alpha_min_positive_view_fraction),
                "--image_l1_bin_alpha_count_tau",
                str(args.image_l1_bin_alpha_count_tau),
                "--image_l1_bin_alpha_fallback_mode",
                str(args.image_l1_bin_alpha_fallback_mode),
                "--image_l1_bin_alpha_max_profile_bins",
                str(args.image_l1_bin_alpha_max_profile_bins),
            ]
        )
    if bool(args.enable_policy_val_image_linear_residual_generator):
        cmd.extend(
            [
                "--enable_policy_val_image_linear_residual_generator",
                "--image_linear_generator_feature_mode",
                str(args.image_linear_generator_feature_mode),
                "--image_linear_generator_ridge",
                str(args.image_linear_generator_ridge),
                "--image_linear_generator_train_max_samples_per_view",
                str(args.image_linear_generator_train_max_samples_per_view),
                "--image_linear_generator_max_train_samples",
                str(args.image_linear_generator_max_train_samples),
                "--image_linear_generator_output_cap",
                str(args.image_linear_generator_output_cap),
                "--image_linear_generator_loss_mode",
                str(args.image_linear_generator_loss_mode),
                "--image_linear_generator_irls_iterations",
                str(args.image_linear_generator_irls_iterations),
                "--image_linear_generator_huber_delta",
                str(args.image_linear_generator_huber_delta),
                "--image_linear_generator_training_sample_policy",
                str(args.image_linear_generator_training_sample_policy),
                "--image_linear_generator_min_descent_margin",
                str(args.image_linear_generator_min_descent_margin),
                "--image_linear_generator_min_training_samples",
                str(args.image_linear_generator_min_training_samples),
                "--image_linear_generator_alpha_grid",
                str(args.image_linear_generator_alpha_grid),
                "--image_linear_generator_expert_mode",
                str(args.image_linear_generator_expert_mode),
                "--image_linear_generator_expert_min_training_samples",
                str(args.image_linear_generator_expert_min_training_samples),
                "--image_linear_generator_expert_shrink_tau",
                str(args.image_linear_generator_expert_shrink_tau),
                "--image_linear_generator_face_reliability_mode",
                str(args.image_linear_generator_face_reliability_mode),
                "--image_linear_generator_face_reliability_min_face_samples",
                str(args.image_linear_generator_face_reliability_min_face_samples),
                "--image_linear_generator_face_reliability_min_relative_gain",
                str(args.image_linear_generator_face_reliability_min_relative_gain),
                "--image_linear_generator_face_reliability_min_positive_view_fraction",
                str(args.image_linear_generator_face_reliability_min_positive_view_fraction),
                "--image_linear_generator_face_reliability_fallback_multiplier",
                str(args.image_linear_generator_face_reliability_fallback_multiplier),
            ]
        )
        if bool(args.image_linear_generator_allow_unvalidated_base_pixels):
            cmd.append("--image_linear_generator_allow_unvalidated_base_pixels")
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
        if bool(args.enable_policy_val_sparse_materialization_frontier):
            cmd.append("--enable_policy_val_sparse_materialization_frontier")
        _append_arg(cmd, "--sparse_materialization_seed_min_relative_gain", args.sparse_materialization_seed_min_relative_gain)
        _append_arg(cmd, "--sparse_materialization_min_bin_samples", args.sparse_materialization_min_bin_samples)
        _append_arg(cmd, "--sparse_materialization_min_bin_views", args.sparse_materialization_min_bin_views)
        _append_arg(cmd, "--sparse_materialization_min_relative_gain", args.sparse_materialization_min_relative_gain)
        _append_arg(cmd, "--sparse_materialization_min_view_relative_gain", args.sparse_materialization_min_view_relative_gain)
        _append_arg(cmd, "--sparse_materialization_min_positive_view_fraction", args.sparse_materialization_min_positive_view_fraction)
        _append_arg(
            cmd,
            "--sparse_materialization_frontier_min_positive_view_fraction",
            args.sparse_materialization_frontier_min_positive_view_fraction,
        )
        _append_arg(
            cmd,
            "--sparse_materialization_frontier_min_risk_adjusted_gain",
            args.sparse_materialization_frontier_min_risk_adjusted_gain,
        )
        _append_arg(
            cmd,
            "--sparse_materialization_frontier_min_sample_quantile",
            args.sparse_materialization_frontier_min_sample_quantile,
        )
        _append_arg(cmd, "--sparse_materialization_max_mean_variance", args.sparse_materialization_max_mean_variance)
        _append_arg(cmd, "--sparse_materialization_min_mean_sign_consistency", args.sparse_materialization_min_mean_sign_consistency)
        if bool(args.enable_sparse_materialization_target_visible_expansion):
            cmd.append("--enable_sparse_materialization_target_visible_expansion")
            _append_arg(
                cmd,
                "--sparse_materialization_target_visible_min_pixels",
                args.sparse_materialization_target_visible_min_pixels,
            )
            _append_arg(
                cmd,
                "--sparse_materialization_target_visible_min_views",
                args.sparse_materialization_target_visible_min_views,
            )
            _append_arg(
                cmd,
                "--sparse_materialization_target_visible_min_policy_samples",
                args.sparse_materialization_target_visible_min_policy_samples,
            )
            _append_arg(
                cmd,
                "--sparse_materialization_target_visible_min_positive_view_fraction",
                args.sparse_materialization_target_visible_min_positive_view_fraction,
            )
            _append_arg(
                cmd,
                "--sparse_materialization_target_visible_max_extra_bins",
                args.sparse_materialization_target_visible_max_extra_bins,
            )
        if bool(args.enable_train_only_target_impact_residual_basis):
            cmd.append("--enable_train_only_target_impact_residual_basis")
            _append_arg(cmd, "--target_impact_min_pixels", args.target_impact_min_pixels)
            _append_arg(cmd, "--target_impact_min_views", args.target_impact_min_views)
            _append_arg(
                cmd,
                "--target_impact_min_policy_samples",
                args.target_impact_min_policy_samples,
            )
            _append_arg(cmd, "--target_impact_max_extra_bins", args.target_impact_max_extra_bins)
            _append_arg(cmd, "--target_impact_max_views", args.target_impact_max_views)
            _append_arg(cmd, "--target_impact_carrier_fill_mode", args.target_impact_carrier_fill_mode)
            _append_arg(cmd, "--target_impact_carrier_fill_blend", args.target_impact_carrier_fill_blend)
            _append_arg(
                cmd,
                "--target_impact_carrier_fill_min_face_samples",
                args.target_impact_carrier_fill_min_face_samples,
            )
            _append_arg(cmd, "--target_impact_carrier_fill_min_norm", args.target_impact_carrier_fill_min_norm)
            _append_arg(
                cmd,
                "--target_impact_carrier_fill_synthetic_count",
                args.target_impact_carrier_fill_synthetic_count,
            )
            _append_arg(cmd, "--target_impact_multisample_fill_mode", args.target_impact_multisample_fill_mode)
            _append_arg(cmd, "--target_impact_multisample_fill_radius", args.target_impact_multisample_fill_radius)
            _append_arg(
                cmd,
                "--target_impact_multisample_fill_min_samples",
                args.target_impact_multisample_fill_min_samples,
            )
            _append_arg(
                cmd,
                "--target_impact_multisample_fill_max_samples_per_bin",
                args.target_impact_multisample_fill_max_samples_per_bin,
            )
            _append_arg(
                cmd,
                "--target_impact_multisample_fill_max_views",
                args.target_impact_multisample_fill_max_views,
            )
            _append_arg(cmd, "--target_impact_multisample_fill_blend", args.target_impact_multisample_fill_blend)
            _append_arg(
                cmd,
                "--target_impact_multisample_fill_kernel_sigma",
                args.target_impact_multisample_fill_kernel_sigma,
            )
            _append_arg(cmd, "--target_impact_multisample_fill_min_norm", args.target_impact_multisample_fill_min_norm)
            _append_arg(
                cmd,
                "--target_impact_multisample_fill_synthetic_count",
                args.target_impact_multisample_fill_synthetic_count,
            )
            _append_arg(cmd, "--target_impact_affine_fill_mode", args.target_impact_affine_fill_mode)
            _append_arg(
                cmd,
                "--target_impact_affine_fill_feature_mode",
                args.target_impact_affine_fill_feature_mode,
            )
            _append_arg(
                cmd,
                "--target_impact_affine_fill_min_samples",
                args.target_impact_affine_fill_min_samples,
            )
            _append_arg(
                cmd,
                "--target_impact_affine_fill_max_samples_per_face",
                args.target_impact_affine_fill_max_samples_per_face,
            )
            _append_arg(cmd, "--target_impact_affine_fill_max_views", args.target_impact_affine_fill_max_views)
            _append_arg(cmd, "--target_impact_affine_fill_blend", args.target_impact_affine_fill_blend)
            _append_arg(cmd, "--target_impact_affine_fill_ridge", args.target_impact_affine_fill_ridge)
            _append_arg(
                cmd,
                "--target_impact_affine_fill_max_condition",
                args.target_impact_affine_fill_max_condition,
            )
            _append_arg(cmd, "--target_impact_affine_fill_min_norm", args.target_impact_affine_fill_min_norm)
            _append_arg(
                cmd,
                "--target_impact_affine_fill_synthetic_count",
                args.target_impact_affine_fill_synthetic_count,
            )
        if bool(args.enable_sparse_materialization_target_connected_region_growth):
            cmd.append("--enable_sparse_materialization_target_connected_region_growth")
            _append_arg(
                cmd,
                "--sparse_materialization_target_connected_radius",
                args.sparse_materialization_target_connected_radius,
            )
            _append_arg(
                cmd,
                "--sparse_materialization_target_connected_min_pixels",
                args.sparse_materialization_target_connected_min_pixels,
            )
            _append_arg(
                cmd,
                "--sparse_materialization_target_connected_min_views",
                args.sparse_materialization_target_connected_min_views,
            )
            _append_arg(
                cmd,
                "--sparse_materialization_target_connected_min_policy_samples",
                args.sparse_materialization_target_connected_min_policy_samples,
            )
            _append_arg(
                cmd,
                "--sparse_materialization_target_connected_min_positive_view_fraction",
                args.sparse_materialization_target_connected_min_positive_view_fraction,
            )
            _append_arg(
                cmd,
                "--sparse_materialization_target_connected_max_negative_relative_gain",
                args.sparse_materialization_target_connected_max_negative_relative_gain,
            )
            _append_arg(
                cmd,
                "--sparse_materialization_target_connected_max_negative_min_view_gain",
                args.sparse_materialization_target_connected_max_negative_min_view_gain,
            )
            _append_arg(
                cmd,
                "--sparse_materialization_target_connected_max_extra_bins",
                args.sparse_materialization_target_connected_max_extra_bins,
            )
    if not bool(args.no_policy_val_bin_uncertainty_guard):
        cmd.extend(
            [
                "--enable_policy_val_bin_uncertainty_guard",
                "--bin_uncertainty_guard_min_bin_samples",
                str(args.bin_uncertainty_guard_min_bin_samples),
                "--bin_uncertainty_guard_min_positive_view_fraction",
                str(args.bin_uncertainty_guard_min_positive_view_fraction),
                "--bin_uncertainty_guard_empty_intersection_policy",
                str(args.bin_uncertainty_guard_empty_intersection_policy),
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
    if bool(args.enable_policy_val_view_consistency_confidence):
        cmd.append("--enable_policy_val_view_consistency_confidence")
    if bool(args.enable_policy_val_view_alpha_cap):
        cmd.append("--enable_policy_val_view_alpha_cap")
    if bool(args.enable_adaptive_low_support_teacher_basis):
        cmd.append("--enable_adaptive_low_support_teacher_basis")
        _append_arg(
            cmd,
            "--adaptive_teacher_basis_min_face_samples_floor",
            args.adaptive_teacher_basis_min_face_samples_floor,
        )
        _append_arg(
            cmd,
            "--adaptive_teacher_basis_support_quantile",
            args.adaptive_teacher_basis_support_quantile,
        )
        _append_arg(
            cmd,
            "--adaptive_teacher_basis_low_support_ridge_scale",
            args.adaptive_teacher_basis_low_support_ridge_scale,
        )
    for attr in (
        "view_confidence_min_relative_gain",
        "view_confidence_min_ssim_gain",
        "view_confidence_min_l1_gain",
        "view_confidence_min_lpips_gain",
        "view_confidence_kernel_sigma",
        "view_confidence_min_confidence",
        "view_alpha_cap_selection_mode",
        "view_alpha_cap_min_relative_gain",
        "view_alpha_cap_min_ssim_gain",
        "view_alpha_cap_min_l1_gain",
        "view_alpha_cap_min_lpips_gain",
        "view_alpha_cap_kernel_sigma",
        "view_alpha_cap_min_confidence",
        "view_alpha_cap_fallback_alpha",
        "view_alpha_cap_seed_stage",
    ):
        value = getattr(args, attr)
        if value is not None:
            _append_arg(cmd, f"--{attr}", value)
    if (
        bool(args.no_policy_val_bin_uncertainty_shrink)
        or bool(args.enable_policy_val_image_l1_bin_alpha_optimization)
        or bool(args.enable_policy_val_image_linear_residual_generator)
    ):
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
    parser.add_argument(
        "--eval_gt_evidence_dir",
        type=Path,
        default=None,
        help=(
            "Optional GT-bearing target evidence used only after apply for final evaluation. "
            "This lets --target_evidence_dir/--prestripped_target_evidence_dir stay GT-free for apply."
        ),
    )
    parser.add_argument("--region_carrier_json", type=Path, required=True)
    parser.add_argument("--teacher_render_dir", type=Path, default=None)
    parser.add_argument("--parent_render_dir", type=Path, default=None)
    parser.add_argument(
        "--distillation_profile",
        choices=("none", "teacher_to_reparented_parent"),
        default="none",
        help=(
            "Explicit teacher-distillation contract. teacher_to_reparented_parent treats "
            "--teacher_render_dir as the strong teacher, uses --parent_render_dir or "
            "--reparent_fit_parent_render_dir as the fit/train parent to bake residuals against, "
            "auto-reparents fit evidence, requires a split-matched --reparent_target_parent_render_dir "
            "for non-train target splits, forces strict no-target-GT apply, and rejects "
            "teacher==parent configurations that would create zero residuals."
        ),
    )
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
    parser.add_argument(
        "--reparent_copy_mode",
        choices=("copy", "hardlink", "symlink", "auto_link"),
        default="copy",
        help=(
            "Copy/link mode for evidence reparenting. Use auto_link in storage-constrained runs to avoid "
            "duplicating unchanged cache files before rewritten NPZ outputs are materialized."
        ),
    )
    parser.add_argument(
        "--teacher_cache_copy_mode",
        choices=("copy", "hardlink", "symlink", "auto_link"),
        default="copy",
        help=(
            "Copy/link mode for teacher surface evidence cache construction. Use auto_link in storage-constrained "
            "distillation runs; rewritten NPZ/text files are atomically replaced."
        ),
    )
    parser.add_argument(
        "--teacher_cache_rewrite_rgb_render_to_parent",
        action="store_true",
        help=(
            "When teacher cache uses --parent_render_dir, rewrite output rgb_render/residual fields to that parent. "
            "This fuses fit-evidence reparenting with teacher-cache construction."
        ),
    )
    parser.add_argument(
        "--skip_reparent_fit_evidence_for_teacher_cache",
        action="store_true",
        help=(
            "Do not create fit_evidence_reparented when a teacher cache will be built with "
            "--teacher_cache_rewrite_rgb_render_to_parent. This saves one full fit-evidence cache."
        ),
    )
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
    parser.add_argument(
        "--prestripped_target_evidence_dir",
        type=Path,
        default=None,
        help=(
            "Reuse an existing target evidence directory that has already been stripped of GT/residual keys. "
            "Requires --strict_no_target_gt_apply and skips the strip step; useful for quota-constrained "
            "repeat runs while keeping target apply GT-free."
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
    parser.add_argument(
        "--enable_adaptive_residual_activity_threshold",
        action="store_true",
        help=(
            "Estimate a train-only residual-active L1 threshold from fit evidence and "
            "use it as the adapter --min_l1. This prevents zero-teacher-residual regions "
            "from dominating persistent residual-surface fitting/certification."
        ),
    )
    parser.add_argument("--adaptive_residual_activity_quantile", type=float, default=0.90)
    parser.add_argument("--adaptive_residual_activity_floor", type=float, default=0.0)
    parser.add_argument("--adaptive_residual_activity_max_samples_per_view", type=int, default=200000)
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
    parser.add_argument("--view_cluster_expert_count", type=int, default=1)
    parser.add_argument("--view_cluster_feature_mode", choices=("none", "camera_center"), default="camera_center")
    parser.add_argument("--view_cluster_min_views", type=int, default=2)
    parser.add_argument("--view_cluster_min_bin_samples", type=int, default=4)
    parser.add_argument("--view_cluster_fallback_mode", choices=("global",), default="global")
    parser.add_argument(
        "--teacher_distilled_basis_mode",
        choices=(
            "none",
            "face_uv_normal_camera_ridge",
            "face_uv_patch_mixture_ridge",
            "low_rank_view_texture_k4",
            "low_rank_view_texture",
            "low_rank_view_texture_rich_k4",
            "low_rank_view_texture_rich",
        ),
        default="face_uv_patch_mixture_ridge",
    )
    parser.add_argument("--teacher_distilled_basis_blend", type=float, default=0.5)
    parser.add_argument("--teacher_distilled_basis_min_face_samples", type=int, default=1024)
    parser.add_argument("--teacher_distilled_basis_ridge", type=float, default=1.0e-2)
    parser.add_argument("--teacher_distilled_low_rank_texture_rank", type=int, default=4)
    parser.add_argument("--teacher_distilled_low_rank_texture_rank_candidates", default="")
    parser.add_argument("--enable_adaptive_low_support_teacher_basis", action="store_true")
    parser.add_argument("--adaptive_teacher_basis_min_face_samples_floor", type=int, default=128)
    parser.add_argument("--adaptive_teacher_basis_support_quantile", type=float, default=0.25)
    parser.add_argument("--adaptive_teacher_basis_low_support_ridge_scale", type=float, default=0.5)
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
    parser.add_argument("--no_policy_val_face_gain_guard", action="store_true")
    parser.add_argument("--enable_policy_val_view_consistency_confidence", action="store_true")
    parser.add_argument("--view_confidence_min_relative_gain", type=float, default=None)
    parser.add_argument("--view_confidence_min_ssim_gain", type=float, default=None)
    parser.add_argument("--view_confidence_min_l1_gain", type=float, default=None)
    parser.add_argument("--view_confidence_min_lpips_gain", type=float, default=None)
    parser.add_argument("--view_confidence_kernel_sigma", type=float, default=None)
    parser.add_argument("--view_confidence_min_confidence", type=float, default=None)
    parser.add_argument("--enable_policy_val_view_alpha_cap", action="store_true")
    parser.add_argument(
        "--view_alpha_cap_selection_mode",
        choices=("smallest_safe", "best_safe"),
        default=None,
    )
    parser.add_argument("--view_alpha_cap_min_relative_gain", type=float, default=None)
    parser.add_argument("--view_alpha_cap_min_ssim_gain", type=float, default=None)
    parser.add_argument("--view_alpha_cap_min_l1_gain", type=float, default=None)
    parser.add_argument("--view_alpha_cap_min_lpips_gain", type=float, default=None)
    parser.add_argument("--view_alpha_cap_kernel_sigma", type=float, default=None)
    parser.add_argument("--view_alpha_cap_min_confidence", type=float, default=None)
    parser.add_argument("--view_alpha_cap_fallback_alpha", type=float, default=None)
    parser.add_argument(
        "--view_alpha_cap_seed_stage",
        choices=("pre_guard", "post_view_confidence"),
        default=None,
    )
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
    parser.add_argument("--enable_policy_val_sparse_materialization_frontier", action="store_true")
    parser.add_argument("--sparse_materialization_seed_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--sparse_materialization_min_bin_samples", type=int, default=16)
    parser.add_argument("--sparse_materialization_min_bin_views", type=int, default=1)
    parser.add_argument("--sparse_materialization_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--sparse_materialization_min_view_relative_gain", type=float, default=-float("inf"))
    parser.add_argument("--sparse_materialization_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument("--sparse_materialization_frontier_min_positive_view_fraction", type=float, default=0.55)
    parser.add_argument("--sparse_materialization_frontier_min_risk_adjusted_gain", type=float, default=0.0)
    parser.add_argument("--sparse_materialization_frontier_min_sample_quantile", type=float, default=0.75)
    parser.add_argument("--sparse_materialization_max_mean_variance", type=float, default=-1.0)
    parser.add_argument("--sparse_materialization_min_mean_sign_consistency", type=float, default=0.0)
    parser.add_argument("--enable_sparse_materialization_target_visible_expansion", action="store_true")
    parser.add_argument("--sparse_materialization_target_visible_min_pixels", type=int, default=1)
    parser.add_argument("--sparse_materialization_target_visible_min_views", type=int, default=1)
    parser.add_argument("--sparse_materialization_target_visible_min_policy_samples", type=int, default=1)
    parser.add_argument(
        "--sparse_materialization_target_visible_min_positive_view_fraction",
        type=float,
        default=-1.0,
    )
    parser.add_argument("--sparse_materialization_target_visible_max_extra_bins", type=int, default=0)
    parser.add_argument("--enable_train_only_target_impact_residual_basis", action="store_true")
    parser.add_argument("--target_impact_min_pixels", type=int, default=1)
    parser.add_argument("--target_impact_min_views", type=int, default=1)
    parser.add_argument("--target_impact_min_policy_samples", type=int, default=0)
    parser.add_argument("--target_impact_max_extra_bins", type=int, default=0)
    parser.add_argument("--target_impact_max_views", type=int, default=0)
    parser.add_argument(
        "--target_impact_carrier_fill_mode",
        choices=("off", "no_policy_rows", "all_added"),
        default="off",
    )
    parser.add_argument("--target_impact_carrier_fill_blend", type=float, default=0.5)
    parser.add_argument("--target_impact_carrier_fill_min_face_samples", type=int, default=128)
    parser.add_argument("--target_impact_carrier_fill_min_norm", type=float, default=1.0e-4)
    parser.add_argument("--target_impact_carrier_fill_synthetic_count", type=int, default=1)
    parser.add_argument(
        "--target_impact_multisample_fill_mode",
        choices=("off", "no_policy_rows", "all_added"),
        default="off",
    )
    parser.add_argument("--target_impact_multisample_fill_radius", type=int, default=1)
    parser.add_argument("--target_impact_multisample_fill_min_samples", type=int, default=4)
    parser.add_argument("--target_impact_multisample_fill_max_samples_per_bin", type=int, default=128)
    parser.add_argument("--target_impact_multisample_fill_max_views", type=int, default=0)
    parser.add_argument("--target_impact_multisample_fill_blend", type=float, default=1.0)
    parser.add_argument("--target_impact_multisample_fill_kernel_sigma", type=float, default=1.0)
    parser.add_argument("--target_impact_multisample_fill_min_norm", type=float, default=1.0e-4)
    parser.add_argument("--target_impact_multisample_fill_synthetic_count", type=int, default=2)
    parser.add_argument(
        "--target_impact_affine_fill_mode",
        choices=("off", "no_policy_rows", "all_added"),
        default="off",
    )
    parser.add_argument(
        "--target_impact_affine_fill_feature_mode",
        choices=("face_uv_normal_camera_ridge", "face_uv_patch_mixture_ridge"),
        default="face_uv_patch_mixture_ridge",
    )
    parser.add_argument("--target_impact_affine_fill_min_samples", type=int, default=64)
    parser.add_argument("--target_impact_affine_fill_max_samples_per_face", type=int, default=4096)
    parser.add_argument("--target_impact_affine_fill_max_views", type=int, default=0)
    parser.add_argument("--target_impact_affine_fill_blend", type=float, default=1.0)
    parser.add_argument("--target_impact_affine_fill_ridge", type=float, default=1.0e-2)
    parser.add_argument("--target_impact_affine_fill_max_condition", type=float, default=1.0e7)
    parser.add_argument("--target_impact_affine_fill_min_norm", type=float, default=1.0e-4)
    parser.add_argument("--target_impact_affine_fill_synthetic_count", type=int, default=4)
    parser.add_argument("--enable_sparse_materialization_target_connected_region_growth", action="store_true")
    parser.add_argument("--sparse_materialization_target_connected_radius", type=int, default=1)
    parser.add_argument("--sparse_materialization_target_connected_min_pixels", type=int, default=1)
    parser.add_argument("--sparse_materialization_target_connected_min_views", type=int, default=1)
    parser.add_argument("--sparse_materialization_target_connected_min_policy_samples", type=int, default=1)
    parser.add_argument(
        "--sparse_materialization_target_connected_min_positive_view_fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--sparse_materialization_target_connected_max_negative_relative_gain",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--sparse_materialization_target_connected_max_negative_min_view_gain",
        type=float,
        default=0.05,
    )
    parser.add_argument("--sparse_materialization_target_connected_max_extra_bins", type=int, default=0)
    parser.add_argument("--no_policy_val_bin_uncertainty_guard", action="store_true")
    parser.add_argument("--no_target_support_candidate_selection", action="store_true")
    parser.add_argument("--no_policy_candidate_dominance_pruning", action="store_true")
    parser.add_argument("--bin_uncertainty_guard_min_bin_samples", type=int, default=16)
    parser.add_argument("--bin_uncertainty_guard_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument(
        "--bin_uncertainty_guard_empty_intersection_policy",
        choices=("reject", "sparse_if_post_accepted"),
        default="sparse_if_post_accepted",
        help=(
            "How vNext handles an empty intersection between sparse-certified bins and "
            "the later raw bin-uncertainty guard. The runner defaults to preserving a "
            "sparse profile only after its own post-gate accepted it; standalone adapter "
            "calls can still use reject for the strict control."
        ),
    )
    parser.add_argument("--no_policy_val_bin_uncertainty_shrink", action="store_true")
    parser.add_argument(
        "--bin_uncertainty_shrink_policy_mode",
        choices=("sparse_positive", "keep_with_downweight", "positive_consensus"),
        default="keep_with_downweight",
    )
    parser.add_argument("--bin_uncertainty_shrink_min_bin_samples", type=int, default=16)
    parser.add_argument("--bin_uncertainty_shrink_min_bin_views", type=int, default=1)
    parser.add_argument("--bin_uncertainty_shrink_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--bin_uncertainty_shrink_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument("--bin_uncertainty_shrink_fallback_shrink", type=float, default=1.0)
    parser.add_argument("--enable_policy_val_image_l1_bin_certificate", action="store_true")
    parser.add_argument(
        "--image_l1_bin_certificate_mode",
        choices=("and", "or", "replace"),
        default="and",
    )
    parser.add_argument("--image_l1_bin_certificate_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--image_l1_bin_certificate_min_positive_view_fraction", type=float, default=0.55)
    parser.add_argument("--image_l1_bin_certificate_gain_tau", type=float, default=0.01)
    parser.add_argument("--image_l1_bin_certificate_pool_radius", type=int, default=0)
    parser.add_argument("--enable_policy_val_image_l1_region_expansion", action="store_true")
    parser.add_argument("--image_l1_region_expansion_radius", type=int, default=1)
    parser.add_argument("--image_l1_region_expansion_max_bins_per_seed", type=int, default=8)
    parser.add_argument("--image_l1_region_expansion_min_neighbor_samples", type=int, default=1)
    parser.add_argument("--image_l1_region_expansion_min_neighbor_views", type=int, default=1)
    parser.add_argument("--image_l1_region_expansion_max_negative_relative_gain", type=float, default=0.02)
    parser.add_argument("--image_l1_region_expansion_max_negative_image_l1_gain", type=float, default=0.02)
    parser.add_argument("--image_l1_region_expansion_shrink_decay", type=float, default=0.5)
    parser.add_argument("--enable_policy_val_image_l1_bin_alpha_optimization", action="store_true")
    parser.add_argument("--image_l1_bin_alpha_grid", default="0,0.0625,0.125,0.25,0.5,0.75,1.0")
    parser.add_argument("--image_l1_bin_alpha_max_alpha", type=float, default=1.0)
    parser.add_argument("--image_l1_bin_alpha_min_bin_samples", type=int, default=8)
    parser.add_argument("--image_l1_bin_alpha_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--image_l1_bin_alpha_min_positive_view_fraction", type=float, default=0.55)
    parser.add_argument("--image_l1_bin_alpha_count_tau", type=float, default=64.0)
    parser.add_argument(
        "--image_l1_bin_alpha_fallback_mode",
        choices=("zero", "global_best"),
        default="zero",
    )
    parser.add_argument("--image_l1_bin_alpha_max_profile_bins", type=int, default=8192)
    parser.add_argument("--enable_policy_val_image_linear_residual_generator", action="store_true")
    parser.add_argument(
        "--image_linear_generator_feature_mode",
        choices=("base", "base_rgb", "base_rgb_bary_view"),
        default="base_rgb",
    )
    parser.add_argument("--image_linear_generator_ridge", type=float, default=1.0e-2)
    parser.add_argument("--image_linear_generator_train_max_samples_per_view", type=int, default=100000)
    parser.add_argument("--image_linear_generator_max_train_samples", type=int, default=1000000)
    parser.add_argument("--image_linear_generator_output_cap", type=float, default=0.12)
    parser.add_argument(
        "--image_linear_generator_loss_mode",
        choices=("mse", "huber_irls", "l1_irls"),
        default="mse",
    )
    parser.add_argument("--image_linear_generator_irls_iterations", type=int, default=4)
    parser.add_argument("--image_linear_generator_huber_delta", type=float, default=0.02)
    parser.add_argument(
        "--image_linear_generator_training_sample_policy",
        choices=("all", "base_l1_descent", "view_balanced", "view_balanced_base_l1_descent"),
        default="all",
    )
    parser.add_argument("--image_linear_generator_min_descent_margin", type=float, default=0.0)
    parser.add_argument("--image_linear_generator_min_training_samples", type=int, default=512)
    parser.add_argument(
        "--image_linear_generator_alpha_grid",
        default="0,0.03125,0.0625,0.125,0.25,0.5,0.75,1.0",
    )
    parser.add_argument(
        "--image_linear_generator_expert_mode",
        choices=("none", "view_cluster"),
        default="none",
    )
    parser.add_argument("--image_linear_generator_expert_min_training_samples", type=int, default=2048)
    parser.add_argument("--image_linear_generator_expert_shrink_tau", type=float, default=8192.0)
    parser.add_argument(
        "--image_linear_generator_face_reliability_mode",
        choices=("none", "global", "view_cluster"),
        default="none",
    )
    parser.add_argument("--image_linear_generator_face_reliability_min_face_samples", type=int, default=256)
    parser.add_argument("--image_linear_generator_face_reliability_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--image_linear_generator_face_reliability_min_positive_view_fraction", type=float, default=0.5)
    parser.add_argument("--image_linear_generator_face_reliability_fallback_multiplier", type=float, default=0.0)
    parser.add_argument("--image_linear_generator_allow_unvalidated_base_pixels", action="store_true")
    parser.add_argument(
        "--enable_view_cluster_local_shrink",
        "--enable_policy_val_cluster_local_shrink",
        dest="enable_view_cluster_local_shrink",
        action="store_true",
        help=(
            "When --view_cluster_expert_count > 1, certify bin uncertainty shrink "
            "inside each GT-free camera-center cluster instead of globally."
        ),
    )
    parser.add_argument(
        "--view_cluster_local_shrink_global_fallback",
        action="store_true",
        help="Expose cluster-local selected bins through the legacy global shrink table as a fallback.",
    )
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
    parser.add_argument(
        "--no_policy_val_prior_bin_gain_hybrid",
        action="store_true",
        help=(
            "Skip the expensive prior-bin-gain/source-mixture hybrid search branch. "
            "This keeps the primary certified atlas candidate and target-support ranking, "
            "but avoids extra policy-val passes for fast ablations."
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
    _apply_distillation_profile(args)
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
    if int(args.teacher_distilled_basis_min_face_samples) <= 0:
        parser.error("--teacher_distilled_basis_min_face_samples must be > 0")
    if int(args.adaptive_teacher_basis_min_face_samples_floor) <= 0:
        parser.error("--adaptive_teacher_basis_min_face_samples_floor must be > 0")
    if not 0.0 <= float(args.adaptive_teacher_basis_support_quantile) <= 1.0:
        parser.error("--adaptive_teacher_basis_support_quantile must be in [0, 1]")
    if float(args.adaptive_teacher_basis_low_support_ridge_scale) < 0.0:
        parser.error("--adaptive_teacher_basis_low_support_ridge_scale must be >= 0")
    if int(args.view_cluster_expert_count) < 1:
        parser.error("--view_cluster_expert_count must be >= 1")
    if int(args.view_cluster_min_views) < 1:
        parser.error("--view_cluster_min_views must be >= 1")
    if int(args.view_cluster_min_bin_samples) < 1:
        parser.error("--view_cluster_min_bin_samples must be >= 1")
    if bool(args.enable_view_cluster_local_shrink) and int(args.view_cluster_expert_count) <= 1:
        parser.error("--enable_view_cluster_local_shrink requires --view_cluster_expert_count > 1")
    if bool(args.enable_policy_val_image_l1_bin_certificate) and bool(args.no_policy_val_bin_uncertainty_shrink):
        parser.error("--enable_policy_val_image_l1_bin_certificate requires bin uncertainty shrink to stay enabled")
    if float(args.image_l1_bin_certificate_min_relative_gain) < 0.0:
        parser.error("--image_l1_bin_certificate_min_relative_gain must be >= 0")
    if not 0.0 <= float(args.image_l1_bin_certificate_min_positive_view_fraction) <= 1.0:
        parser.error("--image_l1_bin_certificate_min_positive_view_fraction must be in [0, 1]")
    if float(args.image_l1_bin_certificate_gain_tau) < 0.0:
        parser.error("--image_l1_bin_certificate_gain_tau must be >= 0")
    if int(args.image_l1_bin_certificate_pool_radius) < 0:
        parser.error("--image_l1_bin_certificate_pool_radius must be >= 0")
    if bool(args.enable_policy_val_image_l1_region_expansion) and not bool(
        args.enable_policy_val_image_l1_bin_certificate
    ):
        parser.error("--enable_policy_val_image_l1_region_expansion requires --enable_policy_val_image_l1_bin_certificate")
    if int(args.image_l1_region_expansion_radius) < 0:
        parser.error("--image_l1_region_expansion_radius must be >= 0")
    if int(args.image_l1_region_expansion_max_bins_per_seed) < 0:
        parser.error("--image_l1_region_expansion_max_bins_per_seed must be >= 0")
    if int(args.image_l1_region_expansion_min_neighbor_samples) < 1:
        parser.error("--image_l1_region_expansion_min_neighbor_samples must be >= 1")
    if int(args.image_l1_region_expansion_min_neighbor_views) < 1:
        parser.error("--image_l1_region_expansion_min_neighbor_views must be >= 1")
    if float(args.image_l1_region_expansion_max_negative_relative_gain) < 0.0:
        parser.error("--image_l1_region_expansion_max_negative_relative_gain must be >= 0")
    if float(args.image_l1_region_expansion_max_negative_image_l1_gain) < 0.0:
        parser.error("--image_l1_region_expansion_max_negative_image_l1_gain must be >= 0")
    if not 0.0 <= float(args.image_l1_region_expansion_shrink_decay) <= 1.0:
        parser.error("--image_l1_region_expansion_shrink_decay must be in [0, 1]")
    if bool(args.enable_policy_val_image_l1_bin_alpha_optimization) and bool(
        args.enable_policy_val_image_l1_bin_certificate
    ):
        parser.error(
            "--enable_policy_val_image_l1_bin_alpha_optimization is a replacement local-alpha mode; "
            "do not combine it with --enable_policy_val_image_l1_bin_certificate"
        )
    if bool(args.enable_policy_val_image_l1_bin_alpha_optimization) and bool(
        args.enable_policy_val_structure_aware_shrink
    ):
        parser.error(
            "--enable_policy_val_image_l1_bin_alpha_optimization is a replacement local-alpha mode; "
            "do not combine it with --enable_policy_val_structure_aware_shrink"
        )
    if bool(args.enable_policy_val_image_linear_residual_generator) and bool(
        args.enable_policy_val_image_l1_bin_alpha_optimization
    ):
        parser.error(
            "--enable_policy_val_image_linear_residual_generator is a replacement generator mode; "
            "do not combine it with --enable_policy_val_image_l1_bin_alpha_optimization"
        )
    if bool(args.enable_policy_val_image_linear_residual_generator) and bool(
        args.enable_policy_val_image_l1_bin_certificate
    ):
        parser.error(
            "--enable_policy_val_image_linear_residual_generator is a replacement generator mode; "
            "do not combine it with --enable_policy_val_image_l1_bin_certificate"
        )
    if (
        bool(args.enable_policy_val_image_linear_residual_generator)
        and str(args.image_linear_generator_face_reliability_mode) == "view_cluster"
        and str(args.image_linear_generator_expert_mode) != "view_cluster"
    ):
        parser.error(
            "--image_linear_generator_face_reliability_mode view_cluster requires "
            "--image_linear_generator_expert_mode view_cluster"
        )
    if bool(args.enable_policy_val_image_linear_residual_generator) and bool(
        args.enable_policy_val_structure_aware_shrink
    ):
        parser.error(
            "--enable_policy_val_image_linear_residual_generator is a replacement generator mode; "
            "do not combine it with --enable_policy_val_structure_aware_shrink"
        )
    if float(args.image_l1_bin_alpha_max_alpha) < 0.0:
        parser.error("--image_l1_bin_alpha_max_alpha must be >= 0")
    if int(args.image_l1_bin_alpha_min_bin_samples) < 1:
        parser.error("--image_l1_bin_alpha_min_bin_samples must be >= 1")
    if float(args.image_l1_bin_alpha_min_relative_gain) < 0.0:
        parser.error("--image_l1_bin_alpha_min_relative_gain must be >= 0")
    if not 0.0 <= float(args.image_l1_bin_alpha_min_positive_view_fraction) <= 1.0:
        parser.error("--image_l1_bin_alpha_min_positive_view_fraction must be in [0, 1]")
    if float(args.image_l1_bin_alpha_count_tau) < 0.0:
        parser.error("--image_l1_bin_alpha_count_tau must be >= 0")
    if int(args.image_l1_bin_alpha_max_profile_bins) < 0:
        parser.error("--image_l1_bin_alpha_max_profile_bins must be >= 0")
    if float(args.image_linear_generator_ridge) < 0.0:
        parser.error("--image_linear_generator_ridge must be >= 0")
    if int(args.image_linear_generator_train_max_samples_per_view) < 1:
        parser.error("--image_linear_generator_train_max_samples_per_view must be >= 1")
    if int(args.image_linear_generator_max_train_samples) < 1:
        parser.error("--image_linear_generator_max_train_samples must be >= 1")
    if float(args.image_linear_generator_output_cap) < 0.0:
        parser.error("--image_linear_generator_output_cap must be >= 0")
    if int(args.image_linear_generator_irls_iterations) < 1:
        parser.error("--image_linear_generator_irls_iterations must be >= 1")
    if float(args.image_linear_generator_huber_delta) <= 0.0:
        parser.error("--image_linear_generator_huber_delta must be > 0")
    if float(args.image_linear_generator_min_descent_margin) < 0.0:
        parser.error("--image_linear_generator_min_descent_margin must be >= 0")
    if int(args.image_linear_generator_min_training_samples) < 1:
        parser.error("--image_linear_generator_min_training_samples must be >= 1")
    if int(args.image_linear_generator_expert_min_training_samples) < 1:
        parser.error("--image_linear_generator_expert_min_training_samples must be >= 1")
    if float(args.image_linear_generator_expert_shrink_tau) < 0.0:
        parser.error("--image_linear_generator_expert_shrink_tau must be >= 0")
    if int(args.image_linear_generator_face_reliability_min_face_samples) < 1:
        parser.error("--image_linear_generator_face_reliability_min_face_samples must be >= 1")
    if not 0.0 <= float(args.image_linear_generator_face_reliability_min_positive_view_fraction) <= 1.0:
        parser.error("--image_linear_generator_face_reliability_min_positive_view_fraction must be in [0, 1]")
    if not 0.0 <= float(args.image_linear_generator_face_reliability_fallback_multiplier) <= 1.0:
        parser.error("--image_linear_generator_face_reliability_fallback_multiplier must be in [0, 1]")
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
    if int(args.sparse_materialization_min_bin_views) < 1:
        parser.error("--sparse_materialization_min_bin_views must be >= 1")
    if float(args.sparse_materialization_min_relative_gain) < 0.0:
        parser.error("--sparse_materialization_min_relative_gain must be >= 0")
    if int(args.bin_uncertainty_shrink_min_bin_views) < 1:
        parser.error("--bin_uncertainty_shrink_min_bin_views must be >= 1")
    if not 0.0 <= float(args.sparse_materialization_min_positive_view_fraction) <= 1.0:
        parser.error("--sparse_materialization_min_positive_view_fraction must be in [0, 1]")
    if not 0.0 <= float(args.sparse_materialization_frontier_min_positive_view_fraction) <= 1.0:
        parser.error("--sparse_materialization_frontier_min_positive_view_fraction must be in [0, 1]")
    if not 0.0 <= float(args.sparse_materialization_frontier_min_sample_quantile) <= 1.0:
        parser.error("--sparse_materialization_frontier_min_sample_quantile must be in [0, 1]")
    if float(args.sparse_materialization_min_mean_sign_consistency) < 0.0:
        parser.error("--sparse_materialization_min_mean_sign_consistency must be >= 0")
    if int(args.sparse_materialization_target_visible_min_pixels) < 1:
        parser.error("--sparse_materialization_target_visible_min_pixels must be >= 1")
    if int(args.sparse_materialization_target_visible_min_views) < 1:
        parser.error("--sparse_materialization_target_visible_min_views must be >= 1")
    if int(args.sparse_materialization_target_visible_min_policy_samples) < 1:
        parser.error("--sparse_materialization_target_visible_min_policy_samples must be >= 1")
    if int(args.sparse_materialization_target_visible_max_extra_bins) < 0:
        parser.error("--sparse_materialization_target_visible_max_extra_bins must be >= 0")
    if (
        float(args.sparse_materialization_target_visible_min_positive_view_fraction) >= 0.0
        and not 0.0 <= float(args.sparse_materialization_target_visible_min_positive_view_fraction) <= 1.0
    ):
        parser.error("--sparse_materialization_target_visible_min_positive_view_fraction must be <0 or in [0, 1]")
    if int(args.target_impact_min_pixels) < 1:
        parser.error("--target_impact_min_pixels must be >= 1")
    if int(args.target_impact_min_views) < 1:
        parser.error("--target_impact_min_views must be >= 1")
    if int(args.target_impact_min_policy_samples) < 0:
        parser.error("--target_impact_min_policy_samples must be >= 0")
    if int(args.target_impact_max_extra_bins) < 0:
        parser.error("--target_impact_max_extra_bins must be >= 0")
    if int(args.target_impact_max_views) < 0:
        parser.error("--target_impact_max_views must be >= 0")
    if not 0.0 <= float(args.target_impact_carrier_fill_blend) <= 1.0:
        parser.error("--target_impact_carrier_fill_blend must be in [0, 1]")
    if int(args.target_impact_carrier_fill_min_face_samples) < 0:
        parser.error("--target_impact_carrier_fill_min_face_samples must be >= 0")
    if float(args.target_impact_carrier_fill_min_norm) < 0.0:
        parser.error("--target_impact_carrier_fill_min_norm must be >= 0")
    if int(args.target_impact_carrier_fill_synthetic_count) < 0:
        parser.error("--target_impact_carrier_fill_synthetic_count must be >= 0")
    if (
        str(args.target_impact_carrier_fill_mode) != "off"
        and not bool(args.enable_train_only_target_impact_residual_basis)
    ):
        parser.error("--target_impact_carrier_fill_mode requires --enable_train_only_target_impact_residual_basis")
    if int(args.target_impact_multisample_fill_radius) < 0:
        parser.error("--target_impact_multisample_fill_radius must be >= 0")
    if int(args.target_impact_multisample_fill_min_samples) < 1:
        parser.error("--target_impact_multisample_fill_min_samples must be >= 1")
    if int(args.target_impact_multisample_fill_max_samples_per_bin) < 0:
        parser.error("--target_impact_multisample_fill_max_samples_per_bin must be >= 0")
    if int(args.target_impact_multisample_fill_max_views) < 0:
        parser.error("--target_impact_multisample_fill_max_views must be >= 0")
    if not 0.0 <= float(args.target_impact_multisample_fill_blend) <= 1.0:
        parser.error("--target_impact_multisample_fill_blend must be in [0, 1]")
    if float(args.target_impact_multisample_fill_kernel_sigma) <= 0.0:
        parser.error("--target_impact_multisample_fill_kernel_sigma must be > 0")
    if float(args.target_impact_multisample_fill_min_norm) < 0.0:
        parser.error("--target_impact_multisample_fill_min_norm must be >= 0")
    if int(args.target_impact_multisample_fill_synthetic_count) < 0:
        parser.error("--target_impact_multisample_fill_synthetic_count must be >= 0")
    if (
        str(args.target_impact_multisample_fill_mode) != "off"
        and not bool(args.enable_train_only_target_impact_residual_basis)
    ):
        parser.error("--target_impact_multisample_fill_mode requires --enable_train_only_target_impact_residual_basis")
    if int(args.target_impact_affine_fill_min_samples) < 1:
        parser.error("--target_impact_affine_fill_min_samples must be >= 1")
    if int(args.target_impact_affine_fill_max_samples_per_face) < 0:
        parser.error("--target_impact_affine_fill_max_samples_per_face must be >= 0")
    if int(args.target_impact_affine_fill_max_views) < 0:
        parser.error("--target_impact_affine_fill_max_views must be >= 0")
    if not 0.0 <= float(args.target_impact_affine_fill_blend) <= 1.0:
        parser.error("--target_impact_affine_fill_blend must be in [0, 1]")
    if float(args.target_impact_affine_fill_ridge) <= 0.0:
        parser.error("--target_impact_affine_fill_ridge must be > 0")
    if float(args.target_impact_affine_fill_max_condition) <= 1.0:
        parser.error("--target_impact_affine_fill_max_condition must be > 1")
    if float(args.target_impact_affine_fill_min_norm) < 0.0:
        parser.error("--target_impact_affine_fill_min_norm must be >= 0")
    if int(args.target_impact_affine_fill_synthetic_count) < 0:
        parser.error("--target_impact_affine_fill_synthetic_count must be >= 0")
    if (
        str(args.target_impact_affine_fill_mode) != "off"
        and not bool(args.enable_train_only_target_impact_residual_basis)
    ):
        parser.error("--target_impact_affine_fill_mode requires --enable_train_only_target_impact_residual_basis")
    if int(args.sparse_materialization_target_connected_radius) < 0:
        parser.error("--sparse_materialization_target_connected_radius must be >= 0")
    if int(args.sparse_materialization_target_connected_min_pixels) < 1:
        parser.error("--sparse_materialization_target_connected_min_pixels must be >= 1")
    if int(args.sparse_materialization_target_connected_min_views) < 1:
        parser.error("--sparse_materialization_target_connected_min_views must be >= 1")
    if int(args.sparse_materialization_target_connected_min_policy_samples) < 1:
        parser.error("--sparse_materialization_target_connected_min_policy_samples must be >= 1")
    if not 0.0 <= float(args.sparse_materialization_target_connected_min_positive_view_fraction) <= 1.0:
        parser.error("--sparse_materialization_target_connected_min_positive_view_fraction must be in [0, 1]")
    if float(args.sparse_materialization_target_connected_max_negative_relative_gain) < 0.0:
        parser.error("--sparse_materialization_target_connected_max_negative_relative_gain must be >= 0")
    if float(args.sparse_materialization_target_connected_max_negative_min_view_gain) < 0.0:
        parser.error("--sparse_materialization_target_connected_max_negative_min_view_gain must be >= 0")
    if int(args.sparse_materialization_target_connected_max_extra_bins) < 0:
        parser.error("--sparse_materialization_target_connected_max_extra_bins must be >= 0")
    if not bool(args.enable_policy_val_image_lpips_gate) and (
        float(args.min_policy_val_effective_lpips_gain) > -1.0
        or float(args.min_policy_val_effective_lpips_cvar20_gain) > -1.0
    ):
        parser.error("LPIPS effective thresholds require --enable_policy_val_image_lpips_gate")
    if args.prestripped_target_evidence_dir is not None and not bool(args.strict_no_target_gt_apply):
        parser.error("--prestripped_target_evidence_dir requires --strict_no_target_gt_apply")
    target_footprint_apply_enabled = (
        bool(args.enable_train_only_target_impact_residual_basis)
        or bool(args.enable_sparse_materialization_target_visible_expansion)
        or bool(args.enable_sparse_materialization_target_connected_region_growth)
    )
    if target_footprint_apply_enabled and not bool(args.strict_no_target_gt_apply):
        parser.error(
            "target-footprint apply paths require --strict_no_target_gt_apply "
            "to keep target/test GT out of the adapter"
        )
    if args.prestripped_target_evidence_dir is not None and not Path(args.prestripped_target_evidence_dir).exists():
        parser.error("--prestripped_target_evidence_dir does not exist")
    if args.eval_gt_evidence_dir is not None and not Path(args.eval_gt_evidence_dir).exists():
        parser.error("--eval_gt_evidence_dir does not exist")
    if args.view_confidence_kernel_sigma is not None and float(args.view_confidence_kernel_sigma) <= 0.0:
        parser.error("--view_confidence_kernel_sigma must be > 0")
    if args.view_confidence_min_confidence is not None and not 0.0 <= float(args.view_confidence_min_confidence) <= 1.0:
        parser.error("--view_confidence_min_confidence must be in [0, 1]")
    if args.view_alpha_cap_kernel_sigma is not None and float(args.view_alpha_cap_kernel_sigma) <= 0.0:
        parser.error("--view_alpha_cap_kernel_sigma must be > 0")
    if args.view_alpha_cap_min_confidence is not None and not 0.0 <= float(args.view_alpha_cap_min_confidence) <= 1.0:
        parser.error("--view_alpha_cap_min_confidence must be in [0, 1]")
    if args.view_alpha_cap_fallback_alpha is not None and float(args.view_alpha_cap_fallback_alpha) < 0.0:
        parser.error("--view_alpha_cap_fallback_alpha must be >= 0")
    if float(args.teacher_distilled_basis_ridge) < 0.0:
        parser.error("--teacher_distilled_basis_ridge must be >= 0")
    if int(args.teacher_distilled_low_rank_texture_rank) <= 0:
        parser.error("--teacher_distilled_low_rank_texture_rank must be > 0")
    for item in str(args.teacher_distilled_low_rank_texture_rank_candidates or "").split(","):
        item = item.strip()
        if item and int(item) <= 0:
            parser.error("--teacher_distilled_low_rank_texture_rank_candidates values must be > 0")
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
    if not 0.0 <= float(args.adaptive_residual_activity_quantile) <= 1.0:
        parser.error("--adaptive_residual_activity_quantile must be in [0, 1]")
    if float(args.adaptive_residual_activity_floor) < 0.0:
        parser.error("--adaptive_residual_activity_floor must be >= 0")
    if int(args.adaptive_residual_activity_max_samples_per_view) < 1:
        parser.error("--adaptive_residual_activity_max_samples_per_view must be >= 1")
    if bool(args.skip_reparent_fit_evidence_for_teacher_cache):
        if bool(args.skip_teacher_cache):
            parser.error("--skip_reparent_fit_evidence_for_teacher_cache requires teacher cache construction")
        if args.reparent_fit_parent_render_dir is None:
            parser.error("--skip_reparent_fit_evidence_for_teacher_cache requires --reparent_fit_parent_render_dir")
        if args.parent_render_dir is None:
            parser.error("--skip_reparent_fit_evidence_for_teacher_cache requires --parent_render_dir")
        if not bool(args.teacher_cache_rewrite_rgb_render_to_parent):
            parser.error(
                "--skip_reparent_fit_evidence_for_teacher_cache requires "
                "--teacher_cache_rewrite_rgb_render_to_parent"
            )
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
    target_apply_no_gt_audit_path = reports_dir / (
        f"{args.scene}_{args.method_name}_{args.target_split}_target_apply_no_gt_verify.json"
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["WANDB_MODE"] = str(args.wandb_mode)
    if args.gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not args.skip_teacher_cache and args.teacher_render_dir is None:
        raise SystemExit("--teacher_render_dir is required unless --skip_teacher_cache is set")

    skip_fit_reparent_for_teacher_cache = bool(args.skip_reparent_fit_evidence_for_teacher_cache) and not bool(
        args.skip_teacher_cache
    )
    args._fit_reparent_execution = {
        "enabled": bool(args.reparent_fit_parent_render_dir is not None),
        "skipped_for_teacher_cache": bool(skip_fit_reparent_for_teacher_cache),
        "reason": "teacher_cache_rewrites_rgb_render_to_parent"
        if skip_fit_reparent_for_teacher_cache
        else "standard_reparent_path",
    }
    args._effective_fit_evidence_dir = (
        Path(args.fit_evidence_dir)
        if skip_fit_reparent_for_teacher_cache
        else (
            reparented_fit_evidence_dir
            if args.reparent_fit_parent_render_dir is not None
            else Path(args.fit_evidence_dir)
        )
    )
    args._effective_target_evidence_dir = (
        reparented_target_evidence_dir
        if args.reparent_target_parent_render_dir is not None
        else Path(args.target_evidence_dir)
    )
    args._effective_eval_gt_evidence_dir = (
        Path(args.eval_gt_evidence_dir)
        if args.eval_gt_evidence_dir is not None
        else Path(args._effective_target_evidence_dir)
    )

    texture_fit_evidence_dir = Path(args._effective_fit_evidence_dir) if args.skip_teacher_cache else teacher_cache_dir
    if bool(args.strict_no_target_gt_apply) and args.prestripped_target_evidence_dir is not None:
        adapter_target_evidence_dir = Path(args.prestripped_target_evidence_dir)
    else:
        adapter_target_evidence_dir = (
            stripped_target_evidence_dir
            if bool(args.strict_no_target_gt_apply)
            else Path(args._effective_target_evidence_dir)
        )
    args._target_apply_forbidden_key_preflight = {
        "enabled": bool(args.strict_no_target_gt_apply),
        "mode": "strict_no_target_gt_apply_forbidden_key_preflight",
        "reason": "strict_no_target_gt_apply_disabled"
        if not bool(args.strict_no_target_gt_apply)
        else "deferred_to_strip_target_evidence_step",
        "passed": True,
    }
    if bool(args.strict_no_target_gt_apply) and args.prestripped_target_evidence_dir is not None:
        forbidden_key_audit = _verify_target_apply_forbidden_keys(adapter_target_evidence_dir)
        args._target_apply_forbidden_key_preflight = forbidden_key_audit
        if not bool(forbidden_key_audit.get("passed", False)):
            reason = str(forbidden_key_audit.get("reason", "forbidden_target_apply_keys_present"))
            bad_preview = forbidden_key_audit.get("bad_views", [])[:3]
            parser.error(
                "strict target apply preflight failed for "
                f"{adapter_target_evidence_dir}: {reason}; bad_views={bad_preview}"
            )
    args._effective_min_l1 = float(args.min_l1)
    args._adaptive_residual_activity_threshold = {
        "enabled": False,
        "mode": "train_only_adaptive_residual_activity_threshold",
        "selected_min_l1": float(args._effective_min_l1),
        "reason": "not_requested",
    }
    if bool(args.enable_adaptive_residual_activity_threshold):
        if texture_fit_evidence_dir.exists():
            effective_min_l1, activity_summary = _compute_adaptive_residual_activity_threshold(
                texture_fit_evidence_dir,
                residual_l1_key="teacher_residual_l1",
                min_alpha=float(args.min_alpha),
                base_min_l1=float(args.min_l1),
                quantile=float(args.adaptive_residual_activity_quantile),
                floor=float(args.adaptive_residual_activity_floor),
                max_samples_per_view=int(args.adaptive_residual_activity_max_samples_per_view),
            )
            args._effective_min_l1 = float(effective_min_l1)
            args._adaptive_residual_activity_threshold = dict(activity_summary)
        else:
            args._adaptive_residual_activity_threshold = {
                "enabled": False,
                "mode": "train_only_adaptive_residual_activity_threshold",
                "selected_min_l1": float(args._effective_min_l1),
                "reason": "fit_evidence_dir_not_available_before_texture_step",
                "evidence_dir": str(texture_fit_evidence_dir),
            }

    commands: list[dict[str, Any]] = []
    if args.reparent_fit_parent_render_dir is not None and not skip_fit_reparent_for_teacher_cache and (
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
    if (
        bool(args.strict_no_target_gt_apply)
        and args.prestripped_target_evidence_dir is None
        and not args.skip_texture
    ):
        commands.append(
            command_record(
                "strip_target_evidence_no_gt",
                _strip_target_evidence_cmd(args, stripped_target_evidence_dir),
                log_path=logs_dir / "01b_strip_target_evidence_no_gt.log",
            )
        )
        commands.append(
            command_record(
                "verify_stripped_target_evidence_no_gt",
                _verify_target_evidence_no_gt_cmd(stripped_target_evidence_dir, target_apply_no_gt_audit_path),
                log_path=logs_dir / "01c_verify_stripped_target_evidence_no_gt.log",
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
        "eval_gt_evidence_dir": path_record(args.eval_gt_evidence_dir)
        if args.eval_gt_evidence_dir
        else None,
        "effective_fit_evidence_dir": path_record(args._effective_fit_evidence_dir),
        "effective_target_evidence_dir": path_record(args._effective_target_evidence_dir),
        "effective_eval_gt_evidence_dir": path_record(args._effective_eval_gt_evidence_dir),
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
        "prestripped_target_evidence_dir": path_record(args.prestripped_target_evidence_dir)
        if args.prestripped_target_evidence_dir
        else None,
    }
    settings = {key: value for key, value in vars(args).items() if key not in {"source_model", "fit_evidence_dir", "target_evidence_dir", "eval_gt_evidence_dir", "region_carrier_json", "teacher_render_dir", "parent_render_dir"}}
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
