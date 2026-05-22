#!/usr/bin/env python3
"""Run Phase-K barycentric residual recovery and train-val gate for scenes.

This is an orchestration script. It uses the fixed Phase-J selected compact
checkpoint and the fixed Phase-J ELA policy for each scene, then evaluates a
single barycentric residual candidate with a train-heldout representation gate.
Held-out test metrics are collected as report-only evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTDOOR_SCENES = {"bicycle", "flowers", "garden", "stump", "treehill"}
PHASEJ_METHOD = "ours_26000_phasej_guarded_adaptedge_ela"
BASE_METHOD = "ours_26000_phasef_extra_compact_base"


def _path_for_summary(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _cleanup_scene_train_artifacts(args: argparse.Namespace, candidate_model: Path) -> None:
    if not bool(args.cleanup_train_artifacts_after_scene):
        return
    for method_name in (args.candidate_base_method, args.candidate_trainval_method):
        train_dir = candidate_model / "train" / str(method_name)
        if train_dir.exists() or train_dir.is_symlink():
            _remove_tree(train_dir)


def _metric(path: Path, method: str) -> dict[str, float]:
    row = _read_json(path).get(method, {})
    out: dict[str, float] = {}
    for key in ("PSNR", "SSIM", "LPIPS"):
        try:
            value = float(row.get(key))
        except Exception:
            value = math.nan
        out[key] = value if math.isfinite(value) else math.nan
    return out


def _has_metric(path: Path, method: str) -> bool:
    values = _metric(path, method)
    return all(math.isfinite(values[key]) for key in ("PSNR", "SSIM", "LPIPS"))


def _run(cmd: list[str], *, gpu: int, log_path: Path, wandb_online: bool = False) -> None:
    env = os.environ.copy()
    if int(gpu) >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if wandb_online:
        env["WANDB_MODE"] = "online"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
        handle.write(f"\n[exit_code] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}")


def _fmt_arg(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12f}".rstrip("0").rstrip(".")
    return str(value)


def _selected_model(policy_root: Path, scene: str) -> Path:
    selected = _read_json(policy_root / scene / "summary.json").get("selected", {})
    model_path = selected.get("model_path")
    if not model_path:
        raise RuntimeError(f"missing selected model for {scene}: {policy_root / scene / 'summary.json'}")
    model = ROOT / model_path
    if not model.is_dir():
        raise FileNotFoundError(model)
    return model


def _scene_format_path(value: str | Path, scene: str) -> str:
    text = str(value)
    if "{scene}" in text:
        return text.format(scene=scene)
    return text


def _image_set(scene: str, args: argparse.Namespace) -> str:
    return args.outdoor_images if scene in OUTDOOR_SCENES else args.indoor_images


def _render_maps(
    args: argparse.Namespace,
    *,
    scene: str,
    model: Path,
    method_name: str,
    log_path: Path,
) -> None:
    train_dir = model / "train" / method_name
    test_dir = model / "test" / method_name
    if (
        not bool(args.force)
        and (train_dir / "camera_index.json").is_file()
        and (test_dir / "camera_index.json").is_file()
        and (train_dir / "depths").is_dir()
        and (test_dir / "depths").is_dir()
    ):
        return
    cmd = [
        sys.executable,
        "scripts/car_model/meshsplatopt_render_evidence_maps.py",
        "-s",
        str(Path(args.dataset_root) / scene),
        "-m",
        str(model),
        "-i",
        _image_set(scene, args),
        "--resolution",
        "-1",
        "--eval",
        "--iteration",
        str(args.iteration),
        "--method_name",
        method_name,
        "--quiet",
    ]
    if bool(args.skip_failed_views):
        cmd.append("--skip_failed_views")
    _run(cmd, gpu=int(args.gpu), log_path=log_path)


def _build_evidence(args: argparse.Namespace, *, scene: str, phasej_model: Path, evidence_dir: Path, log_path: Path) -> None:
    summary = evidence_dir / "surface_evidence_summary.json"
    existing_summary = _read_json(summary)
    operator = str(args.delta_operator)
    has_camera_center = "camera_center" in existing_summary.get("per_view_npz_fields", [])
    evidence_split = str(existing_summary.get("split", "")).strip().lower()
    train_split_ok = evidence_split == "train"
    requires_barycentric = operator == "subdivision" or not bool(args.delta_uniform_barycentric)
    rich_surface_ok = existing_summary.get("barycentric_available") is True or not requires_barycentric
    if (
        not bool(args.force)
        and summary.is_file()
        and train_split_ok
        and rich_surface_ok
        and (operator not in {"sh1", "facelocal_sh1"} or has_camera_center)
    ):
        return
    cmd = [
        sys.executable,
        "scripts/car_model/ecsr_build_surface_evidence_cache.py",
        "-s",
        str(Path(args.dataset_root) / scene),
        "-m",
        str(phasej_model),
        "-i",
        _image_set(scene, args),
        "--resolution",
        "-1",
        "--eval",
        "--iteration",
        str(args.iteration),
        "--scene_name",
        scene,
        "--out_dir",
        str(evidence_dir.parent),
        "--base_method_name",
        BASE_METHOD,
        "--final_method_name",
        PHASEJ_METHOD,
        "--max_views",
        str(args.evidence_max_views),
        "--view_stride",
        str(args.evidence_view_stride),
        "--view_offset",
        str(args.evidence_view_offset),
        "--high_error_quantile",
        str(args.evidence_high_error_quantile),
        "--top_k_faces",
        str(args.delta_top_k),
        "--save_view_npz",
        "--save_residual_rgb",
        "--quiet",
    ]
    if requires_barycentric:
        cmd.append("--save_barycentric")
    _run(cmd, gpu=int(args.gpu), log_path=log_path)


def _apply_delta(
    args: argparse.Namespace,
    *,
    scene: str,
    phasej_model: Path,
    evidence_dir: Path,
    candidate_model: Path,
    log_path: Path,
) -> None:
    if str(args.delta_operator) == "subdivision":
        audit = _candidate_audit_path(args, candidate_model)
        checkpoint = candidate_model / "point_cloud" / f"iteration_{int(args.iteration)}" / "point_cloud_state_dict.pt"
        if not bool(args.force) and audit.is_file() and checkpoint.is_file():
            return
        min_policy_val_relative_gain = args.delta_subdivision_min_policy_val_relative_gain
        if min_policy_val_relative_gain is None:
            min_policy_val_relative_gain = args.delta_min_face_policy_val_relative_gain
        cmd = [
            sys.executable,
            "scripts/car_model/ecsr_apply_surface_residual_subdivision_delta.py",
            "--source_model",
            str(phasej_model),
            "--evidence_dir",
            str(evidence_dir),
            "--output_model",
            str(candidate_model),
            "--iteration",
            str(args.iteration),
            "--top_k",
            str(args.delta_top_k),
            "--min_view_hits",
            str(args.delta_min_view_hits),
            "--min_consistency",
            str(args.delta_min_consistency),
            "--min_pixel_count",
            str(args.delta_min_pixel_count),
            "--max_samples_per_face_view",
            str(args.delta_max_samples_per_face_view),
            "--high_error_quantile",
            str(args.delta_high_error_quantile),
            "--strength",
            str(args.delta_strength),
            "--max_abs_delta_rgb",
            str(args.delta_max_abs_rgb),
            "--feature_mode",
            str(args.delta_subdivision_feature_mode),
            "--materialize_mode",
            str(args.delta_subdivision_materialize_mode),
            "--max_abs_sh_coeff",
            str(args.delta_subdivision_max_abs_sh_coeff),
            "--lambda_ridge",
            str(args.delta_subdivision_lambda_ridge),
            "--min_fit_samples",
            str(args.delta_subdivision_min_fit_samples),
            "--min_val_samples",
            str(args.delta_subdivision_min_val_samples),
            "--min_policy_val_relative_gain",
            str(min_policy_val_relative_gain),
            "--min_policy_val_offsets",
            str(args.delta_min_policy_val_offsets),
            "--min_policy_val_offset_fraction",
            str(args.delta_min_policy_val_offset_fraction),
            "--max_faces_to_apply",
            str(args.delta_max_faces_to_apply),
            "--min_effective_mean_relative_gain",
            str(args.delta_subdivision_min_effective_mean_relative_gain),
            "--min_effective_min_relative_gain",
            str(args.delta_subdivision_min_effective_min_relative_gain),
            "--min_effective_delta_abs_mean",
            str(args.delta_subdivision_min_effective_delta_abs_mean),
            "--min_materialized_attribute_delta",
            str(args.delta_subdivision_min_materialized_attribute_delta),
            "--vertex_delta_min_incident_support_fraction",
            str(args.delta_subdivision_vertex_delta_min_incident_support_fraction),
            "--vertex_delta_max_incident_faces",
            str(args.delta_subdivision_vertex_delta_max_incident_faces),
        ]
        if bool(args.delta_subdivision_allow_no_effect_accept):
            cmd.append("--allow_no_effect_accept")
        if bool(args.delta_subdivision_luma_preserve):
            cmd.extend(
                [
                    "--luma_preserve",
                    "--min_luma_relative_gain",
                    str(args.delta_subdivision_min_luma_relative_gain),
                    "--max_mean_luma_shift",
                    str(args.delta_subdivision_max_mean_luma_shift),
                    "--luma_shrink_grid",
                    str(args.delta_subdivision_luma_shrink_grid),
                    "--luma_shrink_selection",
                    str(args.delta_subdivision_luma_shrink_selection),
                ]
            )
        if bool(args.delta_subdivision_structure_preserve):
            cmd.extend(
                [
                    "--structure_preserve",
                    "--structure_weight_strength",
                    str(args.delta_subdivision_structure_weight_strength),
                    "--min_structure_relative_gain",
                    str(args.delta_subdivision_min_structure_relative_gain),
                    "--max_structure_mean_luma_shift",
                    str(args.delta_subdivision_max_structure_mean_luma_shift),
                    "--structure_shrink_grid",
                    str(args.delta_subdivision_structure_shrink_grid),
                    "--structure_shrink_selection",
                    str(args.delta_subdivision_structure_shrink_selection),
                ]
            )
        if bool(args.delta_subdivision_anchor_support):
            cmd.extend(
                [
                    "--anchor_support",
                    "--anchor_max_error_quantile",
                    str(args.delta_subdivision_anchor_max_error_quantile),
                    "--anchor_samples_per_face_view",
                    str(args.delta_subdivision_anchor_samples_per_face_view),
                    "--anchor_weight",
                    str(args.delta_subdivision_anchor_weight),
                ]
            )
        if str(args.delta_subdivision_candidate_plan_out).strip():
            cmd.extend(["--candidate_plan_out", str(Path(args.delta_subdivision_candidate_plan_out))])
        if str(args.delta_subdivision_materialize_plan_in).strip():
            cmd.extend(
                [
                    "--materialize_plan_in",
                    str(Path(args.delta_subdivision_materialize_plan_in)),
                    "--materialize_plan_limit",
                    str(args.delta_subdivision_materialize_plan_limit),
                ]
            )
        if str(args.delta_policy_val_offsets).strip():
            cmd.extend(["--policy_val_offsets", str(args.delta_policy_val_offsets)])
        if int(args.delta_min_face_gain_certificate_views) > 0:
            cmd.extend(
                [
                    "--min_view_gain_views",
                    str(args.delta_min_face_gain_certificate_views),
                    "--min_view_gain_relative_gain",
                    str(args.delta_min_face_gain_certificate_relative_gain),
                    "--min_view_gain_samples",
                    str(args.delta_min_face_gain_certificate_view_samples),
                    "--min_view_gain_fraction",
                    str(args.delta_min_face_gain_certificate_fraction),
                ]
            )
        _run(cmd, gpu=int(args.gpu), log_path=log_path)
        return

    if str(args.delta_operator) == "facelocal_sh1":
        apply_script = "scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py"
    elif str(args.delta_operator) == "sh1":
        apply_script = "scripts/car_model/ecsr_apply_surface_residual_barycentric_sh1_delta.py"
    else:
        apply_script = "scripts/car_model/ecsr_apply_surface_residual_barycentric_delta.py"
    audit = _candidate_audit_path(args, candidate_model)
    checkpoint = candidate_model / "point_cloud" / f"iteration_{int(args.iteration)}" / "point_cloud_state_dict.pt"
    if not bool(args.force) and audit.is_file() and checkpoint.is_file():
        return
    cmd = [
        sys.executable,
        apply_script,
        "--source_model",
        str(phasej_model),
        "--evidence_dir",
        str(evidence_dir),
        "--output_model",
        str(candidate_model),
        "--iteration",
        str(args.iteration),
        "--top_k",
        str(args.delta_top_k),
        "--min_view_hits",
        str(args.delta_min_view_hits),
        "--min_consistency",
        str(args.delta_min_consistency),
        "--min_pixel_count",
        str(args.delta_min_pixel_count),
        "--max_samples_per_face_view",
        str(args.delta_max_samples_per_face_view),
        "--max_total_samples",
        str(args.delta_max_total_samples),
        "--high_error_quantile",
        str(args.delta_high_error_quantile),
        "--face_score_weight_power",
        _fmt_arg(args.delta_face_score_weight_power),
        "--face_score_weight_max",
        _fmt_arg(args.delta_face_score_weight_max),
        "--strength",
        str(args.delta_strength),
        "--max_abs_delta_rgb",
        str(args.delta_max_abs_rgb),
        "--lambda_mag",
        str(args.delta_lambda_mag),
        "--lambda_smooth",
        str(args.delta_lambda_smooth),
        "--direction_luma_safety_weight",
        _fmt_arg(args.delta_direction_luma_safety_weight),
        "--direction_cosine_weight",
        _fmt_arg(args.delta_direction_cosine_weight),
        "--direction_cosine_margin",
        _fmt_arg(args.delta_direction_cosine_margin),
        "--steps",
        str(args.delta_steps),
        "--shared_residual_field_anchors",
        str(args.delta_shared_residual_field_anchors),
        "--shared_residual_field_sigma",
        _fmt_arg(args.delta_shared_residual_field_sigma),
        "--shared_residual_field_lr",
        _fmt_arg(args.delta_shared_residual_field_lr),
        "--shared_residual_field_weight_l2",
        _fmt_arg(args.delta_shared_residual_field_weight_l2),
        "--shared_residual_field_view_hinge_weight",
        _fmt_arg(args.delta_shared_residual_field_view_hinge_weight),
        "--shared_residual_field_view_hinge_min_samples",
        str(args.delta_shared_residual_field_view_hinge_min_samples),
        "--shared_residual_field_duplicate_smooth_weight",
        _fmt_arg(args.delta_shared_residual_field_duplicate_smooth_weight),
        "--min_policy_val_relative_gain",
        str(args.delta_min_policy_val_relative_gain),
        "--min_policy_val_samples",
        str(args.delta_min_policy_val_samples),
        "--min_policy_val_unique_faces",
        str(args.delta_min_policy_val_unique_faces),
    ]
    if str(args.delta_facelocal_region_carrier_json).strip():
        cmd.extend(
            [
                "--region_carrier_json",
                _scene_format_path(args.delta_facelocal_region_carrier_json, scene),
                "--region_core_weight",
                _fmt_arg(args.delta_region_core_weight),
                "--region_context_weight",
                _fmt_arg(args.delta_region_context_weight),
                "--region_outside_weight",
                _fmt_arg(args.delta_region_outside_weight),
                "--region_boundary_px",
                str(args.delta_region_boundary_px),
            ]
        )
    if bool(args.delta_render_region_objective):
        cmd.extend(
            [
                "--render_region_objective",
                "--render_region_core_weight",
                _fmt_arg(args.delta_render_region_core_weight),
                "--render_region_context_weight",
                _fmt_arg(args.delta_render_region_context_weight),
                "--render_region_outside_penalty",
                _fmt_arg(args.delta_render_region_outside_penalty),
                "--render_region_tail_cvar_weight",
                _fmt_arg(args.delta_render_region_tail_cvar_weight),
                "--render_region_tail_fraction",
                _fmt_arg(args.delta_render_region_tail_fraction),
                "--render_region_min_view_samples",
                str(args.delta_render_region_min_view_samples),
            ]
        )
    else:
        cmd.append("--no-render_region_objective")
    if str(args.delta_operator) in {"sh1", "facelocal_sh1"}:
        cmd.extend(
            [
                "--max_abs_sh_coeff",
                str(args.delta_max_abs_sh_coeff),
                "--lambda_sh1_mag",
                str(args.delta_lambda_sh1_mag),
            ]
        )
        if str(args.delta_operator) == "facelocal_sh1":
            cmd.extend(["--sh_degree", str(args.delta_sh_degree)])
            cmd.extend(
                [
                    "--coefficient_lowpass_mode",
                    str(args.delta_coefficient_lowpass_mode),
                    "--coefficient_lowpass_sh_scale",
                    _fmt_arg(args.delta_coefficient_lowpass_sh_scale),
                ]
            )
            if str(args.delta_facelocal_candidate_plan_out).strip():
                cmd.extend(["--candidate_plan_out", _scene_format_path(args.delta_facelocal_candidate_plan_out, scene)])
            if str(args.delta_facelocal_materialize_plan_in).strip():
                cmd.extend(
                    [
                        "--materialize_plan_in",
                        _scene_format_path(args.delta_facelocal_materialize_plan_in, scene),
                        "--materialize_plan_limit",
                        str(args.delta_facelocal_materialize_plan_limit),
                        "--materialize_plan_scale",
                        str(args.delta_facelocal_materialize_plan_scale),
                    ]
                )
                if str(args.delta_facelocal_materialize_plan_face_ids).strip():
                    cmd.extend(
                        [
                            "--materialize_plan_face_ids",
                            str(args.delta_facelocal_materialize_plan_face_ids),
                        ]
                    )
                if str(args.delta_facelocal_materialize_plan_alpha_json).strip():
                    cmd.extend(
                        [
                            "--materialize_plan_alpha_json",
                            _scene_format_path(args.delta_facelocal_materialize_plan_alpha_json, scene),
                        ]
                    )
                if str(args.delta_facelocal_materialize_plan_render_trust_json).strip():
                    cmd.extend(
                        [
                            "--materialize_plan_render_trust_json",
                            _scene_format_path(args.delta_facelocal_materialize_plan_render_trust_json, scene),
                        ]
                    )
                if bool(args.delta_facelocal_materialize_allow_uncertified_plan):
                    cmd.append("--materialize_allow_uncertified_plan")
                else:
                    cmd.append("--no-materialize_allow_uncertified_plan")
        if bool(args.delta_uniform_barycentric):
            cmd.append("--uniform_barycentric")
        if str(args.delta_operator) == "facelocal_sh1":
            if bool(args.delta_shared_residual_field):
                cmd.append("--shared_residual_field")
            else:
                cmd.append("--no-shared_residual_field")
    if str(args.delta_operator) == "facelocal_sh1":
        cmd.extend(
            [
                "--validation_shrink_mode",
                str(args.delta_validation_shrink_mode),
                "--validation_gain_max_scale",
                _fmt_arg(args.delta_validation_gain_max_scale),
                "--validation_shrink_min_samples",
                str(args.delta_validation_shrink_min_samples),
                "--crossfold_gain_certificate_folds",
                str(args.delta_crossfold_gain_certificate_folds),
                "--crossfold_min_passing_folds",
                str(args.delta_crossfold_min_passing_folds),
                "--crossfold_min_fold_relative_gain",
                _fmt_arg(args.delta_crossfold_min_fold_relative_gain),
                "--crossfold_min_fold_samples",
                str(args.delta_crossfold_min_fold_samples),
                "--patch_cert_rings",
                str(args.delta_patch_cert_rings),
                "--patch_cert_max_faces_per_seed",
                str(args.delta_patch_cert_max_faces_per_seed),
                "--patch_cert_min_direction_cosine",
                _fmt_arg(args.delta_patch_cert_min_direction_cosine),
                "--patch_cert_min_neighbor_policy_val_samples",
                str(args.delta_patch_cert_min_neighbor_policy_val_samples),
                "--patch_cert_min_neighbor_policy_val_relative_gain",
                _fmt_arg(args.delta_patch_cert_min_neighbor_policy_val_relative_gain),
                "--patch_cert_min_policy_val_samples",
                str(args.delta_patch_cert_min_policy_val_samples),
                "--patch_cert_min_relative_gain",
                _fmt_arg(args.delta_patch_cert_min_relative_gain),
                "--patch_cert_neighbor_mode",
                str(args.delta_patch_cert_neighbor_mode),
                "--patch_cert_centroid_candidates_per_seed",
                str(args.delta_patch_cert_centroid_candidates_per_seed),
                "--patch_cert_seed_rescue_min_candidates",
                str(args.delta_patch_cert_seed_rescue_min_candidates),
                "--patch_cert_seed_rescue_max_seeds",
                str(args.delta_patch_cert_seed_rescue_max_seeds),
                "--patch_cert_seed_rescue_min_aux_witnesses",
                str(args.delta_patch_cert_seed_rescue_min_aux_witnesses),
                "--patch_cert_crossfold_folds",
                str(args.delta_patch_cert_crossfold_folds),
                "--patch_cert_crossfold_min_passing_folds",
                str(args.delta_patch_cert_crossfold_min_passing_folds),
                "--patch_cert_crossfold_min_fold_relative_gain",
                _fmt_arg(args.delta_patch_cert_crossfold_min_fold_relative_gain),
                "--patch_cert_crossfold_min_fold_samples",
                str(args.delta_patch_cert_crossfold_min_fold_samples),
                "--patch_cert_cluster_basis_mode",
                str(args.delta_patch_cert_cluster_basis_mode),
                "--patch_cert_cluster_basis_steps",
                str(args.delta_patch_cert_cluster_basis_steps),
                "--patch_cert_cluster_basis_rank",
                str(args.delta_patch_cert_cluster_basis_rank),
                "--patch_cert_cluster_basis_lr",
                _fmt_arg(args.delta_patch_cert_cluster_basis_lr),
                "--patch_cert_cluster_basis_min_samples",
                str(args.delta_patch_cert_cluster_basis_min_samples),
                "--patch_cert_cluster_basis_max_scale",
                _fmt_arg(args.delta_patch_cert_cluster_basis_max_scale),
                "--patch_cert_cluster_basis_max_fit_mse_regression",
                _fmt_arg(args.delta_patch_cert_cluster_basis_max_fit_mse_regression),
                "--patch_cert_cluster_basis_init",
                str(args.delta_patch_cert_cluster_basis_init),
                "--patch_cert_cluster_basis_view_hinge_weight",
                _fmt_arg(args.delta_patch_cert_cluster_basis_view_hinge_weight),
                "--patch_cert_cluster_basis_view_hinge_min_samples",
                str(args.delta_patch_cert_cluster_basis_view_hinge_min_samples),
                "--patch_cert_cluster_basis_geometry_smooth_weight",
                _fmt_arg(args.delta_patch_cert_cluster_basis_geometry_smooth_weight),
                "--patch_cert_carrier_holdout_groups",
                str(args.delta_patch_cert_carrier_holdout_groups),
                "--patch_cert_carrier_holdout_grouping",
                str(args.delta_patch_cert_carrier_holdout_grouping),
                "--patch_cert_carrier_holdout_min_passing_groups",
                str(args.delta_patch_cert_carrier_holdout_min_passing_groups),
                "--patch_cert_carrier_holdout_min_group_relative_gain",
                _fmt_arg(args.delta_patch_cert_carrier_holdout_min_group_relative_gain),
                "--patch_cert_carrier_holdout_min_group_samples",
                str(args.delta_patch_cert_carrier_holdout_min_group_samples),
                "--patch_cert_carrier_holdout_max_mse_regression",
                _fmt_arg(args.delta_patch_cert_carrier_holdout_max_mse_regression),
                "--patch_cert_carrier_holdout_cvar_fraction",
                _fmt_arg(args.delta_patch_cert_carrier_holdout_cvar_fraction),
                "--patch_cert_carrier_holdout_cvar_weight",
                _fmt_arg(args.delta_patch_cert_carrier_holdout_cvar_weight),
                "--patch_cert_carrier_holdout_max_carriers",
                str(args.delta_patch_cert_carrier_holdout_max_carriers),
                "--patch_cert_carrier_holdout_auto_prefix_min_faces",
                str(args.delta_patch_cert_carrier_holdout_auto_prefix_min_faces),
                "--patch_cert_carrier_holdout_auto_prefix_face_bonus",
                _fmt_arg(args.delta_patch_cert_carrier_holdout_auto_prefix_face_bonus),
            ]
        )
        if bool(args.delta_patch_cert_cluster_basis):
            cmd.append("--patch_cert_cluster_basis")
        else:
            cmd.append("--no-patch_cert_cluster_basis")
        if bool(args.delta_patch_cert_neighbor_crossfold):
            cmd.append("--patch_cert_neighbor_crossfold")
        else:
            cmd.append("--no-patch_cert_neighbor_crossfold")
        if bool(args.delta_patch_cert_shrink):
            cmd.append("--patch_cert_shrink")
        else:
            cmd.append("--no-patch_cert_shrink")
        if bool(args.delta_patch_cert_seed_rescue):
            cmd.append("--patch_cert_seed_rescue")
        else:
            cmd.append("--no-patch_cert_seed_rescue")
        if bool(args.delta_patch_cert_carrier_holdout_selector):
            cmd.append("--patch_cert_carrier_holdout_selector")
        else:
            cmd.append("--no-patch_cert_carrier_holdout_selector")
        if bool(args.delta_patch_cert_carrier_holdout_disjoint):
            cmd.append("--patch_cert_carrier_holdout_disjoint")
        else:
            cmd.append("--no-patch_cert_carrier_holdout_disjoint")
        if bool(args.delta_patch_cert_carrier_holdout_auto_prefix):
            cmd.append("--patch_cert_carrier_holdout_auto_prefix")
        else:
            cmd.append("--no-patch_cert_carrier_holdout_auto_prefix")
        if bool(args.delta_patch_cert_carrier_holdout_auto_prefix_positive_tail_safe):
            cmd.append("--patch_cert_carrier_holdout_auto_prefix_positive_tail_safe")
        else:
            cmd.append("--no-patch_cert_carrier_holdout_auto_prefix_positive_tail_safe")
        if bool(args.delta_strict_patchcert_carrier):
            cmd.append("--strict_patchcert_carrier")
    sh1_view_consensus = (
        str(args.delta_operator) in {"sh1", "facelocal_sh1"}
        and float(args.delta_min_face_view_consensus) > 0.0
    )
    facelocal_view_gain_certificate = (
        str(args.delta_operator) == "facelocal_sh1"
        and int(args.delta_min_face_gain_certificate_views) > 0
    )
    facelocal_prediction_safety = (
        str(args.delta_operator) == "facelocal_sh1"
        and float(args.delta_min_face_prediction_safety_fraction) > 0.0
    )
    if str(args.delta_operator) == "facelocal_sh1" or (
        str(args.delta_operator) == "sh1" and (bool(args.delta_sh1_face_policy) or sh1_view_consensus)
    ):
        cmd.extend(
            [
                "--max_faces_to_apply",
                str(args.delta_max_faces_to_apply),
                "--min_face_policy_val_relative_gain",
                str(args.delta_min_face_policy_val_relative_gain),
                "--min_face_policy_val_samples",
                str(args.delta_min_face_policy_val_samples),
            ]
        )
    if sh1_view_consensus:
        cmd.extend(
            [
                "--min_face_view_consensus",
                str(args.delta_min_face_view_consensus),
                "--min_face_consensus_views",
                str(args.delta_min_face_consensus_views),
                "--min_face_consensus_view_samples",
                str(args.delta_min_face_consensus_view_samples),
                "--face_consensus_min_cosine",
                str(args.delta_face_consensus_min_cosine),
            ]
        )
    if facelocal_view_gain_certificate:
        cmd.extend(
            [
                "--min_face_gain_certificate_views",
                str(args.delta_min_face_gain_certificate_views),
                "--min_face_gain_certificate_relative_gain",
                str(args.delta_min_face_gain_certificate_relative_gain),
                "--min_face_gain_certificate_view_samples",
                str(args.delta_min_face_gain_certificate_view_samples),
                "--min_face_gain_certificate_fraction",
                str(args.delta_min_face_gain_certificate_fraction),
            ]
        )
    if facelocal_prediction_safety:
        cmd.extend(
            [
                "--min_face_prediction_safety_fraction",
                str(args.delta_min_face_prediction_safety_fraction),
                "--min_face_prediction_safety_samples",
                str(args.delta_min_face_prediction_safety_samples),
                "--face_prediction_safety_min_cosine",
                str(args.delta_face_prediction_safety_min_cosine),
            ]
        )
    if str(args.delta_operator) == "dc":
        if bool(args.delta_policy_val_filter_faces):
            cmd.extend(
                [
                    "--policy_val_filter_faces",
                    "--policy_val_face_min_samples",
                    str(args.delta_policy_val_face_min_samples),
                    "--policy_val_face_min_relative_gain",
                    str(args.delta_policy_val_face_min_relative_gain),
                    "--policy_val_face_max_keep",
                    str(args.delta_policy_val_face_max_keep),
                ]
            )
        if str(args.delta_candidate_cluster_json).strip():
            cmd.extend(["--candidate_cluster_json", _scene_format_path(args.delta_candidate_cluster_json, scene)])
        if str(args.delta_candidate_cluster_csv).strip():
            cmd.extend(["--candidate_cluster_csv", _scene_format_path(args.delta_candidate_cluster_csv, scene)])
        cmd.extend(
            [
                "--cluster_operator_types",
                str(args.delta_cluster_operator_types),
                "--max_clusters",
                str(args.delta_max_clusters),
                "--cluster_min_redundancy_score",
                str(args.delta_cluster_min_redundancy_score),
                "--cluster_expand_target_faces",
                str(args.delta_cluster_expand_target_faces),
            ]
        )
        if bool(args.delta_cluster_expand_with_top_residual_faces):
            cmd.append("--cluster_expand_with_top_residual_faces")
    _run(cmd, gpu=int(args.gpu), log_path=log_path)


def _candidate_audit_path(args: argparse.Namespace, candidate_model: Path) -> Path:
    if str(args.delta_operator) == "subdivision":
        return candidate_model / "surface_residual_subdivision_delta_audit.json"
    if str(args.delta_operator) == "facelocal_sh1":
        return candidate_model / "surface_residual_facelocal_sh1_delta_audit.json"
    if str(args.delta_operator) == "sh1":
        return candidate_model / "surface_residual_barycentric_sh1_delta_audit.json"
    return candidate_model / "surface_residual_barycentric_delta_audit.json"


def _policy_args(report: dict[str, Any], *, trainval: bool, args: argparse.Namespace) -> list[str]:
    policy = report.get("policy") or {}
    out = [
        "--mode",
        str(policy.get("mode", "residual")),
        "--k",
        str(int(policy.get("k", 4))),
        "--residual_clip",
        str(float(policy.get("residual_clip", 0.2))),
        "--depth_abs_tol",
        str(float(policy.get("depth_abs_tol", 0.02))),
        "--depth_rel_tol",
        str(float(policy.get("depth_rel_tol", 0.06))),
        "--direction_weight",
        str(float(policy.get("direction_weight", 0.35))),
    ]
    if bool(policy.get("edge_gate", False)):
        out.extend(
            [
                "--edge_gate",
                "--edge_gate_quantile",
                str(float(policy.get("edge_gate_quantile", 0.5))),
                "--edge_gate_dilate",
                str(int(policy.get("edge_gate_dilate", 1))),
                "--edge_gate_min",
                str(float(policy.get("edge_gate_min", 0.0))),
            ]
        )
    alpha_policy = str(report.get("alpha_policy", "global"))
    if alpha_policy == "adaptive_bins":
        out.extend(
            [
                "--alpha",
                "0",
                "--skip_fixed_alpha_calibration",
                "--alpha_policy",
                "adaptive_bins",
                "--alpha_feature_mode",
                str(args.alpha_feature_mode),
                "--alpha_default",
                str(args.alpha_default),
            ]
        )
    else:
        out.extend(["--alpha", str(float(report.get("alpha", 0.0))), "--skip_fixed_alpha_calibration"])
    if trainval:
        out.extend(
            [
                "--policy_holdout_fraction",
                str(args.policy_holdout_fraction),
                "--policy_holdout_offset",
                str(args.policy_holdout_offset),
                "--support_policy_fit_only",
                "--calib_sampler",
                args.calib_sampler,
                "--calib_max_views",
                str(args.calib_max_views),
                "--calib_stride",
                str(args.calib_stride),
            ]
        )
    return out


def _apply_ela(
    args: argparse.Namespace,
    *,
    scene: str,
    model: Path,
    base_method: str,
    method_name: str,
    phasej_report: dict[str, Any],
    target_split: str,
    log_path: Path,
) -> None:
    report = model / target_split / method_name / "ela_report.json"
    if not bool(args.force) and report.is_file():
        return
    trainval = target_split == "train"
    cmd = [
        sys.executable,
        "scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py",
        "--base_model_path",
        str(model),
        "--iteration",
        str(args.iteration),
        "--base_method_name",
        base_method,
        "--target_split",
        target_split,
        "--method_name",
        method_name,
        "--wandb",
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        args.wandb_group,
        "--wandb_name",
        f"{args.wandb_name}_{scene}_{method_name}_{target_split}",
    ]
    if str(args.ela_policy_source) == "fixed_phasej":
        cmd.extend(_policy_args(phasej_report, trainval=trainval, args=args))
    elif str(args.ela_policy_source) == "per_model_auto":
        cmd.extend(
            [
                "--auto_policy",
                "--policy_modes",
                str(args.ela_policy_modes),
                "--policy_k_values",
                str(args.ela_policy_k_values),
                "--policy_depth_rel_values",
                str(args.ela_policy_depth_rel_values),
                "--policy_residual_clip_values",
                str(args.ela_policy_residual_clip_values),
                "--policy_direction_weight_values",
                str(args.ela_policy_direction_weight_values),
                "--policy_edge_gate_quantiles",
                str(args.ela_policy_edge_gate_quantiles),
                "--policy_edge_gate_dilates",
                str(args.ela_policy_edge_gate_dilates),
                "--policy_objective",
                str(args.ela_policy_objective),
                "--alpha_grid",
                str(args.ela_alpha_grid),
                "--calib_sampler",
                str(args.calib_sampler),
                "--calib_max_views",
                str(args.calib_max_views),
                "--calib_stride",
                str(args.calib_stride),
                "--policy_holdout_fraction",
                str(args.policy_holdout_fraction),
                "--policy_holdout_offset",
                str(args.policy_holdout_offset),
            ]
        )
        if bool(args.ela_edge_gate):
            cmd.extend(["--edge_gate", "--edge_gate_min", str(args.ela_edge_gate_min)])
        if bool(args.ela_calib_lpips):
            cmd.append("--calib_lpips")
        if trainval and bool(args.support_policy_fit_only):
            cmd.append("--support_policy_fit_only")
    else:
        raise ValueError(f"unknown ELA policy source: {args.ela_policy_source}")
    _run(cmd, gpu=int(args.gpu), log_path=log_path, wandb_online=True)


def _evaluate_trainval(
    args: argparse.Namespace,
    *,
    model: Path,
    method: str,
    view_names_file: Path,
    output: Path,
    per_view_output: Path,
    log_path: Path,
) -> None:
    if not bool(args.force) and _has_metric(output, method) and per_view_output.is_file():
        return
    cmd = [
        sys.executable,
        "scripts/car_model/evaluate_render_split_metrics.py",
        "-m",
        str(model),
        "--split",
        "train",
        "--methods",
        method,
        "--view_names_file",
        str(view_names_file),
        "--view_names_key",
        "policy_val_views",
        "--output",
        str(output),
        "--per_view_output",
        str(per_view_output),
    ]
    _run(cmd, gpu=int(args.gpu), log_path=log_path)


def _evaluate_train_render_region_objective(
    args: argparse.Namespace,
    *,
    scene: str,
    phasej_model: Path,
    candidate_model: Path,
    output_root: Path,
    log_path: Path,
) -> Path:
    source = str(args.train_render_region_eval_source)
    if source == "raw_base":
        out_json = output_root / scene / "train_render_region_objective_raw_base.json"
        out_md = output_root / scene / "train_render_region_objective_raw_base.md"
        baseline_dir = phasej_model / "train" / BASE_METHOD
        candidate_dir = candidate_model / "train" / args.candidate_base_method
        baseline_label = BASE_METHOD
        candidate_label = args.candidate_base_method
    elif source == "post_ela_trainval":
        out_json = output_root / scene / "train_render_region_objective.json"
        out_md = output_root / scene / "train_render_region_objective.md"
        baseline_dir = phasej_model / "train" / args.phasej_trainval_method
        candidate_dir = candidate_model / "train" / args.candidate_trainval_method
        baseline_label = args.phasej_trainval_method
        candidate_label = args.candidate_trainval_method
    else:
        raise ValueError(f"unsupported --train_render_region_eval_source: {source}")
    if not bool(args.force) and out_json.is_file():
        return out_json
    carrier_template = str(args.train_render_region_carrier_json).strip() or str(args.delta_facelocal_region_carrier_json).strip()
    if not carrier_template:
        raise ValueError("--train_render_region_gate_enable requires --train_render_region_carrier_json or --delta_facelocal_region_carrier_json")
    cmd = [
        sys.executable,
        "scripts/car_model/ecsr_eval_train_render_region_objective.py",
        "--scene",
        scene,
        "--carrier_json",
        _scene_format_path(carrier_template, scene),
        "--baseline_dir",
        str(baseline_dir),
        "--candidate_dir",
        str(candidate_dir),
        "--baseline_label",
        baseline_label,
        "--candidate_label",
        candidate_label,
        "--output_json",
        str(out_json),
        "--output_md",
        str(out_md),
        "--max_regions",
        str(args.train_render_region_max_regions),
        "--min_region_pixels",
        str(args.train_render_region_min_pixels),
        "--min_crop_size",
        str(args.train_render_region_min_crop_size),
        "--context_pad",
        str(args.train_render_region_context_pad),
        "--tail_fraction",
        _fmt_arg(args.train_render_region_tail_fraction),
        "--ssim_weight",
        _fmt_arg(args.gate_ssim_weight),
        "--lpips_weight",
        _fmt_arg(args.gate_lpips_weight),
        "--device",
        "cuda",
    ]
    if bool(args.train_render_region_skip_lpips):
        cmd.append("--skip_lpips")
    _run(cmd, gpu=int(args.gpu), log_path=log_path)
    return out_json


def _evaluate_test(
    args: argparse.Namespace,
    *,
    model: Path,
    method: str,
    log_path: Path,
    output: Path | None = None,
    per_view_output: Path | None = None,
    merge_model_results: bool = True,
) -> None:
    metric_path = output if output is not None else model / "results.json"
    if not bool(args.force) and _has_metric(metric_path, method):
        return
    cmd = [
        sys.executable,
        "scripts/car_model/evaluate_render_split_metrics.py",
        "-m",
        str(model),
        "--split",
        "test",
        "--methods",
        method,
    ]
    if output is not None:
        cmd.extend(["--output", str(output)])
    if per_view_output is not None:
        cmd.extend(["--per_view_output", str(per_view_output)])
    if bool(merge_model_results):
        cmd.append("--merge_model_results")
    _run(cmd, gpu=int(args.gpu), log_path=log_path)


def _decide(
    args: argparse.Namespace,
    *,
    scene: str,
    phasej_model: Path,
    candidate_model: Path,
    output_root: Path,
    phasej_trainval_results: Path,
    phasej_test_results: Path,
    log_path: Path,
    render_region_objective_json: Path | None = None,
) -> dict[str, Any]:
    decision_json = output_root / "decisions" / f"{scene}_decision.json"
    decision_md = output_root / "decisions" / f"{scene}_decision.md"
    if not bool(args.force) and decision_json.is_file():
        return _read_json(decision_json)
    cmd = [
        sys.executable,
        "scripts/car_model/ecsr_decide_phasek_trainval_gate.py",
        "--scene",
        scene,
        "--candidate_label",
        args.candidate_label,
        "--fallback_label",
        "phasej_guarded_adaptedge",
        "--base_trainval_results",
        str(phasej_trainval_results),
        "--base_trainval_method",
        args.phasej_trainval_method,
        "--candidate_trainval_results",
        str(candidate_model / "trainval_gate_results.json"),
        "--candidate_trainval_method",
        args.candidate_trainval_method,
        "--base_trainval_per_view",
        str(output_root / scene / "phasej_trainval_gate_per_view.json"),
        "--candidate_trainval_per_view",
        str(candidate_model / "trainval_gate_per_view.json"),
        "--candidate_audit_json",
        str(_candidate_audit_path(args, candidate_model)),
        "--base_test_results",
        str(phasej_test_results),
        "--base_test_method",
        args.phasej_test_method,
        "--candidate_test_results",
        str(candidate_model / "results.json"),
        "--candidate_test_method",
        args.candidate_test_method,
        "--min_psnr_gain",
        _fmt_arg(args.gate_min_psnr_gain),
        "--max_ssim_regression",
        _fmt_arg(args.gate_max_ssim_regression),
        "--max_lpips_regression",
        _fmt_arg(args.gate_max_lpips_regression),
        "--min_balanced_delta",
        _fmt_arg(args.gate_min_balanced_delta),
        "--ssim_weight",
        _fmt_arg(args.gate_ssim_weight),
        "--lpips_weight",
        _fmt_arg(args.gate_lpips_weight),
        "--tail_cvar_fraction",
        _fmt_arg(args.gate_tail_cvar_fraction),
        "--tail_max_balanced_negative_fraction",
        _fmt_arg(args.gate_tail_max_balanced_negative_fraction),
        "--tail_min_balanced_cvar_delta",
        _fmt_arg(args.gate_tail_min_balanced_cvar_delta),
        "--tail_max_lpips_positive_fraction",
        _fmt_arg(args.gate_tail_max_lpips_positive_fraction),
        "--tail_max_worst_lpips_regression",
        _fmt_arg(args.gate_tail_max_worst_lpips_regression),
        "--stratified_group_count",
        str(args.gate_stratified_group_count),
        "--compact_gate_max_faces",
        str(args.gate_compact_max_faces),
        "--compact_gate_max_vertices",
        str(args.gate_compact_max_vertices),
        "--compact_gate_max_face_ratio",
        _fmt_arg(args.gate_compact_max_face_ratio),
        "--compact_gate_min_psnr_gain",
        _fmt_arg(args.gate_compact_min_psnr_gain),
        "--compact_gate_max_ssim_regression",
        _fmt_arg(args.gate_compact_max_ssim_regression),
        "--compact_gate_max_lpips_regression",
        _fmt_arg(args.gate_compact_max_lpips_regression),
        "--compact_gate_max_balanced_negative_fraction",
        _fmt_arg(args.gate_compact_max_balanced_negative_fraction),
        "--compact_gate_min_balanced_cvar_delta",
        _fmt_arg(args.gate_compact_min_balanced_cvar_delta),
        "--compact_gate_max_lpips_positive_fraction",
        _fmt_arg(args.gate_compact_max_lpips_positive_fraction),
        "--compact_gate_max_worst_lpips_regression",
        _fmt_arg(args.gate_compact_max_worst_lpips_regression),
        "--compact_gate_min_stratified_psnr_delta",
        _fmt_arg(args.gate_compact_min_stratified_psnr_delta),
        "--compact_gate_max_stratified_ssim_regression",
        _fmt_arg(args.gate_compact_max_stratified_ssim_regression),
        "--compact_gate_max_stratified_lpips_regression",
        _fmt_arg(args.gate_compact_max_stratified_lpips_regression),
        "--output_json",
        str(decision_json),
        "--output_md",
        str(decision_md),
    ]
    if bool(args.train_render_region_gate_enable):
        cmd.extend(
            [
                "--render_region_gate_enable",
                "--render_region_objective_json",
                str(render_region_objective_json or output_root / scene / "train_render_region_objective.json"),
                "--render_region_min_regions",
                str(args.train_render_region_min_regions),
                "--render_region_min_changed_regions",
                str(args.train_render_region_min_changed_regions),
                "--render_region_min_changed_fraction",
                _fmt_arg(args.train_render_region_min_changed_fraction),
                "--render_region_min_core_balanced_delta",
                _fmt_arg(args.train_render_region_min_core_balanced_delta),
                "--render_region_min_core_psnr_delta",
                _fmt_arg(args.train_render_region_min_core_psnr_delta),
                "--render_region_min_tail_cvar_delta",
                _fmt_arg(args.train_render_region_min_tail_cvar_delta),
                "--render_region_max_context_mse_regression",
                _fmt_arg(args.train_render_region_max_context_mse_regression),
                "--render_region_max_negative_fraction",
                _fmt_arg(args.train_render_region_max_negative_fraction),
            ]
        )
    if bool(args.gate_tail_require_available):
        cmd.append("--tail_require_available")
    if bool(args.gate_compact_enable):
        cmd.append("--compact_gate_enable")
    if bool(args.gate_compact_require):
        cmd.append("--compact_gate_require")
    _run(cmd, gpu=-1, log_path=log_path)
    return _read_json(decision_json)


def run_scene(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    policy_root = ROOT / args.policy_root
    output_root = ROOT / args.output_root
    evidence_root = ROOT / args.evidence_root
    scene_summary_path = output_root / scene / "phasek_scene_summary.json"
    if not bool(args.force) and scene_summary_path.is_file():
        return _read_json(scene_summary_path)
    phasej_model = _selected_model(policy_root, scene)
    phasej_report_path = phasej_model / "test" / PHASEJ_METHOD / "ela_report.json"
    phasej_report = _read_json(phasej_report_path)
    if not phasej_report:
        raise FileNotFoundError(phasej_report_path)

    evidence_dir = evidence_root / scene
    candidate_model = output_root / scene / "model"
    log_path = output_root / scene / "phasek_barycentric_gate.log"
    phasej_test_results = output_root / scene / "phasej_test_results.json"
    phasej_test_per_view = output_root / scene / "phasej_test_per_view.json"
    phasej_trainval_results = output_root / scene / "phasej_trainval_gate_results.json"
    phasej_trainval_per_view = output_root / scene / "phasej_trainval_gate_per_view.json"
    _render_maps(args, scene=scene, model=phasej_model, method_name=BASE_METHOD, log_path=log_path)
    _build_evidence(args, scene=scene, phasej_model=phasej_model, evidence_dir=evidence_dir, log_path=log_path)
    _apply_delta(
        args,
        scene=scene,
        phasej_model=phasej_model,
        evidence_dir=evidence_dir,
        candidate_model=candidate_model,
        log_path=log_path,
    )
    _render_maps(args, scene=scene, model=candidate_model, method_name=args.candidate_base_method, log_path=log_path)
    _evaluate_test(args, model=phasej_model, method=BASE_METHOD, log_path=log_path)
    _evaluate_test(args, model=candidate_model, method=args.candidate_base_method, log_path=log_path)
    _apply_ela(
        args,
        scene=scene,
        model=phasej_model,
        base_method=BASE_METHOD,
        method_name=args.phasej_test_method,
        phasej_report=phasej_report,
        target_split="test",
        log_path=log_path,
    )
    _evaluate_test(
        args,
        model=phasej_model,
        method=args.phasej_test_method,
        output=phasej_test_results,
        per_view_output=phasej_test_per_view,
        merge_model_results=False,
        log_path=log_path,
    )

    _apply_ela(
        args,
        scene=scene,
        model=phasej_model,
        base_method=BASE_METHOD,
        method_name=args.phasej_trainval_method,
        phasej_report=phasej_report,
        target_split="train",
        log_path=log_path,
    )
    _evaluate_trainval(
        args,
        model=phasej_model,
        method=args.phasej_trainval_method,
        view_names_file=phasej_model / "train" / args.phasej_trainval_method / "ela_report.json",
        output=phasej_trainval_results,
        per_view_output=phasej_trainval_per_view,
        log_path=log_path,
    )
    _apply_ela(
        args,
        scene=scene,
        model=candidate_model,
        base_method=args.candidate_base_method,
        method_name=args.candidate_test_method,
        phasej_report=phasej_report,
        target_split="test",
        log_path=log_path,
    )
    _evaluate_test(args, model=candidate_model, method=args.candidate_test_method, log_path=log_path)
    _apply_ela(
        args,
        scene=scene,
        model=candidate_model,
        base_method=args.candidate_base_method,
        method_name=args.candidate_trainval_method,
        phasej_report=phasej_report,
        target_split="train",
        log_path=log_path,
    )
    _evaluate_trainval(
        args,
        model=candidate_model,
        method=args.candidate_trainval_method,
        view_names_file=(
            candidate_model / "train" / args.candidate_trainval_method / "ela_report.json"
            if str(args.ela_policy_source) == "per_model_auto"
            else phasej_model / "train" / args.phasej_trainval_method / "ela_report.json"
        ),
        output=candidate_model / "trainval_gate_results.json",
        per_view_output=candidate_model / "trainval_gate_per_view.json",
        log_path=log_path,
    )
    render_region_objective_json: Path | None = None
    if bool(args.train_render_region_gate_enable):
        render_region_objective_json = _evaluate_train_render_region_objective(
            args,
            scene=scene,
            phasej_model=phasej_model,
            candidate_model=candidate_model,
            output_root=output_root,
            log_path=log_path,
        )
    decision = _decide(
        args,
        scene=scene,
        phasej_model=phasej_model,
        candidate_model=candidate_model,
        output_root=output_root,
        phasej_trainval_results=phasej_trainval_results,
        phasej_test_results=phasej_test_results,
        log_path=log_path,
        render_region_objective_json=render_region_objective_json,
    )
    scene_summary = {
        "scene": scene,
        "phasej_model": _path_for_summary(phasej_model),
        "candidate_model": _path_for_summary(candidate_model),
        "evidence_dir": _path_for_summary(evidence_dir),
        "decision": decision,
        "log_path": _path_for_summary(log_path),
    }
    scene_summary_path.write_text(
        json.dumps(scene_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _cleanup_scene_train_artifacts(args, candidate_model)
    return scene_summary


def _write_aggregate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    accepted = [row for row in rows if row["decision"].get("accepted")]
    payload = {"rows": rows, "accepted_count": len(accepted), "total_count": len(rows), "args": vars(args)}
    (output_root / "phasek_barycentric_gate_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Phase-K Barycentric Gate Summary",
        "",
        f"- scenes: `{len(rows)}`",
        f"- accepted: `{len(accepted)}`",
        "",
        "| scene | selected | accepted | raw dPSNR | raw dSSIM | raw dLPIPS | train-val dPSNR | train-val dSSIM | train-val dLPIPS | test dPSNR | test dSSIM | test dLPIPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        decision = row["decision"]
        td = decision.get("trainval_delta", {})
        hd = decision.get("test_delta_report_only", {})
        raw_delta = {"PSNR": math.nan, "SSIM": math.nan, "LPIPS": math.nan}
        try:
            phasej_model = ROOT / row["phasej_model"]
            candidate_model = ROOT / row["candidate_model"]
            base_raw = _metric(phasej_model / "results.json", BASE_METHOD)
            candidate_raw = _metric(candidate_model / "results.json", args.candidate_base_method)
            raw_delta = {key: candidate_raw[key] - base_raw[key] for key in ("PSNR", "SSIM", "LPIPS")}
        except Exception:
            pass
        lines.append(
            f"| {row['scene']} | {decision.get('selected_label')} | {str(decision.get('accepted')).lower()} | "
            f"{float(raw_delta.get('PSNR', math.nan)):+.6f} | {float(raw_delta.get('SSIM', math.nan)):+.6f} | {float(raw_delta.get('LPIPS', math.nan)):+.6f} | "
            f"{float(td.get('PSNR', math.nan)):+.6f} | {float(td.get('SSIM', math.nan)):+.6f} | {float(td.get('LPIPS', math.nan)):+.6f} | "
            f"{float(hd.get('PSNR', math.nan)):+.6f} | {float(hd.get('SSIM', math.nan)):+.6f} | {float(hd.get('LPIPS', math.nan)):+.6f} |"
        )
    (output_root / "phasek_barycentric_gate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_root", default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix")
    parser.add_argument("--dataset_root", default="/data/peilincai/mesh_datasets/mipnerf360")
    parser.add_argument("--output_root", default="outputs/carnet/meshsplatopt/ecsr_phase_k/bary_delta_v2wide_s08_guarded")
    parser.add_argument("--evidence_root", default="outputs/carnet/meshsplatopt/ecsr_phase_k/surface_evidence_bary_v2wide")
    parser.add_argument("--scenes", default="garden")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--outdoor_images", default="images_4")
    parser.add_argument("--indoor_images", default="images_2")
    parser.add_argument("--skip_failed_views", action="store_true")
    parser.add_argument(
        "--cleanup_train_artifacts_after_scene",
        action="store_true",
        help=(
            "After a scene summary is written, remove candidate train-split render "
            "artifacts for that scene. Metrics, per-view JSONs, test renders, and "
            "checkpoint/audit files are preserved."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evidence_max_views", type=int, default=8)
    parser.add_argument("--evidence_view_stride", type=int, default=6)
    parser.add_argument("--evidence_view_offset", type=int, default=0)
    parser.add_argument("--evidence_high_error_quantile", type=float, default=0.70)
    parser.add_argument("--delta_top_k", type=int, default=4096)
    parser.add_argument("--delta_min_view_hits", type=int, default=2)
    parser.add_argument("--delta_min_consistency", type=float, default=0.85)
    parser.add_argument("--delta_min_pixel_count", type=float, default=6.0)
    parser.add_argument("--delta_max_samples_per_face_view", type=int, default=64)
    parser.add_argument("--delta_max_total_samples", type=int, default=300000)
    parser.add_argument("--delta_high_error_quantile", type=float, default=0.70)
    parser.add_argument(
        "--delta_face_score_weight_power",
        type=float,
        default=0.0,
        help=(
            "Facelocal-only train residual saliency weighting. When >0, the fitter "
            "upweights samples from high-score proposal faces without reading test data."
        ),
    )
    parser.add_argument("--delta_face_score_weight_max", type=float, default=4.0)
    parser.add_argument(
        "--delta_facelocal_region_carrier_json",
        default="",
        help="Facelocal-only render-visible carrier JSON template used for per-view core/context sample weighting.",
    )
    parser.add_argument("--delta_region_core_weight", type=float, default=1.0)
    parser.add_argument("--delta_region_context_weight", type=float, default=1.0)
    parser.add_argument("--delta_region_outside_weight", type=float, default=1.0)
    parser.add_argument("--delta_region_boundary_px", type=int, default=0)
    parser.add_argument("--delta_strength", type=float, default=0.08)
    parser.add_argument("--delta_max_abs_rgb", type=float, default=0.008)
    parser.add_argument("--delta_operator", choices=("dc", "sh1", "facelocal_sh1", "subdivision"), default="dc")
    parser.add_argument(
        "--delta_sh_degree",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="For face-local SH deltas, use a higher residual SH degree while preserving degree-1 defaults.",
    )
    parser.add_argument(
        "--delta_coefficient_lowpass_mode",
        choices=("none", "dc_only", "sh_scale"),
        default="none",
        help="Facelocal-only low-frequency projection of fitted residual coefficients before train-val gating.",
    )
    parser.add_argument("--delta_coefficient_lowpass_sh_scale", type=float, default=1.0)
    parser.add_argument("--candidate_label", default="bary_delta_v2wide_s08")
    parser.add_argument("--delta_max_abs_sh_coeff", type=float, default=0.0)
    parser.add_argument(
        "--delta_uniform_barycentric",
        action="store_true",
        help="For SH1 deltas, use equal face-vertex weights so a broader train-evidence support list can be used without barycentric rerendering.",
    )
    parser.add_argument("--delta_lambda_mag", type=float, default=0.03)
    parser.add_argument("--delta_lambda_sh1_mag", type=float, default=0.06)
    parser.add_argument("--delta_lambda_smooth", type=float, default=0.10)
    parser.add_argument(
        "--delta_direction_luma_safety_weight",
        type=float,
        default=0.0,
        help="Facelocal-only train residual-direction luma safety loss weight.",
    )
    parser.add_argument(
        "--delta_direction_cosine_weight",
        type=float,
        default=0.0,
        help="Facelocal-only train residual-direction cosine alignment loss weight.",
    )
    parser.add_argument("--delta_direction_cosine_margin", type=float, default=0.0)
    parser.add_argument("--delta_render_region_objective", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--delta_render_region_core_weight", type=float, default=1.0)
    parser.add_argument("--delta_render_region_context_weight", type=float, default=0.25)
    parser.add_argument("--delta_render_region_outside_penalty", type=float, default=0.0)
    parser.add_argument("--delta_render_region_tail_cvar_weight", type=float, default=0.0)
    parser.add_argument("--delta_render_region_tail_fraction", type=float, default=0.25)
    parser.add_argument("--delta_render_region_min_view_samples", type=int, default=16)
    parser.add_argument("--delta_steps", type=int, default=800)
    parser.add_argument("--delta_shared_residual_field", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--delta_shared_residual_field_anchors", type=int, default=16)
    parser.add_argument("--delta_shared_residual_field_sigma", type=float, default=0.0)
    parser.add_argument("--delta_shared_residual_field_lr", type=float, default=0.0)
    parser.add_argument("--delta_shared_residual_field_weight_l2", type=float, default=1.0e-4)
    parser.add_argument("--delta_shared_residual_field_view_hinge_weight", type=float, default=0.0)
    parser.add_argument("--delta_shared_residual_field_view_hinge_min_samples", type=int, default=16)
    parser.add_argument("--delta_shared_residual_field_duplicate_smooth_weight", type=float, default=0.0)
    parser.add_argument("--delta_min_policy_val_relative_gain", type=float, default=0.02)
    parser.add_argument("--delta_min_policy_val_samples", type=int, default=512)
    parser.add_argument("--delta_min_policy_val_unique_faces", type=int, default=16)
    parser.add_argument("--delta_policy_val_filter_faces", action="store_true")
    parser.add_argument("--delta_policy_val_face_min_samples", type=int, default=8)
    parser.add_argument("--delta_policy_val_face_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--delta_policy_val_face_max_keep", type=int, default=0)
    parser.add_argument("--delta_candidate_cluster_json", default="")
    parser.add_argument("--delta_candidate_cluster_csv", default="")
    parser.add_argument(
        "--delta_cluster_operator_types",
        default="certificate_cluster_contraction_candidate,surface_attached_attribute_recovery_candidate",
    )
    parser.add_argument("--delta_max_clusters", type=int, default=0)
    parser.add_argument("--delta_cluster_min_redundancy_score", type=float, default=-1.0e30)
    parser.add_argument("--delta_cluster_expand_with_top_residual_faces", action="store_true")
    parser.add_argument("--delta_cluster_expand_target_faces", type=int, default=0)
    parser.add_argument(
        "--delta_validation_shrink_mode",
        choices=("none", "global", "face", "global_gain", "face_gain"),
        default="none",
        help="For face-local SH1 deltas, calibrate residual amplitude on train-only policy-val samples.",
    )
    parser.add_argument("--delta_validation_gain_max_scale", type=float, default=1.0)
    parser.add_argument("--delta_validation_shrink_min_samples", type=int, default=8)
    parser.add_argument("--delta_crossfold_gain_certificate_folds", type=int, default=0)
    parser.add_argument("--delta_crossfold_min_passing_folds", type=int, default=0)
    parser.add_argument("--delta_crossfold_min_fold_relative_gain", type=float, default=0.0)
    parser.add_argument("--delta_crossfold_min_fold_samples", type=int, default=4)
    parser.add_argument("--delta_patch_cert_rings", type=int, default=0)
    parser.add_argument("--delta_patch_cert_max_faces_per_seed", type=int, default=8)
    parser.add_argument("--delta_patch_cert_min_direction_cosine", type=float, default=0.90)
    parser.add_argument("--delta_patch_cert_min_neighbor_policy_val_samples", type=int, default=4)
    parser.add_argument("--delta_patch_cert_min_neighbor_policy_val_relative_gain", type=float, default=-0.02)
    parser.add_argument("--delta_patch_cert_min_policy_val_samples", type=int, default=16)
    parser.add_argument("--delta_patch_cert_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--delta_patch_cert_neighbor_mode", choices=("topology", "centroid", "both"), default="topology")
    parser.add_argument("--delta_patch_cert_centroid_candidates_per_seed", type=int, default=64)
    parser.add_argument("--delta_patch_cert_seed_rescue", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--delta_patch_cert_seed_rescue_min_candidates", type=int, default=1)
    parser.add_argument("--delta_patch_cert_seed_rescue_max_seeds", type=int, default=16)
    parser.add_argument("--delta_patch_cert_seed_rescue_min_aux_witnesses", type=int, default=1)
    parser.add_argument("--delta_patch_cert_crossfold_folds", type=int, default=0)
    parser.add_argument("--delta_patch_cert_crossfold_min_passing_folds", type=int, default=0)
    parser.add_argument("--delta_patch_cert_crossfold_min_fold_relative_gain", type=float, default=0.0)
    parser.add_argument("--delta_patch_cert_crossfold_min_fold_samples", type=int, default=4)
    parser.add_argument("--delta_patch_cert_cluster_basis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--delta_patch_cert_cluster_basis_mode",
        choices=("shared", "scaled", "rank2", "chart_linear", "chart_quad", "field_linear", "field_quad"),
        default="shared",
    )
    parser.add_argument("--delta_patch_cert_cluster_basis_steps", type=int, default=240)
    parser.add_argument("--delta_patch_cert_cluster_basis_rank", type=int, default=2)
    parser.add_argument("--delta_patch_cert_cluster_basis_lr", type=float, default=0.025)
    parser.add_argument("--delta_patch_cert_cluster_basis_min_samples", type=int, default=32)
    parser.add_argument("--delta_patch_cert_cluster_basis_max_scale", type=float, default=2.0)
    parser.add_argument("--delta_patch_cert_cluster_basis_max_fit_mse_regression", type=float, default=0.02)
    parser.add_argument("--delta_patch_cert_cluster_basis_init", choices=("mean", "zero"), default="mean")
    parser.add_argument("--delta_patch_cert_cluster_basis_view_hinge_weight", type=float, default=0.0)
    parser.add_argument("--delta_patch_cert_cluster_basis_view_hinge_min_samples", type=int, default=16)
    parser.add_argument("--delta_patch_cert_cluster_basis_geometry_smooth_weight", type=float, default=0.0)
    parser.add_argument("--delta_patch_cert_carrier_holdout_selector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--delta_patch_cert_carrier_holdout_groups", type=int, default=4)
    parser.add_argument(
        "--delta_patch_cert_carrier_holdout_grouping",
        choices=("view", "sample_balanced"),
        default="view",
    )
    parser.add_argument("--delta_patch_cert_carrier_holdout_disjoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--delta_patch_cert_carrier_holdout_min_passing_groups", type=int, default=3)
    parser.add_argument("--delta_patch_cert_carrier_holdout_min_group_relative_gain", type=float, default=0.0)
    parser.add_argument("--delta_patch_cert_carrier_holdout_min_group_samples", type=int, default=4)
    parser.add_argument("--delta_patch_cert_carrier_holdout_max_mse_regression", type=float, default=0.0)
    parser.add_argument("--delta_patch_cert_carrier_holdout_cvar_fraction", type=float, default=0.25)
    parser.add_argument("--delta_patch_cert_carrier_holdout_cvar_weight", type=float, default=1.0)
    parser.add_argument("--delta_patch_cert_carrier_holdout_max_carriers", type=int, default=0)
    parser.add_argument("--delta_patch_cert_carrier_holdout_auto_prefix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--delta_patch_cert_carrier_holdout_auto_prefix_min_faces", type=int, default=0)
    parser.add_argument("--delta_patch_cert_carrier_holdout_auto_prefix_face_bonus", type=float, default=0.0)
    parser.add_argument(
        "--delta_patch_cert_carrier_holdout_auto_prefix_positive_tail_safe",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--delta_patch_cert_neighbor_crossfold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--delta_patch_cert_shrink", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--delta_strict_patchcert_carrier", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--delta_facelocal_candidate_plan_out",
        default="",
        help="Facelocal-only candidate plan output path template. Use {scene} for scene-specific paths.",
    )
    parser.add_argument(
        "--delta_facelocal_materialize_plan_in",
        default="",
        help="Facelocal-only candidate plan input path template for render-calibrated subset materialization.",
    )
    parser.add_argument("--delta_facelocal_materialize_plan_limit", type=int, default=0)
    parser.add_argument("--delta_facelocal_materialize_plan_scale", type=float, default=1.0)
    parser.add_argument(
        "--delta_facelocal_materialize_plan_face_ids",
        default="",
        help="Optional comma-separated face ids to materialize from the facelocal candidate plan.",
    )
    parser.add_argument(
        "--delta_facelocal_materialize_plan_alpha_json",
        default="",
        help="Optional per-scene JSON template with face_id -> alpha multipliers for facelocal plan materialization.",
    )
    parser.add_argument(
        "--delta_facelocal_materialize_plan_render_trust_json",
        default="",
        help="Optional per-scene render-trust certificate template for strict non-unit facelocal plan scale replay.",
    )
    parser.add_argument(
        "--delta_facelocal_materialize_allow_uncertified_plan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow legacy facelocal plan replay that lacks strict PatchCert carrier metadata.",
    )
    parser.add_argument("--delta_max_faces_to_apply", type=int, default=2048)
    parser.add_argument("--delta_min_face_policy_val_relative_gain", type=float, default=0.0)
    parser.add_argument("--delta_min_face_policy_val_samples", type=int, default=8)
    parser.add_argument(
        "--delta_policy_val_offsets",
        default="",
        help=(
            "For subdivision deltas, require per-face residual gains on these "
            "comma-separated train-only policy-validation offsets before applying."
        ),
    )
    parser.add_argument(
        "--delta_min_policy_val_offsets",
        type=int,
        default=0,
        help="For subdivision deltas, minimum passing offsets. 0 requires all requested offsets.",
    )
    parser.add_argument(
        "--delta_min_policy_val_offset_fraction",
        type=float,
        default=1.0,
        help="For subdivision deltas, minimum passing fraction across requested policy offsets.",
    )
    parser.add_argument(
        "--delta_subdivision_lambda_ridge",
        type=float,
        default=2e-2,
        help="Ridge penalty for the train-only local subdivision residual solve.",
    )
    parser.add_argument(
        "--delta_subdivision_feature_mode",
        choices=("dc", "sh1"),
        default="dc",
        help="Subdivision midpoint residual feature basis. Default dc preserves historical behavior.",
    )
    parser.add_argument(
        "--delta_subdivision_materialize_mode",
        choices=("subdivision", "vertex_delta"),
        default="subdivision",
        help="How the subdivision residual operator writes accepted candidates into the checkpoint.",
    )
    parser.add_argument(
        "--delta_subdivision_max_abs_sh_coeff",
        type=float,
        default=0.0,
        help="SH1 coefficient bound for subdivision feature_mode=sh1; 0 derives from delta_max_abs_rgb.",
    )
    parser.add_argument("--delta_subdivision_luma_preserve", action="store_true")
    parser.add_argument("--delta_subdivision_min_luma_relative_gain", type=float, default=0.0)
    parser.add_argument("--delta_subdivision_max_mean_luma_shift", type=float, default=0.0)
    parser.add_argument("--delta_subdivision_luma_shrink_grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--delta_subdivision_luma_shrink_selection", choices=("min", "max"), default="min")
    parser.add_argument("--delta_subdivision_anchor_support", action="store_true")
    parser.add_argument("--delta_subdivision_anchor_max_error_quantile", type=float, default=0.35)
    parser.add_argument("--delta_subdivision_anchor_samples_per_face_view", type=int, default=0)
    parser.add_argument("--delta_subdivision_anchor_weight", type=float, default=0.25)
    parser.add_argument("--delta_subdivision_candidate_plan_out", default="")
    parser.add_argument("--delta_subdivision_materialize_plan_in", default="")
    parser.add_argument("--delta_subdivision_materialize_plan_limit", type=int, default=0)
    parser.add_argument("--delta_subdivision_vertex_delta_min_incident_support_fraction", type=float, default=0.0)
    parser.add_argument("--delta_subdivision_vertex_delta_max_incident_faces", type=int, default=0)
    parser.add_argument("--delta_subdivision_min_effective_mean_relative_gain", type=float, default=-1.0e30)
    parser.add_argument("--delta_subdivision_min_effective_min_relative_gain", type=float, default=-1.0e30)
    parser.add_argument("--delta_subdivision_min_effective_delta_abs_mean", type=float, default=0.0)
    parser.add_argument("--delta_subdivision_min_materialized_attribute_delta", type=float, default=1.0e-9)
    parser.add_argument("--delta_subdivision_allow_no_effect_accept", action="store_true")
    parser.add_argument("--delta_subdivision_structure_preserve", action="store_true")
    parser.add_argument("--delta_subdivision_structure_weight_strength", type=float, default=2.0)
    parser.add_argument("--delta_subdivision_min_structure_relative_gain", type=float, default=0.0)
    parser.add_argument("--delta_subdivision_max_structure_mean_luma_shift", type=float, default=0.0)
    parser.add_argument("--delta_subdivision_structure_shrink_grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--delta_subdivision_structure_shrink_selection", choices=("min", "max"), default="max")
    parser.add_argument("--delta_subdivision_min_fit_samples", type=int, default=24)
    parser.add_argument("--delta_subdivision_min_val_samples", type=int, default=12)
    parser.add_argument(
        "--delta_subdivision_min_policy_val_relative_gain",
        type=float,
        default=None,
        help=(
            "Subdivision-only per-face policy-val relative gain. If omitted, "
            "falls back to --delta_min_face_policy_val_relative_gain."
        ),
    )
    parser.add_argument(
        "--delta_sh1_face_policy",
        action="store_true",
        help="For shared-vertex SH1 deltas, only write faces that pass fixed policy-val per-face certificates.",
    )
    parser.add_argument(
        "--delta_min_face_view_consensus",
        type=float,
        default=0.0,
        help=(
            "For SH1-family deltas, require this fraction of policy-val train views "
            "to agree with a face residual direction before applying the face update."
        ),
    )
    parser.add_argument("--delta_min_face_consensus_views", type=int, default=2)
    parser.add_argument("--delta_min_face_consensus_view_samples", type=int, default=4)
    parser.add_argument("--delta_face_consensus_min_cosine", type=float, default=0.0)
    parser.add_argument(
        "--delta_min_face_gain_certificate_views",
        type=int,
        default=0,
        help=(
            "For face-local SH1 deltas, require each accepted face to have predicted "
            "residual MSE gain on at least this many policy-val train views. 0 disables it."
        ),
    )
    parser.add_argument("--delta_min_face_gain_certificate_relative_gain", type=float, default=0.0)
    parser.add_argument("--delta_min_face_gain_certificate_view_samples", type=int, default=4)
    parser.add_argument("--delta_min_face_gain_certificate_fraction", type=float, default=0.0)
    parser.add_argument(
        "--delta_min_face_prediction_safety_fraction",
        type=float,
        default=0.0,
        help=(
            "For face-local SH1 deltas, require a train-only policy-val prediction "
            "safety fraction before a face can be materialized. 0 disables it."
        ),
    )
    parser.add_argument("--delta_min_face_prediction_safety_samples", type=int, default=8)
    parser.add_argument("--delta_face_prediction_safety_min_cosine", type=float, default=0.0)
    parser.add_argument(
        "--phasej_test_method",
        default=PHASEJ_METHOD,
        help=(
            "Reference Phase-J test method used for report-only comparison. "
            "Use a unique name to force a fresh same-run replay instead of "
            "reusing stale results from earlier experiments."
        ),
    )
    parser.add_argument("--phasej_trainval_method", default="ours_26000_phasej_trainval_gate")
    parser.add_argument("--candidate_base_method", default="ours_26000_bary_delta_v2wide_s08_base")
    parser.add_argument("--candidate_test_method", default="ours_26000_bary_delta_v2wide_s08_phasej_ela")
    parser.add_argument("--candidate_trainval_method", default="ours_26000_bary_delta_v2wide_s08_phasej_trainval_gate")
    parser.add_argument("--policy_holdout_fraction", type=float, default=0.25)
    parser.add_argument("--policy_holdout_offset", type=int, default=0)
    parser.add_argument("--support_policy_fit_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calib_sampler", choices=("stride_first", "uniform"), default="uniform")
    parser.add_argument("--calib_max_views", type=int, default=32)
    parser.add_argument("--calib_stride", type=int, default=1)
    parser.add_argument(
        "--ela_policy_source",
        choices=("fixed_phasej", "per_model_auto"),
        default="fixed_phasej",
        help=(
            "fixed_phasej replays the selected Phase-J ELA report on both arms. "
            "per_model_auto runs the same train-only auto-policy ELA search for "
            "the Phase-J baseline and candidate checkpoint before the gate."
        ),
    )
    parser.add_argument("--ela_policy_objective", choices=("psnr", "balanced"), default="balanced")
    parser.add_argument("--ela_calib_lpips", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ela_alpha_grid", default="0,0.125,0.25,0.5,0.75,1.0")
    parser.add_argument("--ela_policy_modes", default="residual,color")
    parser.add_argument("--ela_policy_k_values", default="4,8")
    parser.add_argument("--ela_policy_depth_rel_values", default="0.06,0.12")
    parser.add_argument("--ela_policy_residual_clip_values", default="0.25")
    parser.add_argument("--ela_policy_direction_weight_values", default="")
    parser.add_argument("--ela_edge_gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ela_edge_gate_min", type=float, default=0.0)
    parser.add_argument(
        "--ela_policy_edge_gate_quantiles",
        default="",
        help="Optional comma-separated edge-gate quantiles for per-model auto ELA.",
    )
    parser.add_argument(
        "--ela_policy_edge_gate_dilates",
        default="",
        help="Optional comma-separated edge-gate dilation values for per-model auto ELA.",
    )
    parser.add_argument("--alpha_feature_mode", choices=("confidence_magnitude", "confidence_magnitude_edge"), default="confidence_magnitude_edge")
    parser.add_argument("--alpha_default", type=float, default=0.0)
    parser.add_argument("--gate_min_psnr_gain", type=float, default=0.0)
    parser.add_argument("--gate_max_ssim_regression", type=float, default=5e-5)
    parser.add_argument("--gate_max_lpips_regression", type=float, default=1.5e-4)
    parser.add_argument("--gate_min_balanced_delta", type=float, default=0.0)
    parser.add_argument("--gate_ssim_weight", type=float, default=20.0)
    parser.add_argument("--gate_lpips_weight", type=float, default=20.0)
    parser.add_argument("--gate_tail_require_available", action="store_true")
    parser.add_argument("--gate_tail_cvar_fraction", type=float, default=0.20)
    parser.add_argument("--gate_tail_max_balanced_negative_fraction", type=float, default=1.0)
    parser.add_argument("--gate_tail_min_balanced_cvar_delta", type=float, default=-1.0e30)
    parser.add_argument("--gate_tail_max_lpips_positive_fraction", type=float, default=1.0)
    parser.add_argument("--gate_tail_max_worst_lpips_regression", type=float, default=1.0e30)
    parser.add_argument("--gate_stratified_group_count", type=int, default=4)
    parser.add_argument("--gate_compact_enable", action="store_true")
    parser.add_argument("--gate_compact_require", action="store_true")
    parser.add_argument("--gate_compact_max_faces", type=int, default=160)
    parser.add_argument("--gate_compact_max_vertices", type=int, default=512)
    parser.add_argument("--gate_compact_max_face_ratio", type=float, default=1.5e-5)
    parser.add_argument("--gate_compact_min_psnr_gain", type=float, default=2.0e-5)
    parser.add_argument("--gate_compact_max_ssim_regression", type=float, default=1.5e-5)
    parser.add_argument("--gate_compact_max_lpips_regression", type=float, default=5.0e-6)
    parser.add_argument("--gate_compact_max_balanced_negative_fraction", type=float, default=0.70)
    parser.add_argument("--gate_compact_min_balanced_cvar_delta", type=float, default=-0.0012)
    parser.add_argument("--gate_compact_max_lpips_positive_fraction", type=float, default=0.60)
    parser.add_argument("--gate_compact_max_worst_lpips_regression", type=float, default=5.0e-5)
    parser.add_argument("--gate_compact_min_stratified_psnr_delta", type=float, default=-1.0e-5)
    parser.add_argument("--gate_compact_max_stratified_ssim_regression", type=float, default=2.0e-5)
    parser.add_argument("--gate_compact_max_stratified_lpips_regression", type=float, default=1.5e-5)
    parser.add_argument("--train_render_region_gate_enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--train_render_region_carrier_json",
        default="",
        help=(
            "Scene-templated train carrier JSON used to score actual train-rendered "
            "regions before the Phase-K decision. Defaults to --delta_facelocal_region_carrier_json when empty."
        ),
    )
    parser.add_argument("--train_render_region_max_regions", type=int, default=64)
    parser.add_argument("--train_render_region_min_pixels", type=int, default=128)
    parser.add_argument("--train_render_region_min_crop_size", type=int, default=32)
    parser.add_argument("--train_render_region_context_pad", type=int, default=16)
    parser.add_argument("--train_render_region_tail_fraction", type=float, default=0.25)
    parser.add_argument(
        "--train_render_region_skip_lpips",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip crop-level LPIPS in the train-render-region gate. This keeps the "
            "gate fast for dense carrier scenes while the standard full-frame train/test "
            "LPIPS metrics remain unchanged."
        ),
    )
    parser.add_argument(
        "--train_render_region_eval_source",
        choices=("post_ela_trainval", "raw_base"),
        default="post_ela_trainval",
        help=(
            "Render directories compared by the train-only region gate. raw_base compares "
            "the compact Phase-F render to the raw Phase-S candidate before ELA so zero "
            "or invisible edits are detected before the standard train-val gate."
        ),
    )
    parser.add_argument("--train_render_region_min_regions", type=int, default=4)
    parser.add_argument("--train_render_region_min_changed_regions", type=int, default=1)
    parser.add_argument("--train_render_region_min_changed_fraction", type=float, default=0.0)
    parser.add_argument("--train_render_region_min_core_balanced_delta", type=float, default=-1.0e30)
    parser.add_argument("--train_render_region_min_core_psnr_delta", type=float, default=-1.0e30)
    parser.add_argument("--train_render_region_min_tail_cvar_delta", type=float, default=-1.0e30)
    parser.add_argument("--train_render_region_max_context_mse_regression", type=float, default=1.0e30)
    parser.add_argument("--train_render_region_max_negative_fraction", type=float, default=1.0)
    parser.add_argument("--wandb_project", default="mesh-splatting-ecsr")
    parser.add_argument("--wandb_group", default="phasek_barycentric_multiscene")
    parser.add_argument("--wandb_name", default="phasek_barycentric")
    args = parser.parse_args()
    if float(args.delta_patch_cert_cluster_basis_max_scale) <= 0.0:
        parser.error("--delta_patch_cert_cluster_basis_max_scale must be > 0")
    if float(args.delta_patch_cert_cluster_basis_lr) <= 0.0:
        parser.error("--delta_patch_cert_cluster_basis_lr must be > 0")
    if int(args.delta_patch_cert_cluster_basis_steps) < 0:
        parser.error("--delta_patch_cert_cluster_basis_steps must be >= 0")
    if int(args.delta_patch_cert_cluster_basis_rank) < 1:
        parser.error("--delta_patch_cert_cluster_basis_rank must be >= 1")
    if int(args.delta_patch_cert_cluster_basis_min_samples) <= 0:
        parser.error("--delta_patch_cert_cluster_basis_min_samples must be > 0")
    if int(args.delta_patch_cert_seed_rescue_min_candidates) < 0:
        parser.error("--delta_patch_cert_seed_rescue_min_candidates must be >= 0")
    if int(args.delta_patch_cert_seed_rescue_max_seeds) < 0:
        parser.error("--delta_patch_cert_seed_rescue_max_seeds must be >= 0")
    if int(args.delta_patch_cert_seed_rescue_min_aux_witnesses) < 0:
        parser.error("--delta_patch_cert_seed_rescue_min_aux_witnesses must be >= 0")
    if not math.isfinite(float(args.delta_face_score_weight_power)) or float(args.delta_face_score_weight_power) < 0.0:
        parser.error("--delta_face_score_weight_power must be finite and >= 0")
    if not math.isfinite(float(args.delta_face_score_weight_max)) or float(args.delta_face_score_weight_max) < 1.0:
        parser.error("--delta_face_score_weight_max must be finite and >= 1")
    for name in ("delta_region_core_weight", "delta_region_context_weight", "delta_region_outside_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and >= 0")
    if int(args.delta_region_boundary_px) < 0:
        parser.error("--delta_region_boundary_px must be >= 0")
    if (
        not math.isfinite(float(args.delta_coefficient_lowpass_sh_scale))
        or float(args.delta_coefficient_lowpass_sh_scale) < 0.0
    ):
        parser.error("--delta_coefficient_lowpass_sh_scale must be finite and >= 0")
    if (
        str(args.delta_coefficient_lowpass_mode) == "sh_scale"
        and float(args.delta_coefficient_lowpass_sh_scale) > 1.0
    ):
        parser.error("--delta_coefficient_lowpass_sh_scale must be <= 1 for sh_scale lowpass")
    if str(args.ela_policy_source) == "per_model_auto" and float(args.policy_holdout_fraction) <= 0.0:
        parser.error("--ela_policy_source per_model_auto requires --policy_holdout_fraction > 0")
    for name in ("delta_direction_luma_safety_weight", "delta_direction_cosine_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and >= 0")
    for name in (
        "delta_render_region_core_weight",
        "delta_render_region_context_weight",
        "delta_render_region_outside_penalty",
        "delta_render_region_tail_cvar_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and >= 0")
    if (
        not math.isfinite(float(args.delta_render_region_tail_fraction))
        or float(args.delta_render_region_tail_fraction) <= 0.0
        or float(args.delta_render_region_tail_fraction) > 1.0
    ):
        parser.error("--delta_render_region_tail_fraction must be in (0, 1]")
    if int(args.delta_render_region_min_view_samples) < 0:
        parser.error("--delta_render_region_min_view_samples must be >= 0")
    if int(args.train_render_region_max_regions) <= 0:
        parser.error("--train_render_region_max_regions must be > 0")
    if int(args.train_render_region_min_pixels) <= 0:
        parser.error("--train_render_region_min_pixels must be > 0")
    if int(args.train_render_region_min_crop_size) <= 0:
        parser.error("--train_render_region_min_crop_size must be > 0")
    if int(args.train_render_region_context_pad) < 0:
        parser.error("--train_render_region_context_pad must be >= 0")
    if (
        not math.isfinite(float(args.train_render_region_tail_fraction))
        or float(args.train_render_region_tail_fraction) <= 0.0
        or float(args.train_render_region_tail_fraction) > 1.0
    ):
        parser.error("--train_render_region_tail_fraction must be in (0, 1]")
    if int(args.train_render_region_min_regions) < 0:
        parser.error("--train_render_region_min_regions must be >= 0")
    if int(args.train_render_region_min_changed_regions) < 0:
        parser.error("--train_render_region_min_changed_regions must be >= 0")
    if (
        not math.isfinite(float(args.train_render_region_min_changed_fraction))
        or float(args.train_render_region_min_changed_fraction) < 0.0
        or float(args.train_render_region_min_changed_fraction) > 1.0
    ):
        parser.error("--train_render_region_min_changed_fraction must be in [0, 1]")
    for name in (
        "train_render_region_min_core_balanced_delta",
        "train_render_region_min_core_psnr_delta",
        "train_render_region_min_tail_cvar_delta",
        "train_render_region_max_context_mse_regression",
        "train_render_region_max_negative_fraction",
    ):
        if not math.isfinite(float(getattr(args, name))):
            parser.error(f"--{name} must be finite")
    if not math.isfinite(float(args.delta_direction_cosine_margin)):
        parser.error("--delta_direction_cosine_margin must be finite")
    if float(args.delta_direction_cosine_margin) < -1.0 or float(args.delta_direction_cosine_margin) > 1.0:
        parser.error("--delta_direction_cosine_margin must be in [-1, 1]")
    if (
        not math.isfinite(float(args.delta_min_face_prediction_safety_fraction))
        or float(args.delta_min_face_prediction_safety_fraction) < 0.0
        or float(args.delta_min_face_prediction_safety_fraction) > 1.0
    ):
        parser.error("--delta_min_face_prediction_safety_fraction must be in [0, 1]")
    if int(args.delta_min_face_prediction_safety_samples) < 0:
        parser.error("--delta_min_face_prediction_safety_samples must be >= 0")
    if (
        not math.isfinite(float(args.delta_face_prediction_safety_min_cosine))
        or float(args.delta_face_prediction_safety_min_cosine) < -1.0
        or float(args.delta_face_prediction_safety_min_cosine) > 1.0
    ):
        parser.error("--delta_face_prediction_safety_min_cosine must be in [-1, 1]")
    if (
        not math.isfinite(float(args.delta_patch_cert_cluster_basis_max_fit_mse_regression))
        or float(args.delta_patch_cert_cluster_basis_max_fit_mse_regression) < 0.0
    ):
        parser.error("--delta_patch_cert_cluster_basis_max_fit_mse_regression must be finite and >= 0")
    if int(args.delta_patch_cert_carrier_holdout_auto_prefix_min_faces) < 0:
        parser.error("--delta_patch_cert_carrier_holdout_auto_prefix_min_faces must be >= 0")
    if (
        not math.isfinite(float(args.delta_patch_cert_carrier_holdout_auto_prefix_face_bonus))
        or float(args.delta_patch_cert_carrier_holdout_auto_prefix_face_bonus) < 0.0
    ):
        parser.error("--delta_patch_cert_carrier_holdout_auto_prefix_face_bonus must be finite and >= 0")
    if bool(args.delta_patch_cert_carrier_holdout_disjoint) and str(args.delta_patch_cert_carrier_holdout_grouping) != "sample_balanced":
        parser.error(
            "--delta_patch_cert_carrier_holdout_disjoint currently requires "
            "--delta_patch_cert_carrier_holdout_grouping sample_balanced"
        )
    scenes = [scene.strip() for scene in str(args.scenes).replace(" ", ",").split(",") if scene.strip()]
    rows = [run_scene(args, scene) for scene in scenes]
    _write_aggregate(args, rows)
    print(json.dumps({"rows": len(rows), "output_root": str(ROOT / args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
