#!/usr/bin/env python3
"""Evaluate the v169 current-carrier teacher projection upper bound.

This is a read-only diagnostic.  It fits the existing face/UV atlas carrier on
train-fit teacher residual evidence, evaluates it on train-policy-val views
against GT, and writes only JSON/Markdown reports.  It does not apply to target
or test evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ecsr_apply_surface_residual_region_texture_adapter as adapter  # noqa: E402


def parse_csv_floats(text: str) -> list[float]:
    values: list[float] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return sorted(set(values))


def parse_csv_ints(text: str) -> list[int]:
    values: list[int] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit_evidence_dir", type=Path, required=True)
    parser.add_argument("--region_carrier_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    parser.add_argument(
        "--teacher_residual_target_mode",
        choices=("raw_rgb", "luma_only", "edge_luma_mix"),
        default="raw_rgb",
    )
    parser.add_argument("--teacher_residual_target_luma_mix", type=float, default=0.75)
    parser.add_argument("--teacher_residual_target_edge_boost", type=float, default=0.25)
    parser.add_argument("--texture_sizes", default="8,16")
    parser.add_argument("--alpha_grid", default="0,0.03125,0.0625,0.125,0.25,0.5")
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--max_carriers", type=int, default=64)
    parser.add_argument("--max_faces_per_carrier", type=int, default=128)
    parser.add_argument("--max_faces", type=int, default=4096)
    parser.add_argument("--max_samples_per_view", type=int, default=120000)
    parser.add_argument("--min_l1", type=float, default=0.0)
    parser.add_argument("--min_alpha", type=float, default=0.03)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=0.12)
    parser.add_argument("--min_atlas_bin_count", type=int, default=0)
    parser.add_argument("--min_atlas_face_samples", type=int, default=0)
    parser.add_argument("--max_atlas_bin_rgb_variance", type=float, default=-1.0)
    parser.add_argument("--min_atlas_bin_sign_consistency", type=float, default=0.0)
    parser.add_argument("--policy_val_ssim_max_size", type=int, default=512)
    parser.add_argument("--policy_val_l1_max_size", type=int, default=512)
    parser.add_argument("--policy_val_lpips_max_size", type=int, default=384)
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument(
        "--teacher_distilled_basis_mode",
        choices=(
            "none",
            "face_uv_normal_camera_ridge",
            "face_uv_patch_mixture_ridge",
            "surface_feature_rff_ridge",
            "low_rank_view_texture_k4",
            "low_rank_view_texture",
            "low_rank_view_texture_rich_k4",
            "low_rank_view_texture_rich",
        ),
        default="none",
    )
    parser.add_argument("--teacher_distilled_basis_min_face_samples", type=int, default=1024)
    parser.add_argument("--teacher_distilled_basis_ridge", type=float, default=1.0e-2)
    parser.add_argument("--teacher_distilled_basis_apply_mode", choices=("replace_supported", "blend", "fill_empty_only"), default="blend")
    parser.add_argument("--teacher_distilled_basis_blend", type=float, default=0.5)
    parser.add_argument("--teacher_distilled_low_rank_texture_rank", type=int, default=4)
    parser.add_argument(
        "--teacher_distilled_low_rank_texture_ranks",
        default="",
        help="Comma-separated rank ladder for low-rank teacher residual texture diagnostics.",
    )
    parser.add_argument("--enable_adaptive_low_support_teacher_basis", action="store_true")
    parser.add_argument("--adaptive_teacher_basis_min_face_samples_floor", type=int, default=128)
    parser.add_argument("--adaptive_teacher_basis_support_quantile", type=float, default=0.25)
    parser.add_argument("--adaptive_teacher_basis_low_support_ridge_scale", type=float, default=0.5)
    parser.add_argument(
        "--enable_full_image_psnr_rescan",
        action="store_true",
        help=(
            "Compute full-image PSNR by replaying atlas prediction for every policy-val view/alpha. "
            "Default uses the much faster policy-val residual-sample PSNR proxy."
        ),
    )
    parser.add_argument("--phasej_flowers_psnr", type=float, default=20.304358)
    parser.add_argument("--phasej_flowers_ssim", type=float, default=0.557770)
    parser.add_argument("--phasej_flowers_lpips", type=float, default=0.329222)
    return parser.parse_args()


def psnr_from_mse(mse: float) -> float | None:
    if not math.isfinite(float(mse)) or float(mse) <= 0.0:
        return None
    return float(-10.0 * math.log10(float(mse)))


def image_psnr_rows(
    val_views: list[Path],
    atlas: dict[int, adapter.FaceAtlas],
    alpha_grid: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows_by_alpha: dict[str, list[dict[str, Any]]] = {str(float(alpha)): [] for alpha in alpha_grid}
    summary_by_alpha: dict[str, dict[str, Any]] = {}
    for path in val_views:
        with np.load(path, allow_pickle=False) as z:
            if "rgb_render" not in z or "rgb_gt" not in z:
                continue
            render = np.asarray(z["rgb_render"], dtype=np.float32)
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)
            before_mse = float(np.mean((render - gt) ** 2))
            before_psnr = psnr_from_mse(before_mse)
            for alpha in alpha_grid:
                pred, _valid = adapter.predict_delta_for_npz(
                    z,
                    atlas,
                    float(alpha),
                    float(args.min_alpha),
                    min_atlas_bin_count=int(args.min_atlas_bin_count),
                    min_atlas_face_samples=int(args.min_atlas_face_samples),
                    max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
                    min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
                    atlas_confidence_mode="none",
                    atlas_confidence_count_scale=0.0,
                    atlas_confidence_empty_bin=1.0,
                    atlas_confidence_variance_scale=-1.0,
                    atlas_confidence_sign_power=0.0,
                    atlas_confidence_face_sample_scale=0.0,
                    min_atlas_confidence=0.0,
                )
                pred = adapter.clip_delta_rgb(pred, float(args.max_abs_delta_rgb))
                adapted = np.clip(render + pred, 0.0, 1.0)
                after_mse = float(np.mean((adapted - gt) ** 2))
                after_psnr = psnr_from_mse(after_mse)
                rows_by_alpha[str(float(alpha))].append(
                    {
                        "view": path.stem,
                        "psnr_before": before_psnr,
                        "psnr_after": after_psnr,
                        "psnr_gain": (
                            float(after_psnr - before_psnr)
                            if after_psnr is not None and before_psnr is not None
                            else None
                        ),
                        "mse_before": before_mse,
                        "mse_after": after_mse,
                    }
                )
    for alpha in alpha_grid:
        key = str(float(alpha))
        values = [float(row["psnr_gain"]) for row in rows_by_alpha[key] if row.get("psnr_gain") is not None]
        before = [float(row["psnr_before"]) for row in rows_by_alpha[key] if row.get("psnr_before") is not None]
        after = [float(row["psnr_after"]) for row in rows_by_alpha[key] if row.get("psnr_after") is not None]
        arr = np.asarray(values, dtype=np.float64)
        if arr.size:
            sorted_arr = np.sort(arr)
            cvar_count = max(1, int(math.ceil(0.20 * float(sorted_arr.size))))
            summary_by_alpha[key] = {
                "view_count": int(arr.size),
                "psnr_before": float(np.mean(before)),
                "psnr_after": float(np.mean(after)),
                "psnr_gain": float(np.mean(arr)),
                "psnr_positive_view_fraction": float(np.mean(arr > 0.0)),
                "psnr_min_view_gain": float(np.min(arr)),
                "psnr_cvar20_view_gain": float(np.mean(sorted_arr[:cvar_count])),
            }
        else:
            summary_by_alpha[key] = {
                "view_count": 0,
                "psnr_before": None,
                "psnr_after": None,
                "psnr_gain": None,
                "psnr_positive_view_fraction": 0.0,
                "psnr_min_view_gain": None,
                "psnr_cvar20_view_gain": None,
            }
    return {"summary_by_alpha": summary_by_alpha, "per_view_by_alpha": rows_by_alpha}


def atlas_fit_kwargs(args: argparse.Namespace, texture_size: int) -> dict[str, Any]:
    return {
        "residual_rgb_key": str(args.residual_rgb_key),
        "residual_l1_key": str(args.residual_l1_key),
        "texture_size": int(texture_size),
        "policy_val_stride": int(args.policy_val_stride),
        "min_l1": float(args.min_l1),
        "min_alpha": float(args.min_alpha),
        "max_samples_per_view": int(args.max_samples_per_view),
        "fill_empty_with_face_mean": True,
        "atlas_empty_bin_fill_mode": "face_mean",
        "atlas_nearest_fill_max_steps": 32,
        "atlas_nearest_fill_decay": 0.92,
        "atlas_lowpass_passes": 0,
        "atlas_lowpass_neighbor_min_count": 1,
        "surface_multiscale_prior_mode": "none",
        "surface_multiscale_prior_block_sizes": [2, 4, 8],
        "surface_multiscale_prior_min_bin_samples": 8,
        "surface_multiscale_prior_count_tau": 32.0,
        "surface_multiscale_prior_blend": 0.0,
        "surface_multiscale_prior_gate_mode": "none",
        "surface_multiscale_prior_min_prior_weight": 0.0,
        "surface_multiscale_prior_min_direct_samples": 1,
        "surface_multiscale_prior_min_sign_consistency": 0.0,
        "surface_multiscale_prior_max_mean_variance": -1.0,
        "surface_multiscale_prior_min_cosine": 0.0,
        "view_conditioned_basis_mode": "none",
        "view_conditioned_basis_min_bin_samples": 16,
        "view_conditioned_basis_ridge": 1.0e-3,
        "view_conditioned_basis_ood_mode": "none",
        "view_conditioned_basis_ood_max_z": 2.5,
        "view_conditioned_basis_ood_min_std": 5.0e-2,
        "view_cluster_expert_count": 1,
        "view_cluster_feature_mode": "camera_center",
        "view_cluster_min_views": 2,
        "view_cluster_min_bin_samples": 4,
        "view_cluster_fallback_mode": "global",
        "teacher_residual_target_mode": str(args.teacher_residual_target_mode),
        "teacher_residual_target_luma_mix": float(args.teacher_residual_target_luma_mix),
        "teacher_residual_target_edge_boost": float(args.teacher_residual_target_edge_boost),
        "teacher_distilled_basis_mode": str(args.teacher_distilled_basis_mode),
        "teacher_distilled_basis_min_face_samples": int(args.teacher_distilled_basis_min_face_samples),
        "teacher_distilled_basis_ridge": float(args.teacher_distilled_basis_ridge),
        "teacher_distilled_basis_ood_max_z": 3.0,
        "teacher_distilled_basis_ood_min_std": 5.0e-2,
        "teacher_distilled_basis_apply_mode": str(args.teacher_distilled_basis_apply_mode),
        "teacher_distilled_basis_blend": float(args.teacher_distilled_basis_blend),
        "teacher_distilled_low_rank_texture_rank": int(args._current_teacher_low_rank_texture_rank),
        "enable_adaptive_low_support_teacher_basis": bool(args.enable_adaptive_low_support_teacher_basis),
        "adaptive_teacher_basis_min_face_samples_floor": int(args.adaptive_teacher_basis_min_face_samples_floor),
        "adaptive_teacher_basis_support_quantile": float(args.adaptive_teacher_basis_support_quantile),
        "adaptive_teacher_basis_low_support_ridge_scale": float(args.adaptive_teacher_basis_low_support_ridge_scale),
    }


def evaluate_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "residual_rgb_key": str(args.residual_rgb_key),
        "residual_l1_key": str(args.residual_l1_key),
        "min_l1": float(args.min_l1),
        "min_alpha": float(args.min_alpha),
        "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
        "max_samples_per_view": int(args.max_samples_per_view),
        "min_atlas_bin_count": int(args.min_atlas_bin_count),
        "min_atlas_face_samples": int(args.min_atlas_face_samples),
        "max_atlas_bin_rgb_variance": float(args.max_atlas_bin_rgb_variance),
        "min_atlas_bin_sign_consistency": float(args.min_atlas_bin_sign_consistency),
        "atlas_confidence_mode": "none",
        "atlas_confidence_count_scale": 0.0,
        "atlas_confidence_empty_bin": 1.0,
        "atlas_confidence_variance_scale": -1.0,
        "atlas_confidence_sign_power": 0.0,
        "atlas_confidence_face_sample_scale": 0.0,
        "min_atlas_confidence": 0.0,
        "enable_policy_val_image_ssim": True,
        "policy_val_ssim_max_size": int(args.policy_val_ssim_max_size),
        "enable_policy_val_image_l1": True,
        "policy_val_l1_max_size": int(args.policy_val_l1_max_size),
        "enable_policy_val_image_lpips": not bool(args.disable_lpips),
        "policy_val_lpips_max_size": int(args.policy_val_lpips_max_size),
    }


def row_with_psnr(row: dict[str, Any], psnr_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = dict(row)
    full_image_psnr = psnr_summary.get(str(float(row.get("alpha", 0.0))))
    if full_image_psnr:
        out.update(full_image_psnr)
        out["psnr_source"] = "full_image_rescan"
        return out
    # evaluate_policy_val reports mean sum-of-RGB-channel squared residual over
    # sampled policy-val pixels. Divide by 3 to get a per-channel PSNR proxy.
    before = psnr_from_mse(float(row.get("mse_before", 0.0)) / 3.0)
    after = psnr_from_mse(float(row.get("mse_after", 0.0)) / 3.0)
    out.update(
        {
            "view_count": int(row.get("view_count", 0) or 0),
            "psnr_before": before,
            "psnr_after": after,
            "psnr_gain": float(after - before) if before is not None and after is not None else None,
            "psnr_positive_view_fraction": row.get("positive_view_fraction"),
            "psnr_min_view_gain": None,
            "psnr_cvar20_view_gain": None,
            "psnr_source": "policy_val_residual_sample_proxy",
        }
    )
    return out


def select_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nonzero = [row for row in rows if float(row.get("alpha", 0.0)) > 0.0]
    all_axis = [
        row
        for row in nonzero
        if float(row.get("relative_gain", 0.0)) > 0.0
        and (row.get("psnr_gain") is None or float(row.get("psnr_gain", 0.0)) > 0.0)
        and float(row.get("ssim_gain", 0.0)) > 0.0
        and int(row.get("lpips_view_count", 0) or 0) > 0
        and float(row.get("lpips_gain", 0.0)) > 0.0
    ]
    return {
        "best_relative_gain": max(nonzero, key=lambda row: float(row.get("relative_gain", -1.0)), default=None),
        "best_ssim_gain": max(nonzero, key=lambda row: float(row.get("ssim_gain", -1.0)), default=None),
        "best_lpips_gain": max(nonzero, key=lambda row: float(row.get("lpips_gain", -1.0)), default=None),
        "best_all_axis": max(
            all_axis,
            key=lambda row: (
                float(row.get("ssim_gain", 0.0)),
                float(row.get("lpips_gain", 0.0)),
                float(row.get("relative_gain", 0.0)),
            ),
            default=None,
        ),
        "all_axis_candidate_count": int(len(all_axis)),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    view_paths = adapter.evidence_views(args.fit_evidence_dir)
    if not view_paths:
        raise FileNotFoundError(f"no evidence views found under {args.fit_evidence_dir}")
    candidate_faces, carrier_summary = adapter.load_carrier_faces(
        args.region_carrier_json,
        max_carriers=int(args.max_carriers),
        max_faces_per_carrier=int(args.max_faces_per_carrier),
        max_faces=int(args.max_faces),
    )
    if not candidate_faces:
        raise RuntimeError(f"no candidate faces loaded from {args.region_carrier_json}")

    alpha_grid = parse_csv_floats(args.alpha_grid)
    if 0.0 not in alpha_grid:
        alpha_grid = sorted([0.0, *alpha_grid])
    texture_sizes = parse_csv_ints(args.texture_sizes)
    if not texture_sizes:
        raise ValueError("--texture_sizes produced no positive sizes")
    if int(args.teacher_distilled_low_rank_texture_rank) <= 0:
        raise ValueError("--teacher_distilled_low_rank_texture_rank must be > 0")
    rank_candidates = (
        parse_csv_ints(str(args.teacher_distilled_low_rank_texture_ranks))
        if adapter._is_low_rank_teacher_texture_mode(str(args.teacher_distilled_basis_mode))
        and str(args.teacher_distilled_low_rank_texture_ranks).strip()
        else [int(args.teacher_distilled_low_rank_texture_rank)]
    )
    if adapter._is_low_rank_teacher_texture_mode(str(args.teacher_distilled_basis_mode)):
        rank_candidates = sorted(set(int(x) for x in rank_candidates))
        for rank in rank_candidates:
            if int(rank) <= 0:
                raise ValueError("--teacher_distilled_low_rank_texture_ranks values must be > 0")
    else:
        rank_candidates = [0]

    candidate_reports = []
    best_overall = None
    for texture_size in texture_sizes:
        for rank in rank_candidates:
            args._current_teacher_low_rank_texture_rank = int(rank)
            atlas, fit_summary, fit_views, val_views = adapter.fit_atlas(
                view_paths,
                set(candidate_faces),
                **atlas_fit_kwargs(args, texture_size),
            )
            policy = adapter.evaluate_policy_val(
                list(val_views),
                atlas,
                alpha_grid=alpha_grid,
                local_alpha_profile=None,
                face_gain_guard_profile=None,
                bin_uncertainty_guard_profile=None,
                parent_edge_apply_profile=None,
                view_confidence_profile=None,
                **evaluate_kwargs(args),
            )
            psnr = (
                image_psnr_rows(list(val_views), atlas, alpha_grid, args)
                if bool(args.enable_full_image_psnr_rescan)
                else {
                    "summary_by_alpha": {},
                    "per_view_by_alpha": {},
                    "source": "disabled_policy_val_residual_sample_proxy_used_in_rows",
                }
            )
            rows = [row_with_psnr(row, psnr.get("summary_by_alpha", {})) for row in list(policy.get("rows", []))]
            selections = select_rows(rows)
            report = {
                "texture_size": int(texture_size),
                "teacher_distilled_low_rank_texture_rank": int(rank),
                "fit_summary": fit_summary,
                "fit_view_count": int(len(fit_views)),
                "policy_val_view_count": int(len(val_views)),
                "policy_val_rows": rows,
                "policy_val_per_view_by_alpha": policy.get("per_view_by_alpha", {}),
                "policy_val_psnr": psnr,
                "selection": selections,
            }
            candidate_reports.append(report)
            selected = selections.get("best_all_axis")
            if selected is not None:
                selected_with_size = dict(selected)
                selected_with_size["texture_size"] = int(texture_size)
                selected_with_size["teacher_distilled_low_rank_texture_rank"] = int(rank)
                if best_overall is None or (
                    float(selected_with_size.get("ssim_gain", 0.0)),
                    float(selected_with_size.get("lpips_gain", 0.0)),
                    float(selected_with_size.get("relative_gain", 0.0)),
                ) > (
                    float(best_overall.get("ssim_gain", 0.0)),
                    float(best_overall.get("lpips_gain", 0.0)),
                    float(best_overall.get("relative_gain", 0.0)),
                ):
                    best_overall = selected_with_size

    verdict = {
        "policy_val_upper_bound_pass": bool(best_overall is not None),
        "reason": "current_carrier_improves_policy_val_all_axis"
        if best_overall is not None
        else "current_carrier_too_weak_for_policy_val_ssim_lpips",
        "best_all_axis": best_overall,
        "required_next_step": "proceed_to_v169_representation_upgrade"
        if best_overall is not None
        else "do_not_launch_flowers_exact_or_full9_until_representation_changes",
    }
    return adapter.json_safe(
        {
            "operator": "analyze_v169_policy_val_upper_bound",
            "test_usage": "none",
            "target_or_test_gt_usage": "none",
            "fit_evidence_dir": str(args.fit_evidence_dir),
            "region_carrier_json": str(args.region_carrier_json),
            "view_count": int(len(view_paths)),
            "candidate_face_count": int(len(candidate_faces)),
            "carrier_summary": carrier_summary,
            "settings": {
                "residual_rgb_key": str(args.residual_rgb_key),
                "residual_l1_key": str(args.residual_l1_key),
                "texture_sizes": texture_sizes,
                "teacher_distilled_low_rank_texture_rank_candidates": rank_candidates,
                "alpha_grid": alpha_grid,
                "policy_val_stride": int(args.policy_val_stride),
                "max_samples_per_view": int(args.max_samples_per_view),
                "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
                "lpips_enabled": not bool(args.disable_lpips),
                "policy_val_lpips_max_size": int(args.policy_val_lpips_max_size),
                "full_image_psnr_rescan": bool(args.enable_full_image_psnr_rescan),
                "teacher_residual_target_mode": str(args.teacher_residual_target_mode),
                "teacher_residual_target_luma_mix": float(args.teacher_residual_target_luma_mix),
                "teacher_residual_target_edge_boost": float(args.teacher_residual_target_edge_boost),
                "teacher_distilled_basis_mode": str(args.teacher_distilled_basis_mode),
                "teacher_distilled_basis_min_face_samples": int(args.teacher_distilled_basis_min_face_samples),
                "teacher_distilled_basis_apply_mode": str(args.teacher_distilled_basis_apply_mode),
                "teacher_distilled_basis_blend": float(args.teacher_distilled_basis_blend),
                "adaptive_low_support_teacher_basis": bool(args.enable_adaptive_low_support_teacher_basis),
            },
            "phasej_flowers_gate_reference": {
                "PSNR": float(args.phasej_flowers_psnr),
                "SSIM": float(args.phasej_flowers_ssim),
                "LPIPS": float(args.phasej_flowers_lpips),
            },
            "candidates": candidate_reports,
            "verdict": verdict,
        }
    )


def fmt(value: Any, digits: int = 8, signed: bool = False) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    verdict = result["verdict"]
    lines = [
        "# v169 Policy-Val Carrier Upper-Bound Diagnostic",
        "",
        f"- fit evidence: `{result['fit_evidence_dir']}`",
        f"- region carrier: `{result['region_carrier_json']}`",
        f"- candidate faces: `{result['candidate_face_count']}`",
        f"- view count: `{result['view_count']}`",
        f"- verdict: `{verdict['reason']}`",
        "",
        "## Best All-Axis Candidate",
        "",
    ]
    best = verdict.get("best_all_axis")
    if best:
        lines.extend(
            [
                f"- texture size: `{best.get('texture_size')}`",
                f"- teacher low-rank texture rank: `{best.get('teacher_distilled_low_rank_texture_rank')}`",
                f"- alpha: `{fmt(best.get('alpha'))}`",
                f"- relative gain: `{fmt(best.get('relative_gain'), signed=True)}`",
                f"- PSNR gain: `{fmt(best.get('psnr_gain'), signed=True)}`",
                f"- SSIM gain: `{fmt(best.get('ssim_gain'), signed=True)}`",
                f"- LPIPS gain: `{fmt(best.get('lpips_gain'), signed=True)}`",
            ]
        )
    else:
        lines.append("- no nonzero alpha improves relative residual error, PSNR, SSIM, and LPIPS simultaneously.")
    lines.extend(["", "## Candidate Rows", "", "| texture | rank | alpha | rel gain | PSNR gain | SSIM gain | LPIPS gain | all-axis |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for candidate in result.get("candidates", []):
        all_axis_alpha = None
        selected = candidate.get("selection", {}).get("best_all_axis")
        if selected:
            all_axis_alpha = float(selected.get("alpha", 0.0))
        for row in candidate.get("policy_val_rows", []):
            alpha = float(row.get("alpha", 0.0))
            all_axis = bool(all_axis_alpha is not None and abs(alpha - all_axis_alpha) < 1.0e-9)
            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt(candidate.get("texture_size")),
                        fmt(candidate.get("teacher_distilled_low_rank_texture_rank")),
                        fmt(alpha),
                        fmt(row.get("relative_gain"), signed=True),
                        fmt(row.get("psnr_gain"), signed=True),
                        fmt(row.get("ssim_gain"), signed=True),
                        fmt(row.get("lpips_gain"), signed=True),
                        "yes" if all_axis else "",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This diagnostic uses train-fit evidence for fitting and train-policy-val GT for certification only.",
            "- It does not read target/test GT and does not write model artifacts.",
            "- If no all-axis candidate exists, v169 should not promote this carrier to flowers exact/full9.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    if args.output_json.resolve() == args.output_md.resolve():
        raise SystemExit("error: --output_json and --output_md must be different")
    result = analyze(args)
    write_text_atomic(args.output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_text_atomic(args.output_md, render_markdown(result))
    print(json.dumps({"verdict": result["verdict"], "output_json": str(args.output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
