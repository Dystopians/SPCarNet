#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.ecsr_apply_surface_residual_region_texture_adapter import (  # noqa: E402
    evaluate_policy_val,
    evidence_views,
    fit_atlas,
    image_lpips_chw,
    image_ssim_chw,
)


DEFAULT_MODEL = (
    "outputs/carnet/meshsplatopt/ecsr_phase_f/"
    "policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model"
)
DEFAULT_EVIDENCE = "/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence"


def _load_rgb(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.moveaxis(arr, -1, 0)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.mean(diff * diff))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = _mse(a, b)
    if mse <= 1.0e-12:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def _luma_grad(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    gx[:, 1:] = np.abs(luma[:, 1:] - luma[:, :-1])
    gy[1:, :] = np.abs(luma[1:, :] - luma[:-1, :])
    return np.maximum(gx, gy)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "median": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _parse_csv_floats(text: str) -> list[float]:
    vals: list[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            vals.append(float(token))
    return vals or [0.0, 0.125, 0.25, 0.5]


def _parse_csv_modes(text: str) -> list[str]:
    vals: list[str] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            vals.append(token)
    return vals or ["none"]


def _policy_val_paths(paths: list[Path], stride: int) -> list[Path]:
    stride = max(0, int(stride))
    if stride <= 1:
        return []
    return [path for idx, path in enumerate(paths) if idx % stride == 0]


def _image_metrics_for_views(
    *,
    model_path: Path,
    parent_method: str,
    teacher_method: str,
    view_stems: list[str],
    ssim_max_side: int,
    lpips_max_side: int,
    compute_lpips: bool,
) -> dict[str, Any]:
    parent_dir = model_path / "train" / parent_method
    teacher_dir = model_path / "train" / teacher_method
    if not parent_dir.is_dir():
        raise FileNotFoundError(parent_dir)
    if not teacher_dir.is_dir():
        raise FileNotFoundError(teacher_dir)
    lpips_model = None
    if compute_lpips:
        from scripts.car_model.ecsr_apply_surface_residual_region_texture_adapter import build_lpips_model

        lpips_model = build_lpips_model()

    rows: list[dict[str, Any]] = []
    parent_psnr: list[float] = []
    teacher_psnr: list[float] = []
    parent_ssim: list[float] = []
    teacher_ssim: list[float] = []
    parent_lpips: list[float] = []
    teacher_lpips: list[float] = []
    teacher_parent_energy: list[float] = []
    teacher_gt_cosine: list[float] = []
    sign_agreement: list[float] = []
    edge_energy_ratio: list[float] = []

    for stem in view_stems:
        fname = f"{stem}.png"
        parent_path = parent_dir / "renders" / fname
        teacher_path = teacher_dir / "renders" / fname
        gt_path = parent_dir / "gt" / fname
        if not parent_path.is_file() or not teacher_path.is_file() or not gt_path.is_file():
            rows.append(
                {
                    "view": stem,
                    "missing": True,
                    "parent_render": str(parent_path),
                    "teacher_render": str(teacher_path),
                    "gt": str(gt_path),
                }
            )
            continue
        parent = _load_rgb(parent_path)
        teacher = _load_rgb(teacher_path)
        gt = _load_rgb(gt_path)
        p_psnr = _psnr(parent, gt)
        t_psnr = _psnr(teacher, gt)
        p_ssim = image_ssim_chw(parent, gt, int(ssim_max_side))
        t_ssim = image_ssim_chw(teacher, gt, int(ssim_max_side))
        p_lp = image_lpips_chw(parent, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
        t_lp = image_lpips_chw(teacher, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
        teacher_delta = teacher - parent
        gt_delta = gt - parent
        teacher_energy = float(np.mean(np.sum(teacher_delta * teacher_delta, axis=0)))
        gt_energy = float(np.mean(np.sum(gt_delta * gt_delta, axis=0)))
        dot = float(np.sum(teacher_delta.astype(np.float64) * gt_delta.astype(np.float64)))
        denom = math.sqrt(
            float(np.sum(teacher_delta.astype(np.float64) ** 2))
            * float(np.sum(gt_delta.astype(np.float64) ** 2))
        )
        cosine = float(dot / denom) if denom > 1.0e-12 else 0.0
        nonzero = (np.abs(teacher_delta) > 1.0e-6) & (np.abs(gt_delta) > 1.0e-6)
        sign = float(np.mean(np.sign(teacher_delta[nonzero]) == np.sign(gt_delta[nonzero]))) if np.any(nonzero) else 0.0
        parent_edge = float(np.mean(_luma_grad(parent)))
        teacher_delta_edge = float(np.mean(_luma_grad(teacher_delta)))
        edge_ratio = float(teacher_delta_edge / max(parent_edge, 1.0e-12))
        parent_psnr.append(p_psnr)
        teacher_psnr.append(t_psnr)
        parent_ssim.append(p_ssim)
        teacher_ssim.append(t_ssim)
        if p_lp is not None and t_lp is not None:
            parent_lpips.append(float(p_lp))
            teacher_lpips.append(float(t_lp))
        teacher_parent_energy.append(teacher_energy)
        teacher_gt_cosine.append(cosine)
        sign_agreement.append(sign)
        edge_energy_ratio.append(edge_ratio)
        rows.append(
            {
                "view": stem,
                "missing": False,
                "parent_psnr": float(p_psnr),
                "teacher_psnr": float(t_psnr),
                "psnr_gain": float(t_psnr - p_psnr),
                "parent_ssim": float(p_ssim),
                "teacher_ssim": float(t_ssim),
                "ssim_gain": float(t_ssim - p_ssim),
                "parent_lpips": None if p_lp is None else float(p_lp),
                "teacher_lpips": None if t_lp is None else float(t_lp),
                "lpips_gain": None if p_lp is None or t_lp is None else float(p_lp - t_lp),
                "teacher_parent_energy": float(teacher_energy),
                "gt_parent_energy": float(gt_energy),
                "teacher_gt_residual_cosine": float(cosine),
                "teacher_gt_sign_agreement": float(sign),
                "teacher_delta_edge_to_parent_edge": float(edge_ratio),
            }
        )

    summary = {
        "view_count": int(len(rows)),
        "valid_view_count": int(sum(not row.get("missing", False) for row in rows)),
        "parent": {
            "psnr": _mean(parent_psnr),
            "ssim": _mean(parent_ssim),
            "lpips": _mean(parent_lpips),
        },
        "teacher": {
            "psnr": _mean(teacher_psnr),
            "ssim": _mean(teacher_ssim),
            "lpips": _mean(teacher_lpips),
        },
        "teacher_minus_parent": {
            "psnr_gain": _mean([t - p for t, p in zip(teacher_psnr, parent_psnr, strict=False)]),
            "ssim_gain": _mean([t - p for t, p in zip(teacher_ssim, parent_ssim, strict=False)]),
            "lpips_gain": _mean([p - t for p, t in zip(parent_lpips, teacher_lpips, strict=False)]),
            "psnr_positive_view_fraction": float(
                np.mean(np.asarray([t > p for t, p in zip(teacher_psnr, parent_psnr, strict=False)], dtype=np.float32))
            )
            if parent_psnr
            else 0.0,
            "ssim_positive_view_fraction": float(
                np.mean(np.asarray([t > p for t, p in zip(teacher_ssim, parent_ssim, strict=False)], dtype=np.float32))
            )
            if parent_ssim
            else 0.0,
            "lpips_positive_view_fraction": float(
                np.mean(np.asarray([p > t for p, t in zip(parent_lpips, teacher_lpips, strict=False)], dtype=np.float32))
            )
            if parent_lpips
            else 0.0,
            "teacher_parent_energy": _mean(teacher_parent_energy),
            "teacher_parent_energy_quantiles": _quantiles(teacher_parent_energy),
            "teacher_gt_residual_cosine": _mean(teacher_gt_cosine),
            "teacher_gt_sign_agreement": _mean(sign_agreement),
            "teacher_delta_edge_to_parent_edge": _mean(edge_energy_ratio),
        },
        "rows": rows,
    }
    return summary


def _evidence_signal_audit(paths: list[Path], residual_l1_key: str, residual_rgb_key: str) -> dict[str, Any]:
    raw_energy: list[float] = []
    used_energy: list[float] = []
    selected_fraction: list[float] = []
    nonzero_fraction: list[float] = []
    better_fraction: list[float] = []
    support_counts: list[int] = []
    residual_l1_values: list[float] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            if residual_rgb_key not in z or residual_l1_key not in z:
                continue
            face_id = np.asarray(z["face_id"], dtype=np.int64)
            valid = face_id >= 0
            if "barycentric_valid" in z:
                valid &= np.asarray(z["barycentric_valid"]).astype(bool)
            raw_key = f"{residual_rgb_key}_raw"
            raw = np.asarray(z[raw_key if raw_key in z else residual_rgb_key], dtype=np.float32)
            used = np.asarray(z[residual_rgb_key], dtype=np.float32)
            l1 = np.asarray(z[residual_l1_key], dtype=np.float32)
            if not np.any(valid):
                continue
            raw_norm = np.sum(raw * raw, axis=0)
            used_norm = np.sum(used * used, axis=0)
            raw_energy.append(float(np.mean(raw_norm[valid])))
            used_energy.append(float(np.mean(used_norm[valid])))
            selected_fraction.append(float(np.mean(used_norm[valid] > 1.0e-12)))
            nonzero_fraction.append(float(np.mean(l1[valid] > 1.0e-8)))
            if "teacher_better_mask" in z:
                better_fraction.append(float(np.mean(np.asarray(z["teacher_better_mask"]).astype(bool)[valid])))
            support_counts.append(int(np.sum(valid)))
            sample = l1[valid].reshape(-1)
            if sample.size > 8192:
                sample = sample[np.linspace(0, sample.size - 1, 8192, dtype=np.int64)]
            residual_l1_values.extend(float(x) for x in sample.tolist())
    raw_mean = _mean(raw_energy)
    used_mean = _mean(used_energy)
    return {
        "view_count": int(len(paths)),
        "measured_view_count": int(len(raw_energy)),
        "mean_raw_energy": float(raw_mean),
        "mean_used_energy": float(used_mean),
        "used_to_raw_energy_ratio": float(used_mean / max(raw_mean, 1.0e-12)),
        "mean_selected_pixel_fraction": _mean(selected_fraction),
        "mean_nonzero_l1_fraction": _mean(nonzero_fraction),
        "mean_teacher_better_mask_fraction": _mean(better_fraction),
        "support_pixel_count_quantiles": _quantiles([float(x) for x in support_counts]),
        "residual_l1_quantiles": _quantiles(residual_l1_values),
    }


def _rank_fit_candidate_faces(
    paths: list[Path],
    *,
    residual_l1_key: str,
    policy_val_stride: int,
    min_l1: float,
    min_alpha: float,
    max_faces: int,
    max_samples_per_view: int,
) -> tuple[set[int], dict[str, Any]]:
    max_faces = int(max_faces)
    if max_faces <= 0:
        return set(), {
            "enabled": False,
            "reason": "max_candidate_faces_disabled",
            "candidate_faces": 0,
            "selection_scope": "all_valid_faces",
        }
    stride = max(0, int(policy_val_stride))
    rng = np.random.default_rng(173)
    counts: dict[int, int] = {}
    l1_sums: dict[int, float] = {}
    fit_views = 0
    skipped_policy_val_views = 0
    for view_index, path in enumerate(paths):
        if stride > 1 and view_index % stride == 0:
            skipped_policy_val_views += 1
            continue
        fit_views += 1
        with np.load(path, allow_pickle=False) as z:
            if "face_id" not in z or residual_l1_key not in z:
                continue
            face_id = np.asarray(z["face_id"], dtype=np.int64)
            mask = face_id >= 0
            if "barycentric_valid" in z:
                mask &= np.asarray(z["barycentric_valid"]).astype(bool)
            if "alpha" in z:
                mask &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
            l1 = np.asarray(z[residual_l1_key], dtype=np.float32)
            mask &= l1 >= float(min_l1)
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
            if int(max_samples_per_view) > 0 and ys.size > int(max_samples_per_view):
                take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
                ys = ys[take]
                xs = xs[take]
            faces = face_id[ys, xs]
            values = l1[ys, xs].astype(np.float64)
            valid = faces >= 0
            faces = faces[valid]
            values = values[valid]
            if faces.size == 0:
                continue
            unique, inverse = np.unique(faces, return_inverse=True)
            view_counts = np.bincount(inverse, minlength=int(unique.size))
            view_l1 = np.bincount(inverse, weights=values, minlength=int(unique.size))
            for idx, face in enumerate(unique):
                face_i = int(face)
                count = int(view_counts[idx])
                if count <= 0:
                    continue
                counts[face_i] = counts.get(face_i, 0) + count
                l1_sums[face_i] = l1_sums.get(face_i, 0.0) + float(view_l1[idx])
    rows = []
    for face, count in counts.items():
        mean_l1 = float(l1_sums[face] / max(1, count))
        score = mean_l1 * math.log1p(float(count))
        rows.append({"face_id": int(face), "samples": int(count), "mean_l1": mean_l1, "score": float(score)})
    rows.sort(key=lambda row: (float(row["score"]), float(row["mean_l1"]), int(row["samples"])), reverse=True)
    selected = rows[:max_faces]
    faces = {int(row["face_id"]) for row in selected}
    return faces, {
        "enabled": True,
        "selection_scope": "train_fit_only_residual_l1",
        "fit_views": int(fit_views),
        "skipped_policy_val_views": int(skipped_policy_val_views),
        "candidate_faces": int(len(faces)),
        "eligible_faces": int(len(rows)),
        "max_candidate_faces": int(max_faces),
        "top_faces_preview": selected[:32],
    }


def _slim_policy_val(policy_val: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in policy_val.get("rows", []):
        rows.append(
            {
                key: row.get(key)
                for key in [
                    "alpha",
                    "relative_gain",
                    "positive_view_fraction",
                    "min_view_relative_gain",
                    "cvar20_view_relative_gain",
                    "ssim_gain",
                    "ssim_positive_view_fraction",
                    "ssim_min_view_gain",
                    "ssim_cvar20_view_gain",
                    "image_l1_gain",
                    "image_l1_positive_view_fraction",
                    "image_l1_min_view_gain",
                    "image_l1_cvar20_view_gain",
                    "lpips_gain",
                    "lpips_positive_view_fraction",
                    "lpips_min_view_gain",
                    "lpips_cvar20_view_gain",
                ]
                if key in row
            }
        )
    best_all_axis = None
    for row in rows:
        if (
            float(row.get("relative_gain", 0.0) or 0.0) > 0.0
            and float(row.get("ssim_gain", 0.0) or 0.0) > 0.0
            and float(row.get("lpips_gain", 0.0) or 0.0) > 0.0
        ):
            score = (
                float(row.get("relative_gain", 0.0) or 0.0)
                + 20.0 * float(row.get("ssim_gain", 0.0) or 0.0)
                + 20.0 * float(row.get("lpips_gain", 0.0) or 0.0)
            )
            candidate = dict(row)
            candidate["balanced_projection_score"] = float(score)
            if best_all_axis is None or score > float(best_all_axis.get("balanced_projection_score", -1.0)):
                best_all_axis = candidate
    return {
        "enabled": bool(policy_val.get("enabled", False)),
        "samples": int(policy_val.get("samples", 0) or 0),
        "unique_faces": int(policy_val.get("unique_faces", 0) or 0),
        "mse_before": float(policy_val.get("mse_before", 0.0) or 0.0),
        "best_by_mse": dict(policy_val.get("best", {}) or {}),
        "best_all_axis_policy_val": best_all_axis,
        "rows": rows,
    }


def _run_projection_candidate(args: argparse.Namespace, mode: str, view_paths: list[Path]) -> dict[str, Any]:
    alpha_grid = _parse_csv_floats(args.alpha_grid)
    candidate_faces, candidate_face_summary = _rank_fit_candidate_faces(
        view_paths,
        residual_l1_key=str(args.residual_l1_key),
        policy_val_stride=int(args.policy_val_stride),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        max_faces=int(args.max_candidate_faces),
        max_samples_per_view=int(args.max_candidate_face_samples_per_view),
    )
    atlas, fit_summary, fit_views, val_views = fit_atlas(
        view_paths,
        candidate_faces,
        str(args.residual_rgb_key),
        str(args.residual_l1_key),
        int(args.texture_size),
        int(args.policy_val_stride),
        float(args.min_l1),
        float(args.min_alpha),
        int(args.max_samples_per_view),
        False,
        "none",
        0,
        1.0,
        int(args.atlas_lowpass_passes),
        int(args.atlas_lowpass_neighbor_min_count),
        str(args.surface_multiscale_prior_mode),
        [2, 4, 8],
        int(args.surface_multiscale_prior_min_bin_samples),
        float(args.surface_multiscale_prior_count_tau),
        float(args.surface_multiscale_prior_blend),
        str(args.surface_multiscale_prior_gate_mode),
        float(args.surface_multiscale_prior_min_prior_weight),
        int(args.surface_multiscale_prior_min_direct_samples),
        float(args.surface_multiscale_prior_min_sign_consistency),
        float(args.surface_multiscale_prior_max_mean_variance),
        float(args.surface_multiscale_prior_min_cosine),
        str(args.view_conditioned_basis_mode),
        int(args.view_conditioned_basis_min_bin_samples),
        float(args.view_conditioned_basis_ridge),
        str(args.view_conditioned_basis_ood_mode),
        float(args.view_conditioned_basis_ood_max_z),
        float(args.view_conditioned_basis_ood_min_std),
        int(args.view_cluster_expert_count),
        "camera_center" if int(args.view_cluster_expert_count) > 1 else "none",
        int(args.view_cluster_min_views),
        int(args.view_cluster_min_bin_samples),
        "global",
        str(args.teacher_residual_target_mode),
        float(args.teacher_residual_target_luma_mix),
        float(args.teacher_residual_target_edge_boost),
        str(mode),
        int(args.teacher_distilled_basis_min_face_samples),
        float(args.teacher_distilled_basis_ridge),
        float(args.teacher_distilled_basis_ood_max_z),
        float(args.teacher_distilled_basis_ood_min_std),
        str(args.teacher_distilled_basis_apply_mode),
        float(args.teacher_distilled_basis_blend),
        int(args.teacher_distilled_low_rank_texture_rank),
        bool(args.enable_adaptive_low_support_teacher_basis),
        int(args.adaptive_teacher_basis_min_face_samples_floor),
        float(args.adaptive_teacher_basis_support_quantile),
        float(args.adaptive_teacher_basis_low_support_ridge_scale),
    )
    policy_val = evaluate_policy_val(
        val_views,
        atlas,
        str(args.residual_rgb_key),
        str(args.residual_l1_key),
        alpha_grid,
        float(args.min_l1),
        float(args.min_alpha),
        float(args.max_abs_delta_rgb),
        int(args.max_policy_val_samples_per_view),
        int(args.min_atlas_bin_count),
        int(args.min_atlas_face_samples),
        float(args.max_atlas_bin_rgb_variance),
        float(args.min_atlas_bin_sign_consistency),
        str(args.atlas_confidence_mode),
        float(args.atlas_confidence_count_scale),
        float(args.atlas_confidence_empty_bin),
        float(args.atlas_confidence_variance_scale),
        float(args.atlas_confidence_sign_power),
        float(args.atlas_confidence_face_sample_scale),
        float(args.min_atlas_confidence),
        True,
        int(args.policy_val_ssim_max_side),
        True,
        int(args.policy_val_l1_max_side),
        bool(args.compute_lpips),
        int(args.policy_val_lpips_max_side),
    )
    slim = _slim_policy_val(policy_val)
    projection_pass = slim.get("best_all_axis_policy_val") is not None
    return {
        "mode": str(mode),
        "projection_pass_policy_val_all_axis": bool(projection_pass),
        "fit_summary": {
            "input_views": fit_summary.get("input_views"),
            "fit_views": fit_summary.get("fit_views"),
            "policy_val_views": fit_summary.get("policy_val_views"),
            "atlas_faces": fit_summary.get("atlas_faces"),
            "fit_samples": fit_summary.get("fit_samples"),
            "texture_size": fit_summary.get("texture_size"),
            "teacher_residual_target": fit_summary.get("teacher_residual_target"),
            "view_conditioned_basis": fit_summary.get("view_conditioned_basis"),
            "view_cluster_experts": fit_summary.get("view_cluster_experts"),
            "teacher_distilled_basis": fit_summary.get("teacher_distilled_basis"),
            "candidate_face_prefilter": candidate_face_summary,
        },
        "fit_view_count": int(len(fit_views)),
        "policy_val_view_count": int(len(val_views)),
        "policy_val": slim,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    image = payload["teacher_signal_image_audit"]
    teacher = image["teacher"]
    parent = image["parent"]
    delta = image["teacher_minus_parent"]
    lines = [
        "# v176 Phase-J Teacher Residual Projection Audit",
        "",
        f"Date: {payload['created_at']}",
        "",
        "## Verdict",
        "",
        f"- flower Phase-J all-axis gate target: `{payload['phasej_flowers_gate']['psnr']:.6f} / {payload['phasej_flowers_gate']['ssim']:.6f} / {payload['phasej_flowers_gate']['lpips']:.6f}`",
        f"- teacher improves parent on policy-val: `{payload['teacher_signal_pass']}`",
        f"- any carrier projection improves MSE + SSIM + LPIPS on policy-val: `{payload['projection_pass_any_candidate']}`",
        f"- full9 allowed by this audit: `{payload['full9_allowed']}`",
        "",
        "## Teacher Signal",
        "",
        f"- parent policy-val: `{parent['psnr']:.6f} / {parent['ssim']:.6f} / {parent['lpips']:.6f}`",
        f"- Phase-J teacher policy-val: `{teacher['psnr']:.6f} / {teacher['ssim']:.6f} / {teacher['lpips']:.6f}`",
        f"- teacher minus parent gain: `{delta['psnr_gain']:.6f}` PSNR, `{delta['ssim_gain']:.6f}` SSIM, `{delta['lpips_gain']:.6f}` LPIPS-improvement",
        f"- teacher-parent residual energy: `{delta['teacher_parent_energy']:.8f}`",
        f"- teacher/GT residual cosine: `{delta['teacher_gt_residual_cosine']:.6f}`; sign agreement: `{delta['teacher_gt_sign_agreement']:.6f}`",
        "",
        "## Evidence Residual Audit",
        "",
        f"- raw teacher residual energy: `{payload['evidence_signal_audit']['mean_raw_energy']:.8f}`",
        f"- used teacher residual energy: `{payload['evidence_signal_audit']['mean_used_energy']:.8f}`",
        f"- used/raw energy ratio: `{payload['evidence_signal_audit']['used_to_raw_energy_ratio']:.6f}`",
        f"- selected pixel fraction: `{payload['evidence_signal_audit']['mean_selected_pixel_fraction']:.6f}`",
        f"- teacher_better_mask fraction: `{payload['evidence_signal_audit']['mean_teacher_better_mask_fraction']:.6f}`",
        "",
        "## Carrier Projection",
        "",
        "| mode | pass | best alpha | MSE rel gain | SSIM gain | LPIPS gain | pos views | LPIPS pos views |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["projection_candidates"]:
        best = row["policy_val"].get("best_all_axis_policy_val") or row["policy_val"].get("best_by_mse", {})
        lines.append(
            "| {mode} | {passed} | {alpha:.6f} | {rel:.6f} | {ssim:.6f} | {lpips:.6f} | {pv:.3f} | {lpv:.3f} |".format(
                mode=row["mode"],
                passed=str(row["projection_pass_policy_val_all_axis"]),
                alpha=float(best.get("alpha", 0.0) or 0.0),
                rel=float(best.get("relative_gain", 0.0) or 0.0),
                ssim=float(best.get("ssim_gain", 0.0) or 0.0),
                lpips=float(best.get("lpips_gain", 0.0) or 0.0),
                pv=float(best.get("positive_view_fraction", 0.0) or 0.0),
                lpv=float(best.get("lpips_positive_view_fraction", 0.0) or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Artifact Paths",
            "",
            f"- JSON: `{payload['output_json']}`",
            f"- Markdown: `{path}`",
            f"- evidence dir: `{payload['inputs']['fit_evidence_dir']}`",
            f"- model path: `{payload['inputs']['model_path']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Phase-J teacher residual signal and current surface-carrier projection upper bound."
    )
    parser.add_argument("--scene", default="flowers")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--fit_evidence_dir", default=DEFAULT_EVIDENCE)
    parser.add_argument("--parent_method", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument("--teacher_method", default="ours_26000_phasej_trainval_gate")
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--texture_size", type=int, default=16)
    parser.add_argument("--alpha_grid", default="0,0.0625,0.125,0.25,0.5")
    parser.add_argument("--min_l1", type=float, default=0.0)
    parser.add_argument("--min_alpha", type=float, default=0.03)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=0.12)
    parser.add_argument("--max_samples_per_view", type=int, default=8192)
    parser.add_argument("--max_policy_val_samples_per_view", type=int, default=8192)
    parser.add_argument(
        "--max_candidate_faces",
        type=int,
        default=0,
        help="Optional train-fit-only top residual face prefilter. 0 keeps the all-face carrier.",
    )
    parser.add_argument(
        "--max_candidate_face_samples_per_view",
        type=int,
        default=8192,
        help="Per-view sample cap used only for train-fit candidate-face ranking.",
    )
    parser.add_argument("--projection_modes", default="none,low_rank_view_texture_rich")
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--policy_val_ssim_max_side", type=int, default=512)
    parser.add_argument("--policy_val_l1_max_side", type=int, default=512)
    parser.add_argument("--policy_val_lpips_max_side", type=int, default=256)
    parser.add_argument("--min_atlas_bin_count", type=int, default=0)
    parser.add_argument("--min_atlas_face_samples", type=int, default=0)
    parser.add_argument("--max_atlas_bin_rgb_variance", type=float, default=-1.0)
    parser.add_argument("--min_atlas_bin_sign_consistency", type=float, default=0.0)
    parser.add_argument("--atlas_confidence_mode", choices=("none", "count_var_sign"), default="none")
    parser.add_argument("--atlas_confidence_count_scale", type=float, default=0.0)
    parser.add_argument("--atlas_confidence_empty_bin", type=float, default=1.0)
    parser.add_argument("--atlas_confidence_variance_scale", type=float, default=-1.0)
    parser.add_argument("--atlas_confidence_sign_power", type=float, default=0.0)
    parser.add_argument("--atlas_confidence_face_sample_scale", type=float, default=0.0)
    parser.add_argument("--min_atlas_confidence", type=float, default=0.0)
    parser.add_argument("--atlas_lowpass_passes", type=int, default=0)
    parser.add_argument("--atlas_lowpass_neighbor_min_count", type=int, default=1)
    parser.add_argument("--surface_multiscale_prior_mode", choices=("none", "count_pyramid", "local_patch"), default="none")
    parser.add_argument("--surface_multiscale_prior_min_bin_samples", type=int, default=8)
    parser.add_argument("--surface_multiscale_prior_count_tau", type=float, default=32.0)
    parser.add_argument("--surface_multiscale_prior_blend", type=float, default=1.0)
    parser.add_argument("--surface_multiscale_prior_gate_mode", choices=("none", "evidence_consistent"), default="none")
    parser.add_argument("--surface_multiscale_prior_min_prior_weight", type=float, default=0.0)
    parser.add_argument("--surface_multiscale_prior_min_direct_samples", type=int, default=1)
    parser.add_argument("--surface_multiscale_prior_min_sign_consistency", type=float, default=0.0)
    parser.add_argument("--surface_multiscale_prior_max_mean_variance", type=float, default=-1.0)
    parser.add_argument("--surface_multiscale_prior_min_cosine", type=float, default=0.0)
    parser.add_argument(
        "--view_conditioned_basis_mode",
        choices=("none", "camera_center_linear", "normal_camera_linear"),
        default="none",
    )
    parser.add_argument("--view_conditioned_basis_min_bin_samples", type=int, default=4)
    parser.add_argument("--view_conditioned_basis_ridge", type=float, default=1.0e-2)
    parser.add_argument("--view_conditioned_basis_ood_mode", choices=("none", "diag_z"), default="none")
    parser.add_argument("--view_conditioned_basis_ood_max_z", type=float, default=3.0)
    parser.add_argument("--view_conditioned_basis_ood_min_std", type=float, default=5.0e-2)
    parser.add_argument("--view_cluster_expert_count", type=int, default=1)
    parser.add_argument("--view_cluster_min_views", type=int, default=2)
    parser.add_argument("--view_cluster_min_bin_samples", type=int, default=4)
    parser.add_argument("--teacher_residual_target_mode", choices=("raw_rgb", "luma_only", "edge_luma_mix"), default="raw_rgb")
    parser.add_argument("--teacher_residual_target_luma_mix", type=float, default=0.5)
    parser.add_argument("--teacher_residual_target_edge_boost", type=float, default=0.5)
    parser.add_argument("--teacher_distilled_basis_min_face_samples", type=int, default=128)
    parser.add_argument("--teacher_distilled_basis_ridge", type=float, default=1.0e-2)
    parser.add_argument("--teacher_distilled_basis_ood_max_z", type=float, default=3.0)
    parser.add_argument("--teacher_distilled_basis_ood_min_std", type=float, default=5.0e-2)
    parser.add_argument("--teacher_distilled_basis_apply_mode", choices=("replace_supported", "blend", "fill_empty_only"), default="blend")
    parser.add_argument("--teacher_distilled_basis_blend", type=float, default=0.5)
    parser.add_argument("--teacher_distilled_low_rank_texture_rank", type=int, default=4)
    parser.add_argument("--enable_adaptive_low_support_teacher_basis", action="store_true")
    parser.add_argument("--adaptive_teacher_basis_min_face_samples_floor", type=int, default=64)
    parser.add_argument("--adaptive_teacher_basis_support_quantile", type=float, default=0.25)
    parser.add_argument("--adaptive_teacher_basis_low_support_ridge_scale", type=float, default=1.0)
    parser.add_argument("--output_json", default="/tmp/peilincai_spcarnet_v176_phasej_teacher_projection_audit.json")
    parser.add_argument("--output_md", default="docs/car_model/6-29-v176-PhaseJTeacherProjectionAudit.md")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    evidence_dir = Path(args.fit_evidence_dir)
    view_paths = evidence_views(evidence_dir)
    if not view_paths:
        raise FileNotFoundError(f"no evidence views found under {evidence_dir}")
    val_paths = _policy_val_paths(view_paths, int(args.policy_val_stride))
    if not val_paths:
        raise RuntimeError("policy-val view set is empty; use --policy_val_stride > 1")
    val_stems = [path.stem for path in val_paths]

    image_audit = _image_metrics_for_views(
        model_path=model_path,
        parent_method=str(args.parent_method),
        teacher_method=str(args.teacher_method),
        view_stems=val_stems,
        ssim_max_side=int(args.policy_val_ssim_max_side),
        lpips_max_side=int(args.policy_val_lpips_max_side),
        compute_lpips=bool(args.compute_lpips),
    )
    signal_delta = image_audit["teacher_minus_parent"]
    teacher_signal_pass = bool(
        signal_delta["psnr_gain"] > 0.0
        and signal_delta["ssim_gain"] > 0.0
        and (not bool(args.compute_lpips) or signal_delta["lpips_gain"] > 0.0)
        and signal_delta["teacher_parent_energy"] > 1.0e-10
    )
    evidence_audit = _evidence_signal_audit(view_paths, str(args.residual_l1_key), str(args.residual_rgb_key))

    projection_candidates = []
    for mode in _parse_csv_modes(args.projection_modes):
        projection_candidates.append(_run_projection_candidate(args, mode, view_paths))
    projection_pass = any(bool(row.get("projection_pass_policy_val_all_axis", False)) for row in projection_candidates)
    if projection_pass:
        interpretation = (
            "Policy-val projection found at least one candidate that improves residual MSE, SSIM, and LPIPS. "
            "This is enough to justify one flowers exact run, still with strict no-target-GT apply and no full9 promotion "
            "until the Phase-J flowers all-axis gate is beaten."
        )
    else:
        interpretation = (
            "The Phase-J teacher signal is measurable, but the tested baked surface carrier did not produce a policy-val "
            "all-axis projection win. Under the v169 improved prompt this blocks new full9 runs and points to carrier "
            "under-capacity or mask/energy dilution rather than missing experiment packaging."
        )

    payload: dict[str, Any] = {
        "schema": "spcarnet_phasej_teacher_projection_audit_v1",
        "created_at": "2026-06-29",
        "scene": str(args.scene),
        "inputs": {
            "model_path": str(model_path),
            "fit_evidence_dir": str(evidence_dir),
            "parent_method": str(args.parent_method),
            "teacher_method": str(args.teacher_method),
            "policy_val_stride": int(args.policy_val_stride),
            "policy_val_views": val_stems,
            "residual_rgb_key": str(args.residual_rgb_key),
            "residual_l1_key": str(args.residual_l1_key),
        },
        "phasej_flowers_gate": {
            "psnr": 20.304358,
            "ssim": 0.557770,
            "lpips": 0.329222,
        },
        "teacher_signal_pass": bool(teacher_signal_pass),
        "teacher_signal_image_audit": image_audit,
        "evidence_signal_audit": evidence_audit,
        "projection_candidates": projection_candidates,
        "projection_pass_any_candidate": bool(projection_pass),
        "full9_allowed": False,
        "interpretation": interpretation,
        "output_json": str(args.output_json),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["output_json"] = str(output_json)
    _write_markdown(Path(args.output_md), payload)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "teacher_signal_pass": bool(teacher_signal_pass),
        "projection_pass_any_candidate": bool(projection_pass),
        "output_json": str(output_json),
        "output_md": str(args.output_md),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
