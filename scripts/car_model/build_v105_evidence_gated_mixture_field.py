#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from render import _camera_record, _downsample_rend_ids_nearest, _sha256
from triangle_renderer import render

from scripts.car_model.build_v103_surface_affine_residual_field import (
    _assert_strict_camera_matches_bank,
    _dtype,
    _load_delta_bank,
    _load_scene,
)
from scripts.car_model.build_v104b_centered_view_affine_residual_field import (
    BASIS_ORDER,
    XTX_ORDER,
    _accumulate_view_affine_view,
    _center_transform,
    _jsonable_args,
    _raw_xtx_matrix,
    _solve_fallback_affine_coefficients,
    _update_rank_condition_stats,
)


def _condition_score(conditions: torch.Tensor, condition_max: float) -> torch.Tensor:
    score = torch.ones_like(conditions, dtype=torch.float64)
    finite = torch.isfinite(conditions)
    if bool(finite.any().item()):
        log_cond = torch.log10(conditions[finite].clamp_min(1.0))
        hi = torch.log10(torch.tensor(float(condition_max) * 100.0, dtype=torch.float64))
        lo = torch.log10(torch.tensor(float(condition_max), dtype=torch.float64))
        score[finite] = ((hi - log_cond) / (hi - lo).clamp_min(1.0e-8)).clamp(0.0, 1.0)
    score[~finite] = 1.0
    return score


def _normal_error(raw_a: torch.Tensor, xty: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    grad = torch.matmul(raw_a, coeffs) - xty
    return grad.square().sum(dim=(1, 2)).sqrt()


def _accumulate_delta_energy(
    *,
    ids: torch.Tensor,
    delta: torch.Tensor,
    triangle_count: int,
    yty: torch.Tensor,
) -> None:
    if ids.ndim != 2 or delta.ndim != 3:
        raise RuntimeError(f"invalid ids/delta for energy accumulation: ids={tuple(ids.shape)} delta={tuple(delta.shape)}")
    valid = (ids >= 0) & (ids < int(triangle_count))
    if not bool(valid.any().item()):
        return
    local_ids = ids[valid].reshape(-1).long()
    values = delta.permute(1, 2, 0).contiguous()[valid].to(dtype=torch.float64)
    yty.index_add_(0, local_ids, values.square().sum(dim=1))


def _surface_counts_and_view_stats(
    *,
    xtx_flat: torch.Tensor,
    view_std_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = torch.round(xtx_flat[:, 0]).to(dtype=torch.int64)
    observed_mask = counts > 0
    view_means = torch.zeros((int(xtx_flat.shape[0]), 3), dtype=torch.float64)
    view_scales = torch.ones((int(xtx_flat.shape[0]), 3), dtype=torch.float64)
    if bool(observed_mask.any().item()):
        denom = counts[observed_mask].to(dtype=torch.float64).unsqueeze(1)
        view_sums = xtx_flat[:, 3:6]
        view_means[observed_mask] = view_sums[observed_mask] / denom
        view_sq = torch.stack([xtx_flat[:, 15], xtx_flat[:, 18], xtx_flat[:, 20]], dim=1)
        view_vars = torch.zeros_like(view_means)
        view_vars[observed_mask] = (view_sq[observed_mask] / denom) - view_means[observed_mask].square()
        view_scales[observed_mask] = view_vars[observed_mask].clamp_min(0.0).sqrt().clamp_min(float(view_std_floor))
    return counts, observed_mask, view_means, view_scales


def _solve_base_raw_coefficients(
    *,
    xtx_flat: torch.Tensor,
    xty: torch.Tensor,
    view_counts: torch.Tensor,
    min_count: int,
    min_views: int,
    ridge: float,
    view_std_floor: float,
    rank_rtol: float,
    condition_max: float,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    triangle_count = int(xtx_flat.shape[0])
    feature_count = len(BASIS_ORDER)
    counts, observed_mask, view_means, view_scales = _surface_counts_and_view_stats(
        xtx_flat=xtx_flat,
        view_std_floor=float(view_std_floor),
    )
    count_ok = counts >= int(min_count)
    views_ok = view_counts.to(dtype=torch.int64) >= int(min_views)
    centering_ok = observed_mask & torch.isfinite(view_means).all(dim=1) & torch.isfinite(view_scales).all(dim=1)
    candidate_ids_all = (count_ok & views_ok & centering_ok).nonzero(as_tuple=False).reshape(-1)

    fallback_coeffs = torch.zeros((triangle_count, feature_count, 3), dtype=torch.float32)
    _solve_fallback_affine_coefficients(
        xtx_flat=xtx_flat,
        xty=xty,
        tri_ids_all=observed_mask.nonzero(as_tuple=False).reshape(-1),
        ridge=float(ridge),
        coeffs=fallback_coeffs,
    )
    raw_coeffs = fallback_coeffs.clone()
    identity = torch.eye(feature_count, dtype=torch.float64)
    diagnostic_stats = _diagnostic_template(feature_count)
    solve_chunk_triangles = 131_072
    for start in tqdm(range(0, int(candidate_ids_all.numel()), solve_chunk_triangles), desc=desc):
        end = min(start + solve_chunk_triangles, int(candidate_ids_all.numel()))
        tri_ids = candidate_ids_all[start:end]
        if int(tri_ids.numel()) == 0:
            continue
        raw_a = _raw_xtx_matrix(xtx_flat, tri_ids)
        transform = _center_transform(view_means[tri_ids], view_scales[tri_ids])
        centered_a = torch.matmul(transform.transpose(1, 2), torch.matmul(raw_a, transform))
        ranks, conditions = _update_rank_condition_stats(centered_a, diagnostic_stats, float(rank_rtol))
        rank_score = (ranks.to(dtype=torch.float64) / float(feature_count)).clamp(0.0, 1.0)
        view_score = (view_counts[tri_ids].to(dtype=torch.float64) / float(max(1, min_views))).clamp(0.0, 1.0)
        cond_score = _condition_score(conditions, float(condition_max))
        solve_mask = (rank_score * view_score * cond_score) > 0.0
        if not bool(solve_mask.any().item()):
            continue
        tri_ids_ok = tri_ids[solve_mask]
        transform_ok = transform[solve_mask]
        a = centered_a[solve_mask]
        if float(ridge) > 0.0:
            a = a + float(ridge) * identity.unsqueeze(0)
        b = torch.matmul(transform_ok.transpose(1, 2), xty[tri_ids_ok])
        try:
            centered_sol, info = torch.linalg.solve_ex(a, b)
            assign = info == 0
        except AttributeError:
            centered_sol = torch.linalg.solve(a, b)
            assign = torch.ones((int(tri_ids_ok.numel()),), dtype=torch.bool)
        if bool(assign.any().item()):
            raw_coeffs[tri_ids_ok[assign]] = torch.matmul(transform_ok[assign], centered_sol[assign]).to(dtype=torch.float32)
    return fallback_coeffs, raw_coeffs, counts


def _risk_gain_for_ids(
    *,
    eval_xtx_flat: torch.Tensor,
    eval_xty: torch.Tensor,
    eval_yty: torch.Tensor,
    fit_counts: torch.Tensor,
    fit_fallback: torch.Tensor,
    fit_raw: torch.Tensor,
    tri_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gains = torch.zeros((int(tri_ids.numel()),), dtype=torch.float64)
    support = (torch.round(eval_xtx_flat[tri_ids, 0]).to(dtype=torch.int64) > 0) & (fit_counts[tri_ids] > 0)
    if not bool(support.any().item()):
        return gains, support
    local = tri_ids[support]
    raw_a = _raw_xtx_matrix(eval_xtx_flat, local)
    xty_local = eval_xty[local]
    yty_local = eval_yty[local].clamp_min(0.0)
    fallback = fit_fallback[local].to(dtype=torch.float64)
    raw = fit_raw[local].to(dtype=torch.float64)
    fallback_quad = torch.einsum("nfc,nfg,ngc->n", fallback, raw_a, fallback)
    raw_quad = torch.einsum("nfc,nfg,ngc->n", raw, raw_a, raw)
    fallback_lin = torch.einsum("nfc,nfc->n", fallback, xty_local)
    raw_lin = torch.einsum("nfc,nfc->n", raw, xty_local)
    fallback_sse = (yty_local - 2.0 * fallback_lin + fallback_quad).clamp_min(0.0)
    raw_sse = (yty_local - 2.0 * raw_lin + raw_quad).clamp_min(0.0)
    local_gain = ((fallback_sse - raw_sse) / fallback_sse.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    finite = torch.isfinite(local_gain)
    local_gain = torch.where(finite, local_gain, torch.zeros_like(local_gain))
    gains[support] = local_gain
    return gains, support


def _diagnostic_template(feature_count: int) -> dict[str, Any]:
    return {
        "rank_histogram": {str(rank): 0 for rank in range(feature_count + 1)},
        "condition_full_rank_count": 0,
        "condition_finite_count": 0,
        "condition_nonfinite_count": 0,
        "condition_finite_sum": 0.0,
        "condition_finite_max": 0.0,
        "condition_threshold_counts": {str(v): 0 for v in (1.0e4, 1.0e6, 1.0e8, 1.0e10)},
        "diagnostic_failures": 0,
    }


def _solve_mixture_coefficients(
    *,
    xtx_flat: torch.Tensor,
    xty: torch.Tensor,
    view_counts: torch.Tensor,
    crossfit_xtx_even: torch.Tensor | None,
    crossfit_xty_even: torch.Tensor | None,
    crossfit_yty_even: torch.Tensor | None,
    crossfit_view_counts_even: torch.Tensor | None,
    crossfit_xtx_odd: torch.Tensor | None,
    crossfit_xty_odd: torch.Tensor | None,
    crossfit_yty_odd: torch.Tensor | None,
    crossfit_view_counts_odd: torch.Tensor | None,
    gate_source: str,
    min_count: int,
    min_views: int,
    ridge: float,
    view_std_floor: float,
    rank_rtol: float,
    condition_max: float,
    gate_boost: float,
    residual_dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    triangle_count = int(xtx_flat.shape[0])
    feature_count = len(BASIS_ORDER)
    counts, observed_mask, view_means, view_scales = _surface_counts_and_view_stats(
        xtx_flat=xtx_flat,
        view_std_floor=float(view_std_floor),
    )
    count_ok = counts >= int(min_count)
    views_ok = view_counts.to(dtype=torch.int64) >= int(min_views)
    centering_ok = observed_mask & torch.isfinite(view_means).all(dim=1) & torch.isfinite(view_scales).all(dim=1)
    candidate_mask = count_ok & views_ok & centering_ok
    candidate_ids_all = candidate_mask.nonzero(as_tuple=False).reshape(-1)

    fallback_coeffs = torch.zeros((triangle_count, feature_count, 3), dtype=torch.float32)
    fallback_stats = _solve_fallback_affine_coefficients(
        xtx_flat=xtx_flat,
        xty=xty,
        tri_ids_all=observed_mask.nonzero(as_tuple=False).reshape(-1),
        ridge=float(ridge),
        coeffs=fallback_coeffs,
    )
    raw_coeffs = fallback_coeffs.clone()
    gates = torch.zeros((triangle_count,), dtype=torch.float32)
    gain_scores = torch.zeros((triangle_count,), dtype=torch.float32)
    stability_scores = torch.zeros((triangle_count,), dtype=torch.float32)
    mean_residuals = torch.zeros((triangle_count, 3), dtype=torch.float32)
    if bool(observed_mask.any().item()):
        mean_residuals[observed_mask] = (
            xty[observed_mask, 0, :] / counts[observed_mask].to(dtype=torch.float64).unsqueeze(1).clamp_min(1.0)
        ).to(dtype=torch.float32)

    if bool(observed_mask.any().item()):
        median_count = torch.median(counts[observed_mask].to(dtype=torch.float64)).clamp_min(1.0)
    else:
        median_count = torch.tensor(1.0, dtype=torch.float64)

    identity = torch.eye(feature_count, dtype=torch.float64)
    solve_chunk_triangles = 131_072
    diagnostic_stats = _diagnostic_template(feature_count)
    accepted = 0
    solve_failures = 0
    rank_condition_rejected = 0
    gate_positive = 0
    gate_sum = 0.0
    gain_sum = 0.0
    stability_sum = 0.0
    debt_guard_sum = 0.0
    crossfit_gain_sum = 0.0
    crossfit_gain_count = 0

    crossfit_ready = str(gate_source) == "crossfit_risk"
    if crossfit_ready:
        missing = [
            name
            for name, value in {
                "crossfit_xtx_even": crossfit_xtx_even,
                "crossfit_xty_even": crossfit_xty_even,
                "crossfit_yty_even": crossfit_yty_even,
                "crossfit_view_counts_even": crossfit_view_counts_even,
                "crossfit_xtx_odd": crossfit_xtx_odd,
                "crossfit_xty_odd": crossfit_xty_odd,
                "crossfit_yty_odd": crossfit_yty_odd,
                "crossfit_view_counts_odd": crossfit_view_counts_odd,
            }.items()
            if value is None
        ]
        if missing:
            raise RuntimeError("crossfit_risk gate requested but split accumulators are missing: " + ", ".join(missing))
        even_fallback, even_raw, even_counts = _solve_base_raw_coefficients(
            xtx_flat=crossfit_xtx_even,
            xty=crossfit_xty_even,
            view_counts=crossfit_view_counts_even,
            min_count=min_count,
            min_views=min_views,
            ridge=ridge,
            view_std_floor=view_std_floor,
            rank_rtol=rank_rtol,
            condition_max=condition_max,
            desc="solving v105 even split coefficients",
        )
        odd_fallback, odd_raw, odd_counts = _solve_base_raw_coefficients(
            xtx_flat=crossfit_xtx_odd,
            xty=crossfit_xty_odd,
            view_counts=crossfit_view_counts_odd,
            min_count=min_count,
            min_views=min_views,
            ridge=ridge,
            view_std_floor=view_std_floor,
            rank_rtol=rank_rtol,
            condition_max=condition_max,
            desc="solving v105 odd split coefficients",
        )

    for start in tqdm(
        range(0, int(candidate_ids_all.numel()), solve_chunk_triangles),
        desc="solving v105 evidence-gated mixture fields",
    ):
        end = min(start + solve_chunk_triangles, int(candidate_ids_all.numel()))
        tri_ids = candidate_ids_all[start:end]
        if int(tri_ids.numel()) == 0:
            continue
        raw_a = _raw_xtx_matrix(xtx_flat, tri_ids)
        transform = _center_transform(view_means[tri_ids], view_scales[tri_ids])
        centered_a = torch.matmul(transform.transpose(1, 2), torch.matmul(raw_a, transform))
        ranks, conditions = _update_rank_condition_stats(centered_a, diagnostic_stats, float(rank_rtol))
        rank_score = (ranks.to(dtype=torch.float64) / float(feature_count)).clamp(0.0, 1.0)
        view_score = (view_counts[tri_ids].to(dtype=torch.float64) / float(max(1, min_views))).clamp(0.0, 1.0)
        count_score = (torch.log1p(counts[tri_ids].to(dtype=torch.float64)) / torch.log1p(median_count)).clamp(0.0, 1.0)
        cond_score = _condition_score(conditions, float(condition_max))
        stability = (rank_score * view_score * count_score * cond_score).clamp(0.0, 1.0)
        solve_mask = stability > 0.0
        rank_condition_rejected += int((~solve_mask).sum().item())
        if not bool(solve_mask.any().item()):
            continue

        tri_ids_ok = tri_ids[solve_mask]
        transform_ok = transform[solve_mask]
        a = centered_a[solve_mask]
        if float(ridge) > 0.0:
            a = a + float(ridge) * identity.unsqueeze(0)
        b = torch.matmul(transform_ok.transpose(1, 2), xty[tri_ids_ok])
        try:
            centered_sol, info = torch.linalg.solve_ex(a, b)
            failed = info != 0
        except AttributeError:
            centered_sol = torch.linalg.solve(a, b)
            failed = torch.zeros((int(tri_ids_ok.numel()),), dtype=torch.bool)
        if bool(failed.any().item()):
            solve_failures += int(failed.sum().item())
        assign = ~failed
        if not bool(assign.any().item()):
            continue

        assigned_ids = tri_ids_ok[assign]
        assigned_raw = torch.matmul(transform_ok[assign], centered_sol[assign])
        assigned_raw_f32 = assigned_raw.to(dtype=torch.float32)
        raw_coeffs[assigned_ids] = assigned_raw_f32

        raw_a_assigned = raw_a[solve_mask][assign]
        xty_assigned = xty[assigned_ids]
        fallback_assigned = fallback_coeffs[assigned_ids].to(dtype=torch.float64)
        raw_assigned = assigned_raw.to(dtype=torch.float64)
        err_fallback = _normal_error(raw_a_assigned, xty_assigned, fallback_assigned)
        err_raw = _normal_error(raw_a_assigned, xty_assigned, raw_assigned)
        gain = ((err_fallback - err_raw) / err_fallback.clamp_min(1.0e-8)).clamp(0.0, 1.0)
        effective_gain = gain
        if crossfit_ready:
            even_to_odd_gain, even_to_odd_support = _risk_gain_for_ids(
                eval_xtx_flat=crossfit_xtx_odd,
                eval_xty=crossfit_xty_odd,
                eval_yty=crossfit_yty_odd,
                fit_counts=even_counts,
                fit_fallback=even_fallback,
                fit_raw=even_raw,
                tri_ids=assigned_ids,
            )
            odd_to_even_gain, odd_to_even_support = _risk_gain_for_ids(
                eval_xtx_flat=crossfit_xtx_even,
                eval_xty=crossfit_xty_even,
                eval_yty=crossfit_yty_even,
                fit_counts=odd_counts,
                fit_fallback=odd_fallback,
                fit_raw=odd_raw,
                tri_ids=assigned_ids,
            )
            support_count = even_to_odd_support.to(dtype=torch.float64) + odd_to_even_support.to(dtype=torch.float64)
            crossfit_gain = torch.zeros_like(gain)
            supported = support_count > 0.0
            if bool(supported.any().item()):
                crossfit_gain[supported] = (
                    even_to_odd_gain[supported] + odd_to_even_gain[supported]
                ) / support_count[supported]
                crossfit_gain_sum += float(crossfit_gain[supported].sum().item())
                crossfit_gain_count += int(supported.sum().item())
            effective_gain = crossfit_gain
        stable = stability[solve_mask][assign]
        debt_norm = (raw_assigned - fallback_assigned).square().mean(dim=(1, 2)).sqrt()
        fallback_norm = fallback_assigned.square().mean(dim=(1, 2)).sqrt()
        debt_guard = (fallback_norm / (fallback_norm + debt_norm).clamp_min(1.0e-8)).clamp(0.0, 1.0)
        # The gate is evidence-derived: stable triangles may recover residual debt toward the raw
        # view-affine expert, while weak or high-debt triangles remain close to the conservative
        # affine fallback. The debt guard reduces perceptual artifacts from large residual jumps.
        gate = (
            stable
            + float(gate_boost) * torch.sqrt(effective_gain).clamp(0.0, 1.0) * debt_guard * (1.0 - stable)
        ).clamp(0.0, 1.0)
        gates[assigned_ids] = gate.to(dtype=torch.float32)
        gain_scores[assigned_ids] = effective_gain.to(dtype=torch.float32)
        stability_scores[assigned_ids] = stable.to(dtype=torch.float32)
        accepted += int(assigned_ids.numel())
        gate_positive += int((gate > 0.0).sum().item())
        gate_sum += float(gate.sum().item())
        gain_sum += float(effective_gain.sum().item())
        stability_sum += float(stable.sum().item())
        debt_guard_sum += float(debt_guard.sum().item())

    finite_condition_count = int(diagnostic_stats["condition_finite_count"])
    condition_mean = (
        float(diagnostic_stats["condition_finite_sum"]) / float(finite_condition_count)
        if finite_condition_count > 0
        else 0.0
    )
    diagnostic_stats["condition_finite_mean"] = float(condition_mean)
    diagnostic_stats.pop("condition_finite_sum", None)

    valid_count = max(1, accepted)
    stats = {
        "observed_triangles": int(observed_mask.sum().item()),
        "empty_triangles": int((~observed_mask).sum().item()),
        "mixture_candidate_triangles": int(candidate_mask.sum().item()),
        "mixture_triangles": int(accepted),
        "fallback_only_triangles": int((observed_mask & (gates <= 0.0)).sum().item()),
        "rank_condition_rejected_triangles": int(rank_condition_rejected),
        "solve_failure_triangles": int(solve_failures),
        "gate_positive_triangles": int(gate_positive),
        "gate_mean": float(gate_sum / valid_count),
        "gain_score_mean": float(gain_sum / valid_count),
        "stability_score_mean": float(stability_sum / valid_count),
        "debt_guard_mean": float(debt_guard_sum / valid_count),
        "gate_source": str(gate_source),
        "crossfit_gain_mean": float(crossfit_gain_sum / max(1, crossfit_gain_count)),
        "crossfit_gain_supported_triangles": int(crossfit_gain_count),
        "min_count": int(min_count),
        "min_views": int(min_views),
        "ridge": float(ridge),
        "view_std_floor": float(view_std_floor),
        "rank_rtol": float(rank_rtol),
        "condition_max": float(condition_max),
        "gate_boost": float(gate_boost),
        "centered_view_feature_diagnostics": diagnostic_stats,
    }
    stats.update(fallback_stats)
    delta_coeffs = raw_coeffs - fallback_coeffs
    return (
        fallback_coeffs.to(dtype=residual_dtype).contiguous(),
        delta_coeffs.to(dtype=residual_dtype).contiguous(),
        mean_residuals.to(dtype=residual_dtype).contiguous(),
        gates.to(dtype=residual_dtype).contiguous(),
        gain_scores.to(dtype=residual_dtype).contiguous(),
        stability_scores.to(dtype=residual_dtype).contiguous(),
        view_means.to(dtype=residual_dtype).contiguous(),
        view_scales.to(dtype=residual_dtype).contiguous(),
        counts.to(dtype=torch.int32).contiguous(),
        view_counts.to(dtype=torch.int32).contiguous(),
        stats,
    )


def build_field(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model_path)
    delta_bank_path = Path(args.delta_bank_path)
    output_field = Path(args.output_field)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not delta_bank_path.is_file():
        raise FileNotFoundError(delta_bank_path)
    if int(args.renderer_scaling) <= 0 or int(args.chunk_pixels) <= 0:
        raise ValueError("--renderer_scaling and --chunk_pixels must be positive")
    if int(args.min_count) <= 0 or int(args.min_views) <= 0:
        raise ValueError("--min_count and --min_views must be positive")
    if float(args.ridge) < 0.0 or float(args.residual_clip) < 0.0:
        raise ValueError("--ridge and --residual_clip must be non-negative")
    if not (0.0 <= float(args.gate_boost) <= 1.0):
        raise ValueError("--gate_boost must be in [0, 1]")
    if str(args.gate_source) not in {"normal_equation", "crossfit_risk"}:
        raise ValueError("--gate_source must be normal_equation or crossfit_risk")

    residual_dtype = _dtype(args.residual_dtype)
    delta_bank = _load_delta_bank(delta_bank_path, str(args.split), str(args.endpoint_method))
    deltas = delta_bank["deltas"]
    frames = delta_bank["frames"]
    dataset, pipe, triangles, scene, background = _load_scene(model_path, int(args.iteration), int(args.renderer_scaling))
    views = scene.getTestCameras() if str(args.split) == "test" else scene.getTrainCameras()
    if len(deltas) != len(views):
        raise RuntimeError(f"delta bank/view count mismatch: bank={len(deltas)} views={len(views)}")
    extra_frame_meta = [str(key) for key in frames.keys() if str(key) not in deltas]
    if extra_frame_meta:
        raise RuntimeError(f"delta bank has extra frame metadata entries: {len(extra_frame_meta)}")

    faces = triangles.get_triangle_indices.detach().cpu().long().contiguous()
    vertices = triangles.get_vertices.detach().cpu().float().contiguous()
    face_centers = vertices[faces].mean(dim=1).contiguous()
    triangle_count = int(faces.shape[0])
    feature_count = len(BASIS_ORDER)
    xtx_flat = torch.zeros((triangle_count, feature_count * (feature_count + 1) // 2), dtype=torch.float64)
    xty = torch.zeros((triangle_count, feature_count, 3), dtype=torch.float64)
    view_counts = torch.zeros((triangle_count,), dtype=torch.int32)
    use_crossfit = str(args.gate_source) == "crossfit_risk"
    xtx_even = xty_even = yty_even = view_counts_even = None
    xtx_odd = xty_odd = yty_odd = view_counts_odd = None
    if use_crossfit:
        xtx_even = torch.zeros_like(xtx_flat)
        xty_even = torch.zeros_like(xty)
        yty_even = torch.zeros((triangle_count,), dtype=torch.float64)
        view_counts_even = torch.zeros_like(view_counts)
        xtx_odd = torch.zeros_like(xtx_flat)
        xty_odd = torch.zeros_like(xty)
        yty_odd = torch.zeros((triangle_count,), dtype=torch.float64)
        view_counts_odd = torch.zeros_like(view_counts)
    view_reports: list[dict[str, Any]] = []
    started = time.time()

    for idx, view in enumerate(tqdm(views, desc=f"v105 evidence-gated mixture field {args.split}")):
        key = f"{idx:05d}"
        if key not in deltas:
            raise RuntimeError(f"missing delta for target frame {key}")
        _assert_strict_camera_matches_bank(_camera_record(idx, view), frames.get(key, {}).get("target_camera", {}), key)
        with torch.no_grad():
            pkg = render(view, triangles, pipe, background)
        rendering = pkg["render"]
        if "rend_ids" not in pkg or pkg["rend_ids"] is None:
            raise RuntimeError("renderer package missing rend_ids; cannot build v105 field")
        if "image_2D" not in pkg or pkg["image_2D"] is None:
            raise RuntimeError("renderer package missing image_2D; cannot build v105 field")
        ids = _downsample_rend_ids_nearest(pkg["rend_ids"], rendering.shape[-2:]).detach().cpu().long()
        delta = deltas[key].detach().cpu().float()
        if tuple(delta.shape) != tuple(rendering.shape):
            raise RuntimeError(
                f"delta/render shape mismatch for {key}: delta={tuple(delta.shape)} render={tuple(rendering.shape)}"
            )
        report = _accumulate_view_affine_view(
            ids=ids,
            delta=delta,
            projected_xy=pkg["image_2D"].detach().cpu().float().contiguous(),
            faces=faces,
            face_centers=face_centers,
            camera_center=view.camera_center.detach().cpu().float(),
            xtx_flat=xtx_flat,
            xty=xty,
            view_counts=view_counts,
            chunk_pixels=int(args.chunk_pixels),
        )
        if use_crossfit:
            split_xtx = xtx_even if idx % 2 == 0 else xtx_odd
            split_xty = xty_even if idx % 2 == 0 else xty_odd
            split_yty = yty_even if idx % 2 == 0 else yty_odd
            split_view_counts = view_counts_even if idx % 2 == 0 else view_counts_odd
            _accumulate_view_affine_view(
                ids=ids,
                delta=delta,
                projected_xy=pkg["image_2D"].detach().cpu().float().contiguous(),
                faces=faces,
                face_centers=face_centers,
                camera_center=view.camera_center.detach().cpu().float(),
                xtx_flat=split_xtx,
                xty=split_xty,
                view_counts=split_view_counts,
                chunk_pixels=int(args.chunk_pixels),
            )
            _accumulate_delta_energy(
                ids=ids,
                delta=delta,
                triangle_count=triangle_count,
                yty=split_yty,
            )
        report.update({"frame": key, "mean_abs_delta": float(delta.abs().mean().item()), "camera_validated": True})
        view_reports.append(report)
        del pkg, rendering, ids, delta

    (
        base_coeffs,
        delta_coeffs,
        mean_residuals,
        gates,
        gain_scores,
        stability_scores,
        view_means,
        view_scales,
        counts_out,
        view_counts_out,
        solve_stats,
    ) = _solve_mixture_coefficients(
        xtx_flat=xtx_flat,
        xty=xty,
        view_counts=view_counts,
        crossfit_xtx_even=xtx_even,
        crossfit_xty_even=xty_even,
        crossfit_yty_even=yty_even,
        crossfit_view_counts_even=view_counts_even,
        crossfit_xtx_odd=xtx_odd,
        crossfit_xty_odd=xty_odd,
        crossfit_yty_odd=yty_odd,
        crossfit_view_counts_odd=view_counts_odd,
        gate_source=str(args.gate_source),
        min_count=int(args.min_count),
        min_views=int(args.min_views),
        ridge=float(args.ridge),
        view_std_floor=float(args.view_std_floor),
        rank_rtol=float(args.rank_rtol),
        condition_max=float(args.condition_max),
        gate_boost=float(args.gate_boost),
        residual_dtype=residual_dtype,
    )

    valid_mask = counts_out > 0
    render_valid_mask = counts_out >= int(args.min_count)
    endpoint_report = str(delta_bank.get("endpoint_report", "") or "")
    endpoint_report_sha = str(delta_bank.get("endpoint_report_sha256", "") or "")
    source_delta_bank_sha = _sha256(delta_bank_path)
    total_valid_pixels = int(sum(int(row["valid_pixels"]) for row in view_reports))
    total_accumulated_pixels = int(sum(int(row["accumulated_pixels"]) for row in view_reports))
    args_manifest = _jsonable_args(args)
    payload = {
        "schema_version": 1,
        "field_type": "v102_surface_residual_field",
        "basis_type": "affine_barycentric_viewdir_mixture",
        "builder_variant": "v105_evidence_gated_residual_mixture",
        "basis_order": BASIS_ORDER,
        "coefficient_layout": "triangle,basis,rgb",
        "created_at_unix": time.time(),
        "model_path": str(model_path),
        "split": str(args.split),
        "iteration": int(args.iteration),
        "endpoint_method": str(delta_bank.get("endpoint_method", args.endpoint_method)),
        "source_bank_split": str(delta_bank.get("split", "") or ""),
        "endpoint_report": endpoint_report,
        "endpoint_report_sha256": endpoint_report_sha,
        "source_delta_bank": str(delta_bank_path),
        "source_delta_bank_sha256": source_delta_bank_sha,
        "source_target_frames": int(len(views)),
        "triangle_count": int(triangle_count),
        "valid_triangles": int(valid_mask.sum().item()),
        "render_min_count_valid_triangles": int(render_valid_mask.sum().item()),
        "valid_triangle_mask": valid_mask.to(dtype=torch.bool).contiguous(),
        "min_count": int(args.min_count),
        "min_views": int(args.min_views),
        "ridge": float(args.ridge),
        "view_std_floor": float(args.view_std_floor),
        "rank_rtol": float(args.rank_rtol),
        "condition_max": float(args.condition_max),
        "gate_boost": float(args.gate_boost),
        "gate_source": str(args.gate_source),
        "view_gate_temperature": float(args.view_gate_temperature),
        "renderer_scaling": int(args.renderer_scaling),
        "residual_clip": float(args.residual_clip),
        "residual_dtype": str(args.residual_dtype),
        "triangle_base_coefficients": base_coeffs,
        "triangle_delta_coefficients": delta_coeffs,
        "triangle_residuals": mean_residuals,
        "triangle_gate": gates,
        "triangle_gain_score": gain_scores,
        "triangle_stability_score": stability_scores,
        "triangle_view_means": view_means,
        "triangle_view_scales": view_scales,
        "triangle_counts": counts_out,
        "triangle_view_counts": view_counts_out,
        "normal_equation_xtx_order": [f"{BASIS_ORDER[i]}*{BASIS_ORDER[j]}" for i, j in XTX_ORDER],
        "normal_equation_xty_layout": "triangle,basis,rgb",
        "total_valid_pixels": int(total_valid_pixels),
        "total_accumulated_pixels": int(total_accumulated_pixels),
        "solve_stats": solve_stats,
        "view_reports": view_reports,
        "camera_validation": "strict_target_camera_match",
        "elapsed_sec": float(time.time() - started),
        "args": args_manifest,
        "note": (
            "v105 evidence-gated residual-mixture field distilled from v102 target-camera deltas. The field stores "
            "a conservative barycentric affine fallback expert, a view-affine residual-debt expert, and an "
            "evidence-derived gate from support, conditioning, and either cross-fitted teacher risk or teacher "
            "normal-equation gain. It uses no "
            "held-out target GT for the policy, but remains a target-camera endpoint distillation sidecar."
        ),
    }
    output_field.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_field)
    manifest = {
        "schema_version": 1,
        "field_path": str(output_field),
        "field_sha256": _sha256(output_field),
        "field_type": payload["field_type"],
        "basis_type": payload["basis_type"],
        "builder_variant": payload["builder_variant"],
        "basis_order": payload["basis_order"],
        "coefficient_layout": payload["coefficient_layout"],
        "source_delta_bank": str(delta_bank_path),
        "source_delta_bank_sha256": source_delta_bank_sha,
        "triangle_count": int(triangle_count),
        "valid_triangles": int(valid_mask.sum().item()),
        "valid_triangle_fraction": float(valid_mask.float().mean().item()) if int(triangle_count) else 0.0,
        "render_min_count_valid_triangles": int(render_valid_mask.sum().item()),
        "render_min_count_valid_triangle_fraction": (
            float(render_valid_mask.float().mean().item()) if int(triangle_count) else 0.0
        ),
        "source_target_frames": int(len(views)),
        "min_count": int(args.min_count),
        "min_views": int(args.min_views),
        "ridge": float(args.ridge),
        "view_std_floor": float(args.view_std_floor),
        "rank_rtol": float(args.rank_rtol),
        "condition_max": float(args.condition_max),
        "gate_boost": float(args.gate_boost),
        "gate_source": str(args.gate_source),
        "view_gate_temperature": float(args.view_gate_temperature),
        "renderer_scaling": int(args.renderer_scaling),
        "residual_clip": float(args.residual_clip),
        "residual_dtype": str(args.residual_dtype),
        "endpoint_method": str(delta_bank.get("endpoint_method", args.endpoint_method)),
        "source_bank_split": str(delta_bank.get("split", "") or ""),
        "total_valid_pixels": int(total_valid_pixels),
        "total_accumulated_pixels": int(total_accumulated_pixels),
        "solve_stats": solve_stats,
        "args": args_manifest,
        "camera_validation": payload["camera_validation"],
        "elapsed_sec": float(time.time() - started),
        "note": payload["note"],
    }
    manifest_path = output_field.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "field": str(output_field),
                "manifest": str(manifest_path),
                "basis_type": payload["basis_type"],
                "valid_triangles": manifest["valid_triangles"],
                "mixture_triangles": solve_stats.get("mixture_triangles", 0),
                "gate_mean": solve_stats.get("gate_mean", 0.0),
                "total_accumulated_pixels": manifest["total_accumulated_pixels"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a v105 evidence-gated residual-mixture surface field.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--delta_bank_path", required=True)
    parser.add_argument("--output_field", required=True)
    parser.add_argument("--endpoint_method", default="ours_26000_v100_checkpoint_attached_ela_endpoint")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--split", default="test", choices=("test", "train"))
    parser.add_argument("--renderer_scaling", type=int, required=True)
    parser.add_argument("--residual_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--min_views", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--residual_clip", type=float, default=0.08)
    parser.add_argument("--view_std_floor", type=float, default=1e-4)
    parser.add_argument("--rank_rtol", type=float, default=1e-7)
    parser.add_argument("--condition_max", type=float, default=1e8)
    parser.add_argument("--gate_boost", type=float, default=0.5)
    parser.add_argument("--gate_source", default="crossfit_risk", choices=("normal_equation", "crossfit_risk"))
    parser.add_argument("--view_gate_temperature", type=float, default=0.0)
    parser.add_argument("--chunk_pixels", type=int, default=262144)
    return parser.parse_args()


def main() -> int:
    build_field(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
