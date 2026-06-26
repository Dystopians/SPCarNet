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
    _solve_view_affine_coefficients,
    _update_rank_condition_stats,
)


V108_METHOD_VERSION = "v108_mse_descent_locked_pod_moe"
V108_BUILDER_VARIANT = "v108_mse_descent_locked_pod_moe"
V108_EXPERT_MSE_CERTIFICATE = "joint_two_expert_weighted_normal_equation_box_qp_descent_lock"


def _resolved_method_version(field_variant: str, gate_source: str, requested: str = "auto") -> str:
    if str(requested) == V108_METHOD_VERSION:
        return V108_METHOD_VERSION
    if str(field_variant) == "pod_moe" and str(gate_source) == "crossfit_risk":
        return "v107_crossfit_pod_moe_expert_reliability"
    if str(field_variant) == "pod_moe":
        return "v106_perceptual_occlusion_detail_moe"
    return "v105_evidence_gated_residual_mixture"


def _builder_variant(field_variant: str, method_version: str) -> str:
    if str(method_version) == V108_METHOD_VERSION:
        return V108_BUILDER_VARIANT
    return "v106_perceptual_occlusion_detail_moe" if str(field_variant) == "pod_moe" else "v105_evidence_gated_residual_mixture"


def _select_indexed_views(views: list[Any], view_subset: str) -> list[tuple[int, Any]]:
    subset = str(view_subset)
    if subset == "all":
        return list(enumerate(views))
    if subset == "even":
        parity = 0
    elif subset == "odd":
        parity = 1
    else:
        raise ValueError(f"Unsupported view subset: {view_subset}")
    return [(idx, view) for idx, view in enumerate(views) if idx % 2 == parity]


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


def _robust_unit_score(score: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    score = torch.nan_to_num(score.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    sample = score[valid] if valid is not None and bool(valid.any().item()) else score.reshape(-1)
    sample = sample[sample > 0.0]
    if int(sample.numel()) == 0:
        return torch.zeros_like(score)
    scale = torch.quantile(sample, 0.90).clamp_min(1.0e-6)
    return (score / scale).clamp(0.0, 1.0)


def _luminance_detail_score(rendering: torch.Tensor) -> torch.Tensor:
    lum = 0.299 * rendering[0].detach().cpu().float() + 0.587 * rendering[1].detach().cpu().float() + 0.114 * rendering[2].detach().cpu().float()
    score = torch.zeros_like(lum)
    score[:, 1:] = torch.maximum(score[:, 1:], (lum[:, 1:] - lum[:, :-1]).abs())
    score[:, :-1] = torch.maximum(score[:, :-1], (lum[:, 1:] - lum[:, :-1]).abs())
    score[1:, :] = torch.maximum(score[1:, :], (lum[1:, :] - lum[:-1, :]).abs())
    score[:-1, :] = torch.maximum(score[:-1, :], (lum[1:, :] - lum[:-1, :]).abs())
    return _robust_unit_score(score)


def _delta_detail_score(delta: torch.Tensor) -> torch.Tensor:
    mag = delta.detach().cpu().float().abs().mean(dim=0)
    score = torch.zeros_like(mag)
    score[:, 1:] = torch.maximum(score[:, 1:], (mag[:, 1:] - mag[:, :-1]).abs())
    score[:, :-1] = torch.maximum(score[:, :-1], (mag[:, 1:] - mag[:, :-1]).abs())
    score[1:, :] = torch.maximum(score[1:, :], (mag[1:, :] - mag[:-1, :]).abs())
    score[:-1, :] = torch.maximum(score[:-1, :], (mag[1:, :] - mag[:-1, :]).abs())
    return _robust_unit_score(score)


def _boundary_score(ids: torch.Tensor, depth: torch.Tensor | None = None) -> torch.Tensor:
    boundary = torch.zeros_like(ids, dtype=torch.float32)
    id_valid = ids >= 0
    diff_x = (ids[:, 1:] != ids[:, :-1]) & id_valid[:, 1:] & id_valid[:, :-1]
    diff_y = (ids[1:, :] != ids[:-1, :]) & id_valid[1:, :] & id_valid[:-1, :]
    boundary[:, 1:] = torch.maximum(boundary[:, 1:], diff_x.float())
    boundary[:, :-1] = torch.maximum(boundary[:, :-1], diff_x.float())
    boundary[1:, :] = torch.maximum(boundary[1:, :], diff_y.float())
    boundary[:-1, :] = torch.maximum(boundary[:-1, :], diff_y.float())
    if depth is not None:
        d = depth.detach().cpu().float()
        if d.ndim == 3:
            d = d[0]
        if tuple(d.shape) == tuple(ids.shape):
            depth_score = torch.zeros_like(d)
            depth_score[:, 1:] = torch.maximum(depth_score[:, 1:], (d[:, 1:] - d[:, :-1]).abs())
            depth_score[:, :-1] = torch.maximum(depth_score[:, :-1], (d[:, 1:] - d[:, :-1]).abs())
            depth_score[1:, :] = torch.maximum(depth_score[1:, :], (d[1:, :] - d[:-1, :]).abs())
            depth_score[:-1, :] = torch.maximum(depth_score[:-1, :], (d[1:, :] - d[:-1, :]).abs())
            boundary = torch.maximum(boundary, _robust_unit_score(depth_score, id_valid))
    return boundary.clamp(0.0, 1.0)


def _accumulate_weighted_view_affine_view(
    *,
    ids: torch.Tensor,
    delta: torch.Tensor,
    weights: torch.Tensor,
    projected_xy: torch.Tensor,
    faces: torch.Tensor,
    face_centers: torch.Tensor,
    camera_center: torch.Tensor,
    xtx_flat: torch.Tensor,
    xty: torch.Tensor,
    yty: torch.Tensor,
    view_counts: torch.Tensor,
    chunk_pixels: int,
) -> dict[str, Any]:
    if ids.ndim != 2 or weights.ndim != 2:
        raise RuntimeError(f"expected 2D ids/weights, got ids={tuple(ids.shape)} weights={tuple(weights.shape)}")
    if delta.ndim != 3 or int(delta.shape[0]) != 3:
        raise RuntimeError(f"expected delta shape [3,H,W], got {tuple(delta.shape)}")
    if tuple(delta.shape[-2:]) != tuple(ids.shape) or tuple(weights.shape) != tuple(ids.shape):
        raise RuntimeError("weighted accumulation shape mismatch")

    triangle_count = int(faces.shape[0])
    valid = (ids >= 0) & (ids < triangle_count) & torch.isfinite(weights) & (weights > 1.0e-6)
    total_pixels = int(ids.numel())
    valid_pixels = int(valid.sum().item())
    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "accumulated_pixels": 0,
            "valid_fraction": 0.0,
            "accumulated_fraction": 0.0,
            "unique_triangles": 0,
            "weight_sum": 0.0,
        }

    pixel_yx = valid.nonzero(as_tuple=False)
    flat_ids_all = ids[valid].reshape(-1).long()
    weight_all = weights[valid].reshape(-1).to(dtype=torch.float64).clamp(0.0, 1.0)
    unique_triangles = int(torch.unique(flat_ids_all).numel())
    delta_hwc = delta.permute(1, 2, 0).contiguous()
    chunk = max(1, int(chunk_pixels))
    accumulated_pixels = 0
    weight_sum = 0.0
    vertex_count = int(projected_xy.shape[0])
    view_seen = torch.zeros((triangle_count,), dtype=torch.bool)

    for start in range(0, valid_pixels, chunk):
        end = min(start + chunk, valid_pixels)
        local_ids = flat_ids_all[start:end]
        local_yx = pixel_yx[start:end]
        local_w = weight_all[start:end]
        vertex_ids = faces[local_ids]
        vertex_ok = (vertex_ids >= 0).all(dim=1) & (vertex_ids < vertex_count).all(dim=1)
        if not bool(vertex_ok.any().item()):
            continue
        if not bool(vertex_ok.all().item()):
            local_ids = local_ids[vertex_ok]
            local_yx = local_yx[vertex_ok]
            local_w = local_w[vertex_ok]
            vertex_ids = vertex_ids[vertex_ok]

        xy = projected_xy[vertex_ids]
        p = torch.stack(
            [local_yx[:, 1].to(dtype=torch.float32), local_yx[:, 0].to(dtype=torch.float32)],
            dim=1,
        )
        a = xy[:, 0, :]
        b = xy[:, 1, :]
        c = xy[:, 2, :]
        denom = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        safe = denom.abs() > 1e-8
        if not bool(safe.any().item()):
            continue
        safe_denom = torch.where(safe, denom, torch.ones_like(denom))
        w0 = ((b[:, 1] - c[:, 1]) * (p[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (p[:, 1] - c[:, 1])) / safe_denom
        w1 = ((c[:, 1] - a[:, 1]) * (p[:, 0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (p[:, 1] - c[:, 1])) / safe_denom
        direction = camera_center.unsqueeze(0) - face_centers[local_ids]
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
        finite = safe & torch.isfinite(w0) & torch.isfinite(w1) & torch.isfinite(direction).all(dim=1)
        if not bool(finite.any().item()):
            continue
        if not bool(finite.all().item()):
            local_ids = local_ids[finite]
            local_yx = local_yx[finite]
            local_w = local_w[finite]
            w0 = w0[finite]
            w1 = w1[finite]
            direction = direction[finite]

        one = torch.ones_like(w0, dtype=torch.float64)
        values = delta_hwc[local_yx[:, 0], local_yx[:, 1], :].to(dtype=torch.float64)
        basis = torch.cat(
            [
                one[:, None],
                w0.to(dtype=torch.float64)[:, None],
                w1.to(dtype=torch.float64)[:, None],
                direction.to(dtype=torch.float64),
            ],
            dim=1,
        )
        weighted_basis = basis * local_w[:, None]
        xtx_values = torch.stack([weighted_basis[:, i] * basis[:, j] for i, j in XTX_ORDER], dim=1)
        xty_values = weighted_basis[:, :, None] * values[:, None, :]
        xtx_flat.index_add_(0, local_ids, xtx_values)
        xty.index_add_(0, local_ids, xty_values)
        yty.index_add_(0, local_ids, local_w * values.square().sum(dim=1))
        view_seen[torch.unique(local_ids)] = True
        accumulated_pixels += int(local_ids.numel())
        weight_sum += float(local_w.sum().item())

    if bool(view_seen.any().item()):
        view_counts += view_seen.to(dtype=view_counts.dtype)

    return {
        "valid_pixels": int(valid_pixels),
        "accumulated_pixels": int(accumulated_pixels),
        "valid_fraction": float(valid_pixels / max(1, total_pixels)),
        "accumulated_fraction": float(accumulated_pixels / max(1, total_pixels)),
        "unique_triangles": int(unique_triangles),
        "weight_sum": float(weight_sum),
    }


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    solve_success = torch.zeros((triangle_count,), dtype=torch.bool)
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
            solved_ids = tri_ids_ok[assign]
            raw_coeffs[solved_ids] = torch.matmul(transform_ok[assign], centered_sol[assign]).to(dtype=torch.float32)
            solve_success[solved_ids] = True
    return fallback_coeffs, raw_coeffs, counts, solve_success


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


def _weighted_risk_gain_and_scale(
    *,
    eval_xtx_flat: torch.Tensor,
    eval_xty: torch.Tensor,
    eval_yty: torch.Tensor,
    base_coeffs: torch.Tensor,
    expert_coeffs: torch.Tensor,
    tri_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gains = torch.zeros((int(tri_ids.numel()),), dtype=torch.float64)
    full_gains = torch.zeros((int(tri_ids.numel()),), dtype=torch.float64)
    scales = torch.zeros((int(tri_ids.numel()),), dtype=torch.float64)
    support = torch.round(eval_xtx_flat[tri_ids, 0]).to(dtype=torch.int64) > 0
    if not bool(support.any().item()):
        return gains, full_gains, scales
    local = tri_ids[support]
    raw_a = _raw_xtx_matrix(eval_xtx_flat, local)
    xty_local = eval_xty[local]
    yty_local = eval_yty[local].clamp_min(0.0)
    base = base_coeffs[local].to(dtype=torch.float64)
    expert = expert_coeffs[local].to(dtype=torch.float64)
    base_quad = torch.einsum("nfc,nfg,ngc->n", base, raw_a, base)
    expert_quad = torch.einsum("nfc,nfg,ngc->n", expert, raw_a, expert)
    base_lin = torch.einsum("nfc,nfc->n", base, xty_local)
    expert_lin = torch.einsum("nfc,nfc->n", expert, xty_local)
    base_sse = (yty_local - 2.0 * base_lin + base_quad).clamp_min(0.0)
    expert_sse = (yty_local - 2.0 * expert_lin + expert_quad).clamp_min(0.0)
    delta = expert - base
    quad = torch.einsum("nfc,nfg,ngc->n", delta, raw_a, delta).clamp_min(1.0e-12)
    lin = torch.einsum("nfc,nfc->n", delta, torch.matmul(raw_a, base) - xty_local)
    scale = (-lin / quad).clamp(0.0, 1.0)
    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
    optimal_delta = 2.0 * scale * lin + scale.square() * quad
    full_gain = ((base_sse - expert_sse) / base_sse.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    optimal_gain = (-optimal_delta / base_sse.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    gains[support] = torch.where(torch.isfinite(optimal_gain), optimal_gain, torch.zeros_like(optimal_gain))
    full_gains[support] = torch.where(torch.isfinite(full_gain), full_gain, torch.zeros_like(full_gain))
    scales[support] = scale
    return gains, full_gains, scales


def _edge_box_minimizer(linear: torch.Tensor, quad: torch.Tensor) -> torch.Tensor:
    safe_quad = quad > 1.0e-12
    unconstrained = (-linear / quad.clamp_min(1.0e-12)).clamp(0.0, 1.0)
    boundary = torch.where(linear < 0.0, torch.ones_like(linear), torch.zeros_like(linear))
    return torch.where(safe_quad, unconstrained, boundary)


def _joint_two_expert_mse_descent_lock(
    *,
    eval_xtx_flat: torch.Tensor,
    eval_xty: torch.Tensor,
    eval_yty: torch.Tensor,
    base_coeffs: torch.Tensor,
    expert_delta: torch.Tensor,
    tri_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scales = torch.zeros((int(tri_ids.numel()), 2), dtype=torch.float64)
    gains = torch.zeros((int(tri_ids.numel()),), dtype=torch.float64)
    objective_delta = torch.zeros((int(tri_ids.numel()),), dtype=torch.float64)
    support = eval_xtx_flat[tri_ids, 0].to(dtype=torch.float64) > 1.0e-8
    if not bool(support.any().item()):
        return scales, gains, objective_delta, support

    local = tri_ids[support]
    raw_a = _raw_xtx_matrix(eval_xtx_flat, local)
    xty_local = eval_xty[local]
    yty_local = eval_yty[local].clamp_min(0.0)
    base = base_coeffs[local].to(dtype=torch.float64)
    d0 = expert_delta[local, 0].to(dtype=torch.float64)
    d1 = expert_delta[local, 1].to(dtype=torch.float64)
    grad = torch.matmul(raw_a, base) - xty_local

    q00 = torch.einsum("nfc,nfg,ngc->n", d0, raw_a, d0).clamp_min(0.0)
    q11 = torch.einsum("nfc,nfg,ngc->n", d1, raw_a, d1).clamp_min(0.0)
    q01 = torch.einsum("nfc,nfg,ngc->n", d0, raw_a, d1)
    l0 = torch.einsum("nfc,nfc->n", d0, grad)
    l1 = torch.einsum("nfc,nfc->n", d1, grad)

    n = int(local.numel())
    zero = torch.zeros((n,), dtype=torch.float64)
    one = torch.ones((n,), dtype=torch.float64)
    candidates = [
        torch.stack([zero, zero], dim=1),
        torch.stack([one, zero], dim=1),
        torch.stack([zero, one], dim=1),
        torch.stack([one, one], dim=1),
        torch.stack([_edge_box_minimizer(l0, q00), zero], dim=1),
        torch.stack([_edge_box_minimizer(l0 + q01, q00), one], dim=1),
        torch.stack([zero, _edge_box_minimizer(l1, q11)], dim=1),
        torch.stack([one, _edge_box_minimizer(l1 + q01, q11)], dim=1),
    ]

    det = q00 * q11 - q01.square()
    interior0 = (-q11 * l0 + q01 * l1) / det.clamp_min(1.0e-12)
    interior1 = (q01 * l0 - q00 * l1) / det.clamp_min(1.0e-12)
    interior_ok = (
        (det > 1.0e-12)
        & torch.isfinite(interior0)
        & torch.isfinite(interior1)
        & (interior0 >= 0.0)
        & (interior0 <= 1.0)
        & (interior1 >= 0.0)
        & (interior1 <= 1.0)
    )
    candidates.append(
        torch.stack(
            [
                torch.where(interior_ok, interior0, zero),
                torch.where(interior_ok, interior1, zero),
            ],
            dim=1,
        )
    )

    cand = torch.stack(candidates, dim=1)
    lam0 = cand[:, :, 0]
    lam1 = cand[:, :, 1]
    delta = (
        2.0 * (lam0 * l0[:, None] + lam1 * l1[:, None])
        + lam0.square() * q00[:, None]
        + 2.0 * lam0 * lam1 * q01[:, None]
        + lam1.square() * q11[:, None]
    )
    delta = torch.where(torch.isfinite(delta), delta, torch.full_like(delta, float("inf")))
    best_idx = torch.argmin(delta, dim=1)
    row = torch.arange(n)
    best_delta = delta[row, best_idx]
    best_scale = cand[row, best_idx]
    non_increase = best_delta <= 1.0e-10
    best_scale = torch.where(non_increase[:, None], best_scale, torch.zeros_like(best_scale))
    best_delta = torch.where(non_increase, best_delta.clamp_max(0.0), torch.zeros_like(best_delta))

    base_quad = torch.einsum("nfc,nfg,ngc->n", base, raw_a, base)
    base_lin = torch.einsum("nfc,nfc->n", base, xty_local)
    base_sse = (yty_local - 2.0 * base_lin + base_quad).clamp_min(0.0)
    local_gain = (-best_delta / base_sse.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    local_gain = torch.where(torch.isfinite(local_gain), local_gain, torch.zeros_like(local_gain))

    scales[support] = best_scale
    gains[support] = local_gain
    objective_delta[support] = best_delta
    return scales, gains, objective_delta, support


def _crossfit_weighted_risk_gain_and_scale(
    *,
    name: str,
    base_coeffs: torch.Tensor,
    tri_ids: torch.Tensor,
    xtx_even: torch.Tensor,
    xty_even: torch.Tensor,
    yty_even: torch.Tensor,
    view_counts_even: torch.Tensor,
    xtx_odd: torch.Tensor,
    xty_odd: torch.Tensor,
    yty_odd: torch.Tensor,
    view_counts_odd: torch.Tensor,
    min_count: int,
    min_views: int,
    ridge: float,
    view_std_floor: float,
    rank_rtol: float,
    condition_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    even_fallback, even_raw, even_counts, even_solve_success = _solve_base_raw_coefficients(
        xtx_flat=xtx_even,
        xty=xty_even,
        view_counts=view_counts_even,
        min_count=min_count,
        min_views=min_views,
        ridge=ridge,
        view_std_floor=view_std_floor,
        rank_rtol=rank_rtol,
        condition_max=condition_max,
        desc=f"solving v107 {name} even split expert",
    )
    odd_fallback, odd_raw, odd_counts, odd_solve_success = _solve_base_raw_coefficients(
        xtx_flat=xtx_odd,
        xty=xty_odd,
        view_counts=view_counts_odd,
        min_count=min_count,
        min_views=min_views,
        ridge=ridge,
        view_std_floor=view_std_floor,
        rank_rtol=rank_rtol,
        condition_max=condition_max,
        desc=f"solving v107 {name} odd split expert",
    )
    del even_fallback, odd_fallback

    gains = torch.zeros((int(tri_ids.numel()),), dtype=torch.float64)
    full_gains = torch.zeros_like(gains)
    scales = torch.zeros_like(gains)
    both_supported = torch.zeros((int(tri_ids.numel()),), dtype=torch.bool)
    if int(tri_ids.numel()) == 0:
        return gains, full_gains, scales, both_supported, {
            f"{name}_crossfit_supported_triangles": 0,
            f"{name}_crossfit_even_fit_triangles": int(even_solve_success.sum().item()),
            f"{name}_crossfit_odd_fit_triangles": int(odd_solve_success.sum().item()),
        }

    even_to_odd_gain, even_to_odd_full_gain, even_to_odd_scale = _weighted_risk_gain_and_scale(
        eval_xtx_flat=xtx_odd,
        eval_xty=xty_odd,
        eval_yty=yty_odd,
        base_coeffs=base_coeffs,
        expert_coeffs=even_raw,
        tri_ids=tri_ids,
    )
    odd_to_even_gain, odd_to_even_full_gain, odd_to_even_scale = _weighted_risk_gain_and_scale(
        eval_xtx_flat=xtx_even,
        eval_xty=xty_even,
        eval_yty=yty_even,
        base_coeffs=base_coeffs,
        expert_coeffs=odd_raw,
        tri_ids=tri_ids,
    )
    even_fit_support = (
        even_solve_success[tri_ids]
        & (even_counts[tri_ids] >= int(min_count))
        & (view_counts_even[tri_ids].to(dtype=torch.int64) >= int(min_views))
    )
    odd_fit_support = (
        odd_solve_success[tri_ids]
        & (odd_counts[tri_ids] >= int(min_count))
        & (view_counts_odd[tri_ids].to(dtype=torch.int64) >= int(min_views))
    )
    even_to_odd_support = (torch.round(xtx_odd[tri_ids, 0]).to(dtype=torch.int64) > 0) & even_fit_support
    odd_to_even_support = (torch.round(xtx_even[tri_ids, 0]).to(dtype=torch.int64) > 0) & odd_fit_support
    both_supported = even_to_odd_support & odd_to_even_support
    if bool(both_supported.any().item()):
        gains[both_supported] = torch.minimum(even_to_odd_gain[both_supported], odd_to_even_gain[both_supported])
        full_gains[both_supported] = torch.minimum(
            even_to_odd_full_gain[both_supported],
            odd_to_even_full_gain[both_supported],
        )
        scales[both_supported] = torch.minimum(even_to_odd_scale[both_supported], odd_to_even_scale[both_supported])

    stats = {
        f"{name}_crossfit_supported_triangles": int(both_supported.sum().item()),
        f"{name}_crossfit_even_to_odd_supported_triangles": int(even_to_odd_support.sum().item()),
        f"{name}_crossfit_odd_to_even_supported_triangles": int(odd_to_even_support.sum().item()),
        f"{name}_crossfit_even_fit_triangles": int(even_solve_success.sum().item()),
        f"{name}_crossfit_odd_fit_triangles": int(odd_solve_success.sum().item()),
        f"{name}_crossfit_even_weighted_pixels": int(torch.round(xtx_even[:, 0]).sum().item()),
        f"{name}_crossfit_odd_weighted_pixels": int(torch.round(xtx_odd[:, 0]).sum().item()),
        f"{name}_crossfit_gain_mean": (
            float(gains[both_supported].mean().item()) if bool(both_supported.any().item()) else 0.0
        ),
        f"{name}_crossfit_mse_scale_mean": (
            float(scales[both_supported].mean().item()) if bool(both_supported.any().item()) else 0.0
        ),
    }
    return gains, full_gains, scales, both_supported, stats


def _solve_pod_moe_coefficients(
    *,
    xtx_flat: torch.Tensor,
    xty: torch.Tensor,
    view_counts: torch.Tensor,
    detail_xtx: torch.Tensor,
    detail_xty: torch.Tensor,
    detail_yty: torch.Tensor,
    detail_view_counts: torch.Tensor,
    boundary_xtx: torch.Tensor,
    boundary_xty: torch.Tensor,
    boundary_yty: torch.Tensor,
    boundary_view_counts: torch.Tensor,
    detail_xtx_even: torch.Tensor | None,
    detail_xty_even: torch.Tensor | None,
    detail_yty_even: torch.Tensor | None,
    detail_view_counts_even: torch.Tensor | None,
    detail_xtx_odd: torch.Tensor | None,
    detail_xty_odd: torch.Tensor | None,
    detail_yty_odd: torch.Tensor | None,
    detail_view_counts_odd: torch.Tensor | None,
    boundary_xtx_even: torch.Tensor | None,
    boundary_xty_even: torch.Tensor | None,
    boundary_yty_even: torch.Tensor | None,
    boundary_view_counts_even: torch.Tensor | None,
    boundary_xtx_odd: torch.Tensor | None,
    boundary_xty_odd: torch.Tensor | None,
    boundary_yty_odd: torch.Tensor | None,
    boundary_view_counts_odd: torch.Tensor | None,
    gate_source: str,
    min_count: int,
    min_views: int,
    ridge: float,
    view_std_floor: float,
    rank_rtol: float,
    condition_max: float,
    residual_dtype: torch.dtype,
    method_version: str = "auto",
) -> tuple[torch.Tensor, ...]:
    use_crossfit_reliability = str(gate_source) == "crossfit_risk"
    use_descent_lock = str(method_version) == V108_METHOD_VERSION
    if use_crossfit_reliability:
        missing = [
            name
            for name, value in {
                "detail_xtx_even": detail_xtx_even,
                "detail_xty_even": detail_xty_even,
                "detail_yty_even": detail_yty_even,
                "detail_view_counts_even": detail_view_counts_even,
                "detail_xtx_odd": detail_xtx_odd,
                "detail_xty_odd": detail_xty_odd,
                "detail_yty_odd": detail_yty_odd,
                "detail_view_counts_odd": detail_view_counts_odd,
                "boundary_xtx_even": boundary_xtx_even,
                "boundary_xty_even": boundary_xty_even,
                "boundary_yty_even": boundary_yty_even,
                "boundary_view_counts_even": boundary_view_counts_even,
                "boundary_xtx_odd": boundary_xtx_odd,
                "boundary_xty_odd": boundary_xty_odd,
                "boundary_yty_odd": boundary_yty_odd,
                "boundary_view_counts_odd": boundary_view_counts_odd,
            }.items()
            if value is None
        ]
        if missing:
            raise RuntimeError("v107 POD crossfit reliability requested but split accumulators are missing: " + ", ".join(missing))
    base_coeffs, mean_residuals, counts, view_counts_out, base_stats = _solve_view_affine_coefficients(
        xtx_flat=xtx_flat,
        xty=xty,
        view_counts=view_counts,
        min_count=min_count,
        min_views=min_views,
        ridge=ridge,
        view_std_floor=view_std_floor,
        rank_rtol=rank_rtol,
        condition_max=condition_max,
        fallback_mode="shrink",
        residual_dtype=torch.float32,
    )
    detail_fallback, detail_raw, detail_counts, detail_solve_success = _solve_base_raw_coefficients(
        xtx_flat=detail_xtx,
        xty=detail_xty,
        view_counts=detail_view_counts,
        min_count=min_count,
        min_views=min_views,
        ridge=ridge,
        view_std_floor=view_std_floor,
        rank_rtol=rank_rtol,
        condition_max=condition_max,
        desc="solving v106 detail expert",
    )
    boundary_fallback, boundary_raw, boundary_counts, boundary_solve_success = _solve_base_raw_coefficients(
        xtx_flat=boundary_xtx,
        xty=boundary_xty,
        view_counts=boundary_view_counts,
        min_count=min_count,
        min_views=min_views,
        ridge=ridge,
        view_std_floor=view_std_floor,
        rank_rtol=rank_rtol,
        condition_max=condition_max,
        desc="solving v106 boundary expert",
    )
    del detail_fallback, boundary_fallback

    triangle_count = int(xtx_flat.shape[0])
    detail_ids = (detail_counts > 0).nonzero(as_tuple=False).reshape(-1)
    boundary_ids = (boundary_counts > 0).nonzero(as_tuple=False).reshape(-1)
    expert_coeffs = torch.stack([detail_raw, boundary_raw], dim=1)
    expert_delta = expert_coeffs - base_coeffs[:, None, :, :]
    reliability = torch.zeros((triangle_count, 2), dtype=torch.float32)
    gain_scores = torch.zeros((triangle_count, 2), dtype=torch.float32)
    full_gain_scores = torch.zeros((triangle_count, 2), dtype=torch.float32)
    mse_scales = torch.zeros((triangle_count, 2), dtype=torch.float32)
    descent_scales = torch.zeros((triangle_count, 2), dtype=torch.float32)
    descent_gain_scores = torch.zeros((triangle_count,), dtype=torch.float32)
    descent_objective_delta = torch.zeros((triangle_count,), dtype=torch.float32)
    descent_support = torch.zeros((triangle_count,), dtype=torch.bool)
    debt_guards = torch.zeros((triangle_count, 2), dtype=torch.float32)
    positive_counts = counts[counts > 0].to(dtype=torch.float64)
    median_count = positive_counts.median().clamp_min(1.0) if int(positive_counts.numel()) > 0 else torch.tensor(1.0, dtype=torch.float64)
    crossfit_stats: dict[str, Any] = {}
    if int(detail_ids.numel()) > 0:
        if use_crossfit_reliability:
            assert detail_xtx_even is not None and detail_xty_even is not None and detail_yty_even is not None
            assert detail_view_counts_even is not None and detail_xtx_odd is not None and detail_xty_odd is not None
            assert detail_yty_odd is not None and detail_view_counts_odd is not None
            detail_gain, detail_full_gain, detail_mse_scale, detail_crossfit_support, detail_stats = _crossfit_weighted_risk_gain_and_scale(
                name="detail",
                base_coeffs=base_coeffs,
                tri_ids=detail_ids,
                xtx_even=detail_xtx_even,
                xty_even=detail_xty_even,
                yty_even=detail_yty_even,
                view_counts_even=detail_view_counts_even,
                xtx_odd=detail_xtx_odd,
                xty_odd=detail_xty_odd,
                yty_odd=detail_yty_odd,
                view_counts_odd=detail_view_counts_odd,
                min_count=min_count,
                min_views=min_views,
                ridge=ridge,
                view_std_floor=view_std_floor,
                rank_rtol=rank_rtol,
                condition_max=condition_max,
            )
            crossfit_stats.update(detail_stats)
        else:
            detail_gain, detail_full_gain, detail_mse_scale = _weighted_risk_gain_and_scale(
                eval_xtx_flat=detail_xtx,
                eval_xty=detail_xty,
                eval_yty=detail_yty,
                base_coeffs=base_coeffs,
                expert_coeffs=detail_raw,
                tri_ids=detail_ids,
            )
            detail_crossfit_support = torch.ones((int(detail_ids.numel()),), dtype=torch.bool)
        detail_support = (torch.log1p(detail_counts[detail_ids].to(dtype=torch.float64)) / torch.log1p(median_count)).clamp(0.0, 1.0)
        detail_delta_norm = (detail_raw[detail_ids].to(dtype=torch.float64) - base_coeffs[detail_ids].to(dtype=torch.float64)).square().mean(dim=(1, 2)).sqrt()
        detail_base_norm = base_coeffs[detail_ids].to(dtype=torch.float64).square().mean(dim=(1, 2)).sqrt()
        detail_debt_guard = (detail_base_norm / (detail_base_norm + detail_delta_norm).clamp_min(1.0e-8)).clamp(0.0, 1.0)
        detail_rel = (
            torch.sqrt(detail_gain).clamp(0.0, 1.0)
            * detail_support
            * detail_debt_guard
            * detail_crossfit_support.to(dtype=torch.float64)
        ).clamp(0.0, 1.0)
        reliability[detail_ids, 0] = detail_rel.to(dtype=torch.float32)
        gain_scores[detail_ids, 0] = detail_gain.to(dtype=torch.float32)
        full_gain_scores[detail_ids, 0] = detail_full_gain.to(dtype=torch.float32)
        mse_scales[detail_ids, 0] = detail_mse_scale.to(dtype=torch.float32)
        debt_guards[detail_ids, 0] = detail_debt_guard.to(dtype=torch.float32)
    if int(boundary_ids.numel()) > 0:
        if use_crossfit_reliability:
            assert boundary_xtx_even is not None and boundary_xty_even is not None and boundary_yty_even is not None
            assert boundary_view_counts_even is not None and boundary_xtx_odd is not None and boundary_xty_odd is not None
            assert boundary_yty_odd is not None and boundary_view_counts_odd is not None
            (
                boundary_gain,
                boundary_full_gain,
                boundary_mse_scale,
                boundary_crossfit_support,
                boundary_stats,
            ) = _crossfit_weighted_risk_gain_and_scale(
                name="boundary",
                base_coeffs=base_coeffs,
                tri_ids=boundary_ids,
                xtx_even=boundary_xtx_even,
                xty_even=boundary_xty_even,
                yty_even=boundary_yty_even,
                view_counts_even=boundary_view_counts_even,
                xtx_odd=boundary_xtx_odd,
                xty_odd=boundary_xty_odd,
                yty_odd=boundary_yty_odd,
                view_counts_odd=boundary_view_counts_odd,
                min_count=min_count,
                min_views=min_views,
                ridge=ridge,
                view_std_floor=view_std_floor,
                rank_rtol=rank_rtol,
                condition_max=condition_max,
            )
            crossfit_stats.update(boundary_stats)
        else:
            boundary_gain, boundary_full_gain, boundary_mse_scale = _weighted_risk_gain_and_scale(
                eval_xtx_flat=boundary_xtx,
                eval_xty=boundary_xty,
                eval_yty=boundary_yty,
                base_coeffs=base_coeffs,
                expert_coeffs=boundary_raw,
                tri_ids=boundary_ids,
            )
            boundary_crossfit_support = torch.ones((int(boundary_ids.numel()),), dtype=torch.bool)
        boundary_support = (torch.log1p(boundary_counts[boundary_ids].to(dtype=torch.float64)) / torch.log1p(median_count)).clamp(0.0, 1.0)
        boundary_delta_norm = (boundary_raw[boundary_ids].to(dtype=torch.float64) - base_coeffs[boundary_ids].to(dtype=torch.float64)).square().mean(dim=(1, 2)).sqrt()
        boundary_base_norm = base_coeffs[boundary_ids].to(dtype=torch.float64).square().mean(dim=(1, 2)).sqrt()
        boundary_debt_guard = (boundary_base_norm / (boundary_base_norm + boundary_delta_norm).clamp_min(1.0e-8)).clamp(0.0, 1.0)
        boundary_rel = (
            torch.sqrt(boundary_gain).clamp(0.0, 1.0)
            * boundary_support
            * boundary_debt_guard
            * boundary_crossfit_support.to(dtype=torch.float64)
        ).clamp(0.0, 1.0)
        reliability[boundary_ids, 1] = boundary_rel.to(dtype=torch.float32)
        gain_scores[boundary_ids, 1] = boundary_gain.to(dtype=torch.float32)
        full_gain_scores[boundary_ids, 1] = boundary_full_gain.to(dtype=torch.float32)
        mse_scales[boundary_ids, 1] = boundary_mse_scale.to(dtype=torch.float32)
        debt_guards[boundary_ids, 1] = boundary_debt_guard.to(dtype=torch.float32)

    prelock_mse_scales = mse_scales.clone()
    if use_descent_lock:
        joint_ids = ((detail_counts > 0) | (boundary_counts > 0)).nonzero(as_tuple=False).reshape(-1)
        if int(joint_ids.numel()) > 0:
            joint_xtx = detail_xtx + boundary_xtx
            joint_xty = detail_xty + boundary_xty
            joint_yty = detail_yty + boundary_yty
            local_scales, local_gains, local_delta, local_support = _joint_two_expert_mse_descent_lock(
                eval_xtx_flat=joint_xtx,
                eval_xty=joint_xty,
                eval_yty=joint_yty,
                base_coeffs=base_coeffs,
                expert_delta=expert_delta,
                tri_ids=joint_ids,
            )
            descent_scales[joint_ids] = local_scales.to(dtype=torch.float32)
            descent_gain_scores[joint_ids] = local_gains.to(dtype=torch.float32)
            descent_objective_delta[joint_ids] = local_delta.to(dtype=torch.float32)
            descent_support[joint_ids] = local_support
        mse_scales = descent_scales

    occlusion_base_keep = torch.ones((triangle_count,), dtype=torch.float32)
    occlusion_base_keep[boundary_ids] = (1.0 - 0.5 * reliability[boundary_ids, 1]).clamp(0.5, 1.0)
    _, _, view_means, view_scales = _surface_counts_and_view_stats(
        xtx_flat=xtx_flat,
        view_std_floor=float(view_std_floor),
    )
    descent_active = descent_scales > 1.0e-6
    descent_active_triangles = descent_active.any(dim=1)
    stats = {
        "base_variant": "v104c_like_shrink_view_affine",
        "expert_names": ["detail", "occlusion_boundary"],
        "expert_reliability_variant": (
            "v107_crossfit_heldout_weighted_risk" if use_crossfit_reliability else "v106_weighted_normal_equation_risk"
        ),
        "expert_mse_certificate": (
            V108_EXPERT_MSE_CERTIFICATE
            if use_descent_lock
            else (
                "v107_crossfit_heldout_weighted_normal_equation_lambda_star"
                if use_crossfit_reliability
                else "weighted_normal_equation_lambda_star"
            )
        ),
        "expert_reliability_combine": (
            (
                "even_odd_crossfit_reliability_times_joint_two_expert_descent_scale"
                if use_crossfit_reliability
                else "same_stats_weighted_risk_reliability_times_joint_two_expert_descent_scale"
            )
            if use_descent_lock
            else (
                "even_to_odd_and_odd_to_even_min_requires_both_splits"
                if use_crossfit_reliability
                else "same_stats_weighted_risk"
            )
        ),
        "pod_crossfit_split": "target_view_even_odd" if use_crossfit_reliability else "",
        "pod_base_keep_mode": "base_preserving_boundary",
        "pod_view_gate_mode": "temperature_controlled" if use_crossfit_reliability else "implicit_unit_temperature",
        "gate_source": str(gate_source),
        "detail_triangles": int((reliability[:, 0] > 0.0).sum().item()),
        "boundary_triangles": int((reliability[:, 1] > 0.0).sum().item()),
        "detail_solved_triangles": int(detail_solve_success.sum().item()),
        "boundary_solved_triangles": int(boundary_solve_success.sum().item()),
        "detail_reliability_mean": float(reliability[:, 0][reliability[:, 0] > 0.0].mean().item()) if bool((reliability[:, 0] > 0.0).any().item()) else 0.0,
        "boundary_reliability_mean": float(reliability[:, 1][reliability[:, 1] > 0.0].mean().item()) if bool((reliability[:, 1] > 0.0).any().item()) else 0.0,
        "detail_gain_mean": float(gain_scores[:, 0][gain_scores[:, 0] > 0.0].mean().item()) if bool((gain_scores[:, 0] > 0.0).any().item()) else 0.0,
        "boundary_gain_mean": float(gain_scores[:, 1][gain_scores[:, 1] > 0.0].mean().item()) if bool((gain_scores[:, 1] > 0.0).any().item()) else 0.0,
        "detail_full_gain_mean": float(full_gain_scores[:, 0][full_gain_scores[:, 0] > 0.0].mean().item()) if bool((full_gain_scores[:, 0] > 0.0).any().item()) else 0.0,
        "boundary_full_gain_mean": float(full_gain_scores[:, 1][full_gain_scores[:, 1] > 0.0].mean().item()) if bool((full_gain_scores[:, 1] > 0.0).any().item()) else 0.0,
        "detail_mse_scale_mean": float(mse_scales[:, 0][detail_counts > 0].mean().item()) if bool((detail_counts > 0).any().item()) else 0.0,
        "boundary_mse_scale_mean": float(mse_scales[:, 1][boundary_counts > 0].mean().item()) if bool((boundary_counts > 0).any().item()) else 0.0,
        "detail_prelock_mse_scale_mean": float(prelock_mse_scales[:, 0][detail_counts > 0].mean().item()) if bool((detail_counts > 0).any().item()) else 0.0,
        "boundary_prelock_mse_scale_mean": float(prelock_mse_scales[:, 1][boundary_counts > 0].mean().item()) if bool((boundary_counts > 0).any().item()) else 0.0,
        "detail_debt_guard_mean": float(debt_guards[:, 0][detail_counts > 0].mean().item()) if bool((detail_counts > 0).any().item()) else 0.0,
        "boundary_debt_guard_mean": float(debt_guards[:, 1][boundary_counts > 0].mean().item()) if bool((boundary_counts > 0).any().item()) else 0.0,
        "detail_weighted_pixels": int(torch.round(detail_xtx[:, 0]).sum().item()),
        "boundary_weighted_pixels": int(torch.round(boundary_xtx[:, 0]).sum().item()),
        "joint_descent_supported_triangles": int(descent_support.sum().item()) if use_descent_lock else 0,
        "joint_descent_active_triangles": int(descent_active_triangles.sum().item()) if use_descent_lock else 0,
        "joint_descent_detail_active_triangles": int(descent_active[:, 0].sum().item()) if use_descent_lock else 0,
        "joint_descent_boundary_active_triangles": int(descent_active[:, 1].sum().item()) if use_descent_lock else 0,
        "joint_descent_scale_mean": float(descent_scales[descent_support].mean().item()) if use_descent_lock and bool(descent_support.any().item()) else 0.0,
        "joint_descent_active_scale_mean": float(descent_scales[descent_active].mean().item()) if use_descent_lock and bool(descent_active.any().item()) else 0.0,
        "joint_descent_gain_mean": float(descent_gain_scores[descent_support].mean().item()) if use_descent_lock and bool(descent_support.any().item()) else 0.0,
        "joint_descent_active_gain_mean": float(descent_gain_scores[descent_active_triangles].mean().item()) if use_descent_lock and bool(descent_active_triangles.any().item()) else 0.0,
        "joint_descent_objective_delta_mean": float(descent_objective_delta[descent_support].mean().item()) if use_descent_lock and bool(descent_support.any().item()) else 0.0,
        "method_version": str(method_version),
    }
    stats.update(crossfit_stats)
    stats.update({f"base_{key}": value for key, value in base_stats.items()})
    return (
        base_coeffs.to(dtype=residual_dtype).contiguous(),
        expert_delta.to(dtype=residual_dtype).contiguous(),
        mean_residuals.to(dtype=residual_dtype).contiguous(),
        reliability.to(dtype=residual_dtype).contiguous(),
        gain_scores.to(dtype=residual_dtype).contiguous(),
        mse_scales.to(dtype=residual_dtype).contiguous(),
        descent_scales.to(dtype=residual_dtype).contiguous(),
        occlusion_base_keep.to(dtype=residual_dtype).contiguous(),
        view_means.to(dtype=residual_dtype).contiguous(),
        view_scales.to(dtype=residual_dtype).contiguous(),
        counts.to(dtype=torch.int32).contiguous(),
        view_counts_out.to(dtype=torch.int32).contiguous(),
        detail_counts.to(dtype=torch.int32).contiguous(),
        boundary_counts.to(dtype=torch.int32).contiguous(),
        detail_view_counts.to(dtype=torch.int32).contiguous(),
        boundary_view_counts.to(dtype=torch.int32).contiguous(),
        stats,
    )


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
        even_fallback, even_raw, even_counts, even_solve_success = _solve_base_raw_coefficients(
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
        odd_fallback, odd_raw, odd_counts, odd_solve_success = _solve_base_raw_coefficients(
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
        del even_solve_success, odd_solve_success

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
        debt_coeff = raw_assigned - fallback_assigned
        risk_quad = torch.einsum("nfc,nfg,ngc->n", debt_coeff, raw_a_assigned, debt_coeff).clamp_min(1.0e-8)
        risk_grad = torch.matmul(raw_a_assigned, fallback_assigned) - xty_assigned
        risk_lin = torch.einsum("nfc,nfc->n", debt_coeff, risk_grad)
        optimal_gate = (-risk_lin / risk_quad).clamp(0.0, 1.0)
        optimal_gate = torch.where(torch.isfinite(optimal_gate), optimal_gate, torch.zeros_like(optimal_gate))
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
        if str(gate_source) == "optimal_risk":
            gate = optimal_gate
        else:
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
    if str(args.gate_source) not in {"normal_equation", "crossfit_risk", "optimal_risk"}:
        raise ValueError("--gate_source must be normal_equation, crossfit_risk, or optimal_risk")
    if str(args.field_variant) not in {"residual_mixture", "pod_moe"}:
        raise ValueError("--field_variant must be residual_mixture or pod_moe")

    residual_dtype = _dtype(args.residual_dtype)
    method_version = _resolved_method_version(
        str(args.field_variant),
        str(args.gate_source),
        str(getattr(args, "method_version", "auto") or "auto"),
    )
    if str(method_version) == V108_METHOD_VERSION and str(args.field_variant) != "pod_moe":
        raise ValueError(f"--method_version {V108_METHOD_VERSION} requires --field_variant pod_moe")
    if str(method_version) == V108_METHOD_VERSION and str(args.gate_source) not in {"normal_equation", "crossfit_risk"}:
        raise ValueError(f"--method_version {V108_METHOD_VERSION} requires --gate_source normal_equation or crossfit_risk")
    delta_bank = _load_delta_bank(delta_bank_path, str(args.split), str(args.endpoint_method))
    deltas = delta_bank["deltas"]
    frames = delta_bank["frames"]
    dataset, pipe, triangles, scene, background = _load_scene(model_path, int(args.iteration), int(args.renderer_scaling))
    views = list(scene.getTestCameras() if str(args.split) == "test" else scene.getTrainCameras())
    source_available_frames = int(len(views))
    if len(deltas) != source_available_frames:
        raise RuntimeError(f"delta bank/view count mismatch: bank={len(deltas)} views={source_available_frames}")
    extra_frame_meta = [str(key) for key in frames.keys() if str(key) not in deltas]
    if extra_frame_meta:
        raise RuntimeError(f"delta bank has extra frame metadata entries: {len(extra_frame_meta)}")
    selected_views = _select_indexed_views(views, str(getattr(args, "view_subset", "all") or "all"))
    if not selected_views:
        raise RuntimeError(f"view subset {args.view_subset!r} selected no frames from split {args.split!r}")
    selected_frame_indices = [int(idx) for idx, _ in selected_views]
    selected_frame_keys = [f"{idx:05d}" for idx in selected_frame_indices]

    faces = triangles.get_triangle_indices.detach().cpu().long().contiguous()
    vertices = triangles.get_vertices.detach().cpu().float().contiguous()
    face_centers = vertices[faces].mean(dim=1).contiguous()
    triangle_count = int(faces.shape[0])
    feature_count = len(BASIS_ORDER)
    xtx_flat = torch.zeros((triangle_count, feature_count * (feature_count + 1) // 2), dtype=torch.float64)
    xty = torch.zeros((triangle_count, feature_count, 3), dtype=torch.float64)
    view_counts = torch.zeros((triangle_count,), dtype=torch.int32)
    use_pod_moe = str(args.field_variant) == "pod_moe"
    use_crossfit = str(args.gate_source) == "crossfit_risk"
    use_residual_crossfit = use_crossfit and not use_pod_moe
    detail_xtx = detail_xty = detail_yty = detail_view_counts = None
    boundary_xtx = boundary_xty = boundary_yty = boundary_view_counts = None
    detail_xtx_even = detail_xty_even = detail_yty_even = detail_view_counts_even = None
    detail_xtx_odd = detail_xty_odd = detail_yty_odd = detail_view_counts_odd = None
    boundary_xtx_even = boundary_xty_even = boundary_yty_even = boundary_view_counts_even = None
    boundary_xtx_odd = boundary_xty_odd = boundary_yty_odd = boundary_view_counts_odd = None
    if use_pod_moe:
        detail_xtx = torch.zeros_like(xtx_flat)
        detail_xty = torch.zeros_like(xty)
        detail_yty = torch.zeros((triangle_count,), dtype=torch.float64)
        detail_view_counts = torch.zeros_like(view_counts)
        boundary_xtx = torch.zeros_like(xtx_flat)
        boundary_xty = torch.zeros_like(xty)
        boundary_yty = torch.zeros((triangle_count,), dtype=torch.float64)
        boundary_view_counts = torch.zeros_like(view_counts)
        if use_crossfit:
            detail_xtx_even = torch.zeros_like(xtx_flat)
            detail_xty_even = torch.zeros_like(xty)
            detail_yty_even = torch.zeros((triangle_count,), dtype=torch.float64)
            detail_view_counts_even = torch.zeros_like(view_counts)
            detail_xtx_odd = torch.zeros_like(xtx_flat)
            detail_xty_odd = torch.zeros_like(xty)
            detail_yty_odd = torch.zeros((triangle_count,), dtype=torch.float64)
            detail_view_counts_odd = torch.zeros_like(view_counts)
            boundary_xtx_even = torch.zeros_like(xtx_flat)
            boundary_xty_even = torch.zeros_like(xty)
            boundary_yty_even = torch.zeros((triangle_count,), dtype=torch.float64)
            boundary_view_counts_even = torch.zeros_like(view_counts)
            boundary_xtx_odd = torch.zeros_like(xtx_flat)
            boundary_xty_odd = torch.zeros_like(xty)
            boundary_yty_odd = torch.zeros((triangle_count,), dtype=torch.float64)
            boundary_view_counts_odd = torch.zeros_like(view_counts)
    xtx_even = xty_even = yty_even = view_counts_even = None
    xtx_odd = xty_odd = yty_odd = view_counts_odd = None
    if use_residual_crossfit:
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

    desc = f"v105 evidence-gated mixture field {args.split}/{args.view_subset}"
    for idx, view in tqdm(selected_views, desc=desc):
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
        if use_pod_moe:
            boundary = _boundary_score(ids, pkg.get("surf_depth", None))
            base_detail = _luminance_detail_score(rendering)
            teacher_detail = _delta_detail_score(delta)
            detail = (1.0 - (1.0 - base_detail) * (1.0 - teacher_detail)).clamp(0.0, 1.0)
            detail = (detail * (1.0 - boundary)).clamp(0.0, 1.0)
            if use_crossfit:
                split_detail_xtx = detail_xtx_even if idx % 2 == 0 else detail_xtx_odd
                split_detail_xty = detail_xty_even if idx % 2 == 0 else detail_xty_odd
                split_detail_yty = detail_yty_even if idx % 2 == 0 else detail_yty_odd
                split_detail_view_counts = detail_view_counts_even if idx % 2 == 0 else detail_view_counts_odd
                split_boundary_xtx = boundary_xtx_even if idx % 2 == 0 else boundary_xtx_odd
                split_boundary_xty = boundary_xty_even if idx % 2 == 0 else boundary_xty_odd
                split_boundary_yty = boundary_yty_even if idx % 2 == 0 else boundary_yty_odd
                split_boundary_view_counts = boundary_view_counts_even if idx % 2 == 0 else boundary_view_counts_odd
                detail_report = _accumulate_weighted_view_affine_view(
                    ids=ids,
                    delta=delta,
                    weights=detail,
                    projected_xy=pkg["image_2D"].detach().cpu().float().contiguous(),
                    faces=faces,
                    face_centers=face_centers,
                    camera_center=view.camera_center.detach().cpu().float(),
                    xtx_flat=split_detail_xtx,
                    xty=split_detail_xty,
                    yty=split_detail_yty,
                    view_counts=split_detail_view_counts,
                    chunk_pixels=int(args.chunk_pixels),
                )
                boundary_report = _accumulate_weighted_view_affine_view(
                    ids=ids,
                    delta=delta,
                    weights=boundary,
                    projected_xy=pkg["image_2D"].detach().cpu().float().contiguous(),
                    faces=faces,
                    face_centers=face_centers,
                    camera_center=view.camera_center.detach().cpu().float(),
                    xtx_flat=split_boundary_xtx,
                    xty=split_boundary_xty,
                    yty=split_boundary_yty,
                    view_counts=split_boundary_view_counts,
                    chunk_pixels=int(args.chunk_pixels),
                )
            else:
                detail_report = _accumulate_weighted_view_affine_view(
                    ids=ids,
                    delta=delta,
                    weights=detail,
                    projected_xy=pkg["image_2D"].detach().cpu().float().contiguous(),
                    faces=faces,
                    face_centers=face_centers,
                    camera_center=view.camera_center.detach().cpu().float(),
                    xtx_flat=detail_xtx,
                    xty=detail_xty,
                    yty=detail_yty,
                    view_counts=detail_view_counts,
                    chunk_pixels=int(args.chunk_pixels),
                )
                boundary_report = _accumulate_weighted_view_affine_view(
                    ids=ids,
                    delta=delta,
                    weights=boundary,
                    projected_xy=pkg["image_2D"].detach().cpu().float().contiguous(),
                    faces=faces,
                    face_centers=face_centers,
                    camera_center=view.camera_center.detach().cpu().float(),
                    xtx_flat=boundary_xtx,
                    xty=boundary_xty,
                    yty=boundary_yty,
                    view_counts=boundary_view_counts,
                    chunk_pixels=int(args.chunk_pixels),
                )
            report.update(
                {
                    "detail_weight_sum": detail_report.get("weight_sum", 0.0),
                    "boundary_weight_sum": boundary_report.get("weight_sum", 0.0),
                    "detail_unique_triangles": detail_report.get("unique_triangles", 0),
                    "boundary_unique_triangles": boundary_report.get("unique_triangles", 0),
                }
            )
        if use_residual_crossfit:
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

    pod_payload: dict[str, Any] = {}
    if use_pod_moe:
        if use_crossfit:
            detail_xtx.add_(detail_xtx_even).add_(detail_xtx_odd)
            detail_xty.add_(detail_xty_even).add_(detail_xty_odd)
            detail_yty.add_(detail_yty_even).add_(detail_yty_odd)
            detail_view_counts.add_(detail_view_counts_even).add_(detail_view_counts_odd)
            boundary_xtx.add_(boundary_xtx_even).add_(boundary_xtx_odd)
            boundary_xty.add_(boundary_xty_even).add_(boundary_xty_odd)
            boundary_yty.add_(boundary_yty_even).add_(boundary_yty_odd)
            boundary_view_counts.add_(boundary_view_counts_even).add_(boundary_view_counts_odd)
        (
            base_coeffs,
            expert_delta_coeffs,
            mean_residuals,
            expert_reliability,
            expert_gain_scores,
            expert_mse_scales,
            expert_descent_scales,
            occlusion_base_keep,
            view_means,
            view_scales,
            counts_out,
            view_counts_out,
            detail_counts,
            boundary_counts,
            detail_view_counts_out,
            boundary_view_counts_out,
            solve_stats,
        ) = _solve_pod_moe_coefficients(
            xtx_flat=xtx_flat,
            xty=xty,
            view_counts=view_counts,
            detail_xtx=detail_xtx,
            detail_xty=detail_xty,
            detail_yty=detail_yty,
            detail_view_counts=detail_view_counts,
            boundary_xtx=boundary_xtx,
            boundary_xty=boundary_xty,
            boundary_yty=boundary_yty,
            boundary_view_counts=boundary_view_counts,
            detail_xtx_even=detail_xtx_even,
            detail_xty_even=detail_xty_even,
            detail_yty_even=detail_yty_even,
            detail_view_counts_even=detail_view_counts_even,
            detail_xtx_odd=detail_xtx_odd,
            detail_xty_odd=detail_xty_odd,
            detail_yty_odd=detail_yty_odd,
            detail_view_counts_odd=detail_view_counts_odd,
            boundary_xtx_even=boundary_xtx_even,
            boundary_xty_even=boundary_xty_even,
            boundary_yty_even=boundary_yty_even,
            boundary_view_counts_even=boundary_view_counts_even,
            boundary_xtx_odd=boundary_xtx_odd,
            boundary_xty_odd=boundary_xty_odd,
            boundary_yty_odd=boundary_yty_odd,
            boundary_view_counts_odd=boundary_view_counts_odd,
            gate_source=str(args.gate_source),
            min_count=int(args.min_count),
            min_views=int(args.min_views),
            ridge=float(args.ridge),
            view_std_floor=float(args.view_std_floor),
            rank_rtol=float(args.rank_rtol),
            condition_max=float(args.condition_max),
            residual_dtype=residual_dtype,
            method_version=str(method_version),
        )
        pod_payload = {
            "triangle_expert_delta_coefficients": expert_delta_coeffs,
            "triangle_expert_reliability": expert_reliability,
            "triangle_expert_gain_score": expert_gain_scores,
            "triangle_expert_mse_scale": expert_mse_scales,
            "triangle_occlusion_base_keep": occlusion_base_keep,
            "triangle_expert_counts": torch.stack([detail_counts, boundary_counts], dim=1).contiguous(),
            "triangle_expert_view_counts": torch.stack([detail_view_counts_out, boundary_view_counts_out], dim=1).contiguous(),
            "expert_names": ["detail", "occlusion_boundary"],
            "expert_reliability_variant": solve_stats.get("expert_reliability_variant", ""),
            "expert_reliability_combine": solve_stats.get("expert_reliability_combine", ""),
            "expert_mse_certificate": solve_stats.get("expert_mse_certificate", ""),
            "pod_crossfit_split": solve_stats.get("pod_crossfit_split", ""),
            "pod_base_keep_mode": "base_preserving_boundary",
            "pod_view_gate_mode": "temperature_controlled" if use_crossfit else "implicit_unit_temperature",
        }
        if str(method_version) == V108_METHOD_VERSION:
            pod_payload["triangle_expert_descent_scale"] = expert_descent_scales
        delta_coeffs = torch.zeros_like(base_coeffs)
        gates = torch.zeros((int(base_coeffs.shape[0]),), dtype=residual_dtype)
        gain_scores = torch.zeros_like(gates)
        stability_scores = torch.zeros_like(gates)
    else:
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
        "basis_type": "affine_barycentric_viewdir_pod_mixture" if use_pod_moe else "affine_barycentric_viewdir_mixture",
        "builder_variant": _builder_variant(str(args.field_variant), str(method_version)),
        "field_variant": str(args.field_variant),
        "method_version": str(method_version),
        "pod_expert_reliability_variant": solve_stats.get("expert_reliability_variant", "") if use_pod_moe else "",
        "pod_view_gate_mode": pod_payload.get("pod_view_gate_mode", "") if use_pod_moe else "",
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
        "source_available_frames": int(source_available_frames),
        "source_target_frames": int(len(selected_views)),
        "view_subset": str(args.view_subset),
        "selected_frame_indices": selected_frame_indices,
        "selected_frame_keys": selected_frame_keys,
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
            (
                (
                    "v108 MSE-descent-locked POD-MoE field distilled from v102 target-camera deltas. The field stores "
                    "a v104c-like shrink view-affine base plus full-data detail and occlusion-boundary residual experts, "
                    "then scales the rendered expert corrections by a joint two-expert weighted normal-equation box-QP "
                    "descent certificate. It uses no held-out target GT for the policy, but remains a target-camera "
                    "endpoint distillation sidecar."
                )
                if str(method_version) == V108_METHOD_VERSION
                else (
                    "v107 optional cross-fitted POD-MoE expert-reliability field distilled from v102 target-camera deltas. "
                    "The field stores a v104c-like shrink view-affine base plus full-data detail and occlusion-boundary "
                    "residual experts, with reliability/gain/scale certified by even/odd held-out weighted risk and a "
                    "temperature-controlled view gate. It uses "
                    "no held-out target GT for the policy, but remains a target-camera endpoint distillation sidecar."
                    if use_crossfit
                    else "v106 perceptual/occlusion/detail MoE field distilled from v102 target-camera deltas. The field "
                    "stores a v104c-like shrink view-affine base plus detail and occlusion-boundary residual experts. It "
                    "uses no held-out target GT for the policy, but remains a target-camera endpoint distillation sidecar."
                )
            )
            if use_pod_moe
            else "v105 evidence-gated residual-mixture field distilled from v102 target-camera deltas. The field stores "
            "a conservative barycentric affine fallback expert, a view-affine residual-debt expert, and an "
            "evidence-derived gate from support, conditioning, and either cross-fitted teacher risk or teacher "
            "normal-equation gain. It uses no held-out target GT for the policy, but remains a target-camera endpoint "
            "distillation sidecar."
        ),
    }
    payload.update(pod_payload)
    output_field.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_field)
    manifest = {
        "schema_version": 1,
        "field_path": str(output_field),
        "field_sha256": _sha256(output_field),
        "field_type": payload["field_type"],
        "basis_type": payload["basis_type"],
        "builder_variant": payload["builder_variant"],
        "field_variant": payload["field_variant"],
        "method_version": payload["method_version"],
        "pod_expert_reliability_variant": payload["pod_expert_reliability_variant"],
        "pod_view_gate_mode": payload.get("pod_view_gate_mode", ""),
        "expert_reliability_variant": payload.get("expert_reliability_variant", ""),
        "expert_reliability_combine": payload.get("expert_reliability_combine", ""),
        "expert_mse_certificate": payload.get("expert_mse_certificate", ""),
        "pod_crossfit_split": payload.get("pod_crossfit_split", ""),
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
        "source_available_frames": int(source_available_frames),
        "source_target_frames": int(len(selected_views)),
        "view_subset": str(args.view_subset),
        "selected_frame_indices": selected_frame_indices,
        "selected_frame_keys": selected_frame_keys,
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
    parser.add_argument(
        "--view_subset",
        default="all",
        choices=("all", "even", "odd"),
        help=(
            "Subset of the selected split used to fit the sidecar field. "
            "The original camera/delta indices are preserved for strict split audits."
        ),
    )
    parser.add_argument("--renderer_scaling", type=int, required=True)
    parser.add_argument("--residual_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--field_variant", default="residual_mixture", choices=("residual_mixture", "pod_moe"))
    parser.add_argument("--method_version", default="auto", choices=("auto", V108_METHOD_VERSION))
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--min_views", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--residual_clip", type=float, default=0.08)
    parser.add_argument("--view_std_floor", type=float, default=1e-4)
    parser.add_argument("--rank_rtol", type=float, default=1e-7)
    parser.add_argument("--condition_max", type=float, default=1e8)
    parser.add_argument("--gate_boost", type=float, default=0.5)
    parser.add_argument(
        "--gate_source",
        default=None,
        choices=("normal_equation", "crossfit_risk", "optimal_risk"),
        help=(
            "Gate/reliability source. Defaults to normal_equation for pod_moe "
            "to preserve v106 behavior, and crossfit_risk for residual_mixture."
        ),
    )
    parser.add_argument("--view_gate_temperature", type=float, default=0.0)
    parser.add_argument("--chunk_pixels", type=int, default=262144)
    args = parser.parse_args()
    if args.gate_source is None:
        if str(args.method_version) == V108_METHOD_VERSION:
            args.gate_source = "crossfit_risk"
        else:
            args.gate_source = "normal_equation" if str(args.field_variant) == "pod_moe" else "crossfit_risk"
    return args


def main() -> int:
    build_field(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
