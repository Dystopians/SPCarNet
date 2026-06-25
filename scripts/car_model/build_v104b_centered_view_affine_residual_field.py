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


BASIS_ORDER = ["1", "barycentric_0", "barycentric_1", "viewdir_x", "viewdir_y", "viewdir_z"]
XTX_ORDER = [(i, j) for i in range(len(BASIS_ORDER)) for j in range(i, len(BASIS_ORDER))]
AFFINE_XTX_INDICES = [0, 1, 2, 6, 7, 11]
CONDITION_THRESHOLDS = (1.0e4, 1.0e6, 1.0e8, 1.0e10)


def _accumulate_view_affine_view(
    *,
    ids: torch.Tensor,
    delta: torch.Tensor,
    projected_xy: torch.Tensor,
    faces: torch.Tensor,
    face_centers: torch.Tensor,
    camera_center: torch.Tensor,
    xtx_flat: torch.Tensor,
    xty: torch.Tensor,
    view_counts: torch.Tensor | None,
    chunk_pixels: int,
) -> dict[str, Any]:
    if ids.ndim != 2:
        raise RuntimeError(f"expected 2D ids after downsample, got {tuple(ids.shape)}")
    if delta.ndim != 3 or int(delta.shape[0]) != 3:
        raise RuntimeError(f"expected delta shape [3,H,W], got {tuple(delta.shape)}")
    if tuple(delta.shape[-2:]) != tuple(ids.shape):
        raise RuntimeError(f"delta/id shape mismatch: delta={tuple(delta.shape)} ids={tuple(ids.shape)}")

    triangle_count = int(faces.shape[0])
    valid = (ids >= 0) & (ids < triangle_count)
    total_pixels = int(ids.numel())
    valid_pixels = int(valid.sum().item())
    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "accumulated_pixels": 0,
            "valid_fraction": 0.0,
            "accumulated_fraction": 0.0,
            "unique_triangles": 0,
            "invalid_topology_pixels": 0,
            "nonfinite_basis_pixels": 0,
            "degenerate_basis_pixels": 0,
        }

    pixel_yx = valid.nonzero(as_tuple=False)
    flat_ids_all = ids[valid].reshape(-1).long()
    unique_triangles = int(torch.unique(flat_ids_all).numel())
    delta_hwc = delta.permute(1, 2, 0).contiguous()
    chunk = max(1, int(chunk_pixels))
    accumulated_pixels = 0
    invalid_topology_pixels = 0
    nonfinite_basis_pixels = 0
    degenerate_basis_pixels = 0
    vertex_count = int(projected_xy.shape[0])
    view_seen = torch.zeros((triangle_count,), dtype=torch.bool) if view_counts is not None else None

    for start in range(0, valid_pixels, chunk):
        end = min(start + chunk, valid_pixels)
        local_ids = flat_ids_all[start:end]
        local_yx = pixel_yx[start:end]
        vertex_ids = faces[local_ids]
        vertex_ok = (vertex_ids >= 0).all(dim=1) & (vertex_ids < vertex_count).all(dim=1)
        if not bool(vertex_ok.any().item()):
            invalid_topology_pixels += int(vertex_ids.shape[0])
            continue
        if not bool(vertex_ok.all().item()):
            invalid_topology_pixels += int((~vertex_ok).sum().item())
            local_ids = local_ids[vertex_ok]
            local_yx = local_yx[vertex_ok]
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
        safe_denom = torch.where(safe, denom, torch.ones_like(denom))
        w0 = ((b[:, 1] - c[:, 1]) * (p[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (p[:, 1] - c[:, 1])) / safe_denom
        w1 = ((c[:, 1] - a[:, 1]) * (p[:, 0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (p[:, 1] - c[:, 1])) / safe_denom
        w0 = torch.where(safe, w0, torch.zeros_like(w0))
        w1 = torch.where(safe, w1, torch.zeros_like(w1))
        degenerate_basis_pixels += int((~safe).sum().item())

        direction = camera_center.unsqueeze(0) - face_centers[local_ids]
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
        finite = torch.isfinite(w0) & torch.isfinite(w1) & torch.isfinite(direction).all(dim=1)
        if not bool(finite.any().item()):
            nonfinite_basis_pixels += int(w0.shape[0])
            continue
        if not bool(finite.all().item()):
            nonfinite_basis_pixels += int((~finite).sum().item())
            local_ids = local_ids[finite]
            local_yx = local_yx[finite]
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
        xtx_values = torch.stack([basis[:, i] * basis[:, j] for i, j in XTX_ORDER], dim=1)
        xty_values = basis[:, :, None] * values[:, None, :]
        xtx_flat.index_add_(0, local_ids, xtx_values)
        xty.index_add_(0, local_ids, xty_values)
        if view_seen is not None:
            view_seen[torch.unique(local_ids)] = True
        accumulated_pixels += int(local_ids.numel())

    if view_seen is not None and bool(view_seen.any().item()):
        view_counts += view_seen.to(dtype=view_counts.dtype)

    return {
        "valid_pixels": int(valid_pixels),
        "accumulated_pixels": int(accumulated_pixels),
        "valid_fraction": float(valid_pixels / max(1, total_pixels)),
        "accumulated_fraction": float(accumulated_pixels / max(1, total_pixels)),
        "unique_triangles": int(unique_triangles),
        "invalid_topology_pixels": int(invalid_topology_pixels),
        "nonfinite_basis_pixels": int(nonfinite_basis_pixels),
        "degenerate_basis_pixels": int(degenerate_basis_pixels),
    }


def _raw_xtx_matrix(xtx_flat: torch.Tensor, tri_ids: torch.Tensor) -> torch.Tensor:
    feature_count = len(BASIS_ORDER)
    a = torch.zeros((int(tri_ids.numel()), feature_count, feature_count), dtype=torch.float64)
    s = xtx_flat[tri_ids]
    for k, (i, j) in enumerate(XTX_ORDER):
        a[:, i, j] = s[:, k]
        a[:, j, i] = s[:, k]
    return a


def _center_transform(view_mean: torch.Tensor, view_scale: torch.Tensor) -> torch.Tensor:
    feature_count = len(BASIS_ORDER)
    transform = torch.eye(feature_count, dtype=torch.float64).unsqueeze(0).repeat(int(view_mean.shape[0]), 1, 1)
    inv_scale = 1.0 / view_scale
    for j in range(3):
        raw_col = 3 + j
        transform[:, 0, raw_col] = -view_mean[:, j] * inv_scale[:, j]
        transform[:, raw_col, raw_col] = inv_scale[:, j]
    return transform


def _update_rank_condition_stats(
    a: torch.Tensor,
    stats: dict[str, Any],
    rank_rtol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(a.numel()) == 0:
        return (
            torch.zeros((0,), dtype=torch.int64),
            torch.zeros((0,), dtype=torch.float64),
        )
    try:
        evals = torch.linalg.eigvalsh(a)
    except RuntimeError:
        stats["diagnostic_failures"] = int(stats.get("diagnostic_failures", 0)) + int(a.shape[0])
        return (
            torch.zeros((int(a.shape[0]),), dtype=torch.int64),
            torch.full((int(a.shape[0]),), float("inf"), dtype=torch.float64),
        )
    max_eval = evals[:, -1].clamp_min(0.0)
    tol = max_eval * float(rank_rtol)
    ranks = (evals > tol[:, None]).sum(dim=1).to(dtype=torch.int64)
    cond = torch.full((int(a.shape[0]),), float("inf"), dtype=torch.float64)
    for rank in range(len(BASIS_ORDER) + 1):
        stats["rank_histogram"][str(rank)] += int((ranks == rank).sum().item())
    full_rank = ranks == len(BASIS_ORDER)
    if bool(full_rank.any().item()):
        min_eval = evals[full_rank, 0].clamp_min(1.0e-30)
        cond[full_rank] = max_eval[full_rank] / min_eval
        full_rank_cond = cond[full_rank]
        finite = torch.isfinite(full_rank_cond)
        stats["condition_full_rank_count"] += int(full_rank.sum().item())
        stats["condition_finite_count"] += int(finite.sum().item())
        stats["condition_nonfinite_count"] += int((~finite).sum().item())
        if bool(finite.any().item()):
            finite_cond = full_rank_cond[finite]
            stats["condition_finite_sum"] += float(finite_cond.sum().item())
            stats["condition_finite_max"] = max(float(stats["condition_finite_max"]), float(finite_cond.max().item()))
            for threshold in CONDITION_THRESHOLDS:
                stats["condition_threshold_counts"][str(threshold)] += int((finite_cond > threshold).sum().item())
    return ranks, cond


def _solve_fallback_affine_coefficients(
    *,
    xtx_flat: torch.Tensor,
    xty: torch.Tensor,
    tri_ids_all: torch.Tensor,
    ridge: float,
    coeffs: torch.Tensor,
) -> dict[str, Any]:
    solve_failures = 0
    solve_chunks = 0
    identity = torch.eye(3, dtype=torch.float64)
    solve_chunk_triangles = 262_144
    for start in tqdm(
        range(0, int(tri_ids_all.numel()), solve_chunk_triangles),
        desc="solving v104b v103 fallbacks",
    ):
        end = min(start + solve_chunk_triangles, int(tri_ids_all.numel()))
        tri_ids = tri_ids_all[start:end]
        if int(tri_ids.numel()) == 0:
            continue
        s = xtx_flat[tri_ids][:, AFFINE_XTX_INDICES]
        a = torch.zeros((int(tri_ids.numel()), 3, 3), dtype=torch.float64)
        a[:, 0, 0] = s[:, 0]
        a[:, 0, 1] = s[:, 1]
        a[:, 1, 0] = s[:, 1]
        a[:, 0, 2] = s[:, 2]
        a[:, 2, 0] = s[:, 2]
        a[:, 1, 1] = s[:, 3]
        a[:, 1, 2] = s[:, 4]
        a[:, 2, 1] = s[:, 4]
        a[:, 2, 2] = s[:, 5]
        if float(ridge) > 0.0:
            a = a + float(ridge) * identity.unsqueeze(0)
        b = xty[tri_ids, :3, :]
        try:
            sol, info = torch.linalg.solve_ex(a, b)
            failed = info != 0
        except AttributeError:
            sol = torch.linalg.solve(a, b)
            failed = torch.zeros((int(tri_ids.numel()),), dtype=torch.bool)
        if bool(failed.any().item()):
            failed_count = int(failed.sum().item())
            solve_failures += failed_count
            sol[failed] = torch.matmul(torch.linalg.pinv(a[failed]), b[failed])
        coeffs[tri_ids, :3, :] = sol.to(dtype=torch.float32)
        coeffs[tri_ids, 3:, :] = 0.0
        solve_chunks += 1
    return {
        "fallback_solve_failures": int(solve_failures),
        "fallback_solve_chunks": int(solve_chunks),
        "fallback_solve_chunk_triangles": int(solve_chunk_triangles),
    }


def _solve_view_affine_coefficients(
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
    fallback_mode: str,
    residual_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    triangle_count = int(xtx_flat.shape[0])
    feature_count = len(BASIS_ORDER)
    counts = torch.round(xtx_flat[:, 0]).to(dtype=torch.int64)
    observed_mask = counts > 0
    view_sums = xtx_flat[:, 3:6]
    view_means = torch.zeros((triangle_count, 3), dtype=torch.float64)
    view_scales = torch.ones((triangle_count, 3), dtype=torch.float64)
    if bool(observed_mask.any().item()):
        denom = counts[observed_mask].to(dtype=torch.float64).unsqueeze(1)
        view_means[observed_mask] = view_sums[observed_mask] / denom
        view_sq = torch.stack([xtx_flat[:, 15], xtx_flat[:, 18], xtx_flat[:, 20]], dim=1)
        view_vars = torch.zeros((triangle_count, 3), dtype=torch.float64)
        view_vars[observed_mask] = (view_sq[observed_mask] / denom) - view_means[observed_mask].square()
        view_scales[observed_mask] = view_vars[observed_mask].clamp_min(0.0).sqrt().clamp_min(float(view_std_floor))
    centering_ok = observed_mask & torch.isfinite(view_means).all(dim=1) & torch.isfinite(view_scales).all(dim=1)
    count_ok = counts >= int(min_count)
    views_ok = view_counts.to(dtype=torch.int64) >= int(min_views)
    view_affine_mask = count_ok & views_ok & centering_ok
    support_fallback_mask = observed_mask & ~view_affine_mask

    coeffs = torch.zeros((triangle_count, feature_count, 3), dtype=torch.float32)
    mean_residuals = torch.zeros((triangle_count, 3), dtype=torch.float32)
    if bool(observed_mask.any().item()):
        mean_residuals[observed_mask] = (
            xty[observed_mask, 0, :] / counts[observed_mask].to(dtype=torch.float64).unsqueeze(1).clamp_min(1.0)
        ).to(dtype=torch.float32)

    insufficient_count = observed_mask & ~count_ok
    insufficient_views = observed_mask & ~views_ok
    fallback_stats = _solve_fallback_affine_coefficients(
        xtx_flat=xtx_flat,
        xty=xty,
        tri_ids_all=observed_mask.nonzero(as_tuple=False).reshape(-1),
        ridge=float(ridge),
        coeffs=coeffs,
    )

    valid_ids_all = view_affine_mask.nonzero(as_tuple=False).reshape(-1)
    solve_failures = 0
    solve_chunks = 0
    accepted_view_affine = 0
    rank_condition_fallbacks = 0
    solve_failure_fallbacks = 0
    shrink_alpha_sum = 0.0
    shrink_alpha_count = 0
    shrink_alpha_positive = 0
    identity = torch.eye(feature_count, dtype=torch.float64)
    solve_chunk_triangles = 131_072
    diagnostic_stats: dict[str, Any] = {
        "rank_histogram": {str(rank): 0 for rank in range(feature_count + 1)},
        "condition_full_rank_count": 0,
        "condition_finite_count": 0,
        "condition_nonfinite_count": 0,
        "condition_finite_sum": 0.0,
        "condition_finite_max": 0.0,
        "condition_threshold_counts": {str(threshold): 0 for threshold in CONDITION_THRESHOLDS},
        "diagnostic_failures": 0,
    }
    for start in tqdm(
        range(0, int(valid_ids_all.numel()), solve_chunk_triangles),
        desc="solving v104b centered view affine fields",
    ):
        end = min(start + solve_chunk_triangles, int(valid_ids_all.numel()))
        tri_ids = valid_ids_all[start:end]
        if int(tri_ids.numel()) == 0:
            continue
        raw_a = _raw_xtx_matrix(xtx_flat, tri_ids)
        transform = _center_transform(view_means[tri_ids], view_scales[tri_ids])
        centered_a = torch.matmul(transform.transpose(1, 2), torch.matmul(raw_a, transform))
        ranks, conditions = _update_rank_condition_stats(centered_a, diagnostic_stats, float(rank_rtol))
        condition_ok = (
            (ranks >= feature_count)
            & torch.isfinite(conditions)
            & (conditions <= float(condition_max))
        )
        if str(fallback_mode) == "hard":
            rank_condition_fallbacks += int((~condition_ok).sum().item())
            if not bool(condition_ok.any().item()):
                continue
            solve_mask = condition_ok
            alpha = torch.ones((int(condition_ok.sum().item()),), dtype=torch.float64)
        elif str(fallback_mode) == "shrink":
            rank_score = (ranks.to(dtype=torch.float64) / float(feature_count)).clamp(0.0, 1.0)
            view_score = (view_counts[tri_ids].to(dtype=torch.float64) / float(max(1, min_views))).clamp(0.0, 1.0)
            condition_score = torch.ones_like(rank_score)
            finite = torch.isfinite(conditions)
            if bool(finite.any().item()):
                log_cond = torch.log10(conditions[finite].clamp_min(1.0))
                hi = torch.log10(torch.tensor(float(condition_max) * 100.0, dtype=torch.float64))
                lo = torch.log10(torch.tensor(float(condition_max), dtype=torch.float64))
                condition_score[finite] = ((hi - log_cond) / (hi - lo).clamp_min(1.0e-8)).clamp(0.0, 1.0)
            condition_score[~finite] = 1.0
            alpha_all = (rank_score * view_score * condition_score).clamp(0.0, 1.0)
            solve_mask = alpha_all > 0.0
            rank_condition_fallbacks += int((~solve_mask).sum().item())
            if not bool(solve_mask.any().item()):
                continue
            alpha = alpha_all[solve_mask]
            shrink_alpha_sum += float(alpha.sum().item())
            shrink_alpha_count += int(alpha.numel())
            shrink_alpha_positive += int((alpha > 0.0).sum().item())
        else:
            raise ValueError(f"unsupported fallback_mode: {fallback_mode}")
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
            failed = torch.zeros((int(tri_ids.numel()),), dtype=torch.bool)
        if bool(failed.any().item()):
            failed_count = int(failed.sum().item())
            solve_failures += failed_count
            solve_failure_fallbacks += failed_count
        assign = ~failed
        if bool(assign.any().item()):
            raw_sol = torch.matmul(transform_ok[assign], centered_sol[assign])
            if str(fallback_mode) == "shrink":
                alpha_assign = alpha[assign].to(dtype=torch.float32).reshape(-1, 1, 1)
                fallback = coeffs[tri_ids_ok[assign]]
                coeffs[tri_ids_ok[assign]] = fallback + alpha_assign * (raw_sol.to(dtype=torch.float32) - fallback)
            else:
                coeffs[tri_ids_ok[assign]] = raw_sol.to(dtype=torch.float32)
            accepted_view_affine += int(assign.sum().item())
        solve_chunks += 1

    finite_condition_count = int(diagnostic_stats["condition_finite_count"])
    condition_mean = (
        float(diagnostic_stats["condition_finite_sum"]) / float(finite_condition_count)
        if finite_condition_count > 0
        else 0.0
    )
    diagnostic_stats["condition_finite_mean"] = float(condition_mean)
    diagnostic_stats.pop("condition_finite_sum", None)

    stats = {
        "observed_triangles": int(observed_mask.sum().item()),
        "empty_triangles": int((~observed_mask).sum().item()),
        "view_affine_candidate_triangles": int(view_affine_mask.sum().item()),
        "view_affine_triangles": int(accepted_view_affine),
        "fallback_triangles": int(support_fallback_mask.sum().item() + rank_condition_fallbacks + solve_failure_fallbacks),
        "support_fallback_triangles": int(support_fallback_mask.sum().item()),
        "rank_condition_fallback_triangles": int(rank_condition_fallbacks),
        "solve_failure_fallback_triangles": int(solve_failure_fallbacks),
        "fallback_reason_counts": {
            "insufficient_count": int((support_fallback_mask & insufficient_count).sum().item()),
            "insufficient_views": int((support_fallback_mask & insufficient_views).sum().item()),
            "insufficient_count_and_views": int((support_fallback_mask & insufficient_count & insufficient_views).sum().item()),
            "nonfinite_centering": int((observed_mask & count_ok & views_ok & ~centering_ok).sum().item()),
            "rank_or_condition": int(rank_condition_fallbacks),
            "solve_failure": int(solve_failure_fallbacks),
        },
        "view_affine_solve_failures": int(solve_failures),
        "view_affine_solve_chunks": int(solve_chunks),
        "view_affine_solve_chunk_triangles": int(solve_chunk_triangles),
        "min_count": int(min_count),
        "min_views": int(min_views),
        "view_std_floor": float(view_std_floor),
        "rank_rtol": float(rank_rtol),
        "condition_max": float(condition_max),
        "fallback_mode": str(fallback_mode),
        "shrink_alpha_mean": float(shrink_alpha_sum / max(1, shrink_alpha_count)),
        "shrink_alpha_count": int(shrink_alpha_count),
        "shrink_alpha_positive": int(shrink_alpha_positive),
        "centered_view_feature_diagnostics": diagnostic_stats,
    }
    stats.update(fallback_stats)
    return (
        coeffs.to(dtype=residual_dtype).contiguous(),
        mean_residuals.to(dtype=residual_dtype).contiguous(),
        counts.to(dtype=torch.int32).contiguous(),
        view_counts.to(dtype=torch.int32).contiguous(),
        stats,
    )


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if value is None or isinstance(value, (bool, int, float, str)):
            out[str(key)] = value
        elif isinstance(value, Path):
            out[str(key)] = str(value)
        else:
            out[str(key)] = str(value)
    return out


def build_field(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model_path)
    delta_bank_arg = getattr(args, "source_delta_bank_path", None) or getattr(args, "delta_bank_path", None)
    output_field_arg = getattr(args, "output_field_path", None) or getattr(args, "output_field", None)
    delta_bank_path = Path(delta_bank_arg)
    output_field = Path(output_field_arg)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not delta_bank_path.is_file():
        raise FileNotFoundError(delta_bank_path)
    if int(args.renderer_scaling) <= 0:
        raise ValueError("--renderer_scaling must be positive")
    if int(args.chunk_pixels) <= 0:
        raise ValueError("--chunk_pixels must be positive")
    if int(args.min_count) <= 0:
        raise ValueError("--min_count must be positive")
    if int(args.min_views) <= 0:
        raise ValueError("--min_views must be positive")
    if float(args.ridge) < 0.0:
        raise ValueError("--ridge must be non-negative")
    if float(args.residual_clip) < 0.0:
        raise ValueError("--residual_clip must be non-negative")
    if float(args.view_std_floor) <= 0.0:
        raise ValueError("--view_std_floor must be positive")
    if float(args.rank_rtol) <= 0.0:
        raise ValueError("--rank_rtol must be positive")
    if float(args.condition_max) <= 0.0:
        raise ValueError("--condition_max must be positive")
    if str(args.fallback_mode) not in {"hard", "shrink"}:
        raise ValueError("--fallback_mode must be 'hard' or 'shrink'")

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
    view_reports: list[dict[str, Any]] = []
    started = time.time()

    for idx, view in enumerate(tqdm(views, desc=f"v104b centered view-affine surface field {args.split}")):
        key = f"{idx:05d}"
        if key not in deltas:
            raise RuntimeError(f"missing delta for target frame {key}")
        _assert_strict_camera_matches_bank(_camera_record(idx, view), frames.get(key, {}).get("target_camera", {}), key)
        with torch.no_grad():
            pkg = render(view, triangles, pipe, background)
        rendering = pkg["render"]
        if "rend_ids" not in pkg or pkg["rend_ids"] is None:
            raise RuntimeError("renderer package missing rend_ids; cannot build view-affine surface residual field")
        if "image_2D" not in pkg or pkg["image_2D"] is None:
            raise RuntimeError("renderer package missing image_2D; cannot build view-affine surface residual field")
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
        report.update(
            {
                "frame": key,
                "mean_abs_delta": float(delta.abs().mean().item()),
                "camera_validated": True,
            }
        )
        view_reports.append(report)
        del pkg, rendering, ids, delta

    coefficients, mean_residuals, counts_out, view_counts_out, solve_stats = _solve_view_affine_coefficients(
        xtx_flat=xtx_flat,
        xty=xty,
        view_counts=view_counts,
        min_count=int(args.min_count),
        min_views=int(args.min_views),
        ridge=float(args.ridge),
        view_std_floor=float(args.view_std_floor),
        rank_rtol=float(args.rank_rtol),
        condition_max=float(args.condition_max),
        fallback_mode=str(args.fallback_mode),
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
    args_manifest["source_delta_bank_path"] = str(delta_bank_path)
    args_manifest["output_field_path"] = str(output_field)
    payload = {
        "schema_version": 1,
        "field_type": "v102_surface_residual_field",
        "basis_type": "affine_barycentric_viewdir",
        "builder_variant": f"v104b_centered_view_affine_with_v103_{args.fallback_mode}_fallback",
        "basis_order": BASIS_ORDER,
        "fit_basis_order": [
            "1",
            "barycentric_0",
            "barycentric_1",
            "centered_scaled_viewdir_x",
            "centered_scaled_viewdir_y",
            "centered_scaled_viewdir_z",
        ],
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
        "fallback_mode": str(args.fallback_mode),
        "renderer_scaling": int(args.renderer_scaling),
        "residual_clip": float(args.residual_clip),
        "residual_dtype": str(args.residual_dtype),
        "triangle_coefficients": coefficients,
        "triangle_residuals": mean_residuals,
        "triangle_counts": counts_out,
        "triangle_view_counts": view_counts_out,
        "normal_equation_xtx_order": [f"{BASIS_ORDER[i]}*{BASIS_ORDER[j]}" for i, j in XTX_ORDER],
        "normal_equation_xty_layout": "triangle,basis,rgb",
        "view_feature_fit_transform": "per-triangle center/scale fitted basis folded into stored raw viewdir coefficients",
        "total_valid_pixels": int(total_valid_pixels),
        "total_accumulated_pixels": int(total_accumulated_pixels),
        "solve_stats": solve_stats,
        "view_reports": view_reports,
        "camera_validation": "strict_target_camera_match",
        "elapsed_sec": float(time.time() - started),
        "args": args_manifest,
        "note": (
            "v104b view-conditioned face-local affine barycentric residual field distilled from v102 preprojected "
            "deltas. View-direction features are centered/scaled per triangle during fitting and folded back into "
            "the stored [1, barycentric_0, barycentric_1, viewdir_x, viewdir_y, viewdir_z] coefficients. Triangles "
            "without enough count/view support store an embedded v103 affine-barycentric fallback."
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
        "fit_basis_order": payload["fit_basis_order"],
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
        "fallback_mode": str(args.fallback_mode),
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
                "view_affine_triangles": solve_stats.get("view_affine_triangles", 0),
                "fallback_triangles": solve_stats.get("fallback_triangles", 0),
                "total_accumulated_pixels": manifest["total_accumulated_pixels"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a v104b centered view-conditioned face-local residual field from a v102 preprojected delta bank."
        )
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--delta_bank_path", "--source_delta_bank_path", dest="source_delta_bank_path", required=True)
    parser.add_argument("--output_field", "--output_field_path", dest="output_field_path", required=True)
    parser.add_argument("--endpoint_method", default="ours_26000_v100_checkpoint_attached_ela_endpoint")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--split", default="test", choices=("test", "train"))
    parser.add_argument("--renderer_scaling", type=int, required=True)
    parser.add_argument("--residual_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--residual_clip", type=float, default=0.08)
    parser.add_argument("--view_std_floor", type=float, default=1e-4)
    parser.add_argument("--rank_rtol", type=float, default=1e-7)
    parser.add_argument("--condition_max", type=float, default=1e8)
    parser.add_argument("--fallback_mode", default="hard", choices=("hard", "shrink"))
    parser.add_argument("--chunk_pixels", type=int, default=500_000)
    return parser.parse_args()


def main() -> int:
    build_field(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
