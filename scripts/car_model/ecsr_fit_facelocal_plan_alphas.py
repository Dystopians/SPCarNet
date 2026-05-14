#!/usr/bin/env python3
"""Fit train-only per-face alpha multipliers for a face-local residual plan.

This script keeps the selected face set fixed and learns one scalar alpha per
face.  The alphas are later consumed by
``ecsr_apply_surface_residual_facelocal_sh1_delta.py --materialize_plan_alpha_json``.

The objective uses only the existing train evidence cache and its deterministic
fit/policy-val split.  Held-out test renders are not read.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.ecsr_apply_surface_residual_facelocal_sh1_delta import (
    PixelSamples,
    collect_samples,
    evaluate_proxy,
    localize_samples,
    plan_rows_to_facelocal_coeff,
    read_candidate_plan,
    samples_to_tensors,
    split_view_paths,
    _predict,
)
from scripts.car_model.ecsr_run_facelocal_coupled_selector import risk_greedy_rows, train_certificate_score
from ss3dm_prior.meshsplatopt.checkpoint_compaction import checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_plan", "--plan_in", dest="candidate_plan", type=Path, required=True)
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--source_model", type=Path, default=None)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--face_ids", default="", help="Optional comma-separated fixed face ids.")
    parser.add_argument(
        "--selector_decision_json",
        type=Path,
        default=None,
        help="Optional coupled_selector_decision.json to source selected trial face ids from.",
    )
    parser.add_argument(
        "--trial",
        default="selected",
        help="Trial to read from --selector_decision_json. Use 'selected' for the scene-level selected_trial.",
    )
    parser.add_argument("--selector_mode", choices=("top", "score", "risk", "georisk"), default="risk")
    parser.add_argument("--selector_count", type=int, default=4)
    parser.add_argument("--risk_pair_lambda", type=float, default=0.65)
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--high_error_quantile", type=float, default=-1.0)
    parser.add_argument("--min_alpha", type=float, default=-1.0)
    parser.add_argument("--barycentric_tolerance", type=float, default=0.35)
    parser.add_argument("--max_samples_per_face_view", type=int, default=64)
    parser.add_argument("--max_total_samples", type=int, default=240000)
    parser.add_argument("--uniform_barycentric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strength", type=float, default=-1.0)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=-1.0)
    parser.add_argument("--sh_degree", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--alpha_min", type=float, default=0.0)
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=1.0,
        help="Conservative default: shrink selected residual carriers without amplifying them.",
    )
    parser.add_argument("--steps", type=int, default=450)
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--lambda_anchor", type=float, default=0.02)
    parser.add_argument("--lambda_cvar", type=float, default=0.10)
    parser.add_argument("--lambda_view_var", type=float, default=0.03)
    parser.add_argument("--lambda_max_regression", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if float(args.risk_pair_lambda) < 0.0:
        parser.error("--risk_pair_lambda must be non-negative")
    if int(args.selector_count) <= 0:
        parser.error("--selector_count must be positive")
    if float(args.alpha_min) < 0.0:
        parser.error("--alpha_min must be non-negative")
    if float(args.alpha_max) < float(args.alpha_min):
        parser.error("--alpha_max must be >= --alpha_min")
    return args


def parse_face_ids(raw: str) -> list[int]:
    out: list[int] = []
    for item in str(raw or "").replace(" ", ",").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def face_ids_from_selector_decision(path: Path | None, trial: str) -> tuple[list[int], str]:
    if path is None:
        return [], ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected_trial = str(payload.get("selected_trial", "")) if str(trial) == "selected" else str(trial)
    for row in payload.get("trials", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("trial", "")) == selected_trial:
            return [int(fid) for fid in row.get("face_ids", [])], selected_trial
    return [], selected_trial


def plan_defaults(meta: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    filters = meta.get("filters") if isinstance(meta.get("filters"), dict) else {}
    return {
        "high_error_quantile": float(args.high_error_quantile)
        if float(args.high_error_quantile) >= 0.0
        else float(filters.get("high_error_quantile", 0.65)),
        "min_alpha": float(args.min_alpha) if float(args.min_alpha) >= 0.0 else float(filters.get("min_alpha", 0.05)),
        "strength": float(args.strength) if float(args.strength) >= 0.0 else float(meta.get("strength", 0.18)),
        "max_abs_delta_rgb": float(args.max_abs_delta_rgb)
        if float(args.max_abs_delta_rgb) >= 0.0
        else float(meta.get("max_abs_delta_rgb", 0.014)),
    }


def select_rows(
    rows: list[dict[str, Any]],
    *,
    face_ids: list[int],
    mode: str,
    count: int,
    pair_lambda: float,
) -> list[dict[str, Any]]:
    if face_ids:
        wanted = {int(fid) for fid in face_ids}
        selected = [row for row in rows if int(row.get("face_id", -1)) in wanted]
        order = {int(fid): idx for idx, fid in enumerate(face_ids)}
        selected.sort(key=lambda row: order.get(int(row.get("face_id", -1)), len(order)))
        return selected
    if mode == "top":
        return list(rows)[:count]
    if mode == "score":
        return sorted(rows, key=train_certificate_score, reverse=True)[:count]
    if mode == "risk":
        return risk_greedy_rows(rows, count, pair_lambda=pair_lambda)
    if mode == "georisk":
        raise ValueError("georisk selector mode requires explicit --face_ids for alpha fitting")
    raise ValueError(f"unsupported selector mode: {mode}")


def weighted_mse_from_prediction(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=target.device)
    return (((pred - target) ** 2) * weights[:, None]).sum() / (weights.sum().clamp_min(1e-8) * 3.0)


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, beta: float = 0.003) -> torch.Tensor:
    if pred.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=target.device)
    error = pred - target
    abs_error = error.abs()
    beta_t = torch.tensor(float(beta), dtype=pred.dtype, device=pred.device)
    loss = torch.where(abs_error < beta_t, 0.5 * error * error / beta_t.clamp_min(1e-8), abs_error - 0.5 * beta_t)
    return (loss * weights[:, None]).sum() / (weights.sum().clamp_min(1e-8) * 3.0)


def sample_face_indices(samples: PixelSamples, selected_faces: list[int], device: torch.device) -> torch.Tensor:
    face_to_idx = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    return torch.as_tensor([face_to_idx[int(fid)] for fid in samples.face_ids], dtype=torch.long, device=device)


def per_view_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    view_names: list[str],
) -> tuple[list[str], torch.Tensor]:
    if pred.numel() == 0 or not view_names:
        return [], torch.empty((0,), dtype=torch.float32, device=target.device)
    unique = sorted(set(str(name) for name in view_names))
    rows: list[torch.Tensor] = []
    view_array = np.asarray(view_names, dtype=object)
    for name in unique:
        idx_np = np.nonzero(view_array == name)[0]
        if idx_np.size == 0:
            continue
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=target.device)
        rows.append(weighted_mse_from_prediction(pred[idx], target[idx], weights[idx]))
    if not rows:
        return [], torch.empty((0,), dtype=torch.float32, device=target.device)
    return unique, torch.stack(rows)


def alpha_prediction(base_pred: torch.Tensor, alpha: torch.Tensor, face_idx: torch.Tensor) -> torch.Tensor:
    if base_pred.numel() == 0:
        return base_pred
    return base_pred * alpha[face_idx].view(-1, 1)


def evaluate_alpha_proxy(
    base_pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    face_idx: torch.Tensor,
    alpha: torch.Tensor,
) -> dict[str, float]:
    if base_pred.numel() == 0:
        return {"samples": 0, "mse_before": 0.0, "mse_after": 0.0, "relative_gain": 0.0}
    zero = torch.zeros_like(base_pred)
    pred = alpha_prediction(base_pred, alpha, face_idx)
    mse_before = weighted_mse_from_prediction(zero, target, weights)
    mse_after = weighted_mse_from_prediction(pred, target, weights)
    return {
        "samples": int(base_pred.shape[0]),
        "mse_before": float(mse_before.detach().cpu().item()),
        "mse_after": float(mse_after.detach().cpu().item()),
        "relative_gain": float(((mse_before - mse_after) / mse_before.clamp_min(1e-12)).detach().cpu().item()),
    }


def closed_form_alpha(
    base_pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    face_idx: torch.Tensor,
    face_count: int,
    *,
    alpha_min: float,
    alpha_max: float,
) -> torch.Tensor:
    out = torch.ones((face_count,), dtype=torch.float32, device=base_pred.device)
    if base_pred.numel() == 0:
        return out.clamp(float(alpha_min), float(alpha_max))
    w = weights[:, None].clamp_min(1e-8)
    for idx in range(face_count):
        mask = face_idx == idx
        if not bool(mask.any()):
            out[idx] = 0.0
            continue
        p = base_pred[mask]
        y = target[mask]
        ww = w[mask]
        numerator = (ww * p * y).sum()
        denominator = (ww * p * p).sum().clamp_min(1e-12)
        out[idx] = numerator / denominator
    return out.clamp(float(alpha_min), float(alpha_max))


def fit_alphas(
    *,
    fit_base_pred: torch.Tensor,
    fit_target: torch.Tensor,
    fit_weights: torch.Tensor,
    fit_face_idx: torch.Tensor,
    val_base_pred: torch.Tensor,
    val_target: torch.Tensor,
    val_weights: torch.Tensor,
    val_face_idx: torch.Tensor,
    val_view_names: list[str],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    face_count = int(max(int(fit_face_idx.max().item()) + 1 if fit_face_idx.numel() else 0, int(val_face_idx.max().item()) + 1 if val_face_idx.numel() else 0))
    if face_count <= 0:
        return torch.empty((0,), dtype=torch.float32, device=fit_target.device), {"enabled": False, "reason": "no_samples"}

    init = closed_form_alpha(
        fit_base_pred,
        fit_target,
        fit_weights,
        fit_face_idx,
        face_count,
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
    )
    span = max(float(args.alpha_max) - float(args.alpha_min), 1e-6)
    init_unit = ((init - float(args.alpha_min)) / span).clamp(1e-4, 1.0 - 1e-4)
    param = torch.logit(init_unit).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([param], lr=float(args.lr))

    zero_val = torch.zeros_like(val_base_pred)
    _, val_before_per_view = per_view_mse(zero_val, val_target, val_weights, val_view_names)
    final_loss = torch.zeros((), dtype=torch.float32, device=fit_target.device)
    final_terms: dict[str, float] = {}
    for _ in range(int(args.steps)):
        alpha = float(args.alpha_min) + span * torch.sigmoid(param)
        fit_pred = alpha_prediction(fit_base_pred, alpha, fit_face_idx)
        fit_loss = weighted_huber(fit_pred, fit_target, fit_weights)
        anchor = ((alpha - 1.0) ** 2).mean()
        cvar_loss = torch.zeros((), dtype=torch.float32, device=fit_target.device)
        var_loss = torch.zeros((), dtype=torch.float32, device=fit_target.device)
        max_regression = torch.zeros((), dtype=torch.float32, device=fit_target.device)
        if val_base_pred.numel() and val_before_per_view.numel():
            val_pred = alpha_prediction(val_base_pred, alpha, val_face_idx)
            _, val_after_per_view = per_view_mse(val_pred, val_target, val_weights, val_view_names)
            regression = (val_after_per_view - val_before_per_view).clamp_min(0.0)
            if regression.numel():
                k = max(int(math.ceil(0.2 * int(regression.numel()))), 1)
                cvar_loss = torch.topk(regression, k=k).values.mean()
                max_regression = regression.max()
            gains = val_before_per_view - val_after_per_view
            if gains.numel() > 1:
                var_loss = gains.var(unbiased=False)
        final_loss = (
            fit_loss
            + float(args.lambda_anchor) * anchor
            + float(args.lambda_cvar) * cvar_loss
            + float(args.lambda_view_var) * var_loss
            + float(args.lambda_max_regression) * max_regression
        )
        optimizer.zero_grad(set_to_none=True)
        final_loss.backward()
        optimizer.step()
        final_terms = {
            "fit_huber": float(fit_loss.detach().cpu().item()),
            "anchor": float(anchor.detach().cpu().item()),
            "policy_val_cvar_regression": float(cvar_loss.detach().cpu().item()),
            "policy_val_gain_variance": float(var_loss.detach().cpu().item()),
            "policy_val_max_regression": float(max_regression.detach().cpu().item()),
            "total": float(final_loss.detach().cpu().item()),
        }

    alpha = (float(args.alpha_min) + span * torch.sigmoid(param)).detach()
    return alpha, {"enabled": True, "initial_alpha_mean": float(init.mean().detach().cpu().item()), "final_terms": final_terms}


def per_face_report(
    selected_faces: list[int],
    alpha: torch.Tensor,
    fit_samples: PixelSamples,
    val_samples: PixelSamples,
) -> list[dict[str, Any]]:
    fit_counts = {int(fid): 0 for fid in selected_faces}
    val_counts = {int(fid): 0 for fid in selected_faces}
    for fid in fit_samples.face_ids.tolist():
        fit_counts[int(fid)] = fit_counts.get(int(fid), 0) + 1
    for fid in val_samples.face_ids.tolist():
        val_counts[int(fid)] = val_counts.get(int(fid), 0) + 1
    rows: list[dict[str, Any]] = []
    alpha_cpu = alpha.detach().cpu().float().numpy() if alpha.numel() else np.empty((0,), dtype=np.float32)
    for idx, fid in enumerate(selected_faces):
        value = float(alpha_cpu[idx]) if idx < alpha_cpu.shape[0] else 0.0
        rows.append(
            {
                "face_id": int(fid),
                "alpha": value,
                "fit_samples": int(fit_counts.get(int(fid), 0)),
                "policy_val_samples": int(val_counts.get(int(fid), 0)),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    rows, meta = read_candidate_plan(args.candidate_plan)
    if not rows:
        raise RuntimeError(f"candidate plan has no rows: {args.candidate_plan}")
    defaults = plan_defaults(meta, args)
    explicit_face_ids = parse_face_ids(args.face_ids)
    decision_face_ids, decision_trial = face_ids_from_selector_decision(args.selector_decision_json, str(args.trial))
    requested_face_ids = explicit_face_ids or decision_face_ids
    selected_rows = select_rows(
        rows,
        face_ids=requested_face_ids,
        mode=str(args.selector_mode),
        count=int(args.selector_count),
        pair_lambda=float(args.risk_pair_lambda),
    )
    if not selected_rows:
        raise RuntimeError("no selected rows for alpha fitting")

    source_model = args.source_model or Path(str(meta.get("source_model", "")))
    if not str(source_model):
        raise RuntimeError("--source_model is required when candidate plan lacks source_model")
    iteration = int(args.iteration) if int(args.iteration) > 0 else int(meta.get("iteration", 26000))
    state = torch.load(checkpoint_path(source_model, iteration), map_location="cpu")
    faces = state["_triangle_indices"].detach().cpu().long()
    vertices = state["triangles_points"].detach().cpu().float()
    basis_count = int(meta.get("basis_count", (int(args.sh_degree) + 1) ** 2))
    selected_faces, coeff, rejected = plan_rows_to_facelocal_coeff(selected_rows, faces, fallback_basis_count=basis_count)
    if not selected_faces:
        raise RuntimeError(f"selected plan rows were all rejected: {rejected[:3]}")

    view_paths = sorted((args.evidence_dir / "views").glob("*.npz"))
    fit_paths, val_paths = split_view_paths(view_paths, int(args.policy_val_stride))
    face_stats = {int(row["face_id"]): row.get("face_stats", {}) for row in selected_rows if isinstance(row, dict)}
    fit_samples = collect_samples(
        fit_paths,
        selected_faces,
        face_stats,
        high_error_quantile=float(defaults["high_error_quantile"]),
        min_alpha=float(defaults["min_alpha"]),
        barycentric_tolerance=float(args.barycentric_tolerance),
        max_samples_per_face_view=int(args.max_samples_per_face_view),
        max_total_samples=int(args.max_total_samples),
        uniform_barycentric=bool(args.uniform_barycentric),
    )
    val_samples = collect_samples(
        val_paths,
        selected_faces,
        face_stats,
        high_error_quantile=float(defaults["high_error_quantile"]),
        min_alpha=float(defaults["min_alpha"]),
        barycentric_tolerance=float(args.barycentric_tolerance),
        max_samples_per_face_view=int(args.max_samples_per_face_view),
        max_total_samples=max(int(args.max_total_samples // 2), 1),
        uniform_barycentric=bool(args.uniform_barycentric),
    )
    source_vertex_ids, _, fit_sample_vertex_ids = localize_samples(faces, selected_faces, fit_samples)
    _, _, val_sample_vertex_ids = localize_samples(faces, selected_faces, val_samples)
    vertices_local = vertices[source_vertex_ids].float() if source_vertex_ids.numel() else torch.empty((0, 3), dtype=torch.float32)
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    fit_ids, fit_basis, fit_target, fit_weights = samples_to_tensors(
        fit_samples,
        fit_sample_vertex_ids,
        vertices_local,
        strength=float(defaults["strength"]),
        max_abs_delta_rgb=float(defaults["max_abs_delta_rgb"]),
        sh_degree=int(args.sh_degree),
        device=device,
    )
    val_ids, val_basis, val_target, val_weights = samples_to_tensors(
        val_samples,
        val_sample_vertex_ids,
        vertices_local,
        strength=float(defaults["strength"]),
        max_abs_delta_rgb=float(defaults["max_abs_delta_rgb"]),
        sh_degree=int(args.sh_degree),
        device=device,
    )
    coeff = coeff.to(device=device)
    with torch.no_grad():
        fit_base_pred = _predict(coeff, fit_ids, fit_basis)
        val_base_pred = _predict(coeff, val_ids, val_basis)
    fit_face_idx = sample_face_indices(fit_samples, selected_faces, device)
    val_face_idx = sample_face_indices(val_samples, selected_faces, device)
    alpha, opt = fit_alphas(
        fit_base_pred=fit_base_pred,
        fit_target=fit_target,
        fit_weights=fit_weights,
        fit_face_idx=fit_face_idx,
        val_base_pred=val_base_pred,
        val_target=val_target,
        val_weights=val_weights,
        val_face_idx=val_face_idx,
        val_view_names=val_samples.view_names,
        args=args,
    )

    fit_proxy_uniform = evaluate_proxy(coeff, fit_ids, fit_basis, fit_target, fit_weights)
    val_proxy_uniform = evaluate_proxy(coeff, val_ids, val_basis, val_target, val_weights)
    fit_proxy_alpha = evaluate_alpha_proxy(fit_base_pred, fit_target, fit_weights, fit_face_idx, alpha)
    val_proxy_alpha = evaluate_alpha_proxy(val_base_pred, val_target, val_weights, val_face_idx, alpha)
    view_names, val_before = per_view_mse(torch.zeros_like(val_base_pred), val_target, val_weights, val_samples.view_names)
    _, val_after = per_view_mse(alpha_prediction(val_base_pred, alpha, val_face_idx), val_target, val_weights, val_samples.view_names)
    per_view = []
    for idx, name in enumerate(view_names):
        before = float(val_before[idx].detach().cpu().item())
        after = float(val_after[idx].detach().cpu().item()) if idx < int(val_after.shape[0]) else math.nan
        per_view.append(
            {
                "view_name": str(name),
                "mse_before": before,
                "mse_after": after,
                "mse_gain": before - after,
                "relative_gain": (before - after) / max(before, 1e-12),
            }
        )

    alpha_cpu = alpha.detach().cpu().float()
    payload = {
        "operator": "facelocal_plan_per_face_alpha_refit",
        "test_usage": "none",
        "candidate_plan": str(args.candidate_plan),
        "evidence_dir": str(args.evidence_dir),
        "source_model": str(source_model),
        "iteration": int(iteration),
        "selection": {
            "mode": str(args.selector_mode) if not requested_face_ids else "explicit_face_ids",
            "selector_count": int(args.selector_count),
            "risk_pair_lambda": float(args.risk_pair_lambda),
            "selector_decision_json": str(args.selector_decision_json) if args.selector_decision_json else "",
            "selector_decision_trial": decision_trial,
            "face_ids": [int(fid) for fid in selected_faces],
            "rejected_plan_rows": rejected[:20],
        },
        "alpha_bounds": {"min": float(args.alpha_min), "max": float(args.alpha_max)},
        "face_alphas": {str(int(fid)): float(alpha_cpu[idx].item()) for idx, fid in enumerate(selected_faces)},
        "face_alpha_rows": per_face_report(selected_faces, alpha_cpu, fit_samples, val_samples),
        "fit_proxy_uniform": fit_proxy_uniform,
        "policy_val_proxy_uniform": val_proxy_uniform,
        "fit_proxy_alpha": fit_proxy_alpha,
        "policy_val_proxy_alpha": val_proxy_alpha,
        "policy_val_per_view": per_view,
        "policy_val_tail": {
            "view_count": int(len(per_view)),
            "negative_gain_fraction": float(sum(1 for row in per_view if float(row["mse_gain"]) < 0.0) / max(len(per_view), 1)),
            "worst_mse_gain": float(min((float(row["mse_gain"]) for row in per_view), default=0.0)),
            "mean_mse_gain": float(np.mean([float(row["mse_gain"]) for row in per_view])) if per_view else 0.0,
        },
        "objective": opt,
        "args": {
            "policy_val_stride": int(args.policy_val_stride),
            "high_error_quantile": float(defaults["high_error_quantile"]),
            "min_alpha": float(defaults["min_alpha"]),
            "max_samples_per_face_view": int(args.max_samples_per_face_view),
            "max_total_samples": int(args.max_total_samples),
            "uniform_barycentric": bool(args.uniform_barycentric),
            "strength": float(defaults["strength"]),
            "max_abs_delta_rgb": float(defaults["max_abs_delta_rgb"]),
            "steps": int(args.steps),
            "lambda_anchor": float(args.lambda_anchor),
            "lambda_cvar": float(args.lambda_cvar),
            "lambda_view_var": float(args.lambda_view_var),
            "lambda_max_regression": float(args.lambda_max_regression),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "faces": len(selected_faces)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
