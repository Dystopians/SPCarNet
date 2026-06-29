#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evidence_lumigraph_adapter import (
    BenefitCalibrator,
    FrameRecord,
    FrameLoader,
    adapt_frame,
    calibrate_alpha,
    fit_alpha_calibrator,
    fit_benefit_calibrator,
    load_split_frames,
    save_image_tensor,
)


def _parse_alpha_grid(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if 0.0 not in values:
        values.insert(0, 0.0)
    return sorted(set(values))


def _parse_int_grid(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_float_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _selected_calibration_row(calibration: dict, alpha: float) -> dict:
    rows = calibration.get("rows", [])
    for row in rows:
        if abs(float(row.get("alpha", -999.0)) - float(alpha)) < 1e-9:
            return row
    return {}


def _split_policy_frames(train_frames, holdout_fraction: float, holdout_offset: int = 0) -> tuple[list, list]:
    frames = list(train_frames)
    fraction = float(holdout_fraction)
    if fraction <= 0.0 or len(frames) < 3:
        return frames, frames
    step = max(int(round(1.0 / min(max(fraction, 1e-6), 0.5))), 2)
    offset = int(holdout_offset) % step
    policy_val = [frame for idx, frame in enumerate(frames) if idx % step == offset]
    fit = [frame for idx, frame in enumerate(frames) if idx % step != offset]
    if not policy_val or not fit:
        return frames, frames
    return fit, policy_val


def _benefit_feature_modes(args: argparse.Namespace) -> list[str]:
    if args.benefit_feature_mode == "auto":
        return ["confidence_magnitude", "confidence_magnitude_edge"]
    return [str(args.benefit_feature_mode)]


def _fit_optional_benefit(
    args: argparse.Namespace,
    train_frames,
    policy: dict,
    device: torch.device,
    calibration_target_frames=None,
) -> BenefitCalibrator | None:
    if not args.benefit_policy or str(policy["mode"]) != "residual":
        return None
    return fit_benefit_calibrator(
        train_frames,
        calibration_target_frames=calibration_target_frames,
        k=int(policy["k"]),
        mode=str(policy["mode"]),
        calib_stride=args.calib_stride,
        calib_max_views=args.calib_max_views,
        calib_sampler=args.calib_sampler,
        residual_clip=float(policy["residual_clip"]),
        depth_abs_tol=float(policy["depth_abs_tol"]),
        depth_rel_tol=float(policy["depth_rel_tol"]),
        direction_weight=float(policy.get("direction_weight", args.direction_weight)),
        bins=args.benefit_bins,
        min_gain=args.benefit_min_gain,
        min_bin_count=args.benefit_min_bin_count,
        max_pixels_per_view=args.benefit_max_pixels_per_view,
        feature_mode=str(policy.get("benefit_feature_mode", args.benefit_feature_mode)),
        **_local_trust_kwargs(args),
        device=device,
    )


def _fit_optional_alpha(args: argparse.Namespace, train_frames, policy: dict, device: torch.device, calibration_target_frames=None):
    if args.alpha_policy != "adaptive_bins" or str(policy["mode"]) != "residual":
        return None
    return fit_alpha_calibrator(
        train_frames,
        calibration_target_frames=calibration_target_frames,
        k=int(policy["k"]),
        mode=str(policy["mode"]),
        alpha_grid=_parse_alpha_grid(args.alpha_grid),
        calib_stride=args.calib_stride,
        calib_max_views=args.calib_max_views,
        calib_sampler=args.calib_sampler,
        residual_clip=float(policy["residual_clip"]),
        depth_abs_tol=float(policy["depth_abs_tol"]),
        depth_rel_tol=float(policy["depth_rel_tol"]),
        direction_weight=float(policy.get("direction_weight", args.direction_weight)),
        bins=args.alpha_bins,
        min_gain=args.alpha_min_gain,
        min_bin_count=args.alpha_min_bin_count,
        risk_tail_fraction=args.alpha_risk_tail_fraction,
        max_negative_gain_fraction=args.alpha_max_negative_gain_fraction,
        min_tail_gain=args.alpha_min_tail_gain,
        holdout_safe_zero=args.alpha_holdout_safe_zero,
        max_pixels_per_view=args.alpha_max_pixels_per_view,
        feature_mode=args.alpha_feature_mode,
        default_alpha=args.alpha_default,
        region_risk_json=args.alpha_region_risk_json if bool(args.alpha_region_risk_enable) else "",
        region_risk_objective_bad_only=args.alpha_region_risk_objective_bad_only,
        region_risk_objective_max_balanced_delta=args.alpha_region_risk_objective_max_balanced_delta,
        region_risk_objective_max_delta_ssim=args.alpha_region_risk_objective_max_delta_ssim,
        region_risk_objective_min_delta_lpips=args.alpha_region_risk_objective_min_delta_lpips,
        region_risk_min_tail_gain=args.alpha_region_risk_min_tail_gain,
        region_risk_max_negative_fraction=args.alpha_region_risk_max_negative_fraction,
        region_risk_min_regions=args.alpha_region_risk_min_regions,
        view_tail_scale_grid=_parse_float_grid(args.alpha_view_tail_scale_grid),
        view_tail_cvar_fraction=args.alpha_view_tail_cvar_fraction,
        view_tail_min_gain=args.alpha_view_tail_min_gain,
        view_tail_max_negative_fraction=args.alpha_view_tail_max_negative_fraction,
        view_tail_objective=args.alpha_view_tail_objective,
        view_tail_ssim_weight=args.alpha_view_tail_ssim_weight,
        view_tail_lpips_weight=args.alpha_view_tail_lpips_weight,
        view_tail_compute_lpips=args.alpha_view_tail_compute_lpips,
        view_tail_metric_max_side=args.alpha_view_tail_metric_max_side,
        **_local_trust_kwargs(args),
        device=device,
    )


def _build_fd_judge(args: argparse.Namespace, device: torch.device):
    if float(args.fd_weight) <= 0.0 and not bool(args.fd_strict):
        return None
    from utils.fd_loss import FrozenReprConfig, FrozenReprModel

    cfg = FrozenReprConfig(model_name=args.fd_backbone, pool_type=args.fd_pool)
    return FrozenReprModel(cfg, device=device)


def _fd_kwargs(args: argparse.Namespace, fd_judge_cache: list, device: torch.device):
    """Lazy-build the FD judge on first use; reuse across calibrate_alpha calls."""
    if float(args.fd_weight) <= 0.0 and not bool(args.fd_strict):
        judge = None
    elif fd_judge_cache:
        judge = fd_judge_cache[0]
    else:
        judge = _build_fd_judge(args, device)
        fd_judge_cache.append(judge)
    return {
        "fd_judge": judge,
        "fd_weight": float(args.fd_weight),
        "fd_strict": bool(args.fd_strict),
        "fd_strict_tol": float(args.fd_strict_tol),
        "fd_max_views": int(args.fd_max_views),
        "fd_min_views": int(args.fd_min_views),
    }


def _local_trust_kwargs(args: argparse.Namespace) -> dict:
    return {
        "local_trust_gate": bool(args.local_trust_gate),
        "local_trust_min_supports": int(args.local_trust_min_supports),
        "local_trust_max_residual_std": float(args.local_trust_max_residual_std),
        "local_trust_min_agreement": float(args.local_trust_min_agreement),
        "local_trust_agreement_scale": float(args.local_trust_agreement_scale),
        "local_trust_confidence_quantile": float(args.local_trust_confidence_quantile),
        "local_trust_min_confidence": float(args.local_trust_min_confidence),
        "local_trust_mode": str(args.local_trust_mode),
        "local_trust_min_weight": float(args.local_trust_min_weight),
    }


def _choose_policy(
    args: argparse.Namespace,
    train_frames,
    alpha_grid: list[float],
    device: torch.device,
) -> tuple[dict, dict, list[dict], BenefitCalibrator | None]:
    fd_judge_cache: list = []
    benefit_fit_frames, policy_val_frames = _split_policy_frames(
        train_frames,
        args.policy_holdout_fraction,
        args.policy_holdout_offset,
    )
    if not args.auto_policy:
        policy = {
            "mode": args.mode,
            "k": int(args.k),
            "residual_clip": float(args.residual_clip),
            "depth_abs_tol": float(args.depth_abs_tol),
            "depth_rel_tol": float(args.depth_rel_tol),
            "direction_weight": float(args.direction_weight),
            "benefit_feature_mode": _benefit_feature_modes(args)[0],
            "edge_gate": bool(args.edge_gate),
            "edge_gate_quantile": float(args.edge_gate_quantile),
            "edge_gate_min": float(args.edge_gate_min),
            "edge_gate_dilate": int(args.edge_gate_dilate),
            **_local_trust_kwargs(args),
        }
        if args.alpha >= 0.0 and args.skip_fixed_alpha_calibration and not args.benefit_policy:
            calibration = {
                "alpha": float(args.alpha),
                "reason": "fixed_alpha_calibration_skipped",
                "rows": [
                    {
                        "alpha": float(args.alpha),
                        "selection_score": 0.0,
                        "psnr_gain": None,
                        "ssim_gain": None,
                        "lpips_gain": None,
                    }
                ],
            }
            return policy, calibration, [], None
        support_frames = benefit_fit_frames if args.policy_holdout_fraction > 0.0 else train_frames
        target_frames = policy_val_frames if args.policy_holdout_fraction > 0.0 else None
        benefit_calibrator = _fit_optional_benefit(args, benefit_fit_frames, policy, device)
        calibration = calibrate_alpha(
            support_frames,
            calibration_target_frames=target_frames,
            alpha_grid=alpha_grid,
            k=int(policy["k"]),
            mode=str(policy["mode"]),
            calib_stride=args.calib_stride,
            calib_max_views=args.calib_max_views,
            calib_sampler=args.calib_sampler,
            residual_clip=float(policy["residual_clip"]),
            depth_abs_tol=float(policy["depth_abs_tol"]),
            depth_rel_tol=float(policy["depth_rel_tol"]),
            direction_weight=float(policy.get("direction_weight", args.direction_weight)),
            benefit_calibrator=benefit_calibrator,
            edge_gate=bool(policy.get("edge_gate", False)),
            edge_gate_quantile=float(policy.get("edge_gate_quantile", -1.0)),
            edge_gate_min=float(policy.get("edge_gate_min", 0.0)),
            edge_gate_dilate=int(policy.get("edge_gate_dilate", 0)),
            **_local_trust_kwargs(args),
            policy_objective=args.policy_objective,
            ssim_weight=args.policy_ssim_weight,
            lpips_weight=args.policy_lpips_weight,
            compute_lpips=args.calib_lpips,
            device=device,
            **_fd_kwargs(args, fd_judge_cache, device),
        )
        return policy, calibration, [], benefit_calibrator

    candidate_rows: list[dict] = []
    best: tuple[float, int, dict, dict, BenefitCalibrator | None] | None = None
    modes = [m.strip() for m in args.policy_modes.split(",") if m.strip()]
    k_values = _parse_int_grid(args.policy_k_values)
    depth_rel_values = _parse_float_grid(args.policy_depth_rel_values)
    clip_values = _parse_float_grid(args.policy_residual_clip_values)
    benefit_feature_modes = _benefit_feature_modes(args)
    direction_weight_values = (
        _parse_float_grid(args.policy_direction_weight_values)
        if args.policy_direction_weight_values.strip()
        else [float(args.direction_weight)]
    )
    edge_quantile_values = (
        _parse_float_grid(args.policy_edge_gate_quantiles)
        if bool(args.edge_gate) and args.policy_edge_gate_quantiles.strip()
        else [float(args.edge_gate_quantile)]
    )
    edge_dilate_values = (
        _parse_int_grid(args.policy_edge_gate_dilates)
        if bool(args.edge_gate) and args.policy_edge_gate_dilates.strip()
        else [int(args.edge_gate_dilate)]
    )
    order = 0
    for mode in modes:
        for k in k_values:
            for depth_rel in depth_rel_values:
                for residual_clip in clip_values:
                    for direction_weight in direction_weight_values:
                        for edge_quantile in edge_quantile_values:
                            for edge_dilate in edge_dilate_values:
                                feature_modes = benefit_feature_modes if mode == "residual" else ["none"]
                                for benefit_feature_mode in feature_modes:
                                    policy = {
                                        "mode": mode,
                                        "k": int(k),
                                        "residual_clip": float(residual_clip),
                                        "depth_abs_tol": float(args.depth_abs_tol),
                                        "depth_rel_tol": float(depth_rel),
                                        "direction_weight": float(direction_weight),
                                        "benefit_feature_mode": str(benefit_feature_mode),
                                        "edge_gate": bool(args.edge_gate),
                                        "edge_gate_quantile": float(edge_quantile),
                                        "edge_gate_min": float(args.edge_gate_min),
                                        "edge_gate_dilate": int(edge_dilate),
                                        **_local_trust_kwargs(args),
                                    }
                                    support_frames = benefit_fit_frames if args.policy_holdout_fraction > 0.0 else train_frames
                                    target_frames = policy_val_frames if args.policy_holdout_fraction > 0.0 else None
                                    benefit_calibrator = _fit_optional_benefit(args, benefit_fit_frames, policy, device)
                                    calibration = calibrate_alpha(
                                        support_frames,
                                        calibration_target_frames=target_frames,
                                        alpha_grid=alpha_grid,
                                        k=int(k),
                                        mode=mode,
                                        calib_stride=args.calib_stride,
                                        calib_max_views=args.calib_max_views,
                                        calib_sampler=args.calib_sampler,
                                        residual_clip=float(residual_clip),
                                        depth_abs_tol=args.depth_abs_tol,
                                        depth_rel_tol=float(depth_rel),
                                        direction_weight=float(direction_weight),
                                        benefit_calibrator=benefit_calibrator,
                                        edge_gate=bool(policy.get("edge_gate", False)),
                                        edge_gate_quantile=float(policy.get("edge_gate_quantile", -1.0)),
                                        edge_gate_min=float(policy.get("edge_gate_min", 0.0)),
                                        edge_gate_dilate=int(policy.get("edge_gate_dilate", 0)),
                                        **_local_trust_kwargs(args),
                                        policy_objective=args.policy_objective,
                                        ssim_weight=args.policy_ssim_weight,
                                        lpips_weight=args.policy_lpips_weight,
                                        compute_lpips=args.calib_lpips,
                                        device=device,
                                        **_fd_kwargs(args, fd_judge_cache, device),
                                    )
                                    alpha = float(calibration["alpha"])
                                    row = _selected_calibration_row(calibration, alpha)
                                    score = float(row.get("selection_score", row.get("psnr_gain", 0.0)))
                                    candidate_rows.append(
                                        {
                                            **policy,
                                            "alpha": alpha,
                                            "calib_selection_score": score,
                                            "calib_psnr_gain": row.get("psnr_gain"),
                                            "calib_ssim_gain": row.get("ssim_gain"),
                                            "calib_lpips_gain": row.get("lpips_gain"),
                                            "calib_fd_gain": row.get("fd_gain"),
                                            "calib_fd": row.get("fd"),
                                            "calib_base_fd": row.get("base_fd"),
                                            "calib_fd_rejected": row.get("fd_rejected"),
                                            "calib_psnr": row.get("psnr"),
                                            "calib_base_psnr": row.get("base_psnr"),
                                            "benefit_accepted_bins": (
                                                benefit_calibrator.to_json().get("accepted_bins")
                                                if benefit_calibrator is not None
                                                else None
                                            ),
                                        }
                                    )
                                    rank = (score, -order)
                                    if best is None or rank > (best[0], best[1]):
                                        best = (score, -order, policy, calibration, benefit_calibrator)
                                    order += 1
    assert best is not None
    return best[2], best[3], candidate_rows, best[4]


def _copy_gt(target_frames, out_gt: Path) -> None:
    out_gt.mkdir(parents=True, exist_ok=True)
    for frame in target_frames:
        dst = out_gt / frame.render_path.name
        if dst.exists():
            continue
        try:
            os.link(frame.gt_path, dst)
        except OSError:
            shutil.copy2(frame.gt_path, dst)


def _strip_target_gt_for_apply(frame: FrameRecord) -> FrameRecord:
    return FrameRecord(
        idx=int(frame.idx),
        name=str(frame.name),
        render_path=frame.render_path,
        gt_path=Path("__target_gt_forbidden_during_apply__"),
        depth_path=frame.depth_path,
        camera=frame.camera,
    )


def _maybe_wandb(args: argparse.Namespace, report: dict) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[ELA] W&B unavailable, skipping log: {exc}")
        return
    mode = args.wandb_mode or os.environ.get("WANDB_MODE", "online")
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group or None,
        name=args.wandb_name or None,
        mode=mode,
        config={
            "base_model_path": args.base_model_path,
            "iteration": args.iteration,
            "k": args.k,
            "mode": args.mode,
            "residual_clip": args.residual_clip,
            "depth_abs_tol": args.depth_abs_tol,
            "depth_rel_tol": args.depth_rel_tol,
            "direction_weight": args.direction_weight,
            "target_split": args.target_split,
            "method_name": args.method_name,
            "calib_sampler": args.calib_sampler,
            "policy_direction_weight_values": args.policy_direction_weight_values,
            "policy_edge_gate_quantiles": args.policy_edge_gate_quantiles,
            "policy_edge_gate_dilates": args.policy_edge_gate_dilates,
            "policy_objective": args.policy_objective,
            "policy_holdout_fraction": args.policy_holdout_fraction,
            "policy_holdout_offset": args.policy_holdout_offset,
            "support_policy_fit_only": args.support_policy_fit_only,
            "calib_lpips": args.calib_lpips,
            "benefit_policy": args.benefit_policy,
            "benefit_feature_mode": args.benefit_feature_mode,
            "edge_gate": args.edge_gate,
            "edge_gate_quantile": args.edge_gate_quantile,
            "edge_gate_min": args.edge_gate_min,
            "edge_gate_dilate": args.edge_gate_dilate,
            "local_trust_gate": args.local_trust_gate,
            "local_trust_min_supports": args.local_trust_min_supports,
            "local_trust_max_residual_std": args.local_trust_max_residual_std,
            "local_trust_min_agreement": args.local_trust_min_agreement,
            "local_trust_agreement_scale": args.local_trust_agreement_scale,
            "local_trust_confidence_quantile": args.local_trust_confidence_quantile,
            "local_trust_min_confidence": args.local_trust_min_confidence,
            "local_trust_mode": args.local_trust_mode,
            "local_trust_min_weight": args.local_trust_min_weight,
            "min_confidence": args.min_confidence,
            "evidence_max_side": args.evidence_max_side,
            "alpha_view_tail_objective": args.alpha_view_tail_objective,
            "alpha_view_tail_ssim_weight": args.alpha_view_tail_ssim_weight,
            "alpha_view_tail_lpips_weight": args.alpha_view_tail_lpips_weight,
            "alpha_view_tail_compute_lpips": args.alpha_view_tail_compute_lpips,
            "alpha_view_tail_metric_max_side": args.alpha_view_tail_metric_max_side,
            "fd_weight": args.fd_weight,
            "fd_backbone": args.fd_backbone,
            "fd_pool": args.fd_pool,
            "fd_max_views": args.fd_max_views,
            "fd_min_views": args.fd_min_views,
            "fd_strict": args.fd_strict,
            "fd_strict_tol": args.fd_strict_tol,
        },
    )
    flat = {
        "ela/alpha": float(report.get("alpha", 0.0)),
        "ela/k": int(report.get("k", 0)),
        "ela/depth_rel_tol": float(report.get("depth_rel_tol", 0.0)),
        "ela/direction_weight": float(report.get("direction_weight", 0.0)),
        "ela/residual_clip": float(report.get("residual_clip", 0.0)),
        "ela/target_frames": int(report.get("target_frames", 0)),
        "ela/mean_covered_fraction": float(report.get("mean_covered_fraction", 0.0)),
        "ela/mean_confidence": float(report.get("mean_confidence", 0.0)),
        "ela/mean_benefit_accept_fraction": float(report.get("mean_benefit_accept_fraction", 0.0)),
        "ela/mean_edge_accept_fraction": float(report.get("mean_edge_accept_fraction", 0.0)),
        "ela/local_trust_gate": int(bool(report.get("local_trust_gate", False))),
        "ela/mean_local_trust_accept_fraction": float(
            report.get("mean_local_trust_accept_fraction", 0.0)
        ),
        "ela/mean_local_trust_support_count": float(report.get("mean_local_trust_support_count", 0.0)),
        "ela/mean_local_trust_residual_std": float(report.get("mean_local_trust_residual_std", 0.0)),
        "ela/mean_local_trust_agreement": float(report.get("mean_local_trust_agreement", 0.0)),
        "ela/mean_local_trust_confidence_threshold": float(
            report.get("mean_local_trust_confidence_threshold", 0.0)
        ),
        "ela/mean_local_trust_weight": float(report.get("mean_local_trust_weight", 0.0)),
        "ela/mean_local_trust_active_fraction": float(
            report.get("mean_local_trust_active_fraction", 0.0)
        ),
        "ela/mean_alpha": float(report.get("mean_alpha", 0.0)),
        "ela/mean_alpha_active_fraction": float(report.get("mean_alpha_active_fraction", 0.0)),
        "ela/evidence_max_side": int(report.get("evidence_max_side", 0) or 0),
        "ela/mean_evidence_scaled": float(report.get("mean_evidence_scaled", 0.0) or 0.0),
    }
    calibration = report.get("calibration") or {}
    calibration_rows = calibration.get("rows") or []
    chosen_alpha = float(report.get("alpha", 0.0))
    chosen_row = next(
        (
            row
            for row in calibration_rows
            if abs(float(row.get("alpha", -999.0)) - chosen_alpha) < 1e-9
        ),
        {},
    )

    def _wandb_float(value) -> float:
        try:
            out = float(value)
        except Exception:
            return math.nan
        return out if math.isfinite(out) else math.nan

    alpha_calibrator = report.get("alpha_calibrator") or {}
    if isinstance(alpha_calibrator, dict):
        flat.update(
            {
                "ela/alpha_holdout_safe_zero": int(bool(alpha_calibrator.get("holdout_safe_zero", False))),
                "ela/alpha_accepted_bins": int(alpha_calibrator.get("accepted_bins", 0) or 0),
                "ela/alpha_risk_zeroed_bins": int(alpha_calibrator.get("risk_zeroed_bins", 0) or 0),
                "ela/alpha_risk_tail_fraction": _wandb_float(alpha_calibrator.get("risk_tail_fraction")),
                "ela/alpha_max_negative_gain_fraction": _wandb_float(
                    alpha_calibrator.get("max_negative_gain_fraction")
                ),
                "ela/alpha_min_tail_gain": _wandb_float(alpha_calibrator.get("min_tail_gain")),
                "ela/alpha_region_risk_enabled": int(bool(alpha_calibrator.get("region_risk_enabled", False))),
                "ela/alpha_region_risk_zeroed_bins": int(
                    alpha_calibrator.get("region_risk_zeroed_bins", 0) or 0
                ),
                "ela/alpha_region_risk_objective_bad_only": int(
                    bool(alpha_calibrator.get("region_risk_objective_bad_only", False))
                ),
                "ela/alpha_region_risk_min_tail_gain": _wandb_float(
                    alpha_calibrator.get("region_risk_min_tail_gain")
                ),
                "ela/alpha_region_risk_max_negative_fraction": _wandb_float(
                    alpha_calibrator.get("region_risk_max_negative_fraction")
                ),
                "ela/alpha_view_tail_enabled": int(bool(alpha_calibrator.get("view_tail_enabled", False))),
                "ela/alpha_view_tail_scale": _wandb_float(alpha_calibrator.get("view_tail_scale")),
                "ela/alpha_view_tail_safe_scale_found": int(
                    bool(alpha_calibrator.get("view_tail_safe_scale_found", False))
                ),
                "ela/alpha_view_tail_fallback_used": int(
                    bool(alpha_calibrator.get("view_tail_fallback_used", False))
                ),
                "ela/alpha_view_tail_mean_gain": _wandb_float(alpha_calibrator.get("view_tail_mean_gain")),
                "ela/alpha_view_tail_cvar_gain": _wandb_float(alpha_calibrator.get("view_tail_cvar_gain")),
                "ela/alpha_view_tail_negative_fraction": _wandb_float(
                    alpha_calibrator.get("view_tail_negative_fraction")
                ),
                "ela/alpha_view_tail_balanced_objective": int(
                    str(alpha_calibrator.get("view_tail_objective", "mse")) == "balanced"
                ),
                "ela/alpha_view_tail_lpips_enabled": int(
                    bool(alpha_calibrator.get("view_tail_compute_lpips", False))
                ),
            }
        )

    fd_gains = []
    for row in calibration_rows:
        if row.get("fd_gain") is None:
            continue
        gain = _wandb_float(row.get("fd_gain"))
        if math.isfinite(gain):
            fd_gains.append(gain)
    flat.update(
        {
            "ela/fd_requested": int(bool(calibration.get("fd_requested", False))),
            "ela/fd_enabled": int(bool(calibration.get("fd_enabled", False))),
            "ela/fd_views": int(calibration.get("fd_views", 0) or 0),
            "ela/fd_weight": float(calibration.get("fd_weight", 0.0) or 0.0),
            "ela/fd_strict": int(bool(calibration.get("fd_strict", False))),
            "ela/fd_selected_gain": _wandb_float(chosen_row.get("fd_gain")),
            "ela/fd_selected_value": _wandb_float(chosen_row.get("fd")),
            "ela/fd_selected_base": _wandb_float(chosen_row.get("base_fd")),
            "ela/fd_selected_rejected": int(bool(chosen_row.get("fd_rejected", False))),
            "ela/fd_max_gain": max(fd_gains) if fd_gains else math.nan,
            "ela/fd_min_gain": min(fd_gains) if fd_gains else math.nan,
        }
    )
    run.log(flat)
    run.summary.update(flat)
    run.finish()


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model = Path(args.base_model_path)
    output_model = Path(args.output_model_path) if args.output_model_path else base_model
    base_method = args.base_method_name or f"ours_{args.iteration}"
    method_name = args.method_name or f"ours_{args.iteration}_ela_k{args.k}"

    train_frames = load_split_frames(base_model, "train", base_method)
    target_frames = load_split_frames(base_model, args.target_split, base_method)
    benefit_fit_frames, policy_val_frames = _split_policy_frames(
        train_frames,
        args.policy_holdout_fraction,
        args.policy_holdout_offset,
    )
    adapt_support_frames = (
        benefit_fit_frames
        if bool(args.support_policy_fit_only) and float(args.policy_holdout_fraction) > 0.0
        else train_frames
    )
    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    if bool(args.alpha_region_risk_enable):
        risk_path_text = str(args.alpha_region_risk_json).strip()
        if not risk_path_text:
            raise ValueError("--alpha_region_risk_enable requires --alpha_region_risk_json")
        if not Path(risk_path_text).is_file():
            raise FileNotFoundError(f"alpha region-risk JSON not found: {risk_path_text}")
    policy, calibration, policy_candidates, benefit_calibrator = _choose_policy(
        args, train_frames, alpha_grid, device
    )
    alpha_fit_frames = benefit_fit_frames if args.policy_holdout_fraction > 0.0 else train_frames
    alpha_target_frames = policy_val_frames if args.policy_holdout_fraction > 0.0 else None
    alpha_calibrator = _fit_optional_alpha(args, alpha_fit_frames, policy, device, alpha_target_frames)
    alpha = float(args.alpha) if args.alpha >= 0.0 else float(calibration["alpha"])
    out_method = output_model / args.target_split / method_name
    out_render = out_method / "renders"
    out_gt = out_method / "gt"
    out_render.mkdir(parents=True, exist_ok=True)
    copied_gt_before_apply = False
    if not bool(args.strict_no_target_gt_apply):
        _copy_gt(target_frames, out_gt)
        copied_gt_before_apply = True

    loader = FrameLoader(device=device)
    infos = []
    for target in tqdm(target_frames, desc=f"ELA {args.target_split}"):
        apply_target = _strip_target_gt_for_apply(target) if bool(args.strict_no_target_gt_apply) else target
        adapted, info = adapt_frame(
            apply_target,
            adapt_support_frames,
            k=int(policy["k"]),
            alpha=alpha,
            mode=str(policy["mode"]),
            residual_clip=float(policy["residual_clip"]),
            min_confidence=args.min_confidence,
            depth_abs_tol=float(policy["depth_abs_tol"]),
            depth_rel_tol=float(policy["depth_rel_tol"]),
            direction_weight=float(policy.get("direction_weight", args.direction_weight)),
            benefit_calibrator=benefit_calibrator,
            alpha_calibrator=alpha_calibrator,
            edge_gate=bool(policy.get("edge_gate", False)),
            edge_gate_quantile=float(policy.get("edge_gate_quantile", -1.0)),
            edge_gate_min=float(policy.get("edge_gate_min", 0.0)),
            edge_gate_dilate=int(policy.get("edge_gate_dilate", 0)),
            **_local_trust_kwargs(args),
            evidence_max_side=int(args.evidence_max_side),
            loader=loader,
            device=device,
        )
        save_image_tensor(adapted, out_render / target.render_path.name)
        infos.append({"frame": target.name, **info})
    copied_gt_after_apply = False
    if bool(args.strict_no_target_gt_apply):
        _copy_gt(target_frames, out_gt)
        copied_gt_after_apply = True

    report = {
        "method": "Evidence Lumigraph Adapter",
        "base_model_path": str(base_model),
        "output_model_path": str(output_model),
        "base_method": base_method,
        "method_name": method_name,
        "target_split": args.target_split,
        "strict_no_target_gt_apply": bool(args.strict_no_target_gt_apply),
        "target_gt_visible_to_apply": not bool(args.strict_no_target_gt_apply),
        "target_gt_copied_before_apply": bool(copied_gt_before_apply),
        "target_gt_copied_after_apply": bool(copied_gt_after_apply),
        "target_gt_usage": (
            "eval_only_after_apply"
            if bool(args.strict_no_target_gt_apply)
            else "copied_before_apply_but_adapt_frame_does_not_read_target_gt"
        ),
        "target_frames": len(target_frames),
        "train_support_frames": len(train_frames),
        "adapt_support_scope": (
            "policy_fit_train_only"
            if bool(args.support_policy_fit_only) and float(args.policy_holdout_fraction) > 0.0
            else "full_train"
        ),
        "adapt_support_frames": len(adapt_support_frames),
        "adapt_support_view_names": [frame.name for frame in adapt_support_frames],
        "alpha": alpha,
        "alpha_source": "cli" if args.alpha >= 0.0 else "train_calibration",
        "calibration": calibration,
        "auto_policy": bool(args.auto_policy),
        "policy": policy,
        "policy_candidates": policy_candidates,
        "policy_objective": args.policy_objective,
        "policy_ssim_weight": float(args.policy_ssim_weight),
        "policy_lpips_weight": float(args.policy_lpips_weight),
        "calib_lpips": bool(args.calib_lpips),
        "calib_sampler": str(args.calib_sampler),
        "benefit_policy": benefit_calibrator.to_json() if benefit_calibrator is not None else None,
        "alpha_policy": str(args.alpha_policy),
        "alpha_calibrator": alpha_calibrator.to_json() if alpha_calibrator is not None else None,
        "alpha_holdout_safe_zero": bool(args.alpha_holdout_safe_zero),
        "alpha_risk_tail_fraction": float(args.alpha_risk_tail_fraction),
        "alpha_max_negative_gain_fraction": float(args.alpha_max_negative_gain_fraction),
        "alpha_min_tail_gain": float(args.alpha_min_tail_gain),
        "alpha_region_risk_enable": bool(args.alpha_region_risk_enable),
        "alpha_region_risk_json": str(args.alpha_region_risk_json),
        "alpha_region_risk_objective_bad_only": bool(args.alpha_region_risk_objective_bad_only),
        "alpha_region_risk_objective_max_balanced_delta": float(args.alpha_region_risk_objective_max_balanced_delta),
        "alpha_region_risk_objective_max_delta_ssim": float(args.alpha_region_risk_objective_max_delta_ssim),
        "alpha_region_risk_objective_min_delta_lpips": float(args.alpha_region_risk_objective_min_delta_lpips),
        "alpha_region_risk_min_tail_gain": float(args.alpha_region_risk_min_tail_gain),
        "alpha_region_risk_max_negative_fraction": float(args.alpha_region_risk_max_negative_fraction),
        "alpha_region_risk_min_regions": int(args.alpha_region_risk_min_regions),
        "alpha_view_tail_scale_grid": str(args.alpha_view_tail_scale_grid),
        "alpha_view_tail_cvar_fraction": float(args.alpha_view_tail_cvar_fraction),
        "alpha_view_tail_min_gain": float(args.alpha_view_tail_min_gain),
        "alpha_view_tail_max_negative_fraction": float(args.alpha_view_tail_max_negative_fraction),
        "alpha_view_tail_objective": str(args.alpha_view_tail_objective),
        "alpha_view_tail_ssim_weight": float(args.alpha_view_tail_ssim_weight),
        "alpha_view_tail_lpips_weight": float(args.alpha_view_tail_lpips_weight),
        "alpha_view_tail_compute_lpips": bool(args.alpha_view_tail_compute_lpips),
        "alpha_view_tail_metric_max_side": int(args.alpha_view_tail_metric_max_side),
        "benefit_feature_mode": str(policy.get("benefit_feature_mode", args.benefit_feature_mode)),
        "requested_benefit_feature_mode": str(args.benefit_feature_mode),
        "policy_holdout_fraction": float(args.policy_holdout_fraction),
        "policy_holdout_offset": int(args.policy_holdout_offset),
        "policy_fit_views": [frame.name for frame in benefit_fit_frames],
        "policy_val_views": [frame.name for frame in policy_val_frames],
        "edge_gate": bool(policy.get("edge_gate", False)),
        "edge_gate_quantile": float(policy.get("edge_gate_quantile", -1.0)),
        "edge_gate_min": float(policy.get("edge_gate_min", 0.0)),
        "edge_gate_dilate": int(policy.get("edge_gate_dilate", 0)),
        "local_trust_gate": bool(args.local_trust_gate),
        "local_trust_min_supports": int(args.local_trust_min_supports),
        "local_trust_max_residual_std": float(args.local_trust_max_residual_std),
        "local_trust_min_agreement": float(args.local_trust_min_agreement),
        "local_trust_agreement_scale": float(args.local_trust_agreement_scale),
        "local_trust_confidence_quantile": float(args.local_trust_confidence_quantile),
        "local_trust_min_confidence": float(args.local_trust_min_confidence),
        "local_trust_mode": str(args.local_trust_mode),
        "local_trust_min_weight": float(args.local_trust_min_weight),
        "min_confidence": float(args.min_confidence),
        "evidence_max_side": int(args.evidence_max_side),
        "mean_evidence_scaled": float(
            sum(1.0 if bool(x.get("evidence_scaled", False)) else 0.0 for x in infos) / max(len(infos), 1)
        ),
        "mode": str(policy["mode"]),
        "k": int(policy["k"]),
        "residual_clip": float(policy["residual_clip"]),
        "depth_abs_tol": float(policy["depth_abs_tol"]),
        "depth_rel_tol": float(policy["depth_rel_tol"]),
        "direction_weight": float(policy.get("direction_weight", args.direction_weight)),
        "mean_covered_fraction": float(sum(float(x["covered_fraction"]) for x in infos) / max(len(infos), 1)),
        "mean_confidence": float(sum(float(x["mean_confidence"]) for x in infos) / max(len(infos), 1)),
        "mean_benefit_accept_fraction": float(
            sum(float(x.get("benefit_accept_fraction", 0.0)) for x in infos) / max(len(infos), 1)
        ),
        "mean_edge_accept_fraction": float(
            sum(float(x.get("edge_accept_fraction", 0.0)) for x in infos) / max(len(infos), 1)
        ),
        "mean_local_trust_accept_fraction": float(
            sum(float(x.get("local_trust_accept_fraction", 0.0)) for x in infos) / max(len(infos), 1)
        ),
        "mean_local_trust_active_fraction": float(
            sum(float(x.get("local_trust_active_fraction", 0.0)) for x in infos) / max(len(infos), 1)
        ),
        "mean_local_trust_weight": float(
            sum(float(x.get("local_trust_mean_weight", 0.0)) for x in infos) / max(len(infos), 1)
        ),
        "mean_local_trust_support_count": float(
            sum(float(x.get("local_trust_mean_support_count", 0.0)) for x in infos) / max(len(infos), 1)
        ),
        "mean_local_trust_residual_std": float(
            sum(float(x.get("local_trust_mean_residual_std", 0.0)) for x in infos) / max(len(infos), 1)
        ),
        "mean_local_trust_agreement": float(
            sum(float(x.get("local_trust_mean_agreement", 0.0)) for x in infos) / max(len(infos), 1)
        ),
        "mean_local_trust_confidence_threshold": float(
            sum(float(x.get("local_trust_confidence_threshold", 0.0)) for x in infos) / max(len(infos), 1)
        ),
        "mean_alpha": float(sum(float(x.get("alpha_mean", 0.0)) for x in infos) / max(len(infos), 1)),
        "mean_alpha_active_fraction": float(
            sum(float(x.get("alpha_active_fraction", 0.0)) for x in infos) / max(len(infos), 1)
        ),
        "frames": infos,
    }
    (out_method / "ela_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[ELA] Saved adapted renders: {out_render}")
    print(f"[ELA] alpha={alpha:.4f} covered={report['mean_covered_fraction']:.4f} confidence={report['mean_confidence']:.6f}")
    _maybe_wandb(args, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply train-only geometry-aware residual evidence lumigraph adapter.")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--output_model_path", default="")
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--base_method_name", default="")
    parser.add_argument("--target_split", choices=("train", "test"), default="test")
    parser.add_argument("--method_name", default="")
    parser.add_argument(
        "--strict_no_target_gt_apply",
        action="store_true",
        help=(
            "Do not place target/test GT in the output method directory until after adapted renders "
            "have been written. The apply-time FrameRecord is also stripped to a sentinel GT path."
        ),
    )
    parser.add_argument("--k", default=4, type=int)
    parser.add_argument("--mode", choices=("residual", "color"), default="residual")
    parser.add_argument("--auto_policy", action="store_true")
    parser.add_argument("--policy_modes", default="residual,color")
    parser.add_argument("--policy_k_values", default="4,8")
    parser.add_argument("--policy_depth_rel_values", default="0.06,0.12")
    parser.add_argument("--policy_residual_clip_values", default="0.25")
    parser.add_argument("--policy_direction_weight_values", default="")
    parser.add_argument(
        "--policy_edge_gate_quantiles",
        default="",
        help="Optional comma-separated edge-gate quantiles for train-only auto-policy search.",
    )
    parser.add_argument(
        "--policy_edge_gate_dilates",
        default="",
        help="Optional comma-separated edge-gate dilation values for train-only auto-policy search.",
    )
    parser.add_argument("--policy_objective", choices=("psnr", "balanced"), default="psnr")
    parser.add_argument("--policy_ssim_weight", default=20.0, type=float)
    parser.add_argument("--policy_lpips_weight", default=20.0, type=float)
    parser.add_argument(
        "--policy_holdout_fraction",
        default=0.0,
        type=float,
        help="Deterministic train-view holdout fraction for policy selection. 0 keeps the legacy train-only calibration.",
    )
    parser.add_argument(
        "--policy_holdout_offset",
        default=0,
        type=int,
        help=(
            "Offset for deterministic train-view holdout selection. With fraction 0.25, offsets 0..3 "
            "evaluate complementary interleaved trajectory slices."
        ),
    )
    parser.add_argument(
        "--support_policy_fit_only",
        action="store_true",
        help=(
            "For train-policy validation runs, adapt target frames using only the fitting-train subset as support. "
            "This prevents held-out train views from contributing residuals to other held-out train targets."
        ),
    )
    parser.add_argument("--calib_lpips", action="store_true")
    parser.add_argument("--benefit_policy", action="store_true")
    parser.add_argument("--benefit_bins", default=5, type=int)
    parser.add_argument("--benefit_min_gain", default=0.0, type=float)
    parser.add_argument("--benefit_min_bin_count", default=64, type=int)
    parser.add_argument("--benefit_max_pixels_per_view", default=4096, type=int)
    parser.add_argument(
        "--benefit_feature_mode",
        choices=("confidence_magnitude", "confidence_magnitude_edge", "auto"),
        default="confidence_magnitude",
        help="Train-only benefit-gate feature set. auto compares basic and edge modes on the policy holdout split.",
    )
    parser.add_argument("--alpha_policy", choices=("global", "adaptive_bins"), default="global")
    parser.add_argument("--alpha_bins", default=5, type=int)
    parser.add_argument("--alpha_min_gain", default=0.0, type=float)
    parser.add_argument("--alpha_min_bin_count", default=64, type=int)
    parser.add_argument("--alpha_holdout_safe_zero", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--alpha_risk_tail_fraction", default=0.20, type=float)
    parser.add_argument("--alpha_max_negative_gain_fraction", default=1.0, type=float)
    parser.add_argument("--alpha_min_tail_gain", default=-math.inf, type=float)
    parser.add_argument("--alpha_max_pixels_per_view", default=4096, type=int)
    parser.add_argument("--alpha_region_risk_enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--alpha_region_risk_json", default="")
    parser.add_argument("--alpha_region_risk_objective_bad_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--alpha_region_risk_objective_max_balanced_delta", default=0.0, type=float)
    parser.add_argument("--alpha_region_risk_objective_max_delta_ssim", default=0.0, type=float)
    parser.add_argument("--alpha_region_risk_objective_min_delta_lpips", default=0.0, type=float)
    parser.add_argument("--alpha_region_risk_min_tail_gain", default=0.0, type=float)
    parser.add_argument("--alpha_region_risk_max_negative_fraction", default=1.0, type=float)
    parser.add_argument("--alpha_region_risk_min_regions", default=1, type=int)
    parser.add_argument(
        "--alpha_view_tail_scale_grid",
        default="",
        help=(
            "Optional comma-separated global alpha scales selected by policy-view tail safety. "
            "Empty keeps legacy per-bin alpha behavior."
        ),
    )
    parser.add_argument("--alpha_view_tail_cvar_fraction", default=0.25, type=float)
    parser.add_argument("--alpha_view_tail_min_gain", default=-math.inf, type=float)
    parser.add_argument("--alpha_view_tail_max_negative_fraction", default=1.0, type=float)
    parser.add_argument("--alpha_view_tail_objective", choices=("mse", "balanced"), default="mse")
    parser.add_argument("--alpha_view_tail_ssim_weight", default=20.0, type=float)
    parser.add_argument("--alpha_view_tail_lpips_weight", default=20.0, type=float)
    parser.add_argument("--alpha_view_tail_compute_lpips", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--alpha_view_tail_metric_max_side", default=512, type=int)
    parser.add_argument(
        "--alpha_feature_mode",
        choices=("confidence_magnitude", "confidence_magnitude_edge"),
        default="confidence_magnitude_edge",
    )
    parser.add_argument("--alpha_default", default=0.0, type=float)
    parser.add_argument("--edge_gate", action="store_true")
    parser.add_argument("--edge_gate_quantile", default=-1.0, type=float)
    parser.add_argument("--edge_gate_min", default=0.0, type=float)
    parser.add_argument("--edge_gate_dilate", default=0, type=int)
    parser.add_argument("--local_trust_gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local_trust_min_supports", default=2, type=int)
    parser.add_argument("--local_trust_max_residual_std", default=-1.0, type=float)
    parser.add_argument("--local_trust_min_agreement", default=0.0, type=float)
    parser.add_argument("--local_trust_agreement_scale", default=0.04, type=float)
    parser.add_argument("--local_trust_confidence_quantile", default=-1.0, type=float)
    parser.add_argument("--local_trust_min_confidence", default=0.0, type=float)
    parser.add_argument("--local_trust_mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--local_trust_min_weight", default=0.0, type=float)
    parser.add_argument("--alpha", default=-1.0, type=float, help="Override alpha. Default <0 uses train-only calibration.")
    parser.add_argument(
        "--skip_fixed_alpha_calibration",
        action="store_true",
        help="When --alpha is fixed and --auto_policy is disabled, skip redundant alpha calibration.",
    )
    parser.add_argument("--alpha_grid", default="0,0.125,0.25,0.5,0.75,1.0")
    parser.add_argument("--calib_stride", default=16, type=int)
    parser.add_argument("--calib_max_views", default=16, type=int)
    parser.add_argument("--calib_sampler", choices=("stride_first", "uniform"), default="stride_first")
    parser.add_argument("--residual_clip", default=0.25, type=float)
    parser.add_argument("--min_confidence", default=1e-4, type=float)
    parser.add_argument("--depth_abs_tol", default=0.02, type=float)
    parser.add_argument("--depth_rel_tol", default=0.03, type=float)
    parser.add_argument("--direction_weight", default=0.35, type=float)
    parser.add_argument(
        "--evidence_max_side",
        default=0,
        type=int,
        help=(
            "Optional fast adapter path. When >0, compute support evidence warps at this "
            "maximum image side and upsample the residual/confidence maps before the usual gates."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fd_weight",
        default=0.0,
        type=float,
        help=(
            "Weight of FD gain in the alpha selection score. Default 0 keeps legacy behavior. "
            "Raw DINOv2 FD on ~32 train views is typically O(5-30) while PSNR/SSIM/LPIPS terms "
            "are O(1), so values above ~0.05 will dominate the score; prefer --fd_strict first."
        ),
    )
    parser.add_argument(
        "--fd_backbone",
        default="vit_base_patch14_dinov2.lvd142m",
        help="timm model name for the frozen FD judge (DINOv2 ViT-B/14 by default).",
    )
    parser.add_argument(
        "--fd_pool",
        choices=("cls", "mean"),
        default="cls",
        help="Token pooling for ViT FD features.",
    )
    parser.add_argument(
        "--fd_max_views",
        default=32,
        type=int,
        help="Max calibration views used to estimate the FD per-alpha empirical Gaussian.",
    )
    parser.add_argument(
        "--fd_min_views",
        default=8,
        type=int,
        help="Below this many calibration views FD is skipped and treated as inert (no gate, no score change).",
    )
    parser.add_argument(
        "--fd_strict",
        action="store_true",
        help="Reject any alpha>0 whose FD gain falls below -fd_strict_tol (alpha=0 fallback preserved).",
    )
    parser.add_argument(
        "--fd_strict_tol",
        default=0.0,
        type=float,
        help="Tolerance for fd_strict (raw FD units). 0 means any expected-FD regression is rejected.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "spcarnet_meshprior"))
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()
    if int(args.local_trust_min_supports) < 0:
        parser.error("--local_trust_min_supports must be >= 0")
    if not math.isfinite(float(args.local_trust_max_residual_std)):
        parser.error("--local_trust_max_residual_std must be finite; use a negative value to disable the std gate")
    if not 0.0 <= float(args.local_trust_min_agreement) <= 1.0:
        parser.error("--local_trust_min_agreement must be in [0, 1]")
    if float(args.local_trust_agreement_scale) <= 0.0:
        parser.error("--local_trust_agreement_scale must be > 0")
    if not -1.0 <= float(args.local_trust_confidence_quantile) < 1.0:
        parser.error("--local_trust_confidence_quantile must be in [-1, 1)")
    if float(args.local_trust_min_confidence) < 0.0:
        parser.error("--local_trust_min_confidence must be >= 0")
    if float(args.local_trust_min_weight) < 0.0:
        parser.error("--local_trust_min_weight must be >= 0")
    if int(args.evidence_max_side) < 0:
        parser.error("--evidence_max_side must be >= 0")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
