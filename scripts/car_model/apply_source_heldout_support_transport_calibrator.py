#!/usr/bin/env python3
"""Apply a trained support-transport calibrator to a rendered split.

This applies the v302-style constrained hybrid policy to target/test views.  The
rendering step uses target render/depth/camera and train-source support residuals
only.  Target GT is read only after candidate images are written, for metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.train_source_heldout_support_transport_calibrator import (  # noqa: E402
    FEATURE_NAMES,
    SupportTransportCalibrator,
    _build_features,
    _image_metrics,
    _mean,
    _normalize,
    _split_calibrator_train_val,
    _split_source_heldout,
    _summarize_rows,
)
from utils.evidence_lumigraph_adapter import (  # noqa: E402
    FrameLoader,
    compute_evidence_signal,
    load_split_frames,
    save_image_tensor,
    select_support_frames,
    warp_support_residual,
)


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[SupportTransportCalibrator, torch.Tensor, torch.Tensor, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = dict(ckpt.get("config", {}))
    feature_names = list(ckpt.get("feature_names", FEATURE_NAMES))
    if feature_names != FEATURE_NAMES:
        raise RuntimeError(f"feature names mismatch: checkpoint={feature_names} script={FEATURE_NAMES}")
    model = SupportTransportCalibrator(
        len(FEATURE_NAMES),
        hidden_channels=int(config.get("hidden_channels", 32)),
        layers=int(config.get("layers", 3)),
        max_gain=float(config.get("max_gain", 0.75)),
        direct_scale=float(config.get("direct_scale", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["feature_mean"].to(device), ckpt["feature_std"].to(device), config


def _write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "support_transport_apply_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = payload["summary"]
    lines = [
        "# Support-Transport Calibrator Apply Report",
        "",
        "Target GT is used only for this post-apply evaluation report.",
        "",
        "## Summary",
        "",
        f"- split: `{payload['split']['target_split']}`",
        f"- target views: `{payload['split']['target_views']}`",
        f"- support train views: `{payload['split']['support_views']}`",
        f"- anchor alpha / learned scale / blend: `{payload['policy']['anchor_alpha']}` / `{payload['policy']['learned_scale']}` / `{payload['policy']['blend']}`",
        f"- candidate ladder: `{payload['policy'].get('enable_candidate_ladder')}` / `{payload['policy'].get('candidate_ladder_blends')}`",
        f"- candidate variants: `{payload['policy'].get('candidate_variants')}`",
        f"- output variant: `{payload['policy']['output_variant']}`",
        f"- selected variant: `{payload['policy']['selected_variant']}`",
        f"- per-view gate: `{payload['policy'].get('per_view_gate_mode', 'off')}`",
        f"- source perceptual selector: `{payload['policy'].get('source_perceptual_enabled')}`",
        f"- source objective LPIPS / DISTS weight: `{payload['policy'].get('source_objective_lpips_weight')}` / `{payload['policy'].get('source_objective_dists_weight')}`",
        f"- fixed PSNR gain: `{summary['fixed_psnr_gain']}`",
        f"- fixed SSIM gain: `{summary.get('fixed_ssim_gain')}`",
        f"- learned PSNR gain: `{summary['learned_psnr_gain']}`",
        f"- learned SSIM gain: `{summary.get('learned_ssim_gain')}`",
        f"- hybrid PSNR gain: `{summary['hybrid_psnr_gain']}`",
        f"- hybrid SSIM gain: `{summary.get('hybrid_ssim_gain')}`",
        f"- selected PSNR gain: `{summary['selected_psnr_gain']}`",
        f"- selected SSIM gain: `{summary.get('selected_ssim_gain')}`",
        f"- hybrid minus fixed PSNR: `{summary['hybrid_minus_fixed_psnr_gain']}`",
        f"- hybrid minus fixed SSIM: `{summary.get('hybrid_minus_fixed_ssim_gain')}`",
        f"- selected minus fixed PSNR: `{summary['selected_minus_fixed_psnr_gain']}`",
        f"- selected minus fixed SSIM: `{summary.get('selected_minus_fixed_ssim_gain')}`",
        f"- hybrid all-axis vs fixed: `{summary['hybrid_all_axis_vs_fixed']}`",
        f"- selected all-axis safe vs fixed: `{summary['selected_all_axis_safe_vs_fixed']}`",
        f"- per-view no-op fraction: `{summary.get('per_view_noop_fraction')}`",
        "",
        "## Method Summary",
        "",
        "| method | PSNR gain | SSIM gain | changed | positive views | min PSNR gain | CVaR20 PSNR gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    method_rows: list[tuple[str, dict[str, Any]]] = []
    candidate_summaries = payload.get("candidate_summaries")
    if isinstance(candidate_summaries, dict) and candidate_summaries:
        for variant in payload["policy"].get("candidate_variants", []):
            if variant in candidate_summaries:
                method_rows.append((f"candidate:{variant}", candidate_summaries[variant]))
    else:
        method_rows.extend(
            [
                ("fixed raw ELA", payload["fixed_summary"]),
                ("learned only", payload["learned_summary"]),
                ("hybrid", payload["hybrid_summary"]),
            ]
        )
    method_rows.append(("selected output", payload["selected_summary"]))
    for label, row in method_rows:
        tail = row.get("psnr_gain_tail", {})
        lines.append(
            "| {label} | {psnr:+.9f} | {ssim:+.9f} | {changed:.9f} | {pos:.6f} | {minv:+.9f} | {cvar:+.9f} |".format(
                label=label,
                psnr=float(row.get("psnr_gain", 0.0)),
                ssim=float(row.get("ssim_gain", 0.0)),
                changed=float(row.get("mean_changed_fraction", 0.0)),
                pos=float(row.get("positive_view_fraction", 0.0)),
                minv=float(tail.get("min", 0.0)),
                cvar=float(tail.get("cvar", 0.0)),
            )
        )
    lines += ["", "## Verdict", "", str(payload.get("verdict", ""))]
    if payload.get("selector"):
        lines += [
            "",
            "## Source-Heldout Selector",
            "",
            f"- selector variant: `{payload['selector'].get('selected_variant')}`",
            f"- selector views: `{payload['selector'].get('val_views')}`",
            f"- selector verdict: `{payload['selector'].get('verdict')}`",
            f"- source perceptual: `{payload['selector'].get('source_perceptual')}`",
        ]
    if payload.get("per_view_gate"):
        gate = payload["per_view_gate"]
        lines += [
            "",
            "## Per-View Risk Gate",
            "",
            f"- enabled: `{gate.get('enabled')}`",
            f"- selected score: `{gate.get('score_name')}`",
            f"- threshold / polarity: `{gate.get('threshold')}` / `{gate.get('polarity')}`",
            f"- source-heldout accept fraction: `{gate.get('source_accept_fraction')}`",
            f"- target no-op fraction: `{payload['summary'].get('per_view_noop_fraction')}`",
            f"- gate verdict: `{gate.get('verdict')}`",
        ]
    if payload.get("per_view_knn_policy"):
        knn = payload["per_view_knn_policy"]
        lines += [
            "",
            "## Per-View KNN Policy",
            "",
            f"- enabled: `{knn.get('enabled')}`",
            f"- k: `{knn.get('k')}`",
            f"- threshold mode: `{knn.get('threshold_mode')}`",
            f"- selected threshold: `{knn.get('selected_threshold')}`",
            f"- local tail guard: `{knn.get('local_tail_guard', {}).get('enabled')}`",
            f"- local tail k: `{knn.get('local_tail_guard', {}).get('k')}`",
            f"- min score delta vs scene: `{knn.get('min_score_delta_vs_scene')}`",
            f"- forbid fixed when scene non-fixed: `{knn.get('forbid_fixed_when_scene_nonfixed')}`",
            f"- reject variant: `{knn.get('reject_variant')}`",
            f"- source safe vs fixed: `{knn.get('source_safe_vs_fixed')}`",
            f"- source PSNR delta vs scene: `{knn.get('source_mean_psnr_delta_vs_scene_selected')}`",
            f"- source SSIM delta vs scene: `{knn.get('source_mean_ssim_delta_vs_scene_selected')}`",
            f"- source CVaR20 PSNR delta vs scene: `{knn.get('source_cvar_psnr_delta_vs_scene_selected')}`",
            f"- source min PSNR delta vs scene: `{knn.get('source_min_psnr_delta_vs_scene_selected')}`",
            f"- source positive-view delta vs scene: `{knn.get('source_positive_view_fraction_delta_vs_scene_selected')}`",
            f"- source selected counts: `{knn.get('source_selected_counts')}`",
            f"- source local-tail reject count: `{knn.get('source_local_tail_reject_count')}`",
            f"- verdict: `{knn.get('verdict')}`",
        ]
    if payload.get("local_support_policy"):
        local_support = payload["local_support_policy"]
        lines += [
            "",
            "## Source-Heldout Local Support Policy",
            "",
            f"- enabled: `{local_support.get('enabled')}`",
            f"- k: `{local_support.get('k')}`",
            f"- reject variant: `{local_support.get('reject_variant')}`",
            f"- source accept fraction: `{local_support.get('source_accept_fraction')}`",
            f"- source safe vs fixed: `{local_support.get('source_safe_vs_fixed')}`",
            f"- source PSNR delta vs scene: `{local_support.get('source_mean_psnr_delta_vs_scene_selected')}`",
            f"- source SSIM delta vs scene: `{local_support.get('source_mean_ssim_delta_vs_scene_selected')}`",
            f"- source CVaR20 PSNR delta vs scene: `{local_support.get('source_cvar_psnr_delta_vs_scene_selected')}`",
            f"- source min PSNR delta vs scene: `{local_support.get('source_min_psnr_delta_vs_scene_selected')}`",
            f"- source positive-view delta vs scene: `{local_support.get('source_positive_view_fraction_delta_vs_scene_selected')}`",
            f"- source selected counts: `{local_support.get('source_selected_counts')}`",
            f"- min local PSNR delta vs scene: `{local_support.get('min_local_psnr_delta_vs_scene')}`",
            f"- min score delta vs scene: `{local_support.get('min_score_delta_vs_scene')}`",
            f"- forbid fixed when scene non-fixed: `{local_support.get('forbid_fixed_when_scene_nonfixed')}`",
            f"- post incumbent fallback only: `{local_support.get('post_incumbent_fallback_only')}`",
            f"- verdict: `{local_support.get('verdict')}`",
        ]
    if payload.get("pairwise_dominance_policy"):
        pairwise = payload["pairwise_dominance_policy"]
        lines += [
            "",
            "## Pairwise Candidate-vs-Incumbent Dominance Policy",
            "",
            f"- enabled: `{pairwise.get('enabled')}`",
            f"- ridge: `{pairwise.get('ridge')}`",
            f"- k: `{pairwise.get('k')}`",
            f"- source accept fraction: `{pairwise.get('source_accept_fraction')}`",
            f"- source PSNR delta vs incumbent: `{pairwise.get('source_mean_psnr_delta_vs_incumbent')}`",
            f"- source SSIM delta vs incumbent: `{pairwise.get('source_mean_ssim_delta_vs_incumbent')}`",
            f"- source CVaR20 PSNR delta vs incumbent: `{pairwise.get('source_cvar_psnr_delta_vs_incumbent')}`",
            f"- source min PSNR delta vs incumbent: `{pairwise.get('source_min_psnr_delta_vs_incumbent')}`",
            f"- source selected counts: `{pairwise.get('source_selected_counts')}`",
            f"- OOD guard: `{pairwise.get('ood_guard_enabled')}` / `{pairwise.get('ood_quantile')}`",
            f"- OOD threshold: `{pairwise.get('ood_source_distance_threshold')}`",
            f"- verdict: `{pairwise.get('verdict')}`",
        ]
    if payload.get("per_view_risk_model_policy"):
        risk = payload["per_view_risk_model_policy"]
        lines += [
            "",
            "## Per-View Learned Risk Model",
            "",
            f"- enabled: `{risk.get('enabled')}`",
            f"- ridge: `{risk.get('ridge')}`",
            f"- reject variant: `{risk.get('reject_variant')}`",
            f"- auto objective margin: `{risk.get('auto_objective_margin')}`",
            f"- selected objective margin: `{risk.get('selected_objective_margin')}`",
            f"- source accept fraction: `{risk.get('source_accept_fraction')}`",
            f"- source safe vs fixed: `{risk.get('source_safe_vs_fixed')}`",
            f"- source PSNR delta vs scene: `{risk.get('source_mean_psnr_delta_vs_scene_selected')}`",
            f"- source SSIM delta vs scene: `{risk.get('source_mean_ssim_delta_vs_scene_selected')}`",
            f"- source CVaR20 PSNR delta vs scene: `{risk.get('source_cvar_psnr_delta_vs_scene_selected')}`",
            f"- source min PSNR delta vs scene: `{risk.get('source_min_psnr_delta_vs_scene_selected')}`",
            f"- source positive-view delta vs scene: `{risk.get('source_positive_view_fraction_delta_vs_scene_selected')}`",
            f"- source selected counts: `{risk.get('source_selected_counts')}`",
            f"- verdict: `{risk.get('verdict')}`",
        ]
    if payload.get("source_reliability_policy"):
        reliability = payload["source_reliability_policy"]
        lines += [
            "",
            "## Source-Only Reliability Policy",
            "",
            f"- enabled: `{reliability.get('enabled')}`",
            f"- ridge: `{reliability.get('ridge')}`",
            f"- reject variant: `{reliability.get('reject_variant')}`",
            f"- auto objective margin: `{reliability.get('auto_objective_margin')}`",
            f"- selected objective margin: `{reliability.get('selected_objective_margin')}`",
            f"- source accept fraction: `{reliability.get('source_accept_fraction')}`",
            f"- source safe vs fixed: `{reliability.get('source_safe_vs_fixed')}`",
            f"- source PSNR delta vs scene: `{reliability.get('source_mean_psnr_delta_vs_scene_selected')}`",
            f"- source SSIM delta vs scene: `{reliability.get('source_mean_ssim_delta_vs_scene_selected')}`",
            f"- source CVaR20 PSNR delta vs scene: `{reliability.get('source_cvar_psnr_delta_vs_scene_selected')}`",
            f"- source min PSNR delta vs scene: `{reliability.get('source_min_psnr_delta_vs_scene_selected')}`",
            f"- source positive-view delta vs scene: `{reliability.get('source_positive_view_fraction_delta_vs_scene_selected')}`",
            f"- source selected counts: `{reliability.get('source_selected_counts')}`",
            f"- decision prediction source: `{reliability.get('decision_prediction_source')}`",
            f"- calibration: `{reliability.get('source_reliability_calibration')}`",
            f"- verdict: `{reliability.get('verdict')}`",
        ]
    if payload.get("promotion_rollback_policy"):
        promotion = payload["promotion_rollback_policy"]
        lines += [
            "",
            "## Post-Decision Promotion Rollback Certificate",
            "",
            f"- enabled: `{promotion.get('enabled')}`",
            f"- mode: `{promotion.get('mode')}`",
            f"- checked sources: `{promotion.get('sources')}`",
            f"- source sample count: `{promotion.get('source_sample_count')}`",
            f"- calibration quantile / scale: `{promotion.get('calibration_quantile')}` / `{promotion.get('calibration_scale')}`",
            f"- LCB min objective / PSNR / SSIM: `{promotion.get('min_lcb_objective_delta')}` / `{promotion.get('min_lcb_psnr_delta')}` / `{promotion.get('min_lcb_ssim_delta')}`",
            f"- local min CVaR / min / max negative fraction: `{promotion.get('min_local_cvar_delta')}` / `{promotion.get('min_local_min_delta')}` / `{promotion.get('max_local_negative_fraction')}`",
            f"- keep / shadow rollback / rollback: `{promotion.get('keep_count')}` / `{promotion.get('shadow_rollback_count')}` / `{promotion.get('rollback_count')}`",
            f"- reason counts: `{promotion.get('reason_counts')}`",
            f"- verdict: `{promotion.get('verdict')}`",
        ]
    if payload.get("target_neighbor_consistency_policy"):
        target_neighbor = payload["target_neighbor_consistency_policy"]
        lines += [
            "",
            "## Target-Neighbor Consistency Certificate",
            "",
            f"- enabled / mode: `{target_neighbor.get('enabled')}` / `{target_neighbor.get('mode')}`",
            f"- checked sources: `{target_neighbor.get('sources')}`",
            f"- min incumbent-minus-output MAE delta: `{target_neighbor.get('min_incumbent_minus_output_delta')}`",
            f"- neighbor k / max side: `{target_neighbor.get('neighbor_k')}` / `{target_neighbor.get('max_side')}`",
            f"- source contradiction enabled: `{target_neighbor.get('source_contradiction_enabled')}`",
            f"- contradiction source local min / CVaR / positive fraction: `{target_neighbor.get('contradiction_min_source_local_min_delta')}` / `{target_neighbor.get('contradiction_min_source_local_cvar_delta')}` / `{target_neighbor.get('contradiction_min_source_positive_fraction')}`",
            f"- contradiction max incumbent-minus-output delta: `{target_neighbor.get('contradiction_max_incumbent_minus_output_delta')}`",
            f"- target GT used: `{target_neighbor.get('uses_target_gt')}`",
            f"- keep / shadow rollback / rollback: `{target_neighbor.get('keep_count')}` / `{target_neighbor.get('shadow_rollback_count')}` / `{target_neighbor.get('rollback_count')}`",
            f"- reason counts: `{target_neighbor.get('reason_counts')}`",
        ]
    if payload.get("target_neighbor_candidate_unlock_policy"):
        unlock = payload["target_neighbor_candidate_unlock_policy"]
        lines += [
            "",
            "## Target-Neighbor Candidate Unlock",
            "",
            f"- enabled: `{unlock.get('enabled')}`",
            f"- incumbent / candidate: `{unlock.get('incumbent_variant')}` / `{unlock.get('candidate_variant')}`",
            f"- min incumbent-minus-candidate MAE delta: `{unlock.get('min_incumbent_minus_candidate_delta')}`",
            f"- target GT used: `{unlock.get('uses_target_gt')}`",
            f"- keep / promote / skipped: `{unlock.get('keep_count')}` / `{unlock.get('promote_count')}` / `{unlock.get('skipped_count')}`",
            f"- reason counts: `{unlock.get('reason_counts')}`",
        ]
    (output_dir / "support_transport_apply_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


BASE_CANDIDATE_VARIANTS = ("fixed", "learned", "hybrid")


def _variant_name_for_blend(value: float) -> str:
    return f"mix{int(round(float(value) * 1000.0)):04d}"


def _candidate_ladder_blends(args: argparse.Namespace) -> list[float]:
    if not bool(getattr(args, "enable_candidate_ladder", False)):
        return []
    values: list[float] = []
    seen = {0.0, 1.0, float(args.blend)}
    for raw in str(getattr(args, "candidate_ladder_blends", "")).split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = float(raw)
        if value <= 0.0 or value >= 1.0:
            continue
        if any(abs(value - prev) < 1.0e-6 for prev in seen):
            continue
        seen.add(value)
        values.append(value)
    return sorted(values)


def _candidate_variant_blend_map(args: argparse.Namespace) -> dict[str, float]:
    blend_map = {"fixed": 0.0, "learned": 1.0, "hybrid": float(args.blend)}
    for value in _candidate_ladder_blends(args):
        name = _variant_name_for_blend(value)
        if name in blend_map:
            raise ValueError(f"candidate ladder blend name collision for {value}: {name}")
        blend_map[name] = float(value)
    return blend_map


def _candidate_variant_names(args: argparse.Namespace) -> list[str]:
    return list(_candidate_variant_blend_map(args).keys())


def _candidate_count_dict(variants: list[str]) -> dict[str, int]:
    return {variant: 0 for variant in variants} | {"noop": 0, "scene": 0}


def _candidate_deltas(ev: Any, pred_delta: torch.Tensor, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    fixed_delta = float(args.anchor_alpha) * ev.signal
    learned_delta = float(args.learned_scale) * pred_delta
    hybrid_delta = (1.0 - float(args.blend)) * fixed_delta + float(args.blend) * learned_delta
    deltas = {"fixed": fixed_delta, "learned": learned_delta, "hybrid": hybrid_delta}
    for value in _candidate_ladder_blends(args):
        deltas[_variant_name_for_blend(value)] = (1.0 - float(value)) * fixed_delta + float(value) * learned_delta
    return deltas


def _changed_fraction_from_delta(delta: torch.Tensor) -> float:
    changed = torch.any(torch.abs(delta) > (0.5 / 255.0), dim=0)
    return float(torch.mean(changed.to(torch.float32)).detach().cpu().item())


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    value = value.to(dtype=torch.float32)
    if mask is None:
        return float(torch.mean(value).detach().cpu().item())
    weight = mask.to(device=value.device, dtype=value.dtype)
    denom = torch.clamp(weight.sum(), min=1.0)
    return float((value * weight).sum().detach().cpu().item() / float(denom.detach().cpu().item()))


def _candidate_proxy_stats(ev: Any, delta: torch.Tensor) -> dict[str, float]:
    valid = ev.valid.to(device=delta.device, dtype=torch.float32)
    valid3 = valid.expand_as(delta)
    abs_delta = torch.abs(delta)
    signal = ev.signal.to(device=delta.device, dtype=torch.float32)
    abs_signal = torch.abs(signal)
    confidence = ev.confidence.to(device=delta.device, dtype=torch.float32)
    residual_std = (
        ev.residual_std.to(device=delta.device, dtype=torch.float32)
        if getattr(ev, "residual_std", None) is not None
        else torch.zeros_like(valid)
    )
    support_count = (
        ev.support_count.to(device=delta.device, dtype=torch.float32)
        if getattr(ev, "support_count", None) is not None
        else valid
    )
    mean_abs_delta = _masked_mean(abs_delta, valid3)
    mean_abs_signal = _masked_mean(abs_signal, valid3)
    residual_std_mean = _masked_mean(residual_std, valid)
    confidence_mean = _masked_mean(confidence, valid)
    support_count_mean = _masked_mean(support_count, valid)
    covered = float(valid.mean().detach().cpu().item())
    eps = 1.0e-6
    pixel_dot = torch.sum(delta * signal, dim=0, keepdim=True)
    delta_sq = torch.sum(delta.pow(2), dim=0, keepdim=True)
    signal_sq = torch.sum(signal.pow(2), dim=0, keepdim=True)
    valid_weight = valid.to(device=delta.device, dtype=torch.float32)
    global_dot = float((pixel_dot * valid_weight).sum().detach().cpu().item())
    global_delta_energy = float((delta_sq * valid_weight).sum().detach().cpu().item())
    global_signal_energy = float((signal_sq * valid_weight).sum().detach().cpu().item())
    delta_signal_cosine = global_dot / max((global_delta_energy * global_signal_energy) ** 0.5, eps)
    active = (valid_weight > 0.0) & (delta_sq > eps) & (signal_sq > eps)
    active_count = float(active.to(torch.float32).sum().detach().cpu().item())
    if active_count > 0.0:
        opposition_fraction = float(((pixel_dot < 0.0) & active).to(torch.float32).sum().detach().cpu().item() / active_count)
        aligned_fraction = float(((pixel_dot >= 0.0) & active).to(torch.float32).sum().detach().cpu().item() / active_count)
    else:
        opposition_fraction = 0.0
        aligned_fraction = 0.0
    return {
        "covered_fraction": covered,
        "mean_abs_delta": mean_abs_delta,
        "mean_abs_signal": mean_abs_signal,
        "confidence_mean": confidence_mean,
        "residual_std_mean": residual_std_mean,
        "support_count_mean": support_count_mean,
        "changed_fraction": _changed_fraction_from_delta(delta),
        "delta_snr": mean_abs_delta / (residual_std_mean + eps),
        "signal_snr": mean_abs_signal / (residual_std_mean + eps),
        "confidence_snr": confidence_mean * mean_abs_delta / (residual_std_mean + eps),
        "residual_stability": 1.0 / (residual_std_mean + eps),
        "delta_signal_cosine": float(delta_signal_cosine),
        "opposition_fraction": opposition_fraction,
        "aligned_fraction": aligned_fraction,
        "delta_to_signal_ratio": mean_abs_delta / (mean_abs_signal + eps),
        "std_to_signal_ratio": residual_std_mean / (mean_abs_signal + eps),
        "support_confidence": support_count_mean * confidence_mean,
    }


def _parse_score_names(text: str) -> list[str]:
    names = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            names.append(item)
    return names or ["confidence_snr"]


def _parse_feature_names(text: str) -> list[str]:
    return _parse_score_names(text)


def _noop_like(row: dict[str, float]) -> dict[str, float]:
    out = dict(row)
    out["candidate_mse"] = float(row.get("base_mse", 0.0))
    out["mse_reduction"] = 0.0
    out["candidate_psnr"] = float(row.get("base_psnr", 0.0))
    out["psnr_gain"] = 0.0
    out["changed_fraction"] = 0.0
    out["mean_abs_delta"] = 0.0
    if "base_ssim" in row:
        out["candidate_ssim"] = float(row.get("base_ssim", 0.0))
        out["ssim_gain"] = 0.0
    return out


def _score_from_proxy(proxy: dict[str, float], score_name: str) -> float:
    if score_name not in proxy:
        raise KeyError(f"unknown per-view gate score `{score_name}`; available={sorted(proxy.keys())}")
    return float(proxy[score_name])


def _feature_vector(proxy: dict[str, float], feature_names: list[str]) -> list[float]:
    return [_score_from_proxy(proxy, name) for name in feature_names]


def _objective_from_metrics(row: dict[str, float], *, compute_ssim: bool) -> float:
    return float(row.get("psnr_gain", 0.0)) + (20.0 * float(row.get("ssim_gain", 0.0)) if compute_ssim else 0.0)


def _source_objective_from_metrics(row: dict[str, float], *, compute_ssim: bool, args: argparse.Namespace) -> float:
    return (
        _objective_from_metrics(row, compute_ssim=compute_ssim)
        + float(args.source_objective_lpips_weight) * float(row.get("lpips_gain", 0.0))
        + float(args.source_objective_dists_weight) * float(row.get("dists_gain", 0.0))
    )


def _risk_model_objective_from_metrics(row: dict[str, float], *, compute_ssim: bool, args: argparse.Namespace) -> float:
    if bool(args.per_view_risk_model_use_source_perceptual_objective):
        return _source_objective_from_metrics(row, compute_ssim=compute_ssim, args=args)
    return _objective_from_metrics(row, compute_ssim=compute_ssim)


def _tail_values(values: list[float], fraction: float = 0.20) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "cvar": 0.0}
    ordered = sorted(float(v) for v in values)
    count = max(1, int(math.ceil(float(fraction) * len(ordered))))
    return {"min": float(ordered[0]), "cvar": float(sum(ordered[:count]) / count)}


def _downscale_for_perceptual(image: torch.Tensor, max_side: int) -> torch.Tensor:
    max_side = int(max_side)
    if max_side <= 0:
        return image
    h, w = int(image.shape[-2]), int(image.shape[-1])
    current = max(h, w)
    if current <= max_side:
        return image
    scale = float(max_side) / float(current)
    out_h = max(16, int(round(h * scale)))
    out_w = max(16, int(round(w * scale)))
    return F.interpolate(image.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze(0)


def _load_source_perceptual_models(device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    models: dict[str, Any] = {"enabled": bool(args.compute_source_perceptual), "lpips": None, "dists": None}
    if not bool(args.compute_source_perceptual):
        return models
    import lpips

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for param in lpips_model.parameters():
        param.requires_grad_(False)
    models["lpips"] = lpips_model
    try:
        import piq

        dists_model = piq.DISTS(reduction="mean").to(device).eval()
        for param in dists_model.parameters():
            param.requires_grad_(False)
        models["dists"] = dists_model
        models["dists_status"] = "computed_piq_DISTS_reduction_mean"
    except ImportError:
        models["dists_status"] = "not_computed_missing_piq"
    return models


def _augment_source_perceptual_metrics(
    row: dict[str, float],
    *,
    base: torch.Tensor,
    gt: torch.Tensor,
    delta: torch.Tensor,
    models: dict[str, Any],
    max_side: int,
) -> None:
    if not bool(models.get("enabled", False)):
        return
    candidate = torch.clamp(base + delta, 0.0, 1.0)
    base_small = _downscale_for_perceptual(base, int(max_side)).unsqueeze(0)
    candidate_small = _downscale_for_perceptual(candidate, int(max_side)).unsqueeze(0)
    gt_small = _downscale_for_perceptual(gt, int(max_side)).unsqueeze(0)
    with torch.no_grad():
        lpips_model = models.get("lpips")
        if lpips_model is not None:
            base_lpips = float(lpips_model(base_small * 2.0 - 1.0, gt_small * 2.0 - 1.0).detach().cpu().reshape(-1)[0].item())
            candidate_lpips = float(
                lpips_model(candidate_small * 2.0 - 1.0, gt_small * 2.0 - 1.0).detach().cpu().reshape(-1)[0].item()
            )
            row.update(
                {
                    "base_lpips": base_lpips,
                    "candidate_lpips": candidate_lpips,
                    "lpips_gain": float(base_lpips - candidate_lpips),
                }
            )
        dists_model = models.get("dists")
        if dists_model is not None:
            base_dists = float(dists_model(base_small, gt_small).detach().cpu().reshape(-1)[0].item())
            candidate_dists = float(dists_model(candidate_small, gt_small).detach().cpu().reshape(-1)[0].item())
            row.update(
                {
                    "base_dists": base_dists,
                    "candidate_dists": candidate_dists,
                    "dists_gain": float(base_dists - candidate_dists),
                }
            )


def _candidate_learning_features(
    proxy: dict[str, float],
    variant: str,
    feature_names: list[str],
    *,
    variant_blend: float | None = None,
) -> list[float]:
    base = _feature_vector(proxy, feature_names)
    return base + [
        1.0 if variant == "fixed" else 0.0,
        1.0 if variant == "learned" else 0.0,
        1.0 if variant == "hybrid" else 0.0,
        float(variant_blend) if variant_blend is not None else 0.0,
    ]


def _source_reliability_features(
    candidate_proxy: dict[str, float],
    scene_proxy: dict[str, float],
    *,
    variant: str,
    scene_variant: str,
    feature_names: list[str],
    variant_blend: float | None = None,
    scene_variant_blend: float | None = None,
) -> list[float]:
    candidate = _feature_vector(candidate_proxy, feature_names)
    scene = _feature_vector(scene_proxy, feature_names)
    delta = [float(cv) - float(sv) for cv, sv in zip(candidate, scene)]
    return (
        candidate
        + scene
        + delta
        + [
            1.0 if variant == "fixed" else 0.0,
            1.0 if variant == "learned" else 0.0,
            1.0 if variant == "hybrid" else 0.0,
            1.0 if scene_variant == "fixed" else 0.0,
            1.0 if scene_variant == "learned" else 0.0,
            1.0 if scene_variant == "hybrid" else 0.0,
            1.0 if variant == scene_variant else 0.0,
            float(variant_blend) if variant_blend is not None else 0.0,
            float(scene_variant_blend) if scene_variant_blend is not None else 0.0,
            (float(variant_blend) - float(scene_variant_blend))
            if variant_blend is not None and scene_variant_blend is not None
            else 0.0,
        ]
    )


SOURCE_RELIABILITY_TARGET_NAMES = [
    "objective_delta_vs_scene",
    "psnr_delta_vs_scene",
    "ssim_delta_vs_scene",
    "lpips_delta_vs_scene",
    "dists_delta_vs_scene",
]


def _metric_delta(candidate: dict[str, float], scene: dict[str, float], key: str) -> float:
    return float(candidate.get(key, 0.0)) - float(scene.get(key, 0.0))


def _source_reliability_target(
    metrics: dict[str, float],
    scene_metrics: dict[str, float],
    *,
    compute_ssim: bool,
    args: argparse.Namespace,
) -> list[float]:
    scene_objective = _source_objective_from_metrics(scene_metrics, compute_ssim=compute_ssim, args=args)
    return [
        _source_objective_from_metrics(metrics, compute_ssim=compute_ssim, args=args) - scene_objective,
        _metric_delta(metrics, scene_metrics, "psnr_gain"),
        _metric_delta(metrics, scene_metrics, "ssim_gain") if compute_ssim else 0.0,
        _metric_delta(metrics, scene_metrics, "lpips_gain"),
        _metric_delta(metrics, scene_metrics, "dists_gain"),
    ]


def _calibrated_lower_bounds(prediction: list[float], calibration: dict[str, Any] | None) -> list[float]:
    if not calibration or not bool(calibration.get("enabled", False)):
        return [float(value) for value in prediction]
    bounds = [float(value) for value in calibration.get("error_bounds", [])]
    if len(bounds) < len(prediction):
        bounds = bounds + [0.0] * (len(prediction) - len(bounds))
    return [float(value) - float(bound) for value, bound in zip(prediction, bounds)]


def _fit_ridge_predictor(
    examples: list[dict[str, Any]],
    *,
    ridge: float,
) -> dict[str, Any] | None:
    if len(examples) < 2:
        return None
    x = torch.tensor([example["features"] for example in examples], dtype=torch.float64)
    y = torch.tensor([example["target"] for example in examples], dtype=torch.float64)
    mean = x.mean(dim=0, keepdim=True)
    std = torch.clamp(x.std(dim=0, keepdim=True, unbiased=False), min=1.0e-6)
    xn = (x - mean) / std
    ones = torch.ones((xn.shape[0], 1), dtype=xn.dtype)
    design = torch.cat([ones, xn], dim=1)
    reg = torch.eye(design.shape[1], dtype=xn.dtype) * float(ridge)
    reg[0, 0] = 0.0
    lhs = design.T @ design + reg
    rhs = design.T @ y
    try:
        weight = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        weight = torch.linalg.pinv(lhs) @ rhs
    return {
        "mean": mean.squeeze(0).tolist(),
        "std": std.squeeze(0).tolist(),
        "weight": weight.tolist(),
        "feature_dim": int(x.shape[1]),
    }


def _predict_ridge(model: dict[str, Any], features: list[float]) -> list[float]:
    expected_dim = int(model.get("feature_dim", len(model["mean"])))
    if len(features) != expected_dim:
        raise ValueError(f"ridge feature dimension mismatch: got {len(features)}, expected {expected_dim}")
    x = torch.tensor(features, dtype=torch.float64)
    mean = torch.tensor(model["mean"], dtype=torch.float64)
    std = torch.tensor(model["std"], dtype=torch.float64)
    weight = torch.tensor(model["weight"], dtype=torch.float64)
    design = torch.cat([torch.ones(1, dtype=torch.float64), (x - mean) / torch.clamp(std, min=1.0e-6)])
    pred = design @ weight
    return [float(v) for v in pred.tolist()]


def _feature_stats(vectors: list[list[float]]) -> tuple[list[float], list[float]]:
    if not vectors:
        return [], []
    dims = len(vectors[0])
    means = []
    stds = []
    for dim in range(dims):
        vals = [float(v[dim]) for v in vectors]
        mean = sum(vals) / max(len(vals), 1)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)
        means.append(float(mean))
        stds.append(float(max(var, 1.0e-12) ** 0.5))
    return means, stds


def _normalized_distance(a: list[float], b: list[float], mean: list[float], std: list[float]) -> float:
    total = 0.0
    for av, bv, mv, sv in zip(a, b, mean, std):
        da = (float(av) - float(mv)) / max(float(sv), 1.0e-6)
        db = (float(bv) - float(mv)) / max(float(sv), 1.0e-6)
        total += (da - db) ** 2
    return float(total ** 0.5)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("inf")
    ordered = sorted(float(v) for v in values)
    clamped = min(max(float(q), 0.0), 1.0)
    idx = int(round(clamped * (len(ordered) - 1)))
    return float(ordered[idx])


def _summarize_metric_rows(rows: list[dict[str, float]], *, compute_ssim: bool) -> dict[str, Any]:
    if not rows:
        return {}
    summary = dict(_summarize_rows(rows, compute_ssim=compute_ssim))
    if any("lpips_gain" in row for row in rows):
        lpips_gain = [float(row.get("lpips_gain", 0.0)) for row in rows]
        summary.update(
            {
                "base_lpips": _mean([float(row.get("base_lpips", 0.0)) for row in rows]),
                "candidate_lpips": _mean([float(row.get("candidate_lpips", 0.0)) for row in rows]),
                "lpips_gain": _mean(lpips_gain),
                "lpips_gain_tail": _tail_values(lpips_gain),
                "lpips_positive_view_fraction": _mean([1.0 if value > 0.0 else 0.0 for value in lpips_gain]),
            }
        )
    if any("dists_gain" in row for row in rows):
        dists_gain = [float(row.get("dists_gain", 0.0)) for row in rows]
        summary.update(
            {
                "base_dists": _mean([float(row.get("base_dists", 0.0)) for row in rows]),
                "candidate_dists": _mean([float(row.get("candidate_dists", 0.0)) for row in rows]),
                "dists_gain": _mean(dists_gain),
                "dists_gain_tail": _tail_values(dists_gain),
                "dists_positive_view_fraction": _mean([1.0 if value > 0.0 else 0.0 for value in dists_gain]),
            }
        )
    return summary


def _summary_psnr_tail(summary: dict[str, Any], name: str) -> float:
    tail = summary.get("psnr_gain_tail", {})
    return float(tail.get(name, 0.0))


def _positive_view_fraction(summary: dict[str, Any]) -> float:
    return float(summary.get("positive_view_fraction", 0.0))


def _knn_local_metric_summary(
    entries_by_variant: dict[str, list[dict[str, Any]]],
    variant: str,
    vector: list[float],
    *,
    mean: list[float],
    std: list[float],
    k: int,
    compute_ssim: bool,
    exclude_view: str | None = None,
) -> dict[str, Any]:
    pool = [
        entry
        for entry in entries_by_variant.get(variant, [])
        if exclude_view is None or str(entry["view"]) != str(exclude_view)
    ]
    if not pool:
        return {"available": False, "variant": variant, "k": 0, "summary": {}, "neighbors": []}
    ranked = sorted(
        pool,
        key=lambda entry: _normalized_distance(vector, entry["vector"], mean, std),
    )
    local_k = max(1, min(int(k), len(ranked)))
    neighbors = ranked[:local_k]
    rows = [entry["metrics"] for entry in neighbors]
    summary = _summarize_metric_rows(rows, compute_ssim=compute_ssim)
    return {
        "available": True,
        "variant": variant,
        "k": local_k,
        "summary": summary,
        "neighbors": [
            {
                "view": str(entry["view"]),
                "distance": float(_normalized_distance(vector, entry["vector"], mean, std)),
                "score": float(entry["score"]),
                "psnr_gain": float(entry["metrics"].get("psnr_gain", 0.0)),
                "ssim_gain": float(entry["metrics"].get("ssim_gain", 0.0)) if compute_ssim else 0.0,
            }
            for entry in neighbors
        ],
    }


def _knn_local_tail_guard_decision(
    candidate_variant: str,
    scene_variant: str,
    local_summaries: dict[str, dict[str, Any]],
    *,
    compute_ssim: bool,
    min_psnr_delta_vs_scene: float,
    min_ssim_delta_vs_scene: float,
    min_cvar_delta_vs_scene: float,
    min_min_delta_vs_scene: float,
    min_positive_fraction_delta_vs_scene: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": True,
        "candidate_variant": candidate_variant,
        "scene_variant": scene_variant,
        "rejected": False,
        "reason": "passed",
        "deltas_vs_scene": {},
        "candidate_local": local_summaries.get(candidate_variant),
        "scene_local": local_summaries.get(scene_variant),
    }
    if candidate_variant in {"noop", "__scene__"}:
        result["reason"] = "non_candidate"
        return result
    if candidate_variant == scene_variant:
        result["reason"] = "candidate_is_scene"
        return result
    candidate_local = local_summaries.get(candidate_variant, {})
    scene_local = local_summaries.get(scene_variant, {})
    if not bool(candidate_local.get("available", False)) or not bool(scene_local.get("available", False)):
        result["rejected"] = True
        result["reason"] = "missing_local_source_neighbors"
        return result
    candidate_summary = candidate_local.get("summary", {})
    scene_summary = scene_local.get("summary", {})
    deltas = {
        "psnr_gain": float(candidate_summary.get("psnr_gain", 0.0)) - float(scene_summary.get("psnr_gain", 0.0)),
        "cvar_psnr_gain": _summary_psnr_tail(candidate_summary, "cvar") - _summary_psnr_tail(scene_summary, "cvar"),
        "min_psnr_gain": _summary_psnr_tail(candidate_summary, "min") - _summary_psnr_tail(scene_summary, "min"),
        "positive_view_fraction": _positive_view_fraction(candidate_summary) - _positive_view_fraction(scene_summary),
    }
    if compute_ssim:
        deltas["ssim_gain"] = float(candidate_summary.get("ssim_gain", 0.0)) - float(scene_summary.get("ssim_gain", 0.0))
    result["deltas_vs_scene"] = deltas
    failures = []
    if deltas["psnr_gain"] < float(min_psnr_delta_vs_scene):
        failures.append("local_psnr")
    if compute_ssim and deltas.get("ssim_gain", 0.0) < float(min_ssim_delta_vs_scene):
        failures.append("local_ssim")
    if deltas["cvar_psnr_gain"] < float(min_cvar_delta_vs_scene):
        failures.append("local_cvar")
    if deltas["min_psnr_gain"] < float(min_min_delta_vs_scene):
        failures.append("local_min")
    if deltas["positive_view_fraction"] < float(min_positive_fraction_delta_vs_scene):
        failures.append("local_positive_fraction")
    if failures:
        result["rejected"] = True
        result["reason"] = ",".join(failures)
    return result


def _fit_per_view_knn_policy(
    selector_payload: dict[str, Any] | None,
    *,
    compute_ssim: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not bool(args.enable_per_view_knn_policy):
        return {"enabled": False, "verdict": "disabled by CLI"}
    if selector_payload is None or not selector_payload.get("per_view"):
        return {"enabled": False, "verdict": "missing source-heldout selector per-view evidence"}
    selected_variant = str(selector_payload["selected_variant"])
    if selected_variant == "fixed" and not bool(args.per_view_knn_allow_when_scene_fixed):
        return {
            "enabled": False,
            "selected_variant": selected_variant,
            "verdict": "disabled because scene-level source-heldout selector fell back to fixed",
        }
    feature_names = _parse_feature_names(args.per_view_knn_feature_grid)
    variants = list(BASE_CANDIDATE_VARIANTS) if bool(args.per_view_knn_base_variants_only) else _candidate_variant_names(args)
    entries_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    all_vectors: list[list[float]] = []
    for row in selector_payload["per_view"]:
        for variant in variants:
            candidate = row["candidates"][variant]
            vector = _feature_vector(candidate["proxy"], feature_names)
            entry = {
                "view": row["view"],
                "variant": variant,
                "vector": vector,
                "metrics": candidate["metrics"],
                "score": _source_objective_from_metrics(candidate["metrics"], compute_ssim=compute_ssim, args=args),
            }
            entries_by_variant[variant].append(entry)
            all_vectors.append(vector)
    mean, std = _feature_stats(all_vectors)
    if not mean:
        return {"enabled": False, "verdict": "no usable source-heldout proxy vectors"}
    fixed_summary = selector_payload["summaries"]["fixed"]
    selected_variant = str(selector_payload["selected_variant"])
    scene_selected_summary = selector_payload["summaries"][selected_variant]
    local_tail_guard_enabled = bool(args.per_view_knn_enable_local_tail_guard)
    local_tail_k = int(args.per_view_knn_local_tail_k)
    if local_tail_k <= 0:
        local_tail_k = int(args.per_view_knn_k)

    def predict(variant: str, vector: list[float], *, exclude_view: str | None = None) -> float:
        pool = [
            entry
            for entry in entries_by_variant[variant]
            if exclude_view is None or str(entry["view"]) != str(exclude_view)
        ]
        if not pool:
            return float("-inf")
        ranked = sorted(
            pool,
            key=lambda entry: _normalized_distance(vector, entry["vector"], mean, std),
        )
        k = max(1, min(int(args.per_view_knn_k), len(ranked)))
        return _mean([float(entry["score"]) for entry in ranked[:k]])

    source_decisions: list[dict[str, Any]] = []
    for row in selector_payload["per_view"]:
        predictions = {}
        local_summaries = {}
        for variant in variants:
            vector = _feature_vector(row["candidates"][variant]["proxy"], feature_names)
            predictions[variant] = predict(variant, vector, exclude_view=row["view"])
            if local_tail_guard_enabled:
                local_summaries[variant] = _knn_local_metric_summary(
                    entries_by_variant,
                    variant,
                    vector,
                    mean=mean,
                    std=std,
                    k=local_tail_k,
                    compute_ssim=compute_ssim,
                    exclude_view=row["view"],
                )
        best_variant, best_score = max(predictions.items(), key=lambda item: item[1])
        source_decisions.append(
            {
                "view": row["view"],
                "predictions": predictions,
                "best_variant": best_variant,
                "best_score": float(best_score),
                "local_summaries": local_summaries,
                "metrics": row["candidates"][best_variant]["metrics"],
                "scene_metrics": row["candidates"][selected_variant]["metrics"],
            }
        )

    local_tail_guard_payload = {
        "enabled": local_tail_guard_enabled,
        "k": local_tail_k,
        "min_psnr_delta_vs_scene": float(args.per_view_knn_min_local_psnr_delta_vs_scene),
        "min_ssim_delta_vs_scene": float(args.per_view_knn_min_local_ssim_delta_vs_scene),
        "min_cvar_delta_vs_scene": float(args.per_view_knn_min_local_cvar_delta_vs_scene),
        "min_min_delta_vs_scene": float(args.per_view_knn_min_local_min_delta_vs_scene),
        "min_positive_fraction_delta_vs_scene": float(args.per_view_knn_min_local_positive_fraction_delta_vs_scene),
    }

    def evaluate_threshold(
        threshold: float,
    ) -> tuple[dict[str, Any], dict[str, int], list[dict[str, float]], dict[str, Any]]:
        source_policy_rows: list[dict[str, float]] = []
        selected_counts: dict[str, int] = _candidate_count_dict(variants)
        local_tail_reject_views: list[dict[str, Any]] = []
        for decision in source_decisions:
            scene_score = float(decision["predictions"][selected_variant])
            low_absolute_score = float(decision["best_score"]) < float(threshold)
            low_scene_margin = float(decision["best_score"]) < scene_score + float(args.per_view_knn_min_score_delta_vs_scene)
            fixed_downgrade = (
                bool(args.per_view_knn_forbid_fixed_when_scene_nonfixed)
                and selected_variant != "fixed"
                and str(decision["best_variant"]) == "fixed"
            )
            local_tail_guard = (
                _knn_local_tail_guard_decision(
                    str(decision["best_variant"]),
                    selected_variant,
                    decision.get("local_summaries", {}),
                    compute_ssim=compute_ssim,
                    min_psnr_delta_vs_scene=float(args.per_view_knn_min_local_psnr_delta_vs_scene),
                    min_ssim_delta_vs_scene=float(args.per_view_knn_min_local_ssim_delta_vs_scene),
                    min_cvar_delta_vs_scene=float(args.per_view_knn_min_local_cvar_delta_vs_scene),
                    min_min_delta_vs_scene=float(args.per_view_knn_min_local_min_delta_vs_scene),
                    min_positive_fraction_delta_vs_scene=float(args.per_view_knn_min_local_positive_fraction_delta_vs_scene),
                )
                if local_tail_guard_enabled
                else {"enabled": False, "rejected": False}
            )
            local_tail_rejected = bool(local_tail_guard.get("rejected", False))
            if local_tail_rejected:
                local_tail_reject_views.append(
                    {
                        "view": str(decision["view"]),
                        "best_variant": str(decision["best_variant"]),
                        "scene_variant": selected_variant,
                        "reason": str(local_tail_guard.get("reason", "")),
                        "deltas_vs_scene": dict(local_tail_guard.get("deltas_vs_scene", {})),
                    }
                )
            if low_absolute_score or low_scene_margin or fixed_downgrade or local_tail_rejected:
                if str(args.per_view_knn_reject_variant) == "scene":
                    selected_counts["scene"] += 1
                    source_policy_rows.append(decision["scene_metrics"])
                else:
                    selected_counts["noop"] += 1
                    source_policy_rows.append(_noop_like(decision["metrics"]))
            else:
                selected_counts[str(decision["best_variant"])] += 1
                source_policy_rows.append(decision["metrics"])
        local_tail_audit = {
            "reject_count": len(local_tail_reject_views),
            "reject_views": local_tail_reject_views,
        }
        return _summarize_metric_rows(source_policy_rows, compute_ssim=compute_ssim), selected_counts, source_policy_rows, local_tail_audit

    threshold_trials: list[dict[str, Any]] = []
    selected_threshold = float(args.per_view_knn_min_predicted_score)
    if bool(args.per_view_knn_auto_threshold):
        best_trial: dict[str, Any] | None = None
        scores = sorted({float(decision["best_score"]) for decision in source_decisions})
        if scores:
            scores = [min(scores) - 1.0e-9] + scores + [max(scores) + 1.0e-9]
        for threshold in scores:
            source_summary, selected_counts, _, local_tail_audit = evaluate_threshold(threshold)
            reject_count = selected_counts["noop"] + selected_counts["scene"]
            accept_fraction = 1.0 - float(reject_count / max(len(source_decisions), 1))
            mean_delta = float(source_summary.get("psnr_gain", 0.0)) - float(scene_selected_summary.get("psnr_gain", 0.0))
            ssim_delta = (
                float(source_summary.get("ssim_gain", 0.0)) - float(scene_selected_summary.get("ssim_gain", 0.0))
                if compute_ssim
                else 0.0
            )
            cvar_delta = _summary_psnr_tail(source_summary, "cvar") - _summary_psnr_tail(scene_selected_summary, "cvar")
            min_delta = _summary_psnr_tail(source_summary, "min") - _summary_psnr_tail(scene_selected_summary, "min")
            positive_fraction_delta = _positive_view_fraction(source_summary) - _positive_view_fraction(scene_selected_summary)
            safe_vs_fixed = (
                float(source_summary.get("psnr_gain", 0.0)) >= float(fixed_summary.get("psnr_gain", 0.0)) - float(args.selected_safe_tolerance_psnr)
                and (
                    not compute_ssim
                    or float(source_summary.get("ssim_gain", 0.0))
                    >= float(fixed_summary.get("ssim_gain", 0.0)) - float(args.selected_safe_tolerance_ssim)
                )
            )
            trial = {
                "threshold": float(threshold),
                "accept_fraction": accept_fraction,
                "source_summary": source_summary,
                "source_fixed_summary": fixed_summary,
                "source_scene_selected_summary": scene_selected_summary,
                "source_mean_psnr_delta_vs_scene_selected": mean_delta,
                "source_mean_ssim_delta_vs_scene_selected": ssim_delta,
                "source_cvar_psnr_delta_vs_scene_selected": cvar_delta,
                "source_min_psnr_delta_vs_scene_selected": min_delta,
                "source_positive_view_fraction_delta_vs_scene_selected": positive_fraction_delta,
                "source_safe_vs_fixed": bool(safe_vs_fixed),
                "source_selected_counts": selected_counts,
                "source_local_tail_reject_count": int(local_tail_audit["reject_count"]),
                "source_local_tail_reject_views": local_tail_audit["reject_views"],
            }
            threshold_trials.append(trial)
            if accept_fraction < float(args.per_view_knn_min_accept_fraction):
                continue
            if accept_fraction > float(args.per_view_knn_max_accept_fraction):
                continue
            if mean_delta < float(args.per_view_knn_min_source_psnr_delta):
                continue
            if compute_ssim and ssim_delta < float(args.per_view_knn_min_source_ssim_delta):
                continue
            if cvar_delta < float(args.per_view_knn_min_source_cvar_delta):
                continue
            if min_delta < float(args.per_view_knn_min_source_min_delta):
                continue
            if positive_fraction_delta < float(args.per_view_knn_min_source_positive_fraction_delta):
                continue
            if bool(args.per_view_knn_require_source_safe) and not safe_vs_fixed:
                continue
            objective = (
                _source_objective_from_metrics(source_summary, compute_ssim=compute_ssim, args=args)
                + float(args.per_view_knn_source_cvar_weight) * _summary_psnr_tail(source_summary, "cvar")
                + float(args.per_view_knn_source_min_weight) * _summary_psnr_tail(source_summary, "min")
                + float(args.per_view_knn_source_positive_weight) * _positive_view_fraction(source_summary)
            )
            trial["objective"] = float(objective)
            if best_trial is None or float(objective) > float(best_trial["objective"]):
                best_trial = trial
        if best_trial is None:
            return {
                "enabled": False,
                "verdict": "no source-heldout KNN threshold cleared the configured tail-risk constraints",
                "feature_names": feature_names,
                "local_tail_guard": local_tail_guard_payload,
                "threshold_mode": "source_tail_auto",
                "threshold_trials": threshold_trials,
            }
        selected_threshold = float(best_trial["threshold"])

    source_summary, selected_counts, _, local_tail_audit = evaluate_threshold(selected_threshold)
    mean_delta = float(source_summary.get("psnr_gain", 0.0)) - float(scene_selected_summary.get("psnr_gain", 0.0))
    ssim_delta = (
        float(source_summary.get("ssim_gain", 0.0)) - float(scene_selected_summary.get("ssim_gain", 0.0))
        if compute_ssim
        else 0.0
    )
    cvar_delta = _summary_psnr_tail(source_summary, "cvar") - _summary_psnr_tail(scene_selected_summary, "cvar")
    min_delta = _summary_psnr_tail(source_summary, "min") - _summary_psnr_tail(scene_selected_summary, "min")
    positive_fraction_delta = _positive_view_fraction(source_summary) - _positive_view_fraction(scene_selected_summary)
    safe_vs_fixed = (
        float(source_summary.get("psnr_gain", 0.0)) >= float(fixed_summary.get("psnr_gain", 0.0)) - float(args.selected_safe_tolerance_psnr)
        and (
            not compute_ssim
            or float(source_summary.get("ssim_gain", 0.0))
            >= float(fixed_summary.get("ssim_gain", 0.0)) - float(args.selected_safe_tolerance_ssim)
        )
    )

    def disabled_payload(verdict: str) -> dict[str, Any]:
        return {
            "enabled": False,
            "verdict": verdict,
            "feature_names": feature_names,
            "reject_variant": str(args.per_view_knn_reject_variant),
            "local_tail_guard": local_tail_guard_payload,
            "source_summary": source_summary,
            "source_fixed_summary": fixed_summary,
            "source_scene_selected_summary": scene_selected_summary,
            "source_mean_psnr_delta_vs_scene_selected": mean_delta,
            "source_mean_ssim_delta_vs_scene_selected": ssim_delta,
            "source_cvar_psnr_delta_vs_scene_selected": cvar_delta,
            "source_min_psnr_delta_vs_scene_selected": min_delta,
            "source_positive_view_fraction_delta_vs_scene_selected": positive_fraction_delta,
            "source_safe_vs_fixed": bool(safe_vs_fixed),
            "source_selected_counts": selected_counts,
            "source_local_tail_reject_count": int(local_tail_audit["reject_count"]),
            "source_local_tail_reject_views": local_tail_audit["reject_views"],
            "threshold_mode": "source_tail_auto" if bool(args.per_view_knn_auto_threshold) else "fixed",
            "selected_threshold": selected_threshold,
            "threshold_trials": threshold_trials if bool(args.per_view_knn_auto_threshold) else [],
            "source_objective_weights": {
                "lpips": float(args.source_objective_lpips_weight),
                "dists": float(args.source_objective_dists_weight),
            },
        }

    if mean_delta < float(args.per_view_knn_min_source_psnr_delta):
        return disabled_payload("source-heldout KNN policy did not improve scene-selected PSNR enough")
    if compute_ssim and ssim_delta < float(args.per_view_knn_min_source_ssim_delta):
        return disabled_payload("source-heldout KNN policy did not improve scene-selected SSIM enough")
    if cvar_delta < float(args.per_view_knn_min_source_cvar_delta):
        return disabled_payload("source-heldout KNN policy did not improve scene-selected CVaR PSNR enough")
    if min_delta < float(args.per_view_knn_min_source_min_delta):
        return disabled_payload("source-heldout KNN policy did not improve scene-selected min PSNR enough")
    if positive_fraction_delta < float(args.per_view_knn_min_source_positive_fraction_delta):
        return disabled_payload("source-heldout KNN policy did not improve scene-selected positive-view fraction enough")
    if bool(args.per_view_knn_require_source_safe) and not safe_vs_fixed:
        return disabled_payload("source-heldout KNN policy did not clear fixed safety gate")
    return {
        "enabled": True,
        "verdict": "source-heldout KNN per-view policy selected",
        "feature_schema_version": 2,
        "variants": variants,
        "base_variants_only": bool(args.per_view_knn_base_variants_only),
        "feature_names": feature_names,
        "feature_mean": mean,
        "feature_std": std,
        "k": int(args.per_view_knn_k),
        "min_predicted_score": float(selected_threshold),
        "min_score_delta_vs_scene": float(args.per_view_knn_min_score_delta_vs_scene),
        "forbid_fixed_when_scene_nonfixed": bool(args.per_view_knn_forbid_fixed_when_scene_nonfixed),
        "reject_variant": str(args.per_view_knn_reject_variant),
        "local_tail_guard": local_tail_guard_payload,
        "local_tail_guard_enabled": local_tail_guard_enabled,
        "local_tail_k": local_tail_k,
        "min_local_psnr_delta_vs_scene": float(args.per_view_knn_min_local_psnr_delta_vs_scene),
        "min_local_ssim_delta_vs_scene": float(args.per_view_knn_min_local_ssim_delta_vs_scene),
        "min_local_cvar_delta_vs_scene": float(args.per_view_knn_min_local_cvar_delta_vs_scene),
        "min_local_min_delta_vs_scene": float(args.per_view_knn_min_local_min_delta_vs_scene),
        "min_local_positive_fraction_delta_vs_scene": float(args.per_view_knn_min_local_positive_fraction_delta_vs_scene),
        "scene_selected_variant": selected_variant,
        "entries_by_variant": entries_by_variant,
        "source_summary": source_summary,
        "source_fixed_summary": fixed_summary,
        "source_scene_selected_summary": scene_selected_summary,
        "source_mean_psnr_delta_vs_scene_selected": mean_delta,
        "source_mean_ssim_delta_vs_scene_selected": ssim_delta,
        "source_cvar_psnr_delta_vs_scene_selected": cvar_delta,
        "source_min_psnr_delta_vs_scene_selected": min_delta,
        "source_positive_view_fraction_delta_vs_scene_selected": positive_fraction_delta,
        "source_safe_vs_fixed": bool(safe_vs_fixed),
        "source_selected_counts": selected_counts,
        "source_local_tail_reject_count": int(local_tail_audit["reject_count"]),
        "source_local_tail_reject_views": local_tail_audit["reject_views"],
        "threshold_mode": "source_tail_auto" if bool(args.per_view_knn_auto_threshold) else "fixed",
        "selected_threshold": selected_threshold,
        "threshold_trials": threshold_trials if bool(args.per_view_knn_auto_threshold) else [],
        "source_objective_weights": {
            "lpips": float(args.source_objective_lpips_weight),
            "dists": float(args.source_objective_dists_weight),
        },
    }


def _knn_choose_variant(
    proxies_by_variant: dict[str, dict[str, float]],
    policy: dict[str, Any],
    *,
    compute_ssim: bool,
) -> tuple[str, dict[str, float], dict[str, Any]]:
    variants = list(policy.get("variants", BASE_CANDIDATE_VARIANTS))
    feature_names = list(policy["feature_names"])
    mean = [float(x) for x in policy["feature_mean"]]
    std = [float(x) for x in policy["feature_std"]]
    predictions: dict[str, float] = {}
    local_summaries: dict[str, dict[str, Any]] = {}
    for variant in variants:
        vector = _feature_vector(proxies_by_variant[variant], feature_names)
        pool = policy["entries_by_variant"][variant]
        ranked = sorted(
            pool,
            key=lambda entry: _normalized_distance(vector, entry["vector"], mean, std),
        )
        k = max(1, min(int(policy.get("k", 3)), len(ranked)))
        predictions[variant] = _mean([float(entry["score"]) for entry in ranked[:k]])
        if bool(policy.get("local_tail_guard_enabled", False)):
            local_summaries[variant] = _knn_local_metric_summary(
                policy["entries_by_variant"],
                variant,
                vector,
                mean=mean,
                std=std,
                k=int(policy.get("local_tail_k", policy.get("k", 3))),
                compute_ssim=compute_ssim,
            )
    best_variant, best_score = max(predictions.items(), key=lambda item: item[1])
    scene_variant = str(policy.get("scene_selected_variant", "fixed"))
    scene_score = float(predictions.get(scene_variant, float("-inf")))
    diagnostics: dict[str, Any] = {
        "best_variant": best_variant,
        "best_score": float(best_score),
        "scene_variant": scene_variant,
        "scene_score": scene_score,
        "local_tail_guard": {"enabled": False, "rejected": False},
        "local_summaries": local_summaries if bool(policy.get("local_tail_guard_enabled", False)) else {},
        "reject_reason": None,
    }
    if bool(policy.get("local_tail_guard_enabled", False)):
        diagnostics["local_tail_guard"] = _knn_local_tail_guard_decision(
            best_variant,
            scene_variant,
            local_summaries,
            compute_ssim=compute_ssim,
            min_psnr_delta_vs_scene=float(policy.get("min_local_psnr_delta_vs_scene", -1.0e9)),
            min_ssim_delta_vs_scene=float(policy.get("min_local_ssim_delta_vs_scene", -1.0e9)),
            min_cvar_delta_vs_scene=float(policy.get("min_local_cvar_delta_vs_scene", -1.0e9)),
            min_min_delta_vs_scene=float(policy.get("min_local_min_delta_vs_scene", -1.0e9)),
            min_positive_fraction_delta_vs_scene=float(policy.get("min_local_positive_fraction_delta_vs_scene", -1.0e9)),
        )

    def reject() -> tuple[str, dict[str, float], dict[str, Any]]:
        if str(policy.get("reject_variant", "noop")) == "scene":
            return "__scene__", predictions, diagnostics
        return "noop", predictions, diagnostics

    if best_score < float(policy.get("min_predicted_score", 0.0)):
        diagnostics["reject_reason"] = "low_absolute_score"
        return reject()
    if best_score < scene_score + float(policy.get("min_score_delta_vs_scene", 0.0)):
        diagnostics["reject_reason"] = "low_scene_margin"
        return reject()
    if (
        bool(policy.get("forbid_fixed_when_scene_nonfixed", False))
        and scene_variant != "fixed"
        and best_variant == "fixed"
    ):
        diagnostics["reject_reason"] = "fixed_when_scene_nonfixed"
        return reject()
    if bool(diagnostics.get("local_tail_guard", {}).get("rejected", False)):
        diagnostics["reject_reason"] = "local_tail_guard"
        return reject()
    return best_variant, predictions, diagnostics


def _fit_local_support_policy(
    selector_payload: dict[str, Any] | None,
    *,
    compute_ssim: bool,
    args: argparse.Namespace,
    incumbent_policy_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not bool(args.enable_local_support_policy):
        return {"enabled": False, "verdict": "disabled by CLI"}
    if selector_payload is None or not selector_payload.get("per_view"):
        return {"enabled": False, "verdict": "missing source-heldout selector per-view evidence"}
    selected_variant = str(selector_payload["selected_variant"])
    feature_names = _parse_feature_names(args.local_support_feature_grid)
    variants = list(BASE_CANDIDATE_VARIANTS) if bool(args.local_support_base_variants_only) else _candidate_variant_names(args)
    if selected_variant not in variants:
        variants = [selected_variant] + [variant for variant in variants if variant != selected_variant]
    entries_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    all_vectors: list[list[float]] = []
    for row in selector_payload["per_view"]:
        for variant in variants:
            candidate = row["candidates"][variant]
            vector = _feature_vector(candidate["proxy"], feature_names)
            entry = {
                "view": row["view"],
                "variant": variant,
                "vector": vector,
                "metrics": candidate["metrics"],
                "score": _source_objective_from_metrics(candidate["metrics"], compute_ssim=compute_ssim, args=args),
            }
            entries_by_variant[variant].append(entry)
            all_vectors.append(vector)
    mean, std = _feature_stats(all_vectors)
    if not mean:
        return {"enabled": False, "verdict": "no usable local-support proxy vectors"}
    fixed_summary = selector_payload["summaries"]["fixed"]
    scene_summary = selector_payload["summaries"][selected_variant]
    support_k = int(args.local_support_k)
    post_incumbent_fallback_only = bool(args.local_support_post_incumbent_fallback_only)
    incumbent_by_view: dict[str, str] = {}
    if post_incumbent_fallback_only and incumbent_policy_payload and incumbent_policy_payload.get("loo_predictions"):
        for item in incumbent_policy_payload.get("loo_predictions", []):
            chosen = str(item.get("chosen_variant", "__scene__"))
            if chosen == "__scene__":
                incumbent_by_view[str(item.get("view"))] = selected_variant
            else:
                incumbent_by_view[str(item.get("view"))] = chosen

    def local_summaries_for_row(row: dict[str, Any], *, exclude_view: str | None) -> dict[str, dict[str, Any]]:
        summaries: dict[str, dict[str, Any]] = {}
        for variant in variants:
            vector = _feature_vector(row["candidates"][variant]["proxy"], feature_names)
            summaries[variant] = _knn_local_metric_summary(
                entries_by_variant,
                variant,
                vector,
                mean=mean,
                std=std,
                k=support_k,
                compute_ssim=compute_ssim,
                exclude_view=exclude_view,
            )
        return summaries

    def score_candidate(guard: dict[str, Any]) -> float:
        deltas = dict(guard.get("deltas_vs_scene", {}))
        return float(
            float(deltas.get("psnr_gain", 0.0))
            + float(args.local_support_ssim_weight) * float(deltas.get("ssim_gain", 0.0))
            + float(args.local_support_cvar_weight) * float(deltas.get("cvar_psnr_gain", 0.0))
            + float(args.local_support_min_weight) * float(deltas.get("min_psnr_gain", 0.0))
            + float(args.local_support_positive_weight) * float(deltas.get("positive_view_fraction", 0.0))
        )

    def choose_from_local_summaries(local_summaries: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        rejected: list[dict[str, Any]] = []
        for variant in variants:
            if variant == selected_variant:
                continue
            if (
                bool(args.local_support_forbid_fixed_when_scene_nonfixed)
                and selected_variant != "fixed"
                and variant == "fixed"
            ):
                rejected.append({"variant": variant, "reason": "fixed_when_scene_nonfixed"})
                continue
            guard = _knn_local_tail_guard_decision(
                variant,
                selected_variant,
                local_summaries,
                compute_ssim=compute_ssim,
                min_psnr_delta_vs_scene=float(args.local_support_min_local_psnr_delta_vs_scene),
                min_ssim_delta_vs_scene=float(args.local_support_min_local_ssim_delta_vs_scene),
                min_cvar_delta_vs_scene=float(args.local_support_min_local_cvar_delta_vs_scene),
                min_min_delta_vs_scene=float(args.local_support_min_local_min_delta_vs_scene),
                min_positive_fraction_delta_vs_scene=float(args.local_support_min_local_positive_fraction_delta_vs_scene),
            )
            score = score_candidate(guard)
            guard["score"] = float(score)
            if bool(guard.get("rejected", False)):
                rejected.append({"variant": variant, "reason": guard.get("reason"), "guard": guard})
                continue
            if score < float(args.local_support_min_score_delta_vs_scene):
                rejected.append({"variant": variant, "reason": "score_delta_vs_scene", "guard": guard})
                continue
            candidates.append((score, variant, guard))
        if not candidates:
            return "__scene__", {
                "scene_variant": selected_variant,
                "reject_reason": "no_candidate_with_local_support",
                "rejected_candidates": rejected,
                "local_summaries": local_summaries,
            }
        candidates.sort(key=lambda item: item[0], reverse=True)
        score, variant, guard = candidates[0]
        return variant, {
            "scene_variant": selected_variant,
            "best_variant": variant,
            "best_score": float(score),
            "best_guard": guard,
            "candidate_scores": [
                {"variant": name, "score": float(value), "guard": cand_guard}
                for value, name, cand_guard in candidates
            ],
            "rejected_candidates": rejected,
            "local_summaries": local_summaries,
            "reject_reason": None,
        }

    source_policy_rows: list[dict[str, float]] = []
    source_incumbent_rows: list[dict[str, float]] = []
    selected_counts: dict[str, int] = _candidate_count_dict(variants)
    selected_counts["incumbent_skip"] = 0
    source_decisions: list[dict[str, Any]] = []
    for row in selector_payload["per_view"]:
        incumbent_variant = str(incumbent_by_view.get(str(row["view"]), selected_variant))
        if incumbent_variant == "noop":
            incumbent_metrics = _noop_like(row["candidates"][selected_variant]["metrics"])
        elif incumbent_variant in row["candidates"]:
            incumbent_metrics = row["candidates"][incumbent_variant]["metrics"]
        else:
            incumbent_variant = selected_variant
            incumbent_metrics = row["candidates"][selected_variant]["metrics"]
        source_incumbent_rows.append(incumbent_metrics)
        if post_incumbent_fallback_only and incumbent_variant != selected_variant:
            selected_counts["incumbent_skip"] += 1
            source_policy_rows.append(incumbent_metrics)
            source_decisions.append(
                {
                    "view": str(row["view"]),
                    "chosen_variant": "__incumbent__",
                    "incumbent_variant": incumbent_variant,
                    "diagnostics": {"skip_reason": "incumbent_already_refined"},
                }
            )
            continue
        local_summaries = local_summaries_for_row(row, exclude_view=str(row["view"]))
        chosen_variant, diagnostics = choose_from_local_summaries(local_summaries)
        source_decisions.append(
            {
                "view": str(row["view"]),
                "chosen_variant": chosen_variant,
                "incumbent_variant": incumbent_variant,
                "diagnostics": diagnostics,
            }
        )
        if chosen_variant == "__scene__":
            selected_counts["scene"] += 1
            source_policy_rows.append(incumbent_metrics)
        elif chosen_variant == "noop":
            selected_counts["noop"] += 1
            source_policy_rows.append(_noop_like(incumbent_metrics))
        else:
            selected_counts[chosen_variant] += 1
            source_policy_rows.append(row["candidates"][chosen_variant]["metrics"])
    source_summary = _summarize_metric_rows(source_policy_rows, compute_ssim=compute_ssim)
    incumbent_summary = _summarize_metric_rows(source_incumbent_rows, compute_ssim=compute_ssim)
    comparison_summary = incumbent_summary if post_incumbent_fallback_only else scene_summary
    reject_count = int(selected_counts["noop"] + selected_counts["scene"] + selected_counts.get("incumbent_skip", 0))
    accept_fraction = 1.0 - float(reject_count / max(len(source_policy_rows), 1))
    mean_delta = float(source_summary.get("psnr_gain", 0.0)) - float(comparison_summary.get("psnr_gain", 0.0))
    ssim_delta = (
        float(source_summary.get("ssim_gain", 0.0)) - float(comparison_summary.get("ssim_gain", 0.0))
        if compute_ssim
        else 0.0
    )
    cvar_delta = _summary_psnr_tail(source_summary, "cvar") - _summary_psnr_tail(comparison_summary, "cvar")
    min_delta = _summary_psnr_tail(source_summary, "min") - _summary_psnr_tail(comparison_summary, "min")
    positive_fraction_delta = _positive_view_fraction(source_summary) - _positive_view_fraction(comparison_summary)
    safe_vs_fixed = (
        float(source_summary.get("psnr_gain", 0.0)) >= float(fixed_summary.get("psnr_gain", 0.0)) - float(args.selected_safe_tolerance_psnr)
        and (
            not compute_ssim
            or float(source_summary.get("ssim_gain", 0.0))
            >= float(fixed_summary.get("ssim_gain", 0.0)) - float(args.selected_safe_tolerance_ssim)
        )
    )
    base_payload = {
        "feature_schema_version": 1,
        "variants": variants,
        "base_variants_only": bool(args.local_support_base_variants_only),
        "feature_names": feature_names,
        "feature_mean": mean,
        "feature_std": std,
        "k": support_k,
        "scene_selected_variant": selected_variant,
        "entries_by_variant": entries_by_variant,
        "reject_variant": str(args.local_support_reject_variant),
        "source_summary": source_summary,
        "source_fixed_summary": fixed_summary,
        "source_scene_selected_summary": scene_summary,
        "source_incumbent_summary": incumbent_summary,
        "source_comparison_summary_kind": "incumbent_policy" if post_incumbent_fallback_only else "scene_selected",
        "source_mean_psnr_delta_vs_scene_selected": mean_delta,
        "source_mean_ssim_delta_vs_scene_selected": ssim_delta,
        "source_cvar_psnr_delta_vs_scene_selected": cvar_delta,
        "source_min_psnr_delta_vs_scene_selected": min_delta,
        "source_positive_view_fraction_delta_vs_scene_selected": positive_fraction_delta,
        "source_safe_vs_fixed": bool(safe_vs_fixed),
        "source_accept_fraction": accept_fraction,
        "source_selected_counts": selected_counts,
        "source_decisions": source_decisions,
        "min_local_psnr_delta_vs_scene": float(args.local_support_min_local_psnr_delta_vs_scene),
        "min_local_ssim_delta_vs_scene": float(args.local_support_min_local_ssim_delta_vs_scene),
        "min_local_cvar_delta_vs_scene": float(args.local_support_min_local_cvar_delta_vs_scene),
        "min_local_min_delta_vs_scene": float(args.local_support_min_local_min_delta_vs_scene),
        "min_local_positive_fraction_delta_vs_scene": float(args.local_support_min_local_positive_fraction_delta_vs_scene),
        "min_score_delta_vs_scene": float(args.local_support_min_score_delta_vs_scene),
        "forbid_fixed_when_scene_nonfixed": bool(args.local_support_forbid_fixed_when_scene_nonfixed),
        "post_incumbent_fallback_only": post_incumbent_fallback_only,
        "ssim_weight": float(args.local_support_ssim_weight),
        "cvar_weight": float(args.local_support_cvar_weight),
        "min_weight": float(args.local_support_min_weight),
        "positive_weight": float(args.local_support_positive_weight),
    }
    if accept_fraction < float(args.local_support_min_accept_fraction):
        return {"enabled": False, "verdict": "local support accepted too few source views", **base_payload}
    if accept_fraction > float(args.local_support_max_accept_fraction):
        return {"enabled": False, "verdict": "local support accepted too many source views", **base_payload}
    if mean_delta < float(args.local_support_min_source_psnr_delta):
        return {"enabled": False, "verdict": "local support did not improve source PSNR enough", **base_payload}
    if compute_ssim and ssim_delta < float(args.local_support_min_source_ssim_delta):
        return {"enabled": False, "verdict": "local support did not improve source SSIM enough", **base_payload}
    if cvar_delta < float(args.local_support_min_source_cvar_delta):
        return {"enabled": False, "verdict": "local support did not clear source CVaR delta", **base_payload}
    if min_delta < float(args.local_support_min_source_min_delta):
        return {"enabled": False, "verdict": "local support did not clear source min delta", **base_payload}
    if positive_fraction_delta < float(args.local_support_min_source_positive_fraction_delta):
        return {"enabled": False, "verdict": "local support did not clear source positive-view delta", **base_payload}
    if bool(args.local_support_require_source_safe) and not safe_vs_fixed:
        return {"enabled": False, "verdict": "local support did not clear fixed safety gate", **base_payload}
    return {"enabled": True, "verdict": "source-heldout local support certificate selected", **base_payload}


def _local_support_choose_variant(
    proxies_by_variant: dict[str, dict[str, float]],
    policy: dict[str, Any],
    *,
    compute_ssim: bool,
) -> tuple[str, dict[str, Any]]:
    variants = list(policy.get("variants", BASE_CANDIDATE_VARIANTS))
    feature_names = list(policy["feature_names"])
    mean = [float(value) for value in policy.get("feature_mean", [])]
    std = [float(value) for value in policy.get("feature_std", [])]
    scene_variant = str(policy["scene_selected_variant"])
    entries_by_variant = policy.get("entries_by_variant", {})
    support_k = int(policy.get("k", 3))
    local_summaries: dict[str, dict[str, Any]] = {}
    for variant in variants:
        vector = _feature_vector(proxies_by_variant[variant], feature_names)
        local_summaries[variant] = _knn_local_metric_summary(
            entries_by_variant,
            variant,
            vector,
            mean=mean,
            std=std,
            k=support_k,
            compute_ssim=compute_ssim,
        )

    def score_candidate(guard: dict[str, Any]) -> float:
        deltas = dict(guard.get("deltas_vs_scene", {}))
        return float(
            float(deltas.get("psnr_gain", 0.0))
            + float(policy.get("ssim_weight", 0.0)) * float(deltas.get("ssim_gain", 0.0))
            + float(policy.get("cvar_weight", 0.0)) * float(deltas.get("cvar_psnr_gain", 0.0))
            + float(policy.get("min_weight", 0.0)) * float(deltas.get("min_psnr_gain", 0.0))
            + float(policy.get("positive_weight", 0.0)) * float(deltas.get("positive_view_fraction", 0.0))
        )

    candidates: list[tuple[float, str, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for variant in variants:
        if variant == scene_variant:
            continue
        if bool(policy.get("forbid_fixed_when_scene_nonfixed", False)) and scene_variant != "fixed" and variant == "fixed":
            rejected.append({"variant": variant, "reason": "fixed_when_scene_nonfixed"})
            continue
        guard = _knn_local_tail_guard_decision(
            variant,
            scene_variant,
            local_summaries,
            compute_ssim=compute_ssim,
            min_psnr_delta_vs_scene=float(policy.get("min_local_psnr_delta_vs_scene", 0.0)),
            min_ssim_delta_vs_scene=float(policy.get("min_local_ssim_delta_vs_scene", -1.0e9)),
            min_cvar_delta_vs_scene=float(policy.get("min_local_cvar_delta_vs_scene", -1.0e9)),
            min_min_delta_vs_scene=float(policy.get("min_local_min_delta_vs_scene", -1.0e9)),
            min_positive_fraction_delta_vs_scene=float(policy.get("min_local_positive_fraction_delta_vs_scene", -1.0e9)),
        )
        score = score_candidate(guard)
        guard["score"] = float(score)
        if bool(guard.get("rejected", False)):
            rejected.append({"variant": variant, "reason": guard.get("reason"), "guard": guard})
            continue
        if score < float(policy.get("min_score_delta_vs_scene", 0.0)):
            rejected.append({"variant": variant, "reason": "score_delta_vs_scene", "guard": guard})
            continue
        candidates.append((score, variant, guard))
    if not candidates:
        reject = "__scene__" if str(policy.get("reject_variant", "scene")) == "scene" else "noop"
        return reject, {
            "scene_variant": scene_variant,
            "reject_reason": "no_candidate_with_local_support",
            "rejected_candidates": rejected,
            "local_summaries": local_summaries,
        }
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, variant, guard = candidates[0]
    return variant, {
        "scene_variant": scene_variant,
        "best_variant": variant,
        "best_score": float(score),
        "best_guard": guard,
        "candidate_scores": [
            {"variant": name, "score": float(value), "guard": cand_guard}
            for value, name, cand_guard in candidates
        ],
        "rejected_candidates": rejected,
        "local_summaries": local_summaries,
        "reject_reason": None,
    }


PAIRWISE_DOMINANCE_TARGET_NAMES = [
    "objective_delta_vs_incumbent",
    "psnr_delta_vs_incumbent",
    "ssim_delta_vs_incumbent",
    "lpips_delta_vs_incumbent",
    "dists_delta_vs_incumbent",
]


def _pairwise_target(
    candidate_metrics: dict[str, float],
    incumbent_metrics: dict[str, float],
    *,
    compute_ssim: bool,
    args: argparse.Namespace,
) -> list[float]:
    incumbent_objective = _source_objective_from_metrics(incumbent_metrics, compute_ssim=compute_ssim, args=args)
    return [
        _source_objective_from_metrics(candidate_metrics, compute_ssim=compute_ssim, args=args) - incumbent_objective,
        _metric_delta(candidate_metrics, incumbent_metrics, "psnr_gain"),
        _metric_delta(candidate_metrics, incumbent_metrics, "ssim_gain") if compute_ssim else 0.0,
        _metric_delta(candidate_metrics, incumbent_metrics, "lpips_gain"),
        _metric_delta(candidate_metrics, incumbent_metrics, "dists_gain"),
    ]


def _pairwise_local_delta_summary(
    entries: list[dict[str, Any]],
    vector: list[float],
    *,
    mean: list[float],
    std: list[float],
    k: int,
    exclude_view: str | None = None,
) -> dict[str, Any]:
    pool = [entry for entry in entries if exclude_view is None or str(entry.get("view")) != str(exclude_view)]
    if not pool:
        return {"available": False, "k": 0, "neighbors": [], "deltas": {}}
    ranked = sorted(
        (
            (
                _normalized_distance(vector, list(entry["features"]), mean, std),
                entry,
            )
            for entry in pool
        ),
        key=lambda item: item[0],
    )[: max(1, min(int(k), len(pool)))]
    psnr = [float(entry["target"][1]) for _, entry in ranked]
    ssim = [float(entry["target"][2]) for _, entry in ranked]
    objective = [float(entry["target"][0]) for _, entry in ranked]
    return {
        "available": True,
        "k": int(len(ranked)),
        "neighbors": [
            {
                "view": str(entry.get("view")),
                "candidate_variant": str(entry.get("candidate_variant")),
                "incumbent_variant": str(entry.get("incumbent_variant")),
                "distance": float(distance),
                "target": [float(value) for value in entry.get("target", [])],
            }
            for distance, entry in ranked
        ],
        "deltas": {
            "objective": _mean(objective),
            "psnr": _mean(psnr),
            "ssim": _mean(ssim),
            "psnr_min": min(psnr) if psnr else 0.0,
            "psnr_cvar": _tail_values(psnr).get("cvar", 0.0),
            "positive_fraction": _mean([1.0 if value >= 0.0 else 0.0 for value in psnr]),
        },
    }


def _pairwise_blend_step_reject_reason(
    *,
    blend_step: float | None,
    local_deltas: dict[str, Any],
    args: argparse.Namespace,
) -> str | None:
    if blend_step is None or blend_step <= float(args.pairwise_dominance_max_blend_step):
        return None
    if not bool(args.pairwise_dominance_enable_adaptive_blend_step):
        return "blend_step"
    if blend_step > float(args.pairwise_dominance_adaptive_max_blend_step):
        return "blend_step"
    if float(local_deltas.get("psnr", 0.0)) < float(args.pairwise_dominance_large_step_min_local_psnr_delta):
        return "adaptive_blend_local_psnr"
    if float(local_deltas.get("ssim", 0.0)) < float(args.pairwise_dominance_large_step_min_local_ssim_delta):
        return "adaptive_blend_local_ssim"
    if float(local_deltas.get("psnr_cvar", 0.0)) < float(args.pairwise_dominance_large_step_min_local_cvar_delta):
        return "adaptive_blend_local_cvar"
    if float(local_deltas.get("psnr_min", 0.0)) < float(args.pairwise_dominance_large_step_min_local_min_delta):
        return "adaptive_blend_local_min"
    if float(local_deltas.get("positive_fraction", 0.0)) < float(args.pairwise_dominance_large_step_min_positive_fraction):
        return "adaptive_blend_positive_fraction"
    return None


def _source_incumbent_variant_for_view(
    view: str,
    *,
    selected_variant: str,
    source_reliability_payload: dict[str, Any] | None,
) -> str:
    if source_reliability_payload and bool(source_reliability_payload.get("enabled", False)):
        for item in source_reliability_payload.get("loo_predictions", []):
            if str(item.get("view")) != str(view):
                continue
            chosen = str(item.get("chosen_variant", "__scene__"))
            return selected_variant if chosen in {"__scene__", "noop"} else chosen
    return selected_variant


def _fit_pairwise_dominance_policy(
    selector_payload: dict[str, Any] | None,
    *,
    source_reliability_payload: dict[str, Any] | None,
    compute_ssim: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not bool(args.enable_pairwise_dominance_policy):
        return {"enabled": False, "verdict": "disabled by CLI"}
    if selector_payload is None or not selector_payload.get("per_view"):
        return {"enabled": False, "verdict": "missing source-heldout selector per-view evidence"}
    selected_variant = str(selector_payload["selected_variant"])
    feature_names = _parse_feature_names(args.pairwise_dominance_feature_grid)
    variants = _candidate_variant_names(args)
    variant_blend_map = _candidate_variant_blend_map(args)
    examples: list[dict[str, Any]] = []
    rows_by_view: dict[str, dict[str, Any]] = {}
    source_incumbent_rows: list[dict[str, float]] = []
    source_incumbents: dict[str, str] = {}
    entries_by_candidate: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    all_vectors: list[list[float]] = []
    for row in selector_payload["per_view"]:
        view = str(row["view"])
        rows_by_view[view] = row
        incumbent_variant = _source_incumbent_variant_for_view(
            view,
            selected_variant=selected_variant,
            source_reliability_payload=source_reliability_payload,
        )
        if incumbent_variant not in row["candidates"]:
            incumbent_variant = selected_variant
        source_incumbents[view] = incumbent_variant
        incumbent = row["candidates"][incumbent_variant]
        incumbent_metrics = incumbent["metrics"]
        incumbent_proxy = incumbent["proxy"]
        source_incumbent_rows.append(incumbent_metrics)
        for variant in variants:
            if variant == incumbent_variant:
                continue
            candidate = row["candidates"][variant]
            features = _source_reliability_features(
                candidate["proxy"],
                incumbent_proxy,
                variant=variant,
                scene_variant=incumbent_variant,
                feature_names=feature_names,
                variant_blend=variant_blend_map.get(variant),
                scene_variant_blend=variant_blend_map.get(incumbent_variant),
            )
            target = _pairwise_target(candidate["metrics"], incumbent_metrics, compute_ssim=compute_ssim, args=args)
            example = {
                "view": view,
                "candidate_variant": variant,
                "incumbent_variant": incumbent_variant,
                "features": features,
                "target": target,
                "candidate_metrics": candidate["metrics"],
                "incumbent_metrics": incumbent_metrics,
            }
            examples.append(example)
            entries_by_candidate[variant].append(example)
            all_vectors.append(features)
    if len(rows_by_view) < 2 or len(examples) < 2:
        return {"enabled": False, "verdict": "not enough source-heldout pairwise examples"}
    feature_mean, feature_std = _feature_stats(all_vectors)
    ood_source_distances: list[float] = []
    for example in examples:
        pool = [
            entry
            for entry in entries_by_candidate.get(str(example["candidate_variant"]), [])
            if str(entry["view"]) != str(example["view"])
        ]
        if not pool:
            continue
        ood_source_distances.append(
            min(
                _normalized_distance(list(example["features"]), list(entry["features"]), feature_mean, feature_std)
                for entry in pool
            )
        )
    ood_threshold = (
        _quantile(ood_source_distances, float(args.pairwise_dominance_ood_quantile))
        if bool(args.pairwise_dominance_enable_ood_guard)
        else float("inf")
    )

    def choose_from_model(
        row: dict[str, Any],
        incumbent_variant: str,
        model: dict[str, Any],
        *,
        exclude_view: str | None = None,
    ) -> tuple[str, dict[str, list[float]], dict[str, Any]]:
        incumbent = row["candidates"][incumbent_variant]
        predictions: dict[str, list[float]] = {}
        diagnostics: dict[str, Any] = {
            "incumbent_variant": incumbent_variant,
            "candidate_diagnostics": {},
            "reject_reason": None,
        }
        accepted: list[tuple[float, str, dict[str, Any]]] = []
        for variant in variants:
            if variant == incumbent_variant:
                continue
            candidate = row["candidates"][variant]
            features = _source_reliability_features(
                candidate["proxy"],
                incumbent["proxy"],
                variant=variant,
                scene_variant=incumbent_variant,
                feature_names=feature_names,
                variant_blend=variant_blend_map.get(variant),
                scene_variant_blend=variant_blend_map.get(incumbent_variant),
            )
            prediction = _predict_ridge(model, features)
            predictions[variant] = prediction
            local = _pairwise_local_delta_summary(
                entries_by_candidate.get(variant, []),
                features,
                mean=feature_mean,
                std=feature_std,
                k=int(args.pairwise_dominance_k),
                exclude_view=exclude_view,
            )
            pool = [
                entry
                for entry in entries_by_candidate.get(variant, [])
                if exclude_view is None or str(entry["view"]) != str(exclude_view)
            ]
            ood_distance = (
                min(_normalized_distance(features, list(entry["features"]), feature_mean, feature_std) for entry in pool)
                if pool
                else float("inf")
            )
            cand_diag = {
                "prediction": prediction,
                "local": local,
                "ood_distance": float(ood_distance),
                "ood_threshold": float(ood_threshold),
                "reject_reason": None,
            }
            local_deltas = dict(local.get("deltas", {}))
            candidate_blend = variant_blend_map.get(variant)
            incumbent_blend = variant_blend_map.get(incumbent_variant)
            blend_step = (
                abs(float(candidate_blend) - float(incumbent_blend))
                if candidate_blend is not None and incumbent_blend is not None
                else None
            )
            cand_diag["blend_step"] = blend_step
            cand_diag["max_blend_step"] = float(args.pairwise_dominance_max_blend_step)
            cand_diag["adaptive_blend_step_enabled"] = bool(args.pairwise_dominance_enable_adaptive_blend_step)
            cand_diag["adaptive_max_blend_step"] = float(args.pairwise_dominance_adaptive_max_blend_step)
            reject_reason = None
            blend_reject_reason = _pairwise_blend_step_reject_reason(
                blend_step=blend_step,
                local_deltas=local_deltas,
                args=args,
            )
            if blend_reject_reason is not None:
                reject_reason = blend_reject_reason
            elif prediction[0] < float(args.pairwise_dominance_min_predicted_objective_delta):
                reject_reason = "predicted_objective_delta"
            elif prediction[1] < float(args.pairwise_dominance_min_predicted_psnr_delta):
                reject_reason = "predicted_psnr_delta"
            elif compute_ssim and prediction[2] < float(args.pairwise_dominance_min_predicted_ssim_delta):
                reject_reason = "predicted_ssim_delta"
            elif not bool(local.get("available", False)):
                reject_reason = "missing_local_support"
            elif float(local_deltas.get("psnr", 0.0)) < float(args.pairwise_dominance_min_local_psnr_delta):
                reject_reason = "local_psnr_delta"
            elif compute_ssim and float(local_deltas.get("ssim", 0.0)) < float(args.pairwise_dominance_min_local_ssim_delta):
                reject_reason = "local_ssim_delta"
            elif float(local_deltas.get("psnr_cvar", 0.0)) < float(args.pairwise_dominance_min_local_cvar_delta):
                reject_reason = "local_cvar_delta"
            elif float(local_deltas.get("psnr_min", 0.0)) < float(args.pairwise_dominance_min_local_min_delta):
                reject_reason = "local_min_delta"
            elif bool(args.pairwise_dominance_enable_ood_guard) and ood_distance > ood_threshold:
                reject_reason = "ood_distance"
            cand_diag["reject_reason"] = reject_reason
            diagnostics["candidate_diagnostics"][variant] = cand_diag
            if reject_reason is None:
                score = (
                    float(prediction[0])
                    + float(args.pairwise_dominance_psnr_weight) * float(prediction[1])
                    + float(args.pairwise_dominance_ssim_weight) * float(prediction[2])
                    + float(args.pairwise_dominance_local_cvar_weight) * float(local_deltas.get("psnr_cvar", 0.0))
                )
                accepted.append((score, variant, cand_diag))
        if not accepted:
            diagnostics["reject_reason"] = "no_pairwise_candidate"
            return "__incumbent__", predictions, diagnostics
        accepted.sort(key=lambda item: item[0], reverse=True)
        score, variant, cand_diag = accepted[0]
        diagnostics["best_variant"] = variant
        diagnostics["best_score"] = float(score)
        diagnostics["best_diagnostics"] = cand_diag
        return variant, predictions, diagnostics

    source_policy_rows: list[dict[str, float]] = []
    selected_counts: dict[str, int] = _candidate_count_dict(variants)
    selected_counts["incumbent"] = 0
    loo_decisions: list[dict[str, Any]] = []
    for view, row in rows_by_view.items():
        train_examples = [example for example in examples if str(example["view"]) != str(view)]
        model = _fit_ridge_predictor(train_examples, ridge=float(args.pairwise_dominance_ridge))
        incumbent_variant = source_incumbents[view]
        incumbent_metrics = row["candidates"][incumbent_variant]["metrics"]
        if model is None:
            selected_counts["incumbent"] += 1
            source_policy_rows.append(incumbent_metrics)
            loo_decisions.append({"view": view, "chosen_variant": "__incumbent__", "incumbent_variant": incumbent_variant})
            continue
        chosen_variant, predictions, diagnostics = choose_from_model(
            row,
            incumbent_variant,
            model,
            exclude_view=view,
        )
        loo_decisions.append(
            {
                "view": view,
                "chosen_variant": chosen_variant,
                "incumbent_variant": incumbent_variant,
                "predictions": predictions,
                "diagnostics": diagnostics,
            }
        )
        if chosen_variant == "__incumbent__":
            selected_counts["incumbent"] += 1
            source_policy_rows.append(incumbent_metrics)
        else:
            selected_counts[chosen_variant] += 1
            source_policy_rows.append(row["candidates"][chosen_variant]["metrics"])

    source_summary = _summarize_metric_rows(source_policy_rows, compute_ssim=compute_ssim)
    incumbent_summary = _summarize_metric_rows(source_incumbent_rows, compute_ssim=compute_ssim)
    mean_delta = float(source_summary.get("psnr_gain", 0.0)) - float(incumbent_summary.get("psnr_gain", 0.0))
    ssim_delta = (
        float(source_summary.get("ssim_gain", 0.0)) - float(incumbent_summary.get("ssim_gain", 0.0))
        if compute_ssim
        else 0.0
    )
    cvar_delta = _summary_psnr_tail(source_summary, "cvar") - _summary_psnr_tail(incumbent_summary, "cvar")
    min_delta = _summary_psnr_tail(source_summary, "min") - _summary_psnr_tail(incumbent_summary, "min")
    accept_fraction = 1.0 - float(selected_counts["incumbent"] / max(len(rows_by_view), 1))
    base_payload = {
        "feature_schema_version": 1,
        "target_names": PAIRWISE_DOMINANCE_TARGET_NAMES,
        "feature_names": feature_names,
        "variants": variants,
        "variant_blend_map": variant_blend_map,
        "ridge": float(args.pairwise_dominance_ridge),
        "k": int(args.pairwise_dominance_k),
        "source_incumbents": source_incumbents,
        "source_summary": source_summary,
        "source_incumbent_summary": incumbent_summary,
        "source_selected_counts": selected_counts,
        "source_accept_fraction": accept_fraction,
        "source_mean_psnr_delta_vs_incumbent": mean_delta,
        "source_mean_ssim_delta_vs_incumbent": ssim_delta,
        "source_cvar_psnr_delta_vs_incumbent": cvar_delta,
        "source_min_psnr_delta_vs_incumbent": min_delta,
        "loo_decisions": loo_decisions,
        "entries_by_candidate": entries_by_candidate,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "ood_guard_enabled": bool(args.pairwise_dominance_enable_ood_guard),
        "ood_quantile": float(args.pairwise_dominance_ood_quantile),
        "ood_source_distance_threshold": float(ood_threshold),
        "max_blend_step": float(args.pairwise_dominance_max_blend_step),
        "adaptive_blend_step_enabled": bool(args.pairwise_dominance_enable_adaptive_blend_step),
        "adaptive_max_blend_step": float(args.pairwise_dominance_adaptive_max_blend_step),
        "large_step_min_local_psnr_delta": float(args.pairwise_dominance_large_step_min_local_psnr_delta),
        "large_step_min_local_ssim_delta": float(args.pairwise_dominance_large_step_min_local_ssim_delta),
        "large_step_min_local_cvar_delta": float(args.pairwise_dominance_large_step_min_local_cvar_delta),
        "large_step_min_local_min_delta": float(args.pairwise_dominance_large_step_min_local_min_delta),
        "large_step_min_positive_fraction": float(args.pairwise_dominance_large_step_min_positive_fraction),
    }
    if selected_counts["incumbent"] >= len(rows_by_view):
        return {"enabled": False, "verdict": "pairwise dominance accepted no source views", **base_payload}
    if accept_fraction < float(args.pairwise_dominance_min_accept_fraction):
        return {"enabled": False, "verdict": "pairwise dominance accepted too few source views", **base_payload}
    if accept_fraction > float(args.pairwise_dominance_max_accept_fraction):
        return {"enabled": False, "verdict": "pairwise dominance accepted too many source views", **base_payload}
    if mean_delta < float(args.pairwise_dominance_min_source_psnr_delta):
        return {"enabled": False, "verdict": "pairwise dominance did not clear source PSNR delta", **base_payload}
    if compute_ssim and ssim_delta < float(args.pairwise_dominance_min_source_ssim_delta):
        return {"enabled": False, "verdict": "pairwise dominance did not clear source SSIM delta", **base_payload}
    if cvar_delta < float(args.pairwise_dominance_min_source_cvar_delta):
        return {"enabled": False, "verdict": "pairwise dominance did not clear source CVaR delta", **base_payload}
    if min_delta < float(args.pairwise_dominance_min_source_min_delta):
        return {"enabled": False, "verdict": "pairwise dominance did not clear source min delta", **base_payload}
    full_model = _fit_ridge_predictor(examples, ridge=float(args.pairwise_dominance_ridge))
    if full_model is None:
        return {"enabled": False, "verdict": "could not fit full pairwise dominance model", **base_payload}
    return {"enabled": True, "verdict": "pairwise dominance certificate selected", "model": full_model, **base_payload}


def _pairwise_dominance_choose_variant(
    proxies_by_variant: dict[str, dict[str, float]],
    incumbent_variant: str,
    policy: dict[str, Any],
    *,
    compute_ssim: bool,
    args: argparse.Namespace,
) -> tuple[str, dict[str, list[float]], dict[str, Any]]:
    if incumbent_variant not in proxies_by_variant:
        return "__incumbent__", {}, {"reject_reason": "missing_incumbent_proxy", "incumbent_variant": incumbent_variant}
    feature_names = list(policy["feature_names"])
    variants = list(policy["variants"])
    variant_blend_map = dict(policy.get("variant_blend_map", {}))
    feature_mean = [float(value) for value in policy.get("feature_mean", [])]
    feature_std = [float(value) for value in policy.get("feature_std", [])]
    entries_by_candidate = policy.get("entries_by_candidate", {})
    incumbent_proxy = proxies_by_variant[incumbent_variant]
    predictions: dict[str, list[float]] = {}
    diagnostics: dict[str, Any] = {"incumbent_variant": incumbent_variant, "candidate_diagnostics": {}, "reject_reason": None}
    accepted: list[tuple[float, str, dict[str, Any]]] = []
    for variant in variants:
        if variant == incumbent_variant or variant not in proxies_by_variant:
            continue
        features = _source_reliability_features(
            proxies_by_variant[variant],
            incumbent_proxy,
            variant=variant,
            scene_variant=incumbent_variant,
            feature_names=feature_names,
            variant_blend=variant_blend_map.get(variant),
            scene_variant_blend=variant_blend_map.get(incumbent_variant),
        )
        prediction = _predict_ridge(policy["model"], features)
        predictions[variant] = prediction
        local = _pairwise_local_delta_summary(
            list(entries_by_candidate.get(variant, [])),
            features,
            mean=feature_mean,
            std=feature_std,
            k=int(policy.get("k", 3)),
        )
        pool = list(entries_by_candidate.get(variant, []))
        ood_distance = (
            min(_normalized_distance(features, list(entry["features"]), feature_mean, feature_std) for entry in pool)
            if pool
            else float("inf")
        )
        local_deltas = dict(local.get("deltas", {}))
        candidate_blend = variant_blend_map.get(variant)
        incumbent_blend = variant_blend_map.get(incumbent_variant)
        blend_step = (
            abs(float(candidate_blend) - float(incumbent_blend))
            if candidate_blend is not None and incumbent_blend is not None
            else None
        )
        reject_reason = None
        blend_reject_reason = _pairwise_blend_step_reject_reason(
            blend_step=blend_step,
            local_deltas=local_deltas,
            args=args,
        )
        if blend_reject_reason is not None:
            reject_reason = blend_reject_reason
        elif prediction[0] < float(args.pairwise_dominance_min_predicted_objective_delta):
            reject_reason = "predicted_objective_delta"
        elif prediction[1] < float(args.pairwise_dominance_min_predicted_psnr_delta):
            reject_reason = "predicted_psnr_delta"
        elif compute_ssim and prediction[2] < float(args.pairwise_dominance_min_predicted_ssim_delta):
            reject_reason = "predicted_ssim_delta"
        elif not bool(local.get("available", False)):
            reject_reason = "missing_local_support"
        elif float(local_deltas.get("psnr", 0.0)) < float(args.pairwise_dominance_min_local_psnr_delta):
            reject_reason = "local_psnr_delta"
        elif compute_ssim and float(local_deltas.get("ssim", 0.0)) < float(args.pairwise_dominance_min_local_ssim_delta):
            reject_reason = "local_ssim_delta"
        elif float(local_deltas.get("psnr_cvar", 0.0)) < float(args.pairwise_dominance_min_local_cvar_delta):
            reject_reason = "local_cvar_delta"
        elif float(local_deltas.get("psnr_min", 0.0)) < float(args.pairwise_dominance_min_local_min_delta):
            reject_reason = "local_min_delta"
        elif bool(policy.get("ood_guard_enabled", False)) and ood_distance > float(policy.get("ood_source_distance_threshold", float("inf"))):
            reject_reason = "ood_distance"
        cand_diag = {
            "prediction": prediction,
            "local": local,
            "blend_step": blend_step,
            "max_blend_step": float(args.pairwise_dominance_max_blend_step),
            "adaptive_blend_step_enabled": bool(args.pairwise_dominance_enable_adaptive_blend_step),
            "adaptive_max_blend_step": float(args.pairwise_dominance_adaptive_max_blend_step),
            "ood_distance": float(ood_distance),
            "ood_threshold": float(policy.get("ood_source_distance_threshold", float("inf"))),
            "reject_reason": reject_reason,
        }
        diagnostics["candidate_diagnostics"][variant] = cand_diag
        if reject_reason is None:
            score = (
                float(prediction[0])
                + float(args.pairwise_dominance_psnr_weight) * float(prediction[1])
                + float(args.pairwise_dominance_ssim_weight) * float(prediction[2])
                + float(args.pairwise_dominance_local_cvar_weight) * float(local_deltas.get("psnr_cvar", 0.0))
            )
            accepted.append((score, variant, cand_diag))
    if not accepted:
        diagnostics["reject_reason"] = "no_pairwise_candidate"
        return "__incumbent__", predictions, diagnostics
    accepted.sort(key=lambda item: item[0], reverse=True)
    score, variant, cand_diag = accepted[0]
    diagnostics["best_variant"] = variant
    diagnostics["best_score"] = float(score)
    diagnostics["best_diagnostics"] = cand_diag
    return variant, predictions, diagnostics


def _fit_promotion_rollback_policy(
    pairwise_policy: dict[str, Any],
    *,
    compute_ssim: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not bool(args.enable_promotion_rollback_certificate):
        return {"enabled": False, "verdict": "disabled by CLI"}
    if not bool(pairwise_policy.get("enabled", False)):
        return {
            "enabled": False,
            "verdict": "promotion rollback requires an enabled pairwise dominance policy",
            "mode": str(args.promotion_rollback_mode),
        }
    target_names = list(pairwise_policy.get("target_names", PAIRWISE_DOMINANCE_TARGET_NAMES))
    entries_by_candidate = dict(pairwise_policy.get("entries_by_candidate", {}))
    entries_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for variant, entries in entries_by_candidate.items():
        for entry in entries:
            key = (
                str(entry.get("view")),
                str(entry.get("candidate_variant", variant)),
                str(entry.get("incumbent_variant")),
            )
            entries_by_key[key] = entry

    residuals_by_axis: dict[str, list[float]] = {name: [] for name in target_names}
    signed_residuals_by_axis: dict[str, list[float]] = {name: [] for name in target_names}
    calibration_rows: list[dict[str, Any]] = []
    for item in pairwise_policy.get("loo_decisions", []):
        predictions = item.get("predictions")
        if not isinstance(predictions, dict):
            continue
        view = str(item.get("view"))
        incumbent_variant = str(item.get("incumbent_variant"))
        for candidate_variant, prediction in predictions.items():
            entry = entries_by_key.get((view, str(candidate_variant), incumbent_variant))
            if entry is None:
                continue
            target = list(entry.get("target", []))
            if len(target) < len(target_names) or len(prediction) < len(target_names):
                continue
            row_residuals: dict[str, float] = {}
            for axis_idx, axis_name in enumerate(target_names):
                signed = float(prediction[axis_idx]) - float(target[axis_idx])
                signed_residuals_by_axis[axis_name].append(float(signed))
                over = max(0.0, float(signed))
                residuals_by_axis[axis_name].append(float(over))
                row_residuals[axis_name] = float(over)
            calibration_rows.append(
                {
                    "view": view,
                    "candidate_variant": str(candidate_variant),
                    "incumbent_variant": incumbent_variant,
                    "overprediction_residuals": row_residuals,
                }
            )

    sample_count = min((len(values) for values in residuals_by_axis.values()), default=0)
    bounds = {
        axis_name: float(_quantile(values, float(args.promotion_rollback_calibration_quantile)))
        for axis_name, values in residuals_by_axis.items()
    }
    signed_summary = {
        axis_name: {
            "count": int(len(values)),
            "mean": _mean(values),
            "max_overprediction": max(residuals_by_axis.get(axis_name, [0.0])) if residuals_by_axis.get(axis_name) else None,
        }
        for axis_name, values in signed_residuals_by_axis.items()
    }
    sources = [
        part.strip()
        for part in str(args.promotion_rollback_sources).split(",")
        if part.strip()
    ]
    base_payload = {
        "mode": str(args.promotion_rollback_mode),
        "sources": sources,
        "target_names": target_names,
        "source_sample_count": int(sample_count),
        "min_calibration_samples": int(args.promotion_rollback_min_calibration_samples),
        "calibration_quantile": float(args.promotion_rollback_calibration_quantile),
        "calibration_scale": float(args.promotion_rollback_calibration_scale),
        "calibration_error_bounds": bounds,
        "source_loo_summary": signed_summary,
        "min_lcb_objective_delta": float(args.promotion_rollback_min_lcb_objective_delta),
        "min_lcb_psnr_delta": float(args.promotion_rollback_min_lcb_psnr_delta),
        "min_lcb_ssim_delta": float(args.promotion_rollback_min_lcb_ssim_delta),
        "min_local_cvar_delta": float(args.promotion_rollback_min_local_cvar_delta),
        "min_local_min_delta": float(args.promotion_rollback_min_local_min_delta),
        "max_local_negative_fraction": float(args.promotion_rollback_max_local_negative_fraction),
        "calibration_rows": calibration_rows,
    }
    if sample_count < int(args.promotion_rollback_min_calibration_samples):
        return {
            "enabled": False,
            "verdict": "not enough source LOO calibration samples for promotion rollback",
            **base_payload,
        }
    return {
        "enabled": True,
        "verdict": "post-decision promotion rollback certificate fitted",
        **base_payload,
    }


def _promotion_rollback_decision(
    *,
    output_variant: str,
    incumbent_variant: str,
    decision_source: str,
    pairwise_diagnostics: dict[str, Any] | None,
    policy: dict[str, Any],
    compute_ssim: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": bool(policy.get("enabled", False)),
        "mode": str(policy.get("mode", "shadow")),
        "candidate_variant": output_variant,
        "incumbent_variant": incumbent_variant,
        "decision_source": decision_source,
        "decision": "keep",
        "rollback_applied": False,
        "reject_reason": None,
        "prediction": None,
        "calibrated_lower_bounds": None,
        "local_deltas": None,
    }
    if not bool(policy.get("enabled", False)):
        payload["reject_reason"] = "disabled"
        return payload
    if output_variant in {"noop", "__scene__", "__incumbent__"} or output_variant == incumbent_variant:
        payload["reject_reason"] = "no_promotion"
        return payload
    if decision_source not in set(policy.get("sources", [])):
        payload["reject_reason"] = "decision_source_not_checked"
        return payload
    if not pairwise_diagnostics:
        payload["decision"] = "rollback" if str(policy.get("mode", "shadow")) == "enforce" else "shadow_rollback"
        payload["rollback_applied"] = str(policy.get("mode", "shadow")) == "enforce"
        payload["reject_reason"] = "missing_pairwise_diagnostics"
        return payload
    candidate_diagnostics = dict(pairwise_diagnostics.get("candidate_diagnostics", {}))
    candidate_diag = candidate_diagnostics.get(output_variant)
    if not isinstance(candidate_diag, dict):
        payload["decision"] = "rollback" if str(policy.get("mode", "shadow")) == "enforce" else "shadow_rollback"
        payload["rollback_applied"] = str(policy.get("mode", "shadow")) == "enforce"
        payload["reject_reason"] = "missing_candidate_diagnostics"
        return payload
    prediction = [float(value) for value in candidate_diag.get("prediction", [])]
    target_names = list(policy.get("target_names", PAIRWISE_DOMINANCE_TARGET_NAMES))
    bounds_by_name = dict(policy.get("calibration_error_bounds", {}))
    lcb_by_name: dict[str, float] = {}
    for axis_idx, axis_name in enumerate(target_names):
        if axis_idx >= len(prediction):
            continue
        bound = float(bounds_by_name.get(axis_name, 0.0)) * float(policy.get("calibration_scale", 1.0))
        lcb_by_name[axis_name] = float(prediction[axis_idx]) - bound
    local = dict(candidate_diag.get("local", {}))
    local_deltas = dict(local.get("deltas", {}))
    payload["prediction"] = prediction
    payload["calibrated_lower_bounds"] = lcb_by_name
    payload["local_deltas"] = local_deltas

    reject_reason = None
    if lcb_by_name.get("objective_delta_vs_incumbent", 0.0) < float(policy.get("min_lcb_objective_delta", -1.0e9)):
        reject_reason = "lcb_objective_delta"
    elif lcb_by_name.get("psnr_delta_vs_incumbent", 0.0) < float(policy.get("min_lcb_psnr_delta", -1.0e9)):
        reject_reason = "lcb_psnr_delta"
    elif (
        compute_ssim
        and lcb_by_name.get("ssim_delta_vs_incumbent", 0.0) < float(policy.get("min_lcb_ssim_delta", -1.0e9))
    ):
        reject_reason = "lcb_ssim_delta"
    elif float(local_deltas.get("psnr_cvar", 0.0)) < float(policy.get("min_local_cvar_delta", -1.0e9)):
        reject_reason = "local_cvar_delta"
    elif float(local_deltas.get("psnr_min", 0.0)) < float(policy.get("min_local_min_delta", -1.0e9)):
        reject_reason = "local_min_delta"
    else:
        negative_fraction = 1.0 - float(local_deltas.get("positive_fraction", 1.0))
        payload["local_negative_fraction"] = float(negative_fraction)
        if negative_fraction > float(policy.get("max_local_negative_fraction", 1.0)):
            reject_reason = "local_negative_fraction"

    if reject_reason is None:
        payload["decision"] = "keep"
        payload["reject_reason"] = None
        return payload

    payload["reject_reason"] = reject_reason
    if str(policy.get("mode", "shadow")) == "enforce":
        payload["decision"] = "rollback"
        payload["rollback_applied"] = True
    else:
        payload["decision"] = "shadow_rollback"
        payload["rollback_applied"] = False
    return payload


def _tnc_fit_hw(height: int, width: int, max_side: int) -> tuple[int, int] | None:
    if int(max_side) <= 0 or max(height, width) <= int(max_side):
        return None
    scale = float(max_side) / float(max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _tnc_resize_chw(image: torch.Tensor, size: tuple[int, int] | None) -> torch.Tensor:
    if size is None or tuple(image.shape[-2:]) == tuple(size):
        return image
    return F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)


def _tnc_resize_hw(depth: torch.Tensor, size: tuple[int, int] | None) -> torch.Tensor:
    if size is None or tuple(depth.shape[-2:]) == tuple(size):
        return depth
    return F.interpolate(depth[None, None], size=size, mode="bilinear", align_corners=False).squeeze(0).squeeze(0)


def _tnc_weighted_mae(
    warped: torch.Tensor,
    reference: torch.Tensor,
    confidence: torch.Tensor,
    *,
    min_confidence: float,
) -> tuple[float, float]:
    mask = (confidence > float(min_confidence)).to(device=warped.device, dtype=warped.dtype)
    denom = torch.clamp(mask.sum() * float(warped.shape[0]), min=1.0)
    mae = (torch.abs(warped - reference) * mask.unsqueeze(0)).sum() / denom
    return float(mae.detach().cpu().item()), float(mask.mean().detach().cpu().item())


def _target_neighbor_consistency_score(
    *,
    target: Any,
    image: torch.Tensor,
    target_frames: Sequence[Any],
    loader: FrameLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    neighbors = select_support_frames(
        target,
        target_frames,
        k=int(args.target_neighbor_consistency_neighbor_k),
        exclude_names={target.name, target.camera.image_name},
        direction_weight=float(args.target_neighbor_consistency_direction_weight),
    )
    target_depth_full = loader.depth(str(target.depth_path)).to(device=device, dtype=torch.float32)
    target_size = _tnc_fit_hw(
        int(target_depth_full.shape[0]),
        int(target_depth_full.shape[1]),
        int(args.target_neighbor_consistency_max_side),
    )
    target_depth = _tnc_resize_hw(target_depth_full, target_size)
    target_image = _tnc_resize_chw(image.to(device=device, dtype=torch.float32), target_size)

    rows: list[dict[str, Any]] = []
    weighted_error = 0.0
    total_weight = 0.0
    confident_fractions: list[float] = []
    for neighbor, view_weight in neighbors:
        neighbor_depth_full = loader.depth(str(neighbor.depth_path)).to(device=device, dtype=torch.float32)
        neighbor_base_full = loader.render(str(neighbor.render_path)).to(device=device, dtype=torch.float32)
        neighbor_size = _tnc_fit_hw(
            int(neighbor_depth_full.shape[0]),
            int(neighbor_depth_full.shape[1]),
            int(args.target_neighbor_consistency_max_side),
        )
        neighbor_depth = _tnc_resize_hw(neighbor_depth_full, neighbor_size)
        neighbor_base = _tnc_resize_chw(neighbor_base_full, neighbor_size)
        warped, confidence = warp_support_residual(
            neighbor,
            target,
            neighbor_depth,
            target_depth,
            target_image,
            depth_abs_tol=float(args.target_neighbor_consistency_depth_abs_tol),
            depth_rel_tol=float(args.target_neighbor_consistency_depth_rel_tol),
            device=device,
        )
        mae, confident_fraction = _tnc_weighted_mae(
            warped,
            neighbor_base,
            confidence,
            min_confidence=float(args.target_neighbor_consistency_min_confidence),
        )
        effective_weight = float(view_weight) * max(confident_fraction, 0.0)
        rows.append(
            {
                "neighbor": str(neighbor.name),
                "view_weight": float(view_weight),
                "mae_to_neighbor_base": float(mae),
                "confident_fraction": float(confident_fraction),
                "effective_weight": float(effective_weight),
            }
        )
        weighted_error += mae * effective_weight
        total_weight += effective_weight
        confident_fractions.append(confident_fraction)
    return {
        "available": bool(rows),
        "neighbor_count": int(len(rows)),
        "mean_mae_to_neighbor_base": float(weighted_error / max(total_weight, 1.0e-12)) if rows else None,
        "mean_confident_fraction": float(sum(confident_fractions) / len(confident_fractions)) if confident_fractions else 0.0,
        "total_effective_weight": float(total_weight),
        "neighbors": rows,
    }


def _target_neighbor_consistency_decision(
    *,
    output_variant: str,
    incumbent_variant: str,
    decision_source: str,
    pairwise_diagnostics: dict[str, Any] | None,
    ev: Any,
    deltas: dict[str, torch.Tensor],
    target: Any,
    target_frames: Sequence[Any],
    loader: FrameLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    enabled = bool(args.enable_target_neighbor_consistency_certificate)
    payload: dict[str, Any] = {
        "enabled": enabled,
        "mode": str(args.target_neighbor_consistency_mode),
        "candidate_variant": output_variant,
        "incumbent_variant": incumbent_variant,
        "decision_source": decision_source,
        "decision": "keep",
        "rollback_applied": False,
        "reject_reason": None,
        "incumbent_minus_output_mae_delta": None,
        "output_score": None,
        "incumbent_score": None,
        "source_local_deltas": None,
    }
    if not enabled:
        payload["reject_reason"] = "disabled"
        return payload
    if output_variant in {"noop", "__scene__", "__incumbent__"} or output_variant == incumbent_variant:
        payload["reject_reason"] = "no_promotion"
        return payload
    sources = {
        part.strip()
        for part in str(args.target_neighbor_consistency_sources).split(",")
        if part.strip()
    }
    if decision_source not in sources:
        payload["reject_reason"] = "decision_source_not_checked"
        return payload
    if output_variant not in deltas or incumbent_variant not in deltas:
        payload["decision"] = "rollback" if str(args.target_neighbor_consistency_mode) == "enforce" else "shadow_rollback"
        payload["rollback_applied"] = str(args.target_neighbor_consistency_mode) == "enforce"
        payload["reject_reason"] = "missing_candidate_delta"
        return payload
    output_image = torch.clamp(ev.base + deltas[output_variant], 0.0, 1.0)
    incumbent_image = torch.clamp(ev.base + deltas[incumbent_variant], 0.0, 1.0)
    output_score = _target_neighbor_consistency_score(
        target=target,
        image=output_image,
        target_frames=target_frames,
        loader=loader,
        device=device,
        args=args,
    )
    incumbent_score = _target_neighbor_consistency_score(
        target=target,
        image=incumbent_image,
        target_frames=target_frames,
        loader=loader,
        device=device,
        args=args,
    )
    payload["output_score"] = output_score
    payload["incumbent_score"] = incumbent_score
    output_error = output_score.get("mean_mae_to_neighbor_base")
    incumbent_error = incumbent_score.get("mean_mae_to_neighbor_base")
    if output_error is None or incumbent_error is None:
        payload["decision"] = "rollback" if str(args.target_neighbor_consistency_mode) == "enforce" else "shadow_rollback"
        payload["rollback_applied"] = str(args.target_neighbor_consistency_mode) == "enforce"
        payload["reject_reason"] = "score_unavailable"
        return payload
    if float(output_score.get("total_effective_weight", 0.0)) < float(args.target_neighbor_consistency_min_effective_weight):
        payload["decision"] = "rollback" if str(args.target_neighbor_consistency_mode) == "enforce" else "shadow_rollback"
        payload["rollback_applied"] = str(args.target_neighbor_consistency_mode) == "enforce"
        payload["reject_reason"] = "insufficient_target_neighbor_support"
        payload["incumbent_minus_output_mae_delta"] = float(incumbent_error) - float(output_error)
        return payload

    mae_delta = float(incumbent_error) - float(output_error)
    payload["incumbent_minus_output_mae_delta"] = float(mae_delta)
    if mae_delta >= float(args.target_neighbor_consistency_min_incumbent_minus_output_delta):
        if bool(args.target_neighbor_consistency_enable_source_contradiction):
            candidate_diag = (
                dict((pairwise_diagnostics or {}).get("candidate_diagnostics", {}))
                .get(output_variant, {})
            )
            local = dict(candidate_diag.get("local", {})) if isinstance(candidate_diag, dict) else {}
            local_deltas = dict(local.get("deltas", {}))
            payload["source_local_deltas"] = local_deltas
            source_local_min = float(local_deltas.get("psnr_min", -1.0e9))
            source_local_cvar = float(local_deltas.get("psnr_cvar", -1.0e9))
            source_positive_fraction = float(local_deltas.get("positive_fraction", 0.0))
            source_strong = (
                source_local_min >= float(args.target_neighbor_consistency_contradiction_min_source_local_min_delta)
                and source_local_cvar >= float(args.target_neighbor_consistency_contradiction_min_source_local_cvar_delta)
                and source_positive_fraction >= float(args.target_neighbor_consistency_contradiction_min_source_positive_fraction)
            )
            target_negative = mae_delta < float(args.target_neighbor_consistency_contradiction_max_incumbent_minus_output_delta)
            if source_strong and target_negative:
                payload["reject_reason"] = "source_target_neighbor_contradiction"
                if str(args.target_neighbor_consistency_mode) == "enforce":
                    payload["decision"] = "rollback"
                    payload["rollback_applied"] = True
                else:
                    payload["decision"] = "shadow_rollback"
                    payload["rollback_applied"] = False
                return payload
        return payload

    payload["reject_reason"] = "target_neighbor_consistency_delta"
    if str(args.target_neighbor_consistency_mode) == "enforce":
        payload["decision"] = "rollback"
        payload["rollback_applied"] = True
    else:
        payload["decision"] = "shadow_rollback"
        payload["rollback_applied"] = False
    return payload


def _target_neighbor_candidate_unlock_decision(
    *,
    output_variant: str,
    ev: Any,
    deltas: dict[str, torch.Tensor],
    target: Any,
    target_frames: Sequence[Any],
    loader: FrameLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    enabled = bool(args.enable_target_neighbor_candidate_unlock)
    incumbent_variant = str(args.target_neighbor_candidate_unlock_incumbent_variant)
    candidate_variant = str(args.target_neighbor_candidate_unlock_candidate_variant)
    payload: dict[str, Any] = {
        "enabled": enabled,
        "incumbent_variant": incumbent_variant,
        "candidate_variant": candidate_variant,
        "input_variant": output_variant,
        "decision": "keep",
        "promote_applied": False,
        "reject_reason": None,
        "incumbent_minus_candidate_mae_delta": None,
        "candidate_score": None,
        "incumbent_score": None,
    }
    if not enabled:
        payload["reject_reason"] = "disabled"
        return payload
    if output_variant != incumbent_variant:
        payload["reject_reason"] = "input_not_incumbent"
        return payload
    if incumbent_variant not in deltas or candidate_variant not in deltas:
        payload["reject_reason"] = "missing_candidate_delta"
        return payload

    incumbent_image = torch.clamp(ev.base + deltas[incumbent_variant], 0.0, 1.0)
    candidate_image = torch.clamp(ev.base + deltas[candidate_variant], 0.0, 1.0)
    incumbent_score = _target_neighbor_consistency_score(
        target=target,
        image=incumbent_image,
        target_frames=target_frames,
        loader=loader,
        device=device,
        args=args,
    )
    candidate_score = _target_neighbor_consistency_score(
        target=target,
        image=candidate_image,
        target_frames=target_frames,
        loader=loader,
        device=device,
        args=args,
    )
    payload["incumbent_score"] = incumbent_score
    payload["candidate_score"] = candidate_score
    incumbent_error = incumbent_score.get("mean_mae_to_neighbor_base")
    candidate_error = candidate_score.get("mean_mae_to_neighbor_base")
    if incumbent_error is None or candidate_error is None:
        payload["reject_reason"] = "score_unavailable"
        return payload
    if float(candidate_score.get("total_effective_weight", 0.0)) < float(args.target_neighbor_consistency_min_effective_weight):
        payload["reject_reason"] = "insufficient_target_neighbor_support"
        return payload

    mae_delta = float(incumbent_error) - float(candidate_error)
    payload["incumbent_minus_candidate_mae_delta"] = float(mae_delta)
    if mae_delta < float(args.target_neighbor_candidate_unlock_min_incumbent_minus_candidate_delta):
        payload["reject_reason"] = "margin_too_small"
        return payload

    payload["decision"] = "promote"
    payload["promote_applied"] = True
    return payload


def _fit_per_view_risk_model_policy(
    selector_payload: dict[str, Any] | None,
    *,
    compute_ssim: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not bool(args.enable_per_view_risk_model_policy):
        return {"enabled": False, "verdict": "disabled by CLI"}
    if selector_payload is None or not selector_payload.get("per_view"):
        return {"enabled": False, "verdict": "missing source-heldout selector per-view evidence"}
    selected_variant = str(selector_payload["selected_variant"])
    if bool(args.per_view_risk_model_only_when_scene_fixed) and selected_variant != "fixed":
        return {
            "enabled": False,
            "selected_variant": selected_variant,
            "verdict": "disabled because scene-level source-heldout selector was not fixed",
        }
    if selected_variant == "fixed" and not bool(args.per_view_risk_model_allow_when_scene_fixed):
        return {
            "enabled": False,
            "selected_variant": selected_variant,
            "verdict": "disabled because scene-level source-heldout selector fell back to fixed",
        }
    feature_names = _parse_feature_names(args.per_view_risk_model_feature_grid)
    variants = _candidate_variant_names(args)
    variant_blend_map = _candidate_variant_blend_map(args)
    examples: list[dict[str, Any]] = []
    rows_by_view: dict[str, dict[str, Any]] = {}
    entries_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    all_feature_vectors: list[list[float]] = []
    for row in selector_payload["per_view"]:
        rows_by_view[str(row["view"])] = row
        for variant in variants:
            candidate = row["candidates"][variant]
            metrics = candidate["metrics"]
            features = _candidate_learning_features(
                candidate["proxy"],
                variant,
                feature_names,
                variant_blend=variant_blend_map.get(variant),
            )
            target = [
                _risk_model_objective_from_metrics(metrics, compute_ssim=compute_ssim, args=args),
                float(metrics.get("psnr_gain", 0.0)),
                float(metrics.get("ssim_gain", 0.0)) if compute_ssim else 0.0,
            ]
            example = {
                "view": row["view"],
                "variant": variant,
                "features": features,
                "target": target,
                "metrics": metrics,
            }
            examples.append(example)
            entries_by_variant[variant].append(
                {
                    "view": row["view"],
                    "variant": variant,
                    "vector": features,
                    "target": target,
                }
            )
            all_feature_vectors.append(features)
    if len({example["view"] for example in examples}) < 2:
        return {"enabled": False, "verdict": "not enough source-heldout views for leave-one-out risk model"}
    ood_feature_mean, ood_feature_std = _feature_stats(all_feature_vectors)
    ood_source_distances: list[float] = []

    def nearest_source_distance(variant: str, vector: list[float], *, exclude_view: str | None = None) -> float:
        pool = [
            entry
            for entry in entries_by_variant[variant]
            if exclude_view is None or str(entry["view"]) != str(exclude_view)
        ]
        if not pool or not ood_feature_mean:
            return float("inf")
        return min(
            _normalized_distance(vector, entry["vector"], ood_feature_mean, ood_feature_std)
            for entry in pool
        )

    for example in examples:
        dist = nearest_source_distance(
            str(example["variant"]),
            list(example["features"]),
            exclude_view=str(example["view"]),
        )
        if math.isfinite(dist):
            ood_source_distances.append(dist)
    ood_threshold = (
        _quantile(ood_source_distances, float(args.per_view_risk_model_ood_quantile))
        if bool(args.per_view_risk_model_enable_ood_guard)
        else float("inf")
    )

    def predict_for_row(row: dict[str, Any], model: dict[str, Any]) -> dict[str, list[float]]:
        predictions: dict[str, list[float]] = {}
        for variant in variants:
            candidate = row["candidates"][variant]
            predictions[variant] = _predict_ridge(
                model,
                _candidate_learning_features(
                    candidate["proxy"],
                    variant,
                    feature_names,
                    variant_blend=variant_blend_map.get(variant),
                ),
            )
        return predictions

    def choose_from_predictions(
        predictions: dict[str, list[float]],
        *,
        objective_margin: float,
    ) -> str:
        scene_pred = predictions[selected_variant]
        best_variant, best_pred = max(predictions.items(), key=lambda item: item[1][0])
        if best_pred[0] < scene_pred[0] + float(objective_margin):
            return "__scene__" if str(args.per_view_risk_model_reject_variant) == "scene" else "noop"
        if bool(args.per_view_risk_model_require_predicted_scene_axis_nonregression):
            if best_pred[1] < scene_pred[1] + float(args.per_view_risk_model_scene_axis_guard_margin_psnr):
                return "__scene__" if str(args.per_view_risk_model_reject_variant) == "scene" else "noop"
            if (
                compute_ssim
                and best_pred[2] < scene_pred[2] + float(args.per_view_risk_model_scene_axis_guard_margin_ssim)
            ):
                return "__scene__" if str(args.per_view_risk_model_reject_variant) == "scene" else "noop"
        if best_pred[1] < scene_pred[1] + float(args.per_view_risk_model_min_predicted_psnr_delta_vs_scene):
            return "__scene__" if str(args.per_view_risk_model_reject_variant) == "scene" else "noop"
        if (
            compute_ssim
            and best_pred[2] < scene_pred[2] + float(args.per_view_risk_model_min_predicted_ssim_delta_vs_scene)
        ):
            return "__scene__" if str(args.per_view_risk_model_reject_variant) == "scene" else "noop"
        if best_pred[1] < float(args.per_view_risk_model_min_predicted_psnr):
            return "__scene__" if str(args.per_view_risk_model_reject_variant) == "scene" else "noop"
        if compute_ssim and best_pred[2] < float(args.per_view_risk_model_min_predicted_ssim):
            return "__scene__" if str(args.per_view_risk_model_reject_variant) == "scene" else "noop"
        return best_variant

    loo_items: list[dict[str, Any]] = []
    for view, row in rows_by_view.items():
        train_examples = [example for example in examples if str(example["view"]) != str(view)]
        model = _fit_ridge_predictor(train_examples, ridge=float(args.per_view_risk_model_ridge))
        if model is None:
            loo_items.append({"view": view, "row": row, "predictions": None})
            continue
        loo_items.append({"view": view, "row": row, "predictions": predict_for_row(row, model)})

    def evaluate_risk_margin(objective_margin: float) -> tuple[dict[str, Any], dict[str, int], list[dict[str, float]], list[dict[str, Any]]]:
        source_policy_rows: list[dict[str, float]] = []
        selected_counts: dict[str, int] = _candidate_count_dict(variants)
        predictions_log: list[dict[str, Any]] = []
        for item in loo_items:
            row = item["row"]
            predictions = item.get("predictions")
            if predictions is None:
                selected_counts["scene"] += 1
                source_policy_rows.append(row["candidates"][selected_variant]["metrics"])
                predictions_log.append({"view": item["view"], "chosen_variant": "__scene__", "predictions": None})
                continue
            chosen_variant = choose_from_predictions(predictions, objective_margin=float(objective_margin))
            predictions_log.append({"view": item["view"], "chosen_variant": chosen_variant, "predictions": predictions})
            if chosen_variant == "__scene__":
                selected_counts["scene"] += 1
                source_policy_rows.append(row["candidates"][selected_variant]["metrics"])
            elif chosen_variant == "noop":
                selected_counts["noop"] += 1
                source_policy_rows.append(_noop_like(row["candidates"][selected_variant]["metrics"]))
            else:
                selected_counts[chosen_variant] += 1
                source_policy_rows.append(row["candidates"][chosen_variant]["metrics"])
        return (
            _summarize_metric_rows(source_policy_rows, compute_ssim=compute_ssim),
            selected_counts,
            source_policy_rows,
            predictions_log,
        )

    selected_objective_margin = float(args.per_view_risk_model_min_predicted_objective_delta)
    risk_margin_trials: list[dict[str, Any]] = []
    if bool(args.per_view_risk_model_auto_objective_margin):
        candidate_margins = {selected_objective_margin}
        for item in loo_items:
            predictions = item.get("predictions")
            if not predictions:
                continue
            scene_pred = predictions[selected_variant]
            best_variant, best_pred = max(predictions.items(), key=lambda pair: pair[1][0])
            if best_variant != selected_variant:
                candidate_margins.add(max(0.0, float(best_pred[0]) - float(scene_pred[0])))
        if candidate_margins:
            ordered = sorted(candidate_margins)
            candidate_margins.add(max(0.0, ordered[0] - 1.0e-9))
            candidate_margins.add(ordered[-1] + 1.0e-9)
        best_trial: dict[str, Any] | None = None
        for margin in sorted(candidate_margins):
            trial_summary, trial_counts, _, _ = evaluate_risk_margin(float(margin))
            reject_count = int(trial_counts["noop"] + trial_counts["scene"])
            accept_fraction_trial = 1.0 - float(reject_count / max(len(loo_items), 1))
            mean_delta_trial = float(trial_summary.get("psnr_gain", 0.0)) - float(selector_payload["summaries"][selected_variant].get("psnr_gain", 0.0))
            ssim_delta_trial = (
                float(trial_summary.get("ssim_gain", 0.0)) - float(selector_payload["summaries"][selected_variant].get("ssim_gain", 0.0))
                if compute_ssim
                else 0.0
            )
            cvar_delta_trial = _summary_psnr_tail(trial_summary, "cvar") - _summary_psnr_tail(selector_payload["summaries"][selected_variant], "cvar")
            min_delta_trial = _summary_psnr_tail(trial_summary, "min") - _summary_psnr_tail(selector_payload["summaries"][selected_variant], "min")
            positive_fraction_delta_trial = _positive_view_fraction(trial_summary) - _positive_view_fraction(selector_payload["summaries"][selected_variant])
            safe_vs_fixed_trial = (
                float(trial_summary.get("psnr_gain", 0.0)) >= float(selector_payload["summaries"]["fixed"].get("psnr_gain", 0.0)) - float(args.selected_safe_tolerance_psnr)
                and (
                    not compute_ssim
                    or float(trial_summary.get("ssim_gain", 0.0))
                    >= float(selector_payload["summaries"]["fixed"].get("ssim_gain", 0.0)) - float(args.selected_safe_tolerance_ssim)
                )
            )
            objective = (
                _source_objective_from_metrics(trial_summary, compute_ssim=compute_ssim, args=args)
                + float(args.per_view_risk_model_source_cvar_weight) * _summary_psnr_tail(trial_summary, "cvar")
                + float(args.per_view_risk_model_source_min_weight) * _summary_psnr_tail(trial_summary, "min")
                + float(args.per_view_risk_model_source_positive_weight) * _positive_view_fraction(trial_summary)
            )
            trial = {
                "objective_margin": float(margin),
                "objective": float(objective),
                "accept_fraction": float(accept_fraction_trial),
                "source_summary": trial_summary,
                "source_selected_counts": trial_counts,
                "source_mean_psnr_delta_vs_scene_selected": mean_delta_trial,
                "source_mean_ssim_delta_vs_scene_selected": ssim_delta_trial,
                "source_cvar_psnr_delta_vs_scene_selected": cvar_delta_trial,
                "source_min_psnr_delta_vs_scene_selected": min_delta_trial,
                "source_positive_view_fraction_delta_vs_scene_selected": positive_fraction_delta_trial,
                "source_safe_vs_fixed": bool(safe_vs_fixed_trial),
            }
            risk_margin_trials.append(trial)
            if accept_fraction_trial < float(args.per_view_risk_model_min_accept_fraction):
                continue
            if accept_fraction_trial > float(args.per_view_risk_model_max_accept_fraction):
                continue
            if mean_delta_trial < float(args.per_view_risk_model_min_source_psnr_delta):
                continue
            if compute_ssim and ssim_delta_trial < float(args.per_view_risk_model_min_source_ssim_delta):
                continue
            if cvar_delta_trial < float(args.per_view_risk_model_min_source_cvar_delta):
                continue
            if min_delta_trial < float(args.per_view_risk_model_min_source_min_delta):
                continue
            if positive_fraction_delta_trial < float(args.per_view_risk_model_min_source_positive_fraction_delta):
                continue
            if bool(args.per_view_risk_model_require_source_safe) and not safe_vs_fixed_trial:
                continue
            if best_trial is None or float(objective) > float(best_trial["objective"]):
                best_trial = trial
        if best_trial is not None:
            selected_objective_margin = float(best_trial["objective_margin"])

    source_summary, selected_counts, _, loo_predictions = evaluate_risk_margin(selected_objective_margin)
    fixed_summary = selector_payload["summaries"]["fixed"]
    scene_selected_summary = selector_payload["summaries"][selected_variant]
    mean_delta = float(source_summary.get("psnr_gain", 0.0)) - float(scene_selected_summary.get("psnr_gain", 0.0))
    ssim_delta = (
        float(source_summary.get("ssim_gain", 0.0)) - float(scene_selected_summary.get("ssim_gain", 0.0))
        if compute_ssim
        else 0.0
    )
    cvar_delta = _summary_psnr_tail(source_summary, "cvar") - _summary_psnr_tail(scene_selected_summary, "cvar")
    min_delta = _summary_psnr_tail(source_summary, "min") - _summary_psnr_tail(scene_selected_summary, "min")
    positive_fraction_delta = _positive_view_fraction(source_summary) - _positive_view_fraction(scene_selected_summary)
    safe_vs_fixed = (
        float(source_summary.get("psnr_gain", 0.0)) >= float(fixed_summary.get("psnr_gain", 0.0)) - float(args.selected_safe_tolerance_psnr)
        and (
            not compute_ssim
            or float(source_summary.get("ssim_gain", 0.0))
            >= float(fixed_summary.get("ssim_gain", 0.0)) - float(args.selected_safe_tolerance_ssim)
        )
    )
    accept_fraction = 1.0 - float((selected_counts["noop"] + selected_counts["scene"]) / max(len(rows_by_view), 1))
    base_payload = {
        "feature_schema_version": 2,
        "feature_names": feature_names,
        "variants": variants,
        "variant_blend_map": variant_blend_map,
        "ridge": float(args.per_view_risk_model_ridge),
        "reject_variant": str(args.per_view_risk_model_reject_variant),
        "scene_selected_variant": selected_variant,
        "source_summary": source_summary,
        "source_fixed_summary": fixed_summary,
        "source_scene_selected_summary": scene_selected_summary,
        "source_mean_psnr_delta_vs_scene_selected": mean_delta,
        "source_mean_ssim_delta_vs_scene_selected": ssim_delta,
        "source_cvar_psnr_delta_vs_scene_selected": cvar_delta,
        "source_min_psnr_delta_vs_scene_selected": min_delta,
        "source_positive_view_fraction_delta_vs_scene_selected": positive_fraction_delta,
        "source_safe_vs_fixed": bool(safe_vs_fixed),
        "source_selected_counts": selected_counts,
        "source_accept_fraction": accept_fraction,
        "auto_objective_margin": bool(args.per_view_risk_model_auto_objective_margin),
        "selected_objective_margin": float(selected_objective_margin),
        "objective_margin_trials": risk_margin_trials,
        "ood_guard_enabled": bool(args.per_view_risk_model_enable_ood_guard),
        "ood_quantile": float(args.per_view_risk_model_ood_quantile),
        "ood_source_distance_count": int(len(ood_source_distances)),
        "ood_source_distance_threshold": float(ood_threshold),
        "ood_source_distance_summary": {
            "min": min(ood_source_distances) if ood_source_distances else None,
            "mean": _mean(ood_source_distances) if ood_source_distances else None,
            "max": max(ood_source_distances) if ood_source_distances else None,
        },
        "source_objective_weights": {
            "lpips": float(args.source_objective_lpips_weight),
            "dists": float(args.source_objective_dists_weight),
            "used_by_risk_model": bool(args.per_view_risk_model_use_source_perceptual_objective),
        },
        "require_predicted_scene_axis_nonregression": bool(
            args.per_view_risk_model_require_predicted_scene_axis_nonregression
        ),
        "scene_axis_guard_margin_psnr": float(args.per_view_risk_model_scene_axis_guard_margin_psnr),
        "scene_axis_guard_margin_ssim": float(args.per_view_risk_model_scene_axis_guard_margin_ssim),
        "loo_predictions": loo_predictions,
    }
    if (
        bool(args.per_view_risk_model_enable_ood_guard)
        and len(ood_source_distances) < int(args.per_view_risk_model_ood_min_samples)
    ):
        return {
            "enabled": False,
            "verdict": "source-heldout risk model did not have enough OOD calibration distances",
            **base_payload,
        }
    if accept_fraction < float(args.per_view_risk_model_min_accept_fraction):
        return {
            "enabled": False,
            "verdict": "source-heldout risk model accepted too few views",
            **base_payload,
        }
    if mean_delta < float(args.per_view_risk_model_min_source_psnr_delta):
        return {
            "enabled": False,
            "verdict": "source-heldout risk model did not improve scene-selected PSNR enough",
            **base_payload,
        }
    if compute_ssim and ssim_delta < float(args.per_view_risk_model_min_source_ssim_delta):
        return {
            "enabled": False,
            "verdict": "source-heldout risk model did not clear SSIM delta",
            **base_payload,
        }
    if cvar_delta < float(args.per_view_risk_model_min_source_cvar_delta):
        return {
            "enabled": False,
            "verdict": "source-heldout risk model did not clear CVaR tail delta",
            **base_payload,
        }
    if min_delta < float(args.per_view_risk_model_min_source_min_delta):
        return {
            "enabled": False,
            "verdict": "source-heldout risk model did not clear min-gain tail delta",
            **base_payload,
        }
    if positive_fraction_delta < float(args.per_view_risk_model_min_source_positive_fraction_delta):
        return {
            "enabled": False,
            "verdict": "source-heldout risk model did not clear positive-view fraction delta",
            **base_payload,
        }
    if bool(args.per_view_risk_model_require_source_safe) and not safe_vs_fixed:
        return {
            "enabled": False,
            "verdict": "source-heldout risk model did not clear fixed safety gate",
            **base_payload,
        }
    full_model = _fit_ridge_predictor(examples, ridge=float(args.per_view_risk_model_ridge))
    if full_model is None:
        return {
            "enabled": False,
            "verdict": "could not fit full source-heldout risk model",
            **base_payload,
        }
    return {
        "enabled": True,
        "verdict": "source-heldout learned risk model selected",
        "model": full_model,
        "source_entries_by_variant": entries_by_variant,
        "ood_feature_mean": ood_feature_mean,
        "ood_feature_std": ood_feature_std,
        "min_predicted_objective_delta": float(selected_objective_margin),
        "min_predicted_psnr_delta_vs_scene": float(args.per_view_risk_model_min_predicted_psnr_delta_vs_scene),
        "min_predicted_ssim_delta_vs_scene": float(args.per_view_risk_model_min_predicted_ssim_delta_vs_scene),
        "min_predicted_psnr": float(args.per_view_risk_model_min_predicted_psnr),
        "min_predicted_ssim": float(args.per_view_risk_model_min_predicted_ssim),
        **base_payload,
    }


def _risk_model_choose_variant(
    proxies_by_variant: dict[str, dict[str, float]],
    policy: dict[str, Any],
    *,
    compute_ssim: bool,
) -> tuple[str, dict[str, list[float]], dict[str, Any]]:
    variants = list(policy.get("variants", BASE_CANDIDATE_VARIANTS))
    variant_blend_map = {str(k): float(v) for k, v in dict(policy.get("variant_blend_map", {})).items()}
    feature_names = list(policy["feature_names"])
    scene_variant = str(policy["scene_selected_variant"])
    predictions: dict[str, list[float]] = {}
    feature_vectors: dict[str, list[float]] = {}
    for variant in variants:
        feature_vectors[variant] = _candidate_learning_features(
            proxies_by_variant[variant],
            variant,
            feature_names,
            variant_blend=variant_blend_map.get(variant),
        )
        predictions[variant] = _predict_ridge(
            policy["model"],
            feature_vectors[variant],
        )
    scene_pred = predictions[scene_variant]
    best_variant, best_pred = max(predictions.items(), key=lambda item: item[1][0])
    diagnostics: dict[str, Any] = {
        "best_variant": best_variant,
        "scene_variant": scene_variant,
        "best_prediction": best_pred,
        "scene_prediction": scene_pred,
        "reject_reason": None,
        "ood_distance": None,
        "ood_threshold": None,
    }
    min_objective_delta = float(policy.get("min_predicted_objective_delta", 0.0))
    min_psnr_delta_vs_scene = float(policy.get("min_predicted_psnr_delta_vs_scene", -1.0e9))
    min_ssim_delta_vs_scene = float(policy.get("min_predicted_ssim_delta_vs_scene", -1.0e9))
    min_predicted_psnr = float(policy.get("min_predicted_psnr", -1.0e9))
    min_predicted_ssim = float(policy.get("min_predicted_ssim", -1.0e9))
    if best_pred[0] < scene_pred[0] + min_objective_delta:
        diagnostics["reject_reason"] = "objective_delta"
        return "__scene__" if str(policy.get("reject_variant", "scene")) == "scene" else "noop", predictions, diagnostics
    if bool(policy.get("require_predicted_scene_axis_nonregression", False)):
        psnr_margin = float(policy.get("scene_axis_guard_margin_psnr", 0.0))
        ssim_margin = float(policy.get("scene_axis_guard_margin_ssim", 0.0))
        if best_pred[1] < scene_pred[1] + psnr_margin:
            diagnostics["reject_reason"] = "scene_axis_psnr_nonregression"
            return "__scene__" if str(policy.get("reject_variant", "scene")) == "scene" else "noop", predictions, diagnostics
        if compute_ssim and best_pred[2] < scene_pred[2] + ssim_margin:
            diagnostics["reject_reason"] = "scene_axis_ssim_nonregression"
            return "__scene__" if str(policy.get("reject_variant", "scene")) == "scene" else "noop", predictions, diagnostics
    if best_pred[1] < scene_pred[1] + min_psnr_delta_vs_scene:
        diagnostics["reject_reason"] = "psnr_delta_vs_scene"
        return "__scene__" if str(policy.get("reject_variant", "scene")) == "scene" else "noop", predictions, diagnostics
    if compute_ssim and best_pred[2] < scene_pred[2] + min_ssim_delta_vs_scene:
        diagnostics["reject_reason"] = "ssim_delta_vs_scene"
        return "__scene__" if str(policy.get("reject_variant", "scene")) == "scene" else "noop", predictions, diagnostics
    if best_pred[1] < min_predicted_psnr:
        diagnostics["reject_reason"] = "absolute_psnr"
        return "__scene__" if str(policy.get("reject_variant", "scene")) == "scene" else "noop", predictions, diagnostics
    if compute_ssim and best_pred[2] < min_predicted_ssim:
        diagnostics["reject_reason"] = "absolute_ssim"
        return "__scene__" if str(policy.get("reject_variant", "scene")) == "scene" else "noop", predictions, diagnostics
    if bool(policy.get("ood_guard_enabled", False)):
        entries_by_variant = policy.get("source_entries_by_variant", {})
        pool = entries_by_variant.get(best_variant, [])
        mean = [float(x) for x in policy.get("ood_feature_mean", [])]
        std = [float(x) for x in policy.get("ood_feature_std", [])]
        if pool and mean and std:
            ood_distance = min(
                _normalized_distance(feature_vectors[best_variant], entry["vector"], mean, std)
                for entry in pool
            )
        else:
            ood_distance = float("inf")
        ood_threshold = float(policy.get("ood_source_distance_threshold", float("inf")))
        diagnostics["ood_distance"] = float(ood_distance)
        diagnostics["ood_threshold"] = float(ood_threshold)
        if ood_distance > ood_threshold:
            diagnostics["reject_reason"] = "ood_distance"
            return "__scene__" if str(policy.get("reject_variant", "scene")) == "scene" else "noop", predictions, diagnostics
    return best_variant, predictions, diagnostics


def _fit_source_reliability_policy(
    selector_payload: dict[str, Any] | None,
    *,
    compute_ssim: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not bool(args.enable_source_reliability_policy):
        return {"enabled": False, "verdict": "disabled by CLI"}
    if selector_payload is None or not selector_payload.get("per_view"):
        return {"enabled": False, "verdict": "missing source-heldout selector per-view evidence"}
    selected_variant = str(selector_payload["selected_variant"])
    feature_names = _parse_feature_names(args.source_reliability_feature_grid)
    variants = _candidate_variant_names(args)
    variant_blend_map = _candidate_variant_blend_map(args)
    examples: list[dict[str, Any]] = []
    rows_by_view: dict[str, dict[str, Any]] = {}
    entries_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    all_vectors: list[list[float]] = []
    for row in selector_payload["per_view"]:
        view = str(row["view"])
        rows_by_view[view] = row
        scene_candidate = row["candidates"][selected_variant]
        scene_metrics = scene_candidate["metrics"]
        scene_proxy = scene_candidate["proxy"]
        for variant in variants:
            candidate = row["candidates"][variant]
            metrics = candidate["metrics"]
            features = _source_reliability_features(
                candidate["proxy"],
                scene_proxy,
                variant=variant,
                scene_variant=selected_variant,
                feature_names=feature_names,
                variant_blend=variant_blend_map.get(variant),
                scene_variant_blend=variant_blend_map.get(selected_variant),
            )
            target = _source_reliability_target(metrics, scene_metrics, compute_ssim=compute_ssim, args=args)
            example = {
                "view": view,
                "variant": variant,
                "features": features,
                "target": target,
                "metrics": metrics,
                "scene_metrics": scene_metrics,
            }
            examples.append(example)
            entries_by_variant[variant].append(
                {
                    "view": view,
                    "variant": variant,
                    "vector": features,
                    "target": target,
                    "metrics": metrics,
                }
            )
            all_vectors.append(features)
    if len(rows_by_view) < 2:
        return {"enabled": False, "verdict": "not enough source-heldout views for reliability leave-one-out"}
    ood_mean, ood_std = _feature_stats(all_vectors)
    ood_source_distances: list[float] = []

    def nearest_source_distance(variant: str, vector: list[float], *, exclude_view: str | None = None) -> float:
        pool = [
            entry
            for entry in entries_by_variant.get(variant, [])
            if exclude_view is None or str(entry["view"]) != str(exclude_view)
        ]
        if not pool or not ood_mean:
            return float("inf")
        return min(_normalized_distance(vector, entry["vector"], ood_mean, ood_std) for entry in pool)

    for example in examples:
        dist = nearest_source_distance(str(example["variant"]), list(example["features"]), exclude_view=str(example["view"]))
        if math.isfinite(dist):
            ood_source_distances.append(dist)
    ood_threshold = (
        _quantile(ood_source_distances, float(args.source_reliability_ood_quantile))
        if bool(args.source_reliability_enable_ood_guard)
        else float("inf")
    )

    def predict_for_row(row: dict[str, Any], model: dict[str, Any]) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        predictions: dict[str, list[float]] = {}
        vectors: dict[str, list[float]] = {}
        scene_proxy = row["candidates"][selected_variant]["proxy"]
        for variant in variants:
            vector = _source_reliability_features(
                row["candidates"][variant]["proxy"],
                scene_proxy,
                variant=variant,
                scene_variant=selected_variant,
                feature_names=feature_names,
                variant_blend=variant_blend_map.get(variant),
                scene_variant_blend=variant_blend_map.get(selected_variant),
            )
            vectors[variant] = vector
            predictions[variant] = _predict_ridge(model, vector)
        return predictions, vectors

    def choose_from_predictions(
        predictions: dict[str, list[float]],
        vectors: dict[str, list[float]],
        *,
        objective_margin: float,
        exclude_view: str | None = None,
        allow_calibrated_lcb_override: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        def evaluate_decision_predictions(
            decision_predictions: dict[str, list[float]],
            *,
            prediction_source: str,
        ) -> tuple[str, dict[str, Any]]:
            best_variant, best_pred = max(decision_predictions.items(), key=lambda item: item[1][0])
            scene_pred = decision_predictions[selected_variant]
            diagnostics: dict[str, Any] = {
                "best_variant": best_variant,
                "scene_variant": selected_variant,
                "best_prediction": predictions[best_variant],
                "scene_prediction": predictions[selected_variant],
                "best_decision_prediction": best_pred,
                "scene_decision_prediction": scene_pred,
                "decision_prediction_source": prediction_source,
                "calibrated_lower_bounds": decision_predictions
                if prediction_source == "calibrated_lower_bound"
                else None,
                "reject_reason": None,
                "ood_distance": None,
                "ood_threshold": None,
            }

            def reject(reason: str) -> str:
                diagnostics["reject_reason"] = reason
                return "__scene__" if str(args.source_reliability_reject_variant) == "scene" else "noop"

            if best_pred[0] < float(objective_margin):
                return reject("objective_margin"), diagnostics
            if best_pred[1] < float(args.source_reliability_min_predicted_psnr_delta_vs_scene):
                return reject("psnr_delta_vs_scene"), diagnostics
            if compute_ssim and best_pred[2] < float(args.source_reliability_min_predicted_ssim_delta_vs_scene):
                return reject("ssim_delta_vs_scene"), diagnostics
            if best_pred[3] < float(args.source_reliability_min_predicted_lpips_delta_vs_scene):
                return reject("lpips_delta_vs_scene"), diagnostics
            if best_pred[4] < float(args.source_reliability_min_predicted_dists_delta_vs_scene):
                return reject("dists_delta_vs_scene"), diagnostics
            if (
                bool(args.source_reliability_forbid_fixed_when_scene_nonfixed)
                and selected_variant != "fixed"
                and best_variant == "fixed"
            ):
                return reject("fixed_when_scene_nonfixed"), diagnostics
            if bool(args.source_reliability_enable_ood_guard):
                ood_distance = nearest_source_distance(best_variant, vectors[best_variant], exclude_view=exclude_view)
                diagnostics["ood_distance"] = float(ood_distance)
                diagnostics["ood_threshold"] = float(ood_threshold)
                if ood_distance > ood_threshold:
                    return reject("ood_distance"), diagnostics
            return best_variant, diagnostics

        raw_predictions = {variant: [float(value) for value in prediction] for variant, prediction in predictions.items()}
        raw_variant, raw_diagnostics = evaluate_decision_predictions(
            raw_predictions,
            prediction_source="raw_prediction",
        )
        calibration_enabled = bool(calibration_payload.get("enabled", False))
        mode = str(args.source_reliability_calibrated_lcb_mode)
        if not calibration_enabled:
            raw_diagnostics["raw_incumbent_variant"] = raw_variant
            raw_diagnostics["calibrated_lcb_variant"] = None
            raw_diagnostics["final_decision_source"] = "raw_prediction"
            return raw_variant, raw_diagnostics

        lcb_predictions = {
            variant: _calibrated_lower_bounds(prediction, calibration_payload)
            for variant, prediction in predictions.items()
        }
        lcb_variant, lcb_diagnostics = evaluate_decision_predictions(
            lcb_predictions,
            prediction_source="calibrated_lower_bound",
        )
        if mode == "raw_incumbent":
            if raw_variant not in {"__scene__", "noop"} or not allow_calibrated_lcb_override:
                diagnostics = dict(raw_diagnostics)
                diagnostics.update(
                    {
                        "raw_incumbent_variant": raw_variant,
                        "raw_incumbent_diagnostics": raw_diagnostics,
                        "calibrated_lcb_variant": lcb_variant,
                        "calibrated_lcb_diagnostics": lcb_diagnostics,
                        "final_decision_source": (
                            "raw_incumbent"
                            if raw_variant not in {"__scene__", "noop"}
                            else "raw_incumbent_reject_no_lcb_override"
                        ),
                    }
                )
                return raw_variant, diagnostics
            if lcb_variant not in {"__scene__", "noop"}:
                diagnostics = dict(lcb_diagnostics)
                diagnostics.update(
                    {
                        "raw_incumbent_variant": raw_variant,
                        "raw_incumbent_diagnostics": raw_diagnostics,
                        "calibrated_lcb_variant": lcb_variant,
                        "calibrated_lcb_diagnostics": lcb_diagnostics,
                        "final_decision_source": "calibrated_lcb_override",
                    }
                )
                return lcb_variant, diagnostics
            diagnostics = dict(raw_diagnostics)
            diagnostics.update(
                {
                    "raw_incumbent_variant": raw_variant,
                    "raw_incumbent_diagnostics": raw_diagnostics,
                    "calibrated_lcb_variant": lcb_variant,
                    "calibrated_lcb_diagnostics": lcb_diagnostics,
                    "final_decision_source": "raw_incumbent_reject",
                }
            )
            return raw_variant, diagnostics
        lcb_diagnostics["raw_incumbent_variant"] = raw_variant
        lcb_diagnostics["raw_incumbent_diagnostics"] = raw_diagnostics
        lcb_diagnostics["calibrated_lcb_variant"] = lcb_variant
        lcb_diagnostics["calibrated_lcb_diagnostics"] = dict(lcb_diagnostics)
        lcb_diagnostics["final_decision_source"] = "calibrated_lower_bound"
        return lcb_variant, lcb_diagnostics

    loo_items: list[dict[str, Any]] = []
    for view, row in rows_by_view.items():
        train_examples = [example for example in examples if str(example["view"]) != str(view)]
        model = _fit_ridge_predictor(train_examples, ridge=float(args.source_reliability_ridge))
        if model is None:
            loo_items.append({"view": view, "row": row, "predictions": None, "vectors": None})
            continue
        predictions, vectors = predict_for_row(row, model)
        loo_items.append({"view": view, "row": row, "predictions": predictions, "vectors": vectors})

    calibration_errors_by_metric: list[list[float]] = [[] for _ in SOURCE_RELIABILITY_TARGET_NAMES]
    calibration_abs_errors_by_metric: list[list[float]] = [[] for _ in SOURCE_RELIABILITY_TARGET_NAMES]
    for item in loo_items:
        row = item["row"]
        predictions = item.get("predictions")
        if predictions is None:
            continue
        scene_metrics = row["candidates"][selected_variant]["metrics"]
        for variant in variants:
            if variant not in predictions:
                continue
            actual = _source_reliability_target(
                row["candidates"][variant]["metrics"],
                scene_metrics,
                compute_ssim=compute_ssim,
                args=args,
            )
            predicted = predictions[variant]
            for idx, (pred_value, actual_value) in enumerate(zip(predicted, actual)):
                overestimate = max(float(pred_value) - float(actual_value), 0.0)
                calibration_errors_by_metric[idx].append(overestimate)
                calibration_abs_errors_by_metric[idx].append(abs(float(pred_value) - float(actual_value)))
    calibration_count = min((len(values) for values in calibration_errors_by_metric), default=0)
    calibration_requested = bool(args.source_reliability_enable_calibrated_lcb)
    calibration_usable = calibration_count >= int(args.source_reliability_calibration_min_samples)
    calibration_error_bounds = [
        float(_quantile(values, float(args.source_reliability_calibration_quantile)))
        * float(args.source_reliability_calibration_scale)
        if values
        else float("inf")
        for values in calibration_errors_by_metric
    ]
    calibration_abs_error_quantiles = [
        float(_quantile(values, float(args.source_reliability_calibration_quantile))) if values else float("inf")
        for values in calibration_abs_errors_by_metric
    ]
    calibration_coverage_hits = [0 for _ in SOURCE_RELIABILITY_TARGET_NAMES]
    calibration_coverage_counts = [0 for _ in SOURCE_RELIABILITY_TARGET_NAMES]
    if calibration_usable:
        for item in loo_items:
            row = item["row"]
            predictions = item.get("predictions")
            if predictions is None:
                continue
            scene_metrics = row["candidates"][selected_variant]["metrics"]
            for variant in variants:
                if variant not in predictions:
                    continue
                actual = _source_reliability_target(
                    row["candidates"][variant]["metrics"],
                    scene_metrics,
                    compute_ssim=compute_ssim,
                    args=args,
                )
                lower = _calibrated_lower_bounds(predictions[variant], {"enabled": True, "error_bounds": calibration_error_bounds})
                for idx, (lower_value, actual_value) in enumerate(zip(lower, actual)):
                    calibration_coverage_counts[idx] += 1
                    if float(actual_value) >= float(lower_value):
                        calibration_coverage_hits[idx] += 1
    calibration_payload = {
        "enabled": bool(calibration_requested and calibration_usable),
        "requested": bool(calibration_requested),
        "usable": bool(calibration_usable),
        "target_names": list(SOURCE_RELIABILITY_TARGET_NAMES),
        "error_mode": "positive_overprediction",
        "quantile": float(args.source_reliability_calibration_quantile),
        "scale": float(args.source_reliability_calibration_scale),
        "min_samples": int(args.source_reliability_calibration_min_samples),
        "sample_count": int(calibration_count),
        "error_bounds": calibration_error_bounds,
        "abs_error_quantiles": calibration_abs_error_quantiles,
        "empirical_lcb_coverage": {
            name: (
                float(calibration_coverage_hits[idx] / max(calibration_coverage_counts[idx], 1))
                if calibration_coverage_counts[idx]
                else None
            )
            for idx, name in enumerate(SOURCE_RELIABILITY_TARGET_NAMES)
        },
        "coverage_counts": {
            name: int(calibration_coverage_counts[idx])
            for idx, name in enumerate(SOURCE_RELIABILITY_TARGET_NAMES)
        },
    }

    def evaluate_margin(
        objective_margin: float,
        *,
        allow_calibrated_lcb_override: bool = True,
    ) -> tuple[dict[str, Any], dict[str, int], list[dict[str, float]], list[dict[str, Any]]]:
        source_policy_rows: list[dict[str, float]] = []
        selected_counts: dict[str, int] = _candidate_count_dict(variants)
        prediction_log: list[dict[str, Any]] = []
        for item in loo_items:
            row = item["row"]
            predictions = item.get("predictions")
            vectors = item.get("vectors")
            if predictions is None or vectors is None:
                selected_counts["scene"] += 1
                source_policy_rows.append(row["candidates"][selected_variant]["metrics"])
                prediction_log.append({"view": item["view"], "chosen_variant": "__scene__", "predictions": None})
                continue
            chosen_variant, diagnostics = choose_from_predictions(
                predictions,
                vectors,
                objective_margin=float(objective_margin),
                exclude_view=str(item["view"]),
                allow_calibrated_lcb_override=allow_calibrated_lcb_override,
            )
            prediction_log.append(
                {
                    "view": item["view"],
                    "chosen_variant": chosen_variant,
                    "predictions": predictions,
                    "diagnostics": diagnostics,
                }
            )
            if chosen_variant == "__scene__":
                selected_counts["scene"] += 1
                source_policy_rows.append(row["candidates"][selected_variant]["metrics"])
            elif chosen_variant == "noop":
                selected_counts["noop"] += 1
                source_policy_rows.append(_noop_like(row["candidates"][selected_variant]["metrics"]))
            else:
                selected_counts[chosen_variant] += 1
                source_policy_rows.append(row["candidates"][chosen_variant]["metrics"])
        return _summarize_metric_rows(source_policy_rows, compute_ssim=compute_ssim), selected_counts, source_policy_rows, prediction_log

    selected_objective_margin = float(args.source_reliability_min_predicted_objective_delta_vs_scene)
    margin_trials: list[dict[str, Any]] = []
    if bool(args.source_reliability_auto_objective_margin):
        candidate_margins = {selected_objective_margin}
        for item in loo_items:
            predictions = item.get("predictions")
            if not predictions:
                continue
            if (
                bool(calibration_payload.get("enabled", False))
                and str(args.source_reliability_calibrated_lcb_mode) != "raw_incumbent"
            ):
                decision_predictions = {
                    variant: _calibrated_lower_bounds(prediction, calibration_payload)
                    for variant, prediction in predictions.items()
                }
            else:
                decision_predictions = {
                    variant: [float(value) for value in prediction]
                    for variant, prediction in predictions.items()
                }
            best_variant, best_pred = max(decision_predictions.items(), key=lambda pair: pair[1][0])
            if best_variant != selected_variant:
                candidate_margins.add(float(best_pred[0]))
        if candidate_margins:
            ordered = sorted(candidate_margins)
            candidate_margins.add(ordered[0] - 1.0e-9)
            candidate_margins.add(ordered[-1] + 1.0e-9)
        best_trial: dict[str, Any] | None = None
        for margin in sorted(candidate_margins):
            trial_summary, trial_counts, _, _ = evaluate_margin(
                float(margin),
                allow_calibrated_lcb_override=not (
                    bool(calibration_payload.get("enabled", False))
                    and str(args.source_reliability_calibrated_lcb_mode) == "raw_incumbent"
                ),
            )
            reject_count = int(trial_counts["noop"] + trial_counts["scene"])
            accept_fraction_trial = 1.0 - float(reject_count / max(len(loo_items), 1))
            scene_summary = selector_payload["summaries"][selected_variant]
            fixed_summary = selector_payload["summaries"]["fixed"]
            mean_delta_trial = float(trial_summary.get("psnr_gain", 0.0)) - float(scene_summary.get("psnr_gain", 0.0))
            ssim_delta_trial = (
                float(trial_summary.get("ssim_gain", 0.0)) - float(scene_summary.get("ssim_gain", 0.0))
                if compute_ssim
                else 0.0
            )
            cvar_delta_trial = _summary_psnr_tail(trial_summary, "cvar") - _summary_psnr_tail(scene_summary, "cvar")
            min_delta_trial = _summary_psnr_tail(trial_summary, "min") - _summary_psnr_tail(scene_summary, "min")
            positive_fraction_delta_trial = _positive_view_fraction(trial_summary) - _positive_view_fraction(scene_summary)
            safe_vs_fixed_trial = (
                float(trial_summary.get("psnr_gain", 0.0)) >= float(fixed_summary.get("psnr_gain", 0.0)) - float(args.selected_safe_tolerance_psnr)
                and (
                    not compute_ssim
                    or float(trial_summary.get("ssim_gain", 0.0))
                    >= float(fixed_summary.get("ssim_gain", 0.0)) - float(args.selected_safe_tolerance_ssim)
                )
            )
            objective = (
                _source_objective_from_metrics(trial_summary, compute_ssim=compute_ssim, args=args)
                + float(args.source_reliability_source_cvar_weight) * _summary_psnr_tail(trial_summary, "cvar")
                + float(args.source_reliability_source_min_weight) * _summary_psnr_tail(trial_summary, "min")
                + float(args.source_reliability_source_positive_weight) * _positive_view_fraction(trial_summary)
            )
            trial = {
                "objective_margin": float(margin),
                "objective": float(objective),
                "accept_fraction": float(accept_fraction_trial),
                "source_summary": trial_summary,
                "source_selected_counts": trial_counts,
                "source_mean_psnr_delta_vs_scene_selected": mean_delta_trial,
                "source_mean_ssim_delta_vs_scene_selected": ssim_delta_trial,
                "source_cvar_psnr_delta_vs_scene_selected": cvar_delta_trial,
                "source_min_psnr_delta_vs_scene_selected": min_delta_trial,
                "source_positive_view_fraction_delta_vs_scene_selected": positive_fraction_delta_trial,
                "source_safe_vs_fixed": bool(safe_vs_fixed_trial),
            }
            margin_trials.append(trial)
            if accept_fraction_trial < float(args.source_reliability_min_accept_fraction):
                continue
            if accept_fraction_trial > float(args.source_reliability_max_accept_fraction):
                continue
            if mean_delta_trial < float(args.source_reliability_min_source_psnr_delta):
                continue
            if compute_ssim and ssim_delta_trial < float(args.source_reliability_min_source_ssim_delta):
                continue
            if cvar_delta_trial < float(args.source_reliability_min_source_cvar_delta):
                continue
            if min_delta_trial < float(args.source_reliability_min_source_min_delta):
                continue
            if positive_fraction_delta_trial < float(args.source_reliability_min_source_positive_fraction_delta):
                continue
            if bool(args.source_reliability_require_source_safe) and not safe_vs_fixed_trial:
                continue
            if best_trial is None or float(objective) > float(best_trial["objective"]):
                best_trial = trial
        if best_trial is not None:
            selected_objective_margin = float(best_trial["objective_margin"])

    source_summary, selected_counts, _, loo_predictions = evaluate_margin(selected_objective_margin)
    fixed_summary = selector_payload["summaries"]["fixed"]
    scene_summary = selector_payload["summaries"][selected_variant]
    mean_delta = float(source_summary.get("psnr_gain", 0.0)) - float(scene_summary.get("psnr_gain", 0.0))
    ssim_delta = (
        float(source_summary.get("ssim_gain", 0.0)) - float(scene_summary.get("ssim_gain", 0.0))
        if compute_ssim
        else 0.0
    )
    cvar_delta = _summary_psnr_tail(source_summary, "cvar") - _summary_psnr_tail(scene_summary, "cvar")
    min_delta = _summary_psnr_tail(source_summary, "min") - _summary_psnr_tail(scene_summary, "min")
    positive_fraction_delta = _positive_view_fraction(source_summary) - _positive_view_fraction(scene_summary)
    safe_vs_fixed = (
        float(source_summary.get("psnr_gain", 0.0)) >= float(fixed_summary.get("psnr_gain", 0.0)) - float(args.selected_safe_tolerance_psnr)
        and (
            not compute_ssim
            or float(source_summary.get("ssim_gain", 0.0))
            >= float(fixed_summary.get("ssim_gain", 0.0)) - float(args.selected_safe_tolerance_ssim)
        )
    )
    accept_fraction = 1.0 - float((selected_counts["noop"] + selected_counts["scene"]) / max(len(rows_by_view), 1))
    base_payload = {
        "feature_schema_version": 2,
        "feature_names": feature_names,
        "variants": variants,
        "variant_blend_map": variant_blend_map,
        "ridge": float(args.source_reliability_ridge),
        "reject_variant": str(args.source_reliability_reject_variant),
        "scene_selected_variant": selected_variant,
        "source_summary": source_summary,
        "source_fixed_summary": fixed_summary,
        "source_scene_selected_summary": scene_summary,
        "source_mean_psnr_delta_vs_scene_selected": mean_delta,
        "source_mean_ssim_delta_vs_scene_selected": ssim_delta,
        "source_cvar_psnr_delta_vs_scene_selected": cvar_delta,
        "source_min_psnr_delta_vs_scene_selected": min_delta,
        "source_positive_view_fraction_delta_vs_scene_selected": positive_fraction_delta,
        "source_safe_vs_fixed": bool(safe_vs_fixed),
        "source_selected_counts": selected_counts,
        "source_accept_fraction": accept_fraction,
        "auto_objective_margin": bool(args.source_reliability_auto_objective_margin),
        "selected_objective_margin": float(selected_objective_margin),
        "objective_margin_trials": margin_trials,
        "ood_guard_enabled": bool(args.source_reliability_enable_ood_guard),
        "ood_quantile": float(args.source_reliability_ood_quantile),
        "ood_source_distance_count": int(len(ood_source_distances)),
        "ood_source_distance_threshold": float(ood_threshold),
        "ood_source_distance_summary": {
            "min": min(ood_source_distances) if ood_source_distances else None,
            "mean": _mean(ood_source_distances) if ood_source_distances else None,
            "max": max(ood_source_distances) if ood_source_distances else None,
        },
        "calibrated_lcb_mode": str(args.source_reliability_calibrated_lcb_mode),
        "source_reliability_calibration": calibration_payload,
        "fixed_scene_min_source_ssim_delta": float(args.source_reliability_fixed_scene_min_source_ssim_delta),
        "min_accept_fraction": float(args.source_reliability_min_accept_fraction),
        "max_accept_fraction": float(args.source_reliability_max_accept_fraction),
        "min_source_psnr_delta": float(args.source_reliability_min_source_psnr_delta),
        "min_source_ssim_delta": float(args.source_reliability_min_source_ssim_delta),
        "min_source_cvar_delta": float(args.source_reliability_min_source_cvar_delta),
        "min_source_min_delta": float(args.source_reliability_min_source_min_delta),
        "min_source_positive_fraction_delta": float(args.source_reliability_min_source_positive_fraction_delta),
        "min_predicted_objective_delta_vs_scene": float(selected_objective_margin),
        "min_predicted_psnr_delta_vs_scene": float(args.source_reliability_min_predicted_psnr_delta_vs_scene),
        "min_predicted_ssim_delta_vs_scene": float(args.source_reliability_min_predicted_ssim_delta_vs_scene),
        "min_predicted_lpips_delta_vs_scene": float(args.source_reliability_min_predicted_lpips_delta_vs_scene),
        "min_predicted_dists_delta_vs_scene": float(args.source_reliability_min_predicted_dists_delta_vs_scene),
        "forbid_fixed_when_scene_nonfixed": bool(args.source_reliability_forbid_fixed_when_scene_nonfixed),
        "fixed_rollback_certificate_enabled": bool(args.source_reliability_enable_fixed_rollback_certificate),
        "fixed_rollback_min_objective_margin": float(args.source_reliability_fixed_rollback_min_objective_margin),
        "fixed_rollback_min_psnr_margin": float(args.source_reliability_fixed_rollback_min_psnr_margin),
        "fixed_rollback_min_ssim_margin": float(args.source_reliability_fixed_rollback_min_ssim_margin),
        "fixed_rollback_min_best_psnr_delta": float(args.source_reliability_fixed_rollback_min_best_psnr_delta),
        "fixed_rollback_min_best_ssim_delta": float(args.source_reliability_fixed_rollback_min_best_ssim_delta),
        "fixed_rollback_max_scene_opposition_fraction": float(
            args.source_reliability_fixed_rollback_max_scene_opposition_fraction
        ),
        "fixed_rollback_min_scene_aligned_fraction": float(
            args.source_reliability_fixed_rollback_min_scene_aligned_fraction
        ),
        "loo_predictions": loo_predictions,
    }
    if (
        calibration_requested
        and not calibration_usable
        and str(args.source_reliability_calibrated_lcb_mode) != "raw_incumbent"
    ):
        return {"enabled": False, "verdict": "source reliability calibrated LCB did not have enough calibration samples", **base_payload}
    if (
        bool(args.source_reliability_enable_ood_guard)
        and len(ood_source_distances) < int(args.source_reliability_ood_min_samples)
    ):
        return {"enabled": False, "verdict": "not enough OOD calibration distances", **base_payload}
    if accept_fraction < float(args.source_reliability_min_accept_fraction):
        return {"enabled": False, "verdict": "source reliability accepted too few views", **base_payload}
    if accept_fraction > float(args.source_reliability_max_accept_fraction):
        return {"enabled": False, "verdict": "source reliability accepted too many views", **base_payload}
    if mean_delta < float(args.source_reliability_min_source_psnr_delta):
        return {"enabled": False, "verdict": "source reliability did not improve scene-selected PSNR enough", **base_payload}
    if compute_ssim and ssim_delta < float(args.source_reliability_min_source_ssim_delta):
        return {"enabled": False, "verdict": "source reliability did not clear SSIM delta", **base_payload}
    if (
        compute_ssim
        and selected_variant == "fixed"
        and ssim_delta < float(args.source_reliability_fixed_scene_min_source_ssim_delta)
    ):
        return {
            "enabled": False,
            "verdict": "source reliability did not clear fixed-scene SSIM delta",
            **base_payload,
        }
    if cvar_delta < float(args.source_reliability_min_source_cvar_delta):
        return {"enabled": False, "verdict": "source reliability did not clear CVaR tail delta", **base_payload}
    if min_delta < float(args.source_reliability_min_source_min_delta):
        return {"enabled": False, "verdict": "source reliability did not clear min-gain tail delta", **base_payload}
    if positive_fraction_delta < float(args.source_reliability_min_source_positive_fraction_delta):
        return {"enabled": False, "verdict": "source reliability did not clear positive-view fraction delta", **base_payload}
    if bool(args.source_reliability_require_source_safe) and not safe_vs_fixed:
        return {"enabled": False, "verdict": "source reliability did not clear fixed safety gate", **base_payload}
    full_model = _fit_ridge_predictor(examples, ridge=float(args.source_reliability_ridge))
    if full_model is None:
        return {"enabled": False, "verdict": "could not fit full source reliability model", **base_payload}
    return {
        "enabled": True,
        "verdict": "source-only relative reliability model selected",
        "model": full_model,
        "source_entries_by_variant": entries_by_variant,
        "ood_feature_mean": ood_mean,
        "ood_feature_std": ood_std,
        "decision_prediction_source": (
            "raw_incumbent_with_calibrated_lcb_override"
            if bool(calibration_payload.get("enabled", False))
            and str(args.source_reliability_calibrated_lcb_mode) == "raw_incumbent"
            else "calibrated_lower_bound"
            if bool(calibration_payload.get("enabled", False))
            else "raw_prediction"
        ),
        "calibrated_lcb_mode": str(args.source_reliability_calibrated_lcb_mode),
        **base_payload,
    }


def _source_reliability_choose_variant(
    proxies_by_variant: dict[str, dict[str, float]],
    policy: dict[str, Any],
    *,
    compute_ssim: bool,
) -> tuple[str, dict[str, list[float]], dict[str, Any]]:
    variants = list(policy.get("variants", BASE_CANDIDATE_VARIANTS))
    variant_blend_map = {str(k): float(v) for k, v in dict(policy.get("variant_blend_map", {})).items()}
    feature_names = list(policy["feature_names"])
    scene_variant = str(policy["scene_selected_variant"])
    scene_proxy = proxies_by_variant[scene_variant]
    predictions: dict[str, list[float]] = {}
    vectors: dict[str, list[float]] = {}
    for variant in variants:
        vector = _source_reliability_features(
            proxies_by_variant[variant],
            scene_proxy,
            variant=variant,
            scene_variant=scene_variant,
            feature_names=feature_names,
            variant_blend=variant_blend_map.get(variant),
            scene_variant_blend=variant_blend_map.get(scene_variant),
        )
        vectors[variant] = vector
        predictions[variant] = _predict_ridge(policy["model"], vector)
    calibration_payload = policy.get("source_reliability_calibration", {})

    def evaluate_decision_predictions(
        decision_predictions: dict[str, list[float]],
        *,
        prediction_source: str,
    ) -> tuple[str, dict[str, Any]]:
        best_variant, best_pred = max(decision_predictions.items(), key=lambda item: item[1][0])
        scene_pred = decision_predictions[scene_variant]
        diagnostics: dict[str, Any] = {
            "best_variant": best_variant,
            "scene_variant": scene_variant,
            "best_prediction": predictions[best_variant],
            "scene_prediction": predictions[scene_variant],
            "best_decision_prediction": best_pred,
            "scene_decision_prediction": scene_pred,
            "decision_prediction_source": prediction_source,
            "calibrated_lower_bounds": decision_predictions
            if prediction_source == "calibrated_lower_bound"
            else None,
            "reject_reason": None,
            "ood_distance": None,
            "ood_threshold": None,
        }

        def reject(reason: str) -> tuple[str, dict[str, Any]]:
            diagnostics["reject_reason"] = reason
            if str(policy.get("reject_variant", "scene")) == "scene":
                return "__scene__", diagnostics
            return "noop", diagnostics

        def fixed_rollback_certificate() -> dict[str, Any]:
            scene_consistency = proxies_by_variant.get(scene_variant, {})
            payload = {
                "enabled": bool(policy.get("fixed_rollback_certificate_enabled", False)),
                "accepted": False,
                "reason": "disabled",
                "objective_margin": float(best_pred[0]) - float(scene_pred[0]),
                "psnr_margin": float(best_pred[1]) - float(scene_pred[1]),
                "ssim_margin": float(best_pred[2]) - float(scene_pred[2]) if compute_ssim else 0.0,
                "best_psnr_delta": float(best_pred[1]),
                "best_ssim_delta": float(best_pred[2]) if compute_ssim else 0.0,
                "scene_opposition_fraction": float(scene_consistency.get("opposition_fraction", 0.0)),
                "scene_aligned_fraction": float(scene_consistency.get("aligned_fraction", 0.0)),
            }
            if not bool(payload["enabled"]):
                return payload
            if best_variant != "fixed" or scene_variant == "fixed":
                payload["reason"] = "not_fixed_rollback"
                return payload
            if payload["objective_margin"] < float(policy.get("fixed_rollback_min_objective_margin", 0.0)):
                payload["reason"] = "objective_margin"
                return payload
            if payload["psnr_margin"] < float(policy.get("fixed_rollback_min_psnr_margin", 0.0)):
                payload["reason"] = "psnr_margin"
                return payload
            if compute_ssim and payload["ssim_margin"] < float(policy.get("fixed_rollback_min_ssim_margin", 0.0)):
                payload["reason"] = "ssim_margin"
                return payload
            if payload["best_psnr_delta"] < float(policy.get("fixed_rollback_min_best_psnr_delta", 0.0)):
                payload["reason"] = "best_psnr_delta"
                return payload
            if compute_ssim and payload["best_ssim_delta"] < float(policy.get("fixed_rollback_min_best_ssim_delta", 0.0)):
                payload["reason"] = "best_ssim_delta"
                return payload
            if payload["scene_opposition_fraction"] > float(
                policy.get("fixed_rollback_max_scene_opposition_fraction", 1.0)
            ):
                payload["reason"] = "scene_opposition_fraction"
                return payload
            if payload["scene_aligned_fraction"] < float(policy.get("fixed_rollback_min_scene_aligned_fraction", 0.0)):
                payload["reason"] = "scene_aligned_fraction"
                return payload
            payload["accepted"] = True
            payload["reason"] = "passed"
            return payload

        if best_pred[0] < float(policy.get("min_predicted_objective_delta_vs_scene", 0.0)):
            return reject("objective_margin")
        if best_pred[1] < float(policy.get("min_predicted_psnr_delta_vs_scene", -1.0e9)):
            return reject("psnr_delta_vs_scene")
        if compute_ssim and best_pred[2] < float(policy.get("min_predicted_ssim_delta_vs_scene", -1.0e9)):
            return reject("ssim_delta_vs_scene")
        if best_pred[3] < float(policy.get("min_predicted_lpips_delta_vs_scene", -1.0e9)):
            return reject("lpips_delta_vs_scene")
        if best_pred[4] < float(policy.get("min_predicted_dists_delta_vs_scene", -1.0e9)):
            return reject("dists_delta_vs_scene")
        if bool(policy.get("forbid_fixed_when_scene_nonfixed", False)) and scene_variant != "fixed" and best_variant == "fixed":
            rollback_payload = fixed_rollback_certificate()
            diagnostics["fixed_rollback_certificate"] = rollback_payload
            if not bool(rollback_payload.get("accepted", False)):
                return reject("fixed_when_scene_nonfixed")
        if bool(policy.get("ood_guard_enabled", False)):
            pool = policy.get("source_entries_by_variant", {}).get(best_variant, [])
            mean = [float(x) for x in policy.get("ood_feature_mean", [])]
            std = [float(x) for x in policy.get("ood_feature_std", [])]
            if pool and mean and std:
                ood_distance = min(_normalized_distance(vectors[best_variant], entry["vector"], mean, std) for entry in pool)
            else:
                ood_distance = float("inf")
            ood_threshold = float(policy.get("ood_source_distance_threshold", float("inf")))
            diagnostics["ood_distance"] = float(ood_distance)
            diagnostics["ood_threshold"] = float(ood_threshold)
            if ood_distance > ood_threshold:
                return reject("ood_distance")
        return best_variant, diagnostics

    raw_predictions = {variant: [float(value) for value in prediction] for variant, prediction in predictions.items()}
    raw_variant, raw_diagnostics = evaluate_decision_predictions(
        raw_predictions,
        prediction_source="raw_prediction",
    )
    calibration_enabled = bool(calibration_payload.get("enabled", False))
    mode = str(policy.get("calibrated_lcb_mode", "calibrated_lcb"))
    if not calibration_enabled:
        raw_diagnostics["raw_incumbent_variant"] = raw_variant
        raw_diagnostics["calibrated_lcb_variant"] = None
        raw_diagnostics["final_decision_source"] = "raw_prediction"
        return raw_variant, predictions, raw_diagnostics

    lcb_predictions = {
        variant: _calibrated_lower_bounds(prediction, calibration_payload)
        for variant, prediction in predictions.items()
    }
    lcb_variant, lcb_diagnostics = evaluate_decision_predictions(
        lcb_predictions,
        prediction_source="calibrated_lower_bound",
    )
    if mode == "raw_incumbent":
        if raw_variant not in {"__scene__", "noop"}:
            diagnostics = dict(raw_diagnostics)
            diagnostics.update(
                {
                    "raw_incumbent_variant": raw_variant,
                    "raw_incumbent_diagnostics": raw_diagnostics,
                    "calibrated_lcb_variant": lcb_variant,
                    "calibrated_lcb_diagnostics": lcb_diagnostics,
                    "final_decision_source": "raw_incumbent",
                }
            )
            return raw_variant, predictions, diagnostics
        if lcb_variant not in {"__scene__", "noop"}:
            diagnostics = dict(lcb_diagnostics)
            diagnostics.update(
                {
                    "raw_incumbent_variant": raw_variant,
                    "raw_incumbent_diagnostics": raw_diagnostics,
                    "calibrated_lcb_variant": lcb_variant,
                    "calibrated_lcb_diagnostics": lcb_diagnostics,
                    "final_decision_source": "calibrated_lcb_override",
                }
            )
            return lcb_variant, predictions, diagnostics
        diagnostics = dict(raw_diagnostics)
        diagnostics.update(
            {
                "raw_incumbent_variant": raw_variant,
                "raw_incumbent_diagnostics": raw_diagnostics,
                "calibrated_lcb_variant": lcb_variant,
                "calibrated_lcb_diagnostics": lcb_diagnostics,
                "final_decision_source": "raw_incumbent_reject",
            }
        )
        return raw_variant, predictions, diagnostics
    lcb_diagnostics["raw_incumbent_variant"] = raw_variant
    lcb_diagnostics["raw_incumbent_diagnostics"] = raw_diagnostics
    lcb_diagnostics["calibrated_lcb_variant"] = lcb_variant
    lcb_diagnostics["calibrated_lcb_diagnostics"] = dict(lcb_diagnostics)
    lcb_diagnostics["final_decision_source"] = "calibrated_lower_bound"
    return lcb_variant, predictions, lcb_diagnostics


def _gate_accepts(proxy: dict[str, float], gate: dict[str, Any] | None) -> bool:
    if not gate or not bool(gate.get("enabled", False)):
        return True
    score = _score_from_proxy(proxy, str(gate["score_name"]))
    threshold = float(gate["threshold"])
    polarity = str(gate.get("polarity", "ge"))
    return score >= threshold if polarity == "ge" else score <= threshold


def _fit_per_view_gate(
    selector_payload: dict[str, Any] | None,
    selected_variant: str,
    *,
    compute_ssim: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not bool(args.enable_per_view_risk_gate):
        return {"enabled": False, "verdict": "disabled by CLI"}
    if selector_payload is None or not selector_payload.get("per_view"):
        return {"enabled": False, "verdict": "missing source-heldout selector per-view evidence"}
    records = []
    for row in selector_payload["per_view"]:
        candidate = row["candidates"][selected_variant]
        records.append({"metrics": candidate["metrics"], "proxy": candidate["proxy"]})
    if len(records) < 2:
        return {"enabled": False, "verdict": "not enough source-heldout views to fit gate"}

    ungated_rows = [r["metrics"] for r in records]
    ungated_summary = _summarize_rows(ungated_rows, compute_ssim=compute_ssim)
    score_names = _parse_score_names(args.per_view_gate_score_grid)
    best: dict[str, Any] | None = None
    evaluated: list[dict[str, Any]] = []
    for score_name in score_names:
        try:
            scores = [_score_from_proxy(r["proxy"], score_name) for r in records]
        except KeyError:
            continue
        thresholds = sorted(set(scores))
        if not thresholds:
            continue
        for polarity in ["ge", "le"]:
            for threshold in thresholds:
                gated_rows = [
                    r["metrics"] if ((score >= threshold) if polarity == "ge" else (score <= threshold)) else _noop_like(r["metrics"])
                    for r, score in zip(records, scores)
                ]
                accept_fraction = _mean(
                    [1.0 if ((score >= threshold) if polarity == "ge" else (score <= threshold)) else 0.0 for score in scores]
                )
                if accept_fraction < float(args.per_view_gate_min_accept_fraction):
                    continue
                if accept_fraction > float(args.per_view_gate_max_accept_fraction):
                    continue
                summary = _summarize_rows(gated_rows, compute_ssim=compute_ssim)
                mean_delta = float(summary.get("psnr_gain", 0.0)) - float(ungated_summary.get("psnr_gain", 0.0))
                cvar_delta = float(summary["psnr_gain_tail"]["cvar"]) - float(ungated_summary["psnr_gain_tail"]["cvar"])
                min_delta = float(summary["psnr_gain_tail"]["min"]) - float(ungated_summary["psnr_gain_tail"]["min"])
                ssim_delta = (
                    float(summary.get("ssim_gain", 0.0)) - float(ungated_summary.get("ssim_gain", 0.0))
                    if compute_ssim
                    else 0.0
                )
                if mean_delta < float(args.per_view_gate_min_mean_psnr_delta):
                    continue
                if cvar_delta < float(args.per_view_gate_min_cvar_psnr_delta):
                    continue
                if compute_ssim and ssim_delta < float(args.per_view_gate_min_mean_ssim_delta):
                    continue
                objective = (
                    float(summary.get("psnr_gain", 0.0))
                    + 20.0 * (float(summary.get("ssim_gain", 0.0)) if compute_ssim else 0.0)
                    + float(args.per_view_gate_cvar_weight) * float(summary["psnr_gain_tail"]["cvar"])
                    + float(args.per_view_gate_min_weight) * float(summary["psnr_gain_tail"]["min"])
                )
                trial = {
                    "score_name": score_name,
                    "threshold": float(threshold),
                    "polarity": polarity,
                    "source_accept_fraction": accept_fraction,
                    "source_summary": summary,
                    "source_ungated_summary": ungated_summary,
                    "source_mean_psnr_delta": mean_delta,
                    "source_cvar_psnr_delta": cvar_delta,
                    "source_min_psnr_delta": min_delta,
                    "source_mean_ssim_delta": ssim_delta,
                    "objective": objective,
                }
                evaluated.append(trial)
                if best is None or objective > float(best["objective"]):
                    best = trial
    if best is None:
        return {
            "enabled": False,
            "selected_variant": selected_variant,
            "source_ungated_summary": ungated_summary,
            "evaluated_trials": len(evaluated),
            "verdict": "no source-heldout per-view gate cleared the configured safety constraints",
        }
    best["enabled"] = True
    best["selected_variant"] = selected_variant
    best["evaluated_trials"] = len(evaluated)
    best["verdict"] = "source-heldout per-view risk gate selected"
    return best


def _candidate_passes_guard(
    row: dict[str, Any],
    fixed: dict[str, Any],
    *,
    compute_ssim: bool,
    min_psnr_delta: float,
    min_ssim_delta: float,
    min_lpips_delta: float,
    min_dists_delta: float,
) -> bool:
    if float(row.get("psnr_gain", 0.0)) <= 0.0:
        return False
    if float(row.get("psnr_gain", 0.0)) - float(fixed.get("psnr_gain", 0.0)) < float(min_psnr_delta):
        return False
    if compute_ssim:
        if float(row.get("ssim_gain", 0.0)) <= 0.0:
            return False
        if float(row.get("ssim_gain", 0.0)) - float(fixed.get("ssim_gain", 0.0)) < float(min_ssim_delta):
            return False
    if "lpips_gain" in row and "lpips_gain" in fixed:
        if float(row.get("lpips_gain", 0.0)) - float(fixed.get("lpips_gain", 0.0)) < float(min_lpips_delta):
            return False
    if "dists_gain" in row and "dists_gain" in fixed:
        if float(row.get("dists_gain", 0.0)) - float(fixed.get("dists_gain", 0.0)) < float(min_dists_delta):
            return False
    return True


def _select_variant_from_source_heldout(
    *,
    model: SupportTransportCalibrator,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    train_frames: list[Any],
    source_frames: list[Any],
    loader: FrameLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    _, heldout_frames = _split_source_heldout(train_frames, int(args.heldout_stride), int(args.heldout_offset))
    _, selector_val_frames = _split_calibrator_train_val(
        heldout_frames,
        val_stride=int(args.selector_val_stride),
        val_offset=int(args.selector_val_offset),
    )
    if int(args.max_selector_views) > 0:
        selector_val_frames = selector_val_frames[: int(args.max_selector_views)]

    candidate_variants = _candidate_variant_names(args)
    variant_blend_map = _candidate_variant_blend_map(args)
    candidate_rows: dict[str, list[dict[str, float]]] = {variant: [] for variant in candidate_variants}
    per_view: list[dict[str, Any]] = []
    source_perceptual_models = _load_source_perceptual_models(device, args)
    model.eval()
    with torch.no_grad():
        for target in tqdm(selector_val_frames, desc="source-heldout output selector"):
            ev = compute_evidence_signal(
                target,
                source_frames,
                k=int(args.k),
                mode="residual",
                residual_clip=float(args.residual_clip),
                min_confidence=float(args.min_confidence),
                depth_abs_tol=float(args.depth_abs_tol),
                depth_rel_tol=float(args.depth_rel_tol),
                direction_weight=float(args.direction_weight),
                evidence_max_side=int(args.evidence_max_side),
                loader=loader,
                device=device,
            )
            features = _build_features(ev, k=int(args.k)).unsqueeze(0).to(device=device, dtype=torch.float32)
            signal = ev.signal.unsqueeze(0).to(device=device, dtype=torch.float32)
            valid = ev.valid.unsqueeze(0).to(device=device, dtype=torch.float32)
            pred_delta = model(_normalize(features, feature_mean, feature_std), signal, valid).squeeze(0)
            gt = loader.gt(str(target.gt_path)).to(device=device, dtype=torch.float32)
            per_view_candidates: dict[str, Any] = {}
            for variant, delta in _candidate_deltas(ev, pred_delta, args).items():
                row = _image_metrics(
                    ev.base,
                    gt,
                    delta,
                    compute_ssim=bool(args.compute_ssim),
                    ssim_max_side=int(args.ssim_max_side),
                )
                _augment_source_perceptual_metrics(
                    row,
                    base=ev.base,
                    gt=gt,
                    delta=delta,
                    models=source_perceptual_models,
                    max_side=int(args.source_perceptual_max_side),
                )
                row["view"] = target.name
                candidate_rows[variant].append(row)
                per_view_candidates[variant] = {
                    "metrics": row,
                    "proxy": _candidate_proxy_stats(ev, delta),
                }
            per_view.append(
                {
                    "view": target.name,
                    "covered_fraction": float(ev.valid.to(torch.float32).mean().detach().cpu().item()),
                    "support_names": list(ev.support_names),
                    "candidates": per_view_candidates,
                }
            )

    summaries = {
        variant: _summarize_metric_rows(rows, compute_ssim=bool(args.compute_ssim))
        for variant, rows in candidate_rows.items()
    }
    fixed_summary = summaries["fixed"]
    passing = [
        variant
        for variant in candidate_variants
        if variant != "fixed"
        if _candidate_passes_guard(
            summaries[variant],
            fixed_summary,
            compute_ssim=bool(args.compute_ssim),
            min_psnr_delta=float(args.selector_min_vs_fixed_psnr_delta),
            min_ssim_delta=float(args.selector_min_vs_fixed_ssim_delta),
            min_lpips_delta=float(args.selector_min_vs_fixed_lpips_delta),
            min_dists_delta=float(args.selector_min_vs_fixed_dists_delta),
        )
    ]
    if passing:
        selected_variant = max(
            passing,
            key=lambda name: _source_objective_from_metrics(summaries[name], compute_ssim=bool(args.compute_ssim), args=args),
        )
        verdict = "source-heldout selector found a non-fixed candidate that beats fixed on the guard axes"
    else:
        selected_variant = "fixed"
        verdict = "source-heldout selector fell back to fixed because non-fixed candidates did not clear the guard"
    return {
        "selected_variant": selected_variant,
        "candidate_variants": candidate_variants,
        "candidate_variant_blend_map": variant_blend_map,
        "val_views": int(len(selector_val_frames)),
        "summaries": summaries,
        "per_view": per_view,
        "verdict": verdict,
        "min_vs_fixed_psnr_delta": float(args.selector_min_vs_fixed_psnr_delta),
        "min_vs_fixed_ssim_delta": float(args.selector_min_vs_fixed_ssim_delta),
        "min_vs_fixed_lpips_delta": float(args.selector_min_vs_fixed_lpips_delta),
        "min_vs_fixed_dists_delta": float(args.selector_min_vs_fixed_dists_delta),
        "source_perceptual": {
            "enabled": bool(args.compute_source_perceptual),
            "max_side": int(args.source_perceptual_max_side),
            "lpips_status": "computed_lpips_alex" if source_perceptual_models.get("lpips") is not None else "not_computed",
            "dists_status": str(source_perceptual_models.get("dists_status", "not_computed")),
            "objective_lpips_weight": float(args.source_objective_lpips_weight),
            "objective_dists_weight": float(args.source_objective_dists_weight),
        },
    }


def _apply_policy_profile(args: argparse.Namespace) -> None:
    profile = str(getattr(args, "policy_profile", "none"))
    if profile == "none":
        return
    if profile != "v322c_incumbent":
        raise ValueError(f"unsupported policy profile: {profile}")

    args.output_variant = "source_heldout_auto"
    args.enable_candidate_ladder = True
    args.candidate_ladder_blends = "0.25,0.75"
    args.enable_per_view_knn_policy = True
    args.per_view_knn_base_variants_only = True
    args.per_view_knn_reject_variant = "scene"
    args.per_view_knn_forbid_fixed_when_scene_nonfixed = True
    args.per_view_knn_min_score_delta_vs_scene = 5.0e-4

    args.enable_source_reliability_policy = True
    args.source_reliability_reject_variant = "scene"
    args.source_reliability_min_predicted_objective_delta_vs_scene = -1.0e-9
    args.source_reliability_min_predicted_psnr_delta_vs_scene = 0.0
    args.source_reliability_min_predicted_ssim_delta_vs_scene = -2.0e-4
    args.source_reliability_enable_calibrated_lcb = True
    args.source_reliability_calibrated_lcb_mode = "raw_incumbent"
    args.source_reliability_calibration_quantile = 0.5
    args.source_reliability_calibration_scale = 0.5
    args.source_reliability_enable_ood_guard = True
    args.source_reliability_ood_quantile = 0.8
    args.source_reliability_min_source_min_delta = 0.0
    args.source_reliability_fixed_scene_min_source_ssim_delta = -2.0e-5
    args.source_reliability_forbid_fixed_when_scene_nonfixed = True

    args.evidence_max_side = 256
    args.compute_ssim = True
    args.ssim_max_side = 256


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    base_model = Path(args.base_model_path)
    base_method = str(args.base_method_name)
    output_dir = Path(args.output_dir)
    render_dir = output_dir / "renders"
    gt_dir = output_dir / "gt"
    visual_dir = output_dir / "visuals"
    render_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.copy_gt):
        gt_dir.mkdir(parents=True, exist_ok=True)

    model, feature_mean, feature_std, train_config = _load_model(Path(args.checkpoint), device)
    train_frames = load_split_frames(base_model, "train", base_method)
    if str(args.support_source_mode) == "source_split":
        source_frames, _ = _split_source_heldout(train_frames, int(args.heldout_stride), int(args.heldout_offset))
    elif str(args.support_source_mode) == "all_train":
        source_frames = train_frames
    else:
        raise ValueError(f"unsupported support_source_mode: {args.support_source_mode}")
    target_frames = load_split_frames(base_model, str(args.target_split), base_method)
    if int(args.max_target_views) > 0:
        target_frames = target_frames[: int(args.max_target_views)]

    loader = FrameLoader(device=device)
    candidate_variants = _candidate_variant_names(args)
    candidate_variant_blend_map = _candidate_variant_blend_map(args)
    valid_output_variants = set(candidate_variants) | {"source_heldout_auto"}
    if str(args.output_variant) not in valid_output_variants:
        raise ValueError(
            f"unsupported output_variant {args.output_variant!r}; valid variants are {sorted(valid_output_variants)}"
        )
    selector_payload: dict[str, Any] | None = None
    selected_variant = str(args.output_variant)
    if selected_variant == "source_heldout_auto":
        selector_payload = _select_variant_from_source_heldout(
            model=model,
            feature_mean=feature_mean,
            feature_std=feature_std,
            train_frames=train_frames,
            source_frames=source_frames,
            loader=loader,
            device=device,
            args=args,
        )
        selected_variant = str(selector_payload["selected_variant"])
    per_view_gate_payload = _fit_per_view_gate(
        selector_payload,
        selected_variant,
        compute_ssim=bool(args.compute_ssim),
        args=args,
    )
    per_view_knn_payload = _fit_per_view_knn_policy(
        selector_payload,
        compute_ssim=bool(args.compute_ssim),
        args=args,
    )
    per_view_risk_model_payload = _fit_per_view_risk_model_policy(
        selector_payload,
        compute_ssim=bool(args.compute_ssim),
        args=args,
    )
    source_reliability_payload = _fit_source_reliability_policy(
        selector_payload,
        compute_ssim=bool(args.compute_ssim),
        args=args,
    )
    local_support_payload = _fit_local_support_policy(
        selector_payload,
        compute_ssim=bool(args.compute_ssim),
        args=args,
        incumbent_policy_payload=source_reliability_payload,
    )
    pairwise_dominance_payload = _fit_pairwise_dominance_policy(
        selector_payload,
        source_reliability_payload=source_reliability_payload,
        compute_ssim=bool(args.compute_ssim),
        args=args,
    )
    promotion_rollback_payload = _fit_promotion_rollback_policy(
        pairwise_dominance_payload,
        compute_ssim=bool(args.compute_ssim),
        args=args,
    )
    fixed_rows: list[dict[str, float]] = []
    learned_rows: list[dict[str, float]] = []
    hybrid_rows: list[dict[str, float]] = []
    candidate_rows_by_variant: dict[str, list[dict[str, float]]] = {variant: [] for variant in candidate_variants}
    selected_rows: list[dict[str, float]] = []
    per_view: list[dict[str, Any]] = []
    nooped_views = 0
    promotion_rollback_stats = {
        "considered_count": 0,
        "keep_count": 0,
        "shadow_rollback_count": 0,
        "rollback_count": 0,
        "reason_counts": {},
    }
    target_neighbor_consistency_stats = {
        "considered_count": 0,
        "keep_count": 0,
        "shadow_rollback_count": 0,
        "rollback_count": 0,
        "reason_counts": {},
    }
    target_neighbor_candidate_unlock_stats = {
        "promote_count": 0,
        "keep_count": 0,
        "skipped_count": 0,
        "reason_counts": {},
    }

    for idx, target in enumerate(tqdm(target_frames, desc="apply support-transport calibrator")):
        with torch.no_grad():
            ev = compute_evidence_signal(
                target,
                source_frames,
                k=int(args.k),
                mode="residual",
                residual_clip=float(args.residual_clip),
                min_confidence=float(args.min_confidence),
                depth_abs_tol=float(args.depth_abs_tol),
                depth_rel_tol=float(args.depth_rel_tol),
                direction_weight=float(args.direction_weight),
                evidence_max_side=int(args.evidence_max_side),
                loader=loader,
                device=device,
            )
            features = _build_features(ev, k=int(args.k)).unsqueeze(0).to(device=device, dtype=torch.float32)
            signal = ev.signal.unsqueeze(0).to(device=device, dtype=torch.float32)
            valid = ev.valid.unsqueeze(0).to(device=device, dtype=torch.float32)
            pred_delta = model(_normalize(features, feature_mean, feature_std), signal, valid).squeeze(0)
            deltas = _candidate_deltas(ev, pred_delta, args)
            fixed_delta = deltas["fixed"]
            learned_delta = deltas["learned"]
            hybrid_delta = deltas["hybrid"]
            proxies_by_variant = {variant: _candidate_proxy_stats(ev, delta) for variant, delta in deltas.items()}
            selected_delta = deltas[selected_variant]
            selected_proxy = proxies_by_variant[selected_variant]
            knn_predictions: dict[str, float] | None = None
            knn_diagnostics: dict[str, Any] | None = None
            risk_model_predictions: dict[str, list[float]] | None = None
            risk_model_diagnostics: dict[str, Any] | None = None
            local_support_diagnostics: dict[str, Any] | None = None
            pairwise_dominance_predictions: dict[str, list[float]] | None = None
            pairwise_dominance_diagnostics: dict[str, Any] | None = None
            source_reliability_predictions: dict[str, list[float]] | None = None
            source_reliability_diagnostics: dict[str, Any] | None = None
            gate_accepted = True
            output_variant = selected_variant
            selected_proxy_variant = selected_variant
            per_view_knn_rejected = False
            per_view_risk_model_rejected = False
            local_support_rejected = False
            source_reliability_rejected = False
            risk_model_raw_output_variant: str | None = None
            local_support_raw_output_variant: str | None = None
            pairwise_dominance_raw_output_variant: str | None = None
            source_reliability_raw_output_variant: str | None = None
            risk_model_decision = "not_used"
            local_support_decision = "not_used"
            pairwise_dominance_decision = "not_used"
            source_reliability_decision = "not_used"
            local_support_claimed = False
            pairwise_dominance_claimed = False
            source_reliability_claimed = False
            output_decision_source = "scene"
            promotion_incumbent_variant = selected_variant
            promotion_rollback_diagnostics: dict[str, Any] | None = None
            target_neighbor_consistency_diagnostics: dict[str, Any] | None = None
            target_neighbor_candidate_unlock_diagnostics: dict[str, Any] | None = None
            if bool(local_support_payload.get("enabled", False)) and not bool(args.local_support_post_incumbent_fallback_only):
                output_variant, local_support_diagnostics = _local_support_choose_variant(
                    proxies_by_variant,
                    local_support_payload,
                    compute_ssim=bool(args.compute_ssim),
                )
                local_support_raw_output_variant = output_variant
                if output_variant == "__scene__":
                    local_support_rejected = True
                    local_support_decision = "incumbent_fallback"
                    output_variant = selected_variant
                    selected_delta = deltas[selected_variant]
                    selected_proxy = proxies_by_variant[selected_variant]
                    selected_proxy_variant = selected_variant
                    gate_accepted = True
                else:
                    local_support_claimed = True
                    if output_variant == "noop":
                        selected_delta = torch.zeros_like(selected_delta)
                        selected_proxy = _candidate_proxy_stats(ev, selected_delta)
                        selected_proxy_variant = "noop"
                        local_support_decision = "noop"
                    else:
                        selected_delta = deltas[output_variant]
                        selected_proxy = proxies_by_variant[output_variant]
                        selected_proxy_variant = output_variant
                        local_support_decision = "candidate"
                        output_decision_source = "local_support"
                    gate_accepted = output_variant != "noop"
            if (not local_support_claimed) and bool(source_reliability_payload.get("enabled", False)):
                output_variant, source_reliability_predictions, source_reliability_diagnostics = _source_reliability_choose_variant(
                    proxies_by_variant,
                    source_reliability_payload,
                    compute_ssim=bool(args.compute_ssim),
                )
                source_reliability_raw_output_variant = output_variant
                if output_variant == "__scene__":
                    source_reliability_rejected = True
                    source_reliability_decision = "incumbent_fallback"
                    output_variant = selected_variant
                    selected_delta = deltas[selected_variant]
                    selected_proxy = proxies_by_variant[selected_variant]
                    selected_proxy_variant = selected_variant
                    gate_accepted = True
                else:
                    source_reliability_claimed = True
                    if output_variant == "noop":
                        selected_delta = torch.zeros_like(selected_delta)
                        selected_proxy = _candidate_proxy_stats(ev, selected_delta)
                        selected_proxy_variant = "noop"
                        source_reliability_decision = "noop"
                    else:
                        selected_delta = deltas[output_variant]
                        selected_proxy = proxies_by_variant[output_variant]
                        selected_proxy_variant = output_variant
                        source_reliability_decision = "candidate"
                        output_decision_source = "source_reliability"
                    gate_accepted = output_variant != "noop"
            if (not local_support_claimed) and (not source_reliability_claimed) and bool(per_view_risk_model_payload.get("enabled", False)):
                output_variant, risk_model_predictions, risk_model_diagnostics = _risk_model_choose_variant(
                    proxies_by_variant,
                    per_view_risk_model_payload,
                    compute_ssim=bool(args.compute_ssim),
                )
                risk_model_raw_output_variant = output_variant
                if output_variant == "__scene__":
                    per_view_risk_model_rejected = True
                    risk_model_decision = "scene_fallback"
                    output_variant = selected_variant
                    selected_delta = deltas[selected_variant]
                    selected_proxy = proxies_by_variant[selected_variant]
                    selected_proxy_variant = selected_variant
                    gate_accepted = True
                else:
                    if output_variant == "noop":
                        selected_delta = torch.zeros_like(selected_delta)
                        selected_proxy = _candidate_proxy_stats(ev, selected_delta)
                        selected_proxy_variant = "noop"
                        risk_model_decision = "noop"
                    else:
                        selected_delta = deltas[output_variant]
                        selected_proxy = proxies_by_variant[output_variant]
                        selected_proxy_variant = output_variant
                        risk_model_decision = "candidate"
                        output_decision_source = "risk_model"
                    gate_accepted = output_variant != "noop"
            elif (not local_support_claimed) and (not source_reliability_claimed) and bool(per_view_knn_payload.get("enabled", False)):
                output_variant, knn_predictions, knn_diagnostics = _knn_choose_variant(
                    proxies_by_variant,
                    per_view_knn_payload,
                    compute_ssim=bool(args.compute_ssim),
                )
                if output_variant == "__scene__":
                    per_view_knn_rejected = True
                    output_variant = selected_variant
                    selected_delta = deltas[selected_variant]
                    selected_proxy = proxies_by_variant[selected_variant]
                    selected_proxy_variant = selected_variant
                    gate_accepted = True
                else:
                    if output_variant == "noop":
                        selected_delta = torch.zeros_like(selected_delta)
                        selected_proxy = _candidate_proxy_stats(ev, selected_delta)
                        selected_proxy_variant = "noop"
                    else:
                        selected_delta = deltas[output_variant]
                        selected_proxy = proxies_by_variant[output_variant]
                        selected_proxy_variant = output_variant
                        output_decision_source = "knn"
                    gate_accepted = output_variant != "noop"
            elif (not local_support_claimed) and (not source_reliability_claimed) and bool(per_view_gate_payload.get("enabled", False)):
                gate_accepted = _gate_accepts(selected_proxy, per_view_gate_payload)
            if (
                bool(pairwise_dominance_payload.get("enabled", False))
                and output_variant != "noop"
                and gate_accepted
                and output_variant in deltas
            ):
                incumbent_variant = output_variant
                promotion_incumbent_variant = incumbent_variant
                (
                    pairwise_output_variant,
                    pairwise_dominance_predictions,
                    pairwise_dominance_diagnostics,
                ) = _pairwise_dominance_choose_variant(
                    proxies_by_variant,
                    incumbent_variant,
                    pairwise_dominance_payload,
                    compute_ssim=bool(args.compute_ssim),
                    args=args,
                )
                pairwise_dominance_raw_output_variant = pairwise_output_variant
                if pairwise_output_variant == "__incumbent__":
                    pairwise_dominance_decision = "incumbent_fallback"
                else:
                    pairwise_dominance_claimed = True
                    output_variant = pairwise_output_variant
                    selected_delta = deltas[output_variant]
                    selected_proxy = proxies_by_variant[output_variant]
                    selected_proxy_variant = output_variant
                    pairwise_dominance_decision = "candidate"
                    output_decision_source = "pairwise"
            if (
                (not local_support_claimed)
                and (not pairwise_dominance_claimed)
                and bool(args.local_support_post_incumbent_fallback_only)
                and bool(local_support_payload.get("enabled", False))
            ):
                if output_variant == selected_variant and gate_accepted:
                    output_variant, local_support_diagnostics = _local_support_choose_variant(
                        proxies_by_variant,
                        local_support_payload,
                        compute_ssim=bool(args.compute_ssim),
                    )
                    local_support_raw_output_variant = output_variant
                    if output_variant == "__scene__":
                        local_support_rejected = True
                        local_support_decision = "incumbent_fallback"
                        output_variant = selected_variant
                        selected_delta = deltas[selected_variant]
                        selected_proxy = proxies_by_variant[selected_variant]
                        selected_proxy_variant = selected_variant
                        gate_accepted = True
                    else:
                        local_support_claimed = True
                        if output_variant == "noop":
                            selected_delta = torch.zeros_like(selected_delta)
                            selected_proxy = _candidate_proxy_stats(ev, selected_delta)
                            selected_proxy_variant = "noop"
                            local_support_decision = "noop"
                        else:
                            selected_delta = deltas[output_variant]
                            selected_proxy = proxies_by_variant[output_variant]
                            selected_proxy_variant = output_variant
                            local_support_decision = "candidate"
                            output_decision_source = "local_support"
                        gate_accepted = output_variant != "noop"
                elif local_support_decision == "not_used":
                    local_support_decision = "skipped_incumbent_already_refined"
            promotion_rollback_diagnostics = _promotion_rollback_decision(
                output_variant=output_variant,
                incumbent_variant=promotion_incumbent_variant,
                decision_source=output_decision_source,
                pairwise_diagnostics=pairwise_dominance_diagnostics,
                policy=promotion_rollback_payload,
                compute_ssim=bool(args.compute_ssim),
            )
            if bool(promotion_rollback_diagnostics.get("enabled", False)):
                decision = str(promotion_rollback_diagnostics.get("decision", "keep"))
                if decision != "keep":
                    promotion_rollback_stats["considered_count"] += 1
                if decision == "rollback":
                    promotion_rollback_stats["rollback_count"] += 1
                elif decision == "shadow_rollback":
                    promotion_rollback_stats["shadow_rollback_count"] += 1
                else:
                    promotion_rollback_stats["keep_count"] += 1
                reason = str(promotion_rollback_diagnostics.get("reject_reason") or "passed")
                reason_counts = promotion_rollback_stats["reason_counts"]
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
            if bool(promotion_rollback_diagnostics.get("rollback_applied", False)):
                output_variant = promotion_incumbent_variant
                selected_delta = deltas[output_variant]
                selected_proxy = proxies_by_variant[output_variant]
                selected_proxy_variant = output_variant
                gate_accepted = True
            target_neighbor_consistency_diagnostics = _target_neighbor_consistency_decision(
                output_variant=output_variant,
                incumbent_variant=promotion_incumbent_variant,
                decision_source=output_decision_source,
                pairwise_diagnostics=pairwise_dominance_diagnostics,
                ev=ev,
                deltas=deltas,
                target=target,
                target_frames=target_frames,
                loader=loader,
                device=device,
                args=args,
            )
            if bool(target_neighbor_consistency_diagnostics.get("enabled", False)):
                decision = str(target_neighbor_consistency_diagnostics.get("decision", "keep"))
                reason = str(target_neighbor_consistency_diagnostics.get("reject_reason") or "passed")
                actionable = reason not in {"disabled", "no_promotion", "decision_source_not_checked"}
                if actionable:
                    target_neighbor_consistency_stats["considered_count"] += 1
                if decision == "rollback":
                    target_neighbor_consistency_stats["rollback_count"] += 1
                elif decision == "shadow_rollback":
                    target_neighbor_consistency_stats["shadow_rollback_count"] += 1
                elif actionable:
                    target_neighbor_consistency_stats["keep_count"] += 1
                reason_counts = target_neighbor_consistency_stats["reason_counts"]
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
            if bool(target_neighbor_consistency_diagnostics.get("rollback_applied", False)):
                output_variant = promotion_incumbent_variant
                selected_delta = deltas[output_variant]
                selected_proxy = proxies_by_variant[output_variant]
                selected_proxy_variant = output_variant
                gate_accepted = True
            target_neighbor_candidate_unlock_diagnostics = _target_neighbor_candidate_unlock_decision(
                output_variant=output_variant,
                ev=ev,
                deltas=deltas,
                target=target,
                target_frames=target_frames,
                loader=loader,
                device=device,
                args=args,
            )
            if bool(target_neighbor_candidate_unlock_diagnostics.get("enabled", False)):
                decision = str(target_neighbor_candidate_unlock_diagnostics.get("decision", "keep"))
                reason = str(target_neighbor_candidate_unlock_diagnostics.get("reject_reason") or "promoted")
                if decision == "promote":
                    target_neighbor_candidate_unlock_stats["promote_count"] += 1
                elif reason == "input_not_incumbent":
                    target_neighbor_candidate_unlock_stats["skipped_count"] += 1
                else:
                    target_neighbor_candidate_unlock_stats["keep_count"] += 1
                reason_counts = target_neighbor_candidate_unlock_stats["reason_counts"]
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
            if bool(target_neighbor_candidate_unlock_diagnostics.get("promote_applied", False)):
                output_variant = str(args.target_neighbor_candidate_unlock_candidate_variant)
                selected_delta = deltas[output_variant]
                selected_proxy = proxies_by_variant[output_variant]
                selected_proxy_variant = output_variant
                output_decision_source = "target_neighbor_unlock"
                gate_accepted = True
            if not gate_accepted and output_variant != "noop":
                selected_delta = torch.zeros_like(selected_delta)
                output_variant = "noop"
                selected_proxy = _candidate_proxy_stats(ev, selected_delta)
                selected_proxy_variant = "noop"
                nooped_views += 1
            elif output_variant == "noop":
                nooped_views += 1
            selected_image = torch.clamp(ev.base + selected_delta, 0.0, 1.0)
            save_image_tensor(selected_image, render_dir / f"{target.name}.png")
            if bool(args.copy_gt):
                shutil.copy2(target.gt_path, gt_dir / f"{target.name}{target.gt_path.suffix}")

            gt = loader.gt(str(target.gt_path)).to(device=device, dtype=torch.float32)
            candidate_metric_rows = {
                variant: _image_metrics(
                    ev.base,
                    gt,
                    delta,
                    compute_ssim=bool(args.compute_ssim),
                    ssim_max_side=int(args.ssim_max_side),
                )
                for variant, delta in deltas.items()
            }
            fixed_row = candidate_metric_rows["fixed"]
            learned_row = candidate_metric_rows["learned"]
            hybrid_row = candidate_metric_rows["hybrid"]
            selected_row = (
                dict(candidate_metric_rows[output_variant])
                if output_variant in candidate_metric_rows
                else _image_metrics(ev.base, gt, selected_delta, compute_ssim=bool(args.compute_ssim), ssim_max_side=int(args.ssim_max_side))
            )
            for row in candidate_metric_rows.values():
                row["view"] = target.name
            for variant, row in candidate_metric_rows.items():
                candidate_rows_by_variant.setdefault(variant, []).append(row)
            selected_row["view"] = target.name
            fixed_rows.append(fixed_row)
            learned_rows.append(learned_row)
            hybrid_rows.append(hybrid_row)
            selected_rows.append(selected_row)
            per_view.append(
                {
                    "view": target.name,
                    "support_names": list(ev.support_names),
                    "covered_fraction": float(ev.valid.to(torch.float32).mean().detach().cpu().item()),
                    "fixed": fixed_row,
                    "learned": learned_row,
                    "hybrid": hybrid_row,
                    "candidate_metrics": candidate_metric_rows,
                    "selected": selected_row,
                    "selected_variant": selected_variant,
                    "output_variant": output_variant,
                    "per_view_gate_accepted": bool(gate_accepted),
                    "per_view_knn_rejected_to_scene": bool(per_view_knn_rejected),
                    "per_view_risk_model_rejected_to_scene": bool(per_view_risk_model_rejected),
                    "local_support_rejected_to_scene": bool(local_support_rejected),
                    "source_reliability_rejected_to_scene": bool(source_reliability_rejected),
                    "risk_model_raw_output_variant": risk_model_raw_output_variant,
                    "local_support_raw_output_variant": local_support_raw_output_variant,
                    "pairwise_dominance_raw_output_variant": pairwise_dominance_raw_output_variant,
                    "source_reliability_raw_output_variant": source_reliability_raw_output_variant,
                    "risk_model_decision": risk_model_decision,
                    "local_support_decision": local_support_decision,
                    "pairwise_dominance_decision": pairwise_dominance_decision,
                    "source_reliability_decision": source_reliability_decision,
                    "output_decision_source": output_decision_source,
                    "promotion_rollback_diagnostics": promotion_rollback_diagnostics,
                    "selected_proxy_variant": selected_proxy_variant,
                    "selected_proxy": selected_proxy,
                    "per_view_gate_proxy": selected_proxy,
                    "per_view_knn_predictions": knn_predictions,
                    "per_view_knn_diagnostics": knn_diagnostics,
                    "per_view_risk_model_predictions": risk_model_predictions,
                    "per_view_risk_model_diagnostics": risk_model_diagnostics,
                    "local_support_diagnostics": local_support_diagnostics,
                    "pairwise_dominance_predictions": pairwise_dominance_predictions,
                    "pairwise_dominance_diagnostics": pairwise_dominance_diagnostics,
                    "source_reliability_predictions": source_reliability_predictions,
                    "source_reliability_diagnostics": source_reliability_diagnostics,
                    "target_neighbor_consistency_diagnostics": target_neighbor_consistency_diagnostics,
                    "target_neighbor_candidate_unlock_diagnostics": target_neighbor_candidate_unlock_diagnostics,
                }
            )
            if idx < int(args.save_example_views):
                visual_dir.mkdir(parents=True, exist_ok=True)
                save_image_tensor(ev.base, visual_dir / f"{target.name}_base.png")
                save_image_tensor(gt, visual_dir / f"{target.name}_gt.png")
                save_image_tensor(torch.clamp(ev.base + fixed_delta, 0.0, 1.0), visual_dir / f"{target.name}_fixed.png")
                save_image_tensor(torch.clamp(ev.base + learned_delta, 0.0, 1.0), visual_dir / f"{target.name}_learned.png")
                save_image_tensor(torch.clamp(ev.base + hybrid_delta, 0.0, 1.0), visual_dir / f"{target.name}_hybrid.png")
                save_image_tensor(selected_image, visual_dir / f"{target.name}_selected_{output_variant}.png")

    fixed_summary = _summarize_rows(fixed_rows, compute_ssim=bool(args.compute_ssim))
    learned_summary = _summarize_rows(learned_rows, compute_ssim=bool(args.compute_ssim))
    hybrid_summary = _summarize_rows(hybrid_rows, compute_ssim=bool(args.compute_ssim))
    candidate_summaries = {
        variant: _summarize_rows(rows, compute_ssim=bool(args.compute_ssim))
        for variant, rows in candidate_rows_by_variant.items()
    }
    selected_summary = _summarize_rows(selected_rows, compute_ssim=bool(args.compute_ssim))
    hybrid_minus_fixed_psnr = float(hybrid_summary.get("psnr_gain", 0.0)) - float(fixed_summary.get("psnr_gain", 0.0))
    hybrid_minus_fixed_ssim = (
        float(hybrid_summary.get("ssim_gain", 0.0)) - float(fixed_summary.get("ssim_gain", 0.0))
        if bool(args.compute_ssim)
        else None
    )
    selected_minus_fixed_psnr = float(selected_summary.get("psnr_gain", 0.0)) - float(fixed_summary.get("psnr_gain", 0.0))
    selected_minus_fixed_ssim = (
        float(selected_summary.get("ssim_gain", 0.0)) - float(fixed_summary.get("ssim_gain", 0.0))
        if bool(args.compute_ssim)
        else None
    )
    hybrid_all_axis_vs_fixed = (
        float(hybrid_summary.get("psnr_gain", 0.0)) > 0.0
        and hybrid_minus_fixed_psnr >= float(args.min_hybrid_vs_fixed_psnr_delta)
        and (
            not bool(args.compute_ssim)
            or (
                float(hybrid_summary.get("ssim_gain", 0.0)) > 0.0
                and float(hybrid_minus_fixed_ssim) >= float(args.min_hybrid_vs_fixed_ssim_delta)
            )
        )
    )
    selected_all_axis_safe_vs_fixed = (
        float(selected_summary.get("psnr_gain", 0.0)) > 0.0
        and selected_minus_fixed_psnr >= -float(args.selected_safe_tolerance_psnr)
        and (
            not bool(args.compute_ssim)
            or (
                float(selected_summary.get("ssim_gain", 0.0)) > 0.0
                and float(selected_minus_fixed_ssim) >= -float(args.selected_safe_tolerance_ssim)
            )
        )
    )
    active_gate = (
        "source_pairwise_dominance"
        if bool(pairwise_dominance_payload.get("enabled", False))
        else (
            "source_heldout_local_support"
            if bool(local_support_payload.get("enabled", False))
            else (
                "source_heldout_reliability"
                if bool(source_reliability_payload.get("enabled", False))
                else (
                    "source_heldout_risk_model"
                    if bool(per_view_risk_model_payload.get("enabled", False))
                    else (
                        "source_heldout_knn"
                        if bool(per_view_knn_payload.get("enabled", False))
                        else ("source_heldout_threshold" if bool(per_view_gate_payload.get("enabled", False)) else "off")
                    )
                )
            )
        )
    )
    output_label = f"scene `{selected_variant}`"
    if active_gate != "off":
        output_label = f"scene `{selected_variant}` with per-view `{active_gate}` refinement"
    verdict = (
        f"Selected support-transport output {output_label} is all-axis safe versus fixed on this target/test split."
        if selected_all_axis_safe_vs_fixed
        else f"Selected support-transport output {output_label} is not all-axis safe versus fixed on this target/test split."
    )
    online_target_proxy_enabled = bool(args.enable_target_neighbor_consistency) or bool(args.enable_target_neighbor_candidate_unlock)
    payload: dict[str, Any] = {
        "method": "apply v302 constrained hybrid support-transport calibrator",
        "target_gt_usage": "GT read only after candidate images are saved, for evaluation",
        "selection_protocol": {
            "scope": "source_heldout_before_target_loop",
            "target_gt_used_for_selection": False,
            "selection_frozen_before_target_loop": not online_target_proxy_enabled,
            "source_heldout_selector_frozen_before_target_loop": True,
            "online_target_proxy_refinement_enabled": online_target_proxy_enabled,
            "target_gt_first_read_stage": "post_render_eval_after_selected_image_save",
            "source_perceptual_uses_train_heldout_gt_only": bool(args.compute_source_perceptual),
            "local_support_uses_target_gt": False,
            "local_support_fit_scope": "source_heldout_before_target_loop",
            "pairwise_dominance_uses_target_gt": False,
            "pairwise_dominance_fit_scope": "source_heldout_before_target_loop",
            "source_reliability_uses_target_gt": False,
            "source_reliability_fit_scope": "source_heldout_before_target_loop",
            "source_reliability_calibrated_lcb_uses_target_gt": False,
            "promotion_rollback_uses_target_gt": False,
            "promotion_rollback_fit_scope": "source_heldout_pairwise_loo_before_target_loop",
            "target_neighbor_consistency_uses_target_gt": False,
            "target_neighbor_consistency_scope": "target_render_depth_camera_only_post_decision_certificate",
            "target_neighbor_candidate_unlock_uses_target_gt": False,
            "target_neighbor_candidate_unlock_scope": "target_render_depth_camera_only_online_per_view_unlock",
        },
        "checkpoint": str(args.checkpoint),
        "base_model_path": str(base_model),
        "base_method_name": base_method,
        "device": str(device),
        "split": {
            "target_split": str(args.target_split),
            "target_views": int(len(target_frames)),
            "support_views": int(len(source_frames)),
            "support_source_mode": str(args.support_source_mode),
        },
        "policy": {
            "anchor_alpha": float(args.anchor_alpha),
            "learned_scale": float(args.learned_scale),
            "blend": float(args.blend),
            "enable_candidate_ladder": bool(args.enable_candidate_ladder),
            "candidate_ladder_blends": str(args.candidate_ladder_blends),
            "candidate_variants": candidate_variants,
            "candidate_variant_blend_map": candidate_variant_blend_map,
            "candidate_feature_schema_version": 2,
            "output_variant": str(args.output_variant),
            "selected_variant": selected_variant,
            "per_view_gate_mode": active_gate,
            "source_perceptual_enabled": bool(args.compute_source_perceptual),
            "source_perceptual_max_side": int(args.source_perceptual_max_side),
            "source_objective_lpips_weight": float(args.source_objective_lpips_weight),
            "source_objective_dists_weight": float(args.source_objective_dists_weight),
            "policy_profile": str(getattr(args, "policy_profile", "none")),
        },
        "eval_config": {
            "compute_ssim": bool(args.compute_ssim),
            "ssim_max_side": int(args.ssim_max_side),
        },
        "evidence_config": {
            "k": int(args.k),
            "residual_clip": float(args.residual_clip),
            "min_confidence": float(args.min_confidence),
            "depth_abs_tol": float(args.depth_abs_tol),
            "depth_rel_tol": float(args.depth_rel_tol),
            "direction_weight": float(args.direction_weight),
            "evidence_max_side": int(args.evidence_max_side),
        },
        "train_config": train_config,
        "fixed_summary": fixed_summary,
        "learned_summary": learned_summary,
        "hybrid_summary": hybrid_summary,
        "candidate_summaries": candidate_summaries,
        "selected_summary": selected_summary,
        "summary": {
            "fixed_psnr_gain": float(fixed_summary.get("psnr_gain", 0.0)),
            "fixed_ssim_gain": float(fixed_summary.get("ssim_gain", 0.0)) if bool(args.compute_ssim) else None,
            "learned_psnr_gain": float(learned_summary.get("psnr_gain", 0.0)),
            "learned_ssim_gain": float(learned_summary.get("ssim_gain", 0.0)) if bool(args.compute_ssim) else None,
            "hybrid_psnr_gain": float(hybrid_summary.get("psnr_gain", 0.0)),
            "hybrid_ssim_gain": float(hybrid_summary.get("ssim_gain", 0.0)) if bool(args.compute_ssim) else None,
            "selected_psnr_gain": float(selected_summary.get("psnr_gain", 0.0)),
            "selected_ssim_gain": float(selected_summary.get("ssim_gain", 0.0)) if bool(args.compute_ssim) else None,
            "hybrid_minus_fixed_psnr_gain": hybrid_minus_fixed_psnr,
            "hybrid_minus_fixed_ssim_gain": hybrid_minus_fixed_ssim,
            "selected_minus_fixed_psnr_gain": selected_minus_fixed_psnr,
            "selected_minus_fixed_ssim_gain": selected_minus_fixed_ssim,
            "hybrid_all_axis_vs_fixed": bool(hybrid_all_axis_vs_fixed),
            "selected_all_axis_safe_vs_fixed": bool(selected_all_axis_safe_vs_fixed),
            "mean_covered_fraction": _mean([float(row["covered_fraction"]) for row in per_view]),
            "per_view_noop_fraction": float(nooped_views / max(len(per_view), 1)),
            "candidate_psnr_gains": {
                variant: float(row.get("psnr_gain", 0.0))
                for variant, row in candidate_summaries.items()
            },
            "candidate_ssim_gains": {
                variant: float(row.get("ssim_gain", 0.0))
                for variant, row in candidate_summaries.items()
            }
            if bool(args.compute_ssim)
            else None,
        },
        "per_view": per_view,
        "selector": selector_payload,
        "per_view_gate": per_view_gate_payload,
        "per_view_knn_policy": per_view_knn_payload,
        "local_support_policy": local_support_payload,
        "pairwise_dominance_policy": pairwise_dominance_payload,
        "per_view_risk_model_policy": per_view_risk_model_payload,
        "source_reliability_policy": source_reliability_payload,
        "promotion_rollback_policy": {
            **promotion_rollback_payload,
            "keep_count": int(promotion_rollback_stats["keep_count"]),
            "shadow_rollback_count": int(promotion_rollback_stats["shadow_rollback_count"]),
            "rollback_count": int(promotion_rollback_stats["rollback_count"]),
            "considered_count": int(promotion_rollback_stats["considered_count"]),
            "reason_counts": dict(promotion_rollback_stats["reason_counts"]),
        },
        "target_neighbor_consistency_policy": {
            "enabled": bool(args.enable_target_neighbor_consistency_certificate),
            "mode": str(args.target_neighbor_consistency_mode),
            "sources": [
                part.strip()
                for part in str(args.target_neighbor_consistency_sources).split(",")
                if part.strip()
            ],
            "min_incumbent_minus_output_delta": float(args.target_neighbor_consistency_min_incumbent_minus_output_delta),
            "neighbor_k": int(args.target_neighbor_consistency_neighbor_k),
            "direction_weight": float(args.target_neighbor_consistency_direction_weight),
            "max_side": int(args.target_neighbor_consistency_max_side),
            "depth_abs_tol": float(args.target_neighbor_consistency_depth_abs_tol),
            "depth_rel_tol": float(args.target_neighbor_consistency_depth_rel_tol),
            "min_confidence": float(args.target_neighbor_consistency_min_confidence),
            "min_effective_weight": float(args.target_neighbor_consistency_min_effective_weight),
            "source_contradiction_enabled": bool(args.target_neighbor_consistency_enable_source_contradiction),
            "contradiction_min_source_local_min_delta": float(
                args.target_neighbor_consistency_contradiction_min_source_local_min_delta
            ),
            "contradiction_min_source_local_cvar_delta": float(
                args.target_neighbor_consistency_contradiction_min_source_local_cvar_delta
            ),
            "contradiction_min_source_positive_fraction": float(
                args.target_neighbor_consistency_contradiction_min_source_positive_fraction
            ),
            "contradiction_max_incumbent_minus_output_delta": float(
                args.target_neighbor_consistency_contradiction_max_incumbent_minus_output_delta
            ),
            "uses_target_gt": False,
            "reference": "target split render/depth/camera only; compares candidate-vs-incumbent warp MAE to neighboring base renders",
            "keep_count": int(target_neighbor_consistency_stats["keep_count"]),
            "shadow_rollback_count": int(target_neighbor_consistency_stats["shadow_rollback_count"]),
            "rollback_count": int(target_neighbor_consistency_stats["rollback_count"]),
            "considered_count": int(target_neighbor_consistency_stats["considered_count"]),
            "reason_counts": dict(target_neighbor_consistency_stats["reason_counts"]),
        },
        "target_neighbor_candidate_unlock_policy": {
            "enabled": bool(args.enable_target_neighbor_candidate_unlock),
            "incumbent_variant": str(args.target_neighbor_candidate_unlock_incumbent_variant),
            "candidate_variant": str(args.target_neighbor_candidate_unlock_candidate_variant),
            "min_incumbent_minus_candidate_delta": float(args.target_neighbor_candidate_unlock_min_incumbent_minus_candidate_delta),
            "uses_target_gt": False,
            "reference": "target split render/depth/camera only; promotes a fixed incumbent when the candidate is more target-neighbor consistent by a frozen margin",
            "promote_count": int(target_neighbor_candidate_unlock_stats["promote_count"]),
            "keep_count": int(target_neighbor_candidate_unlock_stats["keep_count"]),
            "skipped_count": int(target_neighbor_candidate_unlock_stats["skipped_count"]),
            "reason_counts": dict(target_neighbor_candidate_unlock_stats["reason_counts"]),
        },
        "verdict": verdict,
        "final_status": "APPLY_EVAL_COMPLETE_NOT_PAPER_COMPLETE",
    }
    _write_report(output_dir, payload)

    if bool(args.enable_wandb):
        import wandb

        run = wandb.init(project=str(args.wandb_project), name=str(args.wandb_run_name), dir=str(output_dir))
        flat = {
            "apply/fixed_psnr_gain": float(payload["summary"]["fixed_psnr_gain"]),
            "apply/learned_psnr_gain": float(payload["summary"]["learned_psnr_gain"]),
            "apply/hybrid_psnr_gain": float(payload["summary"]["hybrid_psnr_gain"]),
            "apply/selected_psnr_gain": float(payload["summary"]["selected_psnr_gain"]),
            "apply/hybrid_minus_fixed_psnr_gain": float(payload["summary"]["hybrid_minus_fixed_psnr_gain"]),
            "apply/selected_minus_fixed_psnr_gain": float(payload["summary"]["selected_minus_fixed_psnr_gain"]),
            "apply/hybrid_all_axis_vs_fixed": float(bool(payload["summary"]["hybrid_all_axis_vs_fixed"])),
            "apply/selected_all_axis_safe_vs_fixed": float(bool(payload["summary"]["selected_all_axis_safe_vs_fixed"])),
            "apply/per_view_noop_fraction": float(payload["summary"]["per_view_noop_fraction"]),
        }
        if bool(args.compute_ssim):
            flat["apply/fixed_ssim_gain"] = float(payload["summary"]["fixed_ssim_gain"])
            flat["apply/learned_ssim_gain"] = float(payload["summary"]["learned_ssim_gain"])
            flat["apply/hybrid_ssim_gain"] = float(payload["summary"]["hybrid_ssim_gain"])
            flat["apply/selected_ssim_gain"] = float(payload["summary"]["selected_ssim_gain"])
            flat["apply/hybrid_minus_fixed_ssim_gain"] = float(payload["summary"]["hybrid_minus_fixed_ssim_gain"])
            flat["apply/selected_minus_fixed_ssim_gain"] = float(payload["summary"]["selected_minus_fixed_ssim_gain"])
        for variant, row in payload.get("candidate_summaries", {}).items():
            flat[f"apply/candidate/{variant}/psnr_gain"] = float(row.get("psnr_gain", 0.0))
            if bool(args.compute_ssim):
                flat[f"apply/candidate/{variant}/ssim_gain"] = float(row.get("ssim_gain", 0.0))
        if payload.get("local_support_policy"):
            local_support = payload["local_support_policy"]
            flat["apply/local_support/enabled"] = float(bool(local_support.get("enabled", False)))
            flat["apply/local_support/source_accept_fraction"] = float(local_support.get("source_accept_fraction", 0.0) or 0.0)
            flat["apply/local_support/source_mean_psnr_delta_vs_scene"] = float(
                local_support.get("source_mean_psnr_delta_vs_scene_selected", 0.0) or 0.0
            )
        if payload.get("pairwise_dominance_policy"):
            pairwise = payload["pairwise_dominance_policy"]
            flat["apply/pairwise_dominance/enabled"] = float(bool(pairwise.get("enabled", False)))
            flat["apply/pairwise_dominance/source_accept_fraction"] = float(pairwise.get("source_accept_fraction", 0.0) or 0.0)
            flat["apply/pairwise_dominance/source_mean_psnr_delta_vs_incumbent"] = float(
                pairwise.get("source_mean_psnr_delta_vs_incumbent", 0.0) or 0.0
            )
            flat["apply/pairwise_dominance/source_mean_ssim_delta_vs_incumbent"] = float(
                pairwise.get("source_mean_ssim_delta_vs_incumbent", 0.0) or 0.0
            )
        if payload.get("promotion_rollback_policy"):
            promotion = payload["promotion_rollback_policy"]
            flat["apply/promotion_rollback/enabled"] = float(bool(promotion.get("enabled", False)))
            flat["apply/promotion_rollback/keep_count"] = float(promotion.get("keep_count", 0) or 0)
            flat["apply/promotion_rollback/shadow_rollback_count"] = float(
                promotion.get("shadow_rollback_count", 0) or 0
            )
            flat["apply/promotion_rollback/rollback_count"] = float(promotion.get("rollback_count", 0) or 0)
            flat["apply/promotion_rollback/source_sample_count"] = float(promotion.get("source_sample_count", 0) or 0)
        if payload.get("target_neighbor_consistency_policy"):
            target_neighbor = payload["target_neighbor_consistency_policy"]
            flat["apply/target_neighbor_consistency/enabled"] = float(bool(target_neighbor.get("enabled", False)))
            flat["apply/target_neighbor_consistency/keep_count"] = float(target_neighbor.get("keep_count", 0) or 0)
            flat["apply/target_neighbor_consistency/shadow_rollback_count"] = float(
                target_neighbor.get("shadow_rollback_count", 0) or 0
            )
            flat["apply/target_neighbor_consistency/rollback_count"] = float(target_neighbor.get("rollback_count", 0) or 0)
            flat["apply/target_neighbor_consistency/considered_count"] = float(target_neighbor.get("considered_count", 0) or 0)
        if payload.get("target_neighbor_candidate_unlock_policy"):
            unlock = payload["target_neighbor_candidate_unlock_policy"]
            flat["apply/target_neighbor_candidate_unlock/enabled"] = float(bool(unlock.get("enabled", False)))
            flat["apply/target_neighbor_candidate_unlock/promote_count"] = float(unlock.get("promote_count", 0) or 0)
            flat["apply/target_neighbor_candidate_unlock/keep_count"] = float(unlock.get("keep_count", 0) or 0)
            flat["apply/target_neighbor_candidate_unlock/skipped_count"] = float(unlock.get("skipped_count", 0) or 0)
        run.log(flat)
        run.summary.update(flat)
        run.finish()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", type=Path, required=True)
    parser.add_argument("--base_method_name", default="ours_26000")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_split", default="test")
    parser.add_argument("--support_source_mode", choices=["source_split", "all_train"], default="source_split")
    parser.add_argument("--heldout_stride", type=int, default=4)
    parser.add_argument("--heldout_offset", type=int, default=0)
    parser.add_argument("--max_target_views", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--anchor_alpha", type=float, default=0.25)
    parser.add_argument("--learned_scale", type=float, default=0.5)
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--enable_candidate_ladder", action="store_true")
    parser.add_argument(
        "--candidate_ladder_blends",
        default="0.25,0.75",
        help="Comma-separated residual blend strengths to add as source-heldout candidates when candidate ladder is enabled.",
    )
    parser.add_argument("--output_variant", default="hybrid")
    parser.add_argument(
        "--policy_profile",
        choices=["none", "v322c_incumbent"],
        default="none",
        help=(
            "Apply a named frozen policy profile. v322c_incumbent pins the "
            "archived v322C incumbent-preserving source/KNN policy and its "
            "fair evidence/SSIM evaluation resolution."
        ),
    )
    parser.add_argument("--selector_val_stride", type=int, default=3)
    parser.add_argument("--selector_val_offset", type=int, default=0)
    parser.add_argument("--max_selector_views", type=int, default=0)
    parser.add_argument("--selector_min_vs_fixed_psnr_delta", type=float, default=0.0)
    parser.add_argument("--selector_min_vs_fixed_ssim_delta", type=float, default=0.0)
    parser.add_argument("--selector_min_vs_fixed_lpips_delta", type=float, default=-1.0e9)
    parser.add_argument("--selector_min_vs_fixed_dists_delta", type=float, default=-1.0e9)
    parser.add_argument("--selected_safe_tolerance_psnr", type=float, default=1.0e-12)
    parser.add_argument("--selected_safe_tolerance_ssim", type=float, default=1.0e-12)
    parser.add_argument("--enable_per_view_risk_gate", action="store_true")
    parser.add_argument(
        "--per_view_gate_score_grid",
        default="confidence_snr,delta_snr,signal_snr,covered_fraction,confidence_mean,residual_stability,mean_abs_delta,changed_fraction",
    )
    parser.add_argument("--per_view_gate_min_accept_fraction", type=float, default=0.20)
    parser.add_argument("--per_view_gate_max_accept_fraction", type=float, default=1.00)
    parser.add_argument("--per_view_gate_min_mean_psnr_delta", type=float, default=0.0)
    parser.add_argument("--per_view_gate_min_mean_ssim_delta", type=float, default=0.0)
    parser.add_argument("--per_view_gate_min_cvar_psnr_delta", type=float, default=0.0)
    parser.add_argument("--per_view_gate_cvar_weight", type=float, default=0.25)
    parser.add_argument("--per_view_gate_min_weight", type=float, default=0.10)
    parser.add_argument("--enable_per_view_knn_policy", action="store_true")
    parser.add_argument(
        "--per_view_knn_feature_grid",
        default="covered_fraction,mean_abs_delta,confidence_mean,residual_std_mean,delta_snr,signal_snr,confidence_snr,changed_fraction",
    )
    parser.add_argument("--per_view_knn_k", type=int, default=3)
    parser.add_argument("--per_view_knn_min_predicted_score", type=float, default=0.0)
    parser.add_argument("--per_view_knn_min_score_delta_vs_scene", type=float, default=0.0)
    parser.add_argument("--per_view_knn_base_variants_only", action="store_true")
    parser.add_argument("--per_view_knn_forbid_fixed_when_scene_nonfixed", action="store_true")
    parser.add_argument("--per_view_knn_enable_local_tail_guard", action="store_true")
    parser.add_argument("--per_view_knn_local_tail_k", type=int, default=0)
    parser.add_argument("--per_view_knn_min_local_psnr_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--per_view_knn_min_local_ssim_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--per_view_knn_min_local_cvar_delta_vs_scene", type=float, default=0.0)
    parser.add_argument("--per_view_knn_min_local_min_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--per_view_knn_min_local_positive_fraction_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--per_view_knn_auto_threshold", action="store_true")
    parser.add_argument("--per_view_knn_reject_variant", choices=["noop", "scene"], default="noop")
    parser.add_argument("--per_view_knn_min_accept_fraction", type=float, default=0.0)
    parser.add_argument("--per_view_knn_max_accept_fraction", type=float, default=1.0)
    parser.add_argument("--per_view_knn_min_source_psnr_delta", type=float, default=0.0)
    parser.add_argument("--per_view_knn_min_source_ssim_delta", type=float, default=-1.0e9)
    parser.add_argument("--per_view_knn_min_source_cvar_delta", type=float, default=-1.0e9)
    parser.add_argument("--per_view_knn_min_source_min_delta", type=float, default=-1.0e9)
    parser.add_argument("--per_view_knn_min_source_positive_fraction_delta", type=float, default=-1.0e9)
    parser.add_argument("--per_view_knn_source_cvar_weight", type=float, default=0.25)
    parser.add_argument("--per_view_knn_source_min_weight", type=float, default=0.10)
    parser.add_argument("--per_view_knn_source_positive_weight", type=float, default=0.25)
    parser.add_argument("--per_view_knn_require_source_safe", action="store_true")
    parser.add_argument("--per_view_knn_allow_when_scene_fixed", action="store_true")
    parser.add_argument("--enable_local_support_policy", action="store_true")
    parser.add_argument(
        "--local_support_feature_grid",
        default=(
            "covered_fraction,mean_abs_delta,confidence_mean,residual_std_mean,delta_snr,signal_snr,"
            "confidence_snr,changed_fraction,delta_signal_cosine,opposition_fraction,aligned_fraction,"
            "delta_to_signal_ratio,std_to_signal_ratio,support_confidence,support_count_mean"
        ),
    )
    parser.add_argument("--local_support_k", type=int, default=3)
    parser.add_argument("--local_support_base_variants_only", action="store_true")
    parser.add_argument("--local_support_reject_variant", choices=["noop", "scene"], default="scene")
    parser.add_argument("--local_support_min_local_psnr_delta_vs_scene", type=float, default=0.0)
    parser.add_argument("--local_support_min_local_ssim_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--local_support_min_local_cvar_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--local_support_min_local_min_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--local_support_min_local_positive_fraction_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--local_support_min_score_delta_vs_scene", type=float, default=0.0)
    parser.add_argument("--local_support_min_accept_fraction", type=float, default=0.0)
    parser.add_argument("--local_support_max_accept_fraction", type=float, default=1.0)
    parser.add_argument("--local_support_min_source_psnr_delta", type=float, default=0.0)
    parser.add_argument("--local_support_min_source_ssim_delta", type=float, default=-1.0e9)
    parser.add_argument("--local_support_min_source_cvar_delta", type=float, default=-1.0e9)
    parser.add_argument("--local_support_min_source_min_delta", type=float, default=-1.0e9)
    parser.add_argument("--local_support_min_source_positive_fraction_delta", type=float, default=-1.0e9)
    parser.add_argument("--local_support_require_source_safe", action="store_true")
    parser.add_argument("--local_support_forbid_fixed_when_scene_nonfixed", action="store_true")
    parser.add_argument("--local_support_post_incumbent_fallback_only", action="store_true")
    parser.add_argument("--local_support_ssim_weight", type=float, default=0.0)
    parser.add_argument("--local_support_cvar_weight", type=float, default=0.25)
    parser.add_argument("--local_support_min_weight", type=float, default=0.10)
    parser.add_argument("--local_support_positive_weight", type=float, default=0.25)
    parser.add_argument("--enable_pairwise_dominance_policy", action="store_true")
    parser.add_argument(
        "--pairwise_dominance_feature_grid",
        default=(
            "covered_fraction,mean_abs_delta,confidence_mean,residual_std_mean,delta_snr,signal_snr,"
            "confidence_snr,changed_fraction,delta_signal_cosine,opposition_fraction,aligned_fraction,"
            "delta_to_signal_ratio,std_to_signal_ratio,support_confidence,support_count_mean"
        ),
    )
    parser.add_argument("--pairwise_dominance_ridge", type=float, default=1.0e-3)
    parser.add_argument("--pairwise_dominance_k", type=int, default=3)
    parser.add_argument("--pairwise_dominance_min_predicted_objective_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_min_predicted_psnr_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_min_predicted_ssim_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_min_local_psnr_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_min_local_ssim_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_min_local_cvar_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_min_local_min_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_min_accept_fraction", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_max_accept_fraction", type=float, default=1.0)
    parser.add_argument("--pairwise_dominance_min_source_psnr_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_min_source_ssim_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_min_source_cvar_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_min_source_min_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_enable_ood_guard", action="store_true")
    parser.add_argument("--pairwise_dominance_ood_quantile", type=float, default=0.80)
    parser.add_argument("--pairwise_dominance_max_blend_step", type=float, default=1.0e9)
    parser.add_argument("--pairwise_dominance_enable_adaptive_blend_step", action="store_true")
    parser.add_argument("--pairwise_dominance_adaptive_max_blend_step", type=float, default=1.0e9)
    parser.add_argument("--pairwise_dominance_large_step_min_local_psnr_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_large_step_min_local_ssim_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_large_step_min_local_cvar_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_large_step_min_local_min_delta", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_large_step_min_positive_fraction", type=float, default=1.0)
    parser.add_argument("--pairwise_dominance_psnr_weight", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_ssim_weight", type=float, default=0.0)
    parser.add_argument("--pairwise_dominance_local_cvar_weight", type=float, default=0.25)
    parser.add_argument("--enable_promotion_rollback_certificate", action="store_true")
    parser.add_argument("--promotion_rollback_mode", choices=["shadow", "enforce"], default="shadow")
    parser.add_argument(
        "--promotion_rollback_sources",
        default="pairwise",
        help="Comma-separated decision sources checked by the post-decision promotion rollback certificate.",
    )
    parser.add_argument("--promotion_rollback_min_calibration_samples", type=int, default=4)
    parser.add_argument("--promotion_rollback_calibration_quantile", type=float, default=0.8)
    parser.add_argument("--promotion_rollback_calibration_scale", type=float, default=1.0)
    parser.add_argument("--promotion_rollback_min_lcb_objective_delta", type=float, default=-1.0e9)
    parser.add_argument("--promotion_rollback_min_lcb_psnr_delta", type=float, default=-1.0e9)
    parser.add_argument("--promotion_rollback_min_lcb_ssim_delta", type=float, default=-1.0e9)
    parser.add_argument("--promotion_rollback_min_local_cvar_delta", type=float, default=-1.0e9)
    parser.add_argument("--promotion_rollback_min_local_min_delta", type=float, default=-1.0e9)
    parser.add_argument("--promotion_rollback_max_local_negative_fraction", type=float, default=1.0)
    parser.add_argument("--enable_target_neighbor_consistency_certificate", action="store_true")
    parser.add_argument("--target_neighbor_consistency_mode", choices=["shadow", "enforce"], default="shadow")
    parser.add_argument(
        "--target_neighbor_consistency_sources",
        default="pairwise",
        help="Comma-separated decision sources checked by the target-neighbor render self-consistency certificate.",
    )
    parser.add_argument("--target_neighbor_consistency_min_incumbent_minus_output_delta", type=float, default=-1.0e-4)
    parser.add_argument("--target_neighbor_consistency_neighbor_k", type=int, default=2)
    parser.add_argument("--target_neighbor_consistency_direction_weight", type=float, default=0.35)
    parser.add_argument("--target_neighbor_consistency_max_side", type=int, default=256)
    parser.add_argument("--target_neighbor_consistency_depth_abs_tol", type=float, default=0.03)
    parser.add_argument("--target_neighbor_consistency_depth_rel_tol", type=float, default=0.04)
    parser.add_argument("--target_neighbor_consistency_min_confidence", type=float, default=1.0e-4)
    parser.add_argument("--target_neighbor_consistency_min_effective_weight", type=float, default=0.01)
    parser.add_argument("--target_neighbor_consistency_enable_source_contradiction", action="store_true")
    parser.add_argument("--target_neighbor_consistency_contradiction_min_source_local_min_delta", type=float, default=1.0e9)
    parser.add_argument("--target_neighbor_consistency_contradiction_min_source_local_cvar_delta", type=float, default=1.0e9)
    parser.add_argument("--target_neighbor_consistency_contradiction_min_source_positive_fraction", type=float, default=1.0)
    parser.add_argument("--target_neighbor_consistency_contradiction_max_incumbent_minus_output_delta", type=float, default=-1.0e9)
    parser.add_argument("--enable_target_neighbor_candidate_unlock", action="store_true")
    parser.add_argument("--target_neighbor_candidate_unlock_incumbent_variant", default="fixed")
    parser.add_argument("--target_neighbor_candidate_unlock_candidate_variant", default="learned")
    parser.add_argument("--target_neighbor_candidate_unlock_min_incumbent_minus_candidate_delta", type=float, default=2.0e-4)
    parser.add_argument("--enable_per_view_risk_model_policy", action="store_true")
    parser.add_argument(
        "--per_view_risk_model_feature_grid",
        default="covered_fraction,mean_abs_delta,confidence_mean,residual_std_mean,delta_snr,signal_snr,confidence_snr,changed_fraction",
    )
    parser.add_argument("--per_view_risk_model_ridge", type=float, default=1.0e-3)
    parser.add_argument("--per_view_risk_model_reject_variant", choices=["noop", "scene"], default="scene")
    parser.add_argument("--per_view_risk_model_only_when_scene_fixed", action="store_true")
    parser.add_argument("--per_view_risk_model_allow_when_scene_fixed", action="store_true")
    parser.add_argument("--per_view_risk_model_require_source_safe", action="store_true")
    parser.add_argument("--per_view_risk_model_min_accept_fraction", type=float, default=0.0)
    parser.add_argument("--per_view_risk_model_max_accept_fraction", type=float, default=1.0)
    parser.add_argument("--per_view_risk_model_min_source_psnr_delta", type=float, default=0.0)
    parser.add_argument("--per_view_risk_model_min_source_ssim_delta", type=float, default=-1.0e9)
    parser.add_argument("--per_view_risk_model_min_source_cvar_delta", type=float, default=-1.0e9)
    parser.add_argument("--per_view_risk_model_min_source_min_delta", type=float, default=-1.0e9)
    parser.add_argument("--per_view_risk_model_min_source_positive_fraction_delta", type=float, default=-1.0e9)
    parser.add_argument("--per_view_risk_model_source_cvar_weight", type=float, default=0.25)
    parser.add_argument("--per_view_risk_model_source_min_weight", type=float, default=0.10)
    parser.add_argument("--per_view_risk_model_source_positive_weight", type=float, default=0.25)
    parser.add_argument("--per_view_risk_model_min_predicted_objective_delta", type=float, default=0.0)
    parser.add_argument("--per_view_risk_model_min_predicted_psnr_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--per_view_risk_model_min_predicted_ssim_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--per_view_risk_model_min_predicted_psnr", type=float, default=-1.0e9)
    parser.add_argument("--per_view_risk_model_min_predicted_ssim", type=float, default=-1.0e9)
    parser.add_argument("--per_view_risk_model_require_predicted_scene_axis_nonregression", action="store_true")
    parser.add_argument("--per_view_risk_model_scene_axis_guard_margin_psnr", type=float, default=0.0)
    parser.add_argument("--per_view_risk_model_scene_axis_guard_margin_ssim", type=float, default=0.0)
    parser.add_argument("--per_view_risk_model_enable_ood_guard", action="store_true")
    parser.add_argument("--per_view_risk_model_ood_quantile", type=float, default=0.90)
    parser.add_argument("--per_view_risk_model_ood_min_samples", type=int, default=4)
    parser.add_argument("--per_view_risk_model_use_source_perceptual_objective", action="store_true")
    parser.add_argument("--per_view_risk_model_auto_objective_margin", action="store_true")
    parser.add_argument("--enable_source_reliability_policy", action="store_true")
    parser.add_argument(
        "--source_reliability_feature_grid",
        default="covered_fraction,mean_abs_delta,confidence_mean,residual_std_mean,delta_snr,signal_snr,confidence_snr,changed_fraction,delta_signal_cosine,opposition_fraction,aligned_fraction,delta_to_signal_ratio,std_to_signal_ratio,support_confidence,support_count_mean",
    )
    parser.add_argument("--source_reliability_ridge", type=float, default=1.0e-3)
    parser.add_argument("--source_reliability_reject_variant", choices=["noop", "scene"], default="scene")
    parser.add_argument("--source_reliability_min_accept_fraction", type=float, default=0.0)
    parser.add_argument("--source_reliability_max_accept_fraction", type=float, default=1.0)
    parser.add_argument("--source_reliability_min_source_psnr_delta", type=float, default=0.0)
    parser.add_argument("--source_reliability_min_source_ssim_delta", type=float, default=-1.0e9)
    parser.add_argument("--source_reliability_fixed_scene_min_source_ssim_delta", type=float, default=-1.0e9)
    parser.add_argument("--source_reliability_min_source_cvar_delta", type=float, default=-1.0e9)
    parser.add_argument("--source_reliability_min_source_min_delta", type=float, default=-1.0e9)
    parser.add_argument("--source_reliability_min_source_positive_fraction_delta", type=float, default=-1.0e9)
    parser.add_argument("--source_reliability_source_cvar_weight", type=float, default=0.25)
    parser.add_argument("--source_reliability_source_min_weight", type=float, default=0.10)
    parser.add_argument("--source_reliability_source_positive_weight", type=float, default=0.25)
    parser.add_argument("--source_reliability_require_source_safe", action="store_true")
    parser.add_argument("--source_reliability_auto_objective_margin", action="store_true")
    parser.add_argument("--source_reliability_min_predicted_objective_delta_vs_scene", type=float, default=0.0)
    parser.add_argument("--source_reliability_min_predicted_psnr_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--source_reliability_min_predicted_ssim_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--source_reliability_min_predicted_lpips_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--source_reliability_min_predicted_dists_delta_vs_scene", type=float, default=-1.0e9)
    parser.add_argument("--source_reliability_forbid_fixed_when_scene_nonfixed", action="store_true")
    parser.add_argument("--source_reliability_enable_fixed_rollback_certificate", action="store_true")
    parser.add_argument("--source_reliability_fixed_rollback_min_objective_margin", type=float, default=0.0)
    parser.add_argument("--source_reliability_fixed_rollback_min_psnr_margin", type=float, default=0.0)
    parser.add_argument("--source_reliability_fixed_rollback_min_ssim_margin", type=float, default=0.0)
    parser.add_argument("--source_reliability_fixed_rollback_min_best_psnr_delta", type=float, default=0.0)
    parser.add_argument("--source_reliability_fixed_rollback_min_best_ssim_delta", type=float, default=0.0)
    parser.add_argument("--source_reliability_fixed_rollback_max_scene_opposition_fraction", type=float, default=1.0)
    parser.add_argument("--source_reliability_fixed_rollback_min_scene_aligned_fraction", type=float, default=0.0)
    parser.add_argument("--source_reliability_enable_calibrated_lcb", action="store_true")
    parser.add_argument(
        "--source_reliability_calibrated_lcb_mode",
        choices=["calibrated_lcb", "raw_incumbent"],
        default="calibrated_lcb",
    )
    parser.add_argument("--source_reliability_calibration_quantile", type=float, default=0.80)
    parser.add_argument("--source_reliability_calibration_scale", type=float, default=1.0)
    parser.add_argument("--source_reliability_calibration_min_samples", type=int, default=12)
    parser.add_argument("--source_reliability_enable_ood_guard", action="store_true")
    parser.add_argument("--source_reliability_ood_quantile", type=float, default=0.90)
    parser.add_argument("--source_reliability_ood_min_samples", type=int, default=4)
    parser.add_argument("--residual_clip", type=float, default=0.25)
    parser.add_argument("--min_confidence", type=float, default=1.0e-4)
    parser.add_argument("--depth_abs_tol", type=float, default=0.02)
    parser.add_argument("--depth_rel_tol", type=float, default=0.03)
    parser.add_argument("--direction_weight", type=float, default=0.35)
    parser.add_argument("--evidence_max_side", type=int, default=512)
    parser.add_argument("--compute_ssim", action="store_true")
    parser.add_argument("--ssim_max_side", type=int, default=384)
    parser.add_argument("--compute_source_perceptual", action="store_true")
    parser.add_argument("--source_perceptual_max_side", type=int, default=256)
    parser.add_argument("--source_objective_lpips_weight", type=float, default=0.0)
    parser.add_argument("--source_objective_dists_weight", type=float, default=0.0)
    parser.add_argument("--min_hybrid_vs_fixed_psnr_delta", type=float, default=0.0)
    parser.add_argument("--min_hybrid_vs_fixed_ssim_delta", type=float, default=0.0)
    parser.add_argument("--save_example_views", type=int, default=0)
    parser.add_argument("--copy_gt", action="store_true")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet-transport-diagnostics")
    parser.add_argument("--wandb_run_name", default="apply-v302-support-transport")
    args = parser.parse_args()
    _apply_policy_profile(args)
    payload = run(args)
    print(
        json.dumps(
            {
                "summary": payload["summary"],
                "report": str(Path(args.output_dir) / "support_transport_apply_report.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
