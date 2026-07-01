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
from typing import Any

import torch
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
    _selection_objective,
    _split_calibrator_train_val,
    _split_source_heldout,
    _summarize_rows,
)
from utils.evidence_lumigraph_adapter import (  # noqa: E402
    FrameLoader,
    compute_evidence_signal,
    load_split_frames,
    save_image_tensor,
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
        f"- output variant: `{payload['policy']['output_variant']}`",
        f"- selected variant: `{payload['policy']['selected_variant']}`",
        f"- per-view gate: `{payload['policy'].get('per_view_gate_mode', 'off')}`",
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
    for key, label in [
        ("fixed_summary", "fixed raw ELA"),
        ("learned_summary", "learned only"),
        ("hybrid_summary", "hybrid"),
        ("selected_summary", "selected output"),
    ]:
        row = payload[key]
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
            f"- source safe vs fixed: `{knn.get('source_safe_vs_fixed')}`",
            f"- source selected counts: `{knn.get('source_selected_counts')}`",
            f"- verdict: `{knn.get('verdict')}`",
        ]
    (output_dir / "support_transport_apply_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _candidate_deltas(ev: Any, pred_delta: torch.Tensor, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    fixed_delta = float(args.anchor_alpha) * ev.signal
    learned_delta = float(args.learned_scale) * pred_delta
    hybrid_delta = (1.0 - float(args.blend)) * fixed_delta + float(args.blend) * learned_delta
    return {"fixed": fixed_delta, "learned": learned_delta, "hybrid": hybrid_delta}


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
    abs_signal = torch.abs(ev.signal.to(device=delta.device, dtype=torch.float32))
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


def _summarize_metric_rows(rows: list[dict[str, float]], *, compute_ssim: bool) -> dict[str, Any]:
    return _summarize_rows(rows, compute_ssim=compute_ssim) if rows else {}


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
    variants = ["fixed", "learned", "hybrid"]
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
                "score": _objective_from_metrics(candidate["metrics"], compute_ssim=compute_ssim),
            }
            entries_by_variant[variant].append(entry)
            all_vectors.append(vector)
    mean, std = _feature_stats(all_vectors)
    if not mean:
        return {"enabled": False, "verdict": "no usable source-heldout proxy vectors"}

    def predict(variant: str, vector: list[float], *, exclude_view: str | None = None) -> float:
        pool = [entry for entry in entries_by_variant[variant] if exclude_view is None or entry["view"] != exclude_view]
        if not pool:
            return float("-inf")
        ranked = sorted(
            pool,
            key=lambda entry: _normalized_distance(vector, entry["vector"], mean, std),
        )
        k = max(1, min(int(args.per_view_knn_k), len(ranked)))
        return _mean([float(entry["score"]) for entry in ranked[:k]])

    source_policy_rows = []
    selected_counts: dict[str, int] = {"fixed": 0, "learned": 0, "hybrid": 0, "noop": 0}
    for row in selector_payload["per_view"]:
        predictions = {}
        for variant in variants:
            predictions[variant] = predict(
                variant,
                _feature_vector(row["candidates"][variant]["proxy"], feature_names),
                exclude_view=row["view"],
            )
        best_variant, best_score = max(predictions.items(), key=lambda item: item[1])
        if best_score < float(args.per_view_knn_min_predicted_score):
            selected_counts["noop"] += 1
            source_policy_rows.append(_noop_like(row["candidates"][best_variant]["metrics"]))
        else:
            selected_counts[best_variant] += 1
            source_policy_rows.append(row["candidates"][best_variant]["metrics"])

    source_summary = _summarize_metric_rows(source_policy_rows, compute_ssim=compute_ssim)
    fixed_summary = selector_payload["summaries"]["fixed"]
    selected_variant = str(selector_payload["selected_variant"])
    scene_selected_summary = selector_payload["summaries"][selected_variant]
    mean_delta = float(source_summary.get("psnr_gain", 0.0)) - float(scene_selected_summary.get("psnr_gain", 0.0))
    ssim_delta = (
        float(source_summary.get("ssim_gain", 0.0)) - float(scene_selected_summary.get("ssim_gain", 0.0))
        if compute_ssim
        else 0.0
    )
    safe_vs_fixed = (
        float(source_summary.get("psnr_gain", 0.0)) >= float(fixed_summary.get("psnr_gain", 0.0)) - float(args.selected_safe_tolerance_psnr)
        and (
            not compute_ssim
            or float(source_summary.get("ssim_gain", 0.0))
            >= float(fixed_summary.get("ssim_gain", 0.0)) - float(args.selected_safe_tolerance_ssim)
        )
    )
    if mean_delta < float(args.per_view_knn_min_source_psnr_delta):
        return {
            "enabled": False,
            "verdict": "source-heldout KNN policy did not improve scene-selected PSNR enough",
            "feature_names": feature_names,
            "source_summary": source_summary,
            "source_fixed_summary": fixed_summary,
            "source_scene_selected_summary": scene_selected_summary,
            "source_mean_psnr_delta_vs_scene_selected": mean_delta,
            "source_mean_ssim_delta_vs_scene_selected": ssim_delta,
            "source_safe_vs_fixed": bool(safe_vs_fixed),
            "source_selected_counts": selected_counts,
        }
    if bool(args.per_view_knn_require_source_safe) and not safe_vs_fixed:
        return {
            "enabled": False,
            "verdict": "source-heldout KNN policy did not clear fixed safety gate",
            "feature_names": feature_names,
            "source_summary": source_summary,
            "source_fixed_summary": fixed_summary,
            "source_scene_selected_summary": scene_selected_summary,
            "source_selected_counts": selected_counts,
        }
    return {
        "enabled": True,
        "verdict": "source-heldout KNN per-view policy selected",
        "feature_names": feature_names,
        "feature_mean": mean,
        "feature_std": std,
        "k": int(args.per_view_knn_k),
        "min_predicted_score": float(args.per_view_knn_min_predicted_score),
        "entries_by_variant": entries_by_variant,
        "source_summary": source_summary,
        "source_fixed_summary": fixed_summary,
        "source_scene_selected_summary": scene_selected_summary,
        "source_mean_psnr_delta_vs_scene_selected": mean_delta,
        "source_mean_ssim_delta_vs_scene_selected": ssim_delta,
        "source_safe_vs_fixed": bool(safe_vs_fixed),
        "source_selected_counts": selected_counts,
    }


def _knn_choose_variant(
    proxies_by_variant: dict[str, dict[str, float]],
    policy: dict[str, Any],
    *,
    compute_ssim: bool,
) -> tuple[str, dict[str, float]]:
    del compute_ssim
    variants = ["fixed", "learned", "hybrid"]
    feature_names = list(policy["feature_names"])
    mean = [float(x) for x in policy["feature_mean"]]
    std = [float(x) for x in policy["feature_std"]]
    predictions: dict[str, float] = {}
    for variant in variants:
        vector = _feature_vector(proxies_by_variant[variant], feature_names)
        pool = policy["entries_by_variant"][variant]
        ranked = sorted(
            pool,
            key=lambda entry: _normalized_distance(vector, entry["vector"], mean, std),
        )
        k = max(1, min(int(policy.get("k", 3)), len(ranked)))
        predictions[variant] = _mean([float(entry["score"]) for entry in ranked[:k]])
    best_variant, best_score = max(predictions.items(), key=lambda item: item[1])
    if best_score < float(policy.get("min_predicted_score", 0.0)):
        return "noop", predictions
    return best_variant, predictions


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

    candidate_rows: dict[str, list[dict[str, float]]] = {"fixed": [], "learned": [], "hybrid": []}
    per_view: list[dict[str, Any]] = []
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
        variant: _summarize_rows(rows, compute_ssim=bool(args.compute_ssim))
        for variant, rows in candidate_rows.items()
    }
    fixed_summary = summaries["fixed"]
    passing = [
        variant
        for variant in ["learned", "hybrid"]
        if _candidate_passes_guard(
            summaries[variant],
            fixed_summary,
            compute_ssim=bool(args.compute_ssim),
            min_psnr_delta=float(args.selector_min_vs_fixed_psnr_delta),
            min_ssim_delta=float(args.selector_min_vs_fixed_ssim_delta),
        )
    ]
    if passing:
        selected_variant = max(
            passing,
            key=lambda name: _selection_objective(summaries[name], compute_ssim=bool(args.compute_ssim)),
        )
        verdict = "source-heldout selector found a learned candidate that beats fixed on the guard axes"
    else:
        selected_variant = "fixed"
        verdict = "source-heldout selector fell back to fixed because learned candidates did not clear the guard"
    return {
        "selected_variant": selected_variant,
        "val_views": int(len(selector_val_frames)),
        "summaries": summaries,
        "per_view": per_view,
        "verdict": verdict,
        "min_vs_fixed_psnr_delta": float(args.selector_min_vs_fixed_psnr_delta),
        "min_vs_fixed_ssim_delta": float(args.selector_min_vs_fixed_ssim_delta),
    }


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
    fixed_rows: list[dict[str, float]] = []
    learned_rows: list[dict[str, float]] = []
    hybrid_rows: list[dict[str, float]] = []
    selected_rows: list[dict[str, float]] = []
    per_view: list[dict[str, Any]] = []
    nooped_views = 0

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
            gate_accepted = True
            output_variant = selected_variant
            if bool(per_view_knn_payload.get("enabled", False)):
                output_variant, knn_predictions = _knn_choose_variant(
                    proxies_by_variant,
                    per_view_knn_payload,
                    compute_ssim=bool(args.compute_ssim),
                )
                selected_delta = torch.zeros_like(selected_delta) if output_variant == "noop" else deltas[output_variant]
                selected_proxy = proxies_by_variant[selected_variant] if output_variant == "noop" else proxies_by_variant[output_variant]
                gate_accepted = output_variant != "noop"
            elif bool(per_view_gate_payload.get("enabled", False)):
                gate_accepted = _gate_accepts(selected_proxy, per_view_gate_payload)
            if not gate_accepted and output_variant != "noop":
                selected_delta = torch.zeros_like(selected_delta)
                output_variant = "noop"
                nooped_views += 1
            elif output_variant == "noop":
                nooped_views += 1
            selected_image = torch.clamp(ev.base + selected_delta, 0.0, 1.0)
            save_image_tensor(selected_image, render_dir / f"{target.name}.png")
            if bool(args.copy_gt):
                shutil.copy2(target.gt_path, gt_dir / f"{target.name}{target.gt_path.suffix}")

            gt = loader.gt(str(target.gt_path)).to(device=device, dtype=torch.float32)
            fixed_row = _image_metrics(ev.base, gt, fixed_delta, compute_ssim=bool(args.compute_ssim), ssim_max_side=int(args.ssim_max_side))
            learned_row = _image_metrics(ev.base, gt, learned_delta, compute_ssim=bool(args.compute_ssim), ssim_max_side=int(args.ssim_max_side))
            hybrid_row = _image_metrics(ev.base, gt, hybrid_delta, compute_ssim=bool(args.compute_ssim), ssim_max_side=int(args.ssim_max_side))
            selected_row = _image_metrics(ev.base, gt, selected_delta, compute_ssim=bool(args.compute_ssim), ssim_max_side=int(args.ssim_max_side))
            for row in [fixed_row, learned_row, hybrid_row]:
                row["view"] = target.name
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
                    "selected": selected_row,
                    "selected_variant": selected_variant,
                    "output_variant": output_variant,
                    "per_view_gate_accepted": bool(gate_accepted),
                    "per_view_gate_proxy": selected_proxy,
                    "per_view_knn_predictions": knn_predictions,
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
        "source_heldout_knn"
        if bool(per_view_knn_payload.get("enabled", False))
        else ("source_heldout_threshold" if bool(per_view_gate_payload.get("enabled", False)) else "off")
    )
    output_label = f"scene `{selected_variant}`"
    if active_gate != "off":
        output_label = f"scene `{selected_variant}` with per-view `{active_gate}` refinement"
    verdict = (
        f"Selected support-transport output {output_label} is all-axis safe versus fixed on this target/test split."
        if selected_all_axis_safe_vs_fixed
        else f"Selected support-transport output {output_label} is not all-axis safe versus fixed on this target/test split."
    )
    payload: dict[str, Any] = {
        "method": "apply v302 constrained hybrid support-transport calibrator",
        "target_gt_usage": "GT read only after candidate images are saved, for evaluation",
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
            "output_variant": str(args.output_variant),
            "selected_variant": selected_variant,
            "per_view_gate_mode": active_gate,
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
        },
        "per_view": per_view,
        "selector": selector_payload,
        "per_view_gate": per_view_gate_payload,
        "per_view_knn_policy": per_view_knn_payload,
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
    parser.add_argument("--output_variant", choices=["fixed", "learned", "hybrid", "source_heldout_auto"], default="hybrid")
    parser.add_argument("--selector_val_stride", type=int, default=3)
    parser.add_argument("--selector_val_offset", type=int, default=0)
    parser.add_argument("--max_selector_views", type=int, default=0)
    parser.add_argument("--selector_min_vs_fixed_psnr_delta", type=float, default=0.0)
    parser.add_argument("--selector_min_vs_fixed_ssim_delta", type=float, default=0.0)
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
    parser.add_argument("--per_view_knn_min_source_psnr_delta", type=float, default=0.0)
    parser.add_argument("--per_view_knn_require_source_safe", action="store_true")
    parser.add_argument("--per_view_knn_allow_when_scene_fixed", action="store_true")
    parser.add_argument("--residual_clip", type=float, default=0.25)
    parser.add_argument("--min_confidence", type=float, default=1.0e-4)
    parser.add_argument("--depth_abs_tol", type=float, default=0.02)
    parser.add_argument("--depth_rel_tol", type=float, default=0.03)
    parser.add_argument("--direction_weight", type=float, default=0.35)
    parser.add_argument("--evidence_max_side", type=int, default=512)
    parser.add_argument("--compute_ssim", action="store_true")
    parser.add_argument("--ssim_max_side", type=int, default=384)
    parser.add_argument("--min_hybrid_vs_fixed_psnr_delta", type=float, default=0.0)
    parser.add_argument("--min_hybrid_vs_fixed_ssim_delta", type=float, default=0.0)
    parser.add_argument("--save_example_views", type=int, default=0)
    parser.add_argument("--copy_gt", action="store_true")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet-transport-diagnostics")
    parser.add_argument("--wandb_run_name", default="apply-v302-support-transport")
    args = parser.parse_args()
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
