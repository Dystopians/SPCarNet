#!/usr/bin/env python3
"""Decide whether a representation-level candidate may replace Phase-J.

The gate uses train-heldout render metrics only. Held-out test metrics may be
included in the report as an audit, but they do not affect the decision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")


def _load_metric(path: Path, method: str) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method not in payload:
        raise KeyError(f"{method} not found in {path}")
    row = payload[method]
    return {key: float(row[key]) for key in METRICS}


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path or str(path) in ("", ".") or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(candidate: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - base[key]) for key in METRICS}


def _balanced(delta: dict[str, float], *, ssim_weight: float, lpips_weight: float) -> float:
    return float(delta["PSNR"] + float(ssim_weight) * delta["SSIM"] - float(lpips_weight) * delta["LPIPS"])


def _load_per_view(path: Path, method: str) -> dict[str, dict[str, float]]:
    if not path or str(path) in ("", ".") or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload.get(method) if isinstance(payload, dict) else None
    if not isinstance(row, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = row.get(metric)
        if not isinstance(values, dict):
            continue
        for view_name, value in values.items():
            try:
                out.setdefault(str(view_name), {})[metric] = float(value)
            except Exception:
                continue
    return {key: value for key, value in out.items() if all(metric in value for metric in METRICS)}


def _cvar(values: list[float], fraction: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    count = max(1, int(math.ceil(len(finite) * max(min(float(fraction), 1.0), 1e-6))))
    return float(sum(finite[:count]) / count)


def _per_view_tail(args: argparse.Namespace) -> dict[str, Any]:
    base = _load_per_view(args.base_trainval_per_view, args.base_trainval_method)
    cand = _load_per_view(args.candidate_trainval_per_view, args.candidate_trainval_method)
    keys = sorted(set(base) & set(cand))
    rows: list[dict[str, Any]] = []
    for key in keys:
        delta = _delta(cand[key], base[key])
        rows.append(
            {
                "view_name": key,
                "delta": delta,
                "balanced_delta": _balanced(
                    delta,
                    ssim_weight=float(args.ssim_weight),
                    lpips_weight=float(args.lpips_weight),
                ),
            }
        )
    if not rows:
        return {"available": False, "view_count": 0}
    balanced = [float(row["balanced_delta"]) for row in rows]
    psnr = [float(row["delta"]["PSNR"]) for row in rows]
    ssim = [float(row["delta"]["SSIM"]) for row in rows]
    lpips = [float(row["delta"]["LPIPS"]) for row in rows]
    view_count = len(rows)
    balanced_mean_delta = float(sum(balanced) / view_count)
    balanced_cvar_delta = _cvar(balanced, float(args.tail_cvar_fraction))
    balanced_cvar_loss = max(0.0, -balanced_cvar_delta) if math.isfinite(balanced_cvar_delta) else math.inf
    if balanced_cvar_loss <= 1e-12:
        mean_to_cvar_ratio = math.inf if balanced_mean_delta > 0.0 else 0.0
    else:
        mean_to_cvar_ratio = max(0.0, balanced_mean_delta) / balanced_cvar_loss
    worst_lpips_regression = float(max(lpips))
    return {
        "available": True,
        "view_count": int(view_count),
        "cvar_fraction": float(args.tail_cvar_fraction),
        "balanced_mean_delta": balanced_mean_delta,
        "mean_balanced_delta": balanced_mean_delta,
        "balanced_negative_fraction": float(sum(1 for value in balanced if value < 0.0) / view_count),
        "balanced_cvar_delta": balanced_cvar_delta,
        "balanced_cvar_loss": float(balanced_cvar_loss),
        "mean_to_cvar_ratio": float(mean_to_cvar_ratio),
        "psnr_negative_fraction": float(sum(1 for value in psnr if value < 0.0) / view_count),
        "ssim_negative_fraction": float(sum(1 for value in ssim if value < 0.0) / view_count),
        "lpips_positive_fraction": float(sum(1 for value in lpips if value > 0.0) / view_count),
        "lpips_worst_regression": worst_lpips_regression,
        "worst_lpips_regression": worst_lpips_regression,
        "worst_balanced_rows": sorted(rows, key=lambda row: float(row["balanced_delta"]))[:10],
    }


def _decision(delta: dict[str, float], balanced_delta: float, args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if delta["PSNR"] < float(args.min_psnr_gain):
        reasons.append(f"psnr_gain_below_{args.min_psnr_gain:g}")
    if delta["SSIM"] < -float(args.max_ssim_regression):
        reasons.append(f"ssim_regression_exceeds_{args.max_ssim_regression:g}")
    if delta["LPIPS"] > float(args.max_lpips_regression):
        reasons.append(f"lpips_regression_exceeds_{args.max_lpips_regression:g}")
    if balanced_delta < float(args.min_balanced_delta):
        reasons.append(f"balanced_delta_below_{args.min_balanced_delta:g}")
    return not reasons, reasons


def _tail_decision(tail: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if not bool(tail.get("available", False)):
        if bool(args.tail_require_available):
            return ["trainval_tail_unavailable"]
        return []
    reasons: list[str] = []
    if float(tail.get("balanced_negative_fraction", 0.0)) > float(args.tail_max_balanced_negative_fraction):
        reasons.append(f"tail_balanced_negative_fraction_exceeds_{args.tail_max_balanced_negative_fraction:g}")
    if float(tail.get("balanced_cvar_delta", 0.0)) < float(args.tail_min_balanced_cvar_delta):
        reasons.append(f"tail_balanced_cvar_below_{args.tail_min_balanced_cvar_delta:g}")
    if float(tail.get("lpips_positive_fraction", 0.0)) > float(args.tail_max_lpips_positive_fraction):
        reasons.append(f"tail_lpips_positive_fraction_exceeds_{args.tail_max_lpips_positive_fraction:g}")
    if float(tail.get("lpips_worst_regression", 0.0)) > float(args.tail_max_worst_lpips_regression):
        reasons.append(f"tail_worst_lpips_regression_exceeds_{args.tail_max_worst_lpips_regression:g}")
    return reasons


def _write_md(path: Path, audit: dict[str, Any]) -> None:
    train = audit["trainval_delta"]
    test = audit.get("test_delta_report_only")
    lines = [
        "# Phase-K Train-Val Representation Gate",
        "",
        f"- scene: `{audit['scene']}`",
        f"- candidate: `{audit['candidate_label']}`",
        f"- fallback: `{audit['fallback_label']}`",
        f"- selected: `{audit['selected_label']}`",
        f"- accepted: `{audit['accepted']}`",
        f"- train-val delta PSNR/SSIM/LPIPS: `{train['PSNR']:.9f}` / `{train['SSIM']:.9f}` / `{train['LPIPS']:.9f}`",
        f"- train-val balanced delta: `{audit['trainval_balanced_delta']:.9f}`",
        f"- min PSNR gain: `{audit['thresholds']['min_psnr_gain']}`",
        f"- max SSIM regression: `{audit['thresholds']['max_ssim_regression']}`",
        f"- max LPIPS regression: `{audit['thresholds']['max_lpips_regression']}`",
        f"- min balanced delta: `{audit['thresholds']['min_balanced_delta']}`",
        f"- decision reasons: `{', '.join(audit['decision_reasons']) or 'pass'}`",
    ]
    tail = audit.get("trainval_per_view_tail", {})
    if tail:
        lines.extend(
            [
                "",
                "Train-val per-view tail gate:",
                f"- available / views: `{tail.get('available', False)}` / `{tail.get('view_count', 0)}`",
                f"- balanced mean / CVaR delta: `{float(tail.get('balanced_mean_delta', 0.0)):.9f}` / `{float(tail.get('balanced_cvar_delta', 0.0)):.9f}`",
                f"- balanced negative fraction: `{float(tail.get('balanced_negative_fraction', 0.0)):.6f}`",
                f"- LPIPS positive fraction / worst regression: `{float(tail.get('lpips_positive_fraction', 0.0)):.6f}` / `{float(tail.get('lpips_worst_regression', 0.0)):.9f}`",
            ]
        )
    if test is not None:
        lines.extend(
            [
                "",
                "Held-out test metrics below are report-only and were not used for selection.",
                f"- test delta PSNR/SSIM/LPIPS: `{test['PSNR']:.9f}` / `{test['SSIM']:.9f}` / `{test['LPIPS']:.9f}`",
                f"- test balanced delta: `{audit['test_balanced_delta_report_only']:.9f}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--candidate_label", required=True)
    parser.add_argument("--fallback_label", default="phasej")
    parser.add_argument("--base_trainval_results", type=Path, required=True)
    parser.add_argument("--base_trainval_method", required=True)
    parser.add_argument("--candidate_trainval_results", type=Path, required=True)
    parser.add_argument("--candidate_trainval_method", required=True)
    parser.add_argument("--base_trainval_per_view", type=Path, default=Path(""))
    parser.add_argument("--candidate_trainval_per_view", type=Path, default=Path(""))
    parser.add_argument("--candidate_audit_json", type=Path, default=Path(""))
    parser.add_argument("--base_test_results", type=Path, default=Path(""))
    parser.add_argument("--base_test_method", default="")
    parser.add_argument("--candidate_test_results", type=Path, default=Path(""))
    parser.add_argument("--candidate_test_method", default="")
    parser.add_argument("--min_psnr_gain", type=float, default=0.0)
    parser.add_argument("--max_ssim_regression", type=float, default=5e-5)
    parser.add_argument("--max_lpips_regression", type=float, default=1.5e-4)
    parser.add_argument("--min_balanced_delta", type=float, default=0.0)
    parser.add_argument("--ssim_weight", type=float, default=20.0)
    parser.add_argument("--lpips_weight", type=float, default=20.0)
    parser.add_argument("--tail_require_available", action="store_true")
    parser.add_argument("--tail_cvar_fraction", type=float, default=0.20)
    parser.add_argument("--tail_max_balanced_negative_fraction", type=float, default=1.0)
    parser.add_argument("--tail_min_balanced_cvar_delta", type=float, default=-1.0e30)
    parser.add_argument("--tail_max_lpips_positive_fraction", type=float, default=1.0)
    parser.add_argument("--tail_max_worst_lpips_regression", type=float, default=1.0e30)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    args = parser.parse_args()

    base_train = _load_metric(args.base_trainval_results, args.base_trainval_method)
    cand_train = _load_metric(args.candidate_trainval_results, args.candidate_trainval_method)
    train_delta = _delta(cand_train, base_train)
    train_balanced_delta = _balanced(
        train_delta,
        ssim_weight=float(args.ssim_weight),
        lpips_weight=float(args.lpips_weight),
    )
    accepted, reasons = _decision(train_delta, train_balanced_delta, args)
    trainval_per_view_tail = _per_view_tail(args)
    tail_reasons = _tail_decision(trainval_per_view_tail, args)
    if tail_reasons:
        accepted = False
        reasons.extend(tail_reasons)
    candidate_audit = _load_optional_json(args.candidate_audit_json)
    if candidate_audit:
        audit_accepted = bool(candidate_audit.get("accepted", False))
        audit_no_op = bool(candidate_audit.get("no_op_copy", False))
        if audit_no_op or not audit_accepted:
            accepted = False
            reasons.append("candidate_checkpoint_operator_rejected_or_noop")

    audit: dict[str, Any] = {
        "scene": args.scene,
        "candidate_label": args.candidate_label,
        "fallback_label": args.fallback_label,
        "selected_label": args.candidate_label if accepted else args.fallback_label,
        "accepted": bool(accepted),
        "decision_reasons": reasons,
        "selection_uses_test": False,
        "base_trainval_method": args.base_trainval_method,
        "candidate_trainval_method": args.candidate_trainval_method,
        "base_trainval_metrics": base_train,
        "candidate_trainval_metrics": cand_train,
        "trainval_delta": train_delta,
        "trainval_balanced_delta": train_balanced_delta,
        "trainval_per_view_tail": trainval_per_view_tail,
        "thresholds": {
            "min_psnr_gain": float(args.min_psnr_gain),
            "max_ssim_regression": float(args.max_ssim_regression),
            "max_lpips_regression": float(args.max_lpips_regression),
            "min_balanced_delta": float(args.min_balanced_delta),
            "ssim_weight": float(args.ssim_weight),
            "lpips_weight": float(args.lpips_weight),
            "tail_require_available": bool(args.tail_require_available),
            "tail_cvar_fraction": float(args.tail_cvar_fraction),
            "tail_max_balanced_negative_fraction": float(args.tail_max_balanced_negative_fraction),
            "tail_min_balanced_cvar_delta": float(args.tail_min_balanced_cvar_delta),
            "tail_max_lpips_positive_fraction": float(args.tail_max_lpips_positive_fraction),
            "tail_max_worst_lpips_regression": float(args.tail_max_worst_lpips_regression),
        },
        "candidate_operator_audit": {
            "path": str(args.candidate_audit_json) if args.candidate_audit_json else "",
            "available": bool(candidate_audit),
            "accepted": bool(candidate_audit.get("accepted", False)) if candidate_audit else None,
            "no_op_copy": bool(candidate_audit.get("no_op_copy", False)) if candidate_audit else None,
            "policy_pass": bool(candidate_audit.get("policy_pass", False)) if candidate_audit else None,
        },
    }

    if args.base_test_results and args.candidate_test_results and args.base_test_method and args.candidate_test_method:
        base_test = _load_metric(args.base_test_results, args.base_test_method)
        cand_test = _load_metric(args.candidate_test_results, args.candidate_test_method)
        test_delta = _delta(cand_test, base_test)
        audit.update(
            {
                "base_test_method_report_only": args.base_test_method,
                "candidate_test_method_report_only": args.candidate_test_method,
                "base_test_metrics_report_only": base_test,
                "candidate_test_metrics_report_only": cand_test,
                "test_delta_report_only": test_delta,
                "test_balanced_delta_report_only": _balanced(
                    test_delta,
                    ssim_weight=float(args.ssim_weight),
                    lpips_weight=float(args.lpips_weight),
                ),
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(args.output_md, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
