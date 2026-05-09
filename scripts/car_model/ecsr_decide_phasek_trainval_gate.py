#!/usr/bin/env python3
"""Decide whether a representation-level candidate may replace Phase-J.

The gate uses train-heldout render metrics only. Held-out test metrics may be
included in the report as an audit, but they do not affect the decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")


def _load_metric(path: Path, method: str) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method not in payload:
        raise KeyError(f"{method} not found in {path}")
    row = payload[method]
    return {key: float(row[key]) for key in METRICS}


def _delta(candidate: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - base[key]) for key in METRICS}


def _balanced(delta: dict[str, float], *, ssim_weight: float, lpips_weight: float) -> float:
    return float(delta["PSNR"] + float(ssim_weight) * delta["SSIM"] - float(lpips_weight) * delta["LPIPS"])


def _decision(delta: dict[str, float], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if delta["PSNR"] < float(args.min_psnr_gain):
        reasons.append(f"psnr_gain_below_{args.min_psnr_gain:g}")
    if delta["SSIM"] < -float(args.max_ssim_regression):
        reasons.append(f"ssim_regression_exceeds_{args.max_ssim_regression:g}")
    if delta["LPIPS"] > float(args.max_lpips_regression):
        reasons.append(f"lpips_regression_exceeds_{args.max_lpips_regression:g}")
    return not reasons, reasons


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
        f"- decision reasons: `{', '.join(audit['decision_reasons']) or 'pass'}`",
    ]
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
    parser.add_argument("--base_test_results", type=Path, default=Path(""))
    parser.add_argument("--base_test_method", default="")
    parser.add_argument("--candidate_test_results", type=Path, default=Path(""))
    parser.add_argument("--candidate_test_method", default="")
    parser.add_argument("--min_psnr_gain", type=float, default=0.0)
    parser.add_argument("--max_ssim_regression", type=float, default=5e-5)
    parser.add_argument("--max_lpips_regression", type=float, default=1.5e-4)
    parser.add_argument("--ssim_weight", type=float, default=20.0)
    parser.add_argument("--lpips_weight", type=float, default=20.0)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    args = parser.parse_args()

    base_train = _load_metric(args.base_trainval_results, args.base_trainval_method)
    cand_train = _load_metric(args.candidate_trainval_results, args.candidate_trainval_method)
    train_delta = _delta(cand_train, base_train)
    accepted, reasons = _decision(train_delta, args)

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
        "trainval_balanced_delta": _balanced(
            train_delta,
            ssim_weight=float(args.ssim_weight),
            lpips_weight=float(args.lpips_weight),
        ),
        "thresholds": {
            "min_psnr_gain": float(args.min_psnr_gain),
            "max_ssim_regression": float(args.max_ssim_regression),
            "max_lpips_regression": float(args.max_lpips_regression),
            "ssim_weight": float(args.ssim_weight),
            "lpips_weight": float(args.lpips_weight),
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
