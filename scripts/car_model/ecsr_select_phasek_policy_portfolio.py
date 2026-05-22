#!/usr/bin/env python3
"""Select a fixed Phase-K policy portfolio from train-val evidence only.

The script consumes already computed Phase-K decision JSON files from several
policy variants, for example plain ELA and edge-gated ELA.  It does not use
held-out test metrics for selection; test numbers are copied into the report as
audit-only evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _num(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _metric_score(row: dict[str, Any], *, ssim_weight: float, lpips_weight: float) -> float:
    return (
        _num(row.get("PSNR"), -math.inf)
        + float(ssim_weight) * _num(row.get("SSIM"), -math.inf)
        - float(lpips_weight) * _num(row.get("LPIPS"), math.inf)
    )


def _balanced_delta(delta: dict[str, Any], *, ssim_weight: float, lpips_weight: float) -> float:
    return (
        _num(delta.get("PSNR"), -math.inf)
        + float(ssim_weight) * _num(delta.get("SSIM"), -math.inf)
        - float(lpips_weight) * _num(delta.get("LPIPS"), math.inf)
    )


def _compact_gate(decision: dict[str, Any]) -> dict[str, Any]:
    gate = decision.get("compact_stratified_gate")
    return gate if isinstance(gate, dict) else {}


def _tail(decision: dict[str, Any]) -> dict[str, Any]:
    tail = decision.get("trainval_per_view_tail")
    return tail if isinstance(tail, dict) else {}


def _operator_ok(decision: dict[str, Any]) -> bool:
    audit = decision.get("candidate_operator_audit") or {}
    if not isinstance(audit, dict) or not bool(audit.get("available", False)):
        return False
    return bool(audit.get("accepted", False)) and not bool(audit.get("no_op_copy", False))


def _candidate_valid(decision: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    train_delta = decision.get("trainval_delta") or {}
    train_balanced = _num(
        decision.get("trainval_balanced_delta"),
        _balanced_delta(train_delta, ssim_weight=args.ssim_weight, lpips_weight=args.lpips_weight),
    )
    if not _operator_ok(decision):
        reasons.append("operator_rejected_or_missing")
    if train_balanced < float(args.min_trainval_balanced_delta):
        reasons.append("trainval_balanced_delta_too_low")
    if _num(train_delta.get("PSNR"), -math.inf) < float(args.min_trainval_psnr_delta):
        reasons.append("trainval_psnr_delta_too_low")
    if _num(train_delta.get("SSIM"), -math.inf) < -float(args.max_trainval_ssim_regression):
        reasons.append("trainval_ssim_regression_too_large")
    if _num(train_delta.get("LPIPS"), math.inf) > float(args.max_trainval_lpips_regression):
        reasons.append("trainval_lpips_regression_too_large")

    gate = _compact_gate(decision)
    accepted_faces = int(gate.get("accepted_faces", 0) or 0)
    vertices_added = int(gate.get("vertices_added", 0) or 0)
    face_ratio = _num(gate.get("face_ratio"), math.inf)
    if accepted_faces <= 0:
        reasons.append("no_faces_changed")
    if accepted_faces > int(args.max_faces):
        reasons.append("too_many_faces")
    if vertices_added > int(args.max_vertices):
        reasons.append("too_many_vertices")
    if face_ratio > float(args.max_face_ratio):
        reasons.append("face_ratio_too_high")

    tail = _tail(decision)
    if bool(args.require_tail) and not bool(tail.get("available", False)):
        reasons.append("tail_unavailable")
    if _num(tail.get("balanced_negative_fraction"), 1.0) > float(args.max_balanced_negative_fraction):
        reasons.append("too_many_negative_views")
    if _num(tail.get("balanced_cvar_delta"), -math.inf) < float(args.min_balanced_cvar_delta):
        reasons.append("tail_cvar_too_low")
    if _num(tail.get("lpips_positive_fraction"), 1.0) > float(args.max_lpips_positive_fraction):
        reasons.append("too_many_lpips_regressions")
    if _num(tail.get("lpips_worst_regression"), math.inf) > float(args.max_worst_lpips_regression):
        reasons.append("worst_lpips_regression_too_large")

    audit = {
        "trainval_balanced_delta": float(train_balanced),
        "accepted_faces": int(accepted_faces),
        "vertices_added": int(vertices_added),
        "face_ratio": float(face_ratio),
        "tail_balanced_negative_fraction": _num(tail.get("balanced_negative_fraction"), math.nan),
        "tail_balanced_cvar_delta": _num(tail.get("balanced_cvar_delta"), math.nan),
        "tail_lpips_positive_fraction": _num(tail.get("lpips_positive_fraction"), math.nan),
        "tail_worst_lpips_regression": _num(tail.get("lpips_worst_regression"), math.nan),
    }
    return not reasons, reasons, audit


def _candidate_score(decision: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    train = decision.get("candidate_trainval_metrics") or {}
    delta = decision.get("trainval_delta") or {}
    tail = _tail(decision)
    absolute = _metric_score(train, ssim_weight=args.ssim_weight, lpips_weight=args.lpips_weight)
    gain = _balanced_delta(delta, ssim_weight=args.ssim_weight, lpips_weight=args.lpips_weight)
    cvar_penalty = max(0.0, -_num(tail.get("balanced_cvar_delta"), 0.0))
    lpips_tail_penalty = max(0.0, _num(tail.get("lpips_positive_fraction"), 0.0) - float(args.lpips_positive_target))
    score = (
        absolute
        + float(args.gain_weight) * gain
        - float(args.cvar_penalty_weight) * cvar_penalty
        - float(args.lpips_tail_penalty_weight) * lpips_tail_penalty
    )
    return {
        "score": float(score),
        "absolute_trainval_score": float(absolute),
        "gain_score": float(gain),
        "cvar_penalty": float(cvar_penalty),
        "lpips_tail_penalty": float(lpips_tail_penalty),
    }


def _fallback_score(decision: dict[str, Any], args: argparse.Namespace) -> float:
    train = decision.get("base_trainval_metrics") or {}
    return _metric_score(train, ssim_weight=args.ssim_weight, lpips_weight=args.lpips_weight)


def _load_variant_decision(variant: str, root: Path, scene: str) -> dict[str, Any]:
    decision_path = root / "decisions" / f"{scene}_decision.json"
    decision = _read_json(decision_path)
    if not decision:
        raise FileNotFoundError(decision_path)
    return {"variant": variant, "decision_path": str(decision_path), "decision": decision}


def _select_scene(scene: str, variants: list[tuple[str, Path]], args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    valid_candidates: list[dict[str, Any]] = []
    fallback_options: list[dict[str, Any]] = []
    for variant, root in variants:
        row = _load_variant_decision(variant, root, scene)
        decision = row["decision"]
        valid, reject_reasons, validity_audit = _candidate_valid(decision, args)
        score = _candidate_score(decision, args)
        fallback = _fallback_score(decision, args)
        row.update(
            {
                "candidate_valid": bool(valid),
                "candidate_reject_reasons": reject_reasons,
                "validity_audit": validity_audit,
                "candidate_score": score,
                "fallback_trainval_score": float(fallback),
            }
        )
        rows.append(row)
        if valid:
            valid_candidates.append(row)
        fallback_options.append(row)

    if valid_candidates:
        selected = max(valid_candidates, key=lambda item: float(item["candidate_score"]["score"]))
        selected_kind = "candidate"
        selected_metrics = selected["decision"].get("candidate_test_metrics_report_only") or {}
        baseline_metrics = selected["decision"].get("base_test_metrics_report_only") or {}
        selected_method = selected["decision"].get("candidate_test_method_report_only", "")
        baseline_method = selected["decision"].get("base_test_method_report_only", "")
    else:
        selected = max(fallback_options, key=lambda item: float(item["fallback_trainval_score"]))
        selected_kind = "fallback"
        selected_metrics = selected["decision"].get("base_test_metrics_report_only") or {}
        baseline_metrics = selected_metrics
        selected_method = selected["decision"].get("base_test_method_report_only", "")
        baseline_method = selected_method

    effective_delta = {
        key: _num(selected_metrics.get(key), math.nan) - _num(baseline_metrics.get(key), math.nan)
        for key in METRICS
    }
    return {
        "scene": scene,
        "selected_variant": selected["variant"],
        "selected_kind": selected_kind,
        "selected_method_report_only": selected_method,
        "baseline_method_report_only": baseline_method,
        "selection_uses_test": False,
        "selected_test_metrics_report_only": selected_metrics,
        "baseline_test_metrics_report_only": baseline_metrics,
        "effective_test_delta_report_only": effective_delta,
        "selected_candidate_score": selected.get("candidate_score", {}),
        "selected_fallback_trainval_score": selected.get("fallback_trainval_score"),
        "variant_rows": rows,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [_num(row["effective_test_delta_report_only"].get(key)) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else math.nan


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase-K Train-Val Policy Portfolio",
        "",
        "Selection uses only train-val metrics, compact geometry limits, and per-view tail risk. Held-out test metrics are report-only.",
        "",
        f"- scenes: `{payload['scene_count']}`",
        f"- candidate selections: `{payload['candidate_selection_count']}`",
        f"- fallback selections: `{payload['fallback_selection_count']}`",
        f"- mean report-only effective dPSNR/dSSIM/dLPIPS: `{payload['mean_effective_test_delta']['PSNR']:.9f}` / `{payload['mean_effective_test_delta']['SSIM']:.9f}` / `{payload['mean_effective_test_delta']['LPIPS']:.9f}`",
        "",
        "| scene | selected variant | selected kind | report dPSNR | report dSSIM | report dLPIPS | train-val score | candidate reasons |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        delta = row["effective_test_delta_report_only"]
        selected_variant = row["selected_variant"]
        selected_score = row.get("selected_candidate_score") or {}
        selected_variant_row = next(
            (item for item in row["variant_rows"] if item["variant"] == selected_variant),
            {},
        )
        reasons = selected_variant_row.get("candidate_reject_reasons") or []
        lines.append(
            f"| {row['scene']} | {selected_variant} | {row['selected_kind']} | "
            f"{_num(delta.get('PSNR')):+.9f} | {_num(delta.get('SSIM')):+.9f} | {_num(delta.get('LPIPS')):+.9f} | "
            f"{_num(selected_score.get('score'), _num(row.get('selected_fallback_trainval_score'))):.9f} | "
            f"{', '.join(str(item) for item in reasons) or 'pass'} |"
        )
    lines.extend(
        [
            "",
            "Variant audit:",
            "",
            "| scene | variant | valid | score | abs score | gain score | faces | neg frac | CVaR | LPIPS+ frac | reasons |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        for item in row["variant_rows"]:
            audit = item["validity_audit"]
            score = item["candidate_score"]
            lines.append(
                f"| {row['scene']} | {item['variant']} | {str(item['candidate_valid']).lower()} | "
                f"{score['score']:.9f} | {score['absolute_trainval_score']:.9f} | {score['gain_score']:+.9f} | "
                f"{audit['accepted_faces']} | {_num(audit.get('tail_balanced_negative_fraction')):.6f} | "
                f"{_num(audit.get('tail_balanced_cvar_delta')):+.9f} | {_num(audit.get('tail_lpips_positive_fraction')):.6f} | "
                f"{', '.join(item['candidate_reject_reasons']) or 'pass'} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_variants(values: list[str]) -> list[tuple[str, Path]]:
    variants: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"variant must be name=path, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty variant name in {value!r}")
        variants.append((name, Path(path).expanduser()))
    if not variants:
        raise ValueError("at least one --variant is required")
    return variants


def _jsonable_args(args: argparse.Namespace, variants: list[tuple[str, Path]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key == "variant":
            continue
        payload[key] = str(value) if isinstance(value, Path) else value
    payload["variant"] = [f"{name}={path}" for name, path in variants]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, help="Variant mapping as name=/path/to/root.")
    parser.add_argument("--scenes", required=True, help="Comma-separated scene names.")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--ssim_weight", type=float, default=20.0)
    parser.add_argument("--lpips_weight", type=float, default=20.0)
    parser.add_argument("--gain_weight", type=float, default=1.0)
    parser.add_argument("--cvar_penalty_weight", type=float, default=1.0)
    parser.add_argument("--lpips_tail_penalty_weight", type=float, default=0.25)
    parser.add_argument("--lpips_positive_target", type=float, default=0.50)
    parser.add_argument("--min_trainval_balanced_delta", type=float, default=0.0)
    parser.add_argument("--min_trainval_psnr_delta", type=float, default=0.0)
    parser.add_argument("--max_trainval_ssim_regression", type=float, default=5.0e-5)
    parser.add_argument("--max_trainval_lpips_regression", type=float, default=1.5e-4)
    parser.add_argument("--max_faces", type=int, default=160)
    parser.add_argument("--max_vertices", type=int, default=512)
    parser.add_argument("--max_face_ratio", type=float, default=1.5e-5)
    parser.add_argument("--require_tail", action="store_true")
    parser.add_argument("--max_balanced_negative_fraction", type=float, default=0.75)
    parser.add_argument("--min_balanced_cvar_delta", type=float, default=-0.0012)
    parser.add_argument("--max_lpips_positive_fraction", type=float, default=0.75)
    parser.add_argument("--max_worst_lpips_regression", type=float, default=1.0e-4)
    args = parser.parse_args()

    variants = _parse_variants(args.variant)
    scenes = [item.strip() for item in args.scenes.replace(" ", ",").split(",") if item.strip()]
    rows = [_select_scene(scene, variants, args) for scene in scenes]
    payload = {
        "policy": "phasek_trainval_policy_portfolio_v1",
        "selection_uses_test": False,
        "scene_count": len(rows),
        "candidate_selection_count": sum(1 for row in rows if row["selected_kind"] == "candidate"),
        "fallback_selection_count": sum(1 for row in rows if row["selected_kind"] == "fallback"),
        "mean_effective_test_delta": {key: _mean(rows, key) for key in METRICS},
        "args": _jsonable_args(args, variants),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(args.output_md, payload)
    print(json.dumps({"scenes": len(rows), "output_md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
