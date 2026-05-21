#!/usr/bin/env python3
"""Build a train-val selected Phase-S lowpass policy portfolio.

The script starts from an existing portfolio JSON and considers a fixed set of
lowpass candidate decision JSON templates. Candidate promotion uses train-val
metrics only; held-out test deltas are copied as report-only evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("PSNR", "SSIM", "LPIPS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_portfolio_json", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--scenes", default="")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate spec as label=decision_json_template, with {scene} allowed.",
    )
    parser.add_argument("--min_trainval_balanced_improvement", type=float, default=0.0)
    parser.add_argument("--min_trainval_balanced_delta", type=float, default=0.0)
    parser.add_argument("--min_trainval_psnr_gain", type=float, default=2e-5)
    parser.add_argument("--max_trainval_ssim_regression", type=float, default=1.5e-5)
    parser.add_argument("--max_trainval_lpips_regression", type=float, default=5e-6)
    parser.add_argument("--reject_no_op_operator", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.candidate:
        parser.error("at least one --candidate is required")
    for name in (
        "min_trainval_balanced_improvement",
        "min_trainval_balanced_delta",
        "min_trainval_psnr_gain",
        "max_trainval_ssim_regression",
        "max_trainval_lpips_regression",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and >= 0")
    return args


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scene_list(raw: str, rows: list[dict[str, Any]]) -> list[str]:
    explicit = [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]
    if explicit:
        return explicit
    return [str(row.get("scene")) for row in rows if str(row.get("scene", "")).strip()]


def path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def metric_block(payload: dict[str, Any] | None) -> dict[str, float]:
    payload = payload or {}
    out: dict[str, float] = {}
    for key in METRICS:
        try:
            value = float(payload.get(key))
        except Exception:
            value = math.nan
        out[key] = value if math.isfinite(value) else math.nan
    return out


def num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def parse_candidate_specs(raw_specs: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"invalid candidate spec: {raw}")
        label, template = raw.split("=", 1)
        label = label.strip()
        template = template.strip()
        if not label or not template:
            raise ValueError(f"invalid candidate spec: {raw}")
        out.append((label, template))
    return out


def audit_path_from_decision(path: Path, scene: str) -> Path:
    root = path.parent.parent
    return root / scene / "model" / "surface_residual_facelocal_sh1_delta_audit.json"


def candidate_row(
    label: str,
    template: str,
    scene: str,
    *,
    base_trainval_balanced: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = ROOT / template.format(scene=scene)
    decision = read_json(path)
    audit = read_json(audit_path_from_decision(path, scene))
    train = metric_block(decision.get("trainval_delta"))
    test = metric_block(decision.get("test_delta_report_only"))
    train_balanced = num(decision.get("trainval_balanced_delta"), -math.inf)
    reasons: list[str] = []
    if not decision:
        reasons.append("decision_missing")
    if not bool(decision.get("accepted", False)):
        reasons.append("inner_gate_rejected")
    if train_balanced < float(args.min_trainval_balanced_delta):
        reasons.append(f"trainval_balanced_below_{args.min_trainval_balanced_delta:g}")
    if train_balanced < base_trainval_balanced + float(args.min_trainval_balanced_improvement):
        reasons.append("trainval_balanced_not_above_base")
    if train["PSNR"] < float(args.min_trainval_psnr_gain):
        reasons.append(f"trainval_psnr_below_{args.min_trainval_psnr_gain:g}")
    if train["SSIM"] < -float(args.max_trainval_ssim_regression):
        reasons.append(f"trainval_ssim_regression_exceeds_{args.max_trainval_ssim_regression:g}")
    if train["LPIPS"] > float(args.max_trainval_lpips_regression):
        reasons.append(f"trainval_lpips_regression_exceeds_{args.max_trainval_lpips_regression:g}")
    if bool(args.reject_no_op_operator) and bool(audit.get("no_op_copy", False)):
        reasons.append("operator_no_op_copy")
    return {
        "scene": scene,
        "candidate_label": label,
        "decision_path": path_label(path),
        "operator_audit_path": path_label(audit_path_from_decision(path, scene)),
        "accepted": bool(decision.get("accepted", False)),
        "selector_pass": not reasons,
        "selector_reasons": reasons,
        "selected_label": decision.get("selected_label", ""),
        "trainval_delta": train,
        "trainval_balanced_delta": train_balanced,
        "test_delta_report_only": test,
        "test_balanced_delta_report_only": num(decision.get("test_balanced_delta_report_only"), math.nan),
        "accepted_faces": int(audit.get("accepted_faces", 0) or 0),
        "vertices_added": int(audit.get("vertices_added", 0) or 0),
        "coefficient_lowpass": audit.get("coefficient_lowpass", {}),
    }


def effective_delta(row: dict[str, Any]) -> dict[str, float]:
    return metric_block(row.get("effective_test_delta_report_only") or row.get("test_delta_report_only"))


def balanced_from_delta(delta: dict[str, float]) -> float:
    return float(delta["PSNR"] + 100.0 * delta["SSIM"] - 10.0 * delta["LPIPS"])


def stored_or_fallback_balanced(row: dict[str, Any], delta: dict[str, float]) -> float:
    for key in ("effective_test_balanced_delta_report_only", "test_balanced_delta_report_only"):
        value = row.get(key)
        try:
            out = float(value)
        except Exception:
            continue
        if math.isfinite(out):
            return out
    return balanced_from_delta(delta)


def fmt(value: Any, digits: int = 9) -> str:
    try:
        out = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(out):
        return "n/a"
    return f"{out:+.{digits}f}"


def main() -> int:
    args = parse_args()
    base_path = ROOT / args.base_portfolio_json
    base = read_json(base_path)
    base_rows = base.get("rows") if isinstance(base.get("rows"), list) else []
    base_by_scene = {str(row.get("scene")): row for row in base_rows}
    specs = parse_candidate_specs(args.candidate)
    rows: list[dict[str, Any]] = []
    for scene in scene_list(args.scenes, base_rows):
        base_row = dict(base_by_scene.get(scene, {"scene": scene}))
        base_train_balanced = num(base_row.get("selected_trainval_balanced_delta"), -math.inf)
        candidates = [
            candidate_row(label, template, scene, base_trainval_balanced=base_train_balanced, args=args)
            for label, template in specs
        ]
        promotable = [row for row in candidates if row["selector_pass"]]
        best = max(promotable, key=lambda row: float(row["trainval_balanced_delta"])) if promotable else None
        if best is None:
            effective = effective_delta(base_row)
            effective_balanced = stored_or_fallback_balanced(base_row, effective)
            selected_source = "base_portfolio"
            selected_label = base_row.get("selected_label", "")
            train_delta = metric_block(base_row.get("selected_trainval_delta"))
            train_balanced = base_train_balanced
            accepted = bool(base_row.get("accepted", False))
            lowpass = {}
        else:
            effective = metric_block(best.get("test_delta_report_only"))
            effective_balanced = stored_or_fallback_balanced(best, effective)
            selected_source = str(best["candidate_label"])
            selected_label = best.get("selected_label", "")
            train_delta = metric_block(best.get("trainval_delta"))
            train_balanced = float(best["trainval_balanced_delta"])
            accepted = True
            lowpass = best.get("coefficient_lowpass", {})
        rows.append(
            {
                "scene": scene,
                "selection_uses_test": False,
                "selected_source": selected_source,
                "selected_label": selected_label,
                "accepted": accepted,
                "selected_trainval_delta": train_delta,
                "selected_trainval_balanced_delta": train_balanced,
                "effective_test_delta_report_only": effective,
                "effective_test_balanced_delta_report_only": effective_balanced,
                "base_selected_label": base_row.get("selected_label", ""),
                "base_trainval_balanced_delta": base_train_balanced,
                "base_effective_test_delta_report_only": effective_delta(base_row),
                "base_effective_test_balanced_delta_report_only": stored_or_fallback_balanced(
                    base_row, effective_delta(base_row)
                ),
                "lowpass": lowpass,
                "candidates": candidates,
            }
        )
    scene_count = len(rows)
    mean_delta = {
        key: sum(float(row["effective_test_delta_report_only"][key]) for row in rows) / scene_count
        for key in METRICS
    }
    payload = {
        "selection_uses_test": False,
        "base_portfolio_json": path_label(base_path),
        "candidate_specs": [{"label": label, "path_template": template} for label, template in specs],
        "thresholds": {
            "min_trainval_balanced_improvement": float(args.min_trainval_balanced_improvement),
            "min_trainval_balanced_delta": float(args.min_trainval_balanced_delta),
            "min_trainval_psnr_gain": float(args.min_trainval_psnr_gain),
            "max_trainval_ssim_regression": float(args.max_trainval_ssim_regression),
            "max_trainval_lpips_regression": float(args.max_trainval_lpips_regression),
            "reject_no_op_operator": bool(args.reject_no_op_operator),
        },
        "scene_count": scene_count,
        "lowpass_promoted_count": sum(1 for row in rows if row["selected_source"] != "base_portfolio"),
        "accepted_count": sum(1 for row in rows if row["accepted"]),
        "mean_effective_test_delta_report_only": mean_delta,
        "mean_effective_test_balanced_delta_report_only": sum(
            float(row["effective_test_balanced_delta_report_only"]) for row in rows
        )
        / scene_count,
        "rows": rows,
    }
    output_prefix = ROOT / args.output_prefix
    write_json(output_prefix.with_suffix(".json"), payload)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with output_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "selected_source", "selected_label", "trainval_balanced", "test_dPSNR", "test_dSSIM", "test_dLPIPS", "test_balanced"])
        for row in rows:
            delta = row["effective_test_delta_report_only"]
            writer.writerow(
                [
                    row["scene"],
                    row["selected_source"],
                    row["selected_label"],
                    f"{float(row['selected_trainval_balanced_delta']):.12g}",
                    f"{float(delta['PSNR']):.12g}",
                    f"{float(delta['SSIM']):.12g}",
                    f"{float(delta['LPIPS']):.12g}",
                    f"{float(row['effective_test_balanced_delta_report_only']):.12g}",
                ]
            )
    lines = [
        "# Phase-S Lowpass Policy Portfolio",
        "",
        "Selection uses train-val render gates only; held-out test deltas are report-only.",
        "",
        f"- base portfolio: `{path_label(base_path)}`",
        f"- scenes: `{scene_count}`",
        f"- lowpass promoted scenes: `{payload['lowpass_promoted_count']}`",
        f"- accepted scenes: `{payload['accepted_count']}`",
        f"- mean report-only dPSNR: `{fmt(mean_delta['PSNR'])}`",
        f"- mean report-only dSSIM: `{fmt(mean_delta['SSIM'])}`",
        f"- mean report-only dLPIPS: `{fmt(mean_delta['LPIPS'])}`",
        f"- mean report-only balanced: `{fmt(payload['mean_effective_test_balanced_delta_report_only'])}`",
        "",
        "| scene | selected source | selected label | train-val balanced | test dPSNR | test dSSIM | test dLPIPS | test balanced |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["effective_test_delta_report_only"]
        lines.append(
            f"| {row['scene']} | `{row['selected_source']}` | `{row['selected_label']}` | "
            f"{fmt(row['selected_trainval_balanced_delta'])} | {fmt(delta['PSNR'])} | "
            f"{fmt(delta['SSIM'])} | {fmt(delta['LPIPS'])} | "
            f"{fmt(row['effective_test_balanced_delta_report_only'])} |"
        )
    output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_prefix": path_label(output_prefix), "rows": scene_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
