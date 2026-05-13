#!/usr/bin/env python3
"""Summarize Phase-S face-local render-calibrated gate decisions.

This is a read-only reporting helper for small Phase-S sweeps where each scene
has an independently named output root, for example:

outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_{scene}/decisions/{scene}_decision.json

Acceptance still comes from the train-policy-val decision JSON. Held-out test
deltas are report-only; this collector reports an effective test delta that is
zero for rejected scenes because those rows fall back to the baseline label.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("PSNR", "SSIM", "LPIPS")
DEFAULT_VARIANT = "facelocal_rendercalib_v1_top1_s2_fairreplay_20260513"
DEFAULT_DECISION_TEMPLATE = (
    "outputs/carnet/meshsplatopt/ecsr_phase_s/"
    f"{DEFAULT_VARIANT}_{{scene}}/decisions/{{scene}}_decision.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/carnet/meshsplatopt/ecsr_phase_s") / f"{DEFAULT_VARIANT}_summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant_label", default=DEFAULT_VARIANT)
    parser.add_argument(
        "--decision_path_template",
        default=DEFAULT_DECISION_TEMPLATE,
        help="Decision JSON path template. Use {scene}; {variant_label} is also available.",
    )
    parser.add_argument(
        "--scenes",
        default="bicycle,flowers",
        help="Comma and/or whitespace separated scene names.",
    )
    parser.add_argument("--strict_missing", action="store_true", help="Fail if any requested decision JSON is missing.")
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_ROOT / "summary.json")
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_ROOT / "summary.md")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def parse_scenes(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace(",", " ").split() if item.strip()]


def num(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def fmt(value: Any, digits: int = 9) -> str:
    v = num(value)
    return "n/a" if not math.isfinite(v) else f"{v:+.{digits}f}"


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def metric_block(source: dict[str, Any] | None) -> dict[str, float]:
    source = source or {}
    return {key: num(source.get(key)) for key in METRICS}


def effective_report_only_delta(decision: dict[str, Any]) -> dict[str, float]:
    if not bool(decision.get("accepted")):
        return {key: 0.0 for key in METRICS}
    return metric_block(decision.get("test_delta_report_only"))


def operator_audit_status(decision: dict[str, Any]) -> dict[str, Any]:
    audit = decision.get("candidate_operator_audit") or {}
    available = bool(audit.get("available", False))
    accepted = audit.get("accepted")
    no_op_copy = audit.get("no_op_copy")
    if not available:
        status = "missing"
    elif accepted is False or no_op_copy is True:
        status = "rejected_or_noop"
    else:
        status = "ok"
    return {
        "status": status,
        "available": available,
        "accepted": accepted,
        "no_op_copy": no_op_copy,
        "policy_pass": audit.get("policy_pass"),
        "path": audit.get("path", ""),
    }


def decision_path(template: str, scene: str, variant_label: str) -> Path:
    return Path(template.format(scene=scene, variant_label=variant_label))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scene in parse_scenes(args.scenes):
        path = decision_path(args.decision_path_template, scene, args.variant_label)
        decision = load_json(path)
        if not decision and args.strict_missing:
            raise FileNotFoundError(path)
        row: dict[str, Any] = {
            "scene": scene,
            "decision_path": rel(path),
            "present": bool(decision),
        }
        if decision:
            row.update(
                {
                    "accepted": bool(decision.get("accepted")),
                    "selected_label": decision.get("selected_label", ""),
                    "candidate_label": decision.get("candidate_label", ""),
                    "fallback_label": decision.get("fallback_label", ""),
                    "selection_uses_test": bool(decision.get("selection_uses_test", False)),
                    "decision_reasons": decision.get("decision_reasons", []),
                    "trainval_delta": metric_block(decision.get("trainval_delta")),
                    "report_only_test_delta": metric_block(decision.get("test_delta_report_only")),
                    "effective_report_only_test_delta": effective_report_only_delta(decision),
                    "test_balanced_delta_report_only": num(decision.get("test_balanced_delta_report_only")),
                    "trainval_balanced_delta": num(decision.get("trainval_balanced_delta")),
                    "base_test_method_report_only": decision.get("base_test_method_report_only", ""),
                    "candidate_test_method_report_only": decision.get("candidate_test_method_report_only", ""),
                    "operator_audit": operator_audit_status(decision),
                }
            )
        else:
            row.update(
                {
                    "accepted": False,
                    "selected_label": "",
                    "candidate_label": "",
                    "fallback_label": "",
                    "selection_uses_test": False,
                    "decision_reasons": ["missing_decision_json"],
                    "trainval_delta": {key: math.nan for key in METRICS},
                    "report_only_test_delta": {key: math.nan for key in METRICS},
                    "effective_report_only_test_delta": {key: 0.0 for key in METRICS},
                    "test_balanced_delta_report_only": math.nan,
                    "trainval_balanced_delta": math.nan,
                    "base_test_method_report_only": "",
                    "candidate_test_method_report_only": "",
                    "operator_audit": {"status": "missing_decision", "available": False},
                }
            )
        rows.append(row)

    present_rows = [row for row in rows if row["present"]]
    accepted_rows = [row for row in present_rows if row["accepted"]]
    payload = {
        "variant_label": args.variant_label,
        "decision_path_template": args.decision_path_template,
        "requested_scene_count": len(rows),
        "present_scene_count": len(present_rows),
        "missing_scene_count": len(rows) - len(present_rows),
        "accepted_count": len(accepted_rows),
        "rejected_count": len(present_rows) - len(accepted_rows),
        "mean_effective_report_only_test_delta": {
            key: mean([row["effective_report_only_test_delta"][key] for row in present_rows])
            for key in METRICS
        },
        "mean_accepted_report_only_test_delta": {
            key: mean([row["report_only_test_delta"][key] for row in accepted_rows])
            for key in METRICS
        },
        "operator_audit_status_counts": {
            status: sum(1 for row in present_rows if row["operator_audit"].get("status") == status)
            for status in sorted({row["operator_audit"].get("status") for row in present_rows})
        },
        "rows": rows,
    }
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    eff = payload["mean_effective_report_only_test_delta"]
    accepted = payload["mean_accepted_report_only_test_delta"]
    lines = [
        f"# Phase-S Face-Local Render-Calibrated Summary: {payload['variant_label']}",
        "",
        "Acceptance is copied from the train-policy-val decision JSON. Held-out test deltas are report-only; effective report-only deltas set rejected present scenes to zero because they fall back to the baseline label.",
        "",
        f"- requested scenes: `{payload['requested_scene_count']}`",
        f"- present scenes: `{payload['present_scene_count']}`",
        f"- missing scenes: `{payload['missing_scene_count']}`",
        f"- accepted scenes: `{payload['accepted_count']}`",
        f"- rejected present scenes: `{payload['rejected_count']}`",
        f"- mean effective report-only dPSNR: `{fmt(eff.get('PSNR'))}`",
        f"- mean effective report-only dSSIM: `{fmt(eff.get('SSIM'))}`",
        f"- mean effective report-only dLPIPS: `{fmt(eff.get('LPIPS'))}`",
        f"- mean accepted report-only dPSNR: `{fmt(accepted.get('PSNR'))}`",
        f"- mean accepted report-only dSSIM: `{fmt(accepted.get('SSIM'))}`",
        f"- mean accepted report-only dLPIPS: `{fmt(accepted.get('LPIPS'))}`",
        "",
        "| scene | present | accepted | selected | operator | train-val dPSNR | train-val dSSIM | train-val dLPIPS | report test dPSNR | report test dSSIM | report test dLPIPS | effective dPSNR | effective dSSIM | effective dLPIPS | reasons |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        train = row["trainval_delta"]
        test = row["report_only_test_delta"]
        effective = row["effective_report_only_test_delta"]
        lines.append(
            f"| {row['scene']} | {str(row['present']).lower()} | {str(row['accepted']).lower()} | "
            f"{row['selected_label'] or 'n/a'} | {row['operator_audit'].get('status', 'n/a')} | "
            f"{fmt(train.get('PSNR'))} | {fmt(train.get('SSIM'))} | {fmt(train.get('LPIPS'))} | "
            f"{fmt(test.get('PSNR'))} | {fmt(test.get('SSIM'))} | {fmt(test.get('LPIPS'))} | "
            f"{fmt(effective.get('PSNR'))} | {fmt(effective.get('SSIM'))} | {fmt(effective.get('LPIPS'))} | "
            f"{', '.join(row['decision_reasons']) or 'pass'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload = collect(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(json_safe(payload), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_md, payload)
    print(
        json.dumps(
            {
                "variant_label": payload["variant_label"],
                "present_scene_count": payload["present_scene_count"],
                "accepted_count": payload["accepted_count"],
                "output_json": rel(args.output_json),
                "output_md": rel(args.output_md),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
