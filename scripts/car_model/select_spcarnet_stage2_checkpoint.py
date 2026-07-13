#!/usr/bin/env python3
"""Select the best SP-CarNet Stage-2 checkpoint from eval JSON artifacts.

This is a report-side selector: it does not inspect held-out test images and
does not rerun evaluation. It reads strict eval outputs from
``eval_spcarnet_shape_field_autodecoder.py`` and writes a deterministic JSON
and Markdown audit that can be cited from docs or slides.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_WEIGHTS = {
    "recon_chamfer_l1_mean": 1.0,
    "hidden_chamfer_l1_mean": 0.25,
    "mesh_iou_at_0.5_mean": -0.05,
    "mesh_iou_at_0.5_shell_mean": -0.01,
    "surface_normal_consistency_mean": -0.01,
}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    checkpoint_path: str | None
    eval_json_path: Path


def _strict_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _parse_candidate(raw: str) -> CandidateSpec:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--candidate must be NAME:CHECKPOINT_PATH:EVAL_JSON_PATH; use '-' for an empty checkpoint path"
        )
    name, checkpoint, eval_json = parts
    if not name:
        raise argparse.ArgumentTypeError("candidate name cannot be empty")
    if not eval_json:
        raise argparse.ArgumentTypeError("candidate eval JSON path cannot be empty")
    checkpoint_path = None if checkpoint in {"", "-"} else checkpoint
    return CandidateSpec(name=name, checkpoint_path=checkpoint_path, eval_json_path=Path(eval_json))


def _parse_weights(raw_values: list[str]) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    for raw in raw_values:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(f"invalid --metric_weight {raw!r}; expected metric=value")
        key, value = raw.split("=", 1)
        try:
            weights[key] = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid weight for {key!r}: {value!r}") from exc
    return weights


def _load_eval_summary(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{path} does not contain a dict at top-level key 'summary'")
    return summary


def _score(summary: dict[str, Any], weights: dict[str, float], missing_penalty: float) -> tuple[float, list[str]]:
    total = 0.0
    missing: list[str] = []
    for key, weight in weights.items():
        value = _strict_float(summary.get(key))
        if value is None:
            missing.append(key)
            total += abs(weight) * missing_penalty
        else:
            total += weight * value
    success = _strict_float(summary.get("mesh_extraction_success_rate"))
    if success is None:
        missing.append("mesh_extraction_success_rate")
        total += missing_penalty
    else:
        total += max(0.0, 1.0 - success) * missing_penalty
    return total, missing


def _gate(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    extraction = _strict_float(summary.get("mesh_extraction_success_rate"))
    chamfer = _strict_float(summary.get("recon_chamfer_l1_mean"))
    filled_iou = _strict_float(summary.get("mesh_iou_at_0.5_mean"))
    checks = {
        "mesh_extraction_success_rate": {
            "value": extraction,
            "threshold": args.min_extraction_success,
            "op": ">=",
            "pass": extraction is not None and extraction >= args.min_extraction_success,
        },
        "recon_chamfer_l1_mean": {
            "value": chamfer,
            "threshold": args.max_recon_chamfer,
            "op": "<=",
            "pass": chamfer is not None and chamfer <= args.max_recon_chamfer,
        },
        "mesh_iou_at_0.5_mean": {
            "value": filled_iou,
            "threshold": args.min_filled_iou,
            "op": ">=",
            "pass": filled_iou is not None and filled_iou >= args.min_filled_iou,
        },
    }
    return {
        "pass": all(item["pass"] for item in checks.values()),
        "checks": checks,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    val = _strict_float(value)
    if val is None:
        return "NA"
    return f"{val:.{digits}f}"


def _candidate_rows(
    candidates: list[CandidateSpec], weights: dict[str, float], args: argparse.Namespace
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in candidates:
        summary = _load_eval_summary(spec.eval_json_path)
        score, missing = _score(summary, weights, args.missing_penalty)
        gate = _gate(summary, args)
        rows.append(
            {
                "name": spec.name,
                "checkpoint_path": spec.checkpoint_path,
                "eval_json_path": str(spec.eval_json_path),
                "score": score,
                "score_missing_metrics": missing,
                "gate": gate,
                "summary": {
                    "n_objects_evaluated": summary.get("n_objects_evaluated"),
                    "n_extracted": summary.get("n_extracted"),
                    "mesh_extraction_success_rate": _strict_float(
                        summary.get("mesh_extraction_success_rate")
                    ),
                    "recon_chamfer_l1_mean": _strict_float(summary.get("recon_chamfer_l1_mean")),
                    "hidden_chamfer_l1_mean": _strict_float(summary.get("hidden_chamfer_l1_mean")),
                    "mesh_iou_at_0.5_mean": _strict_float(summary.get("mesh_iou_at_0.5_mean")),
                    "mesh_iou_at_0.5_shell_mean": _strict_float(
                        summary.get("mesh_iou_at_0.5_shell_mean")
                    ),
                    "surface_normal_consistency_mean": _strict_float(
                        summary.get("surface_normal_consistency_mean")
                    ),
                },
            }
        )
    return rows


def _late_degradation_warnings(rows: list[dict[str, Any]], best: dict[str, Any], eps: float) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    best_score = float(best["score"])
    best_summary = best.get("summary", {})
    for row in rows:
        lname = str(row["name"]).lower()
        # Use the candidate label as the contract. Baselines may legitimately
        # point at their own checkpoint_last.pt, but that should not be reported
        # as late-training degradation of the current method.
        is_late = any(token in lname for token in ("final", "last"))
        if not is_late or row["name"] == best["name"]:
            continue
        reasons = []
        if float(row["score"]) > best_score + eps:
            reasons.append(f"score worse by {float(row['score']) - best_score:.9f}")
        summary = row.get("summary", {})
        lower_better = ("recon_chamfer_l1_mean", "hidden_chamfer_l1_mean")
        higher_better = (
            "mesh_iou_at_0.5_mean",
            "mesh_iou_at_0.5_shell_mean",
            "surface_normal_consistency_mean",
        )
        for key in lower_better:
            cur = _strict_float(summary.get(key))
            ref = _strict_float(best_summary.get(key))
            if cur is not None and ref is not None and cur > ref + eps:
                reasons.append(f"{key} worse ({cur:.9f} > {ref:.9f})")
        for key in higher_better:
            cur = _strict_float(summary.get(key))
            ref = _strict_float(best_summary.get(key))
            if cur is not None and ref is not None and cur + eps < ref:
                reasons.append(f"{key} worse ({cur:.9f} < {ref:.9f})")
        if reasons:
            warnings.append({"candidate": row["name"], "reasons": reasons})
    return warnings


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    best = report["best_candidate"]
    lines = [
        "# SP-CarNet Stage-2 Checkpoint Selection",
        "",
        f"Status: `{report['decision']['status']}`",
        "",
        "## Decision",
        "",
        f"- Best candidate: `{best['name']}`",
        f"- Best checkpoint: `{best.get('checkpoint_path') or 'NA'}`",
        f"- Best eval JSON: `{best['eval_json_path']}`",
        f"- Best score: `{best['score']:.9f}`",
        f"- Gate pass: `{best['gate']['pass']}`",
        "",
        report["decision"]["rationale"],
        "",
        "## Candidate Table",
        "",
        "| candidate | score | gate | extracted | recon chamfer | hidden chamfer | filled IoU | shell IoU | normal consistency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["ranked_candidates"]:
        s = row["summary"]
        extracted = f"{s.get('n_extracted', 'NA')}/{s.get('n_objects_evaluated', 'NA')}"
        lines.append(
            "| {name} | {score:.9f} | `{gate}` | {extracted} | {recon} | {hidden} | {filled} | {shell} | {normal} |".format(
                name=row["name"],
                score=float(row["score"]),
                gate=row["gate"]["pass"],
                extracted=extracted,
                recon=_fmt(s.get("recon_chamfer_l1_mean")),
                hidden=_fmt(s.get("hidden_chamfer_l1_mean")),
                filled=_fmt(s.get("mesh_iou_at_0.5_mean")),
                shell=_fmt(s.get("mesh_iou_at_0.5_shell_mean")),
                normal=_fmt(s.get("surface_normal_consistency_mean")),
            )
        )
    lines.extend(["", "## Gate", ""])
    lines.append(
        f"Default gate: extraction success >= `{report['gate_thresholds']['min_extraction_success']}`, "
        f"recon chamfer <= `{report['gate_thresholds']['max_recon_chamfer']}`, "
        f"filled IoU >= `{report['gate_thresholds']['min_filled_iou']}`."
    )
    lines.extend(["", "## Score", ""])
    lines.append("Lower score is better. Default score weights:")
    lines.append("")
    lines.append("| metric | weight | direction |")
    lines.append("|---|---:|---|")
    for key, weight in report["score_weights"].items():
        direction = "higher is better" if float(weight) < 0 else "lower is better"
        lines.append(f"| `{key}` | `{weight}` | {direction} |")
    if report["late_degradation_warnings"]:
        lines.extend(["", "## Late-Checkpoint Degradation Warnings", ""])
        for item in report["late_degradation_warnings"]:
            lines.append(f"- `{item['candidate']}`: " + "; ".join(item["reasons"]))
    lines.extend(["", "## Candidate Artifacts", ""])
    for row in report["ranked_candidates"]:
        lines.append(f"- `{row['name']}` eval: `{row['eval_json_path']}`")
        if row.get("checkpoint_path"):
            lines.append(f"  checkpoint: `{row['checkpoint_path']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    weights = _parse_weights(args.metric_weight)
    rows = _candidate_rows(args.candidate, weights, args)
    if not rows:
        raise ValueError("at least one --candidate is required")
    ranked = sorted(rows, key=lambda r: (float(r["score"]), str(r["name"])))
    best = ranked[0]
    warnings = _late_degradation_warnings(rows, best, args.tie_epsilon)
    status = "GATE_PASS" if best["gate"]["pass"] else "BEST_AVAILABLE_GATE_FAIL"
    if warnings:
        status += "_WITH_LATE_DEGRADATION"
    rationale = (
        f"`{best['name']}` has the lowest deterministic selector score under the configured weights. "
        "The Stage-2 quality gate is "
        + ("passed." if best["gate"]["pass"] else "not passed; this is a best-available checkpoint selection, not a headline quality pass.")
    )
    if warnings:
        rationale += " At least one final/last checkpoint is worse than the selected checkpoint, so validation-driven checkpoint selection is required."
    return _json_sanitize(
        {
            "schema": "spcarnet_stage2_checkpoint_selection_v1",
            "score_weights": weights,
            "gate_thresholds": {
                "min_extraction_success": args.min_extraction_success,
                "max_recon_chamfer": args.max_recon_chamfer,
                "min_filled_iou": args.min_filled_iou,
            },
            "decision": {"status": status, "rationale": rationale},
            "best_candidate": best,
            "ranked_candidates": ranked,
            "late_degradation_warnings": warnings,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        required=True,
        help="Candidate triple NAME:CHECKPOINT_PATH:EVAL_JSON_PATH. Use '-' for no checkpoint path.",
    )
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--metric_weight", action="append", default=[], help="Override score weight as metric=value")
    parser.add_argument("--missing_penalty", type=float, default=1.0)
    parser.add_argument("--tie_epsilon", type=float, default=1e-9)
    parser.add_argument("--min_extraction_success", type=float, default=0.95)
    parser.add_argument("--max_recon_chamfer", type=float, default=0.05)
    parser.add_argument("--min_filled_iou", type=float, default=0.92)
    args = parser.parse_args()

    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.output_md is not None:
        _write_markdown(report, args.output_md)
    print(json.dumps(report["decision"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
