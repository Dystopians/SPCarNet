#!/usr/bin/env python3
"""Collect Phase-S render-visible region-prior decisions into one report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate one or more ecsr_run_phasek_barycentric_gate_scene.py output roots "
            "for the render-visible region-prior Phase-S method. Selection remains exactly "
            "whatever each Phase-K decision chose; this script only reports evidence."
        )
    )
    parser.add_argument("--scenes", required=True, help="Comma/space-separated scene names.")
    parser.add_argument(
        "--phasek_root",
        action="append",
        default=[],
        help="Phase-K output root containing decisions/{scene}_decision.json. Repeatable.",
    )
    parser.add_argument(
        "--carrier_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516"),
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--robust_policy", action="store_true")
    parser.add_argument("--robust_max_trainval_lpips_regression", type=float, default=0.0)
    parser.add_argument("--robust_min_tail_cvar_delta", type=float, default=-1.0e-4)
    parser.add_argument("--robust_min_stratified_balanced_delta", type=float, default=-1.0e-5)
    return parser.parse_args()


def split_scenes(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace(" ", ",").split(",") if item.strip()]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def number(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def fmt(value: float, digits: int = 9) -> str:
    value = number(value)
    return f"{value:+.{digits}f}"


def find_decision(scene: str, roots: list[Path]) -> tuple[Path | None, dict[str, Any] | None]:
    for root in roots:
        candidates = [
            root / "decisions" / f"{scene}_decision.json",
            root / scene / "decision.json",
            root / scene / "coupled_selector_decision.json",
        ]
        for path in candidates:
            data = load_json(path)
            if data is not None:
                return path, data
    return None, None


def carrier_stats(scene: str, carrier_root: Path) -> dict[str, Any]:
    data = load_json(carrier_root / scene / "render_visible_region_carriers.json")
    if not data:
        return {
            "carrier_path": "",
            "carrier_count": 0,
            "region_count": 0,
            "evidence_face_count": 0,
        }
    carriers = data.get("carriers", [])
    regions = data.get("regions", [])
    evidence_faces = data.get("evidence_faces", data.get("top_evidence_faces", []))
    return {
        "carrier_path": str(carrier_root / scene / "render_visible_region_carriers.json"),
        "carrier_count": len(carriers) if isinstance(carriers, list) else number(data.get("carrier_count"), 0),
        "region_count": len(regions) if isinstance(regions, list) else number(data.get("region_count"), 0),
        "evidence_face_count": (
            len(evidence_faces) if isinstance(evidence_faces, list) else number(data.get("evidence_face_count"), 0)
        ),
    }


def decision_row(scene: str, decision_path: Path | None, decision: dict[str, Any] | None, carrier_root: Path) -> dict[str, Any]:
    stats = carrier_stats(scene, carrier_root)
    if decision is None:
        return {
            "scene": scene,
            "present": False,
            "decision_path": "",
            "accepted": False,
            "selected_label": "missing",
            "selection_uses_test": None,
            "trainval_balanced_delta": 0.0,
            "test_balanced_delta_report_only": 0.0,
            "trainval_delta": {metric: 0.0 for metric in METRICS},
            "test_delta": {metric: 0.0 for metric in METRICS},
            "effective_trainval_balanced_delta": 0.0,
            "effective_test_balanced_delta": 0.0,
            "effective_test_delta": {metric: 0.0 for metric in METRICS},
            "trainval_tail_cvar_delta": 0.0,
            "trainval_stratified_min_balanced_delta": 0.0,
            "robust_policy_pass": False,
            "robust_policy_reasons": ["missing_decision"],
            "robust_effective_test_balanced_delta": 0.0,
            "robust_effective_test_delta": {metric: 0.0 for metric in METRICS},
            "decision_reasons": ["missing_decision"],
            "operator_policy_pass": None,
            "operator_no_op_copy": None,
            "operator_accepted_faces": 0,
            "operator_vertices_added": 0,
            **stats,
        }
    trainval_delta = decision.get("trainval_delta") if isinstance(decision.get("trainval_delta"), dict) else {}
    test_delta = decision.get("test_delta_report_only") if isinstance(decision.get("test_delta_report_only"), dict) else {}
    audit = decision.get("candidate_operator_audit") if isinstance(decision.get("candidate_operator_audit"), dict) else {}
    audit_path = Path(str(audit.get("path", ""))) if audit else None
    full_audit = load_json(audit_path) if audit_path and audit_path.is_file() else {}
    if not full_audit:
        full_audit = audit
    accepted = bool(decision.get("accepted", False))
    trainval_balanced = number(decision.get("trainval_balanced_delta"), 0.0)
    test_balanced = number(decision.get("test_balanced_delta_report_only"), 0.0)
    trainval_delta_values = {metric: number(trainval_delta.get(metric), 0.0) for metric in METRICS}
    test_delta_values = {metric: number(test_delta.get(metric), 0.0) for metric in METRICS}
    tail = decision.get("trainval_per_view_tail") if isinstance(decision.get("trainval_per_view_tail"), dict) else {}
    strat = tail.get("stratified") if isinstance(tail.get("stratified"), dict) else {}
    return {
        "scene": scene,
        "present": True,
        "decision_path": str(decision_path or ""),
        "accepted": accepted,
        "selected_label": str(decision.get("selected_label", "")),
        "selection_uses_test": bool(decision.get("selection_uses_test", False))
        if "selection_uses_test" in decision
        else None,
        "trainval_balanced_delta": trainval_balanced,
        "test_balanced_delta_report_only": test_balanced,
        "trainval_delta": trainval_delta_values,
        "test_delta": test_delta_values,
        "effective_trainval_balanced_delta": trainval_balanced if accepted else 0.0,
        "effective_test_balanced_delta": test_balanced if accepted else 0.0,
        "effective_test_delta": {
            metric: (test_delta_values[metric] if accepted else 0.0)
            for metric in METRICS
        },
        "trainval_tail_cvar_delta": number(tail.get("balanced_cvar_delta"), 0.0),
        "trainval_stratified_min_balanced_delta": number(strat.get("min_balanced_mean_delta"), 0.0),
        "robust_policy_pass": False,
        "robust_policy_reasons": [],
        "robust_effective_test_balanced_delta": 0.0,
        "robust_effective_test_delta": {metric: 0.0 for metric in METRICS},
        "decision_reasons": decision.get("decision_reasons", []),
        "operator_policy_pass": full_audit.get("policy_pass") if isinstance(full_audit, dict) else None,
        "operator_no_op_copy": full_audit.get("no_op_copy") if isinstance(full_audit, dict) else None,
        "operator_accepted_faces": int(number(full_audit.get("accepted_faces"), 0)) if isinstance(full_audit, dict) else 0,
        "operator_vertices_added": int(number(full_audit.get("vertices_added"), 0)) if isinstance(full_audit, dict) else 0,
        **stats,
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def apply_robust_policy(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    for row in rows:
        reasons: list[str] = []
        if not row.get("present", False):
            reasons.append("missing_decision")
        if not row.get("accepted", False):
            reasons.append("base_gate_rejected")
        if bool(row.get("selection_uses_test", False)):
            reasons.append("selection_uses_test")
        if float(row.get("trainval_delta", {}).get("LPIPS", 0.0)) > float(args.robust_max_trainval_lpips_regression):
            reasons.append("trainval_lpips_regression")
        if float(row.get("trainval_tail_cvar_delta", 0.0)) < float(args.robust_min_tail_cvar_delta):
            reasons.append("tail_cvar_below_floor")
        if float(row.get("trainval_stratified_min_balanced_delta", 0.0)) < float(
            args.robust_min_stratified_balanced_delta
        ):
            reasons.append("stratified_balanced_below_floor")
        passed = not reasons
        row["robust_policy_pass"] = passed
        row["robust_policy_reasons"] = reasons
        row["robust_effective_test_balanced_delta"] = row["test_balanced_delta_report_only"] if passed else 0.0
        row["robust_effective_test_delta"] = {
            metric: (row["test_delta"][metric] if passed else 0.0)
            for metric in METRICS
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scene",
        "present",
        "accepted",
        "selected_label",
        "selection_uses_test",
        "trainval_balanced_delta",
        "test_balanced_delta_report_only",
        "trainval_PSNR",
        "trainval_SSIM",
        "trainval_LPIPS",
        "test_PSNR",
        "test_SSIM",
        "test_LPIPS",
        "effective_test_balanced_delta",
        "effective_test_PSNR",
        "effective_test_SSIM",
        "effective_test_LPIPS",
        "robust_policy_pass",
        "robust_effective_test_balanced_delta",
        "robust_effective_test_PSNR",
        "robust_effective_test_SSIM",
        "robust_effective_test_LPIPS",
        "robust_policy_reasons",
        "carrier_count",
        "region_count",
        "evidence_face_count",
        "operator_policy_pass",
        "operator_no_op_copy",
        "operator_accepted_faces",
        "operator_vertices_added",
        "decision_path",
        "decision_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scene": row["scene"],
                    "present": row["present"],
                    "accepted": row["accepted"],
                    "selected_label": row["selected_label"],
                    "selection_uses_test": row["selection_uses_test"],
                    "trainval_balanced_delta": row["trainval_balanced_delta"],
                    "test_balanced_delta_report_only": row["test_balanced_delta_report_only"],
                    "trainval_PSNR": row["trainval_delta"]["PSNR"],
                    "trainval_SSIM": row["trainval_delta"]["SSIM"],
                    "trainval_LPIPS": row["trainval_delta"]["LPIPS"],
                    "test_PSNR": row["test_delta"]["PSNR"],
                    "test_SSIM": row["test_delta"]["SSIM"],
                    "test_LPIPS": row["test_delta"]["LPIPS"],
                    "effective_test_balanced_delta": row["effective_test_balanced_delta"],
                    "effective_test_PSNR": row["effective_test_delta"]["PSNR"],
                    "effective_test_SSIM": row["effective_test_delta"]["SSIM"],
                    "effective_test_LPIPS": row["effective_test_delta"]["LPIPS"],
                    "robust_policy_pass": row["robust_policy_pass"],
                    "robust_effective_test_balanced_delta": row["robust_effective_test_balanced_delta"],
                    "robust_effective_test_PSNR": row["robust_effective_test_delta"]["PSNR"],
                    "robust_effective_test_SSIM": row["robust_effective_test_delta"]["SSIM"],
                    "robust_effective_test_LPIPS": row["robust_effective_test_delta"]["LPIPS"],
                    "robust_policy_reasons": ";".join(str(v) for v in row.get("robust_policy_reasons", [])),
                    "carrier_count": row["carrier_count"],
                    "region_count": row["region_count"],
                    "evidence_face_count": row["evidence_face_count"],
                    "operator_policy_pass": row["operator_policy_pass"],
                    "operator_no_op_copy": row["operator_no_op_copy"],
                    "operator_accepted_faces": row["operator_accepted_faces"],
                    "operator_vertices_added": row["operator_vertices_added"],
                    "decision_path": row["decision_path"],
                    "decision_reasons": ";".join(str(v) for v in row.get("decision_reasons", [])),
                }
            )


def write_md(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase-S Render-Visible Region-Prior Summary",
        "",
        "This report aggregates fixed-policy Phase-K decisions. Scene selection uses only each",
        "decision JSON; held-out test deltas are report-only.",
        "",
        "## Aggregate",
        "",
        f"- scenes present: `{summary['present_count']} / {summary['total_count']}`",
        f"- accepted: `{summary['accepted_count']} / {summary['present_count']}`",
        f"- raw mean train-val balanced delta: `{fmt(summary['mean_trainval_balanced_delta'])}`",
        f"- raw mean report-only test balanced delta: `{fmt(summary['mean_test_balanced_delta'])}`",
        f"- effective mean report-only test balanced delta after Phase-J fallback: "
        f"`{fmt(summary['effective_mean_test_balanced_delta'])}`",
        f"- effective mean report-only test delta after fallback: PSNR "
        f"`{fmt(summary['effective_mean_test_delta']['PSNR'])}`, SSIM "
        f"`{fmt(summary['effective_mean_test_delta']['SSIM'])}`, LPIPS "
        f"`{fmt(summary['effective_mean_test_delta']['LPIPS'])}`",
    ]
    if "robust_accepted_count" in summary:
        lines.extend(
            [
                f"- robust promotion accepted: `{summary['robust_accepted_count']} / {summary['present_count']}`",
                f"- robust effective mean report-only test balanced delta: "
                f"`{fmt(summary['robust_effective_mean_test_balanced_delta'])}`",
                f"- robust effective mean report-only test delta: PSNR "
                f"`{fmt(summary['robust_effective_mean_test_delta']['PSNR'])}`, SSIM "
                f"`{fmt(summary['robust_effective_mean_test_delta']['SSIM'])}`, LPIPS "
                f"`{fmt(summary['robust_effective_mean_test_delta']['LPIPS'])}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Per Scene",
            "",
            "| scene | present | accepted | robust | train-val balanced | raw test balanced | effective test balanced | robust effective test balanced | raw test dPSNR | raw test dSSIM | raw test dLPIPS | carriers | accepted faces | decision |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {scene} | {present} | {accepted} | {robust} | `{tv}` | `{tb}` | `{etb}` | `{retb}` | `{dp}` | `{ds}` | `{dl}` | {carriers} | {faces} | `{decision}` |".format(
                scene=row["scene"],
                present=str(row["present"]).lower(),
                accepted=str(row["accepted"]).lower(),
                robust=str(row["robust_policy_pass"]).lower(),
                tv=fmt(row["trainval_balanced_delta"]),
                tb=fmt(row["test_balanced_delta_report_only"]),
                etb=fmt(row["effective_test_balanced_delta"]),
                retb=fmt(row["robust_effective_test_balanced_delta"]),
                dp=fmt(row["test_delta"]["PSNR"]),
                ds=fmt(row["test_delta"]["SSIM"]),
                dl=fmt(row["test_delta"]["LPIPS"]),
                carriers=row["carrier_count"],
                faces=row["operator_accepted_faces"],
                decision=row["selected_label"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    scenes = split_scenes(args.scenes)
    roots = [Path(root) for root in args.phasek_root]
    if not roots:
        raise SystemExit("at least one --phasek_root is required")
    rows = [
        decision_row(scene, *find_decision(scene, roots), carrier_root=args.carrier_root)
        for scene in scenes
    ]
    apply_robust_policy(rows, args)
    present = [row for row in rows if row["present"]]
    accepted = [row for row in present if row["accepted"]]
    robust_accepted = [row for row in present if row["robust_policy_pass"]]
    summary = {
        "total_count": len(rows),
        "present_count": len(present),
        "accepted_count": len(accepted),
        "mean_trainval_balanced_delta": mean([row["trainval_balanced_delta"] for row in present]),
        "mean_test_balanced_delta": mean([row["test_balanced_delta_report_only"] for row in present]),
        "mean_test_delta": {metric: mean([row["test_delta"][metric] for row in present]) for metric in METRICS},
        "effective_mean_trainval_balanced_delta": mean(
            [row["effective_trainval_balanced_delta"] for row in present]
        ),
        "effective_mean_test_balanced_delta": mean(
            [row["effective_test_balanced_delta"] for row in present]
        ),
        "effective_mean_test_delta": {
            metric: mean([row["effective_test_delta"][metric] for row in present])
            for metric in METRICS
        },
    }
    if bool(args.robust_policy):
        summary.update(
            {
                "robust_policy": {
                    "max_trainval_lpips_regression": float(args.robust_max_trainval_lpips_regression),
                    "min_tail_cvar_delta": float(args.robust_min_tail_cvar_delta),
                    "min_stratified_balanced_delta": float(args.robust_min_stratified_balanced_delta),
                },
                "robust_accepted_count": len(robust_accepted),
                "robust_effective_mean_test_balanced_delta": mean(
                    [row["robust_effective_test_balanced_delta"] for row in present]
                ),
                "robust_effective_mean_test_delta": {
                    metric: mean([row["robust_effective_test_delta"][metric] for row in present])
                    for metric in METRICS
                },
            }
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
