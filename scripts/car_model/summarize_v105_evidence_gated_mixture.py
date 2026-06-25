#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SCENES = ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"]
DEFAULT_ROOT = "outputs/carnet/meshsplatopt/ecsr_phase_v105_evidence_gated_mixture_hardtriad_20260625"
DEFAULT_PREFIX = "v105_evidence_gated_mixture_summary"
METRICS = ("PSNR", "SSIM", "LPIPS")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _metric(payload: dict[str, Any], section: str, name: str) -> float | None:
    row = payload.get(section, {})
    if not isinstance(row, dict) or name not in row:
        return None
    return float(row[name])


def _mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _fmt(value: Any, signed: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:+.6f}" if signed else f"{value:.6f}"
    return str(value)


def build_summary(root: Path, scenes: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for scene in scenes:
        report_path = root / scene / f"{scene}_v105_evidence_gated_mixture_report.json"
        if not report_path.is_file():
            row = {"scene": scene, "status": "missing", "present": False, "passed": False, "blocker": "missing_report_json", "report_path": str(report_path)}
            rows.append(row)
            blockers.append(row)
            continue
        report = _read_json(report_path)
        row: dict[str, Any] = {
            "scene": scene,
            "status": "ok" if bool(report.get("passed")) else "failed",
            "present": True,
            "passed": bool(report.get("passed")),
            "blocker": "" if bool(report.get("passed")) else "report_passed_false",
            "report_path": str(report_path),
            "output_method": report.get("output_method"),
            "support_source": report.get("render_stats", {}).get("support_source") if isinstance(report.get("render_stats"), dict) else None,
            "no_test_gt_used_for_policy": report.get("render_stats", {}).get("no_test_gt_used_for_policy") if isinstance(report.get("render_stats"), dict) else None,
        }
        for prefix, section in [
            ("clean", "clean_metrics"),
            ("v104c", "v104c_metrics"),
            ("v105", "metrics"),
            ("endpoint", "reference_metrics"),
            ("v102", "v102_metrics"),
        ]:
            for metric in METRICS:
                row[f"{prefix}_{metric}"] = _metric(report, section, metric)
        for metric in METRICS:
            row[f"d{metric}_vs_clean"] = None if row[f"v105_{metric}"] is None or row[f"clean_{metric}"] is None else row[f"v105_{metric}"] - row[f"clean_{metric}"]
            row[f"d{metric}_vs_v104c"] = None if row[f"v105_{metric}"] is None or row[f"v104c_{metric}"] is None else row[f"v105_{metric}"] - row[f"v104c_{metric}"]
            row[f"d{metric}_vs_endpoint"] = None if row[f"v105_{metric}"] is None or row[f"endpoint_{metric}"] is None else row[f"v105_{metric}"] - row[f"endpoint_{metric}"]
        stats = report.get("field_stats", {})
        if isinstance(stats, dict):
            for key in [
                "valid_triangles",
                "mixture_triangles",
                "fallback_only_triangles",
                "gate_mean",
                "gate_source",
                "gain_score_mean",
                "crossfit_gain_mean",
                "crossfit_gain_supported_triangles",
                "stability_score_mean",
                "debt_guard_mean",
                "elapsed_sec",
                "total_accumulated_pixels",
            ]:
                row[key] = stats.get(key)
        render_stats = report.get("render_stats", {})
        if isinstance(render_stats, dict):
            row["render_elapsed_sec"] = render_stats.get("elapsed_sec")
            row["mean_abs_delta"] = render_stats.get("mean_abs_delta")
            row["mean_surface_valid_fraction"] = render_stats.get("mean_surface_valid_fraction")
        rows.append(row)
        if not bool(report.get("passed")):
            blockers.append(row)

    present_rows = [row for row in rows if row.get("present")]
    ok_rows = [row for row in present_rows if row.get("passed")]
    mean: dict[str, Any] = {}
    for prefix in ("clean", "v104c", "v105", "endpoint", "v102"):
        for metric in METRICS:
            mean[f"{prefix}_{metric}"] = _mean([row.get(f"{prefix}_{metric}") for row in ok_rows])
    for metric in METRICS:
        mean[f"d{metric}_vs_clean"] = _mean([row.get(f"d{metric}_vs_clean") for row in ok_rows])
        mean[f"d{metric}_vs_v104c"] = _mean([row.get(f"d{metric}_vs_v104c") for row in ok_rows])
        mean[f"d{metric}_vs_endpoint"] = _mean([row.get(f"d{metric}_vs_endpoint") for row in ok_rows])
    for key in [
        "valid_triangles",
        "mixture_triangles",
        "fallback_only_triangles",
        "gate_mean",
        "gain_score_mean",
        "crossfit_gain_mean",
        "crossfit_gain_supported_triangles",
        "stability_score_mean",
        "debt_guard_mean",
        "elapsed_sec",
        "render_elapsed_sec",
        "mean_abs_delta",
        "mean_surface_valid_fraction",
    ]:
        mean[key] = _mean([row.get(key) for row in ok_rows])
    return {
        "schema_version": 1,
        "root": str(root),
        "scenes": scenes,
        "present_scenes": len(present_rows),
        "ok_scenes": len(ok_rows),
        "all_present": len(present_rows) == len(scenes),
        "all_ok": len(ok_rows) == len(scenes),
        "missing_scenes": [row["scene"] for row in rows if not row.get("present")],
        "blockers": blockers,
        "mean": mean,
        "rows": rows,
        "claim_boundary": "v105 diagnostic target-delta mixture field; uses no held-out GT for policy but is not train-only unseen-camera generalization.",
    }


def write_outputs(summary: dict[str, Any], out_dir: Path, prefix: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"
    md_path = out_dir / f"{prefix}.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = summary["rows"]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    mean = summary["mean"]
    lines: list[str] = [
        "# v105 Evidence-Gated Mixture Summary",
        "",
        f"- root: `{summary['root']}`",
        f"- scenes: `{len(summary['scenes'])}`",
        f"- present_scenes: `{summary['present_scenes']}`",
        f"- ok_scenes: `{summary['ok_scenes']}`",
        f"- all_present: `{summary['all_present']}`",
        f"- all_ok: `{summary['all_ok']}`",
        f"- claim_boundary: {summary['claim_boundary']}",
        "",
        "## Mean Metrics",
        "",
        "| method | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|",
    ]
    for prefix_name, label in [
        ("clean", "clean"),
        ("v104c", "v104c"),
        ("v105", "v105"),
        ("endpoint", "endpoint/reference"),
        ("v102", "v102"),
    ]:
        lines.append(
            f"| {label} | {_fmt(mean.get(prefix_name + '_PSNR'))} | {_fmt(mean.get(prefix_name + '_SSIM'))} | {_fmt(mean.get(prefix_name + '_LPIPS'))} |"
        )
    lines.extend(
        [
            "",
            "## Mean Deltas",
            "",
            "| comparison | dPSNR | dSSIM | dLPIPS |",
            "|---|---:|---:|---:|",
            f"| v105 - clean | {_fmt(mean.get('dPSNR_vs_clean'), True)} | {_fmt(mean.get('dSSIM_vs_clean'), True)} | {_fmt(mean.get('dLPIPS_vs_clean'), True)} |",
            f"| v105 - v104c | {_fmt(mean.get('dPSNR_vs_v104c'), True)} | {_fmt(mean.get('dSSIM_vs_v104c'), True)} | {_fmt(mean.get('dLPIPS_vs_v104c'), True)} |",
            f"| v105 - endpoint/reference | {_fmt(mean.get('dPSNR_vs_endpoint'), True)} | {_fmt(mean.get('dSSIM_vs_endpoint'), True)} | {_fmt(mean.get('dLPIPS_vs_endpoint'), True)} |",
            "",
            "## Per-Scene",
            "",
            "| scene | status | clean PSNR | v104c PSNR | v105 PSNR | endpoint PSNR | dPSNR clean | dPSNR v104c | dPSNR endpoint | v105 SSIM | v105 LPIPS |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {scene} | {status} | {clean_psnr} | {v104c_psnr} | {v105_psnr} | {endpoint_psnr} | {dclean} | {dv104c} | {dendpoint} | {ssim} | {lpips} |".format(
                scene=row.get("scene", ""),
                status=row.get("status", ""),
                clean_psnr=_fmt(row.get("clean_PSNR")),
                v104c_psnr=_fmt(row.get("v104c_PSNR")),
                v105_psnr=_fmt(row.get("v105_PSNR")),
                endpoint_psnr=_fmt(row.get("endpoint_PSNR")),
                dclean=_fmt(row.get("dPSNR_vs_clean"), True),
                dv104c=_fmt(row.get("dPSNR_vs_v104c"), True),
                dendpoint=_fmt(row.get("dPSNR_vs_endpoint"), True),
                ssim=_fmt(row.get("v105_SSIM")),
                lpips=_fmt(row.get("v105_LPIPS")),
            )
        )
    lines.extend(
        [
            "",
            "## Field Diagnostics",
            "",
            "| scene | status | gate source | valid triangles | mixture triangles | fallback only | gate mean | gain score mean | crossfit gain | crossfit support | stability mean | debt guard mean | field sec | render sec | mean abs delta |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {scene} | {status} | {gate_source} | {valid} | {mix} | {fallback} | {gate} | {gain} | {crossfit_gain} | {crossfit_support} | {stable} | {debt_guard} | {field_sec} | {render_sec} | {mad} |".format(
                scene=row.get("scene", ""),
                status=row.get("status", ""),
                gate_source=row.get("gate_source", ""),
                valid=_fmt(row.get("valid_triangles")),
                mix=_fmt(row.get("mixture_triangles")),
                fallback=_fmt(row.get("fallback_only_triangles")),
                gate=_fmt(row.get("gate_mean")),
                gain=_fmt(row.get("gain_score_mean")),
                crossfit_gain=_fmt(row.get("crossfit_gain_mean")),
                crossfit_support=_fmt(row.get("crossfit_gain_supported_triangles")),
                stable=_fmt(row.get("stability_score_mean")),
                debt_guard=_fmt(row.get("debt_guard_mean")),
                field_sec=_fmt(row.get("elapsed_sec")),
                render_sec=_fmt(row.get("render_elapsed_sec")),
                mad=_fmt(row.get("mean_abs_delta")),
            )
        )
    if summary.get("blockers"):
        lines.extend(["", "## Missing / Blockers", "", "| scene | status | blocker | report |", "|---|---|---|---|"])
        for row in summary["blockers"]:
            lines.append(
                f"| {row.get('scene')} | {row.get('status')} | {row.get('blocker')} | `{row.get('report_path')}` |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize v105 evidence-gated mixture reports.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    outputs = write_outputs(build_summary(root, list(args.scenes)), out_dir, str(args.prefix))
    summary = _read_json(Path(outputs["json"]))
    print(json.dumps({"missing_scenes": summary.get("missing_scenes", []), "outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
