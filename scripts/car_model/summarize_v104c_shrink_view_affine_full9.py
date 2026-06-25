#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")
METRICS = ("PSNR", "SSIM", "LPIPS")
DEFAULT_ROOT = "outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625"
DEFAULT_PREFIX = "v104c_shrink_view_affine_full9_summary"


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        return {}, "missing_report_json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"json_decode_error:{exc}"
    if not isinstance(payload, dict):
        return {}, "json_root_not_object"
    return payload, ""


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metrics_from(report: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        metrics = _as_dict(report.get(key))
        if any(metric in metrics for metric in METRICS):
            return metrics
    return {}


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(lhs: Any, rhs: Any) -> float | None:
    lhs_float = _float_or_none(lhs)
    rhs_float = _float_or_none(rhs)
    if lhs_float is None or rhs_float is None:
        return None
    return lhs_float - rhs_float


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows]
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _fmt(value: Any, signed: bool = False) -> str:
    value_float = _float_or_none(value)
    if value_float is None:
        return ""
    if signed:
        return f"{value_float:+.6f}"
    return f"{value_float:.6f}"


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _collect_row(root: Path, scene: str) -> dict[str, Any]:
    report_path = root / scene / f"{scene}_v104c_shrink_view_affine_report.json"
    report, blocker = _read_json(report_path)

    clean = _metrics_from(report, "clean_metrics", "clean")
    v104c = _metrics_from(report, "metrics", "v104c_metrics", "v104c_shrink_view_affine_metrics")
    reference = _metrics_from(report, "reference_metrics", "endpoint_metrics", "v101_metrics", "v101_v102a_metrics")
    v102 = _metrics_from(report, "v102_metrics", "v102a_metrics")
    endpoint = reference or v102
    field_stats = _as_dict(report.get("field_stats"))
    render_stats = _as_dict(report.get("render_stats"))
    return_codes = _as_dict(report.get("return_codes"))
    deltas = _as_dict(report.get("deltas"))
    delta_vs_clean = _as_dict(deltas.get("vs_clean"))
    delta_vs_reference = _as_dict(deltas.get("vs_reference")) or _as_dict(deltas.get("vs_v101_v102a"))

    if not blocker and not v104c:
        blocker = "missing_v104c_metrics"
    status = "ok" if report and not blocker else ("missing" if blocker == "missing_report_json" else "blocker")

    row: dict[str, Any] = {
        "scene": scene,
        "status": status,
        "present": bool(report),
        "passed": report.get("passed") if report else None,
        "blocker": blocker,
        "report_path": str(report_path),
        "output_method": report.get("output_method"),
        "reference_key": _first_nonempty(report.get("reference_key"), report.get("v102_key"), report.get("v101_key")),
        "endpoint_method": report.get("endpoint_method"),
        "support_source": render_stats.get("support_source"),
        "no_test_gt_used_for_policy": render_stats.get("no_test_gt_used_for_policy"),
        "field": report.get("field"),
        "field_manifest": report.get("field_manifest"),
        "render_report": report.get("render_report"),
        "rc_field": return_codes.get("field"),
        "rc_render": return_codes.get("render"),
        "rc_eval": return_codes.get("eval"),
        "rc_v102": return_codes.get("v102"),
        "field_elapsed_sec": field_stats.get("elapsed_sec"),
        "render_elapsed_sec": render_stats.get("elapsed_sec"),
        "valid_triangles": field_stats.get("valid_triangles"),
        "view_affine_triangles": field_stats.get("view_affine_triangles"),
        "fallback_triangles": field_stats.get("fallback_triangles"),
        "shrink_alpha_mean": field_stats.get("shrink_alpha_mean"),
        "mean_abs_delta": render_stats.get("mean_abs_delta"),
        "mean_surface_valid_fraction": render_stats.get("mean_surface_valid_fraction"),
        "target_frames": render_stats.get("target_frames"),
        "v102_bank": report.get("v102_bank"),
    }

    for metric in METRICS:
        row[f"clean_{metric}"] = clean.get(metric)
        row[f"v104c_{metric}"] = v104c.get(metric)
        row[f"endpoint_{metric}"] = endpoint.get(metric)
        row[f"v102_{metric}"] = v102.get(metric)
        row[f"d{metric}_vs_clean"] = _first_nonempty(delta_vs_clean.get(metric), _delta(v104c.get(metric), clean.get(metric)))
        row[f"d{metric}_vs_endpoint"] = _first_nonempty(
            delta_vs_reference.get(metric),
            _delta(v104c.get(metric), endpoint.get(metric)),
        )
        row[f"d{metric}_vs_v102"] = _delta(v104c.get(metric), v102.get(metric))

    return row


def summarize(root: Path, scenes: list[str]) -> dict[str, Any]:
    rows = [_collect_row(root, scene) for scene in scenes]
    ok_rows = [row for row in rows if row["status"] == "ok"]
    present_rows = [row for row in rows if row["present"]]
    mean_keys = []
    for prefix in ("clean", "v104c", "endpoint", "v102"):
        mean_keys.extend(f"{prefix}_{metric}" for metric in METRICS)
    for suffix in ("vs_clean", "vs_endpoint", "vs_v102"):
        mean_keys.extend(f"d{metric}_{suffix}" for metric in METRICS)
    mean_keys.extend(
        [
            "field_elapsed_sec",
            "render_elapsed_sec",
            "valid_triangles",
            "view_affine_triangles",
            "fallback_triangles",
            "shrink_alpha_mean",
            "mean_abs_delta",
            "mean_surface_valid_fraction",
            "target_frames",
        ]
    )
    blockers = [
        {"scene": row["scene"], "status": row["status"], "blocker": row["blocker"], "report_path": row["report_path"]}
        for row in rows
        if row["status"] != "ok"
    ]
    return {
        "schema_version": 1,
        "root": str(root),
        "scenes": scenes,
        "rows": rows,
        "all_present": all(row["present"] for row in rows),
        "all_ok": all(row["status"] == "ok" for row in rows),
        "all_passed_present": all(bool(row["passed"]) for row in present_rows) if present_rows else False,
        "present_scenes": len(present_rows),
        "ok_scenes": len(ok_rows),
        "missing_scenes": [row["scene"] for row in rows if row["status"] == "missing"],
        "blockers": blockers,
        "mean": {key: _mean(rows, key) for key in mean_keys},
        "claim_boundary": (
            "v104c shrink view-affine full9 summary compares clean MeshSplatting, v104c field results, "
            "and any endpoint/reference v101/v102 metrics present in each report. Missing scene reports are "
            "recorded as missing/blocker rows rather than fatal errors."
        ),
    }


def _write_json(summary: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["scene"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(summary: dict[str, Any], path: Path) -> None:
    mean_values = summary["mean"]
    lines = [
        "# v104c Shrink View-Affine Full9 Summary",
        "",
        f"- root: `{summary['root']}`",
        f"- scenes: `{len(summary['scenes'])}`",
        f"- present_scenes: `{summary['present_scenes']}`",
        f"- ok_scenes: `{summary['ok_scenes']}`",
        f"- all_present: `{summary['all_present']}`",
        f"- all_ok: `{summary['all_ok']}`",
        "",
        "## Mean Metrics",
        "",
        "| method | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|",
        f"| clean | {_fmt(mean_values.get('clean_PSNR'))} | {_fmt(mean_values.get('clean_SSIM'))} | {_fmt(mean_values.get('clean_LPIPS'))} |",
        f"| v104c | {_fmt(mean_values.get('v104c_PSNR'))} | {_fmt(mean_values.get('v104c_SSIM'))} | {_fmt(mean_values.get('v104c_LPIPS'))} |",
        f"| endpoint/reference | {_fmt(mean_values.get('endpoint_PSNR'))} | {_fmt(mean_values.get('endpoint_SSIM'))} | {_fmt(mean_values.get('endpoint_LPIPS'))} |",
        f"| v102 | {_fmt(mean_values.get('v102_PSNR'))} | {_fmt(mean_values.get('v102_SSIM'))} | {_fmt(mean_values.get('v102_LPIPS'))} |",
        "",
        "## Mean Deltas",
        "",
        "| comparison | dPSNR | dSSIM | dLPIPS |",
        "|---|---:|---:|---:|",
        f"| v104c - clean | {_fmt(mean_values.get('dPSNR_vs_clean'), True)} | {_fmt(mean_values.get('dSSIM_vs_clean'), True)} | {_fmt(mean_values.get('dLPIPS_vs_clean'), True)} |",
        f"| v104c - endpoint/reference | {_fmt(mean_values.get('dPSNR_vs_endpoint'), True)} | {_fmt(mean_values.get('dSSIM_vs_endpoint'), True)} | {_fmt(mean_values.get('dLPIPS_vs_endpoint'), True)} |",
        f"| v104c - v102 | {_fmt(mean_values.get('dPSNR_vs_v102'), True)} | {_fmt(mean_values.get('dSSIM_vs_v102'), True)} | {_fmt(mean_values.get('dLPIPS_vs_v102'), True)} |",
        "",
        "## Per-Scene",
        "",
        "| scene | status | blocker | clean PSNR | v104c PSNR | endpoint PSNR | dPSNR clean | dPSNR endpoint | v104c SSIM | v104c LPIPS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['scene']} | {row['status']} | {row['blocker']} | "
            f"{_fmt(row.get('clean_PSNR'))} | {_fmt(row.get('v104c_PSNR'))} | {_fmt(row.get('endpoint_PSNR'))} | "
            f"{_fmt(row.get('dPSNR_vs_clean'), True)} | {_fmt(row.get('dPSNR_vs_endpoint'), True)} | "
            f"{_fmt(row.get('v104c_SSIM'))} | {_fmt(row.get('v104c_LPIPS'))} |"
        )
    lines.extend(
        [
            "",
            "## Field Diagnostics",
            "",
            "| scene | status | valid triangles | fallback triangles | shrink alpha mean | field sec | render sec | mean abs delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            f"| {row['scene']} | {row['status']} | {_fmt(row.get('valid_triangles'))} | "
            f"{_fmt(row.get('fallback_triangles'))} | {_fmt(row.get('shrink_alpha_mean'))} | "
            f"{_fmt(row.get('field_elapsed_sec'))} | {_fmt(row.get('render_elapsed_sec'))} | "
            f"{_fmt(row.get('mean_abs_delta'))} |"
        )
    if summary["blockers"]:
        lines.extend(["", "## Missing / Blockers", "", "| scene | status | blocker | report |", "|---|---|---|---|"])
        for blocker in summary["blockers"]:
            lines.append(
                f"| {blocker['scene']} | {blocker['status']} | {blocker['blocker']} | `{blocker['report_path']}` |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(summary: dict[str, Any], out_dir: Path, prefix: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / f"{prefix}.json",
        "csv": out_dir / f"{prefix}.csv",
        "md": out_dir / f"{prefix}.md",
    }
    _write_json(summary, paths["json"])
    _write_csv(summary["rows"], paths["csv"])
    _write_md(summary, paths["md"])
    return {key: str(value) for key, value in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize v104c shrink view-affine full9 scene reports.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Directory containing per-scene v104c report folders.")
    parser.add_argument("--scenes", nargs="*", default=list(SCENES), help="Scene names to aggregate.")
    parser.add_argument("--out_dir", default=None, help="Output directory. Defaults to --root.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Output filename prefix.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    summary = summarize(root, list(args.scenes))
    outputs = write_outputs(summary, out_dir, args.prefix)
    print(json.dumps({"outputs": outputs, "missing_scenes": summary["missing_scenes"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
