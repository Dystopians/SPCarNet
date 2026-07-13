#!/usr/bin/env python3
"""Compare v107 probe reports against the validated v106 full9 assembly."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return data


def _metric_get(metrics: dict[str, Any], name: str) -> float | None:
    for key in (name, name.lower()):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _signed(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:+.{digits}f}"


def _v106_rows(path: Path) -> dict[str, dict[str, Any]]:
    data = _read_json(path)
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"v106 assembled rows missing: {path}")
    return {str(row.get("scene")): row for row in rows if isinstance(row, dict) and str(row.get("scene"))}


def _find_report(root: Path, scene: str) -> Path | None:
    preferred = root / scene / f"{scene}_v107_crossfit_pod_moe_report.json"
    if preferred.is_file():
        return preferred
    candidates = sorted((root / scene).glob(f"{scene}_*report.json"))
    return candidates[0] if candidates else None


def _identity_ok(report: dict[str, Any]) -> bool:
    checks: list[bool] = []
    field_checks = report.get("field_identity", {}).get("checks", {})
    render_checks = report.get("render_stats", {}).get("identity_checks", {})
    if isinstance(field_checks, dict):
        checks.extend(bool(v) for v in field_checks.values())
    if isinstance(render_checks, dict):
        checks.extend(bool(v) for v in render_checks.values())
    return bool(checks) and all(checks)


def _row(scene: str, report_path: Path | None, v106: dict[str, Any] | None) -> dict[str, Any]:
    if report_path is None:
        return {
            "scene": scene,
            "status": "missing_v107_report",
            "report_path": "",
            "passed": False,
            "identity_ok": False,
        }
    report = _read_json(report_path)
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    clean = report.get("clean_metrics", {}) if isinstance(report.get("clean_metrics"), dict) else {}
    v104c = report.get("v104c_metrics", {}) if isinstance(report.get("v104c_metrics"), dict) else {}
    v106 = v106 or {}
    row: dict[str, Any] = {
        "scene": scene,
        "status": str(report.get("status", "")),
        "report_path": str(report_path),
        "passed": bool(report.get("passed")),
        "identity_ok": _identity_ok(report),
        "method_version": report.get("field_stats", {}).get("method_version")
        if isinstance(report.get("field_stats"), dict)
        else "",
        "pod_view_gate_mode": report.get("field_stats", {}).get("pod_view_gate_mode")
        if isinstance(report.get("field_stats"), dict)
        else "",
    }
    for metric in METRICS:
        row[f"clean_{metric}"] = _metric_get(clean, metric)
        row[f"v104c_{metric}"] = _metric_get(v104c, metric)
        row[f"v106_{metric}"] = float(v106[metric]) if isinstance(v106.get(metric), (int, float)) else None
        row[f"v107_{metric}"] = _metric_get(metrics, metric)
        row[f"d{metric}_v107_vs_clean"] = _delta(row[f"v107_{metric}"], row[f"clean_{metric}"])
        row[f"d{metric}_v107_vs_v104c"] = _delta(row[f"v107_{metric}"], row[f"v104c_{metric}"])
        row[f"d{metric}_v107_vs_v106"] = _delta(row[f"v107_{metric}"], row[f"v106_{metric}"])
    return row


def _mean_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("passed") and r.get("identity_ok")]
    row: dict[str, Any] = {
        "scene": "mean_valid",
        "status": f"{len(valid)}/{len(rows)} valid",
        "report_path": "",
        "passed": len(valid) == len(rows) and bool(rows),
        "identity_ok": len(valid) == len(rows) and bool(rows),
        "method_version": "",
        "pod_view_gate_mode": "",
    }
    for metric in METRICS:
        for prefix in ("clean", "v104c", "v106", "v107"):
            key = f"{prefix}_{metric}"
            values = [r.get(key) for r in valid if isinstance(r.get(key), (int, float))]
            row[key] = sum(values) / len(values) if values else None
        for suffix in ("v107_vs_clean", "v107_vs_v104c", "v107_vs_v106"):
            key = f"d{metric}_{suffix}"
            values = [r.get(key) for r in valid if isinstance(r.get(key), (int, float))]
            row[key] = sum(values) / len(values) if values else None
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "scene",
        "status",
        "passed",
        "identity_ok",
        "method_version",
        "pod_view_gate_mode",
        "v107_PSNR",
        "v106_PSNR",
        "dPSNR_v107_vs_v106",
        "v104c_PSNR",
        "dPSNR_v107_vs_v104c",
        "clean_PSNR",
        "dPSNR_v107_vs_clean",
        "v107_SSIM",
        "v106_SSIM",
        "dSSIM_v107_vs_v106",
        "v104c_SSIM",
        "dSSIM_v107_vs_v104c",
        "clean_SSIM",
        "dSSIM_v107_vs_clean",
        "v107_LPIPS",
        "v106_LPIPS",
        "dLPIPS_v107_vs_v106",
        "v104c_LPIPS",
        "dLPIPS_v107_vs_v104c",
        "clean_LPIPS",
        "dLPIPS_v107_vs_clean",
        "report_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v107 Probe vs v106 Comparison",
        "",
        "Deltas are v107 minus the reference method. LPIPS lower is better, so negative LPIPS deltas are better.",
        "",
        "| scene | status | passed | identity | v107 PSNR | dPSNR vs v106 | dPSNR vs v104c | dPSNR vs clean | v107 SSIM | dSSIM vs v106 | v107 LPIPS | dLPIPS vs v106 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scene} | {status} | {passed} | {identity} | {psnr} | {dpsnr106} | {dpsnr104c} | {dpsnrclean} | {ssim} | {dssim106} | {lpips} | {dlpips106} |".format(
                scene=row.get("scene", ""),
                status=row.get("status", ""),
                passed="yes" if row.get("passed") else "no",
                identity="yes" if row.get("identity_ok") else "no",
                psnr=_fmt(row.get("v107_PSNR")),
                dpsnr106=_signed(row.get("dPSNR_v107_vs_v106")),
                dpsnr104c=_signed(row.get("dPSNR_v107_vs_v104c")),
                dpsnrclean=_signed(row.get("dPSNR_v107_vs_clean")),
                ssim=_fmt(row.get("v107_SSIM")),
                dssim106=_signed(row.get("dSSIM_v107_vs_v106")),
                lpips=_fmt(row.get("v107_LPIPS")),
                dlpips106=_signed(row.get("dLPIPS_v107_vs_v106")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v107_root", required=True, type=Path)
    parser.add_argument(
        "--v106_assembled",
        default=Path("docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.json"),
        type=Path,
    )
    parser.add_argument("--scenes", nargs="+", default=["counter", "flowers", "garden", "bonsai"])
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--prefix", default="v107_probe_vs_v106")
    args = parser.parse_args()

    v106_rows = _v106_rows(args.v106_assembled)
    rows = [_row(scene, _find_report(args.v107_root, scene), v106_rows.get(scene)) for scene in args.scenes]
    rows.append(_mean_row(rows))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.prefix}.json"
    csv_path = args.out_dir / f"{args.prefix}.csv"
    md_path = args.out_dir / f"{args.prefix}.md"
    payload = {
        "v107_root": str(args.v107_root),
        "v106_assembled": str(args.v106_assembled),
        "scenes": list(args.scenes),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_md(md_path, rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
