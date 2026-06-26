#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(results_path: Path, method: str) -> dict[str, float]:
    if not results_path.is_file():
        return {}
    payload = _read_json(results_path)
    row = payload.get(method, {})
    return row if isinstance(row, dict) else {}


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble vNext certified residual texture scene manifests.")
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--method_name", default="ours_26000_vnext_certified_residual_surface_texture")
    args = parser.parse_args()

    manifests = sorted(Path(args.run_root).glob("*/reports/*_vnext_certified_residual_texture_manifest.json"))
    rows: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        outputs = manifest.get("outputs", {}) or {}
        results_path = Path(str(outputs.get("results_path", "")))
        row = {
            "scene": manifest.get("scene", manifest_path.parent.parent.name),
            "status": manifest.get("status", ""),
            "protocol_audit_passed": bool((manifest.get("protocol_audit", {}) or {}).get("passed", False)),
            "results_path": str(results_path),
            "report_path": str(outputs.get("report_path", "")),
            "metrics": _metrics(results_path, str(args.method_name)),
            "errors": manifest.get("errors", []),
        }
        rows.append(row)

    completed = [row for row in rows if row["status"] == "COMPLETE" and row["metrics"]]
    mean = {}
    if completed:
        for key in ("PSNR", "SSIM", "LPIPS"):
            values = [float(row["metrics"][key]) for row in completed if key in row["metrics"]]
            if values:
                mean[key] = sum(values) / len(values)
    payload = {
        "schema_version": 1,
        "run_root": str(args.run_root),
        "method_name": str(args.method_name),
        "scene_count": int(len(rows)),
        "completed_metric_scene_count": int(len(completed)),
        "mean_metrics": mean,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# vNext Certified Residual Surface Texture Summary",
        "",
        f"- run root: `{args.run_root}`",
        f"- scenes found: `{len(rows)}`",
        f"- completed metric scenes: `{len(completed)}`",
        f"- mean PSNR: `{_fmt(mean.get('PSNR'))}`",
        f"- mean SSIM: `{_fmt(mean.get('SSIM'))}`",
        f"- mean LPIPS: `{_fmt(mean.get('LPIPS'))}`",
        "",
        "| scene | status | audit | PSNR | SSIM | LPIPS | report |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row.get("metrics", {}) or {}
        lines.append(
            "| {scene} | {status} | {audit} | {psnr} | {ssim} | {lpips} | `{report}` |".format(
                scene=row.get("scene", ""),
                status=row.get("status", ""),
                audit=row.get("protocol_audit_passed", False),
                psnr=_fmt(metrics.get("PSNR")),
                ssim=_fmt(metrics.get("SSIM")),
                lpips=_fmt(metrics.get("LPIPS")),
                report=row.get("report_path", ""),
            )
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
