#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _metric_delta(results: dict[str, Any], reference: dict[str, Any], key: str) -> float | None:
    if key not in results or key not in reference:
        return None
    return float(results[key]) - float(reference[key])


def collect(report_root: Path, scenes: tuple[str, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        report_path = report_root / f"{scene}_detached_package_report.json"
        report = _read_json(report_path)
        results = report.get("results") if isinstance(report.get("results"), dict) else {}
        reference = report.get("reference_results") if isinstance(report.get("reference_results"), dict) else {}
        hashes = report.get("hash_report") if isinstance(report.get("hash_report"), dict) else {}
        row = {
            "scene": scene,
            "report_path": str(report_path),
            "present": bool(report),
            "passed": bool(report.get("passed")),
            "render_rc": report.get("render_rc"),
            "eval_rc": report.get("eval_rc"),
            "used_required_bank": bool(report.get("used_required_bank")),
            "render_count": hashes.get("render_count"),
            "reference_count": hashes.get("reference_count"),
            "hash_match_count": hashes.get("hash_match_count"),
            "hash_mismatch_count": hashes.get("hash_mismatch_count"),
            "PSNR": results.get("PSNR"),
            "SSIM": results.get("SSIM"),
            "LPIPS": results.get("LPIPS"),
            "reference_PSNR": reference.get("PSNR"),
            "reference_SSIM": reference.get("SSIM"),
            "reference_LPIPS": reference.get("LPIPS"),
            "dPSNR_reference": _metric_delta(results, reference, "PSNR"),
            "dSSIM_reference": _metric_delta(results, reference, "SSIM"),
            "dLPIPS_reference": _metric_delta(results, reference, "LPIPS"),
            "elapsed_sec": report.get("elapsed_sec"),
            "support_source": report.get("render_support_source"),
            "log_path": report.get("log_path"),
        }
        rows.append(row)

    metric_rows = [row for row in rows if row["present"] and row["PSNR"] is not None]
    summary = {
        "schema_version": 1,
        "report_root": str(report_root),
        "scenes": list(scenes),
        "rows": rows,
        "all_present": all(row["present"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "all_used_required_bank": all(row["used_required_bank"] for row in rows),
        "all_hash_exact": all(
            row["hash_mismatch_count"] == 0 and row["hash_match_count"] == row["reference_count"]
            for row in rows
            if row["present"]
        ),
        "mean": {
            "PSNR": mean(float(row["PSNR"]) for row in metric_rows) if metric_rows else None,
            "SSIM": mean(float(row["SSIM"]) for row in metric_rows) if metric_rows else None,
            "LPIPS": mean(float(row["LPIPS"]) for row in metric_rows) if metric_rows else None,
            "dPSNR_reference": mean(float(row["dPSNR_reference"]) for row in metric_rows) if metric_rows else None,
            "dSSIM_reference": mean(float(row["dSSIM_reference"]) for row in metric_rows) if metric_rows else None,
            "dLPIPS_reference": mean(float(row["dLPIPS_reference"]) for row in metric_rows) if metric_rows else None,
        },
        "claim_boundary": (
            "This summary validates detached v101 packages against the bankfp16 render.py endpoint reference. "
            "It checks packaging and evidence-bank closure, not an independent metric improvement over Phase-J."
        ),
    }
    return summary


def write_outputs(summary: dict[str, Any], report_root: Path, prefix: str) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / f"{prefix}.json"
    csv_path = report_root / f"{prefix}.csv"
    md_path = report_root / f"{prefix}.md"
    rows = summary["rows"]

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["scene"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# v101 Detached Package Full9 Summary",
        "",
        f"- all_present: `{summary['all_present']}`",
        f"- all_passed: `{summary['all_passed']}`",
        f"- all_used_required_bank: `{summary['all_used_required_bank']}`",
        f"- all_hash_exact: `{summary['all_hash_exact']}`",
        f"- mean PSNR / SSIM / LPIPS: `{summary['mean']['PSNR']:.6f} / {summary['mean']['SSIM']:.6f} / {summary['mean']['LPIPS']:.6f}`",
        f"- mean delta vs reference: `{summary['mean']['dPSNR_reference']:.6f} / {summary['mean']['dSSIM_reference']:.6f} / {summary['mean']['dLPIPS_reference']:.6f}`",
        "",
        "| scene | passed | bank | hash | PSNR | SSIM | LPIPS | dPSNR ref |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        hash_text = f"{row['hash_match_count']}/{row['reference_count']}"
        lines.append(
            f"| {row['scene']} | {row['passed']} | {row['used_required_bank']} | {hash_text} | "
            f"{float(row['PSNR'] or 0.0):.6f} | {float(row['SSIM'] or 0.0):.6f} | "
            f"{float(row['LPIPS'] or 0.0):.6f} | {float(row['dPSNR_reference'] or 0.0):.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize v101 detached-package full9 validation reports.")
    parser.add_argument("--report_root", default="outputs/carnet/meshsplatopt/ecsr_phase_v101_detached_package_full9_20260625")
    parser.add_argument("--prefix", default="v101_detached_package_full9_summary")
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = collect(Path(args.report_root), tuple(args.scenes))
    write_outputs(summary, Path(args.report_root), args.prefix)
    return 0 if summary["all_present"] and summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
