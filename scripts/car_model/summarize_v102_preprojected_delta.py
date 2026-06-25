#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def summarize(report_root: Path, scenes: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        report_path = report_root / scene / f"{scene}_v102_preprojected_delta_report.json"
        report = _read_json(report_path)
        metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
        hashes = report.get("hash_report") if isinstance(report.get("hash_report"), dict) else {}
        row = {
            "scene": scene,
            "present": bool(report),
            "passed": bool(report.get("passed")),
            "build_rc": report.get("build_rc"),
            "fast_rc": report.get("fast_rc"),
            "eval_rc": report.get("eval_rc"),
            "mode": report.get("mode"),
            "support_source": report.get("support_source"),
            "intermediate_outputs_saved": report.get("intermediate_outputs_saved"),
            "PSNR": metrics.get("PSNR"),
            "SSIM": metrics.get("SSIM"),
            "LPIPS": metrics.get("LPIPS"),
            "dPSNR_reference": report.get("dPSNR_reference"),
            "dSSIM_reference": report.get("dSSIM_reference"),
            "dLPIPS_reference": report.get("dLPIPS_reference"),
            "fast_wall_sec": report.get("fast_wall_sec"),
            "fast_sec_per_view": report.get("fast_sec_per_view"),
            "fast_internal_sec_per_view": report.get("fast_internal_sec_per_view"),
            "hash_match_count": hashes.get("hash_match_count"),
            "reference_count": hashes.get("reference_count"),
            "hash_mismatch_count": hashes.get("hash_mismatch_count"),
            "numerically_exact": hashes.get("numerically_exact"),
            "mean_abs_uint8": hashes.get("mean_abs_uint8"),
            "max_abs_uint8": hashes.get("max_abs_uint8"),
            "v102_bank": report.get("v102_bank"),
            "report_path": str(report_path),
        }
        rows.append(row)
    passed_rows = [row for row in rows if row["present"] and row["passed"]]
    metric_rows = [row for row in rows if row["present"] and row["PSNR"] is not None]
    return {
        "schema_version": 1,
        "report_root": str(report_root),
        "scenes": scenes,
        "rows": rows,
        "all_present": all(row["present"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "all_preprojected": all(str(row["support_source"] or "").startswith("v102_preprojected_delta_bank:") for row in rows),
        "all_numerically_exact": all(bool(row["numerically_exact"]) for row in rows),
        "passed_scenes": len(passed_rows),
        "mean": {
            "PSNR": mean(float(row["PSNR"]) for row in metric_rows) if metric_rows else None,
            "SSIM": mean(float(row["SSIM"]) for row in metric_rows) if metric_rows else None,
            "LPIPS": mean(float(row["LPIPS"]) for row in metric_rows) if metric_rows else None,
            "dPSNR_reference": mean(float(row["dPSNR_reference"]) for row in metric_rows) if metric_rows else None,
            "dSSIM_reference": mean(float(row["dSSIM_reference"]) for row in metric_rows) if metric_rows else None,
            "dLPIPS_reference": mean(float(row["dLPIPS_reference"]) for row in metric_rows) if metric_rows else None,
            "fast_sec_per_view": mean(float(row["fast_sec_per_view"]) for row in metric_rows) if metric_rows else None,
            "fast_internal_sec_per_view": (
                mean(float(row["fast_internal_sec_per_view"]) for row in metric_rows) if metric_rows else None
            ),
        },
        "claim_boundary": (
            "v102 preprojected delta is a target-camera acceleration endpoint generated from v101 train-evidence output. "
            "It is not a vanilla MeshSplatting checkpoint or a general unseen-camera residual field."
        ),
    }


def write_outputs(summary: dict[str, Any], report_root: Path, prefix: str) -> None:
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
        "# v102 Preprojected Delta Summary",
        "",
        f"- all_present: `{summary['all_present']}`",
        f"- all_passed: `{summary['all_passed']}`",
        f"- all_preprojected: `{summary['all_preprojected']}`",
        f"- all_numerically_exact: `{summary['all_numerically_exact']}`",
        f"- mean PSNR / SSIM / LPIPS: `{summary['mean']['PSNR']:.6f} / {summary['mean']['SSIM']:.6f} / {summary['mean']['LPIPS']:.6f}`",
        f"- mean delta vs v101 reference: `{summary['mean']['dPSNR_reference']:.6f} / {summary['mean']['dSSIM_reference']:.6f} / {summary['mean']['dLPIPS_reference']:.6f}`",
        f"- mean fast sec/view: `{summary['mean']['fast_sec_per_view']:.6f}`",
        f"- mean internal sec/view: `{summary['mean']['fast_internal_sec_per_view']:.6f}`",
        "",
        "| scene | passed | PSNR | SSIM | LPIPS | hash | numeric exact | fast sec/view |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene']} | {row['passed']} | {float(row['PSNR'] or 0.0):.6f} | "
            f"{float(row['SSIM'] or 0.0):.6f} | {float(row['LPIPS'] or 0.0):.6f} | "
            f"{row['hash_match_count']}/{row['reference_count']} | {row['numerically_exact']} | "
            f"{float(row['fast_sec_per_view'] or 0.0):.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize v102 preprojected delta validation reports.")
    parser.add_argument("--report_root", default="outputs/carnet/meshsplatopt/ecsr_phase_v102_preprojected_delta_bank_20260625")
    parser.add_argument("--scenes", nargs="*", default=["counter", "kitchen", "bonsai"])
    parser.add_argument("--prefix", default="v102_preprojected_delta_triad_summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_root = Path(args.report_root)
    summary = summarize(report_root, args.scenes)
    write_outputs(summary, report_root, args.prefix)
    return 0 if summary["all_present"] and summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
