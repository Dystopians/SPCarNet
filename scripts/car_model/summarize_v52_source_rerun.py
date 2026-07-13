#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_JSON = Path(
    "outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/"
    "v52_capacity_aware_v48_v51_full9_summary.json"
)
DEFAULT_OUT_ROOT = Path("outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware")
DEFAULT_OUTPUT_JSON = DEFAULT_OUT_ROOT / "v52_capacity_aware_source_rerun_status.json"
DEFAULT_OUTPUT_MD = DEFAULT_OUT_ROOT / "v52_capacity_aware_source_rerun_status.md"
V48_TAG = "v48_autosupport_autocap_guarded_v42calib_region_texture_adapter"
V51_TAG = "v51_fast_support_ladder_tex32_nearest_l1pos05_region_texture_adapter"
METRICS = ("PSNR", "SSIM", "LPIPS")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_method_metrics(results_path: Path) -> tuple[str, dict[str, float]]:
    payload = read_json(results_path)
    if len(payload) != 1:
        raise RuntimeError(f"expected one method in {results_path}, got {list(payload)}")
    method_name = next(iter(payload))
    row = payload[method_name]
    return method_name, {key: float(row[key]) for key in METRICS}


def metric_delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {key: float(a[key] - b[key]) for key in METRICS}


def strict_win(delta: dict[str, float], eps: float) -> bool:
    return delta["PSNR"] > eps and delta["SSIM"] > eps and delta["LPIPS"] < -eps


def nonregressive(delta: dict[str, float], eps: float) -> bool:
    return delta["PSNR"] >= -eps and delta["SSIM"] >= -eps and delta["LPIPS"] <= eps


def fmt(value: float, digits: int = 9) -> str:
    return f"{float(value):+.{digits}f}"


def output_dir_for(source_root: Path, scene: str, selected_source: str) -> Path:
    if selected_source == "v48":
        return source_root / f"{scene}_{V48_TAG}"
    if selected_source == "v51":
        return source_root / f"{scene}_{V51_TAG}"
    raise RuntimeError(f"unsupported selected source {selected_source!r} for {scene}")


def find_output_dir(source_roots: list[Path], scene: str, selected_source: str) -> tuple[Path, bool]:
    candidates = [output_dir_for(root, scene, selected_source) for root in source_roots]
    for candidate in candidates:
        if (candidate / "results.json").is_file():
            return candidate, True
    return candidates[0], False


def baseline_metrics_from_expected(row: dict[str, Any], label: str) -> dict[str, float]:
    expected = {key: float(row["metrics"][key]) for key in METRICS}
    delta = {key: float(row["comparisons"][label]["delta"][key]) for key in METRICS}
    return {key: expected[key] - delta[key] for key in METRICS}


def summarize_deltas(rows: list[dict[str, Any]], label: str, eps: float) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "complete"]
    if not completed:
        return {
            "scene_count": 0,
            "strict_wins": 0,
            "nonregressive_or_tie": 0,
            "mean_dPSNR": None,
            "mean_dSSIM": None,
            "mean_dLPIPS": None,
        }
    deltas = [row["fresh_comparisons"][label]["delta"] for row in completed]
    return {
        "scene_count": len(completed),
        "strict_wins": int(sum(strict_win(delta, eps) for delta in deltas)),
        "nonregressive_or_tie": int(sum(nonregressive(delta, eps) for delta in deltas)),
        "mean_dPSNR": float(sum(delta["PSNR"] for delta in deltas) / len(deltas)),
        "mean_dSSIM": float(sum(delta["SSIM"] for delta in deltas) / len(deltas)),
        "mean_dLPIPS": float(sum(delta["LPIPS"] for delta in deltas) / len(deltas)),
    }


def maybe_audit_status(output_dir: Path) -> dict[str, Any]:
    audit_path = output_dir / "surface_residual_region_texture_adapter_audit.json"
    if not audit_path.is_file():
        return {"audit_path": str(audit_path), "exists": False}
    audit = read_json(audit_path)
    policy = audit.get("policy_val", {}) or {}
    best = policy.get("best", {}) or {}
    target = audit.get("target_apply", {}) or {}
    return {
        "audit_path": str(audit_path),
        "exists": True,
        "accepted": bool(audit.get("accepted", False)),
        "effective_policy": str(audit.get("effective_policy", "")),
        "selected_alpha": audit.get("selected_alpha"),
        "changed_fraction": float(target.get("changed_fraction", 0.0)),
        "policy_val_relative_gain": float(best.get("relative_gain", 0.0)),
        "policy_val_ssim_gain": float(best.get("ssim_gain", 0.0)),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    summary = read_json(args.summary_json)
    expected_rows = summary["rows"]
    rows: list[dict[str, Any]] = []
    for expected_row in expected_rows:
        scene = str(expected_row["scene"])
        selected_source = str(expected_row["selected_source"])
        out_dir, found_result = find_output_dir(args.source_root, scene, selected_source)
        results_path = out_dir / "results.json"
        row: dict[str, Any] = {
            "scene": scene,
            "selected_source": selected_source,
            "output_dir": str(out_dir),
            "searched_roots": [str(root) for root in args.source_root],
            "results_path": str(results_path),
            "expected_metrics": {key: float(expected_row["metrics"][key]) for key in METRICS},
            "status": "missing",
            "fresh_metrics": None,
            "fresh_minus_expected": None,
            "reproduced_expected": False,
            "audit": maybe_audit_status(out_dir),
        }
        if found_result:
            method_name, fresh_metrics = first_method_metrics(results_path)
            diff = metric_delta(fresh_metrics, row["expected_metrics"])
            max_abs_diff = max(abs(diff[key]) for key in METRICS)
            row.update(
                {
                    "status": "complete",
                    "method_name": method_name,
                    "fresh_metrics": fresh_metrics,
                    "fresh_minus_expected": diff,
                    "max_abs_metric_diff": max_abs_diff,
                    "reproduced_expected": bool(max_abs_diff <= args.reproduce_tolerance),
                    "fresh_comparisons": {},
                }
            )
            for label in ("no-op", "v48", "v50"):
                baseline = baseline_metrics_from_expected(expected_row, label)
                delta = metric_delta(fresh_metrics, baseline)
                row["fresh_comparisons"][label] = {
                    "baseline_metrics": baseline,
                    "delta": delta,
                    "strict_win": strict_win(delta, args.metric_eps),
                    "nonregressive_or_tie": nonregressive(delta, args.metric_eps),
                }
        rows.append(row)
    completed = [row for row in rows if row["status"] == "complete"]
    missing = [row for row in rows if row["status"] != "complete"]
    mismatches = [row for row in completed if not row["reproduced_expected"]]
    status = "COMPLETE_REPRODUCED" if len(completed) == len(rows) and not mismatches else "PARTIAL_OR_MISMATCH"
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "method": "v52 source-config rerun status",
        "status": status,
        "source_roots": [str(root) for root in args.source_root],
        "summary_json": str(args.summary_json),
        "expected_scene_count": len(rows),
        "completed_scene_count": len(completed),
        "missing_scene_count": len(missing),
        "mismatch_scene_count": len(mismatches),
        "missing_scenes": [row["scene"] for row in missing],
        "mismatch_scenes": [row["scene"] for row in mismatches],
        "reproduce_tolerance": float(args.reproduce_tolerance),
        "metric_eps": float(args.metric_eps),
        "summary": {
            label: summarize_deltas(rows, label, args.metric_eps) for label in ("no-op", "v48", "v50")
        },
        "rows": rows,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# v52 Capacity-Aware Source Rerun Status",
        "",
        f"Date: `{payload['date']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "- source roots:",
    ]
    for root in payload["source_roots"]:
        lines.append(f"  - `{root}`")
    lines.extend(
        [
        f"- expected scenes: `{payload['expected_scene_count']}`",
        f"- completed scenes: `{payload['completed_scene_count']}`",
        f"- missing scenes: `{payload['missing_scene_count']}`",
        f"- metric mismatch scenes: `{payload['mismatch_scene_count']}`",
        f"- reproduction tolerance: `{payload['reproduce_tolerance']:.1e}`",
        f"- metric win epsilon: `{payload['metric_eps']:.1e}`",
        "",
        "This audit compares fresh source-config rerun `results.json` files against the",
        "existing v52 effective-policy summary. It can be run while the sequential rerun",
        "is still active; missing scenes are reported as incomplete rather than failures.",
        "",
        "## Completed-Scene Aggregate",
        "",
        "| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, stats in payload["summary"].items():
        if stats["scene_count"] == 0:
            lines.append(f"| fresh v52 vs {label} | 0 | 0 | 0 | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| fresh v52 vs {label} | {stats['scene_count']} | {stats['strict_wins']} | "
            f"{stats['nonregressive_or_tie']} | {fmt(stats['mean_dPSNR'])} | "
            f"{fmt(stats['mean_dSSIM'])} | {fmt(stats['mean_dLPIPS'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-Scene Reproduction",
            "",
            "| scene | selected | status | reproduced | PSNR | SSIM | LPIPS | dPSNR expected | dSSIM expected | dLPIPS expected | audit |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        if row["status"] != "complete":
            lines.append(
                f"| {row['scene']} | {row['selected_source']} | `{row['status']}` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | missing |"
            )
            continue
        fresh = row["fresh_metrics"]
        diff = row["fresh_minus_expected"]
        audit = row["audit"]
        audit_status = "accepted" if audit.get("accepted") else str(audit.get("effective_policy", "missing"))
        lines.append(
            f"| {row['scene']} | {row['selected_source']} | complete | "
            f"{int(bool(row['reproduced_expected']))} | {fresh['PSNR']:.6f} | {fresh['SSIM']:.8f} | "
            f"{fresh['LPIPS']:.8f} | {fmt(diff['PSNR'])} | {fmt(diff['SSIM'])} | "
            f"{fmt(diff['LPIPS'])} | {audit_status} |"
        )
    if payload["missing_scenes"]:
        lines.extend(["", "## Missing Scenes", ""])
        for scene in payload["missing_scenes"]:
            lines.append(f"- `{scene}`")
    if payload["mismatch_scenes"]:
        lines.extend(["", "## Metric Mismatch Scenes", ""])
        for scene in payload["mismatch_scenes"]:
            lines.append(f"- `{scene}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize v52 source-config rerun status.")
    parser.add_argument("--source_root", type=Path, action="append", required=True)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--reproduce_tolerance", type=float, default=1e-5)
    parser.add_argument("--metric_eps", type=float, default=1e-5)
    args = parser.parse_args()

    payload = build_payload(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "completed_scene_count": payload["completed_scene_count"],
                "missing_scene_count": payload["missing_scene_count"],
                "mismatch_scene_count": payload["mismatch_scene_count"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
