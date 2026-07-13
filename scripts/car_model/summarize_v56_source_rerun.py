#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware")
DEFAULT_V56_SUMMARY = DEFAULT_ROOT / "v56_face_alpha_guard_full9_summary.json"
DEFAULT_V52_SUMMARY = DEFAULT_ROOT / "v52_capacity_aware_v48_v51_full9_summary.json"
DEFAULT_OUTPUT_JSON = DEFAULT_ROOT / "v56_face_alpha_guard_source_rerun_status.json"
DEFAULT_OUTPUT_MD = DEFAULT_ROOT / "v56_face_alpha_guard_source_rerun_status.md"
V48_TAG = "v48_autosupport_autocap_guarded_v42calib_region_texture_adapter"
V51_TAG = "v51_fast_support_ladder_tex32_nearest_l1pos05_region_texture_adapter"
V55D_TAG = "v55d_policyval_face_alpha_l1pos09_support4096_tex32_nearest_region_texture_adapter"
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


def output_dir_for(root: Path, scene: str, expected_selected: str, v52_underlying: str) -> Path:
    if expected_selected == "v55d_face_alpha":
        return root / f"{scene}_{V55D_TAG}"
    if expected_selected == "v52_fallback" and v52_underlying == "v48":
        return root / f"{scene}_{V48_TAG}"
    if expected_selected == "v52_fallback" and v52_underlying == "v51":
        return root / f"{scene}_{V51_TAG}"
    raise RuntimeError(
        f"unsupported expected/underlying source for {scene}: {expected_selected!r}, {v52_underlying!r}"
    )


def find_output_dir(
    roots: list[Path],
    scene: str,
    expected_selected: str,
    v52_underlying: str,
) -> tuple[Path, bool]:
    candidates = [output_dir_for(root, scene, expected_selected, v52_underlying) for root in roots]
    for candidate in candidates:
        if (candidate / "results.json").is_file():
            return candidate, True
    return candidates[0], False


def maybe_audit_status(output_dir: Path) -> dict[str, Any]:
    audit_path = output_dir / "surface_residual_region_texture_adapter_audit.json"
    if not audit_path.is_file():
        return {"audit_path": str(audit_path), "exists": False}
    audit = read_json(audit_path)
    local = audit.get("local_alpha_profile", {}) or {}
    risk = audit.get("policy_val_risk_gate", {}) or {}
    target = audit.get("target_apply", {}) or {}
    return {
        "audit_path": str(audit_path),
        "exists": True,
        "accepted": bool(audit.get("accepted", False)),
        "effective_policy": str(audit.get("effective_policy", "")),
        "selected_alpha": float(audit.get("selected_alpha", 0.0)),
        "changed_fraction": float(target.get("changed_fraction", 0.0)),
        "local_alpha_enabled": bool(local.get("enabled", False)),
        "local_alpha_mode": str(local.get("mode", "")),
        "face_alpha_count": int(local.get("face_alpha_count", 0) or 0),
        "selected_ssim_min_view_gain": float(risk.get("selected_ssim_min_view_gain", 0.0) or 0.0),
        "selected_image_l1_positive_view_fraction": float(
            risk.get("selected_image_l1_positive_view_fraction", 0.0) or 0.0
        ),
        "selected_image_l1_cvar20_view_gain": float(
            risk.get("selected_image_l1_cvar20_view_gain", 0.0) or 0.0
        ),
    }


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


def baseline_from_expected(row: dict[str, Any], label: str) -> dict[str, float]:
    expected = {key: float(row["selected_metrics"][key]) for key in METRICS}
    delta = {key: float(row["comparisons"][label]["delta"][key]) for key in METRICS}
    return {key: expected[key] - delta[key] for key in METRICS}


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    v56 = read_json(args.v56_summary)
    v52 = read_json(args.v52_summary)
    v52_rows = {str(row["scene"]): row for row in v52["rows"]}
    rows: list[dict[str, Any]] = []
    for expected_row in v56["rows"]:
        scene = str(expected_row["scene"])
        selected_source = str(expected_row["selected_source"])
        v52_underlying = str(v52_rows[scene]["selected_source"])
        out_dir, found = find_output_dir(args.source_root, scene, selected_source, v52_underlying)
        results_path = out_dir / "results.json"
        row: dict[str, Any] = {
            "scene": scene,
            "selected_source": selected_source,
            "v52_underlying_source": v52_underlying,
            "output_dir": str(out_dir),
            "searched_roots": [str(root) for root in args.source_root],
            "results_path": str(results_path),
            "expected_metrics": {key: float(expected_row["selected_metrics"][key]) for key in METRICS},
            "expected_guard_passed": bool(expected_row.get("guard_passed", False)),
            "expected_guard_reject_reasons": expected_row.get("guard_reject_reasons", []),
            "status": "missing",
            "fresh_metrics": None,
            "fresh_minus_expected": None,
            "reproduced_expected": False,
            "audit": maybe_audit_status(out_dir),
        }
        if found:
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
            for label in ("v52", "no-op", "v48", "v50"):
                baseline = baseline_from_expected(expected_row, label)
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
        "method": "v56 face-alpha guard source-rerun status",
        "status": status,
        "source_roots": [str(root) for root in args.source_root],
        "v56_summary": str(args.v56_summary),
        "v52_summary": str(args.v52_summary),
        "expected_scene_count": len(rows),
        "completed_scene_count": len(completed),
        "missing_scene_count": len(missing),
        "mismatch_scene_count": len(mismatches),
        "missing_scenes": [row["scene"] for row in missing],
        "mismatch_scenes": [row["scene"] for row in mismatches],
        "reproduce_tolerance": float(args.reproduce_tolerance),
        "metric_eps": float(args.metric_eps),
        "selection_uses_heldout_metrics": bool(v56.get("selection_uses_heldout_metrics", True)),
        "summary": {
            label: summarize_deltas(rows, label, args.metric_eps)
            for label in ("v52", "no-op", "v48", "v50")
        },
        "rows": rows,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# v56 Face-Alpha Guard Source-Rerun Status",
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
            f"- selection uses held-out metrics: `{payload['selection_uses_heldout_metrics']}`",
            "",
            "This audit compares source-rerun `results.json` files against the fixed v56",
            "effective-policy summary. It can combine a new v56/v55d source root with older",
            "v52 source-rerun roots for fallback scenes.",
            "",
            "## Completed-Scene Aggregate",
            "",
            "| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, stats in payload["summary"].items():
        if stats["scene_count"] == 0:
            lines.append(f"| fresh v56 vs {label} | 0 | 0 | 0 | n/a | n/a | n/a |")
        else:
            lines.append(
                f"| fresh v56 vs {label} | {stats['scene_count']} | {stats['strict_wins']} | "
                f"{stats['nonregressive_or_tie']} | {fmt(stats['mean_dPSNR'])} | "
                f"{fmt(stats['mean_dSSIM'])} | {fmt(stats['mean_dLPIPS'])} |"
            )
    lines.extend(
        [
            "",
            "## Per-Scene Reproduction",
            "",
            "| scene | selected | underlying | status | reproduced | PSNR | SSIM | LPIPS | dPSNR expected | dSSIM expected | dLPIPS expected | audit |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        if row["status"] != "complete":
            lines.append(
                f"| {row['scene']} | {row['selected_source']} | {row['v52_underlying_source']} | "
                "`missing` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | missing |"
            )
            continue
        fresh = row["fresh_metrics"]
        diff = row["fresh_minus_expected"]
        audit = row["audit"]
        if audit.get("exists"):
            audit_status = (
                f"{audit.get('effective_policy', '')}; alpha={audit.get('selected_alpha', 0.0):.4f}; "
                f"face_alpha={audit.get('face_alpha_count', 0)}"
            )
        else:
            audit_status = "missing"
        lines.append(
            f"| {row['scene']} | {row['selected_source']} | {row['v52_underlying_source']} | complete | "
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
    parser = argparse.ArgumentParser(description="Summarize v56 source-rerun reproduction status.")
    parser.add_argument("--source_root", type=Path, action="append", required=True)
    parser.add_argument("--v56_summary", type=Path, default=DEFAULT_V56_SUMMARY)
    parser.add_argument("--v52_summary", type=Path, default=DEFAULT_V52_SUMMARY)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--reproduce_tolerance", type=float, default=1e-5)
    parser.add_argument("--metric_eps", type=float, default=1e-7)
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
