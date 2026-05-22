#!/usr/bin/env python3
"""Filter a face-local residual candidate plan with train-render region evidence.

The input candidate plan is produced by
``ecsr_apply_surface_residual_facelocal_sh1_delta.py --candidate_plan_out``.
The render-region objective is produced on train renders by
``ecsr_eval_train_render_region_objective.py``.  This script maps each
PatchCert carrier in the plan to render-visible train carriers through shared
face ids, then keeps only whole plan carriers whose raw rendered crops show a
nonzero, positive, and tail-safe local effect.

No held-out test images or metrics are read.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_plan", type=Path, required=True)
    parser.add_argument("--render_region_objective", type=Path, required=True)
    parser.add_argument("--carrier_json", type=Path, required=True)
    parser.add_argument("--output_plan", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--max_region_matches_per_plan_carrier", type=int, default=3)
    parser.add_argument("--min_regions", type=int, default=1)
    parser.add_argument("--min_changed_regions", type=int, default=1)
    parser.add_argument("--min_changed_fraction", type=float, default=0.10)
    parser.add_argument("--min_mean_core_balanced_delta", type=float, default=0.0)
    parser.add_argument("--min_mean_delta_psnr", type=float, default=-1.0e30)
    parser.add_argument("--min_tail_core_balanced_delta", type=float, default=-1.0e-8)
    parser.add_argument("--max_context_mse_regression", type=float, default=1.0e-6)
    parser.add_argument("--drop_unmapped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_positive_plan_proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene", default="")
    args = parser.parse_args()
    if int(args.max_region_matches_per_plan_carrier) <= 0:
        parser.error("--max_region_matches_per_plan_carrier must be > 0")
    for name in ("min_regions", "min_changed_regions"):
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name} must be >= 0")
    if not math.isfinite(float(args.min_changed_fraction)) or not 0.0 <= float(args.min_changed_fraction) <= 1.0:
        parser.error("--min_changed_fraction must be in [0, 1]")
    for name in (
        "min_mean_core_balanced_delta",
        "min_mean_delta_psnr",
        "min_tail_core_balanced_delta",
        "max_context_mse_regression",
    ):
        if not math.isfinite(float(getattr(args, name))):
            parser.error(f"--{name} must be finite")
    return args


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def _mean(values: list[float]) -> float:
    vals = _finite(values)
    return float(sum(vals) / len(vals)) if vals else math.nan


def _tail_cvar(values: list[float], fraction: float = 0.25) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return math.nan
    count = max(1, int(math.ceil(float(fraction) * len(vals))))
    return float(sum(vals[:count]) / count)


def _as_int_set(values: Any) -> set[int]:
    out: set[int] = set()
    if isinstance(values, list):
        for value in values:
            try:
                out.add(int(value))
            except Exception:
                continue
    return out


def _load_plan(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json(path)
    if isinstance(payload, list):
        return {}, [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return {}, []
    rows = payload.get("candidates", payload.get("accepted", payload.get("accepted_preview", [])))
    if not isinstance(rows, list):
        rows = []
    meta = dict(payload)
    meta.pop("candidates", None)
    meta.pop("accepted", None)
    meta.pop("accepted_preview", None)
    return meta, [dict(row) for row in rows if isinstance(row, dict)]


def _load_region_carriers(path: Path) -> tuple[dict[str, set[int]], dict[int, set[str]]]:
    payload = _read_json(path)
    raw = payload.get("carriers", []) if isinstance(payload, dict) else []
    carrier_faces: dict[str, set[int]] = {}
    face_to_carriers: dict[int, set[str]] = defaultdict(set)
    for idx, carrier in enumerate(raw):
        if not isinstance(carrier, dict):
            continue
        carrier_id = str(carrier.get("carrier_id", idx))
        faces = _as_int_set(carrier.get("face_ids", []))
        if not faces and isinstance(carrier.get("faces"), list):
            for item in carrier.get("faces", []):
                if not isinstance(item, dict):
                    continue
                try:
                    faces.add(int(item.get("face_id")))
                except Exception:
                    continue
        if not faces:
            continue
        carrier_faces[carrier_id] = faces
        for face_id in faces:
            face_to_carriers[int(face_id)].add(carrier_id)
    return carrier_faces, face_to_carriers


def _load_region_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = _read_json(path)
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    by_carrier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        by_carrier[str(row.get("carrier_id", ""))].append(dict(row))
    return by_carrier


def _plan_carrier_faces(rows: list[dict[str, Any]]) -> set[int]:
    faces: set[int] = set()
    for row in rows:
        faces.update(_as_int_set(row.get("carrier_faces", [])))
        try:
            faces.add(int(row.get("face_id")))
        except Exception:
            pass
    return faces


def _plan_proxy_positive(rows: list[dict[str, Any]]) -> bool:
    values: list[float] = []
    for row in rows:
        proxy = row.get("policy_val_proxy", {})
        if not isinstance(proxy, dict):
            continue
        raw = proxy.get("relative_gain", proxy.get("mean_relative_gain", None))
        try:
            value = float(raw)
        except Exception:
            continue
        if math.isfinite(value):
            values.append(value)
    return (not values) or _mean(values) >= 0.0


def _matched_region_ids(
    *,
    plan_faces: set[int],
    face_to_region_carriers: dict[int, set[str]],
    max_matches: int,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for face_id in plan_faces:
        for carrier_id in face_to_region_carriers.get(int(face_id), set()):
            counts[str(carrier_id)] += 1
    rows = [
        {
            "carrier_id": carrier_id,
            "overlap_faces": int(count),
            "overlap_fraction": float(count) / max(float(len(plan_faces)), 1.0),
        }
        for carrier_id, count in counts.items()
        if int(count) > 0
    ]
    rows.sort(key=lambda row: (-int(row["overlap_faces"]), str(row["carrier_id"])))
    return rows[: int(max_matches)]


def _aggregate_region_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed = [row for row in rows if bool(row.get("crop_changed", False))]
    balanced = [float(row.get("core_balanced_delta", math.nan)) for row in rows]
    dpsnr = [float(row.get("delta_core_psnr", math.nan)) for row in rows]
    dssim = [float(row.get("delta_core_ssim", math.nan)) for row in rows]
    dlpips = [float(row.get("delta_core_lpips", math.nan)) for row in rows]
    context = [float(row.get("context_mse_regression", math.nan)) for row in rows]
    return {
        "regions": int(len(rows)),
        "changed_regions": int(len(changed)),
        "changed_fraction": float(len(changed)) / max(float(len(rows)), 1.0),
        "mean_core_balanced_delta": _mean(balanced),
        "mean_delta_core_psnr": _mean(dpsnr),
        "mean_delta_core_ssim": _mean(dssim),
        "mean_delta_core_lpips": _mean(dlpips),
        "tail_core_balanced_delta": _tail_cvar(balanced, 0.25),
        "max_context_mse_regression": max(_finite(context), default=math.nan),
        "max_crop_abs_diff": max(
            _finite([float(row.get("crop_max_abs_diff", math.nan)) for row in rows]),
            default=math.nan,
        ),
        "mean_crop_abs_diff": _mean([float(row.get("crop_mean_abs_diff", math.nan)) for row in rows]),
    }


def _passes(
    stats: dict[str, Any],
    *,
    args: argparse.Namespace,
    mapped: bool,
    proxy_positive: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not mapped and bool(args.drop_unmapped):
        reasons.append("no_render_region_carrier_overlap")
    if bool(args.require_positive_plan_proxy) and not proxy_positive:
        reasons.append("plan_policy_proxy_negative")
    if int(stats.get("regions", 0)) < int(args.min_regions):
        reasons.append(f"regions_below_{int(args.min_regions)}")
    if int(stats.get("changed_regions", 0)) < int(args.min_changed_regions):
        reasons.append(f"changed_regions_below_{int(args.min_changed_regions)}")
    if float(stats.get("changed_fraction", 0.0)) < float(args.min_changed_fraction):
        reasons.append(f"changed_fraction_below_{float(args.min_changed_fraction):g}")
    if float(stats.get("mean_core_balanced_delta", -math.inf)) < float(args.min_mean_core_balanced_delta):
        reasons.append(f"mean_core_balanced_delta_below_{float(args.min_mean_core_balanced_delta):g}")
    if float(stats.get("mean_delta_core_psnr", -math.inf)) < float(args.min_mean_delta_psnr):
        reasons.append(f"mean_delta_core_psnr_below_{float(args.min_mean_delta_psnr):g}")
    if float(stats.get("tail_core_balanced_delta", -math.inf)) < float(args.min_tail_core_balanced_delta):
        reasons.append(f"tail_core_balanced_delta_below_{float(args.min_tail_core_balanced_delta):g}")
    if float(stats.get("max_context_mse_regression", math.inf)) > float(args.max_context_mse_regression):
        reasons.append(f"context_mse_regression_above_{float(args.max_context_mse_regression):g}")
    return (not reasons), reasons


def main() -> int:
    args = parse_args()
    plan_meta, plan_rows = _load_plan(args.candidate_plan)
    _, face_to_region_carriers = _load_region_carriers(args.carrier_json)
    region_rows_by_carrier = _load_region_rows(args.render_region_objective)

    rows_by_plan_carrier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plan_rows:
        carrier_id = str(row.get("carrier_id", f"face_{row.get('face_id', len(rows_by_plan_carrier))}"))
        rows_by_plan_carrier[carrier_id].append(row)

    kept_rows: list[dict[str, Any]] = []
    carrier_rows: list[dict[str, Any]] = []
    for carrier_id, rows in sorted(rows_by_plan_carrier.items(), key=lambda item: str(item[0])):
        plan_faces = _plan_carrier_faces(rows)
        matches = _matched_region_ids(
            plan_faces=plan_faces,
            face_to_region_carriers=face_to_region_carriers,
            max_matches=int(args.max_region_matches_per_plan_carrier),
        )
        region_rows: list[dict[str, Any]] = []
        for match in matches:
            region_rows.extend(region_rows_by_carrier.get(str(match["carrier_id"]), []))
        stats = _aggregate_region_rows(region_rows)
        proxy_positive = _plan_proxy_positive(rows)
        passed, reasons = _passes(
            stats,
            args=args,
            mapped=bool(matches),
            proxy_positive=proxy_positive,
        )
        row = {
            "plan_carrier_id": carrier_id,
            "plan_faces": sorted(plan_faces),
            "plan_face_count": int(len(plan_faces)),
            "plan_rows": int(len(rows)),
            "matched_region_carriers": matches,
            "render_region_stats": stats,
            "plan_proxy_positive": bool(proxy_positive),
            "accepted": bool(passed),
            "decision_reasons": reasons,
        }
        carrier_rows.append(row)
        if passed:
            kept_rows.extend(rows)

    kept_carriers = {str(row.get("carrier_id", "")) for row in kept_rows}
    out_meta = dict(plan_meta)
    out_meta.update(
        {
            "operator": str(plan_meta.get("operator", "surface_residual_facelocal_plan")),
            "test_usage": "none",
            "plan_export_policy": str(plan_meta.get("plan_export_policy", "final_certified_accepted_faces_only")),
            "source_candidate_plan": str(args.candidate_plan),
            "source_render_region_objective": str(args.render_region_objective),
            "source_region_carrier_json": str(args.carrier_json),
            "render_region_filtered": True,
            "render_region_filter": {
                "scene": str(args.scene),
                "input_rows": int(len(plan_rows)),
                "input_carriers": int(len(rows_by_plan_carrier)),
                "kept_rows": int(len(kept_rows)),
                "kept_carriers": int(len(kept_carriers)),
                "rejected_carriers": int(len(rows_by_plan_carrier) - len(kept_carriers)),
                "max_region_matches_per_plan_carrier": int(args.max_region_matches_per_plan_carrier),
                "drop_unmapped": bool(args.drop_unmapped),
                "require_positive_plan_proxy": bool(args.require_positive_plan_proxy),
                "thresholds": {
                    "min_regions": int(args.min_regions),
                    "min_changed_regions": int(args.min_changed_regions),
                    "min_changed_fraction": float(args.min_changed_fraction),
                    "min_mean_core_balanced_delta": float(args.min_mean_core_balanced_delta),
                    "min_mean_delta_psnr": float(args.min_mean_delta_psnr),
                    "min_tail_core_balanced_delta": float(args.min_tail_core_balanced_delta),
                    "max_context_mse_regression": float(args.max_context_mse_regression),
                },
            },
            "candidate_count": int(len(kept_rows)),
            "carrier_count": int(len(kept_carriers)),
            "candidates": kept_rows,
        }
    )

    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(json.dumps(out_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "scene": str(args.scene),
        "candidate_plan": str(args.candidate_plan),
        "render_region_objective": str(args.render_region_objective),
        "carrier_json": str(args.carrier_json),
        "output_plan": str(args.output_plan),
        "input_rows": int(len(plan_rows)),
        "input_carriers": int(len(rows_by_plan_carrier)),
        "kept_rows": int(len(kept_rows)),
        "kept_carriers": int(len(kept_carriers)),
        "carrier_rows": carrier_rows,
    }
    output_json = args.output_json or args.output_plan.with_suffix(".filter_summary.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Render-Region Filtered Face-Local Plan",
        "",
        f"- scene: `{args.scene}`",
        f"- input rows / carriers: `{len(plan_rows)}` / `{len(rows_by_plan_carrier)}`",
        f"- kept rows / carriers: `{len(kept_rows)}` / `{len(kept_carriers)}`",
        f"- source plan: `{args.candidate_plan}`",
        f"- render-region objective: `{args.render_region_objective}`",
        "",
        "| plan carrier | accepted | rows | faces | matched region carriers | regions | changed | mean balanced | mean dPSNR | tail balanced | max context reg | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in carrier_rows:
        stats = row["render_region_stats"]
        lines.append(
            f"| `{row['plan_carrier_id']}` | {str(row['accepted']).lower()} | "
            f"{int(row['plan_rows'])} | {int(row['plan_face_count'])} | "
            f"{len(row['matched_region_carriers'])} | {int(stats.get('regions', 0))} | "
            f"{int(stats.get('changed_regions', 0))} | "
            f"{float(stats.get('mean_core_balanced_delta', math.nan)):+.9f} | "
            f"{float(stats.get('mean_delta_core_psnr', math.nan)):+.9f} | "
            f"{float(stats.get('tail_core_balanced_delta', math.nan)):+.9f} | "
            f"{float(stats.get('max_context_mse_regression', math.nan)):.9g} | "
            f"`{', '.join(row['decision_reasons']) if row['decision_reasons'] else 'pass'}` |"
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("scene", "input_rows", "input_carriers", "kept_rows", "kept_carriers")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
