#!/usr/bin/env python3
"""Apply a train-only aggregate render-CVaR carrier subset to a filtered plan.

This is a conservative post-filter for face-local residual plans.  It consumes a
plan that already contains ``render_region_filter_decision`` annotations and
keeps only whole carriers that pass fixed train-render region risk checks.  It
then greedily builds a carrier subset whose union of matched train-render
regions remains mean-positive, tail-safe, and context-safe.

No held-out/test images or metrics are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_plan", type=Path, required=True)
    parser.add_argument("--output_plan", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--render_region_objective", type=Path, default=None)
    parser.add_argument("--scene", default="")
    parser.add_argument("--min_regions", type=int, default=1)
    parser.add_argument("--min_changed_regions", type=int, default=1)
    parser.add_argument("--min_changed_fraction", type=float, default=0.05)
    parser.add_argument("--min_mean_core_balanced_delta", type=float, default=0.0)
    parser.add_argument("--min_mean_delta_psnr", type=float, default=0.0)
    parser.add_argument("--min_tail_core_balanced_delta", type=float, default=-2.0e-5)
    parser.add_argument("--max_negative_core_balanced_fraction", type=float, default=0.35)
    parser.add_argument("--max_context_mse_regression", type=float, default=1.0e-6)
    parser.add_argument("--tail_fraction", type=float, default=0.25)
    parser.add_argument("--min_selected_carriers", type=int, default=1)
    parser.add_argument("--expected_view_count", type=int, default=0)
    parser.add_argument("--min_unique_views", type=int, default=0)
    parser.add_argument("--min_changed_unique_views", type=int, default=0)
    parser.add_argument("--min_view_coverage_fraction", type=float, default=0.0)
    parser.add_argument("--min_changed_view_coverage_fraction", type=float, default=0.0)
    parser.add_argument("--min_total_pixels", type=int, default=0)
    parser.add_argument("--min_changed_pixels", type=int, default=0)
    parser.add_argument("--min_changed_pixel_fraction", type=float, default=0.0)
    parser.add_argument("--expected_frame_pixels", type=int, default=0)
    parser.add_argument("--min_full_frame_changed_pixel_fraction", type=float, default=0.0)
    parser.add_argument("--min_area_weighted_core_balanced_delta", type=float, default=-1.0e30)
    parser.add_argument("--min_dilution_adjusted_core_balanced_delta", type=float, default=-1.0e30)
    parser.add_argument("--min_full_frame_visibility_adjusted_delta", type=float, default=-1.0e30)
    parser.add_argument("--prefer_full_frame_visibility", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tail_safe_shrink_carriers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tail_safe_shrink_scales", default="1.0,0.85,0.75,0.6,0.5,0.35,0.2,0.1,0.05,0.035,0.02")
    parser.add_argument("--tail_safe_shrink_min_scale", type=float, default=0.02)
    parser.add_argument("--output_alpha_json", type=Path, default=None)
    args = parser.parse_args()
    for name in (
        "min_regions",
        "min_changed_regions",
        "min_selected_carriers",
        "expected_view_count",
        "min_unique_views",
        "min_changed_unique_views",
        "min_total_pixels",
        "min_changed_pixels",
        "expected_frame_pixels",
    ):
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name} must be >= 0")
    for name in (
        "min_changed_fraction",
        "max_negative_core_balanced_fraction",
        "tail_fraction",
        "min_view_coverage_fraction",
        "min_changed_view_coverage_fraction",
        "min_changed_pixel_fraction",
        "min_full_frame_changed_pixel_fraction",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            parser.error(f"--{name} must be in [0, 1]")
    if float(args.tail_fraction) <= 0.0:
        parser.error("--tail_fraction must be > 0")
    for name in (
        "min_mean_core_balanced_delta",
        "min_mean_delta_psnr",
        "min_tail_core_balanced_delta",
        "max_context_mse_regression",
        "min_area_weighted_core_balanced_delta",
        "min_dilution_adjusted_core_balanced_delta",
        "min_full_frame_visibility_adjusted_delta",
    ):
        if not math.isfinite(float(getattr(args, name))):
            parser.error(f"--{name} must be finite")
    if float(args.max_context_mse_regression) < 0.0:
        parser.error("--max_context_mse_regression must be >= 0")
    if not math.isfinite(float(args.tail_safe_shrink_min_scale)) or not 0.0 <= float(args.tail_safe_shrink_min_scale) <= 1.0:
        parser.error("--tail_safe_shrink_min_scale must be in [0, 1]")
    for raw in str(args.tail_safe_shrink_scales).split(","):
        if not raw.strip():
            continue
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            parser.error("--tail_safe_shrink_scales values must be in [0, 1]")
    return args


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean(values: list[float]) -> float:
    vals = finite(values)
    return float(sum(vals) / len(vals)) if vals else math.nan


def tail_cvar(values: list[float], fraction: float) -> float:
    vals = sorted(finite(values))
    if not vals:
        return math.nan
    count = max(1, int(math.ceil(float(fraction) * len(vals))))
    return float(sum(vals[:count]) / count)


def load_plan(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = read_json(path)
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


def load_region_rows(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None or not path.is_file():
        return {}
    payload = read_json(path)
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        out[str(row.get("carrier_id", ""))].append(dict(row))
    return out


def num(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def row_pixel_count(row: dict[str, Any]) -> float:
    value = num(row.get("pixels"), math.nan)
    if math.isfinite(value) and value > 0.0:
        return float(value)
    bbox = row.get("bbox_xyxy", row.get("bbox", []))
    if isinstance(bbox, list) and len(bbox) >= 4:
        width = max(0.0, num(bbox[2], 0.0) - num(bbox[0], 0.0))
        height = max(0.0, num(bbox[3], 0.0) - num(bbox[1], 0.0))
        area = width * height
        if area > 0.0:
            return float(area)
    return 0.0


def row_changed_pixel_count(row: dict[str, Any]) -> float:
    pixels = row_pixel_count(row)
    value = num(row.get("crop_nonzero_pixels"), math.nan)
    if math.isfinite(value) and value >= 0.0:
        return float(min(value, pixels)) if pixels > 0.0 else float(value)
    fraction = num(row.get("crop_nonzero_fraction"), math.nan)
    if math.isfinite(fraction) and fraction >= 0.0:
        return float(pixels * min(fraction, 1.0))
    return pixels if bool(row.get("crop_changed", False)) else 0.0


def row_frame_pixel_count(row: dict[str, Any], args: argparse.Namespace) -> float:
    value = num(row.get("frame_pixels"), math.nan)
    if math.isfinite(value) and value > 0.0:
        return float(value)
    width = num(row.get("image_width"), math.nan)
    height = num(row.get("image_height"), math.nan)
    if math.isfinite(width) and math.isfinite(height) and width > 0.0 and height > 0.0:
        return float(width * height)
    expected = int(args.expected_frame_pixels)
    return float(expected) if expected > 0 else 0.0


def full_frame_denominator(rows: list[dict[str, Any]], args: argparse.Namespace) -> float:
    expected_views = int(args.expected_view_count)
    expected_frame_pixels = int(args.expected_frame_pixels)
    if expected_frame_pixels > 0 and expected_views > 0:
        return float(expected_frame_pixels * expected_views)

    frame_pixels_by_view: dict[str, float] = {}
    fallback_values: list[float] = []
    for row in rows:
        frame_pixels = row_frame_pixel_count(row, args)
        if frame_pixels <= 0.0:
            continue
        fallback_values.append(float(frame_pixels))
        view = str(row.get("view", row.get("view_name", "")))
        if view:
            frame_pixels_by_view.setdefault(view, float(frame_pixels))
    if expected_views > 0 and fallback_values:
        ordered = sorted(fallback_values)
        return float(ordered[len(ordered) // 2] * expected_views)
    if frame_pixels_by_view:
        return float(sum(frame_pixels_by_view.values()))
    if expected_frame_pixels > 0:
        return float(expected_frame_pixels)
    return math.nan


def weighted_mean(values: list[tuple[float, float]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values:
        if not math.isfinite(float(value)) or not math.isfinite(float(weight)) or float(weight) <= 0.0:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    return float(numerator / denominator) if denominator > 0.0 else math.nan


def parse_scales(raw: str, *, min_scale: float) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        if not item.strip():
            continue
        value = float(item)
        if value + 1.0e-12 < float(min_scale):
            continue
        values.append(float(value))
    values.append(1.0)
    return sorted(set(values), reverse=True)


def scaled_region_rows(rows: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    active = float(scale) > 1.0e-8
    for row in rows:
        item = dict(row)
        for key in (
            "core_balanced_delta",
            "delta_core_psnr",
            "delta_core_ssim",
            "delta_core_lpips",
            "crop_mean_abs_diff",
            "crop_max_abs_diff",
        ):
            if key in item:
                item[key] = num(item.get(key), 0.0) * float(scale)
        if "context_mse_regression" in item:
            item["context_mse_regression"] = max(0.0, num(item.get("context_mse_regression"), 0.0)) * float(scale) ** 2
        if "crop_nonzero_fraction" in item:
            item["crop_nonzero_fraction"] = num(row.get("crop_nonzero_fraction"), 0.0) if active else 0.0
        if "crop_nonzero_pixels" in item:
            item["crop_nonzero_pixels"] = int(round(row_changed_pixel_count(row))) if active else 0
        item["crop_changed"] = bool(row.get("crop_changed", False)) and active
        item["render_cvar_aggregate_subset_shrink_scale"] = float(scale)
        out.append(item)
    return out


def carrier_id_for_row(row: dict[str, Any]) -> str:
    decision = row.get("render_region_filter_decision")
    if isinstance(decision, dict):
        raw = decision.get("plan_carrier_id")
        if raw is not None:
            return str(raw)
    return str(row.get("carrier_id", row.get("carrier_seed_face", row.get("face_id", ""))))


def carrier_info(
    carrier_id: str,
    rows: list[dict[str, Any]],
    *,
    region_rows_by_carrier: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    decision: dict[str, Any] = {}
    for row in rows:
        raw = row.get("render_region_filter_decision")
        if isinstance(raw, dict):
            decision = raw
            break
    stats = decision.get("render_region_stats") if isinstance(decision.get("render_region_stats"), dict) else {}
    matched = decision.get("matched_region_carriers") if isinstance(decision.get("matched_region_carriers"), list) else []
    matched_ids = [str(item.get("carrier_id", "")) for item in matched if isinstance(item, dict)]
    raw_rows: list[dict[str, Any]] = []
    for match_id in matched_ids:
        raw_rows.extend(region_rows_by_carrier.get(match_id, []))

    reasons = []
    if not bool(decision.get("accepted", False)):
        reasons.append("source_filter_rejected")
    if bool(decision.get("severe_tail_rollback", False)):
        reasons.append("source_severe_tail_rollback")
    hard_reasons = risk_reasons(stats, args)
    reasons.extend(hard_reasons)
    return {
        "carrier_id": carrier_id,
        "rows": rows,
        "row_count": int(len(rows)),
        "decision": decision,
        "stats": stats,
        "matched_region_carriers": matched_ids,
        "raw_region_rows": raw_rows,
        "source_pass": bool(decision.get("accepted", False)) and not bool(decision.get("severe_tail_rollback", False)),
        "hard_pass": not reasons,
        "hard_reasons": reasons,
    }


def apply_tail_safe_carrier_shrink(carrier: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not bool(args.tail_safe_shrink_carriers) or not bool(carrier.get("source_pass", False)):
        return carrier
    raw_rows = carrier.get("raw_region_rows", [])
    if not raw_rows:
        return carrier
    original_stats = aggregate_rows(list(raw_rows), args)
    original_reasons = risk_reasons(original_stats, args)
    if not original_reasons:
        carrier["effective_raw_region_rows"] = list(raw_rows)
        carrier["aggregate_subset_shrink_scale"] = 1.0
        carrier["aggregate_subset_shrink"] = {
            "enabled": True,
            "applied": False,
            "scale": 1.0,
            "original_reasons": [],
            "decision_reasons": ["already_hard_pass"],
        }
        return carrier
    attempts: list[dict[str, Any]] = []
    for scale in parse_scales(str(args.tail_safe_shrink_scales), min_scale=float(args.tail_safe_shrink_min_scale)):
        scaled_rows = scaled_region_rows(list(raw_rows), scale)
        stats = aggregate_rows(scaled_rows, args)
        reasons = risk_reasons(stats, args)
        attempts.append({"scale": float(scale), "stats": stats, "reasons": reasons})
        if not reasons:
            out = dict(carrier)
            out["effective_raw_region_rows"] = scaled_rows
            out["stats"] = stats
            out["hard_pass"] = True
            out["hard_reasons"] = []
            out["aggregate_subset_shrink_scale"] = float(scale)
            out["aggregate_subset_shrink"] = {
                "enabled": True,
                "applied": abs(float(scale) - 1.0) > 1.0e-12,
                "scale": float(scale),
                "original_reasons": original_reasons,
                "attempts": attempts,
                "decision_reasons": ["tail_safe_carrier_shrink_passed"],
            }
            return out
    carrier["effective_raw_region_rows"] = list(raw_rows)
    carrier["aggregate_subset_shrink_scale"] = 1.0
    carrier["aggregate_subset_shrink"] = {
        "enabled": True,
        "applied": False,
        "scale": 1.0,
        "original_reasons": original_reasons,
        "attempts": attempts,
        "decision_reasons": ["tail_safe_carrier_shrink_no_safe_scale"],
    }
    return carrier


def risk_reasons(stats: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if int(num(stats.get("regions"), 0.0)) < int(args.min_regions):
        reasons.append(f"regions_below_{int(args.min_regions)}")
    if int(num(stats.get("changed_regions"), 0.0)) < int(args.min_changed_regions):
        reasons.append(f"changed_regions_below_{int(args.min_changed_regions)}")
    if num(stats.get("changed_fraction"), 0.0) < float(args.min_changed_fraction):
        reasons.append(f"changed_fraction_below_{float(args.min_changed_fraction):g}")
    if num(stats.get("mean_core_balanced_delta"), -math.inf) < float(args.min_mean_core_balanced_delta):
        reasons.append(f"mean_core_balanced_delta_below_{float(args.min_mean_core_balanced_delta):g}")
    if num(stats.get("mean_delta_core_psnr"), -math.inf) < float(args.min_mean_delta_psnr):
        reasons.append(f"mean_delta_core_psnr_below_{float(args.min_mean_delta_psnr):g}")
    if num(stats.get("tail_core_balanced_delta"), -math.inf) < float(args.min_tail_core_balanced_delta):
        reasons.append(f"tail_core_balanced_delta_below_{float(args.min_tail_core_balanced_delta):g}")
    if num(stats.get("negative_core_balanced_fraction"), math.inf) > float(args.max_negative_core_balanced_fraction):
        reasons.append(f"negative_core_balanced_fraction_above_{float(args.max_negative_core_balanced_fraction):g}")
    if num(stats.get("max_context_mse_regression"), math.inf) > float(args.max_context_mse_regression):
        reasons.append(f"context_mse_regression_above_{float(args.max_context_mse_regression):g}")
    return reasons


def dilution_reasons(stats: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    expected_views = int(args.expected_view_count)
    if (
        expected_views <= 0
        and (
            float(args.min_view_coverage_fraction) > 0.0
            or float(args.min_changed_view_coverage_fraction) > 0.0
        )
    ):
        reasons.append("expected_view_count_missing_for_coverage")
    if int(num(stats.get("unique_views"), 0.0)) < int(args.min_unique_views):
        reasons.append(f"unique_views_below_{int(args.min_unique_views)}")
    if int(num(stats.get("changed_unique_views"), 0.0)) < int(args.min_changed_unique_views):
        reasons.append(f"changed_unique_views_below_{int(args.min_changed_unique_views)}")
    if num(stats.get("view_coverage_fraction"), 0.0) < float(args.min_view_coverage_fraction):
        reasons.append(f"view_coverage_fraction_below_{float(args.min_view_coverage_fraction):g}")
    if num(stats.get("changed_view_coverage_fraction"), 0.0) < float(args.min_changed_view_coverage_fraction):
        reasons.append(
            f"changed_view_coverage_fraction_below_{float(args.min_changed_view_coverage_fraction):g}"
        )
    if int(num(stats.get("total_pixels"), 0.0)) < int(args.min_total_pixels):
        reasons.append(f"total_pixels_below_{int(args.min_total_pixels)}")
    if int(num(stats.get("changed_pixels"), 0.0)) < int(args.min_changed_pixels):
        reasons.append(f"changed_pixels_below_{int(args.min_changed_pixels)}")
    if num(stats.get("changed_pixel_fraction"), 0.0) < float(args.min_changed_pixel_fraction):
        reasons.append(f"changed_pixel_fraction_below_{float(args.min_changed_pixel_fraction):g}")
    if (
        float(args.min_full_frame_changed_pixel_fraction) > 0.0
        and not math.isfinite(num(stats.get("full_frame_changed_pixel_fraction"), math.nan))
    ):
        reasons.append("full_frame_changed_pixel_fraction_missing")
    elif num(stats.get("full_frame_changed_pixel_fraction"), 0.0) < float(
        args.min_full_frame_changed_pixel_fraction
    ):
        reasons.append(
            "full_frame_changed_pixel_fraction_below_"
            f"{float(args.min_full_frame_changed_pixel_fraction):g}"
        )
    if num(stats.get("area_weighted_core_balanced_delta"), -math.inf) < float(
        args.min_area_weighted_core_balanced_delta
    ):
        reasons.append(
            f"area_weighted_core_balanced_delta_below_{float(args.min_area_weighted_core_balanced_delta):g}"
        )
    if num(stats.get("dilution_adjusted_core_balanced_delta"), -math.inf) < float(
        args.min_dilution_adjusted_core_balanced_delta
    ):
        reasons.append(
            "dilution_adjusted_core_balanced_delta_below_"
            f"{float(args.min_dilution_adjusted_core_balanced_delta):g}"
        )
    if num(stats.get("full_frame_visibility_adjusted_delta"), -math.inf) < float(
        args.min_full_frame_visibility_adjusted_delta
    ):
        reasons.append(
            "full_frame_visibility_adjusted_delta_below_"
            f"{float(args.min_full_frame_visibility_adjusted_delta):g}"
        )
    return reasons


def aggregate_risk_reasons(stats: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return risk_reasons(stats, args) + dilution_reasons(stats, args)


def is_builder_shortfall(reason: str) -> bool:
    return reason.startswith(
        (
            "regions_below_",
            "changed_regions_below_",
            "unique_views_below_",
            "changed_unique_views_below_",
            "view_coverage_fraction_below_",
            "changed_view_coverage_fraction_below_",
            "total_pixels_below_",
            "changed_pixels_below_",
            "full_frame_changed_pixel_fraction_below_",
            "full_frame_visibility_adjusted_delta_below_",
        )
    )


def aggregate_rows(raw_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in raw_rows:
        key = (
            str(row.get("carrier_id", "")),
            str(row.get("view", row.get("view_name", ""))),
            json.dumps(row.get("bbox_xyxy", row.get("bbox", [])), sort_keys=True),
        )
        dedup[key] = row
    rows = list(dedup.values())
    changed = [row for row in rows if bool(row.get("crop_changed", False))]
    balanced = [num(row.get("core_balanced_delta")) for row in rows]
    dpsnr = [num(row.get("delta_core_psnr")) for row in rows]
    dssim = [num(row.get("delta_core_ssim")) for row in rows]
    dlpips = [num(row.get("delta_core_lpips")) for row in rows]
    context = [num(row.get("context_mse_regression")) for row in rows]
    unique_views = {str(row.get("view", row.get("view_name", ""))) for row in rows}
    unique_views.discard("")
    changed_views = {str(row.get("view", row.get("view_name", ""))) for row in changed}
    changed_views.discard("")
    total_pixels = float(sum(row_pixel_count(row) for row in rows))
    changed_pixels = float(sum(row_changed_pixel_count(row) for row in changed))
    full_frame_pixels = full_frame_denominator(rows, args)
    full_frame_changed_fraction = (
        changed_pixels / full_frame_pixels
        if math.isfinite(full_frame_pixels) and full_frame_pixels > 0.0
        else math.nan
    )
    expected_views = int(args.expected_view_count)
    view_denominator = float(expected_views) if expected_views > 0 else float(len(unique_views))
    view_coverage = float(len(unique_views)) / view_denominator if view_denominator > 0.0 else math.nan
    changed_view_coverage = float(len(changed_views)) / view_denominator if view_denominator > 0.0 else math.nan
    changed_pixel_fraction = changed_pixels / total_pixels if total_pixels > 0.0 else math.nan
    area_weighted_balanced = weighted_mean(
        [(num(row.get("core_balanced_delta")), row_pixel_count(row)) for row in rows]
    )
    dilution_adjusted = (
        area_weighted_balanced
        * (changed_view_coverage if math.isfinite(changed_view_coverage) else 1.0)
        * (changed_pixel_fraction if math.isfinite(changed_pixel_fraction) else 1.0)
        if math.isfinite(area_weighted_balanced)
        else math.nan
    )
    full_frame_visibility_adjusted = (
        area_weighted_balanced * full_frame_changed_fraction
        if math.isfinite(area_weighted_balanced) and math.isfinite(full_frame_changed_fraction)
        else math.nan
    )
    return {
        "source": "render_region_objective_rows",
        "regions": int(len(rows)),
        "changed_regions": int(len(changed)),
        "changed_fraction": float(len(changed)) / max(float(len(rows)), 1.0),
        "unique_views": int(len(unique_views)),
        "changed_unique_views": int(len(changed_views)),
        "expected_view_count": int(expected_views),
        "view_coverage_fraction": view_coverage,
        "changed_view_coverage_fraction": changed_view_coverage,
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "changed_pixel_fraction": changed_pixel_fraction,
        "full_frame_denominator_pixels": full_frame_pixels,
        "full_frame_changed_pixel_fraction": full_frame_changed_fraction,
        "mean_core_balanced_delta": mean(balanced),
        "area_weighted_core_balanced_delta": area_weighted_balanced,
        "dilution_adjusted_core_balanced_delta": dilution_adjusted,
        "full_frame_visibility_adjusted_delta": full_frame_visibility_adjusted,
        "mean_delta_core_psnr": mean(dpsnr),
        "mean_delta_core_ssim": mean(dssim),
        "mean_delta_core_lpips": mean(dlpips),
        "tail_core_balanced_delta": tail_cvar(balanced, float(args.tail_fraction)),
        "negative_core_balanced_fraction": (
            float(sum(1 for value in finite(balanced) if value < 0.0)) / max(float(len(finite(balanced))), 1.0)
        ),
        "max_context_mse_regression": max(finite(context), default=math.nan),
    }


def aggregate_stats_from_carriers(carriers: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    raw_rows: list[dict[str, Any]] = []
    for carrier in carriers:
        raw_rows.extend(carrier.get("effective_raw_region_rows", carrier.get("raw_region_rows", [])))
    if raw_rows:
        return aggregate_rows(raw_rows, args)

    weights = [max(1, int(num((carrier.get("stats") or {}).get("regions"), 1.0))) for carrier in carriers]
    total = max(1, sum(weights))

    def weighted(key: str, default: float = math.nan) -> float:
        values: list[float] = []
        for carrier, weight in zip(carriers, weights):
            value = num((carrier.get("stats") or {}).get(key), default)
            if math.isfinite(value):
                values.extend([value] * int(weight))
        return mean(values)

    balanced_values: list[float] = []
    for carrier, weight in zip(carriers, weights):
        stats = carrier.get("stats") or {}
        value = num(stats.get("tail_core_balanced_delta"), math.nan)
        if math.isfinite(value):
            balanced_values.extend([value] * int(weight))
    changed = sum(int(num((carrier.get("stats") or {}).get("changed_regions"), 0.0)) for carrier in carriers)
    regions = sum(int(num((carrier.get("stats") or {}).get("regions"), 0.0)) for carrier in carriers)
    negative = sum(
        int(round(num((carrier.get("stats") or {}).get("negative_core_balanced_fraction"), 0.0) * weight))
        for carrier, weight in zip(carriers, weights)
    )
    unique_views = sum(int(num((carrier.get("stats") or {}).get("unique_views"), 0.0)) for carrier in carriers)
    changed_unique_views = sum(
        int(num((carrier.get("stats") or {}).get("changed_unique_views"), 0.0)) for carrier in carriers
    )
    total_pixels = sum(num((carrier.get("stats") or {}).get("total_pixels"), 0.0) for carrier in carriers)
    changed_pixels = sum(num((carrier.get("stats") or {}).get("changed_pixels"), 0.0) for carrier in carriers)
    full_frame_denominators = [
        num((carrier.get("stats") or {}).get("full_frame_denominator_pixels"), math.nan)
        for carrier in carriers
    ]
    expected_views = int(args.expected_view_count)
    view_denominator = float(expected_views) if expected_views > 0 else float(unique_views)
    area_weighted_balanced = weighted("area_weighted_core_balanced_delta", weighted("mean_core_balanced_delta"))
    changed_view_coverage = (
        float(changed_unique_views) / view_denominator if view_denominator > 0.0 else math.nan
    )
    changed_pixel_fraction = float(changed_pixels) / float(total_pixels) if total_pixels > 0.0 else math.nan
    dilution_adjusted = (
        area_weighted_balanced
        * (changed_view_coverage if math.isfinite(changed_view_coverage) else 1.0)
        * (changed_pixel_fraction if math.isfinite(changed_pixel_fraction) else 1.0)
        if math.isfinite(area_weighted_balanced)
        else math.nan
    )
    full_frame_denominator_value = max(finite(full_frame_denominators), default=math.nan)
    if not math.isfinite(full_frame_denominator_value):
        full_frame_denominator_value = (
            float(int(args.expected_frame_pixels) * expected_views)
            if int(args.expected_frame_pixels) > 0 and expected_views > 0
            else math.nan
        )
    full_frame_changed_fraction = (
        float(changed_pixels) / full_frame_denominator_value
        if math.isfinite(full_frame_denominator_value) and full_frame_denominator_value > 0.0
        else math.nan
    )
    full_frame_visibility_adjusted = (
        area_weighted_balanced * full_frame_changed_fraction
        if math.isfinite(area_weighted_balanced) and math.isfinite(full_frame_changed_fraction)
        else math.nan
    )
    return {
        "source": "carrier_stats_weighted_fallback",
        "regions": int(regions),
        "changed_regions": int(changed),
        "changed_fraction": float(changed) / max(float(regions), 1.0),
        "unique_views": int(unique_views),
        "changed_unique_views": int(changed_unique_views),
        "expected_view_count": int(expected_views),
        "view_coverage_fraction": (
            float(unique_views) / view_denominator if view_denominator > 0.0 else math.nan
        ),
        "changed_view_coverage_fraction": changed_view_coverage,
        "total_pixels": float(total_pixels),
        "changed_pixels": float(changed_pixels),
        "changed_pixel_fraction": changed_pixel_fraction,
        "full_frame_denominator_pixels": full_frame_denominator_value,
        "full_frame_changed_pixel_fraction": full_frame_changed_fraction,
        "mean_core_balanced_delta": weighted("mean_core_balanced_delta"),
        "area_weighted_core_balanced_delta": area_weighted_balanced,
        "dilution_adjusted_core_balanced_delta": dilution_adjusted,
        "full_frame_visibility_adjusted_delta": full_frame_visibility_adjusted,
        "mean_delta_core_psnr": weighted("mean_delta_core_psnr"),
        "mean_delta_core_ssim": weighted("mean_delta_core_ssim"),
        "mean_delta_core_lpips": weighted("mean_delta_core_lpips"),
        "tail_core_balanced_delta": tail_cvar(balanced_values, float(args.tail_fraction)),
        "negative_core_balanced_fraction": float(negative) / float(total),
        "max_context_mse_regression": max(
            finite([num((carrier.get("stats") or {}).get("max_context_mse_regression")) for carrier in carriers]),
            default=math.nan,
        ),
    }


def sort_key(carrier: dict[str, Any], args: argparse.Namespace) -> tuple[Any, ...]:
    stats = carrier.get("stats") or {}
    if bool(args.prefer_full_frame_visibility):
        return (
            num(stats.get("max_context_mse_regression"), math.inf),
            -num(stats.get("full_frame_visibility_adjusted_delta"), -math.inf),
            -num(stats.get("full_frame_changed_pixel_fraction"), -math.inf),
            -num(stats.get("area_weighted_core_balanced_delta"), -math.inf),
            -num(stats.get("tail_core_balanced_delta"), -math.inf),
            -num(stats.get("mean_delta_core_psnr"), -math.inf),
            -num(stats.get("mean_core_balanced_delta"), -math.inf),
            num(stats.get("negative_core_balanced_fraction"), math.inf),
            str(carrier.get("carrier_id", "")),
        )
    return (
        num(stats.get("max_context_mse_regression"), math.inf),
        -num(stats.get("tail_core_balanced_delta"), -math.inf),
        -num(stats.get("mean_delta_core_psnr"), -math.inf),
        -num(stats.get("mean_core_balanced_delta"), -math.inf),
        num(stats.get("negative_core_balanced_fraction"), math.inf),
        str(carrier.get("carrier_id", "")),
    )


def annotate_row(row: dict[str, Any], subset_decision: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["render_cvar_aggregate_subset_decision"] = subset_decision
    return out


def main() -> int:
    args = parse_args()
    meta, rows = load_plan(args.input_plan)
    region_rows_by_carrier = load_region_rows(args.render_region_objective)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[carrier_id_for_row(row)].append(row)

    carriers = [
        carrier_info(
            carrier_id,
            carrier_rows,
            region_rows_by_carrier=region_rows_by_carrier,
            args=args,
        )
        for carrier_id, carrier_rows in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]
    carriers = [apply_tail_safe_carrier_shrink(carrier, args) for carrier in carriers]
    hard_pool = [carrier for carrier in carriers if bool(carrier.get("hard_pass", False))]

    selected: list[dict[str, Any]] = []
    rejected_after_aggregate: dict[str, list[str]] = {}
    aggregate_trace: list[dict[str, Any]] = []
    for carrier in sorted(hard_pool, key=lambda item: sort_key(item, args)):
        trial = selected + [carrier]
        aggregate = aggregate_stats_from_carriers(trial, args)
        reasons = aggregate_risk_reasons(aggregate, args)
        builder_shortfall = bool(reasons) and all(is_builder_shortfall(reason) for reason in reasons)
        trace = {
            "carrier_id": carrier["carrier_id"],
            "trial_selected_carriers": [item["carrier_id"] for item in trial],
            "aggregate_stats": aggregate,
            "aggregate_reasons": reasons,
            "accepted": not reasons or builder_shortfall,
            "builder_shortfall": builder_shortfall,
        }
        aggregate_trace.append(trace)
        if reasons and not builder_shortfall:
            rejected_after_aggregate[str(carrier["carrier_id"])] = reasons
            continue
        selected = trial

    if len(selected) < int(args.min_selected_carriers):
        rejected_after_aggregate["__subset__"] = [f"selected_carriers_below_{int(args.min_selected_carriers)}"]
        selected = []

    final_aggregate = aggregate_stats_from_carriers(selected, args) if selected else {}
    final_aggregate_reasons = aggregate_risk_reasons(final_aggregate, args) if selected else []
    if final_aggregate_reasons:
        rejected_after_aggregate["__subset__"] = final_aggregate_reasons
        selected = []
        final_aggregate = {}
    selected_ids = {str(carrier["carrier_id"]) for carrier in selected}
    alpha_json_path = args.output_alpha_json or args.output_plan.with_name("aggregate_subset_materialize_alpha.json")
    face_alphas: dict[str, float] = {}
    alpha_required = False
    for carrier in selected:
        scale = float(carrier.get("aggregate_subset_shrink_scale", 1.0))
        if abs(scale - 1.0) > 1.0e-12:
            alpha_required = True
        for row in carrier.get("rows", []):
            try:
                face_alphas[str(int(row.get("face_id")))] = float(scale)
            except Exception:
                continue
    selected_rows: list[dict[str, Any]] = []
    carrier_summaries: list[dict[str, Any]] = []
    for carrier in carriers:
        cid = str(carrier["carrier_id"])
        selected_flag = cid in selected_ids
        subset_decision = {
            "selected": bool(selected_flag),
            "policy": "fixed_train_render_aggregate_cvar_whole_carrier_subset",
            "hard_pass": bool(carrier.get("hard_pass", False)),
            "hard_reasons": carrier.get("hard_reasons", []),
            "aggregate_reasons": rejected_after_aggregate.get(cid, []),
            "final_aggregate_stats": final_aggregate if selected_flag else {},
            "shrink": carrier.get("aggregate_subset_shrink", {}),
            "shrink_scale": float(carrier.get("aggregate_subset_shrink_scale", 1.0)),
        }
        carrier_summaries.append(
            {
                "carrier_id": cid,
                "selected": bool(selected_flag),
                "row_count": int(carrier.get("row_count", 0)),
                "source_pass": bool(carrier.get("source_pass", False)),
                "hard_pass": bool(carrier.get("hard_pass", False)),
                "hard_reasons": carrier.get("hard_reasons", []),
                "aggregate_reasons": rejected_after_aggregate.get(cid, []),
                "shrink": carrier.get("aggregate_subset_shrink", {}),
                "shrink_scale": float(carrier.get("aggregate_subset_shrink_scale", 1.0)),
                "stats": carrier.get("stats", {}),
                "matched_region_carriers": carrier.get("matched_region_carriers", []),
            }
        )
        if selected_flag:
            selected_rows.extend(annotate_row(row, subset_decision) for row in carrier.get("rows", []))

    out_meta = dict(meta)
    previous_history = out_meta.get("render_cvar_aggregate_subset_history")
    if not isinstance(previous_history, list):
        previous_history = []
    contract = {
        "scene": str(args.scene),
        "test_usage": "none",
        "input_plan": str(args.input_plan),
        "input_plan_sha256": sha256_file(args.input_plan),
        "render_region_objective": str(args.render_region_objective) if args.render_region_objective else "",
        "render_region_objective_sha256": sha256_file(args.render_region_objective),
        "policy": "fixed_train_render_aggregate_cvar_whole_carrier_subset",
        "thresholds": {
            "min_regions": int(args.min_regions),
            "min_changed_regions": int(args.min_changed_regions),
            "min_changed_fraction": float(args.min_changed_fraction),
            "min_mean_core_balanced_delta": float(args.min_mean_core_balanced_delta),
            "min_mean_delta_psnr": float(args.min_mean_delta_psnr),
            "min_tail_core_balanced_delta": float(args.min_tail_core_balanced_delta),
            "max_negative_core_balanced_fraction": float(args.max_negative_core_balanced_fraction),
            "max_context_mse_regression": float(args.max_context_mse_regression),
            "tail_fraction": float(args.tail_fraction),
            "min_selected_carriers": int(args.min_selected_carriers),
            "expected_view_count": int(args.expected_view_count),
            "min_unique_views": int(args.min_unique_views),
            "min_changed_unique_views": int(args.min_changed_unique_views),
            "min_view_coverage_fraction": float(args.min_view_coverage_fraction),
            "min_changed_view_coverage_fraction": float(args.min_changed_view_coverage_fraction),
            "min_total_pixels": int(args.min_total_pixels),
            "min_changed_pixels": int(args.min_changed_pixels),
            "min_changed_pixel_fraction": float(args.min_changed_pixel_fraction),
            "expected_frame_pixels": int(args.expected_frame_pixels),
            "min_full_frame_changed_pixel_fraction": float(args.min_full_frame_changed_pixel_fraction),
            "min_area_weighted_core_balanced_delta": float(args.min_area_weighted_core_balanced_delta),
            "min_dilution_adjusted_core_balanced_delta": float(
                args.min_dilution_adjusted_core_balanced_delta
            ),
            "min_full_frame_visibility_adjusted_delta": float(
                args.min_full_frame_visibility_adjusted_delta
            ),
            "prefer_full_frame_visibility": bool(args.prefer_full_frame_visibility),
        },
        "input_rows": int(len(rows)),
        "input_carriers": int(len(carriers)),
        "hard_pool_carriers": int(len(hard_pool)),
        "selected_rows": int(len(selected_rows)),
        "selected_carriers": int(len(selected_ids)),
        "selected_carrier_ids": sorted(selected_ids),
        "final_aggregate_stats": final_aggregate,
        "final_aggregate_reasons": rejected_after_aggregate.get("__subset__", []),
        "tail_safe_shrink_carriers": bool(args.tail_safe_shrink_carriers),
        "tail_safe_shrink_scales": str(args.tail_safe_shrink_scales),
        "tail_safe_shrink_min_scale": float(args.tail_safe_shrink_min_scale),
        "prefer_full_frame_visibility": bool(args.prefer_full_frame_visibility),
        "materialize_alpha_json": str(alpha_json_path) if alpha_required else "",
        "materialize_alpha_face_count": int(len(face_alphas)) if alpha_required else 0,
    }
    out_meta.update(
        {
            "test_usage": "none",
            "render_cvar_aggregate_subset_filtered": True,
            "render_cvar_aggregate_subset": contract,
            "render_cvar_aggregate_subset_history": previous_history + [contract],
            "render_cvar_aggregate_subset_materialize_alpha_json": str(alpha_json_path) if alpha_required else "",
            "candidate_count": int(len(selected_rows)),
            "carrier_count": int(len(selected_ids)),
            "candidates": selected_rows,
        }
    )
    write_json(args.output_plan, out_meta)
    if alpha_required:
        write_json(
            alpha_json_path,
            {
                "operator": "render_cvar_aggregate_subset_tail_safe_carrier_shrink",
                "policy": "train_only_per_carrier_monotone_alpha_shrink",
                "test_usage": "none",
                "selection_uses_test": False,
                "candidate_plan": str(args.output_plan),
                "face_alphas": face_alphas,
                "selected_carrier_ids": sorted(selected_ids),
                "tail_safe_shrink_scales": str(args.tail_safe_shrink_scales),
                "tail_safe_shrink_min_scale": float(args.tail_safe_shrink_min_scale),
            },
        )

    summary = {
        **contract,
        "output_plan": str(args.output_plan),
        "carrier_rows": carrier_summaries,
        "aggregate_trace": aggregate_trace,
    }
    output_json = args.output_json or args.output_plan.with_suffix(".aggregate_subset.json")
    write_json(output_json, summary)

    output_md = args.output_md or args.output_plan.with_suffix(".aggregate_subset.md")
    lines = [
        "# Train Render-CVaR Aggregate Carrier Subset",
        "",
        f"- scene: `{args.scene}`",
        f"- input rows / carriers: `{len(rows)}` / `{len(carriers)}`",
        f"- hard-pool carriers: `{len(hard_pool)}`",
        f"- selected rows / carriers: `{len(selected_rows)}` / `{len(selected_ids)}`",
        f"- final aggregate: `{json.dumps(final_aggregate, sort_keys=True)}`",
        f"- final aggregate reasons: `{', '.join(rejected_after_aggregate.get('__subset__', [])) or 'pass'}`",
        f"- prefer full-frame visibility: `{str(bool(args.prefer_full_frame_visibility)).lower()}`",
        "",
        "| carrier | selected | rows | shrink | source pass | hard pass | mean balanced | area balanced | full-frame vis | full-frame changed | changed views | changed px frac | tail balanced | neg frac | max context | hard reasons | aggregate reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in carrier_summaries:
        stats = row.get("stats") or {}
        lines.append(
            f"| `{row['carrier_id']}` | {str(bool(row['selected'])).lower()} | {int(row['row_count'])} | "
            f"{float(row.get('shrink_scale', 1.0)):.6f} | "
            f"{str(bool(row['source_pass'])).lower()} | {str(bool(row['hard_pass'])).lower()} | "
            f"{num(stats.get('mean_core_balanced_delta')):+.9f} | "
            f"{num(stats.get('area_weighted_core_balanced_delta')):+.9f} | "
            f"{num(stats.get('full_frame_visibility_adjusted_delta')):+.9f} | "
            f"{num(stats.get('full_frame_changed_pixel_fraction')):.9f} | "
            f"{int(num(stats.get('changed_unique_views'), 0.0))} | "
            f"{num(stats.get('changed_pixel_fraction')):.6f} | "
            f"{num(stats.get('tail_core_balanced_delta')):+.9f} | "
            f"{num(stats.get('negative_core_balanced_fraction')):.6f} | "
            f"{num(stats.get('max_context_mse_regression')):.9g} | "
            f"`{', '.join(row['hard_reasons']) if row['hard_reasons'] else 'pass'}` | "
            f"`{', '.join(row['aggregate_reasons']) if row['aggregate_reasons'] else 'none'}` |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "scene": args.scene,
                "input_rows": len(rows),
                "input_carriers": len(carriers),
                "selected_rows": len(selected_rows),
                "selected_carriers": len(selected_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
