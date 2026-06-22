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
import hashlib
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
    parser.add_argument("--max_negative_core_balanced_fraction", type=float, default=1.0)
    parser.add_argument("--max_context_mse_regression", type=float, default=1.0e-6)
    parser.add_argument(
        "--min_mean_crop_abs_diff",
        type=float,
        default=0.0,
        help="Reject carriers whose average train-render crop change is below this normalized RGB threshold.",
    )
    parser.add_argument(
        "--min_max_crop_abs_diff",
        type=float,
        default=0.0,
        help="Reject carriers whose maximum train-render crop change is below this normalized RGB threshold.",
    )
    parser.add_argument("--drop_unmapped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_positive_plan_proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tail_safe_shrink_on_tail_fail", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tail_safe_shrink_min_scale", type=float, default=0.5)
    parser.add_argument(
        "--tail_safe_shrink_min_raw_scale",
        type=float,
        default=0.0,
        help=(
            "Reject tail-failure rescues when the analytic mean/tail shrink ratio "
            "falls below this value before applying --tail_safe_shrink_min_scale."
        ),
    )
    parser.add_argument(
        "--rollback_severe_tail_fail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reject whole plan carriers whose train-render tail CVaR deficit exceeds "
            "--rollback_tail_min_cvar_loss instead of admitting them through shrink."
        ),
    )
    parser.add_argument(
        "--rollback_tail_min_cvar_loss",
        type=float,
        default=0.0,
        help="Positive train-render tail CVaR deficit required before severe-tail rollback is applied.",
    )
    parser.add_argument("--risk_safe_shrink_on_train_risk_fail", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--risk_safe_shrink_min_scale", type=float, default=0.25)
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
        "max_negative_core_balanced_fraction",
        "min_mean_crop_abs_diff",
        "min_max_crop_abs_diff",
    ):
        if not math.isfinite(float(getattr(args, name))):
            parser.error(f"--{name} must be finite")
    for name in ("min_mean_crop_abs_diff", "min_max_crop_abs_diff"):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name} must be >= 0")
    if not 0.0 <= float(args.max_negative_core_balanced_fraction) <= 1.0:
        parser.error("--max_negative_core_balanced_fraction must be in [0, 1]")
    if not math.isfinite(float(args.tail_safe_shrink_min_scale)) or not 0.0 < float(args.tail_safe_shrink_min_scale) <= 1.0:
        parser.error("--tail_safe_shrink_min_scale must be in (0, 1]")
    if not math.isfinite(float(args.tail_safe_shrink_min_raw_scale)) or not 0.0 <= float(args.tail_safe_shrink_min_raw_scale) <= 1.0:
        parser.error("--tail_safe_shrink_min_raw_scale must be in [0, 1]")
    if not math.isfinite(float(args.rollback_tail_min_cvar_loss)) or float(args.rollback_tail_min_cvar_loss) < 0.0:
        parser.error("--rollback_tail_min_cvar_loss must be finite and >= 0")
    if not math.isfinite(float(args.risk_safe_shrink_min_scale)) or not 0.0 < float(args.risk_safe_shrink_min_scale) <= 1.0:
        parser.error("--risk_safe_shrink_min_scale must be in (0, 1]")
    return args


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _num(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _row_pixel_count(row: dict[str, Any]) -> float:
    value = _num(row.get("pixels"), math.nan)
    if math.isfinite(value) and value > 0.0:
        return float(value)
    bbox = row.get("bbox_xyxy", row.get("bbox", []))
    if isinstance(bbox, list) and len(bbox) >= 4:
        width = max(0.0, _num(bbox[2], 0.0) - _num(bbox[0], 0.0))
        height = max(0.0, _num(bbox[3], 0.0) - _num(bbox[1], 0.0))
        area = width * height
        if area > 0.0:
            return float(area)
    return 0.0


def _row_changed_pixel_count(row: dict[str, Any]) -> float:
    pixels = _row_pixel_count(row)
    value = _num(row.get("crop_nonzero_pixels"), math.nan)
    if math.isfinite(value) and value >= 0.0:
        return float(min(value, pixels)) if pixels > 0.0 else float(value)
    fraction = _num(row.get("crop_nonzero_fraction"), math.nan)
    if math.isfinite(fraction) and fraction >= 0.0:
        return float(pixels * min(fraction, 1.0))
    return pixels if bool(row.get("crop_changed", False)) else 0.0


def _row_frame_pixel_count(row: dict[str, Any]) -> float:
    value = _num(row.get("frame_pixels"), math.nan)
    if math.isfinite(value) and value > 0.0:
        return float(value)
    width = _num(row.get("image_width"), math.nan)
    height = _num(row.get("image_height"), math.nan)
    if math.isfinite(width) and math.isfinite(height) and width > 0.0 and height > 0.0:
        return float(width * height)
    return 0.0


def _full_frame_denominator(rows: list[dict[str, Any]]) -> float:
    frame_pixels_by_view: dict[str, float] = {}
    for row in rows:
        view = str(row.get("view", row.get("view_name", "")))
        frame_pixels = _row_frame_pixel_count(row)
        if view and frame_pixels > 0.0:
            frame_pixels_by_view.setdefault(view, frame_pixels)
    return float(sum(frame_pixels_by_view.values())) if frame_pixels_by_view else math.nan


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values:
        if not math.isfinite(float(value)) or not math.isfinite(float(weight)) or float(weight) <= 0.0:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    return float(numerator / denominator) if denominator > 0.0 else math.nan


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
    unique_views = {str(row.get("view", row.get("view_name", ""))) for row in rows}
    unique_views.discard("")
    changed_views = {str(row.get("view", row.get("view_name", ""))) for row in changed}
    changed_views.discard("")
    total_pixels = float(sum(_row_pixel_count(row) for row in rows))
    changed_pixels = float(sum(_row_changed_pixel_count(row) for row in changed))
    full_frame_pixels = _full_frame_denominator(rows)
    full_frame_changed_fraction = (
        changed_pixels / full_frame_pixels
        if math.isfinite(full_frame_pixels) and full_frame_pixels > 0.0
        else math.nan
    )
    changed_pixel_fraction = changed_pixels / total_pixels if total_pixels > 0.0 else math.nan
    area_weighted_balanced = _weighted_mean(
        [(float(row.get("core_balanced_delta", math.nan)), _row_pixel_count(row)) for row in rows]
    )
    full_frame_visibility_adjusted = (
        area_weighted_balanced * full_frame_changed_fraction
        if math.isfinite(area_weighted_balanced) and math.isfinite(full_frame_changed_fraction)
        else math.nan
    )
    return {
        "regions": int(len(rows)),
        "changed_regions": int(len(changed)),
        "changed_fraction": float(len(changed)) / max(float(len(rows)), 1.0),
        "unique_views": int(len(unique_views)),
        "changed_unique_views": int(len(changed_views)),
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "changed_pixel_fraction": changed_pixel_fraction,
        "full_frame_denominator_pixels": full_frame_pixels,
        "full_frame_changed_pixel_fraction": full_frame_changed_fraction,
        "mean_core_balanced_delta": _mean(balanced),
        "area_weighted_core_balanced_delta": area_weighted_balanced,
        "full_frame_visibility_adjusted_delta": full_frame_visibility_adjusted,
        "mean_delta_core_psnr": _mean(dpsnr),
        "mean_delta_core_ssim": _mean(dssim),
        "mean_delta_core_lpips": _mean(dlpips),
        "tail_core_balanced_delta": _tail_cvar(balanced, 0.25),
        "negative_core_balanced_fraction": (
            float(sum(1 for value in _finite(balanced) if value < 0.0)) / max(float(len(_finite(balanced))), 1.0)
        ),
        "max_context_mse_regression": max(_finite(context), default=math.nan),
        "max_crop_abs_diff": max(
            _finite([float(row.get("crop_max_abs_diff", math.nan)) for row in rows]),
            default=math.nan,
        ),
        "mean_crop_abs_diff": _mean([float(row.get("crop_mean_abs_diff", math.nan)) for row in rows]),
    }


def _tail_cvar_deficit(stats: dict[str, Any], args: argparse.Namespace) -> float:
    tail_delta = float(stats.get("tail_core_balanced_delta", math.nan))
    min_tail = float(args.min_tail_core_balanced_delta)
    if not (math.isfinite(tail_delta) and math.isfinite(min_tail)):
        return 0.0
    return float(max(0.0, min_tail - tail_delta))


def _bad_tail_attribution(
    rows: list[dict[str, Any]],
    *,
    plan_faces: set[int],
    region_carrier_faces: dict[str, set[int]],
) -> dict[str, Any]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        try:
            score = float(row.get("core_balanced_delta", math.nan))
        except Exception:
            continue
        if math.isfinite(score):
            scored.append((score, row))
    scored.sort(key=lambda item: item[0])
    tail_count = max(1, int(math.ceil(0.25 * len(scored)))) if scored else 0
    bad_rows: list[dict[str, Any]] = []
    bad_region_carriers: set[str] = set()
    bad_plan_faces: set[int] = set()
    for _, row in scored[:tail_count]:
        carrier_id = str(row.get("carrier_id", ""))
        bad_region_carriers.add(carrier_id)
        overlap_faces = sorted(int(face) for face in (region_carrier_faces.get(carrier_id, set()) & plan_faces))
        bad_plan_faces.update(overlap_faces)
        bad_rows.append(
            {
                "carrier_id": carrier_id,
                "view_name": row.get("view_name", row.get("view", "")),
                "bbox_xyxy": row.get("bbox_xyxy", row.get("bbox", [])),
                "core_balanced_delta": float(row.get("core_balanced_delta", math.nan)),
                "delta_core_psnr": float(row.get("delta_core_psnr", math.nan)),
                "delta_core_ssim": float(row.get("delta_core_ssim", math.nan)),
                "delta_core_lpips": float(row.get("delta_core_lpips", math.nan)),
                "context_mse_regression": float(row.get("context_mse_regression", math.nan)),
                "crop_changed": bool(row.get("crop_changed", False)),
                "overlap_plan_faces": overlap_faces,
            }
        )
    return {
        "tail_fraction": 0.25,
        "bad_tail_rows": bad_rows,
        "bad_tail_row_count": int(len(bad_rows)),
        "bad_tail_region_carriers": sorted(bad_region_carriers),
        "bad_tail_plan_faces": sorted(bad_plan_faces),
        "bad_tail_plan_face_count": int(len(bad_plan_faces)),
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
    if not mapped:
        return (not reasons), reasons
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
    if float(stats.get("negative_core_balanced_fraction", 1.0)) > float(args.max_negative_core_balanced_fraction):
        reasons.append(f"negative_core_balanced_fraction_above_{float(args.max_negative_core_balanced_fraction):g}")
    if float(stats.get("max_context_mse_regression", math.inf)) > float(args.max_context_mse_regression):
        reasons.append(f"context_mse_regression_above_{float(args.max_context_mse_regression):g}")
    if float(stats.get("mean_crop_abs_diff", 0.0)) < float(args.min_mean_crop_abs_diff):
        reasons.append(f"mean_crop_abs_diff_below_{float(args.min_mean_crop_abs_diff):g}")
    if float(stats.get("max_crop_abs_diff", 0.0)) < float(args.min_max_crop_abs_diff):
        reasons.append(f"max_crop_abs_diff_below_{float(args.min_max_crop_abs_diff):g}")
    return (not reasons), reasons


def _tail_safe_shrink_scales(stats: dict[str, Any], args: argparse.Namespace) -> tuple[float, float]:
    mean_delta = float(stats.get("mean_core_balanced_delta", math.nan))
    tail_delta = float(stats.get("tail_core_balanced_delta", math.nan))
    min_tail = float(args.min_tail_core_balanced_delta)
    if not (math.isfinite(mean_delta) and math.isfinite(tail_delta)):
        return 1.0, 1.0
    if mean_delta <= 0.0 or tail_delta >= min_tail:
        return 1.0, 1.0
    deficit = abs(tail_delta - min_tail)
    raw_scale = float(min(1.0, mean_delta / max(mean_delta + deficit, 1.0e-12)))
    effective_scale = float(max(float(args.tail_safe_shrink_min_scale), raw_scale))
    return effective_scale, raw_scale


def _tail_safe_shrink_scale(stats: dict[str, Any], args: argparse.Namespace) -> float:
    return _tail_safe_shrink_scales(stats, args)[0]


def _risk_safe_shrink_scale(stats: dict[str, Any], args: argparse.Namespace) -> float:
    scales = [1.0]
    tail_scale = _tail_safe_shrink_scale(stats, args)
    if tail_scale < 1.0:
        scales.append(tail_scale)
    max_context = float(stats.get("max_context_mse_regression", math.nan))
    allowed_context = float(args.max_context_mse_regression)
    if math.isfinite(max_context) and math.isfinite(allowed_context) and max_context > allowed_context > 0.0:
        scales.append(math.sqrt(allowed_context / max(max_context, 1.0e-12)))
    scale = min(scales)
    return float(max(float(args.risk_safe_shrink_min_scale), min(1.0, scale)))


def _scaled_candidate_row(row: dict[str, Any], scale: float, raw_scale: float) -> dict[str, Any]:
    out = dict(row)
    coeff = out.get("delta_coeff")
    if isinstance(coeff, list):
        try:
            out["delta_coeff"] = [
                [
                    [float(value) * float(scale) for value in channel]
                    for channel in basis
                ]
                for basis in coeff
            ]
        except Exception:
            out["delta_coeff"] = coeff
    out["render_region_tail_safe_shrink"] = {
        "enabled": True,
        "scale": float(scale),
        "raw_scale": float(raw_scale),
        "policy": "analytic_mean_tail_ratio_train_only",
    }
    return out


def main() -> int:
    args = parse_args()
    plan_meta, plan_rows = _load_plan(args.candidate_plan)
    region_carrier_faces, face_to_region_carriers = _load_region_carriers(args.carrier_json)
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
        tail_deficit = _tail_cvar_deficit(stats, args)
        tail_attribution = _bad_tail_attribution(
            region_rows,
            plan_faces=plan_faces,
            region_carrier_faces=region_carrier_faces,
        )
        proxy_positive = _plan_proxy_positive(rows)
        passed, reasons = _passes(
            stats,
            args=args,
            mapped=bool(matches),
            proxy_positive=proxy_positive,
        )
        notes: list[str] = []
        if not matches:
            notes.append("render_region_unmapped_proxy_fallback")
        shrink_scale = 1.0
        shrink_raw_scale = 1.0
        shrink_applied = False
        tail_reasons_only = bool(reasons) and all(
            reason.startswith("tail_core_balanced_delta_below_") for reason in reasons
        )
        tail_reason_present = any(reason.startswith("tail_core_balanced_delta_below_") for reason in reasons)
        tail_shrink_scale, tail_shrink_raw_scale = _tail_safe_shrink_scales(stats, args)
        if tail_reason_present:
            shrink_scale = tail_shrink_scale
            shrink_raw_scale = tail_shrink_raw_scale
        severe_tail_rollback = (
            bool(args.rollback_severe_tail_fail)
            and tail_reason_present
            and tail_deficit >= float(args.rollback_tail_min_cvar_loss)
        )
        if severe_tail_rollback:
            notes.append("severe_tail_cvar_rollback")
        tail_shrink_allowed = tail_shrink_raw_scale >= float(args.tail_safe_shrink_min_raw_scale)
        if tail_reason_present and not tail_shrink_allowed:
            notes.append("tail_safe_shrink_raw_scale_below_min")
        if (
            not passed
            and bool(args.tail_safe_shrink_on_tail_fail)
            and bool(matches)
            and tail_reasons_only
            and tail_shrink_allowed
            and not severe_tail_rollback
        ):
            if shrink_scale < 1.0:
                passed = True
                shrink_applied = True
                notes.append("tail_safe_shrink_applied")
        if (
            not passed
            and bool(args.risk_safe_shrink_on_train_risk_fail)
            and bool(matches)
            and reasons
            and all(
                reason.startswith("tail_core_balanced_delta_below_")
                or reason.startswith("context_mse_regression_above_")
                for reason in reasons
            )
            and (not tail_reason_present or tail_shrink_allowed)
            and not severe_tail_rollback
        ):
            shrink_scale = _risk_safe_shrink_scale(stats, args)
            shrink_raw_scale = tail_shrink_raw_scale if tail_reason_present else shrink_scale
            if shrink_scale < 1.0:
                passed = True
                shrink_applied = True
                notes.append("risk_safe_shrink_applied")
        row = {
            "plan_carrier_id": carrier_id,
            "plan_faces": sorted(plan_faces),
            "plan_face_count": int(len(plan_faces)),
            "plan_rows": int(len(rows)),
            "matched_region_carriers": matches,
            "render_region_stats": stats,
            "plan_proxy_positive": bool(proxy_positive),
            "accepted": bool(passed),
            "tail_safe_shrink_applied": bool(shrink_applied),
            "tail_safe_shrink_scale": float(shrink_scale),
            "tail_safe_shrink_raw_scale": float(shrink_raw_scale),
            "tail_cvar_deficit": float(tail_deficit),
            "severe_tail_rollback": bool(severe_tail_rollback),
            "bad_tail_attribution": tail_attribution,
            "decision_reasons": reasons,
            "decision_notes": notes,
        }
        carrier_rows.append(row)
        if passed:
            for candidate_row in rows:
                annotated = (
                    _scaled_candidate_row(candidate_row, shrink_scale, shrink_raw_scale)
                    if shrink_applied
                    else dict(candidate_row)
                )
                annotated["render_region_filter_decision"] = {
                    "accepted": bool(passed),
                    "mapped": bool(matches),
                    "plan_carrier_id": carrier_id,
                    "decision_reasons": reasons,
                    "decision_notes": notes,
                    "tail_safe_shrink_applied": bool(shrink_applied),
                    "tail_safe_shrink_scale": float(shrink_scale),
                    "tail_safe_shrink_raw_scale": float(shrink_raw_scale),
                    "tail_cvar_deficit": float(tail_deficit),
                    "severe_tail_rollback": bool(severe_tail_rollback),
                    "bad_tail_attribution": tail_attribution,
                    "render_region_stats": stats,
                    "matched_region_carriers": matches,
                }
                kept_rows.append(annotated)

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
            "source_candidate_plan_sha256": _sha256_file(args.candidate_plan),
            "source_render_region_objective_sha256": _sha256_file(args.render_region_objective),
            "source_region_carrier_json_sha256": _sha256_file(args.carrier_json),
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
                "unmapped_policy": "reject" if bool(args.drop_unmapped) else "proxy_positive_pass_through",
                "require_positive_plan_proxy": bool(args.require_positive_plan_proxy),
                "tail_safe_shrink_on_tail_fail": bool(args.tail_safe_shrink_on_tail_fail),
                "tail_safe_shrink_min_scale": float(args.tail_safe_shrink_min_scale),
                "tail_safe_shrink_min_raw_scale": float(args.tail_safe_shrink_min_raw_scale),
                "rollback_severe_tail_fail": bool(args.rollback_severe_tail_fail),
                "rollback_tail_min_cvar_loss": float(args.rollback_tail_min_cvar_loss),
                "risk_safe_shrink_on_train_risk_fail": bool(args.risk_safe_shrink_on_train_risk_fail),
                "risk_safe_shrink_min_scale": float(args.risk_safe_shrink_min_scale),
                "thresholds": {
                    "min_regions": int(args.min_regions),
                    "min_changed_regions": int(args.min_changed_regions),
                    "min_changed_fraction": float(args.min_changed_fraction),
                    "min_mean_core_balanced_delta": float(args.min_mean_core_balanced_delta),
                    "min_mean_delta_psnr": float(args.min_mean_delta_psnr),
                    "min_tail_core_balanced_delta": float(args.min_tail_core_balanced_delta),
                    "max_negative_core_balanced_fraction": float(args.max_negative_core_balanced_fraction),
                    "max_context_mse_regression": float(args.max_context_mse_regression),
                    "min_mean_crop_abs_diff": float(args.min_mean_crop_abs_diff),
                    "min_max_crop_abs_diff": float(args.min_max_crop_abs_diff),
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
        "candidate_plan_sha256": _sha256_file(args.candidate_plan),
        "render_region_objective_sha256": _sha256_file(args.render_region_objective),
        "carrier_json_sha256": _sha256_file(args.carrier_json),
        "output_plan": str(args.output_plan),
        "rollback_severe_tail_fail": bool(args.rollback_severe_tail_fail),
        "rollback_tail_min_cvar_loss": float(args.rollback_tail_min_cvar_loss),
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
        "| plan carrier | accepted | rows | faces | matched region carriers | regions | changed | mean balanced | mean dPSNR | tail balanced | tail deficit | rollback | shrink raw/eff | neg frac | max context reg | mean diff | max diff | bad tail faces | reasons | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
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
            f"{float(row.get('tail_cvar_deficit', 0.0)):.9f} | "
            f"{str(bool(row.get('severe_tail_rollback', False))).lower()} | "
            f"{float(row.get('tail_safe_shrink_raw_scale', 1.0)):.6f}/{float(row.get('tail_safe_shrink_scale', 1.0)):.6f} | "
            f"{float(stats.get('negative_core_balanced_fraction', math.nan)):.6f} | "
            f"{float(stats.get('max_context_mse_regression', math.nan)):.9g} | "
            f"{float(stats.get('mean_crop_abs_diff', math.nan)):.9f} | "
            f"{float(stats.get('max_crop_abs_diff', math.nan)):.9f} | "
            f"{int(row.get('bad_tail_attribution', {}).get('bad_tail_plan_face_count', 0))} | "
            f"`{', '.join(row['decision_reasons']) if row['decision_reasons'] else 'pass'}` | "
            f"`{', '.join(row['decision_notes']) if row['decision_notes'] else 'none'}` |"
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("scene", "input_rows", "input_carriers", "kept_rows", "kept_carriers")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
