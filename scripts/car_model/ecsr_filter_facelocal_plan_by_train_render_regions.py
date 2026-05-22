#!/usr/bin/env python3
"""Filter a face-local candidate plan using raw train render-region rows.

This is a standalone JSON post-filter.  It does not render, load a model, or
modify the Phase-K/S pipeline.  The intended use is to take an already written
face-local candidate plan, the corresponding train_render_region_objective JSON,
and the render-visible carrier JSON, then keep only whole carriers whose visible
region rows show positive, nonzero train-render evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_plan_json", type=Path, required=True)
    parser.add_argument("--render_region_objective_json", type=Path, required=True)
    parser.add_argument("--carrier_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument(
        "--metric_policy",
        choices=("balanced_positive", "any_core_positive", "all_core_positive"),
        default="balanced_positive",
        help=(
            "Region pass rule. balanced_positive requires core_balanced_delta > 0. "
            "any_core_positive accepts any positive PSNR/SSIM/LPIPS-improvement/balanced signal. "
            "all_core_positive requires positive PSNR, SSIM, balanced, and non-regressing LPIPS "
            "when LPIPS was computed."
        ),
    )
    parser.add_argument(
        "--carrier_policy",
        choices=("any_region_pass", "all_regions_pass"),
        default="any_region_pass",
        help="Keep a whole carrier if any associated visible region passes, or only if all associated regions pass.",
    )
    parser.add_argument("--min_pass_regions", type=int, default=1)
    parser.add_argument("--require_crop_changed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow_missing_region_rows",
        action="store_true",
        help="Keep carriers with no objective rows. Default drops them because they lack train-render evidence.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def norm_carrier_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def face_id(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("face_id"))
    except Exception:
        return None


def plan_rows(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)], None
    if not isinstance(payload, dict):
        return [], None
    for key in ("candidates", "accepted", "accepted_preview"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)], key
    return [], None


def carrier_face_index(payload: Any) -> tuple[dict[str, set[int]], dict[int, str]]:
    by_carrier: dict[str, set[int]] = {}
    by_face: dict[int, str] = {}
    carriers = payload.get("carriers", []) if isinstance(payload, dict) else []
    if not isinstance(carriers, list):
        return by_carrier, by_face
    for carrier in carriers:
        if not isinstance(carrier, dict):
            continue
        cid = norm_carrier_id(carrier.get("carrier_id", ""))
        if not cid:
            continue
        faces: set[int] = set()
        for value in carrier.get("face_ids", []) if isinstance(carrier.get("face_ids"), list) else []:
            try:
                faces.add(int(value))
            except Exception:
                continue
        for face in carrier.get("faces", []) if isinstance(carrier.get("faces"), list) else []:
            if not isinstance(face, dict):
                continue
            try:
                faces.add(int(face.get("face_id")))
            except Exception:
                continue
        by_carrier[cid] = faces
        for fid in faces:
            by_face.setdefault(fid, cid)
    return by_carrier, by_face


def row_carrier_id(
    row: dict[str, Any],
    face_to_carrier: dict[int, str],
    known_carrier_ids: set[str],
) -> str:
    raw = row.get("carrier_id")
    raw_cid = norm_carrier_id(raw) if raw is not None and str(raw).strip() else ""
    if raw_cid and raw_cid in known_carrier_ids:
        return raw_cid
    fid = face_id(row)
    if fid is not None and fid in face_to_carrier:
        return face_to_carrier[fid]
    carrier_faces = row.get("carrier_faces")
    if isinstance(carrier_faces, list):
        for value in carrier_faces:
            try:
                fid = int(value)
            except Exception:
                continue
            if fid in face_to_carrier:
                return face_to_carrier[fid]
    return raw_cid


def objective_rows_by_carrier(payload: Any) -> dict[str, list[dict[str, Any]]]:
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    out: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        cid = norm_carrier_id(row.get("carrier_id", ""))
        if not cid:
            continue
        out.setdefault(cid, []).append(row)
    return out


def lpips_computed(row: dict[str, Any]) -> bool:
    return math.isfinite(finite_float(row.get("delta_core_lpips")))


def region_passes(row: dict[str, Any], *, metric_policy: str, require_crop_changed: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if require_crop_changed and not bool(row.get("crop_changed", False)):
        reasons.append("crop_not_changed")

    balanced = finite_float(row.get("core_balanced_delta"))
    dpsnr = finite_float(row.get("delta_core_psnr"))
    dssim = finite_float(row.get("delta_core_ssim"))
    dlpips = finite_float(row.get("delta_core_lpips"))
    lpips_ok = (not lpips_computed(row)) or dlpips < 0.0

    if metric_policy == "balanced_positive":
        if not (balanced > 0.0):
            reasons.append("core_balanced_not_positive")
    elif metric_policy == "any_core_positive":
        any_positive = balanced > 0.0 or dpsnr > 0.0 or dssim > 0.0 or (lpips_computed(row) and dlpips < 0.0)
        if not any_positive:
            reasons.append("no_positive_core_metric")
    elif metric_policy == "all_core_positive":
        if not (balanced > 0.0):
            reasons.append("core_balanced_not_positive")
        if not (dpsnr > 0.0):
            reasons.append("core_psnr_not_positive")
        if not (dssim > 0.0):
            reasons.append("core_ssim_not_positive")
        if not lpips_ok:
            reasons.append("core_lpips_regressed")
    else:
        reasons.append(f"unknown_metric_policy:{metric_policy}")
    return not reasons, reasons


def carrier_passes(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    if not rows:
        accepted = bool(args.allow_missing_region_rows)
        return accepted, {
            "accepted": accepted,
            "region_count": 0,
            "passing_region_count": 0,
            "decision_reasons": [] if accepted else ["no_render_region_objective_rows"],
            "region_decisions": [],
        }

    decisions = []
    passing = 0
    reasons: list[str] = []
    for row in rows:
        ok, row_reasons = region_passes(
            row,
            metric_policy=str(args.metric_policy),
            require_crop_changed=bool(args.require_crop_changed),
        )
        passing += int(ok)
        decisions.append(
            {
                "view": row.get("view", ""),
                "bbox_xyxy": row.get("bbox_xyxy", []),
                "accepted": bool(ok),
                "decision_reasons": row_reasons,
                "core_balanced_delta": row.get("core_balanced_delta"),
                "delta_core_psnr": row.get("delta_core_psnr"),
                "delta_core_ssim": row.get("delta_core_ssim"),
                "delta_core_lpips": row.get("delta_core_lpips"),
                "crop_changed": bool(row.get("crop_changed", False)),
                "crop_nonzero_pixels": row.get("crop_nonzero_pixels", 0),
            }
        )

    if passing < int(args.min_pass_regions):
        reasons.append(f"passing_regions_below_{int(args.min_pass_regions)}")
    if str(args.carrier_policy) == "all_regions_pass" and passing < len(rows):
        reasons.append("not_all_regions_passed")
    accepted = not reasons
    return accepted, {
        "accepted": bool(accepted),
        "region_count": int(len(rows)),
        "passing_region_count": int(passing),
        "decision_reasons": reasons,
        "region_decisions": decisions,
    }


def filtered_plan(args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    plan_payload = load_json(args.candidate_plan_json)
    objective_payload = load_json(args.render_region_objective_json)
    carrier_payload = load_json(args.carrier_json)

    rows, row_key = plan_rows(plan_payload)
    carrier_by_id, face_to_carrier = carrier_face_index(carrier_payload)
    objective_by_carrier = objective_rows_by_carrier(objective_payload)
    known_carrier_ids = set(carrier_by_id) | set(objective_by_carrier)

    carrier_audit: dict[str, dict[str, Any]] = {}
    kept_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    kept_carriers: set[str] = set()
    rejected_carriers: set[str] = set()

    for row in rows:
        cid = row_carrier_id(row, face_to_carrier, known_carrier_ids)
        if cid not in carrier_audit:
            accepted, audit = carrier_passes(objective_by_carrier.get(cid, []), args=args)
            audit["carrier_id"] = cid
            carrier_audit[cid] = audit
        accepted = bool(carrier_audit[cid]["accepted"])
        annotated = copy.deepcopy(row)
        annotated.setdefault("render_region_filter", {})
        annotated["render_region_filter"] = {
            "carrier_id": cid,
            "accepted": accepted,
            "carrier_region_count": carrier_audit[cid]["region_count"],
            "carrier_passing_region_count": carrier_audit[cid]["passing_region_count"],
            "decision_reasons": carrier_audit[cid]["decision_reasons"],
        }
        if accepted:
            kept_rows.append(annotated)
            kept_carriers.add(cid)
        else:
            rejected_rows.append(annotated)
            rejected_carriers.add(cid)

    summary = {
        "input_candidate_rows": int(len(rows)),
        "kept_candidate_rows": int(len(kept_rows)),
        "rejected_candidate_rows": int(len(rejected_rows)),
        "input_carriers_seen": int(len(carrier_audit)),
        "kept_carriers": int(len(kept_carriers)),
        "rejected_carriers": int(len(rejected_carriers)),
    }

    if isinstance(plan_payload, list):
        return kept_rows

    out = copy.deepcopy(plan_payload) if isinstance(plan_payload, dict) else {}
    if row_key is None:
        row_key = "candidates"
    out[row_key] = kept_rows
    out["candidate_count"] = int(len(kept_rows))
    if "carrier_count" in out or kept_carriers:
        out["carrier_count"] = int(len(kept_carriers))
    out["render_region_filter_audit"] = {
        "protocol": "facelocal_candidate_plan_filtered_by_raw_train_render_region_objective_rows",
        "test_usage": "none",
        "candidate_plan_json": str(args.candidate_plan_json),
        "render_region_objective_json": str(args.render_region_objective_json),
        "carrier_json": str(args.carrier_json),
        "metric_policy": str(args.metric_policy),
        "carrier_policy": str(args.carrier_policy),
        "min_pass_regions": int(args.min_pass_regions),
        "require_crop_changed": bool(args.require_crop_changed),
        "allow_missing_region_rows": bool(args.allow_missing_region_rows),
        "summary": summary,
        "carrier_decisions": [carrier_audit[key] for key in sorted(carrier_audit)],
    }
    return out


def main() -> int:
    args = parse_args()
    if int(args.min_pass_regions) < 1:
        raise SystemExit("--min_pass_regions must be >= 1")
    for path_arg in ("candidate_plan_json", "render_region_objective_json", "carrier_json"):
        path = getattr(args, path_arg)
        if not path.is_file():
            raise SystemExit(f"--{path_arg} does not exist: {path}")
    write_json(args.output_json, filtered_plan(args))
    print(json.dumps({"output_json": str(args.output_json)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
