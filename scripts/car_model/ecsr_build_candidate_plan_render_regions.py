#!/usr/bin/env python3
"""Build train-only render-region carriers owned by a candidate plan.

The older render-region gate uses a precomputed list of high-residual train
regions.  That list is useful as a generic scene prior, but it can miss the
exact faces selected by a later face-local candidate plan.  This script builds
carrier regions directly from the candidate plan faces and the train evidence
NPZ files, so downstream render-region evaluation can certify the pixels that
the proposed edit actually owns.

No held-out test renders or metrics are read.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_plan", type=Path, required=True)
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--scene", default="")
    parser.add_argument("--max_regions_per_carrier", type=int, default=12)
    parser.add_argument("--min_pixels", type=int, default=16)
    parser.add_argument("--bbox_pad", type=int, default=8)
    parser.add_argument("--min_alpha", type=float, default=0.01)
    parser.add_argument("--high_error_quantile", type=float, default=0.0)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--expand_faces_in_region_bbox", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expand_min_face_pixels", type=int, default=12)
    parser.add_argument("--expand_min_face_views", type=int, default=1)
    parser.add_argument("--expand_max_faces_per_carrier", type=int, default=0)
    parser.add_argument("--frame_aware_ranking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min_frame_support_fraction", type=float, default=0.0)
    parser.add_argument("--min_residual_mass_fraction", type=float, default=0.0)
    parser.add_argument("--max_carriers", type=int, default=0)
    args = parser.parse_args()
    if int(args.max_regions_per_carrier) <= 0:
        parser.error("--max_regions_per_carrier must be > 0")
    if int(args.min_pixels) <= 0:
        parser.error("--min_pixels must be > 0")
    if int(args.bbox_pad) < 0:
        parser.error("--bbox_pad must be >= 0")
    if not math.isfinite(float(args.min_alpha)) or float(args.min_alpha) < 0.0:
        parser.error("--min_alpha must be finite and >= 0")
    if not 0.0 <= float(args.high_error_quantile) <= 1.0:
        parser.error("--high_error_quantile must be in [0, 1]")
    if int(args.max_views) < 0:
        parser.error("--max_views must be >= 0")
    if int(args.expand_min_face_pixels) <= 0:
        parser.error("--expand_min_face_pixels must be > 0")
    if int(args.expand_min_face_views) <= 0:
        parser.error("--expand_min_face_views must be > 0")
    if int(args.expand_max_faces_per_carrier) < 0:
        parser.error("--expand_max_faces_per_carrier must be >= 0")
    for name in ("min_frame_support_fraction", "min_residual_mass_fraction"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and >= 0")
    if int(args.max_carriers) < 0:
        parser.error("--max_carriers must be >= 0")
    return args


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plan_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("candidates", "accepted", "accepted_preview"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def row_faces(row: dict[str, Any]) -> set[int]:
    faces: set[int] = set()
    raw = row.get("carrier_faces")
    if isinstance(raw, list):
        for value in raw:
            try:
                faces.add(int(value))
            except Exception:
                continue
    try:
        faces.add(int(row.get("face_id")))
    except Exception:
        pass
    return faces


def rows_by_carrier(rows: list[dict[str, Any]]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for idx, row in enumerate(rows):
        carrier_id = str(row.get("carrier_id", f"face_{row.get('face_id', idx)}"))
        out[carrier_id].update(row_faces(row))
    return {key: value for key, value in out.items() if value}


def evidence_paths(root: Path, max_views: int) -> list[Path]:
    candidates = sorted((root / "views").glob("*.npz"))
    if not candidates:
        candidates = sorted((root / "per_view_npz").glob("*.npz"))
    if int(max_views) > 0:
        candidates = candidates[: int(max_views)]
    return candidates


def clip_bbox(xs: np.ndarray, ys: np.ndarray, width: int, height: int, pad: int) -> list[int]:
    x0 = max(0, int(xs.min()) - int(pad))
    x1 = min(width, int(xs.max()) + int(pad) + 1)
    y0 = max(0, int(ys.min()) - int(pad))
    y1 = min(height, int(ys.max()) + int(pad) + 1)
    return [x0, y0, x1, y1]


def region_for_carrier(
    *,
    view_path: Path,
    carrier_id: str,
    carrier_faces: set[int],
    face_id: np.ndarray,
    residual_l1: np.ndarray,
    residual_rgb: np.ndarray | None,
    alpha: np.ndarray,
    min_pixels: int,
    bbox_pad: int,
    min_alpha: float,
    high_error_quantile: float,
) -> dict[str, Any] | None:
    if face_id.ndim != 2:
        return None
    height, width = face_id.shape
    mask = np.isin(face_id, np.asarray(sorted(carrier_faces), dtype=np.int64))
    if alpha.shape == face_id.shape:
        mask &= alpha >= float(min_alpha)
    if float(high_error_quantile) > 0.0:
        valid_residual = residual_l1[np.isfinite(residual_l1)]
        if valid_residual.size:
            threshold = float(np.quantile(valid_residual, float(high_error_quantile)))
            mask &= residual_l1 >= threshold
    pixels = int(np.count_nonzero(mask))
    if pixels < int(min_pixels):
        return None
    ys, xs = np.nonzero(mask)
    frame_pixels = int(width) * int(height)
    bbox = clip_bbox(xs, ys, width, height, int(bbox_pad))
    bbox_pixels = int(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))
    present_faces: list[dict[str, int]] = []
    for fid in sorted(set(int(v) for v in np.unique(face_id[mask]).tolist())):
        present_faces.append({"face_id": int(fid), "pixels": int(np.count_nonzero(mask & (face_id == int(fid))))})
    mean_l1 = float(np.mean(residual_l1[mask])) if pixels else 0.0
    residual_mass = float(mean_l1 * pixels)
    if residual_rgb is not None and residual_rgb.ndim == 3 and residual_rgb.shape[1:] == face_id.shape:
        rgb = residual_rgb[:, mask].T
        mean_rgb = [float(v) for v in np.mean(rgb, axis=0).tolist()] if rgb.size else [0.0, 0.0, 0.0]
    else:
        mean_rgb = [0.0, 0.0, 0.0]
    score = float(mean_l1 * math.sqrt(max(pixels, 1)))
    return {
        "view": view_path.stem,
        "carrier_id": str(carrier_id),
        "bbox_xyxy": bbox,
        "pixels": int(pixels),
        "frame_pixels": int(frame_pixels),
        "bbox_pixels": int(bbox_pixels),
        "visible_frame_fraction": float(pixels) / max(float(frame_pixels), 1.0),
        "bbox_frame_fraction": float(bbox_pixels) / max(float(frame_pixels), 1.0),
        "score": score,
        "residual_mass": residual_mass,
        "residual_mass_fraction": residual_mass / max(float(frame_pixels), 1.0),
        "face_coverage": 1.0,
        "face_ids": [int(row["face_id"]) for row in present_faces],
        "faces": present_faces,
        "mean_l1": mean_l1,
        "mean_residual_rgb": mean_rgb,
        "source_npz": str(view_path),
    }


def expanded_face_pixels_for_region(
    *,
    region: dict[str, Any],
    face_id: np.ndarray,
    residual_l1: np.ndarray,
    alpha: np.ndarray,
    min_alpha: float,
    high_error_quantile: float,
) -> dict[int, int]:
    bbox = region.get("bbox_xyxy", [])
    if not isinstance(bbox, list) or len(bbox) != 4 or face_id.ndim != 2:
        return {}
    x0, y0, x1, y1 = [int(v) for v in bbox]
    height, width = face_id.shape
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return {}
    crop_faces = face_id[y0:y1, x0:x1]
    mask = crop_faces >= 0
    if alpha.shape == face_id.shape:
        mask &= alpha[y0:y1, x0:x1] >= float(min_alpha)
    if residual_l1.shape == face_id.shape:
        crop_residual = residual_l1[y0:y1, x0:x1]
        mask &= np.isfinite(crop_residual)
        if float(high_error_quantile) > 0.0:
            valid = crop_residual[mask]
            if valid.size:
                threshold = float(np.quantile(valid, float(high_error_quantile)))
                mask &= crop_residual >= threshold
    if not np.any(mask):
        return {}
    values, counts = np.unique(crop_faces[mask].astype(np.int64), return_counts=True)
    return {int(fid): int(count) for fid, count in zip(values.tolist(), counts.tolist()) if int(fid) >= 0}


def build(args: argparse.Namespace) -> dict[str, Any]:
    plan_payload = read_json(args.candidate_plan)
    carriers = rows_by_carrier(plan_rows(plan_payload))
    paths = evidence_paths(args.evidence_dir, int(args.max_views))
    carrier_regions: dict[str, list[dict[str, Any]]] = {cid: [] for cid in carriers}
    expansion_pixels: dict[str, dict[int, int]] = {cid: defaultdict(int) for cid in carriers}
    expansion_views: dict[str, dict[int, set[str]]] = {cid: defaultdict(set) for cid in carriers}
    skipped_views = 0

    for view_path in paths:
        with np.load(view_path) as z:
            if "face_id" not in z.files or "residual_l1" not in z.files:
                skipped_views += 1
                continue
            face_id = z["face_id"].astype(np.int64)
            residual_l1 = z["residual_l1"].astype(np.float32)
            alpha = z["alpha"].astype(np.float32) if "alpha" in z.files else np.ones_like(face_id, dtype=np.float32)
            if alpha.ndim == 3:
                alpha = np.squeeze(alpha, axis=0)
            residual_rgb = z["residual_rgb"].astype(np.float32) if "residual_rgb" in z.files else None
        for carrier_id, faces in carriers.items():
            region = region_for_carrier(
                view_path=view_path,
                carrier_id=carrier_id,
                carrier_faces=faces,
                face_id=face_id,
                residual_l1=residual_l1,
                residual_rgb=residual_rgb,
                alpha=alpha,
                min_pixels=int(args.min_pixels),
                bbox_pad=int(args.bbox_pad),
                min_alpha=float(args.min_alpha),
                high_error_quantile=float(args.high_error_quantile),
            )
            if region is not None:
                if bool(args.expand_faces_in_region_bbox):
                    expanded_pixels = expanded_face_pixels_for_region(
                        region=region,
                        face_id=face_id,
                        residual_l1=residual_l1,
                        alpha=alpha,
                        min_alpha=float(args.min_alpha),
                        high_error_quantile=float(args.high_error_quantile),
                    )
                    seed_faces = [int(fid) for fid in region.get("face_ids", [])]
                    region["seed_face_ids"] = seed_faces
                    region["expanded_face_pixel_candidates"] = [
                        {"face_id": int(fid), "pixels": int(count)}
                        for fid, count in sorted(expanded_pixels.items(), key=lambda item: (-int(item[1]), int(item[0])))
                    ]
                    for fid, count in expanded_pixels.items():
                        expansion_pixels[carrier_id][int(fid)] += int(count)
                        expansion_views[carrier_id][int(fid)].add(str(view_path.stem))
                carrier_regions[carrier_id].append(region)

    out_carriers: list[dict[str, Any]] = []
    rejected_frame_aware: list[dict[str, Any]] = []
    for carrier_id, faces in sorted(carriers.items()):
        regions = sorted(
            carrier_regions.get(carrier_id, []),
            key=lambda row: (float(row["score"]), int(row["pixels"])),
            reverse=True,
        )[: int(args.max_regions_per_carrier)]
        face_counter: dict[int, int] = defaultdict(int)
        for region in regions:
            for face in region.get("faces", []):
                face_counter[int(face["face_id"])] += int(face["pixels"])
        if not face_counter:
            for fid in sorted(faces):
                face_counter[int(fid)] = 0
        selected_faces = sorted(int(fid) for fid in faces)
        expansion_rows: list[dict[str, Any]] = []
        if bool(args.expand_faces_in_region_bbox):
            for fid, pixels_seen in expansion_pixels.get(carrier_id, {}).items():
                view_count = len(expansion_views.get(carrier_id, {}).get(int(fid), set()))
                if int(pixels_seen) < int(args.expand_min_face_pixels):
                    continue
                if int(view_count) < int(args.expand_min_face_views):
                    continue
                expansion_rows.append(
                    {
                        "face_id": int(fid),
                        "pixels": int(pixels_seen),
                        "views": sorted(expansion_views.get(carrier_id, {}).get(int(fid), set())),
                        "view_count": int(view_count),
                        "seed_face": bool(int(fid) in faces),
                    }
                )
            for fid in faces:
                if not any(int(row["face_id"]) == int(fid) for row in expansion_rows):
                    expansion_rows.append(
                        {
                            "face_id": int(fid),
                            "pixels": int(expansion_pixels.get(carrier_id, {}).get(int(fid), 0)),
                            "views": sorted(expansion_views.get(carrier_id, {}).get(int(fid), set())),
                            "view_count": int(len(expansion_views.get(carrier_id, {}).get(int(fid), set()))),
                            "seed_face": True,
                        }
                    )
            expansion_rows.sort(key=lambda row: (bool(row["seed_face"]), int(row["pixels"]), int(row["view_count"])), reverse=True)
            max_faces = int(args.expand_max_faces_per_carrier)
            if max_faces > 0:
                seed_rows = [row for row in expansion_rows if bool(row["seed_face"])]
                extra_rows = [row for row in expansion_rows if not bool(row["seed_face"])]
                keep_extra = max(max_faces - len(seed_rows), 0)
                expansion_rows = seed_rows + extra_rows[:keep_extra]
            selected_faces = sorted({int(row["face_id"]) for row in expansion_rows} | {int(fid) for fid in faces})
            selected_set = set(selected_faces)
            for region in regions:
                candidates = region.get("expanded_face_pixel_candidates", [])
                if isinstance(candidates, list):
                    expanded_in_region = [
                        {"face_id": int(row["face_id"]), "pixels": int(row["pixels"])}
                        for row in candidates
                        if isinstance(row, dict) and int(row.get("face_id", -1)) in selected_set
                    ]
                    if expanded_in_region:
                        region["expanded_face_ids"] = [int(row["face_id"]) for row in expanded_in_region]
                        region["face_ids"] = [int(row["face_id"]) for row in expanded_in_region]
                        region["faces"] = expanded_in_region
        pixels = int(sum(int(region["pixels"]) for region in regions))
        score_sum = float(sum(float(region["score"]) for region in regions))
        views = sorted({str(region["view"]) for region in regions})
        frame_pixels_by_view: dict[str, int] = {}
        for region in regions:
            view = str(region.get("view", ""))
            frame_pixels = int(region.get("frame_pixels", 0))
            if view and frame_pixels > 0:
                frame_pixels_by_view.setdefault(view, frame_pixels)
        frame_denominator = int(sum(frame_pixels_by_view.values()))
        residual_mass = float(sum(float(region.get("residual_mass", 0.0)) for region in regions))
        bbox_pixels = int(sum(int(region.get("bbox_pixels", 0)) for region in regions))
        frame_support_fraction = float(pixels) / max(float(frame_denominator), 1.0)
        residual_mass_fraction = residual_mass / max(float(frame_denominator), 1.0)
        view_fraction = float(len(views)) / max(float(len(paths)), 1.0)
        frame_aware_score = residual_mass_fraction * (1.0 + view_fraction)
        frame_support = {
            "frame_denominator_pixels": int(frame_denominator),
            "visible_pixels": int(pixels),
            "bbox_pixels": int(bbox_pixels),
            "frame_support_fraction": frame_support_fraction,
            "bbox_frame_fraction": float(bbox_pixels) / max(float(frame_denominator), 1.0),
            "residual_mass": residual_mass,
            "residual_mass_fraction": residual_mass_fraction,
            "view_fraction": view_fraction,
            "frame_aware_score": frame_aware_score,
        }
        carrier_row = {
            "carrier_id": str(carrier_id),
            "face_ids": [int(fid) for fid in selected_faces],
            "seed_face_ids": [int(fid) for fid in sorted(faces)],
            "expanded_face_ids": [int(fid) for fid in selected_faces if int(fid) not in set(faces)],
            "expansion": {
                "enabled": bool(args.expand_faces_in_region_bbox),
                "min_face_pixels": int(args.expand_min_face_pixels),
                "min_face_views": int(args.expand_min_face_views),
                "max_faces_per_carrier": int(args.expand_max_faces_per_carrier),
                "seed_face_count": int(len(faces)),
                "selected_face_count": int(len(selected_faces)),
                "expanded_face_count": int(len([fid for fid in selected_faces if int(fid) not in set(faces)])),
                "rows": expansion_rows[:50],
            },
            "faces": [
                {"face_id": int(fid), "pixels": int(count)}
                for fid, count in sorted(
                    ((fid, face_counter.get(fid, expansion_pixels.get(carrier_id, {}).get(fid, 0))) for fid in selected_faces),
                    key=lambda item: (-int(item[1]), int(item[0])),
                )
            ],
            "regions": regions,
            "views": views,
            "view_count": int(len(views)),
            "pixels": pixels,
            "score_sum": score_sum,
            "score": float(score_sum / max(len(regions), 1)),
            "region_count": int(len(regions)),
            "frame_support": frame_support,
        }
        rejection_reasons: list[str] = []
        if bool(args.frame_aware_ranking):
            if frame_support_fraction < float(args.min_frame_support_fraction):
                rejection_reasons.append(
                    f"frame_support_fraction_below_{float(args.min_frame_support_fraction):g}"
                )
            if residual_mass_fraction < float(args.min_residual_mass_fraction):
                rejection_reasons.append(
                    f"residual_mass_fraction_below_{float(args.min_residual_mass_fraction):g}"
                )
        if rejection_reasons:
            rejected_frame_aware.append(
                {
                    "carrier_id": str(carrier_id),
                    "reasons": rejection_reasons,
                    "frame_support": frame_support,
                    "region_count": int(len(regions)),
                    "view_count": int(len(views)),
                }
            )
            continue
        out_carriers.append(carrier_row)
    if bool(args.frame_aware_ranking):
        out_carriers.sort(
            key=lambda row: (
                float((row.get("frame_support") or {}).get("frame_aware_score", 0.0)),
                float((row.get("frame_support") or {}).get("residual_mass_fraction", 0.0)),
                float((row.get("frame_support") or {}).get("frame_support_fraction", 0.0)),
                int(row["view_count"]),
                int(row["pixels"]),
                float(row["score"]),
            ),
            reverse=True,
        )
    else:
        out_carriers.sort(key=lambda row: (int(row["region_count"]), float(row["score"]), int(row["pixels"])), reverse=True)
    if int(args.max_carriers) > 0:
        rejected_frame_aware.extend(
            {
                "carrier_id": str(row.get("carrier_id", "")),
                "reasons": [f"rank_below_top_{int(args.max_carriers)}"],
                "frame_support": row.get("frame_support", {}),
                "region_count": int(row.get("region_count", 0)),
                "view_count": int(row.get("view_count", 0)),
            }
            for row in out_carriers[int(args.max_carriers) :]
        )
        out_carriers = out_carriers[: int(args.max_carriers)]
    for idx, carrier in enumerate(out_carriers):
        carrier["rank"] = int(idx)

    return {
        "scene": str(args.scene),
        "protocol": "candidate_plan_owned_train_evidence_render_regions",
        "test_usage": "none",
        "candidate_plan": str(args.candidate_plan),
        "evidence_dir": str(args.evidence_dir),
        "settings": {
            "max_regions_per_carrier": int(args.max_regions_per_carrier),
            "min_pixels": int(args.min_pixels),
            "bbox_pad": int(args.bbox_pad),
            "min_alpha": float(args.min_alpha),
            "high_error_quantile": float(args.high_error_quantile),
            "max_views": int(args.max_views),
            "expand_faces_in_region_bbox": bool(args.expand_faces_in_region_bbox),
            "expand_min_face_pixels": int(args.expand_min_face_pixels),
            "expand_min_face_views": int(args.expand_min_face_views),
            "expand_max_faces_per_carrier": int(args.expand_max_faces_per_carrier),
            "frame_aware_ranking": bool(args.frame_aware_ranking),
            "min_frame_support_fraction": float(args.min_frame_support_fraction),
            "min_residual_mass_fraction": float(args.min_residual_mass_fraction),
            "max_carriers": int(args.max_carriers),
        },
        "diagnostics": {
            "input_plan_carriers": int(len(carriers)),
            "input_plan_faces": int(len(set().union(*carriers.values())) if carriers else 0),
            "output_carrier_faces": int(len({int(fid) for carrier in out_carriers for fid in carrier.get("face_ids", [])})),
            "expanded_face_count": int(
                len({int(fid) for carrier in out_carriers for fid in carrier.get("expanded_face_ids", [])})
            ),
            "evidence_views": int(len(paths)),
            "skipped_views": int(skipped_views),
            "carrier_with_regions": int(sum(1 for row in out_carriers if int(row["region_count"]) > 0)),
            "region_count": int(sum(int(row["region_count"]) for row in out_carriers)),
            "rejected_frame_aware_carriers": int(len(rejected_frame_aware)),
            "rank_policy": "full_frame_visible_residual_mass" if bool(args.frame_aware_ranking) else "legacy_region_score",
        },
        "rejected_frame_aware_carriers": rejected_frame_aware,
        "carriers": out_carriers,
    }


def write_md(path: Path, payload: dict[str, Any]) -> None:
    diag = payload["diagnostics"]
    lines = [
        "# Candidate-Owned Render Regions",
        "",
        f"- scene: `{payload['scene']}`",
        f"- protocol: `{payload['protocol']}`",
        f"- test usage: `{payload['test_usage']}`",
        f"- candidate plan: `{payload['candidate_plan']}`",
        f"- evidence dir: `{payload['evidence_dir']}`",
        f"- plan carriers / faces: `{diag['input_plan_carriers']}` / `{diag['input_plan_faces']}`",
        f"- output carrier faces: `{diag.get('output_carrier_faces', diag['input_plan_faces'])}`",
        f"- expanded faces: `{diag.get('expanded_face_count', 0)}`",
        f"- evidence views: `{diag['evidence_views']}`",
        f"- carriers with regions: `{diag['carrier_with_regions']}`",
        f"- total regions: `{diag['region_count']}`",
        f"- rank policy: `{diag.get('rank_policy', 'legacy_region_score')}`",
        f"- rejected frame-aware carriers: `{diag.get('rejected_frame_aware_carriers', 0)}`",
        "",
        "| rank | carrier | faces | expanded | regions | views | pixels | frame support | residual mass frac | score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for carrier in payload.get("carriers", []):
        frame_support = carrier.get("frame_support") or {}
        lines.append(
            f"| {int(carrier.get('rank', 0))} | `{carrier.get('carrier_id', '')}` | "
            f"{len(carrier.get('face_ids', []))} | {len(carrier.get('expanded_face_ids', []))} | "
            f"{int(carrier.get('region_count', 0))} | "
            f"{int(carrier.get('view_count', 0))} | {int(carrier.get('pixels', 0))} | "
            f"{float(frame_support.get('frame_support_fraction', 0.0)):.9f} | "
            f"{float(frame_support.get('residual_mass_fraction', 0.0)):.9f} | "
            f"{float(carrier.get('score', 0.0)):.9f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload = build(args)
    write_json(args.output_json, payload)
    write_md(args.output_md, payload)
    diag = payload["diagnostics"]
    print(
        json.dumps(
            {
                "scene": payload["scene"],
                "carrier_count": len(payload.get("carriers", [])),
                "carrier_with_regions": diag["carrier_with_regions"],
                "region_count": diag["region_count"],
                "output_json": str(args.output_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
