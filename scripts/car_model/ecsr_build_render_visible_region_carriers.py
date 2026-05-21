#!/usr/bin/env python3
"""Build render-visible region carriers from ECSR surface evidence maps.

This is a proposal-side bridge for Phase-S: image-space residual blobs are
merged into face carriers and exported as a region-ranked evidence directory
that the existing face-local fitter can consume without changing the renderer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from scipy import ndimage

        structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        labels, count = ndimage.label(mask, structure=structure)
        return labels.astype(np.int32, copy=False), int(count)
    except Exception:
        labels = np.zeros(mask.shape, dtype=np.int32)
        next_label = 0
        h, w = mask.shape
        for y in range(h):
            for x in range(w):
                if not mask[y, x] or labels[y, x] != 0:
                    continue
                next_label += 1
                stack = [(y, x)]
                labels[y, x] = next_label
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = next_label
                            stack.append((ny, nx))
        return labels, next_label


def _round_float(value: float, digits: int = 8) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(round(float(value), int(digits)))


def _encode_bool_mask_rle(mask: np.ndarray) -> list[int]:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    counts: list[int] = []
    current = 0
    run = 0
    for value_raw in flat:
        value = int(value_raw)
        if value == current:
            run += 1
        else:
            counts.append(run)
            current = value
            run = 1
    counts.append(run)
    return counts


def _view_paths(evidence_dir: Path, *, max_views: int, view_stride: int, view_offset: int) -> list[Path]:
    paths = sorted((evidence_dir / "views").glob("*.npz"))
    if not paths:
        paths = sorted((evidence_dir / "per_view_npz").glob("*.npz"))
    if view_stride > 1:
        paths = [path for idx, path in enumerate(paths) if idx % int(view_stride) == int(view_offset)]
    if max_views > 0:
        paths = paths[: int(max_views)]
    return paths


def _component_regions(
    path: Path,
    *,
    residual_quantile: float,
    min_residual_l1: float,
    min_alpha: float,
    min_pixels: int,
    top_regions_per_view: int,
    max_faces_per_region: int,
    min_face_pixels: int,
    store_region_masks: bool,
) -> list[dict[str, Any]]:
    data = np.load(path)
    face_id = data["face_id"].astype(np.int64, copy=False)
    residual_l1 = data["residual_l1"].astype(np.float32, copy=False)
    alpha = data["alpha"].astype(np.float32, copy=False) if "alpha" in data else np.ones_like(residual_l1)
    residual_rgb = data["residual_rgb"].astype(np.float32, copy=False) if "residual_rgb" in data else None
    valid = (face_id >= 0) & np.isfinite(residual_l1) & (alpha >= float(min_alpha))
    valid_values = residual_l1[valid]
    if valid_values.size == 0:
        return []
    quantile = float(np.clip(residual_quantile, 0.0, 1.0))
    threshold = max(float(min_residual_l1), float(np.quantile(valid_values, quantile)))
    mask = valid & (residual_l1 >= threshold)
    labels, count = _label_components(mask)
    try:
        from scipy import ndimage

        objects = ndimage.find_objects(labels, max_label=count)
    except Exception:
        objects = [None] * count
    rows: list[dict[str, Any]] = []
    view_name = path.stem
    for label in range(1, count + 1):
        slc = objects[label - 1] if label - 1 < len(objects) else None
        if slc is None:
            continue
        local_labels = labels[slc]
        local_mask = local_labels == label
        pixel_count = int(local_mask.sum())
        if pixel_count < int(min_pixels):
            continue
        local_faces = face_id[slc]
        comp_faces = local_faces[local_mask]
        comp_faces = comp_faces[comp_faces >= 0]
        if comp_faces.size == 0:
            continue
        unique, counts = np.unique(comp_faces, return_counts=True)
        order = np.argsort(counts)[::-1]
        face_rows: list[dict[str, Any]] = []
        for idx in order[: int(max_faces_per_region)]:
            count_i = int(counts[idx])
            if count_i < int(min_face_pixels):
                continue
            face_rows.append({"face_id": int(unique[idx]), "pixels": count_i})
        if not face_rows:
            continue
        local_residual = residual_l1[slc]
        values = local_residual[local_mask]
        mean_l1 = float(np.mean(values))
        p90_l1 = float(np.quantile(values, 0.90))
        if residual_rgb is not None:
            y_slice, x_slice = slc
            rgb = residual_rgb[:, y_slice, x_slice][:, local_mask]
            mean_rgb = [float(v) for v in np.mean(rgb, axis=1)]
        else:
            mean_rgb = [0.0, 0.0, 0.0]
        face_pixels = sum(int(row["pixels"]) for row in face_rows)
        face_coverage = float(face_pixels) / max(float(pixel_count), 1.0)
        score = mean_l1 * math.sqrt(float(pixel_count)) * math.log1p(len(face_rows)) * max(face_coverage, 0.05)
        row = {
            "view": view_name,
            "source_npz": str(path),
            "bbox_xyxy": [int(slc[1].start), int(slc[0].start), int(slc[1].stop), int(slc[0].stop)],
            "pixels": pixel_count,
            "threshold": _round_float(threshold),
            "mean_l1": _round_float(mean_l1),
            "p90_l1": _round_float(p90_l1),
            "mean_residual_rgb": [_round_float(v) for v in mean_rgb],
            "face_coverage": _round_float(face_coverage),
            "faces": face_rows,
            "face_ids": [int(row["face_id"]) for row in face_rows],
            "score": _round_float(score),
        }
        if bool(store_region_masks):
            mask_bool = np.asarray(local_mask, dtype=bool)
            row["mask_shape_hw"] = [int(mask_bool.shape[0]), int(mask_bool.shape[1])]
            row["mask_rle_counts"] = _encode_bool_mask_rle(mask_bool)
        rows.append(row)
    rows.sort(key=lambda row: (float(row["score"]), int(row["pixels"])), reverse=True)
    return rows[: int(top_regions_per_view)]


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return float(inter) / float(len(a | b))


def _merge_region_into_carrier(carrier: dict[str, Any], region: dict[str, Any]) -> None:
    pixels = int(region["pixels"])
    old_pixels = int(carrier["pixels"])
    total_pixels = old_pixels + pixels
    carrier["pixels"] = total_pixels
    carrier["score_sum"] = float(carrier.get("score_sum", 0.0)) + float(region["score"])
    carrier["regions"].append(region)
    carrier["views"] = sorted(set(carrier["views"]) | {str(region["view"])})
    carrier["view_count"] = len(carrier["views"])
    face_counter: Counter[int] = carrier["_face_counter"]
    for face in region["faces"]:
        face_counter[int(face["face_id"])] += int(face["pixels"])
    old_mean = float(carrier["mean_l1"])
    carrier["mean_l1"] = (old_mean * old_pixels + float(region["mean_l1"]) * pixels) / max(total_pixels, 1)
    old_rgb = np.asarray(carrier["mean_residual_rgb"], dtype=np.float64)
    new_rgb = np.asarray(region["mean_residual_rgb"], dtype=np.float64)
    carrier["mean_residual_rgb"] = ((old_rgb * old_pixels + new_rgb * pixels) / max(total_pixels, 1)).tolist()
    carrier["face_ids"] = [int(fid) for fid, _ in face_counter.most_common()]
    carrier["faces"] = [{"face_id": int(fid), "pixels": int(count)} for fid, count in face_counter.most_common()]
    carrier["score"] = float(carrier["score_sum"]) * math.log1p(float(carrier["view_count"]))


def merge_regions(
    regions: list[dict[str, Any]],
    *,
    merge_face_jaccard: float,
    min_merge_shared_faces: int,
    max_carriers: int,
) -> list[dict[str, Any]]:
    carriers: list[dict[str, Any]] = []
    for region in sorted(regions, key=lambda row: (float(row["score"]), int(row["pixels"])), reverse=True):
        region_faces = set(int(fid) for fid in region["face_ids"])
        best_idx = -1
        best_key = (0.0, 0)
        for idx, carrier in enumerate(carriers):
            carrier_faces = set(int(fid) for fid in carrier["face_ids"])
            shared = len(region_faces & carrier_faces)
            jac = _jaccard(region_faces, carrier_faces)
            if jac >= float(merge_face_jaccard) or shared >= int(min_merge_shared_faces):
                key = (jac, shared)
                if key > best_key:
                    best_idx = idx
                    best_key = key
        if best_idx < 0:
            face_counter: Counter[int] = Counter()
            for face in region["faces"]:
                face_counter[int(face["face_id"])] += int(face["pixels"])
            carriers.append(
                {
                    "carrier_id": len(carriers),
                    "regions": [region],
                    "views": [str(region["view"])],
                    "view_count": 1,
                    "pixels": int(region["pixels"]),
                    "score_sum": float(region["score"]),
                    "score": float(region["score"]),
                    "mean_l1": float(region["mean_l1"]),
                    "mean_residual_rgb": [float(v) for v in region["mean_residual_rgb"]],
                    "face_ids": [int(fid) for fid, _ in face_counter.most_common()],
                    "faces": [{"face_id": int(fid), "pixels": int(count)} for fid, count in face_counter.most_common()],
                    "_face_counter": face_counter,
                }
            )
        else:
            _merge_region_into_carrier(carriers[best_idx], region)
    for idx, carrier in enumerate(carriers):
        carrier["carrier_id"] = idx
        carrier["score"] = _round_float(float(carrier["score"]))
        carrier["mean_l1"] = _round_float(float(carrier["mean_l1"]))
        carrier["mean_residual_rgb"] = [_round_float(v) for v in carrier["mean_residual_rgb"]]
        carrier.pop("_face_counter", None)
    carriers.sort(key=lambda row: (float(row["score"]), int(row["pixels"])), reverse=True)
    for idx, carrier in enumerate(carriers):
        carrier["rank"] = idx
    return carriers[: int(max_carriers)] if max_carriers > 0 else carriers


def build_face_csv_rows(carriers: list[dict[str, Any]], max_faces: int) -> list[dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    for carrier in carriers:
        carrier_score = float(carrier["score"])
        view_count = int(carrier["view_count"])
        mean_l1 = float(carrier["mean_l1"])
        mean_rgb = [float(v) for v in carrier["mean_residual_rgb"]]
        total_pixels = max(sum(int(face["pixels"]) for face in carrier["faces"]), 1)
        for face in carrier["faces"]:
            fid = int(face["face_id"])
            pixels = int(face["pixels"])
            row = stats.setdefault(
                fid,
                {
                    "face_id": fid,
                    "score": 0.0,
                    "pixel_count": 0.0,
                    "view_hits": 0,
                    "weighted_l1": 0.0,
                    "weighted_rgb": [0.0, 0.0, 0.0],
                    "carrier_count": 0,
                },
            )
            weight = float(pixels) / float(total_pixels)
            row["score"] += carrier_score * weight
            row["pixel_count"] += float(pixels)
            row["view_hits"] = max(int(row["view_hits"]), view_count)
            row["weighted_l1"] += mean_l1 * float(pixels)
            for i in range(3):
                row["weighted_rgb"][i] += mean_rgb[i] * float(pixels)
            row["carrier_count"] += 1
    rows = []
    for row in stats.values():
        pixel_count = max(float(row["pixel_count"]), 1.0)
        mean_l1 = float(row["weighted_l1"]) / pixel_count
        mean_rgb = [float(v) / pixel_count for v in row["weighted_rgb"]]
        consistency = min(1.0, 0.70 + 0.10 * min(int(row["view_hits"]), 3) + 0.02 * min(int(row["carrier_count"]), 5))
        rows.append(
            {
                "face_id": int(row["face_id"]),
                "score": _round_float(float(row["score"])),
                "pixel_count": _round_float(float(row["pixel_count"])),
                "view_hits": int(row["view_hits"]),
                "residual_consistency": _round_float(consistency),
                "mean_l1_error": _round_float(mean_l1),
                "mean_residual_r": _round_float(mean_rgb[0]),
                "mean_residual_g": _round_float(mean_rgb[1]),
                "mean_residual_b": _round_float(mean_rgb[2]),
            }
        )
    rows.sort(key=lambda r: (float(r["score"]), float(r["pixel_count"])), reverse=True)
    return rows[: int(max_faces)] if max_faces > 0 else rows


def write_evidence_dir(out_dir: Path, source_evidence_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "top_residual_supports.csv"
    fieldnames = [
        "face_id",
        "score",
        "pixel_count",
        "view_hits",
        "residual_consistency",
        "mean_l1_error",
        "mean_residual_r",
        "mean_residual_g",
        "mean_residual_b",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    views_src = source_evidence_dir / "views"
    views_dst = out_dir / "views"
    if views_src.is_dir() and not views_dst.exists():
        try:
            os.symlink(views_src.resolve(), views_dst, target_is_directory=True)
        except FileExistsError:
            pass
    summary_src = source_evidence_dir / "surface_evidence_summary.json"
    if summary_src.is_file() and not (out_dir / "surface_evidence_summary.json").exists():
        try:
            os.symlink(summary_src.resolve(), out_dir / "surface_evidence_summary.json")
        except FileExistsError:
            pass


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Render-Visible Region Carrier Proposals",
        "",
        f"- scene: `{payload['scene']}`",
        f"- source evidence: `{payload['source_evidence_dir']}`",
        f"- views scanned: `{payload['views_scanned']}`",
        f"- raw regions: `{payload['raw_region_count']}`",
        f"- merged carriers: `{payload['carrier_count']}`",
        f"- region-ranked evidence dir: `{payload.get('evidence_dir_out', '')}`",
        "",
        "| rank | views | pixels | faces | score | mean L1 | top faces |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for carrier in payload["carriers"][:20]:
        top_faces = ", ".join(str(fid) for fid in carrier["face_ids"][:8])
        lines.append(
            "| {rank} | {view_count} | {pixels} | {faces} | {score:.6f} | {mean_l1:.6f} | `{top_faces}` |".format(
                rank=int(carrier["rank"]),
                view_count=int(carrier["view_count"]),
                pixels=int(carrier["pixels"]),
                faces=len(carrier["face_ids"]),
                score=float(carrier["score"]),
                mean_l1=float(carrier["mean_l1"]),
                top_faces=top_faces,
            )
        )
    lines.extend(
        [
            "",
            "This proposal file is train-evidence only. It does not certify a final",
            "method row; it changes the Phase-S candidate prior from face-score ranking",
            "to render-visible residual-region ranking. Final promotion still requires",
            "the standard train-val render gate with `selection_uses_test=false`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--evidence_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", default="")
    parser.add_argument("--evidence_dir_out", default="")
    parser.add_argument("--max_views", type=int, default=12)
    parser.add_argument("--view_stride", type=int, default=1)
    parser.add_argument("--view_offset", type=int, default=0)
    parser.add_argument("--residual_quantile", type=float, default=0.92)
    parser.add_argument("--min_residual_l1", type=float, default=0.03)
    parser.add_argument("--min_alpha", type=float, default=0.05)
    parser.add_argument("--min_pixels", type=int, default=120)
    parser.add_argument("--min_face_pixels", type=int, default=8)
    parser.add_argument("--top_regions_per_view", type=int, default=8)
    parser.add_argument("--max_faces_per_region", type=int, default=48)
    parser.add_argument("--merge_face_jaccard", type=float, default=0.10)
    parser.add_argument("--min_merge_shared_faces", type=int, default=3)
    parser.add_argument("--max_carriers", type=int, default=64)
    parser.add_argument("--max_evidence_faces", type=int, default=2048)
    parser.add_argument(
        "--store_region_masks",
        action="store_true",
        help=(
            "Store true train residual connected-component masks in the carrier JSON. "
            "Downstream fitters can then distinguish precise core pixels from the "
            "coarser bbox context without reading held-out views."
        ),
    )
    args = parser.parse_args()

    evidence_dir = Path(args.evidence_dir)
    paths = _view_paths(
        evidence_dir,
        max_views=int(args.max_views),
        view_stride=int(args.view_stride),
        view_offset=int(args.view_offset),
    )
    regions: list[dict[str, Any]] = []
    for path in paths:
        regions.extend(
            _component_regions(
                path,
                residual_quantile=float(args.residual_quantile),
                min_residual_l1=float(args.min_residual_l1),
                min_alpha=float(args.min_alpha),
                min_pixels=int(args.min_pixels),
                top_regions_per_view=int(args.top_regions_per_view),
                max_faces_per_region=int(args.max_faces_per_region),
                min_face_pixels=int(args.min_face_pixels),
                store_region_masks=bool(args.store_region_masks),
            )
        )
    carriers = merge_regions(
        regions,
        merge_face_jaccard=float(args.merge_face_jaccard),
        min_merge_shared_faces=int(args.min_merge_shared_faces),
        max_carriers=int(args.max_carriers),
    )
    evidence_dir_out = Path(args.evidence_dir_out) if str(args.evidence_dir_out).strip() else None
    face_rows = build_face_csv_rows(carriers, int(args.max_evidence_faces))
    if evidence_dir_out is not None:
        write_evidence_dir(evidence_dir_out, evidence_dir, face_rows)
    payload = {
        "scene": str(args.scene),
        "source_evidence_dir": str(evidence_dir),
        "views_scanned": len(paths),
        "view_files": [str(path) for path in paths],
        "raw_region_count": len(regions),
        "carrier_count": len(carriers),
        "evidence_face_count": len(face_rows),
        "evidence_dir_out": str(evidence_dir_out) if evidence_dir_out is not None else "",
        "settings": {
            "residual_quantile": float(args.residual_quantile),
            "min_residual_l1": float(args.min_residual_l1),
            "min_alpha": float(args.min_alpha),
            "min_pixels": int(args.min_pixels),
            "top_regions_per_view": int(args.top_regions_per_view),
            "merge_face_jaccard": float(args.merge_face_jaccard),
            "min_merge_shared_faces": int(args.min_merge_shared_faces),
            "store_region_masks": bool(args.store_region_masks),
        },
        "carriers": carriers,
        "evidence_faces_preview": face_rows[:50],
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md = Path(args.out_md) if str(args.out_md).strip() else out_json.with_suffix(".md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(out_md, payload)
    print(json.dumps({"carriers": len(carriers), "regions": len(regions), "evidence_faces": len(face_rows), "out_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
