"""Mine vehicle-like image/COLMAP regions for parking-scene MeshPrior work."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import read_images_binary, read_points3D_binary


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def _resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(im.resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _component_rows(mask: np.ndarray, *, min_area: int, max_components: int) -> list[dict[str, Any]]:
    labels, count = ndimage.label(mask)
    rows = []
    for label in range(1, count + 1):
        comp = labels == label
        area = int(comp.sum())
        if area < min_area:
            continue
        x0, y0, x1, y1 = _bbox(comp)
        rows.append({"label": label, "area": area, "bbox_xyxy": [x0, y0, x1, y1], "mask": comp})
    rows.sort(key=lambda r: -int(r["area"]))
    return rows[:max_components]


def _points_in_bbox(image, points3d: dict, bbox: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    xys = np.asarray(image.xys, dtype=np.float64)
    pids = np.asarray(image.point3D_ids, dtype=np.int64)
    inside = (xys[:, 0] >= x0) & (xys[:, 0] < x1) & (xys[:, 1] >= y0) & (xys[:, 1] < y1) & (pids >= 0)
    pts = []
    for pid in pids[inside]:
        pt = points3d.get(int(pid))
        if pt is not None:
            pts.append(np.asarray(pt.xyz, dtype=np.float64))
    return np.stack(pts, axis=0) if pts else np.zeros((0, 3), dtype=np.float64)


def _score_region(area_frac: float, ground_overlap: float, points3d: np.ndarray) -> tuple[float, list[str]]:
    notes: list[str] = []
    score = 0.0
    if 0.001 <= area_frac <= 0.35:
        score += 0.35
    else:
        notes.append(f"area_fraction_out_of_range={area_frac:.6f}")
    if ground_overlap <= 0.35:
        score += 0.35
    else:
        notes.append(f"ground_overlap_high={ground_overlap:.6f}")
    if len(points3d) >= 8:
        score += 0.30
    else:
        notes.append(f"sparse_points_low={len(points3d)}")
    return float(score), notes


def _region_payload(
    *,
    region_id: str,
    image_name: str,
    image_shape: tuple[int, int],
    bbox_xyxy: list[int],
    area: int,
    ground_overlap: float,
    points3d: np.ndarray,
    score: float,
    notes: list[str],
    threshold: float,
) -> dict[str, Any]:
    h, w = image_shape
    if len(points3d):
        bbox_min = points3d.min(axis=0)
        bbox_max = points3d.max(axis=0)
        centroid = points3d.mean(axis=0)
    else:
        bbox_min = bbox_max = centroid = np.zeros(3, dtype=np.float64)
    return {
        "region_id": region_id,
        "image_name": image_name,
        "bbox_xyxy": [int(x) for x in bbox_xyxy],
        "image_width": int(w),
        "image_height": int(h),
        "mask_area_px": int(area),
        "mask_area_fraction": float(area / max(h * w, 1)),
        "ground_overlap_fraction": float(ground_overlap),
        "sparse_point_count": int(len(points3d)),
        "bbox3d_min": [float(x) for x in bbox_min.tolist()],
        "bbox3d_max": [float(x) for x in bbox_max.tolist()],
        "centroid3d": [float(x) for x in centroid.tolist()],
        "evidence": {
            "segmentation_available": True,
            "segmentation_score": float(score),
            "geometry_score": float(min(1.0, len(points3d) / 32.0)),
            "ground_rejection_score": float(ground_overlap),
            "observed_support_score": float(min(1.0, len(points3d) / 32.0)),
            "car_likeness_score": float(score),
            "eligible_for_posterior": bool(score >= threshold),
            "notes": notes,
        },
        "canonicalization": {
            "mode": "image_bbox_plus_colmap_sparse_points",
            "center": [float(x) for x in centroid.tolist()],
            "scale": float(np.linalg.norm(bbox_max - bbox_min) * 0.5) if len(points3d) else 1.0,
            "confidence": float(score),
            "notes": ["2D ROI first; 3D orientation not reliable until multi-view clustering"],
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    scene_root = Path(args.scene_root)
    seg_dir = Path(args.segmentation_dir or scene_root / "segmentation_dense")
    ground_dir = Path(args.ground_masks_dir or scene_root / "ground_masks")
    sparse_dir = scene_root / "sparse/0"
    images = read_images_binary(sparse_dir / "images.bin")
    points3d = read_points3D_binary(sparse_dir / "points3D.bin")
    by_name = {img.name: img for img in images.values()}
    regions = []
    image_names = sorted(by_name)
    if args.max_images > 0:
        image_names = image_names[: args.max_images]
    for image_name in image_names:
        stem = Path(image_name).stem
        seg_path = seg_dir / f"{stem}.png"
        if not seg_path.is_file():
            continue
        seg = _load_mask(seg_path)
        ground_path = ground_dir / f"{stem}.png"
        ground = _resize_bool(_load_mask(ground_path), seg.shape) if ground_path.is_file() else np.zeros_like(seg, dtype=bool)
        components = _component_rows(seg, min_area=args.min_area_px, max_components=args.max_components_per_image)
        for local_idx, comp in enumerate(components):
            bbox = comp["bbox_xyxy"]
            comp_mask = comp["mask"]
            ground_overlap = float(np.logical_and(comp_mask, ground).sum() / max(int(comp_mask.sum()), 1))
            pts3d = _points_in_bbox(by_name[image_name], points3d, bbox)
            area_frac = float(comp["area"] / max(seg.shape[0] * seg.shape[1], 1))
            score, notes = _score_region(area_frac, ground_overlap, pts3d)
            regions.append(
                _region_payload(
                    region_id=f"{stem}_roi_{local_idx:03d}",
                    image_name=image_name,
                    image_shape=seg.shape,
                    bbox_xyxy=bbox,
                    area=int(comp["area"]),
                    ground_overlap=ground_overlap,
                    points3d=pts3d,
                    score=score,
                    notes=notes,
                    threshold=args.eligibility_threshold,
                )
            )
    regions.sort(key=lambda r: (not r["evidence"]["eligible_for_posterior"], -r["sparse_point_count"], -r["mask_area_px"]))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene_root": str(scene_root),
        "mode": "image_colmap_roi",
        "image_count_considered": len(image_names),
        "region_count": len(regions),
        "eligible_count": sum(1 for r in regions if r["evidence"]["eligible_for_posterior"]),
        "regions": regions,
        "notes": [
            "2D segmentation/ground masks are used to mine ROI candidates.",
            "COLMAP sparse points provide coarse 3D support, not editable scene mesh geometry.",
        ],
    }
    (out / "image_regions.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "regions.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (out / "image_regions_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region_id", "image_name", "x0", "y0", "x1", "y1", "mask_area_px", "ground_overlap", "sparse_point_count", "car_likeness_score", "eligible"])
        for r in regions:
            writer.writerow([r["region_id"], r["image_name"], *r["bbox_xyxy"], r["mask_area_px"], f"{r['ground_overlap_fraction']:.6f}", r["sparse_point_count"], f"{r['evidence']['car_likeness_score']:.6f}", int(r["evidence"]["eligible_for_posterior"])])
    with (out / "image_region_mining_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Parking Image Region Mining Report\n\n")
        f.write(f"- scene_root: `{scene_root}`\n")
        f.write(f"- images considered: `{len(image_names)}`\n")
        f.write(f"- regions: `{len(regions)}`\n")
        f.write(f"- eligible: `{payload['eligible_count']}`\n")
        f.write(f"- segmentation_dir: `{seg_dir}`\n")
        f.write(f"- ground_masks_dir: `{ground_dir}`\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine parking image/COLMAP ROI candidates for MeshPrior.")
    parser.add_argument("--scene_root", default="outputs/carnet/meshprior/parking_phone_tiny/dataset_view")
    parser.add_argument("--segmentation_dir", default="")
    parser.add_argument("--ground_masks_dir", default="")
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/parking_phone_tiny/image_region_mining")
    parser.add_argument("--min_area_px", type=int, default=2000)
    parser.add_argument("--max_components_per_image", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--eligibility_threshold", type=float, default=0.65)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({k: result[k] for k in ("image_count_considered", "region_count", "eligible_count")}, indent=2))


if __name__ == "__main__":
    main()
