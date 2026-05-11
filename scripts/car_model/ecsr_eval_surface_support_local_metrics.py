#!/usr/bin/env python3
"""Evaluate train-defined surface-support local metrics.

The crop/mask protocol is intentionally train-only:

1. read top residual face supports from a Surface Evidence Cache built on train
   views;
2. project those face ids into the target split with pre-rendered surface maps;
3. evaluate any two render methods on the resulting fixed masks/crops.

This script never selects views, masks, or crops from held-out metric gains.
Held-out images are used only after the train-defined support set is fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--surface_maps_dir", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--candidate_dir", type=Path, required=True)
    parser.add_argument("--baseline_label", default="baseline")
    parser.add_argument("--candidate_label", default="candidate")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--top_faces", type=int, default=256)
    parser.add_argument("--min_mask_pixels", type=int, default=768)
    parser.add_argument("--alpha_min", type=float, default=0.05)
    parser.add_argument("--dilate", type=int, default=8)
    parser.add_argument("--crop_pad", type=int, default=24)
    parser.add_argument("--max_views", type=int, default=12)
    parser.add_argument("--save_panels", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_top_faces(path: Path, top_k: int) -> list[int]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append(
                    {
                        "face_id": int(float(row["face_id"])),
                        "score": float(row.get("score", 0.0)),
                        "pixel_count": float(row.get("pixel_count", 0.0)),
                    }
                )
            except Exception:
                continue
    rows.sort(key=lambda row: (row["score"], row["pixel_count"]), reverse=True)
    return [int(row["face_id"]) for row in rows[: int(top_k)]]


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return mask
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None].to(device=device)
    kernel = int(radius) * 2 + 1
    out = F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=int(radius))
    return out[0, 0].detach().cpu().numpy() > 0.5


def load_surface_mask(npz_path: Path, selected_faces: set[int], *, alpha_min: float, dilate: int) -> np.ndarray:
    with np.load(npz_path) as z:
        face_id = z["face_id"].astype(np.int64)
        alpha = z["alpha"].astype(np.float32)
    mask = np.isin(face_id, list(selected_faces)) & (alpha >= float(alpha_min))
    return _dilate_mask(mask, int(dilate))


def bbox_from_mask(mask: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    h, w = mask.shape
    x0 = max(int(xs.min()) - int(pad), 0)
    y0 = max(int(ys.min()) - int(pad), 0)
    x1 = min(int(xs.max()) + int(pad) + 1, w)
    y1 = min(int(ys.max()) + int(pad) + 1, h)
    return (x0, y0, x1, y1)


def read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    return tf.to_tensor(img).unsqueeze(0)[:, :3].to(device=device)


def masked_mse_psnr(render: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return math.nan, math.nan
    diff = render.astype(np.float32) / 255.0 - gt.astype(np.float32) / 255.0
    values = diff[mask]
    mse = float(np.mean(values * values))
    psnr_val = float(-10.0 * math.log10(max(mse, 1e-12)))
    return mse, psnr_val


def masked_mae(render: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    diff = np.abs(render.astype(np.float32) / 255.0 - gt.astype(np.float32) / 255.0)
    return float(np.mean(diff[mask]))


def crop_metrics(render_path: Path, gt_path: Path, bbox: tuple[int, int, int, int], device: torch.device) -> dict[str, float]:
    render = read_rgb(render_path).crop(bbox)
    gt = read_rgb(gt_path).crop(bbox)
    r = to_tensor(render, device)
    g = to_tensor(gt, device)
    with torch.no_grad():
        return {
            "crop_psnr": float(psnr(r, g).detach().cpu().item()),
            "crop_ssim": float(ssim(r, g).detach().cpu().item()),
            "crop_lpips": float(lpips(r, g, net_type="vgg").detach().cpu().item()),
        }


def make_panel(
    *,
    gt: Image.Image,
    baseline: Image.Image,
    candidate: Image.Image,
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    labels: tuple[str, str],
) -> Image.Image:
    crops = [
        ("GT", gt.crop(bbox)),
        (labels[0], baseline.crop(bbox)),
        (labels[1], candidate.crop(bbox)),
    ]
    crop_w, crop_h = crops[0][1].size
    label_h = 24
    panel = Image.new("RGB", (crop_w * 3, crop_h + label_h), "white")
    draw = ImageDraw.Draw(panel)
    for idx, (label, crop) in enumerate(crops):
        panel.paste(crop, (idx * crop_w, label_h))
        draw.text((idx * crop_w + 6, 5), label, fill=(0, 0, 0))
    x0, y0, x1, y1 = bbox
    local_mask = mask[y0:y1, x0:x1]
    edge = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))
    edge_draw = ImageDraw.Draw(edge)
    if np.any(local_mask):
        ys, xs = np.nonzero(local_mask)
        edge_draw.rectangle((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), outline=(0, 255, 0, 255), width=3)
    for idx in range(3):
        region = panel.crop((idx * crop_w, label_h, (idx + 1) * crop_w, label_h + crop_h)).convert("RGBA")
        region.alpha_composite(edge)
        panel.paste(region.convert("RGB"), (idx * crop_w, label_h))
    return panel


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    selected_faces = set(read_top_faces(args.evidence_dir / "top_residual_supports.csv", int(args.top_faces)))
    surface_paths = sorted(Path(args.surface_maps_dir).glob("*.npz"))
    rows: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = args.output_dir / "panels"
    if args.save_panels:
        panels_dir.mkdir(parents=True, exist_ok=True)

    for surface_path in tqdm(surface_paths, desc=f"{args.scene} local masks"):
        stem = surface_path.stem
        mask = load_surface_mask(surface_path, selected_faces, alpha_min=float(args.alpha_min), dilate=int(args.dilate))
        mask_pixels = int(mask.sum())
        if mask_pixels < int(args.min_mask_pixels):
            continue
        bbox = bbox_from_mask(mask, int(args.crop_pad))
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        base_render_path = args.baseline_dir / "renders" / f"{stem}.png"
        cand_render_path = args.candidate_dir / "renders" / f"{stem}.png"
        gt_path = args.baseline_dir / "gt" / f"{stem}.png"
        if not base_render_path.is_file() or not cand_render_path.is_file() or not gt_path.is_file():
            continue
        base_img = read_rgb(base_render_path)
        cand_img = read_rgb(cand_render_path)
        gt_img = read_rgb(gt_path)
        base_np = np.asarray(base_img)
        cand_np = np.asarray(cand_img)
        gt_np = np.asarray(gt_img)
        if mask.shape[:2] != base_np.shape[:2]:
            continue
        base_mse, base_mask_psnr = masked_mse_psnr(base_np, gt_np, mask)
        cand_mse, cand_mask_psnr = masked_mse_psnr(cand_np, gt_np, mask)
        base_crop = crop_metrics(base_render_path, gt_path, bbox, device)
        cand_crop = crop_metrics(cand_render_path, gt_path, bbox, device)
        row = {
            "view": stem,
            "mask_pixels": mask_pixels,
            "bbox": list(map(int, bbox)),
            "baseline_mask_mse": base_mse,
            "candidate_mask_mse": cand_mse,
            "baseline_mask_psnr": base_mask_psnr,
            "candidate_mask_psnr": cand_mask_psnr,
            "baseline_mask_mae": masked_mae(base_np, gt_np, mask),
            "candidate_mask_mae": masked_mae(cand_np, gt_np, mask),
            **{f"baseline_{key}": value for key, value in base_crop.items()},
            **{f"candidate_{key}": value for key, value in cand_crop.items()},
        }
        row.update(
            {
                "delta_mask_psnr": row["candidate_mask_psnr"] - row["baseline_mask_psnr"],
                "delta_mask_mae": row["candidate_mask_mae"] - row["baseline_mask_mae"],
                "delta_crop_psnr": row["candidate_crop_psnr"] - row["baseline_crop_psnr"],
                "delta_crop_ssim": row["candidate_crop_ssim"] - row["baseline_crop_ssim"],
                "delta_crop_lpips": row["candidate_crop_lpips"] - row["baseline_crop_lpips"],
            }
        )
        rows.append(row)
        if args.save_panels and len(rows) <= int(args.max_views):
            panel = make_panel(
                gt=gt_img,
                baseline=base_img,
                candidate=cand_img,
                bbox=bbox,
                mask=mask,
                labels=(args.baseline_label, args.candidate_label),
            )
            panel.save(panels_dir / f"{stem}_panel.png")
        if len(rows) >= int(args.max_views):
            break

    summary: dict[str, Any] = {
        "scene": args.scene,
        "protocol": "train_evidence_top_faces_projected_to_target_split_surface_maps",
        "test_usage": "metrics_only_after_masks_are_fixed",
        "evidence_dir": str(args.evidence_dir),
        "surface_maps_dir": str(args.surface_maps_dir),
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "top_faces": int(args.top_faces),
        "selected_faces": int(len(selected_faces)),
        "rows": rows,
        "view_count": len(rows),
    }
    numeric_keys = [
        "delta_mask_psnr",
        "delta_mask_mae",
        "delta_crop_psnr",
        "delta_crop_ssim",
        "delta_crop_lpips",
        "baseline_mask_psnr",
        "candidate_mask_psnr",
        "baseline_crop_lpips",
        "candidate_crop_lpips",
    ]
    summary["mean"] = {
        key: float(np.mean([row[key] for row in rows])) if rows else math.nan
        for key in numeric_keys
    }
    summary["median"] = {
        key: float(np.median([row[key] for row in rows])) if rows else math.nan
        for key in numeric_keys
    }
    summary["wins"] = {
        "mask_psnr": int(sum(1 for row in rows if row["delta_mask_psnr"] > 0.0)),
        "mask_mae": int(sum(1 for row in rows if row["delta_mask_mae"] < 0.0)),
        "crop_psnr": int(sum(1 for row in rows if row["delta_crop_psnr"] > 0.0)),
        "crop_ssim": int(sum(1 for row in rows if row["delta_crop_ssim"] > 0.0)),
        "crop_lpips": int(sum(1 for row in rows if row["delta_crop_lpips"] < 0.0)),
    }
    return summary


def write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "surface_support_local_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    mean = summary["mean"]
    wins = summary["wins"]
    rows = summary["rows"]
    lines = [
        "# Surface-Support Local Metrics",
        "",
        f"- scene: `{summary['scene']}`",
        f"- protocol: `{summary['protocol']}`",
        f"- test usage: `{summary['test_usage']}`",
        f"- baseline: `{summary['baseline_label']}`",
        f"- candidate: `{summary['candidate_label']}`",
        f"- top train residual faces: `{summary['top_faces']}`",
        f"- evaluated views: `{summary['view_count']}`",
        "",
        "Mean deltas are candidate minus baseline; LPIPS/MAE lower is better.",
        f"- mean delta mask PSNR: `{mean['delta_mask_psnr']:+.6f}`",
        f"- mean delta mask MAE: `{mean['delta_mask_mae']:+.8f}`",
        f"- mean delta crop PSNR: `{mean['delta_crop_psnr']:+.6f}`",
        f"- mean delta crop SSIM: `{mean['delta_crop_ssim']:+.8f}`",
        f"- mean delta crop LPIPS: `{mean['delta_crop_lpips']:+.8f}`",
        f"- wins mask PSNR / mask MAE / crop PSNR / crop SSIM / crop LPIPS: `{wins['mask_psnr']}/{len(rows)}` / `{wins['mask_mae']}/{len(rows)}` / `{wins['crop_psnr']}/{len(rows)}` / `{wins['crop_ssim']}/{len(rows)}` / `{wins['crop_lpips']}/{len(rows)}`",
        "",
        "| view | mask px | dMaskPSNR | dMaskMAE | dCropPSNR | dCropSSIM | dCropLPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['view']} | {row['mask_pixels']} | {row['delta_mask_psnr']:+.6f} | "
            f"{row['delta_mask_mae']:+.8f} | {row['delta_crop_psnr']:+.6f} | "
            f"{row['delta_crop_ssim']:+.8f} | {row['delta_crop_lpips']:+.8f} |"
        )
    (output_dir / "surface_support_local_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    summary = evaluate(args)
    write_outputs(args.output_dir, summary)
    print(json.dumps({"scene": args.scene, "views": summary["view_count"], "mean": summary["mean"], "wins": summary["wins"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
