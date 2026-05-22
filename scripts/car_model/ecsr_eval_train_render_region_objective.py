#!/usr/bin/env python3
"""Evaluate train-only rendered region objective for Phase-K candidates.

Inputs are fixed before held-out test evaluation:

1. render-visible carrier regions generated from train evidence;
2. Phase-J and candidate train renders;
3. train GT images already written by the renderer.

The script scores actual rendered RGB crops/regions on train views only. It is
intended as an additional Phase-K gate after candidate materialization and
train rendering, not as a held-out test selector.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as tf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--carrier_json", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--candidate_dir", type=Path, required=True)
    parser.add_argument("--baseline_label", default="baseline")
    parser.add_argument("--candidate_label", default="candidate")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--max_regions", type=int, default=64)
    parser.add_argument("--min_region_pixels", type=int, default=128)
    parser.add_argument("--min_crop_size", type=int, default=32)
    parser.add_argument("--context_pad", type=int, default=16)
    parser.add_argument("--tail_fraction", type=float, default=0.25)
    parser.add_argument("--ssim_weight", type=float, default=20.0)
    parser.add_argument("--lpips_weight", type=float, default=20.0)
    parser.add_argument(
        "--skip_lpips",
        action="store_true",
        help=(
            "Skip crop-level LPIPS inside this train-only region gate. Full-frame "
            "train/test metrics still use the standard metric script; this option "
            "keeps the local gate fast on scenes with many support crops."
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    return tf.to_tensor(img).unsqueeze(0)[:, :3].to(device=device)


def _crop_metrics(
    render_path: Path,
    gt_path: Path,
    bbox: tuple[int, int, int, int],
    device: torch.device,
    *,
    compute_lpips: bool,
) -> dict[str, float]:
    render = _read_rgb(render_path).crop(bbox)
    gt = _read_rgb(gt_path).crop(bbox)
    r = _to_tensor(render, device)
    g = _to_tensor(gt, device)
    with torch.no_grad():
        out = {
            "psnr": float(psnr(r, g).detach().cpu().item()),
            "ssim": float(ssim(r, g).detach().cpu().item()),
            "lpips": math.nan,
        }
        if compute_lpips:
            out["lpips"] = float(lpips(r, g, net_type="vgg").detach().cpu().item())
        return out


def _mse_region(render: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    diff = render.astype(np.float32) / 255.0 - gt.astype(np.float32) / 255.0
    values = diff[mask]
    return float(np.mean(values * values))


def _crop_diff_stats(base: np.ndarray, cand: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, float | int | bool]:
    x0, y0, x1, y1 = bbox
    crop_base = base[y0:y1, x0:x1]
    crop_cand = cand[y0:y1, x0:x1]
    if crop_base.size == 0 or crop_cand.size == 0:
        return {
            "render_same_bytes": bool(np.array_equal(base, cand)),
            "crop_changed": False,
            "crop_nonzero_pixels": 0,
            "crop_nonzero_fraction": 0.0,
            "crop_max_abs_diff": 0.0,
            "crop_mean_abs_diff": 0.0,
        }
    abs_diff = np.abs(crop_cand.astype(np.int16) - crop_base.astype(np.int16))
    changed_pixels = np.any(abs_diff > 0, axis=-1)
    nonzero_pixels = int(np.count_nonzero(changed_pixels))
    pixel_count = int(changed_pixels.size)
    return {
        "render_same_bytes": bool(np.array_equal(base, cand)),
        "crop_changed": bool(nonzero_pixels > 0),
        "crop_nonzero_pixels": nonzero_pixels,
        "crop_nonzero_fraction": float(nonzero_pixels) / max(float(pixel_count), 1.0),
        "crop_max_abs_diff": float(np.max(abs_diff)) / 255.0,
        "crop_mean_abs_diff": float(np.mean(abs_diff)) / 255.0,
    }


def _clip_bbox(raw: list[Any], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(round(float(v))) for v in raw[:4]]
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    return x0, y0, x1, y1


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - int(pad)),
        max(0, y0 - int(pad)),
        min(width, x1 + int(pad)),
        min(height, y1 + int(pad)),
    )


def _load_regions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    carriers = payload.get("carriers", []) if isinstance(payload, dict) else []
    rows: list[dict[str, Any]] = []
    for carrier in carriers:
        carrier_id = carrier.get("carrier_id")
        for region in carrier.get("regions", []) if isinstance(carrier, dict) else []:
            if not isinstance(region, dict) or not isinstance(region.get("bbox_xyxy"), list):
                continue
            rows.append(
                {
                    "carrier_id": carrier_id,
                    "view": str(region.get("view", "")),
                    "bbox_xyxy": region["bbox_xyxy"],
                    "pixels": int(region.get("pixels", 0)),
                    "score": float(region.get("score", carrier.get("score", 0.0))),
                    "face_coverage": float(region.get("face_coverage", 0.0)),
                }
            )
    rows.sort(key=lambda row: (float(row["score"]), int(row["pixels"])), reverse=True)
    return rows


def _finite_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else math.nan


def _tail_cvar(values: list[float], fraction: float) -> float:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return math.nan
    k = max(1, int(math.ceil(float(fraction) * len(finite))))
    return float(np.mean(finite[:k]))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    compute_lpips = not bool(args.skip_lpips)
    regions = _load_regions(args.carrier_json)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    skipped = {
        "too_few_pixels": 0,
        "missing_images": 0,
        "invalid_bbox": 0,
        "too_small_crop": 0,
        "duplicate_bbox": 0,
    }

    for region in regions:
        if len(rows) >= int(args.max_regions):
            break
        if int(region["pixels"]) < int(args.min_region_pixels):
            skipped["too_few_pixels"] += 1
            continue
        stem = str(region["view"])
        base_render_path = args.baseline_dir / "renders" / f"{stem}.png"
        cand_render_path = args.candidate_dir / "renders" / f"{stem}.png"
        gt_path = args.baseline_dir / "gt" / f"{stem}.png"
        if not (base_render_path.is_file() and cand_render_path.is_file() and gt_path.is_file()):
            skipped["missing_images"] += 1
            continue
        base_img = _read_rgb(base_render_path)
        cand_img = _read_rgb(cand_render_path)
        gt_img = _read_rgb(gt_path)
        width, height = base_img.size
        bbox = _clip_bbox(region["bbox_xyxy"], width, height)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            skipped["invalid_bbox"] += 1
            continue
        if (bbox[2] - bbox[0]) < int(args.min_crop_size) or (bbox[3] - bbox[1]) < int(args.min_crop_size):
            skipped["too_small_crop"] += 1
            continue
        key = (stem, bbox)
        if key in seen:
            skipped["duplicate_bbox"] += 1
            continue
        seen.add(key)

        outer = _expand_bbox(bbox, int(args.context_pad), width, height)
        x0, y0, x1, y1 = bbox
        ox0, oy0, ox1, oy1 = outer
        context_mask = np.zeros((height, width), dtype=bool)
        context_mask[oy0:oy1, ox0:ox1] = True
        context_mask[y0:y1, x0:x1] = False
        base_np = np.asarray(base_img)
        cand_np = np.asarray(cand_img)
        gt_np = np.asarray(gt_img)
        diff_stats = _crop_diff_stats(base_np, cand_np, bbox)
        metrics_skipped_equal_crop = not bool(diff_stats.get("crop_changed", False))
        if metrics_skipped_equal_crop:
            base_core = {"psnr": math.nan, "ssim": math.nan, "lpips": math.nan}
            cand_core = {"psnr": math.nan, "ssim": math.nan, "lpips": math.nan}
            delta_psnr = 0.0
            delta_ssim = 0.0
            delta_lpips = 0.0
            balanced = 0.0
        else:
            base_core = _crop_metrics(base_render_path, gt_path, bbox, device, compute_lpips=compute_lpips)
            cand_core = _crop_metrics(cand_render_path, gt_path, bbox, device, compute_lpips=compute_lpips)
            delta_psnr = cand_core["psnr"] - base_core["psnr"]
            delta_ssim = cand_core["ssim"] - base_core["ssim"]
            delta_lpips = cand_core["lpips"] - base_core["lpips"] if compute_lpips else 0.0
            balanced = delta_psnr + float(args.ssim_weight) * delta_ssim - float(args.lpips_weight) * delta_lpips
        base_context_mse = _mse_region(base_np, gt_np, context_mask)
        cand_context_mse = _mse_region(cand_np, gt_np, context_mask)
        context_mse_regression = (
            max(0.0, cand_context_mse - base_context_mse)
            if math.isfinite(base_context_mse) and math.isfinite(cand_context_mse)
            else math.nan
        )

        rows.append(
            {
                "view": stem,
                "carrier_id": region["carrier_id"],
                "bbox_xyxy": list(bbox),
                "pixels": int(region["pixels"]),
                "score": float(region["score"]),
                "baseline_core_psnr": base_core["psnr"],
                "candidate_core_psnr": cand_core["psnr"],
                "baseline_core_ssim": base_core["ssim"],
                "candidate_core_ssim": cand_core["ssim"],
                "baseline_core_lpips": base_core["lpips"],
                "candidate_core_lpips": cand_core["lpips"],
                "delta_core_psnr": float(delta_psnr),
                "delta_core_ssim": float(delta_ssim),
                "delta_core_lpips": float(delta_lpips),
                "core_balanced_delta": float(balanced),
                "baseline_context_mse": float(base_context_mse),
                "candidate_context_mse": float(cand_context_mse),
                "context_mse_regression": float(context_mse_regression),
                "metrics_skipped_equal_crop": bool(metrics_skipped_equal_crop),
                **diff_stats,
            }
        )

    balanced_values = [row["core_balanced_delta"] for row in rows]
    context_reg = [row["context_mse_regression"] for row in rows]
    changed_region_count = int(sum(1 for row in rows if bool(row.get("crop_changed", False))))
    changed_fraction = float(changed_region_count) / max(float(len(rows)), 1.0)
    same_render_byte_rows = int(sum(1 for row in rows if bool(row.get("render_same_bytes", False))))
    skipped_equal_crop_metrics = int(sum(1 for row in rows if bool(row.get("metrics_skipped_equal_crop", False))))
    max_crop_abs_diff = max((float(row.get("crop_max_abs_diff", 0.0)) for row in rows), default=0.0)
    mean_crop_abs_diff = _finite_mean([float(row.get("crop_mean_abs_diff", 0.0)) for row in rows])
    summary = {
        "scene": args.scene,
        "protocol": "train_render_region_objective_from_train_carrier_bboxes",
        "test_usage": "none",
        "carrier_json": str(args.carrier_json),
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "region_count": int(len(rows)),
        "diagnostics": {
            "baseline_candidate_same_path": bool(args.baseline_dir.resolve() == args.candidate_dir.resolve()),
            "skipped": skipped,
            "changed_region_count": changed_region_count,
            "changed_region_fraction": changed_fraction,
            "unchanged_region_count": int(len(rows) - changed_region_count),
            "same_render_byte_rows": same_render_byte_rows,
            "skipped_equal_crop_metrics": skipped_equal_crop_metrics,
            "max_crop_abs_diff": float(max_crop_abs_diff),
            "mean_crop_abs_diff": float(mean_crop_abs_diff),
        },
        "settings": {
            "max_regions": int(args.max_regions),
            "min_region_pixels": int(args.min_region_pixels),
            "min_crop_size": int(args.min_crop_size),
            "context_pad": int(args.context_pad),
            "tail_fraction": float(args.tail_fraction),
            "ssim_weight": float(args.ssim_weight),
            "lpips_weight": float(args.lpips_weight),
            "skip_lpips": bool(args.skip_lpips),
        },
        "mean": {
            "core_balanced_delta": _finite_mean(balanced_values),
            "delta_core_psnr": _finite_mean([row["delta_core_psnr"] for row in rows]),
            "delta_core_ssim": _finite_mean([row["delta_core_ssim"] for row in rows]),
            "delta_core_lpips": _finite_mean([row["delta_core_lpips"] for row in rows]),
            "context_mse_regression": _finite_mean(context_reg),
        },
        "tail": {
            "core_balanced_cvar_delta": _tail_cvar(balanced_values, float(args.tail_fraction)),
            "worst_core_balanced_delta": min(balanced_values) if balanced_values else math.nan,
            "max_context_mse_regression": max(context_reg) if context_reg else math.nan,
            "negative_core_balanced_fraction": (
                float(sum(1 for v in balanced_values if v < 0.0)) / max(float(len(balanced_values)), 1.0)
            ),
        },
        "wins": {
            "core_balanced": int(sum(1 for row in rows if row["core_balanced_delta"] > 0.0)),
            "core_psnr": int(sum(1 for row in rows if row["delta_core_psnr"] > 0.0)),
            "core_ssim": int(sum(1 for row in rows if row["delta_core_ssim"] > 0.0)),
            "core_lpips": int(sum(1 for row in rows if row["delta_core_lpips"] < 0.0)),
        },
        "rows": rows,
    }
    return summary


def write_outputs(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mean = summary["mean"]
    tail = summary["tail"]
    wins = summary["wins"]
    diagnostics = summary.get("diagnostics", {})
    n = int(summary["region_count"])
    lines = [
        "# Train Render-Region Objective",
        "",
        f"- scene: `{summary['scene']}`",
        f"- protocol: `{summary['protocol']}`",
        f"- test usage: `{summary['test_usage']}`",
        f"- baseline: `{summary['baseline_label']}`",
        f"- candidate: `{summary['candidate_label']}`",
        f"- regions: `{n}`",
        f"- mean core balanced delta: `{mean['core_balanced_delta']:+.9f}`",
        f"- mean dPSNR/dSSIM/dLPIPS: `{mean['delta_core_psnr']:+.9f}` / `{mean['delta_core_ssim']:+.9f}` / `{mean['delta_core_lpips']:+.9f}`",
        f"- tail CVaR / worst core balanced delta: `{tail['core_balanced_cvar_delta']:+.9f}` / `{tail['worst_core_balanced_delta']:+.9f}`",
        f"- mean / max context MSE regression: `{mean['context_mse_regression']:.9g}` / `{tail['max_context_mse_regression']:.9g}`",
        f"- changed regions / fraction: `{diagnostics.get('changed_region_count', 0)}/{n}` / `{float(diagnostics.get('changed_region_fraction', 0.0)):.6f}`",
        f"- max / mean crop abs diff: `{float(diagnostics.get('max_crop_abs_diff', 0.0)):.9f}` / `{float(diagnostics.get('mean_crop_abs_diff', 0.0)):.9f}`",
        f"- same path / same-render rows / skipped equal-crop metrics: `{diagnostics.get('baseline_candidate_same_path', False)}` / `{diagnostics.get('same_render_byte_rows', 0)}` / `{diagnostics.get('skipped_equal_crop_metrics', 0)}`",
        f"- wins balanced / PSNR / SSIM / LPIPS: `{wins['core_balanced']}/{n}` / `{wins['core_psnr']}/{n}` / `{wins['core_ssim']}/{n}` / `{wins['core_lpips']}/{n}`",
        "",
        "| view | carrier | pixels | changed px | max diff | core balanced | dPSNR | dSSIM | dLPIPS | context mse reg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"][:40]:
        lines.append(
            f"| {row['view']} | {row['carrier_id']} | {row['pixels']} | "
            f"{row.get('crop_nonzero_pixels', 0)} | {float(row.get('crop_max_abs_diff', 0.0)):.9f} | "
            f"{row['core_balanced_delta']:+.9f} | {row['delta_core_psnr']:+.9f} | "
            f"{row['delta_core_ssim']:+.9f} | {row['delta_core_lpips']:+.9f} | "
            f"{row['context_mse_regression']:.9g} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    summary = evaluate(args)
    write_outputs(args, summary)
    print(json.dumps({k: summary[k] for k in ("scene", "region_count", "mean", "tail", "wins")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
