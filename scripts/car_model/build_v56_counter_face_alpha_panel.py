#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_V52_RENDER_DIR = Path(
    "outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/"
    "v52_capacity_aware_selected_full9/counter/renders"
)
DEFAULT_V52_GT_DIR = Path(
    "outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/"
    "v52_capacity_aware_selected_full9/counter/gt"
)
DEFAULT_V55D_DIR = Path(
    "/dev/shm/peilincai_spcarnet_v55d_face_alpha_caphit_20260623/"
    "counter_v55d_policyval_face_alpha_l1pos09_support4096_tex32_nearest_region_texture_adapter/"
    "test/ours_26000_counter_v55d_policyval_face_alpha_l1pos09_support4096_tex32_nearest_region_texture_adapter"
)
DEFAULT_OUTPUT = Path("assets/spcarnet_v56_counter_face_alpha_guard_panel.png")
DEFAULT_MANIFEST = Path("assets/spcarnet_v56_counter_face_alpha_guard_panel_manifest.json")


def read_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def to_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def abs_error_image(pred: np.ndarray, gt: np.ndarray, scale: float) -> np.ndarray:
    err = np.mean(np.abs(pred - gt), axis=2)
    gray = np.clip(err * scale, 0.0, 1.0)
    return np.repeat(gray[..., None], 3, axis=2)


def improvement_image(err_a: np.ndarray, err_b: np.ndarray, scale: float) -> np.ndarray:
    improvement = err_a - err_b
    pos = np.clip(improvement * scale, 0.0, 1.0)
    neg = np.clip(-improvement * scale, 0.0, 1.0)
    out = np.zeros((*improvement.shape, 3), dtype=np.float32)
    out[..., 1] = pos
    out[..., 0] = neg
    out[..., 2] = neg
    return out


def list_common_views(a: Path, b: Path, c: Path) -> list[str]:
    names = sorted({p.name for p in a.glob("*.png")} & {p.name for p in b.glob("*.png")} & {p.name for p in c.glob("*.png")})
    if not names:
        raise RuntimeError(f"no common PNG views among {a}, {b}, {c}")
    return names


def crop_candidates(
    names: list[str],
    v52_render_dir: Path,
    v55d_render_dir: Path,
    gt_dir: Path,
    crop: int,
    stride: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for name in names:
        gt = read_rgb(gt_dir / name)
        v52 = read_rgb(v52_render_dir / name)
        v55d = read_rgb(v55d_render_dir / name)
        err52 = np.mean((v52 - gt) ** 2, axis=2)
        err55 = np.mean((v55d - gt) ** 2, axis=2)
        improvement = err52 - err55
        h, w = improvement.shape
        for y in range(0, max(h - crop + 1, 1), stride):
            for x in range(0, max(w - crop + 1, 1), stride):
                patch = improvement[y : y + crop, x : x + crop]
                if patch.shape != (crop, crop):
                    continue
                score = float(np.mean(patch))
                pos_fraction = float(np.mean(patch > 0.0))
                candidates.append(
                    {
                        "view": name,
                        "x": int(x),
                        "y": int(y),
                        "crop": int(crop),
                        "score_mse_reduction": score,
                        "positive_fraction": pos_fraction,
                    }
                )
    candidates.sort(key=lambda row: (row["score_mse_reduction"], row["positive_fraction"]), reverse=True)
    return candidates


def pick_non_overlapping(candidates: list[dict[str, Any]], count: int, min_same_view_iou: float) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for cand in candidates:
        ok = True
        for old in picked:
            if cand["view"] != old["view"]:
                continue
            ax1, ay1, ax2, ay2 = cand["x"], cand["y"], cand["x"] + cand["crop"], cand["y"] + cand["crop"]
            bx1, by1, bx2, by2 = old["x"], old["y"], old["x"] + old["crop"], old["y"] + old["crop"]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = cand["crop"] * cand["crop"] * 2 - inter
            if union > 0 and inter / union > min_same_view_iou:
                ok = False
                break
        if ok:
            picked.append(cand)
        if len(picked) >= count:
            break
    return picked


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def build_panel(
    picked: list[dict[str, Any]],
    v52_render_dir: Path,
    v55d_render_dir: Path,
    gt_dir: Path,
    output: Path,
    scale_err: float,
    scale_improve: float,
) -> list[dict[str, Any]]:
    if not picked:
        raise RuntimeError("no crops selected")
    font = ImageFont.load_default()
    crop = int(picked[0]["crop"])
    gap = 8
    label_h = 18
    cols = ["GT", "v52", "v55d", "err v52", "err v55d", "green=better"]
    width = len(cols) * crop + (len(cols) + 1) * gap
    height = len(picked) * (crop + label_h + gap) + gap
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    manifest_rows: list[dict[str, Any]] = []
    for row_i, item in enumerate(picked):
        name = str(item["view"])
        x0, y0 = int(item["x"]), int(item["y"])
        gt = read_rgb(gt_dir / name)
        v52 = read_rgb(v52_render_dir / name)
        v55d = read_rgb(v55d_render_dir / name)
        sl = np.s_[y0 : y0 + crop, x0 : x0 + crop]
        gt_c = gt[sl]
        v52_c = v52[sl]
        v55d_c = v55d[sl]
        err52 = np.mean((v52_c - gt_c) ** 2, axis=2)
        err55 = np.mean((v55d_c - gt_c) ** 2, axis=2)
        images = [
            gt_c,
            v52_c,
            v55d_c,
            abs_error_image(v52_c, gt_c, scale_err),
            abs_error_image(v55d_c, gt_c, scale_err),
            improvement_image(err52, err55, scale_improve),
        ]
        top = gap + row_i * (crop + label_h + gap)
        for col_i, (title, image) in enumerate(zip(cols, images)):
            left = gap + col_i * (crop + gap)
            canvas.paste(Image.fromarray(to_uint8(image)), (left, top + label_h))
            label(draw, (left + 4, top + 2), title, font)
        label(draw, (gap + 4, top + label_h + 4), f"{name} crop=({x0},{y0})", font)
        manifest_rows.append(
            {
                **item,
                "crop_mse_v52": float(np.mean(err52)),
                "crop_mse_v55d": float(np.mean(err55)),
                "crop_mse_reduction": float(np.mean(err52 - err55)),
                "crop_positive_fraction": float(np.mean((err52 - err55) > 0.0)),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v52_render_dir", type=Path, default=DEFAULT_V52_RENDER_DIR)
    parser.add_argument("--v52_gt_dir", type=Path, default=DEFAULT_V52_GT_DIR)
    parser.add_argument("--v55d_render_dir", type=Path, default=DEFAULT_V55D_DIR / "renders")
    parser.add_argument("--v55d_gt_dir", type=Path, default=DEFAULT_V55D_DIR / "gt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--crop", type=int, default=220)
    parser.add_argument("--stride", type=int, default=110)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--min_same_view_iou", type=float, default=0.2)
    parser.add_argument("--scale_err", type=float, default=8.0)
    parser.add_argument("--scale_improve", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = list_common_views(args.v52_render_dir, args.v55d_render_dir, args.v52_gt_dir)
    gt_names = list_common_views(args.v52_gt_dir, args.v55d_gt_dir, args.v52_render_dir)
    names = sorted(set(names) & set(gt_names))
    candidates = crop_candidates(names, args.v52_render_dir, args.v55d_render_dir, args.v52_gt_dir, args.crop, args.stride)
    picked = pick_non_overlapping(candidates, args.rows, args.min_same_view_iou)
    manifest_rows = build_panel(picked, args.v52_render_dir, args.v55d_render_dir, args.v52_gt_dir, args.output, args.scale_err, args.scale_improve)
    payload = {
        "output": str(args.output),
        "v52_render_dir": str(args.v52_render_dir),
        "v55d_render_dir": str(args.v55d_render_dir),
        "gt_dir": str(args.v52_gt_dir),
        "selection": "mechanical_top_crop_mse_reduction_v52_minus_v55d",
        "crop": int(args.crop),
        "stride": int(args.stride),
        "rows": manifest_rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": str(args.manifest), "selected": manifest_rows}, indent=2))


if __name__ == "__main__":
    main()
