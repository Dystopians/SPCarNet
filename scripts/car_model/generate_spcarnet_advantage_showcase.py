#!/usr/bin/env python3
"""Generate local-error qualitative panels for the current SPCarNet table.

The script uses the same clean MeshSplatting baseline selection recorded in the
full9 CSV report, then searches held-out test renders for crops where SPCarNet
reduces the local RGB error relative to the selected clean baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_SCENES = (
    "bicycle",
    "flowers",
    "garden",
    "stump",
    "treehill",
    "room",
    "counter",
    "kitchen",
    "bonsai",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k"),
    )
    parser.add_argument(
        "--method_root",
        type=Path,
        default=Path(
            "outputs/carnet/meshsplatopt/paper_m360_repro/"
            "compact_ela_sor_adaptive_geo_26k"
        ),
    )
    parser.add_argument(
        "--report_csv",
        type=Path,
        default=Path(
            "outputs/carnet/meshsplatopt/paper_m360_repro/"
            "compact_ela_sor_adaptive_geo_26k/compact_ela_vs_clean.csv"
        ),
    )
    parser.add_argument("--policy_tag", default="sor_adaptive_geo")
    parser.add_argument(
        "--method_name", default="ours_26000_sor_adaptive_geo_compact_ela"
    )
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("assets/spcarnet_m360_where_it_helps_showcase.png"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("assets/spcarnet_m360_where_it_helps_selection.json"),
    )
    parser.add_argument("--max_examples", type=int, default=6)
    parser.add_argument("--crop_w", type=int, default=360)
    parser.add_argument("--crop_h", type=int, default=250)
    parser.add_argument("--cell_w", type=int, default=360)
    parser.add_argument("--cell_h", type=int, default=250)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_baseline_iterations(report_csv: Path) -> dict[str, str]:
    selected: dict[str, str] = {}
    with report_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            selected[row["scene"]] = row["baseline_iteration"]
    return selected


def imread_float(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def integral_image(arr: np.ndarray) -> np.ndarray:
    return np.pad(arr.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def window_sum(integral: np.ndarray, h: int, w: int) -> np.ndarray:
    return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]


def fit_crop(x: int, y: int, w: int, h: int, image_w: int, image_h: int) -> tuple[int, int]:
    return max(0, min(x, image_w - w)), max(0, min(y, image_h - h))


def choose_crop(
    gt: np.ndarray,
    clean: np.ndarray,
    ours: np.ndarray,
    crop_w: int,
    crop_h: int,
) -> dict[str, Any]:
    h, w = gt.shape[:2]
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    err_clean = np.mean(np.abs(clean - gt), axis=2)
    err_ours = np.mean(np.abs(ours - gt), axis=2)
    improvement = err_clean - err_ours

    gray = gt.mean(axis=2)
    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    grad_y[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    texture = grad_x + grad_y
    tex_scale = float(np.percentile(texture, 97)) + 1e-6
    texture = np.clip(texture / tex_scale, 0.0, 1.0)

    positive = np.maximum(improvement, 0.0)
    negative = np.maximum(-improvement, 0.0)
    # Reward visible, textured improvements and penalize regions where ours is worse.
    score_map = positive * (0.35 + 0.65 * texture) - 0.5 * negative

    margin_x = min(max(crop_w // 8, 16), max(w // 8, 16))
    margin_y = min(max(crop_h // 8, 16), max(h // 8, 16))
    score_map[:margin_y, :] *= 0.4
    score_map[-margin_y:, :] *= 0.4
    score_map[:, :margin_x] *= 0.4
    score_map[:, -margin_x:] *= 0.4

    scores = window_sum(integral_image(score_map), crop_h, crop_w)
    best_y, best_x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    x, y = fit_crop(int(best_x), int(best_y), crop_w, crop_h, w, h)

    crop = np.s_[y : y + crop_h, x : x + crop_w]
    clean_mse = float(np.mean((clean[crop] - gt[crop]) ** 2))
    ours_mse = float(np.mean((ours[crop] - gt[crop]) ** 2))
    clean_mae = float(np.mean(err_clean[crop]))
    ours_mae = float(np.mean(err_ours[crop]))
    pos_ratio = float(np.mean(improvement[crop] > 0))
    local_score = float(scores[best_y, best_x] / (crop_w * crop_h))
    return {
        "x": x,
        "y": y,
        "w": crop_w,
        "h": crop_h,
        "local_clean_psnr": psnr_from_mse(clean_mse),
        "local_ours_psnr": psnr_from_mse(ours_mse),
        "local_dpsnr": psnr_from_mse(ours_mse) - psnr_from_mse(clean_mse),
        "local_clean_mae": clean_mae,
        "local_ours_mae": ours_mae,
        "local_mae_drop_pct": 100.0 * (clean_mae - ours_mae) / max(clean_mae, 1e-8),
        "local_positive_pixel_ratio": pos_ratio,
        "local_score": local_score,
    }


def image_crop(path: Path, box: tuple[int, int, int, int], size: tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("RGB").crop(box).resize(size, Image.Resampling.LANCZOS)


def make_delta_map(gt: np.ndarray, clean: np.ndarray, ours: np.ndarray) -> Image.Image:
    err_clean = np.mean(np.abs(clean - gt), axis=2)
    err_ours = np.mean(np.abs(ours - gt), axis=2)
    delta = err_clean - err_ours
    scale = float(np.percentile(np.abs(delta), 99)) + 1e-6
    pos = np.clip(delta / scale, 0.0, 1.0)
    neg = np.clip(-delta / scale, 0.0, 1.0)
    heat = np.zeros((*delta.shape, 3), dtype=np.float32)
    heat[..., 0] = 0.10 + 0.85 * neg
    heat[..., 1] = 0.10 + 0.85 * pos
    heat[..., 2] = 0.10 + 0.25 * np.maximum(pos, neg)
    return Image.fromarray(np.clip(heat * 255.0, 0, 255).astype(np.uint8))


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (240, 240, 240),
) -> None:
    draw.text(xy, text, font=font, fill=fill)


def collect_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    baseline_iters = read_baseline_iterations(args.report_csv)
    candidates: list[dict[str, Any]] = []

    for scene in scenes:
        baseline_iter = baseline_iters.get(scene, "26000")
        clean_key = f"ours_{baseline_iter}"
        clean_metrics = read_json(args.clean_root / scene / "per_view.json")[clean_key]
        method_metrics = read_json(
            args.method_root
            / scene
            / args.policy_tag
            / "compact_model"
            / "per_view.json"
        )[args.method_name]

        clean_dir = args.clean_root / scene / "test" / clean_key
        method_dir = (
            args.method_root
            / scene
            / args.policy_tag
            / "compact_model"
            / "test"
            / args.method_name
        )

        view_names = sorted(clean_metrics["PSNR"].keys())
        for view_name in view_names:
            dpsnr = method_metrics["PSNR"][view_name] - clean_metrics["PSNR"][view_name]
            dssim = method_metrics["SSIM"][view_name] - clean_metrics["SSIM"][view_name]
            dlpips = method_metrics["LPIPS"][view_name] - clean_metrics["LPIPS"][view_name]
            if dpsnr <= 0.0 or dssim <= 0.0 or dlpips >= 0.0:
                continue

            gt_path = clean_dir / "gt" / view_name
            clean_path = clean_dir / "renders" / view_name
            ours_path = method_dir / "renders" / view_name
            if not gt_path.exists() or not clean_path.exists() or not ours_path.exists():
                continue

            gt = imread_float(gt_path)
            clean = imread_float(clean_path)
            ours = imread_float(ours_path)
            if gt.shape != clean.shape or gt.shape != ours.shape:
                continue

            crop = choose_crop(gt, clean, ours, args.crop_w, args.crop_h)
            ranking_score = (
                crop["local_score"]
                * max(crop["local_mae_drop_pct"], 0.0)
                * (1.0 + crop["local_positive_pixel_ratio"])
            )
            candidates.append(
                {
                    "scene": scene,
                    "view": view_name,
                    "baseline_iteration": int(baseline_iter),
                    "dPSNR": dpsnr,
                    "dSSIM": dssim,
                    "dLPIPS": dlpips,
                    "ranking_score": ranking_score,
                    "paths": {
                        "gt": str(gt_path),
                        "clean": str(clean_path),
                        "ours": str(ours_path),
                    },
                    "crop": crop,
                }
            )

    candidates.sort(key=lambda x: x["ranking_score"], reverse=True)
    selected: list[dict[str, Any]] = []
    used_scenes: set[str] = set()
    for candidate in candidates:
        if candidate["scene"] in used_scenes:
            continue
        selected.append(candidate)
        used_scenes.add(candidate["scene"])
        if len(selected) >= args.max_examples:
            break
    if len(selected) < args.max_examples:
        used_pairs = {(x["scene"], x["view"]) for x in selected}
        for candidate in candidates:
            pair = (candidate["scene"], candidate["view"])
            if pair in used_pairs:
                continue
            selected.append(candidate)
            used_pairs.add(pair)
            if len(selected) >= args.max_examples:
                break
    return selected


def compose_showcase(args: argparse.Namespace, selected: list[dict[str, Any]]) -> None:
    font_title = load_font(24)
    font = load_font(18)
    small_font = load_font(15)

    cell_w, cell_h = args.cell_w, args.cell_h
    cols = ("GT crop", "Clean MeshSplatting", "SPCarNet", "Error reduction")
    gap = 18
    left = 32
    top = 92
    row_label_h = 54
    col_label_h = 32
    row_h = row_label_h + col_label_h + cell_h + 22
    width = left * 2 + len(cols) * cell_w + (len(cols) - 1) * gap
    height = top + len(selected) * row_h + 24

    canvas = Image.new("RGB", (width, height), (22, 24, 28))
    draw = ImageDraw.Draw(canvas)
    draw_label(
        draw,
        (left, 22),
        "Where SPCarNet Helps Most: local held-out error reduction",
        font_title,
        (250, 250, 250),
    )
    draw_label(
        draw,
        (left, 56),
        "Green means SPCarNet is closer to GT than the selected clean MeshSplatting baseline; magenta means worse.",
        small_font,
        (190, 196, 205),
    )

    for row_idx, item in enumerate(selected):
        row_y = top + row_idx * row_h
        crop = item["crop"]
        box = (crop["x"], crop["y"], crop["x"] + crop["w"], crop["y"] + crop["h"])
        size = (cell_w, cell_h)

        gt = image_crop(Path(item["paths"]["gt"]), box, size)
        clean = image_crop(Path(item["paths"]["clean"]), box, size)
        ours = image_crop(Path(item["paths"]["ours"]), box, size)

        gt_arr = imread_float(Path(item["paths"]["gt"]))[
            crop["y"] : crop["y"] + crop["h"], crop["x"] : crop["x"] + crop["w"]
        ]
        clean_arr = imread_float(Path(item["paths"]["clean"]))[
            crop["y"] : crop["y"] + crop["h"], crop["x"] : crop["x"] + crop["w"]
        ]
        ours_arr = imread_float(Path(item["paths"]["ours"]))[
            crop["y"] : crop["y"] + crop["h"], crop["x"] : crop["x"] + crop["w"]
        ]
        delta = make_delta_map(gt_arr, clean_arr, ours_arr).resize(
            size, Image.Resampling.NEAREST
        )

        label = (
            f"{item['scene']} / {item['view']}  "
            f"full dPSNR {item['dPSNR']:+.2f}, dSSIM {item['dSSIM']:+.4f}, "
            f"dLPIPS {item['dLPIPS']:+.4f}  |  "
            f"crop dPSNR {crop['local_dpsnr']:+.2f}, "
            f"MAE drop {crop['local_mae_drop_pct']:.1f}%"
        )
        draw_label(draw, (left, row_y), label, font, (245, 245, 245))

        for col_idx, (header, panel) in enumerate(zip(cols, (gt, clean, ours, delta))):
            x = left + col_idx * (cell_w + gap)
            draw_label(draw, (x, row_y + row_label_h), header, small_font, (210, 216, 224))
            canvas.paste(panel, (x, row_y + row_label_h + col_label_h))
            draw.rectangle(
                [x, row_y + row_label_h + col_label_h, x + cell_w, row_y + row_label_h + col_label_h + cell_h],
                outline=(70, 74, 84),
                width=1,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)


def main() -> None:
    args = parse_args()
    selected = collect_candidates(args)
    compose_showcase(args, selected)
    manifest = {
        "description": (
            "Automatically selected local held-out crops where SPCarNet reduces "
            "RGB error relative to the selected clean MeshSplatting baseline."
        ),
        "selection_rule": (
            "Per scene, require full-view dPSNR>0, dSSIM>0, dLPIPS<0 under the "
            "full9 baseline selection, then rank crops by local positive error "
            "reduction weighted by GT texture and penalized by negative error."
        ),
        "showcase": str(args.out),
        "examples": selected,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.manifest}")
    for item in selected:
        c = item["crop"]
        print(
            item["scene"],
            item["view"],
            f"full=({item['dPSNR']:+.3f},{item['dSSIM']:+.4f},{item['dLPIPS']:+.4f})",
            f"crop_dpsnr={c['local_dpsnr']:+.3f}",
            f"mae_drop={c['local_mae_drop_pct']:.1f}%",
            f"pos={100*c['local_positive_pixel_ratio']:.1f}%",
        )


if __name__ == "__main__":
    main()
