#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_SCENES = ("garden", "room", "counter", "bonsai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware"),
    )
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--crop_w", type=int, default=320)
    parser.add_argument("--crop_h", type=int, default=220)
    parser.add_argument("--cell_w", type=int, default=320)
    parser.add_argument("--cell_h", type=int, default=220)
    parser.add_argument("--max_views_per_scene", type=int, default=1)
    parser.add_argument("--max_candidate_views", type=int, default=6)
    parser.add_argument("--search_scale", type=float, default=0.25)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("assets/spcarnet_v42_atlas_qualitative_panel.png"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("assets/spcarnet_v42_atlas_qualitative_panel_manifest.json"),
    )
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_method_dir(model_dir: Path) -> Path:
    test_dir = model_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"missing test dir: {test_dir}")
    dirs = sorted(p for p in test_dir.iterdir() if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"no method dir under {test_dir}")
    return dirs[0]


def method_key(per_view_path: Path) -> str:
    payload = read_json(per_view_path)
    keys = list(payload.keys())
    if len(keys) != 1:
        raise ValueError(f"expected one method key in {per_view_path}, got {keys}")
    return keys[0]


def imread_float(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def integral_image(arr: np.ndarray) -> np.ndarray:
    return np.pad(arr.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def window_sum(integral: np.ndarray, h: int, w: int) -> np.ndarray:
    return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]


def choose_crop(gt: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, crop_w: int, crop_h: int) -> dict[str, Any]:
    h, w = gt.shape[:2]
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    err_base = np.mean(np.abs(baseline - gt), axis=2)
    err_cand = np.mean(np.abs(candidate - gt), axis=2)
    improvement = err_base - err_cand

    gray = gt.mean(axis=2)
    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    grad_y[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    texture = grad_x + grad_y
    texture = np.clip(texture / (float(np.percentile(texture, 97)) + 1e-6), 0.0, 1.0)

    positive = np.maximum(improvement, 0.0)
    negative = np.maximum(-improvement, 0.0)
    score_map = positive * (0.30 + 0.70 * texture) - 0.65 * negative

    margin_x = min(max(crop_w // 8, 16), max(w // 8, 16))
    margin_y = min(max(crop_h // 8, 16), max(h // 8, 16))
    score_map[:margin_y, :] *= 0.4
    score_map[-margin_y:, :] *= 0.4
    score_map[:, :margin_x] *= 0.4
    score_map[:, -margin_x:] *= 0.4

    scores = window_sum(integral_image(score_map), crop_h, crop_w)
    best_y, best_x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    x = max(0, min(int(best_x), w - crop_w))
    y = max(0, min(int(best_y), h - crop_h))
    crop = np.s_[y : y + crop_h, x : x + crop_w]

    base_mse = float(np.mean((baseline[crop] - gt[crop]) ** 2))
    cand_mse = float(np.mean((candidate[crop] - gt[crop]) ** 2))
    base_mae = float(np.mean(err_base[crop]))
    cand_mae = float(np.mean(err_cand[crop]))
    return {
        "x": x,
        "y": y,
        "w": crop_w,
        "h": crop_h,
        "local_base_psnr": psnr_from_mse(base_mse),
        "local_candidate_psnr": psnr_from_mse(cand_mse),
        "local_dpsnr": psnr_from_mse(cand_mse) - psnr_from_mse(base_mse),
        "local_base_mae": base_mae,
        "local_candidate_mae": cand_mae,
        "local_mae_drop_pct": 100.0 * (base_mae - cand_mae) / max(base_mae, 1e-8),
        "local_positive_pixel_ratio": float(np.mean(improvement[crop] > 0.0)),
        "local_score": float(scores[best_y, best_x] / (crop_w * crop_h)),
    }


def crop_metrics(gt: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, crop: dict[str, Any]) -> dict[str, Any]:
    y0 = int(crop["y"])
    y1 = y0 + int(crop["h"])
    x0 = int(crop["x"])
    x1 = x0 + int(crop["w"])
    region = np.s_[y0:y1, x0:x1]
    err_base = np.mean(np.abs(baseline - gt), axis=2)
    err_cand = np.mean(np.abs(candidate - gt), axis=2)
    improvement = err_base - err_cand
    base_mse = float(np.mean((baseline[region] - gt[region]) ** 2))
    cand_mse = float(np.mean((candidate[region] - gt[region]) ** 2))
    base_mae = float(np.mean(err_base[region]))
    cand_mae = float(np.mean(err_cand[region]))
    out = dict(crop)
    out.update(
        {
            "local_base_psnr": psnr_from_mse(base_mse),
            "local_candidate_psnr": psnr_from_mse(cand_mse),
            "local_dpsnr": psnr_from_mse(cand_mse) - psnr_from_mse(base_mse),
            "local_base_mae": base_mae,
            "local_candidate_mae": cand_mae,
            "local_mae_drop_pct": 100.0 * (base_mae - cand_mae) / max(base_mae, 1e-8),
            "local_positive_pixel_ratio": float(np.mean(improvement[region] > 0.0)),
        }
    )
    return out


def improvement_map(gt: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> Image.Image:
    err_base = np.mean(np.abs(baseline - gt), axis=2)
    err_cand = np.mean(np.abs(candidate - gt), axis=2)
    delta = err_base - err_cand
    scale = float(np.percentile(np.abs(delta), 99)) + 1e-6
    pos = np.clip(delta / scale, 0.0, 1.0)
    neg = np.clip(-delta / scale, 0.0, 1.0)
    heat = np.zeros((*delta.shape, 3), dtype=np.float32)
    heat[..., 0] = 0.10 + 0.85 * neg
    heat[..., 1] = 0.10 + 0.85 * pos
    heat[..., 2] = 0.12 + 0.25 * np.maximum(pos, neg)
    return Image.fromarray(np.clip(heat * 255.0, 0, 255).astype(np.uint8))


def crop_image(path: Path, crop: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    box = (crop["x"], crop["y"], crop["x"] + crop["w"], crop["y"] + crop["h"])
    return Image.open(path).convert("RGB").crop(box).resize(size, Image.Resampling.LANCZOS)


def scene_dirs(root: Path, scene: str) -> dict[str, Path]:
    return {
        "noop": root / f"{scene}_evidence_noop_compact_baseline",
        "v41": root / f"{scene}_v41_facemean_expanded_region_texture_adapter",
        "v42": root / f"{scene}_v42_ssimgate_confidence_weighted_region_texture_adapter",
    }


def scale_crop_to_full(crop: dict[str, Any], scaled_size: tuple[int, int], full_size: tuple[int, int]) -> dict[str, Any]:
    sw, sh = scaled_size
    fw, fh = full_size
    sx = fw / max(float(sw), 1.0)
    sy = fh / max(float(sh), 1.0)
    out = dict(crop)
    out["x"] = max(0, min(int(round(crop["x"] * sx)), fw - 1))
    out["y"] = max(0, min(int(round(crop["y"] * sy)), fh - 1))
    out["w"] = min(int(round(crop["w"] * sx)), fw - out["x"])
    out["h"] = min(int(round(crop["h"] * sy)), fh - out["y"])
    return out


def collect_scene_examples(
    root: Path,
    scene: str,
    crop_w: int,
    crop_h: int,
    max_views: int,
    max_candidate_views: int,
    search_scale: float,
) -> list[dict[str, Any]]:
    dirs = scene_dirs(root, scene)
    for key, path in dirs.items():
        if not path.is_dir():
            raise FileNotFoundError(f"missing {key} dir for {scene}: {path}")

    method_dirs = {key: first_method_dir(path) for key, path in dirs.items()}
    keys = {key: method_key(path / "per_view.json") for key, path in dirs.items()}
    per_view = {key: read_json(path / "per_view.json")[keys[key]] for key, path in dirs.items()}

    view_names = sorted(set(per_view["noop"]["PSNR"]) & set(per_view["v41"]["PSNR"]) & set(per_view["v42"]["PSNR"]))
    view_scores: list[tuple[float, str]] = []
    for view in view_names:
        dpsnr = per_view["v42"]["PSNR"][view] - per_view["noop"]["PSNR"][view]
        dssim = per_view["v42"]["SSIM"][view] - per_view["noop"]["SSIM"][view]
        dlpips = per_view["v42"]["LPIPS"][view] - per_view["noop"]["LPIPS"][view]
        strict_bonus = 10.0 if dpsnr > 0.0 and dssim > 0.0 and dlpips < 0.0 else 0.0
        view_scores.append((strict_bonus + dpsnr + 20.0 * dssim - 20.0 * dlpips, view))
    view_scores.sort(reverse=True)
    view_names = [view for _, view in view_scores[: max(1, int(max_candidate_views))]]

    rows: list[dict[str, Any]] = []
    for view in view_names:
        paths = {
            "gt": method_dirs["noop"] / "gt" / view,
            "noop": method_dirs["noop"] / "renders" / view,
            "v41": method_dirs["v41"] / "renders" / view,
            "v42": method_dirs["v42"] / "renders" / view,
        }
        if not all(p.is_file() for p in paths.values()):
            continue

        with Image.open(paths["gt"]) as image:
            full_size = image.size
        scale = min(1.0, max(0.05, float(search_scale)))
        scaled_size = (max(8, int(round(full_size[0] * scale))), max(8, int(round(full_size[1] * scale))))

        gt_small = imread_float(paths["gt"], scaled_size)
        noop_small = imread_float(paths["noop"], scaled_size)
        v42_small = imread_float(paths["v42"], scaled_size)
        if not (gt_small.shape == noop_small.shape == v42_small.shape):
            continue

        crop_small = choose_crop(
            gt_small,
            noop_small,
            v42_small,
            max(8, int(round(crop_w * scale))),
            max(8, int(round(crop_h * scale))),
        )
        crop_full = scale_crop_to_full(crop_small, scaled_size, full_size)

        gt = imread_float(paths["gt"])
        noop = imread_float(paths["noop"])
        v41 = imread_float(paths["v41"])
        v42 = imread_float(paths["v42"])
        if not (gt.shape == noop.shape == v41.shape == v42.shape):
            continue

        crop = crop_metrics(gt, noop, v42, crop_full)
        crop["local_score"] = crop_small.get("local_score", 0.0)
        full_delta = {
            "dPSNR_v42_noop": per_view["v42"]["PSNR"][view] - per_view["noop"]["PSNR"][view],
            "dSSIM_v42_noop": per_view["v42"]["SSIM"][view] - per_view["noop"]["SSIM"][view],
            "dLPIPS_v42_noop": per_view["v42"]["LPIPS"][view] - per_view["noop"]["LPIPS"][view],
            "dPSNR_v42_v41": per_view["v42"]["PSNR"][view] - per_view["v41"]["PSNR"][view],
            "dSSIM_v42_v41": per_view["v42"]["SSIM"][view] - per_view["v41"]["SSIM"][view],
            "dLPIPS_v42_v41": per_view["v42"]["LPIPS"][view] - per_view["v41"]["LPIPS"][view],
        }
        ranking = (
            crop["local_score"]
            * max(crop["local_mae_drop_pct"], 0.0)
            * (1.0 + crop["local_positive_pixel_ratio"])
            * (1.0 if full_delta["dPSNR_v42_noop"] > 0.0 else 0.25)
        )
        rows.append(
            {
                "scene": scene,
                "view": view,
                "ranking_score": ranking,
                "crop": crop,
                "paths": {k: str(v) for k, v in paths.items()},
                "full_delta": full_delta,
            }
        )
    rows.sort(key=lambda x: x["ranking_score"], reverse=True)
    return rows[: max(1, int(max_views))]


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    draw.text(xy, text, font=font, fill=fill)


def compose(examples: list[dict[str, Any]], out: Path, cell_w: int, cell_h: int) -> None:
    title_font = load_font(24, bold=True)
    font = load_font(17)
    small = load_font(14)
    cols = ("GT", "No-op compact", "v41 atlas", "v42 SSIMGate", "v42 error reduction")
    gap = 14
    left = 28
    top = 86
    label_h = 58
    header_h = 28
    row_h = label_h + header_h + cell_h + 20
    width = left * 2 + len(cols) * cell_w + (len(cols) - 1) * gap
    height = top + len(examples) * row_h + 28
    canvas = Image.new("RGB", (width, height), (23, 24, 28))
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (left, 20), "v42 Surface Atlas: same-evidence qualitative comparison", title_font, (246, 246, 246))
    draw_text(
        draw,
        (left, 54),
        "Green: v42 is closer to GT than no-op; magenta: worse. Crops are selected by local held-out error reduction.",
        small,
        (190, 196, 205),
    )
    size = (cell_w, cell_h)
    for idx, item in enumerate(examples):
        y0 = top + idx * row_h
        crop = item["crop"]
        paths = {k: Path(v) for k, v in item["paths"].items()}
        full = item["full_delta"]
        label = (
            f"{item['scene']} / {item['view']}  "
            f"full v42-noop: dPSNR {full['dPSNR_v42_noop']:+.5f}, "
            f"dSSIM {full['dSSIM_v42_noop']:+.8f}, dLPIPS {full['dLPIPS_v42_noop']:+.8f}  |  "
            f"crop dPSNR {crop['local_dpsnr']:+.4f}, MAE drop {crop['local_mae_drop_pct']:.2f}%"
        )
        draw_text(draw, (left, y0), label, font, (242, 242, 242))

        gt_crop = imread_float(paths["gt"])[crop["y"] : crop["y"] + crop["h"], crop["x"] : crop["x"] + crop["w"]]
        noop_crop = imread_float(paths["noop"])[crop["y"] : crop["y"] + crop["h"], crop["x"] : crop["x"] + crop["w"]]
        v42_crop = imread_float(paths["v42"])[crop["y"] : crop["y"] + crop["h"], crop["x"] : crop["x"] + crop["w"]]
        panels = (
            crop_image(paths["gt"], crop, size),
            crop_image(paths["noop"], crop, size),
            crop_image(paths["v41"], crop, size),
            crop_image(paths["v42"], crop, size),
            improvement_map(gt_crop, noop_crop, v42_crop).resize(size, Image.Resampling.NEAREST),
        )
        for col, (header, panel) in enumerate(zip(cols, panels)):
            x0 = left + col * (cell_w + gap)
            draw_text(draw, (x0, y0 + label_h), header, small, (210, 216, 224))
            py = y0 + label_h + header_h
            canvas.paste(panel, (x0, py))
            draw.rectangle([x0, py, x0 + cell_w, py + cell_h], outline=(72, 76, 86), width=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    args = parse_args()
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    examples: list[dict[str, Any]] = []
    for scene in scenes:
        examples.extend(
            collect_scene_examples(
                args.root,
                scene,
                args.crop_w,
                args.crop_h,
                args.max_views_per_scene,
                args.max_candidate_views,
                args.search_scale,
            )
        )
    compose(examples, args.out, args.cell_w, args.cell_h)
    manifest = {
        "description": "Same-evidence no-op/v41/v42 qualitative panel for the v42 surface residual atlas.",
        "root": str(args.root),
        "selection_rule": (
            "For each scene, choose held-out crops with the largest local v42-vs-no-op "
            "RGB error reduction among a small per-view-metric preselection, weighted by GT texture and "
            "penalized by local regressions."
        ),
        "out": str(args.out),
        "examples": examples,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.manifest}")
    for item in examples:
        crop = item["crop"]
        full = item["full_delta"]
        print(
            item["scene"],
            item["view"],
            f"full_d=({full['dPSNR_v42_noop']:+.6f},{full['dSSIM_v42_noop']:+.8f},{full['dLPIPS_v42_noop']:+.8f})",
            f"crop_dpsnr={crop['local_dpsnr']:+.5f}",
            f"mae_drop={crop['local_mae_drop_pct']:.3f}%",
            f"pos={100.0 * crop['local_positive_pixel_ratio']:.1f}%",
        )


if __name__ == "__main__":
    main()
