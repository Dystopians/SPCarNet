#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def imread(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def psnr(mse: float) -> float:
    if mse <= 1.0e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def integral(arr: np.ndarray) -> np.ndarray:
    return np.pad(arr.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def window_sum(ii: np.ndarray, h: int, w: int) -> np.ndarray:
    return ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]


def choose_crop(gt: np.ndarray, base: np.ndarray, cand: np.ndarray, crop_w: int, crop_h: int) -> dict[str, Any]:
    h, w = gt.shape[:2]
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)
    err_base = np.mean(np.abs(base - gt), axis=2)
    err_cand = np.mean(np.abs(cand - gt), axis=2)
    improvement = err_base - err_cand
    positive = np.maximum(improvement, 0.0)
    negative = np.maximum(-improvement, 0.0)
    gray = gt.mean(axis=2)
    grad = np.zeros_like(gray)
    grad[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
    grad[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])
    texture = np.clip(grad / (float(np.percentile(grad, 97)) + 1.0e-6), 0.0, 1.0)
    score_map = positive * (0.35 + 0.65 * texture) - 0.55 * negative
    scores = window_sum(integral(score_map), crop_h, crop_w)
    y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    x = max(0, min(int(x), w - crop_w))
    y = max(0, min(int(y), h - crop_h))
    return {"x": x, "y": y, "w": crop_w, "h": crop_h, "score": float(scores[y, x] / (crop_w * crop_h))}


def crop_arr(arr: np.ndarray, crop: dict[str, Any]) -> np.ndarray:
    x = int(crop["x"])
    y = int(crop["y"])
    w = int(crop["w"])
    h = int(crop["h"])
    return arr[y : y + h, x : x + w]


def crop_metrics(gt: np.ndarray, base: np.ndarray, cand: np.ndarray, crop: dict[str, Any]) -> dict[str, float]:
    gt_c = crop_arr(gt, crop)
    base_c = crop_arr(base, crop)
    cand_c = crop_arr(cand, crop)
    base_mse = float(np.mean((base_c - gt_c) ** 2))
    cand_mse = float(np.mean((cand_c - gt_c) ** 2))
    base_mae = float(np.mean(np.abs(base_c - gt_c)))
    cand_mae = float(np.mean(np.abs(cand_c - gt_c)))
    return {
        "v48_local_psnr": psnr(base_mse),
        "v52_local_psnr": psnr(cand_mse),
        "local_dpsnr": psnr(cand_mse) - psnr(base_mse),
        "v48_local_mae": base_mae,
        "v52_local_mae": cand_mae,
        "local_mae_delta": base_mae - cand_mae,
    }


def heatmap(gt: np.ndarray, base: np.ndarray, cand: np.ndarray, crop: dict[str, Any]) -> Image.Image:
    gt_c = crop_arr(gt, crop)
    base_c = crop_arr(base, crop)
    cand_c = crop_arr(cand, crop)
    err_base = np.mean(np.abs(base_c - gt_c), axis=2)
    err_cand = np.mean(np.abs(cand_c - gt_c), axis=2)
    delta = err_base - err_cand
    scale = float(np.percentile(np.abs(delta), 99)) + 1.0e-6
    pos = np.clip(delta / scale, 0.0, 1.0)
    neg = np.clip(-delta / scale, 0.0, 1.0)
    out = np.zeros((*delta.shape, 3), dtype=np.float32)
    out[..., 0] = 0.12 + 0.82 * neg
    out[..., 1] = 0.12 + 0.82 * pos
    out[..., 2] = 0.12 + 0.20 * np.maximum(pos, neg)
    return Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8))


def to_pil(arr: np.ndarray, crop: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(np.clip(crop_arr(arr, crop) * 255.0, 0, 255).astype(np.uint8))
    return image.resize(size, Image.Resampling.LANCZOS)


def label_cell(image: Image.Image, title: str, subtitle: str, cell_size: tuple[int, int]) -> Image.Image:
    w, h = cell_size
    header_h = 42
    out = Image.new("RGB", (w, h + header_h), (248, 248, 248))
    image = image.resize((w, h), Image.Resampling.LANCZOS)
    out.paste(image, (0, header_h))
    draw = ImageDraw.Draw(out)
    draw.text((8, 5), title, fill=(20, 20, 20), font=load_font(15, bold=True))
    draw.text((8, 24), subtitle, fill=(70, 70, 70), font=load_font(12))
    return out


def hcat(images: list[Image.Image], gap: int = 10) -> Image.Image:
    w = sum(img.width for img in images) + gap * (len(images) - 1)
    h = max(img.height for img in images)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for img in images:
        out.paste(img, (x, 0))
        x += img.width + gap
    return out


def vcat(images: list[Image.Image], gap: int = 14) -> Image.Image:
    w = max(img.width for img in images)
    h = sum(img.height for img in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.height + gap
    return out


def method_dirs_from_audit(model_dir: Path) -> tuple[Path, Path]:
    audit = read_json(model_dir / "surface_residual_region_texture_adapter_audit.json")
    target = audit.get("target_apply", {}) or {}
    render = Path(str(target.get("render_dir", "")))
    gt = Path(str(target.get("gt_dir", "")))
    if render.is_dir() and gt.is_dir():
        return render, gt
    source_model = Path(str(audit.get("source_model", "")))
    base_method = str(audit.get("base_method_name", ""))
    render = source_model / "test" / base_method / "renders"
    gt = source_model / "test" / base_method / "gt"
    if render.is_dir() and gt.is_dir():
        return render, gt
    raise FileNotFoundError(f"cannot locate render/gt dirs for {model_dir}")


def best_view(v48_render: Path, v52_render: Path, gt_dir: Path, max_views: int) -> tuple[str, dict[str, Any], dict[str, float]]:
    names = sorted({p.name for p in v48_render.glob("*.png")} & {p.name for p in v52_render.glob("*.png")} & {p.name for p in gt_dir.glob("*.png")})
    if not names:
        raise RuntimeError(f"no common png views in {v48_render}, {v52_render}, {gt_dir}")
    candidates = names[: max(1, int(max_views))]
    best: tuple[float, str, dict[str, Any], dict[str, float]] | None = None
    for name in candidates:
        gt = imread(gt_dir / name)
        base = imread(v48_render / name)
        cand = imread(v52_render / name)
        crop = choose_crop(gt, base, cand, 360, 240)
        metrics = crop_metrics(gt, base, cand, crop)
        score = float(metrics["local_mae_delta"]) + 0.001 * float(metrics["local_dpsnr"])
        if best is None or score > best[0]:
            best = (score, name, crop, metrics)
    assert best is not None
    return best[1], best[2], best[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v52 capacity-policy qualitative panels for cap-hit scenes.")
    parser.add_argument("--v48_summary", default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v48_autosupport_autocap_guarded_v42calib_full9_summary.json")
    parser.add_argument("--v52_selected_root", default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9")
    parser.add_argument("--scenes", default="counter,kitchen,bonsai")
    parser.add_argument("--max_views", type=int, default=12)
    parser.add_argument("--cell_w", type=int, default=320)
    parser.add_argument("--cell_h", type=int, default=220)
    parser.add_argument("--out", default="assets/spcarnet_v52_capacity_policy_cap_hit_panel.png")
    parser.add_argument("--manifest", default="assets/spcarnet_v52_capacity_policy_cap_hit_panel_manifest.json")
    args = parser.parse_args()

    v48 = read_json(Path(args.v48_summary))
    v48_rows = {row["scene"]: row for row in v48["rows"]}
    selected_root = Path(args.v52_selected_root)
    rows: list[Image.Image] = []
    manifest: list[dict[str, Any]] = []
    cell_size = (int(args.cell_w), int(args.cell_h))
    for scene in [item.strip() for item in str(args.scenes).split(",") if item.strip()]:
        v48_model = Path(v48_rows[scene]["method_dir"])
        v48_render, v48_gt = method_dirs_from_audit(v48_model)
        v52_render = selected_root / scene / "renders"
        v52_gt = selected_root / scene / "gt"
        gt_dir = v52_gt if v52_gt.is_dir() else v48_gt
        view, crop, metrics = best_view(v48_render, v52_render, gt_dir, int(args.max_views))
        gt = imread(gt_dir / view)
        base = imread(v48_render / view)
        cand = imread(v52_render / view)
        heat = heatmap(gt, base, cand, crop).resize(cell_size, Image.Resampling.LANCZOS)
        row = hcat(
            [
                label_cell(to_pil(gt, crop, cell_size), f"{scene} GT", view, cell_size),
                label_cell(to_pil(base, crop, cell_size), "v48 auto-policy", f"PSNR {metrics['v48_local_psnr']:.2f}", cell_size),
                label_cell(to_pil(cand, crop, cell_size), "v52 capacity policy", f"dPSNR {metrics['local_dpsnr']:+.3f}", cell_size),
                label_cell(heat, "green better", f"dMAE {metrics['local_mae_delta']:+.5f}", cell_size),
            ],
            gap=8,
        )
        rows.append(row)
        manifest.append(
            {
                "scene": scene,
                "view": view,
                "crop": crop,
                "metrics": metrics,
                "v48_render": str(v48_render / view),
                "v52_render": str(v52_render / view),
                "gt": str(gt_dir / view),
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vcat(rows, gap=18).save(out)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "manifest": str(manifest_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
