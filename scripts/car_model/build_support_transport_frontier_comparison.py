#!/usr/bin/env python3
"""Build LPIPS and qualitative panels for support-transport frontier methods."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


SCENES = ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"]


def _parse_method_specs(specs: list[str]) -> list[dict[str, Any]]:
    methods = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"bad --method spec `{spec}`; expected name=path")
        name, raw_path = spec.split("=", 1)
        methods.append({"name": name.strip(), "root": Path(raw_path).expanduser().resolve()})
    if not methods:
        raise ValueError("at least one --method is required")
    return methods


def _method_scene_dirs(root: Path, scene: str, clean_iteration: int) -> tuple[Path, Path]:
    direct = root / scene
    if (direct / "renders").is_dir() and (direct / "gt").is_dir():
        return direct / "renders", direct / "gt"
    clean = root / scene / "test" / f"ours_{int(clean_iteration)}"
    if (clean / "renders").is_dir() and (clean / "gt").is_dir():
        return clean / "renders", clean / "gt"
    if (root / "renders").is_dir() and (root / "gt").is_dir():
        return root / "renders", root / "gt"
    raise FileNotFoundError(f"could not find renders/gt for scene `{scene}` under {root}")


def _common_frames(methods: list[dict[str, Any]], scene: str, clean_iteration: int) -> list[str]:
    frame_sets = []
    for method in methods:
        renders_dir, gt_dir = _method_scene_dirs(method["root"], scene, clean_iteration)
        render_names = {p.name for p in renders_dir.glob("*.png")}
        gt_names = {p.name for p in gt_dir.glob("*.png")}
        frame_sets.append(render_names & gt_names)
    if not frame_sets:
        return []
    common = set.intersection(*frame_sets)
    return sorted(common)


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    scale = min(float(max_side) / max(img.width, img.height), 1.0)
    if scale >= 1.0:
        return img
    new_size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
    return img.resize(new_size, Image.BICUBIC)


def _image_to_np(img: Image.Image, *, size: tuple[int, int] | None = None) -> np.ndarray:
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _np_to_lpips_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor * 2.0 - 1.0


def _np_to_unit_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1.0e-12:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def _mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def _lpips_value(model: Any, pred: np.ndarray, gt: np.ndarray, device: torch.device) -> float:
    with torch.no_grad():
        value = model(_np_to_lpips_tensor(pred, device), _np_to_lpips_tensor(gt, device))
    return float(value.detach().cpu().reshape(-1)[0].item())


def _dists_value(model: Any, pred: np.ndarray, gt: np.ndarray, device: torch.device) -> float:
    with torch.no_grad():
        value = model(_np_to_unit_tensor(pred, device), _np_to_unit_tensor(gt, device))
    return float(value.detach().cpu().reshape(-1)[0].item())


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _draw_label(img: Image.Image, label: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    h = 26
    draw.rectangle((0, 0, out.width, h), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255))
    return out


def _stack_horizontal(images: list[Image.Image]) -> Image.Image:
    height = max(im.height for im in images)
    width = sum(im.width for im in images)
    out = Image.new("RGB", (width, height), (20, 20, 20))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def _stack_vertical(images: list[Image.Image]) -> Image.Image:
    width = max(im.width for im in images)
    height = sum(im.height for im in images)
    out = Image.new("RGB", (width, height), (20, 20, 20))
    y = 0
    for im in images:
        out.paste(im, (0, y))
        y += im.height
    return out


def _error_map(pred: np.ndarray, gt: np.ndarray, *, gain: float) -> Image.Image:
    err = np.mean(np.abs(pred - gt), axis=2)
    err = np.clip(err * float(gain), 0.0, 1.0)
    rgb = np.stack([err, 0.25 * err, 1.0 - err], axis=2)
    return Image.fromarray(np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0))


def _best_crop_box(score: np.ndarray, crop_size: int) -> tuple[int, int, int, int]:
    h, w = score.shape
    size = max(32, min(int(crop_size), h, w))
    if h <= size or w <= size:
        return 0, 0, w, h
    integral = np.pad(score, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    sums = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(x), int(y), int(x + size), int(y + size)


def _make_panel(
    *,
    scene: str,
    frame: str,
    methods: list[dict[str, Any]],
    clean_iteration: int,
    out_path: Path,
    max_panel_side: int,
    crop_size: int,
    error_gain: float,
) -> dict[str, Any]:
    first_renders, first_gt = _method_scene_dirs(methods[0]["root"], scene, clean_iteration)
    gt_img = _resize_max_side(_load_rgb(first_gt / frame), max_panel_side)
    size = gt_img.size
    gt = _image_to_np(gt_img)

    rendered_images: list[tuple[str, Image.Image, np.ndarray]] = []
    error_scores = []
    for method in methods:
        renders_dir, _ = _method_scene_dirs(method["root"], scene, clean_iteration)
        img = _resize_max_side(_load_rgb(renders_dir / frame), max_panel_side)
        arr = _image_to_np(img, size=size)
        rendered_images.append((method["name"], Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255.0)), arr))
        error_scores.append(np.mean(np.abs(arr - gt), axis=2))
    crop_score = np.max(np.stack(error_scores, axis=0), axis=0)
    box = _best_crop_box(crop_score, crop_size)

    full_row = [_draw_label(gt_img, "GT")]
    crop_row = [_draw_label(gt_img.crop(box).resize((crop_size, crop_size), Image.BICUBIC), "GT crop")]
    err_row = []
    method_metrics = {}
    for name, img, arr in rendered_images:
        full_row.append(_draw_label(img, name))
        crop = img.crop(box).resize((crop_size, crop_size), Image.BICUBIC)
        crop_row.append(_draw_label(crop, f"{name} crop"))
        err = _error_map(arr, gt, gain=error_gain)
        err_row.append(_draw_label(err, f"{name} error"))
        method_metrics[name] = {"mae": _mae(arr, gt), "psnr": _psnr(arr, gt)}
    panel = _stack_vertical([_stack_horizontal(full_row), _stack_horizontal(crop_row), _stack_horizontal(err_row)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)
    return {
        "scene": scene,
        "frame": frame,
        "panel": str(out_path),
        "crop_box_xyxy": list(box),
        "method_metrics": method_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True, help="Method spec name=path. Can repeat.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--clean_iteration", type=int, default=26000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips_max_side", type=int, default=512)
    parser.add_argument("--panel_max_side", type=int, default=640)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--error_gain", type=float, default=8.0)
    parser.add_argument("--max_panels_per_scene", type=int, default=2)
    parser.add_argument("--panel_scenes", default="garden,flowers,bicycle")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet-transport-diagnostics")
    parser.add_argument("--wandb_run_name", default="frontier-lpips-qualitative")
    args = parser.parse_args()

    import lpips  # Imported lazily because this script is optional evidence tooling.

    try:
        import piq
    except ImportError:
        piq = None

    methods = _parse_method_specs(args.method)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    panel_scenes = {item.strip() for item in str(args.panel_scenes).split(",") if item.strip()}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for param in lpips_model.parameters():
        param.requires_grad_(False)
    dists_model = None
    dists_status = "not_computed_missing_piq"
    if piq is not None:
        dists_model = piq.DISTS(reduction="mean").to(device).eval()
        for param in dists_model.parameters():
            param.requires_grad_(False)
        dists_status = "computed_piq_DISTS_reduction_mean"

    per_scene: dict[str, Any] = {}
    per_view_rows: list[dict[str, Any]] = []
    selected_panel_rows: list[dict[str, Any]] = []
    for scene in scenes:
        frames = _common_frames(methods, scene, int(args.clean_iteration))
        if not frames:
            per_scene[scene] = {"error": "no common frames"}
            continue
        scene_rows = []
        for frame in frames:
            first_renders, first_gt = _method_scene_dirs(methods[0]["root"], scene, int(args.clean_iteration))
            gt_img = _resize_max_side(_load_rgb(first_gt / frame), int(args.lpips_max_side))
            gt = _image_to_np(gt_img)
            size = gt_img.size
            row = {"scene": scene, "frame": frame, "methods": {}}
            for method in methods:
                renders_dir, _ = _method_scene_dirs(method["root"], scene, int(args.clean_iteration))
                pred_img = _resize_max_side(_load_rgb(renders_dir / frame), int(args.lpips_max_side))
                pred = _image_to_np(pred_img, size=size)
                metrics = {
                    "psnr": _psnr(pred, gt),
                    "mae": _mae(pred, gt),
                    "lpips": _lpips_value(lpips_model, pred, gt, device),
                }
                if dists_model is not None:
                    metrics["dists"] = _dists_value(dists_model, pred, gt, device)
                row["methods"][method["name"]] = metrics
            scene_rows.append(row)
            per_view_rows.append(row)

        summary_methods = {}
        for method in methods:
            name = method["name"]
            summary_methods[name] = {
                "psnr": _mean([row["methods"][name]["psnr"] for row in scene_rows]),
                "mae": _mean([row["methods"][name]["mae"] for row in scene_rows]),
                "lpips": _mean([row["methods"][name]["lpips"] for row in scene_rows]),
            }
            if dists_model is not None:
                summary_methods[name]["dists"] = _mean([row["methods"][name]["dists"] for row in scene_rows])
        ref_name = methods[0]["name"]
        for name, metrics in summary_methods.items():
            metrics[f"delta_lpips_vs_{ref_name}"] = metrics["lpips"] - summary_methods[ref_name]["lpips"]
            metrics[f"delta_mae_vs_{ref_name}"] = metrics["mae"] - summary_methods[ref_name]["mae"]
            metrics[f"delta_psnr_vs_{ref_name}"] = metrics["psnr"] - summary_methods[ref_name]["psnr"]
            if "dists" in metrics:
                metrics[f"delta_dists_vs_{ref_name}"] = metrics["dists"] - summary_methods[ref_name]["dists"]
        per_scene[scene] = {"views": len(scene_rows), "methods": summary_methods}

        if scene in panel_scenes:
            non_ref_names = [method["name"] for method in methods[1:]]
            ranked = sorted(
                scene_rows,
                key=lambda row: max(
                    abs(row["methods"][name]["mae"] - row["methods"][ref_name]["mae"]) for name in non_ref_names
                )
                if non_ref_names
                else 0.0,
                reverse=True,
            )
            for row in ranked[: max(0, int(args.max_panels_per_scene))]:
                panel_path = out_dir / "panels" / scene / f"{row['frame'].removesuffix('.png')}_frontier_panel.png"
                selected_panel_rows.append(
                    _make_panel(
                        scene=scene,
                        frame=row["frame"],
                        methods=methods,
                        clean_iteration=int(args.clean_iteration),
                        out_path=panel_path,
                        max_panel_side=int(args.panel_max_side),
                        crop_size=int(args.crop_size),
                        error_gain=float(args.error_gain),
                    )
                )

    aggregate = {}
    for method in methods:
        name = method["name"]
        scene_metrics = [payload["methods"][name] for payload in per_scene.values() if "methods" in payload]
        aggregate[name] = {
            "scene_count": len(scene_metrics),
            "macro_psnr": _mean([m["psnr"] for m in scene_metrics]),
            "macro_mae": _mean([m["mae"] for m in scene_metrics]),
            "macro_lpips": _mean([m["lpips"] for m in scene_metrics]),
        }
        if dists_model is not None:
            aggregate[name]["macro_dists"] = _mean([m["dists"] for m in scene_metrics])
    ref_name = methods[0]["name"]
    for name, metrics in aggregate.items():
        metrics[f"delta_lpips_vs_{ref_name}"] = metrics["macro_lpips"] - aggregate[ref_name]["macro_lpips"]
        metrics[f"delta_mae_vs_{ref_name}"] = metrics["macro_mae"] - aggregate[ref_name]["macro_mae"]
        metrics[f"delta_psnr_vs_{ref_name}"] = metrics["macro_psnr"] - aggregate[ref_name]["macro_psnr"]
        if "macro_dists" in metrics:
            metrics[f"delta_dists_vs_{ref_name}"] = metrics["macro_dists"] - aggregate[ref_name]["macro_dists"]

    payload = {
        "methods": [{"name": method["name"], "root": str(method["root"])} for method in methods],
        "clean_iteration": int(args.clean_iteration),
        "lpips_net": "alex",
        "lpips_max_side": int(args.lpips_max_side),
        "dists_status": dists_status,
        "aggregate": aggregate,
        "per_scene": per_scene,
        "panels": selected_panel_rows,
        "per_view": per_view_rows,
    }
    json_path = out_dir / "frontier_lpips_qualitative_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Support-Transport Frontier LPIPS and Qualitative Evidence",
        "",
        f"LPIPS net: `alex`, max side: `{int(args.lpips_max_side)}`.",
        f"DISTS status: `{dists_status}`.",
        "",
        "## Aggregate",
        "",
    ]
    if dists_model is None:
        md_lines.extend(
            [
                "| method | scenes | PSNR | MAE | LPIPS | dPSNR vs ref | dMAE vs ref | dLPIPS vs ref |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        md_lines.extend(
            [
                "| method | scenes | PSNR | MAE | LPIPS | DISTS | dPSNR vs ref | dMAE vs ref | dLPIPS vs ref | dDISTS vs ref |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    for name, metrics in aggregate.items():
        if dists_model is None:
            md_lines.append(
                "| {name} | {scenes} | {psnr:.6f} | {mae:.6f} | {lpips:.6f} | {dpsnr:+.6f} | {dmae:+.6f} | {dlpips:+.6f} |".format(
                    name=name,
                    scenes=int(metrics["scene_count"]),
                    psnr=float(metrics["macro_psnr"]),
                    mae=float(metrics["macro_mae"]),
                    lpips=float(metrics["macro_lpips"]),
                    dpsnr=float(metrics[f"delta_psnr_vs_{ref_name}"]),
                    dmae=float(metrics[f"delta_mae_vs_{ref_name}"]),
                    dlpips=float(metrics[f"delta_lpips_vs_{ref_name}"]),
                )
            )
        else:
            md_lines.append(
                "| {name} | {scenes} | {psnr:.6f} | {mae:.6f} | {lpips:.6f} | {dists:.6f} | {dpsnr:+.6f} | {dmae:+.6f} | {dlpips:+.6f} | {ddists:+.6f} |".format(
                    name=name,
                    scenes=int(metrics["scene_count"]),
                    psnr=float(metrics["macro_psnr"]),
                    mae=float(metrics["macro_mae"]),
                    lpips=float(metrics["macro_lpips"]),
                    dists=float(metrics["macro_dists"]),
                    dpsnr=float(metrics[f"delta_psnr_vs_{ref_name}"]),
                    dmae=float(metrics[f"delta_mae_vs_{ref_name}"]),
                    dlpips=float(metrics[f"delta_lpips_vs_{ref_name}"]),
                    ddists=float(metrics[f"delta_dists_vs_{ref_name}"]),
                )
            )
    md_lines += ["", "## Selected Panels", ""]
    for row in selected_panel_rows:
        rel = Path(row["panel"]).relative_to(out_dir)
        md_lines.append(f"- `{row['scene']}/{row['frame']}`: ![]({rel.as_posix()})")
    (out_dir / "frontier_lpips_qualitative_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    if bool(args.enable_wandb):
        import wandb

        run = wandb.init(project=str(args.wandb_project), name=str(args.wandb_run_name), dir=str(out_dir))
        log_payload: dict[str, float] = {}
        for name, metrics in aggregate.items():
            prefix = f"frontier/{name}"
            for key, value in metrics.items():
                log_payload[f"{prefix}/{key}"] = float(value)
        wandb.log(log_payload)
        panel_log = {}
        for row in selected_panel_rows[:12]:
            panel_log[f"panel/{row['scene']}_{row['frame'].removesuffix('.png')}"] = wandb.Image(row["panel"])
        if panel_log:
            wandb.log(panel_log)
        run.finish()
    print(json_path)
    print(out_dir / "frontier_lpips_qualitative_summary.md")


if __name__ == "__main__":
    main()
