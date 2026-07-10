#!/usr/bin/env python3
"""Assemble ECR qualitative grids from banked PNG dumps.

This script intentionally does not render anything. It reads existing PNG dumps
under /data/peilincai/gems_stage1, applies the frozen view-selection and crop
rules from the paper audit, and writes montage PNGs plus a merge-safe manifest.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT_DEFAULT = Path("/data/peilincai/gems_stage1")
OUT_DIR_DEFAULT = REPO_ROOT / "RESULTS" / "figures" / "ecr_qual"

SCENES_DEFAULT = ("garden", "bicycle", "bonsai", "treehill", "kitchen")
VIEW_RULES = ("best", "median", "failure")
FINAL_REQUIRED_PLANES = (
    "gt.png",
    "base.png",
    "final.png",
    "err_base.png",
    "err_final.png",
    "conf.png",
    "beta.png",
)

CELL_WIDTH = 440
LABEL_WIDTH = 152
LABEL_STRIP = 22
GAP = 4
CROP_STRIDE = 32

FINAL_ROWS = {
    "gt": ("GT", "gt.png"),
    "base": ("base", "base.png"),
    "final": ("ECR final", "final.png"),
    "beta": ("beta.png", "beta.png"),
    "conf": ("conf.png", "conf.png"),
}


class SceneSkip(Exception):
    """Expected input-data skip for a scene."""

    def __init__(self, reason: str, details: Any | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _metric_paths(stage_root: Path, scene: str) -> tuple[Path, Path]:
    eval_root = stage_root / "eval"
    final_path = eval_root / f"l4_{scene}_cleanfixed30k_routed_v1" / "metrics.json"
    base_path = eval_root / f"{scene}_cleanfixed30k_v1" / "metrics.json"
    return final_path, base_path


def _per_view_arrays(metrics: dict[str, Any], path: Path) -> tuple[list[str], np.ndarray]:
    try:
        per_view = metrics["rendering"]["per_view"]
        names = list(per_view["image_names"])
        psnr = np.asarray(per_view["psnr"], dtype=np.float64)
    except KeyError as exc:
        raise SceneSkip(f"metrics missing rendering.per_view field: {path}", str(exc))
    if len(names) != len(psnr):
        raise SceneSkip(
            f"metrics image_names/psnr length mismatch: {path}",
            {"n_image_names": len(names), "n_psnr": int(len(psnr))},
        )
    if not names:
        raise SceneSkip(f"metrics has no per-view entries: {path}")
    return names, psnr


def select_views(stage_root: Path, scene: str) -> dict[str, Any]:
    """Apply the frozen selection rule exactly against banked PSNR arrays."""
    final_path, base_path = _metric_paths(stage_root, scene)
    if not final_path.is_file():
        raise SceneSkip(f"missing FINAL metrics.json: {final_path}")
    if not base_path.is_file():
        raise SceneSkip(f"missing BASE metrics.json: {base_path}")

    final_metrics = _load_json(final_path)
    base_metrics = _load_json(base_path)
    image_names, final_psnr = _per_view_arrays(final_metrics, final_path)
    base_names, base_psnr = _per_view_arrays(base_metrics, base_path)
    if image_names != base_names:
        raise SceneSkip(
            "FINAL/BASE per-view image_names order mismatch",
            {"final_metrics": str(final_path), "base_metrics": str(base_path)},
        )

    # Tie behavior intentionally follows numpy's default arg* / argsort calls.
    delta = final_psnr - base_psnr
    indices = {
        "best": int(np.argmax(final_psnr)),
        "median": int(np.argsort(final_psnr)[(len(final_psnr) - 1) // 2]),
        "failure": int(np.argmin(delta)),
    }
    expressions = {
        "best": "image_names[argmax(final_row_psnr)]",
        "median": "image_names[argsort(final_row_psnr)[(n-1)//2]]",
        "failure": "image_names[argmin(final_row_psnr - base_row_psnr)]",
    }
    per_view_values = [
        {
            "index": int(i),
            "image_name": name,
            "final_psnr": float(final_psnr[i]),
            "base_psnr": float(base_psnr[i]),
            "delta_psnr": float(delta[i]),
        }
        for i, name in enumerate(image_names)
    ]

    selected: dict[str, Any] = {}
    invocations = []
    for rule in VIEW_RULES:
        idx = indices[rule]
        rec = {
            "rule": rule,
            "expression": expressions[rule],
            "index": int(idx),
            "image_name": image_names[idx],
            "final_psnr": float(final_psnr[idx]),
            "base_psnr": float(base_psnr[idx]),
            "delta_psnr": float(delta[idx]),
        }
        selected[rule] = rec
        invocations.append(
            {
                "rule": rule,
                "expression": expressions[rule],
                "inputs": {
                    "final_metrics_json": str(final_path),
                    "base_metrics_json": str(base_path),
                    "image_names_field": "rendering.per_view.image_names",
                    "final_psnr_field": "rendering.per_view.psnr",
                    "base_psnr_field": "rendering.per_view.psnr",
                    "n_views": len(image_names),
                    "base_final_order_asserted_equal": True,
                },
                "selected": rec,
            }
        )
        print(
            "[ecr_qual][VIEW-SELECT] "
            f"scene={scene} rule={rule} expr='{expressions[rule]}' "
            f"view={image_names[idx]} final_psnr={final_psnr[idx]:.6f} "
            f"base_psnr={base_psnr[idx]:.6f} delta={delta[idx]:.6f}"
        )

    return {
        "final_metrics_json": str(final_path),
        "base_metrics_json": str(base_path),
        "per_view_values": per_view_values,
        "invocations": invocations,
        "selected": selected,
    }


def _final_scene_dir(stage_root: Path, scene: str) -> Path:
    return stage_root / "analysis" / "quals" / f"{scene}_final"


def _pj_scene_dir(stage_root: Path, scene: str) -> Path:
    return stage_root / "analysis" / "quals" / f"{scene}_pj2026"


def validate_final_dumps(stage_root: Path, scene: str, selected: dict[str, Any]) -> None:
    scene_dir = _final_scene_dir(stage_root, scene)
    summary_path = scene_dir / "summary.json"
    if not summary_path.is_file():
        raise SceneSkip(f"missing summary.json: {summary_path}")

    missing = []
    for rule in VIEW_RULES:
        view = selected[rule]["image_name"]
        view_dir = scene_dir / view
        for plane in FINAL_REQUIRED_PLANES:
            path = view_dir / plane
            if not path.is_file():
                missing.append(
                    {
                        "rule": rule,
                        "image_name": view,
                        "plane": plane,
                        "path": str(path),
                    }
                )
    if missing:
        raise SceneSkip("missing required selected-view final dump plane", missing)


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _window_positions(limit: int, window: int, stride: int) -> list[int]:
    last = max(0, limit - window)
    vals = list(range(0, last + 1, stride))
    if not vals or vals[-1] != last:
        vals.append(last)
    return vals


def _best_error_window(err: np.ndarray, want: str) -> tuple[int, int, int, int, float]:
    h_img, w_img = err.shape
    h_win = max(1, h_img // 2)
    w_win = max(1, w_img // 2)
    integral = np.zeros((h_img + 1, w_img + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(err, axis=0), axis=1)

    best_val: float | None = None
    best_xy = (0, 0)
    for y0 in _window_positions(h_img, h_win, CROP_STRIDE):
        for x0 in _window_positions(w_img, w_win, CROP_STRIDE):
            x1 = x0 + w_win
            y1 = y0 + h_win
            score = (
                integral[y1, x1]
                - integral[y0, x1]
                - integral[y1, x0]
                + integral[y0, x0]
            )
            if best_val is None:
                best_val = float(score)
                best_xy = (x0, y0)
            elif want == "min" and score < best_val:
                best_val = float(score)
                best_xy = (x0, y0)
            elif want == "max" and score > best_val:
                best_val = float(score)
                best_xy = (x0, y0)

    x0, y0 = best_xy
    x1 = min(w_img, x0 + w_win)
    y1 = min(h_img, y0 + h_win)
    mean_error = float(best_val / max(1, (x1 - x0) * (y1 - y0)))
    return x0, y0, x1, y1, mean_error


def compute_crop_windows(
    stage_root: Path, scene: str, selected: dict[str, Any]
) -> dict[str, Any]:
    crops: dict[str, Any] = {}
    scene_dir = _final_scene_dir(stage_root, scene)
    for rule in VIEW_RULES:
        view = selected[rule]["image_name"]
        view_dir = scene_dir / view
        final_path = view_dir / "final.png"
        gt_path = view_dir / "gt.png"
        final_arr = np.asarray(_open_rgb(final_path), dtype=np.float32)
        gt_arr = np.asarray(_open_rgb(gt_path), dtype=np.float32)
        if final_arr.shape != gt_arr.shape:
            raise SceneSkip(
                "final.png/gt.png shape mismatch for crop source",
                {
                    "rule": rule,
                    "image_name": view,
                    "final_png": str(final_path),
                    "gt_png": str(gt_path),
                    "final_shape": list(final_arr.shape),
                    "gt_shape": list(gt_arr.shape),
                },
            )

        h_img, w_img = final_arr.shape[:2]
        if rule == "median":
            x0, y0, x1, y1 = 0, 0, w_img, h_img
            mean_error = float(np.abs(final_arr - gt_arr).mean())
            crop_rule = "full_frame_no_crop"
        else:
            err = np.abs(final_arr - gt_arr).mean(axis=2)
            want = "min" if rule == "best" else "max"
            x0, y0, x1, y1, mean_error = _best_error_window(err, want)
            crop_rule = (
                "half_frame_minimize_error_stride32"
                if rule == "best"
                else "half_frame_maximize_error_stride32"
            )

        crops[rule] = {
            "image_name": view,
            "window_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "crop_rule": crop_rule,
            "stride_px": CROP_STRIDE if rule != "median" else None,
            "error_source_final_png": str(final_path),
            "error_source_gt_png": str(gt_path),
            "error_definition": "mean(abs(final.png - gt.png), axis=RGB) from raw PNG arrays",
            "mean_error_in_selected_window_raw_0_255": mean_error,
            "applied_identically_to_all_rows_in_column": True,
        }
        print(
            "[ecr_qual][CROP] "
            f"scene={scene} rule={rule} view={view} "
            f"window=({x0},{y0},{x1},{y1}) crop_rule={crop_rule}"
        )
    return crops


def _resize_to_cell_width(im: Image.Image) -> Image.Image:
    width, height = im.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {im.size}")
    new_h = max(1, int(round(height * CELL_WIDTH / width)))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return im.resize((CELL_WIDTH, new_h), resampling)


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str) -> None:
    draw.text(xy, text, fill=fill)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: str
) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    _draw_text(draw, (x0 + max(0, (x1 - x0 - tw) // 2), y0 + max(0, (y1 - y0 - th) // 2)), text, fill)


def _placeholder(size: tuple[int, int], text: str) -> Image.Image:
    im = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(190, 190, 190))
    _draw_centered_text(draw, (0, 0, size[0], size[1]), text, (90, 90, 90))
    return im


def pj_row_status(stage_root: Path, scene: str, selected: dict[str, Any]) -> dict[str, Any]:
    pj_dir = _pj_scene_dir(stage_root, scene)
    cells = []
    any_present = False
    for rule in VIEW_RULES:
        view = selected[rule]["image_name"]
        path = pj_dir / view / "final.png"
        present = path.is_file()
        any_present = any_present or present
        cells.append(
            {
                "rule": rule,
                "image_name": view,
                "path": str(path),
                "included": present,
                "omitted_reason": None if present else "missing PJ-2026 final.png for selected view",
            }
        )
    if any_present:
        return {
            "included": True,
            "omitted_reason": None,
            "cells": cells,
        }
    return {
        "included": False,
        "omitted_reason": "no PJ-2026 final.png exists for any selected view",
        "cells": cells,
    }


def build_grid(
    stage_root: Path,
    out_dir: Path,
    scene: str,
    selected: dict[str, Any],
    crops: dict[str, Any],
    rows_manifest: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    scene_dir = _final_scene_dir(stage_root, scene)
    pj_dir = _pj_scene_dir(stage_root, scene)
    out_path = out_dir / f"{scene}_ecr_qual_grid.png"

    row_order = ["gt", "base"]
    if rows_manifest["PJ-2026"]["included"]:
        row_order.append("pj2026")
    row_order += ["final", "beta", "conf"]

    placed_images: list[dict[str, Any]] = []
    omitted_cells: list[dict[str, Any]] = []
    row_cells: list[dict[str, Any]] = []
    for row_key in row_order:
        if row_key == "pj2026":
            label = "PJ-2026"
            filename = "final.png"
        else:
            label, filename = FINAL_ROWS[row_key]

        cells = []
        row_h = 1
        for rule in VIEW_RULES:
            view = selected[rule]["image_name"]
            x0, y0, x1, y1 = crops[rule]["window_xyxy"]
            crop_w = max(1, x1 - x0)
            crop_h = max(1, y1 - y0)
            expected_h = max(1, int(round(crop_h * CELL_WIDTH / crop_w)))

            if row_key == "pj2026":
                path = pj_dir / view / filename
            else:
                path = scene_dir / view / filename

            if path.is_file():
                im = _open_rgb(path).crop((x0, y0, x1, y1))
                cell = _resize_to_cell_width(im)
                source_path = str(path)
                placed_images.append(
                    {
                        "row": label,
                        "rule": rule,
                        "image_name": view,
                        "source_path": source_path,
                        "crop_window_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                        "cell_size_px": [int(cell.size[0]), int(cell.size[1])],
                    }
                )
            else:
                cell = _placeholder((CELL_WIDTH, expected_h), "omitted")
                source_path = None
                omitted_cells.append(
                    {
                        "row": label,
                        "rule": rule,
                        "image_name": view,
                        "path": str(path),
                        "reason": "missing source PNG",
                    }
                )
            row_h = max(row_h, cell.size[1])
            cells.append({"rule": rule, "image_name": view, "image": cell, "source_path": source_path})
        row_cells.append({"key": row_key, "label": label, "height": row_h, "cells": cells})

    grid_w = LABEL_WIDTH + len(VIEW_RULES) * CELL_WIDTH + (len(VIEW_RULES) - 1) * GAP
    grid_h = LABEL_STRIP + sum(r["height"] for r in row_cells) + (len(row_cells) - 1) * GAP
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, grid_w, LABEL_STRIP - 1), fill=(238, 238, 238))
    for col, rule in enumerate(VIEW_RULES):
        x = LABEL_WIDTH + col * (CELL_WIDTH + GAP)
        sel = selected[rule]
        header = f"{rule} {sel['image_name']} PSNR {sel['final_psnr']:.3f}"
        _draw_centered_text(draw, (x, 0, x + CELL_WIDTH, LABEL_STRIP), header, (0, 0, 0))

    y = LABEL_STRIP
    for row in row_cells:
        draw.rectangle((0, y, LABEL_WIDTH - 1, y + row["height"] - 1), fill=(248, 248, 248))
        _draw_centered_text(draw, (0, y, LABEL_WIDTH, y + row["height"]), row["label"], (0, 0, 0))
        for col, cell_rec in enumerate(row["cells"]):
            x = LABEL_WIDTH + col * (CELL_WIDTH + GAP)
            cell = cell_rec["image"]
            paste_y = y + (row["height"] - cell.size[1]) // 2
            canvas.paste(cell, (x, paste_y))
            draw.rectangle((x, paste_y, x + cell.size[0] - 1, paste_y + cell.size[1] - 1), outline=(210, 210, 210))
        y += row["height"] + GAP

    canvas.save(out_path)
    return out_path, placed_images, omitted_cells


def rows_manifest_for_scene(stage_root: Path, scene: str, selected: dict[str, Any]) -> dict[str, Any]:
    rows = {
        "GT": {"included": True, "source": f"{scene}_final/*/gt.png", "omitted_reason": None},
        "base": {"included": True, "source": f"{scene}_final/*/base.png", "omitted_reason": None},
        "ECR final": {"included": True, "source": f"{scene}_final/*/final.png", "omitted_reason": None},
        "beta.png": {"included": True, "source": f"{scene}_final/*/beta.png", "omitted_reason": None},
        "conf.png": {"included": True, "source": f"{scene}_final/*/conf.png", "omitted_reason": None},
    }
    rows["PJ-2026"] = pj_row_status(stage_root, scene, selected)
    return rows


def skipped_rows(reason: str) -> dict[str, Any]:
    return {
        "GT": {"included": False, "omitted_reason": f"scene skipped: {reason}"},
        "base": {"included": False, "omitted_reason": f"scene skipped: {reason}"},
        "PJ-2026": {"included": False, "omitted_reason": f"scene skipped: {reason}", "cells": []},
        "ECR final": {"included": False, "omitted_reason": f"scene skipped: {reason}"},
        "beta.png": {"included": False, "omitted_reason": f"scene skipped: {reason}"},
        "conf.png": {"included": False, "omitted_reason": f"scene skipped: {reason}"},
    }


def skipped_crop_windows(selected: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    if not selected:
        return {}
    return {
        rule: {
            "image_name": selected[rule]["image_name"],
            "window_xyxy": None,
            "omitted_reason": f"scene skipped before crop computation: {reason}",
        }
        for rule in VIEW_RULES
    }


def build_scene(stage_root: Path, out_dir: Path, scene: str) -> dict[str, Any]:
    print(f"[ecr_qual] scene={scene} start")
    entry: dict[str, Any] = {
        "scene": scene,
        "stage_root": str(stage_root),
        "output_dir": str(out_dir),
    }
    selection = select_views(stage_root, scene)
    selected = selection["selected"]
    entry["view_selection_inputs"] = {
        "final_metrics_json": selection["final_metrics_json"],
        "base_metrics_json": selection["base_metrics_json"],
        "per_view_values": selection["per_view_values"],
    }
    entry["view_selection_invocations"] = selection["invocations"]
    entry["selected_views"] = selected

    try:
        validate_final_dumps(stage_root, scene, selected)
    except SceneSkip as exc:
        entry.update(
            {
                "status": "skipped",
                "skip_reason": exc.reason,
                "skip_details": exc.details,
                "crop_windows": skipped_crop_windows(selected, exc.reason),
                "rows": skipped_rows(exc.reason),
                "grid_png": None,
                "placed_images": [],
                "omitted_cells": [],
            }
        )
        print(f"[ecr_qual][SKIP] scene={scene} reason={exc.reason}")
        return entry

    crops = compute_crop_windows(stage_root, scene, selected)
    rows = rows_manifest_for_scene(stage_root, scene, selected)
    out_path, placed_images, omitted_cells = build_grid(
        stage_root, out_dir, scene, selected, crops, rows
    )

    pj = rows["PJ-2026"]
    status = "produced" if pj["included"] and all(c["included"] for c in pj["cells"]) else "produced_with_omissions"
    entry.update(
        {
            "status": status,
            "skip_reason": None,
            "crop_windows": crops,
            "rows": rows,
            "grid_png": str(out_path),
            "placed_images": placed_images,
            "omitted_cells": omitted_cells,
        }
    )
    if status == "produced_with_omissions":
        print(f"[ecr_qual][DONE-PARTIAL] scene={scene} output={out_path}")
    else:
        print(f"[ecr_qual][DONE] scene={scene} output={out_path}")
    return entry


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "scenes": {}}
    data = _load_json(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"manifest must be a JSON object: {path}")
    if "scenes" not in data:
        data["scenes"] = {}
    if not isinstance(data["scenes"], dict):
        raise RuntimeError(f"manifest.scenes must be a JSON object: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=STAGE_ROOT_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--scenes", nargs="+", default=list(SCENES_DEFAULT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    for scene in args.scenes:
        try:
            manifest["scenes"][scene] = build_scene(args.stage_root, args.out_dir, scene)
        except SceneSkip as exc:
            manifest["scenes"][scene] = {
                "scene": scene,
                "stage_root": str(args.stage_root),
                "output_dir": str(args.out_dir),
                "status": "skipped",
                "skip_reason": exc.reason,
                "skip_details": exc.details,
                "crop_windows": skipped_crop_windows(None, exc.reason),
                "rows": skipped_rows(exc.reason),
                "grid_png": None,
                "placed_images": [],
                "omitted_cells": [],
            }
            print(f"[ecr_qual][SKIP] scene={scene} reason={exc.reason}")

    manifest["generated_by"] = str(Path(__file__).resolve())
    manifest["schema"] = "ecr_qual_grids.v1"
    _write_json_atomic(manifest_path, manifest)
    print(f"[ecr_qual] manifest={manifest_path}")


if __name__ == "__main__":
    main()
