#!/usr/bin/env python3
"""Assemble edit-aware ECR figure grids from banked panel strips."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path("/data/peilincai/gems_stage1/analysis/edit_aware")
OUT_DIR = REPO_ROOT / "RESULTS" / "figures" / "edit_aware"

SCENES = ("garden", "toy_parking", "garden_recolor")
TILE_WIDTH_OUT = 420
LABEL_STRIP_H = 20

COLUMNS_5 = [
    ("ORIG", "original ECR"),
    ("C1", "edited base (C1)"),
    ("C2", "stale cache (C2)"),
    ("C4", "full rebuild (C4)"),
    ("C5", "local invalidation (C5, ours)"),
]
COLUMNS_4 = [
    ("ORIG", "original ECR"),
    ("C1", "edited base (C1)"),
    ("C2", "stale cache (C2)"),
    ("C5", "local invalidation (C5, ours)"),
]

METHOD_SYNONYMS = {
    "ORIG": (("orig", "ecr"),),
    "C1": (("edited", "base"), ("editedbase",)),
    "C2": (("stale",), ("cache",)),
    "C4": (("full", "rebuild"), ("rebuild",)),
    "C5": (("ours",), ("local", "invalidation")),
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def norm_key(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value)


def find_method_key(per_method: dict[str, Any], code: str) -> str | None:
    code_upper = code.upper()
    for key in per_method:
        if key.upper().startswith(code_upper):
            return key

    normalized = {key: norm_key(key) for key in per_method}
    for tokens in METHOD_SYNONYMS.get(code_upper, ()):
        for key, key_norm in normalized.items():
            if all(token in key_norm for token in tokens):
                return key
    return None


def leak_values(per_method: dict[str, Any], columns: list[tuple[str, str]]) -> tuple[dict[str, float | None], dict[str, str | None], dict[str, str | None]]:
    raw: dict[str, float | None] = {}
    formatted: dict[str, str | None] = {}
    method_keys: dict[str, str | None] = {}
    for code, label in columns:
        if code == "ORIG":
            raw[label] = None
            formatted[label] = None
            method_keys[label] = find_method_key(per_method, code)
            continue
        method_key = find_method_key(per_method, code)
        method_keys[label] = method_key
        value = None
        if method_key is not None:
            method_record = per_method.get(method_key)
            if isinstance(method_record, dict) and "leak_R" in method_record:
                value = float(method_record["leak_R"])
        raw[label] = value
        formatted[label] = None if value is None else f"leak={value:.3g}"
    return raw, formatted, method_keys


def scene_tile_count(scene: str) -> int:
    return 4 if "recolor" in scene else 5


def columns_for_count(tile_count: int) -> list[tuple[str, str]]:
    if tile_count == 5:
        return COLUMNS_5
    if tile_count == 4:
        return COLUMNS_4
    raise ValueError(f"unsupported tile count: {tile_count}")


def split_strip(path: Path, tile_count: int) -> tuple[list[Image.Image], tuple[int, int]]:
    with Image.open(path) as im:
        strip = im.convert("RGB")
    width, height = strip.size
    if width % tile_count != 0:
        raise ValueError(f"{path}: width {width} is not divisible by {tile_count} tiles")
    tile_width = width // tile_count
    tiles = [
        strip.crop((idx * tile_width, 0, (idx + 1) * tile_width, height))
        for idx in range(tile_count)
    ]
    return tiles, (tile_width, height)


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    scale = width / image.width
    height = max(1, int(round(image.height * scale)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def candidate_positions(total: int, window: int) -> list[int]:
    last = max(0, total - window)
    vals = [int(round(x)) for x in np.linspace(0, last, 8)]
    return sorted(set(vals))


def best_zoom_window(tiles: list[Image.Image], tile_count: int) -> list[int]:
    if tile_count == 5:
        left_idx, right_idx = 1, 3  # C1 edited-base vs C4 full-rebuild.
    elif tile_count == 4:
        left_idx, right_idx = 1, 2  # C1 edited-base vs C2 stale-cache.
    else:
        raise ValueError(f"unsupported tile count: {tile_count}")

    left = np.asarray(tiles[left_idx], dtype=np.int16)
    right = np.asarray(tiles[right_idx], dtype=np.int16)
    diff = np.abs(left - right).mean(axis=2)
    height, width = diff.shape
    crop_h = max(1, height // 2)
    crop_w = max(1, width // 2)

    best_score = -1.0
    best_xy = (0, 0)
    for y0 in candidate_positions(height, crop_h):
        for x0 in candidate_positions(width, crop_w):
            score = float(diff[y0 : y0 + crop_h, x0 : x0 + crop_w].mean())
            if score > best_score:
                best_score = score
                best_xy = (x0, y0)

    x0, y0 = best_xy
    return [int(x0), int(y0), int(x0 + crop_w), int(y0 + crop_h)]


def draw_column_labels(draw: ImageDraw.ImageDraw, y0: int, labels: list[str], leak_labels: dict[str, str | None], font: ImageFont.ImageFont) -> None:
    for idx, label in enumerate(labels):
        x0 = idx * TILE_WIDTH_OUT
        x1 = x0 + TILE_WIDTH_OUT
        leak_label = leak_labels.get(label)
        lines = [label] if leak_label is None else [label, leak_label]
        if len(lines) == 1:
            ys = [y0 + 6]
        else:
            ys = [y0 + 1, y0 + 10]
        for text, text_y in zip(lines, ys):
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_x = x0 + max(0, (x1 - x0 - text_w) // 2)
            draw.text((text_x, text_y), text, fill="black", font=font)


def build_grid_image(tiles: list[Image.Image], zoom_window: list[int], labels: list[str], leak_labels: dict[str, str | None]) -> Image.Image:
    font = ImageFont.load_default()
    full_tiles = [resize_to_width(tile, TILE_WIDTH_OUT) for tile in tiles]
    crops = [resize_to_width(tile.crop(tuple(zoom_window)), TILE_WIDTH_OUT) for tile in tiles]

    full_h = max(tile.height for tile in full_tiles)
    crop_h = max(tile.height for tile in crops)
    width = TILE_WIDTH_OUT * len(tiles)
    height = LABEL_STRIP_H + full_h + LABEL_STRIP_H + crop_h
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)

    y = 0
    draw_column_labels(draw, y, labels, leak_labels, font)
    y += LABEL_STRIP_H
    for idx, tile in enumerate(full_tiles):
        grid.paste(tile, (idx * TILE_WIDTH_OUT, y))
    y += full_h
    draw_column_labels(draw, y, labels, leak_labels, font)
    y += LABEL_STRIP_H
    for idx, tile in enumerate(crops):
        grid.paste(tile, (idx * TILE_WIDTH_OUT, y))
    return grid


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"manifest is not a JSON object: {path}")
    return data


def process_scene(scene: str, manifest: dict[str, Any]) -> tuple[list[Path], list[str]]:
    scene_dir = INPUT_ROOT / scene
    panels_dir = scene_dir / "panels"
    eval_path = scene_dir / "edit_eval.json"
    skips: list[str] = []
    if not scene_dir.is_dir():
        return [], [f"{scene}: missing scene directory {scene_dir}"]
    if not panels_dir.is_dir():
        return [], [f"{scene}: missing panels directory {panels_dir}"]
    if not eval_path.is_file():
        return [], [f"{scene}: missing edit_eval.json {eval_path}"]

    panel_paths = sorted(panels_dir.glob("*.png"))
    if not panel_paths:
        return [], [f"{scene}: no panel PNGs in {panels_dir}"]

    eval_data = load_json(eval_path)
    per_method = eval_data.get("per_method")
    if not isinstance(per_method, dict):
        return [], [f"{scene}: edit_eval.json missing object field per_method"]

    tile_count = scene_tile_count(scene)
    columns = columns_for_count(tile_count)
    labels = [label for _, label in columns]
    leak_raw, leak_formatted, method_keys = leak_values(per_method, columns)
    produced: list[Path] = []

    for panel_path in panel_paths:
        try:
            tiles, tile_size = split_strip(panel_path, tile_count)
            zoom_window = best_zoom_window(tiles, tile_count)
            grid = build_grid_image(tiles, zoom_window, labels, leak_formatted)
            out_path = OUT_DIR / f"{scene}__{panel_path.stem}_grid.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            grid.save(out_path)

            grid_key = f"{scene}__{panel_path.stem}"
            manifest.setdefault("grids", {})[grid_key] = {
                "source_panel_path": str(panel_path),
                "output_path": str(out_path),
                "scene": scene,
                "view": panel_path.stem,
                "tile_count": tile_count,
                "source_tile_size": [int(tile_size[0]), int(tile_size[1])],
                "tile_order": labels,
                "zoom_window": zoom_window,
                "leak_R": leak_raw,
                "leak_R_formatted": leak_formatted,
                "method_keys": method_keys,
                "edit_eval_json": str(eval_path),
            }
            produced.append(out_path)
            print(f"[edit_grids] wrote {out_path}")
        except Exception as exc:
            skips.append(f"{scene}/{panel_path.name}: {exc}")

    return produced, skips


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    manifest = load_manifest(manifest_path)
    all_produced: list[Path] = []
    all_skips: list[str] = []

    for scene in SCENES:
        produced, skips = process_scene(scene, manifest)
        all_produced.extend(produced)
        all_skips.extend(skips)

    manifest["skips"] = all_skips
    manifest["script"] = str(Path(__file__).resolve())
    write_json_atomic(manifest_path, manifest)

    print(f"[edit_grids] manifest {manifest_path}")
    for skip in all_skips:
        print(f"[edit_grids][SKIP] {skip}")
    print(f"[edit_grids] produced {len(all_produced)} grid(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
