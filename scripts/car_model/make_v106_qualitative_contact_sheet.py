#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL_PATH = Path("/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625")
DEFAULT_BASE_METHOD = "ours_26000_v104c_shrink_view_affine_min1_minviews1_{scene}"
DEFAULT_CANDIDATE_METHOD = "ours_26000_v106_podmoe_basepreserve_{scene}"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency guard
    np = None  # type: ignore[assignment]

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional dependency guard
    imageio = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional dependency guard
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CropBox:
    x0: int
    y0: int
    x1: int
    y1: int
    label: str


@dataclass(frozen=True)
class FrameTriplet:
    scene: str
    frame_name: str
    base_path: Path
    candidate_path: Path
    gt_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact qualitative contact sheet comparing v104c baseline, "
            "v106 candidate, GT, and simple absolute-error maps."
        )
    )
    parser.add_argument(
        "--scene",
        action="append",
        required=True,
        help="Scene name. May be repeated or comma-separated, e.g. --scene garden,bonsai.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base-method", default=DEFAULT_BASE_METHOD)
    parser.add_argument("--candidate-method", default=DEFAULT_CANDIDATE_METHOD)
    parser.add_argument("--frame-glob", default="*.png", help="Glob used inside render directories.")
    parser.add_argument(
        "--frame-index",
        action="append",
        type=int,
        help="0-based index into sorted frame matches. May be repeated. Defaults to 0.",
    )
    parser.add_argument("--max-frames", type=int, default=1, help="Maximum frames per scene when no frame index is given.")
    parser.add_argument(
        "--crop",
        action="append",
        default=[],
        help="Optional crop box as x,y,w,h by default. May be repeated.",
    )
    parser.add_argument(
        "--crop-mode",
        choices=("xywh", "xyxy"),
        default="xywh",
        help="Interpret --crop as x,y,w,h or x0,y0,x1,y1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/car_model/assets/v106_qualitative/v106_qualitative_contact_sheet.png"),
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=240,
        help="Maximum panel width in pixels; keeps generated previews small.",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=180,
        help="Maximum panel height in pixels; keeps generated previews small.",
    )
    parser.add_argument(
        "--error-scale",
        type=float,
        default=0.25,
        help="Absolute RGB error value mapped to full heat-map intensity.",
    )
    parser.add_argument("--no-manifest", action="store_true", help="Do not write a sidecar JSON manifest.")
    return parser.parse_args()


def require_image_stack() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if Image is None:
        missing.append("PIL/Pillow")
    if imageio is None and Image is None:
        missing.append("imageio or PIL/Pillow")
    if missing:
        raise SystemExit(
            "cannot create PNG contact sheets; missing optional image dependency: " + ", ".join(sorted(set(missing)))
        )


def expand_scenes(values: Iterable[str]) -> list[str]:
    scenes: list[str] = []
    for value in values:
        for part in value.split(","):
            scene = part.strip()
            if scene and scene not in scenes:
                scenes.append(scene)
    if not scenes:
        raise SystemExit("no scenes provided")
    return scenes


def render_method(template: str, scene: str) -> str:
    return template.format(scene=scene) if "{scene}" in template else template


def method_root(model_path: Path, scene: str, method_template: str) -> Path:
    return model_path / scene / "detached_model" / "test" / render_method(method_template, scene)


def natural_key(path: Path) -> tuple[object, ...]:
    parts: list[object] = []
    token = ""
    numeric = path.stem.isdigit()
    if numeric:
        return (int(path.stem), path.name)
    for char in path.name:
        if char.isdigit():
            token += char
        else:
            if token:
                parts.append(int(token))
                token = ""
            parts.append(char)
    if token:
        parts.append(int(token))
    return tuple(parts)


def sorted_frames(render_dir: Path, frame_glob: str) -> list[Path]:
    if not render_dir.is_dir():
        raise SystemExit(f"missing renders directory: {render_dir}")
    frames = [path for path in render_dir.glob(frame_glob) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(frames, key=natural_key)


def choose_frames(frames: list[Path], indexes: list[int] | None, max_frames: int, render_dir: Path) -> list[Path]:
    if not frames:
        raise SystemExit(f"no frames matched in {render_dir}")
    if indexes:
        selected = []
        for index in indexes:
            if index < 0 or index >= len(frames):
                raise SystemExit(f"frame index {index} out of range for {render_dir}; matched {len(frames)} frames")
            selected.append(frames[index])
        return selected
    return frames[: max(1, max_frames)]


def paired_frame(path: Path, peer_dir: Path) -> Path:
    candidate = peer_dir / path.name
    if candidate.exists():
        return candidate
    same_stem = sorted(peer_dir.glob(path.stem + ".*"), key=natural_key)
    for peer in same_stem:
        if peer.is_file() and peer.suffix.lower() in IMAGE_SUFFIXES:
            return peer
    raise SystemExit(f"could not find frame {path.name} in {peer_dir}")


def collect_triplets(args: argparse.Namespace, scenes: list[str]) -> list[FrameTriplet]:
    triplets: list[FrameTriplet] = []
    for scene in scenes:
        base_root = method_root(args.model_path, scene, args.base_method)
        candidate_root = method_root(args.model_path, scene, args.candidate_method)
        base_render_dir = base_root / "renders"
        candidate_render_dir = candidate_root / "renders"
        gt_dir = candidate_root / "gt"
        if not gt_dir.is_dir():
            gt_dir = base_root / "gt"
        frames = choose_frames(
            sorted_frames(candidate_render_dir, args.frame_glob),
            args.frame_index,
            args.max_frames,
            candidate_render_dir,
        )
        for candidate_path in frames:
            triplets.append(
                FrameTriplet(
                    scene=scene,
                    frame_name=candidate_path.name,
                    base_path=paired_frame(candidate_path, base_render_dir),
                    candidate_path=candidate_path,
                    gt_path=paired_frame(candidate_path, gt_dir),
                )
            )
    return triplets


def parse_crop(raw: str, mode: str, ordinal: int) -> CropBox:
    pieces = [piece.strip() for piece in raw.replace(":", ",").split(",")]
    if len(pieces) != 4:
        raise SystemExit(f"invalid crop {raw!r}; expected four comma-separated integers")
    try:
        a, b, c, d = [int(round(float(piece))) for piece in pieces]
    except ValueError as exc:
        raise SystemExit(f"invalid crop {raw!r}; expected numeric values") from exc
    if mode == "xywh":
        x0, y0, x1, y1 = a, b, a + c, b + d
    else:
        x0, y0, x1, y1 = a, b, c, d
    if x1 <= x0 or y1 <= y0:
        raise SystemExit(f"invalid crop {raw!r}; non-positive crop extent")
    return CropBox(x0=x0, y0=y0, x1=x1, y1=y1, label=f"crop{ordinal}:{raw}")


def read_rgb(path: Path) -> object:
    if imageio is not None:
        arr = imageio.imread(path)
    else:
        arr = np.asarray(Image.open(path))
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise SystemExit(f"unsupported image shape {arr.shape} for {path}")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        max_value = float(np.max(arr)) if arr.size else 1.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def clamp_crop(crop: CropBox | None, width: int, height: int) -> tuple[int, int, int, int]:
    if crop is None:
        return 0, 0, width, height
    x0 = min(max(0, crop.x0), width)
    y0 = min(max(0, crop.y0), height)
    x1 = min(max(0, crop.x1), width)
    y1 = min(max(0, crop.y1), height)
    if x1 <= x0 or y1 <= y0:
        raise SystemExit(f"crop {crop.label} is outside image bounds {width}x{height}")
    return x0, y0, x1, y1


def fit_tile(arr: object, max_width: int, max_height: int) -> object:
    height, width = arr.shape[:2]
    scale = min(max_width / max(1, width), max_height / max(1, height), 1.0)
    if scale >= 1.0:
        return arr
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    image = Image.fromarray(arr)
    return np.asarray(image.resize(new_size, Image.Resampling.LANCZOS))


def normalize_to_rgb(arr: object) -> object:
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return arr[:, :, :3]


def make_error_map(render: object, gt: object, scale: float) -> object:
    error = np.mean(np.abs(render.astype(np.float32) - gt.astype(np.float32)) / 255.0, axis=2)
    denom = scale if scale > 0 else max(float(np.percentile(error, 99)), 1.0 / 255.0)
    intensity = np.clip(error / max(denom, 1.0 / 255.0), 0.0, 1.0)
    heat = np.zeros((*intensity.shape, 3), dtype=np.uint8)
    heat[:, :, 0] = np.clip(255.0 * intensity, 0, 255).astype(np.uint8)
    heat[:, :, 1] = np.clip(255.0 * np.sqrt(intensity) * 0.65, 0, 180).astype(np.uint8)
    heat[:, :, 2] = np.clip(255.0 * (1.0 - intensity) * 0.12, 0, 45).astype(np.uint8)
    return heat


def fit_text(draw: object, text: str, font: object, max_width: int) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = "..."
    if draw.textlength(suffix, font=font) > max_width:
        return ""
    keep = text
    while keep and draw.textlength(keep + suffix, font=font) > max_width:
        keep = keep[:-1]
    return keep + suffix if keep else suffix


def label_tile(arr: object, label: str, sublabel: str = "") -> object:
    pad = 6
    label_height = 34 if sublabel else 22
    image = Image.new("RGB", (arr.shape[1], arr.shape[0] + label_height), (255, 255, 255))
    image.paste(Image.fromarray(normalize_to_rgb(arr)), (0, label_height))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    max_text_width = max(1, arr.shape[1] - pad * 2)
    draw.text((pad, 4), fit_text(draw, label, font, max_text_width), fill=(0, 0, 0), font=font)
    if sublabel:
        draw.text((pad, 18), fit_text(draw, sublabel, font, max_text_width), fill=(80, 80, 80), font=font)
    return np.asarray(image)


def pad_to(arr: object, width: int, height: int) -> object:
    out = np.full((height, width, 3), 255, dtype=np.uint8)
    out[: arr.shape[0], : arr.shape[1], :] = normalize_to_rgb(arr)
    return out


def hstack_tiles(tiles: list[object], gap: int = 8) -> object:
    height = max(tile.shape[0] for tile in tiles)
    padded = [pad_to(tile, tile.shape[1], height) for tile in tiles]
    separators = [np.full((height, gap, 3), 240, dtype=np.uint8) for _ in range(max(0, len(padded) - 1))]
    pieces: list[object] = []
    for index, tile in enumerate(padded):
        pieces.append(tile)
        if index < len(separators):
            pieces.append(separators[index])
    return np.concatenate(pieces, axis=1)


def vstack_rows(rows: list[object], gap: int = 12) -> object:
    width = max(row.shape[1] for row in rows)
    padded = [pad_to(row, width, row.shape[0]) for row in rows]
    separators = [np.full((gap, width, 3), 248, dtype=np.uint8) for _ in range(max(0, len(padded) - 1))]
    pieces: list[object] = []
    for index, row in enumerate(padded):
        pieces.append(row)
        if index < len(separators):
            pieces.append(separators[index])
    return np.concatenate(pieces, axis=0)


def crop_arrays(base: object, candidate: object, gt: object, crop: CropBox | None) -> tuple[object, object, object]:
    height = min(base.shape[0], candidate.shape[0], gt.shape[0])
    width = min(base.shape[1], candidate.shape[1], gt.shape[1])
    x0, y0, x1, y1 = clamp_crop(crop, width, height)
    return base[y0:y1, x0:x1], candidate[y0:y1, x0:x1], gt[y0:y1, x0:x1]


def build_contact_sheet(triplets: list[FrameTriplet], crops: list[CropBox], args: argparse.Namespace) -> object:
    rows: list[object] = []
    crop_items: list[CropBox | None] = crops if crops else [None]
    for triplet in triplets:
        base = read_rgb(triplet.base_path)
        candidate = read_rgb(triplet.candidate_path)
        gt = read_rgb(triplet.gt_path)
        for crop in crop_items:
            crop_base, crop_candidate, crop_gt = crop_arrays(base, candidate, gt, crop)
            sublabel = f"{triplet.scene}/{triplet.frame_name}"
            if crop is not None:
                sublabel += f" {crop.label}"
            panels = [
                ("GT", crop_gt),
                ("v104c baseline", crop_base),
                ("v106 candidate", crop_candidate),
                ("|v104c-GT|", make_error_map(crop_base, crop_gt, args.error_scale)),
                ("|v106-GT|", make_error_map(crop_candidate, crop_gt, args.error_scale)),
            ]
            tiles = [
                label_tile(fit_tile(panel, args.tile_width, args.tile_height), label, sublabel if idx == 0 else "")
                for idx, (label, panel) in enumerate(panels)
            ]
            rows.append(hstack_tiles(tiles))
    if not rows:
        raise SystemExit("no rows generated")
    return vstack_rows(rows)


def write_png(path: Path, arr: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if imageio is not None:
        imageio.imwrite(path, arr)
        return
    Image.fromarray(arr).save(path)


def write_manifest(path: Path, triplets: list[FrameTriplet], crops: list[CropBox], args: argparse.Namespace) -> None:
    manifest_path = path.with_suffix(".json")
    payload = {
        "output": str(path),
        "model_path": str(args.model_path),
        "base_method": args.base_method,
        "candidate_method": args.candidate_method,
        "frame_glob": args.frame_glob,
        "frame_index": args.frame_index,
        "max_frames": args.max_frames,
        "tile_width": args.tile_width,
        "tile_height": args.tile_height,
        "error_scale": args.error_scale,
        "crops": [crop.__dict__ for crop in crops],
        "frames": [
            {
                "scene": triplet.scene,
                "frame_name": triplet.frame_name,
                "base_path": str(triplet.base_path),
                "candidate_path": str(triplet.candidate_path),
                "gt_path": str(triplet.gt_path),
            }
            for triplet in triplets
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    require_image_stack()
    if args.tile_width < 32 or args.tile_height < 32:
        raise SystemExit("--tile-width and --tile-height must be at least 32")
    if not math.isfinite(args.error_scale):
        raise SystemExit("--error-scale must be finite")
    scenes = expand_scenes(args.scene)
    crops = [parse_crop(raw, args.crop_mode, index + 1) for index, raw in enumerate(args.crop)]
    triplets = collect_triplets(args, scenes)
    sheet = build_contact_sheet(triplets, crops, args)
    write_png(args.output, sheet)
    if not args.no_manifest:
        write_manifest(args.output, triplets, crops, args)
    print(f"wrote {args.output} ({sheet.shape[1]}x{sheet.shape[0]})")
    if not args.no_manifest:
        print(f"wrote {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
