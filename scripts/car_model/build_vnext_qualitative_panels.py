#!/usr/bin/env python3
"""Build compact qualitative comparison panels for SPCarNet/vNext renders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageStat


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _parse_label_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Empty label in: {value}")
    return label, Path(path).expanduser()


def _list_images(path: Path) -> Dict[str, Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {path}")
    images = {
        p.name: p
        for p in sorted(path.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }
    if not images:
        raise FileNotFoundError(f"No images found in: {path}")
    return images


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _resize_for_tile(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image.copy()
    height = max(1, round(image.height * width / image.width))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize((width, height), resampling)


def _labelled_tile(image: Image.Image, label: str, width: int, label_height: int) -> Image.Image:
    body = _resize_for_tile(image, width)
    tile = Image.new("RGB", (width, body.height + label_height), (255, 255, 255))
    tile.paste(body, (0, label_height))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, width - 1, label_height - 1), fill=(245, 245, 245))
    draw.text((8, max(2, (label_height - 11) // 2)), label[:80], fill=(20, 20, 20))
    return tile


def _diff_image(lhs: Image.Image, rhs: Image.Image, scale: float) -> Image.Image:
    if lhs.size != rhs.size:
        raise ValueError(f"Cannot diff images with different native sizes: {lhs.size} vs {rhs.size}")
    diff = ImageChops.difference(lhs, rhs)
    if scale != 1.0:
        diff = diff.point(lambda value: max(0, min(255, int(round(value * scale)))))
    return diff


def _mean_l1(lhs_path: Path, rhs_path: Path) -> float:
    lhs = _load_rgb(lhs_path)
    rhs = _load_rgb(rhs_path)
    diff = _diff_image(lhs, rhs, 1.0)
    return float(sum(ImageStat.Stat(diff).mean) / (3.0 * 255.0))


def _select_frames(
    common_names: Sequence[str],
    gt_images: Dict[str, Path],
    method_images: Dict[str, Dict[str, Path]],
    reference_label: str | None,
    candidate_label: str | None,
    num_views: int,
    selection_mode: str,
    explicit_frames: Sequence[str] | None,
) -> Tuple[List[str], Dict[str, float]]:
    if explicit_frames:
        missing = [name for name in explicit_frames if name not in common_names]
        if missing:
            raise RuntimeError(f"Requested frames are not common to all inputs: {missing}")
        return list(explicit_frames), {}

    if selection_mode == "candidate_worst_gt_l1" and not candidate_label:
        raise ValueError("--candidate_label is required for candidate_worst_gt_l1 selection")
    if selection_mode == "largest_candidate_reference_delta" and (
        not candidate_label or not reference_label
    ):
        raise ValueError(
            "--candidate_label and --reference_label are required for largest_candidate_reference_delta"
        )

    ranked: List[Tuple[float, str]] = []
    scores: Dict[str, float] = {}
    for name in common_names:
        score = 0.0
        if selection_mode == "first":
            score = -float(len(ranked))
        elif selection_mode == "candidate_worst_gt_l1" and candidate_label:
            score = _mean_l1(method_images[candidate_label][name], gt_images[name])
        elif (
            selection_mode == "largest_candidate_reference_delta"
            and candidate_label
            and reference_label
        ):
            score = _mean_l1(
                method_images[candidate_label][name],
                method_images[reference_label][name],
            )
        else:
            score = -float(len(ranked))
        ranked.append((score, name))
        scores[name] = score

    if selection_mode == "first":
        selected = [name for _, name in ranked[:num_views]]
    else:
        selected = [name for _, name in sorted(ranked, reverse=True)[:num_views]]
    return selected, scores


def _make_row(
    frame_name: str,
    gt_path: Path,
    method_paths: Dict[str, Path],
    method_order: Sequence[str],
    reference_label: str | None,
    candidate_label: str | None,
    tile_width: int,
    label_height: int,
    diff_scale: float,
) -> Image.Image:
    gt = _load_rgb(gt_path)
    tiles = [_labelled_tile(gt, f"GT | {frame_name}", tile_width, label_height)]
    loaded_methods = {label: _load_rgb(method_paths[label]) for label in method_order}

    for label in method_order:
        tiles.append(_labelled_tile(loaded_methods[label], label, tile_width, label_height))

    if candidate_label in loaded_methods:
        candidate = loaded_methods[candidate_label]
        tiles.append(
            _labelled_tile(
                _diff_image(candidate, gt, diff_scale),
                f"|{candidate_label}-GT| x{diff_scale:g}",
                tile_width,
                label_height,
            )
        )

    if reference_label in loaded_methods:
        reference = loaded_methods[reference_label]
        tiles.append(
            _labelled_tile(
                _diff_image(reference, gt, diff_scale),
                f"|{reference_label}-GT| x{diff_scale:g}",
                tile_width,
                label_height,
            )
        )

    if candidate_label in loaded_methods and reference_label in loaded_methods:
        tiles.append(
            _labelled_tile(
                _diff_image(loaded_methods[candidate_label], loaded_methods[reference_label], diff_scale),
                f"|{candidate_label}-{reference_label}| x{diff_scale:g}",
                tile_width,
                label_height,
            )
        )

    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    row = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for tile in tiles:
        row.paste(tile, (x, 0))
        x += tile.width
    return row


def build_panel(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = Path(args.gt_dir).expanduser()
    gt_images = _list_images(gt_dir)
    methods = [_parse_label_path(value) for value in args.method]
    method_order = [label for label, _ in methods]
    if len(method_order) != len(set(method_order)):
        raise ValueError(f"Duplicate method labels: {method_order}")
    if args.reference_label and args.reference_label not in method_order:
        raise ValueError(f"--reference_label {args.reference_label!r} is not in methods {method_order}")
    if args.candidate_label and args.candidate_label not in method_order:
        raise ValueError(f"--candidate_label {args.candidate_label!r} is not in methods {method_order}")

    method_images = {label: _list_images(path) for label, path in methods}
    common_names = sorted(
        set(gt_images).intersection(*(set(images) for images in method_images.values()))
    )
    if not common_names:
        missing = {
            label: sorted(set(gt_images).symmetric_difference(set(images)))[:25]
            for label, images in method_images.items()
        }
        raise RuntimeError(
            "No common frame names across GT and methods. "
            f"Example symmetric differences: {json.dumps(missing, indent=2)}"
        )

    explicit_frames = None
    if args.frames:
        explicit_frames = [item.strip() for item in args.frames.split(",") if item.strip()]

    selected, scores = _select_frames(
        common_names,
        gt_images,
        method_images,
        args.reference_label,
        args.candidate_label,
        args.num_views,
        args.selection_mode,
        explicit_frames,
    )
    if not selected:
        raise RuntimeError("No frames selected for the qualitative panel.")

    selected_file_audit: Dict[str, Dict[str, Dict[str, object]]] = {}
    for name in selected:
        selected_file_audit[name] = {
            "gt": {
                "path": str(gt_images[name]),
                "sha1": _sha1(gt_images[name]),
                "size": list(_image_size(gt_images[name])),
            }
        }
        for label in method_order:
            path = method_images[label][name]
            selected_file_audit[name][label] = {
                "path": str(path),
                "sha1": _sha1(path),
                "size": list(_image_size(path)),
            }

    rows = [
        _make_row(
            frame_name=name,
            gt_path=gt_images[name],
            method_paths={label: method_images[label][name] for label in method_order},
            method_order=method_order,
            reference_label=args.reference_label,
            candidate_label=args.candidate_label,
            tile_width=args.tile_width,
            label_height=args.label_height,
            diff_scale=args.diff_scale,
        )
        for name in selected
    ]

    panel_width = max(row.width for row in rows)
    panel_height = sum(row.height for row in rows) + args.row_gap * (len(rows) - 1)
    panel = Image.new("RGB", (panel_width, panel_height), (255, 255, 255))
    y = 0
    for row in rows:
        panel.paste(row, (0, y))
        y += row.height + args.row_gap

    panel_path = output_dir / f"{args.panel_name}.png"
    manifest_path = output_dir / f"{args.panel_name}_manifest.json"
    summary_path = output_dir / f"{args.panel_name}_summary.md"
    panel.save(panel_path)

    manifest: Dict[str, object] = {
        "schema_version": 2,
        "panel_path": str(panel_path),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "argv": list(getattr(args, "command", None) or sys.argv),
        "gt_dir": str(gt_dir),
        "methods": {label: str(path) for label, path in methods},
        "method_order": method_order,
        "reference_label": args.reference_label,
        "candidate_label": args.candidate_label,
        "selection_mode": args.selection_mode,
        "diff_scale": args.diff_scale,
        "tile_width": args.tile_width,
        "common_frame_count": len(common_names),
        "selected_count": len(selected),
        "selected_frames": selected,
        "selected_file_audit": selected_file_audit,
        "selection_scores": {name: scores.get(name) for name in selected if name in scores},
        "alignment_policy": "filename_intersection_with_selected_file_hashes",
        "strict_native_size_diff": True,
        "missing_counts": {
            label: {
                "missing_from_method": len(set(gt_images) - set(images)),
                "extra_in_method": len(set(images) - set(gt_images)),
            }
            for label, images in method_images.items()
        },
    }
    provenance_json = getattr(args, "provenance_json", None)
    if provenance_json:
        manifest["provenance"] = json.loads(provenance_json)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary = [
        f"# {args.panel_name}",
        "",
        f"- Panel: `{panel_path}`",
        f"- Common frames: `{len(common_names)}`",
        f"- Selected frames: `{', '.join(selected)}`",
        f"- Selection mode: `{args.selection_mode}`",
        f"- Reference: `{args.reference_label}`",
        f"- Candidate: `{args.candidate_label}`",
        "",
        "This panel is generated from preserved render outputs and is intended for README/PPT qualitative evidence.",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt_dir", required=True, help="Directory containing GT images.")
    parser.add_argument(
        "--method",
        required=True,
        action="append",
        help="Method render directory as LABEL=PATH. Repeat for baseline/current/improved.",
    )
    parser.add_argument("--reference_label", default=None, help="Label used as the reference baseline.")
    parser.add_argument("--candidate_label", default=None, help="Label used as the improved candidate.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--panel_name", default="qualitative_panel")
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--tile_width", type=int, default=360)
    parser.add_argument("--label_height", type=int, default=26)
    parser.add_argument("--row_gap", type=int, default=8)
    parser.add_argument("--diff_scale", type=float, default=4.0)
    parser.add_argument("--frames", default=None, help="Comma-separated frame names to use exactly.")
    parser.add_argument(
        "--selection_mode",
        choices=("first", "candidate_worst_gt_l1", "largest_candidate_reference_delta"),
        default="largest_candidate_reference_delta",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_views < 1:
        raise ValueError("--num_views must be positive")
    if args.tile_width < 16:
        raise ValueError("--tile_width must be at least 16")
    manifest = build_panel(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
