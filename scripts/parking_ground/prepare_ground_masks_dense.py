#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _stem(p: Path) -> str:
    return p.stem


def _numeric_suffix(name: str) -> Optional[int]:
    m = re.search(r"(\d+)$", name)
    if m is None:
        return None
    return int(m.group(1))


def _to_binary_mask(arr: np.ndarray, threshold: int) -> np.ndarray:
    # Keep behavior simple and deterministic for this dataset.
    if arr.ndim == 2:
        return (arr > threshold).astype(np.uint8) * 255
    if arr.ndim == 3:
        rgb = arr[..., :3]
        # Fast path: non-black as foreground (avoids expensive per-image unique scan).
        mask = np.any(rgb > 0, axis=-1)
        # If mask collapses to all false (rare), fallback to grayscale threshold.
        if not np.any(mask):
            gray = rgb.astype(np.float32).mean(axis=-1)
            mask = gray > float(threshold)
        return mask.astype(np.uint8) * 255
    raise ValueError(f"Unsupported mask array shape: {arr.shape}")


def _candidate_source_stems(image_stem: str) -> List[str]:
    cands = [image_stem]
    if image_stem.startswith("images_"):
        cands.append(image_stem[len("images_"):])
    num = _numeric_suffix(image_stem)
    if num is not None:
        cands.append(str(num).zfill(5))
        cands.append(str(num))
    out = []
    seen = set()
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _build_source_index(mask_dir: Path, suffix: str) -> Dict[str, Path]:
    idx = {}
    for p in sorted(mask_dir.glob(f"*{suffix}")):
        idx[p.stem] = p
    return idx


def _nearest_source_by_numeric(src_index: Dict[str, Path], query_num: int) -> Optional[Tuple[str, Path, int]]:
    numeric_items = []
    for stem, p in src_index.items():
        n = _numeric_suffix(stem)
        if n is None:
            continue
        numeric_items.append((n, stem, p))
    if len(numeric_items) == 0:
        return None
    numeric_items.sort(key=lambda x: x[0])
    vals = np.array([x[0] for x in numeric_items], dtype=np.int64)
    pos = int(np.argmin(np.abs(vals - int(query_num))))
    nearest = numeric_items[pos]
    gap = abs(int(nearest[0]) - int(query_num))
    return nearest[1], nearest[2], gap


def main():
    parser = argparse.ArgumentParser(description="Prepare dense, training-aligned ground masks for all images.")
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--source_mask_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--suffix", default=".png", type=str)
    parser.add_argument("--threshold", default=127, type=int)
    parser.add_argument("--fill_missing_nearest", action="store_true")
    parser.add_argument("--nearest_max_gap", default=1000000, type=int)
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    source_mask_dir = Path(args.source_mask_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not source_mask_dir.is_dir():
        raise FileNotFoundError(f"source_mask_dir not found: {source_mask_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    src_index = _build_source_index(source_mask_dir, args.suffix)
    image_files = sorted([p for p in image_dir.iterdir() if p.is_file()])

    summary = {
        "image_dir": str(image_dir),
        "source_mask_dir": str(source_mask_dir),
        "output_dir": str(output_dir),
        "num_images": len(image_files),
        "num_source_masks": len(src_index),
        "matched_exact": 0,
        "matched_nearest": 0,
        "missing": 0,
        "coverage_ratio": 0.0,
        "fill_missing_nearest": bool(args.fill_missing_nearest),
        "nearest_max_gap": int(args.nearest_max_gap),
    }
    per_image = []

    for img in image_files:
        img_stem = _stem(img)
        src_path = None
        match_mode = "missing"

        for cand in _candidate_source_stems(img_stem):
            p = src_index.get(cand, None)
            if p is not None:
                src_path = p
                match_mode = "exact"
                break

        if (src_path is None) and bool(args.fill_missing_nearest):
            qn = _numeric_suffix(img_stem)
            if qn is not None:
                near = _nearest_source_by_numeric(src_index, qn)
                if near is not None:
                    near_stem, near_path, gap = near
                    if int(gap) <= int(args.nearest_max_gap):
                        src_path = near_path
                        match_mode = f"nearest(gap={gap},stem={near_stem})"

        out_path = output_dir / f"{img_stem}{args.suffix}"
        rec = {
            "image_stem": img_stem,
            "matched": bool(src_path is not None),
            "mode": match_mode,
            "source_mask": str(src_path) if src_path is not None else "",
            "output_mask": str(out_path),
        }

        if src_path is None:
            summary["missing"] += 1
            per_image.append(rec)
            continue

        arr = np.array(Image.open(src_path))
        bin_mask = _to_binary_mask(arr=arr, threshold=int(args.threshold))
        Image.fromarray(bin_mask, mode="L").save(out_path)
        if match_mode == "exact":
            summary["matched_exact"] += 1
        else:
            summary["matched_nearest"] += 1
        per_image.append(rec)

    matched_total = int(summary["matched_exact"]) + int(summary["matched_nearest"])
    summary["coverage_ratio"] = float(matched_total / max(1, int(summary["num_images"])))

    with open(output_dir / "dense_mask_prep_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=2)

    with open(output_dir / "dense_mask_prep_summary.md", "w", encoding="utf-8") as f:
        f.write("# Dense Ground Mask Preparation Summary\n\n")
        for k, v in summary.items():
            f.write(f"- {k}: {v}\n")

    print(f"[DenseMaskPrep] output_dir={output_dir}")
    print(
        "[DenseMaskPrep] matched_exact={} matched_nearest={} missing={} coverage={:.4f}".format(
            summary["matched_exact"],
            summary["matched_nearest"],
            summary["missing"],
            summary["coverage_ratio"],
        )
    )


if __name__ == "__main__":
    main()
