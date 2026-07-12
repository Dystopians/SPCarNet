#!/usr/bin/env python
"""Build edit-aware ECR cache variants with transformed validity masks."""
from __future__ import annotations

import argparse
import copy
import errno
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


VALID_VARIANTS = ("dilate4", "dilate16", "box2d")


def parse_variants(spec: str) -> list[str]:
    variants = [v.strip() for v in str(spec).split(",") if v.strip()]
    unknown = [v for v in variants if v not in VALID_VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; valid={list(VALID_VARIANTS)}")
    if len(set(variants)) != len(variants):
        raise ValueError(f"duplicate variants in {variants}")
    if not variants:
        raise ValueError("at least one variant is required")
    return variants


def dir_file_sizes(root: Path) -> dict[str, int]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(root))] = path.stat().st_size
    return files


def load_manifest(cache_dir: Path) -> dict:
    path = cache_dir / "manifest.json"
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    masks = manifest.get("edit", {}).get("masks", {})
    if not isinstance(masks, dict) or not masks:
        raise ValueError(f"{path} does not contain edit.masks")
    return manifest


def hardlink_cache_payload(src: Path, dst: Path):
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if str(rel) == "manifest.json" or rel.parts[0] == "masks":
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, out)
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                shutil.copy2(path, out)
            else:
                raise


def mask_to_stale(mask_path: Path) -> np.ndarray:
    arr = np.array(Image.open(mask_path).convert("L"))
    valid = arr >= 128
    return ~valid


def stale_to_mask(stale: np.ndarray) -> np.ndarray:
    stale = np.asarray(stale, dtype=bool)
    return np.where(stale, 0, 255).astype(np.uint8)


def dilate_stale(stale: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(stale, dtype=bool).copy()
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return binary_dilation(np.asarray(stale, dtype=bool), structure=structure)


def box_stale(stale: np.ndarray) -> np.ndarray:
    stale = np.asarray(stale, dtype=bool)
    ys, xs = np.nonzero(stale)
    out = np.zeros_like(stale, dtype=bool)
    if ys.size == 0:
        return out
    out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return out


def transform_stale(stale: np.ndarray, variant: str) -> np.ndarray:
    if variant == "dilate4":
        return dilate_stale(stale, 4)
    if variant == "dilate16":
        return dilate_stale(stale, 16)
    if variant == "box2d":
        return box_stale(stale)
    raise ValueError(f"unknown variant: {variant}")


def write_variant_masks(src: Path, dst: Path, masks: dict[str, str], variant: str):
    for _, rel in sorted(masks.items()):
        rel_path = Path(rel)
        stale = mask_to_stale(src / rel_path)
        out_mask = stale_to_mask(transform_stale(stale, variant))
        out_path = dst / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out_mask, mode="L").save(out_path)


def amend_manifest(manifest: dict, dst: Path, variant: str) -> dict:
    out = copy.deepcopy(manifest)
    edit = dict(out.get("edit", {}))
    policy = str(edit.get("policy", "")).strip()
    suffix = f"mask_variant={variant}"
    edit["policy"] = f"{policy}; {suffix}" if policy else suffix
    out["edit"] = edit

    files = dir_file_sizes(dst)
    sizes = dict(out.get("sizes", {}))
    sizes["files"] = files
    sizes["n_files"] = len(files)
    sizes["cache_mb_raw"] = sum(files.values()) / (1024.0 * 1024.0)
    out["sizes"] = sizes
    return out


def build_variant(edited_cache: Path, out_root: Path, manifest: dict, variant: str):
    dst = out_root / variant
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty directory: {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    masks = dict(manifest["edit"]["masks"])
    hardlink_cache_payload(edited_cache, dst)
    write_variant_masks(edited_cache, dst, masks, variant)
    out_manifest = amend_manifest(manifest, dst, variant)
    with (dst / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(out_manifest, fh, indent=1)
    return out_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edited-cache", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--variants", required=True,
                    help="comma-separated subset of: dilate4,dilate16,box2d")
    args = ap.parse_args()

    edited_cache = Path(args.edited_cache)
    out_root = Path(args.out_root)
    try:
        variants = parse_variants(args.variants)
        manifest = load_manifest(edited_cache)
        out_root.mkdir(parents=True, exist_ok=True)
        for variant in variants:
            out_manifest = build_variant(edited_cache, out_root, manifest, variant)
            print(f"[mask_variants] {variant}: {out_root / variant} "
                  f"files={out_manifest['sizes']['n_files']}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
