#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _stem(path: Path) -> str:
    return path.stem


def _numeric_suffix(name: str) -> Optional[str]:
    m = re.search(r"(\d+)$", name)
    return m.group(1) if m else None


def _candidate_mask_stems(image_stem: str) -> List[str]:
    cands = [image_stem]
    if image_stem.startswith("images_"):
        cands.append(image_stem[len("images_"):])
    num = _numeric_suffix(image_stem)
    if num is not None:
        cands.append(num)

    seen = set()
    out = []
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _build_mask_index(mask_dir: Path, suffix: str) -> Dict[str, Path]:
    out = {}
    for p in sorted(mask_dir.glob(f"*{suffix}")):
        out[p.stem] = p
    return out


def _link_or_copy(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src.resolve(), dst)
    else:
        import shutil

        shutil.copy2(src.resolve(), dst)


def main():
    parser = argparse.ArgumentParser(description="Align sparse ground masks to image stems expected by training.")
    parser.add_argument("--image_dir", required=True, type=str, help="Directory with training images (e.g. .../images).")
    parser.add_argument("--mask_dir", required=True, type=str, help="Original mask directory (e.g. SegmentationClass).")
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Aligned mask directory. Default: <mask_dir>_aligned_for_images",
    )
    parser.add_argument("--suffix", default=".png", type=str)
    parser.add_argument("--mode", default="symlink", choices=["symlink", "hardlink", "copy"])
    parser.add_argument("--clean_output", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    mask_dir = Path(args.mask_dir).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"mask_dir not found: {mask_dir}")

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = mask_dir.parent / f"{mask_dir.name}_aligned_for_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        for p in out_dir.glob(f"*{args.suffix}"):
            p.unlink()

    mask_index = _build_mask_index(mask_dir=mask_dir, suffix=args.suffix)
    image_files = sorted([p for p in image_dir.iterdir() if p.is_file()])

    matched = 0
    missing = 0
    per_image = []
    for img in image_files:
        img_stem = _stem(img)
        found_src = None
        for cand in _candidate_mask_stems(img_stem):
            if cand in mask_index:
                found_src = mask_index[cand]
                break
        rec = {
            "image_stem": img_stem,
            "matched": bool(found_src is not None),
            "source_mask": str(found_src) if found_src is not None else "",
        }
        if found_src is not None:
            dst = out_dir / f"{img_stem}{args.suffix}"
            _link_or_copy(src=found_src, dst=dst, mode=args.mode)
            matched += 1
        else:
            missing += 1
        per_image.append(rec)

    summary = {
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "output_dir": str(out_dir),
        "mode": args.mode,
        "suffix": args.suffix,
        "num_images": len(image_files),
        "num_masks_source": len(mask_index),
        "num_matched": matched,
        "num_missing": missing,
        "coverage_ratio": float(matched / max(1, len(image_files))),
    }

    summary_json = out_dir / "alignment_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=2)

    summary_md = out_dir / "alignment_summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Ground Mask Alignment Summary\n\n")
        for k, v in summary.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Notes\n")
        f.write("- Output filenames are aligned to training image stems (e.g. `images_00001.png`).\n")
        f.write("- Missing images keep no file; training fallback behavior remains unchanged.\n")

    print(f"[MaskAlign] output_dir={out_dir}")
    print(f"[MaskAlign] matched={matched}/{len(image_files)} coverage={summary['coverage_ratio']:.4f}")
    print(f"[MaskAlign] summary_json={summary_json}")
    print(f"[MaskAlign] summary_md={summary_md}")


if __name__ == "__main__":
    main()
