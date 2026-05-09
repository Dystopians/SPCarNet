#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


def _read_images(renders_dir: Path, gt_dir: Path, view_names: set[str] | None = None):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        if view_names is not None and fname not in view_names and Path(fname).stem not in view_names:
            continue
        render_path = renders_dir / fname
        gt_path = gt_dir / fname
        if not render_path.is_file() or not gt_path.is_file():
            continue
        render = Image.open(render_path)
        gt = Image.open(gt_path)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate_split(
    model_path: Path,
    split: str,
    methods: set[str] | None = None,
    view_names: set[str] | None = None,
) -> tuple[dict, dict]:
    split_dir = model_path / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    full: dict[str, dict[str, float]] = {}
    per_view: dict[str, dict[str, dict[str, float]]] = {}
    for method_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        method = method_dir.name
        if methods is not None and method not in methods:
            continue
        renders_dir = method_dir / "renders"
        gt_dir = method_dir / "gt"
        if not renders_dir.is_dir() or not gt_dir.is_dir():
            continue
        renders, gts, image_names = _read_images(renders_dir, gt_dir, view_names=view_names)
        if not renders:
            continue
        ssims = []
        psnrs = []
        lpipss = []
        for idx in tqdm(range(len(renders)), desc=f"{split}/{method} metrics"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))
        full[method] = {
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
        }
        per_view[method] = {
            "SSIM": {name: val for name, val in zip(image_names, torch.tensor(ssims).tolist())},
            "PSNR": {name: val for name, val in zip(image_names, torch.tensor(psnrs).tolist())},
            "LPIPS": {name: val for name, val in zip(image_names, torch.tensor(lpipss).tolist())},
        }
        print(f"{split}/{method}: {json.dumps(full[method], sort_keys=True)}")
    return full, per_view


def _load_view_names(args: argparse.Namespace) -> set[str] | None:
    names: set[str] = set()
    for item in args.view_names or []:
        for token in str(item).replace(",", " ").split():
            token = token.strip()
            if token:
                names.add(token)
                names.add(Path(token).stem)
    if args.view_names_file:
        payload = json.loads(Path(args.view_names_file).read_text(encoding="utf-8"))
        value = payload
        if args.view_names_key:
            for key in args.view_names_key.split("."):
                if key:
                    value = value[key]
        if not isinstance(value, list):
            raise TypeError(f"view names in {args.view_names_file} must be a list, got {type(value).__name__}")
        for item in value:
            token = str(item).strip()
            if token:
                names.add(token)
                names.add(Path(token).stem)
    return names or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate render metrics on a selected split directory.")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--view_names", nargs="*", default=None)
    parser.add_argument(
        "--view_names_file",
        default="",
        help="Optional JSON file containing a list of split-local view stems to evaluate.",
    )
    parser.add_argument(
        "--view_names_key",
        default="policy_val_views",
        help="Dot-separated key inside --view_names_file. Use an empty string if the file itself is a list.",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--per_view_output", default="")
    parser.add_argument(
        "--merge_model_results",
        action="store_true",
        help="Merge selected-method test metrics back into model_path/results.json and per_view.json.",
    )
    args = parser.parse_args()
    torch.cuda.set_device(torch.device("cuda:0"))
    methods = set(args.methods) if args.methods else None
    view_names = _load_view_names(args)
    if view_names is not None:
        print(f"[eval] restricting {args.split} evaluation to requested view names/stems: {len(view_names)} tokens")
    full, per_view = evaluate_split(Path(args.model_path), args.split, methods, view_names=view_names)
    output = Path(args.output) if args.output else Path(args.model_path) / f"{args.split}_results.json"
    per_view_output = (
        Path(args.per_view_output) if args.per_view_output else Path(args.model_path) / f"{args.split}_per_view.json"
    )
    output.write_text(json.dumps(full, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    per_view_output.write_text(json.dumps(per_view, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.merge_model_results and args.split == "test":
        model_path = Path(args.model_path)
        result_path = model_path / "results.json"
        existing = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        existing.update(full)
        result_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        per_view_path = model_path / "per_view.json"
        existing_per_view = json.loads(per_view_path.read_text(encoding="utf-8")) if per_view_path.is_file() else {}
        existing_per_view.update(per_view)
        per_view_path.write_text(json.dumps(existing_per_view, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
