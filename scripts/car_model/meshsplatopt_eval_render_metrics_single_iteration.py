#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lpipsPyTorch import lpips  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


def _read_image(path: Path) -> torch.Tensor:
    return tf.to_tensor(Image.open(path)).unsqueeze(0)[:, :3, :, :].cuda()


def evaluate_method(model_path: Path, iteration: int) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    method = f"ours_{int(iteration)}"
    renders_dir = model_path / "test" / method / "renders"
    gt_dir = model_path / "test" / method / "gt"
    if not renders_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing render/gt directories for {model_path}/test/{method}")

    image_names = sorted(path.name for path in renders_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not image_names:
        raise FileNotFoundError(f"No rendered images found in {renders_dir}")

    ssims: list[float] = []
    psnrs: list[float] = []
    lpipss: list[float] = []
    per_view = {"SSIM": {}, "PSNR": {}, "LPIPS": {}}
    for name in tqdm(image_names, desc=f"metrics {method}"):
        gt_path = gt_dir / name
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing GT image for render {name}: {gt_path}")
        render = _read_image(renders_dir / name)
        gt = _read_image(gt_path)
        ssim_value = float(ssim(render, gt).item())
        psnr_value = float(psnr(render, gt).item())
        lpips_value = float(lpips(render, gt, net_type="vgg").item())
        ssims.append(ssim_value)
        psnrs.append(psnr_value)
        lpipss.append(lpips_value)
        per_view["SSIM"][name] = ssim_value
        per_view["PSNR"][name] = psnr_value
        per_view["LPIPS"][name] = lpips_value

    summary = {
        "SSIM": float(torch.tensor(ssims).mean().item()),
        "PSNR": float(torch.tensor(psnrs).mean().item()),
        "LPIPS": float(torch.tensor(lpipss).mean().item()),
    }
    return summary, per_view


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate render metrics for exactly one saved iteration.")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    model = Path(args.model_path)
    method = f"ours_{int(args.iteration)}"
    summary, per_view = evaluate_method(model, args.iteration)

    results_path = model / "results.json"
    per_view_path = model / "per_view.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else {}
    per_view_payload = json.loads(per_view_path.read_text(encoding="utf-8")) if per_view_path.is_file() else {}
    results[method] = summary
    per_view_payload[method] = per_view
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    per_view_path.write_text(json.dumps(per_view_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({method: summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
