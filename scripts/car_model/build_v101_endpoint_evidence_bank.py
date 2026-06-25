#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.benchmark_ela_postprocess_runtime import _read_report, _select_support_frames
from utils.evidence_lumigraph_adapter import load_split_frames, read_depth_tensor, read_image_tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_payload(frame) -> dict[str, Any]:
    camera = frame.camera
    return {
        "idx": int(camera.idx),
        "image_name": str(camera.image_name),
        "width": int(camera.width),
        "height": int(camera.height),
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "camera_center": [float(x) for x in camera.camera_center],
        "world_view_transform": [[float(v) for v in row] for row in camera.world_view_transform],
    }


def _tensor_dtype(name: str) -> torch.dtype:
    name = str(name).strip().lower()
    if name in {"float32", "fp32"}:
        return torch.float32
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"unsupported tensor dtype: {name}")


def build_bank(args: argparse.Namespace) -> dict[str, Any]:
    base_model = Path(args.base_model_path).resolve()
    endpoint_dir = (
        Path(args.output_model_path).resolve()
        / "point_cloud"
        / f"iteration_{int(args.iteration)}"
        / "render_residual_endpoint"
        / args.endpoint_method
    )
    report_path = Path(args.source_report).resolve() if args.source_report else endpoint_dir / "ela_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = _read_report(str(report_path))
    train_frames = load_split_frames(base_model, "train", args.base_method_name)
    support_frames, support_source, missing_support_names = _select_support_frames(train_frames, report)
    if int(args.max_support_frames) > 0:
        support_frames = support_frames[: int(args.max_support_frames)]
    if not support_frames:
        raise RuntimeError("no support frames selected for v101 evidence bank")

    residual_dtype = _tensor_dtype(args.residual_dtype)
    depth_dtype = _tensor_dtype(args.depth_dtype)
    residuals: dict[str, torch.Tensor] = {}
    depths: dict[str, torch.Tensor] = {}
    frames = []
    started = time.time()
    for frame in tqdm(support_frames, desc="v101 evidence bank"):
        render = read_image_tensor(frame.render_path, device="cpu")
        gt = read_image_tensor(frame.gt_path, device="cpu")
        depth = read_depth_tensor(frame.depth_path, device="cpu")
        residual = gt - render
        clip = float(args.residual_clip)
        if clip > 0:
            residual = torch.clamp(residual, -clip, clip)
        residuals[frame.name] = residual.to(dtype=residual_dtype).contiguous()
        depths[frame.name] = depth.to(dtype=depth_dtype).contiguous()
        frames.append(
            {
                "idx": int(frame.idx),
                "name": str(frame.name),
                "camera": _camera_payload(frame),
                "render_sha256": _sha256(frame.render_path),
                "gt_sha256": _sha256(frame.gt_path),
                "depth_sha256": _sha256(frame.depth_path),
                "height": int(depth.shape[0]),
                "width": int(depth.shape[1]),
            }
        )

    endpoint_dir.mkdir(parents=True, exist_ok=True)
    bank_path = Path(args.output_bank).resolve() if args.output_bank else endpoint_dir / "v101_evidence_bank.pt"
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = bank_path.with_suffix(".manifest.json")
    payload = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "base_model": str(base_model),
        "base_method": str(args.base_method_name),
        "source_report": str(report_path),
        "source_report_sha256": _sha256(report_path),
        "support_source": str(support_source),
        "missing_report_support_names": missing_support_names,
        "residual_clip": float(args.residual_clip),
        "residual_dtype": str(args.residual_dtype),
        "depth_dtype": str(args.depth_dtype),
        "frames": frames,
        "residuals": residuals,
        "depths": depths,
    }
    torch.save(payload, bank_path)
    manifest = {
        "schema_version": 1,
        "bank_path": str(bank_path),
        "bank_sha256": _sha256(bank_path),
        "base_model": str(base_model),
        "base_method": str(args.base_method_name),
        "source_report": str(report_path),
        "source_report_sha256": _sha256(report_path),
        "support_source": str(support_source),
        "support_frames": int(len(frames)),
        "missing_report_support_names": missing_support_names,
        "residual_clip": float(args.residual_clip),
        "residual_dtype": str(args.residual_dtype),
        "depth_dtype": str(args.depth_dtype),
        "elapsed_sec": float(time.time() - started),
        "note": "Train-derived residual/depth/camera evidence bank. It contains no held-out target GT.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (endpoint_dir / "v101_evidence_bank_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a checkpoint-attached v101 train evidence bank.")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--endpoint_method", default="ours_26000_v100_checkpoint_attached_ela_endpoint")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--base_method_name", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument("--source_report", default="")
    parser.add_argument("--output_bank", default="")
    parser.add_argument("--residual_clip", type=float, default=0.25)
    parser.add_argument("--residual_dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--depth_dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--max_support_frames", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    manifest = build_bank(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
