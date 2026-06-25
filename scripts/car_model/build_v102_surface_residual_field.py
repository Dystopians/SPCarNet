#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from render import _assert_camera_matches_bank, _camera_record, _downsample_rend_ids_nearest, _sha256, _torch_load
from scene import Scene
from triangle_renderer import TriangleModel, render


def _dtype(name: str) -> torch.dtype:
    name = str(name).strip().lower()
    if name in {"float32", "fp32"}:
        return torch.float32
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"unsupported residual dtype: {name}")


def _load_scene(model_path: Path, iteration: int, renderer_scaling: int):
    old_argv = sys.argv[:]
    try:
        sys.argv = ["build_v102_surface_residual_field", "-m", str(model_path)]
        parser = ArgumentParser()
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        args = get_combined_args(parser)
        dataset = model.extract(args)
        pipe = pipeline.extract(args)
    finally:
        sys.argv = old_argv
    triangles = TriangleModel(dataset.sh_degree)
    triangles.scaling = int(renderer_scaling)
    scene = Scene(args=dataset, triangles=triangles, init_opacity=None, set_sigma=None, load_iteration=iteration, shuffle=False)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    return dataset, pipe, triangles, scene, background


def build_field(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model_path)
    delta_bank_path = Path(args.delta_bank_path)
    output_field = Path(args.output_field)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not delta_bank_path.is_file():
        raise FileNotFoundError(delta_bank_path)

    delta_bank = _torch_load(delta_bank_path)
    if str(delta_bank.get("bank_type", "")) != "v102_preprojected_delta_bank":
        raise RuntimeError("delta bank is not a v102 preprojected delta bank")
    bank_endpoint = str(delta_bank.get("endpoint_method", "") or "")
    if bank_endpoint and bank_endpoint != str(args.endpoint_method):
        raise RuntimeError(
            f"delta bank endpoint_method mismatch: bank={bank_endpoint} cli={args.endpoint_method}"
        )
    bank_split = str(delta_bank.get("split", "") or "")
    if bank_split and bank_split != str(args.split):
        raise RuntimeError(f"delta bank split mismatch: bank={bank_split} cli={args.split}")
    deltas = delta_bank.get("deltas", {})
    if not isinstance(deltas, dict) or not deltas:
        raise RuntimeError("delta bank contains no deltas")
    frames = delta_bank.get("frames", {})
    if isinstance(frames, list):
        frames = {str(row.get("frame", row.get("name", idx))): row for idx, row in enumerate(frames)}
    if not isinstance(frames, dict):
        frames = {}
    missing_frame_meta = [str(key) for key in deltas.keys() if str(key) not in frames]
    if missing_frame_meta:
        raise RuntimeError(f"delta bank is missing frame metadata for {len(missing_frame_meta)} deltas")

    dataset, pipe, triangles, scene, background = _load_scene(
        model_path,
        int(args.iteration),
        int(args.renderer_scaling),
    )
    views = scene.getTestCameras() if str(args.split) == "test" else scene.getTrainCameras()
    triangle_count = int(triangles.get_triangle_indices.shape[0])
    sums = torch.zeros((triangle_count, 3), dtype=torch.float32, device="cpu")
    counts = torch.zeros((triangle_count,), dtype=torch.float32, device="cpu")
    view_reports = []
    started = time.time()
    for idx, view in enumerate(tqdm(views, desc=f"v102 surface field {args.split}")):
        key = f"{idx:05d}"
        if key not in deltas:
            raise RuntimeError(f"missing delta for target frame {key}")
        _assert_camera_matches_bank(_camera_record(idx, view), frames.get(key, {}).get("target_camera", {}), key)
        with torch.no_grad():
            pkg = render(view, triangles, pipe, background)
        rendering = pkg["render"]
        if "rend_ids" not in pkg or pkg["rend_ids"] is None:
            raise RuntimeError("renderer package missing rend_ids; cannot build surface residual field")
        ids = _downsample_rend_ids_nearest(pkg["rend_ids"], rendering.shape[-2:]).detach().cpu().long()
        delta = deltas[key].detach().cpu().float()
        if tuple(delta.shape) != tuple(rendering.shape):
            raise RuntimeError(f"delta/render shape mismatch for {key}: delta={tuple(delta.shape)} render={tuple(rendering.shape)}")
        valid = (ids >= 0) & (ids < triangle_count)
        flat_ids = ids[valid].reshape(-1)
        values = delta.permute(1, 2, 0)[valid].reshape(-1, 3)
        sums.index_add_(0, flat_ids, values)
        counts.index_add_(0, flat_ids, torch.ones((int(flat_ids.numel()),), dtype=torch.float32))
        view_reports.append(
            {
                "frame": key,
                "valid_fraction": float(valid.float().mean().item()),
                "unique_triangles": int(torch.unique(flat_ids).numel()) if flat_ids.numel() else 0,
                "mean_abs_delta": float(delta.abs().mean().item()),
                "camera_validated": True,
            }
        )
        del pkg, rendering, ids, delta, valid, flat_ids, values

    min_count = int(args.min_count)
    valid_triangles = counts >= min_count
    residuals = torch.zeros_like(sums)
    residuals[valid_triangles] = sums[valid_triangles] / counts[valid_triangles].unsqueeze(1).clamp_min(1.0)
    residuals = residuals.to(dtype=_dtype(args.residual_dtype)).contiguous()
    counts_out = counts.to(dtype=torch.int32).contiguous()
    endpoint_report = Path(str(delta_bank.get("endpoint_report", "")))
    endpoint_report_sha = str(delta_bank.get("endpoint_report_sha256", "") or "")
    payload = {
        "schema_version": 1,
        "field_type": "v102_surface_residual_field",
        "created_at_unix": time.time(),
        "model_path": str(model_path),
        "split": str(args.split),
        "iteration": int(args.iteration),
        "endpoint_method": str(delta_bank.get("endpoint_method", args.endpoint_method)),
        "source_bank_split": bank_split,
        "endpoint_report": str(endpoint_report),
        "endpoint_report_sha256": endpoint_report_sha,
        "source_delta_bank": str(delta_bank_path),
        "source_delta_bank_sha256": _sha256(delta_bank_path),
        "source_target_frames": int(len(views)),
        "triangle_count": int(triangle_count),
        "valid_triangles": int(valid_triangles.sum().item()),
        "min_count": int(min_count),
        "residual_dtype": str(args.residual_dtype),
        "triangle_residuals": residuals,
        "triangle_counts": counts_out,
        "view_reports": view_reports,
        "camera_validation": "strict_target_camera_match",
        "elapsed_sec": float(time.time() - started),
        "note": "Surface-addressed residual field distilled from v102 preprojected deltas and renderer triangle ids. It stores no target GT.",
    }
    output_field.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_field)
    manifest = {
        "schema_version": 1,
        "field_path": str(output_field),
        "field_sha256": _sha256(output_field),
        "source_delta_bank": str(delta_bank_path),
        "source_delta_bank_sha256": _sha256(delta_bank_path),
        "triangle_count": int(triangle_count),
        "valid_triangles": int(valid_triangles.sum().item()),
        "valid_triangle_fraction": float(valid_triangles.float().mean().item()),
        "source_target_frames": int(len(views)),
        "min_count": int(min_count),
        "residual_dtype": str(args.residual_dtype),
        "endpoint_method": str(delta_bank.get("endpoint_method", args.endpoint_method)),
        "source_bank_split": bank_split,
        "camera_validation": "strict_target_camera_match",
        "elapsed_sec": float(time.time() - started),
        "note": payload["note"],
    }
    manifest_path = output_field.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"field": str(output_field), "manifest": str(manifest_path), "valid_triangles": manifest["valid_triangles"]}, indent=2, sort_keys=True))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a v102 surface-addressed residual field from a preprojected delta bank.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--delta_bank_path", required=True)
    parser.add_argument("--output_field", required=True)
    parser.add_argument("--endpoint_method", default="ours_26000_v100_checkpoint_attached_ela_endpoint")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--split", default="test", choices=("test", "train"))
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--residual_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--renderer_scaling", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    build_field(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
