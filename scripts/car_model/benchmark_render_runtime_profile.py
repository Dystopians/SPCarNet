#!/usr/bin/env python3
"""Measure render-only runtime and CUDA memory for MeshSplatting checkpoints.

This benchmark intentionally does not save PNG renders. It times the renderer
forward pass on selected train/test cameras and records CUDA peak memory. Use it
for clean/compact/checkpoint-baked artifacts; render-time adapters that apply
extra image-space or evidence-lumigraph work need a separate end-to-end profile.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from triangle_renderer import TriangleModel, render
from utils.general_utils import safe_state


def _count_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return int(total)


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def _stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _select_views(views: list[Any], max_views: int, stride: int) -> list[Any]:
    stride = max(1, int(stride))
    selected = list(views)[::stride]
    if int(max_views) > 0:
        selected = selected[: int(max_views)]
    return selected


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Render Runtime Profile",
        "",
        "This is a render-only CUDA profile. It does not save PNG files and does not include `metrics.py`, LPIPS, disk I/O, or render-time SPCarNet Evidence Lumigraph Adapter post-processing.",
        "",
        "## Summary",
        "",
        f"- label: `{payload['label']}`",
        f"- model path: `{payload['model_path']}`",
        f"- split: `{payload['split']}`",
        f"- iteration: `{payload['loaded_iteration']}`",
        f"- views: `{payload['num_views']}`",
        f"- repeats: `{payload['repeats']}`",
        f"- mean elapsed sec: `{payload['elapsed_sec_mean']:.6f}`",
        f"- mean ms/view: `{payload['ms_per_view_mean']:.6f}`",
        f"- mean FPS: `{payload['fps_mean']:.6f}`",
        f"- peak allocated MiB max: `{payload['peak_allocated_mib_max']:.3f}`",
        f"- peak reserved MiB max: `{payload['peak_reserved_mib_max']:.3f}`",
        f"- triangles: `{payload['triangles']}`",
        f"- vertices: `{payload['vertices']}`",
        f"- checkpoint bytes: `{payload['checkpoint_bytes']}`",
        "",
        "## Repeats",
        "",
        "| repeat | elapsed sec | ms/view | FPS | peak allocated MiB | peak reserved MiB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["repeat_rows"]:
        lines.append(
            "| {repeat} | {elapsed_sec:.6f} | {ms_per_view:.6f} | {fps:.6f} | {peak_allocated_mib:.3f} | {peak_reserved_mib:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Scope Note",
            "",
            "- Use this table for clean MeshSplatting, compact-only checkpoints, and checkpoint-baked candidates.",
            "- Do not use it to claim Phase-J adapter end-to-end speed unless the adapter itself is profiled in the measured command.",
            "- For paper-facing numbers, rerun with all test views, at least 3 repeats, and a fixed visible GPU.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = ArgumentParser(description="Render-only runtime/FPS/VRAM benchmark for MeshSplatting checkpoints")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--max_views", type=int, default=0, help="0 means all selected split views")
    parser.add_argument("--view_stride", type=int, default=1)
    parser.add_argument("--warmup_views", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--out_md", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    torch.cuda.set_device(int(args.gpu))
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    with torch.no_grad():
        triangles = TriangleModel(dataset.sh_degree)
        triangles.scaling = 4
        scene = Scene(
            args=dataset,
            triangles=triangles,
            init_opacity=None,
            set_sigma=None,
            load_iteration=int(args.iteration),
            shuffle=False,
        )
        views = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
        selected_views = _select_views(views, int(args.max_views), int(args.view_stride))
        if not selected_views:
            raise RuntimeError(f"No views selected for split={args.split}")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        warmup = selected_views[: max(0, min(int(args.warmup_views), len(selected_views)))]
        for view in warmup:
            _ = render(view, triangles, pipe, background)["render"]
        torch.cuda.synchronize()

        repeat_rows: list[dict[str, float]] = []
        checksum = 0.0
        for repeat_idx in range(max(1, int(args.repeats))):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            last_render = None
            for view in tqdm(selected_views, desc=f"render profile repeat {repeat_idx + 1}"):
                last_render = render(view, triangles, pipe, background)["render"]
            torch.cuda.synchronize()
            elapsed = max(1e-9, time.perf_counter() - start)
            if last_render is not None:
                checksum += float(last_render.detach().mean().item())
            peak_alloc = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            peak_reserved = float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0)
            repeat_rows.append(
                {
                    "repeat": int(repeat_idx + 1),
                    "elapsed_sec": float(elapsed),
                    "ms_per_view": float(elapsed * 1000.0 / len(selected_views)),
                    "fps": float(len(selected_views) / elapsed),
                    "peak_allocated_mib": peak_alloc,
                    "peak_reserved_mib": peak_reserved,
                }
            )

    elapsed_values = [float(row["elapsed_sec"]) for row in repeat_rows]
    fps_values = [float(row["fps"]) for row in repeat_rows]
    ms_values = [float(row["ms_per_view"]) for row in repeat_rows]
    alloc_values = [float(row["peak_allocated_mib"]) for row in repeat_rows]
    reserved_values = [float(row["peak_reserved_mib"]) for row in repeat_rows]
    point_cloud_dir = Path(dataset.model_path) / "point_cloud" / f"iteration_{scene.loaded_iter}"

    payload: dict[str, Any] = {
        "label": args.label or Path(dataset.model_path).name,
        "model_path": str(dataset.model_path),
        "split": str(args.split),
        "requested_iteration": int(args.iteration),
        "loaded_iteration": int(scene.loaded_iter),
        "num_views": int(len(selected_views)),
        "view_stride": int(args.view_stride),
        "warmup_views": int(len(warmup)),
        "repeats": int(len(repeat_rows)),
        "elapsed_sec_mean": _mean(elapsed_values),
        "elapsed_sec_stdev": _stdev(elapsed_values),
        "ms_per_view_mean": _mean(ms_values),
        "ms_per_view_stdev": _stdev(ms_values),
        "fps_mean": _mean(fps_values),
        "fps_stdev": _stdev(fps_values),
        "peak_allocated_mib_mean": _mean(alloc_values),
        "peak_allocated_mib_max": max(alloc_values) if alloc_values else float("nan"),
        "peak_reserved_mib_mean": _mean(reserved_values),
        "peak_reserved_mib_max": max(reserved_values) if reserved_values else float("nan"),
        "triangles": int(triangles._triangle_indices.shape[0]),
        "vertices": int(triangles.vertices.shape[0]),
        "checkpoint_bytes": _count_bytes(point_cloud_dir),
        "point_cloud_dir": str(point_cloud_dir),
        "checksum": float(checksum),
        "repeat_rows": repeat_rows,
        "scope": "render_only_no_png_no_metrics_no_adapter_postprocess",
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(out_md, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
