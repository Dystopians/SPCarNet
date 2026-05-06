#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torchvision
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from triangle_renderer import TriangleModel, render
from utils.evidence_lumigraph_adapter import CameraRecord, save_camera_index
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal


def _camera_record(idx: int, view) -> CameraRecord:
    width = int(view.image_width)
    height = int(view.image_height)
    return CameraRecord(
        idx=int(idx),
        image_name=str(getattr(view, "image_name", f"{idx:05d}")),
        width=width,
        height=height,
        fx=float(fov2focal(float(view.FoVx), width)),
        fy=float(fov2focal(float(view.FoVy), height)),
        camera_center=tuple(float(x) for x in view.camera_center.detach().cpu().tolist()),
        world_view_transform=tuple(
            tuple(float(v) for v in row)
            for row in view.world_view_transform.detach().cpu().tolist()
        ),
    )


def render_set(model_path: str, name: str, iteration: int, views, triangles, pipeline, background, *, save_depth: bool):
    method_dir = Path(model_path) / name / f"ours_{iteration}"
    render_path = method_dir / "renders"
    gts_path = method_dir / "gt"
    depth_path = method_dir / "depths"
    render_path.mkdir(parents=True, exist_ok=True)
    gts_path.mkdir(parents=True, exist_ok=True)
    if save_depth:
        depth_path.mkdir(parents=True, exist_ok=True)

    camera_records: list[CameraRecord] = []
    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        pkg = render(view, triangles, pipeline, background)
        rendering = pkg["render"]
        gt = view.original_image[0:3, :, :]
        key = f"{idx:05d}"
        torchvision.utils.save_image(rendering, render_path / f"{key}.png")
        torchvision.utils.save_image(gt, gts_path / f"{key}.png")
        if save_depth:
            depth = pkg.get("surf_depth", None)
            if depth is None:
                raise RuntimeError("render package did not include surf_depth")
            depth_np = depth[0].detach().float().cpu().numpy().astype(np.float32)
            np.save(depth_path / f"{key}.npy", depth_np)
        camera_records.append(_camera_record(idx, view))
    save_camera_index(camera_records, method_dir / "camera_index.json")


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, save_depth: bool):
    with torch.no_grad():
        triangles = TriangleModel(dataset.sh_degree)
        triangles.scaling = 4
        scene = Scene(
            args=dataset,
            triangles=triangles,
            init_opacity=None,
            set_sigma=None,
            load_iteration=iteration,
            shuffle=False,
        )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), triangles, pipeline, background, save_depth=save_depth)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), triangles, pipeline, background, save_depth=save_depth)


def main() -> int:
    parser = ArgumentParser(description="Render RGB/GT/depth/camera evidence maps for ELA.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--no_depth", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering evidence maps " + args.model_path)
    safe_state(args.quiet)
    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        save_depth=not bool(args.no_depth),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
