#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
import json
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


def _frustum_active_mask(triangles: TriangleModel, view, margin: float) -> torch.Tensor:
    vertices = triangles.get_vertices
    faces = triangles.get_triangle_indices.long()
    m = view.full_proj_transform.flatten()
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    clip_x = m[0] * x + m[4] * y + m[8] * z + m[12]
    clip_y = m[1] * x + m[5] * y + m[9] * z + m[13]
    clip_w = m[3] * x + m[7] * y + m[11] * z + m[15]
    inv_w = 1.0 / (clip_w + 1e-8)
    ndc_x = clip_x * inv_w
    ndc_y = clip_y * inv_w
    finite = torch.isfinite(ndc_x) & torch.isfinite(ndc_y) & torch.isfinite(clip_w) & (clip_w > 1e-6)

    ndc_x = torch.where(finite, ndc_x, torch.full_like(ndc_x, 1e6))
    ndc_y = torch.where(finite, ndc_y, torch.full_like(ndc_y, 1e6))
    fx = ndc_x[faces]
    fy = ndc_y[faces]
    fvalid = finite[faces]
    lo = -1.0 - float(margin)
    hi = 1.0 + float(margin)
    bbox_overlap = (
        (torch.amin(fx, dim=1) <= hi)
        & (torch.amax(fx, dim=1) >= lo)
        & (torch.amin(fy, dim=1) <= hi)
        & (torch.amax(fy, dim=1) >= lo)
    )
    return (torch.any(fvalid, dim=1) & bbox_overlap).contiguous()


def render_set(
    model_path: str,
    name: str,
    iteration: int,
    views,
    triangles,
    pipeline,
    background,
    *,
    save_depth: bool,
    method_name: str | None = None,
    skip_failed_views: bool = False,
    frustum_cull: bool = False,
    frustum_margin: float = 0.5,
    save_surface_maps: bool = False,
):
    method_dir = Path(model_path) / name / (method_name or f"ours_{iteration}")
    render_path = method_dir / "renders"
    gts_path = method_dir / "gt"
    depth_path = method_dir / "depths"
    surface_map_path = method_dir / "surface_maps"
    render_path.mkdir(parents=True, exist_ok=True)
    gts_path.mkdir(parents=True, exist_ok=True)
    if save_depth:
        depth_path.mkdir(parents=True, exist_ok=True)
    if save_surface_maps:
        surface_map_path.mkdir(parents=True, exist_ok=True)

    camera_records: list[CameraRecord] = []
    failures: list[dict[str, str | int]] = []
    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        if frustum_cull:
            active_mask = _frustum_active_mask(triangles, view, float(frustum_margin))
            triangles.set_temporary_active_mask(active_mask)
        try:
            pkg = render(view, triangles, pipeline, background)
        except RuntimeError as exc:
            triangles.clear_temporary_active_mask()
            if not skip_failed_views:
                raise
            message = str(exc).splitlines()[0]
            failures.append(
                {
                    "idx": int(idx),
                    "image_name": str(getattr(view, "image_name", f"{idx:05d}")),
                    "error": message,
                }
            )
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            print(f"[EvidenceRender] skipped {name}/{idx:05d}: {message}", flush=True)
            continue
        finally:
            triangles.clear_temporary_active_mask()
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
        if save_surface_maps:
            face_ids = pkg.get("rend_ids", None)
            alpha = pkg.get("rend_alpha", None)
            surf_depth = pkg.get("surf_depth", None)
            if face_ids is None or alpha is None or surf_depth is None:
                raise RuntimeError("render package did not include rend_ids/rend_alpha/surf_depth")
            face_ids_t = face_ids.detach().float()
            if face_ids_t.ndim == 3:
                face_ids_t = face_ids_t.unsqueeze(0)
            face_ids_hw = torch.nn.functional.interpolate(
                face_ids_t,
                size=(int(view.image_height), int(view.image_width)),
                mode="nearest",
            ).squeeze().detach().cpu().numpy().astype(np.int32)
            alpha_hw = alpha.detach().float().squeeze().cpu().numpy().astype(np.float16)
            depth_hw = surf_depth.detach().float().squeeze().cpu().numpy().astype(np.float32)
            np.savez_compressed(
                surface_map_path / f"{key}.npz",
                face_id=face_ids_hw,
                alpha=alpha_hw,
                depth=depth_hw,
                camera_center=view.camera_center.detach().float().cpu().numpy().astype(np.float32),
                image_name=str(getattr(view, "image_name", key)),
            )
        camera_records.append(_camera_record(idx, view))
        del pkg, rendering, gt
        if save_depth:
            del depth, depth_np
        torch.cuda.empty_cache()
    (method_dir / "render_failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    if not camera_records:
        first_error = failures[0].get("error", "unknown") if failures else "no failures recorded"
        raise RuntimeError(
            f"no evidence views rendered for split={name}; "
            f"skipped_failures={len(failures)}; first_error={first_error}"
        )
    save_camera_index(camera_records, method_dir / "camera_index.json")


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    save_depth: bool,
        method_name: str | None = None,
        skip_failed_views: bool = False,
        frustum_cull: bool = False,
        frustum_margin: float = 0.5,
        save_surface_maps: bool = False,
):
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
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                triangles,
                pipeline,
                background,
                save_depth=save_depth,
                method_name=method_name,
                skip_failed_views=skip_failed_views,
                frustum_cull=frustum_cull,
                frustum_margin=frustum_margin,
                save_surface_maps=save_surface_maps,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                triangles,
                pipeline,
                background,
                save_depth=save_depth,
                method_name=method_name,
                skip_failed_views=skip_failed_views,
                frustum_cull=frustum_cull,
                frustum_margin=frustum_margin,
                save_surface_maps=save_surface_maps,
            )


def main() -> int:
    parser = ArgumentParser(description="Render RGB/GT/depth/camera evidence maps for ELA.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--no_depth", action="store_true")
    parser.add_argument("--method_name", default="")
    parser.add_argument("--skip_failed_views", action="store_true")
    parser.add_argument("--frustum_cull", action="store_true")
    parser.add_argument("--frustum_margin", default=0.5, type=float)
    parser.add_argument(
        "--save_surface_maps",
        action="store_true",
        help="Also save face-id/alpha/depth surface maps for surface-attached residual adapters.",
    )
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
        method_name=args.method_name or None,
        skip_failed_views=bool(args.skip_failed_views),
        frustum_cull=bool(args.frustum_cull),
        frustum_margin=float(args.frustum_margin),
        save_surface_maps=bool(args.save_surface_maps),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
