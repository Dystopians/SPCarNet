#!/usr/bin/env python3
"""Benchmark integrated MeshSplatting render + Phase-J ELA postprocess.

This profiler runs the renderer forward pass and the Evidence Lumigraph Adapter
in the same process for the same target views. It does not write PNGs and does
not run image metrics. Train/support evidence is read from existing artifacts,
while the target frame's base RGB/depth are supplied from the freshly rendered
tensor outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
import re
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from triangle_renderer import TriangleModel, render
from utils.evidence_lumigraph_adapter import FrameLoader, FrameRecord, adapt_frame, load_split_frames
from utils.general_utils import safe_state

from scripts.car_model.benchmark_ela_postprocess_runtime import (
    _alpha_calibrator_from_report,
    _alpha_from_report,
    _benefit_calibrator_from_report,
    _json_safe,
    _policy_from_report,
    _read_report,
    _select_support_frames,
)


class IntegratedFrameLoader(FrameLoader):
    """FrameLoader that overrides the current target render/depth tensors."""

    def __init__(
        self,
        *,
        device: torch.device,
    ) -> None:
        super().__init__(device=device)
        self._target_render_path: str | None = None
        self._target_depth_path: str | None = None
        self._target_render: torch.Tensor | None = None
        self._target_depth: torch.Tensor | None = None

    def set_target(
        self,
        *,
        target_render_path: Path,
        target_depth_path: Path,
        target_render: torch.Tensor,
        target_depth: torch.Tensor,
    ) -> None:
        self._target_render_path = str(target_render_path)
        self._target_depth_path = str(target_depth_path)
        self._target_render = target_render.to(device=self.device, dtype=torch.float32)
        depth = target_depth.to(device=self.device, dtype=torch.float32)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth.squeeze(0)
        self._target_depth = depth

    def render(self, path: str) -> torch.Tensor:  # type: ignore[override]
        if str(path) == self._target_render_path:
            if self._target_render is None:
                raise RuntimeError("Integrated target render was requested before set_target()")
            return self._target_render
        return super().render(path)

    def depth(self, path: str) -> torch.Tensor:  # type: ignore[override]
        if str(path) == self._target_depth_path:
            if self._target_depth is None:
                raise RuntimeError("Integrated target depth was requested before set_target()")
            return self._target_depth
        return super().depth(path)


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def _stdev(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _select_pairs(views: Sequence[Any], frames: Sequence[FrameRecord], max_views: int, stride: int) -> list[tuple[Any, FrameRecord]]:
    if len(views) != len(frames):
        raise RuntimeError(f"Scene cameras/views mismatch: {len(views)} cameras vs {len(frames)} FrameRecords")
    for idx, (view, frame) in enumerate(zip(views, frames)):
        view_name = str(getattr(view, "image_name", ""))
        frame_names = {str(frame.name), str(frame.camera.image_name), Path(str(frame.camera.image_name)).stem}
        if view_name not in frame_names and Path(view_name).stem not in frame_names:
            raise RuntimeError(
                "Scene camera and evidence frame order mismatch at index "
                f"{idx}: view.image_name={view_name!r}, frame.name={frame.name!r}, "
                f"frame.camera.image_name={frame.camera.image_name!r}"
            )
    stride = max(1, int(stride))
    pairs = list(zip(views, frames))[::stride]
    if int(max_views) > 0:
        pairs = pairs[: int(max_views)]
    if not pairs:
        raise RuntimeError("No integrated runtime views selected")
    return pairs


def _method_iteration(method_name: str) -> int | None:
    match = re.search(r"(?:^|_)ours_(\d+)(?:_|$)", method_name)
    return int(match.group(1)) if match else None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_peak_row(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"cuda_peak_allocated_mib": None, "cuda_peak_reserved_mib": None}
    return {
        "cuda_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
        "cuda_peak_reserved_mib": float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Integrated Phase-J Runtime Profile",
        "",
        "This profile measures renderer forward + Phase-J ELA `adapt_frame` in one process.",
        "It excludes PNG writes, image metrics, LPIPS, and policy calibration.",
        "",
        "## Summary",
        "",
        f"- label: `{payload['label']}`",
        f"- model path: `{payload['model_path']}`",
        f"- split: `{payload['split']}`",
        f"- iteration: `{payload['loaded_iteration']}`",
        f"- views: `{payload['num_views']}`",
        f"- repeats: `{payload['repeats']}`",
        f"- evidence max side: `{payload['evidence_max_side']}`",
        f"- mean ms/view: `{payload['ms_per_view_mean']:.6f}`",
        f"- mean FPS: `{payload['fps_mean']:.6f}`",
        f"- mean render ms/view: `{payload['render_ms_per_view_mean']:.6f}`",
        f"- mean adapter ms/view: `{payload['adapter_ms_per_view_mean']:.6f}`",
        f"- adapter/render ratio: `{payload['adapter_over_render_ratio_mean']:.6f}`",
        f"- peak allocated MiB max: `{payload['peak_allocated_mib_max']:.3f}`",
        f"- peak reserved MiB max: `{payload['peak_reserved_mib_max']:.3f}`",
        f"- triangles: `{payload['triangles']}`",
        f"- vertices: `{payload['vertices']}`",
        "",
        "## Repeats",
        "",
        "| repeat | elapsed sec | ms/view | FPS | render ms/view | adapter ms/view | peak allocated MiB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["repeat_rows"]:
        lines.append(
            "| {repeat} | {elapsed_sec:.6f} | {ms_per_view:.6f} | {fps:.6f} | {render_ms_per_view:.6f} | {adapter_ms_per_view:.6f} | {cuda_peak_allocated_mib:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Scope Note",
            "",
            "- Uses fresh renderer RGB/depth tensors for each target view.",
            "- Uses existing train/support evidence artifacts for support residuals.",
            "- Does not write adapted PNGs, so wall-clock image export remains a separate deployment concern.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated render + Phase-J ELA runtime benchmark")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--base_method_name", required=True)
    parser.add_argument("--ela_report", required=True)
    parser.add_argument("--max_views", type=int, default=1, help="0 means all selected split views")
    parser.add_argument("--view_stride", type=int, default=1)
    parser.add_argument("--warmup_views", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--evidence_max_side",
        type=int,
        default=0,
        help="Optional fast adapter path: compute ELA evidence warps at this maximum side before upsampling.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", default="")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    if int(args.max_views) < 0:
        parser.error("--max_views must be >= 0")
    if int(args.repeats) < 1:
        parser.error("--repeats must be >= 1")
    if int(args.evidence_max_side) < 0:
        parser.error("--evidence_max_side must be >= 0")

    torch.cuda.set_device(int(args.gpu))
    device = torch.device("cuda", int(args.gpu))
    safe_state(args.quiet)

    report = _read_report(args.ela_report)
    policy, policy_source = _policy_from_report(report)
    alpha, alpha_source = _alpha_from_report(report)
    benefit_calibrator = _benefit_calibrator_from_report(report)
    alpha_calibrator = _alpha_calibrator_from_report(report)
    expected_iteration = _method_iteration(str(args.base_method_name))
    if expected_iteration is not None and int(args.iteration) >= 0 and int(args.iteration) != expected_iteration:
        raise RuntimeError(
            f"--iteration {args.iteration} does not match --base_method_name {args.base_method_name!r}; "
            f"expected {expected_iteration}."
        )

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    model_path = Path(dataset.model_path)

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
        target_frames = load_split_frames(model_path, args.split, args.base_method_name)
        train_frames = load_split_frames(model_path, "train", args.base_method_name)
        support_frames, support_source, missing_support_names = _select_support_frames(train_frames, report)
        selected_pairs = _select_pairs(views, target_frames, int(args.max_views), int(args.view_stride))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)

        warmup = selected_pairs[: max(0, min(int(args.warmup_views), len(selected_pairs)))]
        for view, _frame in warmup:
            _ = render(view, triangles, pipe, background)["render"]
        _sync(device)

        repeat_rows: list[dict[str, Any]] = []
        checksum = 0.0
        for repeat_idx in range(max(1, int(args.repeats))):
            loader = IntegratedFrameLoader(device=device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            _sync(device)
            start = time.perf_counter()
            render_time = 0.0
            adapter_time = 0.0
            frame_infos: list[dict[str, Any]] = []
            for view, target in tqdm(selected_pairs, desc=f"integrated repeat {repeat_idx + 1}"):
                render_start = time.perf_counter()
                rendered = render(view, triangles, pipe, background)
                _sync(device)
                render_time += time.perf_counter() - render_start

                loader.set_target(
                    target_render_path=target.render_path,
                    target_depth_path=target.depth_path,
                    target_render=rendered["render"].detach(),
                    target_depth=rendered["surf_depth"].detach(),
                )
                adapter_start = time.perf_counter()
                adapted, info = adapt_frame(
                    target,
                    support_frames,
                    k=int(policy["k"]),
                    alpha=float(alpha),
                    mode=str(policy["mode"]),
                    residual_clip=float(policy["residual_clip"]),
                    min_confidence=float(policy["min_confidence"]),
                    depth_abs_tol=float(policy["depth_abs_tol"]),
                    depth_rel_tol=float(policy["depth_rel_tol"]),
                    direction_weight=float(policy["direction_weight"]),
                    benefit_calibrator=benefit_calibrator,
                    alpha_calibrator=alpha_calibrator,
                    edge_gate=bool(policy["edge_gate"]),
                    edge_gate_quantile=float(policy["edge_gate_quantile"]),
                    edge_gate_min=float(policy["edge_gate_min"]),
                    edge_gate_dilate=int(policy["edge_gate_dilate"]),
                    local_trust_gate=bool(policy["local_trust_gate"]),
                    local_trust_min_supports=int(policy["local_trust_min_supports"]),
                    local_trust_max_residual_std=float(policy["local_trust_max_residual_std"]),
                    local_trust_min_agreement=float(policy["local_trust_min_agreement"]),
                    local_trust_agreement_scale=float(policy["local_trust_agreement_scale"]),
                    local_trust_confidence_quantile=float(policy["local_trust_confidence_quantile"]),
                    local_trust_min_confidence=float(policy["local_trust_min_confidence"]),
                    local_trust_mode=str(policy["local_trust_mode"]),
                    local_trust_min_weight=float(policy["local_trust_min_weight"]),
                    evidence_max_side=int(args.evidence_max_side),
                    loader=loader,
                    device=device,
                )
                _sync(device)
                adapter_time += time.perf_counter() - adapter_start
                checksum += float(adapted.detach().mean().item())
                frame_infos.append({"frame": target.name, **info})
            _sync(device)
            elapsed = max(time.perf_counter() - start, 1e-9)
            row: dict[str, Any] = {
                "repeat": int(repeat_idx + 1),
                "elapsed_sec": float(elapsed),
                "ms_per_view": float(elapsed * 1000.0 / len(selected_pairs)),
                "fps": float(len(selected_pairs) / elapsed),
                "render_elapsed_sec": float(render_time),
                "adapter_elapsed_sec": float(adapter_time),
                "render_ms_per_view": float(render_time * 1000.0 / len(selected_pairs)),
                "adapter_ms_per_view": float(adapter_time * 1000.0 / len(selected_pairs)),
                "adapter_over_render_ratio": float(adapter_time / max(render_time, 1e-9)),
                "frames": frame_infos,
            }
            row.update(_cuda_peak_row(device))
            repeat_rows.append(row)

    elapsed_values = [float(row["elapsed_sec"]) for row in repeat_rows]
    fps_values = [float(row["fps"]) for row in repeat_rows]
    ms_values = [float(row["ms_per_view"]) for row in repeat_rows]
    render_ms_values = [float(row["render_ms_per_view"]) for row in repeat_rows]
    adapter_ms_values = [float(row["adapter_ms_per_view"]) for row in repeat_rows]
    ratio_values = [float(row["adapter_over_render_ratio"]) for row in repeat_rows]
    alloc_values = [float(row["cuda_peak_allocated_mib"]) for row in repeat_rows if row.get("cuda_peak_allocated_mib") is not None]
    reserved_values = [float(row["cuda_peak_reserved_mib"]) for row in repeat_rows if row.get("cuda_peak_reserved_mib") is not None]

    payload: dict[str, Any] = {
        "benchmark": "integrated_phasej_runtime",
        "scope": "renderer_forward_plus_adapt_frame_no_png_no_metrics_no_policy_calibration",
        "command": sys.argv,
        "label": args.label or model_path.name,
        "model_path": str(model_path),
        "split": str(args.split),
        "base_method_name": str(args.base_method_name),
        "ela_report": str(args.ela_report),
        "policy_source": policy_source,
        "alpha_source": alpha_source,
        "requested_iteration": int(args.iteration),
        "loaded_iteration": int(scene.loaded_iter),
        "num_views": int(len(selected_pairs)),
        "available_target_frame_count": int(len(target_frames)),
        "view_stride": int(args.view_stride),
        "warmup_views": int(len(warmup)),
        "repeats": int(len(repeat_rows)),
        "evidence_max_side": int(args.evidence_max_side),
        "support_frame_count": int(len(support_frames)),
        "support_source": support_source,
        "missing_report_support_names": missing_support_names,
        "target_frame_names": [frame.name for _view, frame in selected_pairs],
        "alpha": float(alpha),
        "k": int(policy["k"]),
        "mode": str(policy["mode"]),
        "depth_abs_tol": float(policy["depth_abs_tol"]),
        "depth_rel_tol": float(policy["depth_rel_tol"]),
        "residual_clip": float(policy["residual_clip"]),
        "direction_weight": float(policy["direction_weight"]),
        "benefit_calibrator_loaded": benefit_calibrator is not None,
        "alpha_calibrator_loaded": alpha_calibrator is not None,
        "elapsed_sec_mean": _mean(elapsed_values),
        "elapsed_sec_stdev": _stdev(elapsed_values),
        "ms_per_view_mean": _mean(ms_values),
        "ms_per_view_stdev": _stdev(ms_values),
        "fps_mean": _mean(fps_values),
        "fps_stdev": _stdev(fps_values),
        "render_ms_per_view_mean": _mean(render_ms_values),
        "adapter_ms_per_view_mean": _mean(adapter_ms_values),
        "adapter_over_render_ratio_mean": _mean(ratio_values),
        "peak_allocated_mib_mean": _mean(alloc_values),
        "peak_allocated_mib_max": max(alloc_values) if alloc_values else float("nan"),
        "peak_reserved_mib_mean": _mean(reserved_values),
        "peak_reserved_mib_max": max(reserved_values) if reserved_values else float("nan"),
        "triangles": int(triangles._triangle_indices.shape[0]),
        "vertices": int(triangles.vertices.shape[0]),
        "checksum": float(checksum),
        "repeat_rows": repeat_rows,
    }
    payload = _json_safe(payload)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        _write_markdown(Path(args.out_md), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
