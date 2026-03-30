#!/usr/bin/env python3
"""
One-stop pipeline for MeshSplatting: train -> render -> visual/geometry eval.

Design goals:
1) Put commonly used train/eval args in one place.
2) Make repeated experiments reproducible with dataset/method presets.
3) Keep flexibility with passthrough args for advanced cases.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Classic dataset presets (local paths already imported in this workspace).
#
# Use case hints:
# - bonsai_local: indoor scene, dense texture + many close views.
# - flowers_local: outdoor scene, richer depth range and geometry variation.
#
# Recommended default image folder:
# - images_4: good speed/quality balance for most quick iterations.
# ---------------------------------------------------------------------------
DATASET_PRESETS: Dict[str, Dict[str, object]] = {
    "bonsai_local": {
        "scene_path": "/data2/peilincai/mesh_datasets/mipnerf360/bonsai",
        "indoor": True,
        "outdoor": False,
        "recommended_images": "images_4",
        "notes": "Indoor benchmark scene. Good for geometry stabilization checks.",
    },
    "flowers_local": {
        "scene_path": "/data2/peilincai/mesh_datasets/mipnerf360/flowers",
        "indoor": False,
        "outdoor": True,
        "recommended_images": "images_4",
        "notes": "Outdoor benchmark scene. Good for generalization and speed checks.",
    },
}


# ---------------------------------------------------------------------------
# Classic method presets.
#
# baseline_origin:
# - Closest to original baseline behavior: no extra ground-regularization flags.
#
# current_ground_method:
# - Enables your current ground-aware regularization stack.
# - These defaults are intentionally moderate and can be overridden via
#   --train_extra.
# ---------------------------------------------------------------------------
METHOD_PRESETS: Dict[str, Dict[str, object]] = {
    "baseline_origin": {
        "train_args": [],
        "notes": "Original baseline style (no ground-aware incremental regularization).",
    },
    "current_ground_method": {
        "train_args": [
            "--enable_ground_plane_estimation",
            "--enable_ground_regularization",
            "--enable_ground_plane_loss",
            "--enable_ground_normal_loss",
            "--enable_ground_smoothness_loss",
            "--enable_ground_mesh_assignment",
            "--ground_reg_start_iter",
            "2000",
            "--ground_reg_warmup_iters",
            "3000",
            "--lambda_ground_plane",
            "0.02",
            "--lambda_ground_normal",
            "0.01",
            "--lambda_ground_smoothness",
            "0.005",
            "--ground_reg_global_scale",
            "1.0",
        ],
        "notes": "Current method style with ground-plane/normal/smoothness regularization.",
    },
}


def _cmd_to_str(cmd: List[str]) -> str:
    return shlex.join(cmd)


def _run_command(cmd: List[str], dry_run: bool) -> None:
    print(f"\n[Pipeline] { _cmd_to_str(cmd) }")
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _resolve_scene_and_mode(args: argparse.Namespace) -> tuple[str, bool, bool, str]:
    scene_path = args.scene_path
    indoor = bool(args.indoor)
    outdoor = bool(args.outdoor)
    images = args.images

    if args.dataset != "custom":
        preset = DATASET_PRESETS[args.dataset]
        if not scene_path:
            scene_path = str(preset["scene_path"])
        if not args.indoor and not args.outdoor:
            indoor = bool(preset["indoor"])
            outdoor = bool(preset["outdoor"])
        if args.images == "":
            images = str(preset["recommended_images"])

    if not scene_path:
        raise ValueError("scene path is empty. Use --dataset preset or provide --scene_path.")

    if indoor and outdoor:
        raise ValueError("indoor/outdoor cannot both be True.")

    if images == "":
        images = "images_4"

    return scene_path, indoor, outdoor, images


def build_train_cmd(args: argparse.Namespace, scene_path: str, indoor: bool, outdoor: bool, images: str) -> List[str]:
    cmd = [
        args.python_bin,
        "train.py",
        "-s",
        scene_path,
        "-m",
        args.model_path,
        "--eval",
        "--iterations",
        str(args.iterations),
        "--images",
        images,
        "--split_strategy",
        args.split_strategy,
    ]

    if args.split_file:
        cmd += ["--split_file", args.split_file]
    if indoor:
        cmd += ["--indoor"]
    if outdoor:
        cmd += ["--outdoor"]
    if args.load_iteration is not None:
        cmd += ["--load_iteration", str(args.load_iteration)]
    if args.scene_name:
        cmd += ["--scene_name", args.scene_name]

    if args.method != "custom":
        cmd += list(METHOD_PRESETS[args.method]["train_args"])  # type: ignore[index]

    if args.train_extra:
        cmd += shlex.split(args.train_extra)

    return cmd


def build_render_cmd(args: argparse.Namespace, scene_path: str, eval_iteration: int, images: str) -> List[str]:
    cmd = [
        args.python_bin,
        "render.py",
        "--iteration",
        str(eval_iteration),
        "-s",
        scene_path,
        "-m",
        args.model_path,
        "--eval",
        "--images",
        images,
    ]
    if not args.render_train:
        cmd += ["--skip_train"]
    if args.render_extra:
        cmd += shlex.split(args.render_extra)
    return cmd


def build_metrics_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [args.python_bin, "metrics.py", "-m", args.model_path]
    if args.metrics_extra:
        cmd += shlex.split(args.metrics_extra)
    return cmd


def build_geometry_cmd(args: argparse.Namespace, scene_path: str, eval_iteration: int, images: str) -> List[str]:
    cmd = [
        args.python_bin,
        "evaluate_geometry_colmap.py",
        "--iteration",
        str(eval_iteration),
        "-s",
        scene_path,
        "-m",
        args.model_path,
        "--eval",
        "--images",
        images,
    ]
    if args.geometry_output:
        cmd += ["--output", args.geometry_output]
    if args.geometry_extra:
        cmd += shlex.split(args.geometry_extra)
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified train+eval pipeline for MeshSplatting experiments."
    )

    parser.add_argument(
        "--dataset",
        default="bonsai_local",
        choices=["custom", *DATASET_PRESETS.keys()],
        help="Dataset preset. Use custom with --scene_path for arbitrary scenes.",
    )
    parser.add_argument(
        "--method",
        default="baseline_origin",
        choices=["custom", *METHOD_PRESETS.keys()],
        help="Method preset for train flags.",
    )

    parser.add_argument("--scene_path", default="", help="Scene directory path.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Output model directory (same value used by train/render/metrics).",
    )
    parser.add_argument(
        "--images",
        default="",
        help="Image folder name inside scene (images/images_2/images_4/images_8).",
    )
    parser.add_argument("--indoor", action="store_true", help="Force indoor training mode.")
    parser.add_argument("--outdoor", action="store_true", help="Force outdoor training mode.")
    parser.add_argument("--scene_name", default="", help="Optional scene_name passed to train.py.")

    parser.add_argument("--iterations", type=int, default=30_000, help="Train iterations.")
    parser.add_argument(
        "--eval_iteration",
        type=int,
        default=-1,
        help="Iteration for render/eval. -1 means use --iterations value.",
    )
    parser.add_argument(
        "--load_iteration",
        type=int,
        default=None,
        help="Optional checkpoint iteration to resume from in train.py.",
    )

    parser.add_argument(
        "--split_strategy",
        default="llff",
        choices=["llff", "file"],
        help="COLMAP split strategy.",
    )
    parser.add_argument("--split_file", default="", help="Split JSON used when split_strategy=file.")

    parser.add_argument("--skip_train", action="store_true", help="Skip train stage.")
    parser.add_argument("--skip_render", action="store_true", help="Skip render stage.")
    parser.add_argument("--skip_metrics", action="store_true", help="Skip metrics stage.")
    parser.add_argument("--skip_geometry", action="store_true", help="Skip geometry stage.")
    parser.add_argument(
        "--render_train",
        action="store_true",
        help="Render train views too (default only renders test views).",
    )

    parser.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python interpreter used for subprocess calls.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands only.")

    # Passthrough args for advanced usage without editing this file.
    parser.add_argument("--train_extra", default="", help="Extra raw args appended to train.py.")
    parser.add_argument("--render_extra", default="", help="Extra raw args appended to render.py.")
    parser.add_argument("--metrics_extra", default="", help="Extra raw args appended to metrics.py.")
    parser.add_argument("--geometry_extra", default="", help="Extra raw args appended to evaluate_geometry_colmap.py.")
    parser.add_argument("--geometry_output", default="", help="Optional output path for geometry json.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    scene_path, indoor, outdoor, images = _resolve_scene_and_mode(args)
    eval_iteration = args.iterations if args.eval_iteration < 0 else args.eval_iteration

    print("[Pipeline] Config summary:")
    print(f"  dataset      : {args.dataset}")
    print(f"  method       : {args.method}")
    print(f"  scene_path   : {scene_path}")
    print(f"  model_path   : {args.model_path}")
    print(f"  images       : {images}")
    print(f"  indoor       : {indoor}")
    print(f"  outdoor      : {outdoor}")
    print(f"  iterations   : {args.iterations}")
    print(f"  eval_iter    : {eval_iteration}")
    print(f"  split        : {args.split_strategy}")
    if args.split_file:
        print(f"  split_file   : {args.split_file}")

    if not args.skip_train:
        _run_command(build_train_cmd(args, scene_path, indoor, outdoor, images), args.dry_run)
    if not args.skip_render:
        _run_command(build_render_cmd(args, scene_path, eval_iteration, images), args.dry_run)
    if not args.skip_metrics:
        _run_command(build_metrics_cmd(args), args.dry_run)
    if not args.skip_geometry:
        _run_command(build_geometry_cmd(args, scene_path, eval_iteration, images), args.dry_run)

    print("\n[Pipeline] Done.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Example commands (requested templates)
#
# Original baseline sample:
# python pipeline.py \
#   --dataset bonsai_local \
#   --method baseline_origin \
#   --model_path models/bonsai_baseline_origin \
#   --iterations 30000 \
#   --images images_4 \
#   --indoor
#
# Current method sample:
# python pipeline.py \
#   --dataset bonsai_local \
#   --method current_ground_method \
#   --model_path models/bonsai_current_ground \
#   --iterations 30000 \
#   --images images_4 \
#   --indoor \
#   --split_strategy file \
#   --split_file /data2/peilincai/mesh_datasets/mipnerf360/bonsai/sparse/0/split_outoftrain_v1.json \
#   --train_extra "--enable_ground_masks --ground_mask_dir segmentation/ground_masks/bonsai"
# ---------------------------------------------------------------------------
