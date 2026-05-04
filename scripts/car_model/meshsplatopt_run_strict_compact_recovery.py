#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import checkpoint_path  # noqa: E402


PRESETS = ("compact_render_only", "compact_sparse_low_lambda", "compact_sparse_decay")


def _topology(model_path: Path, iteration: int) -> dict[str, int | str]:
    import torch

    checkpoint = checkpoint_path(model_path, iteration)
    state = torch.load(checkpoint, map_location="cpu")
    return {
        "checkpoint": str(checkpoint),
        "iteration": int(iteration),
        "triangles": int(state["_triangle_indices"].shape[0]),
        "vertices": int(state["triangles_points"].shape[0]),
    }


def _train_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        "train.py",
        "-s",
        args.source_path,
        "-m",
        args.output_path,
        "--images",
        args.images,
        "--resolution",
        str(args.resolution),
        "--eval",
        "--load_iteration",
        str(args.load_iteration),
        "--iterations",
        str(args.final_iteration),
        "--test_iterations",
        str(args.final_iteration),
        "--save_iterations",
        str(args.final_iteration),
        "--checkpoint_iterations",
        str(args.final_iteration),
        "--densify_until_iter",
        str(args.load_iteration),
        "--skip_restricted_delaunay",
        "--freeze_topology_updates",
        "--enable_wandb",
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        args.wandb_group,
        "--wandb_name",
        args.wandb_name,
        "--wandb_image_log_interval",
        "1000",
        "--wandb_scalar_log_interval",
        "50",
    ]
    if args.preset in ("compact_sparse_low_lambda", "compact_sparse_decay"):
        cmd.extend(
            [
                "--enable_sparse_colmap_depth_loss",
                "--lambda_sparse_colmap_depth",
                str(args.sparse_lambda),
                "--sparse_colmap_depth_start_iter",
                str(args.sparse_start_iter),
                "--sparse_colmap_depth_warmup_iters",
                str(args.sparse_warmup_iters),
                "--sparse_colmap_depth_min_matches",
                str(args.sparse_min_matches),
                "--sparse_colmap_depth_sample_mode",
                args.sparse_sample_mode,
                "--sparse_colmap_depth_low_error_fraction",
                str(args.sparse_fraction),
            ]
        )
    if args.preset == "compact_sparse_decay":
        cmd.extend(
            [
                "--sparse_colmap_depth_decay_start_iter",
                str(args.sparse_decay_start),
                "--sparse_colmap_depth_decay_end_iter",
                str(args.sparse_decay_end),
                "--sparse_colmap_depth_decay_final_mult",
                str(args.sparse_decay_final_mult),
            ]
        )
    return cmd


def _render_args(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "render.py",
        "-s",
        args.source_path,
        "-m",
        args.output_path,
        "--images",
        args.images,
        "--resolution",
        str(args.resolution),
        "--eval",
        "--iteration",
        str(args.final_iteration),
        "--skip_train",
    ]


def _geometry_args(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "evaluate_geometry_colmap.py",
        "-s",
        args.source_path,
        "-m",
        args.output_path,
        "--images",
        args.images,
        "--eval",
        "--iteration",
        str(args.final_iteration),
        "--max_points_per_view",
        "500",
        "--output",
        f"{args.output_path}/geometry_eval_colmap/iter_{args.final_iteration}_max500.json",
    ]


def run(args: argparse.Namespace) -> int:
    out = Path(args.contract_out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_cmd = _train_args(args)
    render_cmd = _render_args(args)
    metrics_cmd = [args.python, "metrics.py", "-m", args.output_path]
    geometry_cmd = _geometry_args(args)

    load_topology = _topology(Path(args.output_path), args.load_iteration)
    summary = {
        "preset": args.preset,
        "source_path": args.source_path,
        "output_path": args.output_path,
        "load_iteration": args.load_iteration,
        "final_iteration": args.final_iteration,
        "wandb_project": args.wandb_project,
        "wandb_group": args.wandb_group,
        "wandb_name": args.wandb_name,
        "execute": bool(args.execute),
        "topology_unchanged": None,
    }
    (out / "exact_train_command.txt").write_text(shlex.join(train_cmd) + "\n", encoding="utf-8")
    (out / "render_command.txt").write_text(shlex.join(render_cmd) + "\n", encoding="utf-8")
    (out / "metrics_command.txt").write_text(shlex.join(metrics_cmd) + "\n", encoding="utf-8")
    (out / "geometry_command.txt").write_text(shlex.join(geometry_cmd) + "\n", encoding="utf-8")
    (out / "wandb_url.txt").write_text(
        f"https://wandb.ai/karamazovaniki-university-of-southern-california/{args.wandb_project}/runs/{args.wandb_name}\n",
        encoding="utf-8",
    )
    (out / "recovery_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.execute:
        subprocess.run(train_cmd, cwd=ROOT, check=True)
        subprocess.run(render_cmd, cwd=ROOT, check=True)
        subprocess.run(metrics_cmd, cwd=ROOT, check=True)
        subprocess.run(geometry_cmd, cwd=ROOT, check=True)
    try:
        final_checkpoint_present = checkpoint_path(Path(args.output_path), args.final_iteration).is_file()
    except FileNotFoundError:
        final_checkpoint_present = False
    final_topology = _topology(Path(args.output_path), args.final_iteration) if final_checkpoint_present else {}
    topology_unchanged = bool(
        final_topology
        and load_topology["triangles"] == final_topology["triangles"]
        and load_topology["vertices"] == final_topology["vertices"]
    )
    topology_audit = {
        "load": load_topology,
        "final": final_topology,
        "topology_unchanged": topology_unchanged,
        "required_flags": ["--freeze_topology_updates", "--skip_restricted_delaunay"],
        "sparse_depth_enabled": "--enable_sparse_colmap_depth_loss" in train_cmd,
    }
    summary["topology_unchanged"] = topology_unchanged
    (out / "topology_audit.json").write_text(json.dumps(topology_audit, indent=2) + "\n", encoding="utf-8")
    (out / "recovery_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "topology_audit": topology_audit}, indent=2))
    return 0 if topology_unchanged or args.allow_missing_final else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write and optionally execute strict topology-frozen compact recovery.")
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--load_iteration", type=int, required=True)
    parser.add_argument("--final_iteration", type=int, required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--preset", choices=PRESETS, default="compact_render_only")
    parser.add_argument("--sparse_lambda", type=float, default=0.001)
    parser.add_argument("--sparse_start_iter", type=int, default=22000)
    parser.add_argument("--sparse_warmup_iters", type=int, default=300)
    parser.add_argument("--sparse_min_matches", type=int, default=16)
    parser.add_argument("--sparse_sample_mode", default="mixed_low_error")
    parser.add_argument("--sparse_fraction", type=float, default=0.5)
    parser.add_argument("--sparse_decay_start", type=int, default=0)
    parser.add_argument("--sparse_decay_end", type=int, default=0)
    parser.add_argument("--sparse_decay_final_mult", type=float, default=0.0)
    parser.add_argument("--wandb_project", default="spcarnet_meshprior")
    parser.add_argument("--wandb_group", default="finalF6_strict_recovery")
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument("--contract_out_dir", required=True)
    parser.add_argument("--python", default="/home/peilincai/micromamba/envs/mesh_splatting/bin/python")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow_missing_final", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
