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


def _iteration_schedule(args: argparse.Namespace) -> list[int]:
    values = {int(args.final_iteration)}
    for item in str(args.milestone_iterations or "").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if int(args.load_iteration) < value <= int(args.final_iteration):
            values.add(value)
    return sorted(values)


def _extend_iteration_arg(cmd: list[str], flag: str, values: list[int]) -> None:
    cmd.append(flag)
    cmd.extend(str(v) for v in values)


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
    eval_iterations = _iteration_schedule(args)
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
        "--split_strategy",
        args.split_strategy,
        "--split_file",
        args.split_file,
        "--load_iteration",
        str(args.load_iteration),
        "--seed",
        str(args.train_seed),
        "--iterations",
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
    _extend_iteration_arg(cmd, "--test_iterations", eval_iterations)
    _extend_iteration_arg(cmd, "--save_iterations", eval_iterations)
    _extend_iteration_arg(cmd, "--checkpoint_iterations", eval_iterations)
    if args.lr_triangles_points_init is not None:
        cmd.extend(["--lr_triangles_points_init", str(args.lr_triangles_points_init)])
    if args.feature_lr is not None:
        cmd.extend(["--feature_lr", str(args.feature_lr)])
    if args.weight_lr is not None:
        cmd.extend(["--weight_lr", str(args.weight_lr)])
    if args.indoor:
        cmd.append("--indoor")
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
    if args.lpips_lambda > 0.0:
        cmd.extend(
            [
                "--lambda_lpips_loss",
                str(args.lpips_lambda),
                "--lpips_loss_start_iter",
                str(args.lpips_start_iter),
                "--lpips_loss_warmup_iters",
                str(args.lpips_warmup_iters),
                "--lpips_loss_max_side",
                str(args.lpips_max_side),
            ]
        )
    if args.teacher_render_lambda > 0.0:
        cmd.extend(
            [
                "--enable_teacher_render_loss",
                "--teacher_render_dir",
                args.teacher_render_dir,
                "--lambda_teacher_render",
                str(args.teacher_render_lambda),
                "--teacher_render_dssim",
                str(args.teacher_render_dssim),
                "--teacher_render_mask_mode",
                args.teacher_render_mask_mode,
                "--teacher_render_error_margin",
                str(args.teacher_render_error_margin),
                "--teacher_render_start_iter",
                str(args.teacher_render_start_iter),
                "--teacher_render_warmup_iters",
                str(args.teacher_render_warmup_iters),
                "--teacher_render_decay_start_iter",
                str(args.teacher_render_decay_start_iter),
                "--teacher_render_decay_end_iter",
                str(args.teacher_render_decay_end_iter),
                "--teacher_render_decay_final_mult",
                str(args.teacher_render_decay_final_mult),
            ]
        )
    if args.parent_render_rollback_lambda > 0.0:
        cmd.extend(
            [
                "--enable_parent_render_rollback_loss",
                "--parent_render_rollback_dir",
                args.parent_render_rollback_dir,
                "--lambda_parent_render_rollback",
                str(args.parent_render_rollback_lambda),
                "--parent_render_rollback_start_iter",
                str(args.parent_render_rollback_start_iter),
                "--parent_render_rollback_warmup_iters",
                str(args.parent_render_rollback_warmup_iters),
                "--parent_render_rollback_decay_start_iter",
                str(args.parent_render_rollback_decay_start_iter),
                "--parent_render_rollback_decay_end_iter",
                str(args.parent_render_rollback_decay_end_iter),
                "--parent_render_rollback_decay_final_mult",
                str(args.parent_render_rollback_decay_final_mult),
                "--parent_render_rollback_margin_abs",
                str(args.parent_render_rollback_margin_abs),
                "--parent_render_rollback_margin_rel",
                str(args.parent_render_rollback_margin_rel),
                "--parent_render_rollback_huber_delta",
                str(args.parent_render_rollback_huber_delta),
                "--parent_render_rollback_aggregation",
                args.parent_render_rollback_aggregation,
                "--parent_render_rollback_cvar_fraction",
                str(args.parent_render_rollback_cvar_fraction),
                "--parent_render_rollback_cvar_min_pixels",
                str(args.parent_render_rollback_cvar_min_pixels),
                "--parent_render_rollback_patch_radius",
                str(args.parent_render_rollback_patch_radius),
                "--parent_render_rollback_patch_reduce",
                args.parent_render_rollback_patch_reduce,
                "--parent_render_rollback_error_space",
                args.parent_render_rollback_error_space,
                "--parent_render_rollback_dssim_weight",
                str(args.parent_render_rollback_dssim_weight),
                "--parent_render_rollback_edge_weight",
                str(args.parent_render_rollback_edge_weight),
                "--parent_render_rollback_ssim_window",
                str(args.parent_render_rollback_ssim_window),
                "--parent_render_rollback_edge_guidance_weight",
                str(args.parent_render_rollback_edge_guidance_weight),
            ]
        )
    if args.checkpoint_geometry_anchor_lambda > 0.0:
        cmd.extend(
            [
                "--enable_checkpoint_geometry_anchor",
                "--lambda_checkpoint_geometry_anchor",
                str(args.checkpoint_geometry_anchor_lambda),
                "--checkpoint_geometry_anchor_start_iter",
                str(args.checkpoint_geometry_anchor_start_iter),
                "--checkpoint_geometry_anchor_warmup_iters",
                str(args.checkpoint_geometry_anchor_warmup_iters),
                "--checkpoint_geometry_anchor_decay_start_iter",
                str(args.checkpoint_geometry_anchor_decay_start_iter),
                "--checkpoint_geometry_anchor_decay_end_iter",
                str(args.checkpoint_geometry_anchor_decay_end_iter),
                "--checkpoint_geometry_anchor_decay_final_mult",
                str(args.checkpoint_geometry_anchor_decay_final_mult),
                "--checkpoint_geometry_anchor_huber_delta",
                str(args.checkpoint_geometry_anchor_huber_delta),
            ]
        )
    if args.checkpoint_render_depth_anchor_lambda > 0.0 or args.checkpoint_render_normal_anchor_lambda > 0.0:
        cmd.extend(
            [
                "--enable_checkpoint_render_geometry_anchor",
                "--lambda_checkpoint_render_depth_anchor",
                str(args.checkpoint_render_depth_anchor_lambda),
                "--lambda_checkpoint_render_normal_anchor",
                str(args.checkpoint_render_normal_anchor_lambda),
                "--checkpoint_render_geometry_anchor_start_iter",
                str(args.checkpoint_render_geometry_anchor_start_iter),
                "--checkpoint_render_geometry_anchor_warmup_iters",
                str(args.checkpoint_render_geometry_anchor_warmup_iters),
                "--checkpoint_render_geometry_anchor_huber_delta",
                str(args.checkpoint_render_geometry_anchor_huber_delta),
            ]
        )
    if args.sparse_depth_parent_rollback_lambda > 0.0:
        cmd.extend(
            [
                "--enable_sparse_depth_parent_rollback_loss",
                "--sparse_depth_parent_rollback_cache",
                args.sparse_depth_parent_rollback_cache,
                "--lambda_sparse_depth_parent_rollback",
                str(args.sparse_depth_parent_rollback_lambda),
                "--sparse_depth_parent_rollback_start_iter",
                str(args.sparse_depth_parent_rollback_start_iter),
                "--sparse_depth_parent_rollback_warmup_iters",
                str(args.sparse_depth_parent_rollback_warmup_iters),
                "--sparse_depth_parent_rollback_margin_abs",
                str(args.sparse_depth_parent_rollback_margin_abs),
                "--sparse_depth_parent_rollback_margin_rel",
                str(args.sparse_depth_parent_rollback_margin_rel),
                "--sparse_depth_parent_rollback_huber_delta",
                str(args.sparse_depth_parent_rollback_huber_delta),
                "--sparse_depth_parent_rollback_combined_mae_beta",
                str(args.sparse_depth_parent_rollback_combined_mae_beta),
                "--sparse_depth_parent_rollback_max_points_per_view",
                str(args.sparse_depth_parent_rollback_max_points_per_view),
                "--sparse_depth_parent_rollback_loss_space",
                args.sparse_depth_parent_rollback_loss_space,
                "--sparse_depth_parent_rollback_aggregation",
                args.sparse_depth_parent_rollback_aggregation,
                "--sparse_depth_parent_rollback_cvar_fraction",
                str(args.sparse_depth_parent_rollback_cvar_fraction),
                "--sparse_depth_parent_rollback_cvar_min_points",
                str(args.sparse_depth_parent_rollback_cvar_min_points),
                "--sparse_depth_parent_rollback_pixel_radius",
                str(args.sparse_depth_parent_rollback_pixel_radius),
                "--sparse_depth_parent_rollback_patch_reduce",
                args.sparse_depth_parent_rollback_patch_reduce,
            ]
        )
        if args.sparse_depth_parent_rollback_cluster_balance:
            cmd.append("--sparse_depth_parent_rollback_cluster_balance")
        if args.sparse_depth_parent_rollback_regressed_only:
            cmd.append("--sparse_depth_parent_rollback_regressed_only")
        if args.sparse_depth_parent_rollback_cluster_top_k > 0:
            cmd.extend(["--sparse_depth_parent_rollback_cluster_top_k", str(args.sparse_depth_parent_rollback_cluster_top_k)])
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
        "--split_strategy",
        args.split_strategy,
        "--split_file",
        args.split_file,
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
        "--resolution",
        str(args.resolution),
        "--eval",
        "--split_strategy",
        args.split_strategy,
        "--split_file",
        args.split_file,
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
        "train_seed": int(args.train_seed),
        "milestone_iterations": _iteration_schedule(args),
        "split_strategy": args.split_strategy,
        "split_file": args.split_file,
        "indoor": bool(args.indoor),
        "teacher_render_lambda": float(args.teacher_render_lambda),
        "teacher_render_dir": args.teacher_render_dir,
        "teacher_render_mask_mode": args.teacher_render_mask_mode,
        "parent_render_rollback_lambda": float(args.parent_render_rollback_lambda),
        "parent_render_rollback_dir": args.parent_render_rollback_dir,
        "parent_render_rollback_aggregation": args.parent_render_rollback_aggregation,
        "parent_render_rollback_cvar_fraction": float(args.parent_render_rollback_cvar_fraction),
        "parent_render_rollback_cvar_min_pixels": int(args.parent_render_rollback_cvar_min_pixels),
        "parent_render_rollback_patch_radius": int(args.parent_render_rollback_patch_radius),
        "parent_render_rollback_patch_reduce": args.parent_render_rollback_patch_reduce,
        "parent_render_rollback_error_space": args.parent_render_rollback_error_space,
        "parent_render_rollback_dssim_weight": float(args.parent_render_rollback_dssim_weight),
        "parent_render_rollback_edge_weight": float(args.parent_render_rollback_edge_weight),
        "parent_render_rollback_ssim_window": int(args.parent_render_rollback_ssim_window),
        "parent_render_rollback_edge_guidance_weight": float(args.parent_render_rollback_edge_guidance_weight),
        "checkpoint_geometry_anchor_lambda": float(args.checkpoint_geometry_anchor_lambda),
        "checkpoint_geometry_anchor_huber_delta": float(args.checkpoint_geometry_anchor_huber_delta),
        "checkpoint_render_depth_anchor_lambda": float(args.checkpoint_render_depth_anchor_lambda),
        "checkpoint_render_normal_anchor_lambda": float(args.checkpoint_render_normal_anchor_lambda),
        "checkpoint_render_geometry_anchor_huber_delta": float(args.checkpoint_render_geometry_anchor_huber_delta),
        "sparse_depth_parent_rollback_cache": args.sparse_depth_parent_rollback_cache,
        "sparse_depth_parent_rollback_lambda": float(args.sparse_depth_parent_rollback_lambda),
        "sparse_depth_parent_rollback_loss_space": args.sparse_depth_parent_rollback_loss_space,
        "sparse_depth_parent_rollback_combined_mae_beta": float(args.sparse_depth_parent_rollback_combined_mae_beta),
        "sparse_depth_parent_rollback_regressed_only": bool(args.sparse_depth_parent_rollback_regressed_only),
        "sparse_depth_parent_rollback_cluster_top_k": int(args.sparse_depth_parent_rollback_cluster_top_k),
        "sparse_depth_parent_rollback_margin_abs": float(args.sparse_depth_parent_rollback_margin_abs),
        "sparse_depth_parent_rollback_margin_rel": float(args.sparse_depth_parent_rollback_margin_rel),
        "lr_triangles_points_init": args.lr_triangles_points_init,
        "feature_lr": args.feature_lr,
        "weight_lr": args.weight_lr,
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
    parser.add_argument(
        "--milestone_iterations",
        default="",
        help="Comma-separated extra iterations to test/save/checkpoint before final_iteration.",
    )
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--split_strategy", default="llff")
    parser.add_argument("--split_file", default="")
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
    parser.add_argument("--lpips_lambda", type=float, default=0.0)
    parser.add_argument("--lpips_start_iter", type=int, default=22000)
    parser.add_argument("--lpips_warmup_iters", type=int, default=300)
    parser.add_argument("--lpips_max_side", type=int, default=512)
    parser.add_argument("--teacher_render_dir", default="")
    parser.add_argument("--teacher_render_lambda", type=float, default=0.0)
    parser.add_argument("--teacher_render_dssim", type=float, default=0.0)
    parser.add_argument("--teacher_render_mask_mode", default="teacher_better")
    parser.add_argument("--teacher_render_error_margin", type=float, default=0.0)
    parser.add_argument("--teacher_render_start_iter", type=int, default=22000)
    parser.add_argument("--teacher_render_warmup_iters", type=int, default=300)
    parser.add_argument("--teacher_render_decay_start_iter", type=int, default=-1)
    parser.add_argument("--teacher_render_decay_end_iter", type=int, default=-1)
    parser.add_argument("--teacher_render_decay_final_mult", type=float, default=1.0)
    parser.add_argument("--parent_render_rollback_dir", default="")
    parser.add_argument("--parent_render_rollback_lambda", type=float, default=0.0)
    parser.add_argument("--parent_render_rollback_start_iter", type=int, default=22000)
    parser.add_argument("--parent_render_rollback_warmup_iters", type=int, default=300)
    parser.add_argument("--parent_render_rollback_decay_start_iter", type=int, default=-1)
    parser.add_argument("--parent_render_rollback_decay_end_iter", type=int, default=-1)
    parser.add_argument("--parent_render_rollback_decay_final_mult", type=float, default=1.0)
    parser.add_argument("--parent_render_rollback_margin_abs", type=float, default=0.0)
    parser.add_argument("--parent_render_rollback_margin_rel", type=float, default=0.0)
    parser.add_argument("--parent_render_rollback_huber_delta", type=float, default=0.02)
    parser.add_argument("--parent_render_rollback_aggregation", choices=("mean", "cvar"), default="mean")
    parser.add_argument("--parent_render_rollback_cvar_fraction", type=float, default=0.1)
    parser.add_argument("--parent_render_rollback_cvar_min_pixels", type=int, default=1024)
    parser.add_argument("--parent_render_rollback_patch_radius", type=int, default=0)
    parser.add_argument("--parent_render_rollback_patch_reduce", choices=("center", "max_violation", "mean_violation"), default="center")
    parser.add_argument(
        "--parent_render_rollback_error_space",
        choices=("l1", "l2", "channel_max", "l1_dssim", "l1_edge", "l1_dssim_edge"),
        default="l1",
    )
    parser.add_argument("--parent_render_rollback_dssim_weight", type=float, default=0.0)
    parser.add_argument("--parent_render_rollback_edge_weight", type=float, default=0.0)
    parser.add_argument("--parent_render_rollback_ssim_window", type=int, default=11)
    parser.add_argument("--parent_render_rollback_edge_guidance_weight", type=float, default=0.0)
    parser.add_argument("--checkpoint_geometry_anchor_lambda", type=float, default=0.0)
    parser.add_argument("--checkpoint_geometry_anchor_start_iter", type=int, default=0)
    parser.add_argument("--checkpoint_geometry_anchor_warmup_iters", type=int, default=300)
    parser.add_argument("--checkpoint_geometry_anchor_decay_start_iter", type=int, default=-1)
    parser.add_argument("--checkpoint_geometry_anchor_decay_end_iter", type=int, default=-1)
    parser.add_argument("--checkpoint_geometry_anchor_decay_final_mult", type=float, default=1.0)
    parser.add_argument("--checkpoint_geometry_anchor_huber_delta", type=float, default=0.01)
    parser.add_argument("--checkpoint_render_depth_anchor_lambda", type=float, default=0.0)
    parser.add_argument("--checkpoint_render_normal_anchor_lambda", type=float, default=0.0)
    parser.add_argument("--checkpoint_render_geometry_anchor_start_iter", type=int, default=0)
    parser.add_argument("--checkpoint_render_geometry_anchor_warmup_iters", type=int, default=300)
    parser.add_argument("--checkpoint_render_geometry_anchor_huber_delta", type=float, default=0.02)
    parser.add_argument("--sparse_depth_parent_rollback_cache", default="")
    parser.add_argument("--sparse_depth_parent_rollback_lambda", type=float, default=0.0)
    parser.add_argument("--sparse_depth_parent_rollback_start_iter", type=int, default=0)
    parser.add_argument("--sparse_depth_parent_rollback_warmup_iters", type=int, default=300)
    parser.add_argument("--sparse_depth_parent_rollback_margin_abs", type=float, default=0.0)
    parser.add_argument("--sparse_depth_parent_rollback_margin_rel", type=float, default=0.0)
    parser.add_argument("--sparse_depth_parent_rollback_huber_delta", type=float, default=0.05)
    parser.add_argument("--sparse_depth_parent_rollback_combined_mae_beta", type=float, default=1.0)
    parser.add_argument("--sparse_depth_parent_rollback_cluster_balance", action="store_true")
    parser.add_argument("--sparse_depth_parent_rollback_regressed_only", action="store_true")
    parser.add_argument("--sparse_depth_parent_rollback_cluster_top_k", type=int, default=0)
    parser.add_argument("--sparse_depth_parent_rollback_max_points_per_view", type=int, default=500)
    parser.add_argument("--sparse_depth_parent_rollback_loss_space", choices=("absrel", "mae", "combined"), default="combined")
    parser.add_argument("--sparse_depth_parent_rollback_aggregation", choices=("mean", "cvar", "cluster_cvar"), default="mean")
    parser.add_argument("--sparse_depth_parent_rollback_cvar_fraction", type=float, default=0.2)
    parser.add_argument("--sparse_depth_parent_rollback_cvar_min_points", type=int, default=16)
    parser.add_argument("--sparse_depth_parent_rollback_pixel_radius", type=int, default=0)
    parser.add_argument("--sparse_depth_parent_rollback_patch_reduce", choices=("center", "max_violation", "mean_violation"), default="center")
    parser.add_argument("--lr_triangles_points_init", type=float, default=None)
    parser.add_argument("--feature_lr", type=float, default=None)
    parser.add_argument("--weight_lr", type=float, default=None)
    parser.add_argument("--wandb_project", default="spcarnet_meshprior")
    parser.add_argument("--wandb_group", default="finalF6_strict_recovery")
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--indoor", action="store_true")
    parser.add_argument("--contract_out_dir", required=True)
    parser.add_argument("--python", default="/home/peilincai/micromamba/envs/mesh_splatting/bin/python")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow_missing_final", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
