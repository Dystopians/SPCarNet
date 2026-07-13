#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


DEFAULT_V48_ROOTS = (
    "outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,"
    "/dev/shm/peilincai_spcarnet_v48_full9_20260623"
)
DEFAULT_V48_TAG = "v48_autosupport_autocap_guarded_v42calib_region_texture_adapter"
DEFAULT_TEACHER_METHOD_CANDIDATES = (
    "ours_26000_phasej_trainval_gate_lowpass_dconly_tailprefix",
    "ours_26000_phasej_trainval_gate_lowpass_dc_counter",
    "ours_26000_phasej_trainval_gate_lowpass_sh050_s025_counter",
    "ours_26000_phasej_trainval_gate",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_v48_audit(scene: str, roots: list[Path], tag: str) -> Path:
    paths: list[Path] = []
    for root in roots:
        paths.extend(root.glob(f"**/{scene}_{tag}/surface_residual_region_texture_adapter_audit.json"))
    if not paths:
        raise FileNotFoundError(f"missing v48 audit for {scene} under {', '.join(str(x) for x in roots)}")
    return sorted(paths, key=lambda path: len(str(path)))[0]


def cfg_value(text: str, key: str, default: str = "") -> str:
    match = re.search(rf"{re.escape(key)}=('[^']*'|\"[^\"]*\"|[^,)]+)", text)
    if match is None:
        return default
    raw = match.group(1).strip()
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return raw[1:-1]
    return raw


def parse_cfg_args(model_path: Path) -> dict[str, str]:
    text = (model_path / "cfg_args").read_text(encoding="utf-8")
    return {
        "source_path": cfg_value(text, "source_path"),
        "images": cfg_value(text, "images", "images"),
        "resolution": cfg_value(text, "resolution", "-1"),
        "split_strategy": cfg_value(text, "split_strategy", "llff"),
        "split_file": cfg_value(text, "split_file", ""),
    }


def ensure_link(src: Path, dst: Path, *, force: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def prepare_staged_model(source_model: Path, staged_model: Path, load_iteration: int, *, force: bool) -> None:
    staged_model.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "cameras.json", "input.ply"):
        src = source_model / name
        if src.exists():
            ensure_link(src, staged_model / name, force=force)
    source_iter = source_model / "point_cloud" / f"iteration_{load_iteration}"
    if not source_iter.is_dir():
        raise FileNotFoundError(f"missing source checkpoint iteration: {source_iter}")
    ensure_link(source_iter, staged_model / "point_cloud" / f"iteration_{load_iteration}", force=force)


def choose_teacher_dir(source_model: Path, requested: str) -> tuple[str, Path]:
    candidates = [requested] if requested else list(DEFAULT_TEACHER_METHOD_CANDIDATES)
    for method in candidates:
        render_dir = source_model / "train" / method / "renders"
        if render_dir.is_dir():
            return method, render_dir
    raise FileNotFoundError(
        "missing teacher train render dir; tried "
        + ", ".join(str(source_model / "train" / method / "renders") for method in candidates)
    )


def build_sentinel_cmd(args: argparse.Namespace, cfg: dict[str, str], staged_model: Path, cache_path: Path) -> list[str]:
    cmd = [
        args.python,
        "scripts/car_model/meshsplatopt_build_sparse_depth_sentinel_cache.py",
        "-s",
        cfg["source_path"],
        "-m",
        str(staged_model),
        "--images",
        cfg["images"],
        "--resolution",
        cfg["resolution"],
        "--eval",
        "--split_strategy",
        cfg["split_strategy"],
        "--parent_model_path",
        str(staged_model),
        "--parent_iteration",
        str(args.load_iteration),
        "--split",
        args.sentinel_split,
        "--num_views",
        str(args.sentinel_num_views),
        "--max_points_per_view",
        str(args.sentinel_max_points_per_view),
        "--sample_mode",
        args.sentinel_sample_mode,
        "--low_error_fraction",
        str(args.sentinel_low_error_fraction),
        "--seed",
        str(args.seed),
        "--output",
        str(cache_path),
    ]
    if cfg["split_file"]:
        cmd.extend(["--split_file", cfg["split_file"]])
    if args.sentinel_prefer_hard_views:
        cmd.append("--prefer_hard_views")
    if args.sentinel_prefer_observable_views:
        cmd.append("--prefer_observable_views")
    if args.sentinel_cluster_balance:
        cmd.append("--cluster_balance")
    return cmd


def build_recovery_cmd(
    args: argparse.Namespace,
    cfg: dict[str, str],
    staged_model: Path,
    contract_dir: Path,
    teacher_render_dir: Path,
    parent_render_dir: Path,
    cache_path: Path,
) -> list[str]:
    cmd = [
        args.python,
        "scripts/car_model/meshsplatopt_run_strict_compact_recovery.py",
        "--source_path",
        cfg["source_path"],
        "--output_path",
        str(staged_model),
        "--load_iteration",
        str(args.load_iteration),
        "--final_iteration",
        str(args.final_iteration),
        "--milestone_iterations",
        args.milestone_iterations,
        "--images",
        cfg["images"],
        "--resolution",
        cfg["resolution"],
        "--split_strategy",
        cfg["split_strategy"],
        "--preset",
        args.preset,
        "--teacher_render_dir",
        str(teacher_render_dir),
        "--teacher_render_lambda",
        str(args.teacher_render_lambda),
        "--teacher_render_dssim",
        str(args.teacher_render_dssim),
        "--teacher_render_mask_mode",
        args.teacher_render_mask_mode,
        "--teacher_render_error_margin",
        str(args.teacher_render_error_margin),
        "--teacher_render_parent_delta_min",
        str(args.teacher_render_parent_delta_min),
        "--teacher_render_start_iter",
        str(args.load_iteration),
        "--teacher_render_warmup_iters",
        str(args.teacher_render_warmup_iters),
        "--teacher_render_decay_start_iter",
        str(args.teacher_render_decay_start_iter),
        "--teacher_render_decay_end_iter",
        str(args.teacher_render_decay_end_iter),
        "--teacher_render_decay_final_mult",
        str(args.teacher_render_decay_final_mult),
        "--parent_render_rollback_dir",
        str(parent_render_dir),
        "--parent_render_rollback_lambda",
        str(args.parent_render_rollback_lambda),
        "--parent_render_rollback_start_iter",
        str(args.load_iteration),
        "--parent_render_rollback_warmup_iters",
        str(args.parent_render_rollback_warmup_iters),
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
        "--parent_render_rollback_edge_guidance_weight",
        str(args.parent_render_rollback_edge_guidance_weight),
        "--checkpoint_render_depth_anchor_lambda",
        str(args.checkpoint_render_depth_anchor_lambda),
        "--checkpoint_render_normal_anchor_lambda",
        str(args.checkpoint_render_normal_anchor_lambda),
        "--checkpoint_render_geometry_anchor_start_iter",
        str(args.load_iteration),
        "--checkpoint_render_geometry_anchor_warmup_iters",
        str(args.checkpoint_render_geometry_anchor_warmup_iters),
        "--sparse_depth_parent_rollback_cache",
        str(cache_path),
        "--sparse_depth_parent_rollback_lambda",
        str(args.sparse_depth_parent_rollback_lambda),
        "--sparse_depth_parent_rollback_start_iter",
        str(args.load_iteration),
        "--sparse_depth_parent_rollback_warmup_iters",
        str(args.sparse_depth_parent_rollback_warmup_iters),
        "--sparse_depth_parent_rollback_loss_space",
        args.sparse_depth_parent_rollback_loss_space,
        "--sparse_depth_parent_rollback_aggregation",
        args.sparse_depth_parent_rollback_aggregation,
        "--sparse_depth_parent_rollback_cvar_fraction",
        str(args.sparse_depth_parent_rollback_cvar_fraction),
        "--sparse_depth_parent_rollback_patch_reduce",
        args.sparse_depth_parent_rollback_patch_reduce,
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        args.wandb_group,
        "--wandb_name",
        args.wandb_name,
        "--train_seed",
        str(args.seed),
        "--contract_out_dir",
        str(contract_dir),
        "--python",
        args.python,
        "--allow_missing_final",
    ]
    if args.lr_triangles_points_init is not None:
        cmd.extend(["--lr_triangles_points_init", str(args.lr_triangles_points_init)])
    if args.feature_lr is not None:
        cmd.extend(["--feature_lr", str(args.feature_lr)])
    if args.weight_lr is not None:
        cmd.extend(["--weight_lr", str(args.weight_lr)])
    if cfg["split_file"]:
        cmd.extend(["--split_file", cfg["split_file"]])
    if args.execute:
        cmd.append("--execute")
    if args.sparse_depth_parent_rollback_cluster_balance:
        cmd.append("--sparse_depth_parent_rollback_cluster_balance")
    if args.sparse_depth_parent_rollback_regressed_only:
        cmd.append("--sparse_depth_parent_rollback_regressed_only")
    return cmd


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v96 checkpoint-baked certified ELA recovery for one scene.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output_root", default="/dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625")
    parser.add_argument("--tag", default="v96_checkpoint_baked_certified_repair")
    parser.add_argument("--v48_roots", default=DEFAULT_V48_ROOTS)
    parser.add_argument("--v48_tag", default=DEFAULT_V48_TAG)
    parser.add_argument("--teacher_method_name", default="")
    parser.add_argument("--load_iteration", type=int, default=26000)
    parser.add_argument("--final_iteration", type=int, default=30000)
    parser.add_argument("--milestone_iterations", default="28000,30000")
    parser.add_argument("--preset", choices=("compact_render_only", "compact_sparse_low_lambda", "compact_sparse_decay"), default="compact_render_only")
    parser.add_argument("--teacher_render_lambda", type=float, default=0.02)
    parser.add_argument("--teacher_render_dssim", type=float, default=0.10)
    parser.add_argument("--teacher_render_mask_mode", default="teacher_better")
    parser.add_argument("--teacher_render_error_margin", type=float, default=0.0)
    parser.add_argument("--teacher_render_parent_delta_min", type=float, default=0.0)
    parser.add_argument("--teacher_render_warmup_iters", type=int, default=300)
    parser.add_argument("--teacher_render_decay_start_iter", type=int, default=-1)
    parser.add_argument("--teacher_render_decay_end_iter", type=int, default=-1)
    parser.add_argument("--teacher_render_decay_final_mult", type=float, default=1.0)
    parser.add_argument("--parent_render_rollback_lambda", type=float, default=1.0)
    parser.add_argument("--parent_render_rollback_warmup_iters", type=int, default=300)
    parser.add_argument("--parent_render_rollback_aggregation", choices=("mean", "cvar"), default="cvar")
    parser.add_argument("--parent_render_rollback_cvar_fraction", type=float, default=0.10)
    parser.add_argument("--parent_render_rollback_cvar_min_pixels", type=int, default=1024)
    parser.add_argument("--parent_render_rollback_patch_radius", type=int, default=1)
    parser.add_argument("--parent_render_rollback_patch_reduce", choices=("center", "max_violation", "mean_violation"), default="max_violation")
    parser.add_argument("--parent_render_rollback_error_space", default="l1_dssim_edge")
    parser.add_argument("--parent_render_rollback_dssim_weight", type=float, default=0.25)
    parser.add_argument("--parent_render_rollback_edge_weight", type=float, default=0.10)
    parser.add_argument("--parent_render_rollback_edge_guidance_weight", type=float, default=0.10)
    parser.add_argument("--checkpoint_render_depth_anchor_lambda", type=float, default=0.01)
    parser.add_argument("--checkpoint_render_normal_anchor_lambda", type=float, default=0.005)
    parser.add_argument("--checkpoint_render_geometry_anchor_warmup_iters", type=int, default=300)
    parser.add_argument("--sparse_depth_parent_rollback_lambda", type=float, default=0.02)
    parser.add_argument("--sparse_depth_parent_rollback_warmup_iters", type=int, default=300)
    parser.add_argument("--sparse_depth_parent_rollback_loss_space", choices=("absrel", "mae", "combined"), default="combined")
    parser.add_argument("--sparse_depth_parent_rollback_aggregation", choices=("mean", "cvar", "cluster_cvar"), default="cvar")
    parser.add_argument("--sparse_depth_parent_rollback_cvar_fraction", type=float, default=0.20)
    parser.add_argument("--sparse_depth_parent_rollback_patch_reduce", choices=("center", "max_violation", "mean_violation"), default="max_violation")
    parser.add_argument("--sparse_depth_parent_rollback_cluster_balance", action="store_true")
    parser.add_argument("--sparse_depth_parent_rollback_regressed_only", action="store_true")
    parser.add_argument("--lr_triangles_points_init", type=float, default=None)
    parser.add_argument("--feature_lr", type=float, default=None)
    parser.add_argument("--weight_lr", type=float, default=None)
    parser.add_argument("--sentinel_split", choices=("train", "calibration"), default="train")
    parser.add_argument("--sentinel_num_views", type=int, default=24)
    parser.add_argument("--sentinel_max_points_per_view", type=int, default=500)
    parser.add_argument("--sentinel_sample_mode", default="mixed_low_error")
    parser.add_argument("--sentinel_low_error_fraction", type=float, default=0.5)
    parser.add_argument("--sentinel_prefer_hard_views", action="store_true")
    parser.add_argument("--sentinel_prefer_observable_views", action="store_true")
    parser.add_argument("--sentinel_cluster_balance", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet_meshprior")
    parser.add_argument("--wandb_group", default="v96_checkpoint_baked_certified_repair")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument("--wandb_dir", default="/dev/shm/wandb_spcarnet_v96")
    parser.add_argument("--python", default="/home/peilincai/micromamba/envs/mesh_splatting/bin/python")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force_staging", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    roots = [Path(item) for item in str(args.v48_roots).split(",") if item]
    audit_path = find_v48_audit(args.scene, roots, args.v48_tag)
    audit = read_json(audit_path)
    source_model = Path(audit["source_model"])
    cfg = parse_cfg_args(source_model)
    if not cfg["source_path"]:
        raise ValueError(f"could not parse source_path from {source_model / 'cfg_args'}")
    parent_render_dir = source_model / "train" / str(audit["base_method_name"]) / "renders"
    if not parent_render_dir.is_dir():
        raise FileNotFoundError(f"missing parent train render dir: {parent_render_dir}")
    teacher_method, teacher_render_dir = choose_teacher_dir(source_model, str(args.teacher_method_name))

    output_root = Path(args.output_root)
    scene_root = output_root / f"{args.scene}_{args.tag}"
    staged_model = scene_root / "recovery_model"
    contract_dir = scene_root / "contract"
    cache_path = scene_root / f"sparse_depth_parent_rollback_{args.sentinel_split}_cache.npz"
    args.wandb_name = args.wandb_name or f"{args.scene}_{args.tag}_{args.final_iteration}"

    prepare_staged_model(source_model, staged_model, int(args.load_iteration), force=bool(args.force_staging))
    sentinel_cmd = build_sentinel_cmd(args, cfg, staged_model, cache_path)
    recovery_cmd = build_recovery_cmd(
        args,
        cfg,
        staged_model,
        contract_dir,
        teacher_render_dir,
        parent_render_dir,
        cache_path,
    )
    manifest = {
        "method": "v96 checkpoint-baked certified ELA recovery",
        "scene": args.scene,
        "audit_path": str(audit_path),
        "source_model": str(source_model),
        "staged_model": str(staged_model),
        "teacher_method_name": teacher_method,
        "teacher_render_dir": str(teacher_render_dir),
        "parent_render_dir": str(parent_render_dir),
        "cache_path": str(cache_path),
        "contract_dir": str(contract_dir),
        "load_iteration": int(args.load_iteration),
        "final_iteration": int(args.final_iteration),
        "wandb_mode": args.wandb_mode,
        "wandb_dir": args.wandb_dir,
        "sentinel_command": sentinel_cmd,
        "recovery_command": recovery_cmd,
        "execute": bool(args.execute),
    }
    write_text(scene_root / "v96_manifest.json", json.dumps(manifest, indent=2) + "\n")
    write_text(scene_root / "build_sparse_depth_sentinel_command.txt", shlex.join(sentinel_cmd) + "\n")
    write_text(scene_root / "strict_recovery_command.txt", shlex.join(recovery_cmd) + "\n")
    print(json.dumps(manifest, indent=2))

    if args.execute:
        env = os.environ.copy()
        env["WANDB_MODE"] = str(args.wandb_mode)
        env["WANDB_DIR"] = str(args.wandb_dir)
        env.setdefault("PYTHONUNBUFFERED", "1")
        subprocess.run(sentinel_cmd, cwd=ROOT, check=True, env=env)
        subprocess.run(recovery_cmd, cwd=ROOT, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
