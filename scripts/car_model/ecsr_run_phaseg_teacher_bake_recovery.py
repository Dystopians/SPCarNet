#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import checkpoint_path  # noqa: E402


OUTDOOR_SCENES = {"bicycle", "flowers", "garden", "stump", "treehill"}
INDOOR_SCENES = {"room", "counter", "kitchen", "bonsai"}
DEFAULT_SCENES = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
IGNORE_COPY_NAMES = {
    "train",
    "test",
    "results.json",
    "per_view.json",
    "test_results.json",
    "test_per_view.json",
    "policy_val_results.json",
    "policy_val_per_view.json",
    "geometry_eval_colmap",
}


def _parse_scenes(text: str) -> list[str]:
    if not text or text.strip().lower() in {"all", "*"}:
        return list(DEFAULT_SCENES)
    return [x.strip() for x in text.split(",") if x.strip()]


def _run(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None, execute: bool = True) -> None:
    print(shlex.join(cmd), flush=True)
    if execute:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_phasef_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    rows = payload.get("rows", [])
    by_scene = {}
    for row in rows:
        scene = str(row.get("scene", "")).strip()
        if scene:
            by_scene[scene] = row
    return by_scene


def _copy_model_shell(source_model: Path, output_model: Path, *, force: bool = False) -> None:
    if output_model.exists():
        if not force:
            return
        shutil.rmtree(output_model)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in IGNORE_COPY_NAMES}

    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_model, output_model, ignore=ignore)


def _copy_base_train_evidence(source_model: Path, output_model: Path, base_method_name: str, *, force: bool = False) -> Path:
    src = source_model / "train" / base_method_name
    dst = output_model / "train" / base_method_name
    if not src.is_dir():
        raise FileNotFoundError(f"missing Phase-F train evidence: {src}")
    if dst.exists() and force:
        shutil.rmtree(dst)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    return dst / "renders"


def _topology(model_path: Path, iteration: int) -> dict[str, int | str]:
    import torch

    ckpt = checkpoint_path(model_path, iteration)
    state = torch.load(ckpt, map_location="cpu")
    return {
        "checkpoint": str(ckpt),
        "iteration": int(iteration),
        "triangles": int(state["_triangle_indices"].shape[0]),
        "vertices": int(state["triangles_points"].shape[0]),
    }


def _checkpoint_file_if_present(model_path: Path, iteration: int) -> Path | None:
    try:
        ckpt = checkpoint_path(model_path, iteration)
    except FileNotFoundError:
        return None
    return ckpt if ckpt.is_file() else None


def _iteration_schedule(load_iteration: int, final_iteration: int, milestone_iterations: str) -> list[int]:
    values = {int(final_iteration)}
    for item in str(milestone_iterations or "").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if int(load_iteration) < value <= int(final_iteration):
            values.add(value)
    return sorted(values)


def _extend_iteration_arg(cmd: list[str], flag: str, values: list[int]) -> None:
    cmd.append(flag)
    cmd.extend(str(v) for v in values)


def _scene_images(scene: str, outdoor_images: str, indoor_images: str) -> str:
    return indoor_images if scene in INDOOR_SCENES else outdoor_images


def _scene_indoor(scene: str) -> bool:
    return scene in INDOOR_SCENES


def _source_path(dataset_root: Path, scene: str) -> Path:
    path = dataset_root / scene
    if not path.is_dir():
        raise FileNotFoundError(f"missing dataset scene: {path}")
    return path


def _split_file(split_root: Path, scene: str) -> Path:
    path = split_root / scene / "split_file.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing split file: {path}")
    return path


def _split_args(strategy: str, split_file: Path) -> list[str]:
    strategy = str(strategy)
    if strategy == "file":
        return ["--split_strategy", "file", "--split_file", str(split_file)]
    return ["--split_strategy", "llff", "--split_file", ""]


def _phasef_fixed_policy(source_model: Path, phasef: dict[str, Any]) -> dict[str, Any]:
    method_name = str(phasef.get("method_name") or "")
    if not method_name:
        raise KeyError("Phase-F row does not contain method_name for fixed-policy teacher generation")
    report_path = source_model / "test" / method_name / "ela_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing Phase-F ELA policy report: {report_path}")
    report = _load_json(report_path)
    policy = dict(report.get("policy") or {})
    if not policy:
        policy = {
            "mode": report.get("mode", "residual"),
            "k": report.get("k", 8),
            "residual_clip": report.get("residual_clip", 0.2),
            "depth_abs_tol": report.get("depth_abs_tol", 0.02),
            "depth_rel_tol": report.get("depth_rel_tol", 0.12),
            "direction_weight": report.get("direction_weight", 0.2),
            "edge_gate": report.get("edge_gate", True),
            "edge_gate_quantile": report.get("edge_gate_quantile", 0.7),
            "edge_gate_min": report.get("edge_gate_min", 0.0),
            "edge_gate_dilate": report.get("edge_gate_dilate", 1),
        }
    return {
        "report_path": str(report_path),
        "alpha": float(report.get("alpha", 0.875)),
        "mode": str(policy.get("mode", "residual")),
        "k": int(policy.get("k", 8)),
        "residual_clip": float(policy.get("residual_clip", 0.2)),
        "depth_abs_tol": float(policy.get("depth_abs_tol", 0.02)),
        "depth_rel_tol": float(policy.get("depth_rel_tol", 0.12)),
        "direction_weight": float(policy.get("direction_weight", 0.2)),
        "edge_gate": bool(policy.get("edge_gate", True)),
        "edge_gate_quantile": float(policy.get("edge_gate_quantile", 0.7)),
        "edge_gate_min": float(policy.get("edge_gate_min", 0.0)),
        "edge_gate_dilate": int(policy.get("edge_gate_dilate", 1)),
    }


def _ela_train_command(
    args: argparse.Namespace,
    *,
    scene: str,
    source_model: Path,
    output_model: Path,
    phasef: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    cmd = [
        args.python,
        "scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py",
        "--base_model_path",
        str(source_model),
        "--output_model_path",
        str(output_model),
        "--iteration",
        str(args.load_iteration),
        "--base_method_name",
        args.base_method_name,
        "--target_split",
        "train",
        "--method_name",
        args.teacher_method_name,
        "--calib_sampler",
        args.calib_sampler,
        "--calib_max_views",
        str(args.calib_max_views),
        "--calib_stride",
        str(args.calib_stride),
        "--device",
        "cuda",
        "--wandb",
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        args.wandb_group,
        "--wandb_name",
        f"{args.wandb_prefix}_{scene}_train_teacher",
    ]
    metadata: dict[str, Any] = {"source": args.teacher_policy_source}
    if args.teacher_policy_source == "phasef_report":
        fixed = _phasef_fixed_policy(source_model, phasef)
        metadata.update(fixed)
        cmd.extend(
            [
                "--mode",
                fixed["mode"],
                "--k",
                str(fixed["k"]),
                "--alpha",
                str(fixed["alpha"]),
                "--residual_clip",
                str(fixed["residual_clip"]),
                "--depth_abs_tol",
                str(fixed["depth_abs_tol"]),
                "--depth_rel_tol",
                str(fixed["depth_rel_tol"]),
                "--direction_weight",
                str(fixed["direction_weight"]),
                "--edge_gate_quantile",
                str(fixed["edge_gate_quantile"]),
                "--edge_gate_min",
                str(fixed["edge_gate_min"]),
                "--edge_gate_dilate",
                str(fixed["edge_gate_dilate"]),
                "--skip_fixed_alpha_calibration",
            ]
        )
        if fixed["edge_gate"]:
            cmd.append("--edge_gate")
    else:
        cmd.extend(
            [
                "--auto_policy",
                "--policy_modes",
                args.policy_modes,
                "--policy_k_values",
                args.policy_k_values,
                "--policy_depth_rel_values",
                args.policy_depth_rel_values,
                "--policy_residual_clip_values",
                args.policy_residual_clip_values,
                "--policy_direction_weight_values",
                args.policy_direction_weight_values,
                "--policy_objective",
                args.policy_objective,
                "--policy_ssim_weight",
                str(args.policy_ssim_weight),
                "--policy_lpips_weight",
                str(args.policy_lpips_weight),
                "--alpha_grid",
                args.alpha_grid,
                "--edge_gate",
                "--edge_gate_quantile",
                str(args.edge_gate_quantile),
                "--edge_gate_dilate",
                str(args.edge_gate_dilate),
            ]
        )
        if args.calib_lpips:
            cmd.append("--calib_lpips")
    return cmd, metadata


def _train_command(
    args: argparse.Namespace,
    *,
    scene: str,
    source_path: Path,
    output_model: Path,
    split_file: Path,
    images: str,
    parent_render_dir: Path,
    teacher_render_dir: Path,
) -> list[str]:
    eval_iterations = _iteration_schedule(args.load_iteration, args.final_iteration, args.milestone_iterations)
    cmd = [
        args.python,
        "train.py",
        "-s",
        str(source_path),
        "-m",
        str(output_model),
        "--images",
        images,
        "--resolution",
        str(args.resolution),
        "--eval",
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
        f"{args.wandb_prefix}_{scene}_bake_{args.final_iteration}",
        "--wandb_image_log_interval",
        str(args.wandb_image_log_interval),
        "--wandb_scalar_log_interval",
        str(args.wandb_scalar_log_interval),
        "--lr_triangles_points_init",
        str(args.lr_triangles_points_init),
        "--feature_lr",
        str(args.feature_lr),
        "--weight_lr",
        str(args.weight_lr),
        "--enable_teacher_render_loss",
        "--teacher_render_dir",
        str(teacher_render_dir),
        "--lambda_teacher_render",
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
        "--enable_parent_render_rollback_loss",
        "--parent_render_rollback_dir",
        str(parent_render_dir),
        "--lambda_parent_render_rollback",
        str(args.parent_render_rollback_lambda),
        "--parent_render_rollback_start_iter",
        str(args.load_iteration),
        "--parent_render_rollback_warmup_iters",
        str(args.parent_render_rollback_warmup_iters),
        "--parent_render_rollback_huber_delta",
        str(args.parent_render_rollback_huber_delta),
        "--parent_render_rollback_aggregation",
        args.parent_render_rollback_aggregation,
        "--parent_render_rollback_cvar_fraction",
        str(args.parent_render_rollback_cvar_fraction),
        "--parent_render_rollback_cvar_min_pixels",
        str(args.parent_render_rollback_cvar_min_pixels),
        "--parent_render_rollback_error_space",
        args.parent_render_rollback_error_space,
        "--parent_render_rollback_dssim_weight",
        str(args.parent_render_rollback_dssim_weight),
        "--parent_render_rollback_edge_weight",
        str(args.parent_render_rollback_edge_weight),
    ]
    cmd.extend(_split_args(args.train_split_strategy, split_file))
    _extend_iteration_arg(cmd, "--test_iterations", eval_iterations)
    _extend_iteration_arg(cmd, "--save_iterations", eval_iterations)
    _extend_iteration_arg(cmd, "--checkpoint_iterations", eval_iterations)
    if _scene_indoor(scene):
        cmd.append("--indoor")
    if args.lpips_lambda > 0.0:
        cmd.extend(
            [
                "--lambda_lpips_loss",
                str(args.lpips_lambda),
                "--lpips_loss_start_iter",
                str(args.load_iteration),
                "--lpips_loss_warmup_iters",
                str(args.lpips_warmup_iters),
                "--lpips_loss_max_side",
                str(args.lpips_max_side),
            ]
        )
    return cmd


def _render_command(
    args: argparse.Namespace,
    *,
    source_path: Path,
    output_model: Path,
    split_file: Path,
    images: str,
    iteration: int,
) -> list[str]:
    cmd = [
        args.python,
        "render.py",
        "-s",
        str(source_path),
        "-m",
        str(output_model),
        "--images",
        images,
        "--resolution",
        str(args.resolution),
        "--eval",
        "--iteration",
        str(iteration),
        "--skip_train",
        "--quiet",
    ]
    cmd.extend(_split_args(args.eval_split_strategy, split_file))
    return cmd


def _geometry_command(
    args: argparse.Namespace,
    *,
    source_path: Path,
    output_model: Path,
    split_file: Path,
    images: str,
    iteration: int,
) -> list[str]:
    cmd = [
        args.python,
        "evaluate_geometry_colmap.py",
        "-s",
        str(source_path),
        "-m",
        str(output_model),
        "--images",
        images,
        "--resolution",
        str(args.resolution),
        "--eval",
        "--iteration",
        str(iteration),
        "--max_points_per_view",
        str(args.geometry_max_points_per_view),
        "--output",
        str(output_model / "geometry_eval_colmap" / f"iter_{iteration}_max{args.geometry_max_points_per_view}.json"),
    ]
    cmd.extend(_split_args(args.eval_split_strategy, split_file))
    return cmd


def _method_metrics(model_path: Path, method: str) -> dict[str, float] | None:
    results = model_path / "results.json"
    if not results.is_file():
        return None
    data = _load_json(results)
    row = data.get(method)
    if row is None:
        return None
    return {k: float(row[k]) for k in ("PSNR", "SSIM", "LPIPS") if k in row}


def _clean_metrics(clean_root: Path, scene: str) -> dict[str, dict[str, float]]:
    path = clean_root / scene / "results.json"
    if not path.is_file():
        return {}
    payload = _load_json(path)
    out = {}
    for method, row in payload.items():
        if all(k in row for k in ("PSNR", "SSIM", "LPIPS")):
            out[str(method)] = {k: float(row[k]) for k in ("PSNR", "SSIM", "LPIPS")}
    return out


def _score(metrics: dict[str, float]) -> float:
    return float(metrics["PSNR"]) + 20.0 * float(metrics["SSIM"]) - 20.0 * float(metrics["LPIPS"])


def _best_clean(clean: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]] | tuple[None, None]:
    if not clean:
        return None, None
    method = max(clean.keys(), key=lambda key: _score(clean[key]))
    return method, clean[method]


def _delta(method: dict[str, float] | None, baseline: dict[str, float] | None) -> dict[str, float] | None:
    if method is None or baseline is None:
        return None
    return {
        "dPSNR": float(method["PSNR"]) - float(baseline["PSNR"]),
        "dSSIM": float(method["SSIM"]) - float(baseline["SSIM"]),
        "dLPIPS": float(method["LPIPS"]) - float(baseline["LPIPS"]),
    }


def _beats_all(delta: dict[str, float] | None) -> bool:
    if delta is None:
        return False
    return bool(delta["dPSNR"] > 0.0 and delta["dSSIM"] > 0.0 and delta["dLPIPS"] < 0.0)


def _teacher_render_dir(output_model: Path, method_name: str) -> Path:
    return output_model / "train" / method_name / "renders"


def run_scene(args: argparse.Namespace, scene: str, phasef_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if scene not in phasef_rows:
        raise KeyError(f"scene {scene!r} not found in {args.phasef_summary}")
    phasef = phasef_rows[scene]
    source_model = Path(phasef["model"])
    if not source_model.is_dir():
        raise FileNotFoundError(f"missing Phase-F selected model: {source_model}")

    scene_out = Path(args.out_root) / scene
    output_model = scene_out / "recovery_model"
    contract_dir = scene_out / "contract"
    contract_dir.mkdir(parents=True, exist_ok=True)

    src_path = _source_path(Path(args.dataset_root), scene)
    split = _split_file(Path(args.split_root), scene)
    images = _scene_images(scene, args.outdoor_images, args.indoor_images)
    final_method = f"ours_{args.final_iteration}"

    _copy_model_shell(source_model, output_model, force=args.force_copy)
    parent_render_dir = _copy_base_train_evidence(source_model, output_model, args.base_method_name, force=args.force_evidence)

    teacher_dir = _teacher_render_dir(output_model, args.teacher_method_name)
    ela_cmd, teacher_policy = _ela_train_command(
        args,
        scene=scene,
        source_model=source_model,
        output_model=output_model,
        phasef=phasef,
    )
    train_cmd = _train_command(
        args,
        scene=scene,
        source_path=src_path,
        output_model=output_model,
        split_file=split,
        images=images,
        parent_render_dir=parent_render_dir,
        teacher_render_dir=teacher_dir,
    )
    render_cmd = _render_command(
        args,
        source_path=src_path,
        output_model=output_model,
        split_file=split,
        images=images,
        iteration=args.final_iteration,
    )
    metrics_cmd = [args.python, "metrics.py", "-m", str(output_model)]
    geometry_cmd = _geometry_command(
        args,
        source_path=src_path,
        output_model=output_model,
        split_file=split,
        images=images,
        iteration=args.final_iteration,
    )

    (contract_dir / "ela_train_teacher_command.txt").write_text(shlex.join(ela_cmd) + "\n", encoding="utf-8")
    (contract_dir / "exact_train_command.txt").write_text(shlex.join(train_cmd) + "\n", encoding="utf-8")
    (contract_dir / "render_command.txt").write_text(shlex.join(render_cmd) + "\n", encoding="utf-8")
    (contract_dir / "metrics_command.txt").write_text(shlex.join(metrics_cmd) + "\n", encoding="utf-8")
    (contract_dir / "geometry_command.txt").write_text(shlex.join(geometry_cmd) + "\n", encoding="utf-8")

    load_topology = _topology(output_model, args.load_iteration)
    if args.execute:
        env = os.environ.copy()
        if args.gpu >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if args.wandb_mode:
            env["WANDB_MODE"] = str(args.wandb_mode)
        if args.force_teacher or not teacher_dir.is_dir() or not any(teacher_dir.glob("*.png")):
            _run(ela_cmd, env=env, execute=True)
        else:
            print(f"[Phase-G] reuse teacher renders: {teacher_dir}", flush=True)
        final_checkpoint = _checkpoint_file_if_present(output_model, args.final_iteration)
        if args.force_train or final_checkpoint is None:
            _run(train_cmd, env=env, execute=True)
        else:
            print(f"[Phase-G] reuse final checkpoint: {final_checkpoint}", flush=True)
        if args.force_render or not (output_model / "test" / final_method / "renders").is_dir():
            _run(render_cmd, env=env, execute=True)
        else:
            print(f"[Phase-G] reuse final test renders: {output_model / 'test' / final_method / 'renders'}", flush=True)
        _run(metrics_cmd, env=env, execute=True)
        if not args.skip_geometry:
            _run(geometry_cmd, env=env, execute=True)

    final_topology = {}
    try:
        if checkpoint_path(output_model, args.final_iteration).is_file():
            final_topology = _topology(output_model, args.final_iteration)
    except FileNotFoundError:
        final_topology = {}
    topology_unchanged = bool(
        final_topology
        and final_topology.get("triangles") == load_topology.get("triangles")
        and final_topology.get("vertices") == load_topology.get("vertices")
    )

    method_metrics = _method_metrics(output_model, final_method)
    clean_all = _clean_metrics(Path(args.clean_root), scene)
    clean_method, clean_best = _best_clean(clean_all)
    delta_clean = _delta(method_metrics, clean_best)
    delta_phasef_ela = _delta(method_metrics, phasef.get("method"))
    delta_phasef_raw = _delta(method_metrics, phasef.get("source_ela"))

    summary = {
        "scene": scene,
        "source_model": str(source_model),
        "output_model": str(output_model),
        "images": images,
        "train_split_strategy": args.train_split_strategy,
        "eval_split_strategy": args.eval_split_strategy,
        "split_file": str(split) if args.train_split_strategy == "file" or args.eval_split_strategy == "file" else "",
        "load_iteration": int(args.load_iteration),
        "final_iteration": int(args.final_iteration),
        "final_method": final_method,
        "base_method_name": args.base_method_name,
        "teacher_method_name": args.teacher_method_name,
        "teacher_policy": teacher_policy,
        "teacher_render_dir": str(teacher_dir),
        "parent_render_dir": str(parent_render_dir),
        "wandb_project": args.wandb_project,
        "wandb_group": args.wandb_group,
        "wandb_prefix": args.wandb_prefix,
        "train_recipe": {
            "feature_lr": float(args.feature_lr),
            "weight_lr": float(args.weight_lr),
            "lr_triangles_points_init": float(args.lr_triangles_points_init),
            "teacher_render_lambda": float(args.teacher_render_lambda),
            "teacher_render_dssim": float(args.teacher_render_dssim),
            "teacher_render_mask_mode": args.teacher_render_mask_mode,
            "teacher_render_parent_delta_min": float(args.teacher_render_parent_delta_min),
            "parent_render_rollback_lambda": float(args.parent_render_rollback_lambda),
            "parent_render_rollback_aggregation": args.parent_render_rollback_aggregation,
            "parent_render_rollback_error_space": args.parent_render_rollback_error_space,
            "lpips_lambda": float(args.lpips_lambda),
        },
        "phasef": {
            "method": phasef.get("method"),
            "source_ela": phasef.get("source_ela"),
            "clean": phasef.get("clean"),
            "clean_baseline_method": phasef.get("clean_baseline_method"),
            "total_removed_fraction": phasef.get("total_removed_fraction"),
        },
        "clean_candidates": clean_all,
        "clean_best_method_by_score": clean_method,
        "clean_best_metrics": clean_best,
        "method_metrics": method_metrics,
        "delta_vs_clean_best": delta_clean,
        "delta_vs_phasef_render_time_ela": delta_phasef_ela,
        "delta_vs_phasef_source_ela": delta_phasef_raw,
        "beats_clean_best_all_three": _beats_all(delta_clean),
        "beats_phasef_render_time_ela_all_three": _beats_all(delta_phasef_ela),
        "topology": {
            "load": load_topology,
            "final": final_topology,
            "topology_unchanged": topology_unchanged,
            "required_flags": ["--freeze_topology_updates", "--skip_restricted_delaunay"],
        },
    }
    _write_json(contract_dir / "phaseg_scene_summary.json", summary)
    _write_json(output_model / "phaseg_scene_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def aggregate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("method_metrics")]
    mean = {}
    for key in ("delta_vs_clean_best", "delta_vs_phasef_render_time_ela", "delta_vs_phasef_source_ela"):
        deltas = [row.get(key) for row in valid if row.get(key)]
        if deltas:
            mean[key] = {
                "dPSNR": sum(float(x["dPSNR"]) for x in deltas) / len(deltas),
                "dSSIM": sum(float(x["dSSIM"]) for x in deltas) / len(deltas),
                "dLPIPS": sum(float(x["dLPIPS"]) for x in deltas) / len(deltas),
            }
    payload = {
        "args": vars(args),
        "rows": rows,
        "num_valid": len(valid),
        "mean_deltas": mean,
        "beats_clean_best_all_three": sum(1 for row in valid if row.get("beats_clean_best_all_three")),
        "beats_phasef_render_time_ela_all_three": sum(
            1 for row in valid if row.get("beats_phasef_render_time_ela_all_three")
        ),
        "topology_unchanged": sum(1 for row in valid if row.get("topology", {}).get("topology_unchanged")),
    }
    out_root = Path(args.out_root)
    _write_json(out_root / "phaseg_teacher_bake_summary.json", payload)
    md = [
        "# Phase-G Teacher-Bake Summary",
        "",
        f"- Scenes: {', '.join(row['scene'] for row in rows)}",
        f"- Valid metrics: {len(valid)}/{len(rows)}",
        f"- Beats clean-best on PSNR/SSIM/LPIPS: {payload['beats_clean_best_all_three']}/{len(valid)}",
        f"- Beats render-time Phase-F ELA on PSNR/SSIM/LPIPS: {payload['beats_phasef_render_time_ela_all_three']}/{len(valid)}",
        f"- Topology unchanged: {payload['topology_unchanged']}/{len(valid)}",
        "",
        "| scene | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | dPSNR Phase-F ELA | dSSIM Phase-F ELA | dLPIPS Phase-F ELA | topo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row.get("method_metrics") or {}
        dc = row.get("delta_vs_clean_best") or {}
        de = row.get("delta_vs_phasef_render_time_ela") or {}
        topo = row.get("topology", {}).get("topology_unchanged")
        md.append(
            "| {scene} | {psnr:.6f} | {ssim:.6f} | {lpips:.6f} | {dc_psnr:.6f} | {dc_ssim:.6f} | {dc_lpips:.6f} | {de_psnr:.6f} | {de_ssim:.6f} | {de_lpips:.6f} | {topo} |".format(
                scene=row.get("scene", ""),
                psnr=float(m.get("PSNR", float("nan"))),
                ssim=float(m.get("SSIM", float("nan"))),
                lpips=float(m.get("LPIPS", float("nan"))),
                dc_psnr=float(dc.get("dPSNR", float("nan"))),
                dc_ssim=float(dc.get("dSSIM", float("nan"))),
                dc_lpips=float(dc.get("dLPIPS", float("nan"))),
                de_psnr=float(de.get("dPSNR", float("nan"))),
                de_ssim=float(de.get("dSSIM", float("nan"))),
                de_lpips=float(de.get("dLPIPS", float("nan"))),
                topo=str(topo),
            )
        )
    (out_root / "phaseg_teacher_bake_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase-G representation-level teacher-baked ECSR recovery.")
    parser.add_argument("--scenes", default="bicycle")
    parser.add_argument("--dataset_root", default="/data/peilincai/mesh_datasets/mipnerf360")
    parser.add_argument("--split_root", default="outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train_colmap_file")
    parser.add_argument("--phasef_summary", default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_alpha0875_full9.json")
    parser.add_argument("--clean_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k")
    parser.add_argument("--out_root", default="outputs/carnet/meshsplatopt/ecsr_phase_g/teacher_bake_alpha0875_v1")
    parser.add_argument("--base_method_name", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument("--teacher_method_name", default="ours_26000_phaseg_alpha0875_train_teacher")
    parser.add_argument("--load_iteration", type=int, default=26000)
    parser.add_argument("--final_iteration", type=int, default=26200)
    parser.add_argument("--milestone_iterations", default="")
    parser.add_argument("--outdoor_images", default="images_4")
    parser.add_argument("--indoor_images", default="images_2")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--train_split_strategy", choices=("llff", "file"), default="llff")
    parser.add_argument("--eval_split_strategy", choices=("llff", "file"), default="llff")
    parser.add_argument("--alpha_grid", default="0,0.125,0.25,0.5,0.75,0.875,1.0")
    parser.add_argument("--policy_modes", default="residual")
    parser.add_argument("--policy_k_values", default="4,8")
    parser.add_argument("--policy_depth_rel_values", default="0.06,0.12")
    parser.add_argument("--policy_residual_clip_values", default="0.2,0.25")
    parser.add_argument("--policy_direction_weight_values", default="0.2,0.35")
    parser.add_argument("--policy_objective", choices=("psnr", "balanced"), default="balanced")
    parser.add_argument("--policy_ssim_weight", type=float, default=20.0)
    parser.add_argument("--policy_lpips_weight", type=float, default=20.0)
    parser.add_argument("--edge_gate_quantile", type=float, default=0.7)
    parser.add_argument("--edge_gate_dilate", type=int, default=1)
    parser.add_argument("--calib_sampler", choices=("stride_first", "uniform"), default="stride_first")
    parser.add_argument("--calib_max_views", type=int, default=16)
    parser.add_argument("--calib_stride", type=int, default=16)
    parser.add_argument("--calib_lpips", action="store_true", default=True)
    parser.add_argument("--no_calib_lpips", action="store_false", dest="calib_lpips")
    parser.add_argument("--teacher_policy_source", choices=("phasef_report", "auto"), default="phasef_report")
    parser.add_argument("--feature_lr", type=float, default=3e-5)
    parser.add_argument("--weight_lr", type=float, default=0.0)
    parser.add_argument("--lr_triangles_points_init", type=float, default=0.0)
    parser.add_argument("--teacher_render_lambda", type=float, default=0.05)
    parser.add_argument("--teacher_render_dssim", type=float, default=0.10)
    parser.add_argument("--teacher_render_mask_mode", default="teacher_better")
    parser.add_argument("--teacher_render_error_margin", type=float, default=0.0)
    parser.add_argument("--teacher_render_parent_delta_min", type=float, default=0.0)
    parser.add_argument("--teacher_render_warmup_iters", type=int, default=50)
    parser.add_argument("--teacher_render_decay_start_iter", type=int, default=-1)
    parser.add_argument("--teacher_render_decay_end_iter", type=int, default=-1)
    parser.add_argument("--teacher_render_decay_final_mult", type=float, default=1.0)
    parser.add_argument("--parent_render_rollback_lambda", type=float, default=3.0)
    parser.add_argument("--parent_render_rollback_warmup_iters", type=int, default=50)
    parser.add_argument("--parent_render_rollback_huber_delta", type=float, default=0.02)
    parser.add_argument("--parent_render_rollback_aggregation", choices=("mean", "cvar"), default="cvar")
    parser.add_argument("--parent_render_rollback_cvar_fraction", type=float, default=0.1)
    parser.add_argument("--parent_render_rollback_cvar_min_pixels", type=int, default=1024)
    parser.add_argument(
        "--parent_render_rollback_error_space",
        choices=("l1", "l2", "channel_max", "l1_dssim", "l1_edge", "l1_dssim_edge"),
        default="l1_dssim_edge",
    )
    parser.add_argument("--parent_render_rollback_dssim_weight", type=float, default=0.25)
    parser.add_argument("--parent_render_rollback_edge_weight", type=float, default=0.10)
    parser.add_argument("--lpips_lambda", type=float, default=0.0)
    parser.add_argument("--lpips_warmup_iters", type=int, default=50)
    parser.add_argument("--lpips_max_side", type=int, default=512)
    parser.add_argument("--geometry_max_points_per_view", type=int, default=500)
    parser.add_argument("--wandb_project", default="mesh-splatting-ecsr")
    parser.add_argument("--wandb_group", default="phase_g_teacher_bake_alpha0875")
    parser.add_argument("--wandb_prefix", default="phaseg_alpha0875_v1")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb_image_log_interval", type=int, default=1000)
    parser.add_argument("--wandb_scalar_log_interval", type=int, default=50)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--python", default="/home/peilincai/micromamba/envs/mesh_splatting/bin/python")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force_copy", action="store_true")
    parser.add_argument("--force_evidence", action="store_true")
    parser.add_argument("--force_teacher", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_render", action="store_true")
    parser.add_argument("--skip_geometry", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scenes = _parse_scenes(args.scenes)
    phasef_rows = _load_phasef_rows(Path(args.phasef_summary))
    rows = [run_scene(args, scene, phasef_rows) for scene in scenes]
    aggregate(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
