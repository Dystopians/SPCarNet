#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _default_run_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d")


def _default_parent_method(scene: str, iteration: int) -> str:
    return f"ours_{int(iteration)}_v106_podmoe_basepreserve_{scene}"


def _default_candidate_method(scene: str, iteration: int) -> str:
    return f"ours_{int(iteration)}_v110_strict_train_even_candidate_{scene}"


def _default_gate_method(scene: str, iteration: int) -> str:
    return f"ours_{int(iteration)}_v110_strict_train_even_odd_parent_gate_{scene}"


def _resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path:
        return Path(args.model_path).expanduser().resolve()
    return (Path(args.package_root).expanduser() / args.scene / "detached_model").resolve()


def _resolve_train_bank(args: argparse.Namespace) -> Path:
    if args.train_v102_bank_path:
        return Path(args.train_v102_bank_path).expanduser().resolve()
    scene_dir = Path(args.train_v102_bank_root).expanduser() / args.scene
    names = [args.train_v102_bank_name]
    if args.train_v102_bank_name != "v102_preprojected_delta_bank_train.pt":
        names.append("v102_preprojected_delta_bank_train.pt")
    if args.train_v102_bank_name != "v102_preprojected_delta_bank.pt":
        names.append("v102_preprojected_delta_bank.pt")
    for name in names:
        path = scene_dir / name
        if path.is_file():
            return path.resolve()
    return (scene_dir / args.train_v102_bank_name).resolve()


def _split_method_dir(model_path: Path, split: str, method: str) -> Path:
    return model_path / split / method


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _method_metrics(results: dict[str, Any], method: str) -> dict[str, Any]:
    value = results.get(method, {})
    return value if isinstance(value, dict) else {}


def _step(name: str, cmd: list[str], log_path: Path) -> dict[str, Any]:
    return {
        "name": name,
        "cmd": [str(part) for part in cmd],
        "cmd_string": _shell_join([str(part) for part in cmd]),
        "log_path": str(log_path),
        "returncode": None,
        "elapsed_sec": None,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    scene = str(args.scene)
    iteration = int(args.iteration)
    python = str(args.python)
    model_path = _resolve_model_path(args)
    train_bank = _resolve_train_bank(args)
    output_root = Path(args.output_root).expanduser().resolve()
    run_root = output_root / scene
    fields_dir = run_root / "fields"
    logs_dir = run_root / "logs"
    reports_dir = run_root / "reports"

    parent_method = args.parent_method_name or _default_parent_method(scene, iteration)
    candidate_method = args.candidate_method_name or _default_candidate_method(scene, iteration)
    gate_method = args.gate_method_name or _default_gate_method(scene, iteration)
    gate_output_model_path = Path(args.gate_output_model_path).expanduser().resolve() if args.gate_output_model_path else model_path
    candidate_field = (
        Path(args.candidate_field_path).expanduser().resolve()
        if args.candidate_field_path
        else fields_dir / f"{candidate_method}_field.pt"
    )

    build_cmd = [
        python,
        "scripts/car_model/build_v105_evidence_gated_mixture_field.py",
        "--model_path",
        str(model_path),
        "--delta_bank_path",
        str(train_bank),
        "--output_field",
        str(candidate_field),
        "--endpoint_method",
        str(args.endpoint_method),
        "--iteration",
        str(iteration),
        "--split",
        str(args.field_split),
        "--view_subset",
        str(args.field_view_subset),
        "--renderer_scaling",
        str(args.renderer_scaling),
        "--residual_dtype",
        str(args.residual_dtype),
        "--field_variant",
        str(args.field_variant),
        "--method_version",
        str(args.method_version),
        "--gate_source",
        str(args.gate_source),
        "--view_gate_temperature",
        str(args.view_gate_temperature),
        "--min_count",
        str(args.min_count),
        "--min_views",
        str(args.min_views),
        "--ridge",
        str(args.ridge),
        "--residual_clip",
        str(args.residual_clip),
        "--view_std_floor",
        str(args.view_std_floor),
        "--rank_rtol",
        str(args.rank_rtol),
        "--condition_max",
        str(args.condition_max),
        "--gate_boost",
        str(args.gate_boost),
        "--chunk_pixels",
        str(args.chunk_pixels),
    ]

    render_cmd = [
        python,
        "render.py",
        "-m",
        str(model_path),
        "--iteration",
        str(iteration),
        "--checkpoint_endpoint_method",
        str(args.endpoint_method),
        "--checkpoint_endpoint_output_method",
        candidate_method,
        "--checkpoint_endpoint_surface_field_path",
        str(candidate_field),
        "--checkpoint_endpoint_require_surface_field",
        "--checkpoint_endpoint_no_intermediate_outputs",
        "--quiet",
    ]
    if args.render_skip_train:
        render_cmd.append("--skip_train")
    if args.render_skip_test:
        render_cmd.append("--skip_test")

    gate_cmd = [
        python,
        "scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py",
        "--parent_model_path",
        str(model_path),
        "--parent_method_name",
        parent_method,
        "--candidate_model_path",
        str(model_path),
        "--candidate_method_name",
        candidate_method,
        "--output_model_path",
        str(gate_output_model_path),
        "--method_name",
        gate_method,
        "--calib_split",
        str(args.calib_split),
        "--calib_view_subset",
        str(args.calib_view_subset),
        "--target_split",
        str(args.target_split),
        "--threshold_grid",
        str(args.threshold_grid),
        "--frame_threshold_grid",
        str(args.frame_threshold_grid),
        f"--frame_threshold_quantile={args.frame_threshold_quantile}",
        "--softness_grid",
        str(args.softness_grid),
        "--max_blend_grid",
        str(args.max_blend_grid),
        "--local_kernels",
        str(args.local_kernels),
        "--mask_dilate",
        str(args.mask_dilate),
        "--calib_max_views",
        str(args.calib_max_views),
        "--calib_sampler",
        str(args.calib_sampler),
        "--objective",
        str(args.objective),
        "--ssim_weight",
        str(args.ssim_weight),
        "--lpips_weight",
        str(args.lpips_weight),
        "--min_mask_mean",
        str(args.min_mask_mean),
        "--target_mask_mean",
        str(args.target_mask_mean),
        "--max_mean_mse_increase",
        str(args.max_mean_mse_increase),
        "--max_p95_mse_increase",
        str(args.max_p95_mse_increase),
        f"--min_mean_psnr_gain={args.min_mean_psnr_gain}",
        f"--min_mean_ssim_gain={args.min_mean_ssim_gain}",
        f"--min_mean_lpips_gain={args.min_mean_lpips_gain}",
        f"--min_p05_score_gain={args.min_p05_score_gain}",
        f"--min_p05_psnr_gain={args.min_p05_psnr_gain}",
        f"--min_p05_ssim_gain={args.min_p05_ssim_gain}",
        f"--min_p05_lpips_gain={args.min_p05_lpips_gain}",
        "--oot_gate_mode",
        str(args.oot_gate_mode),
        "--oot_source_manifest",
        str(candidate_field.with_suffix(".manifest.json")),
        "--oot_source_view_subset",
        str(args.oot_source_view_subset),
        "--oot_center_quantile",
        str(args.oot_center_quantile),
        "--oot_center_rel_margin",
        str(args.oot_center_rel_margin),
        "--oot_center_abs_margin",
        str(args.oot_center_abs_margin),
        "--oot_max_frame_fraction",
        str(args.oot_max_frame_fraction),
        "--oot_max_mask_weighted_fraction",
        str(args.oot_max_mask_weighted_fraction),
        "--oot_min_mask_mean_for_scene_check",
        str(args.oot_min_mask_mean_for_scene_check),
        "--device",
        str(args.device),
        "--wandb_mode",
        str(args.wandb_mode),
        "--wandb_project",
        str(args.wandb_project),
        "--wandb_group",
        str(args.wandb_group),
        "--wandb_name",
        str(args.wandb_name or f"v110-strict-{scene}"),
    ]
    if args.calib_lpips:
        gate_cmd.append("--calib_lpips")
    if args.allow_failed_policy:
        gate_cmd.append("--allow_failed_policy")
    if args.wandb:
        gate_cmd.append("--wandb")

    eval_output = reports_dir / f"{scene}_{gate_method}_{args.target_split}_results.json"
    per_view_output = reports_dir / f"{scene}_{gate_method}_{args.target_split}_per_view.json"
    eval_cmd = [
        python,
        "scripts/car_model/evaluate_render_split_metrics.py",
        "-m",
        str(gate_output_model_path),
        "--split",
        str(args.target_split),
        "--methods",
        gate_method,
        "--output",
        str(eval_output),
        "--per_view_output",
        str(per_view_output),
    ]
    if args.merge_model_results:
        eval_cmd.append("--merge_model_results")

    steps = [
        _step("build_train_even_candidate_field", build_cmd, logs_dir / "01_build_candidate_field.log"),
        _step("render_candidate_train_and_test", render_cmd, logs_dir / "02_render_candidate.log"),
        _step("gate_train_odd_to_test", gate_cmd, logs_dir / "03_parent_gate.log"),
        _step("evaluate_gated_test", eval_cmd, logs_dir / "04_evaluate_gated_test.log"),
    ]
    return {
        "schema_version": 1,
        "method": "v110 strict-split parent-gated orchestration",
        "scene": scene,
        "iteration": iteration,
        "model_path": str(model_path),
        "train_v102_bank_path": str(train_bank),
        "output_root": str(output_root),
        "run_root": str(run_root),
        "fields_dir": str(fields_dir),
        "logs_dir": str(logs_dir),
        "reports_dir": str(reports_dir),
        "candidate_field_path": str(candidate_field),
        "parent_method_name": parent_method,
        "candidate_method_name": candidate_method,
        "gate_method_name": gate_method,
        "gate_output_model_path": str(gate_output_model_path),
        "strict_split": {
            "field_split": args.field_split,
            "field_view_subset": args.field_view_subset,
            "calib_split": args.calib_split,
            "calib_view_subset": args.calib_view_subset,
            "target_split": args.target_split,
            "test_gt_usage": "final evaluation only",
        },
        "paths": {
            "parent_train_dir": str(_split_method_dir(model_path, args.calib_split, parent_method)),
            "candidate_train_dir": str(_split_method_dir(model_path, args.calib_split, candidate_method)),
            "parent_test_dir": str(_split_method_dir(model_path, args.target_split, parent_method)),
            "candidate_test_dir": str(_split_method_dir(model_path, args.target_split, candidate_method)),
            "gated_test_dir": str(_split_method_dir(gate_output_model_path, args.target_split, gate_method)),
            "gate_report": str(
                _split_method_dir(gate_output_model_path, args.target_split, gate_method)
                / "v109_render_realized_parent_gate_report.json"
            ),
            "eval_output": str(eval_output),
            "per_view_output": str(per_view_output),
            "orchestration_report_json": str(reports_dir / f"{scene}_v110_strict_split_parent_gate_report.json"),
            "orchestration_report_md": str(reports_dir / f"{scene}_v110_strict_split_parent_gate_report.md"),
        },
        "settings": {
            "endpoint_method": args.endpoint_method,
            "renderer_scaling": args.renderer_scaling,
            "residual_dtype": args.residual_dtype,
            "field_variant": args.field_variant,
            "method_version": args.method_version,
            "gate_source": args.gate_source,
            "view_gate_temperature": args.view_gate_temperature,
            "threshold_grid": args.threshold_grid,
            "frame_threshold_grid": args.frame_threshold_grid,
            "softness_grid": args.softness_grid,
            "max_blend_grid": args.max_blend_grid,
            "local_kernels": args.local_kernels,
            "objective": args.objective,
            "min_p05_psnr_gain": args.min_p05_psnr_gain,
            "min_p05_ssim_gain": args.min_p05_ssim_gain,
            "min_p05_lpips_gain": args.min_p05_lpips_gain,
            "oot_gate_mode": args.oot_gate_mode,
            "oot_center_quantile": args.oot_center_quantile,
            "oot_center_rel_margin": args.oot_center_rel_margin,
            "oot_center_abs_margin": args.oot_center_abs_margin,
            "wandb": bool(args.wandb),
            "wandb_mode": args.wandb_mode,
            "gpu": args.gpu,
        },
        "steps": steps,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    metrics = report.get("metrics", {})
    lines = [
        f"# v110 strict split parent gate - {report.get('scene')}",
        "",
        f"- status: `{report.get('status')}`",
        f"- model: `{report.get('model_path')}`",
        f"- train bank: `{report.get('train_v102_bank_path')}`",
        f"- parent method: `{report.get('parent_method_name')}`",
        f"- candidate method: `{report.get('candidate_method_name')}`",
        f"- gated method: `{report.get('gate_method_name')}`",
        f"- strict split: `{json.dumps(report.get('strict_split', {}), sort_keys=True)}`",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Commands",
        "",
    ]
    for step in report.get("steps", []):
        lines.extend(
            [
                f"### {step.get('name')}",
                "",
                f"- returncode: `{step.get('returncode')}`",
                f"- elapsed_sec: `{step.get('elapsed_sec')}`",
                f"- log: `{step.get('log_path')}`",
                "",
                "```bash",
                str(step.get("cmd_string", "")),
                "```",
                "",
            ]
        )
    if report.get("errors"):
        lines.extend(["## Errors", "", "```json", json.dumps(report["errors"], indent=2, sort_keys=True), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _preflight(plan: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if args.dry_run:
        return []
    missing: list[str] = []
    for key in ("model_path", "train_v102_bank_path"):
        if not Path(str(plan[key])).exists():
            missing.append(f"missing {key}: {plan[key]}")
    for rel in (
        "render.py",
        "scripts/car_model/build_v105_evidence_gated_mixture_field.py",
        "scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py",
        "scripts/car_model/evaluate_render_split_metrics.py",
    ):
        path = ROOT / rel
        if not path.is_file():
            missing.append(f"missing script: {path}")
    paths = plan.get("paths", {})
    for key in ("parent_train_dir", "parent_test_dir"):
        if not Path(str(paths.get(key, ""))).is_dir():
            missing.append(f"missing required parent render directory {key}: {paths.get(key)}")
    return missing


def _run_step(step: dict[str, Any], env: dict[str, str], dry_run: bool) -> dict[str, Any]:
    log_path = Path(str(step["log_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if dry_run:
        step["returncode"] = 0
        step["elapsed_sec"] = 0.0
        log_path.write_text("[dry-run] command was not executed\n" + step["cmd_string"] + "\n", encoding="utf-8")
        return step
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            step["cmd"],
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    step["returncode"] = int(proc.returncode)
    step["elapsed_sec"] = float(time.time() - start)
    return step


def run_plan(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    reports_dir = Path(str(plan["reports_dir"]))
    fields_dir = Path(str(plan["fields_dir"]))
    logs_dir = Path(str(plan["logs_dir"]))
    for path in (reports_dir, fields_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    report = dict(plan)
    report["created_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    report["dry_run"] = bool(args.dry_run)
    report["status"] = "DRY_RUN" if args.dry_run else "RUNNING"
    report["errors"] = []

    preflight_errors = _preflight(plan, args)
    if preflight_errors:
        report["status"] = "FAILED_PREFLIGHT"
        report["errors"].extend(preflight_errors)
        return _finalize_report(report)

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", str(args.wandb_mode))
    env["WANDB_MODE"] = str(args.wandb_mode)
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    completed_steps: list[dict[str, Any]] = []
    for step in report["steps"]:
        completed = _run_step(step, env, args.dry_run)
        completed_steps.append(completed)
        if int(completed.get("returncode") or 0) != 0:
            report["status"] = "FAILED"
            report["errors"].append(
                {
                    "step": completed.get("name"),
                    "returncode": completed.get("returncode"),
                    "log_path": completed.get("log_path"),
                }
            )
            if not args.continue_on_error:
                break
    report["steps"] = completed_steps + report["steps"][len(completed_steps) :]

    if not report["errors"]:
        report["status"] = "DRY_RUN" if args.dry_run else "COMPLETE"

    metrics_path = Path(str(report["paths"]["eval_output"]))
    per_view_path = Path(str(report["paths"]["per_view_output"]))
    gate_report_path = Path(str(report["paths"]["gate_report"]))
    metrics_json = _read_json(metrics_path)
    report["metrics"] = {
        "results_path": str(metrics_path),
        "per_view_path": str(per_view_path),
        "gate_report_path": str(gate_report_path),
        "gated_method": _method_metrics(metrics_json, str(report["gate_method_name"])),
        "all_methods_in_eval_output": metrics_json,
    }
    report["gate_report_summary"] = _read_json(gate_report_path)
    return _finalize_report(report)


def _finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    report["finished_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    report_path = Path(str(report["paths"]["orchestration_report_json"]))
    md_path = Path(str(report["paths"]["orchestration_report_md"]))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    report["report_json"] = str(report_path)
    report["report_md"] = str(md_path)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-scene v110 strict-split parent-gated orchestration.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--package_root", default="/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--train_v102_bank_root", default="/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_train_20260625")
    parser.add_argument("--train_v102_bank_name", default="v102_preprojected_delta_bank_train.pt")
    parser.add_argument("--train_v102_bank_path", default="")
    parser.add_argument("--output_root", default=f"/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_{_default_run_tag()}")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "spcarnet_meshprior"))
    parser.add_argument("--wandb_group", default="v110_strict_split_parent_gate")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb", action="store_true")

    parser.add_argument("--parent_method_name", default="")
    parser.add_argument("--candidate_method_name", default="")
    parser.add_argument("--gate_method_name", default="")
    parser.add_argument("--gate_output_model_path", default="")
    parser.add_argument("--candidate_field_path", default="")

    parser.add_argument("--endpoint_method", default="ours_26000_v100_checkpoint_attached_ela_endpoint")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--field_split", choices=("train", "test"), default="train")
    parser.add_argument("--field_view_subset", choices=("all", "even", "odd"), default="even")
    parser.add_argument("--renderer_scaling", type=int, default=4)
    parser.add_argument("--residual_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--field_variant", choices=("residual_mixture", "pod_moe"), default="pod_moe")
    parser.add_argument("--method_version", default="v108_mse_descent_locked_pod_moe")
    parser.add_argument("--gate_source", choices=("normal_equation", "crossfit_risk", "optimal_risk"), default="normal_equation")
    parser.add_argument("--view_gate_temperature", type=float, default=0.0)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--min_views", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--residual_clip", type=float, default=0.08)
    parser.add_argument("--view_std_floor", type=float, default=1e-4)
    parser.add_argument("--rank_rtol", type=float, default=1e-7)
    parser.add_argument("--condition_max", type=float, default=1e8)
    parser.add_argument("--gate_boost", type=float, default=0.5)
    parser.add_argument("--chunk_pixels", type=int, default=262144)

    parser.add_argument("--render_skip_train", action="store_true")
    parser.add_argument("--render_skip_test", action="store_true")
    parser.add_argument("--calib_split", choices=("train", "test"), default="train")
    parser.add_argument("--calib_view_subset", choices=("all", "even", "odd"), default="odd")
    parser.add_argument("--target_split", choices=("train", "test"), default="test")
    parser.add_argument("--threshold_grid", default="0.0005,0.001,0.002,0.004,0.006,0.008,0.010,0.014,0.020")
    parser.add_argument("--frame_threshold_grid", default="0")
    parser.add_argument("--frame_threshold_quantile", type=float, default=-1.0)
    parser.add_argument("--softness_grid", default="0,0.0005,0.001,0.002")
    parser.add_argument("--max_blend_grid", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--local_kernels", default="1,9,25")
    parser.add_argument("--mask_dilate", type=int, default=0)
    parser.add_argument("--calib_max_views", type=int, default=64)
    parser.add_argument("--calib_sampler", choices=("uniform", "first"), default="uniform")
    parser.add_argument("--objective", choices=("psnr", "balanced"), default="balanced")
    parser.add_argument("--ssim_weight", type=float, default=20.0)
    parser.add_argument("--lpips_weight", type=float, default=20.0)
    parser.add_argument("--calib_lpips", action="store_true")
    parser.add_argument("--min_mask_mean", type=float, default=0.0)
    parser.add_argument("--target_mask_mean", type=float, default=0.25)
    parser.add_argument("--max_mean_mse_increase", type=float, default=0.0)
    parser.add_argument("--max_p95_mse_increase", type=float, default=0.0)
    parser.add_argument("--min_mean_psnr_gain", type=float, default=0.5)
    parser.add_argument("--min_mean_ssim_gain", type=float, default=-1e-6)
    parser.add_argument("--min_mean_lpips_gain", type=float, default=-1e9)
    parser.add_argument("--min_p05_score_gain", type=float, default=-1e-4)
    parser.add_argument("--min_p05_psnr_gain", type=float, default=0.0)
    parser.add_argument("--min_p05_ssim_gain", type=float, default=-1e-6)
    parser.add_argument("--min_p05_lpips_gain", type=float, default=-1e9)
    parser.add_argument("--oot_gate_mode", choices=("off", "report", "scene_fallback", "frame_fallback"), default="scene_fallback")
    parser.add_argument("--oot_source_view_subset", choices=("all", "even", "odd"), default="even")
    parser.add_argument("--oot_center_quantile", type=float, default=0.95)
    parser.add_argument("--oot_center_rel_margin", type=float, default=0.0)
    parser.add_argument("--oot_center_abs_margin", type=float, default=0.0)
    parser.add_argument("--oot_max_frame_fraction", type=float, default=0.10)
    parser.add_argument("--oot_max_mask_weighted_fraction", type=float, default=0.05)
    parser.add_argument("--oot_min_mask_mean_for_scene_check", type=float, default=0.05)
    parser.add_argument("--allow_failed_policy", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--merge_model_results", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    report = run_plan(plan, args)
    print(json.dumps({
        "status": report.get("status"),
        "report_json": report.get("report_json"),
        "report_md": report.get("report_md"),
        "errors": report.get("errors", []),
        "metrics": report.get("metrics", {}),
    }, indent=2, sort_keys=True))
    return 0 if report.get("status") in {"COMPLETE", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
