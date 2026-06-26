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
    return f"ours_{int(iteration)}_v113b_oot_strict_parent_gate_{scene}"


def _shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path:
        return Path(args.model_path).expanduser().resolve()
    return (Path(args.package_root).expanduser() / args.scene / "detached_model").resolve()


def _resolve_candidate_manifest(args: argparse.Namespace, candidate_field: Path) -> Path:
    if args.candidate_manifest_path:
        return Path(args.candidate_manifest_path).expanduser().resolve()
    return candidate_field.with_suffix(".manifest.json").resolve()


def _resolve_candidate_field(args: argparse.Namespace, output_root: Path, candidate_method: str) -> Path:
    if args.candidate_field_path:
        return Path(args.candidate_field_path).expanduser().resolve()
    return (output_root / args.scene / "fields" / f"{candidate_method}_field.pt").resolve()


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    scene = str(args.scene)
    iteration = int(args.iteration)
    python = str(args.python)
    model_path = _resolve_model_path(args)
    output_root = Path(args.output_root).expanduser().resolve()
    run_root = output_root / scene
    logs_dir = run_root / "logs_v113b"
    reports_dir = run_root / "reports"

    parent_method = args.parent_method_name or _default_parent_method(scene, iteration)
    candidate_method = args.candidate_method_name or _default_candidate_method(scene, iteration)
    gate_method = args.gate_method_name or _default_gate_method(scene, iteration)
    report_stem = f"{scene}_{gate_method}_oot_tail_gate_replay"
    candidate_field = _resolve_candidate_field(args, output_root, candidate_method)
    candidate_manifest = _resolve_candidate_manifest(args, candidate_field)
    gate_output_model_path = Path(args.gate_output_model_path).expanduser().resolve() if args.gate_output_model_path else model_path

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
        "--min_mean_psnr_gain",
        str(args.min_mean_psnr_gain),
        f"--min_mean_ssim_gain={args.min_mean_ssim_gain}",
        f"--min_mean_lpips_gain={args.min_mean_lpips_gain}",
        f"--min_p05_score_gain={args.min_p05_score_gain}",
        f"--min_p05_psnr_gain={args.min_p05_psnr_gain}",
        f"--min_p05_ssim_gain={args.min_p05_ssim_gain}",
        f"--min_p05_lpips_gain={args.min_p05_lpips_gain}",
        "--oot_gate_mode",
        str(args.oot_gate_mode),
        "--oot_source_manifest",
        str(candidate_manifest),
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
        str(args.wandb_name or f"v113b-oot-tail-{scene}"),
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
        _step("gate_train_odd_to_test_v113b", gate_cmd, logs_dir / "01_v113b_parent_gate.log"),
        _step("evaluate_v113b_test", eval_cmd, logs_dir / "02_v113b_evaluate_test.log"),
    ]
    return {
        "schema_version": 1,
        "method": "v113b OOT tail-safe parent-gate replay",
        "scene": scene,
        "iteration": iteration,
        "model_path": str(model_path),
        "output_root": str(output_root),
        "run_root": str(run_root),
        "logs_dir": str(logs_dir),
        "reports_dir": str(reports_dir),
        "parent_method_name": parent_method,
        "candidate_method_name": candidate_method,
        "gate_method_name": gate_method,
        "candidate_field_path": str(candidate_field),
        "candidate_manifest_path": str(candidate_manifest),
        "gate_output_model_path": str(gate_output_model_path),
        "strict_split": {
            "calib_split": args.calib_split,
            "calib_view_subset": args.calib_view_subset,
            "target_split": args.target_split,
            "test_gt_usage": "final evaluation only",
            "candidate_field_source": "prebuilt train/even candidate",
        },
        "paths": {
            "parent_train_dir": str(model_path / args.calib_split / parent_method),
            "candidate_train_dir": str(model_path / args.calib_split / candidate_method),
            "parent_test_dir": str(model_path / args.target_split / parent_method),
            "candidate_test_dir": str(model_path / args.target_split / candidate_method),
            "gated_test_dir": str(gate_output_model_path / args.target_split / gate_method),
            "gate_report": str(gate_output_model_path / args.target_split / gate_method / "v109_render_realized_parent_gate_report.json"),
            "eval_output": str(eval_output),
            "per_view_output": str(per_view_output),
            "orchestration_report_json": str(reports_dir / f"{report_stem}_report.json"),
            "orchestration_report_md": str(reports_dir / f"{report_stem}_report.md"),
        },
        "settings": {
            "oot_gate_mode": args.oot_gate_mode,
            "min_p05_psnr_gain": args.min_p05_psnr_gain,
            "oot_center_quantile": args.oot_center_quantile,
            "oot_center_rel_margin": args.oot_center_rel_margin,
            "oot_center_abs_margin": args.oot_center_abs_margin,
            "oot_max_mask_weighted_fraction": args.oot_max_mask_weighted_fraction,
            "wandb": bool(args.wandb),
            "wandb_mode": args.wandb_mode,
            "gpu": args.gpu,
        },
        "steps": steps,
    }


def _preflight(plan: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if args.dry_run:
        return []
    missing: list[str] = []
    for key in ("model_path", "candidate_field_path", "candidate_manifest_path"):
        if not Path(str(plan[key])).exists():
            missing.append(f"missing {key}: {plan[key]}")
    for key in ("parent_train_dir", "candidate_train_dir", "parent_test_dir", "candidate_test_dir"):
        if not Path(str(plan["paths"][key])).is_dir():
            missing.append(f"missing render directory {key}: {plan['paths'][key]}")
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


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# v113b OOT tail-safe replay - {report.get('scene')}",
        "",
        f"- status: `{report.get('status')}`",
        f"- model: `{report.get('model_path')}`",
        f"- parent method: `{report.get('parent_method_name')}`",
        f"- candidate method: `{report.get('candidate_method_name')}`",
        f"- gated method: `{report.get('gate_method_name')}`",
        f"- candidate manifest: `{report.get('candidate_manifest_path')}`",
        f"- strict split: `{json.dumps(report.get('strict_split', {}), sort_keys=True)}`",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(report.get("metrics", {}), indent=2, sort_keys=True),
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


def run_plan(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for path in (Path(str(plan["logs_dir"])), Path(str(plan["reports_dir"]))):
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
    gate_report_path = Path(str(report["paths"]["gate_report"]))
    metrics_json = _read_json(metrics_path)
    report["metrics"] = {
        "results_path": str(metrics_path),
        "per_view_path": str(report["paths"]["per_view_output"]),
        "gate_report_path": str(gate_report_path),
        "gated_method": _method_metrics(metrics_json, str(report["gate_method_name"])),
        "all_methods_in_eval_output": metrics_json,
    }
    report["gate_report_summary"] = _read_json(gate_report_path)
    return _finalize_report(report)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay v113b OOT tail-safe gate/eval from prebuilt strict candidate renders.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--package_root", default="/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--output_root", default=f"/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_{_default_run_tag()}")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "spcarnet_meshprior"))
    parser.add_argument("--wandb_group", default="v113_oot_tail_parent_gate")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb", action="store_true")

    parser.add_argument("--parent_method_name", default="")
    parser.add_argument("--candidate_method_name", default="")
    parser.add_argument("--gate_method_name", default="")
    parser.add_argument("--gate_output_model_path", default="")
    parser.add_argument("--candidate_field_path", default="")
    parser.add_argument("--candidate_manifest_path", default="")
    parser.add_argument("--iteration", type=int, default=26000)

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
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "report_json": report.get("report_json"),
                "report_md": report.get("report_md"),
                "errors": report.get("errors", []),
                "metrics": report.get("metrics", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.get("status") in {"COMPLETE", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
