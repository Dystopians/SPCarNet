#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from plan_v52_capacity_aware_source_rerun import (  # noqa: E402
    build_v48_commands,
    build_v51_commands,
    quote_cmd,
)


DEFAULT_ROOT = Path("outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware")
DEFAULT_V56_SUMMARY = DEFAULT_ROOT / "v56_face_alpha_guard_full9_summary.json"
DEFAULT_V52_SUMMARY = DEFAULT_ROOT / "v52_capacity_aware_v48_v51_full9_summary.json"
DEFAULT_V55D_ROOT = Path("/dev/shm/peilincai_spcarnet_v55d_face_alpha_caphit_20260623")
DEFAULT_PLAN_JSON = DEFAULT_ROOT / "v56_face_alpha_guard_source_rerun_plan.json"
DEFAULT_PLAN_MD = DEFAULT_ROOT / "v56_face_alpha_guard_source_rerun_plan.md"
DEFAULT_PLAN_SH = DEFAULT_ROOT / "v56_face_alpha_guard_source_rerun_plan.sh"
DEFAULT_REFRESH_JSON = DEFAULT_ROOT / "v56_face_alpha_guard_with_source_rerun_summary.json"
DEFAULT_REFRESH_MD = DEFAULT_ROOT / "v56_face_alpha_guard_with_source_rerun_summary.md"
V55D_TAG = "v55d_policyval_face_alpha_l1pos09_support4096_tex32_nearest_region_texture_adapter"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_arg(argv: list[str], name: str, value: Any) -> None:
    text = str(value)
    if text.startswith("-"):
        argv.append(f"--{name}={text}")
    else:
        argv.extend([f"--{name}", text])


def build_v55d_candidate_command(
    scene: str,
    output_root: Path,
    python: str,
    gpu: str,
    force: bool,
    v48_roots: str,
    min_target_changed_fraction: float,
    wandb_project: str,
    wandb_group: str,
    wandb_run_prefix: str,
    wandb_mode: str,
) -> dict[str, Any]:
    cmd = [
        python,
        "scripts/car_model/run_l1risk_fairnoop_scene.py",
        "--scene",
        scene,
        "--gpu",
        gpu,
        "--output_root",
        str(output_root),
        "--tag",
        V55D_TAG,
        "--v48_roots",
        v48_roots,
        "--support_expansion_max_extra_faces_candidates",
        "4096",
        "--texture_size_candidates",
        "32",
        "--atlas_empty_bin_fill_mode",
        "nearest_observed",
        "--enable_policy_val_face_alpha_calibration",
        "--face_alpha_calibration_max_alpha",
        "0.5",
        "--face_alpha_calibration_min_alpha",
        "0.0",
        "--face_alpha_calibration_multipliers",
        "0.5,0.75,1.0,1.25",
        "--face_alpha_calibration_min_face_samples",
        "256",
        "--min_policy_val_l1_positive_view_fraction",
        "0.9",
        "--min_target_changed_fraction",
        str(float(min_target_changed_fraction)),
    ]
    if wandb_project:
        cmd.extend(
            [
                "--wandb_project",
                wandb_project,
                "--wandb_group",
                wandb_group or "v56_face_alpha_guard_source_rerun",
                "--wandb_run_name",
                f"{wandb_run_prefix or 'v56_v55d'}_{scene}_{time.strftime('%Y%m%d_%H%M%S')}",
                "--wandb_mode",
                wandb_mode,
            ]
        )
    if force:
        cmd.append("--force")
    env = {"CUDA_VISIBLE_DEVICES": gpu, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    return {
        "scene": scene,
        "selected_source": "v55d_face_alpha",
        "underlying_source": "v55d_candidate",
        "output_model": str(output_root / f"{scene}_{V55D_TAG}"),
        "method_name": f"ours_26000_{scene}_{V55D_TAG}",
        "gpu": gpu,
        "commands": [{"name": "runner_apply_metrics", "argv": cmd, "env": env}],
        "required_missing": [],
    }


def build_v52_fallback_job(
    v52_row: dict[str, Any],
    output_root: Path,
    python: str,
    gpu: str,
    force: bool,
    v48_roots: str,
) -> dict[str, Any]:
    scene = str(v52_row["scene"])
    selected = str(v52_row["selected_source"])
    if selected == "v48":
        job = build_v48_commands(
            scene=scene,
            source_dir=Path(str(v52_row["selected_source_dir"])),
            output_root=output_root,
            python=python,
            gpu=gpu,
            force=force,
        )
    elif selected == "v51":
        job = build_v51_commands(
            scene=scene,
            output_root=output_root,
            python=python,
            gpu=gpu,
            force=force,
            v48_roots=v48_roots,
        )
    else:
        raise RuntimeError(f"unsupported v52 selected source {selected!r} for {scene}")
    job["selected_source"] = "v52_fallback"
    job["underlying_source"] = selected
    return job


def write_shell(path: Path, plan: dict[str, Any]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Generated: {plan['date']}",
        f"# Execute from: {ROOT}",
        "",
        f"cd {shlex.quote(str(ROOT))}",
        "",
    ]
    log_root = Path(plan["output_root"]) / "logs"
    lines.append(f"mkdir -p {shlex.quote(str(log_root))}")
    lines.append("")
    for item in plan["jobs"]:
        lines.append(
            f"# scene={item['scene']} selected={item['selected_source']} underlying={item['underlying_source']}"
        )
        for index, command in enumerate(item["commands"]):
            log_path = log_root / f"{item['scene']}_{item['underlying_source']}_{index}_{command['name']}.log"
            lines.append(quote_cmd(command["argv"], command.get("env", {}), str(log_path)))
        lines.append("")
    lines.append("# Refresh fixed v56 guard summary from this source-rerun root plus previous v55d roots.")
    lines.append(quote_cmd(plan["refresh_command"]))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def write_markdown(path: Path, plan: dict[str, Any]) -> None:
    status_line = (
        "Status: `EXECUTED_SOURCE_RERUN_COMMANDS`."
        if plan["executed"]
        else "Status: `COMMAND_PLAN_ONLY`."
    )
    lines = [
        "# v56 Face-Alpha Guard Source-Rerun Plan",
        "",
        f"Date: `{plan['date']}`",
        "",
        status_line,
        "",
        "This plan promotes v56 from pure artifact replay toward source-config validation.",
        "It can either rerun the effective selected rows (`selected`) or probe v55d face-alpha",
        "candidates on additional scenes (`v55d_candidates`) before replaying the fixed guard.",
        "",
        "## Scope",
        "",
        f"- mode: `{plan['mode']}`",
        f"- output root: `{plan['output_root']}`",
        f"- selection uses held-out metrics: `{plan['selection_uses_heldout_metrics']}`",
        "- v55d min target changed fraction: "
        f"`{plan.get('v55d_command_policy', {}).get('min_target_changed_fraction', 'n/a')}`",
        f"- W&B wrapper enabled: `{plan['wandb']['enabled']}`",
        "",
        "## Jobs",
        "",
        "| scene | selected | underlying | gpu | output model | missing inputs |",
        "|---|---|---|---:|---|---|",
    ]
    for item in plan["jobs"]:
        missing = ", ".join(item.get("required_missing", [])) or "none"
        lines.append(
            f"| {item['scene']} | `{item['selected_source']}` | `{item['underlying_source']}` | "
            f"`{item['gpu']}` | `{item['output_model']}` | `{missing}` |"
        )
    lines.extend(["", "## Commands", ""])
    for item in plan["jobs"]:
        lines.extend(
            [
                f"### {item['scene']} -> {item['selected_source']} / {item['underlying_source']}",
                "",
                "```bash",
            ]
        )
        for command in item["commands"]:
            lines.append(quote_cmd(command["argv"], command.get("env", {})))
        lines.extend(["```", ""])
    lines.extend(
        [
            "## Refresh Command",
            "",
            "```bash",
            quote_cmd(plan["refresh_command"]),
            "```",
            "",
        ]
    )
    if plan.get("execution_results"):
        lines.extend(["## Execution Results", "", "| scene | command | returncode | elapsed sec | log |", "|---|---|---:|---:|---|"])
        for result in plan["execution_results"]:
            lines.append(
                f"| {result['scene']} | `{result['name']}` | {result['returncode']} | "
                f"{result['elapsed_sec']:.2f} | `{result['log_path']}` |"
            )
        lines.append("")
    if plan.get("refresh_result"):
        result = plan["refresh_result"]
        lines.extend(
            [
                "## Refresh Result",
                "",
                f"- returncode: `{result['returncode']}`",
                f"- elapsed sec: `{result['elapsed_sec']:.2f}`",
                f"- output JSON: `{plan['refresh_output_json']}`",
                f"- output MD: `{plan['refresh_output_md']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Honest Boundary",
            "",
            "A v55d candidate probe is not automatically a promoted v56 endpoint. Promotion still",
            "requires the fixed train/policy-val guard to pass and the held-out metrics to be",
            "reported without being used for branch selection.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_one(command: dict[str, Any], log_path: Path) -> tuple[int, float]:
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in command.get("env", {}).items()})
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command:\n" + quote_cmd(command["argv"], command.get("env", {})) + "\n\n")
        log.flush()
        proc = subprocess.run(command["argv"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode), float(time.time() - start)


def execute_plan(plan: dict[str, Any], wandb_run: Any | None = None) -> list[dict[str, Any]]:
    log_root = Path(plan["output_root"]) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for item in plan["jobs"]:
        for index, command in enumerate(item["commands"]):
            log_path = log_root / f"{item['scene']}_{item['underlying_source']}_{index}_{command['name']}.log"
            returncode, elapsed = run_one(command, log_path)
            result = {
                "scene": item["scene"],
                "selected_source": item["selected_source"],
                "underlying_source": item["underlying_source"],
                "name": command["name"],
                "returncode": returncode,
                "elapsed_sec": elapsed,
                "log_path": str(log_path),
            }
            results.append(result)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "command/elapsed_sec": elapsed,
                        "command/returncode": returncode,
                        "command/scene_index": len(results),
                    }
                )
                try:
                    wandb_run.save(str(log_path), policy="now")
                except Exception as exc:
                    wandb_run.summary[f"wandb_save_warning_{len(results)}"] = str(exc)
            if returncode != 0:
                raise RuntimeError(json.dumps(result, indent=2))
    return results


def build_refresh_command(args: argparse.Namespace, output_root: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/car_model/summarize_v56_face_alpha_guard_policy.py",
        "--v52_summary",
        str(args.v52_summary),
        "--output_json",
        str(args.refresh_output_json),
        "--output_md",
        str(args.refresh_output_md),
        "--v55d_root",
        str(output_root),
    ]
    for root in args.previous_v55d_root:
        command.extend(["--v55d_root", str(root)])
    return command


def run_refresh(command: list[str]) -> dict[str, Any]:
    start = time.time()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    result = {
        "argv": command,
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.time() - start),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan or execute v56 face-alpha guard source reruns.")
    parser.add_argument("--v56_summary", type=Path, default=DEFAULT_V56_SUMMARY)
    parser.add_argument("--v52_summary", type=Path, default=DEFAULT_V52_SUMMARY)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--plan_json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--plan_md", type=Path, default=DEFAULT_PLAN_MD)
    parser.add_argument("--plan_sh", type=Path, default=DEFAULT_PLAN_SH)
    parser.add_argument("--refresh_output_json", type=Path, default=DEFAULT_REFRESH_JSON)
    parser.add_argument("--refresh_output_md", type=Path, default=DEFAULT_REFRESH_MD)
    parser.add_argument("--previous_v55d_root", type=Path, action="append", default=[DEFAULT_V55D_ROOT])
    parser.add_argument("--mode", choices=("selected", "v55d_candidates", "both"), default="selected")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, assigned round-robin.")
    parser.add_argument("--scene", action="append", default=[], help="Optional scene subset.")
    parser.add_argument(
        "--v48_roots",
        default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623",
    )
    parser.add_argument(
        "--v55d_min_target_changed_fraction",
        type=float,
        default=0.001,
        help="Forwarded to v55d candidate runs; use 0.0 for the stricter paper-policy ablation.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--refresh_after_execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--wandb_group", default="v56_face_alpha_guard_source_rerun")
    parser.add_argument("--wandb_mode", default="online", choices=("online", "offline", "disabled"))
    args = parser.parse_args()

    v56_payload = read_json(args.v56_summary)
    v52_payload = read_json(args.v52_summary)
    if v56_payload.get("selection_uses_heldout_metrics") is not False:
        raise RuntimeError("v56 selection must not use held-out metrics")
    v52_rows = {str(row["scene"]): row for row in v52_payload["rows"]}
    selected_subset = set(args.scene or [])
    rows = [row for row in v56_payload["rows"] if not selected_subset or row["scene"] in selected_subset]
    if selected_subset and len(rows) != len(selected_subset):
        found = {str(row["scene"]) for row in rows}
        raise RuntimeError(f"missing requested scenes: {sorted(selected_subset - found)}")
    gpus = [item.strip() for item in str(args.gpus).split(",") if item.strip()]
    if not gpus:
        raise RuntimeError("--gpus must contain at least one GPU id")
    output_root = args.output_root
    if output_root is None:
        output_root = Path(f"/dev/shm/peilincai_spcarnet_v56_source_rerun_{time.strftime('%Y%m%d_%H%M%S')}")

    jobs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        scene = str(row["scene"])
        gpu = gpus[index % len(gpus)]
        if args.mode in {"selected", "both"}:
            if str(row["selected_source"]) == "v55d_face_alpha":
                job = build_v55d_candidate_command(
                    scene=scene,
                    output_root=output_root,
                    python=sys.executable,
                    gpu=gpu,
                    force=bool(args.force),
                    v48_roots=str(args.v48_roots),
                    min_target_changed_fraction=float(args.v55d_min_target_changed_fraction),
                    wandb_project=str(args.wandb_project),
                    wandb_group=str(args.wandb_group),
                    wandb_run_prefix=str(args.wandb_run_name or "v56_selected_v55d"),
                    wandb_mode=str(args.wandb_mode),
                )
            else:
                job = build_v52_fallback_job(
                    v52_row=v52_rows[scene],
                    output_root=output_root,
                    python=sys.executable,
                    gpu=gpu,
                    force=bool(args.force),
                    v48_roots=str(args.v48_roots),
                )
            key = (scene, str(job["underlying_source"]))
            if key not in seen:
                jobs.append(job)
                seen.add(key)
        if args.mode in {"v55d_candidates", "both"}:
            job = build_v55d_candidate_command(
                scene=scene,
                output_root=output_root,
                python=sys.executable,
                gpu=gpu,
                force=bool(args.force),
                v48_roots=str(args.v48_roots),
                min_target_changed_fraction=float(args.v55d_min_target_changed_fraction),
                wandb_project=str(args.wandb_project),
                wandb_group=str(args.wandb_group),
                wandb_run_prefix=str(args.wandb_run_name or "v56_candidate_v55d"),
                wandb_mode=str(args.wandb_mode),
            )
            key = (scene, str(job["underlying_source"]))
            if key not in seen:
                jobs.append(job)
                seen.add(key)

    refresh_command = build_refresh_command(args, output_root)
    plan = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "COMMAND_PLAN_ONLY",
        "executed": bool(args.execute),
        "mode": str(args.mode),
        "selection_uses_heldout_metrics": False,
        "v56_summary": str(args.v56_summary),
        "v52_summary": str(args.v52_summary),
        "output_root": str(output_root),
        "plan_json": str(args.plan_json),
        "plan_md": str(args.plan_md),
        "plan_sh": str(args.plan_sh),
        "refresh_output_json": str(args.refresh_output_json),
        "refresh_output_md": str(args.refresh_output_md),
        "previous_v55d_roots": [str(root) for root in args.previous_v55d_root],
        "gpus": gpus,
        "wandb": {
            "enabled": bool(args.execute and args.wandb_project and args.wandb_mode != "disabled"),
            "project": str(args.wandb_project),
            "run_name": str(args.wandb_run_name),
            "group": str(args.wandb_group),
            "mode": str(args.wandb_mode),
        },
        "v55d_command_policy": {
            "min_target_changed_fraction": float(args.v55d_min_target_changed_fraction),
        },
        "jobs": jobs,
        "refresh_command": refresh_command,
    }
    if args.execute:
        plan["status"] = "EXECUTED_SOURCE_RERUN_COMMANDS"
        wandb_run = None
        if args.wandb_project and args.wandb_mode != "disabled":
            import wandb

            wandb_run = wandb.init(
                project=str(args.wandb_project),
                name=str(args.wandb_run_name or f"v56_source_rerun_{time.strftime('%Y%m%d_%H%M%S')}"),
                group=str(args.wandb_group),
                mode=str(args.wandb_mode),
                config={
                    "mode": str(args.mode),
                    "v56_summary": str(args.v56_summary),
                    "v52_summary": str(args.v52_summary),
                    "output_root": str(output_root),
                    "gpus": gpus,
                    "scene_count": len(jobs),
                    "selection_uses_heldout_metrics": False,
                    "v55d_min_target_changed_fraction": float(args.v55d_min_target_changed_fraction),
                },
            )
        try:
            plan["execution_results"] = execute_plan(plan, wandb_run=wandb_run)
            if bool(args.refresh_after_execute):
                plan["refresh_result"] = run_refresh(refresh_command)
            if wandb_run is not None:
                wandb_run.summary["status"] = plan["status"]
                wandb_run.summary["completed_commands"] = len(plan["execution_results"])
                if plan.get("refresh_result"):
                    wandb_run.summary["refresh_returncode"] = plan["refresh_result"]["returncode"]
        finally:
            if wandb_run is not None:
                wandb_run.finish()
    args.plan_json.parent.mkdir(parents=True, exist_ok=True)
    args.plan_json.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_shell(args.plan_sh, plan)
    write_markdown(args.plan_md, plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "mode": plan["mode"],
                "jobs": len(jobs),
                "output_root": str(output_root),
                "plan_json": str(args.plan_json),
                "plan_md": str(args.plan_md),
                "plan_sh": str(args.plan_sh),
                "executed": bool(args.execute),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
