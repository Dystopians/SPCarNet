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
DEFAULT_SUMMARY = Path(
    "outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/"
    "v52_capacity_aware_v48_v51_full9_summary.json"
)
DEFAULT_PLAN_ROOT = Path("outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware")
DEFAULT_PLAN_JSON = DEFAULT_PLAN_ROOT / "v52_capacity_aware_source_rerun_plan.json"
DEFAULT_PLAN_MD = DEFAULT_PLAN_ROOT / "v52_capacity_aware_source_rerun_plan.md"
DEFAULT_PLAN_SH = DEFAULT_PLAN_ROOT / "v52_capacity_aware_source_rerun_plan.sh"
V48_TAG = "v48_autosupport_autocap_guarded_v42calib_region_texture_adapter"
V51_TAG = "v51_fast_support_ladder_tex32_nearest_l1pos05_region_texture_adapter"


APPLY_SETTING_ARGS = (
    "source_model",
    "fit_evidence_dir",
    "target_evidence_dir",
    "region_carrier_json",
    "target_split",
    "base_method_name",
    "residual_rgb_key",
    "residual_l1_key",
    "texture_size",
    "texture_size_candidates",
    "max_carriers",
    "max_faces_per_carrier",
    "max_faces",
    "support_expansion_mode",
    "support_expansion_max_extra_faces",
    "support_expansion_max_extra_faces_candidates",
    "support_expansion_min_face_samples",
    "support_expansion_min_mean_l1",
    "policy_val_stride",
    "alpha_grid",
    "min_l1",
    "min_alpha",
    "min_atlas_bin_count",
    "min_atlas_face_samples",
    "max_atlas_bin_rgb_variance",
    "min_atlas_bin_sign_consistency",
    "atlas_confidence_mode",
    "atlas_confidence_count_scale",
    "atlas_confidence_empty_bin",
    "atlas_confidence_variance_scale",
    "atlas_confidence_sign_power",
    "atlas_confidence_face_sample_scale",
    "min_atlas_confidence",
    "atlas_lowpass_passes",
    "atlas_lowpass_neighbor_min_count",
    "atlas_empty_bin_fill_mode",
    "atlas_nearest_fill_max_steps",
    "atlas_nearest_fill_decay",
    "max_samples_per_view",
    "max_abs_delta_rgb",
    "min_policy_val_samples",
    "min_policy_val_relative_gain",
    "min_policy_val_positive_view_fraction",
    "min_policy_val_cvar20_relative_gain",
    "min_policy_val_min_view_relative_gain",
    "policy_val_ssim_max_size",
    "min_policy_val_ssim_mean_gain",
    "min_policy_val_ssim_positive_view_fraction",
    "min_policy_val_ssim_min_view_gain",
    "policy_val_l1_max_size",
    "min_policy_val_l1_mean_gain",
    "min_policy_val_l1_positive_view_fraction",
    "min_policy_val_l1_min_view_gain",
    "min_policy_val_l1_cvar20_view_gain",
    "min_target_changed_fraction",
    "noop_fallback_source",
)
APPLY_BOOLEAN_FLAGS = (
    "select_alpha_by_risk_gate",
    "enable_policy_val_image_ssim_gate",
    "enable_policy_val_image_l1_gate",
    "write_noop_on_reject",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quote_cmd(argv: list[str], env: dict[str, str] | None = None, log_path: str | None = None) -> str:
    prefix = ""
    if env:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items()) + " "
    command = prefix + " ".join(shlex.quote(str(part)) for part in argv)
    if log_path:
        return f"{command} > {shlex.quote(log_path)} 2>&1"
    return command


def path_exists(value: Any) -> bool:
    if value in (None, ""):
        return False
    return Path(str(value)).exists()


def append_cli_value(argv: list[str], key: str, value: Any) -> None:
    text = str(value)
    if text.startswith("-"):
        argv.append(f"--{key}={text}")
    else:
        argv.extend([f"--{key}", text])


def build_v48_commands(
    scene: str,
    source_dir: Path,
    output_root: Path,
    python: str,
    gpu: str,
    force: bool,
) -> dict[str, Any]:
    audit_path = source_dir / "surface_residual_region_texture_adapter_audit.json"
    audit = read_json(audit_path)
    settings = dict(audit.get("settings", {}) or {})
    output_model = output_root / f"{scene}_{V48_TAG}"
    method_name = f"ours_26000_{scene}_{V48_TAG}"
    apply_cmd = [python, "scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py"]
    required_missing: list[str] = []
    for key in APPLY_SETTING_ARGS:
        if key not in settings or settings[key] in (None, ""):
            continue
        append_cli_value(apply_cmd, key, settings[key])
        if key in {"source_model", "fit_evidence_dir", "target_evidence_dir", "region_carrier_json"} and not path_exists(settings[key]):
            required_missing.append(f"{key}:{settings[key]}")
    apply_cmd.extend(["--output_model", str(output_model), "--method_name", method_name])
    for key in APPLY_BOOLEAN_FLAGS:
        if bool(settings.get(key, False)):
            apply_cmd.append(f"--{key}")
    if not bool(settings.get("write_noop_on_reject", False)):
        apply_cmd.append("--write_noop_on_reject")
    if "fill_empty_with_face_mean" in settings and not bool(settings["fill_empty_with_face_mean"]):
        apply_cmd.append("--no-fill_empty_with_face_mean")
    if force:
        apply_cmd.append("--force")
    metrics_cmd = [python, "metrics.py", "-m", str(output_model)]
    env = {"CUDA_VISIBLE_DEVICES": gpu, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    return {
        "scene": scene,
        "selected_source": "v48",
        "audit_path": str(audit_path),
        "output_model": str(output_model),
        "method_name": method_name,
        "gpu": gpu,
        "commands": [
            {"name": "apply", "argv": apply_cmd, "env": env},
            {"name": "metrics", "argv": metrics_cmd, "env": env},
        ],
        "required_missing": required_missing,
    }


def build_v51_commands(
    scene: str,
    output_root: Path,
    python: str,
    gpu: str,
    force: bool,
    v48_roots: str,
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
        V51_TAG,
        "--v48_roots",
        v48_roots,
        "--min_policy_val_l1_positive_view_fraction",
        "0.5",
        "--min_target_changed_fraction",
        "0.0",
        "--support_expansion_max_extra_faces_candidates",
        "2048,4096",
        "--texture_size_candidates",
        "32",
        "--atlas_empty_bin_fill_mode",
        "nearest_observed",
    ]
    if force:
        cmd.append("--force")
    env = {"CUDA_VISIBLE_DEVICES": gpu, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    return {
        "scene": scene,
        "selected_source": "v51",
        "output_model": str(output_root / f"{scene}_{V51_TAG}"),
        "method_name": f"ours_26000_{scene}_{V51_TAG}",
        "gpu": gpu,
        "commands": [{"name": "runner_apply_metrics", "argv": cmd, "env": env}],
        "required_missing": [],
    }


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
        lines.append(f"# scene={item['scene']} selected={item['selected_source']}")
        for index, command in enumerate(item["commands"]):
            log_path = log_root / f"{item['scene']}_{item['selected_source']}_{index}_{command['name']}.log"
            lines.append(quote_cmd(command["argv"], command.get("env", {}), str(log_path)))
        lines.append("")
    lines.append("# Refresh v52 selected artifacts after rerun outputs have been reviewed or promoted.")
    lines.append(quote_cmd(plan["refresh_command"]))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def write_markdown(path: Path, plan: dict[str, Any]) -> None:
    lines = [
        "# v52 Capacity-Aware Source Rerun Plan",
        "",
        f"Date: `{plan['date']}`",
        "",
        "Status: `COMMAND_PLAN_ONLY`." if not plan["executed"] else "Status: `EXECUTED_SEQUENTIAL_SOURCE_RERUN`.",
        "",
        "This file records the source/audit-config commands needed to regenerate the selected",
        "v52 rows. v48 rows are rebuilt from their saved atlas audit settings; v51 rows reuse",
        "`run_l1risk_fairnoop_scene.py` with the fixed support-ladder policy.",
        "",
        "## Important Boundary",
        "",
        "The default mode only writes a command plan. It does not consume GPU and does not prove",
        "fresh metrics. Use `--execute --gpus <ids> --wandb_project <project>` only when",
        "ready to launch a medium/full rerun. The wrapper logs command status and logs to W&B;",
        "the underlying apply/metrics scripts still do not emit fine-grained W&B metrics.",
        "",
        "## Plan Files",
        "",
        f"- JSON: `{plan['plan_json']}`",
        f"- shell: `{plan['plan_sh']}`",
        f"- output root: `{plan['output_root']}`",
        "",
        "## Jobs",
        "",
        "| scene | selected | gpu | output model | missing inputs |",
        "|---|---|---:|---|---|",
    ]
    for item in plan["jobs"]:
        missing = ", ".join(item.get("required_missing", [])) or "none"
        lines.append(
            f"| {item['scene']} | `{item['selected_source']}` | `{item['gpu']}` | "
            f"`{item['output_model']}` | `{missing}` |"
        )
    lines.extend(["", "## Commands", ""])
    for item in plan["jobs"]:
        lines.extend([f"### {item['scene']} -> {item['selected_source']}", "", "```bash"])
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
        lines.extend(["## Execution Results", "", "| scene | command | returncode | elapsed sec |", "|---|---|---:|---:|"])
        for result in plan["execution_results"]:
            lines.append(
                f"| {result['scene']} | `{result['name']}` | {result['returncode']} | {result['elapsed_sec']:.2f} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def execute_plan(plan: dict[str, Any], wandb_run: Any | None = None) -> list[dict[str, Any]]:
    log_root = Path(plan["output_root"]) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for item in plan["jobs"]:
        for index, command in enumerate(item["commands"]):
            log_path = log_root / f"{item['scene']}_{item['selected_source']}_{index}_{command['name']}.log"
            env = os.environ.copy()
            env.update({str(k): str(v) for k, v in command.get("env", {}).items()})
            start = time.time()
            with log_path.open("w", encoding="utf-8") as log:
                log.write("command:\n" + quote_cmd(command["argv"], command.get("env", {})) + "\n\n")
                log.flush()
                proc = subprocess.run(command["argv"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.time() - start
            result = {
                "scene": item["scene"],
                "selected_source": item["selected_source"],
                "name": command["name"],
                "returncode": proc.returncode,
                "elapsed_sec": elapsed,
                "log_path": str(log_path),
            }
            results.append(result)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "command/elapsed_sec": elapsed,
                        "command/returncode": proc.returncode,
                        "command/scene_index": len(results),
                    }
                )
                try:
                    wandb_run.save(str(log_path), policy="now")
                except Exception as exc:
                    wandb_run.summary[f"wandb_save_warning_{len(results)}"] = str(exc)
            if proc.returncode != 0:
                raise RuntimeError(json.dumps(result, indent=2))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan or execute v52 selected-row source reruns.")
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--plan_json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--plan_md", type=Path, default=DEFAULT_PLAN_MD)
    parser.add_argument("--plan_sh", type=Path, default=DEFAULT_PLAN_SH)
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, assigned round-robin.")
    parser.add_argument("--scene", action="append", default=[], help="Optional scene subset.")
    parser.add_argument(
        "--v48_roots",
        default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623",
    )
    parser.add_argument("--execute", action="store_true", help="Actually run planned commands sequentially.")
    parser.add_argument("--force", action="store_true", help="Forward --force to generated scene commands.")
    parser.add_argument("--wandb_project", default="", help="Optional W&B project for --execute runs.")
    parser.add_argument("--wandb_run_name", default="", help="Optional W&B run name for --execute runs.")
    parser.add_argument("--wandb_mode", default="online", choices=("online", "offline", "disabled"))
    args = parser.parse_args()

    payload = read_json(args.summary_json)
    if payload.get("selection_uses_heldout_metrics") is not False:
        raise RuntimeError("v52 selection must not use held-out metrics")
    selected_subset = set(args.scene or [])
    rows = [row for row in payload["rows"] if not selected_subset or row["scene"] in selected_subset]
    if selected_subset and len(rows) != len(selected_subset):
        found = {row["scene"] for row in rows}
        raise RuntimeError(f"missing requested scenes: {sorted(selected_subset - found)}")
    gpus = [item.strip() for item in str(args.gpus).split(",") if item.strip()]
    if not gpus:
        raise RuntimeError("--gpus must contain at least one GPU id")
    output_root = args.output_root
    if output_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_root = Path(f"/dev/shm/peilincai_spcarnet_v52_source_rerun_{stamp}")
    jobs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        scene = str(row["scene"])
        selected = str(row["selected_source"])
        gpu = gpus[index % len(gpus)]
        if selected == "v48":
            jobs.append(
                build_v48_commands(
                    scene=scene,
                    source_dir=Path(str(row["selected_source_dir"])),
                    output_root=output_root,
                    python=sys.executable,
                    gpu=gpu,
                    force=bool(args.force),
                )
            )
        elif selected == "v51":
            jobs.append(
                build_v51_commands(
                    scene=scene,
                    output_root=output_root,
                    python=sys.executable,
                    gpu=gpu,
                    force=bool(args.force),
                    v48_roots=str(args.v48_roots),
                )
            )
        else:
            raise RuntimeError(f"unsupported selected source {selected!r} for {scene}")
    refresh_command = [
        sys.executable,
        "scripts/car_model/run_v52_capacity_aware_pipeline.py",
    ]
    plan = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "COMMAND_PLAN_ONLY",
        "executed": bool(args.execute),
        "summary_json": str(args.summary_json),
        "selection_uses_heldout_metrics": False,
        "output_root": str(output_root),
        "plan_json": str(args.plan_json),
        "plan_md": str(args.plan_md),
        "plan_sh": str(args.plan_sh),
        "gpus": gpus,
        "wandb": {
            "enabled": bool(args.execute and args.wandb_project and args.wandb_mode != "disabled"),
            "project": str(args.wandb_project),
            "run_name": str(args.wandb_run_name),
            "mode": str(args.wandb_mode),
        },
        "jobs": jobs,
        "refresh_command": refresh_command,
    }
    if args.execute:
        plan["status"] = "EXECUTED_SEQUENTIAL_SOURCE_RERUN"
        wandb_run = None
        if args.wandb_project and args.wandb_mode != "disabled":
            import wandb

            wandb_run = wandb.init(
                project=str(args.wandb_project),
                name=str(args.wandb_run_name or f"v52_source_rerun_{time.strftime('%Y%m%d_%H%M%S')}"),
                mode=str(args.wandb_mode),
                config={
                    "summary_json": str(args.summary_json),
                    "output_root": str(output_root),
                    "gpus": gpus,
                    "scene_count": len(jobs),
                    "selection_uses_heldout_metrics": False,
                },
            )
        try:
            plan["execution_results"] = execute_plan(plan, wandb_run=wandb_run)
            if wandb_run is not None:
                wandb_run.summary["status"] = plan["status"]
                wandb_run.summary["completed_commands"] = len(plan["execution_results"])
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
