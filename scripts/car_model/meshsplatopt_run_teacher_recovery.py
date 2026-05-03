#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.teacher_recovery import run_teacher_recovery_contract


def load_model_cfg(model_path: Path) -> argparse.Namespace:
    cfg_path = model_path / "cfg_args"
    if not cfg_path.exists():
        return argparse.Namespace()
    return eval(cfg_path.read_text(encoding="utf-8"), {"Namespace": argparse.Namespace})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--edit_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--run_real_tiny", action="store_true")
    parser.add_argument("--load_iteration", type=int, default=200)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wandb_project", default="spcarnet_meshprior")
    parser.add_argument("--wandb_group", default="meshsplatopt_teacher_recovery")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--train_densify_until_iter", type=int, default=None)
    parser.add_argument("--train_densify_from_iter", type=int, default=None)
    parser.add_argument("--train_densification_interval", type=int, default=None)
    parser.add_argument("--train_skip_restricted_delaunay", action="store_true")
    parser.add_argument(
        "--train_extra_args",
        default="",
        help="Optional shell-style extra arguments appended to train.py for diagnostic recovery runs.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"\n[exit_code] {proc.returncode}\n")
    return int(proc.returncode)


def copy_recovery_model(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    ignore = shutil.ignore_patterns("train", "test", "results.json", "per_view.json", "geometry_eval_colmap")
    shutil.copytree(source, target, ignore=ignore)


def run_real_tiny_recovery(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    recovery_model = output_dir / "recovery_model"
    source_model = Path(args.model_path)
    source_cfg = load_model_cfg(source_model)
    copy_recovery_model(source_model, recovery_model)
    env = os.environ.copy()
    env["WANDB_PROJECT"] = args.wandb_project
    env["WANDB_MODE"] = "online"
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    logs = output_dir / "logs"
    nvidia_code = run_command(["nvidia-smi"], cwd=REPO_ROOT, env=env, log_path=logs / "nvidia_smi.log")
    train_until = int(args.load_iteration) + int(args.iterations)
    wandb_name = args.wandb_name or f"meshsplatopt_recovery_{Path(args.output_dir).name}"
    train_cmd = [
        args.python,
        "train.py",
        "-s",
        str(getattr(source_cfg, "source_path", "")),
        "-m",
        str(recovery_model),
        "-i",
        str(getattr(source_cfg, "images", "images")),
        "-r",
        str(getattr(source_cfg, "resolution", -1)),
        "--load_iteration",
        str(args.load_iteration),
        "--iterations",
        str(train_until),
        "--test_iterations",
        str(train_until),
        "--save_iterations",
        str(train_until),
        "--enable_wandb",
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        args.wandb_group,
        "--wandb_name",
        wandb_name,
        "--wandb_scalar_log_interval",
        "10",
        "--wandb_disable_fixed_views",
        "--quiet",
    ]
    if bool(getattr(source_cfg, "eval", False)):
        train_cmd.append("--eval")
    if bool(getattr(source_cfg, "white_background", False)):
        train_cmd.append("--white_background")
    if args.train_densify_until_iter is not None:
        train_cmd.extend(["--densify_until_iter", str(int(args.train_densify_until_iter))])
    if args.train_densify_from_iter is not None:
        train_cmd.extend(["--densify_from_iter", str(int(args.train_densify_from_iter))])
    if args.train_densification_interval is not None:
        train_cmd.extend(["--densification_interval", str(int(args.train_densification_interval))])
    if bool(args.train_skip_restricted_delaunay):
        train_cmd.append("--skip_restricted_delaunay")
    if str(args.train_extra_args).strip():
        train_cmd.extend(shlex.split(str(args.train_extra_args)))
    train_code = run_command(train_cmd, cwd=REPO_ROOT, env=env, log_path=logs / "train_recovery.log")
    render_code = metrics_code = None
    if train_code == 0:
        render_code = run_command(
            [
                args.python,
                "render.py",
                "-m",
                str(recovery_model),
                "--iteration",
                str(train_until),
                "--skip_train",
                "--quiet",
            ],
            cwd=REPO_ROOT,
            env=env,
            log_path=logs / "render_recovery.log",
        )
        if render_code == 0:
            metrics_code = run_command(
                [args.python, "metrics.py", "-m", str(recovery_model)],
                cwd=REPO_ROOT,
                env=env,
                log_path=logs / "metrics_recovery.log",
            )
    result = {
        "real_recovery_run": train_code == 0,
        "recovery_model": str(recovery_model),
        "load_iteration": int(args.load_iteration),
        "train_until_iteration": train_until,
        "gpu": args.gpu,
        "wandb_project": args.wandb_project,
        "wandb_group": args.wandb_group,
        "wandb_name": wandb_name,
        "train_overrides": {
            "densify_until_iter": args.train_densify_until_iter,
            "densify_from_iter": args.train_densify_from_iter,
            "densification_interval": args.train_densification_interval,
            "skip_restricted_delaunay": bool(args.train_skip_restricted_delaunay),
            "extra_args": str(args.train_extra_args),
        },
        "exit_codes": {
            "nvidia_smi": nvidia_code,
            "train": train_code,
            "render": render_code,
            "metrics": metrics_code,
        },
        "logs": {
            "nvidia_smi": str(logs / "nvidia_smi.log"),
            "train": str(logs / "train_recovery.log"),
            "render": str(logs / "render_recovery.log"),
            "metrics": str(logs / "metrics_recovery.log"),
        },
    }
    (output_dir / "real_tiny_recovery_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if train_code != 0:
        raise SystemExit(f"real tiny teacher recovery failed; see {logs / 'train_recovery.log'}")
    return result


def main() -> None:
    args = parse_args()
    plan = run_teacher_recovery_contract(
        model_path=args.model_path,
        edit_json=args.edit_json,
        output_dir=args.output_dir,
        iterations=args.iterations,
    )
    report = {"contract": plan.to_dict(), "real_tiny_recovery": None}
    if args.run_real_tiny:
        report["real_tiny_recovery"] = run_real_tiny_recovery(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "teacher_recovery_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
