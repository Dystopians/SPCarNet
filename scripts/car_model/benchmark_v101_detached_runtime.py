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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _run_cmd(cmd: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        elapsed = time.perf_counter() - start
        handle.write(f"\n[exit_code] {proc.returncode}\n[wall_sec] {elapsed:.6f}\n")
    return int(proc.returncode), float(elapsed)


def _count_pngs(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.is_dir() else 0


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model_path)
    if not model.is_dir():
        raise FileNotFoundError(model)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    rows = []
    baseline_method = f"ours_{int(args.iteration)}"
    baseline_log = report_root / f"{args.scene}_runtime_baseline.log"
    if baseline_log.exists():
        baseline_log.unlink()
    baseline_rc, baseline_wall = _run_cmd(
        [
            sys.executable,
            "render.py",
            "-m",
            str(model),
            "--iteration",
            str(args.iteration),
            "--skip_train",
            "--quiet",
        ],
        baseline_log,
        env,
    )
    baseline_count = _count_pngs(model / "test" / baseline_method / "renders")
    rows.append(
        {
            "name": "standard_render",
            "method": baseline_method,
            "returncode": baseline_rc,
            "wall_sec": baseline_wall,
            "views": baseline_count,
            "sec_per_view": baseline_wall / baseline_count if baseline_count else None,
            "log_path": str(baseline_log),
        }
    )

    endpoint_log = report_root / f"{args.scene}_runtime_v101_bank.log"
    if endpoint_log.exists():
        endpoint_log.unlink()
    endpoint_bank = (
        model
        / "point_cloud"
        / f"iteration_{int(args.iteration)}"
        / "render_residual_endpoint"
        / args.endpoint_method
        / "v101_evidence_bank.pt"
    )
    endpoint_rc, endpoint_wall = _run_cmd(
        [
            sys.executable,
            "render.py",
            "-m",
            str(model),
            "--iteration",
            str(args.iteration),
            "--skip_train",
            "--checkpoint_endpoint_method",
            args.endpoint_method,
            "--checkpoint_endpoint_output_method",
            args.endpoint_method_name,
            "--checkpoint_endpoint_bank_path",
            str(endpoint_bank),
            "--checkpoint_endpoint_require_bank",
            "--quiet",
        ],
        endpoint_log,
        env,
    )
    endpoint_count = _count_pngs(model / "test" / args.endpoint_method_name / "renders")
    endpoint_report = _read_json(model / "test" / args.endpoint_method_name / "render_py_endpoint_report.json")
    endpoint_support_source = str(endpoint_report.get("support_source", "") or "")
    endpoint_used_required_bank = endpoint_support_source.startswith("v101_evidence_bank:")
    rows.append(
        {
            "name": "v101_require_bank_render",
            "method": args.endpoint_method_name,
            "returncode": endpoint_rc,
            "wall_sec": endpoint_wall,
            "views": endpoint_count,
            "sec_per_view": endpoint_wall / endpoint_count if endpoint_count else None,
            "internal_elapsed_sec": endpoint_report.get("elapsed_sec"),
            "internal_sec_per_view": (
                float(endpoint_report["elapsed_sec"]) / int(endpoint_report["target_frames"])
                if endpoint_report.get("elapsed_sec") is not None and endpoint_report.get("target_frames")
                else None
            ),
            "mean_abs_delta": endpoint_report.get("mean_abs_delta"),
            "mean_changed_fraction": endpoint_report.get("mean_changed_fraction"),
            "support_source": endpoint_support_source,
            "used_required_bank": endpoint_used_required_bank,
            "bank_path": str(endpoint_bank),
            "log_path": str(endpoint_log),
        }
    )

    baseline_spv = rows[0].get("sec_per_view")
    endpoint_spv = rows[1].get("sec_per_view")
    payload = {
        "schema_version": 1,
        "scene": args.scene,
        "model_path": str(model),
        "iteration": int(args.iteration),
        "rows": rows,
        "slowdown_wall": (
            float(endpoint_spv) / float(baseline_spv)
            if baseline_spv and endpoint_spv
            else None
        ),
        "passed": bool(
            baseline_rc == 0
            and endpoint_rc == 0
            and endpoint_used_required_bank
            and baseline_count > 0
            and endpoint_count == baseline_count
        ),
        "claim_boundary": (
            "This benchmark compares standard render.py against v101 require-bank render.py on the same detached package. "
            "It measures deployment overhead, not training time or quality."
        ),
    }
    json_path = report_root / f"{args.scene}_runtime_audit.json"
    md_path = report_root / f"{args.scene}_runtime_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# v101 Detached Runtime Audit: {args.scene}",
        "",
        f"- model: `{model}`",
        f"- passed: `{payload['passed']}`",
        f"- wall slowdown: `{payload['slowdown_wall']}`",
        "",
        "| path | views | wall sec | sec/view | return code |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['views']} | {row['wall_sec']:.6f} | "
            f"{float(row['sec_per_view'] or 0.0):.6f} | {row['returncode']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), "passed": payload["passed"]}, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark standard render.py against v101 require-bank render.py.")
    parser.add_argument("--scene", default="counter")
    parser.add_argument("--model_path", default="/dev/shm/peilincai_spcarnet_v101_detached_package_20260625/counter/detached_model")
    parser.add_argument("--report_root", default="outputs/carnet/meshsplatopt/ecsr_phase_v101_runtime_audit_20260625")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--endpoint_method", default="ours_26000_v100_checkpoint_attached_ela_endpoint")
    parser.add_argument("--endpoint_method_name", default="ours_26000_v101_detached_runtime_bank")
    parser.add_argument("--gpu", default="")
    return parser.parse_args()


def main() -> int:
    payload = benchmark(parse_args())
    return 0 if payload.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
