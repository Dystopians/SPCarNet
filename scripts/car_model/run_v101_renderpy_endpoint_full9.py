#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")
V100_METHOD = "ours_26000_v100_checkpoint_attached_ela_endpoint"
V101_METHOD = "ours_26000_v101_renderpy_endpoint_full9"


def _num(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_v100_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    rows = {}
    for row in payload.get("rows", []):
        scene = str(row.get("scene", ""))
        if scene:
            rows[scene] = row
    return rows


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "dPSNR": _num(candidate.get("PSNR")) - _num(baseline.get("PSNR")),
        "dSSIM": _num(candidate.get("SSIM")) - _num(baseline.get("SSIM")),
        "dLPIPS": _num(candidate.get("LPIPS")) - _num(baseline.get("LPIPS")),
    }


def _run_cmd(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.write(f"\n[exit_code] {proc.returncode}\n")
    return int(proc.returncode)


def _run_scene(args: argparse.Namespace, scene: str, gpu: int, v100_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    model = Path(args.v100_output_root) / scene / "recovery_model"
    if not model.is_dir():
        raise FileNotFoundError(model)
    log_path = Path(args.report_root) / "logs" / f"{scene}_v101_renderpy.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    method = str(args.method_name)
    started = time.time()
    bank_rc = 0
    if str(args.bank_root).strip():
        bank_path = Path(args.bank_root) / scene / "v101_evidence_bank.pt"
    else:
        bank_path = (
            model
            / "point_cloud"
            / f"iteration_{int(args.iteration)}"
            / "render_residual_endpoint"
            / str(args.endpoint_method)
            / "v101_evidence_bank.pt"
        )
    if args.build_banks and (args.force_rebuild_banks or not bank_path.is_file()):
        base_model = Path(args.base_model_root) / scene / "ratio_0200" / "compact_model"
        bank_cmd = [
            sys.executable,
            "scripts/car_model/build_v101_endpoint_evidence_bank.py",
            "--base_model_path",
            str(base_model),
            "--output_model_path",
            str(model),
            "--endpoint_method",
            str(args.endpoint_method),
            "--iteration",
            str(args.iteration),
            "--base_method_name",
            str(args.base_method_name),
            "--residual_dtype",
            str(args.bank_residual_dtype),
            "--depth_dtype",
            str(args.bank_depth_dtype),
            "--output_bank",
            str(bank_path),
        ]
        bank_rc = _run_cmd(bank_cmd, log_path, env)
    render_cmd = [
        sys.executable,
        "render.py",
        "-m",
        str(model),
        "--iteration",
        str(args.iteration),
        "--skip_train",
        "--checkpoint_endpoint_method",
        str(args.endpoint_method),
        "--checkpoint_endpoint_output_method",
        method,
        "--quiet",
    ]
    if args.build_banks or str(args.bank_root).strip() or args.require_bank:
        render_cmd.extend(["--checkpoint_endpoint_bank_path", str(bank_path)])
    if args.require_bank:
        render_cmd.append("--checkpoint_endpoint_require_bank")
    render_rc = 999
    if bank_rc == 0:
        render_rc = _run_cmd(render_cmd, log_path, env)
    eval_rc = 999
    if render_rc == 0:
        eval_cmd = [
            sys.executable,
            "scripts/car_model/evaluate_render_split_metrics.py",
            "-m",
            str(model),
            "--split",
            "test",
            "--methods",
            method,
            "--merge_model_results",
        ]
        eval_rc = _run_cmd(eval_cmd, log_path, env)
    results = _read_json(model / "results.json") if eval_rc == 0 else {}
    metrics = results.get(method, {}) if eval_rc == 0 else {}
    missing_metrics_error = ""
    if eval_rc == 0 and not metrics:
        eval_rc = 998
        missing_metrics_error = f"method {method} missing from results.json"
    report = _read_json(model / "test" / method / "render_py_endpoint_report.json")
    v100 = v100_rows[scene]
    clean = {"PSNR": v100["clean_PSNR"], "SSIM": v100["clean_SSIM"], "LPIPS": v100["clean_LPIPS"]}
    legacy = {"PSNR": v100["source_PSNR"], "SSIM": v100["source_SSIM"], "LPIPS": v100["source_LPIPS"]}
    phasej = {"PSNR": v100["phasej_PSNR"], "SSIM": v100["phasej_SSIM"], "LPIPS": v100["phasej_LPIPS"]}
    dc = _delta(metrics, clean)
    ds = _delta(metrics, legacy)
    dp = _delta(metrics, phasej)
    return {
        "scene": scene,
        "gpu": int(gpu),
        "bank_rc": int(bank_rc),
        "render_rc": int(render_rc),
        "eval_rc": int(eval_rc),
        "elapsed_sec": float(time.time() - started),
        "model": str(model),
        "method": method,
        "log_path": str(log_path),
        "PSNR": _num(metrics.get("PSNR")),
        "SSIM": _num(metrics.get("SSIM")),
        "LPIPS": _num(metrics.get("LPIPS")),
        "dPSNR_vs_clean": dc["dPSNR"],
        "dSSIM_vs_clean": dc["dSSIM"],
        "dLPIPS_vs_clean": dc["dLPIPS"],
        "dPSNR_vs_legacy_source": ds["dPSNR"],
        "dSSIM_vs_legacy_source": ds["dSSIM"],
        "dLPIPS_vs_legacy_source": ds["dLPIPS"],
        "dPSNR_vs_phasej": dp["dPSNR"],
        "dSSIM_vs_phasej": dp["dSSIM"],
        "dLPIPS_vs_phasej": dp["dLPIPS"],
        "mean_abs_delta": _num(report.get("mean_abs_delta"), 0.0),
        "mean_changed_fraction": _num(report.get("mean_changed_fraction"), 0.0),
        "support_source": str(report.get("support_source", "")),
        "evidence_bank": report.get("evidence_bank", {}),
        "bank_path": str(bank_path),
        "error": missing_metrics_error,
    }


def _write_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    preferred_fields = [
        "scene",
        "gpu",
        "render_rc",
        "eval_rc",
        "bank_rc",
        "elapsed_sec",
        "model",
        "method",
        "log_path",
        "PSNR",
        "SSIM",
        "LPIPS",
        "dPSNR_vs_clean",
        "dSSIM_vs_clean",
        "dLPIPS_vs_clean",
        "dPSNR_vs_legacy_source",
        "dSSIM_vs_legacy_source",
        "dLPIPS_vs_legacy_source",
        "dPSNR_vs_phasej",
        "dSSIM_vs_phasej",
        "dLPIPS_vs_phasej",
        "mean_abs_delta",
        "mean_changed_fraction",
        "support_source",
        "evidence_bank",
        "bank_path",
        "error",
    ]
    fields = preferred_fields + sorted({key for row in rows for key in row} - set(preferred_fields))
    csv_path = report_root / "v101_renderpy_endpoint_full9_summary.csv"
    json_path = report_root / "v101_renderpy_endpoint_full9_summary.json"
    md_path = report_root / "v101_renderpy_endpoint_full9_summary.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "method": str(args.method_name),
        "endpoint_method": str(args.endpoint_method),
        "v100_output_root": str(args.v100_output_root),
        "build_banks": bool(args.build_banks),
        "require_bank": bool(args.require_bank),
        "bank_residual_dtype": str(args.bank_residual_dtype),
        "bank_depth_dtype": str(args.bank_depth_dtype),
        "bank_root": str(args.bank_root),
        "all_returncodes_zero": all(
            int(r.get("bank_rc", 0)) == 0 and int(r["render_rc"]) == 0 and int(r["eval_rc"]) == 0
            for r in rows
        ),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# v101 Render.py Endpoint Full9 Summary",
        "",
        f"- method: `{args.method_name}`",
        f"- endpoint method: `{args.endpoint_method}`",
        f"- build banks: `{bool(args.build_banks)}`",
        f"- require bank: `{bool(args.require_bank)}`",
        f"- all return codes zero: `{payload['all_returncodes_zero']}`",
        "",
        "| scene | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | dPSNR Phase-J | dSSIM Phase-J | dLPIPS Phase-J | support source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if int(r.get("bank_rc", 0)) != 0 or int(r.get("render_rc", 1)) != 0 or int(r.get("eval_rc", 1)) != 0:
            lines.append(
                f"| {r.get('scene', '')} | nan | nan | nan | nan | nan | nan | nan | nan | nan | "
                f"ERROR: {r.get('error', 'see log')} |"
            )
            continue
        lines.append(
            f"| {r['scene']} | {r['PSNR']:.6f} | {r['SSIM']:.6f} | {r['LPIPS']:.6f} | "
            f"{r['dPSNR_vs_clean']:+.6f} | {r['dSSIM_vs_clean']:+.6f} | {r['dLPIPS_vs_clean']:+.6f} | "
            f"{r['dPSNR_vs_phasej']:+.6f} | {r['dSSIM_vs_phasej']:+.6f} | {r['dLPIPS_vs_phasej']:+.6f} | "
            f"{r['support_source']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.wandb:
        _wandb_log(args, rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, indent=2))


def _wandb_log(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    try:
        import wandb
    except Exception as exc:
        print(f"[v101-full9] wandb unavailable: {exc}")
        return
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        mode=args.wandb_mode,
        dir=args.wandb_dir or None,
        config={"method": args.method_name, "endpoint_method": args.endpoint_method},
    )
    means = {}
    for key in ("dPSNR_vs_clean", "dSSIM_vs_clean", "dLPIPS_vs_clean", "dPSNR_vs_phasej", "dSSIM_vs_phasej", "dLPIPS_vs_phasej"):
        values = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
        means[f"mean/{key}"] = sum(values) / len(values) if values else math.nan
    run.log(means)
    run.summary.update(means)
    run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run render.py-consuming v101 endpoint over fixed full9 v100 models.")
    parser.add_argument("--v100_output_root", default="/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625")
    parser.add_argument("--v100_summary_json", default="outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_full9_fixed_20260625/v100_checkpoint_attached_ela_full9_summary.json")
    parser.add_argument("--report_root", required=True)
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--gpus", default="1,2,3,5")
    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--endpoint_method", default=V100_METHOD)
    parser.add_argument("--method_name", default=V101_METHOD)
    parser.add_argument("--base_model_root", default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix")
    parser.add_argument("--base_method_name", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument("--build_banks", action="store_true")
    parser.add_argument("--force_rebuild_banks", action="store_true")
    parser.add_argument("--require_bank", action="store_true")
    parser.add_argument("--bank_root", default="")
    parser.add_argument("--bank_residual_dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--bank_depth_dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet_meshprior")
    parser.add_argument("--wandb_group", default="v101_renderpy_endpoint_full9")
    parser.add_argument("--wandb_name", default="v101_renderpy_endpoint_full9")
    parser.add_argument("--wandb_mode", default="offline")
    parser.add_argument("--wandb_dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes = [scene.strip() for scene in args.scenes.replace(" ", ",").split(",") if scene.strip()]
    gpus = [int(item) for item in args.gpus.replace(" ", ",").split(",") if item.strip()]
    if not scenes or not gpus:
        raise RuntimeError("missing scenes or GPUs")
    v100_rows = _load_v100_rows(ROOT / args.v100_summary_json)
    missing = [scene for scene in scenes if scene not in v100_rows]
    if missing:
        raise RuntimeError(f"missing v100 summary rows: {missing}")
    Path(args.report_root).mkdir(parents=True, exist_ok=True)
    pending = list(scenes)
    gpu_slots = gpus[: max(1, min(len(gpus), int(args.max_parallel)))]
    free_gpus = list(gpu_slots)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_slots)) as pool:
        active: dict[Future, tuple[str, int]] = {}
        while pending or active:
            while pending and free_gpus:
                scene = pending.pop(0)
                gpu = free_gpus.pop(0)
                active[pool.submit(_run_scene, args, scene, gpu, v100_rows)] = (scene, gpu)
                print(f"[v101-full9] launched {scene} gpu={gpu}", flush=True)
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                scene, gpu = active.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"scene": scene, "gpu": gpu, "bank_rc": 999, "render_rc": 999, "eval_rc": 999, "error": str(exc)}
                print(f"[v101-full9] finished {scene} gpu={gpu} rc={row.get('render_rc')}/{row.get('eval_rc')}", flush=True)
                results.append(row)
                free_gpus.append(gpu)
    results.sort(key=lambda row: scenes.index(row["scene"]))
    _write_summary(args, results)
    return (
        0
        if all(
            int(r.get("bank_rc", 0)) == 0 and int(r.get("render_rc", 1)) == 0 and int(r.get("eval_rc", 1)) == 0
            for r in results
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
