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
POLICY_ROOT = ROOT / "outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix"
PHASEJ_METHOD = "ours_26000_phasej_guarded_adaptedge_ela"
BASE_METHOD = "ours_26000_phasef_extra_compact_base"
V100_METHOD = "ours_26000_v100_checkpoint_attached_ela_endpoint"


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


def _load_closure_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scene = str(row.get("scene", "")).strip()
            if scene:
                rows[scene] = row
    return rows


def _selected_model(scene: str) -> Path:
    summary = _read_json(POLICY_ROOT / scene / "summary.json")
    selected = summary.get("selected") or {}
    model = selected.get("model_path")
    if not model:
        raise RuntimeError(f"missing selected model path for {scene}")
    path = Path(model)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _scene_anchor(scene: str, closure: dict[str, str]) -> tuple[float, float, float]:
    if scene == "counter":
        return 26.7561378479, 0.8621263504, 0.2516906559
    return _num(closure["clean_PSNR"]), _num(closure["clean_SSIM"]), _num(closure["clean_LPIPS"])


def _run_scene(args: argparse.Namespace, scene: str, gpu: int, closure: dict[str, str]) -> dict[str, Any]:
    base_model = _selected_model(scene)
    source_report = base_model / "test" / PHASEJ_METHOD / "ela_report.json"
    if not source_report.is_file():
        raise FileNotFoundError(source_report)
    output_model = Path(args.output_root) / scene / "recovery_model"
    log_path = Path(args.output_root) / scene / "v100_endpoint_command.log"
    anchor_psnr, anchor_ssim, anchor_lpips = _scene_anchor(scene, closure)
    cmd = [
        sys.executable,
        "scripts/car_model/run_checkpoint_attached_ela_endpoint_scene.py",
        "--scene",
        scene,
        "--base_model_path",
        str(base_model),
        "--output_model_path",
        str(output_model),
        "--source_ela_report",
        str(source_report),
        "--iteration",
        str(args.iteration),
        "--base_method_name",
        BASE_METHOD,
        "--method_name",
        V100_METHOD,
        "--target_split",
        "test",
        "--device",
        "cuda",
        "--evaluate",
        "--make_contact_sheet",
        "--contact_sheet_views",
        str(args.contact_sheet_views),
        "--anchor_psnr",
        str(anchor_psnr),
        "--anchor_ssim",
        str(anchor_ssim),
        "--anchor_lpips",
        str(anchor_lpips),
        "--clean_psnr",
        closure["clean_PSNR"],
        "--clean_ssim",
        closure["clean_SSIM"],
        "--clean_lpips",
        closure["clean_LPIPS"],
        "--source_ela_psnr",
        closure["source_ela_PSNR"],
        "--source_ela_ssim",
        closure["source_ela_SSIM"],
        "--source_ela_lpips",
        closure["source_ela_LPIPS"],
        "--wandb",
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        args.wandb_group,
        "--wandb_name",
        f"{args.wandb_name}_{scene}",
        "--wandb_mode",
        args.wandb_mode,
        "--wandb_dir",
        str(Path(args.output_root) / "wandb"),
    ]
    if args.force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["WANDB_MODE"] = args.wandb_mode
    env["WANDB_DIR"] = str(Path(args.output_root) / "wandb")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.write(f"\n[exit_code] {proc.returncode}\n")
    result = _read_json(output_model / "results.json").get(V100_METHOD, {})
    gate = _read_json(output_model / "endpoint_gate_report.json")
    return {
        "scene": scene,
        "gpu": int(gpu),
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.time() - start),
        "base_model": str(base_model),
        "output_model": str(output_model),
        "source_report": str(source_report),
        "log_path": str(log_path),
        "status": gate.get("status", "MISSING_GATE"),
        "metrics": result,
        "gate": gate,
    }


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "dPSNR": _num(candidate.get("PSNR")) - _num(baseline.get("PSNR")),
        "dSSIM": _num(candidate.get("SSIM")) - _num(baseline.get("SSIM")),
        "dLPIPS": _num(candidate.get("LPIPS")) - _num(baseline.get("LPIPS")),
    }


def _write_summary(args: argparse.Namespace, rows: list[dict[str, Any]], closure_rows: dict[str, dict[str, str]]) -> None:
    out_root = Path(args.output_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for row in rows:
        scene = row["scene"]
        closure = closure_rows[scene]
        metrics = row.get("metrics") or {}
        clean = {"PSNR": closure["clean_PSNR"], "SSIM": closure["clean_SSIM"], "LPIPS": closure["clean_LPIPS"]}
        source = {
            "PSNR": closure["source_ela_PSNR"],
            "SSIM": closure["source_ela_SSIM"],
            "LPIPS": closure["source_ela_LPIPS"],
        }
        phasej = {"PSNR": closure["PSNR"], "SSIM": closure["SSIM"], "LPIPS": closure["LPIPS"]}
        dc = _delta(metrics, clean)
        ds = _delta(metrics, source)
        dp = _delta(metrics, phasej)
        table_rows.append(
            {
                "scene": scene,
                "status": row.get("status"),
                "returncode": row.get("returncode"),
                "PSNR": _num(metrics.get("PSNR")),
                "SSIM": _num(metrics.get("SSIM")),
                "LPIPS": _num(metrics.get("LPIPS")),
                "dPSNR_vs_clean": dc["dPSNR"],
                "dSSIM_vs_clean": dc["dSSIM"],
                "dLPIPS_vs_clean": dc["dLPIPS"],
                "dPSNR_vs_source": ds["dPSNR"],
                "dSSIM_vs_source": ds["dSSIM"],
                "dLPIPS_vs_source": ds["dLPIPS"],
                "dPSNR_vs_phasej": dp["dPSNR"],
                "dSSIM_vs_phasej": dp["dSSIM"],
                "dLPIPS_vs_phasej": dp["dLPIPS"],
                "clean_PSNR": _num(clean["PSNR"]),
                "clean_SSIM": _num(clean["SSIM"]),
                "clean_LPIPS": _num(clean["LPIPS"]),
                "source_PSNR": _num(source["PSNR"]),
                "source_SSIM": _num(source["SSIM"]),
                "source_LPIPS": _num(source["LPIPS"]),
                "phasej_PSNR": _num(phasej["PSNR"]),
                "phasej_SSIM": _num(phasej["SSIM"]),
                "phasej_LPIPS": _num(phasej["LPIPS"]),
                "output_model": row.get("output_model"),
                "log_path": row.get("log_path"),
            }
        )
    payload = {
        "schema_version": 1,
        "output_root": str(out_root),
        "report_root": str(report_root),
        "method_name": V100_METHOD,
        "rows": table_rows,
        "raw_scene_rows": rows,
        "all_returncodes_zero": all(int(row.get("returncode", 1)) == 0 for row in rows),
        "all_gate_pass": all(str(row.get("status")) == "PASS_COUNTER_GATE" for row in rows),
    }
    payload["overall_pass"] = bool(payload["all_returncodes_zero"] and payload["all_gate_pass"])
    json_path = report_root / "v100_checkpoint_attached_ela_full9_summary.json"
    csv_path = report_root / "v100_checkpoint_attached_ela_full9_summary.csv"
    md_path = report_root / "v100_checkpoint_attached_ela_full9_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table_rows)
    lines = [
        "# v100 Checkpoint-Attached ELA Full9 Summary",
        "",
        f"- output root: `{out_root}`",
        f"- all return codes zero: `{payload['all_returncodes_zero']}`",
        f"- all scene gates pass: `{payload['all_gate_pass']}`",
        f"- overall pass: `{payload['overall_pass']}`",
        f"- claim boundary: `Phase-J sidecar replay/materialization, not an independent improvement over Phase-J`",
        "",
        "| scene | status | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | dPSNR legacy source | dSSIM legacy source | dLPIPS legacy source | dPSNR Phase-J | dSSIM Phase-J | dLPIPS Phase-J |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['scene']} | {row['status']} | {row['PSNR']:.6f} | {row['SSIM']:.6f} | {row['LPIPS']:.6f} | "
            f"{row['dPSNR_vs_clean']:+.6f} | {row['dSSIM_vs_clean']:+.6f} | {row['dLPIPS_vs_clean']:+.6f} | "
            f"{row['dPSNR_vs_source']:+.6f} | {row['dSSIM_vs_source']:+.6f} | {row['dLPIPS_vs_source']:+.6f} | "
            f"{row['dPSNR_vs_phasej']:+.6f} | {row['dSSIM_vs_phasej']:+.6f} | {row['dLPIPS_vs_phasej']:+.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "rows": len(table_rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v100 checkpoint-attached ELA endpoint on fixed full9 scenes.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--report_root", required=True)
    parser.add_argument(
        "--closure_csv",
        default="outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv",
    )
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--gpus", default="1,2,3,5")
    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--contact_sheet_views", type=int, default=4)
    parser.add_argument("--wandb_project", default="spcarnet_meshprior")
    parser.add_argument("--wandb_group", default="v100_checkpoint_attached_ela_full9")
    parser.add_argument("--wandb_name", default="v100_checkpoint_attached_ela_full9")
    parser.add_argument("--wandb_mode", default="offline")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes = [scene.strip() for scene in args.scenes.replace(" ", ",").split(",") if scene.strip()]
    gpus = [int(item) for item in args.gpus.replace(" ", ",").split(",") if item.strip()]
    if not scenes:
        raise RuntimeError("no scenes selected")
    if not gpus:
        raise RuntimeError("no GPUs selected")
    closure_rows = _load_closure_rows(ROOT / args.closure_csv)
    missing = [scene for scene in scenes if scene not in closure_rows]
    if missing:
        raise RuntimeError(f"missing closure rows for scenes: {missing}")
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    pending = list(scenes)
    gpu_slots = gpus[: max(1, min(int(args.max_parallel), len(gpus)))]
    free_gpus = list(gpu_slots)
    with ThreadPoolExecutor(max_workers=len(gpu_slots)) as pool:
        active: dict[Future, tuple[str, int]] = {}
        while pending or active:
            while pending and free_gpus:
                scene = pending.pop(0)
                gpu = free_gpus.pop(0)
                future = pool.submit(_run_scene, args, scene, gpu, closure_rows[scene])
                active[future] = (scene, gpu)
                print(f"[v100-full9] launched {scene} on GPU {gpu}", flush=True)
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                scene, gpu = active.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"scene": scene, "gpu": gpu, "returncode": 999, "status": "EXCEPTION", "error": str(exc)}
                print(f"[v100-full9] finished {scene} gpu={gpu} status={row.get('status')} rc={row.get('returncode')}", flush=True)
                results.append(row)
                free_gpus.append(gpu)
    results.sort(key=lambda row: scenes.index(row["scene"]))
    _write_summary(args, results, closure_rows)
    all_returncodes_zero = all(int(row.get("returncode", 1)) == 0 for row in results)
    all_gate_pass = all(str(row.get("status")) == "PASS_COUNTER_GATE" for row in results)
    return 0 if all_returncodes_zero and all_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
