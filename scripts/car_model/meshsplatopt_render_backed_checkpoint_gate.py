#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_adapter import apply_edit_to_checkpoint_copy, load_checkpoint_state
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit


@dataclass(frozen=True)
class ModelEvidence:
    model_path: str
    checkpoint_path: str
    triangles: int
    vertices: int
    render_metrics: dict[str, float] | None
    geometry_metrics: dict[str, float] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", required=True)
    parser.add_argument("--candidate_model")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--edit_json")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--iteration", type=int, default=200)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_points_per_view", type=int, default=500)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--max_psnr_drop", type=float, default=0.02)
    parser.add_argument("--max_ssim_drop", type=float, default=0.002)
    parser.add_argument("--max_lpips_increase", type=float, default=0.005)
    parser.add_argument("--max_absrel_increase", type=float, default=0.02)
    parser.add_argument("--max_depth_mae_increase", type=float, default=0.1)
    parser.add_argument("--max_normal_angle_increase", type=float, default=1.0)
    return parser.parse_args()


def checkpoint_path(model_path: Path, iteration: int) -> Path:
    return model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"


def load_topology(path: Path) -> tuple[int, int]:
    payload = load_checkpoint_state(path)
    return int(payload["_triangle_indices"].shape[0]), int(payload["triangles_points"].shape[0])


def load_render_metrics(model_path: Path, iteration: int) -> dict[str, float] | None:
    path = model_path / "results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    row = data.get(f"ours_{iteration}")
    if row is None and data:
        row = next(iter(data.values()))
    if row is None:
        return None
    return {k: float(row[k]) for k in ("PSNR", "SSIM", "LPIPS") if k in row}


def load_geometry_metrics(model_path: Path, iteration: int, max_points_per_view: int) -> dict[str, float] | None:
    candidates = [
        model_path / "geometry_eval_colmap" / f"iter_{iteration}_max{max_points_per_view}.json",
        model_path / "geometry_eval_colmap" / f"iter_{iteration}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, float] = {}
        if isinstance(data.get("depth"), dict):
            depth = data["depth"]
            if "count" in depth:
                out["Points"] = float(depth["count"])
            if "abs_rel" in depth:
                out["AbsRel"] = float(depth["abs_rel"])
            if "mae" in depth:
                out["DepthMAE"] = float(depth["mae"])
        if isinstance(data.get("normal"), dict):
            normal = data["normal"]
            if "mean_ang_deg" in normal:
                out["NormalMeanDeg"] = float(normal["mean_ang_deg"])
        for src, dst in [
            ("sparse_abs_rel", "AbsRel"),
            ("sparse_depth_mae", "DepthMAE"),
            ("normal_mean_angle_deg", "NormalMeanDeg"),
            ("num_points_evaluated", "Points"),
        ]:
            if src in data:
                out[dst] = float(data[src])
        for key in ("AbsRel", "DepthMAE", "NormalMeanDeg", "Points"):
            if key in data:
                out[key] = float(data[key])
        return out
    return None


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"\n[exit_code] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}; see {log_path}")


def ensure_render_metrics(model_path: Path, args: argparse.Namespace, output_root: Path, label: str, env: dict[str, str]) -> None:
    test_render_dir = model_path / "test" / f"ours_{args.iteration}" / "renders"
    if args.force_rerun or not test_render_dir.exists():
        run_command(
            [args.python, "render.py", "-m", str(model_path), "--iteration", str(args.iteration), "--skip_train"],
            output_root / "logs" / f"{label}_render.log",
            env,
        )
    if args.force_rerun or not (model_path / "results.json").exists():
        run_command(
            [args.python, "metrics.py", "-m", str(model_path)],
            output_root / "logs" / f"{label}_metrics.log",
            env,
        )


def ensure_geometry_metrics(model_path: Path, args: argparse.Namespace, output_root: Path, label: str, env: dict[str, str]) -> None:
    out = model_path / "geometry_eval_colmap" / f"iter_{args.iteration}_max{args.max_points_per_view}.json"
    if args.force_rerun or not out.exists():
        run_command(
            [
                args.python,
                "evaluate_geometry_colmap.py",
                "--model_path",
                str(model_path),
                "--iteration",
                str(args.iteration),
                "--max_points_per_view",
                str(args.max_points_per_view),
                "--output",
                str(out),
            ],
            output_root / "logs" / f"{label}_geometry.log",
            env,
        )


def materialize_candidate(args: argparse.Namespace, output_root: Path) -> Path:
    if args.candidate_model:
        return Path(args.candidate_model)
    if not args.checkpoint_path or not args.edit_json:
        raise SystemExit("Provide either --candidate_model or both --checkpoint_path and --edit_json")
    baseline_model = Path(args.baseline_model)
    model_out = output_root / "candidate_model"
    iter_dir = model_out / "point_cloud" / f"iteration_{args.iteration}"
    edit = MeshEdit(**json.loads(Path(args.edit_json).read_text(encoding="utf-8")))
    apply_edit_to_checkpoint_copy(args.checkpoint_path, edit, iter_dir)
    for name in ("cfg_args", "cameras.json", "input.ply"):
        src = baseline_model / name
        if src.exists():
            shutil.copy2(src, model_out / name)
    return model_out


def evidence_for(model_path: Path, iteration: int, max_points_per_view: int) -> ModelEvidence:
    ckpt = checkpoint_path(model_path, iteration)
    triangles, vertices = load_topology(ckpt)
    return ModelEvidence(
        model_path=str(model_path),
        checkpoint_path=str(ckpt),
        triangles=triangles,
        vertices=vertices,
        render_metrics=load_render_metrics(model_path, iteration),
        geometry_metrics=load_geometry_metrics(model_path, iteration, max_points_per_view),
    )


def compare(args: argparse.Namespace, baseline: ModelEvidence, candidate: ModelEvidence) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    deltas: dict[str, Any] = {
        "triangles": candidate.triangles - baseline.triangles,
        "vertices": candidate.vertices - baseline.vertices,
    }
    if baseline.render_metrics is None or candidate.render_metrics is None:
        reasons.append("render_metrics_missing")
    else:
        deltas["PSNR"] = candidate.render_metrics["PSNR"] - baseline.render_metrics["PSNR"]
        deltas["SSIM"] = candidate.render_metrics["SSIM"] - baseline.render_metrics["SSIM"]
        deltas["LPIPS"] = candidate.render_metrics["LPIPS"] - baseline.render_metrics["LPIPS"]
        if deltas["PSNR"] < -args.max_psnr_drop:
            reasons.append("psnr_drop_exceeds_threshold")
        if deltas["SSIM"] < -args.max_ssim_drop:
            reasons.append("ssim_drop_exceeds_threshold")
        if deltas["LPIPS"] > args.max_lpips_increase:
            reasons.append("lpips_increase_exceeds_threshold")
    if baseline.geometry_metrics is None or candidate.geometry_metrics is None:
        reasons.append("geometry_metrics_missing")
    else:
        for key, threshold, reason in [
            ("AbsRel", args.max_absrel_increase, "absrel_increase_exceeds_threshold"),
            ("DepthMAE", args.max_depth_mae_increase, "depth_mae_increase_exceeds_threshold"),
            ("NormalMeanDeg", args.max_normal_angle_increase, "normal_angle_increase_exceeds_threshold"),
        ]:
            if key in baseline.geometry_metrics and key in candidate.geometry_metrics:
                deltas[key] = candidate.geometry_metrics[key] - baseline.geometry_metrics[key]
                if deltas[key] > threshold:
                    reasons.append(reason)
            else:
                reasons.append(f"{key.lower()}_missing")
    return not reasons, reasons, deltas


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_command(["nvidia-smi"], output_root / "logs" / "nvidia_smi.log", env)
    baseline_model = Path(args.baseline_model)
    candidate_model = materialize_candidate(args, output_root)
    ensure_render_metrics(baseline_model, args, output_root, "baseline", env)
    ensure_render_metrics(candidate_model, args, output_root, "candidate", env)
    ensure_geometry_metrics(baseline_model, args, output_root, "baseline", env)
    ensure_geometry_metrics(candidate_model, args, output_root, "candidate", env)
    baseline = evidence_for(baseline_model, args.iteration, args.max_points_per_view)
    candidate = evidence_for(candidate_model, args.iteration, args.max_points_per_view)
    accepted, reasons, deltas = compare(args, baseline, candidate)
    report = {
        "status": "PASS" if accepted else "FAIL",
        "accepted": accepted,
        "reasons": reasons,
        "iteration": args.iteration,
        "gpu": args.gpu,
        "thresholds": {
            "max_psnr_drop": args.max_psnr_drop,
            "max_ssim_drop": args.max_ssim_drop,
            "max_lpips_increase": args.max_lpips_increase,
            "max_absrel_increase": args.max_absrel_increase,
            "max_depth_mae_increase": args.max_depth_mae_increase,
            "max_normal_angle_increase": args.max_normal_angle_increase,
        },
        "baseline": asdict(baseline),
        "candidate": asdict(candidate),
        "deltas": deltas,
    }
    (output_root / "render_backed_checkpoint_gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not accepted:
        raise SystemExit("render-backed checkpoint gate failed")


if __name__ == "__main__":
    main()
