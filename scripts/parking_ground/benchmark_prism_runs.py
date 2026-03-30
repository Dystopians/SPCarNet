#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _run(cmd: List[str], cwd: str):
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _latest_iteration(model_path: Path) -> int:
    point_cloud_dir = model_path / "point_cloud"
    if not point_cloud_dir.exists():
        return -1
    best = -1
    for p in point_cloud_dir.glob("iteration_*"):
        m = re.match(r"iteration_(\d+)$", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best


def _load_metrics_results(model_path: Path) -> Dict[str, float]:
    out = {"PSNR": float("nan"), "SSIM": float("nan"), "LPIPS": float("nan")}
    fp = model_path / "results.json"
    if not fp.exists():
        return out
    with open(fp, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # metrics.py output format may be either:
    # 1) { "<scene_dir>": { "ours_xxx": {...} } }  (older / nested)
    # 2) { "ours_xxx": {...} }                      (flat)
    by_method: Dict = {}
    by_scene = payload.get(str(model_path), {})
    if isinstance(by_scene, dict) and len(by_scene) > 0:
        by_method = by_scene
    elif isinstance(payload, dict):
        by_method = payload
    else:
        return out

    # Pick highest iteration method when available.
    best_iter = -1
    best_vals = None
    for k, v in by_method.items():
        if not isinstance(v, dict):
            continue
        m = re.search(r"(\d+)$", str(k))
        it = int(m.group(1)) if m else -1
        if it >= best_iter:
            best_iter = it
            best_vals = v
    if isinstance(best_vals, dict):
        out["PSNR"] = float(best_vals.get("PSNR", float("nan")))
        out["SSIM"] = float(best_vals.get("SSIM", float("nan")))
        out["LPIPS"] = float(best_vals.get("LPIPS", float("nan")))
    return out


def _load_geometry_eval(path: Path) -> Dict[str, float]:
    out = {
        "AbsRel": float("nan"),
        "Delta1.25": float("nan"),
        "MeanAngle": float("nan"),
        "AbsCos": float("nan"),
        "DepthMAE": float("nan"),
    }
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    depth = payload.get("depth", {}) or {}
    normal = payload.get("normal", {}) or {}
    out["AbsRel"] = float(depth.get("abs_rel", float("nan")))
    out["Delta1.25"] = float(depth.get("delta_1.25", float("nan")))
    out["DepthMAE"] = float(depth.get("mae", float("nan")))
    out["MeanAngle"] = float(normal.get("mean_ang_deg", float("nan"))) if isinstance(normal, dict) else float("nan")
    out["AbsCos"] = float(normal.get("mean_abs_cos", float("nan"))) if isinstance(normal, dict) else float("nan")
    return out


def _latest_validation_json(model_path: Path) -> Optional[Path]:
    vals = sorted((model_path / "prism_validation").glob("validation_iter_*.json"))
    return vals[-1] if len(vals) > 0 else None


def _artifact_paths(model_path: Path) -> Dict[str, str]:
    out = {
        "prism_round_checkpoints": str(model_path / "prism_round_checkpoints"),
        "prism_validation": str(model_path / "prism_validation"),
        "prism_debug": str(model_path / "prism_debug"),
        "geometry_eval_colmap": str(model_path / "geometry_eval_colmap"),
        "test_renders": str(model_path / "test"),
        "metrics_json": str(model_path / "results.json"),
    }
    val_latest = _latest_validation_json(model_path)
    if val_latest is not None:
        out["latest_validation_json"] = str(val_latest)
    return out


def _count_round_debug(model_path: Path) -> Dict[str, int]:
    base = model_path / "prism_round_checkpoints"
    if not base.exists():
        return {"pre_ckpt": 0, "post_ckpt": 0}
    pre = len(glob.glob(str(base / "*round_pre_*")))
    post = len(glob.glob(str(base / "*round_post_*")))
    return {"pre_ckpt": pre, "post_ckpt": post}


def _write_markdown(out_path: Path, rows: List[Dict]):
    lines = []
    lines.append("# PRISM Parking-Ground Benchmark\n")
    lines.append("| Run | Iter | PSNR | SSIM | LPIPS | MAE | AbsRel | Delta<1.25 | MeanAngle | AbsCos | RollbackByVal |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {name} | {it} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {mae:.4f} | {absrel:.4f} | {delta:.4f} | {ang:.4f} | {abscos:.4f} | {rb} |".format(
                name=r["run_name"],
                it=r["iteration"],
                psnr=r["metrics"]["PSNR"],
                ssim=r["metrics"]["SSIM"],
                lpips=r["metrics"]["LPIPS"],
                mae=r["geometry"]["DepthMAE"],
                absrel=r["geometry"]["AbsRel"],
                delta=r["geometry"]["Delta1.25"],
                ang=r["geometry"]["MeanAngle"],
                abscos=r["geometry"]["AbsCos"],
                rb=r.get("latest_validation_rollback", "n/a"),
            )
        )
    lines.append("")
    lines.append("## Debug Artifact Paths")
    for r in rows:
        lines.append("")
        lines.append(f"### {r['run_name']}")
        for k, v in r["artifacts"].items():
            lines.append(f"- {k}: `{v}`")
        lines.append(
            f"- round_checkpoint_counts: pre={r['round_debug_counts']['pre_ckpt']}, post={r['round_debug_counts']['post_ckpt']}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run fair PRISM benchmark on multiple trained runs.")
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="benchmarks/prism_parking_ground")
    parser.add_argument("--iteration", type=int, default=-1, help="-1 means latest per model")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in format name=/abs/or/relative/model_path; can be repeated",
    )
    parser.add_argument("--skip_render_metrics", action="store_true")
    parser.add_argument("--skip_geometry_eval", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (repo_root / args.output_root / ts).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"Bad --run spec: {spec}")
        run_name, model_path_raw = spec.split("=", 1)
        model_path = Path(model_path_raw).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        iteration = int(args.iteration)
        if iteration < 0:
            iteration = _latest_iteration(model_path)
        if iteration < 0:
            raise RuntimeError(f"Cannot infer iteration for model: {model_path}")

        if not args.skip_render_metrics:
            _run(
                [
                    "python",
                    "render.py",
                    "-s",
                    args.scene_path,
                    "-m",
                    str(model_path),
                    "--iteration",
                    str(iteration),
                    "--eval",
                    "--split_strategy",
                    "file",
                    "--split_file",
                    args.split_file,
                    "--skip_train",
                ],
                cwd=str(repo_root),
            )
            _run(["python", "metrics.py", "-m", str(model_path)], cwd=str(repo_root))

        geom_out = model_path / "geometry_eval_colmap" / f"iter_{iteration}_bench.json"
        if not args.skip_geometry_eval:
            _run(
                [
                    "python",
                    "evaluate_geometry_colmap.py",
                    "-s",
                    args.scene_path,
                    "-m",
                    str(model_path),
                    "--iteration",
                    str(iteration),
                    "--eval",
                    "--split_strategy",
                    "file",
                    "--split_file",
                    args.split_file,
                    "--output",
                    str(geom_out),
                ],
                cwd=str(repo_root),
            )

        latest_val = _latest_validation_json(model_path)
        latest_val_rollback = "n/a"
        if latest_val is not None:
            try:
                payload = json.loads(latest_val.read_text(encoding="utf-8"))
                latest_val_rollback = "1" if (not bool(payload.get("pass_gate", True))) else "0"
            except Exception:
                pass

        row = {
            "run_name": run_name,
            "model_path": str(model_path),
            "iteration": int(iteration),
            "metrics": _load_metrics_results(model_path),
            "geometry": _load_geometry_eval(geom_out),
            "artifacts": _artifact_paths(model_path),
            "round_debug_counts": _count_round_debug(model_path),
            "latest_validation_rollback": latest_val_rollback,
        }
        rows.append(row)

    result = {
        "created_at": ts,
        "scene_path": args.scene_path,
        "split_file": args.split_file,
        "rows": rows,
    }
    json_path = out_dir / "benchmark_results.json"
    md_path = out_dir / "benchmark_summary.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_markdown(md_path, rows)

    print(f"[Benchmark] JSON: {json_path}")
    print(f"[Benchmark] Markdown: {md_path}")


if __name__ == "__main__":
    main()
