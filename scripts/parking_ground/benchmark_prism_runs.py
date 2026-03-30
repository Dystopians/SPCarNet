#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
except Exception:
    torch = None


DEFAULT_CHECKPOINTS = [15000, 16000, 18000, 20000, 21000, 24000, 30000]


def _run(cmd: List[str], cwd: str):
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _run_timed(cmd: List[str], cwd: str) -> float:
    print("[CMD]", " ".join(cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, check=True)
    return max(1e-9, time.perf_counter() - t0)


def _safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _is_nan(v: float) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _list_available_iterations(model_path: Path) -> List[int]:
    out: List[int] = []
    point_cloud_dir = model_path / "point_cloud"
    if not point_cloud_dir.exists():
        return out
    for p in point_cloud_dir.glob("iteration_*"):
        m = re.match(r"iteration_(\d+)$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def _load_metrics_results(model_path: Path) -> Dict[str, Dict[str, float]]:
    fp = model_path / "results.json"
    out: Dict[str, Dict[str, float]] = {}
    if not fp.exists():
        return out
    payload = json.loads(fp.read_text(encoding="utf-8"))
    by_method: Dict = {}
    by_scene = payload.get(str(model_path), {})
    if isinstance(by_scene, dict) and len(by_scene) > 0:
        by_method = by_scene
    elif isinstance(payload, dict):
        by_method = payload
    else:
        return out
    for k, v in by_method.items():
        if not isinstance(v, dict):
            continue
        m = re.search(r"(\d+)$", str(k))
        if not m:
            continue
        out[m.group(1)] = {
            "PSNR": _safe_float(v.get("PSNR")),
            "SSIM": _safe_float(v.get("SSIM")),
            "LPIPS": _safe_float(v.get("LPIPS")),
        }
    return out


def _compute_mae_for_iteration(model_path: Path, iteration: int) -> Tuple[float, int]:
    method_dir = model_path / "test" / f"ours_{iteration}"
    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    if (not renders_dir.exists()) or (not gt_dir.exists()):
        return float("nan"), 0
    render_files = sorted([p for p in renders_dir.glob("*.png") if p.is_file()])
    if len(render_files) == 0:
        return float("nan"), 0
    total_abs = 0.0
    total_cnt = 0
    paired = 0
    for rp in render_files:
        gp = gt_dir / rp.name
        if not gp.exists():
            continue
        r = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32) / 255.0
        g = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0
        if r.shape != g.shape:
            continue
        diff = np.abs(r - g)
        total_abs += float(np.sum(diff))
        total_cnt += int(diff.size)
        paired += 1
    if total_cnt <= 0:
        return float("nan"), 0
    return float(total_abs / float(total_cnt)), paired


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    depth = payload.get("depth", {}) or {}
    normal = payload.get("normal", {}) or {}
    out["AbsRel"] = _safe_float(depth.get("abs_rel"))
    out["Delta1.25"] = _safe_float(depth.get("delta_1.25"))
    out["DepthMAE"] = _safe_float(depth.get("mae"))
    if isinstance(normal, dict):
        out["MeanAngle"] = _safe_float(normal.get("mean_ang_deg"))
        out["AbsCos"] = _safe_float(normal.get("mean_abs_cos"))
    return out


def _count_vertices_triangles(model_path: Path, iteration: int) -> Dict[str, float]:
    out = {"vertices": float("nan"), "triangles": float("nan")}
    state = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if (torch is None) or (not state.exists()):
        return out
    try:
        payload = torch.load(str(state), map_location="cpu")
        verts = payload.get("triangles_points", None)
        tris = payload.get("_triangle_indices", None)
        if verts is not None and hasattr(verts, "shape"):
            out["vertices"] = float(int(verts.shape[0]))
        if tris is not None and hasattr(tris, "shape"):
            out["triangles"] = float(int(tris.shape[0]))
    except Exception:
        pass
    return out


def _parse_sparse_depth_enabled(model_path: Path) -> bool:
    cfg = model_path / "cfg_args"
    if not cfg.exists():
        return False
    text = cfg.read_text(encoding="utf-8", errors="ignore")
    needles = [
        "enable_sparse_colmap_depth_loss=True",
        "'enable_sparse_colmap_depth_loss': True",
        '"enable_sparse_colmap_depth_loss": true',
    ]
    return any(n in text for n in needles)


def _collect_run_flags(model_path: Path) -> Dict[str, object]:
    cleanup_summary = model_path / "prism_debug" / "final_cleanup_summary.json"
    cleanup_enabled = False
    cleanup_pruned = 0
    if cleanup_summary.exists():
        try:
            payload = json.loads(cleanup_summary.read_text(encoding="utf-8"))
            cleanup_enabled = bool(payload.get("cleanup_enabled", False))
            cleanup_pruned = int(payload.get("pruned_triangles", 0))
        except Exception:
            pass

    rollback_happened = False
    val_dir = model_path / "prism_validation"
    if val_dir.exists():
        for p in sorted(val_dir.glob("validation_iter_*.json")):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                if not bool(payload.get("pass_gate", True)):
                    rollback_happened = True
                    break
            except Exception:
                continue

    return {
        "final_cleanup_enabled": cleanup_enabled,
        "final_cleanup_pruned": int(cleanup_pruned),
        "rollback_happened": rollback_happened,
        "sparse_colmap_depth_supervision": _parse_sparse_depth_enabled(model_path),
    }


def _geometry_key(row: Dict) -> Tuple[float, float, float, float]:
    g = row["geometry"]
    if any(_is_nan(g[k]) for k in ["AbsRel", "MeanAngle", "DepthMAE", "Delta1.25"]):
        return (float("inf"), float("inf"), float("inf"), float("inf"))
    return (g["AbsRel"], g["MeanAngle"], g["DepthMAE"], -g["Delta1.25"])


def _psnr_key(row: Dict) -> float:
    v = row["metrics"]["PSNR"]
    if _is_nan(v):
        return -float("inf")
    return v


def _pick_run_summaries(rows: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    by_run: Dict[str, List[Dict]] = {}
    for r in rows:
        by_run.setdefault(r["run_name"], []).append(r)

    for run_name, items in by_run.items():
        sorted_items = sorted(items, key=lambda x: int(x["checkpoint"]))
        best_geo = min(sorted_items, key=_geometry_key)
        best_psnr = max(sorted_items, key=_psnr_key)
        final_ckpt = max(sorted_items, key=lambda x: int(x["checkpoint"]))
        out[run_name] = {
            "best_by_geometry": int(best_geo["checkpoint"]),
            "best_by_psnr": int(best_psnr["checkpoint"]),
            "final_checkpoint": int(final_ckpt["checkpoint"]),
            "best_by_geometry_row": best_geo,
            "best_by_psnr_row": best_psnr,
            "final_row": final_ckpt,
        }
    return out


def _write_markdown(out_path: Path, rows: List[Dict], run_summaries: Dict[str, Dict]):
    lines: List[str] = []
    lines.append("# PRISM GeoGate Benchmark")
    lines.append("")
    lines.append("## Per-Checkpoint Results")
    lines.append("")
    lines.append(
        "| Run | Checkpoint | FPS | PSNR | SSIM | LPIPS | MAE | AbsRel | Delta<1.25 | MeanAngle | AbsCos | Depth MAE | triangles | vertices | final cleanup | rollback | sparse depth sup |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"
    )
    for r in rows:
        lines.append(
            "| {run} | {ckpt} | {fps:.3f} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {mae:.4f} | {absrel:.4f} | {delta:.4f} | {ang:.4f} | {abscos:.4f} | {dmae:.4f} | {tri:.0f} | {vert:.0f} | {cleanup} | {rollback} | {sparse} |".format(
                run=r["run_name"],
                ckpt=r["checkpoint"],
                fps=r["fps"],
                psnr=r["metrics"]["PSNR"],
                ssim=r["metrics"]["SSIM"],
                lpips=r["metrics"]["LPIPS"],
                mae=r["metrics"]["MAE"],
                absrel=r["geometry"]["AbsRel"],
                delta=r["geometry"]["Delta1.25"],
                ang=r["geometry"]["MeanAngle"],
                abscos=r["geometry"]["AbsCos"],
                dmae=r["geometry"]["DepthMAE"],
                tri=r["mesh"]["triangles"],
                vert=r["mesh"]["vertices"],
                cleanup=f"{int(r['run_flags']['final_cleanup_enabled'])}/{int(r['run_flags']['final_cleanup_pruned'])}",
                rollback=int(r["run_flags"]["rollback_happened"]),
                sparse=int(r["run_flags"]["sparse_colmap_depth_supervision"]),
            )
        )

    lines.append("")
    lines.append("## Run-Level Highlights")
    lines.append("")
    lines.append("| Run | best-by-geometry | best-by-psnr | final checkpoint |")
    lines.append("|---|---:|---:|---:|")
    for run_name in sorted(run_summaries.keys()):
        s = run_summaries[run_name]
        lines.append(
            f"| {run_name} | {s['best_by_geometry']} | {s['best_by_psnr']} | {s['final_checkpoint']} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def main():
    parser = argparse.ArgumentParser(description="Benchmark PRISM runs across multiple checkpoints.")
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="benchmarks/prism_parking_ground")
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help="Checkpoint iterations to benchmark for each run.",
    )
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
    geometry_root = out_dir / "geometry_eval"
    geometry_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    run_specs: List[Tuple[str, Path]] = []
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"Bad --run spec: {spec}")
        run_name, model_path_raw = spec.split("=", 1)
        model_path = Path(model_path_raw).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        run_specs.append((run_name, model_path))

    for run_name, model_path in run_specs:
        print(f"[Benchmark] run={run_name} model={model_path}")
        available = set(_list_available_iterations(model_path))
        selected = [int(x) for x in args.checkpoints if int(x) in available]
        if len(selected) == 0:
            print(f"[WARN] run={run_name} has no requested checkpoints, skipping.")
            continue

        render_times: Dict[int, float] = {}
        view_counts: Dict[int, int] = {}
        if not args.skip_render_metrics:
            for ckpt in selected:
                elapsed = _run_timed(
                    [
                        "python",
                        "render.py",
                        "-s",
                        args.scene_path,
                        "-m",
                        str(model_path),
                        "--iteration",
                        str(ckpt),
                        "--eval",
                        "--split_strategy",
                        "file",
                        "--split_file",
                        args.split_file,
                        "--skip_train",
                    ],
                    cwd=str(repo_root),
                )
                render_times[ckpt] = elapsed
                renders_dir = model_path / "test" / f"ours_{ckpt}" / "renders"
                view_counts[ckpt] = len(list(renders_dir.glob("*.png"))) if renders_dir.exists() else 0
            _run(["python", "metrics.py", "-m", str(model_path)], cwd=str(repo_root))

        metrics_by_iter = _load_metrics_results(model_path)
        run_flags = _collect_run_flags(model_path)
        run_geo_dir = geometry_root / run_name
        run_geo_dir.mkdir(parents=True, exist_ok=True)

        for ckpt in selected:
            geom_out = run_geo_dir / f"iter_{ckpt}.json"
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
                        str(ckpt),
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

            metrics = metrics_by_iter.get(str(ckpt), {})
            mae, paired = _compute_mae_for_iteration(model_path, ckpt)
            elapsed = render_times.get(ckpt, float("nan"))
            count = view_counts.get(ckpt, 0)
            fps = float(count) / elapsed if (count > 0 and (not _is_nan(elapsed))) else float("nan")
            row = {
                "run_name": run_name,
                "model_path": str(model_path),
                "checkpoint": int(ckpt),
                "fps": float(fps),
                "num_eval_views": int(count),
                "render_elapsed_sec": float(elapsed) if not _is_nan(elapsed) else float("nan"),
                "metrics": {
                    "PSNR": _safe_float(metrics.get("PSNR")),
                    "SSIM": _safe_float(metrics.get("SSIM")),
                    "LPIPS": _safe_float(metrics.get("LPIPS")),
                    "MAE": float(mae),
                    "mae_paired_views": int(paired),
                },
                "geometry": _load_geometry_eval(geom_out),
                "mesh": _count_vertices_triangles(model_path, ckpt),
                "run_flags": run_flags,
                "geometry_eval_output": str(geom_out),
            }
            rows.append(row)

    rows = sorted(rows, key=lambda x: (x["run_name"], int(x["checkpoint"])))
    run_summaries = _pick_run_summaries(rows)
    result = {
        "created_at": ts,
        "scene_path": args.scene_path,
        "split_file": args.split_file,
        "checkpoints_requested": [int(x) for x in args.checkpoints],
        "rows": rows,
        "run_summaries": run_summaries,
    }

    json_path = out_dir / "benchmark_results.json"
    md_path = out_dir / "benchmark_summary.md"
    json_path.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
    _write_markdown(md_path, rows, run_summaries)

    print(f"[Benchmark] JSON: {json_path}")
    print(f"[Benchmark] Markdown: {md_path}")


if __name__ == "__main__":
    main()
