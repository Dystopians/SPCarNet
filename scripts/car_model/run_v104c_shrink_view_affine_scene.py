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
ENDPOINT_METHOD = "ours_26000_v100_checkpoint_attached_ela_endpoint"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _run_cmd(cmd: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        elapsed = time.perf_counter() - started
        handle.write(f"\n[exit_code] {proc.returncode}\n[wall_sec] {elapsed:.6f}\n")
    return int(proc.returncode), float(elapsed)


def _metrics_delta(lhs: dict[str, Any], rhs: dict[str, Any], key: str) -> float | None:
    if key not in lhs or key not in rhs:
        return None
    return float(lhs[key]) - float(rhs[key])


def _load_clean_metrics(scene: str, clean_root: Path) -> dict[str, Any]:
    payload = _read_json(clean_root / scene / "results.json")
    metrics = payload.get("ours_26000", {})
    return metrics if isinstance(metrics, dict) else {}


def run_scene(args: argparse.Namespace) -> dict[str, Any]:
    scene = str(args.scene)
    model = Path(args.package_root) / scene / "detached_model"
    if not model.is_dir():
        raise FileNotFoundError(model)
    v102_bank = Path(args.v102_bank_root) / scene / "v102_preprojected_delta_bank.pt"
    field = Path(args.field_root) / scene / "v104c_shrink_view_affine_min1_minviews1_field.pt"
    report_root = Path(args.report_root) / scene
    report_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if str(args.gpu).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")

    v102_rc = 0
    v102_wall = 0.0
    if (args.force_v102 or not v102_bank.is_file()) and bool(args.build_v102_if_missing):
        v102_cmd = [
            sys.executable,
            "scripts/car_model/run_v102_preprojected_delta_scene.py",
            "--scene",
            scene,
            "--package_root",
            str(args.package_root),
            "--v102_bank_root",
            str(args.v102_bank_root),
            "--report_root",
            str(args.v102_report_root),
            "--endpoint_method",
            str(args.endpoint_method),
            "--iteration",
            str(args.iteration),
            "--delta_dtype",
            "float32",
            "--gpu",
            str(args.gpu),
            "--no_intermediate_outputs",
        ]
        if args.force_v102:
            v102_cmd.append("--force")
        v102_rc, v102_wall = _run_cmd(v102_cmd, report_root / f"{scene}_v102.log", env)
    elif not v102_bank.is_file():
        raise FileNotFoundError(f"missing v102 bank and --build_v102_if_missing not set: {v102_bank}")

    field_rc = 999
    field_wall = 0.0
    if v102_rc == 0 and (args.force_field or not field.is_file()):
        field_rc, field_wall = _run_cmd(
            [
                sys.executable,
                "scripts/car_model/build_v104b_centered_view_affine_residual_field.py",
                "--model_path",
                str(model),
                "--delta_bank_path",
                str(v102_bank),
                "--output_field",
                str(field),
                "--endpoint_method",
                str(args.endpoint_method),
                "--iteration",
                str(args.iteration),
                "--split",
                "test",
                "--renderer_scaling",
                str(args.renderer_scaling),
                "--residual_dtype",
                str(args.residual_dtype),
                "--min_count",
                "1",
                "--min_views",
                "1",
                "--ridge",
                str(args.ridge),
                "--residual_clip",
                str(args.residual_clip),
                "--view_std_floor",
                str(args.view_std_floor),
                "--rank_rtol",
                str(args.rank_rtol),
                "--condition_max",
                str(args.condition_max),
                "--fallback_mode",
                "shrink",
                "--chunk_pixels",
                str(args.chunk_pixels),
            ],
            report_root / f"{scene}_field.log",
            env,
        )
    elif field.is_file():
        field_rc = 0

    output_method = args.output_method or f"ours_{int(args.iteration)}_v104c_shrink_view_affine_min1_minviews1_{scene}"
    render_rc = 999
    render_wall = 0.0
    render_report_path = model / "test" / output_method / "render_py_endpoint_report.json"
    if field_rc == 0 and (args.force_render or not render_report_path.is_file()):
        render_rc, render_wall = _run_cmd(
            [
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
                output_method,
                "--checkpoint_endpoint_surface_field_path",
                str(field),
                "--checkpoint_endpoint_require_surface_field",
                "--checkpoint_endpoint_no_intermediate_outputs",
                "--quiet",
            ],
            report_root / f"{scene}_render.log",
            env,
        )
    elif render_report_path.is_file():
        render_rc = 0

    eval_rc = 999
    eval_wall = 0.0
    results_before = _read_json(model / "results.json")
    if render_rc == 0 and (args.force_eval or output_method not in results_before):
        eval_rc, eval_wall = _run_cmd(
            [
                sys.executable,
                "scripts/car_model/evaluate_render_split_metrics.py",
                "-m",
                str(model),
                "--split",
                "test",
                "--methods",
                output_method,
                "--merge_model_results",
            ],
            report_root / f"{scene}_eval.log",
            env,
        )
    elif output_method in results_before:
        eval_rc = 0

    results = _read_json(model / "results.json")
    metrics = results.get(output_method, {}) if isinstance(results.get(output_method), dict) else {}
    clean_metrics = _load_clean_metrics(scene, Path(args.clean_root))
    reference_key = args.reference_method or f"ours_{int(args.iteration)}_v101_detached_package_full9_{scene}"
    reference_metrics = results.get(reference_key, {}) if isinstance(results.get(reference_key), dict) else {}
    v102_key = f"ours_{int(args.iteration)}_v102_delta_fast_{scene}"
    v102_metrics = results.get(v102_key, {}) if isinstance(results.get(v102_key), dict) else {}
    manifest = _read_json(field.with_suffix(".manifest.json"))
    render_report = _read_json(render_report_path)
    solve_stats = manifest.get("solve_stats", {}) if isinstance(manifest.get("solve_stats"), dict) else {}
    payload = {
        "schema_version": 1,
        "scene": scene,
        "model": str(model),
        "endpoint_method": str(args.endpoint_method),
        "iteration": int(args.iteration),
        "output_method": output_method,
        "v102_bank": str(v102_bank),
        "field": str(field),
        "field_manifest": str(field.with_suffix(".manifest.json")),
        "render_report": str(render_report_path),
        "build_v102_if_missing": bool(args.build_v102_if_missing),
        "return_codes": {
            "v102": int(v102_rc),
            "field": int(field_rc),
            "render": int(render_rc),
            "eval": int(eval_rc),
        },
        "wall_sec": {
            "v102": float(v102_wall),
            "field": float(field_wall),
            "render": float(render_wall),
            "eval": float(eval_wall),
        },
        "metrics": metrics,
        "clean_metrics": clean_metrics,
        "reference_key": reference_key,
        "reference_metrics": reference_metrics,
        "v102_key": v102_key,
        "v102_metrics": v102_metrics,
        "deltas": {
            "vs_clean": {
                "PSNR": _metrics_delta(metrics, clean_metrics, "PSNR"),
                "SSIM": _metrics_delta(metrics, clean_metrics, "SSIM"),
                "LPIPS": _metrics_delta(metrics, clean_metrics, "LPIPS"),
            },
            "vs_v101_v102a": {
                "PSNR": _metrics_delta(metrics, reference_metrics, "PSNR"),
                "SSIM": _metrics_delta(metrics, reference_metrics, "SSIM"),
                "LPIPS": _metrics_delta(metrics, reference_metrics, "LPIPS"),
            },
        },
        "field_stats": {
            "valid_triangles": manifest.get("valid_triangles"),
            "total_accumulated_pixels": manifest.get("total_accumulated_pixels"),
            "view_affine_triangles": solve_stats.get("view_affine_triangles"),
            "fallback_triangles": solve_stats.get("fallback_triangles"),
            "shrink_alpha_mean": solve_stats.get("shrink_alpha_mean"),
            "elapsed_sec": manifest.get("elapsed_sec"),
        },
        "render_stats": {
            "mode": render_report.get("mode"),
            "support_source": render_report.get("support_source"),
            "target_frames": render_report.get("target_frames"),
            "elapsed_sec": render_report.get("elapsed_sec"),
            "mean_abs_delta": render_report.get("mean_abs_delta"),
            "mean_changed_fraction": render_report.get("mean_changed_fraction"),
            "mean_surface_valid_fraction": render_report.get("mean_surface_valid_fraction"),
            "no_test_gt_used_for_policy": render_report.get("no_test_gt_used_for_policy"),
        },
        "logs": {
            "v102": str(report_root / f"{scene}_v102.log"),
            "field": str(report_root / f"{scene}_field.log"),
            "render": str(report_root / f"{scene}_render.log"),
            "eval": str(report_root / f"{scene}_eval.log"),
        },
    }
    payload["passed"] = bool(
        v102_rc == 0
        and field_rc == 0
        and render_rc == 0
        and eval_rc == 0
        and str(render_report.get("mode")) == "surface_residual_field_endpoint"
        and bool(render_report.get("no_test_gt_used_for_policy"))
        and str(render_report.get("support_source", "")).startswith("v102_surface_residual_field:")
        and {"PSNR", "SSIM", "LPIPS"}.issubset(metrics.keys())
    )
    out_json = report_root / f"{scene}_v104c_shrink_view_affine_report.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md = report_root / f"{scene}_v104c_shrink_view_affine_report.md"
    out_md.write_text(
        "\n".join(
            [
                f"# v104c Shrink View-Affine Report: {scene}",
                "",
                f"- passed: `{payload['passed']}`",
                f"- metrics: `{metrics}`",
                f"- delta vs clean: `{payload['deltas']['vs_clean']}`",
                f"- delta vs v101/v102a: `{payload['deltas']['vs_v101_v102a']}`",
                f"- field stats: `{payload['field_stats']}`",
                f"- render stats: `{payload['render_stats']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(out_json), "passed": payload["passed"], "metrics": metrics}, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/render/evaluate v104c shrink view-affine field for one scene.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--package_root", default="/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625")
    parser.add_argument("--v102_bank_root", default="/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625")
    parser.add_argument("--field_root", default="/dev/shm/peilincai_spcarnet_v104c_shrink_view_affine_field_20260625")
    parser.add_argument("--report_root", default="outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625")
    parser.add_argument("--v102_report_root", default="outputs/carnet/meshsplatopt/ecsr_phase_v102_preprojected_delta_bank_20260625")
    parser.add_argument("--clean_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k")
    parser.add_argument("--endpoint_method", default=ENDPOINT_METHOD)
    parser.add_argument("--reference_method", default="")
    parser.add_argument("--output_method", default="")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--renderer_scaling", type=int, default=4)
    parser.add_argument("--residual_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--residual_clip", type=float, default=0.08)
    parser.add_argument("--view_std_floor", type=float, default=1e-4)
    parser.add_argument("--rank_rtol", type=float, default=1e-7)
    parser.add_argument("--condition_max", type=float, default=1e8)
    parser.add_argument("--chunk_pixels", type=int, default=262144)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--build_v102_if_missing", action="store_true")
    parser.add_argument("--force_v102", action="store_true")
    parser.add_argument("--force_field", action="store_true")
    parser.add_argument("--force_render", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = run_scene(parse_args())
    return 0 if payload.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
