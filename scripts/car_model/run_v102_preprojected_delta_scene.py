#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _compare_render_hashes(reference_dir: Path, output_dir: Path) -> dict[str, Any]:
    from PIL import Image
    import numpy as np

    ref_names = sorted(path.name for path in reference_dir.glob("*.png"))
    out_names = sorted(path.name for path in output_dir.glob("*.png"))
    common = sorted(set(ref_names) & set(out_names))
    mismatches = []
    diff_parts = []
    for name in common:
        if _sha256(reference_dir / name) != _sha256(output_dir / name):
            mismatches.append(name)
        ref = np.asarray(Image.open(reference_dir / name).convert("RGB"), dtype=np.int16)
        out = np.asarray(Image.open(output_dir / name).convert("RGB"), dtype=np.int16)
        diff_parts.append(np.abs(ref - out).reshape(-1))
    diff = np.concatenate(diff_parts) if diff_parts else np.array([], dtype=np.int16)
    mean_abs_uint8 = float(diff.mean()) if diff.size else None
    p99_abs_uint8 = float(np.percentile(diff, 99)) if diff.size else None
    max_abs_uint8 = int(diff.max()) if diff.size else None
    nonzero_fraction = float((diff > 0).mean()) if diff.size else None
    numerically_exact = bool(
        diff.size
        and (mean_abs_uint8 is not None and mean_abs_uint8 <= 1e-4)
        and (p99_abs_uint8 is not None and p99_abs_uint8 <= 0.0)
        and (max_abs_uint8 is not None and max_abs_uint8 <= 1)
        and (nonzero_fraction is not None and nonzero_fraction <= 1e-6)
    )
    return {
        "reference_count": len(ref_names),
        "output_count": len(out_names),
        "common_count": len(common),
        "hash_match_count": len(common) - len(mismatches),
        "hash_mismatch_count": len(mismatches),
        "hash_mismatches": mismatches[:20],
        "mean_abs_uint8": mean_abs_uint8,
        "p99_abs_uint8": p99_abs_uint8,
        "max_abs_uint8": max_abs_uint8,
        "nonzero_fraction": nonzero_fraction,
        "numerically_exact": numerically_exact,
    }


def run_scene(args: argparse.Namespace) -> dict[str, Any]:
    scene = args.scene
    model = Path(args.package_root) / scene / "detached_model"
    if not model.is_dir():
        raise FileNotFoundError(model)
    endpoint_dir = model / "point_cloud" / f"iteration_{int(args.iteration)}" / "render_residual_endpoint" / args.endpoint_method
    v101_bank = endpoint_dir / "v101_evidence_bank.pt"
    if not v101_bank.is_file():
        raise FileNotFoundError(v101_bank)
    report_root = Path(args.report_root) / scene
    report_root.mkdir(parents=True, exist_ok=True)
    bank_root = Path(args.v102_bank_root) / scene
    bank_root.mkdir(parents=True, exist_ok=True)
    v102_bank = bank_root / "v102_preprojected_delta_bank.pt"

    build_method = args.build_method or f"ours_{int(args.iteration)}_v102_delta_build_{scene}"
    fast_method = args.fast_method or f"ours_{int(args.iteration)}_v102_delta_fast_{scene}"
    reference_method = args.reference_method or f"ours_{int(args.iteration)}_v101_detached_package_full9_{scene}"

    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    build_log = report_root / f"{scene}_build.log"
    fast_log = report_root / f"{scene}_fast.log"
    eval_log = report_root / f"{scene}_eval.log"
    for log_path in (build_log, fast_log, eval_log):
        if log_path.exists() and args.force:
            log_path.unlink()

    build_rc = 0
    build_wall = 0.0
    if args.force or not v102_bank.is_file():
        build_rc, build_wall = _run_cmd(
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
                build_method,
                "--checkpoint_endpoint_base_model",
                "/__spcarnet_v102_preprojected_build_must_not_read_train_evidence__",
                "--checkpoint_endpoint_bank_path",
                str(v101_bank),
                "--checkpoint_endpoint_require_bank",
                "--checkpoint_endpoint_write_preprojected_bank",
                str(v102_bank),
                "--checkpoint_endpoint_preprojected_delta_dtype",
                args.delta_dtype,
                "--quiet",
            ],
            build_log,
            env,
        )

    fast_rc = 999
    fast_wall = 0.0
    if build_rc == 0:
        cmd = [
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
            fast_method,
            "--checkpoint_endpoint_preprojected_bank_path",
            str(v102_bank),
            "--checkpoint_endpoint_require_preprojected_bank",
            "--quiet",
        ]
        if args.no_intermediate_outputs:
            cmd.append("--checkpoint_endpoint_no_intermediate_outputs")
        fast_rc, fast_wall = _run_cmd(cmd, fast_log, env)

    eval_rc = 999
    if fast_rc == 0:
        eval_rc, _ = _run_cmd(
            [
                sys.executable,
                "scripts/car_model/evaluate_render_split_metrics.py",
                "-m",
                str(model),
                "--split",
                "test",
                "--methods",
                fast_method,
                "--merge_model_results",
            ],
            eval_log,
            env,
        )

    results = _read_json(model / "results.json")
    fast_metrics = results.get(fast_method, {}) if isinstance(results.get(fast_method), dict) else {}
    reference_metrics = results.get(reference_method, {}) if isinstance(results.get(reference_method), dict) else {}
    render_report = _read_json(model / "test" / fast_method / "render_py_endpoint_report.json")
    bank_manifest = _read_json(v102_bank.with_suffix(".manifest.json"))
    hash_report = _compare_render_hashes(
        model / "test" / reference_method / "renders",
        model / "test" / fast_method / "renders",
    )
    sec_per_view = fast_wall / hash_report["output_count"] if hash_report["output_count"] else None
    payload = {
        "schema_version": 1,
        "scene": scene,
        "model": str(model),
        "endpoint_method": args.endpoint_method,
        "reference_method": reference_method,
        "build_method": build_method,
        "fast_method": fast_method,
        "v101_bank": str(v101_bank),
        "v102_bank": str(v102_bank),
        "v102_bank_bytes": int(v102_bank.stat().st_size) if v102_bank.is_file() else 0,
        "v102_bank_manifest": bank_manifest,
        "delta_dtype": args.delta_dtype,
        "no_intermediate_outputs": bool(args.no_intermediate_outputs),
        "build_rc": int(build_rc),
        "fast_rc": int(fast_rc),
        "eval_rc": int(eval_rc),
        "build_wall_sec": float(build_wall),
        "fast_wall_sec": float(fast_wall),
        "fast_sec_per_view": sec_per_view,
        "fast_internal_elapsed_sec": render_report.get("elapsed_sec"),
        "fast_internal_sec_per_view": (
            float(render_report["elapsed_sec"]) / int(render_report["target_frames"])
            if render_report.get("elapsed_sec") is not None and render_report.get("target_frames")
            else None
        ),
        "support_source": render_report.get("support_source"),
        "mode": render_report.get("mode"),
        "intermediate_outputs_saved": render_report.get("intermediate_outputs_saved"),
        "mean_abs_delta": render_report.get("mean_abs_delta"),
        "mean_changed_fraction": render_report.get("mean_changed_fraction"),
        "metrics": fast_metrics,
        "reference_metrics": reference_metrics,
        "dPSNR_reference": (
            float(fast_metrics["PSNR"]) - float(reference_metrics["PSNR"])
            if "PSNR" in fast_metrics and "PSNR" in reference_metrics
            else None
        ),
        "dSSIM_reference": (
            float(fast_metrics["SSIM"]) - float(reference_metrics["SSIM"])
            if "SSIM" in fast_metrics and "SSIM" in reference_metrics
            else None
        ),
        "dLPIPS_reference": (
            float(fast_metrics["LPIPS"]) - float(reference_metrics["LPIPS"])
            if "LPIPS" in fast_metrics and "LPIPS" in reference_metrics
            else None
        ),
        "hash_report": hash_report,
        "passed": bool(
            build_rc == 0
            and fast_rc == 0
            and eval_rc == 0
            and str(render_report.get("mode")) == "preprojected_delta_endpoint"
            and str(render_report.get("support_source", "")).startswith("v102_preprojected_delta_bank:")
            and hash_report["reference_count"] > 0
            and (
                hash_report["hash_mismatch_count"] == 0
                or bool(hash_report.get("numerically_exact", False))
            )
            and hash_report["common_count"] == hash_report["reference_count"]
        ),
        "logs": {
            "build": str(build_log),
            "fast": str(fast_log),
            "eval": str(eval_log),
        },
    }
    out_json = report_root / f"{scene}_v102_preprojected_delta_report.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md = report_root / f"{scene}_v102_preprojected_delta_report.md"
    out_md.write_text(
        "\n".join(
            [
                f"# v102 Preprojected Delta Report: {scene}",
                "",
                f"- passed: `{payload['passed']}`",
                f"- fast wall sec/view: `{payload['fast_sec_per_view']}`",
                f"- hash exact: `{hash_report['hash_match_count']}/{hash_report['reference_count']}`",
                f"- metrics: `{fast_metrics}`",
                f"- delta vs reference: `{payload['dPSNR_reference']} / {payload['dSSIM_reference']} / {payload['dLPIPS_reference']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(out_json), "passed": payload["passed"]}, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate a v102 preprojected delta bank for one scene.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--package_root", default="/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625")
    parser.add_argument("--v102_bank_root", default="/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625")
    parser.add_argument("--report_root", default="outputs/carnet/meshsplatopt/ecsr_phase_v102_preprojected_delta_bank_20260625")
    parser.add_argument("--endpoint_method", default=ENDPOINT_METHOD)
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--reference_method", default="")
    parser.add_argument("--build_method", default="")
    parser.add_argument("--fast_method", default="")
    parser.add_argument("--delta_dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--gpu", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_intermediate_outputs", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = run_scene(parse_args())
    return 0 if payload.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
