#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCENE_BASE_ROOT = Path("/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625")
BANK_ROOT = Path("/dev/shm/peilincai_spcarnet_v101_bankfp16_full9_fixed_20260625")
ENDPOINT_METHOD = "ours_26000_v100_checkpoint_attached_ela_endpoint"
REFERENCE_METHOD = "ours_26000_v101_bankfp16_renderpy_endpoint_full9_fixed"
DETACHED_MISSING_ROOT = Path("/__spcarnet_detached_package_must_not_read_train_evidence__")
ENDPOINT_BANK_SIDECARS = {
    "v101_evidence_bank.pt",
    "v101_evidence_bank.manifest.json",
    "v101_evidence_bank_manifest.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_or_link(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst, follow_symlinks=True)
    elif mode == "hardlink":
        try:
            os.link(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst, follow_symlinks=True)
    else:
        raise ValueError(f"unsupported copy mode: {mode}")


def _copy_tree_files(src_dir: Path, dst_dir: Path, *, skip_names: set[str] | None = None) -> None:
    if not src_dir.is_dir():
        return
    skip_names = skip_names or set()
    for src in src_dir.rglob("*"):
        if src.is_file():
            if src.name in skip_names:
                continue
            rel = src.relative_to(src_dir)
            dst = dst_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst, follow_symlinks=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _prepare_package(args: argparse.Namespace) -> dict[str, Any]:
    scene = args.scene
    source_model = Path(args.source_root) / scene / "recovery_model"
    if not source_model.is_dir():
        raise FileNotFoundError(source_model)
    package_model = Path(args.package_root) / scene / "detached_model"
    if package_model.exists() and args.force:
        shutil.rmtree(package_model)
    package_model.mkdir(parents=True, exist_ok=True)

    for name in ("cfg_args", "cameras.json", "input.ply"):
        _copy_or_link(source_model / name, package_model / name, mode="copy")
    for name in ("topology_audit.json", "topology_audit.md"):
        src = source_model / name
        if src.exists() or src.is_symlink():
            _copy_or_link(src, package_model / name, mode="copy")

    iteration_dir = package_model / "point_cloud" / f"iteration_{int(args.iteration)}"
    _copy_or_link(
        source_model / "point_cloud" / f"iteration_{int(args.iteration)}" / "point_cloud_state_dict.pt",
        iteration_dir / "point_cloud_state_dict.pt",
        mode=args.large_file_mode,
    )

    src_endpoint = (
        source_model
        / "point_cloud"
        / f"iteration_{int(args.iteration)}"
        / "render_residual_endpoint"
        / args.endpoint_method
    )
    dst_endpoint = iteration_dir / "render_residual_endpoint" / args.endpoint_method
    src_endpoint_report = src_endpoint / "ela_report.json"
    if not src_endpoint_report.is_file():
        raise FileNotFoundError(src_endpoint_report)
    _copy_tree_files(src_endpoint, dst_endpoint, skip_names=ENDPOINT_BANK_SIDECARS)
    endpoint_report = dst_endpoint / "ela_report.json"
    _copy_or_link(src_endpoint_report, endpoint_report, mode="copy")

    bank_src = Path(args.bank_root) / scene / "v101_evidence_bank.pt"
    if not bank_src.is_file():
        raise FileNotFoundError(bank_src)
    bank_dst = dst_endpoint / "v101_evidence_bank.pt"
    for sidecar_name in ENDPOINT_BANK_SIDECARS - {"v101_evidence_bank.pt"}:
        sidecar_path = dst_endpoint / sidecar_name
        if sidecar_path.exists() or sidecar_path.is_symlink():
            sidecar_path.unlink()
    _copy_or_link(bank_src, bank_dst, mode=args.large_file_mode)
    bank_manifest_src = bank_src.with_suffix(".manifest.json")
    if bank_manifest_src.is_file():
        _copy_or_link(bank_manifest_src, dst_endpoint / "v101_evidence_bank.manifest.json", mode="copy")

    return {
        "scene": scene,
        "source_model": str(source_model),
        "package_model": str(package_model),
        "endpoint_dir": str(dst_endpoint),
        "endpoint_report": str(endpoint_report),
        "endpoint_report_sha256": _sha256(endpoint_report),
        "bank_path": str(bank_dst),
        "bank_sha256": _sha256(bank_dst),
        "checkpoint_sha256": _sha256(iteration_dir / "point_cloud_state_dict.pt"),
        "detached_base_model_override": str(DETACHED_MISSING_ROOT),
    }


def _run_cmd(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.write(f"\n[exit_code] {proc.returncode}\n")
    return int(proc.returncode)


def _compare_hashes(package_model: Path, source_model: Path, method: str, reference_method: str) -> dict[str, Any]:
    out_dir = package_model / "test" / method / "renders"
    ref_dir = source_model / "test" / reference_method / "renders"
    names = sorted(path.name for path in out_dir.glob("*.png"))
    ref_names = sorted(path.name for path in ref_dir.glob("*.png"))
    common = sorted(set(names) & set(ref_names))
    mismatches = []
    for name in common:
        if _sha256(out_dir / name) != _sha256(ref_dir / name):
            mismatches.append(name)
    return {
        "render_count": len(names),
        "reference_count": len(ref_names),
        "common_count": len(common),
        "hash_match_count": len(common) - len(mismatches),
        "hash_mismatch_count": len(mismatches),
        "hash_mismatches": mismatches[:20],
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    package = _prepare_package(args)
    package_model = Path(package["package_model"])
    source_model = Path(package["source_model"])
    method = args.method_name or f"ours_{int(args.iteration)}_v101_detached_package_smoke"
    log_path = Path(args.report_root) / f"{args.scene}_detached_package.log"
    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    started = time.time()
    if log_path.exists():
        log_path.unlink()
    render_rc = _run_cmd(
        [
            sys.executable,
            "render.py",
            "-m",
            str(package_model),
            "--iteration",
            str(args.iteration),
            "--skip_train",
            "--checkpoint_endpoint_method",
            args.endpoint_method,
            "--checkpoint_endpoint_output_method",
            method,
            "--checkpoint_endpoint_base_model",
            str(DETACHED_MISSING_ROOT),
            "--checkpoint_endpoint_bank_path",
            str(package["bank_path"]),
            "--checkpoint_endpoint_require_bank",
            "--quiet",
        ],
        log_path,
        env,
    )
    eval_rc = 999
    if render_rc == 0:
        eval_rc = _run_cmd(
            [
                sys.executable,
                "scripts/car_model/evaluate_render_split_metrics.py",
                "-m",
                str(package_model),
                "--split",
                "test",
                "--methods",
                method,
                "--merge_model_results",
            ],
            log_path,
            env,
        )
    package_results = _read_json(package_model / "results.json").get(method, {})
    reference_results = _read_json(source_model / "results.json").get(args.reference_method, {})
    render_report_path = package_model / "test" / method / "render_py_endpoint_report.json"
    render_report = _read_json(render_report_path) if render_rc == 0 else {}
    support_source = str(render_report.get("support_source", "") or "")
    used_required_bank = support_source.startswith("v101_evidence_bank:")
    hash_report = _compare_hashes(package_model, source_model, method, args.reference_method) if eval_rc == 0 else {}
    report = {
        "schema_version": 1,
        "scene": args.scene,
        "method": method,
        "reference_method": args.reference_method,
        "render_rc": int(render_rc),
        "eval_rc": int(eval_rc),
        "elapsed_sec": float(time.time() - started),
        "package": package,
        "results": package_results,
        "reference_results": reference_results,
        "render_report_path": str(render_report_path),
        "render_support_source": support_source,
        "used_required_bank": used_required_bank,
        "hash_report": hash_report,
        "passed": bool(
            render_rc == 0
            and eval_rc == 0
            and used_required_bank
            and hash_report.get("render_count", 0) == hash_report.get("reference_count", -1)
            and hash_report.get("hash_mismatch_count", 1) == 0
        ),
        "claim": (
            "render.py was run with --checkpoint_endpoint_require_bank, an explicit package bank path, "
            "and --checkpoint_endpoint_base_model set to a nonexistent path, so support evidence must come from the package bank."
        ),
        "log_path": str(log_path),
    }
    Path(args.report_root).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.report_root) / f"{args.scene}_detached_package_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_path), "passed": report["passed"]}, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a detached v101 endpoint package with an evidence bank.")
    parser.add_argument("--scene", default="counter")
    parser.add_argument("--source_root", default=str(SCENE_BASE_ROOT))
    parser.add_argument("--bank_root", default=str(BANK_ROOT))
    parser.add_argument("--package_root", default="/dev/shm/peilincai_spcarnet_v101_detached_package_20260625")
    parser.add_argument("--report_root", default="outputs/carnet/meshsplatopt/ecsr_phase_v101_detached_package_20260625")
    parser.add_argument("--endpoint_method", default=ENDPOINT_METHOD)
    parser.add_argument("--reference_method", default=REFERENCE_METHOD)
    parser.add_argument("--method_name", default="")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--large_file_mode", default="hardlink", choices=("hardlink", "copy"))
    parser.add_argument("--gpu", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    report = validate(parse_args())
    return 0 if report.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
