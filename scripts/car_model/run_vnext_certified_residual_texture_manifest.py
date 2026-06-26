#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


REQUIRED_KEYS = ("scene", "source_model", "fit_evidence_dir", "target_evidence_dir", "region_carrier_json")


def _python() -> str:
    return sys.executable


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scene_rows(config: Any) -> list[dict[str, Any]]:
    if isinstance(config, dict) and isinstance(config.get("scenes"), list):
        rows = config["scenes"]
    elif isinstance(config, list):
        rows = config
    elif isinstance(config, dict):
        rows = []
        for scene, payload in sorted(config.items()):
            if not isinstance(payload, dict):
                raise ValueError(f"scene config for {scene!r} must be an object")
            rows.append({"scene": scene, **payload})
    else:
        raise ValueError("scene config must be a list, a {scenes: [...]} object, or a scene mapping")
    parsed: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"scene row {idx} must be an object")
        missing = [key for key in REQUIRED_KEYS if not row.get(key)]
        if missing:
            raise ValueError(f"scene row {idx} is missing required keys: {missing}")
        parsed.append(dict(row))
    return parsed


def _path_exists(row: dict[str, Any], key: str) -> bool:
    return Path(str(row.get(key, ""))).exists()


def _preflight_row(row: dict[str, Any]) -> dict[str, Any]:
    checks = {key: _path_exists(row, key) for key in REQUIRED_KEYS if key != "scene"}
    return {
        "scene": str(row["scene"]),
        "gpu": str(row.get("gpu", "")),
        "input_exists": checks,
        "ready": all(checks.values()),
        "source_model": str(row["source_model"]),
        "fit_evidence_dir": str(row["fit_evidence_dir"]),
        "target_evidence_dir": str(row["target_evidence_dir"]),
        "region_carrier_json": str(row["region_carrier_json"]),
    }


def _manifest_path(output_root: Path, scene: str) -> Path:
    return output_root / scene / "reports" / f"{scene}_vnext_certified_residual_texture_manifest.json"


def _existing_complete(output_root: Path, scene: str) -> bool:
    path = _manifest_path(output_root, scene)
    if not path.is_file():
        return False
    try:
        return _read_json(path).get("status") == "COMPLETE"
    except Exception:
        return False


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_dir_files(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.is_dir():
        return 0
    copied = 0
    for src in sorted(path for path in src_dir.rglob("*") if path.is_file()):
        rel = src.relative_to(src_dir)
        if _copy_file(src, dst_dir / rel):
            copied += 1
    return copied


def _compact_scene_artifacts(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    scene_root = Path(args.output_root) / scene
    compact_root = Path(args.compact_artifact_root) / scene
    copied: dict[str, Any] = {
        "scene": scene,
        "scene_root": str(scene_root),
        "compact_scene_root": str(compact_root),
        "copied_file_count": 0,
        "copied_groups": {},
    }

    for name in ("reports", "logs"):
        count = _copy_dir_files(scene_root / name, compact_root / name)
        copied["copied_file_count"] += count
        copied["copied_groups"][name] = count

    model_audits = [
        "surface_residual_region_texture_adapter_audit.json",
        "surface_residual_region_texture_adapter_audit.md",
        "topology_audit.json",
        "topology_audit.md",
        "trainval_gate_results.json",
        "trainval_gate_per_view.json",
    ]
    model_count = 0
    for name in model_audits:
        if _copy_file(scene_root / "model" / name, compact_root / "model_audits" / name):
            model_count += 1
    copied["copied_file_count"] += model_count
    copied["copied_groups"]["model_audits"] = model_count

    selector_count = _copy_dir_files(scene_root / "model" / "selector", compact_root / "selector")
    copied["copied_file_count"] += selector_count
    copied["copied_groups"]["selector"] = selector_count

    target_count = 0
    if _copy_file(
        scene_root / "target_evidence_no_gt" / "target_evidence_no_gt_audit.json",
        compact_root / "model_audits" / "target_evidence_no_gt_audit.json",
    ):
        target_count += 1
    copied["copied_file_count"] += target_count
    copied["copied_groups"]["target_no_gt_audit"] = target_count

    manifest_log = Path(args.output_root) / "_manifest_logs" / f"{scene}_vnext_scene.log"
    manifest_count = 0
    if _copy_file(manifest_log, compact_root / "manifest_logs" / manifest_log.name):
        manifest_count += 1
    copied["copied_file_count"] += manifest_count
    copied["copied_groups"]["manifest_logs"] = manifest_count
    return copied


def _cleanup_scene_outputs(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    scene_root = Path(args.output_root) / scene
    protected = {"reports", "logs"}
    removed: list[str] = []
    skipped: list[str] = []
    if not scene_root.is_dir():
        return {"scene": scene, "scene_root": str(scene_root), "removed": removed, "skipped": skipped}
    for child in sorted(scene_root.iterdir()):
        if child.name in protected:
            skipped.append(str(child))
            continue
        removed.append(str(child))
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
    return {"scene": scene, "scene_root": str(scene_root), "removed": removed, "skipped": skipped}


def _scene_cmd(
    args: argparse.Namespace,
    row: dict[str, Any],
    unknown: list[str],
) -> list[str]:
    scene = str(row["scene"])
    cmd = [
        _python(),
        "scripts/car_model/run_vnext_certified_residual_texture_scene.py",
        "--scene",
        scene,
        "--source_model",
        str(row["source_model"]),
        "--fit_evidence_dir",
        str(row["fit_evidence_dir"]),
        "--target_evidence_dir",
        str(row["target_evidence_dir"]),
        "--region_carrier_json",
        str(row["region_carrier_json"]),
        "--output_root",
        str(args.output_root),
        "--target_split",
        str(args.target_split),
        "--base_method_name",
        str(args.base_method_name),
        "--method_name",
        str(args.method_name),
        "--wandb_mode",
        str(args.wandb_mode),
    ]
    if row.get("teacher_render_dir"):
        cmd.extend(["--teacher_render_dir", str(row["teacher_render_dir"])])
    if row.get("parent_render_dir"):
        cmd.extend(["--parent_render_dir", str(row["parent_render_dir"])])
    gpu = row.get("gpu", args.gpu)
    if str(gpu) != "":
        cmd.extend(["--gpu", str(gpu)])
    if bool(args.dry_run):
        cmd.append("--dry_run")
    if bool(args.skip_eval):
        cmd.append("--skip_eval")
    if bool(args.wandb):
        cmd.extend(
            [
                "--wandb",
                "--wandb_project",
                str(args.wandb_project),
                "--wandb_group",
                str(args.wandb_group),
                "--wandb_name",
                str(row.get("wandb_name") or f"{args.wandb_name_prefix}{scene}"),
            ]
        )
    cmd.extend(unknown)
    return cmd


def _run_scene(
    args: argparse.Namespace,
    row: dict[str, Any],
    unknown: list[str],
) -> dict[str, Any]:
    scene = str(row["scene"])
    log_path = Path(args.output_root) / "_manifest_logs" / f"{scene}_vnext_scene.log"
    preflight = _preflight_row(row)
    if not preflight["ready"]:
        return {
            "scene": scene,
            "status": "MISSING_INPUT",
            "returncode": None,
            "elapsed_sec": 0.0,
            "log_path": "",
            "preflight": preflight,
        }
    if bool(args.skip_existing_complete) and _existing_complete(Path(args.output_root), scene):
        return {
            "scene": scene,
            "status": "SKIPPED_EXISTING_COMPLETE",
            "returncode": 0,
            "elapsed_sec": 0.0,
            "log_path": "",
            "preflight": preflight,
        }

    cmd = _scene_cmd(args, row, unknown)
    env = os.environ.copy()
    if args.wandb_dir:
        env["WANDB_DIR"] = str(args.wandb_dir)
    if args.wandb_mode:
        env["WANDB_MODE"] = str(args.wandb_mode)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(str(token) for token in cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    compact_artifacts = None
    cleanup = None
    scene_succeeded = proc.returncode == 0
    if scene_succeeded and args.compact_artifact_root and not bool(args.dry_run):
        compact_artifacts = _compact_scene_artifacts(args, scene)
    if scene_succeeded and bool(args.cleanup_scene_outputs) and not bool(args.dry_run):
        cleanup = _cleanup_scene_outputs(args, scene)
    return {
        "scene": scene,
        "status": "COMPLETE" if proc.returncode == 0 else "FAILED",
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.time() - start),
        "log_path": str(log_path),
        "cmd": cmd,
        "cmd_string": " ".join(str(token) for token in cmd),
        "preflight": preflight,
        "compact_artifacts": compact_artifacts,
        "cleanup": cleanup,
    }


def _assemble(args: argparse.Namespace) -> dict[str, Any]:
    reports_dir = Path(args.output_root) / "summary"
    logs_dir = Path(args.output_root) / "_manifest_logs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_json = reports_dir / "vnext_manifest_summary.json"
    output_md = reports_dir / "vnext_manifest_summary.md"
    log_path = logs_dir / "assemble_vnext_manifest_summary.log"
    cmd = [
        _python(),
        "scripts/car_model/assemble_vnext_certified_residual_texture_report.py",
        "--run_root",
        str(args.output_root),
        "--output_json",
        str(output_json),
        "--output_md",
        str(output_md),
        "--method_name",
        str(args.method_name),
    ]
    if args.compact_artifact_root:
        cmd.extend(["--compact_artifact_root", str(args.compact_artifact_root)])
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(str(token) for token in cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, text=True)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "output_json": str(output_json),
        "output_md": str(output_md),
        "log_path": str(log_path),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# vNext Manifest Runner Summary",
        "",
        f"- status: `{payload.get('status')}`",
        f"- config: `{payload.get('config')}`",
        f"- output root: `{payload.get('output_root')}`",
        f"- preflight-only: `{payload.get('preflight_only')}`",
        f"- max parallel: `{payload.get('max_parallel')}`",
        f"- ready scenes: `{payload.get('ready_scene_count')}`",
        f"- missing-input scenes: `{payload.get('missing_input_scene_count')}`",
        f"- failed scenes: `{payload.get('failed_scene_count')}`",
        "",
        "| scene | status | ready | returncode | elapsed sec | gpu | log |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("rows", []):
        preflight = row.get("preflight", {}) or {}
        lines.append(
            "| {scene} | {status} | {ready} | {returncode} | {elapsed:.3f} | {gpu} | `{log}` |".format(
                scene=row.get("scene", ""),
                status=row.get("status", ""),
                ready=preflight.get("ready", ""),
                returncode="" if row.get("returncode") is None else row.get("returncode"),
                elapsed=float(row.get("elapsed_sec") or 0.0),
                gpu=preflight.get("gpu", ""),
                log=row.get("log_path", ""),
            )
        )
    if payload.get("missing_inputs"):
        lines.extend(["", "## Missing Inputs", ""])
        for scene, checks in payload["missing_inputs"].items():
            missing = [key for key, ok in checks.items() if not ok]
            lines.append(f"- `{scene}`: {', '.join(missing)}")
    if payload.get("assembly"):
        lines.extend(
            [
                "",
                "## Assembly",
                "",
                f"- returncode: `{payload['assembly'].get('returncode')}`",
                f"- json: `{payload['assembly'].get('output_json')}`",
                f"- md: `{payload['assembly'].get('output_md')}`",
                f"- log: `{payload['assembly'].get('log_path')}`",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run vNext certified residual surface texture from a per-scene manifest."
    )
    parser.add_argument("--scene_config_json", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--target_split", choices=("train", "test"), default="test")
    parser.add_argument("--base_method_name", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument("--method_name", default="ours_26000_vnext_certified_residual_surface_texture")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--max_parallel", type=int, default=1)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_existing_complete", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument("--wandb_dir", type=Path, default=None)
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "spcarnet_meshprior"))
    parser.add_argument("--wandb_group", default="vnext_certified_residual_texture_manifest")
    parser.add_argument("--wandb_name_prefix", default="vnext-manifest-")
    parser.add_argument(
        "--compact_artifact_root",
        type=Path,
        default=None,
        help="Optional root where each completed scene copies compact reports, audits, logs, and summaries.",
    )
    parser.add_argument(
        "--cleanup_scene_outputs",
        action="store_true",
        help=(
            "After optional compact artifact copy, delete bulky per-scene outputs under output_root/scene "
            "while preserving output_root/scene/reports and output_root/scene/logs for final assembly."
        ),
    )
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--summary_md", type=Path, default=None)
    return parser.parse_known_args()


def main() -> int:
    args, unknown = parse_args()
    if int(args.max_parallel) < 1:
        raise SystemExit("--max_parallel must be >= 1")
    if bool(args.cleanup_scene_outputs) and int(args.max_parallel) != 1:
        raise SystemExit("--cleanup_scene_outputs requires --max_parallel 1")
    rows_config = _scene_rows(_read_json(args.scene_config_json))
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    preflight = [_preflight_row(row) for row in rows_config]

    rows: list[dict[str, Any]] = []
    if bool(args.preflight_only):
        rows = [
            {
                "scene": row["scene"],
                "status": "READY" if row["ready"] else "MISSING_INPUT",
                "returncode": 0 if row["ready"] else None,
                "elapsed_sec": 0.0,
                "log_path": "",
                "preflight": row,
            }
            for row in preflight
        ]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.max_parallel)) as executor:
            future_to_scene = {
                executor.submit(_run_scene, args, row, unknown): str(row["scene"]) for row in rows_config
            }
            for future in concurrent.futures.as_completed(future_to_scene):
                result = future.result()
                rows.append(result)
                if (
                    result.get("status") not in {"COMPLETE", "SKIPPED_EXISTING_COMPLETE"}
                    and not bool(args.continue_on_error)
                ):
                    for other in future_to_scene:
                        other.cancel()
                    break
        order = {str(row["scene"]): idx for idx, row in enumerate(rows_config)}
        rows.sort(key=lambda row: order.get(str(row.get("scene")), 10**9))

    missing_inputs = {
        row["scene"]: row["input_exists"]
        for row in preflight
        if not bool(row.get("ready", False))
    }
    failed_rows = [row for row in rows if row.get("status") == "FAILED"]
    missing_rows = [row for row in rows if row.get("status") == "MISSING_INPUT"]
    assembly = None
    if not bool(args.preflight_only):
        assembly = _assemble(args)

    status = "PREFLIGHT" if bool(args.preflight_only) else "COMPLETE"
    if failed_rows or missing_rows:
        status = "FAILED" if failed_rows else "MISSING_INPUT"
    payload = {
        "schema_version": 1,
        "status": status,
        "config": str(args.scene_config_json),
        "output_root": str(args.output_root),
        "preflight_only": bool(args.preflight_only),
        "dry_run": bool(args.dry_run),
        "max_parallel": int(args.max_parallel),
        "ready_scene_count": int(sum(1 for row in preflight if row.get("ready"))),
        "missing_input_scene_count": int(len(missing_inputs)),
        "failed_scene_count": int(len(failed_rows)),
        "rows": rows,
        "missing_inputs": missing_inputs,
        "unknown_passthrough_args": unknown,
        "assembly": assembly,
    }
    reports_dir = Path(args.output_root) / "summary"
    summary_json = Path(args.summary_json) if args.summary_json else reports_dir / "vnext_manifest_runner_summary.json"
    summary_md = Path(args.summary_md) if args.summary_md else reports_dir / "vnext_manifest_runner_summary.md"
    _write_json(summary_json, payload)
    _write_md(summary_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if failed_rows or missing_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
