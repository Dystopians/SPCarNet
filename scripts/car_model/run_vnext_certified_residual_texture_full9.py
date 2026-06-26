#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


DEFAULT_SCENES = "bicycle,bonsai,counter,flowers,garden,kitchen,room,stump,treehill"


def _python() -> str:
    return sys.executable


def _parse_scenes(text: str) -> list[str]:
    scenes = [token.strip() for token in str(text).replace(" ", ",").split(",") if token.strip()]
    if not scenes:
        raise ValueError("at least one scene is required")
    return scenes


def _format_template(template: str, scene: str) -> str:
    if "{scene}" not in str(template):
        raise ValueError(f"template must contain '{{scene}}': {template}")
    return str(template).format(scene=scene)


def _scene_cmd(args: argparse.Namespace, scene: str, unknown: list[str]) -> list[str]:
    cmd = [
        _python(),
        "scripts/car_model/run_vnext_certified_residual_texture_scene.py",
        "--scene",
        scene,
        "--source_model",
        _format_template(args.source_model_template, scene),
        "--fit_evidence_dir",
        _format_template(args.fit_evidence_template, scene),
        "--target_evidence_dir",
        _format_template(args.target_evidence_template, scene),
        "--region_carrier_json",
        _format_template(args.region_carrier_template, scene),
        "--teacher_render_dir",
        _format_template(args.teacher_render_template, scene),
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
    if args.parent_render_template:
        cmd.extend(["--parent_render_dir", _format_template(args.parent_render_template, scene)])
    if args.gpu != "":
        cmd.extend(["--gpu", str(args.gpu)])
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
                f"{args.wandb_name_prefix}{scene}" if args.wandb_name_prefix else f"vnext-{scene}",
            ]
        )
    cmd.extend(unknown)
    return cmd


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# vNext Certified Residual Texture Full9 Runner",
        "",
        f"- status: `{payload.get('status')}`",
        f"- output root: `{payload.get('output_root')}`",
        f"- scenes: `{', '.join(payload.get('scenes', []))}`",
        f"- dry run: `{payload.get('dry_run')}`",
        f"- failed scenes: `{payload.get('failed_scene_count')}`",
        "",
        "| scene | returncode | elapsed sec | log |",
        "|---|---:|---:|---|",
    ]
    for row in payload.get("rows", []):
        lines.append(
            "| {scene} | {returncode} | {elapsed:.3f} | `{log}` |".format(
                scene=row.get("scene", ""),
                returncode=row.get("returncode"),
                elapsed=float(row.get("elapsed_sec") or 0.0),
                log=row.get("log_path", ""),
            )
        )
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
        description=(
            "Run vNext certified residual surface texture over multiple scenes. "
            "Path arguments are templates and must contain {scene}."
        )
    )
    parser.add_argument("--scenes", default=DEFAULT_SCENES)
    parser.add_argument("--source_model_template", required=True)
    parser.add_argument("--fit_evidence_template", required=True)
    parser.add_argument("--target_evidence_template", required=True)
    parser.add_argument("--region_carrier_template", required=True)
    parser.add_argument("--teacher_render_template", required=True)
    parser.add_argument("--parent_render_template", default="")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--target_split", choices=("train", "test"), default="test")
    parser.add_argument("--base_method_name", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument("--method_name", default="ours_26000_vnext_certified_residual_surface_texture")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--wandb_mode", default="offline")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet_meshprior")
    parser.add_argument("--wandb_group", default="vnext_certified_residual_texture_full9")
    parser.add_argument("--wandb_name_prefix", default="vnext-")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--summary_md", type=Path, default=None)
    return parser.parse_known_args()


def main() -> int:
    args, unknown = parse_args()
    scenes = _parse_scenes(args.scenes)
    output_root = Path(args.output_root)
    logs_dir = output_root / "_full9_logs"
    reports_dir = output_root / "summary"
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json) if args.summary_json else reports_dir / "vnext_full9_runner_summary.json"
    summary_md = Path(args.summary_md) if args.summary_md else reports_dir / "vnext_full9_runner_summary.md"

    rows: list[dict[str, Any]] = []
    failed = 0
    for scene in scenes:
        cmd = _scene_cmd(args, scene, unknown)
        log_path = logs_dir / f"{scene}_vnext_scene.log"
        start = time.time()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(str(x) for x in cmd) + "\n\n")
            handle.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        row = {
            "scene": scene,
            "cmd": cmd,
            "cmd_string": " ".join(str(x) for x in cmd),
            "returncode": int(proc.returncode),
            "elapsed_sec": float(time.time() - start),
            "log_path": str(log_path),
        }
        rows.append(row)
        if proc.returncode != 0:
            failed += 1
            if not bool(args.continue_on_error):
                break

    assembly_json = reports_dir / "vnext_certified_residual_texture_summary.json"
    assembly_md = reports_dir / "vnext_certified_residual_texture_summary.md"
    assembly_log = logs_dir / "assemble_vnext_summary.log"
    assemble_cmd = [
        _python(),
        "scripts/car_model/assemble_vnext_certified_residual_texture_report.py",
        "--run_root",
        str(output_root),
        "--output_json",
        str(assembly_json),
        "--output_md",
        str(assembly_md),
        "--method_name",
        str(args.method_name),
    ]
    assembly: dict[str, Any] | None = None
    if rows:
        with assembly_log.open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(assemble_cmd) + "\n\n")
            handle.flush()
            proc = subprocess.run(
                assemble_cmd,
                cwd=str(ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        assembly = {
            "cmd": assemble_cmd,
            "returncode": int(proc.returncode),
            "output_json": str(assembly_json),
            "output_md": str(assembly_md),
            "log_path": str(assembly_log),
        }

    payload = {
        "schema_version": 1,
        "status": "FAILED" if failed else ("DRY_RUN" if args.dry_run else "COMPLETE"),
        "output_root": str(output_root),
        "scenes": scenes,
        "dry_run": bool(args.dry_run),
        "failed_scene_count": int(failed),
        "rows": rows,
        "assembly": assembly,
        "unknown_passthrough_args": unknown,
    }
    _write_json(summary_json, payload)
    _write_md(summary_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
