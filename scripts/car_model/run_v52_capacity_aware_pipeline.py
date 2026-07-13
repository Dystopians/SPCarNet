#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware")
DEFAULT_SELECTED_ROOT = DEFAULT_ROOT / "v52_capacity_aware_selected_full9"
DEFAULT_SUMMARY_JSON = DEFAULT_ROOT / "v52_capacity_aware_v48_v51_full9_summary.json"
DEFAULT_SUMMARY_MD = DEFAULT_ROOT / "v52_capacity_aware_v48_v51_full9_summary.md"
DEFAULT_GALLERY = DEFAULT_SELECTED_ROOT / "qualitative_gallery.html"
DEFAULT_PIPELINE_MANIFEST = DEFAULT_SELECTED_ROOT / "v52_capacity_aware_pipeline_manifest.json"
DEFAULT_PIPELINE_REPORT = DEFAULT_SELECTED_ROOT / "v52_capacity_aware_pipeline_report.md"
DEFAULT_PANEL = Path("assets/spcarnet_v52_capacity_policy_cap_hit_panel.png")
DEFAULT_PANEL_MANIFEST = Path("assets/spcarnet_v52_capacity_policy_cap_hit_panel_manifest.json")
DEFAULT_SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(argv: list[str], cwd: Path) -> dict[str, Any]:
    start = time.time()
    proc = subprocess.run(argv, cwd=str(cwd), text=True, capture_output=True, check=False)
    elapsed = time.time() - start
    record = {
        "argv": argv,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "stdout_tail": proc.stdout[-6000:],
        "stderr_tail": proc.stderr[-6000:],
    }
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2))
    return record


def require_path(path: Path, kind: str) -> None:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"missing required directory: {path}")


def validate_summary(summary_json: Path, selected_root: Path, expected_scenes: tuple[str, ...]) -> dict[str, Any]:
    require_path(summary_json, "file")
    payload = read_json(summary_json)
    rows = payload.get("rows", []) or []
    scenes = [str(row.get("scene", "")) for row in rows]
    if scenes != list(expected_scenes):
        raise RuntimeError(f"unexpected v52 scene order/list: {scenes}")
    if payload.get("selection_uses_heldout_metrics") is not False:
        raise RuntimeError("v52 summary must declare selection_uses_heldout_metrics=false")
    selected_sources = {str(row["scene"]): str(row.get("selected_source", "")) for row in rows}
    summary = payload.get("summary", {}) or {}
    manifest_path = selected_root / "manifest.json"
    require_path(manifest_path, "file")
    selected_manifest = read_json(manifest_path)
    if int(selected_manifest.get("scene_count", -1)) != len(expected_scenes):
        raise RuntimeError(f"selected tree scene_count is not {len(expected_scenes)}: {manifest_path}")
    render_linked = int(selected_manifest.get("render_linked_scene_count", -1))
    if render_linked != len(expected_scenes):
        raise RuntimeError(f"selected tree render_linked_scene_count is not {len(expected_scenes)}: {render_linked}")
    per_scene: list[dict[str, Any]] = []
    for scene in expected_scenes:
        scene_root = selected_root / scene
        require_path(scene_root, "dir")
        require_path(scene_root / "selection_manifest.json", "file")
        require_path(scene_root / "results.json", "file")
        require_path(scene_root / "renders", "dir")
        require_path(scene_root / "gt", "dir")
        selection = read_json(scene_root / "selection_manifest.json")
        if selection.get("selection_uses_heldout_metrics") is not False:
            raise RuntimeError(f"{scene} selection manifest must declare selection_uses_heldout_metrics=false")
        per_scene.append(
            {
                "scene": scene,
                "selected_source": selected_sources[scene],
                "scene_root": str(scene_root),
                "render_dir": str((scene_root / "renders").resolve()),
                "gt_dir": str((scene_root / "gt").resolve()),
                "copied_files": selection.get("copied_files", []),
            }
        )
    return {
        "summary_status": payload.get("status", ""),
        "selection_uses_heldout_metrics": payload.get("selection_uses_heldout_metrics"),
        "scene_count": len(rows),
        "selected_sources": selected_sources,
        "summary": summary,
        "selected_manifest": selected_manifest,
        "per_scene": per_scene,
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    validation = manifest["validation"]
    summary = validation["summary"]
    lines = [
        "# v52 Capacity-Aware Pipeline Report",
        "",
        f"Date: `{manifest['date']}`",
        "",
        "Status: `ARTIFACT_PIPELINE_COMPLETE_NOT_FULL_GPU_RERENDER`.",
        "",
        "This report records a one-command artifact pipeline for the fixed v52 capacity-aware",
        "policy. It refreshes the v52 summary, selected small-artifact tree, qualitative HTML",
        "gallery, cap-hit local panel, and manifest from existing v48/v51 materialized outputs.",
        "It does not rerun the original GPU render/eval jobs from source configs.",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(manifest["self_command"]),
        "```",
        "",
        "## Outputs",
        "",
        f"- summary JSON: `{manifest['outputs']['summary_json']}`",
        f"- summary MD: `{manifest['outputs']['summary_md']}`",
        f"- selected root: `{manifest['outputs']['selected_root']}`",
        f"- selected manifest: `{manifest['outputs']['selected_manifest']}`",
        f"- qualitative gallery: `{manifest['outputs']['gallery_html']}`",
        f"- cap-hit panel: `{manifest['outputs']['panel_png']}`",
        f"- cap-hit panel manifest: `{manifest['outputs']['panel_manifest']}`",
        f"- pipeline manifest: `{manifest['outputs']['pipeline_manifest']}`",
        "",
        "## Validation",
        "",
        f"- scene count: `{validation['scene_count']}`",
        f"- render/GT linked scenes: `{validation['selected_manifest']['render_linked_scene_count']}`",
        f"- selection uses held-out metrics: `{validation['selection_uses_heldout_metrics']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("no-op", "v48", "v50"):
        stats = summary[label]
        lines.append(
            f"| v52 vs {label} | {stats['scene_count']} | {stats['strict_wins']} | "
            f"{stats['nonregressive_or_tie']} | {stats['mean_dPSNR']:+.9f} | "
            f"{stats['mean_dSSIM']:+.9f} | {stats['mean_dLPIPS']:+.9f} |"
        )
    lines.extend(
        [
            "",
            "## Scene Decisions",
            "",
            "| scene | selected source | render dir |",
            "|---|---|---|",
        ]
    )
    for row in validation["per_scene"]:
        lines.append(f"| {row['scene']} | `{row['selected_source']}` | `{row['render_dir']}` |")
    lines.extend(
        [
            "",
            "## Executed Steps",
            "",
            "| step | returncode | elapsed sec |",
            "|---|---:|---:|",
        ]
    )
    for command in manifest["commands"]:
        lines.append(f"| `{command['name']}` | {command['returncode']} | {command['elapsed_sec']:.2f} |")
    lines.extend(
        [
            "",
            "## Honest Boundary",
            "",
            "This is an engineering closure improvement for v52 artifact reproducibility. It is not",
            "the final paper endpoint because v52 still selects among existing v48/v51 outputs and",
            "does not yet launch fresh GPU render/eval jobs from source configs. The next closure",
            "step is a true source-config launcher that can regenerate selected renders and metrics",
            "under W&B-logged medium/full runs.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the v52 capacity-aware artifact pipeline.")
    parser.add_argument("--selected_root", type=Path, default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary_md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--gallery_html", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--gallery_limit", type=int, default=3)
    parser.add_argument("--panel_png", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--panel_manifest", type=Path, default=DEFAULT_PANEL_MANIFEST)
    parser.add_argument("--pipeline_manifest", type=Path, default=DEFAULT_PIPELINE_MANIFEST)
    parser.add_argument("--pipeline_report", type=Path, default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--skip_gallery", action="store_true")
    parser.add_argument("--skip_panel", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    python = sys.executable
    commands: list[dict[str, Any]] = []

    summarize_cmd = [
        python,
        "scripts/car_model/summarize_v52_capacity_aware_policy.py",
        "--output_json",
        str(args.summary_json),
        "--output_md",
        str(args.summary_md),
        "--materialize_root",
        str(args.selected_root),
    ]
    record = run_command(summarize_cmd, repo)
    record["name"] = "summarize_and_materialize"
    commands.append(record)

    if not args.skip_gallery:
        gallery_cmd = [
            python,
            "scripts/car_model/final_build_stageSCE10_qualitative_gallery.py",
            "--output_html",
            str(args.gallery_html),
            "--limit",
            str(int(args.gallery_limit)),
        ]
        for scene in DEFAULT_SCENES:
            gallery_cmd.extend(["--entry", scene, str(args.selected_root / scene / "renders")])
        record = run_command(gallery_cmd, repo)
        record["name"] = "build_selected_gallery"
        commands.append(record)

    if not args.skip_panel:
        panel_cmd = [
            python,
            "scripts/car_model/build_v52_capacity_policy_panels.py",
            "--v52_selected_root",
            str(args.selected_root),
            "--out",
            str(args.panel_png),
            "--manifest",
            str(args.panel_manifest),
        ]
        record = run_command(panel_cmd, repo)
        record["name"] = "build_cap_hit_panel"
        commands.append(record)

    validation = validate_summary(args.summary_json, args.selected_root, DEFAULT_SCENES)
    if not args.skip_gallery:
        require_path(args.gallery_html, "file")
    if not args.skip_panel:
        require_path(args.panel_png, "file")
        require_path(args.panel_manifest, "file")
        panel_manifest = read_json(args.panel_manifest)
        if len(panel_manifest) != 3:
            raise RuntimeError(f"expected 3 cap-hit panel rows, got {len(panel_manifest)}")
    manifest = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "method": "v52 capacity-aware artifact pipeline",
        "status": "ARTIFACT_PIPELINE_COMPLETE_NOT_FULL_GPU_RERENDER",
        "self_command": [python, "scripts/car_model/run_v52_capacity_aware_pipeline.py", *sys.argv[1:]],
        "commands": commands,
        "validation": validation,
        "outputs": {
            "summary_json": str(args.summary_json),
            "summary_md": str(args.summary_md),
            "selected_root": str(args.selected_root),
            "selected_manifest": str(args.selected_root / "manifest.json"),
            "gallery_html": "" if args.skip_gallery else str(args.gallery_html),
            "panel_png": "" if args.skip_panel else str(args.panel_png),
            "panel_manifest": "" if args.skip_panel else str(args.panel_manifest),
            "pipeline_manifest": str(args.pipeline_manifest),
            "pipeline_report": str(args.pipeline_report),
        },
    }
    args.pipeline_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.pipeline_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(args.pipeline_report, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "summary_json": str(args.summary_json),
                "selected_root": str(args.selected_root),
                "gallery_html": "" if args.skip_gallery else str(args.gallery_html),
                "panel_png": "" if args.skip_panel else str(args.panel_png),
                "pipeline_manifest": str(args.pipeline_manifest),
                "pipeline_report": str(args.pipeline_report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

