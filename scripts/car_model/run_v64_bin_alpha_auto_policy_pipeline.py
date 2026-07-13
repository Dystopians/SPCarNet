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
DEFAULT_SELECTED_ROOT = DEFAULT_ROOT / "v64_bin_alpha_auto_policy_selected_full9"
DEFAULT_SUMMARY_JSON = DEFAULT_ROOT / "v64_bin_alpha_auto_policy_full9_summary.json"
DEFAULT_SUMMARY_MD = DEFAULT_ROOT / "v64_bin_alpha_auto_policy_full9_summary.md"
DEFAULT_GALLERY = DEFAULT_SELECTED_ROOT / "qualitative_gallery.html"
DEFAULT_PIPELINE_MANIFEST = DEFAULT_SELECTED_ROOT / "v64_bin_alpha_auto_policy_pipeline_manifest.json"
DEFAULT_PIPELINE_REPORT = DEFAULT_SELECTED_ROOT / "v64_bin_alpha_auto_policy_pipeline_report.md"
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


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)


def validate_selected_tree(summary: dict[str, Any], selected_root: Path) -> dict[str, Any]:
    rows = summary.get("rows", []) or []
    scenes = [str(row.get("scene", "")) for row in rows]
    if scenes != list(DEFAULT_SCENES):
        raise RuntimeError(f"unexpected v64 scene order/list: {scenes}")
    if summary.get("selection_uses_heldout_metrics") is not False:
        raise RuntimeError("v64 summary must declare selection_uses_heldout_metrics=false")
    manifest_path = selected_root / "manifest.json"
    require_file(manifest_path)
    manifest = read_json(manifest_path)
    if int(manifest.get("scene_count", -1)) != len(DEFAULT_SCENES):
        raise RuntimeError(f"selected tree scene_count mismatch: {manifest_path}")
    if int(manifest.get("render_linked_scene_count", -1)) != len(DEFAULT_SCENES):
        raise RuntimeError(f"selected tree render/GT link count mismatch: {manifest_path}")
    per_scene: list[dict[str, Any]] = []
    for scene in DEFAULT_SCENES:
        scene_root = selected_root / scene
        require_dir(scene_root)
        require_file(scene_root / "selection_manifest.json")
        require_file(scene_root / "results.json")
        require_dir(scene_root / "renders")
        require_dir(scene_root / "gt")
        selection = read_json(scene_root / "selection_manifest.json")
        if selection.get("selection_uses_heldout_metrics") is not False:
            raise RuntimeError(f"{scene} selection manifest must declare selection_uses_heldout_metrics=false")
        per_scene.append(
            {
                "scene": scene,
                "selected_source": selection.get("selected_source", ""),
                "scene_root": str(scene_root),
                "render_dir": str((scene_root / "renders").resolve()),
                "gt_dir": str((scene_root / "gt").resolve()),
                "guard_passed": bool(selection.get("guard_passed", False)),
            }
        )
    return {
        "scene_count": len(rows),
        "candidate_complete_scene_count": int(summary.get("candidate_complete_scene_count", 0)),
        "selected_manifest": manifest,
        "selection_uses_heldout_metrics": summary.get("selection_uses_heldout_metrics"),
        "summary": summary.get("summary", {}),
        "per_scene": per_scene,
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    validation = manifest["validation"]
    summary = validation["summary"]
    if int(validation.get("candidate_complete_scene_count", 0)) == len(DEFAULT_SCENES):
        boundary = (
            "The v63b candidate set is complete for full9 and the selected tree is materialized. "
            "v64 remains a report-only fixed-policy candidate because the guard was designed "
            "after the initial counter/kitchen probes and needs fresh blind/long-run validation "
            "before paper-level promotion."
        )
    else:
        boundary = (
            "v64 is a fixed auto-policy artifact, not a final paper endpoint yet. Candidate "
            "completion is below 9 scenes, so the policy summary is partial."
        )
    lines = [
        "# v64 Bin-Alpha Auto Policy Pipeline Report",
        "",
        f"Date: `{manifest['date']}`",
        "",
        f"Status: `{manifest['status']}`.",
        "",
        "This pipeline refreshes the fixed v64 auto-policy summary, materialized selected",
        "small-artifact tree, optional selected-render HTML gallery, and validation manifest.",
        "Selection uses train/policy-val audit fields only and falls back to v56 when v63b",
        "bin-alpha evidence is not strong enough.",
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
        f"- pipeline manifest: `{manifest['outputs']['pipeline_manifest']}`",
        "",
        "## Validation",
        "",
        f"- scene count: `{validation['scene_count']}`",
        f"- v63b candidate complete scenes: `{validation['candidate_complete_scene_count']}`",
        f"- render/GT linked scenes: `{validation['selected_manifest']['render_linked_scene_count']}`",
        f"- selection uses held-out metrics: `{validation['selection_uses_heldout_metrics']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("v56", "v52", "no-op", "v48", "v50"):
        stats = summary[label]
        lines.append(
            f"| v64 vs {label} | {stats['scene_count']} | {stats['strict_wins']} | "
            f"{stats['nonregressive_or_tie']} | {stats['mean_dPSNR']:+.9f} | "
            f"{stats['mean_dSSIM']:+.9f} | {stats['mean_dLPIPS']:+.9f} |"
        )
    lines.extend(
        [
            "",
            "## Scene Decisions",
            "",
            "| scene | selected source | guard passed | render dir |",
            "|---|---|---:|---|",
        ]
    )
    for row in validation["per_scene"]:
        lines.append(
            f"| {row['scene']} | `{row['selected_source']}` | {int(row['guard_passed'])} | `{row['render_dir']}` |"
        )
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
            boundary,
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v64 bin-alpha fixed auto-policy artifact pipeline.")
    parser.add_argument("--selected_root", type=Path, default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary_md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--gallery_html", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--gallery_limit", type=int, default=3)
    parser.add_argument("--pipeline_manifest", type=Path, default=DEFAULT_PIPELINE_MANIFEST)
    parser.add_argument("--pipeline_report", type=Path, default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--skip_gallery", action="store_true")
    parser.add_argument("--max_copy_bytes", type=int, default=5_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    python = sys.executable
    commands: list[dict[str, Any]] = []
    summarize_cmd = [
        python,
        "scripts/car_model/summarize_v64_bin_alpha_auto_policy.py",
        "--output_json",
        str(args.summary_json),
        "--output_md",
        str(args.summary_md),
        "--materialize_root",
        str(args.selected_root),
        "--max_copy_bytes",
        str(int(args.max_copy_bytes)),
    ]
    record = run_command(summarize_cmd, repo)
    record["name"] = "summarize_and_materialize_v64"
    commands.append(record)
    summary = read_json(args.summary_json)

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

    validation = validate_selected_tree(summary, args.selected_root)
    if not args.skip_gallery:
        require_file(args.gallery_html)
    manifest = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "method": "v64 bin-alpha auto policy artifact pipeline",
        "status": "ARTIFACT_PIPELINE_COMPLETE_REPORT_ONLY_CANDIDATE",
        "self_command": [python, "scripts/car_model/run_v64_bin_alpha_auto_policy_pipeline.py", *sys.argv[1:]],
        "commands": commands,
        "validation": validation,
        "outputs": {
            "summary_json": str(args.summary_json),
            "summary_md": str(args.summary_md),
            "selected_root": str(args.selected_root),
            "selected_manifest": str(args.selected_root / "manifest.json"),
            "gallery_html": "" if args.skip_gallery else str(args.gallery_html),
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
                "selected_root": str(args.selected_root),
                "pipeline_manifest": str(args.pipeline_manifest),
                "pipeline_report": str(args.pipeline_report),
                "candidate_complete_scene_count": validation["candidate_complete_scene_count"],
                "render_linked_scene_count": validation["selected_manifest"]["render_linked_scene_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
