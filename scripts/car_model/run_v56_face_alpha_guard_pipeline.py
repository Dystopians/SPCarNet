#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware")
DEFAULT_SELECTED_ROOT = DEFAULT_ROOT / "v56_face_alpha_guard_selected_full9"
DEFAULT_SUMMARY_JSON = DEFAULT_ROOT / "v56_face_alpha_guard_full9_summary.json"
DEFAULT_SUMMARY_MD = DEFAULT_ROOT / "v56_face_alpha_guard_full9_summary.md"
DEFAULT_V52_SELECTED_ROOT = DEFAULT_ROOT / "v52_capacity_aware_selected_full9"
DEFAULT_GALLERY = DEFAULT_SELECTED_ROOT / "qualitative_gallery.html"
DEFAULT_PIPELINE_MANIFEST = DEFAULT_SELECTED_ROOT / "v56_face_alpha_guard_pipeline_manifest.json"
DEFAULT_PIPELINE_REPORT = DEFAULT_SELECTED_ROOT / "v56_face_alpha_guard_pipeline_report.md"
DEFAULT_COUNTER_PANEL = Path("assets/spcarnet_v56_counter_face_alpha_guard_panel.png")
DEFAULT_COUNTER_PANEL_MANIFEST = Path("assets/spcarnet_v56_counter_face_alpha_guard_panel_manifest.json")
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


def copy_if_small(src: Path, dst: Path, max_bytes: int) -> bool:
    if not src.is_file():
        return False
    if src.stat().st_size > max_bytes:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(target.resolve(), target_is_directory=True)


def dir_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_dir() else None


def render_gt_from_audit(audit_path: Path) -> tuple[Path | None, Path | None, str]:
    if not audit_path.is_file():
        return None, None, "missing_audit"
    audit = read_json(audit_path)
    target = audit.get("target_apply", {}) or {}
    render_dir = dir_or_none(target.get("render_dir"))
    gt_dir = dir_or_none(target.get("gt_dir"))
    if render_dir is not None and gt_dir is not None:
        return render_dir, gt_dir, "audit_target_apply"
    source_model = dir_or_none(audit.get("source_model"))
    base_method = str(audit.get("base_method_name", ""))
    if source_model is not None and base_method:
        render_dir = source_model / "test" / base_method / "renders"
        gt_dir = source_model / "test" / base_method / "gt"
        if render_dir.is_dir() and gt_dir.is_dir():
            return render_dir, gt_dir, "source_model_base_method"
    return None, None, "not_found"


def fallback_source_for_v52(v52_selected_root: Path, scene: str) -> tuple[Path, Path | None, Path | None, str]:
    scene_root = v52_selected_root / scene
    require_dir(scene_root)
    manifest_path = scene_root / "selection_manifest.json"
    require_file(manifest_path)
    manifest = read_json(manifest_path)
    render_dir = dir_or_none(manifest.get("render_dir")) or dir_or_none(scene_root / "renders")
    gt_dir = dir_or_none(manifest.get("gt_dir")) or dir_or_none(scene_root / "gt")
    return scene_root, render_dir, gt_dir, "v52_selected_manifest"


def materialize_selected_tree(
    summary: dict[str, Any],
    selected_root: Path,
    v52_selected_root: Path,
    max_copy_bytes: int,
) -> dict[str, Any]:
    selected_root.mkdir(parents=True, exist_ok=True)
    small_names = (
        "results.json",
        "surface_residual_region_texture_adapter_audit.json",
        "surface_residual_region_texture_adapter_audit.md",
        "topology_audit.json",
        "topology_audit.md",
    )
    scene_records: list[dict[str, Any]] = []
    for row in summary["rows"]:
        scene = str(row["scene"])
        scene_root = selected_root / scene
        scene_root.mkdir(parents=True, exist_ok=True)
        selected_source = str(row["selected_source"])
        copied_files: list[str] = []
        if selected_source == "v55d_face_alpha":
            source_dir = Path(str(row["v55d_result_path"])).parent
            audit_path = source_dir / "surface_residual_region_texture_adapter_audit.json"
            render_dir, gt_dir, render_source = render_gt_from_audit(audit_path)
        else:
            source_dir, render_dir, gt_dir, render_source = fallback_source_for_v52(v52_selected_root, scene)
            audit_path = source_dir / "surface_residual_region_texture_adapter_audit.json"
        for name in small_names:
            if copy_if_small(source_dir / name, scene_root / name, max_copy_bytes):
                copied_files.append(name)
        log_path = source_dir / f"apply_metrics_{scene}.log"
        if copy_if_small(log_path, scene_root / log_path.name, max_copy_bytes):
            copied_files.append(log_path.name)
        render_linked = False
        gt_linked = False
        if render_dir is not None:
            replace_symlink(scene_root / "renders", render_dir)
            render_linked = True
        if gt_dir is not None:
            replace_symlink(scene_root / "gt", gt_dir)
            gt_linked = True
        record = {
            "scene": scene,
            "selected_source": selected_source,
            "source_dir": str(source_dir),
            "copied_files": copied_files,
            "render_source": render_source,
            "render_dir": "" if render_dir is None else str(render_dir),
            "gt_dir": "" if gt_dir is None else str(gt_dir),
            "render_symlink": str(scene_root / "renders") if render_linked else "",
            "gt_symlink": str(scene_root / "gt") if gt_linked else "",
            "render_linked": render_linked,
            "gt_linked": gt_linked,
            "metrics": row["selected_metrics"],
            "v52_metrics": row["v52_metrics"],
            "guard_passed": bool(row["guard_passed"]),
            "guard_reject_reasons": row["guard_reject_reasons"],
            "selection_uses_heldout_metrics": False,
            "caveat": summary.get("caveat", ""),
        }
        (scene_root / "selection_manifest.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        scene_records.append(record)
    manifest = {
        "materialized_root": str(selected_root),
        "scene_count": len(scene_records),
        "max_copy_bytes": int(max_copy_bytes),
        "render_linked_scene_count": int(sum(1 for item in scene_records if item["render_linked"] and item["gt_linked"])),
        "selection_uses_heldout_metrics": False,
        "status": "V56_SELECTED_TREE_MATERIALIZED",
        "scenes": scene_records,
    }
    (selected_root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def validate_selected_tree(summary: dict[str, Any], selected_root: Path) -> dict[str, Any]:
    rows = summary.get("rows", []) or []
    scenes = [str(row.get("scene", "")) for row in rows]
    if scenes != list(DEFAULT_SCENES):
        raise RuntimeError(f"unexpected v56 scene order/list: {scenes}")
    if summary.get("selection_uses_heldout_metrics") is not False:
        raise RuntimeError("v56 summary must declare selection_uses_heldout_metrics=false")
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
        "selected_manifest": manifest,
        "selection_uses_heldout_metrics": summary.get("selection_uses_heldout_metrics"),
        "summary": summary.get("summary", {}),
        "per_scene": per_scene,
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    validation = manifest["validation"]
    summary = validation["summary"]
    lines = [
        "# v56 Face-Alpha Guard Pipeline Report",
        "",
        f"Date: `{manifest['date']}`",
        "",
        "Status: `ARTIFACT_PIPELINE_COMPLETE_REPORT_ONLY_CANDIDATE`.",
        "",
        "This one-command pipeline refreshes the v56 summary, selected small-artifact tree,",
        "selected-render HTML gallery, and counter face-alpha qualitative panel from existing",
        "v52/v55d materialized outputs. It does not rerun GPU render/eval jobs from source configs.",
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
        f"- counter panel: `{manifest['outputs']['counter_panel_png']}`",
        f"- counter panel manifest: `{manifest['outputs']['counter_panel_manifest']}`",
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
    for label in ("v52", "no-op", "v48", "v50"):
        stats = summary[label]
        lines.append(
            f"| v56 vs {label} | {stats['scene_count']} | {stats['strict_wins']} | "
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
            "v56 is safer than raw v55d but remains a report-only candidate. The guard was",
            "designed after inspecting v55d cap-hit held-out results, so the next closure step",
            "is a fresh source-config rerun or blind validation split with W&B logging.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v56 face-alpha guard artifact pipeline.")
    parser.add_argument("--selected_root", type=Path, default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary_md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--v52_selected_root", type=Path, default=DEFAULT_V52_SELECTED_ROOT)
    parser.add_argument("--gallery_html", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--gallery_limit", type=int, default=3)
    parser.add_argument("--counter_panel_png", type=Path, default=DEFAULT_COUNTER_PANEL)
    parser.add_argument("--counter_panel_manifest", type=Path, default=DEFAULT_COUNTER_PANEL_MANIFEST)
    parser.add_argument("--pipeline_manifest", type=Path, default=DEFAULT_PIPELINE_MANIFEST)
    parser.add_argument("--pipeline_report", type=Path, default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--max_copy_bytes", type=int, default=5_000_000)
    parser.add_argument("--skip_gallery", action="store_true")
    parser.add_argument("--skip_panel", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    python = sys.executable
    commands: list[dict[str, Any]] = []
    summarize_cmd = [
        python,
        "scripts/car_model/summarize_v56_face_alpha_guard_policy.py",
        "--output_json",
        str(args.summary_json),
        "--output_md",
        str(args.summary_md),
    ]
    record = run_command(summarize_cmd, repo)
    record["name"] = "summarize_v56_guard"
    commands.append(record)
    summary = read_json(args.summary_json)
    selected_manifest = materialize_selected_tree(summary, args.selected_root, args.v52_selected_root, args.max_copy_bytes)

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
            "scripts/car_model/build_v56_counter_face_alpha_panel.py",
            "--output",
            str(args.counter_panel_png),
            "--manifest",
            str(args.counter_panel_manifest),
        ]
        record = run_command(panel_cmd, repo)
        record["name"] = "build_counter_panel"
        commands.append(record)

    validation = validate_selected_tree(summary, args.selected_root)
    if not args.skip_gallery:
        require_file(args.gallery_html)
    if not args.skip_panel:
        require_file(args.counter_panel_png)
        require_file(args.counter_panel_manifest)
    manifest = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "method": "v56 face-alpha guard artifact pipeline",
        "status": "ARTIFACT_PIPELINE_COMPLETE_REPORT_ONLY_CANDIDATE",
        "self_command": [python, "scripts/car_model/run_v56_face_alpha_guard_pipeline.py", *sys.argv[1:]],
        "commands": commands,
        "selected_manifest": selected_manifest,
        "validation": validation,
        "outputs": {
            "summary_json": str(args.summary_json),
            "summary_md": str(args.summary_md),
            "selected_root": str(args.selected_root),
            "selected_manifest": str(args.selected_root / "manifest.json"),
            "gallery_html": "" if args.skip_gallery else str(args.gallery_html),
            "counter_panel_png": "" if args.skip_panel else str(args.counter_panel_png),
            "counter_panel_manifest": "" if args.skip_panel else str(args.counter_panel_manifest),
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
                "render_linked_scene_count": selected_manifest["render_linked_scene_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
