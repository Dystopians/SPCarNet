#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


LIGHTWEIGHT_SUFFIXES = {".json", ".md", ".txt", ".log", ".csv", ".tsv", ".yaml", ".yml"}
SUMMARY_SUFFIXES = {".json", ".md"}
SUMMARY_HINTS = ("summary", "manifest_summary", "runner_summary")
ROOT_EVIDENCE_HINTS = ("report", "audit", "log", "result", "manifest")
MAX_COPY_BYTES = 20 * 1024 * 1024

SCENE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("reports", ("reports",)),
    ("model_audits", ("model_audits", "audits")),
    ("logs", ("logs", "manifest_logs")),
    ("selector", ("selector",)),
    ("results", ("results",)),
)

HEAVY_DIR_NAMES = {
    "checkpoints",
    "checkpoint",
    "ckpts",
    "gt",
    "images",
    "model",
    "models",
    "renders",
    "render",
    "rendered",
    "target_evidence",
    "target_evidence_no_gt",
    "teacher_renders",
    "train",
    "val",
    "videos",
}


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-") or "vnext_full9_cleanup_artifacts"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _as_posix(path: Path) -> str:
    return path.as_posix()


def _rel_or_name(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def _has_heavy_part(parts: Iterable[str]) -> bool:
    return any(part.lower() in HEAVY_DIR_NAMES for part in parts)


def _looks_like_summary(path: Path, summary_prefix: str) -> bool:
    name = path.name.lower()
    prefix = summary_prefix.lower()
    return (
        path.suffix.lower() in SUMMARY_SUFFIXES
        and (name.startswith(prefix) or any(hint in name for hint in SUMMARY_HINTS))
    )


def _looks_like_root_evidence(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in LIGHTWEIGHT_SUFFIXES and any(hint in name for hint in ROOT_EVIDENCE_HINTS)


def _iter_dir_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_file())


def _iter_lightweight_tree(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root)
        if _has_heavy_part(rel.parts[:-1]):
            continue
        if path.suffix.lower() in LIGHTWEIGHT_SUFFIXES:
            files.append(path)
    return sorted(files)


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return -1


def _copy_or_plan(
    *,
    src: Path,
    dst: Path,
    source_root: Path,
    output_dir: Path,
    category: str,
    group: str,
    scene: str,
    dry_run: bool,
    copied: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    planned_destinations: set[Path],
) -> None:
    src = src.resolve()
    dst = dst.resolve()
    size = _file_size(src)
    base_record = {
        "category": category,
        "group": group,
        "scene": scene,
        "source": _as_posix(src),
        "source_relative": _as_posix(_rel_or_name(src, source_root.resolve())),
        "destination": _as_posix(dst),
        "destination_relative": _as_posix(_rel_or_name(dst, output_dir.resolve())),
        "bytes": size,
    }
    if _is_relative_to(src, output_dir):
        skipped.append({**base_record, "reason": "source_inside_output_dir"})
        return
    if src.is_symlink():
        skipped.append({**base_record, "reason": "symlink_skipped"})
        return
    if size < 0:
        skipped.append({**base_record, "reason": "stat_failed"})
        return
    if size > MAX_COPY_BYTES:
        skipped.append({**base_record, "reason": "too_large_for_lightweight_policy"})
        return
    if dst in planned_destinations:
        skipped.append({**base_record, "reason": "duplicate_destination_in_plan"})
        return
    if dst.exists():
        skipped.append({**base_record, "reason": "destination_exists"})
        return

    planned_destinations.add(dst)
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    copied.append({**base_record, "status": "planned" if dry_run else "copied"})


def _summary_sources(run_root: Path, compact_root: Path) -> list[tuple[str, Path, Path]]:
    sources: list[tuple[str, Path, Path]] = []
    for label, root in (("run_root", run_root), ("compact_artifact_root", compact_root)):
        for rel in (Path("."), Path("summary"), Path("summaries"), Path("reports")):
            directory = root / rel
            if directory.is_dir():
                sources.append((label, root, directory))
    return sources


def _copy_summaries(
    *,
    run_root: Path,
    compact_root: Path,
    output_dir: Path,
    summary_prefix: str,
    dry_run: bool,
    copied: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    planned_destinations: set[Path],
) -> int:
    count = 0
    for label, source_root, directory in _summary_sources(run_root, compact_root):
        for src in _iter_dir_files(directory):
            if not _looks_like_summary(src, summary_prefix):
                continue
            rel = _rel_or_name(src, source_root)
            if rel.parts and rel.parts[0] in {"summary", "summaries"}:
                dst = output_dir / rel
            else:
                dst = output_dir / "summary" / rel.name
            before = len(copied)
            _copy_or_plan(
                src=src,
                dst=dst,
                source_root=source_root,
                output_dir=output_dir,
                category="summary",
                group=label,
                scene="",
                dry_run=dry_run,
                copied=copied,
                skipped=skipped,
                planned_destinations=planned_destinations,
            )
            if len(copied) > before:
                count += 1
    return count


def _copy_root_logs(
    *,
    run_root: Path,
    compact_root: Path,
    output_dir: Path,
    dry_run: bool,
    copied: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    planned_destinations: set[Path],
) -> int:
    count = 0
    for label, source_root in (("run_root", run_root), ("compact_artifact_root", compact_root)):
        for rel_dir in (Path("_manifest_logs"), Path("manifest_logs"), Path("logs")):
            source_dir = source_root / rel_dir
            if not source_dir.is_dir():
                continue
            for src in _iter_lightweight_tree(source_dir):
                rel = src.relative_to(source_dir)
                dst = output_dir / "_manifest_logs" / label / rel_dir.name / rel
                before = len(copied)
                _copy_or_plan(
                    src=src,
                    dst=dst,
                    source_root=source_root,
                    output_dir=output_dir,
                    category="root_log",
                    group=f"{label}:{rel_dir.name}",
                    scene="",
                    dry_run=dry_run,
                    copied=copied,
                    skipped=skipped,
                    planned_destinations=planned_destinations,
                )
                if len(copied) > before:
                    count += 1
    return count


def _scene_dirs(compact_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for child in sorted(path for path in compact_root.iterdir() if path.is_dir()):
        if child.name.startswith(".") or child.name in {"reports", "summaries", "manifests"}:
            continue
        has_group = any((child / group_dir).is_dir() for _, group_dirs in SCENE_GROUPS for group_dir in group_dirs)
        has_root_evidence = any(_looks_like_root_evidence(path) for path in _iter_dir_files(child))
        if has_group or has_root_evidence:
            dirs.append(child)
    return dirs


def _copy_scene_artifacts(
    *,
    compact_root: Path,
    output_dir: Path,
    dry_run: bool,
    copied: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    planned_destinations: set[Path],
) -> dict[str, dict[str, int]]:
    scene_counts: dict[str, dict[str, int]] = {}
    for scene_dir in _scene_dirs(compact_root):
        scene = scene_dir.name
        scene_counts[scene] = {canonical_group: 0 for canonical_group, _ in SCENE_GROUPS}
        scene_counts[scene]["root_files"] = 0
        for canonical_group, group_dirs in SCENE_GROUPS:
            for group_dir_name in group_dirs:
                source_dir = scene_dir / group_dir_name
                if not source_dir.is_dir():
                    continue
                for src in _iter_lightweight_tree(source_dir):
                    rel = src.relative_to(source_dir)
                    if group_dir_name == canonical_group:
                        dst = output_dir / scene / canonical_group / rel
                    else:
                        dst = output_dir / scene / canonical_group / group_dir_name / rel
                    before = len(copied)
                    _copy_or_plan(
                        src=src,
                        dst=dst,
                        source_root=compact_root,
                        output_dir=output_dir,
                        category="scene_artifact",
                        group=canonical_group,
                        scene=scene,
                        dry_run=dry_run,
                        copied=copied,
                        skipped=skipped,
                        planned_destinations=planned_destinations,
                    )
                    if len(copied) > before:
                        scene_counts[scene][canonical_group] += 1

        for src in _iter_dir_files(scene_dir):
            if not _looks_like_root_evidence(src):
                continue
            dst = output_dir / scene / "root_files" / src.name
            before = len(copied)
            _copy_or_plan(
                src=src,
                dst=dst,
                source_root=compact_root,
                output_dir=output_dir,
                category="scene_root_artifact",
                group="root_files",
                scene=scene,
                dry_run=dry_run,
                copied=copied,
                skipped=skipped,
                planned_destinations=planned_destinations,
            )
            if len(copied) > before:
                scene_counts[scene]["root_files"] += 1
    return scene_counts


def _available_manifest_path(output_dir: Path, stem: str, suffix: str) -> Path:
    candidate = output_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        candidate = output_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find available manifest path for {stem}{suffix}")


def _markdown_manifest(manifest: dict[str, Any]) -> str:
    copied_rows = manifest.get("copied", [])
    skipped_rows = manifest.get("skipped", [])
    lines = [
        "# vNext Full9 Cleanup Artifact Promotion",
        "",
        f"- created at UTC: `{manifest.get('created_at_utc', '')}`",
        f"- dry run: `{manifest.get('dry_run', False)}`",
        f"- run root: `{manifest['source_roots'].get('run_root', '')}`",
        f"- compact artifact root: `{manifest['source_roots'].get('compact_artifact_root', '')}`",
        f"- output dir: `{manifest.get('output_dir', '')}`",
        f"- copied/planned files: `{len(copied_rows)}`",
        f"- skipped files: `{len(skipped_rows)}`",
        "",
        "## Scene Counts",
        "",
        "| scene | reports | model audits | logs | selector | results | root files |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, counts in sorted((manifest.get("scene_counts") or {}).items()):
        lines.append(
            "| {scene} | {reports} | {model_audits} | {logs} | {selector} | {results} | {root_files} |".format(
                scene=scene,
                reports=counts.get("reports", 0),
                model_audits=counts.get("model_audits", 0),
                logs=counts.get("logs", 0),
                selector=counts.get("selector", 0),
                results=counts.get("results", 0),
                root_files=counts.get("root_files", 0),
            )
        )
    if not manifest.get("scene_counts"):
        lines.append("|  | 0 | 0 | 0 | 0 | 0 | 0 |")
    lines.extend(
        [
            "",
            "## Copied Paths",
            "",
            "| category | scene | group | bytes | source | destination |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for row in copied_rows:
        lines.append(
            "| {category} | {scene} | {group} | {bytes} | `{source}` | `{destination}` |".format(
                category=row.get("category", ""),
                scene=row.get("scene", ""),
                group=row.get("group", ""),
                bytes=row.get("bytes", ""),
                source=row.get("source", ""),
                destination=row.get("destination_relative", row.get("destination", "")),
            )
        )
    if not copied_rows:
        lines.append("|  |  |  | 0 |  |  |")
    if skipped_rows:
        lines.extend(
            [
                "",
                "## Skipped Paths",
                "",
                "| reason | category | scene | group | bytes | source | destination |",
                "|---|---|---|---|---:|---|---|",
            ]
        )
        for row in skipped_rows:
            lines.append(
                "| {reason} | {category} | {scene} | {group} | {bytes} | `{source}` | `{destination}` |".format(
                    reason=row.get("reason", ""),
                    category=row.get("category", ""),
                    scene=row.get("scene", ""),
                    group=row.get("group", ""),
                    bytes=row.get("bytes", ""),
                    source=row.get("source", ""),
                    destination=row.get("destination_relative", row.get("destination", "")),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    compact_root = args.compact_artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    summary_prefix = _safe_name(args.summary_prefix)

    copied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    missing_optional: list[str] = []
    planned_destinations: set[Path] = set()

    summary_count = _copy_summaries(
        run_root=run_root,
        compact_root=compact_root,
        output_dir=output_dir,
        summary_prefix=summary_prefix,
        dry_run=bool(args.dry_run),
        copied=copied,
        skipped=skipped,
        planned_destinations=planned_destinations,
    )
    if summary_count == 0:
        missing_optional.append("summary markdown/json files under run_root or compact_artifact_root")

    root_log_count = _copy_root_logs(
        run_root=run_root,
        compact_root=compact_root,
        output_dir=output_dir,
        dry_run=bool(args.dry_run),
        copied=copied,
        skipped=skipped,
        planned_destinations=planned_destinations,
    )
    if root_log_count == 0:
        missing_optional.append("root manifest/log files under run_root or compact_artifact_root")

    scene_counts = _copy_scene_artifacts(
        compact_root=compact_root,
        output_dir=output_dir,
        dry_run=bool(args.dry_run),
        copied=copied,
        skipped=skipped,
        planned_destinations=planned_destinations,
    )
    if not scene_counts:
        missing_optional.append("per-scene compact artifact directories")

    return {
        "schema": "vnext_full9_cleanup_artifact_promotion_v1",
        "created_at_utc": _utc_timestamp(),
        "dry_run": bool(args.dry_run),
        "summary_prefix": summary_prefix,
        "source_roots": {
            "run_root": _as_posix(run_root),
            "compact_artifact_root": _as_posix(compact_root),
        },
        "output_dir": _as_posix(output_dir),
        "copy_policy": {
            "allowed_suffixes": sorted(LIGHTWEIGHT_SUFFIXES),
            "summary_suffixes": sorted(SUMMARY_SUFFIXES),
            "max_copy_bytes": MAX_COPY_BYTES,
            "scene_groups": {canonical: list(group_dirs) for canonical, group_dirs in SCENE_GROUPS},
            "heavy_dirs_not_traversed": sorted(HEAVY_DIR_NAMES),
            "overwrite_existing_destinations": False,
        },
        "root_log_count": root_log_count,
        "scene_counts": scene_counts,
        "missing_optional": missing_optional,
        "copied": copied,
        "skipped": skipped,
    }


def _write_manifests(manifest: dict[str, Any], output_dir: Path, summary_prefix: str) -> tuple[Path, Path]:
    manifest_stem = f"{summary_prefix}_promotion_manifest"
    json_path = _available_manifest_path(output_dir, manifest_stem, ".json")
    md_path = _available_manifest_path(output_dir, manifest_stem, ".md")
    manifest["manifest_paths"] = {
        "json": _as_posix(json_path),
        "markdown": _as_posix(md_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_manifest(manifest), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Promote lightweight vNext full9 fixed-policy cleanup evidence from run and compact roots "
            "into a docs artifact directory without copying heavy render/model trees."
        )
    )
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--compact_artifact_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--summary_prefix", default="vnext_full9_cleanup_artifacts")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for label in ("run_root", "compact_artifact_root"):
        path = getattr(args, label)
        if not path.is_dir():
            parser.error(f"--{label} must be an existing directory: {path}")

    manifest = _build_manifest(args)
    output_dir = args.output_dir.resolve()
    summary_prefix = str(manifest["summary_prefix"])

    if args.dry_run:
        planned_json = output_dir / f"{summary_prefix}_promotion_manifest.json"
        planned_md = output_dir / f"{summary_prefix}_promotion_manifest.md"
        manifest["manifest_paths"] = {
            "json": _as_posix(planned_json),
            "markdown": _as_posix(planned_md),
        }
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    json_path, md_path = _write_manifests(manifest, output_dir, summary_prefix)
    print(
        json.dumps(
            {
                "copied_file_count": len(manifest.get("copied", [])),
                "skipped_file_count": len(manifest.get("skipped", [])),
                "manifest_json": _as_posix(json_path),
                "manifest_markdown": _as_posix(md_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
