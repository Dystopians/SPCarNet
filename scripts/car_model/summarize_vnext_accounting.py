#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


AUDIT_NAMES = {
    "surface_residual_region_texture_adapter_audit.json",
}
MANIFEST_SUFFIX = "_vnext_certified_residual_texture_manifest.json"
TOPOLOGY_NAME = "topology_audit.json"
COMPACTION_NAME = "compaction_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize vNext residual texture accounting from manifests and audits."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Run roots or artifact directories.")
    parser.add_argument("--json_output", type=Path, default=None, help="Optional machine-readable JSON output.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def fnum(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def get(payload: Any, *keys: str) -> Any:
    cur = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def first(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def rel(path: Path | str | None) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return str(path)


def read_compaction_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    out: dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = row.get("key")
                if key:
                    out[key] = row.get("value")
    except OSError:
        return {}
    return out


def scene_root(path: Path) -> Path:
    for parent in [path.parent, *path.parents]:
        if parent.name in {"reports", "model", "model_audits", "selector"}:
            return parent.parent
    return path.parent if path.is_file() else path


def discover(paths: list[Path]) -> dict[Path, dict[str, Path]]:
    bundles: dict[Path, dict[str, Path]] = {}
    for raw in paths:
        root = raw.expanduser()
        candidates: list[Path] = []
        if root.is_file():
            candidates.append(root)
        elif root.is_dir():
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "renders", "rgb", "images", "gt", "wandb"}]
                candidates.extend(Path(dirpath) / name for name in filenames)
        for path in candidates:
            kind = None
            if path.name in AUDIT_NAMES:
                kind = "adapter_audit"
            elif path.name.endswith(MANIFEST_SUFFIX):
                kind = "manifest"
            elif path.name == TOPOLOGY_NAME:
                kind = "topology"
            elif path.name == COMPACTION_NAME:
                kind = "compaction"
            if kind is None:
                continue
            bundle = bundles.setdefault(scene_root(path), {})
            bundle.setdefault(kind, path)
    return bundles


def step_elapsed(manifest: dict[str, Any], name: str) -> float | None:
    steps = []
    for key in ("steps", "commands"):
        value = manifest.get(key, [])
        if isinstance(value, list):
            steps.extend(value)
    for step in steps:
        if isinstance(step, dict) and step.get("name") == name:
            return fnum(step.get("elapsed_sec"))
    return None


def estimate_texture_bytes(audit: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    faces = fnum(first(get(audit, "fit_summary", "atlas_faces"), get(audit, "fit_summary", "candidate_faces")))
    tex = fnum(first(get(audit, "fit_summary", "selected_texture_size"), get(audit, "fit_summary", "texture_size"), audit.get("selected_texture_size")))
    if faces is None or tex is None:
        return None, None, None
    bins = int(faces) * int(tex) * int(tex)
    rgb_params = bins * 3
    rgb_bytes_float32 = rgb_params * 4
    return bins, rgb_params, rgb_bytes_float32


def build_row(root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    manifest = read_json(paths.get("manifest", Path()))
    audit = read_json(paths.get("adapter_audit", Path()))
    topology = read_json(paths.get("topology", Path()))
    compaction = read_compaction_csv(paths.get("compaction", Path()))
    bins, rgb_params, rgb_bytes = estimate_texture_bytes(audit)
    row = {
        "scene": first(manifest.get("scene"), get(audit, "settings", "scene"), root.name),
        "method": first(manifest.get("method"), get(manifest, "settings", "method_name"), get(audit, "settings", "method_name")),
        "run_root": manifest.get("run_root"),
        "artifact_root": str(root),
        "status": manifest.get("status"),
        "protocol_passed": get(manifest, "protocol_audit", "passed"),
        "accepted": audit.get("accepted"),
        "effective_policy": audit.get("effective_policy"),
        "fallback_written": audit.get("fallback_written"),
        "selected_alpha": first(audit.get("selected_alpha"), get(audit, "fit_summary", "selected_alpha")),
        "selected_texture_size": first(get(audit, "fit_summary", "selected_texture_size"), get(audit, "fit_summary", "texture_size"), audit.get("selected_texture_size")),
        "atlas_faces": first(get(audit, "fit_summary", "atlas_faces"), get(audit, "fit_summary", "candidate_faces")),
        "changed_fraction": first(get(audit, "target_apply", "changed_fraction"), get(audit, "target_coverage_gate", "changed_fraction")),
        "texture_bins": bins,
        "estimated_residual_rgb_params": rgb_params,
        "estimated_residual_rgb_bytes_float32": rgb_bytes,
        "pre_triangles": first(topology.get("pre_triangles"), compaction.get("face_count")),
        "post_triangles": topology.get("post_triangles"),
        "removed_triangles": first(topology.get("removed_triangles"), compaction.get("selected_count")),
        "removed_fraction": first(topology.get("removed_fraction"), compaction.get("selected_fraction")),
        "strip_elapsed_sec": step_elapsed(manifest, "strip_target_evidence_no_gt"),
        "apply_elapsed_sec": step_elapsed(manifest, "apply_certified_residual_texture"),
        "eval_elapsed_sec": step_elapsed(manifest, "evaluate_vnext_target"),
        "manifest_path": rel(paths.get("manifest")),
        "audit_path": rel(paths.get("adapter_audit")),
    }
    return row


def fmt(value: Any, digits: int = 6) -> str:
    number = fnum(value)
    return "" if number is None else f"{number:.{digits}f}"


def bool_cell(value: Any) -> str:
    return "yes" if value is True else "no" if value is False else ""


def md(value: Any) -> str:
    return ("" if value is None else str(value)).replace("\n", " ").replace("|", "\\|")


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| scene | method | status | protocol | accepted | policy | alpha | tex | atlas faces | changed frac | est tex MB | tri reduction | apply sec | eval sec |")
    print("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda item: (str(item.get("scene") or ""), str(item.get("method") or ""), str(item.get("artifact_root") or ""))):
        est_mb = fnum(row.get("estimated_residual_rgb_bytes_float32"))
        tri_reduction = fnum(row.get("removed_fraction"))
        print(
            "| "
            + " | ".join(
                [
                    md(row.get("scene")),
                    md(row.get("method")),
                    md(row.get("status")),
                    bool_cell(row.get("protocol_passed")),
                    bool_cell(row.get("accepted")),
                    md(row.get("effective_policy")),
                    fmt(row.get("selected_alpha")),
                    fmt(row.get("selected_texture_size"), 0),
                    fmt(row.get("atlas_faces"), 0),
                    fmt(row.get("changed_fraction"), 9),
                    "" if est_mb is None else f"{est_mb / (1024 * 1024):.3f}",
                    "" if tri_reduction is None else f"{100.0 * tri_reduction:.3f}%",
                    fmt(row.get("apply_elapsed_sec"), 3),
                    fmt(row.get("eval_elapsed_sec"), 3),
                ]
            )
            + " |"
        )
    if rows:
        accepted = sum(1 for row in rows if row.get("accepted") is True)
        fallback = sum(1 for row in rows if row.get("fallback_written") is True or row.get("accepted") is False)
        protocol = sum(1 for row in rows if row.get("protocol_passed") is True)
        changed_values = [fnum(row.get("changed_fraction")) for row in rows]
        changed_values = [value for value in changed_values if value is not None]
        texture_bytes = [fnum(row.get("estimated_residual_rgb_bytes_float32")) for row in rows]
        texture_bytes = [value for value in texture_bytes if value is not None]
        print()
        print("## Aggregate")
        print()
        print(f"- scenes: `{len(rows)}`")
        print(f"- protocol pass: `{protocol}/{len(rows)}`")
        print(f"- accepted nonzero/policy accepted: `{accepted}/{len(rows)}`")
        print(f"- fallback or rejected: `{fallback}/{len(rows)}`")
        if changed_values:
            print(f"- mean changed fraction: `{sum(changed_values) / len(changed_values):.9f}`")
        if texture_bytes:
            print(f"- estimated residual RGB texture storage: `{sum(texture_bytes) / (1024 * 1024):.3f} MB float32`")


def main() -> int:
    args = parse_args()
    rows = [
        build_row(root, paths)
        for root, paths in discover(args.paths).items()
        if paths.get("manifest") is not None or paths.get("adapter_audit") is not None
    ]
    print_markdown(rows)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
