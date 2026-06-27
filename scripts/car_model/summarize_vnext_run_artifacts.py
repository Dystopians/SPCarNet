#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")
SKIP_DIRS = {".git", "__pycache__", "gt", "images", "media", "point_cloud", "renders", "rgb", "wandb"}
BUNDLE_DIRS = {"reports", "model", "model_audits", "target_evidence_no_gt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a compact Markdown summary of vNext run artifact JSONs."
    )
    parser.add_argument(
        "--reference_compare_json",
        type=Path,
        default=None,
        help=(
            "Optional comparison JSON, e.g. the v106 full9 compare artifact. "
            "When provided, add per-scene deltas versus clean/v106 references."
        ),
    )
    parser.add_argument(
        "--reference_methods",
        nargs="*",
        default=("clean", "v106_podmoe_basepreserve"),
        help="Reference method names to include from --reference_compare_json.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more run roots, scene directories, or matching JSON artifact files.",
    )
    return parser.parse_args()


def read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"
    except OSError as exc:
        return None, f"read_error:{exc}"


def fnum(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return str(path)


def artifact_kind(path: Path) -> str | None:
    name = path.name
    parent = path.parent.name
    if parent == "reports":
        if name.endswith("results.json"):
            return "result"
        if "per_view" in name and name.endswith(".json"):
            return "per_view"
        if "manifest" in name and name.endswith(".json"):
            return "manifest"
    if parent in {"model", "model_audits"} and name == "surface_residual_region_texture_adapter_audit.json":
        return "adapter"
    if parent == "target_evidence_no_gt" and name.endswith("audit.json"):
        return "target_audit"
    if parent == "model_audits" and name.endswith("target_evidence_no_gt_audit.json"):
        return "target_audit"

    # Compact docs copies and early scene dumps keep these files at the scene root.
    if name == "surface_residual_region_texture_adapter_audit.json":
        return "adapter"
    if name.endswith("target_evidence_no_gt_audit.json"):
        return "target_audit"
    if name.endswith("_test_results.json"):
        return "result"
    if "per_view" in name and name.endswith(".json"):
        return "per_view"
    if "manifest" in name and name.endswith("manifest.json"):
        return "manifest"
    return None


def scene_root(path: Path) -> Path:
    return path.parent.parent if path.parent.name in BUNDLE_DIRS else path.parent


def find_artifacts(paths: list[Path]) -> dict[Path, dict[str, list[Path]]]:
    bundles: dict[Path, dict[str, list[Path]]] = {}
    seen: set[Path] = set()
    for raw in paths:
        root = raw.expanduser()
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                candidates.extend(Path(dirpath) / name for name in filenames)
        else:
            continue
        for path in candidates:
            kind = artifact_kind(path)
            if kind is None:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            bundle = bundles.setdefault(
                scene_root(path),
                {"result": [], "per_view": [], "manifest": [], "adapter": [], "target_audit": []},
            )
            bundle[kind].append(path)
    for bundle in bundles.values():
        for paths_for_kind in bundle.values():
            paths_for_kind.sort(key=lambda p: str(p))
    return bundles


def method_from_manifest(payload: dict[str, Any]) -> str:
    return str(first(get(payload, "settings", "method_name"), payload.get("method_name"), payload.get("method")) or "")


def method_from_audit(payload: dict[str, Any]) -> str:
    return str(first(payload.get("method_name"), get(payload, "settings", "method_name")) or "")


def load_manifests(paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        payload, error = read_json(path)
        payload = payload if isinstance(payload, dict) else {}
        out.append(
            {
                "path": path,
                "payload": payload,
                "error": error,
                "method": method_from_manifest(payload),
                "result_name": Path(str(get(payload, "outputs", "results_path") or "")).name,
            }
        )
    return out


def load_audits(paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        payload, error = read_json(path)
        payload = payload if isinstance(payload, dict) else {}
        out.append({"path": path, "payload": payload, "error": error, "method": method_from_audit(payload)})
    return out


def result_metrics(payload: Any, fallback_method: str) -> dict[str, dict[str, float]]:
    if not isinstance(payload, dict):
        return {}
    direct = {metric: fnum(payload.get(metric)) for metric in METRICS}
    if any(value is not None for value in direct.values()):
        return {fallback_method: {key: value for key, value in direct.items() if value is not None}}
    rows = {}
    for method, maybe_metrics in payload.items():
        if not isinstance(maybe_metrics, dict):
            continue
        metrics = {metric: fnum(maybe_metrics.get(metric)) for metric in METRICS}
        metrics = {key: value for key, value in metrics.items() if value is not None}
        if metrics:
            rows[str(method)] = metrics
    return rows


def normalized_metrics(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for source_key, target_key in (
        ("PSNR", "PSNR"),
        ("psnr", "PSNR"),
        ("SSIM", "SSIM"),
        ("ssim", "SSIM"),
        ("LPIPS", "LPIPS"),
        ("lpips", "LPIPS"),
    ):
        value = fnum(payload.get(source_key))
        if value is not None:
            out[target_key] = value
    return out


def load_reference_compare_json(path: Path | None, requested_methods: list[str] | tuple[str, ...]) -> dict[str, dict[str, dict[str, float]]]:
    if path is None:
        return {}
    payload, error = read_json(path)
    if error is not None or not isinstance(payload, dict):
        return {}
    requested = set(requested_methods)
    refs: dict[str, dict[str, dict[str, float]]] = {method: {} for method in requested_methods}
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        scene = str(row.get("scene") or "")
        if not scene:
            continue
        if "clean" in requested:
            clean = normalized_metrics(row.get("clean_metrics"))
            if clean:
                refs.setdefault("clean", {}).setdefault(scene, clean)
        method = str(row.get("method") or "")
        if method in requested:
            metrics = normalized_metrics(row.get("metrics"))
            if metrics:
                refs.setdefault(method, {})[scene] = metrics
    return {method: scene_metrics for method, scene_metrics in refs.items() if scene_metrics}


def apply_reference_deltas(rows: list[dict[str, Any]], refs: dict[str, dict[str, dict[str, float]]]) -> None:
    if not refs:
        return
    for row in rows:
        scene = str(row.get("scene") or "")
        metric_values = normalized_metrics(row)
        delta_by_ref: dict[str, dict[str, float]] = {}
        strict_wins: dict[str, bool] = {}
        nonregressive: dict[str, bool] = {}
        for ref_name, scene_refs in refs.items():
            ref_metrics = scene_refs.get(scene)
            if not ref_metrics:
                continue
            deltas = {}
            for metric in METRICS:
                current = metric_values.get(metric)
                ref_value = ref_metrics.get(metric)
                if current is not None and ref_value is not None:
                    deltas[metric] = current - ref_value
            if deltas:
                delta_by_ref[ref_name] = deltas
                strict_wins[ref_name] = (
                    deltas.get("PSNR", -math.inf) > 0.0
                    and deltas.get("SSIM", -math.inf) > 0.0
                    and deltas.get("LPIPS", math.inf) < 0.0
                )
                nonregressive[ref_name] = (
                    deltas.get("PSNR", -math.inf) >= -1e-9
                    and deltas.get("SSIM", -math.inf) >= -1e-9
                    and deltas.get("LPIPS", math.inf) <= 1e-9
                )
        row["delta_by_ref"] = delta_by_ref
        row["strict_wins"] = strict_wins
        row["nonregressive"] = nonregressive


def choose_manifest(manifests: list[dict[str, Any]], method: str, result_path: Path | None = None) -> dict[str, Any] | None:
    if result_path is not None:
        for manifest in manifests:
            if manifest["result_name"] == result_path.name:
                return manifest
    for manifest in manifests:
        if manifest["method"] == method:
            return manifest
    return None


def choose_audit(audits: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    for audit in audits:
        if audit["method"] == method:
            return audit
    return None


def selected_candidate(audit: dict[str, Any]) -> dict[str, Any]:
    candidates = [c for c in audit.get("fill_mode_candidates", []) if isinstance(c, dict)]
    selected_alpha = fnum(audit.get("selected_alpha"))
    accepted = [c for c in candidates if c.get("accepted") is True]
    for candidate in accepted + candidates:
        if selected_alpha is not None and fnum(candidate.get("selected_alpha")) == selected_alpha:
            return candidate
    return accepted[0] if accepted else candidates[0] if candidates else {}


def texture_candidates(audit: dict[str, Any]) -> str:
    settings_value = first(
        get(audit, "fit_summary", "texture_size_candidates"),
        get(audit, "fit_summary", "policy_candidate_control", "adaptive_texture_size_ladder", "planned_texture_size_candidates"),
        get(audit, "settings", "texture_size_candidates"),
        audit.get("texture_size_candidates"),
    )
    if settings_value is not None:
        return str(settings_value).replace(" ", "")
    values = []
    for candidate in audit.get("fill_mode_candidates", []):
        if isinstance(candidate, dict) and candidate.get("texture_size") is not None:
            value = str(candidate["texture_size"])
            if value not in values:
                values.append(value)
    return ",".join(values)


def selected_support(audit: dict[str, Any]) -> str:
    fit = audit.get("fit_summary") if isinstance(audit.get("fit_summary"), dict) else {}
    candidate = selected_candidate(audit)
    support = candidate.get("support_summary") if isinstance(candidate.get("support_summary"), dict) else {}
    mode = first(
        fit.get("selected_support_mode"),
        fit.get("support_mode"),
        candidate.get("support_mode"),
        support.get("mode"),
        get(audit, "settings", "support_expansion_mode"),
    )
    added = first(fit.get("selected_support_added_faces"), fit.get("support_added_faces"), support.get("added_faces"))
    if mode is None and added is None:
        return ""
    return str(mode) if added is None else f"{mode} (+{added})"


def infer_scene(root: Path, manifests: list[dict[str, Any]], audits: list[dict[str, Any]], result_path: Path | None) -> str:
    for manifest in manifests:
        if get(manifest["payload"], "scene"):
            return str(get(manifest["payload"], "scene"))
    for audit in audits:
        if get(audit["payload"], "settings", "scene"):
            return str(get(audit["payload"], "settings", "scene"))
    if result_path:
        match = re.match(r"([A-Za-z0-9-]+)_", result_path.name)
        if match:
            return match.group(1)
    return root.name


def make_row(
    root: Path,
    scene: str,
    method: str,
    metrics: dict[str, float],
    result_path: Path | None,
    manifest: dict[str, Any] | None,
    audit: dict[str, Any] | None,
    status_hint: str,
) -> dict[str, Any]:
    manifest_payload = manifest["payload"] if manifest else {}
    audit_payload = audit["payload"] if audit else {}
    result_from_manifest = get(manifest_payload, "outputs", "results_path")
    displayed_result = result_path or (Path(str(result_from_manifest)) if result_from_manifest else None)
    return {
        "scene": scene or str(first(get(manifest_payload, "scene"), root.name) or ""),
        "method": method,
        "status": str(first(get(manifest_payload, "status"), status_hint) or ""),
        "protocol_passed": get(manifest_payload, "protocol_audit", "passed"),
        "PSNR": metrics.get("PSNR"),
        "SSIM": metrics.get("SSIM"),
        "LPIPS": metrics.get("LPIPS"),
        "selected_alpha": first(audit_payload.get("selected_alpha"), get(audit_payload, "fit_summary", "selected_alpha")),
        "texture_candidates": texture_candidates(audit_payload) if audit_payload else "",
        "selected_support": selected_support(audit_payload) if audit_payload else "",
        "changed_fraction": first(
            get(audit_payload, "target_apply", "changed_fraction"),
            get(audit_payload, "target_coverage_gate", "changed_fraction"),
        ),
        "result_path": rel(displayed_result) if displayed_result else "",
    }


def rows_for_bundle(root: Path, paths: dict[str, list[Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifests = load_manifests(paths["manifest"])
    audits = load_audits(paths["adapter"])
    seen_methods: set[str] = set()

    for result_path in paths["result"]:
        payload, error = read_json(result_path)
        method_rows = result_metrics(payload, result_path.stem) if error is None else {}
        if not method_rows:
            method_rows = {result_path.stem: {}}
        for method, metrics in method_rows.items():
            manifest = choose_manifest(manifests, method, result_path)
            audit = choose_audit(audits, method)
            scene = infer_scene(root, manifests, audits, result_path)
            rows.append(make_row(root, scene, method, metrics, result_path, manifest, audit, error or "results"))
            seen_methods.add(method)

    partial_methods: list[str] = []
    for item in manifests + audits:
        method = str(item.get("method") or "")
        if method and method not in seen_methods and method not in partial_methods:
            partial_methods.append(method)
    if not partial_methods and not paths["result"]:
        for per_view_path in paths["per_view"]:
            payload, _error = read_json(per_view_path)
            if isinstance(payload, dict):
                partial_methods.extend(str(k) for k, v in payload.items() if isinstance(v, dict) and str(k) not in partial_methods)
    if not partial_methods and not paths["result"] and paths["target_audit"]:
        partial_methods.append("")

    for method in partial_methods:
        manifest = choose_manifest(manifests, method)
        audit = choose_audit(audits, method)
        status = "target_audit" if not method else "manifest" if manifest else "audit" if audit else "per_view"
        scene = infer_scene(root, manifests, audits, None)
        rows.append(make_row(root, scene, method, {}, None, manifest, audit, status))
        if method:
            seen_methods.add(method)
    return rows


def fmt(value: Any, digits: int = 6) -> str:
    number = fnum(value)
    return "" if number is None else f"{number:.{digits}f}"


def bool_cell(value: Any) -> str:
    return "yes" if value is True else "no" if value is False else ""


def md(value: Any) -> str:
    return ("" if value is None else str(value)).replace("\n", " ").replace("|", "\\|")


def delta_cell(row: dict[str, Any], ref_name: str, metric: str) -> str:
    value = get(row, "delta_by_ref", ref_name, metric)
    number = fnum(value)
    if number is None:
        return ""
    return f"{number:+.6f}"


def win_cell(row: dict[str, Any], key: str, ref_name: str) -> str:
    value = get(row, key, ref_name)
    return "yes" if value is True else "no" if value is False else ""


def print_table(rows: list[dict[str, Any]]) -> None:
    has_ref = any(row.get("delta_by_ref") for row in rows)
    if has_ref:
        print("| scene | method | status | protocol_passed | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | strict win clean | dPSNR v106 | dSSIM v106 | dLPIPS v106 | non-reg v106 | selected alpha | texture candidates | selected support | changed fraction | result path |")
        print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|---:|---|")
    else:
        print("| scene | method | status | protocol_passed | PSNR | SSIM | LPIPS | selected alpha | texture candidates | selected support | changed fraction | result path |")
        print("|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---|")
    for row in sorted(rows, key=lambda r: (str(r["scene"]), str(r["method"]), str(r["result_path"]))):
        result_path = f"`{md(row['result_path'])}`" if row["result_path"] else ""
        cells = [
            md(row["scene"]),
            md(row["method"]),
            md(row["status"]),
            bool_cell(row["protocol_passed"]),
            fmt(row["PSNR"]),
            fmt(row["SSIM"]),
            fmt(row["LPIPS"]),
        ]
        if has_ref:
            cells.extend(
                [
                    delta_cell(row, "clean", "PSNR"),
                    delta_cell(row, "clean", "SSIM"),
                    delta_cell(row, "clean", "LPIPS"),
                    win_cell(row, "strict_wins", "clean"),
                    delta_cell(row, "v106_podmoe_basepreserve", "PSNR"),
                    delta_cell(row, "v106_podmoe_basepreserve", "SSIM"),
                    delta_cell(row, "v106_podmoe_basepreserve", "LPIPS"),
                    win_cell(row, "nonregressive", "v106_podmoe_basepreserve"),
                ]
            )
        cells.extend(
            [
                fmt(row["selected_alpha"]),
                md(row["texture_candidates"]),
                md(row["selected_support"]),
                fmt(row["changed_fraction"], 9),
                result_path,
            ]
        )
        print("| " + " | ".join(cells) + " |")


def print_reference_summary(rows: list[dict[str, Any]]) -> None:
    rows_with_refs = [row for row in rows if row.get("delta_by_ref")]
    if not rows_with_refs:
        return
    print()
    print("## Reference Summary")
    print()
    print("| method | rows | mean dPSNR clean | mean dSSIM clean | mean dLPIPS clean | strict wins clean | non-regressive v106 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows_with_refs:
        by_method.setdefault(str(row.get("method") or ""), []).append(row)
    for method, method_rows in sorted(by_method.items()):
        clean_deltas = [
            get(row, "delta_by_ref", "clean")
            for row in method_rows
            if isinstance(get(row, "delta_by_ref", "clean"), dict)
        ]
        def mean_delta(metric: str) -> str:
            values = [fnum(delta.get(metric)) for delta in clean_deltas if isinstance(delta, dict)]
            values = [value for value in values if value is not None]
            if not values:
                return ""
            return f"{sum(values) / len(values):+.6f}"

        strict_clean = sum(1 for row in method_rows if get(row, "strict_wins", "clean") is True)
        nonreg_v106 = sum(1 for row in method_rows if get(row, "nonregressive", "v106_podmoe_basepreserve") is True)
        print(
            f"| {md(method)} | {len(method_rows)} | {mean_delta('PSNR')} | {mean_delta('SSIM')} | {mean_delta('LPIPS')} | {strict_clean}/{len(method_rows)} | {nonreg_v106}/{len(method_rows)} |"
        )


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for root, paths in find_artifacts(args.paths).items():
        rows.extend(rows_for_bundle(root, paths))
    refs = load_reference_compare_json(args.reference_compare_json, args.reference_methods)
    apply_reference_deltas(rows, refs)
    print_table(rows)
    print_reference_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
