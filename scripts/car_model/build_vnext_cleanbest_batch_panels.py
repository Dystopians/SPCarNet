#!/usr/bin/env python3
"""Build clean-best/base/vNext qualitative panels from preserved vNext scene runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

from build_vnext_qualitative_panels import build_panel


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric_score(row: Dict[str, float], policy: str) -> float:
    if policy == "composite_psnr_ssim_lpips":
        return float(row["PSNR"]) + 20.0 * float(row["SSIM"]) - 20.0 * float(row["LPIPS"])
    if policy == "psnr":
        return float(row["PSNR"])
    if policy == "ssim":
        return float(row["SSIM"])
    if policy == "lpips":
        return -float(row["LPIPS"])
    raise ValueError(f"unsupported clean selection policy: {policy}")


def _find_clean_best(clean_root: Path, scene: str, args: argparse.Namespace) -> Dict[str, Any]:
    results_path = clean_root / scene / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing clean results: {results_path}")
    results = _read_json(results_path)
    candidates = []
    for method, metrics in results.items():
        render_dir = clean_root / scene / "test" / method / "renders"
        if method.startswith("ours_") and render_dir.is_dir():
            candidates.append(
                {
                    "method": method,
                    "render_dir": str(render_dir),
                    "metrics": metrics,
                    "score": _metric_score(metrics, str(args.clean_selection_policy)),
                }
            )
    if not candidates:
        raise RuntimeError(f"No clean candidates with renders for scene: {scene}")
    if args.clean_method:
        explicit = [row for row in candidates if row["method"] == args.clean_method]
        if not explicit:
            raise RuntimeError(
                f"Requested --clean_method {args.clean_method!r} is not available for scene {scene}. "
                f"Candidates: {[row['method'] for row in candidates]}"
            )
        selected = explicit[0]
    else:
        selected = max(candidates, key=lambda row: row["score"])
    selected = dict(selected)
    selected["selection"] = {
        "policy": str(args.clean_selection_policy),
        "explicit_clean_method": str(args.clean_method or ""),
        "results_path": str(results_path),
        "candidate_count": int(len(candidates)),
        "candidate_scores": {
            str(row["method"]): {
                "score": float(row["score"]),
                "metrics": dict(row["metrics"]),
                "render_dir": str(row["render_dir"]),
            }
            for row in candidates
        },
    }
    return selected


def _load_metrics(path: Path, method: str) -> Dict[str, float] | None:
    if not path.is_file():
        return None
    data = _read_json(path)
    if method in data:
        return data[method]
    if len(data) == 1:
        only = next(iter(data.values()))
        if isinstance(only, dict):
            return only
    return None


def _scene_row(args: argparse.Namespace, scene: str) -> Dict[str, Any]:
    clean = _find_clean_best(Path(args.clean_root), scene, args)
    base_render_dir = (
        Path(args.base_root)
        / scene
        / "ratio_0200"
        / "compact_model"
        / "test"
        / args.base_method_name
        / "renders"
    )
    vnext_method_dir = Path(args.run_root) / scene / "model" / "test" / args.method_name
    vnext_render_dir = vnext_method_dir / "renders"
    vnext_gt_dir = vnext_method_dir / "gt"

    missing = []
    for label, path in (
        ("base_render_dir", base_render_dir),
        ("vnext_render_dir", vnext_render_dir),
        ("vnext_gt_dir", vnext_gt_dir),
    ):
        if not path.is_dir():
            missing.append({"label": label, "path": str(path)})
    if missing:
        return {
            "scene": scene,
            "status": "MISSING_RENDER_INPUT",
            "missing": missing,
            "clean": clean,
        }

    clean_suffix = clean["method"].replace("ours_", "")
    panel_name = f"{scene}_cleanbest_base_vnext_panel"
    panel_output_dir = Path(args.output_root) / scene
    base_results_path = (
        Path(args.base_root) / scene / "ratio_0200" / "compact_model" / "test_results.json"
    )
    vnext_results_path = (
        Path(args.run_root)
        / scene
        / "reports"
        / f"{scene}_{args.method_name}_test_results.json"
    )
    vnext_manifest_path = (
        Path(args.run_root)
        / scene
        / "reports"
        / f"{scene}_vnext_certified_residual_texture_manifest.json"
    )
    adapter_audit_path = (
        Path(args.run_root) / scene / "model" / "surface_residual_region_texture_adapter_audit.json"
    )
    provenance = {
        "scene": scene,
        "clean_selection": dict(clean.get("selection", {})),
        "clean_results_path": str(Path(args.clean_root) / scene / "results.json"),
        "base_results_path": str(base_results_path),
        "vnext_results_path": str(vnext_results_path),
        "vnext_manifest_path": str(vnext_manifest_path),
        "adapter_audit_path": str(adapter_audit_path),
    }
    panel_args = SimpleNamespace(
        gt_dir=str(vnext_gt_dir),
        method=[
            f"cleanbest{clean_suffix}={clean['render_dir']}",
            f"compact_base={base_render_dir}",
            f"vnext={vnext_render_dir}",
        ],
        reference_label="compact_base",
        candidate_label="vnext",
        output_dir=str(panel_output_dir),
        panel_name=panel_name,
        num_views=int(args.num_views),
        tile_width=int(args.tile_width),
        label_height=int(args.label_height),
        row_gap=int(args.row_gap),
        diff_scale=float(args.diff_scale),
        frames=None,
        selection_mode=str(args.selection_mode),
        command=list(sys.argv),
        provenance_json=json.dumps(provenance, sort_keys=True),
    )
    try:
        manifest = build_panel(panel_args)
    except ValueError as exc:
        return {
            "scene": scene,
            "status": "FRAME_CONTRACT_MISMATCH",
            "error": str(exc),
            "clean": clean,
            "base_metrics": _load_metrics(base_results_path, args.base_method_name),
            "vnext_metrics": _load_metrics(vnext_results_path, args.method_name),
            "source_paths": provenance,
            "panel_manifest": None,
        }
    return {
        "scene": scene,
        "status": "COMPLETE",
        "clean": clean,
        "base_metrics": _load_metrics(base_results_path, args.base_method_name),
        "vnext_metrics": _load_metrics(vnext_results_path, args.method_name),
        "source_paths": provenance,
        "panel_manifest": manifest,
    }


def _delta(vnext: Dict[str, float] | None, ref: Dict[str, float] | None, key: str) -> str:
    if not vnext or not ref or key not in vnext or key not in ref:
        return ""
    value = float(vnext[key]) - float(ref[key])
    return f"{value:+.9f}"


def _metric(row: Dict[str, float] | None, key: str) -> str:
    if not row or key not in row:
        return ""
    return f"{float(row[key]):.9f}"


def _write_summary_md(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# vNext Clean-Best Qualitative Panel Batch",
        "",
        f"- schema version: `{payload['schema_version']}`",
        f"- run root: `{payload['run_root']}`",
        f"- output root: `{payload['output_root']}`",
        f"- scenes: `{', '.join(payload['scenes'])}`",
        f"- clean selection policy: `{payload['clean_selection_policy']}`",
        f"- explicit clean method: `{payload.get('clean_method') or ''}`",
        "",
        "| scene | status | clean best | vNext PSNR | vNext SSIM | vNext LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | dPSNR base | dSSIM base | dLPIPS base | panel |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        clean = row.get("clean") or {}
        clean_metrics = clean.get("metrics")
        base_metrics = row.get("base_metrics")
        vnext_metrics = row.get("vnext_metrics")
        panel_path = ""
        if row.get("panel_manifest"):
            panel_path = row["panel_manifest"].get("panel_path", "")
        lines.append(
            "| {scene} | {status} | {clean_method} | {psnr} | {ssim} | {lpips} | {dc_psnr} | {dc_ssim} | {dc_lpips} | {db_psnr} | {db_ssim} | {db_lpips} | `{panel}` |".format(
                scene=row.get("scene", ""),
                status=row.get("status", ""),
                clean_method=clean.get("method", ""),
                psnr=_metric(vnext_metrics, "PSNR"),
                ssim=_metric(vnext_metrics, "SSIM"),
                lpips=_metric(vnext_metrics, "LPIPS"),
                dc_psnr=_delta(vnext_metrics, clean_metrics, "PSNR"),
                dc_ssim=_delta(vnext_metrics, clean_metrics, "SSIM"),
                dc_lpips=_delta(vnext_metrics, clean_metrics, "LPIPS"),
                db_psnr=_delta(vnext_metrics, base_metrics, "PSNR"),
                db_ssim=_delta(vnext_metrics, base_metrics, "SSIM"),
                db_lpips=_delta(vnext_metrics, base_metrics, "LPIPS"),
                panel=panel_path,
            )
        )
    lines.extend(
        [
            "",
            "Interpretation note: LPIPS deltas are better when negative. Panels use the clean checkpoint "
            "selection policy stated above from local test metrics and are intended for qualitative diagnosis, "
            "not parameter search.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--scenes", required=True, help="Comma-separated scenes.")
    parser.add_argument(
        "--clean_root",
        default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k",
    )
    parser.add_argument(
        "--base_root",
        default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix",
    )
    parser.add_argument("--method_name", default="ours_26000_vnext_structure_aware_shrink")
    parser.add_argument("--base_method_name", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument(
        "--clean_selection_policy",
        choices=("composite_psnr_ssim_lpips", "psnr", "ssim", "lpips"),
        default="composite_psnr_ssim_lpips",
        help=(
            "Checkpoint selection contract for clean MeshSplatting renders. The default composite is "
            "PSNR + 20*SSIM - 20*LPIPS and is written into every manifest."
        ),
    )
    parser.add_argument(
        "--clean_method",
        default="",
        help="Optional explicit clean method/checkpoint such as ours_30000; if set, no best-score search is used.",
    )
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--tile_width", type=int, default=300)
    parser.add_argument("--label_height", type=int, default=26)
    parser.add_argument("--row_gap", type=int, default=8)
    parser.add_argument("--diff_scale", type=float, default=4.0)
    parser.add_argument(
        "--selection_mode",
        choices=("first", "candidate_worst_gt_l1", "largest_candidate_reference_delta"),
        default="largest_candidate_reference_delta",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    rows = [_scene_row(args, scene) for scene in scenes]
    payload = {
        "schema_version": 2,
        "argv": list(sys.argv),
        "run_root": str(args.run_root),
        "output_root": str(args.output_root),
        "scenes": scenes,
        "clean_selection_policy": str(args.clean_selection_policy),
        "clean_method": str(args.clean_method or ""),
        "rows": rows,
    }
    output_root = Path(args.output_root)
    _write_json(output_root / "cleanbest_qualitative_batch_summary.json", payload)
    _write_summary_md(output_root / "cleanbest_qualitative_batch_summary.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
