"""Collect Stage26 cross-scene PRISM evidence into reproducible tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


TEST_RE = re.compile(
    r"\[ITER (?P<iteration>\d+)\] Evaluating test: "
    r"L1 (?P<l1>[-+0-9.eE]+) "
    r"PSNR (?P<psnr>[-+0-9.eE]+) "
    r"SSIM (?P<ssim>[-+0-9.eE]+) "
    r"LPIPS (?P<lpips>[-+0-9.eE]+) "
    r"FPS (?P<fps>[-+0-9.eE]+)"
)
WANDB_RE = re.compile(r"https://wandb\.ai/[^\s]+/runs/[A-Za-z0-9_-]+")
WANDB_SUMMARY_RE = re.compile(r"wandb:\s+(?P<key>mesh/(?:triangle_count|vertex_count))\s+(?P<value>[-+0-9.eE]+)")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _parse_train_log(path: Path) -> tuple[dict[str, Any], str, dict[str, float]]:
    metrics: dict[str, Any] = {}
    wandb_url = ""
    wandb_summary: dict[str, float] = {}
    if not path.exists():
        return metrics, wandb_url, wandb_summary
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TEST_RE.search(line)
        if match:
            iteration = match.group("iteration")
            metrics[iteration] = {
                "l1": float(match.group("l1")),
                "psnr": float(match.group("psnr")),
                "ssim": float(match.group("ssim")),
                "lpips": float(match.group("lpips")),
                "fps": float(match.group("fps")),
            }
        url_match = WANDB_RE.search(line)
        if url_match:
            wandb_url = url_match.group(0)
        summary_match = WANDB_SUMMARY_RE.search(line)
        if summary_match:
            wandb_summary[summary_match.group("key")] = float(summary_match.group("value"))
    return metrics, wandb_url, wandb_summary


def _run_label(run_dir: Path) -> tuple[str, str]:
    name = run_dir.name
    if name.startswith("mipnerf360_bonsai_"):
        scene = "mipnerf360_bonsai"
        variant = name.removeprefix("mipnerf360_bonsai_").removesuffix("_2000iter")
    elif name.startswith("eth3d_courtyard_"):
        scene = "eth3d_courtyard"
        variant = name.removeprefix("eth3d_courtyard_").removesuffix("_2000iter")
    else:
        parts = name.split("_")
        scene = "_".join(parts[:2])
        variant = "_".join(parts[2:]).removesuffix("_2000iter")
    return scene, variant


def _collect_prism(model_dir: Path) -> dict[str, Any]:
    candidate_files = sorted((model_dir / "prism_round_checkpoints").glob("*_candidate_meta.json"))
    rows = [_load_json(path) for path in candidate_files]
    effective = [row for row in rows if not int(row.get("no_candidates") or 0)]
    commits = [row for row in effective if bool(row.get("committed"))]
    rollbacks = [row for row in effective if int(row.get("rollback") or 0)]
    no_candidates = [row for row in rows if int(row.get("no_candidates") or 0)]
    return {
        "round_count": len(effective),
        "commit_count": len(commits),
        "rollback_count": len(rollbacks),
        "no_candidate_retry_count": len(no_candidates),
        "decisions": rows,
    }


def _collect_validation(model_dir: Path) -> dict[str, Any]:
    validation_files = sorted((model_dir / "prism_validation").glob("validation_iter_*.json"))
    rows = [_clean(_load_json(path)) for path in validation_files]
    observable = [
        row
        for row in rows
        if float((row.get("current") or {}).get("geometry_observable") or 0.0) > 0
    ]
    passes = [row for row in rows if bool(row.get("pass_gate"))]
    stage_best = rows[-1].get("stage_best") if rows else None
    return {
        "validation_count": len(rows),
        "observable_count": len(observable),
        "pass_count": len(passes),
        "latest_stage_best": stage_best,
        "rows": rows,
    }


def _collect_run(run_dir: Path) -> dict[str, Any]:
    scene, variant = _run_label(run_dir)
    model_dir = run_dir / "model"
    train_metrics, wandb_url, wandb_summary = _parse_train_log(run_dir / "logs" / "train.log")

    independent_metrics = {}
    results_path = model_dir / "results.json"
    if results_path.exists():
        payload = _load_json(results_path)
        independent_metrics = payload.get("ours_2000", {})

    final_cleanup = {}
    final_cleanup_path = model_dir / "prism_debug" / "final_cleanup_summary.json"
    if final_cleanup_path.exists():
        final_cleanup = _load_json(final_cleanup_path)

    return {
        "run": run_dir.name,
        "scene": scene,
        "variant": variant,
        "model_path": str(model_dir),
        "wandb_url": wandb_url,
        "wandb_summary": wandb_summary,
        "train_test_metrics": train_metrics,
        "independent_metrics": independent_metrics,
        "final_cleanup": final_cleanup,
        "prism": _collect_prism(model_dir),
        "validation": _collect_validation(model_dir),
    }


def _pct_delta(method: float | None, baseline: float | None) -> float | None:
    if method is None or baseline in (None, 0):
        return None
    return 100.0 * (method - baseline) / baseline


def _get_nested(row: dict[str, Any], *keys: str) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _build_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scene = {row["scene"]: [] for row in rows}
    for row in rows:
        by_scene[row["scene"]].append(row)

    pairs: list[dict[str, Any]] = []
    for scene, scene_rows in sorted(by_scene.items()):
        baseline = next((r for r in scene_rows if r["variant"].startswith("baseline")), None)
        method = next((r for r in scene_rows if "prism" in r["variant"]), None)
        if not baseline or not method:
            continue
        base_test = baseline["train_test_metrics"].get("2000", {})
        meth_test = method["train_test_metrics"].get("2000", {})
        base_ind = baseline["independent_metrics"]
        meth_ind = method["independent_metrics"]
        base_tri = _get_nested(baseline, "final_cleanup", "pre_prune_triangle_count")
        meth_tri = _get_nested(method, "final_cleanup", "pre_prune_triangle_count")
        base_wandb_tri = _get_nested(baseline, "wandb_summary", "mesh/triangle_count")
        meth_wandb_tri = _get_nested(method, "wandb_summary", "mesh/triangle_count")
        pairs.append(
            {
                "scene": scene,
                "baseline_run": baseline["run"],
                "method_run": method["run"],
                "train_psnr_delta": (meth_test.get("psnr") or 0.0) - (base_test.get("psnr") or 0.0),
                "train_ssim_delta": (meth_test.get("ssim") or 0.0) - (base_test.get("ssim") or 0.0),
                "train_lpips_delta": (meth_test.get("lpips") or 0.0) - (base_test.get("lpips") or 0.0),
                "independent_psnr_delta": (meth_ind.get("PSNR") or 0.0) - (base_ind.get("PSNR") or 0.0),
                "independent_ssim_delta": (meth_ind.get("SSIM") or 0.0) - (base_ind.get("SSIM") or 0.0),
                "independent_lpips_delta": (meth_ind.get("LPIPS") or 0.0) - (base_ind.get("LPIPS") or 0.0),
                "checkpoint_triangle_delta_pct": _pct_delta(meth_tri, base_tri),
                "wandb_triangle_delta_pct": _pct_delta(meth_wandb_tri, base_wandb_tri),
                "method_prism_commits": method["prism"]["commit_count"],
                "method_prism_rollbacks": method["prism"]["rollback_count"],
                "method_validation_passes": method["validation"]["pass_count"],
                "method_validation_observable": method["validation"]["observable_count"],
            }
        )
    return pairs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage26 Cross-Scene Evidence Summary",
        "",
        "Generated by `scripts/car_model/meshprior_collect_stage26_cross_scene.py`.",
        "",
        "## Runs",
        "",
        "| scene | variant | W&B | train PSNR/SSIM/LPIPS | independent PSNR/SSIM/LPIPS | W&B triangles | checkpoint triangles | PRISM decisions | validation |",
        "|---|---|---|---|---|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["scene"], r["variant"])):
        test = row["train_test_metrics"].get("2000", {})
        ind = row["independent_metrics"]
        triangles = _get_nested(row, "final_cleanup", "pre_prune_triangle_count")
        wandb_triangles = _get_nested(row, "wandb_summary", "mesh/triangle_count")
        prism = row["prism"]
        validation = row["validation"]
        lines.append(
            "| {scene} | {variant} | [run]({url}) | {tpsnr}/{tssim}/{tlpips} | "
            "{ipsnr}/{issim}/{ilpips} | {wandb_triangles} | {triangles} | {commits} commit, {rollbacks} rollback, {retry} retry | "
            "{passes}/{validations} pass, {observable} observable |".format(
                scene=row["scene"],
                variant=row["variant"],
                url=row["wandb_url"],
                tpsnr=_fmt(test.get("psnr")),
                tssim=_fmt(test.get("ssim")),
                tlpips=_fmt(test.get("lpips")),
                ipsnr=_fmt(ind.get("PSNR")),
                issim=_fmt(ind.get("SSIM")),
                ilpips=_fmt(ind.get("LPIPS")),
                wandb_triangles=_fmt(wandb_triangles, 0),
                triangles=triangles,
                commits=prism["commit_count"],
                rollbacks=prism["rollback_count"],
                retry=prism["no_candidate_retry_count"],
                passes=validation["pass_count"],
                validations=validation["validation_count"],
                observable=validation["observable_count"],
            )
        )

    lines += [
        "",
        "## Paired Deltas",
        "",
        "| scene | train dPSNR/dSSIM/dLPIPS | independent dPSNR/dSSIM/dLPIPS | W&B triangle delta | checkpoint triangle delta | PRISM commits/rollbacks | validation observable/pass |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for pair in pairs:
        lines.append(
            "| {scene} | {tdp}/{tds}/{tdl} | {idp}/{ids}/{idl} | {wandb_tri}% | {tri}% | {commits}/{rollbacks} | {obs}/{passes} |".format(
                scene=pair["scene"],
                tdp=_fmt(pair["train_psnr_delta"]),
                tds=_fmt(pair["train_ssim_delta"]),
                tdl=_fmt(pair["train_lpips_delta"]),
                idp=_fmt(pair["independent_psnr_delta"]),
                ids=_fmt(pair["independent_ssim_delta"]),
                idl=_fmt(pair["independent_lpips_delta"]),
                wandb_tri=_fmt(pair["wandb_triangle_delta_pct"], 2),
                tri=_fmt(pair["checkpoint_triangle_delta_pct"], 2),
                commits=pair["method_prism_commits"],
                rollbacks=pair["method_prism_rollbacks"],
                obs=pair["method_validation_observable"],
                passes=pair["method_validation_passes"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/carnet/meshprior/stage26_cross_scene"),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    out = args.out or root / "summary"
    run_dirs = sorted(path for path in root.iterdir() if (path / "model").is_dir())
    rows = [_collect_run(path) for path in run_dirs]
    pairs = _build_pairs(rows)

    out.mkdir(parents=True, exist_ok=True)
    (out / "stage26_cross_scene_summary.json").write_text(
        json.dumps({"runs": _clean(rows), "paired_deltas": _clean(pairs)}, indent=2),
        encoding="utf-8",
    )
    flat_rows = []
    for row in rows:
        test = row["train_test_metrics"].get("2000", {})
        ind = row["independent_metrics"]
        flat_rows.append(
            {
                "scene": row["scene"],
                "variant": row["variant"],
                "wandb_url": row["wandb_url"],
                "train_psnr": test.get("psnr"),
                "train_ssim": test.get("ssim"),
                "train_lpips": test.get("lpips"),
                "independent_psnr": ind.get("PSNR"),
                "independent_ssim": ind.get("SSIM"),
                "independent_lpips": ind.get("LPIPS"),
                "wandb_triangles": _get_nested(row, "wandb_summary", "mesh/triangle_count"),
                "wandb_vertices": _get_nested(row, "wandb_summary", "mesh/vertex_count"),
                "checkpoint_triangles": _get_nested(row, "final_cleanup", "pre_prune_triangle_count"),
                "checkpoint_vertices": _get_nested(row, "final_cleanup", "pre_prune_vertex_count"),
                "prism_commits": row["prism"]["commit_count"],
                "prism_rollbacks": row["prism"]["rollback_count"],
                "prism_no_candidate_retries": row["prism"]["no_candidate_retry_count"],
                "validation_count": row["validation"]["validation_count"],
                "validation_observable_count": row["validation"]["observable_count"],
                "validation_pass_count": row["validation"]["pass_count"],
            }
        )
    _write_csv(out / "stage26_cross_scene_runs.csv", flat_rows)
    _write_csv(out / "stage26_cross_scene_paired_deltas.csv", pairs)
    _write_markdown(out / "stage26_cross_scene_summary.md", rows, pairs)
    print(f"Wrote Stage26 summary to {out}")


if __name__ == "__main__":
    main()
