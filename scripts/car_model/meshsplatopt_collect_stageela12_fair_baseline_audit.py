#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BaselineCandidate:
    label: str
    model: str
    iteration: int
    method: str


@dataclass(frozen=True)
class SceneSpec:
    scene: str
    dataset: str
    baseline_candidates: tuple[BaselineCandidate, ...]
    method_label: str
    method_model: str
    method_iteration: int
    method_name: str
    wandb: str
    method_rgb_model: str | None = None


SCENES: tuple[SceneSpec, ...] = (
    SceneSpec(
        scene="bonsai",
        dataset="Mip-NeRF 360",
        baseline_candidates=(
            BaselineCandidate("clean7000", "outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000", 7000, "ours_7000"),
            BaselineCandidate("clean9000", "outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000", 9000, "ours_9000"),
            BaselineCandidate("clean22000", "outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000", 22000, "ours_22000"),
        ),
        method_label="SOR10 + ELA safe",
        method_model="outputs/carnet/meshsplatopt/stageELA11_sparse_occluder_policy/bonsai/sor10_clean9000/compact_model",
        method_iteration=9000,
        method_name="ours_9000_sor10_ela_safe",
        wandb="vmai8bls",
    ),
    SceneSpec(
        scene="courtyard",
        dataset="ETH3D",
        baseline_candidates=(
            BaselineCandidate("clean7000", "outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000", 7000, "ours_7000"),
            BaselineCandidate("clean9000", "outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000", 9000, "ours_9000"),
            BaselineCandidate("clean22000", "outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000", 22000, "ours_22000"),
        ),
        method_label="SOR10 + ELA safe",
        method_model="outputs/carnet/meshsplatopt/stageELA11_sparse_occluder_policy/courtyard/sor10_clean9000/compact_model",
        method_iteration=9000,
        method_name="ours_9000_sor10_ela_safe",
        wandb="xcoa2n7y",
    ),
    SceneSpec(
        scene="room",
        dataset="Mip-NeRF 360",
        baseline_candidates=(
            BaselineCandidate("clean7000", "outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000", 7000, "ours_7000"),
            BaselineCandidate("clean9000", "outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000", 9000, "ours_9000"),
            BaselineCandidate("clean22000", "outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000", 22000, "ours_22000"),
        ),
        method_label="QEM50 parent-rollback + ELA safe",
        method_model="outputs/carnet/meshsplatopt/stageELA10_geometry_preserving_compact_recovery/room/qem50_sparse_parentrollback_9000to12000/recovery_model",
        method_iteration=12000,
        method_name="ours_12000_qem50_parentrollback_ela_safe",
        wandb="9t01dwd8",
    ),
    SceneSpec(
        scene="counter",
        dataset="Mip-NeRF 360",
        baseline_candidates=(
            BaselineCandidate("clean7000", "outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000", 7000, "ours_7000"),
            BaselineCandidate("clean9000", "outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000", 9000, "ours_9000"),
            BaselineCandidate("clean22000", "outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000", 22000, "ours_22000"),
        ),
        method_label="QEM50 parent-rollback + ELA safe",
        method_model="outputs/carnet/meshsplatopt/stageELA10_geometry_preserving_compact_recovery/counter/qem50_sparse_parentrollback_9000to12000/recovery_model",
        method_iteration=12000,
        method_name="ours_12000_qem50_parentrollback_ela_safe",
        wandb="zcc5inc0",
    ),
    SceneSpec(
        scene="parking_phone_tiny",
        dataset="Phone parking COLMAP",
        baseline_candidates=(
            BaselineCandidate(
                "clean22000",
                "outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_7000to22000/model",
                22000,
                "ours_22000",
            ),
            BaselineCandidate(
                "clean30000",
                "outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_22000to30000/model",
                30000,
                "ours_30000",
            ),
        ),
        method_label="CSEF70 sparse-depth + train-calibrated parent-gated ELA",
        method_model="outputs/carnet/meshsplatopt/final_stageF33_parking_csef_sparse_depth/prune70/recovery_model",
        method_iteration=26000,
        method_name="ours_26000_outdoor_parentgate_traincalib_v7",
        wandb="ts6721g0",
        method_rgb_model="outputs/carnet/meshsplatopt/stageOUT1_parking_visual_tail_recovery/f33_outdoor_parentgate_traincalib_v7_eval",
    ),
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _rel(path: str | Path, base: Path = ROOT) -> str:
    return os.path.relpath(Path(path), start=base)


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _metrics_from_file(model: Path, split: str, method: str) -> dict[str, float]:
    filename = f"{split}_results.json" if split != "test" else "results.json"
    payload = _read_json(model / filename)
    row = payload.get(method, {})
    return {
        "psnr": _finite(row.get("PSNR")),
        "ssim": _finite(row.get("SSIM")),
        "lpips": _finite(row.get("LPIPS")),
    }


def _per_view(model: Path, method: str) -> dict[str, Any]:
    return _read_json(model / "per_view.json").get(method, {})


def _geometry(model: Path, iteration: int) -> dict[str, float]:
    path = model / "geometry_eval_colmap" / f"iter_{iteration}_max500.json"
    if not path.exists():
        return {"abs_rel": math.nan, "depth_mae": math.nan, "normal": math.nan}
    payload = _read_json(path)
    return {
        "abs_rel": _finite(payload.get("depth", {}).get("abs_rel")),
        "depth_mae": _finite(payload.get("depth", {}).get("mae")),
        "normal": _finite(payload.get("normal", {}).get("mean_ang_deg")),
    }


def _topology(model: Path, iteration: int) -> dict[str, int]:
    import torch

    ckpt = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    state = torch.load(ckpt, map_location="cpu")
    return {
        "triangles": int(state["_triangle_indices"].shape[0]),
        "vertices": int(state["triangles_points"].shape[0]),
    }


def _score(metrics: dict[str, float]) -> float:
    # Train-only checkpoint selection. PSNR carries the scale, SSIM/LPIPS break perceptual ties.
    return float(metrics["psnr"] + 20.0 * metrics["ssim"] - 20.0 * metrics["lpips"])


def _candidate_row(scene: str, candidate: BaselineCandidate) -> dict[str, Any]:
    model = ROOT / candidate.model
    train = _metrics_from_file(model, "train", candidate.method)
    test = _metrics_from_file(model, "test", candidate.method)
    geom = _geometry(model, candidate.iteration)
    topo = _topology(model, candidate.iteration)
    return {
        "scene": scene,
        "label": candidate.label,
        "model": candidate.model,
        "iteration": candidate.iteration,
        "method": candidate.method,
        "train_psnr": train["psnr"],
        "train_ssim": train["ssim"],
        "train_lpips": train["lpips"],
        "train_selection_score": _score(train),
        "test_psnr": test["psnr"],
        "test_ssim": test["ssim"],
        "test_lpips": test["lpips"],
        "abs_rel": geom["abs_rel"],
        "depth_mae": geom["depth_mae"],
        "normal": geom["normal"],
        "triangles": topo["triangles"],
        "vertices": topo["vertices"],
    }


def _select_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite_rows = [row for row in rows if math.isfinite(float(row["train_selection_score"]))]
    if not finite_rows:
        raise RuntimeError("No finite train-selection rows")
    return max(finite_rows, key=lambda row: (float(row["train_selection_score"]), float(row["train_psnr"])))


def _comparison_row(spec: SceneSpec, baseline: dict[str, Any]) -> dict[str, Any]:
    method_model = ROOT / spec.method_model
    method_rgb_model = ROOT / (spec.method_rgb_model or spec.method_model)
    method_rgb = _metrics_from_file(method_rgb_model, "test", spec.method_name)
    method_geom = _geometry(method_model, spec.method_iteration)
    method_topo = _topology(method_model, spec.method_iteration)
    out: dict[str, Any] = {
        "scene": spec.scene,
        "dataset": spec.dataset,
        "baseline_label": baseline["label"],
        "baseline_model": baseline["model"],
        "baseline_iteration": baseline["iteration"],
        "baseline_train_selection_score": baseline["train_selection_score"],
        "baseline_test_psnr": baseline["test_psnr"],
        "baseline_test_ssim": baseline["test_ssim"],
        "baseline_test_lpips": baseline["test_lpips"],
        "baseline_abs_rel": baseline["abs_rel"],
        "baseline_depth_mae": baseline["depth_mae"],
        "baseline_normal": baseline["normal"],
        "baseline_triangles": baseline["triangles"],
        "baseline_vertices": baseline["vertices"],
        "method_label": spec.method_label,
        "method_model": spec.method_model,
        "method_rgb_model": spec.method_rgb_model or spec.method_model,
        "method_iteration": spec.method_iteration,
        "method_name": spec.method_name,
        "wandb": spec.wandb,
        "method_psnr": method_rgb["psnr"],
        "method_ssim": method_rgb["ssim"],
        "method_lpips": method_rgb["lpips"],
        "method_abs_rel": method_geom["abs_rel"],
        "method_depth_mae": method_geom["depth_mae"],
        "method_normal": method_geom["normal"],
        "method_triangles": method_topo["triangles"],
        "method_vertices": method_topo["vertices"],
    }
    out.update(
        {
            "d_psnr": out["method_psnr"] - out["baseline_test_psnr"],
            "d_ssim": out["method_ssim"] - out["baseline_test_ssim"],
            "d_lpips": out["method_lpips"] - out["baseline_test_lpips"],
            "d_abs_rel": out["method_abs_rel"] - out["baseline_abs_rel"],
            "d_depth_mae": out["method_depth_mae"] - out["baseline_depth_mae"],
            "d_normal": out["method_normal"] - out["baseline_normal"],
            "triangle_reduction": 1.0 - (out["method_triangles"] / out["baseline_triangles"]),
            "vertex_reduction": 1.0 - (out["method_vertices"] / out["baseline_vertices"]),
        }
    )
    out["strict_full_pass"] = (
        out["d_psnr"] > 0.0
        and out["d_ssim"] > 0.0
        and out["d_lpips"] < 0.0
        and out["d_abs_rel"] < 0.0
        and out["d_depth_mae"] < 0.0
        and out["d_normal"] < 0.0
        and out["triangle_reduction"] > 0.0
    )
    return out


def _common_images(model_a: Path, method_a: str, model_b: Path, method_b: str) -> list[str]:
    a = model_a / "test" / method_a / "renders"
    b = model_b / "test" / method_b / "renders"
    names_a = {p.name for p in a.glob("*.png")} if a.is_dir() else set()
    names_b = {p.name for p in b.glob("*.png")} if b.is_dir() else set()
    return sorted(names_a & names_b)


def _collect_per_view(spec: SceneSpec, baseline: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_model = ROOT / str(baseline["model"])
    method_model = ROOT / (spec.method_rgb_model or spec.method_model)
    baseline_method = str(baseline["method"])
    method_name = spec.method_name
    base_per = _per_view(baseline_model, baseline_method)
    method_per = _per_view(method_model, method_name)
    rows = []
    for name in _common_images(baseline_model, baseline_method, method_model, method_name):
        b_psnr = _finite(base_per.get("PSNR", {}).get(name))
        m_psnr = _finite(method_per.get("PSNR", {}).get(name))
        b_ssim = _finite(base_per.get("SSIM", {}).get(name))
        m_ssim = _finite(method_per.get("SSIM", {}).get(name))
        b_lpips = _finite(base_per.get("LPIPS", {}).get(name))
        m_lpips = _finite(method_per.get("LPIPS", {}).get(name))
        rows.append(
            {
                "scene": spec.scene,
                "image": name,
                "baseline_psnr": b_psnr,
                "method_psnr": m_psnr,
                "d_psnr": m_psnr - b_psnr,
                "baseline_ssim": b_ssim,
                "method_ssim": m_ssim,
                "d_ssim": m_ssim - b_ssim,
                "baseline_lpips": b_lpips,
                "method_lpips": m_lpips,
                "d_lpips": m_lpips - b_lpips,
                "rgb_full_pass": m_psnr > b_psnr and m_ssim > b_ssim and m_lpips < b_lpips,
                "gt": _rel(method_model / "test" / method_name / "gt" / name),
                "baseline_render": _rel(baseline_model / "test" / baseline_method / "renders" / name),
                "method_render": _rel(method_model / "test" / method_name / "renders" / name),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, precision: int = 6) -> str:
    if isinstance(value, bool):
        return "`True`" if value else "`False`"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "nan"
    return f"{f:.{precision}f}"


def _pick_gallery_rows(per_view_rows: list[dict[str, Any]], per_scene: int) -> list[dict[str, Any]]:
    selected = []
    for spec in SCENES:
        rows = [row for row in per_view_rows if row["scene"] == spec.scene and math.isfinite(float(row["d_psnr"]))]
        if not rows:
            continue
        ordered = sorted(rows, key=lambda row: float(row["d_psnr"]))
        if per_scene <= 1:
            indexes = [len(ordered) // 2]
        else:
            indexes = [round(i * (len(ordered) - 1) / (per_scene - 1)) for i in range(per_scene)]
        seen = set()
        for index in indexes:
            row = ordered[index]
            if row["image"] in seen:
                continue
            selected.append(row)
            seen.add(row["image"])
        for row in ordered:
            if len([x for x in selected if x["scene"] == spec.scene]) >= per_scene:
                break
            if row["image"] in seen:
                continue
            selected.append(row)
            seen.add(row["image"])
    return selected


def _write_gallery(out_dir: Path, selected_rows: list[dict[str, Any]]) -> None:
    gallery_dir = out_dir / "qualitative_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    _write_json(gallery_dir / "selected_views.json", selected_rows)
    lines = [
        "# Stage ELA12 Fair Baseline Qualitative Manifest",
        "",
        "Each selected held-out view aligns GT, train-selected clean Mesh Splatting baseline, and the current method.",
        "Views are selected mechanically per scene from method-minus-baseline per-view PSNR: worst, middle, and best cases.",
        "",
        "| scene | view | baseline PSNR | method PSNR | dPSNR | baseline LPIPS | method LPIPS | dLPIPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['scene']} | {row['image']} | {_fmt(row['baseline_psnr'], 4)} | {_fmt(row['method_psnr'], 4)} | "
            f"{float(row['d_psnr']):+.4f} | {_fmt(row['baseline_lpips'], 4)} | {_fmt(row['method_lpips'], 4)} | {float(row['d_lpips']):+.4f} |"
        )
    (gallery_dir / "gallery_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sections = []
    for row in selected_rows:
        gt = _rel(ROOT / row["gt"], gallery_dir)
        baseline = _rel(ROOT / row["baseline_render"], gallery_dir)
        method = _rel(ROOT / row["method_render"], gallery_dir)
        sections.append(
            "<section>"
            f"<h2>{row['scene']} / {row['image']} / dPSNR {float(row['d_psnr']):+.4f} / dLPIPS {float(row['d_lpips']):+.4f}</h2>"
            "<div class='grid'>"
            f"<figure><img src='{gt}'><figcaption>GT</figcaption></figure>"
            f"<figure><img src='{baseline}'><figcaption>Train-selected clean Mesh Splatting</figcaption></figure>"
            f"<figure><img src='{method}'><figcaption>SPCarNet adaptive policy</figcaption></figure>"
            "</div>"
            "</section>"
        )
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage ELA12 fair baseline gallery</title>
<style>
body { font-family: system-ui, sans-serif; margin: 24px; background: #f6f6f4; color: #202020; }
h1 { font-size: 24px; margin: 0 0 8px; }
h2 { font-size: 16px; margin: 28px 0 10px; }
p { max-width: 980px; color: #444; }
.grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; align-items: start; }
figure { margin: 0; background: #fff; border: 1px solid #ddd; padding: 8px; }
img { width: 100%; height: auto; display: block; }
figcaption { font-size: 13px; margin-top: 6px; color: #555; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>Stage ELA12 fair baseline gallery</h1>
<p>GT, train-selected clean Mesh Splatting baseline, and current method are shown for identical held-out views. Selection is mechanical: worst, middle, and best method-minus-baseline PSNR per scene.</p>
""" + "\n".join(sections) + "\n</body>\n</html>\n"
    (gallery_dir / "gallery.html").write_text(html, encoding="utf-8")


def _write_report(path: Path, candidate_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]], per_view_rows: list[dict[str, Any]], out_dir: Path) -> None:
    pass_rows = [row for row in comparison_rows if row["strict_full_pass"]]
    view_summary = []
    for spec in SCENES:
        rows = [row for row in per_view_rows if row["scene"] == spec.scene]
        view_summary.append(
            {
                "scene": spec.scene,
                "views": len(rows),
                "rgb_full_pass_views": sum(1 for row in rows if row["rgb_full_pass"]),
                "min_d_psnr": min(float(row["d_psnr"]) for row in rows) if rows else math.nan,
                "mean_d_psnr": sum(float(row["d_psnr"]) for row in rows) / max(len(rows), 1),
                "max_d_lpips": max(float(row["d_lpips"]) for row in rows) if rows else math.nan,
            }
        )
    lines = [
        "# Stage ELA12 Fair Baseline Audit",
        "",
        "Baseline checkpoint selection is now train-only: for each scene, all available clean Mesh Splatting candidate checkpoints are rendered on train and scored by `PSNR + 20 * SSIM - 20 * LPIPS`. Test metrics are used only after selecting the clean baseline checkpoint.",
        "",
        f"Strict full-pass rows against train-selected clean baselines: `{len(pass_rows)}/{len(comparison_rows)}`.",
        "",
        "## Train-Selected Baseline Comparison",
        "",
        "| scene | selected baseline | method | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal | tri reduction | full |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['scene']} | {row['baseline_label']}@{row['baseline_iteration']} | {row['method_label']} | "
            f"{_fmt(row['d_psnr'])} | {_fmt(row['d_ssim'])} | {_fmt(row['d_lpips'])} | {_fmt(row['d_abs_rel'])} | "
            f"{_fmt(row['d_depth_mae'])} | {_fmt(row['d_normal'])} | {_fmt(100.0 * row['triangle_reduction'], 2)}% | {_fmt(row['strict_full_pass'])} |"
        )
    lines.extend(
        [
            "",
            "## Clean Baseline Candidate Table",
            "",
            "| scene | candidate | train score | train PSNR | train SSIM | train LPIPS | test PSNR | test SSIM | test LPIPS |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in candidate_rows:
        lines.append(
            f"| {row['scene']} | {row['label']}@{row['iteration']} | {_fmt(row['train_selection_score'])} | {_fmt(row['train_psnr'])} | "
            f"{_fmt(row['train_ssim'])} | {_fmt(row['train_lpips'])} | {_fmt(row['test_psnr'])} | {_fmt(row['test_ssim'])} | {_fmt(row['test_lpips'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-View RGB Stress Test",
            "",
            "| scene | views | RGB full-pass views | min dPSNR | mean dPSNR | worst dLPIPS |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in view_summary:
        lines.append(
            f"| {row['scene']} | {row['views']} | {row['rgb_full_pass_views']} | {_fmt(row['min_d_psnr'])} | {_fmt(row['mean_d_psnr'])} | {_fmt(row['max_d_lpips'])} |"
        )
    lines.extend(
        [
            "",
            "## Scope Note",
            "",
            "This audit covers the current validated scene set with complete method artifacts: `parking_phone_tiny`, `bonsai`, `courtyard`, `room`, and `counter`. Raw dataset folders that do not yet have complete method artifacts are intentionally not folded into the headline table.",
            "",
            "## Artifacts",
            "",
            f"- summary JSON: `{_rel(out_dir / 'fair_baseline_audit.json')}`",
            f"- baseline candidate CSV: `{_rel(out_dir / 'baseline_candidate_rows.csv')}`",
            f"- comparison CSV: `{_rel(out_dir / 'fair_selected_baseline_comparison.csv')}`",
            f"- per-view CSV: `{_rel(out_dir / 'per_view_rgb_deltas.csv')}`",
            f"- qualitative gallery: `{_rel(out_dir / 'qualitative_gallery' / 'gallery.html')}`",
            f"- qualitative manifest: `{_rel(out_dir / 'qualitative_gallery' / 'gallery_manifest.md')}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _maybe_log_wandb(args: argparse.Namespace, comparison_rows: list[dict[str, Any]], per_view_rows: list[dict[str, Any]], report_path: str) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[WARN] W&B unavailable: {exc}")
        return
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config={
            "selection_rule": "train_psnr_plus_20ssim_minus_20lpips",
            "report": report_path,
        },
    )
    pass_count = sum(1 for row in comparison_rows if row["strict_full_pass"])
    run.log(
        {
            "fair_baseline/strict_full_pass_count": pass_count,
            "fair_baseline/scene_count": len(comparison_rows),
            "fair_baseline/per_view_rgb_pass_count": sum(1 for row in per_view_rows if row["rgb_full_pass"]),
            "fair_baseline/per_view_count": len(per_view_rows),
            "fair_baseline/min_d_psnr": min(float(row["d_psnr"]) for row in per_view_rows),
            "fair_baseline/max_d_lpips": max(float(row["d_lpips"]) for row in per_view_rows),
        }
    )
    run.summary.update(
        {
            "strict_full_pass": f"{pass_count}/{len(comparison_rows)}",
            "per_view_rgb_pass": f"{sum(1 for row in per_view_rows if row['rgb_full_pass'])}/{len(per_view_rows)}",
        }
    )
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect fair train-selected baseline audit for current MeshSplatOpt scene set.")
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/stageELA12_fair_baseline_audit")
    parser.add_argument("--report", default="docs/car_model/stageELA12_fair_baseline_audit_report.md")
    parser.add_argument("--gallery-per-scene", default=3, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet_meshprior")
    parser.add_argument("--wandb_group", default="ELA12_fair_baseline_audit")
    parser.add_argument("--wandb_name", default="ELA12_fair_baseline_audit")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    per_view_rows: list[dict[str, Any]] = []
    selected_by_scene: dict[str, dict[str, Any]] = {}
    for spec in SCENES:
        scene_candidates = [_candidate_row(spec.scene, candidate) for candidate in spec.baseline_candidates]
        candidate_rows.extend(scene_candidates)
        selected = _select_baseline(scene_candidates)
        selected_by_scene[spec.scene] = selected
        comparison_rows.append(_comparison_row(spec, selected))
        per_view_rows.extend(_collect_per_view(spec, selected))

    _write_csv(out_dir / "baseline_candidate_rows.csv", candidate_rows)
    _write_csv(out_dir / "fair_selected_baseline_comparison.csv", comparison_rows)
    _write_csv(out_dir / "per_view_rgb_deltas.csv", per_view_rows)
    selected_gallery_rows = _pick_gallery_rows(per_view_rows, args.gallery_per_scene)
    _write_gallery(out_dir, selected_gallery_rows)
    payload = {
        "decision": "FAIR_TRAIN_SELECTED_BASELINE_AUDIT_READY",
        "selection_rule": "train_psnr_plus_20ssim_minus_20lpips",
        "selected_by_scene": selected_by_scene,
        "baseline_candidates": candidate_rows,
        "comparison_rows": comparison_rows,
        "per_view_rgb_deltas": per_view_rows,
        "selected_gallery_rows": selected_gallery_rows,
    }
    _write_json(out_dir / "fair_baseline_audit.json", payload)
    _write_report(ROOT / args.report, candidate_rows, comparison_rows, per_view_rows, out_dir)
    _maybe_log_wandb(args, comparison_rows, per_view_rows, args.report)
    print(json.dumps({"decision": payload["decision"], "report": args.report, "out_dir": args.out_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
