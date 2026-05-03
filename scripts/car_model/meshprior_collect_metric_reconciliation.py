"""Collect paper-facing MeshPrior metric/topology evidence rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional report convenience
    Image = None
    ImageDraw = None


PROJECT = "https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs"


@dataclass(frozen=True)
class EvidenceRow:
    label: str
    scene: str
    stage: str
    method: str
    model_path: str
    iteration: int
    wandb_run: str
    metric_path: str = "independent_render_py_metrics_py"
    notes: str = ""


ROWS: tuple[EvidenceRow, ...] = (
    EvidenceRow(
        label="parking_m24_2_retention_7000",
        scene="parking_phone_tiny",
        stage="M24.2",
        method="late_prism_freeze_after_first_commit",
        model_path="outputs/carnet/meshprior/parking_phone_tiny/stage24_2_topology_retention/freeze_after_first_commit_7000iter/model",
        iteration=7000,
        wandb_run="vsv2bs79",
        notes="single parking-scene integrated topology-retention row",
    ),
    EvidenceRow(
        label="bonsai_m26_sparse_depth_baseline",
        scene="mipnerf360_bonsai",
        stage="M26",
        method="sparse_depth_baseline",
        model_path="outputs/carnet/meshprior/stage26_cross_scene/mipnerf360_bonsai_baseline_sparse_depth_2000iter/model",
        iteration=2000,
        wandb_run="xdct9uys",
    ),
    EvidenceRow(
        label="bonsai_m29_cap512",
        scene="mipnerf360_bonsai",
        stage="M29",
        method="candidate_cap512_adaptive",
        model_path="outputs/carnet/meshprior/stage29_candidate_selection/mipnerf360_bonsai_cap512_adaptive_ratio0p02_geom1400_2000iter/model",
        iteration=2000,
        wandb_run="ck157wtl",
    ),
    EvidenceRow(
        label="bonsai_m33_diverse_calib",
        scene="mipnerf360_bonsai",
        stage="M33",
        method="diverse_calib_measured_rank_cap512",
        model_path="outputs/carnet/meshprior/stage33_calibration_diversity/mipnerf360_bonsai_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model",
        iteration=2000,
        wandb_run="kg5htc8u",
        notes="Stage35 bonsai reference row",
    ),
    EvidenceRow(
        label="bonsai_m34_relaxed_v3",
        scene="mipnerf360_bonsai",
        stage="M34",
        method="post_commit_relaxed_score_v3",
        model_path="outputs/carnet/meshprior/stage34_post_commit_refresh/mipnerf360_bonsai_refresh_v3_relaxed_score_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model",
        iteration=2000,
        wandb_run="lt1v4652",
    ),
    EvidenceRow(
        label="bonsai_m35_retained_relaxed",
        scene="mipnerf360_bonsai",
        stage="M35",
        method="retained_relaxed_cap1_strict_gate",
        model_path="outputs/carnet/meshprior/stage35_retained_refresh/mipnerf360_bonsai_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter_retry1/model",
        iteration=2000,
        wandb_run="rszvl7gn",
        notes="current best retained-edit bonsai row",
    ),
    EvidenceRow(
        label="courtyard_m26_sparse_depth_baseline",
        scene="eth3d_courtyard",
        stage="M26",
        method="sparse_depth_baseline",
        model_path="outputs/carnet/meshprior/stage26_cross_scene/eth3d_courtyard_baseline_sparse_depth_2000iter/model",
        iteration=2000,
        wandb_run="mdan8yc2",
    ),
    EvidenceRow(
        label="courtyard_m32_measured_rank",
        scene="eth3d_courtyard",
        stage="M32",
        method="measured_rank_cap512",
        model_path="outputs/carnet/meshprior/stage32_measured_candidate_rank/eth3d_courtyard_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model",
        iteration=2000,
        wandb_run="fb7jfcaj",
    ),
    EvidenceRow(
        label="courtyard_m33_diverse_calib",
        scene="eth3d_courtyard",
        stage="M33",
        method="diverse_calib_measured_rank_cap512",
        model_path="outputs/carnet/meshprior/stage33_calibration_diversity/eth3d_courtyard_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model",
        iteration=2000,
        wandb_run="w9c0b65f",
    ),
    EvidenceRow(
        label="courtyard_m35_retained_relaxed",
        scene="eth3d_courtyard",
        stage="M35",
        method="retained_relaxed_cap1_strict_gate",
        model_path="outputs/carnet/meshprior/stage35_retained_refresh/eth3d_courtyard_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model",
        iteration=2000,
        wandb_run="u2s15ok0",
        notes="cross-scene retained-edit mechanism check",
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _nan() -> float:
    return float("nan")


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, _nan()))
    except Exception:
        return _nan()


def _independent_metrics(model_path: Path, iteration: int) -> dict[str, float]:
    results = _load_json(model_path / "results.json")
    metrics = results.get(f"ours_{int(iteration)}", {})
    return {
        "independent_psnr": _metric_value(metrics, "PSNR"),
        "independent_ssim": _metric_value(metrics, "SSIM"),
        "independent_lpips": _metric_value(metrics, "LPIPS"),
    }


def _cleanup_summary(model_path: Path) -> dict[str, Any]:
    return _load_json(model_path / "prism_debug" / "final_cleanup_summary.json")


def _retained_audit(model_path: Path, cleanup: dict[str, Any]) -> dict[str, Any]:
    audit = cleanup.get("relaxed_retained_topology_audit", {})
    if audit:
        return dict(audit)
    return _load_json(model_path / "prism_debug" / "relaxed_retained_topology_audit.json")


def _candidate_counts(model_path: Path) -> dict[str, int]:
    counts = {
        "candidate_meta_count": 0,
        "candidate_committed_count": 0,
        "candidate_rollback_count": 0,
        "candidate_no_candidate_count": 0,
        "candidate_relaxed_used_count": 0,
        "candidate_relaxed_cap_reached_count": 0,
    }
    for path in sorted((model_path / "prism_round_checkpoints").glob("*candidate_meta.json")):
        data = _load_json(path)
        counts["candidate_meta_count"] += 1
        counts["candidate_committed_count"] += int(bool(data.get("committed", False)))
        counts["candidate_rollback_count"] += int(bool(data.get("rollback", False)))
        counts["candidate_no_candidate_count"] += int(bool(data.get("no_candidates", False)))
        counts["candidate_relaxed_used_count"] += int(bool(data.get("candidate_relaxed_refresh_used", 0)))
        if str(data.get("candidate_relaxed_reject_reason", "")) == "relaxed_commit_cap_reached":
            counts["candidate_relaxed_cap_reached_count"] += 1
    return counts


def _wandb_url(run_id: str) -> str:
    return f"{PROJECT}/{run_id}" if run_id else ""


def _parse_training_internal_metrics(model_path: Path, iteration: int) -> dict[str, float]:
    train_log = model_path.parent / "logs" / "train.log"
    out = {
        "training_test_psnr": _nan(),
        "training_test_ssim": _nan(),
        "training_test_lpips": _nan(),
        "training_train_psnr": _nan(),
        "training_train_ssim": _nan(),
        "training_train_lpips": _nan(),
    }
    if not train_log.exists():
        return out
    pattern = re.compile(
        rf"\[ITER {int(iteration)}\] Evaluating (test|train): .*?PSNR ([0-9.eE+-]+) SSIM ([0-9.eE+-]+) LPIPS ([0-9.eE+-]+)"
    )
    for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        split = match.group(1)
        out[f"training_{split}_psnr"] = float(match.group(2))
        out[f"training_{split}_ssim"] = float(match.group(3))
        out[f"training_{split}_lpips"] = float(match.group(4))
    return out


def _collect_row(spec: EvidenceRow) -> dict[str, Any]:
    model_path = Path(spec.model_path)
    cleanup = _cleanup_summary(model_path)
    audit = _retained_audit(model_path, cleanup)
    metrics = _independent_metrics(model_path, spec.iteration)
    candidates = _candidate_counts(model_path)
    training = _parse_training_internal_metrics(model_path, spec.iteration)
    final_triangles = cleanup.get("post_prune_triangle_count", audit.get("final_triangle_count", ""))
    final_vertices = cleanup.get("post_prune_vertex_count", "")
    relaxed_records = audit.get("relaxed_commit_records", [])
    return {
        "label": spec.label,
        "scene": spec.scene,
        "stage": spec.stage,
        "method": spec.method,
        "model_path": str(model_path),
        "iteration": int(spec.iteration),
        "wandb_url": _wandb_url(spec.wandb_run),
        "metric_path": spec.metric_path,
        "final_triangles": final_triangles,
        "final_vertices": final_vertices,
        "active_prism_commits": int(candidates["candidate_committed_count"]),
        "rolled_back_candidate_rounds": int(candidates["candidate_rollback_count"]),
        "no_candidate_rounds": int(candidates["candidate_no_candidate_count"]),
        "relaxed_used_rounds": int(candidates["candidate_relaxed_used_count"]),
        "relaxed_cap_reached_rounds": int(candidates["candidate_relaxed_cap_reached_count"]),
        "relaxed_commit_count": int(audit.get("relaxed_commit_count", 0)),
        "active_relaxed_commit_count": int(audit.get("active_relaxed_commit_count", 0)),
        "validation_rolled_back_relaxed_commit_count": int(
            audit.get("validation_rolled_back_relaxed_commit_count", 0)
        ),
        "relaxed_topology_retained": audit.get("relaxed_topology_retained", ""),
        "relaxed_topology_erased": audit.get("relaxed_topology_erased", ""),
        "relaxed_commit_iterations": ",".join(str(r.get("iteration", "")) for r in relaxed_records),
        "notes": spec.notes,
        **metrics,
        **training,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Metric Reconciliation Evidence Table\n\n")
        f.write("Metric path: independent `render.py + metrics.py` unless otherwise stated. ")
        f.write("Training-time eval fields are retained separately and must not be mixed with independent metrics.\n\n")
        f.write(
            "| label | scene | method | triangles | PSNR | SSIM | LPIPS | commits | relaxed active/rolled back | W&B |\n"
        )
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                "| {label} | {scene} | {method} | {tri} | {psnr} | {ssim} | {lpips} | {commits} | {active}/{rolled} | {wandb} |\n".format(
                    label=row["label"],
                    scene=row["scene"],
                    method=row["method"],
                    tri=row["final_triangles"],
                    psnr=_fmt(row["independent_psnr"]),
                    ssim=_fmt(row["independent_ssim"]),
                    lpips=_fmt(row["independent_lpips"]),
                    commits=row["active_prism_commits"],
                    active=row["active_relaxed_commit_count"],
                    rolled=row["validation_rolled_back_relaxed_commit_count"],
                    wandb=row["wandb_url"],
                )
            )
        f.write("\n## Metric-Path Warning\n\n")
        f.write(
            "The `training_test_*` and `training_train_*` fields come from train logs. "
            "They use the training evaluation path and are diagnostic only. The paper table above uses independent metrics.\n\n"
        )
        f.write("## Failure Taxonomy\n\n")
        for item in FAILURE_TAXONOMY:
            f.write(f"- {item}\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


FAILURE_TAXONOMY = (
    "no-candidate after topology sync: recent protection can mask every survivor after a commit",
    "validation rollback: a relaxed commit can pass the counterfactual proxy yet fail the recovery-window validation",
    "retained relaxed cap reached: conservative M35 behavior intentionally blocks further relaxed fallback once one active relaxed edit survives",
    "metric-path mismatch: training eval metrics and independent render metrics are different rows and must not be averaged or substituted",
    "dataset geometry observability: COLMAP-track geometry proxies are meaningful for Mip-NeRF 360/ETH3D converted scenes, but not for Tanks mirrors without true sparse tracks",
)


def _make_panel(model_path: Path, iteration: int, out_path: Path, max_views: int = 3) -> bool:
    if Image is None or ImageDraw is None:
        return False
    root = model_path / "test" / f"ours_{int(iteration)}"
    render_dir = root / "renders"
    gt_dir = root / "gt"
    render_files = sorted(render_dir.glob("*.png"))[:max_views]
    pairs = [(p, gt_dir / p.name) for p in render_files if (gt_dir / p.name).exists()]
    if not pairs:
        return False
    thumb_w, thumb_h = 320, 180
    label_h = 24
    canvas = Image.new("RGB", (thumb_w * len(pairs), (thumb_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (render_path, gt_path) in enumerate(pairs):
        x = idx * thumb_w
        for row_idx, (kind, img_path) in enumerate((("render", render_path), ("gt", gt_path))):
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            y = row_idx * (thumb_h + label_h) + label_h
            canvas.paste(img, (x + (thumb_w - img.width) // 2, y + (thumb_h - img.height) // 2))
            draw.text((x + 6, row_idx * (thumb_h + label_h) + 4), f"{kind}: {img_path.name}", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_collect_row(spec) for spec in ROWS]
    report = {
        "status": "PASS",
        "metric_policy": "paper-facing tables use independent render.py + metrics.py results only",
        "rows": rows,
        "failure_taxonomy": list(FAILURE_TAXONOMY),
    }
    panel_paths: list[str] = []
    for spec in ROWS:
        if spec.stage != "M35":
            continue
        panel = out_dir / "visual_panels" / f"{spec.label}.png"
        if _make_panel(Path(spec.model_path), spec.iteration, panel):
            panel_paths.append(str(panel))
    report["visual_panels"] = panel_paths
    (out_dir / "metric_reconciliation_report.json").write_text(
        json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(rows, out_dir / "metric_reconciliation_table.csv")
    _write_markdown(rows, out_dir / "metric_reconciliation_table.md")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect paper-facing MeshPrior metric evidence.")
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/stage36_metric_reconciliation")
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps({"status": report["status"], "rows": len(report["rows"]), "visual_panels": report["visual_panels"]}, indent=2))


if __name__ == "__main__":
    main()
