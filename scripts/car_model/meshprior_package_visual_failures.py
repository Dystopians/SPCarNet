"""Package MeshPrior visual panels, failure cases, and paper-safe claims."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None


PANEL_ROWS: tuple[dict[str, Any], ...] = (
    {
        "label": "parking_m24_2_retention_7000",
        "scene": "parking_phone_tiny",
        "iteration": 7000,
        "model_path": "outputs/carnet/meshprior/parking_phone_tiny/stage24_2_topology_retention/freeze_after_first_commit_7000iter/model",
    },
    {
        "label": "bonsai_m35_retained_relaxed",
        "scene": "mipnerf360_bonsai",
        "iteration": 2000,
        "model_path": "outputs/carnet/meshprior/stage35_retained_refresh/mipnerf360_bonsai_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter_retry1/model",
    },
    {
        "label": "courtyard_m35_retained_relaxed",
        "scene": "eth3d_courtyard",
        "iteration": 2000,
        "model_path": "outputs/carnet/meshprior/stage35_retained_refresh/eth3d_courtyard_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model",
    },
)


FAILURE_CASES: tuple[dict[str, str], ...] = (
    {
        "failure_type": "post_commit_no_candidate",
        "example": "Stage34 root-cause diagnostics",
        "artifact": "docs/car_model/meshprior_stage34_post_commit_refresh_report.md",
        "paper_note": "After topology sync, recent protection can mark all survivors and erase the normal candidate pool.",
        "decision": "Use relaxed post-commit discovery only behind strict retained-edit controls.",
    },
    {
        "failure_type": "validation_rollback",
        "example": "M35 bonsai first four relaxed commits",
        "artifact": "outputs/carnet/meshprior/stage35_retained_refresh/mipnerf360_bonsai_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter_retry1/model/prism_debug/relaxed_retained_topology_audit.json",
        "paper_note": "A relaxed edit can pass the local proxy gate but still be rejected by recovery-window validation.",
        "decision": "Report active retained commits separately from total relaxed attempts.",
    },
    {
        "failure_type": "relaxed_commit_cap_reached",
        "example": "M35 courtyard retained cap",
        "artifact": "outputs/carnet/meshprior/stage35_retained_refresh/eth3d_courtyard_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model/prism_round_checkpoints/iter_001683_candidate_meta.json",
        "paper_note": "The conservative cap prevents further relaxed pruning once one active relaxed edit survives.",
        "decision": "Treat cap-1 as the safe default row; sweep higher caps only as future work.",
    },
    {
        "failure_type": "metric_path_mismatch",
        "example": "M36 training versus independent metric separation",
        "artifact": "outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_table.csv",
        "paper_note": "Training-time eval and independent render metrics differ and should not be substituted.",
        "decision": "Use independent render.py + metrics.py values in paper tables.",
    },
    {
        "failure_type": "dataset_geometry_observability",
        "example": "Tanks mirror lacks true sparse COLMAP tracks",
        "artifact": "docs/car_model/meshprior_stage25_multidataset_validation_report.md",
        "paper_note": "The available Tanks mirror can test trainability but not sparse-track geometry claims.",
        "decision": "Use Mip-NeRF 360 and ETH3D for geometry-observable claims until Tanks COLMAP tracks are rebuilt.",
    },
    {
        "failure_type": "perceptual_metric_tradeoff",
        "example": "M35 courtyard improves PSNR/SSIM/topology but not LPIPS versus M32/M33",
        "artifact": "outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_table.md",
        "paper_note": "The method should be framed as topology-quality control, not universal dominance on every image metric.",
        "decision": "State metric tradeoffs explicitly in claim wording.",
    },
)


CLAIM_WORDING = {
    "safe_claim": (
        "PRISM is an auditable topology-control layer for mesh-splatting optimization. "
        "On the selected public COLMAP scenes, the retained relaxed refresh variant reduces final mesh topology "
        "while preserving or improving key independent render metrics on bonsai and improving topology/PSNR/SSIM on courtyard."
    ),
    "do_not_claim": (
        "Do not claim universal image-quality dominance: courtyard LPIPS is worse than selected earlier rows, "
        "and Tanks geometry evidence is not yet paper-grade without real sparse tracks."
    ),
    "next_training_decision": (
        "Do not start full-budget public-scene training yet. The next highest-value step is to polish visual/failure figures "
        "and then run one full-budget Stage35 public scene only after the paper table/figure needs are fixed."
    ),
}


def _make_panel(model_path: Path, iteration: int, out_path: Path, max_views: int = 4) -> bool:
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
            y0 = row_idx * (thumb_h + label_h)
            canvas.paste(img, (x + (thumb_w - img.width) // 2, y0 + label_h + (thumb_h - img.height) // 2))
            draw.text((x + 6, y0 + 4), f"{kind}: {img_path.name}", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return True


def _write_failure_table(out_dir: Path) -> None:
    csv_path = out_dir / "failure_case_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(FAILURE_CASES[0].keys()))
        writer.writeheader()
        writer.writerows(FAILURE_CASES)
    with (out_dir / "failure_case_table.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Failure Case Table\n\n")
        f.write("| failure type | example | artifact | paper note | decision |\n")
        f.write("|---|---|---|---|---|\n")
        for row in FAILURE_CASES:
            f.write(
                "| {failure_type} | {example} | `{artifact}` | {paper_note} | {decision} |\n".format(**row)
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels: list[dict[str, str]] = []
    for row in PANEL_ROWS:
        panel_path = out_dir / "visual_panels" / f"{row['label']}.png"
        ok = _make_panel(Path(row["model_path"]), int(row["iteration"]), panel_path)
        panels.append(
            {
                "label": row["label"],
                "scene": row["scene"],
                "panel_path": str(panel_path) if ok else "",
                "status": "created" if ok else "missing_render_or_pil",
            }
        )
    _write_failure_table(out_dir)
    package = {
        "status": "PASS",
        "panels": panels,
        "failure_cases": list(FAILURE_CASES),
        "claim_wording": CLAIM_WORDING,
    }
    (out_dir / "visual_failure_package.json").write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "paper_claim_wording.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Paper Claim Wording\n\n")
        for key, value in CLAIM_WORDING.items():
            f.write(f"## {key}\n\n{value}\n\n")
    return package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package MeshPrior visual panels and failure cases.")
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/stage37_visual_failure_package")
    return parser


def main() -> None:
    package = run(build_parser().parse_args())
    print(json.dumps({"status": package["status"], "panels": package["panels"]}, indent=2))


if __name__ == "__main__":
    main()
