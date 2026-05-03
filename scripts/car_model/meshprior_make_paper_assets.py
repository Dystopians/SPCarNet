"""Build final paper-facing MeshPrior table, captions, and limitations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SELECTED_LABELS = (
    "parking_m24_2_retention_7000",
    "bonsai_m33_diverse_calib",
    "bonsai_m35_retained_relaxed",
    "courtyard_m32_measured_rank",
    "courtyard_m35_retained_relaxed",
)


CAPTIONS = {
    "parking_m24_2_retention_7000": (
        "Parking-phone scene, M24.2 topology-retention row. The panel compares independent renders against held-out "
        "ground truth after late PRISM editing and densification freeze, illustrating the single-scene long-budget "
        "topology-control behavior."
    ),
    "bonsai_m35_retained_relaxed": (
        "Mip-NeRF 360 bonsai, M35 retained relaxed refresh. One relaxed post-commit edit survives final validation, "
        "reducing final topology versus Stage33 while improving independent PSNR, SSIM, and LPIPS."
    ),
    "courtyard_m35_retained_relaxed": (
        "ETH3D courtyard, M35 retained relaxed refresh. The method keeps one active relaxed edit and improves topology, "
        "PSNR, and SSIM among selected rows, while LPIPS remains a reported tradeoff."
    ),
}


LIMITATIONS = (
    "The evidence supports PRISM as an auditable topology-control layer, not a universal image-quality optimizer.",
    "Independent render.py + metrics.py values are the paper-facing metrics; training-time eval values are diagnostic.",
    "M35 improves all selected independent metrics on bonsai, but courtyard LPIPS is worse than selected M32/M33 rows.",
    "The available Tanks and Temples mirror is not geometry-observable enough for sparse-track geometry claims.",
    "The current full-budget public-scene decision is no-go until the table and figure story is stable; one full-budget Stage35 public-scene run should be launched only if it fills a concrete paper-table gap.",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def _selected_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_label = {row["label"]: row for row in rows}
    missing = [label for label in SELECTED_LABELS if label not in by_label]
    if missing:
        raise FileNotFoundError(f"missing selected labels: {missing}")
    return [by_label[label] for label in SELECTED_LABELS]


def _write_final_table(rows: list[dict[str, str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Final MeshPrior Paper Table\n\n")
        f.write("All image metrics are independent `render.py + metrics.py` values. Lower LPIPS is better.\n\n")
        f.write("| scene | row | topology | PSNR | SSIM | LPIPS | audit note |\n")
        f.write("|---|---|---:|---:|---:|---:|---|\n")
        for row in rows:
            audit_note = "baseline/earlier row"
            if row["label"] == "bonsai_m35_retained_relaxed":
                audit_note = "1 active relaxed edit; 4 validation rollbacks recorded"
            elif row["label"] == "courtyard_m35_retained_relaxed":
                audit_note = "1 active relaxed edit; cap reached later"
            elif row["label"] == "parking_m24_2_retention_7000":
                audit_note = "long-budget parking topology-retention evidence"
            f.write(
                "| {scene} | {method} | {triangles} | {psnr:.6f} | {ssim:.6f} | {lpips:.6f} | {audit_note} |\n".format(
                    scene=row["scene"],
                    method=row["method"],
                    triangles=row["final_triangles"],
                    psnr=_float(row, "independent_psnr"),
                    ssim=_float(row, "independent_ssim"),
                    lpips=_float(row, "independent_lpips"),
                    audit_note=audit_note,
                )
            )


def _write_captions(out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Figure Captions\n\n")
        for label, caption in CAPTIONS.items():
            f.write(f"## {label}\n\n{caption}\n\n")


def _write_limitations(out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Limitations\n\n")
        for item in LIMITATIONS:
            f.write(f"- {item}\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _selected_rows(_read_rows(Path(args.metric_table)))
    _write_final_table(rows, out_dir / "final_paper_table.md")
    _write_captions(out_dir / "figure_captions.md")
    _write_limitations(out_dir / "limitations.md")
    decision = {
        "full_budget_public_scene_training": "NO_GO_FOR_NOW",
        "reason": (
            "The immediate blocker is paper-asset clarity, not missing short-run evidence. "
            "Run one full-budget Stage35 public scene only after the final table identifies a specific missing row."
        ),
        "requires_wandb_if_revisited": True,
        "requires_gpu_check_if_revisited": True,
    }
    package = {
        "status": "PASS",
        "selected_labels": list(SELECTED_LABELS),
        "training_decision": decision,
        "outputs": {
            "final_table": str(out_dir / "final_paper_table.md"),
            "captions": str(out_dir / "figure_captions.md"),
            "limitations": str(out_dir / "limitations.md"),
        },
    }
    (out_dir / "paper_assets_package.json").write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    return package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final MeshPrior paper assets.")
    parser.add_argument(
        "--metric_table",
        default="outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_table.csv",
    )
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/stage38_paper_assets")
    return parser


def main() -> None:
    package = run(build_parser().parse_args())
    print(json.dumps({"status": package["status"], "training_decision": package["training_decision"]}, indent=2))


if __name__ == "__main__":
    main()
