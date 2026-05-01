"""Generate MeshPrior NeurIPS-style report tables from matrix results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


OBJECT_COLUMNS = [
    "method",
    "output_type",
    "recon_chamfer_l1",
    "hidden_chamfer_l1",
    "visible_preservation_error",
    "zero_corruption_chamfer",
    "free_space_violation",
    "mesh_extraction_success",
    "inference_time",
    "status",
]
SYNTH_COLUMNS = [
    "method",
    "damage_type",
    "hole_closure",
    "floater_prune_precision",
    "floater_prune_recall",
    "valid_surface_protect_recall",
    "visible_preservation",
    "free_space_violation",
    "triangle_count_delta",
    "status",
]
SCENE_COLUMNS = [
    "method",
    "scene",
    "checkpoint_iteration",
    "psnr",
    "ssim",
    "lpips",
    "colmap_absrel",
    "sparse_depth_mae",
    "normal_mean_angle",
    "triangle_count",
    "controlled_fps",
    "car_roi_hole_floater_metrics",
    "accepted_proposals",
    "rejected_proposals",
    "status",
]
ABLATION_COLUMNS = ["row", "expected_status", "available_evidence", "risk"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def _object_rows(matrix: dict) -> list[dict]:
    rows = []
    for exp in matrix["experiments"]:
        if exp["group"] != "object":
            continue
        m = exp.get("metrics", {})
        rows.append({"method": exp["method"], "output_type": exp.get("output_type", ""), "status": exp["status"], **m})
    return rows


def _synthetic_rows(matrix: dict) -> list[dict]:
    rows = []
    for exp in matrix["experiments"]:
        if exp["group"] != "synthetic":
            continue
        if exp.get("synthetic_rows"):
            for row in exp["synthetic_rows"]:
                rows.append({"method": row.get("method", exp["method"]), "status": exp["status"], **row})
        else:
            m = exp.get("metrics", {})
            rows.append(
                {
                    "method": exp["method"],
                    "damage_type": m.get("damage_type"),
                    "hole_closure": m.get("boundary_edge_delta_sum"),
                    "floater_prune_precision": m.get("floater_prune_precision"),
                    "floater_prune_recall": m.get("floater_prune_recall"),
                    "valid_surface_protect_recall": m.get("valid_surface_protect_recall"),
                    "visible_preservation": m.get("visible_preservation_error"),
                    "free_space_violation": m.get("free_space_violation_delta_max"),
                    "triangle_count_delta": m.get("triangle_count_delta_sum"),
                    "status": exp["status"],
                }
            )
    return rows


def _scene_rows(matrix: dict) -> list[dict]:
    rows = []
    for exp in matrix["experiments"]:
        if exp["group"] != "scene":
            continue
        rows.append({"method": exp["method"], "status": exp["status"], **exp.get("metrics", {})})
    return rows


def _ablation_rows(matrix: dict) -> list[dict]:
    available_ids = {exp["id"]: exp for exp in matrix["experiments"] if exp["status"] == "AVAILABLE"}
    return [
        {"row": "direct insert, no gate", "expected_status": "not approved", "available_evidence": "not run", "risk": "hallucination/free-space"},
        {"row": "prior score only", "expected_status": "diagnostic", "available_evidence": "object prior metrics", "risk": "object confidence without scene evidence"},
        {"row": "prior + free-space gate", "expected_status": "partial", "available_evidence": "dry-run free-space deltas", "risk": "no render gate"},
        {"row": "prior + geometry gate", "expected_status": "available" if "scene_baseline_meshprior_proposals" in available_ids else "missing", "available_evidence": "M9/M11 dry-run gate", "risk": "dry-run only"},
        {"row": "prior + render gate", "expected_status": "missing", "available_evidence": "not connected", "risk": "render degradation unknown"},
        {"row": "full gated method", "expected_status": "partial", "available_evidence": "proposal gate + cleanup-repaired training smoke", "risk": "real scene integration incomplete"},
    ]


def _markdown_table(columns: list[str], rows: list[dict], limit: int = 12) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:limit]:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(out)


def run(args: argparse.Namespace) -> dict[str, str]:
    matrix = _load_json(Path(args.matrix_results))
    out_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    object_rows = _object_rows(matrix)
    synthetic_rows = _synthetic_rows(matrix)
    scene_rows = _scene_rows(matrix)
    ablation_rows = _ablation_rows(matrix)
    failures = [exp for exp in matrix["experiments"] if exp["status"] == "MISSING"]

    paths = {
        "object_table": out_dir / "object_table.csv",
        "synthetic_damage_table": out_dir / "synthetic_damage_table.csv",
        "scene_table": out_dir / "scene_table.csv",
        "ablation_table": out_dir / "ablation_table.csv",
        "failure_cases": out_dir / "failure_cases.md",
        "main_report": report_dir / "meshprior_neurips_main_report.md",
    }
    _write_csv(paths["object_table"], OBJECT_COLUMNS, object_rows)
    _write_csv(paths["synthetic_damage_table"], SYNTH_COLUMNS, synthetic_rows)
    _write_csv(paths["scene_table"], SCENE_COLUMNS, scene_rows)
    _write_csv(paths["ablation_table"], ABLATION_COLUMNS, ablation_rows)

    paths["failure_cases"].parent.mkdir(parents=True, exist_ok=True)
    with paths["failure_cases"].open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Failure Cases\n\n")
        if not failures:
            f.write("No missing experiments in this matrix run.\n")
        for exp in failures:
            f.write(f"- `{exp['id']}`: {exp.get('missing_reason', 'missing')}\n")

    paths["main_report"].parent.mkdir(parents=True, exist_ok=True)
    with paths["main_report"].open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Evaluation Matrix Report\n\n")
        f.write("This report separates object-prior quality, synthetic repair, scene metrics, and safety ablations. Missing experiments are retained as `MISSING` rather than dropped.\n\n")
        f.write("## Matrix Status\n\n")
        f.write(json.dumps(matrix.get("counts", {}), indent=2) + "\n\n")
        f.write("## Table 1 — Object Prior Quality\n\n")
        f.write(_markdown_table(OBJECT_COLUMNS, object_rows) + "\n\n")
        f.write("## Table 2 — Synthetic Mesh Repair\n\n")
        f.write(_markdown_table(SYNTH_COLUMNS, synthetic_rows) + "\n\n")
        f.write("## Table 3 — Scene Mesh Optimization\n\n")
        f.write(_markdown_table(SCENE_COLUMNS, scene_rows) + "\n\n")
        f.write("## Table 4 — Safety Ablation\n\n")
        f.write(_markdown_table(ABLATION_COLUMNS, ablation_rows) + "\n\n")
        f.write("## Failure Cases\n\n")
        f.write(f"See `{paths['failure_cases']}`.\n")

    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return {k: str(v) for k, v in paths.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MeshPrior NeurIPS report tables.")
    parser.add_argument("--matrix_results", default="outputs/carnet/meshprior/experiment_matrix/matrix_results.json")
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/reports")
    parser.add_argument("--report_dir", default="docs/car_model/reports")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
