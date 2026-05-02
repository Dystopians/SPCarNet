"""Prepare an evaluation-ready model directory for a copied parking checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_model = Path(args.source_model)
    copied_checkpoint = Path(args.copied_checkpoint)
    output_model = Path(args.output_model)
    iteration_dir = output_model / "point_cloud" / f"iteration_{args.iteration}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    for name in ("cfg_args", "cameras.json", "input.ply"):
        src = source_model / name
        if src.is_file():
            shutil.copy2(src, output_model / name)
    shutil.copy2(copied_checkpoint, iteration_dir / "point_cloud_state_dict.pt")

    state = torch.load(iteration_dir / "point_cloud_state_dict.pt", map_location="cpu")
    report = {
        "source_model": str(source_model),
        "copied_checkpoint": str(copied_checkpoint),
        "output_model": str(output_model),
        "iteration": int(args.iteration),
        "source_model_edited": False,
        "recovery_model_written": True,
        "triangles": int(state["_triangle_indices"].shape[0]),
        "vertices": int(state["triangles_points"].shape[0]),
        "files": {
            "cfg_args": str(output_model / "cfg_args"),
            "cameras_json": str(output_model / "cameras.json"),
            "input_ply": str(output_model / "input.ply"),
            "checkpoint": str(iteration_dir / "point_cloud_state_dict.pt"),
        },
        "notes": [
            "This directory mirrors the trained model layout expected by Scene(load_iteration=...).",
            "It is a recovery/evaluation model for the copied checkpoint; the baseline model is not overwritten.",
        ],
    }
    output_model.mkdir(parents=True, exist_ok=True)
    (output_model / "meshprior_recovery_model_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_model / "meshprior_recovery_model_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Parking Recovery Model Report\n\n")
        f.write("- source model edited: `false`\n")
        f.write("- recovery model written: `true`\n")
        f.write(f"- iteration: `{args.iteration}`\n")
        f.write(f"- triangles: `{report['triangles']}`\n")
        f.write(f"- vertices: `{report['vertices']}`\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare an evaluation-ready model directory for a copied parking checkpoint.")
    parser.add_argument("--source_model", default="outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model")
    parser.add_argument("--copied_checkpoint", default="outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/point_cloud_state_dict.pt")
    parser.add_argument("--output_model", default="outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup")
    parser.add_argument("--iteration", type=int, default=200)
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps({"output_model": report["output_model"], "triangles": report["triangles"], "vertices": report["vertices"]}, indent=2))


if __name__ == "__main__":
    main()
