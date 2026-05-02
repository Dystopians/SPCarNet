"""Create topology-controlled checkpoint-copy ablations for MeshPrior parking runs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import torch


VERTEX_KEYS = ("triangles_points", "vertex_weight", "features_dc", "features_rest")
FACE_KEYS = ("importance_score", "image_size", "pixel_count")


def _triangle_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    tri = vertices.float()[faces.long()]
    return torch.linalg.norm(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1), dim=1) * 0.5


def _compact_state(state: dict[str, Any], keep_faces: torch.Tensor) -> tuple[dict[str, Any], dict[str, int]]:
    faces = state["_triangle_indices"].detach().cpu().long()
    face_count_before = int(faces.shape[0])
    kept_faces = faces[keep_faces]
    used_vertices = torch.unique(kept_faces.reshape(-1), sorted=True)
    remap = torch.full((int(faces.max().item()) + 1,), -1, dtype=torch.long)
    remap[used_vertices] = torch.arange(len(used_vertices), dtype=torch.long)
    new_faces = remap[kept_faces].to(dtype=state["_triangle_indices"].dtype)

    out: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            if key == "_triangle_indices":
                out[key] = new_faces.clone()
            elif key in VERTEX_KEYS and value.shape[0] == remap.shape[0]:
                out[key] = value.detach().cpu()[used_vertices].clone()
            elif key in FACE_KEYS and value.shape[0] == face_count_before:
                out[key] = value.detach().cpu()[keep_faces].clone()
            else:
                out[key] = value.detach().cpu().clone()
        else:
            out[key] = value
    stats = {
        "face_count_before": face_count_before,
        "face_count_after": int(new_faces.shape[0]),
        "faces_removed": int(face_count_before - new_faces.shape[0]),
        "vertex_count_before": int(state["triangles_points"].shape[0]),
        "vertex_count_after": int(out["triangles_points"].shape[0]),
        "vertices_removed": int(state["triangles_points"].shape[0] - out["triangles_points"].shape[0]),
    }
    return out, stats


def _copy_model_metadata(source_model: Path, output_model: Path) -> None:
    output_model.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "cameras.json", "input.ply"):
        src = source_model / name
        if src.is_file():
            shutil.copy2(src, output_model / name)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_model = Path(args.source_model)
    source_checkpoint = Path(args.source_checkpoint)
    output_model = Path(args.output_model)
    output_iteration_dir = output_model / "point_cloud" / f"iteration_{args.iteration}"
    output_iteration_dir.mkdir(parents=True, exist_ok=True)
    _copy_model_metadata(source_model, output_model)

    state = torch.load(source_checkpoint, map_location="cpu")
    faces = state["_triangle_indices"].detach().cpu().long()
    vertices = state["triangles_points"].detach().cpu().float()
    areas = _triangle_areas(vertices, faces)

    prune_fraction = float(args.prune_fraction)
    if not 0.0 <= prune_fraction < 1.0:
        raise ValueError("--prune_fraction must be in [0, 1)")
    keep_count = max(int(args.min_keep_triangles), int(round(faces.shape[0] * (1.0 - prune_fraction))))
    keep_count = min(keep_count, int(faces.shape[0]))
    keep_ids = torch.topk(areas, k=keep_count, largest=True, sorted=False).indices
    keep_faces = torch.zeros((faces.shape[0],), dtype=torch.bool)
    keep_faces[keep_ids] = True

    out_state, stats = _compact_state(state, keep_faces)
    output_checkpoint = output_iteration_dir / "point_cloud_state_dict.pt"
    torch.save(out_state, output_checkpoint)

    removed_areas = areas[~keep_faces]
    kept_areas = areas[keep_faces]
    report = {
        "source_model": str(source_model),
        "source_checkpoint": str(source_checkpoint),
        "output_model": str(output_model),
        "output_checkpoint": str(output_checkpoint),
        "iteration": int(args.iteration),
        "source_model_edited": False,
        "checkpoint_copy_edited": True,
        "method": "remove_smallest_area_triangles",
        "prune_fraction_requested": prune_fraction,
        "min_keep_triangles": int(args.min_keep_triangles),
        **stats,
        "actual_prune_fraction": float(stats["faces_removed"] / max(stats["face_count_before"], 1)),
        "area_summary": {
            "all_min": float(areas.min().item()),
            "all_max": float(areas.max().item()),
            "all_mean": float(areas.mean().item()),
            "kept_min": float(kept_areas.min().item()) if kept_areas.numel() else float("nan"),
            "kept_mean": float(kept_areas.mean().item()) if kept_areas.numel() else float("nan"),
            "removed_max": float(removed_areas.max().item()) if removed_areas.numel() else float("nan"),
            "removed_mean": float(removed_areas.mean().item()) if removed_areas.numel() else float("nan"),
        },
        "notes": [
            "This is a post-training checkpoint-copy ablation; the source model is not overwritten.",
            "The ranking signal is triangle world-space area because saved runtime importance/image_size/pixel_count are zero in the current 7000 checkpoint.",
            "Render, COLMAP proxy geometry, FPS/topology, and W&B/external logging are required before treating this as an improvement.",
        ],
    }
    (output_model / "topology_control_ablation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_model / "topology_control_ablation_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Topology Control Ablation Report\n\n")
        f.write("- source model edited: `false`\n")
        f.write("- checkpoint copy edited: `true`\n")
        f.write("- method: `remove_smallest_area_triangles`\n")
        f.write(f"- iteration: `{args.iteration}`\n")
        f.write(f"- prune fraction requested: `{prune_fraction}`\n")
        f.write(f"- triangles: `{stats['face_count_before']}` -> `{stats['face_count_after']}`\n")
        f.write(f"- vertices: `{stats['vertex_count_before']}` -> `{stats['vertex_count_after']}`\n")
    with (output_model / "topology_control_ablation_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "prune_fraction_requested",
                "actual_prune_fraction",
                "face_count_before",
                "face_count_after",
                "vertex_count_before",
                "vertex_count_after",
            ],
        )
        writer.writeheader()
        writer.writerow({key: report[key] for key in writer.fieldnames})
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create topology-controlled checkpoint-copy ablations.")
    parser.add_argument(
        "--source_model",
        default="outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/current_branch_7000iter/model",
    )
    parser.add_argument(
        "--source_checkpoint",
        default="outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/current_branch_7000iter/model/point_cloud/iteration_7000/point_cloud_state_dict.pt",
    )
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--prune_fraction", type=float, required=True)
    parser.add_argument("--min_keep_triangles", type=int, default=1)
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "output_model": report["output_model"],
                "triangles": [report["face_count_before"], report["face_count_after"]],
                "vertices": [report["vertex_count_before"], report["vertex_count_after"]],
                "actual_prune_fraction": report["actual_prune_fraction"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
