"""Apply accepted parking patch cleanup proposals to a copied triangle checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


VERTEX_KEYS = ("triangles_points", "vertex_weight", "features_dc", "features_rest")
FACE_KEYS = ("importance_score", "image_size", "pixel_count")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _accepted_cleanup_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in report.get("results", [])
        if bool(row.get("accepted")) and str(row.get("proposal_type")) == "component_cleanup_candidate"
    ]


def _removed_local_face_indices(before_npz: str, after_npz: str) -> np.ndarray:
    before = np.load(before_npz, allow_pickle=False)
    after = np.load(after_npz, allow_pickle=False)
    before_faces = np.asarray(before["faces"], dtype=np.int64)
    after_faces = {tuple(face.tolist()) for face in np.asarray(after["faces"], dtype=np.int64)}
    removed = [i for i, face in enumerate(before_faces) if tuple(face.tolist()) not in after_faces]
    return np.asarray(removed, dtype=np.int64)


def _patch_original_faces(patch_path: str) -> np.ndarray:
    patch = np.load(patch_path, allow_pickle=False)
    return np.asarray(patch["original_face_indices"], dtype=np.int64)


def _collect_removed_faces(cleanup_rows: list[dict[str, Any]], patch_rows: dict[str, dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    removed_global: list[int] = []
    application_rows: list[dict[str, Any]] = []
    for row in cleanup_rows:
        region_id = str(row["region_id"])
        patch_path = patch_rows[region_id]["patch_path"]
        original_faces = _patch_original_faces(patch_path)
        removed_local = _removed_local_face_indices(row["before_npz"], row["after_npz"])
        global_faces = original_faces[removed_local]
        removed_global.extend(int(x) for x in global_faces.tolist())
        application_rows.append(
            {
                "region_id": region_id,
                "proposal_id": row["proposal_id"],
                "patch_path": patch_path,
                "removed_local_faces": int(len(removed_local)),
                "removed_global_faces": int(len(global_faces)),
                "min_global_face": int(global_faces.min()) if len(global_faces) else -1,
                "max_global_face": int(global_faces.max()) if len(global_faces) else -1,
            }
        )
    return np.asarray(sorted(set(removed_global)), dtype=np.int64), application_rows


def _copy_checkpoint_with_removed_faces(state: dict[str, Any], removed_faces: np.ndarray) -> tuple[dict[str, Any], dict[str, int]]:
    faces = state["_triangle_indices"].detach().cpu().long()
    face_count_before = int(faces.shape[0])
    keep_faces = torch.ones((face_count_before,), dtype=torch.bool)
    if len(removed_faces):
        keep_faces[torch.as_tensor(removed_faces, dtype=torch.long)] = False
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    proposal_report = _load_json(Path(args.patch_proposal_report))
    patch_summary = _load_json(Path(args.mesh_patch_summary))
    patch_rows = {str(row["region_id"]): row for row in patch_summary.get("patches", [])}
    cleanup_rows = _accepted_cleanup_rows(proposal_report)
    if args.max_applications > 0:
        cleanup_rows = cleanup_rows[: args.max_applications]
    removed_faces, application_rows = _collect_removed_faces(cleanup_rows, patch_rows)
    state = torch.load(args.triangle_state, map_location="cpu")
    copied_state, stats = _copy_checkpoint_with_removed_faces(state, removed_faces)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_checkpoint = out / "point_cloud_state_dict.pt"
    torch.save(copied_state, output_checkpoint)
    report = {
        "source_triangle_state": str(args.triangle_state),
        "output_triangle_state": str(output_checkpoint),
        "source_model_edited": False,
        "checkpoint_copy_edited": True,
        "accepted_cleanup_applications": len(cleanup_rows),
        "unique_removed_faces": int(len(removed_faces)),
        **stats,
        "applications": application_rows,
        "notes": [
            "The baseline checkpoint is not overwritten.",
            "The output checkpoint compacts vertices and per-face arrays after removing accepted copied-patch cleanup faces.",
            "This is an integrity/writeback test; render and geometry evaluation are required before treating it as a scene improvement.",
        ],
    }
    (out / "checkpoint_copy_application_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (out / "checkpoint_copy_application_rows.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "region_id",
            "proposal_id",
            "patch_path",
            "removed_local_faces",
            "removed_global_faces",
            "min_global_face",
            "max_global_face",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(application_rows)
    with (out / "checkpoint_copy_application_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Parking Checkpoint-Copy Application Report\n\n")
        f.write("- source model edited: `false`\n")
        f.write("- checkpoint copy edited: `true`\n")
        f.write(f"- cleanup applications: `{len(cleanup_rows)}`\n")
        f.write(f"- unique removed faces: `{len(removed_faces)}`\n")
        f.write(f"- faces: `{stats['face_count_before']}` -> `{stats['face_count_after']}`\n")
        f.write(f"- vertices: `{stats['vertex_count_before']}` -> `{stats['vertex_count_after']}`\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply accepted parking patch cleanup proposals to a copied checkpoint.")
    parser.add_argument("--patch_proposal_report", default="outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_report.json")
    parser.add_argument("--mesh_patch_summary", default="outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json")
    parser.add_argument(
        "--triangle_state",
        default="outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model/point_cloud/iteration_200/point_cloud_state_dict.pt",
    )
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup")
    parser.add_argument("--max_applications", type=int, default=0)
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "accepted_cleanup_applications": report["accepted_cleanup_applications"],
                "unique_removed_faces": report["unique_removed_faces"],
                "faces": [report["face_count_before"], report["face_count_after"]],
                "vertices": [report["vertex_count_before"], report["vertex_count_after"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
