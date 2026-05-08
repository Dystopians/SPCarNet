#!/usr/bin/env python3
"""Apply a bounded surface-attached residual DC delta to a MeshSplatting checkpoint.

This is an ECSR Phase-D Version-2 MVP. It does not edit rendered images. It
reads train-only Phase-A residual support statistics, attaches a small bounded
RGB residual to the SH DC coefficient of vertices incident to those surface
faces, and writes a new checkpoint that can be rendered by the existing
MeshSplatting renderer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import copy_model_metadata, checkpoint_path, validate_faces
from utils.sh_utils import C0


VERTEX_KEYS = ("triangles_points", "vertex_weight", "features_dc", "features_rest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model", type=Path, required=True)
    parser.add_argument("--evidence_csv", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--top_k", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=0.025)
    parser.add_argument("--min_view_hits", type=int, default=1)
    parser.add_argument("--min_consistency", type=float, default=0.0)
    parser.add_argument("--copy_results", action="store_true")
    return parser.parse_args()


def read_evidence(path: Path, *, top_k: int, min_view_hits: int, min_consistency: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            view_hits = int(float(row.get("view_hits", 0)))
            consistency = float(row.get("residual_consistency", 0.0))
            if view_hits < int(min_view_hits) or consistency < float(min_consistency):
                continue
            rows.append(
                {
                    "rank": int(row["rank"]),
                    "face_id": int(row["face_id"]),
                    "score": float(row["score"]),
                    "pixel_count": float(row["pixel_count"]),
                    "view_hits": view_hits,
                    "consistency": consistency,
                    "residual_rgb": np.asarray(
                        [
                            float(row["mean_residual_r"]),
                            float(row["mean_residual_g"]),
                            float(row["mean_residual_b"]),
                        ],
                        dtype=np.float32,
                    ),
                }
            )
    rows.sort(key=lambda x: (float(x["score"]), float(x["pixel_count"])), reverse=True)
    return rows[: int(top_k)]


def copy_extra_metadata(source_model: Path, output_model: Path, *, copy_results: bool) -> None:
    copy_model_metadata(source_model, output_model)
    if copy_results:
        for name in ("results.json", "per_view.json"):
            src = source_model / name
            if src.is_file():
                shutil.copy2(src, output_model / name)


def main() -> int:
    args = parse_args()
    source_checkpoint = checkpoint_path(args.source_model, args.iteration)
    output_checkpoint = args.output_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud_state_dict.pt"
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    copy_extra_metadata(args.source_model, args.output_model, copy_results=bool(args.copy_results))

    state = torch.load(source_checkpoint, map_location="cpu")
    faces = state["_triangle_indices"].detach().cpu().long()
    vertices = state["triangles_points"].detach().cpu()
    features_dc = state["features_dc"].detach().cpu().clone().float()
    if features_dc.ndim != 3 or features_dc.shape[1] != 1 or features_dc.shape[2] != 3:
        raise ValueError(f"expected features_dc shape [V,1,3], got {tuple(features_dc.shape)}")

    evidence = read_evidence(
        args.evidence_csv,
        top_k=int(args.top_k),
        min_view_hits=int(args.min_view_hits),
        min_consistency=float(args.min_consistency),
    )
    accum = torch.zeros((features_dc.shape[0], 3), dtype=torch.float32)
    weights = torch.zeros((features_dc.shape[0], 1), dtype=torch.float32)
    used_faces = []
    skipped_faces = []
    for row in evidence:
        face_id = int(row["face_id"])
        if face_id < 0 or face_id >= int(faces.shape[0]):
            skipped_faces.append(face_id)
            continue
        vertex_ids = faces[face_id]
        if int(vertex_ids.min()) < 0 or int(vertex_ids.max()) >= int(vertices.shape[0]):
            skipped_faces.append(face_id)
            continue
        residual_rgb = np.clip(
            float(args.strength) * row["residual_rgb"],
            -float(args.max_abs_delta_rgb),
            float(args.max_abs_delta_rgb),
        ).astype(np.float32)
        delta_sh = torch.from_numpy(residual_rgb / float(C0)).to(torch.float32)
        weight = float(max(row["pixel_count"], 1.0) * max(row["consistency"], 1e-3))
        accum[vertex_ids] += delta_sh[None, :] * weight
        weights[vertex_ids] += weight
        used_faces.append(
            {
                "face_id": face_id,
                "rank": int(row["rank"]),
                "weight": weight,
                "residual_rgb": row["residual_rgb"].tolist(),
                "applied_delta_rgb": residual_rgb.tolist(),
                "vertices": [int(x) for x in vertex_ids.tolist()],
            }
        )

    active_vertices = weights.squeeze(-1) > 0
    vertex_delta = torch.zeros_like(accum)
    vertex_delta[active_vertices] = accum[active_vertices] / weights[active_vertices].clamp_min(1e-8)
    max_delta_sh = float(args.max_abs_delta_rgb) / float(C0)
    vertex_delta = torch.clamp(vertex_delta, min=-max_delta_sh, max=max_delta_sh)
    features_dc[active_vertices, 0, :] += vertex_delta[active_vertices]

    out_state: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            out_state[key] = value.detach().cpu().clone()
        else:
            out_state[key] = value
    out_state["features_dc"] = features_dc.to(dtype=state["features_dc"].dtype)
    torch.save(out_state, output_checkpoint)

    degenerate, invalid = validate_faces(out_state["triangles_points"], out_state["_triangle_indices"])
    delta_rgb = vertex_delta[active_vertices] * float(C0)
    audit = {
        "operator": "surface_residual_dc_delta",
        "test_usage": "none",
        "source_model": str(args.source_model),
        "source_checkpoint": str(source_checkpoint),
        "output_model": str(args.output_model),
        "output_checkpoint": str(output_checkpoint),
        "iteration": int(args.iteration),
        "evidence_csv": str(args.evidence_csv),
        "top_k_requested": int(args.top_k),
        "faces_used": len(used_faces),
        "faces_skipped": len(skipped_faces),
        "vertices_modified": int(active_vertices.sum().item()),
        "strength": float(args.strength),
        "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
        "delta_rgb_abs_mean": float(delta_rgb.abs().mean().item()) if delta_rgb.numel() else 0.0,
        "delta_rgb_abs_max": float(delta_rgb.abs().max().item()) if delta_rgb.numel() else 0.0,
        "topology_before": {
            "triangles": int(faces.shape[0]),
            "vertices": int(vertices.shape[0]),
        },
        "topology_after": {
            "triangles": int(out_state["_triangle_indices"].shape[0]),
            "vertices": int(out_state["triangles_points"].shape[0]),
            "degenerate_face_count": int(degenerate),
            "invalid_index_count": int(invalid),
        },
        "used_faces_preview": used_faces[:20],
    }
    (args.output_model / "surface_residual_delta_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_model / "surface_residual_delta_audit.md").write_text(
        "\n".join(
            [
                "# ECSR Surface Residual Delta Audit",
                "",
                f"- operator: `{audit['operator']}`",
                f"- iteration: `{audit['iteration']}`",
                f"- faces used: `{audit['faces_used']}`",
                f"- vertices modified: `{audit['vertices_modified']}`",
                f"- strength: `{audit['strength']}`",
                f"- max abs delta RGB: `{audit['max_abs_delta_rgb']}`",
                f"- delta RGB abs mean: `{audit['delta_rgb_abs_mean']:.6f}`",
                f"- delta RGB abs max: `{audit['delta_rgb_abs_max']:.6f}`",
                f"- topology unchanged: `{audit['topology_before'] == {k: audit['topology_after'][k] for k in ('triangles', 'vertices')}}`",
                f"- degenerate faces: `{degenerate}`",
                f"- invalid indices: `{invalid}`",
                "",
                "The residual is attached to checkpoint SH DC coefficients. No rendered image is edited.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))
    return 0 if degenerate == 0 and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
