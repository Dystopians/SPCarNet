#!/usr/bin/env python3
"""Add bounded surface-attached residual microfacets from train-only evidence.

This Phase-D operator is intentionally different from a hyperparameter scan:
it changes the representation by adding a small number of residual carrier
triangles on top of multi-view stable high-error faces. The added facets are
shrunk copies of their source faces, inherit local appearance/opacity, and only
store a clipped SH-DC residual. Net triangle count can still stay below the
clean MeshSplatting baseline when this is applied after compaction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
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
FACE_KEYS = ("importance_score", "image_size", "pixel_count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model", type=Path, required=True)
    parser.add_argument("--evidence_csv", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--top_k", type=int, default=96)
    parser.add_argument("--min_view_hits", type=int, default=2)
    parser.add_argument("--min_consistency", type=float, default=0.95)
    parser.add_argument("--min_pixel_count", type=float, default=20.0)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=0.04)
    parser.add_argument("--micro_scale", type=float, default=0.70)
    parser.add_argument("--normal_offset", type=float, default=0.0005)
    parser.add_argument("--weight_logit_boost", type=float, default=0.5)
    return parser.parse_args()


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def read_evidence(
    path: Path,
    *,
    top_k: int,
    min_view_hits: int,
    min_consistency: float,
    min_pixel_count: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            view_hits = int(_float(row, "view_hits"))
            consistency = _float(row, "residual_consistency")
            pixel_count = _float(row, "pixel_count")
            if view_hits < int(min_view_hits):
                continue
            if consistency < float(min_consistency):
                continue
            if pixel_count < float(min_pixel_count):
                continue
            rows.append(
                {
                    "rank": int(_float(row, "rank")),
                    "face_id": int(_float(row, "face_id")),
                    "score": _float(row, "score"),
                    "pixel_count": pixel_count,
                    "view_hits": view_hits,
                    "consistency": consistency,
                    "residual_rgb": np.asarray(
                        [
                            _float(row, "mean_residual_r"),
                            _float(row, "mean_residual_g"),
                            _float(row, "mean_residual_b"),
                        ],
                        dtype=np.float32,
                    ),
                }
            )
    rows.sort(key=lambda x: (float(x["score"]), float(x["pixel_count"])), reverse=True)
    return rows[: int(top_k)]


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
        else:
            out[key] = value
    return out


def face_normals(points: torch.Tensor) -> torch.Tensor:
    normals = torch.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0], dim=1)
    norm = normals.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return normals / norm


def mean_edge_length(points: torch.Tensor) -> torch.Tensor:
    e01 = (points[:, 1] - points[:, 0]).norm(dim=1)
    e12 = (points[:, 2] - points[:, 1]).norm(dim=1)
    e20 = (points[:, 0] - points[:, 2]).norm(dim=1)
    return torch.stack([e01, e12, e20], dim=1).mean(dim=1, keepdim=True)


def main() -> int:
    args = parse_args()
    source_checkpoint = checkpoint_path(args.source_model, args.iteration)
    output_checkpoint = args.output_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud_state_dict.pt"
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    copy_model_metadata(args.source_model, args.output_model)

    state = torch.load(source_checkpoint, map_location="cpu")
    vertices = state["triangles_points"].detach().cpu().float()
    faces = state["_triangle_indices"].detach().cpu().long()
    features_dc = state["features_dc"].detach().cpu().float()
    features_rest = state["features_rest"].detach().cpu().float()
    vertex_weight = state["vertex_weight"].detach().cpu().float()

    evidence = read_evidence(
        args.evidence_csv,
        top_k=int(args.top_k),
        min_view_hits=int(args.min_view_hits),
        min_consistency=float(args.min_consistency),
        min_pixel_count=float(args.min_pixel_count),
    )
    valid_rows: list[dict[str, Any]] = []
    skipped_faces: list[int] = []
    for row in evidence:
        face_id = int(row["face_id"])
        if face_id < 0 or face_id >= int(faces.shape[0]):
            skipped_faces.append(face_id)
            continue
        vertex_ids = faces[face_id]
        if int(vertex_ids.min().item()) < 0 or int(vertex_ids.max().item()) >= int(vertices.shape[0]):
            skipped_faces.append(face_id)
            continue
        valid_rows.append(row)

    if valid_rows:
        face_ids = torch.as_tensor([int(row["face_id"]) for row in valid_rows], dtype=torch.long)
        src_faces = faces[face_ids]
        src_points = vertices[src_faces]
        centers = src_points.mean(dim=1, keepdim=True)
        normals = face_normals(src_points).unsqueeze(1)
        offsets = normals * mean_edge_length(src_points).view(-1, 1, 1) * float(args.normal_offset)
        micro_points = centers + float(args.micro_scale) * (src_points - centers) + offsets
        new_vertices = micro_points.reshape(-1, 3)

        residual_rgb = torch.as_tensor(
            np.stack([row["residual_rgb"] for row in valid_rows], axis=0),
            dtype=torch.float32,
        )
        delta_rgb = (residual_rgb * float(args.strength)).clamp(
            -float(args.max_abs_delta_rgb),
            float(args.max_abs_delta_rgb),
        )
        src_fdc = features_dc[src_faces].mean(dim=1)
        src_frest = features_rest[src_faces].mean(dim=1)
        src_weight = vertex_weight[src_faces].mean(dim=1) + float(args.weight_logit_boost)
        micro_fdc = (src_fdc + delta_rgb[:, None, :] / float(C0)).repeat_interleave(3, dim=0)
        micro_frest = src_frest.repeat_interleave(3, dim=0)
        micro_weight = src_weight.repeat_interleave(3, dim=0)

        start = int(vertices.shape[0])
        offsets_idx = torch.arange(0, 3 * len(valid_rows), 3, dtype=torch.long)[:, None]
        new_faces = start + offsets_idx + torch.tensor([[0, 1, 2]], dtype=torch.long)
    else:
        new_vertices = torch.empty((0, 3), dtype=torch.float32)
        micro_fdc = torch.empty((0,) + tuple(features_dc.shape[1:]), dtype=features_dc.dtype)
        micro_frest = torch.empty((0,) + tuple(features_rest.shape[1:]), dtype=features_rest.dtype)
        micro_weight = torch.empty((0,) + tuple(vertex_weight.shape[1:]), dtype=vertex_weight.dtype)
        new_faces = torch.empty((0, 3), dtype=torch.long)
        delta_rgb = torch.empty((0, 3), dtype=torch.float32)
        face_ids = torch.empty((0,), dtype=torch.long)

    out = clone_state(state)
    out["triangles_points"] = torch.cat([vertices, new_vertices], dim=0).to(dtype=state["triangles_points"].dtype)
    out["features_dc"] = torch.cat([features_dc, micro_fdc], dim=0).to(dtype=state["features_dc"].dtype)
    out["features_rest"] = torch.cat([features_rest, micro_frest], dim=0).to(dtype=state["features_rest"].dtype)
    out["vertex_weight"] = torch.cat([vertex_weight, micro_weight], dim=0).to(dtype=state["vertex_weight"].dtype)
    out["_triangle_indices"] = torch.cat([faces, new_faces], dim=0).to(dtype=state["_triangle_indices"].dtype)
    for key in FACE_KEYS:
        value = state.get(key)
        if not torch.is_tensor(value):
            continue
        if value.shape[0] != faces.shape[0]:
            continue
        if len(valid_rows):
            copied = value.detach().cpu()[face_ids].clone()
        else:
            copied = value.detach().cpu()[:0].clone()
        out[key] = torch.cat([value.detach().cpu(), copied], dim=0).to(dtype=value.dtype)

    torch.save(out, output_checkpoint)
    degenerate, invalid = validate_faces(out["triangles_points"], out["_triangle_indices"])
    added_faces = int(new_faces.shape[0])
    added_vertices = int(new_vertices.shape[0])
    audit = {
        "operator": "surface_residual_microfacets",
        "test_usage": "none",
        "source_model": str(args.source_model),
        "source_checkpoint": str(source_checkpoint),
        "output_model": str(args.output_model),
        "output_checkpoint": str(output_checkpoint),
        "iteration": int(args.iteration),
        "evidence_csv": str(args.evidence_csv),
        "evidence_rows_read": int(len(evidence)),
        "faces_used": int(len(valid_rows)),
        "faces_skipped": int(len(skipped_faces)),
        "vertices_added": added_vertices,
        "triangles_added": added_faces,
        "top_k_requested": int(args.top_k),
        "filters": {
            "min_view_hits": int(args.min_view_hits),
            "min_consistency": float(args.min_consistency),
            "min_pixel_count": float(args.min_pixel_count),
        },
        "strength": float(args.strength),
        "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
        "micro_scale": float(args.micro_scale),
        "normal_offset": float(args.normal_offset),
        "weight_logit_boost": float(args.weight_logit_boost),
        "delta_rgb_abs_mean": float(delta_rgb.abs().mean().item()) if delta_rgb.numel() else 0.0,
        "delta_rgb_abs_max": float(delta_rgb.abs().max().item()) if delta_rgb.numel() else 0.0,
        "topology_before": {
            "triangles": int(faces.shape[0]),
            "vertices": int(vertices.shape[0]),
        },
        "topology_after": {
            "triangles": int(out["_triangle_indices"].shape[0]),
            "vertices": int(out["triangles_points"].shape[0]),
            "degenerate_face_count": int(degenerate),
            "invalid_index_count": int(invalid),
        },
        "used_faces_preview": [
            {
                "rank": int(row["rank"]),
                "face_id": int(row["face_id"]),
                "score": float(row["score"]),
                "pixel_count": float(row["pixel_count"]),
                "view_hits": int(row["view_hits"]),
                "consistency": float(row["consistency"]),
                "residual_rgb": [float(x) for x in row["residual_rgb"].tolist()],
            }
            for row in valid_rows[:20]
        ],
    }
    (args.output_model / "surface_residual_microfacets_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    md = [
        "# ECSR Surface Residual Microfacets Audit",
        "",
        f"- operator: `{audit['operator']}`",
        f"- source model: `{audit['source_model']}`",
        f"- output model: `{audit['output_model']}`",
        f"- faces used: `{audit['faces_used']}`",
        f"- triangles added: `{audit['triangles_added']}`",
        f"- vertices added: `{audit['vertices_added']}`",
        f"- strength: `{audit['strength']}`",
        f"- max abs delta RGB: `{audit['max_abs_delta_rgb']}`",
        f"- micro scale: `{audit['micro_scale']}`",
        f"- normal offset: `{audit['normal_offset']}`",
        f"- weight logit boost: `{audit['weight_logit_boost']}`",
        f"- delta RGB abs mean: `{audit['delta_rgb_abs_mean']:.6f}`",
        f"- delta RGB abs max: `{audit['delta_rgb_abs_max']:.6f}`",
        f"- triangles: `{audit['topology_before']['triangles']}` -> `{audit['topology_after']['triangles']}`",
        f"- vertices: `{audit['topology_before']['vertices']}` -> `{audit['topology_after']['vertices']}`",
        f"- degenerate faces: `{degenerate}`",
        f"- invalid indices: `{invalid}`",
        "",
        "The added triangles are train-evidence residual carriers attached to existing surface faces.",
    ]
    (args.output_model / "surface_residual_microfacets_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if degenerate == 0 and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
