#!/usr/bin/env python3
"""Static topology certificates for ECSR Phase-C candidate contractions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import checkpoint_path


DEFAULT_SCENES = (
    "bicycle",
    "flowers",
    "garden",
    "stump",
    "treehill",
    "room",
    "counter",
    "kitchen",
    "bonsai",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k"),
    )
    parser.add_argument("--policy_tag", default="sor_adaptive_geo")
    parser.add_argument(
        "--preflight_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_c/candidate_preflight"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_c/static_topology_certificate"),
    )
    parser.add_argument(
        "--doc_out",
        type=Path,
        default=Path("docs/car_model/5-8-ECSR-PhaseC-StaticTopologyCertificate.md"),
    )
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--max_candidates_per_scene", type=int, default=64)
    parser.add_argument("--area_eps", type=float, default=1e-12)
    parser.add_argument("--max_inverted_local_faces", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def face_area_and_normal(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tri = vertices[faces.long()]
    normal = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    area = 0.5 * torch.linalg.norm(normal, dim=1)
    denom = torch.linalg.norm(normal, dim=1, keepdim=True).clamp_min(1e-20)
    return area, normal / denom


def candidate_edges(faces: torch.Tensor) -> list[tuple[int, int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for tri in faces.tolist():
        for a, b in ((tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])):
            key = (int(min(a, b)), int(max(a, b)))
            counts[key] = counts.get(key, 0) + 1
    return [(a, b, count) for (a, b), count in counts.items()]


def choose_edge(vertices: torch.Tensor, candidate_face_vertices: torch.Tensor) -> dict[str, Any]:
    edges = candidate_edges(candidate_face_vertices)
    if not edges:
        raise ValueError("candidate has no edges")
    scored = []
    for a, b, count in edges:
        length = float(torch.linalg.norm(vertices[a] - vertices[b]).item())
        scored.append((count < 2, length, -count, a, b, count))
    scored.sort()
    _, length, neg_count, a, b, count = scored[0]
    return {
        "v_keep": int(min(a, b)),
        "v_remove": int(max(a, b)),
        "edge_length": float(length),
        "edge_support_in_candidate_faces": int(count),
        "edge_selection": "shortest_internal_edge" if count >= 2 else "shortest_candidate_edge",
    }


def simulate_edge_collapse(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    affected_faces = [int(x) for x in candidate["affected_faces"]]
    if not affected_faces:
        raise ValueError("candidate has no affected faces")
    if min(affected_faces) < 0 or max(affected_faces) >= int(faces.shape[0]):
        raise ValueError("affected face id out of range")
    candidate_face_vertices = faces[torch.as_tensor(affected_faces, dtype=torch.long)]
    edge = choose_edge(vertices, candidate_face_vertices)
    v_keep = int(edge["v_keep"])
    v_remove = int(edge["v_remove"])
    contains_keep = (faces == v_keep).any(dim=1)
    contains_remove = (faces == v_remove).any(dim=1)
    local_mask = contains_keep | contains_remove
    removed_mask = contains_keep & contains_remove
    local_ids = torch.nonzero(local_mask, as_tuple=False).flatten()
    removed_ids = torch.nonzero(removed_mask, as_tuple=False).flatten()

    local_faces_before = faces[local_ids].clone()
    local_vertices_after = vertices.clone()
    midpoint = 0.5 * (vertices[v_keep] + vertices[v_remove])
    local_vertices_after[v_keep] = midpoint
    local_faces_after = local_faces_before.clone()
    local_faces_after[local_faces_after == v_remove] = v_keep
    repeated = (
        (local_faces_after[:, 0] == local_faces_after[:, 1])
        | (local_faces_after[:, 0] == local_faces_after[:, 2])
        | (local_faces_after[:, 1] == local_faces_after[:, 2])
    )
    keep_local = ~repeated

    before_area, before_normal = face_area_and_normal(vertices, local_faces_before)
    after_area = torch.zeros_like(before_area)
    after_normal = torch.zeros_like(before_normal)
    if int(keep_local.sum().item()) > 0:
        area_kept, normal_kept = face_area_and_normal(local_vertices_after, local_faces_after[keep_local])
        after_area[keep_local] = area_kept
        after_normal[keep_local] = normal_kept
    normal_dot = (before_normal * after_normal).sum(dim=1)
    inverted = keep_local & (normal_dot < 0.0)
    zero_area_after = keep_local & (after_area <= float(args.area_eps))
    valence_keep_before = int(contains_keep.sum().item())
    valence_remove_before = int(contains_remove.sum().item())
    valence_keep_after = int((local_mask.sum() - removed_mask.sum()).item())
    removed_faces = int(removed_ids.numel())
    post_triangles = int(faces.shape[0]) - removed_faces
    post_vertices_upper = int(vertices.shape[0]) - 1

    risk_flags: list[str] = []
    if removed_faces <= 0:
        risk_flags.append("no_triangle_reduction")
    if int(zero_area_after.sum().item()) > 0:
        risk_flags.append("zero_area_after_collapse")
    if int(inverted.sum().item()) > int(args.max_inverted_local_faces):
        risk_flags.append("local_normal_flip")
    if valence_keep_after > max(valence_keep_before + valence_remove_before, 1):
        risk_flags.append("valence_spike")
    if float(edge["edge_length"]) <= 0.0 or not math.isfinite(float(edge["edge_length"])):
        risk_flags.append("invalid_edge_length")

    status = "PASS_STATIC" if not risk_flags else "REJECT_STATIC"
    return {
        "operator_type": "edge_collapse_static_simulation",
        "edge": edge,
        "topology_before": {
            "triangles": int(faces.shape[0]),
            "vertices": int(vertices.shape[0]),
            "local_faces": int(local_ids.numel()),
            "candidate_faces": len(affected_faces),
            "valence_keep_before": valence_keep_before,
            "valence_remove_before": valence_remove_before,
        },
        "topology_after_estimate": {
            "triangles": post_triangles,
            "vertices_upper_bound": post_vertices_upper,
            "removed_faces": removed_faces,
            "removed_fraction": float(removed_faces / max(int(faces.shape[0]), 1)),
            "valence_keep_after": valence_keep_after,
            "zero_area_after_local_faces": int(zero_area_after.sum().item()),
            "inverted_local_faces": int(inverted.sum().item()),
            "degenerate_faces_removed": int(repeated.sum().item()),
        },
        "risk_flags": risk_flags,
        "status": status,
    }


def process_scene(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    model_path = args.method_root / scene / args.policy_tag / "compact_model"
    state = torch.load(checkpoint_path(model_path, int(args.iteration)), map_location="cpu")
    vertices = state["triangles_points"].detach().cpu().float()
    faces = state["_triangle_indices"].detach().cpu().long()
    preflight_dir = args.preflight_root / scene / "certificates"
    cert_paths = sorted(preflight_dir.glob("*.json"))[: int(args.max_candidates_per_scene)]
    out_dir = args.out_root / scene / "certificates"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in cert_paths:
        preflight = load_json(path)
        if preflight.get("preflight_status") != "preflight_pass":
            continue
        record = {
            "scene": scene,
            "candidate_id": preflight["candidate_id"],
            "phase_b_operator_type": preflight["operator_type"],
            "affected_faces": preflight["affected_faces"],
            "seed": preflight.get("seed"),
            "test_usage": "none",
            "source_preflight_certificate": str(path),
        }
        try:
            static = simulate_edge_collapse(vertices, faces, preflight, args)
            record.update(static)
        except Exception as exc:
            record.update(
                {
                    "operator_type": "edge_collapse_static_simulation",
                    "status": "REJECT_STATIC",
                    "risk_flags": ["exception"],
                    "exception": str(exc),
                }
            )
        (out_dir / f"{preflight['candidate_id']}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        records.append(record)

    passed = [r for r in records if r.get("status") == "PASS_STATIC"]
    contraction_pass = [
        r
        for r in passed
        if r.get("phase_b_operator_type") == "certificate_cluster_contraction_candidate"
    ]
    summary = {
        "scene": scene,
        "model_path": str(model_path),
        "candidates_checked": len(records),
        "static_pass": len(passed),
        "static_reject": len(records) - len(passed),
        "contraction_static_pass": len(contraction_pass),
        "certificate_dir": str(out_dir),
    }
    scene_out = args.out_root / scene
    scene_out.mkdir(parents=True, exist_ok=True)
    (scene_out / "static_topology_certificate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_doc(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> str:
    rows = [
        [
            s["scene"],
            s["candidates_checked"],
            s["static_pass"],
            s["static_reject"],
            s["contraction_static_pass"],
            f"`{s['certificate_dir']}`",
        ]
        for s in summaries
    ]
    total_checked = sum(int(s["candidates_checked"]) for s in summaries)
    total_pass = sum(int(s["static_pass"]) for s in summaries)
    total_contraction = sum(int(s["contraction_static_pass"]) for s in summaries)
    return "\n".join(
        [
            "# ECSR Phase-C Static Topology Certificate",
            "",
            "This is the first static Layer-1 certificate for ECSR contraction",
            "candidates. It simulates a conservative edge-collapse operator on",
            "the real checkpoint topology and rejects candidates that would create",
            "local zero-area faces, normal flips, invalid edge lengths, or no",
            "triangle reduction. No checkpoint is modified and no held-out test",
            "view is used.",
            "",
            "## Aggregate",
            "",
            md_table(
                ["metric", "value"],
                [
                    ["candidates checked", total_checked],
                    ["static pass", total_pass],
                    ["static reject", total_checked - total_pass],
                    ["contraction static pass", total_contraction],
                ],
            ),
            "",
            "## Per-Scene",
            "",
            md_table(
                [
                    "scene",
                    "checked",
                    "static pass",
                    "static reject",
                    "contraction pass",
                    "certificate dir",
                ],
                rows,
            ),
            "",
            "## Interpretation",
            "",
            "A PASS_STATIC candidate is still not an accepted ECSR edit. It is only",
            "eligible for materialized checkpoint smoke rendering and policy-val",
            "before/after metrics. This prevents expensive long runs on candidates",
            "that are already topologically unsafe.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    summaries = [process_scene(args, scene) for scene in scenes]
    aggregate = {"protocol": vars(args), "summaries": summaries}
    (args.out_root / "phase_c_static_topology_certificate_summary.json").write_text(
        json.dumps(aggregate, indent=2, default=str) + "\n", encoding="utf-8"
    )
    md = build_doc(args, summaries)
    (args.out_root / "phase_c_static_topology_certificate_summary.md").write_text(
        md + "\n", encoding="utf-8"
    )
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(md + "\n", encoding="utf-8")
    print(f"[ECSR] wrote {args.doc_out}")
    print(f"[ECSR] wrote {args.out_root / 'phase_c_static_topology_certificate_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
