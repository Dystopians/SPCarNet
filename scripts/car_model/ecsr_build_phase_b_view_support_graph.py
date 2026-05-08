#!/usr/bin/env python3
"""Build the ECSR Phase-B View-Support Redundancy Graph.

This is a train-cache-only candidate generator.  It does not edit checkpoints
and does not accept candidates for the final method.  Its job is to replace
single-face prune scores with auditable surface-support groups that can later
enter certificate contraction or surface-attached appearance recovery.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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

OUTDOOR = {"bicycle", "flowers", "garden", "stump", "treehill"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--surface_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph"),
    )
    parser.add_argument(
        "--doc_out",
        type=Path,
        default=Path("docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md"),
    )
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--top_k", type=int, default=256)
    parser.add_argument("--neighbor_stride", type=int, default=8)
    parser.add_argument("--edge_score_threshold", type=float, default=0.52)
    parser.add_argument("--cluster_score_threshold", type=float, default=0.58)
    parser.add_argument("--min_shared_views", type=int, default=1)
    parser.add_argument("--max_cluster_size", type=int, default=12)
    return parser.parse_args()


class UnionFind:
    def __init__(self, values: list[int]):
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: int, b: int, max_size: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.size[ra] + self.size[rb] > max_size:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def clusters(self) -> list[list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for value in self.parent:
            out[self.find(value)].append(value)
        return [sorted(values) for values in out.values()]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_top_supports(path: Path, top_k: int) -> dict[int, dict[str, float]]:
    supports: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in itertools.islice(csv.DictReader(f), top_k):
            face_id = int(row["face_id"])
            supports[face_id] = {
                "rank": float(row["rank"]),
                "score": float(row["score"]),
                "pixel_count": float(row["pixel_count"]),
                "view_hits": float(row["view_hits"]),
                "mean_l1_error": float(row["mean_l1_error"]),
                "mean_texture": float(row["mean_texture"]),
                "residual_consistency": float(row["residual_consistency"]),
                "mean_residual_r": float(row["mean_residual_r"]),
                "mean_residual_g": float(row["mean_residual_g"]),
                "mean_residual_b": float(row["mean_residual_b"]),
            }
    return supports


def residual_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    va = np.asarray(
        [a["mean_residual_r"], a["mean_residual_g"], a["mean_residual_b"]],
        dtype=np.float64,
    )
    vb = np.asarray(
        [b["mean_residual_r"], b["mean_residual_g"], b["mean_residual_b"]],
        dtype=np.float64,
    )
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-10:
        return 0.5
    return float(np.clip((float(np.dot(va, vb)) / denom + 1.0) * 0.5, 0.0, 1.0))


def scalar_similarity(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-8)
    return float(np.clip(1.0 - abs(a - b) / denom, 0.0, 1.0))


def npz_files(scene_dir: Path) -> list[Path]:
    view_dir = scene_dir / "views"
    return sorted(path for path in view_dir.glob("*.npz") if path.stem.isdigit())


def collect_view_support(scene_dir: Path, face_ids: set[int], neighbor_stride: int) -> tuple[dict[int, set[int]], dict[int, int], dict[tuple[int, int], int]]:
    view_sets: dict[int, set[int]] = {fid: set() for fid in face_ids}
    pixel_counts: dict[int, int] = defaultdict(int)
    adjacency_counts: dict[tuple[int, int], int] = defaultdict(int)
    face_id_list = np.asarray(sorted(face_ids), dtype=np.int64)

    for view_idx, path in enumerate(npz_files(scene_dir)):
        with np.load(path) as data:
            face_map = data["face_id"].astype(np.int64, copy=False)
        top_mask = np.isin(face_map, face_id_list)
        if not np.any(top_mask):
            continue
        visible, counts = np.unique(face_map[top_mask], return_counts=True)
        visible = [int(x) for x in visible if int(x) in face_ids]
        for fid, count in zip(visible, counts):
            view_sets[int(fid)].add(view_idx)
            pixel_counts[int(fid)] += int(count)

        sampled = face_map[::neighbor_stride, ::neighbor_stride]
        sampled_mask = np.isin(sampled, face_id_list)
        for a, b in (
            (sampled[:, :-1], sampled[:, 1:]),
            (sampled[:-1, :], sampled[1:, :]),
        ):
            mask = sampled_mask[: a.shape[0], : a.shape[1]] & np.isin(b, face_id_list) & (a != b)
            if not np.any(mask):
                continue
            pairs = np.stack([a[mask], b[mask]], axis=1).astype(np.int64)
            pairs.sort(axis=1)
            unique_pairs = np.unique(pairs, axis=0)
            for p0, p1 in unique_pairs:
                adjacency_counts[(int(p0), int(p1))] += 1

    return view_sets, pixel_counts, adjacency_counts


def edge_record(
    fid_a: int,
    fid_b: int,
    supports: dict[int, dict[str, float]],
    view_sets: dict[int, set[int]],
    adjacency_counts: dict[tuple[int, int], int],
) -> dict[str, Any]:
    a, b = supports[fid_a], supports[fid_b]
    views_a, views_b = view_sets[fid_a], view_sets[fid_b]
    union = views_a | views_b
    inter = views_a & views_b
    visibility_overlap = float(len(inter) / max(len(union), 1))
    adjacency_views = int(adjacency_counts.get((min(fid_a, fid_b), max(fid_a, fid_b)), 0))
    adjacency_strength = float(adjacency_views / max(len(inter), 1))
    adjacency_strength = float(np.clip(adjacency_strength, 0.0, 1.0))
    residual = residual_similarity(a, b)
    error = scalar_similarity(a["mean_l1_error"], b["mean_l1_error"])
    texture = scalar_similarity(a["mean_texture"], b["mean_texture"])
    min_consistency = min(a["residual_consistency"], b["residual_consistency"])

    risk_flags: list[str] = []
    if len(inter) < 2:
        risk_flags.append("low_shared_views")
    if adjacency_views == 0:
        risk_flags.append("no_projected_adjacency")
    if min_consistency < 0.8:
        risk_flags.append("weak_residual_direction")
    if max(a["mean_texture"], b["mean_texture"]) > 0.65 and texture < 0.55:
        risk_flags.append("texture_mismatch")

    penalty = 0.0
    if "low_shared_views" in risk_flags:
        penalty += 0.10
    if "no_projected_adjacency" in risk_flags:
        penalty += 0.08
    if "weak_residual_direction" in risk_flags:
        penalty += 0.06
    if "texture_mismatch" in risk_flags:
        penalty += 0.04

    score = (
        0.30 * visibility_overlap
        + 0.25 * adjacency_strength
        + 0.20 * residual
        + 0.15 * error
        + 0.10 * texture
        - penalty
    )
    return {
        "face_i": fid_a,
        "face_j": fid_b,
        "shared_views": len(inter),
        "union_views": len(union),
        "visibility_overlap": visibility_overlap,
        "projected_adjacency_views": adjacency_views,
        "projected_adjacency_strength": adjacency_strength,
        "residual_similarity": residual,
        "error_similarity": error,
        "texture_similarity": texture,
        "min_residual_consistency": min_consistency,
        "redundancy_score": float(score),
        "risk_flags": risk_flags,
    }


def cluster_records(
    clusters: list[list[int]],
    supports: dict[int, dict[str, float]],
    edge_by_pair: dict[tuple[int, int], dict[str, Any]],
    num_unique_faces: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, faces in enumerate(clusters):
        if len(faces) <= 1:
            continue
        pair_edges = [
            edge_by_pair[(min(a, b), max(a, b))]
            for a, b in itertools.combinations(faces, 2)
            if (min(a, b), max(a, b)) in edge_by_pair
        ]
        if not pair_edges:
            continue
        mean_score = float(np.mean([e["redundancy_score"] for e in pair_edges]))
        mean_shared = float(np.mean([e["shared_views"] for e in pair_edges]))
        mean_adjacency = float(np.mean([e["projected_adjacency_strength"] for e in pair_edges]))
        mean_consistency = float(np.mean([supports[f]["residual_consistency"] for f in faces]))
        mean_error = float(np.mean([supports[f]["mean_l1_error"] for f in faces]))
        all_risks = sorted({flag for edge in pair_edges for flag in edge["risk_flags"]})
        if mean_adjacency > 0.0 and mean_shared >= 2 and mean_consistency >= 0.9:
            operator = "certificate_cluster_contraction_candidate"
        elif mean_consistency >= 0.9:
            operator = "surface_attached_attribute_recovery_candidate"
        else:
            operator = "diagnostic_only_high_risk_cluster"
        out.append(
            {
                "candidate_id": f"C{idx:04d}",
                "faces": faces,
                "num_faces": len(faces),
                "operator_type": operator,
                "mean_redundancy_score": mean_score,
                "mean_shared_views": mean_shared,
                "mean_projected_adjacency_strength": mean_adjacency,
                "mean_residual_consistency": mean_consistency,
                "mean_l1_error": mean_error,
                "risk_flags": all_risks,
                "expected_triangle_reduction_upper_bound": float((len(faces) - 1) / max(num_unique_faces, 1)),
                "certificate_status": "not_evaluated_phase_b_candidate_only",
            }
        )
    out.sort(key=lambda item: (item["operator_type"], item["mean_redundancy_score"]), reverse=True)
    return out


def build_scene_graph(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    scene_dir = args.surface_root / scene
    summary = load_json(scene_dir / "surface_evidence_summary.json")
    supports = load_top_supports(scene_dir / "top_residual_supports.csv", args.top_k)
    face_ids = set(supports)
    view_sets, pixel_counts, adjacency_counts = collect_view_support(
        scene_dir, face_ids, args.neighbor_stride
    )

    edges: list[dict[str, Any]] = []
    for fid_a, fid_b in itertools.combinations(sorted(face_ids), 2):
        record = edge_record(fid_a, fid_b, supports, view_sets, adjacency_counts)
        if record["shared_views"] >= args.min_shared_views:
            edges.append(record)
    edges.sort(key=lambda item: item["redundancy_score"], reverse=True)
    edge_by_pair = {
        (min(int(e["face_i"]), int(e["face_j"])), max(int(e["face_i"]), int(e["face_j"]))): e
        for e in edges
    }

    uf = UnionFind(sorted(face_ids))
    accepted_edges = []
    for edge in edges:
        if float(edge["redundancy_score"]) < args.cluster_score_threshold:
            continue
        if "weak_residual_direction" in edge["risk_flags"]:
            continue
        if "no_projected_adjacency" in edge["risk_flags"] and edge["shared_views"] < 3:
            continue
        if uf.union(int(edge["face_i"]), int(edge["face_j"]), args.max_cluster_size):
            accepted_edges.append(edge)
    clusters = cluster_records(uf.clusters(), supports, edge_by_pair, int(summary["num_unique_faces"]))

    node_records = []
    for fid, stats in supports.items():
        node_records.append(
            {
                "face_id": fid,
                "rank": int(stats["rank"]),
                "view_hits": len(view_sets.get(fid, set())),
                "pixel_count_phase_a": int(stats["pixel_count"]),
                "pixel_count_recomputed": int(pixel_counts.get(fid, 0)),
                "mean_l1_error": stats["mean_l1_error"],
                "mean_texture": stats["mean_texture"],
                "residual_consistency": stats["residual_consistency"],
            }
        )
    node_records.sort(key=lambda item: item["rank"])

    edge_candidates = [
        e
        for e in edges
        if e["redundancy_score"] >= args.edge_score_threshold
        and not ("weak_residual_direction" in e["risk_flags"])
    ]
    contraction_candidates = [
        c
        for c in clusters
        if c["operator_type"] == "certificate_cluster_contraction_candidate"
    ]
    attribute_candidates = [
        c
        for c in clusters
        if c["operator_type"] == "surface_attached_attribute_recovery_candidate"
    ]
    next_action = (
        "candidate_certification_ready"
        if contraction_candidates
        else "attribute_recovery_first"
        if attribute_candidates
        else "increase_cluster_context_or_geometry_cache"
    )

    scene_out = args.out_root / scene
    scene_out.mkdir(parents=True, exist_ok=True)
    graph = {
        "scene": scene,
        "scene_type": "outdoor" if scene in OUTDOOR else "indoor",
        "phase_a_summary": {
            "top_error_addressable_fraction": summary["mean_top_error_addressable_fraction"],
            "top_support_multiview_fraction": summary["top_support_multiview_fraction"],
            "top_support_mean_multiview_consistency": summary["top_support_mean_multiview_consistency"],
            "diagnostic_a": summary["diagnostic_a"],
            "diagnostic_b": summary["diagnostic_b"],
        },
        "protocol": {
            "top_k": args.top_k,
            "neighbor_stride": args.neighbor_stride,
            "edge_score_threshold": args.edge_score_threshold,
            "cluster_score_threshold": args.cluster_score_threshold,
            "min_shared_views": args.min_shared_views,
            "max_cluster_size": args.max_cluster_size,
            "test_usage": "none",
            "certificate_status": "candidate_generation_only",
        },
        "stats": {
            "nodes": len(node_records),
            "tested_edges": len(edges),
            "edge_candidates": len(edge_candidates),
            "cluster_candidates": len(clusters),
            "contraction_candidates": len(contraction_candidates),
            "attribute_recovery_candidates": len(attribute_candidates),
            "expected_triangle_reduction_upper_bound": float(
                sum(c["num_faces"] - 1 for c in clusters) / max(int(summary["num_unique_faces"]), 1)
            ),
            "next_action": next_action,
        },
        "nodes": node_records,
        "edges_top": edges[:500],
        "candidate_clusters": clusters,
    }
    (scene_out / "view_support_graph.json").write_text(
        json.dumps(graph, indent=2) + "\n", encoding="utf-8"
    )
    with (scene_out / "candidate_clusters.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "candidate_id",
                "operator_type",
                "num_faces",
                "mean_redundancy_score",
                "mean_shared_views",
                "mean_projected_adjacency_strength",
                "mean_residual_consistency",
                "expected_triangle_reduction_upper_bound",
                "risk_flags",
                "faces",
            ]
        )
        for c in clusters:
            writer.writerow(
                [
                    c["candidate_id"],
                    c["operator_type"],
                    c["num_faces"],
                    c["mean_redundancy_score"],
                    c["mean_shared_views"],
                    c["mean_projected_adjacency_strength"],
                    c["mean_residual_consistency"],
                    c["expected_triangle_reduction_upper_bound"],
                    ";".join(c["risk_flags"]),
                    " ".join(str(face) for face in c["faces"]),
                ]
            )
    report = [
        f"# ECSR Phase-B View-Support Redundancy Graph: {scene}",
        "",
        f"- scene type: `{graph['scene_type']}`",
        f"- nodes: `{graph['stats']['nodes']}`",
        f"- tested edges: `{graph['stats']['tested_edges']}`",
        f"- edge candidates: `{graph['stats']['edge_candidates']}`",
        f"- cluster candidates: `{graph['stats']['cluster_candidates']}`",
        f"- contraction candidates: `{graph['stats']['contraction_candidates']}`",
        f"- attribute recovery candidates: `{graph['stats']['attribute_recovery_candidates']}`",
        f"- expected triangle reduction upper bound: `{100.0 * graph['stats']['expected_triangle_reduction_upper_bound']:.4f}%`",
        f"- next action: `{next_action}`",
        "",
        "This graph is train-cache-only and candidate-generation-only. It does not",
        "edit checkpoints, use held-out test views, or certify final acceptance.",
        "",
        "Artifacts: `view_support_graph.json`, `candidate_clusters.csv`.",
    ]
    (scene_out / "view_support_graph_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return graph


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def fmt_pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def build_doc(args: argparse.Namespace, graphs: list[dict[str, Any]]) -> str:
    rows = []
    for graph in graphs:
        stats = graph["stats"]
        phase_a = graph["phase_a_summary"]
        rows.append(
            [
                graph["scene"],
                graph["scene_type"],
                stats["nodes"],
                stats["edge_candidates"],
                stats["cluster_candidates"],
                stats["contraction_candidates"],
                stats["attribute_recovery_candidates"],
                fmt_pct(stats["expected_triangle_reduction_upper_bound"], 4),
                f"{100.0 * phase_a['top_support_multiview_fraction']:.2f}%",
                stats["next_action"],
            ]
        )

    total_clusters = sum(int(g["stats"]["cluster_candidates"]) for g in graphs)
    total_contraction = sum(int(g["stats"]["contraction_candidates"]) for g in graphs)
    total_attribute = sum(int(g["stats"]["attribute_recovery_candidates"]) for g in graphs)
    mean_upper = float(
        np.mean([float(g["stats"]["expected_triangle_reduction_upper_bound"]) for g in graphs])
    )
    ready = sum(1 for g in graphs if g["stats"]["next_action"] == "candidate_certification_ready")
    md = [
        "# ECSR Phase-B View-Support Redundancy Graph",
        "",
        "This report is generated from the Phase-A train-only surface evidence",
        "cache. It upgrades the unit of reasoning from isolated face deletion to",
        "auditable local support groups. The output is still candidate-only:",
        "no checkpoint is modified and no held-out test view participates.",
        "",
        "## Fixed Policy",
        "",
        f"- top-K residual supports per scene: `{args.top_k}`",
        f"- projected adjacency sampling stride: `{args.neighbor_stride}`",
        f"- edge score threshold: `{args.edge_score_threshold}`",
        f"- cluster score threshold: `{args.cluster_score_threshold}`",
        f"- min shared train views: `{args.min_shared_views}`",
        f"- max cluster size: `{args.max_cluster_size}`",
        "- test usage: `none`",
        "",
        "## Aggregate",
        "",
        md_table(
            ["metric", "value"],
            [
                ["scenes", len(graphs)],
                ["cluster candidates", total_clusters],
                ["certificate-contraction candidates", total_contraction],
                ["surface-attribute recovery candidates", total_attribute],
                ["scenes with certification-ready candidates", f"{ready} / {len(graphs)}"],
                ["mean triangle reduction upper bound from Phase-B clusters", fmt_pct(mean_upper, 4)],
            ],
        ),
        "",
        "## Per-Scene Result",
        "",
        md_table(
            [
                "scene",
                "type",
                "nodes",
                "edge cand.",
                "cluster cand.",
                "contraction cand.",
                "attribute cand.",
                "tri-red upper",
                "Phase-A multiview",
                "next action",
            ],
            rows,
        ),
        "",
        "## Interpretation",
        "",
        "Phase B confirms that the correct next unit is a local support group, not",
        "a hand-picked per-scene parameter. The graph finds auditable candidate",
        "groups from train evidence only, but the expected direct triangle",
        "reduction of the top residual supports is still tiny. Therefore, Phase C",
        "must not overclaim compression from these clusters alone. It should use",
        "the graph as a safe candidate front-end for certificate contraction and",
        "Phase D surface-attached recovery.",
        "",
        "The strong research direction is now fixed:",
        "",
        "1. use Phase-B groups as policy-defined local masks;",
        "2. run certificate checks on train/policy-val only;",
        "3. start with attribute-only recovery where contraction evidence is weak;",
        "4. reserve held-out test for final full9 validation.",
        "",
        "## Artifacts",
        "",
    ]
    artifact_rows = []
    for graph in graphs:
        scene_dir = args.out_root / str(graph["scene"])
        artifact_rows.append(
            [
                graph["scene"],
                f"`{scene_dir / 'view_support_graph.json'}`",
                f"`{scene_dir / 'candidate_clusters.csv'}`",
                f"`{scene_dir / 'view_support_graph_report.md'}`",
            ]
        )
    md.append(md_table(["scene", "graph JSON", "candidates CSV", "report"], artifact_rows))
    md.append("")
    return "\n".join(md)


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    graphs = [build_scene_graph(args, scene) for scene in scenes]
    summary = {
        "protocol": {
            "surface_root": str(args.surface_root),
            "out_root": str(args.out_root),
            "scenes": scenes,
            "top_k": args.top_k,
            "neighbor_stride": args.neighbor_stride,
            "edge_score_threshold": args.edge_score_threshold,
            "cluster_score_threshold": args.cluster_score_threshold,
            "min_shared_views": args.min_shared_views,
            "max_cluster_size": args.max_cluster_size,
            "test_usage": "none",
        },
        "graphs": [
            {
                "scene": graph["scene"],
                "scene_type": graph["scene_type"],
                "phase_a_summary": graph["phase_a_summary"],
                "stats": graph["stats"],
            }
            for graph in graphs
        ],
    }
    (args.out_root / "phase_b_view_support_graph_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    md = build_doc(args, graphs)
    (args.out_root / "phase_b_view_support_graph_summary.md").write_text(
        md + "\n", encoding="utf-8"
    )
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(md + "\n", encoding="utf-8")
    print(f"[ECSR] wrote {args.doc_out}")
    print(f"[ECSR] wrote {args.out_root / 'phase_b_view_support_graph_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
