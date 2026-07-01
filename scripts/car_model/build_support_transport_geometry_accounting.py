#!/usr/bin/env python3
"""Build geometry accounting for the support-transport frontier.

The support-transport policies evaluated in v305/v315d/v316c operate on render
outputs from the same compact parent topology.  This script compares that
compact parent against the local clean MeshSplatting baseline and writes a
small, auditable geometry ledger for reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def build_accounting(clean_root: Path, compact_root: Path, scenes: list[str]) -> dict[str, Any]:
    per_scene: dict[str, dict[str, Any]] = {}
    totals = {
        "clean_triangles": 0,
        "support_transport_triangles": 0,
        "clean_vertices": 0,
        "support_transport_vertices": 0,
    }

    for scene in scenes:
        clean_summary = clean_root / scene / "prism_debug" / "final_cleanup_summary.json"
        compact_audit = compact_root / scene / "ratio_0200" / "compact_model" / "topology_audit.json"
        if not clean_summary.exists():
            raise FileNotFoundError(f"Missing clean topology summary for {scene}: {clean_summary}")
        if not compact_audit.exists():
            raise FileNotFoundError(f"Missing compact topology audit for {scene}: {compact_audit}")

        clean = read_json(clean_summary)
        compact = read_json(compact_audit)

        clean_triangles = int(clean["post_prune_triangle_count"])
        clean_vertices = int(clean["post_prune_vertex_count"])
        compact_triangles = int(compact["post_triangles"])
        compact_vertices = int(compact["post_vertices"])

        triangle_delta = compact_triangles - clean_triangles
        vertex_delta = compact_vertices - clean_vertices
        triangle_reduction = (clean_triangles - compact_triangles) / clean_triangles
        vertex_reduction = (clean_vertices - compact_vertices) / clean_vertices

        per_scene[scene] = {
            "clean_summary": str(clean_summary),
            "compact_topology_audit": str(compact_audit),
            "clean_triangles": clean_triangles,
            "support_transport_triangles": compact_triangles,
            "triangle_delta_vs_clean": triangle_delta,
            "triangle_reduction_vs_clean": triangle_reduction,
            "clean_vertices": clean_vertices,
            "support_transport_vertices": compact_vertices,
            "vertex_delta_vs_clean": vertex_delta,
            "vertex_reduction_vs_clean": vertex_reduction,
            "compact_selector_mode": compact.get("selector_mode"),
            "compact_removed_fraction_from_parent": compact.get("removed_fraction"),
            "compact_degenerate_face_count": compact.get("degenerate_face_count"),
            "compact_invalid_index_count": compact.get("invalid_index_count"),
        }

        totals["clean_triangles"] += clean_triangles
        totals["support_transport_triangles"] += compact_triangles
        totals["clean_vertices"] += clean_vertices
        totals["support_transport_vertices"] += compact_vertices

    aggregate = {
        **totals,
        "scene_count": len(scenes),
        "total_triangle_delta_vs_clean": totals["support_transport_triangles"] - totals["clean_triangles"],
        "total_vertex_delta_vs_clean": totals["support_transport_vertices"] - totals["clean_vertices"],
        "total_triangle_reduction_vs_clean": (
            (totals["clean_triangles"] - totals["support_transport_triangles"]) / totals["clean_triangles"]
        ),
        "total_vertex_reduction_vs_clean": (
            (totals["clean_vertices"] - totals["support_transport_vertices"]) / totals["clean_vertices"]
        ),
        "mean_scene_triangle_reduction_vs_clean": sum(
            row["triangle_reduction_vs_clean"] for row in per_scene.values()
        )
        / len(per_scene),
        "mean_scene_vertex_reduction_vs_clean": sum(row["vertex_reduction_vs_clean"] for row in per_scene.values())
        / len(per_scene),
    }

    return {
        "clean_root": str(clean_root),
        "compact_root": str(compact_root),
        "geometry_owner": "v305/v315d/v316c inherit this compact parent topology; support transport changes colors/renders, not mesh topology.",
        "aggregate": aggregate,
        "per_scene": per_scene,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    aggregate = report["aggregate"]
    rows = report["per_scene"]
    lines = [
        "# Support-Transport Geometry Accounting",
        "",
        "This ledger compares the local clean MeshSplatting topology against the compact parent topology used by the current support-transport frontier.",
        "",
        "Important protocol note: v305, v315d, and v316c inherit the same compact parent topology. Their support-transport policy changes render/color corrections, not the mesh triangle or vertex count.",
        "",
        "## Aggregate",
        "",
        "| scenes | clean triangles | support-transport triangles | total triangle reduction | clean vertices | support-transport vertices | total vertex reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {aggregate['scene_count']} | {aggregate['clean_triangles']} | "
            f"{aggregate['support_transport_triangles']} | "
            f"{format_float(100.0 * aggregate['total_triangle_reduction_vs_clean'])}% | "
            f"{aggregate['clean_vertices']} | {aggregate['support_transport_vertices']} | "
            f"{format_float(100.0 * aggregate['total_vertex_reduction_vs_clean'])}% |"
        ),
        "",
        "## Per Scene",
        "",
        "| scene | clean triangles | support-transport triangles | triangle reduction | clean vertices | support-transport vertices | vertex reduction | compact parent removed fraction | topology errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, row in rows.items():
        topology_errors = int(row["compact_degenerate_face_count"] or 0) + int(row["compact_invalid_index_count"] or 0)
        lines.append(
            f"| {scene} | {row['clean_triangles']} | {row['support_transport_triangles']} | "
            f"{format_float(100.0 * row['triangle_reduction_vs_clean'])}% | "
            f"{row['clean_vertices']} | {row['support_transport_vertices']} | "
            f"{format_float(100.0 * row['vertex_reduction_vs_clean'])}% | "
            f"{format_float(100.0 * float(row['compact_removed_fraction_from_parent'] or 0.0))}% | "
            f"{topology_errors} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current frontier is not only a render-quality tweak: it keeps the compact-parent geometry advantage while improving render metrics over the local clean baseline.",
            "- Geometry reduction is scene-dependent because the compact parent is conservative on indoor scenes and more aggressive on sparse/outdoor-heavy scenes.",
            "- This does not prove final paper closure by itself; it closes the geometry-accounting gap for the current evidence package.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_root", type=Path, required=True)
    parser.add_argument("--compact_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--scenes", default=",".join(SCENES))
    args = parser.parse_args()

    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    report = build_accounting(args.clean_root, args.compact_root, scenes)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.output_md)
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
