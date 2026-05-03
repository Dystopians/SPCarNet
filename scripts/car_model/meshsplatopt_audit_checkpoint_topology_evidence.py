#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_adapter import checkpoint_to_mesh_state, load_checkpoint_state
from ss3dm_prior.meshsplatopt.csef_builder import edge_ownership, triangle_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_component_ratio_for_edge_csef", type=float, default=0.25)
    parser.add_argument("--max_boundary_face_fraction_for_edge_csef", type=float, default=0.80)
    return parser.parse_args()


def connected_components(face_count: int, owners: dict[tuple[int, int], list[int]]) -> list[list[int]]:
    neighbors: list[set[int]] = [set() for _ in range(face_count)]
    for face_ids in owners.values():
        if len(face_ids) < 2:
            continue
        for a in face_ids:
            for b in face_ids:
                if a != b:
                    neighbors[a].add(b)
    seen = np.zeros((face_count,), dtype=bool)
    components: list[list[int]] = []
    for start in range(face_count):
        if seen[start]:
            continue
        queue: deque[int] = deque([start])
        seen[start] = True
        comp: list[int] = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in neighbors[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    queue.append(nxt)
        components.append(comp)
    return components


def summarize(state) -> dict[str, Any]:
    owners = edge_ownership(state.faces)
    components = connected_components(len(state.faces), owners)
    component_sizes = [len(c) for c in components]
    boundary_faces = set()
    shared_edges = 0
    boundary_edges = 0
    nonmanifold_edges = 0
    for face_ids in owners.values():
        if len(face_ids) == 1:
            boundary_edges += 1
            boundary_faces.add(int(face_ids[0]))
        elif len(face_ids) == 2:
            shared_edges += 1
        else:
            nonmanifold_edges += 1
            boundary_faces.update(int(x) for x in face_ids)
    _, _, areas = triangle_geometry(state.vertices, state.faces)
    repeated_vertex_refs = int(state.faces.size - len(set(int(x) for x in state.faces.reshape(-1))))
    return {
        "vertices": int(len(state.vertices)),
        "triangles": int(len(state.faces)),
        "components": int(len(components)),
        "component_size_histogram": dict(Counter(component_sizes).most_common(20)),
        "largest_component_faces": int(max(component_sizes) if component_sizes else 0),
        "largest_component_fraction": float(max(component_sizes) / max(len(state.faces), 1) if component_sizes else 0.0),
        "single_face_component_fraction": float(sum(1 for x in component_sizes if x == 1) / max(len(components), 1)),
        "boundary_edges": int(boundary_edges),
        "shared_edges": int(shared_edges),
        "nonmanifold_edges": int(nonmanifold_edges),
        "boundary_faces": int(len(boundary_faces)),
        "boundary_face_fraction": float(len(boundary_faces) / max(len(state.faces), 1)),
        "repeated_vertex_refs": repeated_vertex_refs,
        "mean_triangle_area": float(np.mean(areas)) if len(areas) else 0.0,
        "median_triangle_area": float(np.median(areas)) if len(areas) else 0.0,
        "max_triangle_area": float(np.max(areas)) if len(areas) else 0.0,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = checkpoint_to_mesh_state(load_checkpoint_state(args.checkpoint_path))
    stats = summarize(state)
    edge_csef_valid = (
        stats["largest_component_fraction"] >= args.max_component_ratio_for_edge_csef
        and stats["boundary_face_fraction"] <= args.max_boundary_face_fraction_for_edge_csef
    )
    report = {
        "status": "PASS_EDGE_CSEF_VALID" if edge_csef_valid else "FAIL_EDGE_CSEF_INVALID_TRIANGLE_SOUP",
        "checkpoint_path": args.checkpoint_path,
        "edge_topology_stats": stats,
        "gate_thresholds": {
            "max_component_ratio_for_edge_csef": args.max_component_ratio_for_edge_csef,
            "max_boundary_face_fraction_for_edge_csef": args.max_boundary_face_fraction_for_edge_csef,
        },
        "decision": (
            "edge-connected CSEF may be used"
            if edge_csef_valid
            else "do not use shared-edge boundary-loop CSEF for real checkpoint proposal selection; use spatial adjacency or render/sparse evidence"
        ),
    }
    (out / "checkpoint_topology_evidence_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# MeshSplatOpt Checkpoint Topology Evidence Audit",
        "",
        f"- status: `{report['status']}`",
        f"- checkpoint: `{args.checkpoint_path}`",
        f"- triangles: `{stats['triangles']}`",
        f"- vertices: `{stats['vertices']}`",
        f"- components: `{stats['components']}`",
        f"- largest component fraction: `{stats['largest_component_fraction']}`",
        f"- single-face component fraction: `{stats['single_face_component_fraction']}`",
        f"- boundary face fraction: `{stats['boundary_face_fraction']}`",
        f"- decision: {report['decision']}",
    ]
    (out / "checkpoint_topology_evidence_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not edge_csef_valid:
        raise SystemExit("edge-topology CSEF invalid for this checkpoint")


if __name__ == "__main__":
    main()
