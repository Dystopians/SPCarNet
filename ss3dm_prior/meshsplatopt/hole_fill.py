from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .csef_builder import edge_ownership, write_ascii_ply
from .edit_types import MeshEdit, MeshSplatOptEditType, MeshState


@dataclass(frozen=True)
class FillProposal:
    proposal_id: str
    fill_mode: str
    edit: MeshEdit | None
    certificate: dict[str, Any]
    rejected_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["edit"] = self.edit.to_dict() if self.edit is not None else None
        return data


def find_boundary_loops(faces: np.ndarray) -> list[list[int]]:
    owners = edge_ownership(faces)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for (a, b), face_ids in owners.items():
        if len(face_ids) == 1:
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
    loops: list[list[int]] = []
    used_edges: set[tuple[int, int]] = set()
    for start in sorted(adjacency):
        for nxt in adjacency[start]:
            edge = tuple(sorted((start, nxt)))
            if edge in used_edges:
                continue
            loop = [start, nxt]
            used_edges.add(edge)
            prev, cur = start, nxt
            while True:
                candidates = [v for v in adjacency[cur] if v != prev]
                if not candidates:
                    break
                next_v = candidates[0]
                edge = tuple(sorted((cur, next_v)))
                if edge in used_edges:
                    break
                used_edges.add(edge)
                if next_v == loop[0]:
                    loops.append(loop)
                    break
                loop.append(next_v)
                prev, cur = cur, next_v
    return [loop for loop in loops if len(loop) >= 3]


def loop_area_xy(vertices: np.ndarray, loop: list[int]) -> float:
    pts = vertices[np.asarray(loop, dtype=np.int64)]
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def make_boundary_loop_fill(state: MeshState, loop: list[int], *, proposal_id: str = "fill_loop_0000") -> FillProposal:
    if len(set(loop)) < 3:
        return FillProposal(
            proposal_id=proposal_id,
            fill_mode="boundary_loop_fill",
            edit=None,
            certificate={"boundary_loop_support": False},
            rejected_reason="degenerate_boundary_loop",
        )
    loop_vertices = state.vertices[np.asarray(loop, dtype=np.int64)]
    area = loop_area_xy(state.vertices, loop)
    if area <= 1e-8:
        return FillProposal(
            proposal_id=proposal_id,
            fill_mode="boundary_loop_fill",
            edit=None,
            certificate={"boundary_loop_support": False, "expected_area_repaired": area},
            rejected_reason="degenerate_boundary_loop",
        )
    centroid = loop_vertices.mean(axis=0)
    center_index = len(state.vertices)
    faces = []
    for i, a in enumerate(loop):
        b = loop[(i + 1) % len(loop)]
        faces.append([int(a), int(b), center_index])
    cert = {
        "boundary_loop_support": True,
        "neighboring_surface_support": True,
        "sparse_depth_support": False,
        "free_space_risk": 0.1,
        "semantic_ground_object_support": "unknown",
        "camera_coverage_score": 0.5,
        "prior_only_flag": False,
        "expected_topology_cost": len(faces),
        "expected_area_repaired": area,
    }
    edit = MeshEdit(
        edit_id=f"{proposal_id}_edit",
        edit_type=MeshSplatOptEditType.FILL_PATCH.value,
        defect_id="unknown",
        inserted_vertices=[[float(x) for x in centroid]],
        inserted_faces=faces,
        topology_cost_delta=float(len(faces)),
        evidence_summary=cert,
        risk_summary={"free_space_risk": cert["free_space_risk"]},
        attribute_changes={"faces_are_global_indices": True},
    )
    return FillProposal(proposal_id=proposal_id, fill_mode="boundary_loop_fill", edit=edit, certificate=cert)


def write_fill_outputs(
    state: MeshState,
    proposals: list[FillProposal],
    output_dir: str | Path,
    *,
    preview_state: MeshState | None = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fill_proposals.json").write_text(json.dumps([p.to_dict() for p in proposals], indent=2), encoding="utf-8")
    with (out / "fill_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["proposal_id", "fill_mode", "prior_only_flag", "expected_area_repaired", "expected_topology_cost", "rejected_reason"])
        for p in proposals:
            writer.writerow(
                [
                    p.proposal_id,
                    p.fill_mode,
                    p.certificate.get("prior_only_flag", False),
                    p.certificate.get("expected_area_repaired", 0.0),
                    p.certificate.get("expected_topology_cost", 0),
                    p.rejected_reason,
                ]
            )
    lines = ["# Fill Certificate Report", "", f"- proposals: `{len(proposals)}`", "", "## Proposals", ""]
    for p in proposals:
        lines.append(
            f"- `{p.proposal_id}` mode `{p.fill_mode}` prior_only `{p.certificate.get('prior_only_flag', False)}` "
            f"rejected `{p.rejected_reason or 'no'}`"
        )
    (out / "fill_certificate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_ascii_ply(out / "fill_debug_before.ply", state.vertices, state.faces)
    if preview_state is not None:
        write_ascii_ply(out / "fill_debug_after.ply", preview_state.vertices, preview_state.faces)
