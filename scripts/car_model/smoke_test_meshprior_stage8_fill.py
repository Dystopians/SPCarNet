"""Smoke test for MeshPrior Stage 8 guarded fill."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.fill import build_fill_proposal, evaluate_fill_risk, find_boundary_loops
from ss3dm_prior.meshprior.synthetic_damage import compute_hole_boundary_metrics, damage_mesh_local_hole, make_box_mesh


def box_surface_field(points: torch.Tensor) -> torch.Tensor:
    linf = torch.maximum(torch.maximum(torch.abs(points[:, 0] / 1.0), torch.abs(points[:, 1] / 0.5)), torch.abs(points[:, 2] / 0.25))
    support = torch.exp(-torch.abs(linf - 1.0) * 16.0)
    eps = 1e-5
    return torch.log(torch.clamp(support, eps, 1 - eps) / torch.clamp(1 - support, eps, 1 - eps))


def main() -> None:
    vertices, faces = make_box_mesh()
    damaged = damage_mesh_local_hole(vertices, faces, remove_count=2)
    loops = find_boundary_loops((damaged.vertices, damaged.faces))
    assert len(loops) == 1, loops
    assert loops[0].closed
    before_hole = compute_hole_boundary_metrics(damaged.faces)
    proposal = build_fill_proposal((damaged.vertices, damaged.faces), loops[0], box_surface_field, min_support=0.45)
    risk = evaluate_fill_risk(proposal)
    after_hole = compute_hole_boundary_metrics(proposal.faces_after)
    assert proposal.accepted
    assert int(risk["added_face_count"]) == 4, risk
    assert risk["boundary_edge_count_after"] == 0.0, risk
    assert risk["component_count_delta"] == 0.0, risk
    assert after_hole["boundary_edge_count"] < before_hole["boundary_edge_count"]
    print("[meshprior-stage8-smoke] PASS")
    print(json.dumps(risk, indent=2))


if __name__ == "__main__":
    main()
