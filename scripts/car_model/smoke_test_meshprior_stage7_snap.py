"""Smoke test for MeshPrior Stage 7 conservative snap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.snap import evaluate_snap_risk, propose_vertex_snap


def sphere_field(points: torch.Tensor) -> torch.Tensor:
    dist = torch.linalg.norm(points, dim=-1)
    occ = torch.sigmoid((1.0 - dist) * 24.0)
    return torch.log(torch.clamp(occ, 1e-5, 1 - 1e-5) / torch.clamp(1 - occ, 1e-5, 1 - 1e-5))


def surface_distance(points: np.ndarray) -> np.ndarray:
    return np.abs(np.linalg.norm(points, axis=1) - 1.0)


def main() -> None:
    vertices = np.asarray(
        [
            [1.10, 0.00, 0.00],
            [-1.10, 0.00, 0.00],
            [0.00, 1.10, 0.00],
            [0.00, -1.10, 0.00],
            [0.00, 0.00, 1.10],
            [0.00, 0.00, -1.10],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int64,
    )
    protect = np.zeros(len(vertices), dtype=bool)
    protect[0] = True
    proposal = propose_vertex_snap(
        vertices,
        faces,
        sphere_field,
        protect_mask=protect,
        max_disp=0.02,
        allow_boundary=False,
    )
    risk = evaluate_snap_risk(proposal, distance_fn=surface_distance)
    assert risk["surface_distance_after_mean"] < risk["surface_distance_before_mean"], risk
    assert risk["max_displacement"] <= 0.020001, risk
    assert np.linalg.norm(proposal.displacement[0]) == 0.0
    print("[meshprior-stage7-smoke] PASS")
    print(json.dumps(risk, indent=2))


if __name__ == "__main__":
    main()
