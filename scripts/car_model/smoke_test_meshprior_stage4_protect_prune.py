"""Smoke test for MeshPrior Stage 4 protect/prune scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.protect_prune import build_protect_prune_proposals, compute_triangle_scores


def _logit(p: torch.Tensor) -> torch.Tensor:
    eps = 1e-5
    return torch.log(torch.clamp(p, eps, 1.0 - eps) / torch.clamp(1.0 - p, eps, 1.0 - eps))


def fake_box_field(points: torch.Tensor) -> torch.Tensor:
    # High support near the surface of the canonical unit cube, low far away.
    linf = points.abs().amax(dim=-1)
    support = torch.exp(-torch.abs(linf - 1.0) * 16.0)
    return _logit(support)


def main() -> None:
    vertices = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
            [4, 4, 4],
            [4.5, 4, 4],
            [4, 4.5, 4],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
            [8, 9, 10],
        ],
        dtype=np.int64,
    )
    table = compute_triangle_scores(
        vertices=vertices,
        faces=faces,
        decoder=fake_box_field,
        z=None,
        samples_per_face=4,
        uncertainty_penalty=0.0,
    )
    cube_protect = float(np.mean(table.protect_scores[:12]))
    floater_protect = float(table.protect_scores[12])
    cube_prune = float(np.mean(table.prune_scores[:12]))
    floater_prune = float(table.prune_scores[12])
    assert cube_protect > floater_protect + 0.5, (cube_protect, floater_protect)
    assert floater_prune > cube_prune + 0.5, (floater_prune, cube_prune)
    batch = build_protect_prune_proposals(table, region_id="synthetic_region", protect_threshold=0.5, prune_threshold=0.5)
    types = {p.proposal_type for p in batch.proposals}
    assert "protect" in types
    assert "prune" in types
    print("[meshprior-stage4-smoke] PASS")
    print(
        json.dumps(
            {
                "cube_protect": cube_protect,
                "floater_protect": floater_protect,
                "cube_prune": cube_prune,
                "floater_prune": floater_prune,
                "proposal_types": sorted(types),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
