#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import _compact_state, validate_faces


def main() -> int:
    state = {
        "triangles_points": torch.arange(15, dtype=torch.float32).reshape(5, 3),
        "_triangle_indices": torch.tensor([[0, 1, 2], [2, 3, 0]], dtype=torch.int32),
        "vertex_weight": torch.ones((5, 1), dtype=torch.float32),
        "features_dc": torch.zeros((5, 1, 3), dtype=torch.float32),
        "features_rest": torch.zeros((5, 15, 3), dtype=torch.float32),
        "importance_score": torch.ones((2,), dtype=torch.float32),
        "image_size": torch.ones((2,), dtype=torch.float32),
        "pixel_count": torch.ones((2,), dtype=torch.int32),
        "sigma": -2.0,
        "active_sh_degree": 3,
    }
    compact, stats = _compact_state(state, np.zeros((0,), dtype=np.int64))
    assert int(stats["post_vertices"]) == 4, stats
    assert torch.equal(compact["_triangle_indices"], state["_triangle_indices"])
    assert int(compact["triangles_points"].shape[0]) == 4
    degenerate, invalid = validate_faces(compact["triangles_points"], compact["_triangle_indices"])
    assert degenerate == 0 and invalid == 0
    print("checkpoint compaction trailing-unused-vertex smoke PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
