#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.sce_local_surgery import apply_synthetic_sce_local_surgery, propose_sce_local_surgery, write_sce_local_surgery_outputs  # noqa: E402


def main() -> int:
    rows = [
        {"cluster_id": 0, "depth_conflict": 2.0, "surface_support": 1.0},
        {"cluster_id": 1, "in_triangle_depth_variation": 2.0, "render_debt": 2.0, "surface_support": 1.0},
        {"cluster_id": 2, "hole_score": 1.0, "render_debt": 2.0, "surface_support": 1.0},
        {"cluster_id": 3, "hole_score": 1.0, "render_debt": 2.0, "surface_support": 0.0, "prior_only_flag": True},
        {"cluster_id": 4, "appearance_ghost_score": 1.0, "surface_support": 1.0},
    ]
    proposals = propose_sce_local_surgery(rows)
    actions = [p["action"] for p in proposals]
    assert actions[:5] == ["SNAP_VERTICES", "SPLIT_TRIANGLES", "FILL_PATCH", "REJECT", "APPEARANCE_RESET"], actions
    snap_result = apply_synthetic_sce_local_surgery(proposals[0], 10.0)
    split_result = apply_synthetic_sce_local_surgery(proposals[1], 10.0)
    assert snap_result["sentinel_error_after"] < 10.0
    assert split_result["sentinel_error_after"] < 10.0
    assert not proposals[3]["accepted"]
    with tempfile.TemporaryDirectory() as td:
        write_sce_local_surgery_outputs(proposals, Path(td))
        assert (Path(td) / "sce_local_surgery_proposals.json").is_file()
    print("SCE9 local surgery smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

