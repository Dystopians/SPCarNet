#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.certificate_edit_planner import plan_certificate_edits, write_certificate_edit_plan  # noqa: E402


def main() -> int:
    ecg = {
        "cluster_summary": [
            {"cluster_id": 0, "certificate_pressure": 8, "gate_critical_count": 3, "bad_tradeoff_count": 2},
            {"cluster_id": 1, "certificate_pressure": 6, "gate_critical_count": 2, "snap_score": 7, "surface_support": 0.8},
            {"cluster_id": 2, "certificate_pressure": 0, "render_debt": 3, "surface_support": 0.8},
            {"cluster_id": 3, "certificate_pressure": 0, "render_debt": 3, "surface_support": 0.8, "hole_score": 1},
            {"cluster_id": 4, "certificate_pressure": 0, "redundancy": 2, "surface_support": 0.0},
            {"cluster_id": 5, "certificate_pressure": 0, "prior_only_risk": 1, "surface_support": 0.0},
            {"cluster_id": 6, "certificate_pressure": 0, "render_gain": 1},
        ]
    }
    plan = plan_certificate_edits(ecg)
    actions = {p["action"] for p in plan["plans"]}
    required = {"ROLLBACK_ONLY", "SNAP_LOCAL", "SPLIT_ALLOCATE", "FILL_PATCH_LOCAL", "DELETE_OR_COLLAPSE", "REJECT_UNOBSERVED", "APPEARANCE_ONLY"}
    assert required.issubset(actions), actions
    with tempfile.TemporaryDirectory() as td:
        write_certificate_edit_plan(plan, Path(td))
        assert (Path(td) / "certificate_edit_plan.json").is_file()
        assert (Path(td) / "certificate_edit_plan.csv").is_file()
    print("SCE13 certificate edit planner smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

