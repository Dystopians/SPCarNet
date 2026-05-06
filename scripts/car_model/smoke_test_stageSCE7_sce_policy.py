#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sce_recovery_policy import (  # noqa: E402
    SCEPolicyConfig,
    decide_sce_policy_action,
    select_early_stop_candidate,
)


def main() -> int:
    cfg = SCEPolicyConfig()
    degrading_gate = {
        "metrics": {"delta_absrel": 0.001, "delta_mae": -0.1},
        "checks": {"mean_absrel_nonregression": False, "mean_mae_nonregression": True},
    }
    action = decide_sce_policy_action(sentinel_gate=degrading_gate, cfg=cfg)
    assert action["activate_rollback"] is True
    assert action["action"] == "run_targeted_rollback"

    passing_gate = {
        "metrics": {"delta_absrel": -0.001, "delta_mae": -0.1},
        "checks": {"mean_absrel_nonregression": True, "mean_mae_nonregression": True},
    }
    action = decide_sce_policy_action(sentinel_gate=passing_gate, cfg=cfg)
    assert action["activate_rollback"] is False

    parent = {
        "psnr": 10.0,
        "ssim": 0.2,
        "lpips": 0.5,
        "absrel": 0.3,
        "depth_mae": 3.0,
        "normal": 40.0,
    }
    history = [
        {"iteration": 100, "metrics": {"psnr": 10.5, "ssim": 0.25, "lpips": 0.49, "absrel": 0.29, "depth_mae": 2.9, "normal": 39.0}, "sentinel_pass": True},
        {"iteration": 200, "metrics": {"psnr": 11.0, "ssim": 0.30, "lpips": 0.48, "absrel": 0.31, "depth_mae": 2.8, "normal": 39.0}, "sentinel_pass": True},
        {"iteration": 300, "metrics": {"psnr": 11.5, "ssim": 0.31, "lpips": 0.47, "absrel": 0.32, "depth_mae": 2.7, "normal": 38.0}, "sentinel_pass": True},
    ]
    selected = select_early_stop_candidate(history, parent_metrics=parent, cfg=cfg)
    assert selected["decision"] == "PARENT_PARETO_EARLY_STOP"
    assert selected["selected"]["iteration"] == 100, selected
    print("SCE7 automatic SCE policy smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

