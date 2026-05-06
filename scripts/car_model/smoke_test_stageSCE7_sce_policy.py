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
    parent_pareto_guard_pass,
    render_guard_pass,
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

    guarded = SCEPolicyConfig(
        require_sentinel_gate_for_recovery=True,
        require_measured_candidate_for_recovery=True,
        max_psnr_drop=0.0,
        max_ssim_drop=0.0,
        max_lpips_increase=0.0,
        min_render_score_delta=0.0,
    )
    action = decide_sce_policy_action(sentinel_gate=None, cfg=guarded)
    assert action["action"] == "accept_parent_noop"
    assert action["execute_recovery"] is False
    parent_render = {"psnr": 11.0, "ssim": 0.24, "lpips": 0.57}
    bad_render = {"psnr": 10.8, "ssim": 0.20, "lpips": 0.60}
    ok, reasons, deltas = render_guard_pass(bad_render, parent_render, guarded)
    assert ok is False
    assert "psnr_drop_exceeds_guard" in reasons
    assert deltas["delta_lpips"] > 0.0
    action = decide_sce_policy_action(
        sentinel_gate=passing_gate,
        cfg=guarded,
        candidate_metrics=bad_render,
        parent_metrics=parent_render,
    )
    assert action["action"] == "accept_parent_noop"
    assert action["reason"] == "render_guard_failed"

    pareto_guarded = SCEPolicyConfig(require_parent_pareto_for_acceptance=True)
    parent_all = {"psnr": 10.0, "ssim": 0.2, "lpips": 0.5, "absrel": 0.3, "depth_mae": 3.0, "normal": 40.0}
    rgb_better_geom_worse = {"psnr": 10.1, "ssim": 0.21, "lpips": 0.49, "absrel": 0.31, "depth_mae": 3.1, "normal": 39.0}
    ok, reasons, deltas = parent_pareto_guard_pass(rgb_better_geom_worse, parent_all, pareto_guarded)
    assert ok is False
    assert "absrel_above_parent" in reasons
    assert "depth_mae_above_parent" in reasons
    assert deltas["delta_psnr"] > 0.0
    action = decide_sce_policy_action(
        sentinel_gate=passing_gate,
        cfg=pareto_guarded,
        candidate_metrics=rgb_better_geom_worse,
        parent_metrics=parent_all,
    )
    assert action["action"] == "accept_parent_noop"
    assert action["reason"] == "parent_pareto_guard_failed"

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
