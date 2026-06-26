#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.run_v111_end_to_end_strict_parent_gate_scene import build_plan, parse_args


def _cmd_for(plan: dict, step_name: str) -> list[str]:
    for step in plan["steps"]:
        if step["name"] == step_name:
            return step["cmd"]
    raise AssertionError(f"missing step {step_name}")


def main() -> int:
    args = parse_args(
        [
            "--scene",
            "flowers",
            "--package_root",
            "/dev/shm/pkg",
            "--train_v102_bank_root",
            "/dev/shm/train_bank",
            "--output_root",
            "/dev/shm/v111",
            "--gpu",
            "2",
            "--dry_run",
        ]
    )
    plan = build_plan(args)
    parent_build_cmd = _cmd_for(plan, "build_train_all_parent_field")
    parent_render_cmd = _cmd_for(plan, "render_parent_train_and_test")
    candidate_build_cmd = _cmd_for(plan, "build_train_even_candidate_field")
    candidate_render_cmd = _cmd_for(plan, "render_candidate_train_and_test")
    gate_cmd = _cmd_for(plan, "gate_train_odd_to_test")
    eval_cmd = _cmd_for(plan, "evaluate_gated_test")

    assert "--split" in parent_build_cmd and parent_build_cmd[parent_build_cmd.index("--split") + 1] == "train"
    assert "--view_subset" in parent_build_cmd and parent_build_cmd[parent_build_cmd.index("--view_subset") + 1] == "all"
    assert "--split" in candidate_build_cmd and candidate_build_cmd[candidate_build_cmd.index("--split") + 1] == "train"
    assert "--view_subset" in candidate_build_cmd and candidate_build_cmd[candidate_build_cmd.index("--view_subset") + 1] == "even"
    assert "--skip_train" not in parent_render_cmd
    assert "--skip_test" not in parent_render_cmd
    assert "--skip_train" not in candidate_render_cmd
    assert "--skip_test" not in candidate_render_cmd
    assert "--checkpoint_endpoint_surface_field_path" in parent_render_cmd
    assert "--checkpoint_endpoint_surface_field_path" in candidate_render_cmd
    assert "--calib_split" in gate_cmd and gate_cmd[gate_cmd.index("--calib_split") + 1] == "train"
    assert "--calib_view_subset" in gate_cmd and gate_cmd[gate_cmd.index("--calib_view_subset") + 1] == "odd"
    assert "--target_split" in gate_cmd and gate_cmd[gate_cmd.index("--target_split") + 1] == "test"
    assert "--parent_method_name" in gate_cmd and gate_cmd[gate_cmd.index("--parent_method_name") + 1] == plan["parent_method_name"]
    assert "--candidate_method_name" in gate_cmd and gate_cmd[gate_cmd.index("--candidate_method_name") + 1] == plan["candidate_method_name"]
    assert "--min_p05_psnr_gain=0.0" in gate_cmd
    assert "--min_p05_ssim_gain=-1e-06" in gate_cmd
    assert "--min_p05_lpips_gain=-1000000000.0" in gate_cmd
    assert "--oot_gate_mode" in gate_cmd and gate_cmd[gate_cmd.index("--oot_gate_mode") + 1] == "scene_fallback"
    assert "--oot_source_manifest" in gate_cmd
    assert gate_cmd[gate_cmd.index("--oot_source_manifest") + 1].endswith(
        "ours_26000_v111_train_even_candidate_flowers_field.manifest.json"
    )
    assert "--split" in eval_cmd and eval_cmd[eval_cmd.index("--split") + 1] == "test"
    assert plan["train_v102_bank_path"].endswith("/flowers/v102_preprojected_delta_bank_train.pt")
    assert plan["parent_method_name"] == "ours_26000_v111_train_all_parent_flowers"
    assert plan["candidate_method_name"] == "ours_26000_v111_train_even_candidate_flowers"
    assert plan["gate_method_name"] == "ours_26000_v111_train_even_odd_parent_gate_flowers"
    assert plan["strict_fairness"]["parent_build"]["split"] == "train"
    assert plan["strict_fairness"]["parent_build"]["view_subset"] == "all"
    assert plan["strict_fairness"]["candidate_build"]["split"] == "train"
    assert plan["strict_fairness"]["candidate_build"]["view_subset"] == "even"
    assert plan["strict_fairness"]["gate"]["calib_split"] == "train"
    assert plan["strict_fairness"]["gate"]["calib_view_subset"] == "odd"
    assert plan["strict_fairness"]["gate"]["target_split"] == "test"

    print("v111 end-to-end strict runner arg smoke: OK")
    print(json.dumps({"steps": [step["name"] for step in plan["steps"]]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
