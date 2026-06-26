#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.run_v110_strict_split_parent_gate_scene import build_plan, parse_args


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
            "/dev/shm/v110",
            "--gpu",
            "2",
            "--dry_run",
        ]
    )
    plan = build_plan(args)
    build_cmd = _cmd_for(plan, "build_train_even_candidate_field")
    render_cmd = _cmd_for(plan, "render_candidate_train_and_test")
    gate_cmd = _cmd_for(plan, "gate_train_odd_to_test")
    eval_cmd = _cmd_for(plan, "evaluate_gated_test")

    assert "--split" in build_cmd and build_cmd[build_cmd.index("--split") + 1] == "train"
    assert "--view_subset" in build_cmd and build_cmd[build_cmd.index("--view_subset") + 1] == "even"
    assert "--skip_train" not in render_cmd
    assert "--skip_test" not in render_cmd
    assert "--checkpoint_endpoint_surface_field_path" in render_cmd
    assert "--calib_split" in gate_cmd and gate_cmd[gate_cmd.index("--calib_split") + 1] == "train"
    assert "--calib_view_subset" in gate_cmd and gate_cmd[gate_cmd.index("--calib_view_subset") + 1] == "odd"
    assert "--target_split" in gate_cmd and gate_cmd[gate_cmd.index("--target_split") + 1] == "test"
    assert "--split" in eval_cmd and eval_cmd[eval_cmd.index("--split") + 1] == "test"
    assert plan["train_v102_bank_path"].endswith("/flowers/v102_preprojected_delta_bank_train.pt")
    assert plan["parent_method_name"] == "ours_26000_v106_podmoe_basepreserve_flowers"
    assert plan["candidate_method_name"] == "ours_26000_v110_strict_train_even_candidate_flowers"
    assert plan["gate_method_name"] == "ours_26000_v110_strict_train_even_odd_parent_gate_flowers"

    print("v110 strict runner arg smoke: OK")
    print(json.dumps({"steps": [step["name"] for step in plan["steps"]]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
