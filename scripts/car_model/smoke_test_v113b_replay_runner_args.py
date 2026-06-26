#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.run_v113b_oot_tail_gate_replay_scene import build_plan, parse_args


def _cmd_for(plan: dict, step_name: str) -> list[str]:
    for step in plan["steps"]:
        if step["name"] == step_name:
            return step["cmd"]
    raise AssertionError(f"missing step {step_name}")


def main() -> int:
    args = parse_args(
        [
            "--scene",
            "counter",
            "--package_root",
            "/dev/shm/pkg",
            "--output_root",
            "/dev/shm/v110",
            "--gpu",
            "1",
            "--dry_run",
        ]
    )
    plan = build_plan(args)
    gate_cmd = _cmd_for(plan, "gate_train_odd_to_test_v113b")
    eval_cmd = _cmd_for(plan, "evaluate_v113b_test")

    assert plan["parent_method_name"] == "ours_26000_v106_podmoe_basepreserve_counter"
    assert plan["candidate_method_name"] == "ours_26000_v110_strict_train_even_candidate_counter"
    assert plan["gate_method_name"] == "ours_26000_v113b_oot_strict_parent_gate_counter"
    assert plan["candidate_manifest_path"].endswith(
        "/counter/fields/ours_26000_v110_strict_train_even_candidate_counter_field.manifest.json"
    )
    assert "--calib_split" in gate_cmd and gate_cmd[gate_cmd.index("--calib_split") + 1] == "train"
    assert "--calib_view_subset" in gate_cmd and gate_cmd[gate_cmd.index("--calib_view_subset") + 1] == "odd"
    assert "--target_split" in gate_cmd and gate_cmd[gate_cmd.index("--target_split") + 1] == "test"
    assert "--min_p05_psnr_gain=0.0" in gate_cmd
    assert "--oot_gate_mode" in gate_cmd and gate_cmd[gate_cmd.index("--oot_gate_mode") + 1] == "scene_fallback"
    assert "--oot_source_manifest" in gate_cmd and gate_cmd[gate_cmd.index("--oot_source_manifest") + 1] == plan["candidate_manifest_path"]
    assert "--split" in eval_cmd and eval_cmd[eval_cmd.index("--split") + 1] == "test"
    assert "--methods" in eval_cmd and eval_cmd[eval_cmd.index("--methods") + 1] == plan["gate_method_name"]

    print("v113b replay runner arg smoke: OK")
    print(json.dumps({"steps": [step["name"] for step in plan["steps"]]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

