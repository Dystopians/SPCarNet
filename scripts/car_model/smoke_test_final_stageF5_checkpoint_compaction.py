#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/peilincai/micromamba/envs/mesh_splatting/bin/python"


def _run(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)


def main() -> int:
    source = "outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_7000to22000/model"
    out_root = "outputs/carnet/meshsplatopt/final_stageF5_checkpoint_compaction"
    area = _run(
        [
            PYTHON,
            "scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py",
            "--source_model",
            source,
            "--iteration",
            "22000",
            "--output_model",
            f"{out_root}/parking_area70_repro/model",
            "--selector_mode",
            "area_smallest",
            "--target_prune_fraction",
            "0.70",
        ]
    )
    csef = _run(
        [
            PYTHON,
            "scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py",
            "--source_model",
            source,
            "--iteration",
            "22000",
            "--output_model",
            f"{out_root}/parking_csef70_smoke/model",
            "--selector_mode",
            "csef_low_evidence_boundary_protected",
            "--target_prune_fraction",
            "0.70",
        ]
    )
    expected = 2_564_473
    tolerance = 2
    assert abs(int(area["post_triangles"]) - expected) <= tolerance, area
    assert int(area["invalid_index_count"]) == 0 and int(area["degenerate_face_count"]) == 0
    assert abs(int(csef["post_triangles"]) - expected) <= tolerance, csef
    assert int(csef["invalid_index_count"]) == 0 and int(csef["degenerate_face_count"]) == 0
    assert Path(ROOT / csef["output_checkpoint"]).is_file()
    print(
        "F5 checkpoint compaction smoke PASS: "
        f"area_triangles={area['post_triangles']} csef_triangles={csef['post_triangles']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
