#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.synthetic_damage import run_synthetic_repair_benchmark, write_synthetic_benchmark_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/carnet/meshsplatopt/stageR13_synthetic_repair_benchmark")
    args = parser.parse_args()
    result = run_synthetic_repair_benchmark()
    write_synthetic_benchmark_outputs(result, args.output_dir)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit("Synthetic repair benchmark gate failed")


if __name__ == "__main__":
    main()
