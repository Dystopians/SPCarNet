#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.synthetic_damage import DAMAGE_CATEGORIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", default="outputs/carnet/meshsplatopt/stageR13_synthetic_repair_benchmark/benchmark_spec.json")
    args = parser.parse_args()
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"damage_categories": DAMAGE_CATEGORIES}, indent=2), encoding="utf-8")
    print(f"Wrote synthetic benchmark spec to {path}")


if __name__ == "__main__":
    main()
