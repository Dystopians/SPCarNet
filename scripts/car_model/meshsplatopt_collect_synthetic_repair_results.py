#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", default="outputs/carnet/meshsplatopt/stageR13_synthetic_repair_benchmark/synthetic_repair_results.json")
    args = parser.parse_args()
    data = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    print(json.dumps({"status": data["status"], "gate": data["gate"]}, indent=2))


if __name__ == "__main__":
    main()
