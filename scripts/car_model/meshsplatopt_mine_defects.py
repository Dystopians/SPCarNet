#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.defect_mining import load_csef_result, mine_defects, write_defect_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine MeshSplatOpt defects from CSEF regions.")
    parser.add_argument("--csef_regions_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--giant_area_threshold", type=float, default=8.0)
    parser.add_argument("--unknown_void_hints_json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csef = load_csef_result(args.csef_regions_json)
    hints = []
    if args.unknown_void_hints_json:
        hints = json.loads(Path(args.unknown_void_hints_json).read_text(encoding="utf-8"))
    defects = mine_defects(csef, giant_area_threshold=args.giant_area_threshold, unknown_void_hints=hints)
    write_defect_outputs(defects, args.output_dir)
    print(f"Wrote {len(defects)} defects to {args.output_dir}")


if __name__ == "__main__":
    main()
