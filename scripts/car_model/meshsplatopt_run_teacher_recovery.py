#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.teacher_recovery import run_teacher_recovery_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--edit_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = run_teacher_recovery_contract(
        model_path=args.model_path,
        edit_json=args.edit_json,
        output_dir=args.output_dir,
        iterations=args.iterations,
    )
    print(json.dumps(plan.to_dict(), indent=2))


if __name__ == "__main__":
    main()
