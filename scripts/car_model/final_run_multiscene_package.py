#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    commands = [
        [sys.executable, "scripts/car_model/final_collect_ablation_suite.py"],
        [sys.executable, "scripts/car_model/final_collect_multiscene_package.py"],
        [sys.executable, "scripts/car_model/final_make_paper_assets.py"],
    ]
    for cmd in commands:
        print("+", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

