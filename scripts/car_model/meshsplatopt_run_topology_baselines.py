#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.csef_builder import load_mesh
from ss3dm_prior.meshsplatopt.edit_types import MeshState
from ss3dm_prior.meshsplatopt.topology_baselines import run_topology_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--budgets", default="0.90,0.75,0.50,0.25")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vertices, faces = load_mesh(args.mesh_path)
    budgets = [float(x) for x in args.budgets.split(",") if x]
    runs = run_topology_baselines(MeshState(vertices, faces), args.output_dir, budgets=budgets)
    print(f"Wrote {len(runs)} topology baseline runs to {args.output_dir}")


if __name__ == "__main__":
    main()
