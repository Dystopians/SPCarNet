#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.edit_portfolio import PortfolioItem
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshState
from ss3dm_prior.meshsplatopt.csef_builder import load_mesh
from ss3dm_prior.meshsplatopt.repair_state_machine import run_repair_state_machine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--portfolio_json", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vertices, faces = load_mesh(args.mesh_path)
    payload = json.loads(Path(args.portfolio_json).read_text(encoding="utf-8"))
    items = [PortfolioItem(edit=MeshEdit(**p["edit"]), **{k: v for k, v in p.items() if k != "edit"}) for p in payload]
    result = run_repair_state_machine(MeshState(vertices, faces), items, args.output_dir)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
