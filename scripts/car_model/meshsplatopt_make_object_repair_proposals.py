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
from ss3dm_prior.meshsplatopt.object_prior_repair import make_object_prior_repair_proposals, write_object_repair_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--canonicalization_confidence", type=float, default=0.8)
    parser.add_argument("--posterior_uncertainty", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vertices, faces = load_mesh(args.mesh_path)
    proposals = make_object_prior_repair_proposals(
        MeshState(vertices, faces),
        canonicalization_confidence=args.canonicalization_confidence,
        posterior_uncertainty=args.posterior_uncertainty,
    )
    write_object_repair_outputs(proposals, args.output_dir)
    print(f"Wrote {len(proposals)} object repair proposals to {args.output_dir}")


if __name__ == "__main__":
    main()
