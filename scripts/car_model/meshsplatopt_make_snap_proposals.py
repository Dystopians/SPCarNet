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
from ss3dm_prior.meshsplatopt.snap_proposals import make_snap_proposals, write_snap_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vertices, faces = load_mesh(args.mesh_path)
    state = MeshState(vertices, faces)
    proposals = make_snap_proposals(state)
    best = min([p for p in proposals if not p.rejected_reason], key=lambda p: p.expected_error_after, default=None)
    write_snap_outputs(state, proposals, args.output_dir, preview_proposal=best)
    print(f"Wrote {len(proposals)} snap proposals to {args.output_dir}")


if __name__ == "__main__":
    main()
