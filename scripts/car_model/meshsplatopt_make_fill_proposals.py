#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.csef_builder import load_mesh
from ss3dm_prior.meshsplatopt.edit_apply import apply_edit
from ss3dm_prior.meshsplatopt.edit_types import MeshState
from ss3dm_prior.meshsplatopt.hole_fill import find_boundary_loops, make_boundary_loop_fill, write_fill_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vertices, faces = load_mesh(args.mesh_path)
    state = MeshState(vertices, faces)
    loops = find_boundary_loops(state.faces)
    proposals = [make_boundary_loop_fill(state, loop, proposal_id=f"boundary_loop_fill_{i:04d}") for i, loop in enumerate(loops)]
    preview = state.copy()
    for proposal in proposals[:1]:
        if proposal.edit is not None:
            apply_edit(preview, proposal.edit)
    write_fill_outputs(state, proposals, args.output_dir, preview_state=preview)
    print(f"Wrote {len(proposals)} fill proposals to {args.output_dir}")


if __name__ == "__main__":
    main()
