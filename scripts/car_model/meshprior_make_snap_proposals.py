"""Create conservative MeshPrior snap proposals for a mesh NPZ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.snap import accept_snap_proposal, evaluate_snap_risk, propose_vertex_snap


def _sphere_field(points: torch.Tensor) -> torch.Tensor:
    dist = torch.linalg.norm(points, dim=-1)
    occ = torch.sigmoid((1.0 - dist) * 24.0)
    return torch.log(torch.clamp(occ, 1e-5, 1 - 1e-5) / torch.clamp(1 - occ, 1e-5, 1 - 1e-5))


def run(args: argparse.Namespace) -> dict[str, object]:
    payload = np.load(args.mesh_npz)
    vertices = payload["vertices"]
    faces = payload["faces"]
    proposal = propose_vertex_snap(vertices, faces, _sphere_field, max_disp=args.max_disp, allow_boundary=args.allow_boundary)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "snap_proposal.npz", vertices_before=proposal.vertices_before, vertices_after=proposal.vertices_after, displacement=proposal.displacement, eligible_mask=proposal.eligible_mask, faces=faces)
    risk = evaluate_snap_risk(proposal)
    risk["accepted_by_default_gate"] = accept_snap_proposal(risk)
    with (out_dir / "snap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(risk, f, indent=2)
        f.write("\n")
    return risk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make conservative snap proposals.")
    parser.add_argument("--mesh_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_disp", type=float, default=0.02)
    parser.add_argument("--allow_boundary", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
