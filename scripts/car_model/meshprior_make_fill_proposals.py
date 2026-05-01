"""Create guarded MeshPrior fill proposals for a mesh NPZ."""

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

from ss3dm_prior.meshprior.fill import build_fill_proposal, evaluate_fill_risk, find_boundary_loops, score_hole_candidates


def _box_surface_field(points: torch.Tensor) -> torch.Tensor:
    linf = torch.maximum(torch.maximum(torch.abs(points[:, 0] / 1.0), torch.abs(points[:, 1] / 0.5)), torch.abs(points[:, 2] / 0.25))
    support = torch.exp(-torch.abs(linf - 1.0) * 16.0)
    eps = 1e-5
    return torch.log(torch.clamp(support, eps, 1 - eps) / torch.clamp(1 - support, eps, 1 - eps))


def run(args: argparse.Namespace) -> dict[str, object]:
    payload = np.load(args.mesh_npz)
    vertices = payload["vertices"]
    faces = payload["faces"]
    loops = find_boundary_loops((vertices, faces))
    scores = score_hole_candidates((vertices, faces), loops)
    accepted_scores = [s for s in scores if bool(s["accepted"])]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"loop_count": len(loops), "candidate_scores": scores, "proposal_written": False}
    if accepted_scores:
        loop_idx = int(max(accepted_scores, key=lambda x: float(x["score"]))["loop_index"])
        proposal = build_fill_proposal((vertices, faces), loops[loop_idx], _box_surface_field, min_support=args.min_support)
        risk = evaluate_fill_risk(proposal)
        np.savez(
            out_dir / "fill_proposal.npz",
            vertices_before=proposal.vertices_before,
            faces_before=proposal.faces_before,
            vertices_after=proposal.vertices_after,
            faces_after=proposal.faces_after,
            added_vertex_indices=np.asarray(proposal.added_vertex_indices, dtype=np.int64),
            added_face_indices=np.asarray(proposal.added_face_indices, dtype=np.int64),
        )
        summary.update({"proposal_written": True, "risk": risk, "confidence": proposal.confidence})
    with (out_dir / "fill_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make guarded fill proposals.")
    parser.add_argument("--mesh_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_support", type=float, default=0.45)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
