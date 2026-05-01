"""Create MeshPrior protect/prune proposal scores from M2/M3 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.protect_prune import build_protect_prune_proposals, compute_triangle_scores
from ss3dm_prior.meshprior.scene_region_posterior import load_ply_mesh, load_spcarnet_completion_model, save_json


def _device_or_cpu(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def run(args: argparse.Namespace) -> dict[str, object]:
    regions_payload = json.loads(Path(args.regions_json).read_text(encoding="utf-8"))
    regions = regions_payload.get("regions", [])[: max(0, int(args.limit))]
    device = _device_or_cpu(str(args.device))
    model = load_spcarnet_completion_model(args.posterior_checkpoint, device=device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    proposals = []
    score_tables = []
    mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for region in regions:
        if not region.get("evidence", {}).get("eligible_for_posterior", False) and not args.include_ineligible:
            continue
        region_id = str(region.get("region_id"))
        posterior_region_dir = Path(args.posterior_dir) / region_id
        z_path = posterior_region_dir / "z_mean.npy"
        summary_path = posterior_region_dir / "posterior_summary.json"
        mesh_path = region.get("source_mesh_path")
        if not z_path.is_file() or not mesh_path:
            continue
        if mesh_path not in mesh_cache:
            mesh_cache[mesh_path] = load_ply_mesh(mesh_path)
        z = torch.from_numpy(np.load(z_path)).float()
        if z.ndim == 2:
            z = z[0]
        uncertainty = 0.0
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            uncertainty = float(summary.get("uncertainty_score", 0.0))
        vertices, faces = mesh_cache[mesh_path]
        table = compute_triangle_scores(
            vertices=vertices,
            faces=faces,
            decoder=model.decoder,
            z=z,
            face_indices=region.get("face_indices", []),
            samples_per_face=int(args.samples_per_face),
            uncertainty_penalty=min(1.0, uncertainty),
            device=device,
        )
        batch = build_protect_prune_proposals(
            table,
            region_id=region_id,
            protect_threshold=float(args.protect_threshold),
            prune_threshold=float(args.prune_threshold),
        )
        score_tables.extend(batch.score_tables)
        proposals.extend(batch.proposals)
        for face, protect, prune, support, violation in zip(
            table.face_indices,
            table.protect_scores,
            table.prune_scores,
            table.surface_support,
            table.prior_violation,
            strict=True,
        ):
            rows.append([region_id, int(face), float(protect), float(prune), float(support), float(violation)])
    dtype = [("region_id", "U64"), ("face_index", "i8"), ("protect", "f4"), ("prune", "f4"), ("support", "f4"), ("violation", "f4")]
    arr = np.asarray([tuple(r) for r in rows], dtype=dtype)
    np.savez(output_dir / "triangle_scores.npz", scores=arr)
    save_json(
        output_dir / "proposals.json",
        {
            "regions_json": str(args.regions_json),
            "posterior_dir": str(args.posterior_dir),
            "proposal_count": len(proposals),
            "proposals": [p.to_dict() for p in proposals],
            "score_tables": [t.to_dict() for t in score_tables],
        },
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region_id", "face_index", "protect", "prune", "support", "violation"])
        writer.writerows(rows)
    return {"triangles": len(rows), "proposal_count": len(proposals), "output_dir": str(output_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create MeshPrior protect/prune proposals.")
    parser.add_argument("--regions_json", required=True)
    parser.add_argument("--posterior_dir", required=True)
    parser.add_argument("--posterior_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples_per_face", type=int, default=4)
    parser.add_argument("--protect_threshold", type=float, default=0.5)
    parser.add_argument("--prune_threshold", type=float, default=0.5)
    parser.add_argument("--include_ineligible", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
