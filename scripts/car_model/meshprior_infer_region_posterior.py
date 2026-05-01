"""Run SP-CarNet posterior inference for mined MeshPrior regions."""

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

from ss3dm_prior.meshprior.scene_region_posterior import (
    canonicalize_region_points,
    decode_region_field,
    estimate_canonical_transform,
    estimate_posterior_uncertainty,
    extract_patch_mesh,
    load_ply_mesh,
    make_grid,
    run_spcarnet_posterior,
    sample_region_points,
    save_json,
)


def _device_or_cpu(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def run(args: argparse.Namespace) -> dict[str, object]:
    regions_json = Path(args.regions_json)
    if not regions_json.is_file():
        raise FileNotFoundError(f"regions_json not found: {regions_json}")
    checkpoint = Path(args.posterior_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"posterior_checkpoint not found: {checkpoint}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(regions_json.read_text(encoding="utf-8"))
    regions = payload.get("regions", [])
    selected = [r for r in regions if r.get("evidence", {}).get("eligible_for_posterior", False)]
    if args.include_ineligible:
        selected = regions
    selected = selected[: max(0, int(args.limit))]
    device = _device_or_cpu(str(args.device))
    summaries = []
    mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for idx, region in enumerate(selected):
        mesh_path = region.get("source_mesh_path")
        region_id = str(region.get("region_id", f"region_{idx:04d}"))
        region_dir = output_dir / region_id
        region_dir.mkdir(parents=True, exist_ok=True)
        if not mesh_path or not Path(mesh_path).is_file():
            summaries.append({"region_id": region_id, "status": "missing_mesh", "mesh_path": mesh_path})
            continue
        if mesh_path not in mesh_cache:
            mesh_cache[mesh_path] = load_ply_mesh(mesh_path)
        sampled = sample_region_points(mesh_cache[mesh_path], region, int(args.n_points), seed=int(args.seed) + idx)
        transform = estimate_canonical_transform(sampled)
        canonical = canonicalize_region_points(sampled, transform)
        model, post = run_spcarnet_posterior(checkpoint, canonical, device=device)
        z_mean = post["z_mean"].detach().cpu().numpy()
        z_logvar_t = post["z_logvar"]
        z_logvar = None if z_logvar_t is None else z_logvar_t.detach().cpu().numpy()
        np.save(region_dir / "sampled_region_points.npy", canonical)
        np.save(region_dir / "z_mean.npy", z_mean)
        if z_logvar is not None:
            np.save(region_dir / "z_logvar.npy", z_logvar)
        save_json(region_dir / "canonical_transform.json", transform)

        grid = make_grid(resolution=int(args.grid_resolution))
        logits = decode_region_field(model.decoder, post["z_mean"], grid, device=device)
        occ = torch.sigmoid(logits).detach().cpu().numpy().reshape(
            int(args.grid_resolution), int(args.grid_resolution), int(args.grid_resolution)
        )
        np.save(region_dir / f"occupancy_grid_{int(args.grid_resolution)}.npy", occ)
        uncertainty = estimate_posterior_uncertainty(post["z_mean"], post["z_logvar"], K=int(args.uncertainty_samples))
        extraction = {"extraction_success": False, "vertex_count": 0, "face_count": 0, "watertight": False}
        try:
            mesh_res = extract_patch_mesh(
                occupancy_fn=lambda q: torch.sigmoid(
                    model.decoder(q[None].to(torch.device(device)), post["z_mean"].to(torch.device(device)))
                ).reshape(-1),
                device=torch.device(device),
                resolution=int(args.grid_resolution),
                iso_level=0.5,
            )
            extraction = {
                "extraction_success": mesh_res.mesh is not None,
                "vertex_count": mesh_res.vertex_count,
                "face_count": mesh_res.face_count,
                "watertight": mesh_res.watertight,
            }
        except Exception as exc:
            extraction = {"extraction_success": False, "error": repr(exc), "vertex_count": 0, "face_count": 0}
        summary = {
            "region_id": region_id,
            "status": "ok",
            "mesh_path": mesh_path,
            "n_points": int(args.n_points),
            "field_occupancy_ratio": float((occ >= 0.5).mean()),
            **uncertainty,
            **extraction,
        }
        save_json(region_dir / "posterior_summary.json", summary)
        summaries.append(summary)
    top = {
        "regions_json": str(regions_json),
        "posterior_checkpoint": str(checkpoint),
        "device": device,
        "processed_regions": len(summaries),
        "ok_regions": sum(1 for s in summaries if s.get("status") == "ok"),
        "regions": summaries,
    }
    save_json(output_dir / "posterior_index.json", top)
    return top


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer SP-CarNet posteriors for mined scene regions.")
    parser.add_argument("--regions_json", required=True)
    parser.add_argument("--posterior_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--n_points", type=int, default=768)
    parser.add_argument("--grid_resolution", type=int, default=32)
    parser.add_argument("--uncertainty_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include_ineligible", action="store_true")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
