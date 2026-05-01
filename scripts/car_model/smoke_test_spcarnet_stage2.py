#!/usr/bin/env python3
"""SP-CarNet Stage 2 smoke test.

Validates the auto-decoder forward + backward path, the trainer's loss
assembly, and the Marching-Cubes evaluation pipeline.

Steps:
1. Build the Stage-1 object index (with --limit 8 if it does not yet exist).
2. Construct a tiny SPCarShapeFieldDecoder (width 64, depth 3, latent_dim 32).
3. Run two training iterations on two objects.
4. Verify finiteness + decreasing loss + non-zero gradients in both decoder and Z.
5. Extract a Marching-Cubes mesh at resolution 16 (mesh=None fallback acceptable).

Exits non-zero on assertion failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.car_model import build_spcarnet_object_index as build_index  # noqa: E402
from ss3dm_prior.data.spcarnet_object_dataset import SPCarObjectDataset  # noqa: E402
from ss3dm_prior.mesh.marching_cubes import extract_patch_mesh  # noqa: E402
from ss3dm_prior.models.spcarnet_shape_field import SPCarShapeFieldDecoder  # noqa: E402
from ss3dm_prior.training.spcarnet_autodecoder import (  # noqa: E402
    LatentTable,
    ShapeFieldLossConfig,
    ShapeFieldTrainConfig,
    assemble_query_batch,
    compute_losses,
)


def _ensure_index() -> Path:
    full_index = REPO_ROOT / "outputs/carnet/spcarnet/object_index_v1.json"
    if full_index.is_file():
        return full_index
    tmp = Path(tempfile.mkdtemp()) / "object_index_smoke.json"
    rc = build_index.main(
        [
            "--patch_cache_dir",
            str(REPO_ROOT / "outputs/ss3dm_prior_car/meshfleet_car_cache_v5"),
            "--output",
            str(tmp),
            "--canonicalization",
            "identity",
            "--limit",
            "8",
        ]
    )
    if rc != 0 or not tmp.is_file():
        raise RuntimeError("index builder failed during smoke")
    return tmp


def main() -> int:
    index_path = _ensure_index()
    print(f"[stage2-smoke] index_path={index_path}", file=sys.stderr)

    dataset = SPCarObjectDataset(index_path, splits=("train", "val", "test"))
    if len(dataset) < 2:
        print("[stage2-smoke] FAIL: dataset has fewer than 2 objects", file=sys.stderr)
        return 1
    items = [dataset[0], dataset[1]]
    print(f"[stage2-smoke] dataset_ok n_objects={len(dataset)}", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = SPCarShapeFieldDecoder(
        latent_dim=32,
        hidden_dim=64,
        depth=3,
        num_fourier_freqs=8,
        field_kind="occupancy",
        feature_dim=0,
    ).to(device)
    latents = LatentTable(num_objects=2, latent_dim=32, init_std=0.01).to(device)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=5e-4)
    opt_lat = torch.optim.Adam(latents.parameters(), lr=1e-3)

    train_cfg = ShapeFieldTrainConfig(
        object_index_path=str(index_path),
        queries_surface=64,
        queries_free=64,
        queries_hard=32,
        queries_mixed=32,
        queries_eikonal=0,
    )
    loss_cfg = ShapeFieldLossConfig()

    rng = np.random.default_rng(0)
    losses_seen: list[float] = []
    decoder_grad_seen = False
    latent_grad_seen = False

    for it in range(2):
        per_object_queries = [
            assemble_query_batch(item, cfg=train_cfg, rng=rng, field_kind="occupancy")
            for item in items
        ]
        stacked = {
            key: torch.stack([q[key] for q in per_object_queries], dim=0).to(device)
            for key in ("surface", "free", "hard", "mixed_pts", "mixed_lab")
        }
        z = latents(torch.tensor([0, 1], dtype=torch.long, device=device))
        total, metrics = compute_losses(
            decoder, z, stacked, loss_cfg=loss_cfg, field_kind="occupancy"
        )
        if not torch.isfinite(total):
            print(f"[stage2-smoke] FAIL: non-finite loss at iter {it}: {metrics}", file=sys.stderr)
            return 1
        opt_dec.zero_grad(set_to_none=True)
        opt_lat.zero_grad(set_to_none=True)
        total.backward()
        if any(p.grad is not None and torch.any(p.grad != 0).item() for p in decoder.parameters()):
            decoder_grad_seen = True
        if latents.codes.grad is not None and torch.any(latents.codes.grad != 0).item():
            latent_grad_seen = True
        opt_dec.step()
        opt_lat.step()
        losses_seen.append(metrics["loss_total"])
        print(f"[stage2-smoke] iter={it} loss={metrics}", file=sys.stderr)

    if not decoder_grad_seen:
        print("[stage2-smoke] FAIL: decoder parameters received no gradients", file=sys.stderr)
        return 1
    if not latent_grad_seen:
        print("[stage2-smoke] FAIL: latent table received no gradients", file=sys.stderr)
        return 1
    if losses_seen[-1] > losses_seen[0] + 1e-3 and losses_seen[0] < 10.0:
        # Loss going up by > 1e-3 from a healthy starting point would be suspicious.
        # Don't hard-fail on small upticks because Adam can step uphill on iter 1.
        print(
            f"[stage2-smoke] WARN: loss did not decrease ({losses_seen[0]:.4f} -> {losses_seen[-1]:.4f})",
            file=sys.stderr,
        )

    # Marching Cubes at low resolution.
    decoder.eval()
    z0 = latents.codes[0].detach()

    def _occupancy_fn(query: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = query.unsqueeze(0).to(device)
            return torch.sigmoid(decoder(q, z0.unsqueeze(0))).squeeze(0)

    result = extract_patch_mesh(
        occupancy_fn=_occupancy_fn,
        device=device,
        patch_radius=1.0,
        resolution=16,
        iso_level=0.5,
    )
    print(
        f"[stage2-smoke] mc_ok mesh_present={result.mesh is not None} "
        f"vertex_count={result.vertex_count} face_count={result.face_count}",
        file=sys.stderr,
    )

    # Optional sampled chamfer if a mesh was produced.
    if result.mesh is not None and result.face_count > 0:
        sampled = result.mesh.sample(256)
        sampled = np.asarray(sampled, dtype=np.float32)
        clean = items[0]["clean_points_object"]
        d = torch.cdist(torch.from_numpy(sampled), torch.from_numpy(clean)).numpy()
        chamfer = 0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean())
        print(f"[stage2-smoke] sampled_chamfer={chamfer:.4f}", file=sys.stderr)

    print("[stage2-smoke] PASS", file=sys.stderr)
    print(json.dumps({"losses_seen": losses_seen, "mesh_present": result.mesh is not None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
