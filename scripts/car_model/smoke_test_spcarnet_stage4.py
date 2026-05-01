#!/usr/bin/env python3
"""SP-CarNet Stage 4 smoke test.

Steps:
1. Build a tiny encoder + tiny decoder (mirrors the Stage-3 smoke sizes).
2. Pick 2 objects from the train split.
3. Run 3 refinement steps on each:
   a. init z from encoder mean,
   b. compute observation loss (Tier-1 only, no scanner pose),
   c. backward; verify finite z.grad,
   d. step Adam on [z].
4. Verify decoder gradients remain zero throughout.
5. Verify the script tolerates ``scanner_pose=None``.

Exits non-zero on assertion failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.data.spcarnet_object_dataset import SPCarObjectDataset  # noqa: E402
from ss3dm_prior.losses_spcarnet_observation import (  # noqa: E402
    compute_observation_loss,
    free_space_violation_rate,
)
from ss3dm_prior.models.spcarnet_posterior import (  # noqa: E402
    SPCarPosteriorCompletionModel,
    SPCarPosteriorEncoder,
)
from ss3dm_prior.models.spcarnet_shape_field import SPCarShapeFieldDecoder  # noqa: E402


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    object_index = REPO_ROOT / "outputs/carnet/spcarnet/object_index_v1.json"
    if not object_index.is_file():
        print(f"[stage4-smoke] FAIL: object index missing at {object_index}", file=sys.stderr)
        return 1

    dataset = SPCarObjectDataset(object_index, splits=("train",))
    if len(dataset) < 2:
        print("[stage4-smoke] FAIL: dataset has fewer than 2 objects", file=sys.stderr)
        return 1
    items = [dataset[0], dataset[1]]
    print(f"[stage4-smoke] dataset_ok n_objects={len(dataset)}", file=sys.stderr)

    encoder = SPCarPosteriorEncoder(
        latent_dim=32, feature_dim=48, num_xattn_layers=2, num_self_attn_layers=1,
        num_latent_queries=8, attention_heads=4, ffn_dim=96, dropout=0.0,
        posterior_kind="variational", use_normals=False, use_conditioning_adapter=True,
    ).to(device)
    decoder = SPCarShapeFieldDecoder(
        latent_dim=32, hidden_dim=64, depth=3, num_fourier_freqs=8,
        field_kind="occupancy", feature_dim=0,
    ).to(device)
    completion = SPCarPosteriorCompletionModel(
        encoder=encoder, decoder=decoder, decoder_finetune_enabled=False,
    ).to(device)
    completion.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    weights = {"w_surf": 1.0, "w_free": 1.0, "w_mixed": 0.5, "w_ray": 0.0,
               "w_incidence": 0.0, "lambda_prior": 1e-3}
    deltas = {"delta_surf": 0.5, "delta_free": 0.5, "delta_mixed": 0.5}

    per_obj_summary: list[dict] = []

    for idx, item in enumerate(items):
        partial = item.get("partial_observed_points")
        if partial is None or len(partial) == 0:
            partial = item["clean_points_object"]
        partial_t = torch.from_numpy(np.asarray(partial, dtype=np.float32)[:64]).to(device)
        free_np = item.get("free_space_query_points")
        free_t = (
            torch.from_numpy(np.asarray(free_np, dtype=np.float32)[:64]).to(device)
            if free_np is not None and len(free_np) > 0
            else None
        )
        qall = item.get("occupancy_query_points")
        qlab = item.get("occupancy_query_labels")
        qall_t = (
            torch.from_numpy(np.asarray(qall, dtype=np.float32)[:64]).to(device)
            if qall is not None and len(qall) > 0
            else None
        )
        qlab_t = (
            torch.from_numpy(np.asarray(qlab[:64], dtype=np.float32)).to(device)
            if qlab is not None and len(qlab) > 0
            else None
        )

        with torch.no_grad():
            post = completion.encode(partial_t.unsqueeze(0), sample=False)
            z0 = post.z_mean.squeeze(0).clone()

        z = z0.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=1e-2)

        history = []
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            loss, metrics = compute_observation_loss(
                decoder=decoder,
                z=z.unsqueeze(0),
                observed_points=partial_t.unsqueeze(0),
                free_points=free_t.unsqueeze(0) if free_t is not None else None,
                hard_negatives=None,
                mixed_points=qall_t.unsqueeze(0) if qall_t is not None else None,
                mixed_labels=qlab_t.unsqueeze(0) if qlab_t is not None else None,
                mixed_ignore=None,
                scanner_pose=None,    # explicitly None — verifies fallback path
                weights=weights, deltas=deltas, field_kind="occupancy",
                enable_ray_loss=False, enable_incidence=False,
            )
            assert torch.isfinite(loss), f"non-finite loss at step {step}"
            loss.backward()
            assert z.grad is not None and torch.isfinite(z.grad).all(), (
                f"z grad non-finite at step {step}"
            )
            assert torch.any(z.grad != 0).item(), f"z grad is all zeros at step {step}"
            for p in decoder.parameters():
                assert p.grad is None or torch.all(p.grad == 0).item(), (
                    f"decoder param received gradient at step {step}"
                )
            opt.step()
            history.append({"step": step, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})

        violation = free_space_violation_rate(
            decoder, z.detach().unsqueeze(0),
            free_t.unsqueeze(0) if free_t is not None else partial_t.unsqueeze(0),
        )
        drift = float(torch.norm(z.detach() - z0).item())
        per_obj_summary.append({
            "object_id": item["object_id"],
            "history": history,
            "z_drift": drift,
            "free_space_violation": violation,
        })
        print(
            f"[stage4-smoke] obj_{idx}_ok object_id={item['object_id']} "
            f"loss0={history[0]['loss_total']:.4f} loss2={history[2]['loss_total']:.4f} "
            f"z_drift={drift:.6f} violation={violation:.4f}",
            file=sys.stderr,
        )

    print("[stage4-smoke] PASS", file=sys.stderr)
    print(json.dumps({"objects": per_obj_summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
