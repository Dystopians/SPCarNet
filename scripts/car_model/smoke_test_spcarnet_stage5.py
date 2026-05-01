#!/usr/bin/env python3
"""SP-CarNet Stage 5 smoke test.

Steps (per design §5):
1. Build a tiny encoder + tiny decoder.
2. Pick 1 train object.
3. Sample K=4 candidates from q(z|O) with sample=True.
4. Verify pairwise latent L2 > 1e-4 (candidates are distinct).
5. Compute score for each — verify finite.
6. Compute pairwise diversity_latent_l2 — verify > 0.
7. Print [stage5-smoke] PASS.

Mesh extraction is not asserted here — tiny untrained decoder gives uniform
field which legitimately yields no MC mesh.

Exits non-zero on assertion failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.data.spcarnet_object_dataset import SPCarObjectDataset  # noqa: E402
from ss3dm_prior.models.spcarnet_posterior import (  # noqa: E402
    SPCarPosteriorCompletionModel,
    SPCarPosteriorEncoder,
)
from ss3dm_prior.models.spcarnet_shape_field import SPCarShapeFieldDecoder  # noqa: E402


def _score_no_grad(decoder, z, *, observed, free) -> float:
    with torch.no_grad():
        loss = 0.0
        if observed.numel() > 0:
            f_obs = decoder(observed.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
            loss += float(F.binary_cross_entropy_with_logits(
                f_obs, torch.ones_like(f_obs)
            ).item())
        if free is not None and free.numel() > 0:
            f_free = decoder(free.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
            loss += float(F.binary_cross_entropy_with_logits(
                f_free, torch.zeros_like(f_free)
            ).item())
        return -loss + float(-0.5 * z.pow(2).sum().item())


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    object_index = REPO_ROOT / "outputs/carnet/spcarnet/object_index_v1.json"
    if not object_index.is_file():
        print(f"[stage5-smoke] FAIL: object index missing at {object_index}", file=sys.stderr)
        return 1
    dataset = SPCarObjectDataset(object_index, splits=("train",))
    if len(dataset) < 1:
        print("[stage5-smoke] FAIL: empty dataset", file=sys.stderr)
        return 1
    item = dataset[0]
    print(f"[stage5-smoke] dataset_ok n_objects={len(dataset)}", file=sys.stderr)

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

    partial_np = item.get("partial_observed_points")
    if partial_np is None or partial_np.size == 0:
        partial_np = item["clean_points_object"]
    partial_t = torch.from_numpy(np.asarray(partial_np[:64], dtype=np.float32)).to(device)
    free_np = item.get("free_space_query_points")
    free_t = (
        torch.from_numpy(np.asarray(free_np[:64], dtype=np.float32)).to(device)
        if free_np is not None and len(free_np) > 0
        else None
    )

    K = 4
    z_list: list[torch.Tensor] = []
    scores: list[float] = []
    for k in range(K):
        torch.manual_seed(1000 + k)
        with torch.no_grad():
            post = completion.encode(partial_t.unsqueeze(0), sample=True)
            z_k = post.z.squeeze(0).clone()
        z_list.append(z_k)
        score = _score_no_grad(decoder, z_k, observed=partial_t, free=free_t)
        scores.append(score)
        assert np.isfinite(score), f"non-finite score at k={k}"
    print(f"[stage5-smoke] sampled_K={K}", file=sys.stderr)

    # Pairwise latent L2 — assert >1e-4 (distinct).
    z_stack = torch.stack(z_list, dim=0)
    d_lat = torch.cdist(z_stack, z_stack)
    idx = torch.triu_indices(K, K, offset=1)
    pairs = d_lat[idx[0], idx[1]]
    diversity = float(pairs.mean().item())
    min_pair = float(pairs.min().item())
    assert min_pair > 1e-4, f"two candidates collapsed (min pairwise L2 = {min_pair})"
    print(
        f"[stage5-smoke] diversity_latent_l2={diversity:.6f} min_pair={min_pair:.6f}",
        file=sys.stderr,
    )

    # Score finiteness + ranking sanity.
    assert all(np.isfinite(s) for s in scores), "some scores were non-finite"
    print(
        f"[stage5-smoke] scores={['%.4f' % s for s in scores]} "
        f"top1_idx={int(np.argmax(scores))}",
        file=sys.stderr,
    )

    print("[stage5-smoke] PASS", file=sys.stderr)
    print(json.dumps({
        "K": K,
        "scores": scores,
        "diversity_latent_l2": diversity,
        "min_pairwise_latent_l2": min_pair,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
