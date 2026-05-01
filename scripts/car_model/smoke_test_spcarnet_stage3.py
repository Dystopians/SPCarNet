#!/usr/bin/env python3
"""SP-CarNet Stage 3 smoke test.

Validates the posterior encoder + frozen decoder forward / backward path and
checks the four contract items in the design doc §5.

Steps:
1. Build a tiny encoder + tiny decoder (latent_dim=32, hidden_dim=64, depth=3,
   num_xattn_layers=2, num_self_attn_layers=1, num_latent_queries=8).
2. Forward B=2 partial-points batch through the encoder.
3. Sample K=2 latent candidates via reparameterisation; assert distinct.
4. Decode occupancy on a (2, 64, 3) query grid; assert finite and shape (2, 64).
5. Backward pass through L_total; assert non-zero gradients on encoder, zero on decoder.

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

from ss3dm_prior.models.spcarnet_posterior import (  # noqa: E402
    SPCarPosteriorCompletionModel,
    SPCarPosteriorEncoder,
)
from ss3dm_prior.models.spcarnet_shape_field import SPCarShapeFieldDecoder  # noqa: E402


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    encoder = SPCarPosteriorEncoder(
        latent_dim=32,
        feature_dim=48,
        num_xattn_layers=2,
        num_self_attn_layers=1,
        num_latent_queries=8,
        attention_heads=4,
        ffn_dim=96,
        dropout=0.0,
        posterior_kind="variational",
        use_normals=False,
        use_conditioning_adapter=True,
    ).to(device)
    decoder = SPCarShapeFieldDecoder(
        latent_dim=32,
        hidden_dim=64,
        depth=3,
        num_fourier_freqs=8,
        field_kind="occupancy",
        feature_dim=0,
    ).to(device)

    completion = SPCarPosteriorCompletionModel(
        encoder=encoder,
        decoder=decoder,
        decoder_finetune_enabled=False,
    ).to(device)

    # ---------------- Step 2: forward B=2 ----------------
    B, N_partial, Q = 2, 64, 64
    partial = torch.from_numpy(
        rng.normal(0.0, 0.5, size=(B, N_partial, 3)).astype(np.float32)
    ).to(device)
    queries = torch.from_numpy(
        rng.uniform(-1.0, 1.0, size=(B, Q, 3)).astype(np.float32)
    ).to(device)

    completion.train()
    post_a = completion.encode(partial, sample=True)
    assert post_a.z_mean.shape == (B, 32), post_a.z_mean.shape
    assert post_a.z_logvar is not None and post_a.z_logvar.shape == (B, 32), (
        post_a.z_logvar.shape if post_a.z_logvar is not None else "None"
    )
    assert torch.isfinite(post_a.z_mean).all()
    assert torch.isfinite(post_a.z_logvar).all()
    print(
        f"[stage3-smoke] encoder_forward_ok z_mean.shape={tuple(post_a.z_mean.shape)} "
        f"logvar.mean={float(post_a.z_logvar.mean().item()):.4f}",
        file=sys.stderr,
    )

    # ---------------- Step 3: K=2 samples distinct ----------------
    post_b = completion.encode(partial, sample=True)
    delta = (post_a.z - post_b.z).abs().mean().item()
    assert delta > 1e-6, "two reparameterised samples were identical"
    print(f"[stage3-smoke] sampling_ok pairwise_delta={delta:.6f}", file=sys.stderr)

    # ---------------- Step 4: decode through frozen decoder ----------------
    logits_a = completion.decode_field(post_a.z, queries)
    assert logits_a.shape == (B, Q), logits_a.shape
    assert torch.isfinite(logits_a).all()
    print(
        f"[stage3-smoke] decode_ok logits.shape={tuple(logits_a.shape)} "
        f"sigmoid.mean={float(torch.sigmoid(logits_a).mean().item()):.4f}",
        file=sys.stderr,
    )

    # ---------------- Step 5: backward + grad checks ----------------
    z_target = torch.zeros(B, 32, device=device)
    target_lab = torch.full((B, Q), 0.5, device=device)
    out = completion(
        observation={"partial_points": partial},
        query_points={"surf": queries, "free": queries, "hard": queries, "mixed": queries},
        sample=True,
    )
    l_z = ((out["z_mean"] - z_target) ** 2).mean()
    l_kl = 0.5 * (out["z_mean"].pow(2) + out["z_logvar"].exp() - 1 - out["z_logvar"]).sum(dim=-1).mean()
    l_surf = F.binary_cross_entropy_with_logits(out["surf_logits"], torch.ones_like(out["surf_logits"]))
    l_free = F.binary_cross_entropy_with_logits(out["free_logits"], torch.zeros_like(out["free_logits"]))
    total = 10.0 * l_z + 1e-3 * l_kl + l_surf + l_free
    assert torch.isfinite(total)
    completion.zero_grad(set_to_none=True)
    total.backward()
    enc_has_grad = any(
        p.grad is not None and torch.any(p.grad != 0).item()
        for p in encoder.parameters()
        if p.requires_grad
    )
    dec_has_grad = any(
        p.grad is not None and torch.any(p.grad != 0).item()
        for p in decoder.parameters()
    )
    assert enc_has_grad, "encoder parameters did not receive gradients"
    assert not dec_has_grad, "decoder parameters received gradients but should be frozen"
    print(
        f"[stage3-smoke] backward_ok encoder_grad=True decoder_grad=False "
        f"loss={total.detach().item():.4f} l_z={l_z.detach().item():.4f} "
        f"l_kl={l_kl.detach().item():.4f} l_surf={l_surf.detach().item():.4f}",
        file=sys.stderr,
    )

    print("[stage3-smoke] PASS", file=sys.stderr)
    print(
        json.dumps(
            {
                "loss_total": float(total.detach().item()),
                "z_logvar_mean": float(post_a.z_logvar.mean().item()),
                "encoder_grad_flowed": True,
                "decoder_grad_flowed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
