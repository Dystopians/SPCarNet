"""Smoke test for CarNet_v0.8.1 — verifies the v12 model with EdgeConv +
corrupted warm-start + K=8 constructs, runs forward+backward, and produces
a finite ``point_flow_matching_loss``.

Loads the actual production yaml so any field-name mismatch surfaces here
rather than 30 minutes into a training run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path("/data/peilincai/mesh-splatting")
sys.path.insert(0, str(ROOT))

from ss3dm_prior.models.patch_denoiser import LocalPatchDenoiser  # noqa: E402

CFG = ROOT / "configs/ss3dm_prior/carnet_v0_8_1/model_carnet_v0_8_1.yaml"


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text())
    model_kwargs = dict(cfg["model"])
    print(f"[smoke] model_type={model_kwargs.get('model_type')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LocalPatchDenoiser(**model_kwargs).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] params={n_params/1e6:.2f}M")

    # Synthetic batch matching v0.8.1 patch size (recon_point_count=2048).
    B = 2
    N = 2048
    torch.manual_seed(0)
    clean = torch.randn(B, N, 3, device=device) * 0.3
    # corrupted = clean + small noise + dropout some points (replace with random)
    corrupted = clean + 0.06 * torch.randn_like(clean)
    drop = torch.rand(B, N, 1, device=device) < 0.25
    corrupted = torch.where(drop, torch.randn_like(clean) * 0.3, corrupted)
    cn = torch.nn.functional.normalize(torch.randn(B, N, 3, device=device), dim=-1)
    observed = clean + 0.02 * torch.randn_like(clean)

    print(f"[smoke] batch B={B} N={N} device={device}")

    outputs = model(
        corrupted_points=corrupted,
        corrupted_normals=cn,
        clean_points=clean,
        observed_points=observed,
    )

    # Required v0.8.1 outputs
    assert "point_flow_matching_loss" in outputs, "missing point_flow_matching_loss"
    pf = outputs["point_flow_matching_loss"]
    print(f"[smoke] point_flow_matching_loss={pf.item():.4f} finite={torch.isfinite(pf).item()}")
    assert torch.isfinite(pf), "non-finite point_flow_matching_loss"

    # recon should match patch size; finite
    recon = outputs.get("recon_points")
    assert recon is not None and recon.shape == (B, N, 3), f"recon shape={recon.shape}"
    assert torch.isfinite(recon).all(), "recon contains non-finite"
    print(f"[smoke] recon shape={tuple(recon.shape)} mean_abs={recon.abs().mean().item():.4f}")

    # Backward (step 1: zero-init final linear → only last layer has grad)
    pf.backward()
    n1 = sum(
        1
        for p in model.parameters()
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    )
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[smoke] step1 params_with_grad={n1}/{n_total} (zero-init expected)")

    # One Adam step, then re-forward — verifies gradient propagates through
    # the WHOLE graph (encoder + decoder) once the zero-init wears off.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad()
    out2 = model(
        corrupted_points=corrupted,
        corrupted_normals=cn,
        clean_points=clean,
        observed_points=observed,
    )
    pf2 = out2["point_flow_matching_loss"]
    pf2.backward()
    n2 = sum(
        1
        for p in model.parameters()
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    )
    print(f"[smoke] step2 params_with_grad={n2}/{n_total} (must be >> step1)")
    assert n2 > 100, f"encoder appears disconnected: only {n2} params receive grad"

    # Quick sanity check: edge_conv module is present
    pf_dec = model.impl.point_flow_decoder
    print(
        f"[smoke] decoder.use_edge_conv={pf_dec.use_edge_conv} "
        f"k={pf_dec.edge_conv_k} steps={model.impl.point_flow_steps} "
        f"init_mode={model.impl.point_flow_init_mode!r} "
        f"init_noise_scale={model.impl.point_flow_init_noise_scale}"
    )

    print("[smoke] OK")


if __name__ == "__main__":
    main()
