"""GEMS Stage-4 ladder L2 transport: K-source occlusion-aware aggregation
with Laplacian multi-band fusion (PROTOCOL 1.2.0 §4E; LEDGER GOAL #E-03).

Differences vs the frozen PJ-2026 single-band transport (both pre-registered,
2 mechanisms = the rung cap):
  1. K is selected ONCE per scene on train-side leave-one-out from the
     pre-registered grid {2,4,8}, jointly with alpha over the same frozen
     alpha grid (same candidates/stride/objective as the PJ-2026 alpha
     calibration; ties break toward smaller K).
  2. The confidence-weighted average of warped support residuals is replaced
     by a 4-band Laplacian (Burt-Adelson) fusion: residual bands are merged
     with Gaussian-pyramid confidence weights, which removes support-boundary
     seams; band count and kernel are FROZEN (no tuning axis).

Everything else is inherited unchanged from the ELA kernel: nearest-K
selection by distance+direction score, depth-consistency soft z-test
confidence, additive clamped application. Same D4 boundary: reads only
through the caller's confined loader; the target's gt_path is never read.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from utils.evidence_lumigraph_adapter import (
    FrameRecord,
    _make_target_world_grid,
    mse_to_psnr,
    select_support_frames,
    warp_support_residual,
)

_BINOMIAL5 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _blur(x: torch.Tensor) -> torch.Tensor:
    """Separable 5-tap binomial blur, channel-wise. x: [C,H,W]."""
    c = x.shape[0]
    k = _BINOMIAL5.to(device=x.device, dtype=x.dtype)
    kx = k.view(1, 1, 1, 5).expand(c, 1, 1, 5)
    ky = k.view(1, 1, 5, 1).expand(c, 1, 5, 1)
    y = x.unsqueeze(0)
    y = F.conv2d(F.pad(y, (2, 2, 0, 0), mode="reflect"), kx, groups=c)
    y = F.conv2d(F.pad(y, (0, 0, 2, 2), mode="reflect"), ky, groups=c)
    return y.squeeze(0)


def _down(x: torch.Tensor) -> torch.Tensor:
    return _blur(x)[:, ::2, ::2]


def _up(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x.unsqueeze(0), size=size, mode="bilinear",
                         align_corners=False).squeeze(0)


def _gaussian_pyramid(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    pyr = [x]
    for _ in range(levels - 1):
        pyr.append(_down(pyr[-1]))
    return pyr


def _laplacian_pyramid(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    gauss = _gaussian_pyramid(x, levels)
    pyr = []
    for i in range(levels - 1):
        h, w = gauss[i].shape[-2:]
        pyr.append(gauss[i] - _up(gauss[i + 1], (h, w)))
    pyr.append(gauss[-1])
    return pyr


def multiband_fuse(residuals: list[torch.Tensor],
                   confidences: list[torch.Tensor],
                   bands: int = 4,
                   min_confidence: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge warped support residuals with Gaussian-weighted Laplacian bands.

    residuals: list of [3,H,W]; confidences: list of [1,H,W] (view weight
    already folded in). Returns (signal [3,H,W], weight_den [1,H,W]).
    """
    weight_den = torch.zeros_like(confidences[0])
    for conf in confidences:
        weight_den = weight_den + conf
    lap_pyrs = [_laplacian_pyramid(r, bands) for r in residuals]
    conf_pyrs = [_gaussian_pyramid(c, bands) for c in confidences]
    fused_bands = []
    for b in range(bands):
        num = torch.zeros_like(lap_pyrs[0][b])
        den = torch.zeros_like(conf_pyrs[0][b])
        for lp, cp in zip(lap_pyrs, conf_pyrs):
            num = num + lp[b] * cp[b]
            den = den + cp[b]
        fused_bands.append(num / torch.clamp(den, min=1e-8))
    signal = fused_bands[-1]
    for b in range(bands - 2, -1, -1):
        h, w = fused_bands[b].shape[-2:]
        signal = fused_bands[b] + _up(signal, (h, w))
    valid = weight_den > float(min_confidence)
    signal = torch.where(valid, signal, torch.zeros_like(signal))
    return signal, weight_den


def adapt_frame_l2(
    target: FrameRecord,
    support_frames,
    *,
    k: int = 4,
    alpha: float = 1.0,
    residual_clip: float = 0.25,
    min_confidence: float = 1e-4,
    depth_abs_tol: float = 0.02,
    depth_rel_tol: float = 0.03,
    direction_weight: float = 0.35,
    bands: int = 4,
    loader=None,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, dict]:
    """L2 transport for one target view. Same call/return contract as the
    ELA adapt_frame (residual mode only); loader must provide
    render/gt/depth/residual (the ECR ConfinedFrameLoader duck-type)."""
    device = torch.device(device)
    base = loader.render(str(target.render_path)).to(device)
    target_depth = loader.depth(str(target.depth_path)).to(device)
    support = select_support_frames(
        target, support_frames, k=int(k),
        exclude_names={target.name, target.camera.image_name},
        direction_weight=float(direction_weight))
    world_grid = _make_target_world_grid(target.camera, target_depth,
                                         device=device)
    residuals, confidences, used = [], [], []
    for frame, view_weight in support:
        warped, conf = warp_support_residual(
            target, frame, target_depth,
            loader.depth(str(frame.depth_path)),
            loader.residual(frame, residual_clip=residual_clip),
            depth_abs_tol=float(depth_abs_tol),
            depth_rel_tol=float(depth_rel_tol),
            target_world_grid=world_grid, device=device)
        weight = conf.unsqueeze(0) * float(view_weight)
        if float(weight.mean().item()) <= 0.0:
            continue
        residuals.append(warped)
        confidences.append(weight)
        used.append(frame.name)
    if not residuals:
        info = {"support_count": 0, "support_names": [],
                "mean_confidence": 0.0, "covered_fraction": 0.0}
        return base, info
    signal, weight_den = multiband_fuse(
        residuals, confidences, bands=int(bands),
        min_confidence=float(min_confidence))
    valid = weight_den > float(min_confidence)
    adapted = torch.clamp(base + float(alpha) * signal, 0.0, 1.0)
    info = {
        "support_count": len(used),
        "support_names": used,
        "mean_confidence": float(weight_den.mean().detach().cpu().item()),
        "covered_fraction": float(valid.to(torch.float32).mean()
                                  .detach().cpu().item()),
        "bands": int(bands),
    }
    return adapted, info


def calibrate_k_alpha(
    train_frames,
    *,
    k_grid,
    alpha_grid,
    calib_stride: int,
    calib_max_views: int,
    residual_clip: float,
    depth_abs_tol: float,
    depth_rel_tol: float,
    direction_weight: float,
    bands: int,
    min_confidence: float = 1e-4,
    loader=None,
    device: torch.device | str = "cuda",
) -> dict:
    """Joint (K, alpha) train-side leave-one-out calibration for the L2
    transport. Same candidate rule as the PJ-2026 alpha calibration
    (stride_first: every calib_stride-th train frame, capped at
    calib_max_views), same PSNR objective; alpha sweep is analytic in the
    alpha=1 delta. Ties break toward smaller K, then smaller alpha."""
    from utils.evidence_lumigraph_adapter import FrameLoader

    device = torch.device(device)
    loader = loader or FrameLoader(device=device)
    candidates = train_frames[::max(int(calib_stride), 1)][:int(calib_max_views)]
    scores = {(int(kk), float(a)): {"mse": 0.0, "base_mse": 0.0, "count": 0}
              for kk in k_grid for a in alpha_grid}
    for tgt in candidates:
        support = [f for f in train_frames if f.name != tgt.name]
        gt = loader.gt(str(tgt.gt_path))
        base = loader.render(str(tgt.render_path))
        base_mse = float(torch.mean((base - gt) ** 2).item())
        for kk in k_grid:
            adapted1, _ = adapt_frame_l2(
                tgt, support, k=int(kk), alpha=1.0,
                residual_clip=residual_clip, min_confidence=min_confidence,
                depth_abs_tol=depth_abs_tol, depth_rel_tol=depth_rel_tol,
                direction_weight=direction_weight, bands=bands,
                loader=loader, device=device)
            delta = adapted1 - base
            for a in alpha_grid:
                pred = torch.clamp(base + float(a) * delta, 0.0, 1.0)
                cell = scores[(int(kk), float(a))]
                cell["mse"] += float(torch.mean((pred - gt) ** 2).item())
                cell["base_mse"] += base_mse
                cell["count"] += 1
    rows = []
    best = None
    for (kk, a), cell in sorted(scores.items()):
        n = max(cell["count"], 1)
        gain = mse_to_psnr(cell["mse"] / n) - mse_to_psnr(cell["base_mse"] / n)
        rows.append({"k": kk, "alpha": a, "psnr_gain": gain, "views": n})
        key = (-gain, kk, a)
        if best is None or key < best[0]:
            best = (key, kk, a, gain)
    _, best_k, best_alpha, best_gain = best
    return {
        "k": int(best_k),
        "alpha": float(best_alpha),
        "psnr_gain": float(best_gain),
        "rows": rows,
        "calibration_views": [f.name for f in candidates],
        "k_grid": [int(x) for x in k_grid],
        "alpha_grid": [float(x) for x in alpha_grid],
        "objective": "psnr",
        "sampler": "stride_first",
    }
