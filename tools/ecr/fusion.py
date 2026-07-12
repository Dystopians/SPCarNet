"""GEMS Stage-4 ladder L3: learned per-pixel fusion (the Deep-Blending move).

A small U-Net is trained ONCE per scene on TRAIN views only (leave-one-out
transport features -> per-pixel blend weight), then FROZEN and applied
identically to every test view (LEDGER GOAL #E-04 pre-registration; legal
per prompt §1 — a per-scene MODEL, not per-test-view arbitration).

Inputs (9 channels, all render-time-legal, from the evidence machinery whose
error-prediction power is banked at Spearman rho ~= 0.7):
    base render (3) · fused transport signal (3) · total warp confidence (1)
    · support count / 8 (1) · cross-support residual std (1)
Output: alpha map in [0,1]; composed frame = clamp(base + alpha * signal).

The net rides the incumbent transport's fuse for the SIGNAL (single-band
PJ-2026 or L2 multiband — frozen per cache manifest); confidence statistics
are always the single-band accumulators (same definitions as the ELA
EvidenceSignal fields).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.evidence_lumigraph_adapter import (
    _frame_valid_mask,
    _make_target_world_grid,
    select_support_frames,
    warp_support_residual,
)
from tools.ecr.transport_l2 import multiband_fuse


def compute_transport_features(
    target,
    support_frames,
    *,
    k: int,
    residual_clip: float,
    min_confidence: float,
    depth_abs_tol: float,
    depth_rel_tol: float,
    direction_weight: float,
    fuse: str = "single",
    bands: int = 4,
    with_color: bool = False,
    loader=None,
    device: torch.device | str = "cuda",
) -> dict:
    """Warp + fuse WITHOUT applying: returns base, signal and the confidence
    feature planes for one target view. Reads only through `loader`.
    with_color additionally warps + fuses the support TRAIN GT colors
    (L4 routing input)."""
    device = torch.device(device)
    base = loader.render(str(target.render_path)).to(device)
    target_depth = loader.depth(str(target.depth_path)).to(device)
    support = select_support_frames(
        target, support_frames, k=int(k),
        exclude_names={target.name, target.camera.image_name},
        direction_weight=float(direction_weight))
    world_grid = _make_target_world_grid(target.camera, target_depth,
                                         device=device)
    h, w = base.shape[-2:]
    residuals, colors, confidences, used = [], [], [], []
    for frame, view_weight in support:
        support_depth = loader.depth(str(frame.depth_path))
        frame_valid = _frame_valid_mask(loader, frame)
        warped, conf = warp_support_residual(
            target, frame, target_depth,
            support_depth,
            loader.residual(frame, residual_clip=residual_clip),
            depth_abs_tol=float(depth_abs_tol),
            depth_rel_tol=float(depth_rel_tol),
            target_world_grid=world_grid, device=device,
            support_valid=frame_valid)
        weight = conf.unsqueeze(0) * float(view_weight)
        if float(weight.mean().item()) <= 0.0:
            continue
        if with_color:
            warped_color, _ = warp_support_residual(
                target, frame, target_depth,
                support_depth,
                loader.gt(str(frame.gt_path)),
                depth_abs_tol=float(depth_abs_tol),
                depth_rel_tol=float(depth_rel_tol),
                target_world_grid=world_grid, device=device,
                support_valid=frame_valid)
            colors.append(warped_color)
        residuals.append(warped)
        confidences.append(weight)
        used.append(frame.name)
    zeros1 = torch.zeros(1, h, w, device=device)
    if not residuals:
        out = {"base": base, "signal": torch.zeros_like(base),
               "weight_den": zeros1, "support_count": zeros1,
               "residual_std": zeros1, "support_names": []}
        if with_color:
            out["color"] = torch.zeros_like(base)
        return out
    # single-band accumulators (ELA EvidenceSignal definitions)
    sig_num = torch.zeros_like(base)
    sig_sq = torch.zeros_like(base)
    weight_den = torch.zeros_like(zeros1)
    support_count = torch.zeros_like(zeros1)
    for warped, weight in zip(residuals, confidences):
        sig_num = sig_num + warped * weight
        sig_sq = sig_sq + warped.pow(2) * weight
        weight_den = weight_den + weight
        support_count = support_count + (weight > float(min_confidence)).to(
            torch.float32)
    valid = weight_den > float(min_confidence)
    mean = torch.where(valid, sig_num / torch.clamp(weight_den, min=1e-8),
                       torch.zeros_like(sig_num))
    mean_sq = torch.where(valid, sig_sq / torch.clamp(weight_den, min=1e-8),
                          torch.zeros_like(sig_sq))
    variance = torch.clamp(mean_sq - mean.pow(2), min=0.0)
    residual_std = torch.sqrt(torch.mean(variance, dim=0, keepdim=True) + 1e-12)
    residual_std = torch.where(valid, residual_std,
                               torch.zeros_like(residual_std))
    if fuse == "multiband":
        signal, _ = multiband_fuse(residuals, confidences, bands=int(bands),
                                   min_confidence=float(min_confidence))
    else:
        signal = mean
    out = {"base": base, "signal": signal, "weight_den": weight_den,
           "support_count": support_count, "residual_std": residual_std,
           "support_names": used}
    if with_color:
        if fuse == "multiband":
            color, _ = multiband_fuse(colors, confidences, bands=int(bands),
                                      min_confidence=float(min_confidence))
        else:
            color_num = torch.zeros_like(base)
            for wc, weight in zip(colors, confidences):
                color_num = color_num + wc * weight
            color = torch.where(valid,
                                color_num / torch.clamp(weight_den, min=1e-8),
                                torch.zeros_like(color_num))
        out["color"] = torch.clamp(color, 0.0, 1.0)
    return out


def _conf_planes(feats: dict, ablate_conf: bool) -> list:
    """The 3 confidence-derived net-input planes; zeroed under the CR4
    ablation (net loses access to certified confidence as INPUT — the
    transport/compose machinery itself is untouched)."""
    if ablate_conf:
        z = torch.zeros_like(feats["weight_den"])
        return [z, z.clone(), z.clone()]
    return [
        torch.clamp(feats["weight_den"], 0.0, 4.0) / 4.0,
        torch.clamp(feats["support_count"], 0.0, 8.0) / 8.0,
        torch.clamp(feats["residual_std"], 0.0, 0.5) * 2.0,
    ]


def features_to_input(feats: dict, ablate_conf: bool = False) -> torch.Tensor:
    """Stack the 9-channel net input from compute_transport_features."""
    return torch.cat([
        feats["base"],
        feats["signal"],
        *_conf_planes(feats, ablate_conf),
    ], dim=0)


def features_to_input_routed(feats: dict,
                             ablate_conf: bool = False) -> torch.Tensor:
    """Stack the 12-channel routed net input (L4): 9-ch + fused color."""
    return torch.cat([
        feats["base"],
        feats["signal"],
        feats["color"],
        *_conf_planes(feats, ablate_conf),
    ], dim=0)


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.norm2 = nn.GroupNorm(8, cout)

    def forward(self, x):
        x = F.silu(self.norm1(self.conv1(x)))
        return F.silu(self.norm2(self.conv2(x)))


class FusionNet(nn.Module):
    """U-Net-S -> per-pixel maps in [0,1]. ~0.9M params; frozen arch.

    L3 (default): cin=9, out=1 (alpha), head bias +1.0 -> alpha ~= 0.73 at
    init (close to the PJ-2026 calibrated global alphas).
    L4 (routed): cin=12, out=2 (alpha, beta), head bias (+1.0, -4.0) ->
    routing starts OFF (beta ~= 0.018), i.e. training starts at L3 behavior.
    """

    def __init__(self, cin: int = 9, base: int = 32, out_channels: int = 1,
                 head_bias: tuple = (1.0,)):
        super().__init__()
        self.enc1 = _Block(cin, base)
        self.enc2 = _Block(base, base * 2)
        self.enc3 = _Block(base * 2, base * 4)
        self.mid = _Block(base * 4, base * 4)
        self.dec3 = _Block(base * 4 + base * 4, base * 2)
        self.dec2 = _Block(base * 2 + base * 2, base)
        self.dec1 = _Block(base + base, base)
        self.head = nn.Conv2d(base, int(out_channels), 3, padding=1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            for c, b in enumerate(head_bias):
                self.head.bias[c] = float(b)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        m = self.mid(F.avg_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([
            F.interpolate(m, size=e3.shape[-2:], mode="bilinear",
                          align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([
            F.interpolate(d3, size=e2.shape[-2:], mode="bilinear",
                          align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=e1.shape[-2:], mode="bilinear",
                          align_corners=False), e1], dim=1))
        alpha = torch.sigmoid(self.head(d1))
        if ph or pw:
            alpha = alpha[..., :h, :w]
        return alpha


def apply_fusion(net: FusionNet, feats: dict,
                 ablate_conf: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    x = features_to_input(feats, ablate_conf=ablate_conf).unsqueeze(0)
    alpha = net(x).squeeze(0)
    return torch.clamp(feats["base"] + alpha * feats["signal"], 0.0, 1.0), alpha


def compose_routed(base: torch.Tensor, signal: torch.Tensor,
                   color: torch.Tensor, valid: torch.Tensor,
                   maps: torch.Tensor) -> torch.Tensor:
    """L4 composition: out = (1-beta·valid)·(base + alpha·signal)
    + beta·valid·color. maps: [2,H,W] (alpha, beta) or [B,2,h,w]."""
    if maps.dim() == 3:
        alpha, beta = maps[0:1], maps[1:2]
    else:
        alpha, beta = maps[:, 0:1], maps[:, 1:2]
    beta = beta * valid.to(maps.dtype)
    corrected = torch.clamp(base + alpha * signal, 0.0, 1.0)
    return torch.clamp((1.0 - beta) * corrected + beta * color, 0.0, 1.0)


def apply_routed_fusion(net: FusionNet, feats: dict,
                        min_confidence: float = 1e-4,
                        ablate_conf: bool = False
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    x = features_to_input_routed(feats, ablate_conf=ablate_conf).unsqueeze(0)
    maps = net(x).squeeze(0)
    valid = feats["weight_den"] > float(min_confidence)
    out = compose_routed(feats["base"], feats["signal"], feats["color"],
                         valid, maps)
    return out, maps
