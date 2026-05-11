#!/usr/bin/env python3
"""Smoke-test utils/fd_loss against a numpy reference and a real backbone forward."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.fd_loss import (
    FrozenReprConfig,
    FrozenReprModel,
    frechet_distance,
    frechet_distance_loss,
    empirical_gaussian,
    sqrtm_psd,
)


def numpy_fd(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    ca = a - mu_a
    cb = b - mu_b
    sa = ca.T @ ca / (a.shape[0] - 1) + eps * np.eye(a.shape[1])
    sb = cb.T @ cb / (b.shape[0] - 1) + eps * np.eye(b.shape[1])
    wa, va = np.linalg.eigh(0.5 * (sa + sa.T))
    sqrt_sa = (va * np.sqrt(np.clip(wa, 0.0, None))) @ va.T
    middle = sqrt_sa @ sb @ sqrt_sa
    wm, vm = np.linalg.eigh(0.5 * (middle + middle.T))
    sqrt_middle = (vm * np.sqrt(np.clip(wm, 0.0, None))) @ vm.T
    mean_term = float(np.sum((mu_a - mu_b) ** 2))
    trace_term = float(np.trace(sa) + np.trace(sb) - 2.0 * np.trace(sqrt_middle))
    return mean_term + trace_term


def test_fd_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(64, 32)).astype(np.float64)
    b = rng.normal(loc=0.5, scale=1.2, size=(80, 32)).astype(np.float64)
    np_fd = numpy_fd(a, b)
    torch_fd = frechet_distance(torch.from_numpy(a), torch.from_numpy(b))["fd"]
    rel = abs(np_fd - torch_fd) / max(abs(np_fd), 1e-8)
    print(f"[fd numpy ref] np={np_fd:.6f} torch={torch_fd:.6f} rel_err={rel:.3e}")
    assert rel < 1e-4, f"FD mismatch: np={np_fd}, torch={torch_fd}"


def test_fd_zero_for_equal_batches() -> None:
    rng = np.random.default_rng(1)
    a = torch.from_numpy(rng.normal(size=(48, 64)).astype(np.float64))
    fd = frechet_distance(a, a.clone())["fd"]
    print(f"[fd self] {fd:.3e}")
    assert fd < 1e-6, f"Self-FD should be ~0, got {fd}"


def test_fd_loss_matches_two_batch() -> None:
    rng = np.random.default_rng(2)
    a = torch.from_numpy(rng.normal(size=(96, 48)).astype(np.float64))
    b = torch.from_numpy(rng.normal(loc=0.3, size=(96, 48)).astype(np.float64))
    two_batch = frechet_distance(a, b)["fd"]
    mu_b, sigma_b = empirical_gaussian(b)
    sqrt_b = sqrtm_psd(sigma_b)
    loss = float(frechet_distance_loss(a, mu_b, sigma_b, sqrt_b).detach().cpu().item())
    rel = abs(two_batch - loss) / max(abs(two_batch), 1e-8)
    print(f"[fd loss vs 2-batch] two={two_batch:.6f} loss={loss:.6f} rel_err={rel:.3e}")
    assert rel < 1e-4, f"frechet_distance_loss != two-batch FD"


def test_backbone_forward(model_name: str = "vit_base_patch14_dinov2.lvd142m") -> None:
    if not torch.cuda.is_available():
        print("[backbone] CUDA unavailable, skipping")
        return
    cfg = FrozenReprConfig(model_name=model_name, pool_type="cls")
    judge = FrozenReprModel(cfg, device="cuda")
    fake_render = torch.rand(4, 3, 512, 640)
    fake_gt = torch.rand(4, 3, 512, 640)
    feats_r = judge.encode(fake_render)
    feats_g = judge.encode(fake_gt)
    print(
        f"[backbone] {model_name} image_size={judge.image_size} "
        f"D={judge.feature_dim} pool={judge.pool_type} "
        f"feats_r={tuple(feats_r.shape)} feats_g={tuple(feats_g.shape)}"
    )
    assert feats_r.shape == (4, judge.feature_dim)
    fd = frechet_distance(feats_r, feats_g)
    print(f"[backbone fd] random-vs-random fd={fd['fd']:.4f} mean={fd['mean_term']:.4f} trace={fd['trace_term']:.4f}")


if __name__ == "__main__":
    test_fd_matches_numpy()
    test_fd_zero_for_equal_batches()
    test_fd_loss_matches_two_batch()
    test_backbone_forward()
    print("OK")
