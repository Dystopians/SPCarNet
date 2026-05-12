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


class StubJudge:
    """Deterministic FD judge for integration tests: maps each [3,H,W] image to a
    8-d feature derived from its per-channel mean and crude texture stats."""

    image_size = 4

    def prepare(self, image):
        if image.ndim == 3:
            x = image.unsqueeze(0)
        else:
            x = image
        x = torch.nn.functional.interpolate(
            x.float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        return x.squeeze(0)

    def encode(self, images):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        flat = images.float().reshape(images.shape[0], 3, -1)
        means = flat.mean(dim=2)
        stds = flat.std(dim=2)
        maxes = flat.amax(dim=2)
        minus = flat.amin(dim=2)
        feats = torch.cat([means, stds, maxes, minus], dim=1)
        return feats[:, :8].contiguous()


def _make_synthetic_frame(tmp, split, idx, base_value, residual_strength):
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from utils.evidence_lumigraph_adapter import CameraRecord, FrameRecord

    root = Path(tmp)
    method = root / split / "ours_1"
    h = w = 8
    base = np.full((h, w, 3), base_value, dtype=np.float32)
    residual = np.zeros((h, w, 3), dtype=np.float32)
    residual[:, 2:6, 0] = residual_strength
    gt = base + residual
    depth = np.ones((h, w), dtype=np.float32)
    render = method / "renders" / f"{idx:05d}.png"
    target = method / "gt" / f"{idx:05d}.png"
    depth_path = method / "depths" / f"{idx:05d}.npy"
    for p in (render, target, depth_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(base * 255, 0, 255).astype(np.uint8)).save(render)
    Image.fromarray(np.clip(gt * 255, 0, 255).astype(np.uint8)).save(target)
    np.save(depth_path, depth)
    cam = CameraRecord(
        idx=idx,
        image_name=f"cam{idx}",
        width=w,
        height=h,
        fx=float(w),
        fy=float(h),
        camera_center=(0.0, 0.0, 0.0),
        world_view_transform=tuple(tuple(float(v) for v in row) for row in __import__("numpy").eye(4, dtype="float32")),
    )
    return FrameRecord(
        idx=idx, name=f"{idx:05d}", render_path=render, gt_path=target, depth_path=depth_path, camera=cam
    )


def test_calibrate_alpha_fd_integration() -> None:
    """End-to-end: FD path runs, alpha=0 is exempt from strict rejection, fd_min_views guard works."""
    import tempfile

    from utils.evidence_lumigraph_adapter import calibrate_alpha

    judge = StubJudge()
    with tempfile.TemporaryDirectory() as tmp:
        frames = [
            _make_synthetic_frame(tmp, "train", i, base_value=0.25, residual_strength=0.20)
            for i in range(4)
        ]
        # FD enabled but fd_views_taken (2) is below default fd_min_views (8) -> should skip cleanly
        calib_skip = calibrate_alpha(
            frames,
            alpha_grid=[0.0, 0.5, 1.0],
            k=1, mode="residual",
            calib_stride=1, calib_max_views=2, calib_sampler="uniform",
            residual_clip=1.0,
            depth_abs_tol=0.001, depth_rel_tol=0.001, direction_weight=0.0,
            device="cpu",
            fd_judge=judge, fd_weight=1.0, fd_strict=True, fd_max_views=4, fd_min_views=8,
        )
        assert calib_skip["fd_requested"] is True
        assert calib_skip["fd_enabled"] is False, calib_skip
        assert "insufficient" in (calib_skip["fd_skipped_reason"] or ""), calib_skip
        # Lower fd_min_views to allow FD to run on the available 2 views
        calib_on = calibrate_alpha(
            frames,
            alpha_grid=[0.0, 0.5, 1.0],
            k=1, mode="residual",
            calib_stride=1, calib_max_views=2, calib_sampler="uniform",
            residual_clip=1.0,
            depth_abs_tol=0.001, depth_rel_tol=0.001, direction_weight=0.0,
            device="cpu",
            fd_judge=judge, fd_weight=1.0, fd_strict=True, fd_max_views=4, fd_min_views=2,
        )
        assert calib_on["fd_enabled"] is True, calib_on
        rows = {float(r["alpha"]): r for r in calib_on["rows"]}
        # R3: alpha=0 fd_gain must be exactly zero because we reuse base feats.
        assert rows[0.0]["fd_gain"] == 0.0, rows[0.0]
        # R3: alpha=0 must never be marked fd_rejected.
        assert rows[0.0]["fd_rejected"] is False
        # FD fields exist for all alphas.
        for a in (0.0, 0.5, 1.0):
            assert rows[a]["fd"] is not None, rows[a]
            assert rows[a]["base_fd"] is not None, rows[a]
            assert rows[a]["fd_views"] == 2
    print("[fd integration] passed")


if __name__ == "__main__":
    test_fd_matches_numpy()
    test_fd_zero_for_equal_batches()
    test_fd_loss_matches_two_batch()
    test_backbone_forward()
    test_calibrate_alpha_fd_integration()
    print("OK")
