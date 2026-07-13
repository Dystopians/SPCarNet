#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evidence_lumigraph_adapter import (
    CameraRecord,
    FrameRecord,
    adapt_frame,
    calibrate_alpha,
    fit_alpha_calibrator,
    fit_benefit_calibrator,
    save_camera_index,
    warp_support_residual,
)


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).save(path)


def _cam(idx: int) -> CameraRecord:
    view = np.eye(4, dtype=np.float32)
    return CameraRecord(
        idx=idx,
        image_name=f"cam{idx}",
        width=8,
        height=8,
        fx=8.0,
        fy=8.0,
        camera_center=(0.0, 0.0, 0.0),
        world_view_transform=tuple(tuple(float(v) for v in row) for row in view),
    )


def _frame(root: Path, split: str, idx: int, base: np.ndarray, gt: np.ndarray, depth: np.ndarray) -> FrameRecord:
    method = root / split / "ours_1"
    render = method / "renders" / f"{idx:05d}.png"
    target = method / "gt" / f"{idx:05d}.png"
    depth_path = method / "depths" / f"{idx:05d}.npy"
    _save_rgb(render, base)
    _save_rgb(target, gt)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(depth_path, depth.astype(np.float32))
    return FrameRecord(idx=idx, name=f"{idx:05d}", render_path=render, gt_path=target, depth_path=depth_path, camera=_cam(idx))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        h = w = 8
        base = np.full((h, w, 3), 0.25, dtype=np.float32)
        residual = np.zeros((h, w, 3), dtype=np.float32)
        residual[:, 2:6, 0] = 0.20
        gt = base + residual
        depth = np.ones((h, w), dtype=np.float32)

        support = _frame(root, "train", 0, base, gt, depth)
        target = _frame(root, "train", 1, base, gt, depth)
        for split in ("train",):
            save_camera_index([_cam(0), _cam(1)], root / split / "ours_1" / "camera_index.json")

        warped, confidence = warp_support_residual(
            target,
            support,
            torch.from_numpy(depth),
            torch.from_numpy(depth),
            torch.from_numpy((gt - base).transpose(2, 0, 1)),
            device="cpu",
            depth_abs_tol=0.001,
            depth_rel_tol=0.001,
        )
        err = torch.mean(torch.abs(warped.cpu() - torch.from_numpy((gt - base).transpose(2, 0, 1)))).item()
        assert err < 1e-5, err
        assert float(confidence.mean().item()) > 0.99

        calib = calibrate_alpha(
            [support, target],
            alpha_grid=[0.0, 0.5, 1.0],
            k=1,
            mode="residual",
            calib_stride=1,
            calib_max_views=2,
            calib_sampler="uniform",
            residual_clip=1.0,
            depth_abs_tol=0.001,
            depth_rel_tol=0.001,
            direction_weight=0.0,
            device="cpu",
        )
        assert float(calib["alpha"]) == 1.0, calib

        benefit = fit_benefit_calibrator(
            [support, target],
            k=1,
            mode="residual",
            calib_stride=1,
            calib_max_views=2,
            calib_sampler="uniform",
            residual_clip=1.0,
            depth_abs_tol=0.001,
            depth_rel_tol=0.001,
            direction_weight=0.0,
            bins=2,
            min_bin_count=1,
            device="cpu",
        )
        assert benefit.to_json()["accepted_bins"] > 0, benefit.to_json()
        alpha_calibrator = fit_alpha_calibrator(
            [support, target],
            k=1,
            mode="residual",
            alpha_grid=[0.0, 0.5, 1.0],
            calib_stride=1,
            calib_max_views=2,
            calib_sampler="uniform",
            residual_clip=1.0,
            depth_abs_tol=0.001,
            depth_rel_tol=0.001,
            direction_weight=0.0,
            bins=2,
            min_bin_count=1,
            view_tail_scale_grid=[1.0, 0.5, 0.0],
            view_tail_min_gain=0.0,
            view_tail_max_negative_fraction=0.0,
            device="cpu",
        )
        alpha_json = alpha_calibrator.to_json()
        assert bool(alpha_json["view_tail_enabled"]), alpha_json
        assert float(alpha_json["view_tail_scale"]) in {0.0, 0.5, 1.0}, alpha_json
        assert "view_tail_safe_scale_found" in alpha_json, alpha_json
        assert "view_tail_fallback_used" in alpha_json, alpha_json
        assert isinstance(alpha_json["view_tail_candidate_stats"], list), alpha_json
        assert len(alpha_json["view_tail_candidate_stats"]) == 3, alpha_json
        assert alpha_json["accepted_bins"] > 0, alpha_json
        balanced_alpha = fit_alpha_calibrator(
            [support, target],
            k=1,
            mode="residual",
            alpha_grid=[0.0, 0.5, 1.0],
            calib_stride=1,
            calib_max_views=2,
            calib_sampler="uniform",
            residual_clip=1.0,
            depth_abs_tol=0.001,
            depth_rel_tol=0.001,
            direction_weight=0.0,
            bins=2,
            min_bin_count=1,
            view_tail_scale_grid=[1.0, 0.5, 0.0],
            view_tail_min_gain=0.0,
            view_tail_max_negative_fraction=0.0,
            view_tail_objective="balanced",
            view_tail_compute_lpips=False,
            view_tail_metric_max_side=8,
            device="cpu",
        )
        balanced_json = balanced_alpha.to_json()
        assert balanced_json["view_tail_objective"] == "balanced", balanced_json
        assert bool(balanced_json["view_tail_enabled"]), balanced_json
        assert isinstance(balanced_json["view_tail_candidate_stats"], list), balanced_json
        assert len(balanced_json["view_tail_candidate_stats"]) == 3, balanced_json
        assert "mean_psnr_gain" in balanced_json["view_tail_candidate_stats"][0], balanced_json
        assert "mean_ssim_gain" in balanced_json["view_tail_candidate_stats"][0], balanced_json
        adapted, info = adapt_frame(
            target,
            [support],
            k=1,
            alpha=1.0,
            mode="residual",
            residual_clip=1.0,
            depth_abs_tol=0.001,
            depth_rel_tol=0.001,
            direction_weight=0.0,
            benefit_calibrator=benefit,
            device="cpu",
        )
        target_tensor = torch.from_numpy(gt.transpose(2, 0, 1))
        assert torch.mean(torch.abs(adapted.cpu() - target_tensor)).item() < 1e-2
        assert float(info["covered_fraction"]) > 0.1, info

        trusted, trusted_info = adapt_frame(
            target,
            [support],
            k=1,
            alpha=1.0,
            mode="residual",
            residual_clip=1.0,
            depth_abs_tol=0.001,
            depth_rel_tol=0.001,
            direction_weight=0.0,
            benefit_calibrator=benefit,
            local_trust_gate=True,
            local_trust_min_supports=1,
            local_trust_max_residual_std=0.01,
            local_trust_min_agreement=0.5,
            local_trust_confidence_quantile=0.0,
            device="cpu",
        )
        assert torch.mean(torch.abs(trusted.cpu() - target_tensor)).item() < 1e-2
        assert bool(trusted_info["local_trust_enabled"]), trusted_info
        assert float(trusted_info["local_trust_accept_fraction"]) > 0.1, trusted_info
        assert float(trusted_info["local_trust_mean_support_count"]) >= 1.0, trusted_info

        rejected, rejected_info = adapt_frame(
            target,
            [support],
            k=1,
            alpha=1.0,
            mode="residual",
            residual_clip=1.0,
            depth_abs_tol=0.001,
            depth_rel_tol=0.001,
            direction_weight=0.0,
            benefit_calibrator=benefit,
            local_trust_gate=True,
            local_trust_min_supports=2,
            device="cpu",
        )
        assert float(rejected_info["local_trust_accept_fraction"]) == 0.0, rejected_info
        assert torch.mean(torch.abs(rejected.cpu() - target_tensor)).item() > 0.02

        softened, softened_info = adapt_frame(
            target,
            [support],
            k=1,
            alpha=1.0,
            mode="residual",
            residual_clip=1.0,
            depth_abs_tol=0.001,
            depth_rel_tol=0.001,
            direction_weight=0.0,
            local_trust_gate=True,
            local_trust_mode="soft",
            local_trust_min_supports=2,
            local_trust_min_weight=0.0,
            device="cpu",
        )
        assert softened_info["local_trust_mode"] == "soft", softened_info
        assert float(softened_info["local_trust_mean_weight"]) > 0.0, softened_info
        assert float(softened_info["local_trust_active_fraction"]) > 0.1, softened_info
        softened_err = torch.mean(torch.abs(softened.cpu() - target_tensor)).item()
        rejected_err = torch.mean(torch.abs(rejected.cpu() - target_tensor)).item()
        assert softened_err < rejected_err, (softened_info, softened_err, rejected_err)
    print("[ELA smoke] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
