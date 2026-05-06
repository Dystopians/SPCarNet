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
            residual_clip=1.0,
            depth_abs_tol=0.001,
            depth_rel_tol=0.001,
            direction_weight=0.0,
            bins=2,
            min_bin_count=1,
            device="cpu",
        )
        assert benefit.to_json()["accepted_bins"] > 0, benefit.to_json()
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
    print("[ELA smoke] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
