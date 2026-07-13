#!/usr/bin/env python3
from __future__ import annotations

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import _compute_teacher_render_loss


class _Camera:
    image_name = "00000.png"


def _loss_for(parent_value: float, teacher_value: float, image_value: float, *, delta_min: float):
    gt = torch.zeros((3, 4, 4), dtype=torch.float32)
    image = torch.full_like(gt, float(image_value))
    teacher = torch.full_like(gt, float(teacher_value))
    parent = torch.full_like(gt, float(parent_value))
    return _compute_teacher_render_loss(
        viewpoint_cam=_Camera(),
        image=image,
        gt_image=gt,
        teacher_cache={"00000.png": teacher},
        lam=1.0,
        dssim_weight=0.0,
        mask_mode="teacher_better_current_parent_changed",
        error_margin=0.0,
        parent_cache={"00000.png": parent},
        parent_delta_min=float(delta_min),
    )


def main() -> int:
    active = _loss_for(parent_value=0.50, teacher_value=0.10, image_value=0.40, delta_min=0.05)
    assert active is not None
    assert float(active["mask_fraction"]) == 1.0
    assert torch.isfinite(active["loss_pure"])
    assert float(active["loss_pure"]) > 0.0

    no_parent_gain = _loss_for(parent_value=0.08, teacher_value=0.10, image_value=0.40, delta_min=0.0)
    assert no_parent_gain is None

    no_current_gain = _loss_for(parent_value=0.50, teacher_value=0.30, image_value=0.20, delta_min=0.0)
    assert no_current_gain is None

    no_teacher_delta = _loss_for(parent_value=0.12, teacher_value=0.10, image_value=0.40, delta_min=0.05)
    assert no_teacher_delta is None

    print("[teacher triadic smoke] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
