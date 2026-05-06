#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import _compute_parent_render_rollback_loss  # noqa: E402


class _Cam:
    image_name = "view"


def _run(aggregation: str) -> dict:
    gt = torch.zeros((3, 4, 4), dtype=torch.float32)
    parent = torch.zeros((3, 4, 4), dtype=torch.float32)
    image = torch.zeros((3, 4, 4), dtype=torch.float32)
    image[:, 0, 0] = 0.04
    image[:, 0, 1] = 0.10
    image[:, 0, 2] = 0.90
    cache = {"view": parent}
    return _compute_parent_render_rollback_loss(
        viewpoint_cam=_Cam(),
        image=image,
        gt_image=gt,
        parent_cache=cache,
        lam=1.0,
        margin_abs=0.0,
        huber_delta=0.02,
        aggregation=aggregation,
        cvar_fraction=0.25,
        cvar_min_pixels=1,
        patch_radius=0,
        patch_reduce="center",
        error_space="l1",
    )


def main() -> int:
    mean = _run("mean")
    cvar = _run("cvar")
    assert mean is not None and cvar is not None
    assert int(mean["active_pixels"]) == 3, mean
    assert int(cvar["tail_pixels"]) == 1, cvar
    assert float(cvar["loss_pure"]) > float(mean["loss_pure"]), (mean, cvar)

    gt = torch.zeros((3, 2, 2), dtype=torch.float32)
    parent = torch.zeros((3, 2, 2), dtype=torch.float32)
    image = torch.zeros((3, 2, 2), dtype=torch.float32)
    no_regress = _compute_parent_render_rollback_loss(
        viewpoint_cam=_Cam(),
        image=image,
        gt_image=gt,
        parent_cache={"view": parent},
        lam=1.0,
    )
    assert no_regress is None, no_regress
    print("SCE23 parent-render tail rollback smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
