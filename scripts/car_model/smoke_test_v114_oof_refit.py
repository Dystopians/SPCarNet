#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.build_v105_evidence_gated_mixture_field import (
    BASIS_ORDER,
    XTX_ORDER,
    _crossfit_weighted_risk_gain_and_scale,
)


def _identity_xtx(triangle_count: int) -> torch.Tensor:
    feature_count = len(BASIS_ORDER)
    out = torch.zeros((triangle_count, len(XTX_ORDER)), dtype=torch.float64)
    for slot, (row, col) in enumerate(XTX_ORDER):
        if row == col:
            out[:, slot] = 1.0
    if feature_count <= 0:
        raise AssertionError("empty basis")
    return out


def main() -> int:
    triangle_count = 3
    feature_count = len(BASIS_ORDER)
    xtx_even = _identity_xtx(triangle_count)
    xtx_odd = _identity_xtx(triangle_count)
    xty_even = torch.zeros((triangle_count, feature_count, 3), dtype=torch.float64)
    xty_odd = torch.zeros_like(xty_even)
    yty_even = torch.ones((triangle_count,), dtype=torch.float64)
    yty_odd = torch.ones_like(yty_even)
    view_counts_even = torch.ones((triangle_count,), dtype=torch.int32)
    view_counts_odd = torch.ones((triangle_count,), dtype=torch.int32)

    xty_even[:, 0, 0] = 1.0
    xty_odd[:, 0, 0] = 1.0
    view_counts_odd[1] = 0
    view_counts_even[2] = 0
    base = torch.zeros((triangle_count, feature_count, 3), dtype=torch.float32)
    tri_ids = torch.arange(triangle_count, dtype=torch.long)

    strict_gain, _, strict_scale, strict_support, strict_stats = _crossfit_weighted_risk_gain_and_scale(
        name="smoke",
        base_coeffs=base,
        tri_ids=tri_ids,
        xtx_even=xtx_even,
        xty_even=xty_even,
        yty_even=yty_even,
        view_counts_even=view_counts_even,
        xtx_odd=xtx_odd,
        xty_odd=xty_odd,
        yty_odd=yty_odd,
        view_counts_odd=view_counts_odd,
        min_count=1,
        min_views=1,
        ridge=0.0,
        view_std_floor=1e-4,
        rank_rtol=1e-7,
        condition_max=1e8,
        combine_mode="strict_min",
    )
    oof_gain, _, oof_scale, oof_support, oof_stats = _crossfit_weighted_risk_gain_and_scale(
        name="smoke",
        base_coeffs=base,
        tri_ids=tri_ids,
        xtx_even=xtx_even,
        xty_even=xty_even,
        yty_even=yty_even,
        view_counts_even=view_counts_even,
        xtx_odd=xtx_odd,
        xty_odd=xty_odd,
        yty_odd=yty_odd,
        view_counts_odd=view_counts_odd,
        min_count=1,
        min_views=1,
        ridge=0.0,
        view_std_floor=1e-4,
        rank_rtol=1e-7,
        condition_max=1e8,
        combine_mode="oof_positive_cap",
    )

    if int(strict_support.sum().item()) != 1:
        raise AssertionError(f"strict support should keep only both-fold triangle: {strict_support}")
    if int(oof_support.sum().item()) != 3:
        raise AssertionError(f"oof support should keep both positive one-sided triangles: {oof_support}")
    if not (oof_gain[1] > 0.0 and oof_gain[1] < oof_gain[0]):
        raise AssertionError(f"one-sided gain should be positive and downweighted: {oof_gain.tolist()}")
    if not (oof_scale[1] > 0.0 and oof_scale[1] < strict_scale[0]):
        raise AssertionError(f"one-sided scale should be downweighted: {oof_scale.tolist()} vs {strict_scale.tolist()}")
    if strict_stats["smoke_crossfit_combine_mode"] != "strict_min":
        raise AssertionError(strict_stats)
    if oof_stats["smoke_crossfit_combine_mode"] != "oof_positive_cap":
        raise AssertionError(oof_stats)

    print("v114 OOF refit smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
