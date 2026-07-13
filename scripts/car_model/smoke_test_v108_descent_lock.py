#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.build_v105_evidence_gated_mixture_field import (
    _joint_two_expert_mse_descent_lock,
)


def _toy_system() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xtx = torch.zeros((2, 21), dtype=torch.float64)
    xtx[:, 0] = 1.0
    xty = torch.zeros((2, 6, 3), dtype=torch.float64)
    xty[:, 0, 0] = 1.0
    yty = torch.ones((2,), dtype=torch.float64)
    base = torch.zeros((2, 6, 3), dtype=torch.float64)
    delta = torch.zeros((2, 2, 6, 3), dtype=torch.float64)
    delta[0, 0, 0, 0] = -1.0
    delta[0, 1, 0, 0] = 1.0
    delta[1, 0, 0, 0] = -1.0
    delta[1, 1, 0, 0] = -0.5
    return xtx, xty, yty, base, delta


def _sse(xty: torch.Tensor, yty: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    return yty - 2.0 * (coeff * xty).sum(dim=(1, 2)) + coeff[:, 0, :].square().sum(dim=1)


def main() -> int:
    xtx, xty, yty, base, delta = _toy_system()
    scales, gains, objective_delta, support = _joint_two_expert_mse_descent_lock(
        eval_xtx_flat=xtx,
        eval_xty=xty,
        eval_yty=yty,
        base_coeffs=base,
        expert_delta=delta,
        tri_ids=torch.arange(2),
    )

    if not bool(support.all().item()):
        raise AssertionError(f"expected both toy triangles to be supported, got {support.tolist()}")

    corrected = base + scales[:, 0, None, None] * delta[:, 0] + scales[:, 1, None, None] * delta[:, 1]
    base_sse = _sse(xty, yty, base)
    corrected_sse = _sse(xty, yty, corrected)
    if bool((corrected_sse > base_sse + 1.0e-8).any().item()):
        raise AssertionError(f"descent lock increased SSE: base={base_sse.tolist()} corrected={corrected_sse.tolist()}")

    if float(scales[0, 0].item()) > 1.0e-8:
        raise AssertionError(f"bad detail expert should be suppressed on tri0, scales={scales[0].tolist()}")
    if float(scales[0, 1].item()) <= 0.5:
        raise AssertionError(f"good boundary expert should be retained on tri0, scales={scales[0].tolist()}")
    if float(scales[1].abs().max().item()) > 1.0e-8:
        raise AssertionError(f"all-bad tri1 experts should be suppressed, scales={scales[1].tolist()}")
    if float(gains[0].item()) <= 0.0 or float(objective_delta[0].item()) >= 0.0:
        raise AssertionError(
            f"tri0 should have a positive descent gain, gain={gains[0].item()} delta={objective_delta[0].item()}"
        )

    print(
        "v108 descent-lock smoke passed: "
        f"scales={scales.tolist()}, gains={gains.tolist()}, corrected_sse={corrected_sse.tolist()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
