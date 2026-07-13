#!/usr/bin/env python3
"""Smoke-test POD-MoE view-gate semantics in render.py.

The test is intentionally synthetic and CPU-only. It verifies that v107
`temperature_controlled` fields with `view_gate_temperature=0.0` skip the POD
view gate, while legacy v106 fields without `pod_view_gate_mode` keep the old
implicit unit-temperature behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from render import _apply_surface_residual_field


@dataclass
class _Triangles:
    get_triangle_indices: torch.Tensor
    get_vertices: torch.Tensor


@dataclass
class _View:
    camera_center: torch.Tensor


def _scene_inputs():
    rendering = torch.zeros((3, 2, 2), dtype=torch.float32)
    rendering[0, :, 1] = 1.0
    pkg = {
        "rend_ids": torch.zeros((2, 2), dtype=torch.long),
        "image_2D": torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=torch.float32,
        ),
    }
    triangles = _Triangles(
        get_triangle_indices=torch.tensor([[0, 1, 2]], dtype=torch.long),
        get_vertices=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    view = _View(camera_center=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
    return rendering, pkg, triangles, view


def _field(mode: str | None, temperature: float):
    base = torch.zeros((1, 6, 3), dtype=torch.float32)
    experts = torch.zeros((1, 2, 6, 3), dtype=torch.float32)
    experts[0, 0, 0, 0] = 0.2
    field = {
        "basis_type": "affine_barycentric_viewdir_pod_mixture",
        "triangle_base_coefficients": base,
        "triangle_expert_delta_coefficients": experts,
        "triangle_expert_reliability": torch.ones((1, 2), dtype=torch.float32),
        "triangle_expert_mse_scale": torch.ones((1, 2), dtype=torch.float32),
        "triangle_occlusion_base_keep": torch.ones((1,), dtype=torch.float32),
        "triangle_view_means": torch.full((1, 3), 10.0, dtype=torch.float32),
        "triangle_view_scales": torch.full((1, 3), 0.01, dtype=torch.float32),
        "valid_triangles": 1,
        "triangle_count": 1,
        "min_count": 1,
        "view_gate_temperature": temperature,
        "pod_base_keep_mode": "base_preserving_boundary",
    }
    if mode is not None:
        field["pod_view_gate_mode"] = mode
    return field


def _red_gain(field) -> float:
    rendering, pkg, triangles, view = _scene_inputs()
    adapted, _info = _apply_surface_residual_field(
        rendering,
        pkg,
        field,
        device=torch.device("cpu"),
        triangles=triangles,
        view=view,
    )
    return float((adapted[0] - rendering[0]).mean().item())


def main() -> None:
    tc_zero_gain = _red_gain(_field("temperature_controlled", 0.0))
    implicit_gain = _red_gain(_field("implicit_unit_temperature", 0.0))
    legacy_missing_gain = _red_gain(_field(None, 0.0))

    if tc_zero_gain <= 0.02:
        raise AssertionError(f"temperature_controlled vgt=0 should keep POD delta; got {tc_zero_gain:.8f}")
    if implicit_gain >= tc_zero_gain * 0.01:
        raise AssertionError(
            "implicit_unit_temperature should suppress the far-view synthetic POD delta; "
            f"implicit={implicit_gain:.8f}, tc_zero={tc_zero_gain:.8f}"
        )
    if abs(legacy_missing_gain - implicit_gain) > 1.0e-8:
        raise AssertionError(
            "missing pod_view_gate_mode should preserve v106 implicit behavior; "
            f"missing={legacy_missing_gain:.8f}, implicit={implicit_gain:.8f}"
        )

    print(
        "POD view-gate smoke passed: "
        f"temperature_controlled_vgt0_gain={tc_zero_gain:.8f}, "
        f"implicit_gain={implicit_gain:.8f}, "
        f"legacy_missing_gain={legacy_missing_gain:.8f}"
    )


if __name__ == "__main__":
    main()
