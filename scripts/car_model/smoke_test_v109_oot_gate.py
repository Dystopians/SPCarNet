#!/usr/bin/env python3
"""Smoke tests for v109 out-of-trajectory camera-support gating."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().with_name("meshsplatopt_v109_render_realized_parent_gate.py")
SPEC = importlib.util.spec_from_file_location("meshsplatopt_v109_render_realized_parent_gate", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_camera_index(path: Path, centers: list[tuple[float, float, float]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "idx": idx,
            "image_name": f"frame_{idx:05d}",
            "camera_center": [float(x), float(y), float(z)],
        }
        for idx, (x, y, z) in enumerate(centers)
    ]
    (path / "camera_index.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _args(manifest: Path) -> argparse.Namespace:
    return argparse.Namespace(
        oot_gate_mode="scene_fallback",
        oot_source_manifest=str(manifest),
        oot_source_view_subset="even",
        oot_center_quantile=0.95,
        oot_center_rel_margin=0.0,
        oot_center_abs_margin=0.0,
        oot_max_frame_fraction=0.10,
        oot_max_mask_weighted_fraction=0.05,
        oot_min_mask_mean_for_scene_check=0.05,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        calib = root / "calib"
        target = root / "target"
        manifest = root / "field.manifest.json"
        manifest.write_text(json.dumps({"selected_frame_keys": ["00000", "00002"]}) + "\n", encoding="utf-8")

        _write_camera_index(
            calib,
            [
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.5, 0.0, 0.0),
            ],
        )
        _write_camera_index(target, [(0.4, 0.0, 0.0), (2.4, 0.0, 0.0)])
        close = MODULE._evaluate_out_of_trajectory_gate(
            _args(manifest),
            calib,
            target,
            ["00001.png", "00003.png"],
            ["00000.png", "00001.png"],
            {"mean_mask": 0.5, "frames": [{"image": "00000.png", "mask_mean": 0.5}, {"image": "00001.png", "mask_mean": 0.5}]},
        )
        if not close["pass"]:
            raise AssertionError(f"in-support target should pass: {close}")

        _write_camera_index(target, [(10.0, 0.0, 0.0), (11.0, 0.0, 0.0)])
        far = MODULE._evaluate_out_of_trajectory_gate(
            _args(manifest),
            calib,
            target,
            ["00001.png", "00003.png"],
            ["00000.png", "00001.png"],
            {"mean_mask": 0.5, "frames": [{"image": "00000.png", "mask_mean": 0.5}, {"image": "00001.png", "mask_mean": 0.5}]},
        )
        if far["pass"] or far.get("fallback_reason") != "target_frame_fraction_exceeds_support,mask_weighted_fraction_exceeds_support":
            raise AssertionError(f"out-of-support target should fail both OOT checks: {far}")

        zero_mask = MODULE._evaluate_out_of_trajectory_gate(
            _args(manifest),
            calib,
            target,
            ["00001.png", "00003.png"],
            ["00000.png", "00001.png"],
            {"mean_mask": 0.0, "frames": [{"image": "00000.png", "mask_mean": 0.0}, {"image": "00001.png", "mask_mean": 0.0}]},
        )
        if not zero_mask["pass"]:
            raise AssertionError(f"zero-mask parent fallback should not fail OOT: {zero_mask}")

        frame_disabled = {str(row.get("image")) for row in far.get("frames", []) if row.get("oot_center_ood")}
        if frame_disabled != {"00000.png", "00001.png"}:
            raise AssertionError(f"frame-fallback disabled set mismatch: {frame_disabled}")

    print("v109 OOT gate smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
