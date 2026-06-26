#!/usr/bin/env python3
"""Smoke tests for v109 calibration-view subset filtering."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().with_name("meshsplatopt_v109_render_realized_parent_gate.py")
SPEC = importlib.util.spec_from_file_location("meshsplatopt_v109_render_realized_parent_gate", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_filter_names_by_view_subset = MODULE._filter_names_by_view_subset


def _assert_equal(actual: list[str], expected: list[str], label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def main() -> int:
    names = [
        "00000.png",
        "00001.png",
        "00002.png",
        "00009.png",
        "frame.png",
        "00010.png",
        "00011.extra.png",
    ]

    _assert_equal(_filter_names_by_view_subset(names, "all"), names, "all keeps compatibility")
    _assert_equal(
        _filter_names_by_view_subset(names, "even"),
        ["00000.png", "00002.png", "00010.png"],
        "even numeric stems",
    )
    _assert_equal(
        _filter_names_by_view_subset(names, "odd"),
        ["00001.png", "00009.png"],
        "odd numeric stems",
    )

    try:
        _filter_names_by_view_subset(names, "bad")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid subset should raise ValueError")

    print("v109 gate subset smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
