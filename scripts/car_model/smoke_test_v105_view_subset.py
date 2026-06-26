#!/usr/bin/env python3
"""Smoke-test strict view-subset indexing for v105+ field builders."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.build_v105_evidence_gated_mixture_field import _select_indexed_views


def _assert_equal(actual, expected) -> None:
    if actual != expected:
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


def main() -> int:
    views = ["v0", "v1", "v2", "v3", "v4"]
    _assert_equal(_select_indexed_views(views, "all"), [(0, "v0"), (1, "v1"), (2, "v2"), (3, "v3"), (4, "v4")])
    _assert_equal(_select_indexed_views(views, "even"), [(0, "v0"), (2, "v2"), (4, "v4")])
    _assert_equal(_select_indexed_views(views, "odd"), [(1, "v1"), (3, "v3")])
    try:
        _select_indexed_views(views, "bad")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid subset should raise ValueError")
    print("v105 view_subset smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
