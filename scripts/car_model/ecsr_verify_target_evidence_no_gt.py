#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


FORBIDDEN_KEYS = {
    "rgb_gt",
    "residual_rgb",
    "residual_l1",
    "teacher_residual_rgb",
    "teacher_residual_l1",
    "teacher_residual_rgb_raw",
    "teacher_better_mask",
    "teacher_gain_l1",
    "teacher_parent_delta_l1",
}


def evidence_views(evidence_dir: Path) -> list[Path]:
    views_dir = evidence_dir / "views"
    if views_dir.is_dir():
        return sorted(path for path in views_dir.glob("*.npz") if path.is_file())
    return sorted(path for path in evidence_dir.glob("*.npz") if path.is_file())


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that target evidence passed to vNext apply contains no GT/residual keys."
    )
    parser.add_argument("--target_evidence_dir", type=Path, required=True)
    parser.add_argument("--audit_path", type=Path, default=None)
    args = parser.parse_args()

    evidence_dir = Path(args.target_evidence_dir)
    paths = evidence_views(evidence_dir)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "evidence_dir": str(evidence_dir),
        "view_count": int(len(paths)),
        "forbidden_keys": sorted(FORBIDDEN_KEYS),
        "bad_view_count": 0,
        "bad_views": [],
        "sample_keys": [],
        "target_gt_visible_to_apply": False,
        "target_residual_visible_to_apply": False,
        "passed": False,
    }
    if not paths:
        audit["reason"] = "no_target_evidence_views"
    else:
        bad_views: list[dict[str, Any]] = []
        sample_keys: list[dict[str, Any]] = []
        for idx, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as z:
                keys = sorted(str(key) for key in z.files)
            forbidden_present = sorted(set(keys) & FORBIDDEN_KEYS)
            if idx < 4:
                sample_keys.append({"path": str(path), "keys": keys})
            if forbidden_present:
                bad_views.append({"path": str(path), "forbidden_keys": forbidden_present})
        audit["bad_view_count"] = int(len(bad_views))
        audit["bad_views"] = bad_views[:32]
        audit["sample_keys"] = sample_keys
        audit["passed"] = not bad_views
        if bad_views:
            audit["target_gt_visible_to_apply"] = any(
                "rgb_gt" in set(row.get("forbidden_keys", [])) for row in bad_views
            )
            audit["target_residual_visible_to_apply"] = any(
                set(row.get("forbidden_keys", []))
                & {
                    "residual_rgb",
                    "residual_l1",
                    "teacher_residual_rgb",
                    "teacher_residual_l1",
                    "teacher_residual_rgb_raw",
                }
                for row in bad_views
            )
            audit["reason"] = "forbidden_target_apply_keys_present"
        else:
            audit["reason"] = "ok"

    text = json.dumps(json_safe(audit), indent=2, sort_keys=True) + "\n"
    if args.audit_path is not None:
        Path(args.audit_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_path).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if bool(audit.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
