#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


DEFAULT_ALLOWED_KEYS = {
    "alpha",
    "barycentric",
    "barycentric_valid",
    "camera_center",
    "depth",
    "face_id",
    "image_name",
    "normal",
    "rgb_render",
    "texture",
}
ALWAYS_FORBIDDEN_KEYS = {
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


def parse_allowed_keys(text: str) -> set[str]:
    if not str(text).strip():
        return set(DEFAULT_ALLOWED_KEYS)
    return {item.strip() for item in str(text).split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a target evidence view cache that hides target GT/residual keys from vNext apply."
    )
    parser.add_argument("--target_evidence_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--allowed_keys", default=",".join(sorted(DEFAULT_ALLOWED_KEYS)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.target_evidence_dir)
    out_dir = Path(args.out_dir)
    source_views = evidence_views(source_dir)
    if not source_views:
        raise FileNotFoundError(f"no target evidence npz views found in {source_dir}")
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"{out_dir} exists; pass --force to overwrite")
        shutil.rmtree(out_dir)
    output_views_dir = out_dir / "views"
    output_views_dir.mkdir(parents=True, exist_ok=True)

    allowed_keys = parse_allowed_keys(args.allowed_keys)
    removed_counts: dict[str, int] = {}
    kept_counts: dict[str, int] = {}
    view_rows: list[dict[str, Any]] = []
    for source_path in tqdm(source_views, desc="strip target evidence"):
        with np.load(source_path, allow_pickle=False) as z:
            present = set(str(key) for key in z.files)
            forbidden_present = sorted(present & ALWAYS_FORBIDDEN_KEYS)
            kept = sorted((present & allowed_keys) - ALWAYS_FORBIDDEN_KEYS)
            if "rgb_render" not in kept:
                raise KeyError(f"{source_path} missing required rgb_render key after stripping")
            payload = {key: np.asarray(z[key]) for key in kept}
            for key in kept:
                kept_counts[key] = kept_counts.get(key, 0) + 1
            for key in forbidden_present:
                removed_counts[key] = removed_counts.get(key, 0) + 1
        out_path = output_views_dir / source_path.name
        np.savez_compressed(out_path, **payload)
        view_rows.append(
            {
                "source": str(source_path),
                "output": str(out_path),
                "kept_keys": kept,
                "removed_forbidden_keys": forbidden_present,
            }
        )

    audit = {
        "schema_version": 1,
        "source_target_evidence_dir": str(source_dir),
        "stripped_target_evidence_dir": str(out_dir),
        "view_count": int(len(view_rows)),
        "allowed_keys": sorted(allowed_keys),
        "always_forbidden_keys": sorted(ALWAYS_FORBIDDEN_KEYS),
        "removed_key_counts": removed_counts,
        "kept_key_counts": kept_counts,
        "target_gt_visible_to_apply": False,
        "target_residual_visible_to_apply": False,
        "views": view_rows,
    }
    audit_path = out_dir / "target_evidence_no_gt_audit.json"
    audit_path.write_text(json.dumps(json_safe(audit), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(audit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
