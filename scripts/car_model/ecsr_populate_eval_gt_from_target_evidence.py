#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


def evidence_views(evidence_dir: Path) -> list[Path]:
    views_dir = evidence_dir / "views"
    if views_dir.is_dir():
        return sorted(path for path in views_dir.glob("*.npz") if path.is_file())
    return sorted(path for path in evidence_dir.glob("*.npz") if path.is_file())


def save_image_chw(path: Path, image: np.ndarray) -> None:
    arr = np.clip(np.moveaxis(np.asarray(image, dtype=np.float32), 0, -1), 0.0, 1.0)
    arr_u8 = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8).save(path)


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
        description="Populate evaluation GT images after vNext target apply, keeping target GT out of apply/selection."
    )
    parser.add_argument("--target_evidence_dir", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--audit_path", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    target_views = evidence_views(Path(args.target_evidence_dir))
    if not target_views:
        raise FileNotFoundError(f"no target evidence npz views found in {args.target_evidence_dir}")

    method_dir = Path(args.output_model) / str(args.split) / str(args.method_name)
    render_dir = method_dir / "renders"
    if not render_dir.is_dir():
        raise FileNotFoundError(f"render dir does not exist: {render_dir}")
    gt_dir = method_dir / "gt"
    if gt_dir.exists():
        if not args.force:
            raise FileExistsError(f"{gt_dir} exists; pass --force to overwrite")
        shutil.rmtree(gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_render = 0
    rows: list[dict[str, Any]] = []
    for path in tqdm(target_views, desc="populate eval gt"):
        with np.load(path, allow_pickle=False) as z:
            if "rgb_gt" not in z:
                raise KeyError(f"{path} missing rgb_gt for final evaluation")
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)
        name = f"{path.stem}.png"
        render_path = render_dir / name
        if not render_path.is_file():
            missing_render += 1
            rows.append({"view": str(path.name), "gt_written": False, "reason": "missing_render"})
            continue
        save_image_chw(gt_dir / name, gt)
        written += 1
        rows.append({"view": str(path.name), "gt_written": True, "render": str(render_path)})

    audit = {
        "schema_version": 1,
        "target_evidence_dir": str(args.target_evidence_dir),
        "output_model": str(args.output_model),
        "split": str(args.split),
        "method_name": str(args.method_name),
        "render_dir": str(render_dir),
        "gt_dir": str(gt_dir),
        "target_gt_visible_to_eval": True,
        "target_gt_visible_to_apply": False,
        "view_count": int(len(target_views)),
        "written_gt_images": int(written),
        "missing_render_images": int(missing_render),
        "views": rows,
    }
    audit_path = args.audit_path or (method_dir / "eval_gt_population_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(json_safe(audit), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(audit), indent=2, sort_keys=True))
    return 1 if missing_render else 0


if __name__ == "__main__":
    raise SystemExit(main())
