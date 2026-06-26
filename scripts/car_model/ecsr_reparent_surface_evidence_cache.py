#!/usr/bin/env python3
"""Re-anchor an ECSR surface evidence cache to a new parent render.

This utility is the missing bridge for v115-style experiments: existing surface
evidence can keep its geometry, face ids, barycentric coordinates, and GT fields,
while `rgb_render` and parent residual fields are recomputed against a stronger
parent such as v106. It is intentionally strict by default: frame names and
native sizes must match, and no held-out target GT is required unless residual
fields are explicitly present in the input cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_evidence_dir", type=Path, required=True)
    parser.add_argument("--parent_render_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow_resize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--parent_label", default="reparented_parent")
    parser.add_argument("--rgb_render_key", default="rgb_render")
    parser.add_argument("--rgb_gt_key", default="rgb_gt")
    parser.add_argument("--residual_rgb_key", default="residual_rgb")
    parser.add_argument("--residual_l1_key", default="residual_l1")
    parser.add_argument("--audit_name", default="surface_evidence_reparent_audit.json")
    return parser.parse_args()


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_cache(base: Path, out: Path, *, force: bool) -> None:
    if not base.is_dir():
        raise FileNotFoundError(f"missing base evidence dir: {base}")
    if out.resolve() == base.resolve():
        raise ValueError("--out_dir must differ from --base_evidence_dir")
    if out.exists():
        if not force:
            raise FileExistsError(f"output exists; pass --force to replace: {out}")
        shutil.rmtree(out)
    shutil.copytree(base, out)


def _per_view_dir(cache_dir: Path) -> Path:
    for name in ("views", "per_view_npz"):
        path = cache_dir / name
        if path.is_dir():
            return path
    npz_files = sorted(cache_dir.glob("*.npz"))
    if npz_files:
        return cache_dir
    raise FileNotFoundError(f"{cache_dir} has no views/, per_view_npz/, or root-level .npz files")


def _image_index(root: Path) -> dict[str, Path]:
    candidates = [root]
    if (root / "renders").is_dir():
        candidates.insert(0, root / "renders")
    mapping: dict[str, Path] = {}
    for directory in candidates:
        if not directory.is_dir():
            continue
        for ext in IMAGE_EXTS:
            for path in directory.glob(f"*{ext}"):
                mapping.setdefault(path.stem, path)
    if not mapping:
        raise FileNotFoundError(f"no render images found under {root}")
    return mapping


def _load_rgb(path: Path, *, shape_chw: tuple[int, int, int], allow_resize: bool) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    c, h, w = shape_chw
    if c != 3:
        raise ValueError(f"expected CHW RGB shape, got {shape_chw}")
    if image.size != (w, h):
        if not allow_resize:
            raise ValueError(f"{path} has size {image.size}, expected {(w, h)}; pass --allow_resize to resize")
        image = image.resize((w, h), Image.Resampling.BILINEAR)
    return (np.asarray(image, dtype=np.float32) / 255.0).transpose(2, 0, 1).astype(np.float32)


def _safe_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp.replace(path)


def _as_rgb_chw(value: np.ndarray, *, key: str, path: Path) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"{path} field {key!r} must be CHW RGB, got {arr.shape}")
    return np.clip(arr, 0.0, 1.0)


def _infer_rgb_shape(payload: dict[str, np.ndarray], *, path: Path, rgb_render_key: str, rgb_gt_key: str) -> tuple[int, int, int]:
    for key in (rgb_render_key, rgb_gt_key):
        if key in payload:
            return _as_rgb_chw(payload[key], key=key, path=path).shape
    for key in ("face_id", "face_ids"):
        if key in payload:
            face = np.asarray(payload[key])
            if face.ndim != 2:
                raise ValueError(f"{path} field {key!r} must be HxW when inferring RGB shape, got {face.shape}")
            return (3, int(face.shape[0]), int(face.shape[1]))
    raise KeyError(f"{path} needs {rgb_render_key}/{rgb_gt_key} or face_id to infer RGB shape")


def _update_summary(summary_path: Path, *, audit: dict[str, Any], args: argparse.Namespace) -> None:
    if not summary_path.is_file():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return
    fields = list(summary.get("per_view_npz_fields", []))
    for field in (args.rgb_render_key, args.residual_rgb_key, args.residual_l1_key):
        if field not in fields:
            fields.append(field)
    summary["per_view_npz_fields"] = fields
    summary["reparented_parent"] = {
        "enabled": True,
        "parent_label": str(args.parent_label),
        "parent_render_dir": str(args.parent_render_dir),
        "audit_path": str(audit.get("audit_path", "")),
        "processed_views": int(audit.get("processed_views", 0)),
        "residuals_recomputed": int(audit.get("residual_recomputed_views", 0)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if int(args.max_views) < 0:
        raise SystemExit("--max_views must be >= 0")
    _copy_cache(args.base_evidence_dir, args.out_dir, force=bool(args.force))
    view_dir = _per_view_dir(args.out_dir)
    view_paths = sorted(view_dir.glob("*.npz"))
    if int(args.max_views) > 0:
        view_paths = view_paths[: int(args.max_views)]
    if not view_paths:
        raise FileNotFoundError(f"no .npz views found under {view_dir}")

    parent_images = _image_index(args.parent_render_dir)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    residual_recomputed = 0
    for path in view_paths:
        parent = parent_images.get(path.stem)
        if parent is None:
            skipped.append({"view": path.stem, "reason": "missing_parent_render"})
            continue
        payload = _safe_npz_payload(path)
        shape = _infer_rgb_shape(
            payload,
            path=path,
            rgb_render_key=str(args.rgb_render_key),
            rgb_gt_key=str(args.rgb_gt_key),
        )
        parent_rgb = _load_rgb(parent, shape_chw=shape, allow_resize=bool(args.allow_resize))
        old_rgb = (
            _as_rgb_chw(payload[str(args.rgb_render_key)], key=str(args.rgb_render_key), path=path)
            if str(args.rgb_render_key) in payload
            else None
        )
        payload[str(args.rgb_render_key)] = parent_rgb.astype(np.float16)
        row: dict[str, Any] = {
            "view": path.stem,
            "parent_image": str(parent),
            "parent_sha1": _sha1_file(parent),
            "shape_chw": list(parent_rgb.shape),
            "old_parent_mean_abs_delta": None,
            "residual_recomputed": False,
        }
        if old_rgb is not None:
            row["old_parent_mean_abs_delta"] = float(np.mean(np.abs(parent_rgb - old_rgb)))
        if str(args.rgb_gt_key) in payload:
            gt = _as_rgb_chw(payload[str(args.rgb_gt_key)], key=str(args.rgb_gt_key), path=path)
            residual = (gt - parent_rgb).astype(np.float32)
            payload[str(args.residual_rgb_key)] = residual.astype(np.float16)
            payload[str(args.residual_l1_key)] = np.mean(np.abs(residual), axis=0).astype(np.float16)
            row["residual_recomputed"] = True
            row["mean_residual_l1"] = float(np.mean(payload[str(args.residual_l1_key)].astype(np.float32)))
            residual_recomputed += 1
        _write_npz(path, payload)
        rows.append(row)

    audit_path = args.out_dir / str(args.audit_name)
    audit = {
        "schema_version": 1,
        "base_evidence_dir": str(args.base_evidence_dir),
        "out_dir": str(args.out_dir),
        "parent_render_dir": str(args.parent_render_dir),
        "parent_label": str(args.parent_label),
        "allow_resize": bool(args.allow_resize),
        "view_dir": str(view_dir),
        "candidate_views": int(len(view_paths)),
        "processed_views": int(len(rows)),
        "skipped_views": int(len(skipped)),
        "residual_recomputed_views": int(residual_recomputed),
        "audit_path": str(audit_path),
        "rows": rows,
        "skipped": skipped,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _update_summary(args.out_dir / "surface_evidence_summary.json", audit=audit, args=args)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if not skipped else 2


if __name__ == "__main__":
    raise SystemExit(main())
