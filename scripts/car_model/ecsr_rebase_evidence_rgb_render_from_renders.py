#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


FORBIDDEN_TARGET_KEYS = {
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

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

MINIMAL_FIELDS = {
    "alpha",
    "barycentric",
    "barycentric_valid",
    "camera_center",
    "depth",
    "face_id",
    "normal",
    "texture",
}


def evidence_views(evidence_dir: Path) -> list[Path]:
    views_dir = evidence_dir / "views"
    if views_dir.is_dir():
        return sorted(path for path in views_dir.glob("*.npz") if path.is_file())
    return sorted(path for path in evidence_dir.glob("*.npz") if path.is_file())


def image_index(render_dir: Path) -> dict[str, Path]:
    if not render_dir.is_dir():
        raise FileNotFoundError(render_dir)
    out: dict[str, Path] = {}
    for ext in IMAGE_EXTS:
        for path in render_dir.glob(f"*{ext}"):
            out[path.stem] = path
    return out


def as_float_chw(arr: np.ndarray, *, key: str, path: Path) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 3 or out.shape[0] != 3:
        raise ValueError(f"{path} field {key!r} must be CHW RGB, got {out.shape}")
    return np.clip(out, 0.0, 1.0)


def infer_rgb_shape(payload: dict[str, np.ndarray], *, path: Path) -> tuple[int, int, int]:
    for key in ("rgb_render", "rgb_gt"):
        if key in payload:
            return as_float_chw(payload[key], key=key, path=path).shape
    if "face_id" in payload:
        h, w = np.asarray(payload["face_id"]).shape[-2:]
        return (3, int(h), int(w))
    raise KeyError(f"{path} needs rgb_render/rgb_gt or face_id to infer RGB shape")


def load_rgb(path: Path, *, shape_chw: tuple[int, int, int], allow_resize: bool) -> np.ndarray:
    c, h, w = shape_chw
    if c != 3:
        raise ValueError(f"expected CHW RGB shape, got {shape_chw}")
    image = Image.open(path).convert("RGB")
    if image.size != (w, h):
        if not allow_resize:
            raise ValueError(f"{path} has size {image.size}, expected {(w, h)}")
        image = image.resize((w, h), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1).astype(np.float32)


def load_rgb_native(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1).astype(np.float32)


def resize_rgb_chw(image: np.ndarray, *, shape_chw: tuple[int, int, int]) -> np.ndarray:
    c, h, w = shape_chw
    arr = as_float_chw(image, key="resize_rgb_chw", path=Path("<array>"))
    if arr.shape == shape_chw:
        return arr
    if c != 3:
        raise ValueError(f"expected CHW RGB shape, got {shape_chw}")
    pil = Image.fromarray(np.clip(np.moveaxis(arr, 0, -1) * 255.0, 0, 255).astype(np.uint8))
    pil = pil.resize((w, h), Image.Resampling.BILINEAR)
    return (np.asarray(pil, dtype=np.float32) / 255.0).transpose(2, 0, 1).astype(np.float32)


def _resize_hw_field(arr: np.ndarray, *, h: int, w: int, nearest: bool) -> np.ndarray:
    src = np.asarray(arr)
    if src.shape == (h, w):
        return src
    dtype = src.dtype
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if nearest:
        image = Image.fromarray(src.astype(np.int32, copy=False))
        out = np.asarray(image.resize((w, h), resampling))
        if np.issubdtype(dtype, np.bool_):
            return (out > 0).astype(dtype)
        return out.astype(dtype, copy=False)
    image = Image.fromarray(src.astype(np.float32, copy=False), mode="F")
    out = np.asarray(image.resize((w, h), resampling), dtype=np.float32)
    return out.astype(dtype, copy=False)


def _resize_chw_field(arr: np.ndarray, *, h: int, w: int, nearest: bool) -> np.ndarray:
    src = np.asarray(arr)
    if src.ndim != 3:
        raise ValueError(f"expected CHW field, got shape {src.shape}")
    if src.shape[-2:] == (h, w):
        return src
    channels = [_resize_hw_field(src[idx], h=h, w=w, nearest=nearest) for idx in range(src.shape[0])]
    return np.stack(channels, axis=0).astype(src.dtype, copy=False)


def resize_minimal_geometry(payload: dict[str, np.ndarray], *, shape_chw: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """Resize real geometry buffers to match a rebased native-resolution parent.

    RGB parents can be rebased from a different render resolution than the
    source evidence cache.  Surface-texture models need the dense face/UV
    buffers at the same resolution as `rgb_render`; otherwise they silently
    collapse back to image-only behavior or fail on shape mismatches.
    """

    _, h, w = shape_chw
    out = dict(payload)
    nearest_keys = {"face_id", "barycentric_valid"}
    chw_keys = {"barycentric", "normal"}
    for key in sorted(MINIMAL_FIELDS - {"camera_center"}):
        arr = np.asarray(out[key])
        nearest = key in nearest_keys
        if key in chw_keys:
            out[key] = _resize_chw_field(arr, h=h, w=w, nearest=nearest)
        else:
            out[key] = _resize_hw_field(arr, h=h, w=w, nearest=nearest)
    return out


def safe_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {str(key): z[key] for key in z.files}


def write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    tmp.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite surface evidence rgb_render from an external render directory. "
            "For fit evidence this can recompute residual supervision against rgb_gt; "
            "for target evidence it strips GT/residual keys before apply."
        )
    )
    parser.add_argument("--input_evidence_dir", type=Path, required=True)
    parser.add_argument("--render_dir", type=Path, required=True)
    parser.add_argument(
        "--gt_render_dir",
        type=Path,
        default=None,
        help="Optional RGB GT image directory keyed by evidence stem, used only with --recompute_residual_from_gt.",
    )
    parser.add_argument("--output_evidence_dir", type=Path, required=True)
    parser.add_argument("--audit_path", type=Path, default=None)
    parser.add_argument("--allow_resize", action="store_true")
    parser.add_argument(
        "--match_render_resolution",
        action="store_true",
        help="Use the external render image's native resolution for output evidence.",
    )
    parser.add_argument("--recompute_residual_from_gt", action="store_true")
    parser.add_argument("--strip_target_gt_and_residuals", action="store_true")
    parser.add_argument(
        "--minimal_fields",
        action="store_true",
        help=(
            "Write only fields needed by the image-space U-Net trainer/apply path. "
            "This intentionally omits old residual diagnostics and is not sufficient "
            "for surface-texture face selection."
        ),
    )
    parser.add_argument(
        "--constant_geometry_fields",
        action="store_true",
        help=(
            "With --minimal_fields, replace dense geometry buffers with deterministic constants. "
            "This creates an image-space anchored residual dataset for storage-limited diagnostics."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy_sidecars", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_evidence_dir)
    output_dir = Path(args.output_evidence_dir)
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"output exists; pass --force to replace files: {output_dir}")
    output_views_dir = output_dir / "views" if (input_dir / "views").is_dir() else output_dir
    output_views_dir.mkdir(parents=True, exist_ok=True)

    renders = image_index(Path(args.render_dir))
    gt_renders = image_index(Path(args.gt_render_dir)) if args.gt_render_dir is not None else {}
    paths = evidence_views(input_dir)
    if not paths:
        raise FileNotFoundError(f"no evidence views in {input_dir}")

    audit: dict[str, Any] = {
        "schema": "spcarnet_rebase_evidence_rgb_render_v1",
        "input_evidence_dir": str(input_dir),
        "render_dir": str(args.render_dir),
        "gt_render_dir": str(args.gt_render_dir) if args.gt_render_dir is not None else "",
        "output_evidence_dir": str(output_dir),
        "view_count": int(len(paths)),
        "recompute_residual_from_gt": bool(args.recompute_residual_from_gt),
        "strip_target_gt_and_residuals": bool(args.strip_target_gt_and_residuals),
        "minimal_fields": bool(args.minimal_fields),
        "constant_geometry_fields": bool(args.constant_geometry_fields),
        "match_render_resolution": bool(args.match_render_resolution),
        "target_gt_visible_to_apply": False,
        "target_residual_visible_to_apply": False,
        "missing_render_count": 0,
        "rewritten_view_count": 0,
        "geometry_resized_view_count": 0,
        "sample_views": [],
    }

    forbidden_seen: set[str] = set()
    missing: list[str] = []
    for idx, src in enumerate(paths):
        render_path = renders.get(src.stem)
        if render_path is None:
            missing.append(src.stem)
            continue
        payload = safe_npz_payload(src)
        if bool(args.match_render_resolution):
            new_parent = load_rgb_native(render_path)
            shape = tuple(int(x) for x in new_parent.shape)
        else:
            shape = infer_rgb_shape(payload, path=src)
            new_parent = load_rgb(render_path, shape_chw=shape, allow_resize=bool(args.allow_resize))
        old_parent = (
            as_float_chw(payload["rgb_render"], key="rgb_render", path=src)
            if "rgb_render" in payload
            else np.zeros_like(new_parent)
        )
        if old_parent.shape != new_parent.shape:
            old_parent = resize_rgb_chw(old_parent, shape_chw=shape)
        if bool(args.minimal_fields) and bool(args.constant_geometry_fields):
            _, h, w = new_parent.shape
            if "camera_center" not in payload:
                raise KeyError(f"{src} missing required minimal field: camera_center")
            payload = {
                "alpha": np.ones((h, w), dtype=np.float16),
                "barycentric": np.zeros((3, h, w), dtype=np.float16),
                "barycentric_valid": np.ones((h, w), dtype=np.uint8),
                "camera_center": np.asarray(payload["camera_center"], dtype=np.float32),
                "depth": np.zeros((h, w), dtype=np.float16),
                "face_id": np.zeros((h, w), dtype=np.int32),
                "normal": np.zeros((3, h, w), dtype=np.float16),
                "texture": np.zeros((h, w), dtype=np.float16),
            }
        elif bool(args.minimal_fields):
            missing_required = sorted(key for key in MINIMAL_FIELDS if key not in payload)
            if missing_required:
                raise KeyError(f"{src} missing required minimal fields: {missing_required}")
            payload = {key: payload[key] for key in sorted(MINIMAL_FIELDS)}
            before_shape = tuple(int(x) for x in np.asarray(payload["face_id"]).shape[-2:])
            if bool(args.match_render_resolution) and before_shape != tuple(shape[1:]):
                payload = resize_minimal_geometry(payload, shape_chw=shape)
                audit["geometry_resized_view_count"] = int(audit["geometry_resized_view_count"]) + 1
        payload["rgb_render"] = new_parent.astype(np.float16)

        residual_mean_l1 = None
        if bool(args.recompute_residual_from_gt):
            if args.gt_render_dir is not None:
                gt_path = gt_renders.get(src.stem)
                if gt_path is None:
                    raise KeyError(f"{src} missing GT image in {args.gt_render_dir}")
                gt = load_rgb(gt_path, shape_chw=shape, allow_resize=bool(args.allow_resize) or bool(args.match_render_resolution))
            else:
                source_payload = safe_npz_payload(src) if bool(args.minimal_fields) else payload
                if "rgb_gt" not in source_payload:
                    raise KeyError(f"{src} cannot recompute residuals without rgb_gt")
                gt = as_float_chw(source_payload["rgb_gt"], key="rgb_gt", path=src)
                if gt.shape != new_parent.shape:
                    if not (bool(args.allow_resize) or bool(args.match_render_resolution)):
                        raise ValueError(f"{src} rgb_gt shape {gt.shape} differs from parent {new_parent.shape}")
                    gt = resize_rgb_chw(gt, shape_chw=shape)
            residual = (gt - new_parent).astype(np.float32)
            residual_l1 = np.mean(np.abs(residual), axis=0).astype(np.float32)
            payload["rgb_gt"] = gt.astype(np.float16)
            payload["teacher_residual_rgb"] = residual.astype(np.float16)
            if not bool(args.minimal_fields):
                payload["residual_rgb"] = residual.astype(np.float16)
                payload["residual_l1"] = residual_l1.astype(np.float16)
                payload["teacher_residual_rgb_raw"] = residual.astype(np.float16)
                payload["teacher_residual_l1"] = residual_l1.astype(np.float16)
                payload["teacher_better_mask"] = (residual_l1 > 0.0).astype(np.uint8)
                payload["teacher_gain_l1"] = np.maximum(
                    np.mean(np.abs(old_parent - gt), axis=0) - residual_l1,
                    0.0,
                ).astype(np.float16)
                payload["teacher_parent_delta_l1"] = np.mean(np.abs(new_parent - old_parent), axis=0).astype(np.float16)
            residual_mean_l1 = float(np.mean(residual_l1))

        if bool(args.strip_target_gt_and_residuals):
            for key in sorted(FORBIDDEN_TARGET_KEYS):
                payload.pop(key, None)
            forbidden_seen.update(set(payload) & FORBIDDEN_TARGET_KEYS)

        dst = output_views_dir / src.name
        write_npz(dst, payload)
        audit["rewritten_view_count"] = int(audit["rewritten_view_count"]) + 1
        if idx < 8:
            audit["sample_views"].append(
                {
                    "view": src.stem,
                    "input": str(src),
                    "render": str(render_path),
                    "output": str(dst),
                    "mean_abs_parent_change": float(np.mean(np.abs(new_parent - old_parent))),
                    "mean_residual_l1": residual_mean_l1,
                    "keys": sorted(payload.keys()),
                }
            )

    if missing:
        audit["missing_render_count"] = int(len(missing))
        audit["missing_renders"] = missing[:32]
        audit["passed"] = False
        audit["reason"] = "missing_render_for_evidence_view"
    else:
        audit["missing_render_count"] = 0
        audit["target_gt_visible_to_apply"] = bool("rgb_gt" in forbidden_seen)
        audit["target_residual_visible_to_apply"] = bool(
            forbidden_seen
            & {
                "residual_rgb",
                "residual_l1",
                "teacher_residual_rgb",
                "teacher_residual_l1",
                "teacher_residual_rgb_raw",
            }
        )
        audit["passed"] = not forbidden_seen
        audit["reason"] = "ok" if not forbidden_seen else "forbidden_target_keys_after_strip"

    if bool(args.copy_sidecars):
        copied: list[str] = []
        for src in sorted(input_dir.iterdir()):
            if src.name == "views":
                continue
            if src.is_file():
                dst = output_dir / src.name
                if not dst.exists() or bool(args.force):
                    dst.write_bytes(src.read_bytes())
                copied.append(src.name)
        audit["copied_sidecars"] = copied

    text = json.dumps(json_safe(audit), indent=2, sort_keys=True) + "\n"
    if args.audit_path is not None:
        Path(args.audit_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_path).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if bool(audit.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
