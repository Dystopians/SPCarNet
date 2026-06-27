#!/usr/bin/env python3
"""Augment an ECSR surface evidence cache with teacher residual targets.

This is the bridge from Phase-G image-level teacher evidence to Phase-K/Phase-S
surface-addressed residual fitting.  It never reads held-out test views.  It
copies an existing train surface evidence cache, adds conservative teacher
residual fields to each per-view NPZ, and rebuilds `top_residual_supports.csv`
using the teacher target rather than the original GT-render residual.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_evidence_dir", type=Path, required=True)
    parser.add_argument("--teacher_render_dir", type=Path, required=True)
    parser.add_argument(
        "--parent_render_dir",
        type=Path,
        default=None,
        help="Optional parent render directory. If omitted, per-view `rgb_render` is used.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow_resize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--teacher_render_error_margin", type=float, default=0.0)
    parser.add_argument("--teacher_parent_delta_min", type=float, default=0.0)
    parser.add_argument(
        "--mask_target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When GT is available, zero teacher residual outside pixels where the "
            "teacher beats the parent and changes the parent enough."
        ),
    )
    parser.add_argument(
        "--selection_mode",
        choices=("better_masked_residual", "teacher_gain", "residual_magnitude"),
        default="better_masked_residual",
        help="Signal written to --teacher_residual_l1_key for sample selection.",
    )
    parser.add_argument("--teacher_residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--teacher_residual_l1_key", default="teacher_residual_l1")
    parser.add_argument("--teacher_residual_rgb_raw_key", default="teacher_residual_rgb_raw")
    parser.add_argument("--teacher_better_mask_key", default="teacher_better_mask")
    parser.add_argument("--teacher_gain_l1_key", default="teacher_gain_l1")
    parser.add_argument("--teacher_parent_delta_l1_key", default="teacher_parent_delta_l1")
    parser.add_argument("--rebuild_top_supports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top_support_min_alpha", type=float, default=0.0)
    parser.add_argument(
        "--top_support_limit",
        type=int,
        default=4096,
        help="Maximum teacher-ranked face rows to write. Set <=0 to write all nonzero faces.",
    )
    return parser.parse_args()


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
    raise FileNotFoundError(f"{cache_dir} has neither views/ nor per_view_npz/")


def _image_index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"missing render dir: {root}")
    mapping: dict[str, Path] = {}
    for ext in IMAGE_EXTS:
        for path in root.glob(f"*{ext}"):
            mapping[path.stem] = path
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
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1).astype(np.float32)


def _safe_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    tmp.replace(path)


def _as_float_chw(arr: np.ndarray, *, key: str, path: Path) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 3 or out.shape[0] != 3:
        raise ValueError(f"{path} field {key!r} must be CHW RGB, got {out.shape}")
    return np.clip(out, 0.0, 1.0)


def _rgb_shape_from_payload(payload: dict[str, np.ndarray], *, path: Path) -> tuple[int, int, int]:
    for key in ("rgb_render", "rgb_gt"):
        if key in payload:
            return _as_float_chw(payload[key], key=key, path=path).shape
    if "face_id" not in payload:
        raise KeyError(f"{path} needs rgb_render/rgb_gt or face_id to infer teacher render shape")
    face_id = np.asarray(payload["face_id"])
    if face_id.ndim != 2:
        raise ValueError(f"{path} field 'face_id' must be HxW when inferring RGB shape, got {face_id.shape}")
    return (3, int(face_id.shape[0]), int(face_id.shape[1]))


def _teacher_mask(
    *,
    teacher_rgb: np.ndarray,
    parent_rgb: np.ndarray,
    gt_rgb: np.ndarray | None,
    margin: float,
    delta_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parent_delta_l1 = np.mean(np.abs(teacher_rgb - parent_rgb), axis=0).astype(np.float32)
    if gt_rgb is None:
        mask = parent_delta_l1 >= float(delta_min)
        gain = np.zeros_like(parent_delta_l1, dtype=np.float32)
        return mask, gain, parent_delta_l1
    parent_err = np.mean(np.abs(parent_rgb - gt_rgb), axis=0).astype(np.float32)
    teacher_err = np.mean(np.abs(teacher_rgb - gt_rgb), axis=0).astype(np.float32)
    gain = (parent_err - teacher_err).astype(np.float32)
    mask = (teacher_err + float(margin) < parent_err) & (parent_delta_l1 >= float(delta_min))
    return mask, gain, parent_delta_l1


def _augment_one(
    path: Path,
    *,
    teacher_image: Path,
    parent_image: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = _safe_npz_payload(path)
    rgb_shape = _rgb_shape_from_payload(payload, path=path)
    if "rgb_render" not in payload and parent_image is None:
        raise KeyError(f"{path} needs rgb_render when --parent_render_dir is not supplied")
    parent_rgb = (
        _load_rgb(parent_image, shape_chw=rgb_shape, allow_resize=bool(args.allow_resize))
        if parent_image is not None
        else _as_float_chw(payload["rgb_render"], key="rgb_render", path=path)
    )
    teacher_rgb = _load_rgb(teacher_image, shape_chw=parent_rgb.shape, allow_resize=bool(args.allow_resize))
    gt_rgb = _as_float_chw(payload["rgb_gt"], key="rgb_gt", path=path) if "rgb_gt" in payload else None

    raw_residual = (teacher_rgb - parent_rgb).astype(np.float32)
    better_mask, gain_l1, parent_delta_l1 = _teacher_mask(
        teacher_rgb=teacher_rgb,
        parent_rgb=parent_rgb,
        gt_rgb=gt_rgb,
        margin=float(args.teacher_render_error_margin),
        delta_min=float(args.teacher_parent_delta_min),
    )
    if gt_rgb is not None and bool(args.mask_target):
        target_rgb = raw_residual * better_mask[None, :, :].astype(np.float32)
    else:
        target_rgb = raw_residual

    residual_mag_l1 = np.mean(np.abs(target_rgb), axis=0).astype(np.float32)
    if args.selection_mode == "teacher_gain" and gt_rgb is not None:
        target_l1 = np.maximum(gain_l1, 0.0) * better_mask.astype(np.float32)
    elif args.selection_mode == "residual_magnitude":
        target_l1 = np.mean(np.abs(raw_residual), axis=0).astype(np.float32)
    else:
        target_l1 = residual_mag_l1

    payload[str(args.teacher_residual_rgb_key)] = target_rgb.astype(np.float16)
    payload[str(args.teacher_residual_l1_key)] = target_l1.astype(np.float16)
    payload[str(args.teacher_residual_rgb_raw_key)] = raw_residual.astype(np.float16)
    payload[str(args.teacher_better_mask_key)] = better_mask.astype(np.uint8)
    payload[str(args.teacher_gain_l1_key)] = gain_l1.astype(np.float16)
    payload[str(args.teacher_parent_delta_l1_key)] = parent_delta_l1.astype(np.float16)
    _write_npz(path, payload)

    return {
        "view": path.stem,
        "teacher_image": str(teacher_image),
        "parent_image": str(parent_image) if parent_image is not None else "npz:rgb_render",
        "has_gt": gt_rgb is not None,
        "active_fraction": float(np.mean(better_mask)),
        "mean_target_l1": float(np.mean(target_l1)),
        "mean_raw_parent_delta_l1": float(np.mean(parent_delta_l1)),
        "mean_positive_teacher_gain_l1": float(np.mean(np.maximum(gain_l1, 0.0))) if gt_rgb is not None else None,
    }


def _rebuild_top_supports(
    view_paths: list[Path],
    out_dir: Path,
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_alpha: float,
    top_support_limit: int,
) -> dict[str, Any]:
    max_face = -1
    for path in view_paths:
        with np.load(path) as z:
            required = {"face_id", "alpha", residual_l1_key, residual_rgb_key}
            missing = sorted(required - set(z.files))
            if missing:
                raise KeyError(f"{path} missing fields for teacher top-support rebuild: {missing}")
            face_id = z["face_id"].astype(np.int64)
            if face_id.size:
                local_max = int(face_id.max())
                if local_max > max_face:
                    max_face = local_max
    if max_face < 0:
        return {
            "top_residual_supports_csv": str(out_dir / "top_residual_supports.csv"),
            "parent_top_residual_supports_csv": None,
            "rows": 0,
            "nonzero_faces": 0,
            "top_preview": [],
        }

    size = int(max_face) + 1
    total_count = np.zeros((size,), dtype=np.int64)
    total_l1 = np.zeros((size,), dtype=np.float64)
    total_rgb = np.zeros((3, size), dtype=np.float64)
    view_count = np.zeros((size,), dtype=np.int32)

    for path in view_paths:
        with np.load(path) as z:
            face_id = z["face_id"].astype(np.int64)
            alpha = z["alpha"].astype(np.float32)
            if alpha.ndim == 3:
                alpha = np.squeeze(alpha, axis=0)
            residual_l1 = z[residual_l1_key].astype(np.float32)
            residual_rgb = z[residual_rgb_key].astype(np.float32)
        valid = (face_id >= 0) & (alpha >= float(min_alpha)) & np.isfinite(residual_l1) & (residual_l1 > 0.0)
        if not np.any(valid):
            continue
        fids = face_id[valid].reshape(-1)
        l1 = residual_l1[valid].reshape(-1)
        rgb = residual_rgb[:, valid].T.reshape(-1, 3).astype(np.float64)
        count = np.bincount(fids, minlength=size)
        total_count += count.astype(np.int64, copy=False)
        total_l1 += np.bincount(fids, weights=l1.astype(np.float64, copy=False), minlength=size)
        total_rgb[0] += np.bincount(fids, weights=rgb[:, 0], minlength=size)
        total_rgb[1] += np.bincount(fids, weights=rgb[:, 1], minlength=size)
        total_rgb[2] += np.bincount(fids, weights=rgb[:, 2], minlength=size)
        view_count += (count > 0).astype(np.int32, copy=False)

    used = total_count > 0
    used_count = int(used.sum())
    if used_count <= 0:
        rows = []
    else:
        face_ids = np.nonzero(used)[0].astype(np.int64)
        counts = total_count[used].astype(np.float64)
        mean_l1 = total_l1[used] / np.maximum(counts, 1.0)
        mean_rgb = (total_rgb[:, used] / np.maximum(counts[None, :], 1.0)).T
        consistency = (np.linalg.norm(mean_rgb, axis=1) > 1e-8).astype(np.float64)
        score = mean_l1 * np.log1p(counts) * np.sqrt(np.maximum(view_count[used].astype(np.float64), 1.0)) * np.maximum(consistency, 1e-3)
        limit = int(top_support_limit)
        if limit > 0 and score.shape[0] > limit:
            keep = np.argpartition(-score, limit - 1)[:limit]
            order = keep[np.argsort(-score[keep])]
        else:
            order = np.argsort(-score)
        rows = [
            {
                "face_id": int(face_ids[idx]),
                "score": float(score[idx]),
                "pixel_count": int(total_count[int(face_ids[idx])]),
                "view_hits": int(view_count[int(face_ids[idx])]),
                "mean_l1_error": float(mean_l1[idx]),
                "mean_texture": 0.0,
                "residual_consistency": float(consistency[idx]),
                "mean_residual_r": float(mean_rgb[idx, 0]),
                "mean_residual_g": float(mean_rgb[idx, 1]),
                "mean_residual_b": float(mean_rgb[idx, 2]),
            }
            for idx in order.tolist()
        ]
    csv_path = out_dir / "top_residual_supports.csv"
    parent_csv = out_dir / "top_residual_supports_parent.csv"
    if csv_path.is_file() and not parent_csv.exists():
        csv_path.replace(parent_csv)
    fieldnames = [
        "rank",
        "face_id",
        "score",
        "pixel_count",
        "view_hits",
        "mean_l1_error",
        "mean_texture",
        "residual_consistency",
        "mean_residual_r",
        "mean_residual_g",
        "mean_residual_b",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})
    return {
        "top_residual_supports_csv": str(csv_path),
        "parent_top_residual_supports_csv": str(parent_csv) if parent_csv.exists() else None,
        "rows": int(len(rows)),
        "nonzero_faces": int(used_count),
        "top_support_limit": int(top_support_limit),
        "top_preview": rows[:10],
    }


def _update_summary(out_dir: Path, payload: dict[str, Any]) -> None:
    summary_path = out_dir / "surface_evidence_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fields = list(summary.get("per_view_npz_fields", []))
    for key in payload["fields_written"]:
        if key not in fields:
            fields.append(key)
    summary["per_view_npz_fields"] = fields
    summary["teacher_surface_evidence_augmentation"] = payload
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    md = [
        "# Teacher Surface Evidence Cache Augmentation",
        "",
        f"- base evidence dir: `{summary['base_evidence_dir']}`",
        f"- teacher render dir: `{summary['teacher_render_dir']}`",
        f"- parent source: `{summary['parent_source']}`",
        f"- processed views: `{summary['processed_views']}`",
        f"- skipped views: `{summary['skipped_views']}`",
        f"- mean active fraction: `{summary['mean_active_fraction']:.6f}`",
        f"- mean target L1: `{summary['mean_target_l1']:.6f}`",
        f"- selection mode: `{summary['selection_mode']}`",
        f"- mask target: `{summary['mask_target']}`",
        "",
        "Fields written:",
        "",
    ]
    md.extend(f"- `{field}`" for field in summary["fields_written"])
    if summary.get("top_support_rebuild"):
        top = summary["top_support_rebuild"]
        md.extend(
            [
                "",
                "Top-support rebuild:",
                "",
                f"- rows: `{top['rows']}`",
                f"- csv: `{top['top_residual_supports_csv']}`",
                f"- parent csv: `{top.get('parent_top_residual_supports_csv')}`",
            ]
        )
    (out_dir / "teacher_surface_evidence_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    _copy_cache(args.base_evidence_dir, args.out_dir, force=bool(args.force))
    view_dir = _per_view_dir(args.out_dir)
    view_paths = sorted(view_dir.glob("*.npz"))
    if int(args.max_views) > 0:
        view_paths = view_paths[: int(args.max_views)]
    teacher_images = _image_index(args.teacher_render_dir)
    parent_images = _image_index(args.parent_render_dir) if args.parent_render_dir is not None else {}

    view_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in view_paths:
        teacher = teacher_images.get(path.stem)
        if teacher is None:
            skipped.append({"view": path.stem, "reason": "missing_teacher_render"})
            continue
        parent = parent_images.get(path.stem) if parent_images else None
        if args.parent_render_dir is not None and parent is None:
            skipped.append({"view": path.stem, "reason": "missing_parent_render"})
            continue
        view_summaries.append(_augment_one(path, teacher_image=teacher, parent_image=parent, args=args))

    if not view_summaries:
        raise RuntimeError("no per-view NPZ files were augmented")

    top_support_rebuild = None
    if bool(args.rebuild_top_supports):
        top_support_rebuild = _rebuild_top_supports(
            [view_dir / f"{row['view']}.npz" for row in view_summaries],
            args.out_dir,
            residual_rgb_key=str(args.teacher_residual_rgb_key),
            residual_l1_key=str(args.teacher_residual_l1_key),
            min_alpha=float(args.top_support_min_alpha),
            top_support_limit=int(args.top_support_limit),
        )

    fields_written = [
        str(args.teacher_residual_rgb_key),
        str(args.teacher_residual_l1_key),
        str(args.teacher_residual_rgb_raw_key),
        str(args.teacher_better_mask_key),
        str(args.teacher_gain_l1_key),
        str(args.teacher_parent_delta_l1_key),
    ]
    summary = {
        "operator": "ecsr_build_teacher_surface_evidence_cache",
        "test_usage": "none",
        "base_evidence_dir": str(args.base_evidence_dir),
        "teacher_render_dir": str(args.teacher_render_dir),
        "parent_render_dir": str(args.parent_render_dir) if args.parent_render_dir else None,
        "parent_source": "parent_render_dir" if args.parent_render_dir else "npz:rgb_render",
        "out_dir": str(args.out_dir),
        "processed_views": int(len(view_summaries)),
        "skipped_views": int(len(skipped)),
        "skipped": skipped[:50],
        "selection_mode": str(args.selection_mode),
        "mask_target": bool(args.mask_target),
        "teacher_render_error_margin": float(args.teacher_render_error_margin),
        "teacher_parent_delta_min": float(args.teacher_parent_delta_min),
        "fields_written": fields_written,
        "mean_active_fraction": float(np.mean([row["active_fraction"] for row in view_summaries])),
        "mean_target_l1": float(np.mean([row["mean_target_l1"] for row in view_summaries])),
        "mean_raw_parent_delta_l1": float(np.mean([row["mean_raw_parent_delta_l1"] for row in view_summaries])),
        "mean_positive_teacher_gain_l1": float(
            np.mean([row["mean_positive_teacher_gain_l1"] for row in view_summaries if row["mean_positive_teacher_gain_l1"] is not None])
        )
        if any(row["mean_positive_teacher_gain_l1"] is not None for row in view_summaries)
        else None,
        "view_summaries": view_summaries,
        "top_support_rebuild": top_support_rebuild,
    }
    (args.out_dir / "teacher_surface_evidence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _update_summary(args.out_dir, summary)
    _write_report(args.out_dir, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
