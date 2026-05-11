#!/usr/bin/env python3
"""Expand Surface Evidence Cache top supports from saved train-view NPZ files.

This avoids rerendering when we only need a broader train-defined support list
for representation-level residual fitting.  It never reads held-out test views;
it only aggregates fields already present in a train Surface Evidence Cache.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_evidence_dir", type=Path, required=True)
    parser.add_argument("--output_evidence_dir", type=Path, required=True)
    parser.add_argument("--top_k_faces", type=int, default=4096)
    parser.add_argument("--high_error_quantile", type=float, default=0.65)
    parser.add_argument(
        "--max_face_id",
        type=int,
        default=-1,
        help="Optional exclusive upper bound for valid face ids; filters stale positive invalid ids in older NPZ caches.",
    )
    parser.add_argument("--link_views", action="store_true", default=True)
    parser.add_argument("--copy_views", action="store_true")
    return parser.parse_args()


def _unique_reduce(face_ids: np.ndarray, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    face_ids = np.asarray(face_ids, dtype=np.int64)
    if face_ids.size == 0:
        return {"face_id": np.empty((0,), dtype=np.int64)}
    if int(face_ids.min()) >= 0 and int(face_ids.max()) <= 50_000_000:
        max_face = int(face_ids.max())
        hit_count = np.bincount(face_ids, minlength=max_face + 1)
        used = hit_count > 0
        unique = np.nonzero(used)[0].astype(np.int64)
        reduced: dict[str, np.ndarray] = {"face_id": unique}
        for key, value in values.items():
            value = np.asarray(value)
            if value.ndim == 1:
                out_full = np.bincount(face_ids, weights=value.astype(np.float64), minlength=max_face + 1)
                reduced[key] = out_full[used]
            else:
                out = np.empty((len(unique), value.shape[1]), dtype=np.float64)
                for channel in range(value.shape[1]):
                    out_full = np.bincount(
                        face_ids,
                        weights=value[:, channel].astype(np.float64),
                        minlength=max_face + 1,
                    )
                    out[:, channel] = out_full[used]
                reduced[key] = out
        return reduced

    unique, inv = np.unique(face_ids, return_inverse=True)
    reduced = {"face_id": unique.astype(np.int64)}
    for key, value in values.items():
        value = np.asarray(value)
        if value.ndim == 1:
            out = np.zeros((len(unique),), dtype=np.float64)
            np.add.at(out, inv, value.astype(np.float64))
        else:
            out = np.zeros((len(unique), value.shape[1]), dtype=np.float64)
            for channel in range(value.shape[1]):
                np.add.at(out[:, channel], inv, value[:, channel].astype(np.float64))
        reduced[key] = out
    return reduced


def _top_k_indices(score: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0:
        return np.empty((0,), dtype=np.int64)
    if top_k >= len(score):
        return np.argsort(score)[::-1]
    idx = np.argpartition(score, -top_k)[-top_k:]
    return idx[np.argsort(score[idx])[::-1]]


def _prepare_views(source: Path, output: Path, *, link_views: bool, copy_views: bool) -> None:
    source_views = source / "views"
    output_views = output / "views"
    if output_views.exists() or output_views.is_symlink():
        if output_views.is_symlink() or output_views.is_file():
            output_views.unlink()
        elif copy_views:
            shutil.rmtree(output_views)
    if output_views.exists():
        return
    if copy_views:
        shutil.copytree(source_views, output_views)
    elif link_views:
        output_views.symlink_to(source_views.resolve(), target_is_directory=True)
    else:
        output_views.mkdir(parents=True, exist_ok=True)


def expand(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_evidence_dir
    output = args.output_evidence_dir
    output.mkdir(parents=True, exist_ok=True)
    _prepare_views(source, output, link_views=bool(args.link_views), copy_views=bool(args.copy_views))

    view_paths = sorted((source / "views").glob("*.npz"))
    if not view_paths:
        raise FileNotFoundError(f"no view npz files under {source / 'views'}")

    face_chunks: list[np.ndarray] = []
    count_chunks: list[np.ndarray] = []
    err_chunks: list[np.ndarray] = []
    high_err_chunks: list[np.ndarray] = []
    tex_chunks: list[np.ndarray] = []
    residual_pixel_sum_chunks: list[np.ndarray] = []
    residual_view_mean_chunks: list[np.ndarray] = []
    residual_view_norm_chunks: list[np.ndarray] = []
    view_hit_chunks: list[np.ndarray] = []
    per_view: list[dict[str, Any]] = []

    for view_path in view_paths:
        with np.load(view_path) as z:
            required = {"face_id", "residual_l1", "texture", "residual_rgb"}
            missing = sorted(required - set(z.files))
            if missing:
                raise RuntimeError(f"{view_path} missing required fields: {missing}")
            face_id = z["face_id"].astype(np.int64)
            residual_l1 = z["residual_l1"].astype(np.float32)
            texture = z["texture"].astype(np.float32)
            residual_rgb = z["residual_rgb"].astype(np.float32)
        valid = face_id >= 0
        if int(args.max_face_id) > 0:
            valid &= face_id < int(args.max_face_id)
        if not np.any(valid):
            continue
        threshold = float(np.quantile(residual_l1.reshape(-1), float(args.high_error_quantile)))
        flat_valid = valid.reshape(-1)
        fids = face_id.reshape(-1)[flat_valid]
        err = residual_l1.reshape(-1)[flat_valid]
        tex = texture.reshape(-1)[flat_valid]
        high = (residual_l1.reshape(-1)[flat_valid] >= threshold).astype(np.float32)
        res = residual_rgb.transpose(1, 2, 0).reshape(-1, 3)[flat_valid]
        reduced = _unique_reduce(
            fids,
            {
                "count": np.ones_like(err, dtype=np.float32),
                "error_sum": err,
                "high_error_sum": high,
                "texture_sum": tex,
                "residual_sum": res,
            },
        )
        mean_res = reduced["residual_sum"] / np.maximum(reduced["count"][:, None], 1.0)
        face_chunks.append(reduced["face_id"])
        count_chunks.append(reduced["count"])
        err_chunks.append(reduced["error_sum"])
        high_err_chunks.append(reduced["high_error_sum"])
        tex_chunks.append(reduced["texture_sum"])
        residual_pixel_sum_chunks.append(reduced["residual_sum"])
        residual_view_mean_chunks.append(mean_res)
        residual_view_norm_chunks.append(np.linalg.norm(mean_res, axis=1))
        view_hit_chunks.append(np.ones((len(reduced["face_id"]),), dtype=np.float64))
        per_view.append(
            {
                "view": view_path.stem,
                "valid_face_pixels": int(flat_valid.sum()),
                "high_error_threshold": threshold,
            }
        )

    if not face_chunks:
        raise RuntimeError("no valid surface evidence found")

    reduced = _unique_reduce(
        np.concatenate(face_chunks),
        {
            "count": np.concatenate(count_chunks),
            "error_sum": np.concatenate(err_chunks),
            "high_error_sum": np.concatenate(high_err_chunks),
            "texture_sum": np.concatenate(tex_chunks),
            "residual_pixel_sum": np.concatenate(residual_pixel_sum_chunks),
            "residual_view_sum": np.concatenate(residual_view_mean_chunks),
            "residual_view_norm_sum": np.concatenate(residual_view_norm_chunks),
            "view_hits": np.concatenate(view_hit_chunks),
        },
    )
    counts = reduced["count"]
    mean_error = reduced["error_sum"] / np.maximum(counts, 1.0)
    mean_texture = reduced["texture_sum"] / np.maximum(counts, 1.0)
    residual_mean = reduced["residual_pixel_sum"] / np.maximum(counts[:, None], 1.0)
    consistency = np.linalg.norm(reduced["residual_view_sum"], axis=1) / np.maximum(
        reduced["residual_view_norm_sum"],
        1e-8,
    )
    score = mean_error * np.log1p(counts) * (0.35 + 0.65 * mean_texture) * (0.5 + 0.5 * np.clip(consistency, 0, 1))
    top_k = min(int(args.top_k_faces), len(score))
    top_idx = _top_k_indices(score, top_k)

    top_csv = output / "top_residual_supports.csv"
    with top_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
        )
        for rank, idx in enumerate(top_idx, start=1):
            writer.writerow(
                [
                    rank,
                    int(reduced["face_id"][idx]),
                    float(score[idx]),
                    int(counts[idx]),
                    int(reduced["view_hits"][idx]),
                    float(mean_error[idx]),
                    float(mean_texture[idx]),
                    float(consistency[idx]),
                    float(residual_mean[idx, 0]),
                    float(residual_mean[idx, 1]),
                    float(residual_mean[idx, 2]),
                ]
            )

    source_summary_path = source / "surface_evidence_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.is_file() else {}
    summary = {
        "scene": source_summary.get("scene", source.name),
        "operator": "expand_surface_evidence_top_supports",
        "test_usage": "none",
        "source_evidence_dir": str(source),
        "output_evidence_dir": str(output),
        "view_source": "symlink" if (output / "views").is_symlink() else "copy",
        "num_views": len(view_paths),
        "num_unique_faces": int(len(reduced["face_id"])),
        "top_k_faces": int(top_k),
        "high_error_quantile": float(args.high_error_quantile),
        "source_top_k_faces": source_summary.get("top_k_faces"),
        "per_view_npz_fields": source_summary.get(
            "per_view_npz_fields",
            ["face_id", "residual_l1", "texture", "alpha", "depth", "normal", "camera_center", "residual_rgb"],
        ),
        "barycentric_available": bool(source_summary.get("barycentric_available", False)),
        "uniform_barycentric_compatible": True,
        "artifacts": {
            "top_residual_supports_csv": str(top_csv),
            "views": str(output / "views"),
        },
        "view_summaries": per_view,
    }
    (output / "surface_evidence_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# Expanded Surface Evidence Supports: {summary['scene']}",
        "",
        f"- source: `{source}`",
        f"- output: `{output}`",
        f"- top supports: `{top_k}`",
        f"- unique faces: `{summary['num_unique_faces']}`",
        f"- high-error quantile: `{summary['high_error_quantile']}`",
        f"- test usage: `{summary['test_usage']}`",
        "",
        "This expansion reuses train-view NPZ evidence and does not rerender or inspect held-out test views.",
    ]
    (output / "surface_evidence_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = expand(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
