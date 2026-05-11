#!/usr/bin/env python3
"""Phase-A Surface Evidence Cache for ECSR.

The cache renders train views with MeshSplatting's triangle renderer, stores
per-pixel face ids/residual signals for a bounded set of views, and aggregates
the residual signal onto surface supports.  It is a diagnostic cache: it does
not accept/reject method candidates and never reads held-out test views unless
explicitly requested for a final-report-only visualization.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from triangle_renderer import TriangleModel, render
from utils.general_utils import safe_state


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _to_uint8_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))


def _heatmap(values: np.ndarray, color: str = "magma") -> Image.Image:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        norm = np.zeros_like(values, dtype=np.float32)
    else:
        vmax = float(np.percentile(values[finite], 99.5)) + 1e-8
        norm = np.clip(values / vmax, 0.0, 1.0)
    if color == "green":
        rgb = np.stack([0.08 + 0.15 * norm, 0.08 + 0.90 * norm, 0.08 + 0.25 * norm], axis=-1)
    else:
        rgb = np.stack([0.08 + 0.92 * norm, 0.08 + 0.50 * np.sqrt(norm), 0.10 + 0.15 * (1.0 - norm)], axis=-1)
    return _to_uint8_image(rgb)


def _texture_strength(gt: np.ndarray) -> np.ndarray:
    gray = gt.mean(axis=0)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    tex = gx + gy
    scale = float(np.percentile(tex, 97)) + 1e-8
    return np.clip(tex / scale, 0.0, 1.0).astype(np.float32)


def _save_contact_sheet(
    out_path: Path,
    view_panels: list[dict[str, Any]],
    top_faces: set[int],
    *,
    max_rows: int = 6,
) -> None:
    if not view_panels:
        return
    rows = view_panels[:max_rows]
    font = _load_font(16)
    small = _load_font(13)
    cell_w = 260
    cell_h = 170
    gap = 14
    label_h = 48
    headers = ["GT", "Render", "Error", "Top residual supports"]
    width = 24 * 2 + len(headers) * cell_w + (len(headers) - 1) * gap
    row_h = label_h + 26 + cell_h + 16
    height = 24 + len(rows) * row_h + 18
    canvas = Image.new("RGB", (width, height), (22, 24, 28))
    draw = ImageDraw.Draw(canvas)
    for row_idx, panel in enumerate(rows):
        y = 24 + row_idx * row_h
        draw.text(
            (24, y),
            f"{panel['key']}  addressable={100*panel['top_addressable_fraction']:.1f}%  "
            f"valid={100*panel['valid_fraction']:.1f}%",
            font=font,
            fill=(245, 245, 245),
        )
        face_ids = panel["face_ids"]
        top_mask = np.isin(face_ids, list(top_faces))
        gt = _to_uint8_image(panel["gt"].transpose(1, 2, 0))
        render_img = _to_uint8_image(panel["render"].transpose(1, 2, 0))
        err = _heatmap(panel["abs_error"])
        overlay = panel["render"].transpose(1, 2, 0).copy()
        overlay[top_mask] = 0.35 * overlay[top_mask] + 0.65 * np.array([0.0, 1.0, 0.25], dtype=np.float32)
        overlay_img = _to_uint8_image(overlay)
        for col, (header, image) in enumerate(zip(headers, [gt, render_img, err, overlay_img])):
            x = 24 + col * (cell_w + gap)
            draw.text((x, y + label_h), header, font=small, fill=(210, 216, 224))
            image = image.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y + label_h + 24))
            draw.rectangle([x, y + label_h + 24, x + cell_w, y + label_h + 24 + cell_h], outline=(70, 74, 84))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _unique_reduce(face_ids: np.ndarray, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    face_ids = np.asarray(face_ids, dtype=np.int64)
    if face_ids.size == 0:
        return {"face_id": np.empty((0,), dtype=np.int64)}
    if int(face_ids.min()) >= 0 and int(face_ids.max()) <= 50_000_000:
        max_face = int(face_ids.max())
        # MeshSplatting face ids are dense enough for bincount and this avoids
        # the pathological np.unique/add.at path on high-resolution outdoor scenes.
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


def _read_audit_scene(audit_json: Path | None, scene_name: str) -> dict[str, Any]:
    if audit_json is None or not audit_json.exists():
        return {}
    payload = json.loads(audit_json.read_text(encoding="utf-8"))
    for record in payload.get("scene_records", []):
        if str(record.get("scene")) == scene_name:
            return record
    return {}


def _diagnostic_b(record: dict[str, Any], compact_result: dict[str, Any], method_result: dict[str, Any]) -> dict[str, Any]:
    row = record.get("row", {})
    if not row:
        return {
            "verdict": "unknown",
            "reason": "current-state audit row unavailable",
        }
    clean_psnr = float(row.get("baseline_psnr", np.nan))
    clean_ssim = float(row.get("baseline_ssim", np.nan))
    clean_lpips = float(row.get("baseline_lpips", np.nan))
    compact_dpsnr = float(compact_result.get("PSNR", np.nan)) - clean_psnr
    compact_dssim = float(compact_result.get("SSIM", np.nan)) - clean_ssim
    compact_dlpips = float(compact_result.get("LPIPS", np.nan)) - clean_lpips
    method_dpsnr = float(method_result.get("PSNR", np.nan)) - clean_psnr
    method_dlpips = float(method_result.get("LPIPS", np.nan)) - clean_lpips
    geom_safe = (
        float(row.get("d_abs_rel", 0.0)) <= 1e-6
        and float(row.get("d_depth_mae", 0.0)) <= 1e-6
        and float(row.get("d_normal", 0.0)) <= 1e-6
    )
    ela_gain_lpips = compact_dlpips - method_dlpips
    if geom_safe and ela_gain_lpips > 0.005:
        verdict = "appearance-relocation-promising"
        reason = "geometry is safe while ELA recovers LPIPS beyond compact-only"
    elif not geom_safe:
        verdict = "geometry-first"
        reason = "geometry metrics regress; contraction/topology must be solved before residual relocation"
    else:
        verdict = "weak-relocation-signal"
        reason = "ELA gain over compact-only is small under current aggregate metrics"
    return {
        "verdict": verdict,
        "reason": reason,
        "compact_dpsnr": compact_dpsnr,
        "compact_dssim": compact_dssim,
        "compact_dlpips": compact_dlpips,
        "method_dpsnr": method_dpsnr,
        "method_dlpips": method_dlpips,
        "ela_gain_lpips_over_compact": ela_gain_lpips,
    }


def _parse_view_indices(spec: str) -> list[int]:
    indices: list[int] = []
    for token in str(spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        indices.append(int(token))
    return indices


def _compute_top_face_barycentric(
    face_ids: np.ndarray,
    projected_vertices_xy: np.ndarray,
    faces_np: np.ndarray,
    selected_faces: set[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute image-plane barycentric coordinates for selected visible faces.

    The CUDA renderer currently exposes the winning face id per pixel but not
    barycentric coordinates. For the residual-fitting cache we only need stable
    local coordinates on the top residual supports, so this routine reconstructs
    a conservative 2D barycentric map from projected triangle vertices.
    """

    h, w = face_ids.shape
    bary = np.zeros((3, h, w), dtype=np.float32)
    valid = np.zeros((h, w), dtype=bool)
    if not selected_faces:
        return bary, valid, 0

    yy, xx = np.indices((h, w), dtype=np.float32)
    used_faces = 0
    max_face = int(faces_np.shape[0])
    visible = set(int(x) for x in np.unique(face_ids[face_ids >= 0]))
    for face_id in sorted(selected_faces & visible):
        if face_id < 0 or face_id >= max_face:
            continue
        vertex_ids = faces_np[face_id].astype(np.int64)
        if np.any(vertex_ids < 0) or np.any(vertex_ids >= projected_vertices_xy.shape[0]):
            continue
        mask = face_ids == face_id
        if not np.any(mask):
            continue
        p0, p1, p2 = projected_vertices_xy[vertex_ids]
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-8 or not np.isfinite(denom):
            local = np.full((3, int(mask.sum())), 1.0 / 3.0, dtype=np.float32)
        else:
            mx = xx[mask]
            my = yy[mask]
            b0 = ((y1 - y2) * (mx - x2) + (x2 - x1) * (my - y2)) / denom
            b1 = ((y2 - y0) * (mx - x2) + (x0 - x2) * (my - y2)) / denom
            b2 = 1.0 - b0 - b1
            local = np.stack([b0, b1, b2], axis=0).astype(np.float32)
            local = np.nan_to_num(local, nan=1.0 / 3.0, posinf=1.0 / 3.0, neginf=1.0 / 3.0)
        bary[:, mask] = local
        valid[mask] = True
        used_faces += 1
    return bary, valid, used_faces


def build_cache(args, dataset, pipeline) -> dict[str, Any]:
    scene_name = args.scene_name or Path(dataset.model_path).parts[-3 if Path(dataset.model_path).name == "compact_model" else -1]
    out_dir = Path(args.out_dir) / scene_name
    per_view_dir = out_dir / "views"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_view_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        triangles = TriangleModel(dataset.sh_degree)
        triangles.scaling = int(args.internal_upsample)
        scene = Scene(
            args=dataset,
            triangles=triangles,
            init_opacity=None,
            set_sigma=None,
            load_iteration=int(args.iteration),
            shuffle=False,
        )
        views = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
        faces_np = triangles.get_triangle_indices.detach().cpu().long().numpy()
        requested_indices = _parse_view_indices(getattr(args, "view_indices", ""))
        if requested_indices:
            indexed_views = []
            for idx in requested_indices:
                if idx < 0 or idx >= len(views):
                    raise ValueError(f"view index {idx} out of range for split={args.split} with {len(views)} views")
                indexed_views.append((idx, views[idx]))
        else:
            indexed_views = list(enumerate(views))[int(args.view_offset) :: int(args.view_stride)]
        indexed_views = indexed_views[: int(args.max_views)]
        if not indexed_views:
            raise RuntimeError(f"no views selected for split={args.split}")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        face_chunks: list[np.ndarray] = []
        count_chunks: list[np.ndarray] = []
        err_chunks: list[np.ndarray] = []
        high_err_chunks: list[np.ndarray] = []
        tex_chunks: list[np.ndarray] = []
        residual_pixel_sum_chunks: list[np.ndarray] = []
        residual_view_mean_chunks: list[np.ndarray] = []
        residual_view_norm_chunks: list[np.ndarray] = []
        view_hit_chunks: list[np.ndarray] = []
        view_panels: list[dict[str, Any]] = []
        view_summaries: list[dict[str, Any]] = []
        bary_view_cache: list[dict[str, Any]] = []

        for original_idx, view in tqdm(indexed_views, desc=f"ECSR evidence {scene_name}/{args.split}"):
            pkg = render(view, triangles, pipeline, background)
            rendering_t = pkg["render"].detach().float().clamp(0, 1).cpu()
            gt_t = view.original_image[0:3, :, :].detach().float().clamp(0, 1).cpu()
            if rendering_t.shape != gt_t.shape:
                raise RuntimeError(f"render/GT shape mismatch: {rendering_t.shape} vs {gt_t.shape}")
            h, w = int(gt_t.shape[1]), int(gt_t.shape[2])
            face_ids_t = pkg["rend_ids"].detach().float()
            if face_ids_t.ndim == 3:
                face_ids_t = face_ids_t.unsqueeze(0)
            face_ids_low = F.interpolate(face_ids_t, size=(h, w), mode="nearest").squeeze().detach().cpu().numpy().astype(np.int64)
            alpha_np = pkg["rend_alpha"].detach().float().squeeze().cpu().numpy().astype(np.float32)

            residual = (gt_t - rendering_t).numpy().astype(np.float32)
            abs_error = np.mean(np.abs(residual), axis=0).astype(np.float32)
            gt_np = gt_t.numpy().astype(np.float32)
            render_np = rendering_t.numpy().astype(np.float32)
            texture = _texture_strength(gt_np)
            valid = (face_ids_low >= 0) & (face_ids_low < int(faces_np.shape[0]))
            face_ids_clean = np.where(valid, face_ids_low, -1)
            valid_fraction = float(np.mean(valid))
            threshold = float(np.quantile(abs_error.reshape(-1), float(args.high_error_quantile)))
            high_error = abs_error >= threshold
            high_error_pixels = int(np.sum(high_error))
            top_addressable_fraction = float(np.sum(valid & high_error) / max(high_error_pixels, 1))

            key = f"{original_idx:05d}"
            if args.save_view_npz:
                view_payload = {
                    "face_id": face_ids_clean.astype(np.int32),
                    "residual_l1": abs_error.astype(np.float16),
                    "texture": texture.astype(np.float16),
                    "alpha": alpha_np.astype(np.float16),
                    "depth": pkg["surf_depth"].detach().float().squeeze().cpu().numpy().astype(np.float32),
                    "normal": pkg["surf_normal"].detach().float().cpu().numpy().astype(np.float16),
                    "camera_center": view.camera_center.detach().float().cpu().numpy().astype(np.float32),
                }
                if bool(args.save_residual_rgb):
                    view_payload["residual_rgb"] = residual.astype(np.float16)
                if bool(args.save_rgb):
                    view_payload["rgb_render"] = render_np.astype(np.float16)
                    view_payload["rgb_gt"] = gt_np.astype(np.float16)
                np.savez_compressed(
                    per_view_dir / f"{key}.npz",
                    **view_payload,
                )
            if bool(args.save_barycentric):
                bary_view_cache.append(
                    {
                        "key": key,
                        "face_ids": face_ids_clean.astype(np.int32),
                        "projected_vertices_xy": pkg["image_2D"].detach().float().cpu().numpy().astype(np.float32),
                    }
                )
            Image.fromarray((np.clip(abs_error / (np.percentile(abs_error, 99.5) + 1e-8), 0, 1) * 255).astype(np.uint8)).save(
                per_view_dir / f"{key}_error.png"
            )

            if np.any(valid):
                flat_valid = valid.reshape(-1)
                fids = face_ids_clean.reshape(-1)[flat_valid]
                err = abs_error.reshape(-1)[flat_valid]
                tex = texture.reshape(-1)[flat_valid]
                high = high_error.reshape(-1)[flat_valid].astype(np.float32)
                res = residual.transpose(1, 2, 0).reshape(-1, 3)[flat_valid]
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

            view_panels.append(
                {
                    "key": key,
                    "gt": gt_np,
                    "render": render_np,
                    "abs_error": abs_error,
                    "face_ids": face_ids_clean,
                    "valid_fraction": valid_fraction,
                    "top_addressable_fraction": top_addressable_fraction,
                }
            )
            view_summaries.append(
                {
                    "view_index": int(original_idx),
                    "image_name": str(getattr(view, "image_name", key)),
                    "width": w,
                    "height": h,
                    "valid_face_id_fraction": valid_fraction,
                    "high_error_threshold": threshold,
                    "top_error_addressable_fraction": top_addressable_fraction,
                    "mean_l1_error": float(abs_error.mean()),
                    "p95_l1_error": float(np.percentile(abs_error, 95)),
                }
            )
            del pkg
            torch.cuda.empty_cache()

    if not face_chunks:
        raise RuntimeError("no valid face-id evidence was collected")

    all_faces = np.concatenate(face_chunks)
    reduced = _unique_reduce(
        all_faces,
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
    error_sum = reduced["error_sum"]
    texture_sum = reduced["texture_sum"]
    residual_pixel_sum = reduced["residual_pixel_sum"]
    residual_view_sum = reduced["residual_view_sum"]
    residual_view_norm_sum = reduced["residual_view_norm_sum"]
    view_hits = reduced["view_hits"]
    mean_error = error_sum / np.maximum(counts, 1.0)
    mean_texture = texture_sum / np.maximum(counts, 1.0)
    residual_mean = residual_pixel_sum / np.maximum(counts[:, None], 1.0)
    consistency = np.linalg.norm(residual_view_sum, axis=1) / np.maximum(residual_view_norm_sum, 1e-8)
    score = mean_error * np.log1p(counts) * (0.35 + 0.65 * mean_texture) * (0.5 + 0.5 * np.clip(consistency, 0, 1))
    top_k = min(int(args.top_k_faces), len(score))
    top_idx = _top_k_indices(score, top_k)
    top_faces = set(int(x) for x in reduced["face_id"][top_idx])

    barycentric_written_views = 0
    barycentric_used_faces_total = 0
    if bool(args.save_barycentric):
        if not bool(args.save_view_npz):
            raise RuntimeError("--save_barycentric requires --save_view_npz")
        for item in bary_view_cache:
            bary, bary_valid, used_faces = _compute_top_face_barycentric(
                item["face_ids"],
                item["projected_vertices_xy"],
                faces_np,
                top_faces,
            )
            npz_path = per_view_dir / f"{item['key']}.npz"
            with np.load(npz_path) as old:
                payload = {name: old[name] for name in old.files}
            payload["barycentric"] = bary.astype(np.float16)
            payload["barycentric_valid"] = bary_valid.astype(np.bool_)
            np.savez_compressed(npz_path, **payload)
            barycentric_written_views += 1
            barycentric_used_faces_total += int(used_faces)

    top_csv = out_dir / "top_residual_supports.csv"
    with top_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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
                    int(view_hits[idx]),
                    float(mean_error[idx]),
                    float(mean_texture[idx]),
                    float(consistency[idx]),
                    float(residual_mean[idx, 0]),
                    float(residual_mean[idx, 1]),
                    float(residual_mean[idx, 2]),
                ]
            )
    _save_contact_sheet(out_dir / "surface_residual_contact_sheet.png", view_panels, top_faces)

    total_error = float(np.sum(error_sum))
    top_error = float(np.sum(error_sum[top_idx]))
    top_high = float(np.sum(reduced["high_error_sum"][top_idx]))
    total_high = float(np.sum(reduced["high_error_sum"]))
    top_view_hits = view_hits[top_idx] if len(top_idx) else np.asarray([0.0])
    top_multiview_mask = top_view_hits >= 2
    stable_top = consistency[top_idx][top_multiview_mask] if np.any(top_multiview_mask) else np.asarray([0.0])
    top_multiview_fraction = float(np.mean(top_multiview_mask)) if len(top_idx) else 0.0

    compact_results_path = Path(dataset.model_path) / "results.json"
    compact_results = json.loads(compact_results_path.read_text(encoding="utf-8")) if compact_results_path.exists() else {}
    audit_record = _read_audit_scene(args.audit_json, scene_name)
    diagnostic_b = _diagnostic_b(
        audit_record,
        compact_results.get(args.base_method_name, {}),
        compact_results.get(args.final_method_name, {}),
    )
    per_view_npz_fields = [
        "face_id",
        "residual_l1",
        "texture",
        "alpha",
        "depth",
        "normal",
        "camera_center",
    ]
    if bool(args.save_residual_rgb):
        per_view_npz_fields.append("residual_rgb")
    if bool(args.save_rgb):
        per_view_npz_fields.extend(["rgb_render", "rgb_gt"])
    if bool(args.save_barycentric):
        per_view_npz_fields.extend(["barycentric", "barycentric_valid"])

    summary = {
        "scene": scene_name,
        "model_path": str(dataset.model_path),
        "source_path": str(dataset.source_path),
        "split": args.split,
        "iteration": int(args.iteration),
        "selected_views": [int(i) for i, _ in indexed_views],
        "num_views": len(indexed_views),
        "num_unique_faces": int(len(reduced["face_id"])),
        "top_k_faces": int(top_k),
        "mean_valid_face_id_fraction": float(np.mean([v["valid_face_id_fraction"] for v in view_summaries])),
        "mean_top_error_addressable_fraction": float(np.mean([v["top_error_addressable_fraction"] for v in view_summaries])),
        "top_support_error_fraction": top_error / max(total_error, 1e-8),
        "top_support_high_error_fraction": top_high / max(total_high, 1e-8),
        "top_support_multiview_fraction": top_multiview_fraction,
        "top_support_mean_multiview_consistency": float(np.mean(stable_top)),
        "top_support_median_multiview_consistency": float(np.median(stable_top)),
        "diagnostic_a": {
            "surface_addressability": "pass"
            if float(np.mean([v["top_error_addressable_fraction"] for v in view_summaries])) >= float(args.min_addressable_fraction)
            else "weak",
            "residual_multiview_consistency": "pass"
            if top_multiview_fraction >= float(args.min_multiview_top_fraction)
            and float(np.mean(stable_top)) >= float(args.min_consistency)
            else "weak",
        },
        "diagnostic_b": diagnostic_b,
        "view_summaries": view_summaries,
        "artifacts": {
            "top_residual_supports_csv": str(top_csv),
            "contact_sheet": str(out_dir / "surface_residual_contact_sheet.png"),
            "per_view_dir": str(per_view_dir),
        },
        "per_view_npz_fields": per_view_npz_fields,
        "barycentric_available": bool(args.save_barycentric),
        "barycentric_scope": "top_residual_supports" if bool(args.save_barycentric) else "none",
        "barycentric_written_views": int(barycentric_written_views),
        "barycentric_used_faces_total": int(barycentric_used_faces_total),
    }
    (out_dir / "surface_evidence_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md = [
        f"# ECSR Phase-A Surface Evidence Cache: {scene_name}",
        "",
        f"- model: `{dataset.model_path}`",
        f"- split: `{args.split}`",
        f"- selected views: `{summary['selected_views']}`",
        f"- unique surface supports: `{summary['num_unique_faces']}`",
        f"- mean valid face-id fraction: `{summary['mean_valid_face_id_fraction']:.4f}`",
        f"- top-error addressable fraction: `{summary['mean_top_error_addressable_fraction']:.4f}`",
        f"- top-support error fraction: `{summary['top_support_error_fraction']:.4f}`",
        f"- top-support high-error fraction: `{summary['top_support_high_error_fraction']:.4f}`",
        f"- top-support multi-view fraction: `{summary['top_support_multiview_fraction']:.4f}`",
        f"- top-support mean multi-view residual consistency: `{summary['top_support_mean_multiview_consistency']:.4f}`",
        f"- Diagnostic A addressability: `{summary['diagnostic_a']['surface_addressability']}`",
        f"- Diagnostic A consistency: `{summary['diagnostic_a']['residual_multiview_consistency']}`",
        f"- Diagnostic B: `{diagnostic_b['verdict']}` - {diagnostic_b['reason']}",
        "",
        "![surface residual contact sheet](surface_residual_contact_sheet.png)",
        "",
        "Top supports are saved in `top_residual_supports.csv`; per-view face-id/error arrays are saved under `views/`.",
    ]
    (out_dir / "surface_evidence_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = ArgumentParser(description="Build ECSR Phase-A surface evidence cache.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=26000, type=int)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--scene_name", default="")
    parser.add_argument("--out_dir", default="outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence")
    parser.add_argument("--audit_json", type=Path, default=Path("outputs/carnet/meshsplatopt/ecsr_phase_a/current_state_audit/current_state_audit.json"))
    parser.add_argument("--base_method_name", default="ours_26000")
    parser.add_argument("--final_method_name", default="ours_26000_sor_adaptive_geo_compact_ela")
    parser.add_argument("--max_views", default=8, type=int)
    parser.add_argument("--view_stride", default=6, type=int)
    parser.add_argument("--view_offset", default=0, type=int)
    parser.add_argument("--view_indices", default="", help="Comma-separated split-local view indices; overrides stride/offset.")
    parser.add_argument("--internal_upsample", default=4, type=int)
    parser.add_argument("--high_error_quantile", default=0.90, type=float)
    parser.add_argument("--top_k_faces", default=256, type=int)
    parser.add_argument("--min_addressable_fraction", default=0.85, type=float)
    parser.add_argument("--min_consistency", default=0.25, type=float)
    parser.add_argument("--min_multiview_top_fraction", default=0.25, type=float)
    parser.add_argument("--save_view_npz", action="store_true")
    parser.add_argument(
        "--save_residual_rgb",
        action="store_true",
        help="Store per-pixel RGB residuals in each view NPZ for representation-level relocation fitting.",
    )
    parser.add_argument(
        "--save_rgb",
        action="store_true",
        help="Store render and GT RGB tensors in each view NPZ. This is larger and intended for fitting diagnostics.",
    )
    parser.add_argument(
        "--save_barycentric",
        action="store_true",
        help="Store reconstructed 2D barycentric coordinates for top residual supports in each view NPZ.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    if args.split != "train":
        print("[ECSR] WARNING: non-train split requested. Do not use this for policy selection.", flush=True)
    safe_state(args.quiet)
    summary = build_cache(args, model.extract(args), pipeline.extract(args))
    print(f"[ECSR] wrote {Path(args.out_dir) / summary['scene']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
