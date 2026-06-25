#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from render import _camera_record, _downsample_rend_ids_nearest, _sha256
from triangle_renderer import render

from scripts.car_model.build_v103_surface_affine_residual_field import (
    _assert_strict_camera_matches_bank,
    _dtype,
    _load_delta_bank,
    _load_scene,
)


BASIS_ORDER = ["1", "barycentric_0", "barycentric_1", "viewdir_x", "viewdir_y", "viewdir_z"]
XTX_ORDER = [(i, j) for i in range(len(BASIS_ORDER)) for j in range(i, len(BASIS_ORDER))]


def _accumulate_view_affine_view(
    *,
    ids: torch.Tensor,
    delta: torch.Tensor,
    projected_xy: torch.Tensor,
    faces: torch.Tensor,
    face_centers: torch.Tensor,
    camera_center: torch.Tensor,
    xtx_flat: torch.Tensor,
    xty: torch.Tensor,
    chunk_pixels: int,
) -> dict[str, Any]:
    if ids.ndim != 2:
        raise RuntimeError(f"expected 2D ids after downsample, got {tuple(ids.shape)}")
    if delta.ndim != 3 or int(delta.shape[0]) != 3:
        raise RuntimeError(f"expected delta shape [3,H,W], got {tuple(delta.shape)}")
    if tuple(delta.shape[-2:]) != tuple(ids.shape):
        raise RuntimeError(f"delta/id shape mismatch: delta={tuple(delta.shape)} ids={tuple(ids.shape)}")

    triangle_count = int(faces.shape[0])
    valid = (ids >= 0) & (ids < triangle_count)
    total_pixels = int(ids.numel())
    valid_pixels = int(valid.sum().item())
    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "accumulated_pixels": 0,
            "valid_fraction": 0.0,
            "accumulated_fraction": 0.0,
            "unique_triangles": 0,
            "invalid_topology_pixels": 0,
            "nonfinite_basis_pixels": 0,
            "degenerate_basis_pixels": 0,
        }

    pixel_yx = valid.nonzero(as_tuple=False)
    flat_ids_all = ids[valid].reshape(-1).long()
    unique_triangles = int(torch.unique(flat_ids_all).numel())
    delta_hwc = delta.permute(1, 2, 0).contiguous()
    chunk = max(1, int(chunk_pixels))
    accumulated_pixels = 0
    invalid_topology_pixels = 0
    nonfinite_basis_pixels = 0
    degenerate_basis_pixels = 0
    vertex_count = int(projected_xy.shape[0])

    for start in range(0, valid_pixels, chunk):
        end = min(start + chunk, valid_pixels)
        local_ids = flat_ids_all[start:end]
        local_yx = pixel_yx[start:end]
        vertex_ids = faces[local_ids]
        vertex_ok = (vertex_ids >= 0).all(dim=1) & (vertex_ids < vertex_count).all(dim=1)
        if not bool(vertex_ok.any().item()):
            invalid_topology_pixels += int(vertex_ids.shape[0])
            continue
        if not bool(vertex_ok.all().item()):
            invalid_topology_pixels += int((~vertex_ok).sum().item())
            local_ids = local_ids[vertex_ok]
            local_yx = local_yx[vertex_ok]
            vertex_ids = vertex_ids[vertex_ok]

        xy = projected_xy[vertex_ids]
        p = torch.stack(
            [local_yx[:, 1].to(dtype=torch.float32), local_yx[:, 0].to(dtype=torch.float32)],
            dim=1,
        )
        a = xy[:, 0, :]
        b = xy[:, 1, :]
        c = xy[:, 2, :]
        denom = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        safe = denom.abs() > 1e-8
        safe_denom = torch.where(safe, denom, torch.ones_like(denom))
        w0 = ((b[:, 1] - c[:, 1]) * (p[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (p[:, 1] - c[:, 1])) / safe_denom
        w1 = ((c[:, 1] - a[:, 1]) * (p[:, 0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (p[:, 1] - c[:, 1])) / safe_denom
        w0 = torch.where(safe, w0, torch.zeros_like(w0))
        w1 = torch.where(safe, w1, torch.zeros_like(w1))
        degenerate_basis_pixels += int((~safe).sum().item())

        direction = camera_center.unsqueeze(0) - face_centers[local_ids]
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
        finite = torch.isfinite(w0) & torch.isfinite(w1) & torch.isfinite(direction).all(dim=1)
        if not bool(finite.any().item()):
            nonfinite_basis_pixels += int(w0.shape[0])
            continue
        if not bool(finite.all().item()):
            nonfinite_basis_pixels += int((~finite).sum().item())
            local_ids = local_ids[finite]
            local_yx = local_yx[finite]
            w0 = w0[finite]
            w1 = w1[finite]
            direction = direction[finite]

        one = torch.ones_like(w0, dtype=torch.float64)
        values = delta_hwc[local_yx[:, 0], local_yx[:, 1], :].to(dtype=torch.float64)
        basis = torch.cat(
            [
                one[:, None],
                w0.to(dtype=torch.float64)[:, None],
                w1.to(dtype=torch.float64)[:, None],
                direction.to(dtype=torch.float64),
            ],
            dim=1,
        )
        xtx_values = torch.stack([basis[:, i] * basis[:, j] for i, j in XTX_ORDER], dim=1)
        xty_values = basis[:, :, None] * values[:, None, :]
        xtx_flat.index_add_(0, local_ids, xtx_values)
        xty.index_add_(0, local_ids, xty_values)
        accumulated_pixels += int(local_ids.numel())

    return {
        "valid_pixels": int(valid_pixels),
        "accumulated_pixels": int(accumulated_pixels),
        "valid_fraction": float(valid_pixels / max(1, total_pixels)),
        "accumulated_fraction": float(accumulated_pixels / max(1, total_pixels)),
        "unique_triangles": int(unique_triangles),
        "invalid_topology_pixels": int(invalid_topology_pixels),
        "nonfinite_basis_pixels": int(nonfinite_basis_pixels),
        "degenerate_basis_pixels": int(degenerate_basis_pixels),
    }


def _solve_view_affine_coefficients(
    *,
    xtx_flat: torch.Tensor,
    xty: torch.Tensor,
    min_count: int,
    ridge: float,
    residual_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    triangle_count = int(xtx_flat.shape[0])
    feature_count = len(BASIS_ORDER)
    counts = torch.round(xtx_flat[:, 0]).to(dtype=torch.int64)
    valid_mask = counts >= int(min_count)
    coeffs = torch.zeros((triangle_count, feature_count, 3), dtype=torch.float32)
    mean_residuals = torch.zeros((triangle_count, 3), dtype=torch.float32)
    if bool(valid_mask.any().item()):
        mean_residuals[valid_mask] = (
            xty[valid_mask, 0, :] / counts[valid_mask].to(dtype=torch.float64).unsqueeze(1).clamp_min(1.0)
        ).to(dtype=torch.float32)

    valid_ids_all = valid_mask.nonzero(as_tuple=False).reshape(-1)
    solve_failures = 0
    solve_chunks = 0
    identity = torch.eye(feature_count, dtype=torch.float64)
    solve_chunk_triangles = 131_072
    for start in tqdm(range(0, int(valid_ids_all.numel()), solve_chunk_triangles), desc="solving v104 view affine fields"):
        end = min(start + solve_chunk_triangles, int(valid_ids_all.numel()))
        tri_ids = valid_ids_all[start:end]
        if int(tri_ids.numel()) == 0:
            continue
        a = torch.zeros((int(tri_ids.numel()), feature_count, feature_count), dtype=torch.float64)
        s = xtx_flat[tri_ids]
        for k, (i, j) in enumerate(XTX_ORDER):
            a[:, i, j] = s[:, k]
            a[:, j, i] = s[:, k]
        if float(ridge) > 0.0:
            a = a + float(ridge) * identity.unsqueeze(0)
        b = xty[tri_ids]
        try:
            sol, info = torch.linalg.solve_ex(a, b)
            failed = info != 0
        except AttributeError:
            sol = torch.linalg.solve(a, b)
            failed = torch.zeros((int(tri_ids.numel()),), dtype=torch.bool)
        if bool(failed.any().item()):
            failed_count = int(failed.sum().item())
            solve_failures += failed_count
            sol[failed] = torch.matmul(torch.linalg.pinv(a[failed]), b[failed])
        coeffs[tri_ids] = sol.to(dtype=torch.float32)
        solve_chunks += 1

    coeffs[~valid_mask] = 0.0
    mean_residuals[~valid_mask] = 0.0
    stats = {
        "valid_triangles": int(valid_mask.sum().item()),
        "invalid_triangles": int((~valid_mask).sum().item()),
        "solve_failures": int(solve_failures),
        "solve_chunks": int(solve_chunks),
        "solve_chunk_triangles": int(solve_chunk_triangles),
    }
    return (
        coeffs.to(dtype=residual_dtype).contiguous(),
        mean_residuals.to(dtype=residual_dtype).contiguous(),
        counts.to(dtype=torch.int32).contiguous(),
        stats,
    )


def build_field(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model_path)
    delta_bank_path = Path(args.delta_bank_path)
    output_field = Path(args.output_field)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not delta_bank_path.is_file():
        raise FileNotFoundError(delta_bank_path)
    if int(args.renderer_scaling) <= 0:
        raise ValueError("--renderer_scaling must be positive")
    if int(args.chunk_pixels) <= 0:
        raise ValueError("--chunk_pixels must be positive")
    if int(args.min_count) <= 0:
        raise ValueError("--min_count must be positive")
    if float(args.ridge) < 0.0:
        raise ValueError("--ridge must be non-negative")
    if float(args.residual_clip) < 0.0:
        raise ValueError("--residual_clip must be non-negative")

    residual_dtype = _dtype(args.residual_dtype)
    delta_bank = _load_delta_bank(delta_bank_path, str(args.split), str(args.endpoint_method))
    deltas = delta_bank["deltas"]
    frames = delta_bank["frames"]

    dataset, pipe, triangles, scene, background = _load_scene(model_path, int(args.iteration), int(args.renderer_scaling))
    views = scene.getTestCameras() if str(args.split) == "test" else scene.getTrainCameras()
    if len(deltas) != len(views):
        raise RuntimeError(f"delta bank/view count mismatch: bank={len(deltas)} views={len(views)}")
    extra_frame_meta = [str(key) for key in frames.keys() if str(key) not in deltas]
    if extra_frame_meta:
        raise RuntimeError(f"delta bank has extra frame metadata entries: {len(extra_frame_meta)}")

    faces = triangles.get_triangle_indices.detach().cpu().long().contiguous()
    vertices = triangles.get_vertices.detach().cpu().float().contiguous()
    face_centers = vertices[faces].mean(dim=1).contiguous()
    triangle_count = int(faces.shape[0])
    feature_count = len(BASIS_ORDER)
    xtx_flat = torch.zeros((triangle_count, feature_count * (feature_count + 1) // 2), dtype=torch.float64)
    xty = torch.zeros((triangle_count, feature_count, 3), dtype=torch.float64)
    view_reports: list[dict[str, Any]] = []
    started = time.time()

    for idx, view in enumerate(tqdm(views, desc=f"v104 view-affine surface field {args.split}")):
        key = f"{idx:05d}"
        if key not in deltas:
            raise RuntimeError(f"missing delta for target frame {key}")
        _assert_strict_camera_matches_bank(_camera_record(idx, view), frames.get(key, {}).get("target_camera", {}), key)
        with torch.no_grad():
            pkg = render(view, triangles, pipe, background)
        rendering = pkg["render"]
        if "rend_ids" not in pkg or pkg["rend_ids"] is None:
            raise RuntimeError("renderer package missing rend_ids; cannot build view-affine surface residual field")
        if "image_2D" not in pkg or pkg["image_2D"] is None:
            raise RuntimeError("renderer package missing image_2D; cannot build view-affine surface residual field")
        ids = _downsample_rend_ids_nearest(pkg["rend_ids"], rendering.shape[-2:]).detach().cpu().long()
        delta = deltas[key].detach().cpu().float()
        if tuple(delta.shape) != tuple(rendering.shape):
            raise RuntimeError(
                f"delta/render shape mismatch for {key}: delta={tuple(delta.shape)} render={tuple(rendering.shape)}"
            )
        report = _accumulate_view_affine_view(
            ids=ids,
            delta=delta,
            projected_xy=pkg["image_2D"].detach().cpu().float().contiguous(),
            faces=faces,
            face_centers=face_centers,
            camera_center=view.camera_center.detach().cpu().float(),
            xtx_flat=xtx_flat,
            xty=xty,
            chunk_pixels=int(args.chunk_pixels),
        )
        report.update(
            {
                "frame": key,
                "mean_abs_delta": float(delta.abs().mean().item()),
                "camera_validated": True,
            }
        )
        view_reports.append(report)
        del pkg, rendering, ids, delta

    coefficients, mean_residuals, counts_out, solve_stats = _solve_view_affine_coefficients(
        xtx_flat=xtx_flat,
        xty=xty,
        min_count=int(args.min_count),
        ridge=float(args.ridge),
        residual_dtype=residual_dtype,
    )
    valid_mask = counts_out >= int(args.min_count)
    endpoint_report = str(delta_bank.get("endpoint_report", "") or "")
    endpoint_report_sha = str(delta_bank.get("endpoint_report_sha256", "") or "")
    source_delta_bank_sha = _sha256(delta_bank_path)
    total_valid_pixels = int(sum(int(row["valid_pixels"]) for row in view_reports))
    total_accumulated_pixels = int(sum(int(row["accumulated_pixels"]) for row in view_reports))
    payload = {
        "schema_version": 1,
        "field_type": "v102_surface_residual_field",
        "basis_type": "affine_barycentric_viewdir",
        "basis_order": BASIS_ORDER,
        "coefficient_layout": "triangle,basis,rgb",
        "created_at_unix": time.time(),
        "model_path": str(model_path),
        "split": str(args.split),
        "iteration": int(args.iteration),
        "endpoint_method": str(delta_bank.get("endpoint_method", args.endpoint_method)),
        "source_bank_split": str(delta_bank.get("split", "") or ""),
        "endpoint_report": endpoint_report,
        "endpoint_report_sha256": endpoint_report_sha,
        "source_delta_bank": str(delta_bank_path),
        "source_delta_bank_sha256": source_delta_bank_sha,
        "source_target_frames": int(len(views)),
        "triangle_count": int(triangle_count),
        "valid_triangles": int(valid_mask.sum().item()),
        "valid_triangle_mask": valid_mask.to(dtype=torch.bool).contiguous(),
        "min_count": int(args.min_count),
        "ridge": float(args.ridge),
        "renderer_scaling": int(args.renderer_scaling),
        "residual_clip": float(args.residual_clip),
        "residual_dtype": str(args.residual_dtype),
        "triangle_coefficients": coefficients,
        "triangle_residuals": mean_residuals,
        "triangle_counts": counts_out,
        "normal_equation_xtx_order": [f"{BASIS_ORDER[i]}*{BASIS_ORDER[j]}" for i, j in XTX_ORDER],
        "normal_equation_xty_layout": "triangle,basis,rgb",
        "total_valid_pixels": int(total_valid_pixels),
        "total_accumulated_pixels": int(total_accumulated_pixels),
        "solve_stats": solve_stats,
        "view_reports": view_reports,
        "camera_validation": "strict_target_camera_match",
        "elapsed_sec": float(time.time() - started),
        "note": (
            "View-conditioned face-local affine barycentric residual field distilled from v102 preprojected deltas. "
            "Basis uses [1, barycentric_0, barycentric_1, viewdir_x, viewdir_y, viewdir_z] and stores no target GT."
        ),
    }
    output_field.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_field)
    manifest = {
        "schema_version": 1,
        "field_path": str(output_field),
        "field_sha256": _sha256(output_field),
        "field_type": payload["field_type"],
        "basis_type": payload["basis_type"],
        "basis_order": payload["basis_order"],
        "coefficient_layout": payload["coefficient_layout"],
        "source_delta_bank": str(delta_bank_path),
        "source_delta_bank_sha256": source_delta_bank_sha,
        "triangle_count": int(triangle_count),
        "valid_triangles": int(valid_mask.sum().item()),
        "valid_triangle_fraction": float(valid_mask.float().mean().item()) if int(triangle_count) else 0.0,
        "source_target_frames": int(len(views)),
        "min_count": int(args.min_count),
        "ridge": float(args.ridge),
        "renderer_scaling": int(args.renderer_scaling),
        "residual_clip": float(args.residual_clip),
        "residual_dtype": str(args.residual_dtype),
        "endpoint_method": str(delta_bank.get("endpoint_method", args.endpoint_method)),
        "source_bank_split": str(delta_bank.get("split", "") or ""),
        "total_valid_pixels": int(total_valid_pixels),
        "total_accumulated_pixels": int(total_accumulated_pixels),
        "solve_stats": solve_stats,
        "camera_validation": payload["camera_validation"],
        "elapsed_sec": float(time.time() - started),
        "note": payload["note"],
    }
    manifest_path = output_field.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "field": str(output_field),
                "manifest": str(manifest_path),
                "basis_type": payload["basis_type"],
                "valid_triangles": manifest["valid_triangles"],
                "total_accumulated_pixels": manifest["total_accumulated_pixels"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a v104 view-conditioned face-local residual field from a v102 preprojected delta bank."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--delta_bank_path", required=True)
    parser.add_argument("--output_field", required=True)
    parser.add_argument("--endpoint_method", default="ours_26000_v100_checkpoint_attached_ela_endpoint")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--split", default="test", choices=("test", "train"))
    parser.add_argument("--renderer_scaling", type=int, required=True)
    parser.add_argument("--residual_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--residual_clip", type=float, default=0.08)
    parser.add_argument("--chunk_pixels", type=int, default=500_000)
    return parser.parse_args()


def main() -> int:
    build_field(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
